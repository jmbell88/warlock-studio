"""``Worker``'s tile-sheet stage: one generation on a grid, cut into tiles.

Its own module for ``_q_sprite``'s reason: a distinct subject with a distinct
publish contract, and folding it into the restyle module would put another
unrelated thing in a class that was split apart for having too many.

What makes it different from every other generating kind here: **it publishes
its own ``input.png`` and that file is the finished artifact**, not a reference
something else is reconstructed from. There is no mesh stage behind it, which is
why its rows are born at ``stage="tilesheet"`` -- a stage no promote, remesh or
editable gate accepts, so the library's buttons refuse it without knowing it
exists.

What is genuinely new is that **the grid is imposed on the model rather than
asked of it**. A canny ControlNet gets a picture of the sixty-four rectangles,
so the cells land where the slicer is already going to cut. The alternative --
prompting for a grid and hoping -- puts every seam a few pixels out, and since
the slice happens on fixed rectangles either way, "a few pixels out" means every
tile carries a sliver of its neighbour.

This module reaches into no other layer: the compositing it needs is
sixty-four crops and a paste, and it lives in ``pipelines.tilesheet``, which
is where the queue is allowed to look.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import guidance, models

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class TileSheetOps:
    """The tile-sheet stage, mixed into :class:`~.queue.Worker`."""

    def _tile_sheet_step(self: Worker, job_id: str, step: int, steps: int) -> None:
        """``_t2i_step`` for a kind with exactly one generation.

        Deliberately not ``self._t2i_step``: that helper is shared with the
        text path, and this kind's ``t2i_sample`` window is a different slice
        of a different table. One pass means the mapping is the identity, which
        is why this is three lines -- a kind that samples N times has to spend
        the one window across all N, and ``_sprite_step`` is that shape.
        """
        self._step_progress(job_id, "t2i_sample", "Drawing the sheet", step, steps)

    async def _tile_sheet(self: Worker, job: dict[str, Any]) -> None:
        """Draw an 8x8 grid of tiles, cut it up, and publish one sheet.

        One generation, then a CPU tail that reduces each of the sixty-four
        cells to exactly the tile size and quantizes the whole sheet to one
        palette. One job rather than sixty-four, because the deliverable is the
        *sheet*: tiles dispatched minutes apart behind different work would not
        share a style, a light direction or a palette, and the expensive part
        (loading the checkpoint and the LoRA) is paid once for all of them.
        """
        import numpy as np
        from PIL import Image

        from . import queue as queue_mod
        from .asset_workflows import tile_plan
        from .pipelines import control, pixel, tilesheet
        from .pipelines.conditioning import Conditioning

        job_id = job["id"]
        params = job["params"]
        # Re-derived rather than trusted: params outlive the door that wrote
        # them, and this raises on a block that no longer validates -- the rule
        # ``_sprite_synthesis`` follows about ``spritesynth.geometry``.
        block = dict(params.get("sheet") or {})
        # The geometry is *required*, not defaulted. Defaulting to 32 is the
        # shape of a silent failure rather than a tolerance: there is exactly
        # one writer (``service.tilesheets.create_tile_sheet``) and it always
        # writes both keys, so absence is corruption and the honest answer is
        # to say so before the card is spent.
        try:
            tile_w = int(block["tile_w"])
            projection = str(block["projection"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("this tile sheet does not say what it is a sheet of") from None
        # ``geometry`` is the single authority on what pairs are buildable, and
        # it raises with a sentence naming the offending value -- so a stored
        # block the door would refuse today costs the request rather than
        # sixty-four cells of nothing.
        geom = tilesheet.geometry(tile_w, projection)
        normalized_plan = None
        request_payload = params.get("generation_request")
        if (
            isinstance(request_payload, dict)
            and request_payload.get("generation_type") == "tileset"
        ):
            request_tile = request_payload.get("tile") or {}
            normalized_plan = tile_plan(
                mode=str(request_tile.get("mode") or "collection"),
                view=str(request_tile.get("view") or projection),
                prompt_items=request_tile.get("prompt_items") or (),
                variants=int(request_tile.get("variants") or 1),
                inner_terrain=str(request_tile.get("inner_terrain") or ""),
                outer_terrain=str(request_tile.get("outer_terrain") or ""),
                boundary=str(request_tile.get("boundary") or ""),
                ground=str(request_tile.get("ground") or ""),
                path=str(request_tile.get("path") or ""),
                edge=str(request_tile.get("edge") or ""),
                target_cell_px=request_tile.get("target_cell_px"),
            )
            params["tile_plan"] = normalized_plan
            await asyncio.to_thread(self.store.set_params, job_id, params)
        colors = int(params.get("colors", 64))
        palette_name = str(params.get("palette") or "")
        dither = bool(params.get("dither"))
        # Read from disk **here**, before the card is spent, and not down at the
        # quantize phase where it is used. It is one small text file, and the
        # alternative is a job that generates the whole sheet and then fails on
        # a palette the user deleted after submitting it -- the same rule the
        # geometry above obeys, for the same reason: params outlive the door
        # that wrote them. ``()`` when no palette was named, which is what the
        # common request says.
        designed = await asyncio.to_thread(
            queue_mod._palette_entries, self.config, palette_name
        )
        seed = int(params.get("seed", 42))
        subject = tilesheet.sheet_subject(job["prompt"] or "", geom.view)

        base_key = self._resolve_base_key(params, default="sdxl_cfg")
        spec = models.BASE_MODELS[base_key]
        pixel_style = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
        lora: str | None = models.PIXEL_SHEET_LORA
        if not models.lora_fits(spec, pixel_style):
            # The same tolerance ``_pixel_sheet`` applies, and needed here for
            # its reason: params outlive the service that wrote them, and a
            # sheet whose sidecar claimed a LoRA that never loaded would be a
            # recipe nobody can reproduce. Drawn bare instead, and said so.
            log.warning(
                "base model %s (%s) cannot take the pixel-art LoRA %s (%s); "
                "drawing the tile sheet without it",
                spec.key, spec.family, pixel_style.key, pixel_style.family,
            )
            lora = None

        job_dir = self.config.job_dir(job_id)
        await asyncio.to_thread(
            functools.partial(job_dir.mkdir, parents=True, exist_ok=True)
        )
        # Written by the door when the user attached one, and simply absent
        # otherwise -- which is the common path. Read from disk rather than
        # from params because params carry the *selection* and this is the
        # image itself.
        ref_png = job_dir / "ref.png"
        has_reference = await asyncio.to_thread(ref_png.exists)

        with tempfile.TemporaryDirectory(prefix="warlock-tilesheet-") as tmp:
            scratch = Path(tmp)
            self.progress.update(
                job_id, phase="guide", label="Ruling the grid",
                inner=0.0, inner_next=1.0, nominal=1.0, detail="",
            )
            guide_path = scratch / "guide.png"
            guide = await asyncio.to_thread(tilesheet.render_guide, geom)
            await asyncio.to_thread(guide.save, guide_path, "PNG")
            # Recorded for the same reason ``_sprite_synthesis`` records it: a
            # guide that drew nothing is otherwise an unanswerable question
            # about why the cells came back in the wrong places.
            guide_edges = await asyncio.to_thread(control.edge_fraction, guide)

            control_scale = float(
                params.get("control_scale", models.DEFAULT_CONTROL_SCALE)
            )
            control_end = float(params.get("control_end", models.DEFAULT_CONTROL_END))
            ip_scale = float(params.get("ip_scale", models.DEFAULT_IP_SCALE))
            cond = Conditioning(
                # The grid guide, always. Deliberately no init image: an img2img
                # start would fight the guide in every cell, and there is no
                # strength axis on this kind for exactly that reason.
                control="canny",
                control_image=guide_path,
                control_scale=control_scale,
                control_end=control_end,
                # And the user's picture, only if there is one. This kind's
                # reference is optional -- it carries the *style* the sheet
                # should be drawn in, not what any cell contains -- so the
                # common request reaches the pipe with one adapter, and
                # ``vram.estimate_parts`` gates the encoder's cost on the same
                # fact.
                **(
                    {"ip_adapter": "plus", "ip_image": ref_png, "ip_scale": ip_scale}
                    if has_reference
                    else {}
                ),
            )

            # The default sentinel is *not* taken: this kind hands over a real
            # ``Conditioning`` and ``_needs_handoff`` reads it directly. Trellis
            # is therefore stopped when one is running, and ``vram.estimate``
            # charges ``TRELLIS_GIB`` under coexist anyway -- an over-charge in
            # the safe direction, and the same shape ``sprite_synthesis`` has.
            t2i, _handoff = await self._acquire_t2i(spec, base_key, cond)
            composed = guidance.compose_prompt(subject, params)
            out_path = scratch / "sheet.png"
            try:
                if self._cancel is not None and self._cancel.event.is_set():
                    # Before the generation, not only after: this is ~20 s of
                    # GPU a cancelled job should not spend, and nothing has been
                    # published, so returning here leaves the run loop's
                    # ``_discard_artifacts`` with nothing to undo.
                    return
                await asyncio.to_thread(
                    functools.partial(
                        t2i.generate,
                        composed,
                        out_path,
                        seed=seed,
                        lora=lora,
                        lora_weight=pixel_style.default_weight,
                        negative_prompt=str(params.get("negative_prompt") or ""),
                        conditioning=cond,
                        on_state=lambda s: self._t2i_state(job_id, s),
                        on_step=functools.partial(self._tile_sheet_step, job_id),
                        cancel_event=self._cancel.event if self._cancel else None,
                        # The grid template, and the frame it is drawn on. An
                        # isometric sheet is 2:1 by definition of the
                        # projection, so generating it square and squashing it
                        # afterwards would return sixty-four ellipses.
                        tilesheet=True,
                        size=geom.source_size,
                    )
                )
            finally:
                # Every reference dropped before the reclaim, which is why the
                # decode below happens after this and not inside the try.
                await self._release_t2i(t2i, spec)

            if self._cancel is not None and self._cancel.event.is_set():
                return

            self.progress.update(
                job_id, phase="slice", label="Cutting the sheet up",
                inner=0.0, inner_next=1.0, nominal=2.0, detail="",
            )
            with Image.open(out_path) as generated:
                generated.load()
                full = generated.convert("RGBA")
            # On the whole generated frame and before the reduction, which is
            # the only place the number means anything: the lattice belongs to
            # the generation, and ``reduce_sheet`` is about to resample it away.
            grid = await asyncio.to_thread(pixel.lattice, full)
            reduced_array = await asyncio.to_thread(
                tilesheet.reduce_sheet, np.asarray(full), geom
            )

        self.progress.update(
            job_id, phase="quantize", label="Sharing one palette",
            inner=0.0, inner_next=1.0, nominal=2.0, detail="",
        )
        # One palette across every tile -- authored or derived, the property is
        # the same and ``quantize_tiles`` carries the argument: sixty-four tiles
        # quantized separately read as sixty-four pictures pasted together,
        # which is exactly what a tile sheet must not. With no palette and no
        # dither this is byte-for-byte the median cut it has always been.
        reduced, palette, palette_source = await asyncio.to_thread(
            functools.partial(
                tilesheet.quantize_tiles,
                Image.fromarray(reduced_array, "RGBA"),
                colors=colors,
                entries=designed,
                dither=dither,
            )
        )

        if self._cancel is not None and self._cancel.event.is_set():
            # Nothing is published for a cancelled draw. The run loop is about
            # to call ``_discard_artifacts``, and writing the sheet first would
            # leave it deleting a file it had just been told to make.
            return

        # The provenance copy first: it is not served, so a failure part way
        # through leaves nothing a reader would mistake for a finished sheet.
        await asyncio.to_thread(full.save, job_dir / "sheet.png", "PNG")

        out_png = job_dir / "input.png"
        # Staged and renamed rather than saved in place: ``input.png`` is a
        # served name, and a second draw of the same row (or a library reader
        # mid-read) must never see a partial file under it.
        out_tmp = out_png.with_name(f".{out_png.name}.tmp")
        try:
            await asyncio.to_thread(reduced.save, out_tmp, "PNG")
            await asyncio.to_thread(os.replace, out_tmp, out_png)
        finally:
            with contextlib.suppress(OSError):
                out_tmp.unlink(missing_ok=True)

        recipe: dict[str, Any] = {
            "base_model": base_key,
            "seed": seed,
            "prompt": t2i.last_prompt or composed,
            "control": "canny",
            "control_scale": control_scale,
            "control_end": control_end,
            "guide_edge_fraction": guide_edges,
            "colors": colors,
            # Empty unless a palette was named or dither asked for, so a sheet
            # drawn the way every sheet before today was drawn writes the
            # sidecar it wrote before. ``palette_record`` holds that rule and
            # the reason ``colors`` stays beside these keys.
            **tilesheet.palette_record(
                name=palette_name,
                entries=designed,
                source=palette_source,
                dither=dither,
            ),
        }
        if has_reference:
            recipe["ip_adapter"] = "plus"
            recipe["ip_scale"] = ip_scale
        if lora is not None:
            recipe["style_lora"] = lora
            recipe["lora_weight"] = pixel_style.default_weight
        doc = tilesheet.sheet_sidecar(
            prompt=job["prompt"] or "",
            tile_w=geom.tile_w,
            tile_h=geom.tile_h,
            view=geom.view,
            colors=colors,
            palette=palette,
            recipe=recipe,
            created=time.time(),
            working_cell_px=(geom.cell_w, geom.cell_h),
            target_cell_px=job.get("params", {}).get("target_cell_px"),
            reduction=(job.get("params", {}).get("reduction_method") or "measured_pixel_reducer"),
            source_seed=seed,
            workflow=normalized_plan,
            grid=grid,
        )
        # Last, and only after the sheet: this file is the completion marker, so
        # publishing it first would advertise a sheet still being written.
        await asyncio.to_thread(
            queue_mod._publish_text, job_dir / "sheet.json", json.dumps(doc, indent=2)
        )
        # On the row as well, so the library can say what it drew without
        # opening a file. Derived, so DERIVED_PARAMS carries it.
        await asyncio.to_thread(
            self.store.merge_params,
            job_id,
            {
                "sheet_report": {
                    "tiles": geom.tiles,
                    "palette": len(palette),
                    "tile_w": geom.tile_w,
                    "tile_h": geom.tile_h,
                    "sheet_w": geom.sheet_size[0],
                    "sheet_h": geom.sheet_size[1],
                    "projection": geom.view,
                    "recipe": recipe,
                }
            },
        )
        log.info(
            "drew tile sheet %s: %d tiles, %d colours at %dx%d (%s)",
            job_id, geom.tiles, len(palette), geom.tile_w, geom.tile_h, geom.view,
        )

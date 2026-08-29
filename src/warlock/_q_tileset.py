"""``Worker``'s tileset stage: N seamless materials, one atlas.

Its own module for ``_q_tilesheet``'s reason -- a distinct subject with a
distinct publish contract -- and deliberately *beside* that one rather than
inside it, because the two are opposites at the only point that matters. The
grid path imposes sixty-four cells on **one** generation with a canny guide and
cuts them out; this path runs **N** generations that each wrap, and lays them
out. ``docs/measurements/2026-08-18-tile-sheet-grid.md`` is why both exist: the
guide is obeyed and the cells still come back identical, because every cell of
the guide is identical and there is no per-cell signal for variety. Variety is a
property of the request, not of the model's composition.

So the mechanism here is not a guide at all. ``tile=True`` selects
``text2image``'s circular-padding path (``text2image.py:99``, applied at
``:975-988`` to the UNet, the VAE and any ControlNet), which is what makes one
material tile against itself -- and it is *refused* rather than degraded on a
non-SDXL checkpoint at ``:955``, because a DiT has no Conv2d to wrap. There is
no ControlNet on this path at all, which is why ``vram.estimate_parts`` gates
that term on ``params["control"]`` and the door writes that key only for the
grid mode.

Two modes, one bracket:

``materials``
    N material descriptions, N generations, reduced and laid out in a plain
    grid by ``tileatlas.assemble``.

``terrain``
    Two materials -- an inner and an outer -- composited into a blob-47 autotile
    row by ``tilemask.blob_atlas``. The boundary is a scalar field rather than a
    drawing, so the model is never asked to draw an edge.

**Both modes are the same job with a different tail**, which is why the body
below is one coroutine that the two entry points name. The acquire/release
bracket and the publish ordering are invariants; two copies of them is how they
would come to disagree.

This module reaches into no other layer: everything it needs to think with is in
``pipelines.tileatlas``, ``pipelines.tilemask`` and -- for the quantize tail the
two tile kinds share -- ``pipelines.tilesheet``, which is where the queue is
allowed to look. That last one is the grid path's module and is imported here on
purpose: the one-palette-across-the-whole-atlas property is the same property in
both kinds and is not about how the cells were produced, so it is written once
in ``tilesheet.quantize_tiles`` rather than twice in the two ``_q_`` modules
that would then be free to disagree.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import guidance, models

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)

#: Where the un-reduced generations are kept beside the finished atlas. A
#: directory rather than the grid path's single ``sheet.png``, because there are
#: N of them -- and it is on ``_discard_artifacts``' tile-sheet list for the
#: reason that file is: a cancelled draw leaves nothing.
MATERIALS_DIR = "materials"


class TileSetOps:
    """The seamless-material tileset stage, mixed into :class:`~.queue.Worker`."""

    def _material_step(
        self: Worker, job_id: str, index: int, count: int, step: int, steps: int
    ) -> None:
        """``_tile_sheet_step`` for a kind that samples N times.

        ``progress.PHASES_TILE_SHEET`` has one ``t2i_sample`` window and this
        kind spends it across every pass, which is the shape ``_sprite_step``
        already has and the reason ``_q_tilesheet``'s own three-line version
        says a kind that samples N times cannot use it: the mapping is no longer
        the identity. Material ``i``'s step ``s`` sits at ``i*steps + s`` out of
        ``count*steps``, so the bar crosses the window once rather than N times.
        """
        self._step_progress(
            job_id,
            "t2i_sample",
            f"Drawing material {index + 1} of {count}",
            index * steps + step,
            count * steps,
        )

    async def _tile_sheet_materials(self: Worker, job: dict[str, Any]) -> None:
        """N seamless materials laid out in a plain grid. See :meth:`_tile_set`."""
        await self._tile_set(job, "materials")

    async def _tile_sheet_terrain(self: Worker, job: dict[str, Any]) -> None:
        """Two seamless materials composited into a blob-47 row. See
        :meth:`_tile_set`."""
        await self._tile_set(job, "terrain")

    async def _tile_set(self: Worker, job: dict[str, Any], mode: str) -> None:
        """Draw N seamless materials and publish one atlas.

        N generations rather than N jobs, because the deliverable is the *set*:
        materials dispatched minutes apart behind different work would not share
        a style, a light direction or a palette -- ``_tile_sheet``'s argument for
        one job, which does not stop being true when the sheet stops being one
        frame.

        **One ``_acquire_t2i``/``_release_t2i`` bracket around all N passes.**
        The expensive part is loading the checkpoint and the LoRA, and this pays
        it once for the whole set -- ``_sprite_synthesis``' reason for putting
        two candidates behind one load, applied to sixty-four. The bracket is
        also the only reason the loop is allowed to be this long: a stage that
        acquired per pass would hold and release ~7 GiB N times and give the
        cancel path N places to leave a pipe resident.

        **Deliberately not batched.** ``num_images_per_prompt`` stays 1 and no
        pass sees another's latent. Batching would multiply a 1024 latent's
        activation peak against a ``vram.estimate`` figure computed for exactly
        one frame, which is the OOM that only reproduces on a full card.
        """
        import numpy as np
        from PIL import Image

        from . import queue as queue_mod
        from .pipelines import pixel, seam, tileatlas, tilemask, tilesheet
        from .pipelines.conditioning import Conditioning

        job_id = job["id"]
        params = job["params"]
        # Re-derived rather than trusted, and re-derived *before* the card is
        # spent: params outlive the door that wrote them, so a block the door
        # would refuse today has to cost the request rather than N generations
        # of nothing. ``_tile_sheet``'s rule, and ``_sprite_synthesis``' before
        # it.
        block = dict(params.get("sheet") or {})
        geom, entries, seeds, subjects = _plan(block, mode)

        colors = int(params.get("colors", 64))
        palette_name = str(params.get("palette") or "")
        dither = bool(params.get("dither"))
        # Read from disk **here**, before the card is spent, and not down at the
        # quantize phase where it is used. It is one small text file, and the
        # alternative is a set that runs N full generations and then fails on a
        # palette the user deleted after submitting it -- ``_plan``'s rule three
        # lines up, for its reason: params outlive the door that wrote them.
        # ``()`` when no palette was named, which is what the common request
        # says.
        designed = await asyncio.to_thread(
            queue_mod._palette_entries, self.config, palette_name
        )
        # The request's own seed, recorded so the set is reproducible as a set.
        # The seeds that actually run are the per-material ones above --
        # ``tileatlas.material_seeds`` derives them from this one so that
        # material ``i`` is reproducible on its own from the pair ``(seed, i)``,
        # and the door stores the result rather than making the worker guess
        # which derivation it used.
        seed = int(params.get("seed", 42))
        style_lock = bool(block.get("style_lock"))
        count = len(subjects)

        base_key = self._resolve_base_key(params, default="sdxl_cfg")
        spec = models.BASE_MODELS[base_key]
        pixel_style = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
        lora: str | None = models.PIXEL_SHEET_LORA
        if not models.lora_fits(spec, pixel_style):
            # ``_tile_sheet``'s tolerance verbatim, for its reason: params
            # outlive the service that wrote them, and a set whose sidecar
            # claimed a LoRA that never loaded would be a recipe nobody can
            # reproduce. Drawn bare instead, and said so.
            #
            # A base that cannot take the LoRA is usually also a base that
            # cannot wrap at all -- ``text2image.generate`` refuses ``tile=True``
            # on anything outside the SDXL family rather than degrading it -- so
            # the job will fail a few lines below with that sentence. That
            # refusal is the right one to surface and this is not the place to
            # pre-empt it.
            log.warning(
                "base model %s (%s) cannot take the pixel-art LoRA %s (%s); "
                "drawing the materials without it",
                spec.key, spec.family, pixel_style.key, pixel_style.family,
            )
            lora = None

        job_dir = self.config.job_dir(job_id)
        await asyncio.to_thread(
            functools.partial(job_dir.mkdir, parents=True, exist_ok=True)
        )
        # Written by the door when the user attached one, and simply absent
        # otherwise -- which is the common path. Read from disk rather than from
        # params because params carry the *selection* and this is the image
        # itself, and because a reroll copies the row before it copies the file.
        ref_png = job_dir / "ref.png"
        has_reference = await asyncio.to_thread(ref_png.exists)
        ip_scale = float(params.get("ip_scale", models.DEFAULT_IP_SCALE))

        with tempfile.TemporaryDirectory(prefix="warlock-tileset-") as tmp:
            scratch = Path(tmp)
            # No guide is rendered on this path -- there is no grid to impose --
            # but the phase exists in ``PHASES_TILE_SHEET`` and skipping it
            # would start the bar at 4%. One update, so the window is crossed
            # rather than jumped.
            self.progress.update(
                job_id, phase="guide", label="Planning the materials",
                inner=0.0, inner_next=1.0, nominal=1.0, detail="",
            )
            paths = [scratch / f"material-{index:02d}.png" for index in range(count)]

            # **``None``, not an empty ``Conditioning()``.** The bit-identity
            # contract is asserted at the fake pipeline's boundary in
            # ``tests/conftest.py``: an unconditioned pass must hand the pipe
            # ``conditioning=None``. It is also what ``_needs_handoff`` reads --
            # an empty object is not None, so it would stop a warm trellis that
            # ``vram.estimate`` has already priced as co-resident, and the two
            # halves of one rule would disagree.
            first_cond = (
                Conditioning(ip_adapter="plus", ip_image=ref_png, ip_scale=ip_scale)
                if has_reference
                else None
            )
            # Style lock: the first material becomes the IP reference for every
            # one after it, so N independent samples come back as one set rather
            # than as N pictures of the right things in N different hands. Pass
            # 1 has nothing to lock onto yet and so carries only the user's own
            # reference, if there is one -- which is why this is the one pass
            # whose conditioning differs.
            later_cond = first_cond
            if style_lock and count > 1:
                later_cond = Conditioning(
                    ip_adapter="plus", ip_image=paths[0], ip_scale=ip_scale
                )
            # What ``_acquire_t2i`` is asked about is whether *this job* will put
            # an encoder beside a resident trellis, so a style-locked job says
            # yes even though its first pass is bare.
            acquire_cond = first_cond if first_cond is not None else later_cond

            t2i, _handoff = await self._acquire_t2i(spec, base_key, acquire_cond)
            composed = [guidance.compose_prompt(subject, params) for subject in subjects]
            reports: list[dict[str, Any]] = []
            try:
                for index in range(count):
                    if self._cancel is not None and self._cancel.event.is_set():
                        # Before each pass, not only after the last: every one is
                        # ~20 s of GPU a cancelled job should not spend, and
                        # nothing has been published yet, so returning here
                        # leaves ``_discard_artifacts`` with nothing to undo.
                        return
                    await asyncio.to_thread(
                        functools.partial(
                            t2i.generate,
                            composed[index],
                            paths[index],
                            seed=seeds[index],
                            lora=lora,
                            lora_weight=pixel_style.default_weight,
                            negative_prompt=str(params.get("negative_prompt") or ""),
                            conditioning=first_cond if index == 0 else later_cond,
                            # Only the first pass. ``_t2i_state`` emits
                            # "t2i_load", and a later pass re-emitting it would
                            # drag the bar back out of ``t2i_sample`` N-1 times
                            # for a checkpoint that is already resident -- the
                            # failure ``_sprite_synthesis`` avoids by passing no
                            # ``on_state`` at all.
                            on_state=(
                                (lambda s: self._t2i_state(job_id, s))
                                if index == 0
                                else None
                            ),
                            on_step=functools.partial(
                                self._material_step, job_id, index, count
                            ),
                            cancel_event=self._cancel.event if self._cancel else None,
                            # The whole seamless mechanism. Not ``tilesheet=``:
                            # that flag is the grid template and carries an
                            # explicit no-wrap rule, because a sheet's leftmost
                            # and rightmost columns are *different tiles*.
                            tile=True,
                            # One SDXL frame. Stated rather than defaulted
                            # because ``tileatlas.reduce_material`` refuses any
                            # factor that is not exact, and 1024 is the
                            # numerator that makes the tile sizes divide.
                            size=geom.source_size,
                        )
                    )
                    # Advisory, never a rejection. ``seam.SEAM_MAX`` was measured
                    # on turbo at 4 steps and ``seam.py:36-37`` says outright to
                    # re-measure it per checkpoint, so a CFG base at 30 steps is
                    # outside the corpus that produced the threshold. The number
                    # goes in the sidecar and on the row; the user decides.
                    try:
                        reports.append(await asyncio.to_thread(seam.report, paths[index]))
                    except Exception:
                        log.exception(
                            "seam measurement failed for material %d of job %s",
                            index, job_id,
                        )
                        reports.append({})
            finally:
                # Every reference dropped before the reclaim, which is why the
                # decode below happens after this and not inside the try.
                await self._release_t2i(t2i, spec)

            if self._cancel is not None and self._cancel.event.is_set():
                return

            self.progress.update(
                job_id, phase="slice", label="Reducing the materials",
                inner=0.0, inner_next=1.0, nominal=2.0, detail="",
            )
            tiles = []
            grids: list[dict[str, Any]] = []
            for index, path in enumerate(paths):
                with Image.open(path) as generated:
                    generated.load()
                    full = generated.convert("RGBA")
                # Per material, on the whole frame, before ``reduce_material``
                # resamples it: each material is its own generation and can
                # plainly land on its own lattice.
                grids.append(
                    {
                        "material": index,
                        **await asyncio.to_thread(pixel.lattice, full),
                    }
                )
                tiles.append(
                    await asyncio.to_thread(
                        tileatlas.reduce_material,
                        np.asarray(full),
                        geom.tile_w,
                        geom.tile_h,
                    )
                )

            if mode == tileatlas.MODE_TERRAIN:
                mask = _mask_block(block)
                atlas = await asyncio.to_thread(
                    functools.partial(
                        tilemask.blob_atlas,
                        tiles[0],
                        tiles[1],
                        geom.tile_w,
                        seed=int(mask["seed"]),
                        inset=mask["inset"],
                        amplitude=mask["amplitude"],
                        feather=mask["feather"],
                    )
                )
            else:
                mask = None
                atlas = await asyncio.to_thread(tileatlas.assemble, tiles, geom)

            self.progress.update(
                job_id, phase="quantize", label="Sharing one palette",
                inner=0.0, inner_next=1.0, nominal=2.0, detail="",
            )
            # One palette across every tile -- authored or derived, the property
            # is the same and ``quantize_tiles`` carries the argument: materials
            # quantized separately read as N pictures pasted together, which is
            # exactly what a tileset must not. It matters more here than on the
            # grid path, where the cells at least came out of one frame. With no
            # palette and no dither this is byte-for-byte the median cut it has
            # always been.
            reduced, palette, palette_source = await asyncio.to_thread(
                functools.partial(
                    tilesheet.quantize_tiles,
                    Image.fromarray(atlas, "RGBA"),
                    colors=colors,
                    entries=designed,
                    dither=dither,
                )
            )

            if self._cancel is not None and self._cancel.event.is_set():
                # Nothing is published for a cancelled draw. The run loop is
                # about to call ``_discard_artifacts``, and writing the atlas
                # first would leave it deleting a file it had just been told to
                # make.
                return

            # The provenance copies first: they are not served, so a failure
            # part way through leaves nothing a reader would mistake for a
            # finished set. Copied from the scratch files rather than re-encoded
            # from images held across the whole job -- sixty-four decoded 1024px
            # RGBA frames is ~256 MiB carried for the length of a generation
            # queue, and these are byte-for-byte what the model returned, which
            # is the entire point of keeping them.
            materials_dir = job_dir / MATERIALS_DIR
            await asyncio.to_thread(
                functools.partial(materials_dir.mkdir, parents=True, exist_ok=True)
            )
            for index, path in enumerate(paths):
                await asyncio.to_thread(
                    shutil.copyfile, path, materials_dir / f"{index:02d}.png"
                )

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

        seams = [
            {
                "index": index,
                "worst": float(report.get("worst", 0.0)),
                "seamless": bool(report.get("seamless", False)),
            }
            for index, report in enumerate(reports)
        ]
        recipe: dict[str, Any] = {
            "base_model": base_key,
            "seed": seed,
            # The last pass's finished prompt, as ``prompt.TILE_TEMPLATE`` built
            # it -- the same field every other recipe in the tree carries, and
            # the only one of the N that the pipe still remembers. The whole set
            # is beside it, because on this kind one prompt is not the recipe:
            # ``subjects`` is what was handed over and the sidecar's
            # ``materials`` list is what the user typed.
            "prompt": t2i.last_prompt or (composed[-1] if composed else ""),
            "subjects": list(composed),
            "colors": colors,
            "style_lock": style_lock,
            "seams": seams,
            # The worst material decides, because a set is only as seamless as
            # the tile somebody notices.
            "seam_worst": max((entry["worst"] for entry in seams), default=0.0),
            "seam_threshold": seam.SEAM_MAX,
            # Empty unless a palette was named or dither asked for, so a set
            # drawn the way every set before today was drawn writes the sidecar
            # it wrote before. ``palette_record`` holds that rule and the reason
            # ``colors`` stays beside these keys.
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
        if mode == tileatlas.MODE_TERRAIN:
            # The two halves by name, because the sidecar's ``materials`` list
            # cannot carry them: it is one record per *column* and there are
            # forty-seven columns of two materials.
            recipe["terrain"] = {
                "inner": {"prompt": entries[0]["prompt"], "seed": seeds[0]},
                "outer": {"prompt": entries[1]["prompt"], "seed": seeds[1]},
            }

        doc = tileatlas.atlas_sidecar(
            geom,
            created=time.time(),
            materials=_bind(geom, entries, seeds, mode),
            terrains=(block.get("terrains") or ()) if mode == tileatlas.MODE_TERRAIN else (),
            mask=mask,
            recipe=recipe,
            grids=grids,
        )
        # Last, and only after the atlas: this file is the completion marker, so
        # publishing it first would advertise a set still being written.
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
                    "mode": geom.mode,
                    "layout": geom.layout,
                    "tiles": geom.tiles,
                    "materials": count,
                    "palette": len(palette),
                    "tile_w": geom.tile_w,
                    "tile_h": geom.tile_h,
                    "sheet_w": geom.atlas_size[0],
                    "sheet_h": geom.atlas_size[1],
                    "projection": geom.view,
                    "seams": seams,
                    "seam_worst": recipe["seam_worst"],
                    "recipe": recipe,
                }
            },
        )
        log.info(
            "drew %s tileset %s: %d materials, %d tiles, %d colours at %dx%d",
            geom.mode, job_id, count, geom.tiles, len(palette), geom.tile_w, geom.tile_h,
        )


# --- the stored block --------------------------------------------------------
#
# Module-level and pure, so the refusals are one sentence in one place and a
# test can reach them without a worker. Everything here raises rather than
# defaults, for ``_tile_sheet``'s reason: there is exactly one writer and it
# always writes these keys, so absence is corruption and the honest answer is to
# say so before the card is spent.


def _plan(
    block: dict[str, Any], mode: str
) -> tuple[Any, list[dict[str, Any]], list[int], list[str]]:
    """``(geometry, material entries, seeds, subjects)`` from a stored block.

    The geometry is re-derived from ``tile_w``/``projection`` rather than read
    from the stored ``columns``/``rows``: those are a *record* of what the door
    laid out, and ``tileatlas`` is the single authority on what is buildable. A
    stored pair the module refuses today -- a 48px tile, which does not divide
    1024, or a 3/4 view, which cannot tile vertically -- raises here with a
    sentence naming the value.
    """
    from .pipelines import tileatlas

    try:
        tile_w = int(block["tile_w"])
        projection = str(block["projection"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("this tileset does not say what it is a set of") from None

    raw = block.get("materials")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "a tileset names the materials it is made of; this row names none"
        )
    entries = [_material_entry(entry, index) for index, entry in enumerate(raw)]

    if mode == tileatlas.MODE_TERRAIN:
        if len(entries) != 2:
            # An inner and an outer, in that order -- ``blob_rects`` makes the
            # centre cell a member always, so which is which is not a convention
            # but which of the two the forty-seven pictures are of.
            raise ValueError(
                f"a terrain set is drawn from an inner and an outer material; "
                f"this row names {len(entries)}"
            )
        geom = tileatlas.terrain_geometry(tile_w, projection)
        subjects = list(
            tileatlas.terrain_subjects(
                entries[0]["prompt"],
                entries[1]["prompt"],
                str(block.get("boundary") or ""),
            )
        )
    else:
        geom = tileatlas.material_geometry(tile_w, projection, len(entries))
        subjects = [
            tileatlas.material_subject(
                entry["prompt"], index=index, total=len(entries)
            )
            for index, entry in enumerate(entries)
        ]
    return geom, entries, [entry["seed"] for entry in entries], subjects


def _material_entry(entry: Any, index: int) -> dict[str, Any]:
    """One stored material, with its words and its seed, or a refusal."""
    if not isinstance(entry, dict):
        raise ValueError(f"material {index} is not a record; got {entry!r}")
    prompt = str(entry.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(
            f"material {index + 1} has no words; a material is described or it "
            f"is not generated"
        )
    try:
        seed = int(entry["seed"])
    except (KeyError, TypeError, ValueError):
        # Not defaulted, and not re-derived from the request seed either. The
        # door runs ``tileatlas.material_seeds`` and stores the result, so a
        # missing seed means the row was written by something else -- and a
        # worker that guessed one would publish a sidecar claiming a seed that
        # cannot be re-run.
        raise ValueError(
            f"material {index + 1} does not say which seed it was drawn from"
        ) from None
    return {"prompt": prompt, "variant": int(entry.get("variant") or 1), "seed": seed}


def _mask_block(block: dict[str, Any]) -> dict[str, Any]:
    """The terrain field's parameters, in pixels or as ``None`` for the ratio.

    ``tilemask._resolve`` turns a ``None`` into the ratio-derived default, which
    is the same *shape* of boundary at every tile size -- an absolute 6px inset
    is a coastline at 32px and a solid band at 16px. So absent is a real value
    here and not a missing one, and only the seed is required.
    """
    from .pipelines import tilemask

    raw = block.get("mask")
    if not isinstance(raw, dict):
        raise ValueError(
            "a terrain set records the mask field that drew it; without it "
            "nothing can say which noise produced these boundaries"
        )
    try:
        seed = int(raw["seed"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("a mask record names the seed its noise was drawn from") from None
    stored = raw.get("version")
    if stored is not None and int(stored) != tilemask.MASK_VERSION:
        # Warned rather than refused, the tolerance every stored-params reader
        # here applies: the field implementation moved under a row the user can
        # no longer edit, and redrawing it with today's field beats failing it.
        # The sidecar stamps the version that actually ran, so the record still
        # says which one these pixels came from.
        log.warning(
            "this terrain's mask was planned at field version %s and will be "
            "drawn at %s", stored, tilemask.MASK_VERSION,
        )
    return {
        "seed": seed,
        "inset": None if raw.get("inset") is None else float(raw["inset"]),
        "amplitude": None if raw.get("amplitude") is None else float(raw["amplitude"]),
        "feather": None if raw.get("feather") is None else float(raw["feather"]),
    }


def _bind(geom: Any, entries: list[dict[str, Any]], seeds: list[int], mode: str) -> tuple[Any, ...]:
    """The geometry's cells with the request's words on them.

    ``material_geometry`` and ``terrain_geometry`` both leave the words empty --
    the layout is arithmetic and knows nothing about the request -- and
    ``atlas_sidecar`` refuses an unbound cell, so this is the join.

    **A terrain's forty-seven columns all carry the inner material's record.**
    That is not a placeholder: each column *is* a picture of the inner terrain
    against the outer, so the inner's words and seed are the true answer to
    "what is in this cell", and the outer is named in the recipe where it can be
    stated once rather than forty-seven times.
    """
    from .pipelines import tileatlas

    if mode == tileatlas.MODE_TERRAIN:
        inner = entries[0]
        return tuple(
            dataclasses.replace(
                cell, prompt=inner["prompt"], variant=inner["variant"], seed=seeds[0]
            )
            for cell in geom.cells
        )
    return tuple(
        dataclasses.replace(
            cell, prompt=entry["prompt"], variant=entry["variant"], seed=seed
        )
        for cell, entry, seed in zip(geom.cells, entries, seeds, strict=True)
    )

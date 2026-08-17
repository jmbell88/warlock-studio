"""``Worker``'s ground-set stage: two seamless textures per terrain, composited.

The one job in this file, and its own module for ``_q_sprite``'s reason: it is a
distinct subject with a distinct publish contract, and folding it into the
restyle module would put a sixth unrelated thing in a class that was split apart
for having five.

What makes it different from every other generating kind here: **it publishes
its own ``input.png`` and that file is the finished artifact**, not a reference
something else is reconstructed from. There is no mesh stage behind it, which is
why its rows are born at ``stage="ground"`` -- a stage no promote, remesh or
editable gate accepts, so the library's buttons refuse it without knowing it
exists.

**The one cross-layer import in the queue's half of the tree.** This module
imports ``studio.plotter.groundtex`` to composite the atlas. That direction is
allowed and the reverse is not: the plotter package is pinned pure (it may not
import a pipeline, the service or the queue), and the compositor is the piece
both halves need -- duplicating it in ``pipelines/`` is how the worker's
forty-seven cases come to disagree with the editor's. ``tests/test_ground_worker.py``
pins this file's ``warlock.studio`` imports to exactly that one module, so a
second one is a failing test rather than a precedent.
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
from .pipelines import prompt as prompt_lib

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)

#: What one 1024px SDXL frame is. Every texture is generated square and full
#: size and *then* reduced to the tile, because the seam is a property of the
#: generation (circular padding over ``Conv2d``) and generating at 32px would
#: simply produce a small bad picture.
TEXTURE_PX = 1024


class GroundOps:
    """The ground-set stage, mixed into :class:`~.queue.Worker`."""

    def _ground_step(
        self: Worker, job_id: str, painted: int, textures: int, step: int, steps: int
    ) -> None:
        """``_t2i_step`` for a stage that samples N times through one pipe.

        ``_sprite_step``'s job with a different arithmetic. The sprite path has
        two generations and gives each its own phase; a ground set has up to
        sixteen and the count is the user's, so the phases cannot be written
        down in advance. What can be is the *position*: texture ``painted`` of
        ``textures``, ``step`` of ``steps`` through its own sampling, is
        ``(painted * steps + step)`` of ``(textures * steps)`` through the set.
        Reported that way, the one ``t2i_sample`` window is spent once over the
        whole job instead of once per texture.
        """
        self._step_progress(
            job_id,
            "t2i_sample",
            "Painting materials",
            painted * steps + step,
            textures * steps,
        )

    async def _ground_set(self: Worker, job: dict[str, Any]) -> None:
        """Paint a forty-seven-case terrain atlas with generated materials.

        Two generations per terrain -- a fill material and a border material --
        then a CPU tail that reduces each to exactly the tile size, composites
        the forty-seven cases, and quantizes the whole sheet to one palette.

        One job rather than one per texture, because the deliverable is the
        *sheet*: sixteen rows that could be dispatched minutes apart behind
        different work would not be a tileset, and the expensive part (loading
        the checkpoint and the LoRA) is paid once for all of them.
        """
        import numpy as np
        from PIL import Image

        from . import queue as queue_mod
        from .pipelines import ground
        from .pipelines.pixelsheet import quantize_shared
        from .studio.plotter import groundtex

        job_id = job["id"]
        params = job["params"]
        # Re-derived rather than trusted: params outlive the door that wrote
        # them, and this raises on a block that no longer validates -- the rule
        # ``_sprite_synthesis`` follows about ``spritesynth.geometry``.
        block = dict(params.get("ground") or {})
        rows = [dict(entry) for entry in block.get("terrains") or []]
        if not rows:
            raise ValueError("this ground set names no terrains")
        # The ceiling read from the compositor rather than restated. A stored
        # block naming fifty terrains is a hundred SDXL generations, and the
        # door that refused it is minutes behind us. ``groundtex.GroundSpec``
        # is where this rule and the border rule below actually live -- it is
        # not constructed here because it wants ``TerrainSpec`` objects and
        # these rows are the params' own dicts, so inventing an outline colour
        # to satisfy a validator would be worse than reading its two numbers.
        if len(rows) > groundtex.MAX_TERRAINS:
            raise ValueError(
                f"this ground set names {len(rows)} terrains; "
                f"at most {groundtex.MAX_TERRAINS} can be painted"
            )
        # The geometry is *required*, not defaulted. It was defaulted to 32, and
        # that is the shape of a silent failure rather than a tolerance: the
        # adoption step reads the same two keys with a default of 0 and refuses
        # a mismatch against the map, so a block that had lost them would paint
        # a 32px set that no map could ever take -- twenty minutes of card time
        # spent on a file with no destination. There is exactly one writer
        # (``service.grounds.create_ground_set``) and it always writes them, so
        # absence is corruption and the honest answer is to say so now.
        try:
            tile_w = int(block["tile_w"])
            tile_h = int(block["tile_h"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "this ground set does not say what size its tiles are"
            ) from None
        if tile_w < 1 or tile_h < 1:
            raise ValueError("a ground set's tiles are at least one pixel across")
        projection = str(block.get("projection") or "orthogonal")
        stored_band = int(block.get("border") or 0)
        # Re-checked here and not merely at the door, which is what the comment
        # above claims for the whole block and used to be true of the terrain
        # list alone. This is the one stored value whose violation is *silent*:
        # a band of half the tile or more leaves no fill visible, so all
        # forty-seven cases composite to the same square and the job finishes
        # looking like a success. ``create_ground_set`` refuses it with a
        # sentence; the worker cannot raise that sentence (it may not import the
        # service) but it can decline to spend the card on it.
        if stored_band and 2 * stored_band >= min(tile_w, tile_h):
            raise ValueError(
                f"a {tile_w}x{tile_h} tile cannot take a {stored_band}px border: "
                "no fill would be left to tell the forty-seven cases apart"
            )
        border = stored_band or groundtex.default_border(tile_w, tile_h)
        # Default 1, so a job stored before phases existed still runs; anything
        # else the door computed, and a value outside the table is corruption.
        phases = int(block.get("phases") or 1)
        if phases not in (1, 2, 4):
            raise ValueError("a ground set's phase count per axis is 1, 2 or 4")
        if phases * max(tile_w, tile_h) > TEXTURE_PX:
            raise ValueError(
                f"{phases} phases of a {tile_w}x{tile_h} tile run past the "
                f"{TEXTURE_PX}px generation"
            )
        colors = int(params.get("colors", 32))
        seed = int(params.get("seed", 42))
        theme = job["prompt"] or ""

        base_key = self._resolve_base_key(params, default="sdxl_cfg")
        spec = models.BASE_MODELS[base_key]
        pixel_style = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
        lora: str | None = models.PIXEL_SHEET_LORA
        if not models.lora_fits(spec, pixel_style):
            # The same tolerance ``_pixel_sheet`` applies, and needed here for
            # its reason: params outlive the service that wrote them, and a set
            # whose sidecar claimed a LoRA that never loaded would be a recipe
            # nobody can reproduce. Painted bare instead, and said so.
            log.warning(
                "base model %s (%s) cannot take the pixel-art LoRA %s (%s); "
                "painting the ground set without it",
                spec.key, spec.family, pixel_style.key, pixel_style.family,
            )
            lora = None

        job_dir = self.config.job_dir(job_id)
        await asyncio.to_thread(
            functools.partial(job_dir.mkdir, parents=True, exist_ok=True)
        )

        # ``cond=None`` explicitly, and this is the one caller that means it: a
        # seamless texture is conditioned on nothing but its prompt, so the
        # conditioning-beside-a-running-trellis term of ``_needs_handoff`` must
        # not fire. Taking the default sentinel here would stop a warm trellis
        # for every ground set -- ~20 s of ``stop()`` plus a full cold reload on
        # the next mesh job -- while ``vram.estimate`` prices this kind as
        # co-resident (``vram.py`` "Never trellis"). The accounting and the
        # admission would then disagree with what the worker actually does.
        t2i, _handoff = await self._acquire_t2i(spec, base_key, None)
        total = len(rows) * ground.TEXTURES_PER_TERRAIN
        fills: list[Any] = []
        borders: list[Any] = []
        # ``(file name, role, decoded 1024px image)`` in generation order, which
        # is fill-then-border per terrain. Kept whole rather than reduced inside
        # the loop so nothing but ``t2i.generate`` runs while the pipe is
        # resident -- the CPU tail is minutes of card time if it does not.
        full: list[tuple[str, str, Any]] = []
        # Every composed prompt, in generation order, for the recipe below. The
        # recipe used to record ``t2i.last_prompt`` under a singular ``prompt``
        # key -- which is the *border* subject of the last terrain, sitting
        # beside ``"textures": 16`` and reading as though it described all
        # sixteen. A set is N different prompts by construction; one of them is
        # not a recipe for it.
        composed_prompts: list[str] = []
        try:
            with tempfile.TemporaryDirectory(prefix="warlock-ground-") as tmp:
                scratch = Path(tmp)
                step = 0
                for index, row in enumerate(rows):
                    subjects = ground.texture_prompts(
                        theme,
                        str(row.get("name") or ""),
                        str(row.get("material") or ""),
                        str(row.get("edge") or ""),
                        tuple(row.get("color") or (128, 128, 128, 255)),
                    )
                    for role, subject in zip(ground.ROLES, subjects, strict=True):
                        if self._cancel is not None and self._cancel.event.is_set():
                            # Between textures, not only at the end: each one is
                            # ~20 s of GPU a cancelled job should not spend, and
                            # nothing has been published, so returning here
                            # leaves the run loop's ``_discard_artifacts`` with
                            # nothing to undo.
                            return
                        self.progress.update(
                            job_id, phase="paint", label="Painting materials",
                            inner=step / total, inner_next=(step + 1) / total,
                            nominal=20.0,
                            detail=f"{row.get('name') or index + 1} {role}",
                        )
                        out_path = scratch / f"t{index}-{role}.png"
                        composed = guidance.compose_prompt(
                            subject, params, fields=prompt_lib.TILE_FIELDS
                        )
                        composed_prompts.append(composed)
                        await asyncio.to_thread(
                            functools.partial(
                                t2i.generate,
                                composed,
                                out_path,
                                # Derived, never stored: one seed reproduces the
                                # whole sheet, and two adjacent textures still
                                # come back unrelated.
                                seed=ground.texture_seed(seed, index, role),
                                lora=lora,
                                lora_weight=pixel_style.default_weight,
                                negative_prompt=str(params.get("negative_prompt") or ""),
                                # Only the first texture's state reaches the
                                # bar. The pipe is loaded once and stays
                                # resident for the whole set, so every later
                                # call re-announcing "Preparing" would relabel a
                                # bar that is three quarters through with the
                                # caption of a job that has not started.
                                on_state=(
                                    (lambda s: self._t2i_state(job_id, s))
                                    if step == 0
                                    else None
                                ),
                                # Global steps, not this texture's. Each call
                                # reports i-of-n for its own generation, and
                                # feeding that straight to ``_t2i_step`` walked
                                # the whole ``t2i_sample`` window (0.14..0.88)
                                # once *per texture* -- so texture one finished
                                # at 88% and the never-regress creep held it
                                # there for the remaining fifteen sixteenths of
                                # the job. Offsetting by the textures already
                                # painted spends the window once across the set,
                                # which is what the window is for. The per-pass
                                # phase tables ``_sprite_synthesis`` uses are the
                                # other answer to this; they do not scale to a
                                # count the form chooses.
                                on_step=functools.partial(
                                    self._ground_step, job_id, step, total
                                ),
                                cancel_event=self._cancel.event if self._cancel else None,
                                # The whole feature: circular padding over the
                                # convolutions, so the texture is seamless on
                                # its own torus and the reduction below keeps it
                                # seamless on the tile's.
                                tile=True,
                            )
                        )
                        with Image.open(out_path) as opened:
                            opened.load()
                            full.append(
                                (f"t{index}-{role}.png", role, opened.convert("RGBA"))
                            )
                        step += 1
        finally:
            # Every reference dropped before the reclaim, which is why the
            # decoded textures above are host-side PIL images and nothing here
            # holds a tensor.
            await self._release_t2i(t2i, spec)

        if self._cancel is not None and self._cancel.event.is_set():
            return

        self.progress.update(
            job_id, phase="compose", label="Composing the tileset",
            inner=0.0, inner_next=1.0, nominal=3.0, detail="",
        )
        # The period is ``phases`` tiles: reduced to k * tile and sliced into
        # k**2 tile-sized phase variants by the compositor.
        for _name, role, image in full:
            small = await asyncio.to_thread(
                ground.reduce_texture,
                np.asarray(image),
                tile_w * phases,
                tile_h * phases,
            )
            (fills if role == ground.FILL else borders).append(small)
        atlas = await asyncio.to_thread(
            groundtex.compose_atlas,
            tile_w,
            tile_h,
            projection,
            border,
            fills,
            borders,
            phases,
        )

        self.progress.update(
            job_id, phase="quantize", label="Sharing one palette",
            inner=0.0, inner_next=1.0, nominal=3.0, detail="",
        )
        # One palette across every terrain, for ``quantize_shared``'s reason
        # applied to a different axis: N terrains quantized separately read as N
        # tilesets pasted together, which is exactly what a ground set must not.
        reduced, palette = await asyncio.to_thread(
            quantize_shared, Image.fromarray(atlas, "RGBA"), colors
        )

        if self._cancel is not None and self._cancel.event.is_set():
            # Nothing is published for a cancelled paint. The run loop is about
            # to call ``_discard_artifacts``, and writing the atlas first would
            # leave it deleting a file it had just been told to make.
            return

        # The provenance copies first: they are not served, so a failure part
        # way through leaves nothing a reader would mistake for a finished set.
        textures_dir = job_dir / "textures"
        await asyncio.to_thread(
            functools.partial(textures_dir.mkdir, parents=True, exist_ok=True)
        )
        for name, _role, image in full:
            await asyncio.to_thread(image.save, textures_dir / name, "PNG")

        out_png = job_dir / "input.png"
        # Staged and renamed rather than saved in place: ``input.png`` is a
        # served name, and a second paint of the same row (or a library reader
        # mid-adoption) must never see a partial file under it.
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
            # Plural, and all of them: see ``composed_prompts`` above. In
            # generation order, which is fill-then-border per terrain, so the
            # nth pair belongs to ``terrains[n]``.
            "prompts": composed_prompts,
            "textures": total,
        }
        if lora is not None:
            recipe["style_lora"] = lora
            recipe["lora_weight"] = pixel_style.default_weight
        doc = ground.ground_sidecar(
            theme=theme,
            tile_w=tile_w,
            tile_h=tile_h,
            projection=projection,
            border=border,
            colors=colors,
            phases=phases,
            terrains=rows,
            palette=palette,
            recipe=recipe,
            created=time.time(),
        )
        # Last, and only after the atlas: this file is the completion marker, so
        # publishing it first would advertise a sheet still being written.
        await asyncio.to_thread(
            queue_mod._publish_text, job_dir / "ground.json", json.dumps(doc, indent=2)
        )
        # On the row as well, so the library can say what it painted without
        # opening a file. Derived, so DERIVED_PARAMS carries it.
        await asyncio.to_thread(
            self.store.merge_params,
            job_id,
            {
                "ground_report": {
                    "textures": total,
                    "palette": len(palette),
                    "border": border,
                    "phases": phases,
                    "tile_w": tile_w,
                    "tile_h": tile_h,
                    "recipe": recipe,
                }
            },
        )
        log.info(
            "painted ground set %s: %d terrains, %d textures, %d colours at %dx%d",
            job_id, len(rows), total, len(palette), tile_w, tile_h,
        )

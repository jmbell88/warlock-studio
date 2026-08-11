"""``Worker``'s restyle stages: pixel sheets, sprite drafts, re-texture.

The three jobs that take something already finished and give it a new surface.
Split out of ``queue.py`` for the reason every mixin here is: the class had 35
methods and five unrelated subjects, and the loop core -- which is the part
with the invariants -- was buried among them.

What these share, and why they are one module rather than three: every one of
them is an **img2img pass on the resident SDXL pipe**, so every one of them
makes the VRAM handoff (``Worker._acquire_t2i``) and gives the checkpoint back
in a ``finally`` (``Worker._release_t2i``) -- and every one of them publishes
into the *source* job's directory, because a restyle depicts that asset and
belongs beside it. That is also why they are queue jobs at all rather than
derivations: a TaskRunner thread racing the worker for VRAM is the OOM that
only reproduces under load.

Queue.py module-level names are reached through ``queue_mod`` rather than
imported: a call-time lookup keeps ``monkeypatch.setattr(queue_mod, ...)``
landing, and a module-scope import of ``.queue`` here would be circular.
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

from . import guidance, models, rigging
from .pipelines import control

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class SpriteOps:
    """The restyle stages, mixed into :class:`~.queue.Worker`."""

    async def _pixel_sheet(self: Worker, job: dict[str, Any]) -> None:
        """Restyle a finished sprite sheet into pixel art, one band at a time.

        A queue job rather than a derivation, for the reason every other GPU
        path here is one: it needs the resident SDXL pipe, and a TaskRunner
        thread racing the worker for VRAM is the OOM that only reproduces under
        load. And a job of its own rather than a stage of the sheet render: the
        EEVEE render is slow and deterministic, while the restyle is the part a
        user iterates on -- strength, palette, seed -- so paying for the render
        again on every attempt would be the whole cost for none of the change.

        The output belongs to the render, so it lands beside it in the *source*
        job's directory, and the JSON is written last as the completion marker
        -- the same rule ``_sheet`` and ``_rig`` follow.
        """
        from PIL import Image

        from . import queue as queue_mod
        from .pipelines import pixelsheet
        from .pipelines.conditioning import Conditioning

        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        sheet_id = str(params.get("sheet_id") or "")
        if not rigging.is_valid_id(sheet_id):
            raise ValueError(f"sheet_id is not a sheet id: {sheet_id!r}")
        source_dir = self.config.job_dir(source_id)

        meta = await asyncio.to_thread(rigging.read_sheet, source_dir, sheet_id)
        if meta is None:
            # Deleted between queueing and running -- the same failure `poses`
            # gets in _sheet, and for the same reason: a restyle of a sheet
            # that is gone would depict nothing.
            raise RuntimeError(f"sheet {sheet_id} no longer exists")
        png = rigging.sheet_png_path(source_dir, sheet_id)
        if not png.exists():
            raise RuntimeError(f"sheet {sheet_id} has no rendered atlas")

        logical = int(params.get("logical_size", 32))
        colors = int(params.get("colors", 32))
        strength = float(params.get("strength", models.DEFAULT_IMG2IMG_STRENGTH))
        structure = bool(params.get("structure_lock", True))
        seed = int(params.get("seed", 42))
        # Re-checked here as well as at the door: params outlive the sheet they
        # name, and a re-rendered sheet is a different grid.
        pixelsheet.check_restylable(meta, logical)

        base_key = self._resolve_base_key(params, default="sdxl_cfg")
        spec = models.BASE_MODELS[base_key]
        pixel_style = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
        lora: str | None = models.PIXEL_SHEET_LORA
        if not models.lora_fits(spec, pixel_style):
            # The same tolerance _generate applies to a stored style the base
            # cannot take, and needed here for the same reason: params outlive
            # the service that wrote them, and create_pixel_sheet's refusal
            # guards only the door. Without this, a klein base ran the restyle
            # bare while the sidecar below claimed the pixel LoRA styled it.
            log.warning(
                "base model %s (%s) cannot take the pixel-sheet LoRA %s (%s); "
                "restyling without it",
                spec.key, spec.family, pixel_style.key, pixel_style.family,
            )
            lora = None
        # The same handoff a text job makes, through the same preamble.
        t2i, _handoff = await self._acquire_t2i(spec, base_key)

        with Image.open(png) as opened:
            opened.load()
            atlas = opened.convert("RGBA")

        prompt = guidance.compose_prompt(job["prompt"] or "", params)
        plan = pixelsheet.bands(meta)
        styled = Image.new("RGBA", atlas.size, (0, 0, 0, 0))
        try:
            for band in plan:
                self.progress.update(
                    job_id, phase="restyle", label="Restyling sheet",
                    inner=band.index / len(plan),
                    inner_next=(band.index + 1) / len(plan),
                    nominal=30.0, detail=f"band {band.index + 1}/{len(plan)}",
                )
                with tempfile.TemporaryDirectory(prefix="warlock-pixel-") as tmp:
                    scratch = Path(tmp)
                    init_path = scratch / "init.png"
                    out_path = scratch / "out.png"
                    await asyncio.to_thread(
                        pixelsheet.band_init(atlas, band).save, init_path, "PNG"
                    )
                    control_path = None
                    if structure and spec.controlnet:
                        # Canny off the flat render, which is clean unlit
                        # albedo -- so the hint is each cell's real silhouette
                        # rather than an edge map of its lighting.
                        control_path = scratch / "hint.png"
                        await asyncio.to_thread(
                            functools.partial(
                                control.write_hint,
                                init_path,
                                control_path,
                                kind="canny",
                                size=pixelsheet.BAND_PX,
                            )
                        )
                    cond = Conditioning(
                        init_image=init_path,
                        strength=strength,
                        control="canny" if control_path else None,
                        control_image=control_path,
                    )
                    await asyncio.to_thread(
                        functools.partial(
                            t2i.generate,
                            prompt,
                            out_path,
                            # One seed for every band, so a multi-band sheet
                            # keeps one identity all the way down the atlas.
                            seed=seed,
                            lora=lora,
                            lora_weight=pixel_style.default_weight,
                            conditioning=cond,
                            on_state=lambda s: self._t2i_state(job_id, s),
                            on_step=lambda i, n: self._t2i_step(job_id, i, n),
                            cancel_event=self._cancel.event if self._cancel else None,
                            sheet=True,
                        )
                    )
                    with Image.open(out_path) as generated:
                        generated.load()
                        piece = pixelsheet.crop_back(generated, band)
                top = band.first_row * band.frame
                source_band = atlas.crop((0, top, band.width, top + band.height))
                styled.paste(pixelsheet.remask(piece, source_band), (0, top))
        finally:
            await self._release_t2i(t2i, spec)

        self.progress.update(
            job_id, phase="quantize", label="Reducing to pixels",
            inner=0.0, inner_next=1.0, nominal=3.0, detail="",
        )
        factor = int(meta["frame_size"]) // logical
        small = await asyncio.to_thread(pixelsheet.downscale, styled, factor)
        reduced, palette = await asyncio.to_thread(
            pixelsheet.quantize_shared, small, colors
        )

        if self._cancel is not None and self._cancel.event.is_set():
            # Nothing is published for a cancelled restyle. The run loop is
            # about to call _discard_artifacts, and writing the pair first
            # would leave it deleting files it had just been told to make --
            # the same reason _rig skips finalize_rig on a cancel.
            return

        out_png = rigging.sheet_pixel_png_path(source_dir, sheet_id)
        # Staged and renamed, not saved in place. ``sheets.py`` serves this
        # name as soon as a *previous* restyle's sidecar exists, and restyling
        # the same sheet again is allowed -- so a bare save tears a file that
        # is already being served, with the old sidecar still calling it ready.
        out_tmp = out_png.with_name(f".{out_png.name}.tmp")
        try:
            await asyncio.to_thread(reduced.save, out_tmp, "PNG")
            await asyncio.to_thread(os.replace, out_tmp, out_png)
        finally:
            with contextlib.suppress(OSError):
                out_tmp.unlink(missing_ok=True)
        # The recipe records what ran, not what was asked for -- the rule
        # guidance.normalize applies to a stored lora_weight. A style dropped
        # above is simply absent, exactly as structure_lock already reads
        # False when the base has no ControlNet to lock with.
        recipe: dict[str, Any] = {
            "base_model": base_key,
            "strength": strength,
            "structure_lock": bool(structure and spec.controlnet),
            "seed": seed,
            "colors": colors,
            "prompt": t2i.last_prompt or prompt,
            "bands": len(plan),
        }
        if lora is not None:
            recipe["style_lora"] = lora
        doc = pixelsheet.pixel_sidecar(
            meta,
            image=out_png.name,
            logical_size=logical,
            palette=palette,
            recipe=recipe,
            created=time.time(),
        )
        # Last, and only after the PNG: this file is what the service treats as
        # the completion marker, so publishing it first would advertise an
        # atlas that is still being written.
        await asyncio.to_thread(
            queue_mod._publish_text,
            rigging.sheet_pixel_path(source_dir, sheet_id),
            json.dumps(doc, indent=2),
        )
        log.info(
            "restyled sheet %s for job %s: %d bands, %d colours at %dpx",
            sheet_id, source_id, len(plan), len(palette), logical,
        )

    async def _sprite_synthesis(self: Worker, job: dict[str, Any]) -> None:
        """Two candidate sprite atlases from one finished 2D reference.

        A queue job for ``_pixel_sheet``'s reason exactly -- it needs the
        resident SDXL pipe, and a TaskRunner thread racing the worker for VRAM
        is the OOM that only reproduces under load -- and it writes into the
        *source* job's directory for ``_sheet``'s reason: a draft depicts that
        reference and belongs beside it.

        Two candidates in one job rather than two jobs, because the expensive
        part (loading the checkpoint, both adapters and the LoRA) is paid once
        and the pair is the deliverable: a user comparing two guesses at three
        views they have never seen is the entire point, and two rows that could
        be dispatched minutes apart behind different work would not be a pair.

        Nothing is written until both candidates are assembled. A draft is a
        trio -- two PNGs then the sidecar last, which is the completion marker
        -- and a half-published draft is one the pane would list and the user
        would open.
        """
        from PIL import Image

        from . import queue as queue_mod
        from .pipelines import spritesynth
        from .pipelines.conditioning import Conditioning

        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        draft_id = str(params.get("draft_id") or "")
        if not rigging.is_valid_id(draft_id):
            raise ValueError(f"draft_id is not a draft id: {draft_id!r}")
        sheet_type = str(params.get("sheet_type") or "")
        # Re-derived rather than trusted: this raises on an unknown type, and
        # params outlive the door that validated them.
        geom = spritesynth.geometry(sheet_type)

        source_dir = self.config.job_dir(source_id)
        src_png = source_dir / "input.png"
        if not src_png.exists():
            raise RuntimeError(f"reference {source_id} no longer has an image")

        logical = int(params.get("logical_size", 64))
        colors = int(params.get("colors", 32))
        seeds = (("a", int(params.get("seed_a", 42))), ("b", int(params.get("seed_b", 43))))

        base_key = self._resolve_base_key(params, default="sdxl_cfg")
        spec = models.BASE_MODELS[base_key]
        pixel_style = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
        lora: str | None = models.PIXEL_SHEET_LORA
        if not models.lora_fits(spec, pixel_style):
            # The same tolerance _pixel_sheet applies, for its reason: params
            # outlive the service that wrote them, and a draft whose recipe
            # claimed a LoRA that never ran would be evidence about a style
            # nobody generated.
            log.warning(
                "base model %s (%s) cannot take the pixel-sheet LoRA %s (%s); "
                "generating without it",
                spec.key, spec.family, pixel_style.key, pixel_style.family,
            )
            lora = None

        with tempfile.TemporaryDirectory(prefix="warlock-sprite-") as tmp:
            scratch = Path(tmp)
            self.progress.update(
                job_id, phase="condition", label="Reading the reference",
                inner=0.0, inner_next=1.0, nominal=12.0, detail="",
            )
            # The heavy matte, once, on the source only -- ~11.5 s an image, so
            # sixteen of them is a different feature. Every generated cell gets
            # the cheap corner fill inside spritesynth instead, which is what
            # the "plain background" clause in the prompt is there to earn.
            source_rgba, source_report, matte_source = await asyncio.to_thread(
                queue_mod._sprite_source, src_png, self.config
            )
            ip_path = scratch / "ip.png"
            ip_image = await asyncio.to_thread(queue_mod._sprite_ip_image, source_rgba)
            await asyncio.to_thread(ip_image.save, ip_path, "PNG")
            template = await asyncio.to_thread(
                spritesynth.load_guide_template, sheet_type
            )
            guide_path = scratch / "guide.png"
            guide = await asyncio.to_thread(spritesynth.render_guide, geom, template)
            await asyncio.to_thread(guide.save, guide_path, "PNG")
            guide_edges = await asyncio.to_thread(control.edge_fraction, guide)

            t2i, _handoff = await self._acquire_t2i(spec, base_key)

            prompt = guidance.compose_prompt(job["prompt"] or "", params)
            ip_scale = float(params.get("ip_scale", models.DEFAULT_IP_SCALE))
            control_scale = float(
                params.get("control_scale", models.DEFAULT_CONTROL_SCALE)
            )
            control_end = float(params.get("control_end", models.DEFAULT_CONTROL_END))

            assembled: list[tuple[str, Any, dict[str, Any]]] = []
            try:
                for letter, seed in seeds:
                    if self._cancel is not None and self._cancel.event.is_set():
                        # Before B, not only at the end: the first generation is
                        # ~20 s of GPU that a cancelled job should not spend.
                        return
                    phase = f"generate_{letter}"
                    self.progress.update(
                        job_id, phase=phase, label="Drawing the sheet",
                        inner=0.0, inner_next=0.05, nominal=20.0, detail="",
                    )
                    out_path = scratch / f"{letter}.png"
                    cond = Conditioning(
                        # Both adapters at once: the IP-Adapter carries who the
                        # character is and the ControlNet carries where the
                        # limbs go. Deliberately no init image -- an img2img
                        # start from the reference would fight the pose guides
                        # in fifteen of the sixteen cells, and there is no
                        # strength axis on this kind for exactly that reason.
                        ip_adapter="plus",
                        ip_image=ip_path,
                        ip_scale=ip_scale,
                        control="canny",
                        control_image=guide_path,
                        control_scale=control_scale,
                        control_end=control_end,
                    )
                    await asyncio.to_thread(
                        functools.partial(
                            t2i.generate,
                            prompt,
                            out_path,
                            seed=seed,
                            lora=lora,
                            lora_weight=pixel_style.default_weight,
                            conditioning=cond,
                            # Deliberately not self._t2i_state: it emits
                            # "t2i_load"/"t2i_sample", which are not in
                            # PHASES_SPRITE, and update() falls back to
                            # (0.0, 1.0) for an unknown phase -- which would
                            # drag the bar back to zero twice per job.
                            on_step=functools.partial(
                                self._sprite_step, job_id, phase
                            ),
                            cancel_event=self._cancel.event if self._cancel else None,
                            sheet=True,
                        )
                    )
                    self.progress.update(
                        job_id, phase=f"assemble_{letter}", label="Cutting the sheet up",
                        inner=0.0, inner_next=1.0, nominal=6.0, detail="",
                    )
                    with Image.open(out_path) as generated:
                        generated.load()
                        atlas = generated.convert("RGB")
                    reduced, record = await asyncio.to_thread(
                        queue_mod._sprite_assemble,
                        atlas,
                        geom,
                        logical,
                        colors,
                        source_rgba,
                        source_report,
                    )
                    record["image"] = f"{draft_id}.{letter}.png"
                    record["seed"] = seed
                    assembled.append((letter, reduced, record))
            finally:
                await self._release_t2i(t2i, spec)

            if self._cancel is not None and self._cancel.event.is_set():
                # Nothing is published for a cancelled synthesis, exactly as
                # _pixel_sheet says: the run loop is about to call
                # _discard_artifacts, and writing the trio first would leave it
                # deleting files it had just been told to make.
                return

            recipe: dict[str, Any] = {
                "base_model": base_key,
                "prompt": t2i.last_prompt or prompt,
                "ip_adapter": "plus",
                "ip_scale": ip_scale,
                "control": "canny",
                "control_scale": control_scale,
                "control_end": control_end,
                "guide_template": sheet_type,
                "guide_edge_fraction": guide_edges,
                "matte_source": matte_source,
                "colors": colors,
            }
            if lora is not None:
                recipe["style_lora"] = lora
                recipe["lora_weight"] = pixel_style.default_weight

            sprite_dir = rigging.sprite_dir(source_dir)
            await asyncio.to_thread(
                functools.partial(sprite_dir.mkdir, parents=True, exist_ok=True)
            )
            for letter, reduced, _ in assembled:
                await asyncio.to_thread(
                    reduced.save,
                    rigging.sprite_draft_png_path(source_dir, draft_id, letter),
                    "PNG",
                )
            doc = spritesynth.draft_sidecar(
                draft_id=draft_id,
                source_job=source_id,
                created=time.time(),
                geom=geom,
                logical_size=logical,
                colors=colors,
                candidates=[record for _, _, record in assembled],
                recipe=recipe,
            )
            # Last, and only after both PNGs: this file is what
            # rigging.list_sprite_drafts treats as the completion marker.
            await asyncio.to_thread(
                queue_mod._publish_text,
                rigging.sprite_draft_path(source_dir, draft_id),
                json.dumps(doc, indent=2),
            )

        log.info(
            "synthesised sprite draft %s for job %s: %s, %d cells at %dpx",
            draft_id, source_id, sheet_type, len(geom.cells), logical,
        )

    def _sprite_step(self: Worker, job_id: str, phase: str, step: int, total: int) -> None:
        """``_t2i_step`` for a phase table that has no ``t2i_sample``."""
        self._step_progress(job_id, phase, "Drawing the sheet", step, total)

    async def _retexture(self: Worker, job: dict[str, Any]) -> None:
        """Give a finished mesh a new skin: render, restyle, project, swap.

        Four stages and only the middle one is GPU-generation: ``op_views``
        renders the mesh flat from ``retexture.VIEWS``, one SDXL img2img pass
        restyles each render, ``op_project`` bakes each restyled view back into
        the mesh's own UV atlas with a weight image beside it, and
        ``retexture.assemble`` does the weighted mean on the host. Two Blender
        ops rather than one because the restyle sits between them and Blender
        is a separate interpreter with no way to call back into this one.

        A queue job for ``_pixel_sheet``'s reason: it needs the resident SDXL
        pipe, and a TaskRunner thread racing the worker for VRAM is the OOM
        that only reproduces under load.

        **The new mesh belongs to the old one**, so like a rig it is published
        into the *source* job's directory rather than into this job's -- and by
        the same mechanism, a temp name renamed into place, so a concurrent
        reader of ``model.glb`` never sees a partial file. ``source.glb`` is
        untouched: a re-texture is derivation, never authorship, and the
        reconstruction has to stay the thing a retarget rebuilds from.
        """
        from .pipelines import retexture
        from .pipelines.conditioning import Conditioning

        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        source_dir = self.config.job_dir(source_id)
        model_glb = source_dir / "model.glb"
        if not model_glb.exists():
            raise RuntimeError("source job has no mesh to re-texture")

        views = list(retexture.VIEWS)
        # Absent means "match the mesh", which is the default and is measured:
        # a fixed size changed nothing a view covered and halved the resolution
        # of everything no view did.
        asked = params.get("texture_size")
        texture_size = (
            int(asked)
            if asked
            else (
                await asyncio.to_thread(retexture.atlas_size, model_glb)
                or retexture.TEXTURE_PX
            )
        )
        strength = float(params.get("strength", models.DEFAULT_IMG2IMG_STRENGTH))
        seed = int(params.get("seed", 42))
        base_key = self._resolve_base_key(params)
        spec = models.BASE_MODELS[base_key]
        view_px = spec.image_size

        job_dir = self.config.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        # This job's own directory, not a TemporaryDirectory: six renders, six
        # restyles and twelve bakes are what a user asks about when the result
        # is wrong, and they are exactly what ``_discard_artifacts`` removes on
        # a cancel. Nothing here is in MEDIA or LISTED, so none of it is served.
        views_dir = job_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        self.progress.update(
            job_id, phase="views", label="Rendering views", inner=0.0,
            inner_next=0.2, nominal=20.0, detail=f"{len(views)} directions",
        )
        await asyncio.to_thread(
            functools.partial(
                rigging.run_worker,
                rigging.views_spec(model_glb, views_dir, views, size=view_px),
                on_progress=lambda f, label: self.progress.update(
                    job_id, phase="views", label=label, inner=f * 0.2,
                    inner_next=min(f * 0.2 + 0.03, 0.2), nominal=20.0, detail="",
                ),
                on_start=self._note_blender,
                # A render of six frames, which is the sentence sheet_timeout
                # already describes -- a second knob would be two ceilings on
                # one kind of Blender run.
                timeout=self.config.sheet_timeout,
            )
        )

        prompt = guidance.compose_prompt(job["prompt"] or "", params)
        # Same preamble as the text branch and the pixel-sheet restyle.
        t2i, _handoff = await self._acquire_t2i(spec, base_key)
        assert self._cancel is not None
        try:
            for index in range(len(views)):
                init_path = views_dir / f"view_{index:02d}.png"
                if not init_path.exists():
                    continue
                self.progress.update(
                    job_id, phase="restyle", label="Restyling views",
                    inner=0.2 + 0.55 * index / len(views),
                    inner_next=0.2 + 0.55 * (index + 1) / len(views),
                    nominal=25.0, detail=f"view {index + 1}/{len(views)}",
                )
                await asyncio.to_thread(
                    functools.partial(
                        t2i.generate,
                        prompt,
                        views_dir / f"restyled_{index:02d}.png",
                        # One seed for every view. Six independent seeds is six
                        # unrelated interpretations of one prompt, and the
                        # weighted mean of those is mud exactly where two views
                        # overlap -- which is most of the mesh.
                        seed=seed,
                        conditioning=Conditioning(
                            init_image=init_path, strength=strength
                        ),
                        on_state=lambda s: self._t2i_state(job_id, s),
                        on_step=lambda i, n: self._t2i_step(job_id, i, n),
                        cancel_event=self._cancel.event if self._cancel else None,
                    )
                )
        finally:
            await self._release_t2i(t2i, spec)

        if self._cancel is not None and self._cancel.event.is_set():
            return

        self.progress.update(
            job_id, phase="project", label="Baking projections", inner=0.75,
            inner_next=0.95, nominal=30.0, detail="",
        )
        await asyncio.to_thread(
            functools.partial(
                rigging.run_worker,
                rigging.project_spec(
                    model_glb, views_dir, views_dir, views,
                    size=view_px, texture_size=texture_size,
                ),
                on_progress=lambda f, label: self.progress.update(
                    job_id, phase="project", label=label, inner=0.75 + f * 0.2,
                    inner_next=min(0.75 + f * 0.2 + 0.03, 0.95), nominal=30.0,
                    detail="",
                ),
                on_start=self._note_blender,
                timeout=self.config.sheet_timeout,
            )
        )

        self.progress.update(
            job_id, phase="assemble", label="Combining projections", inner=0.95,
            inner_next=1.0, nominal=5.0, detail="",
        )
        base_png = views_dir / "base.png"
        # The mesh's existing albedo, at the bake's resolution: a texel no view
        # could see keeps its old colour, and that is only expressible if the
        # old colours are addressable at the size the projections were baked at.
        has_base = await asyncio.to_thread(
            functools.partial(
                retexture.extract_base_colour, model_glb, base_png, size=texture_size
            )
        )
        atlas = views_dir / "atlas.png"
        report = await asyncio.to_thread(
            functools.partial(
                retexture.assemble,
                views_dir,
                base_png if has_base else None,
                atlas,
                count=len(views),
            )
        )
        if report is None:
            raise RuntimeError("not every projection could be read back from Blender")

        if self._cancel is not None and self._cancel.event.is_set():
            # Nothing is published for a cancelled re-texture -- the run loop is
            # about to call _discard_artifacts, and the mesh on disk is still a
            # different, successful job's. The same rule _rig follows about
            # finalize_rig.
            return

        temp = source_dir / rigging.RETEXTURE_GLB_TMP
        try:
            if not await asyncio.to_thread(
                functools.partial(retexture.swap_base_colour, model_glb, atlas, temp)
            ):
                raise RuntimeError("this mesh has no base-colour texture to replace")
            # Rename rather than write: model.glb is served, and os.replace is
            # atomic on both platforms as long as the two share a filesystem --
            # which they do, being siblings.
            await asyncio.to_thread(os.replace, temp, model_glb)
        finally:
            # The staging name is in the *source* job's directory, beside the
            # mesh it is a candidate replacement for. A failure that left it
            # there would put a stale half-skinned GLB next to model.glb until
            # the next re-texture happened to overwrite it.
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
        await asyncio.to_thread(self._drop_surface_artifacts, source_dir)

        report["base_model"] = base_key
        report["strength"] = strength
        report["seed"] = seed
        report["texture_size"] = texture_size
        report["prompt"] = t2i.last_prompt or prompt
        params["retexture"] = report
        await asyncio.to_thread(self.store.set_params, job_id, params)
        # On the mesh's own row as well, because that is where a reader looking
        # at the asset asks what its surface is -- and it is derived, so
        # DERIVED_PARAMS carries it and a reroll does not inherit it.
        await asyncio.to_thread(
            self.store.merge_params, source_id, {"retexture": report}
        )
        log.info(
            "re-textured job %s from %s: %d views, %.0f%% coverage",
            source_id, job_id, report["views"], report["coverage"] * 100.0,
        )

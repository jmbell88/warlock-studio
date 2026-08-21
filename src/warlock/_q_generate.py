"""``Worker``'s generation stage: conditioning, text/image -> mesh, the pipe.

The original job kind, and the only one that runs *both* models: SDXL draws a
reference and trellis-server reconstructs it. Split out of ``queue.py`` for the
reason every mixin here is: the class had 35 methods and five unrelated
subjects, and the loop core -- which is the part with the invariants -- was
buried among them.

``_generate`` is the one stage whose teardown is deliberately **not**
``Worker._release_t2i``: every other t2i stage ends when the image does, while
this one hands straight on to the mesh stage inside the same job, so what it
trades the VRAM against is different. The comment at its own ``finally`` is the
argument; ``docs/INVARIANTS.md`` records that the asymmetry is intended.

``_get_text2image`` can live here even though ``conftest`` patches the image
model: its body import (``from .pipelines.text2image import Text2Image``) reads
the patch off the *text2image module*, which is namespace-insensitive.
``Worker.__init__`` cannot, because ``TrellisServer`` is patched on ``queue``
itself -- so it stays there.

Queue.py module-level names are reached through ``queue_mod`` rather than
imported: a call-time lookup keeps ``monkeypatch.setattr(queue_mod, ...)``
landing -- ``_log_mem`` and ``_require_commit_headroom`` are exactly the kind
of thing a VRAM test rebinds -- and a module-scope import of ``.queue`` here
would be circular.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import fetch, guidance, models, provenance
from .pipelines import control, reference, seam

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class GenerateOps:
    """The generation stage, mixed into :class:`~.queue.Worker`."""

    async def _conditioning(self: Worker, job_dir: Path, params: dict[str, Any], spec):
        """Resolve a job's conditioning selection into a Conditioning, or None.

        Returns None whenever there is nothing to attach, which is what keeps
        the unconditioned path bit-identical: the pipeline is handed None, not
        an empty object it would have to interpret.

        An unknown registry key is dropped with a warning rather than failing
        the job -- the same tolerance base_model and the style LoRAs already
        get, for the same reason: params can predate a rename and the user
        cannot fix a row that already exists.
        """
        from .pipelines.conditioning import Conditioning

        # A guide the caller drew itself, rather than a hint derived from a
        # reference. Troupe's T-pose reference stage writes ``control.png``
        # at the door (``spritesynth.render_tpose_guide``) and asks for it by
        # name here, for the reason ``render_guide``'s docstring gives: the
        # guide is already line art in canny space, and running the detector
        # over it would return the outline of each stroke -- two lines where
        # the guide means one, which is the "why does my character have four
        # legs" failure. ``_sprite_synthesis`` builds its Conditioning by hand
        # for exactly this reason; a reference job cannot, because it is the
        # ordinary text path and must keep the candidates, the reroll and the
        # promote gate that path already has.
        prerendered = str(params.get("control_hint_source") or "") == "guide"

        ref = job_dir / "ref.png"
        if not ref.exists() and not prerendered:
            return None

        ip_key = params.get("ip_adapter") or None
        if ip_key is not None and ip_key not in models.IP_ADAPTERS:
            log.warning("unknown ip_adapter %r; generating without it", ip_key)
            ip_key = None

        control_key = params.get("control") or None
        if control_key is not None and control_key not in models.CONTROLNETS:
            log.warning("unknown control %r; generating without it", control_key)
            control_key = None
        if control_key is not None and not spec.controlnet:
            # Reachable when a stored base_model fell back to the registry
            # default above, or when a base loses its controlnet flag.
            log.warning(
                "base model %s cannot run a ControlNet; generating without it", spec.key
            )
            control_key = None
        if control_key is not None and models.CONTROLNETS[control_key].preprocessor is None:
            # A pipeline-fed entry (the re-texture depth control) in a text
            # job's params: guidance.normalize refuses it at the door, so this
            # is a row that never went through the door. write_hint(kind=None)
            # would fail the job with the checkpoint already in VRAM.
            log.warning(
                "control %s takes a pipeline-rendered hint; generating without it",
                control_key,
            )
            control_key = None

        if ip_key is None and control_key is None:
            return None

        control_image = None
        if control_key is not None:
            cn = models.CONTROLNETS[control_key]
            control_image = job_dir / "control.png"
            if prerendered:
                # Drawn here rather than copied from the door's job directory,
                # and that is the point: a guide is a pure function of its
                # variant, so a rerolled reference -- a *new* job id and a new,
                # empty directory -- draws the same guide instead of failing on
                # a file the door wrote into the row it rerolled. It is also
                # what keeps ``params`` the whole description of the run.
                #
                # ``edge_fraction`` is still recorded against it, so "the guide
                # drew nothing" stays an answerable question -- the same number
                # ``write_hint`` would have written.
                variant = str(params.get("guide_variant") or "")

                def _draw(variant: str = variant, dest: Path = control_image) -> Any:
                    from .pipelines import spritesynth

                    guide = spritesynth.render_tpose_guide(variant)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(f".{dest.name}.tmp")
                    guide.save(tmp, format="PNG")
                    os.replace(tmp, dest)
                    return control.hint_report(dest)

                params["control_hint"] = await asyncio.to_thread(_draw)
            else:
                params["control_hint"] = await asyncio.to_thread(
                    functools.partial(
                        control.write_hint,
                        ref,
                        control_image,
                        kind=cn.preprocessor,
                        size=spec.image_size,
                    )
                )

        return Conditioning(
            ip_adapter=ip_key,
            ip_image=ref if ip_key else None,
            ip_scale=float(params.get("ip_scale", models.DEFAULT_IP_SCALE)),
            control=control_key,
            control_image=control_image,
            control_scale=float(params.get("control_scale", models.DEFAULT_CONTROL_SCALE)),
            control_end=float(params.get("control_end", models.DEFAULT_CONTROL_END)),
        )

    async def _generate(self: Worker, job: dict[str, Any]) -> None:
        from . import queue as queue_mod

        if job["kind"] == "rig":
            await self._rig(job)
            return
        if job["kind"] == "sheet":
            await self._sheet(job)
            return
        if job["kind"] == "charsheet":
            await self._charsheet(job)
            return
        if job["kind"] == "pixel_sheet":
            await self._pixel_sheet(job)
            return
        if job["kind"] == "sprite_synthesis":
            await self._sprite_synthesis(job)
            return
        if job["kind"] == "retexture":
            await self._retexture(job)
            return
        if job["kind"] == "tile_sheet":
            await self._tile_sheet(job)
            return
        job_dir = self.config.job_dir(job["id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        params = job["params"]
        # Two seeds, one per stage, falling back to the single legacy seed so a
        # job row written before the split still reproduces exactly.
        legacy_seed = int(params.get("seed", 42))
        reference_seed = int(params.get("reference_seed", legacy_seed))
        mesh_seed = int(params.get("mesh_seed", legacy_seed))
        resolution = int(params.get("resolution", 1024))
        image_path = job_dir / "input.png"

        job_id = job["id"]
        assert self._cancel is not None
        if job["kind"] == "text":
            base_key = self._resolve_base_key(params)
            style_lora = params.get("style_lora") or None
            lora_weight = float(params.get("lora_weight", models.DEFAULT_LORA_WEIGHT))
            spec = models.BASE_MODELS[base_key]
            style = models.STYLE_LORAS.get(style_lora or "")
            if style is not None and not models.lora_fits(spec, style):
                # The same tolerance _conditioning applies to a ControlNet the
                # base cannot run, and reachable the same two ways: a stored
                # base_model fell back to the default above, or a row predates
                # the base changing family. Dropping the style beats failing a
                # job whose params the user can no longer edit.
                #
                # A style_lora naming a key the registry no longer holds
                # resolves to None here and passes through untouched, which is
                # today's behaviour exactly -- _apply_adapters logs "not
                # downloaded" for it. That is a pre-existing tolerance.
                log.warning(
                    "base model %s (%s) cannot take the style LoRA %s (%s); "
                    "generating without it",
                    spec.key, spec.family, style.key, style.family,
                )
                style_lora = None
            cond = await self._conditioning(job_dir, params, spec)
            # The one preamble, with this branch's real `cond` -- the only
            # caller that has one. `handoff` is kept for the finally below.
            t2i, handoff = await self._acquire_t2i(spec, base_key, cond)
            is_tile = job.get("stage") == "tile"
            # text2image's own PROMPT_TEMPLATE (or TILE_TEMPLATE, off the
            # tile= flag below) wraps this with the TRELLIS-friendly
            # single-object scaffolding.
            composed = guidance.compose_prompt(job["prompt"] or "", params)
            # Expansion happens once, here, outside the reroll loop: the
            # expansion is part of the *prompt*, so every reroll attempt draws
            # the same text at a fresh image seed rather than a fresh prompt
            # each time. Never on a tile -- the whitelist is object-aesthetic
            # vocabulary and a texture has no subject to describe.
            is_scene = params.get("expand") == "scene"
            if params.get("expand") and not is_tile:
                composed = await self._expand_prompt(
                    job_id, composed, params, reference_seed
                )
            # The reroll lives *inside* the try below, so however many samples
            # it draws there is still one load and one unload around all of
            # them: the VRAM handoff is a property of the stage, not of an
            # attempt.
            attempts: list[dict[str, Any]] = []
            seed = reference_seed
            retries = max(0, int(self.config.reference_retries))
            is_reference = job.get("stage") == "reference"
            try:
                while True:
                    await asyncio.to_thread(
                        functools.partial(
                            t2i.generate,
                            composed,
                            image_path,
                            seed=seed,
                            lora=style_lora,
                            lora_weight=lora_weight,
                            negative_prompt=str(params.get("negative_prompt") or ""),
                            conditioning=cond,
                            on_state=lambda s: self._t2i_state(job_id, s),
                            on_step=lambda i, n: self._t2i_step(job_id, i, n),
                            cancel_event=self._cancel.event,
                            tile=is_tile,
                            scene=is_scene,
                        )
                    )
                    params["composed_prompt"] = t2i.last_prompt or composed
                    if is_tile:
                        # A seam verdict, never a composition one: the rejection
                        # rules below are all about where a *subject* sits, and
                        # a tile has none -- so entering the reroll loop would
                        # redraw it for failing a test it cannot pass. Advisory
                        # and swallowed on failure, the same rule _audit_mesh
                        # follows: the PNG is on disk and fine.
                        try:
                            params["seam_report"] = await asyncio.to_thread(
                                seam.report, image_path
                            )
                        except Exception:
                            log.exception("seam measurement failed for job %s", job_id)
                        break
                    if not (is_reference or retries):
                        # Nothing to measure it for: a model-stage job with the
                        # retry off is measured a few lines further down by
                        # reference.prepare anyway, and paying for a second
                        # flood fill here would be pure cost.
                        break
                    # Measure only, and never a rejection: the user is judging
                    # the image, and the mesh stage is where the cost is. On a
                    # reference job this is what promote_to_model's soft check
                    # reads; on a model-stage job with the reroll on it is not
                    # stored at all (see the `if is_reference` below) and exists
                    # purely to decide whether to redraw, because
                    # reference.prepare measures again a few lines further down.
                    try:
                        report = await asyncio.to_thread(reference.measure_file, image_path)
                    except Exception:
                        # Advisory, exactly as the ranking below is: the image
                        # is already on disk and fine, so no verdict means no
                        # reroll -- never a job failed by its own second
                        # opinion. measure_file catches its own read errors, so
                        # reaching here means the measurement itself broke.
                        log.exception("measuring the reference failed for job %s", job_id)
                        # The attempt still happened and its image is the one on
                        # disk, so it is recorded before the break or the
                        # provenance would name an earlier seed that no longer
                        # reproduces what shipped. ok is True with measured
                        # False, following reference.unmeasured: "we could not
                        # measure this" is not "this was refused", and the
                        # sample is the one being kept.
                        attempts.append(
                            {"seed": seed, "ok": True, "reasons": [], "measured": False}
                        )
                        break
                    if is_reference:
                        params["reference_report"] = report.as_dict()
                    attempts.append(
                        {"seed": seed, "ok": report.ok, "reasons": list(report.reasons)}
                    )
                    if report.ok or len(attempts) > retries or self._cancel.event.is_set():
                        # A budget, not a loop: the report's rules are
                        # heuristics, so past the ceiling the user gets the
                        # last attempt rather than a job that refuses to end.
                        break
                    seed = queue_mod._fresh_seed()
                    log.info(
                        "job %s: rerolling the reference (%s)",
                        job_id,
                        "; ".join(report.reasons),
                    )
                if len(attempts) > 1:
                    # Only when it actually retried: a single-attempt job's
                    # provenance is already the seed in params.
                    params["reference_attempts"] = attempts
                    params["reference_seed"] = seed
                if is_reference:
                    try:
                        params["rank"] = await asyncio.to_thread(
                            self._rank_reference, image_path, params
                        )
                    except Exception:
                        # Advisory: the image is on disk and fine. The UI shows
                        # no score rather than a wrong one.
                        log.exception("ranking failed for job %s", job_id)
                if t2i.last_recipe:
                    params.setdefault("recipe", {})["reference"] = t2i.last_recipe
                await asyncio.to_thread(self.store.set_params, job_id, params)
            finally:
                if handoff:
                    # unload(), not trim(): trim() returns the CUDA caching
                    # allocator's pool, and an offloaded pipe's cost is not in
                    # the CUDA pool at all -- it is ~16 GiB of host weights that
                    # only dropping the pipe releases. For the exclusive and
                    # conditioning reasons it is the original stop-before-load /
                    # unload-before-next-start choreography.
                    #
                    # Deliberately *not* followed by `self._text2image = None`,
                    # which the three later teardown sites (_pixel_sheet,
                    # _sprite_synthesis, _retexture) do. That asymmetry looks
                    # like drift and is not: every reader of the field asks
                    # `is not None and .loaded`, so `.loaded` is what answers
                    # "is a pipe resident" -- including the idle-timeout sweep
                    # and the VRAM credit at dispatch -- and an unloaded
                    # instance is reusable rather than stale, because load()
                    # rebuilds from `_pipe is None`. Keeping it means the next
                    # text job at the same base reuses this object instead of
                    # constructing a second one, which is what makes the credit
                    # comment above ("unload before loading the new") true of
                    # one identity rather than two.
                    await asyncio.to_thread(t2i.unload)
                else:
                    # Coexist used to keep the pipeline resident here and merely
                    # ``trim()`` the caching allocator's pool -- the card had
                    # room for both models, so why throw away a checkpoint the
                    # next job may want warm.
                    #
                    # Because the card was never the binding constraint. The
                    # *host* is: the pool trim gave back the CUDA blocks, and
                    # the ~7 GiB of weights it left resident kept both their
                    # WDDM backing and the pipe's own arenas charged against
                    # system commit, for the full 600 s idle timeout after a job
                    # that had already finished. ``_require_commit_headroom``
                    # then refuses the *next* job on a percentage of that
                    # commit: a 63.5 GiB machine with a 14.2 GiB pagefile sat at
                    # 96% and refused everything while 24 GiB of RAM was free.
                    #
                    # So both branches unload now and the difference between
                    # them is gone. Kept as two branches anyway, because the
                    # reasons are still two -- ``handoff`` must unload before
                    # trellis restarts, and this one unloads because nothing is
                    # owed a warm pipe -- and collapsing them would leave one
                    # comment claiming to explain both.
                    #
                    # ``self._text2image`` is deliberately *not* cleared, for
                    # the reason spelled out in the branch above: ``.loaded`` is
                    # what every reader asks, and keeping the identity is what
                    # lets the next job at the same base reuse this object.
                    await asyncio.to_thread(t2i.unload)

            if job.get("stage") in ("reference", "tile"):
                # The whole point of the split: the user judges the image before
                # anything pays for a trellis run. The job is finished here --
                # promotion creates a separate child job (service.jobs.promote_to_model).
                # A tile never has a mesh stage at all, and is not promotable.
                queue_mod._log_mem("after image-only job")
                return
        elif not image_path.exists():
            raise RuntimeError("image job has no uploaded input.png")

        if job.get("stage") in ("reference", "tile"):
            # The text branch returns here itself, having just generated the
            # image. This catches the other kind: a reference whose pixels came
            # from an upload or the paint editor. Its stage says the job stops
            # before the mesh, and honouring that is a property of the *stage*,
            # not of how the image was made -- reading it off the kind is how a
            # painted reference used to buy an unasked-for trellis run.
            #
            # "tile" is here for the same reason rather than because it can
            # happen: create_job refuses a non-text tile, so nothing should
            # reach this line with that stage. But the check that stops a job
            # short of trellis must be about the stage and nothing else, or a
            # future door onto a tile inherits a two-minute reconstruction.
            queue_mod._log_mem("after image-only job")
            return

        if self._cancel.event.is_set():
            return

        # The stage boundary, and the first thing on this path that can lead to
        # a spawn. _check_resources asked this once, before the image stage --
        # which is exactly the stage that moves the number, and on an offloaded
        # checkpoint moves it by ~16 GiB of host weights. Until now the only
        # observation point here was _log_mem's log.critical, which records the
        # wall and then walks into it.
        #
        # The wording is the requirement: the image is finished and on disk, and
        # a user told "out of memory" after a successful two-minute generation
        # will assume they lost it.
        queue_mod._require_commit_headroom(
            " after the image stage",
            "The reference image is finished and saved with this job; only the "
            "3D stage was withheld. Close other applications or restart "
            "Warlock, then re-run this job to reconstruct the mesh.",
        )

        # The server's launch settings are per-job now, so a job that pins
        # either one needs the resident server restarted before it runs. Two
        # things keep this from disturbing anything above it. It is called with
        # *resolved* values on every model-stage job, not only on a job that
        # pins one -- so an ordinary job following a sweep unit restores the
        # config's own settings, and nothing has to remember that the sweep
        # changed them. And it is a no-op against a server that is not running,
        # which is exactly the state exclusive mode's handoff has just left
        # behind (stop -> SDXL -> unload), so the stop-before-load /
        # unload-before-next-start choreography is byte-for-byte unchanged.
        #
        # A restart that then fails to come back is nothing new: ensure_started
        # inside generate() raises, the job errors, and _check_backoff throttles
        # the respawn.
        await asyncio.to_thread(
            self.trellis.ensure_config,
            tex_res=int(params["trellis_tex_res"])
            if "trellis_tex_res" in params
            else self.config.trellis_tex_res,
            band=int(params["trellis_band"])
            if "trellis_band" in params
            else self.config.trellis_band,
        )

        self.progress.update(
            job_id,
            phase="trellis",
            label="Starting 3D engine" if not self.trellis.running else "Sending image",
            inner=0.0,
            inner_next=0.02,
            nominal=6.0,
            detail="",
        )
        # The reconstruction is kept verbatim as source.glb and never
        # overwritten: model.glb is derived from it, so re-targeting a triangle
        # budget later (service.jobs.optimize_job) never has to pay for
        # another trellis run.
        source_glb = job_dir / "source.glb"
        glb_path = job_dir / "model.glb"

        # What trellis actually sees, measured and recorded before it is
        # uploaded. The rejection happens *here*, not after: a reference that
        # cannot reconstruct should cost the request, not two minutes of GPU.
        trellis_input = job_dir / "reference.png"
        report = await asyncio.to_thread(
            functools.partial(
                reference.prepare,
                image_path,
                trellis_input,
                enabled=bool(params.get("reference_prep", queue_mod.DEFAULT_REFERENCE_PREP)),
            )
        )
        params["reference_report"] = report.as_dict()
        # Off the loop thread: the recipe fingerprints trellis-server.exe and
        # the models directory, which is real disk I/O, and this thread is the
        # one hosting cancellation.
        params.setdefault("recipe", {})["trellis"] = await asyncio.to_thread(
            functools.partial(
                provenance.trellis_recipe, self.config, params, mesh_seed=mesh_seed
            )
        )
        await asyncio.to_thread(self.store.set_params, job_id, params)
        if not report.ok:
            raise RuntimeError("; ".join(report.reasons))

        queue_mod._log_mem("before trellis generate")
        await self.trellis.generate(
            trellis_input,
            source_glb,
            seed=mesh_seed,
            resolution=resolution,
            bg_removal=str(params.get("bg_removal") or guidance.FALLBACK_BG_REMOVAL),
        )
        # The remesh loop, and it re-enters only the trellis half. Everything
        # above it -- the VRAM handoff, the reference measurement, the
        # composition gate -- has already happened and is deliberately not
        # repeated: under exclusive mode the handoff is a property of the
        # stage, not of an attempt, exactly as the reference reroll's is.
        #
        # Off unless mesh_retries says otherwise, in which case a mesh whose
        # worst view is more see-through than mesh_hole_max is reconstructed
        # again at a fresh seed. See config.mesh_hole_max for where that number
        # came from.
        best: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        retries = max(0, int(self.config.mesh_retries))
        keep = job_dir / "best.glb"
        keep_source = job_dir / "best.source.glb"
        # Whether keep/keep_source currently hold the best attempt's pair. The
        # loop only ever retries from a staged state, so this is True whenever
        # a restore could be needed.
        staged = False
        # Which attempt's reconstruction is sitting at source.glb/model.glb
        # right now, which is the only thing that decides whether the scratch
        # copies have to be put back. None means "no longer any attempt's" --
        # a retry that raised part-way through writing source.glb.
        on_disk_seed: int | None = mesh_seed
        # The scratch pair is released in a finally, not as a trailing
        # statement. best.glb and best.source.glb are two whole
        # reconstructions -- tens of MB -- and every path out of the block
        # below that is not a clean fall-through used to leak them: an
        # exception from _optimize, _apply_scale, _audit_mesh or the params
        # write, and a cancel that propagates. Nothing sweeps them
        # afterwards; _discard_artifacts runs only on a cancelled job, so on
        # an error they sat in the job directory for its whole life.
        try:
            while True:
                await self._optimize(job_id, source_glb, glb_path, params)
                await self._apply_scale(job_id, glb_path, params)
                audit = await self._audit_mesh(job_id, glb_path, params)
                worst = None if audit is None else audit.get("worst")
                attempts.append({"seed": mesh_seed, "worst": worst})
                new_best = best is None or (
                    worst is not None
                    and best["worst"] is not None
                    and worst < best["worst"]
                )
                if new_best:
                    # params is snapshotted shallowly, which is exact today and
                    # deliberately so: _optimize, _apply_scale and _audit_mesh all
                    # *rebind* their top-level keys rather than mutating a nested
                    # value in place, so a shallow copy of the mapping isolates
                    # every value that can change between here and the restore. A
                    # callee that grew an in-place nested mutation would defeat it,
                    # which is the one thing to check before adding to this loop.
                    best = {"seed": mesh_seed, "worst": worst, "params": dict(params)}
                if (
                    worst is None
                    or worst <= self.config.mesh_hole_max
                    or len(attempts) > retries
                    or self._cancel.event.is_set()
                ):
                    # A budget, not a loop, and an unmeasurable mesh is not a bad
                    # one: no verdict means the mesh already on disk is kept.
                    break
                if new_best:
                    # Below the break, not above it: a copy taken on the attempt
                    # that turns out to be the last one is never read, and this is
                    # two whole GLBs. Reaching here means a retry is definitely
                    # about to overwrite them.
                    #
                    # Kept aside rather than trusted to be last, because a reroll
                    # can be worse and then two minutes of GPU would have made the
                    # asset worse than it already was. Both files, because
                    # source.glb is what a later retarget re-derives model.glb
                    # from -- a source left behind by the losing attempt would
                    # quietly undo the choice made here the first time someone
                    # changed a triangle budget.
                    try:
                        # Hard links, one executor hop for the pair (C37): see
                        # _stage_link for why a link cannot be scribbled on.
                        def _stage_pair() -> None:
                            queue_mod._stage_link(glb_path, keep)
                            queue_mod._stage_link(source_glb, keep_source)

                        await asyncio.to_thread(_stage_pair)
                    except OSError:
                        # A full disk is the realistic way a tens-of-megabytes copy
                        # fails, and this feature is what doubled the footprint. So
                        # it gives up on retrying rather than failing a job whose
                        # mesh is already on disk: without a copy of the attempt in
                        # hand there is nothing to fall back to, and running trellis
                        # again would risk overwriting the only good result.
                        log.exception(
                            "job %s: could not stage the mesh for a remesh; keeping it", job_id
                        )
                        break
                    staged = True
                mesh_seed = queue_mod._fresh_seed()
                log.info(
                    "job %s: mesh audited %.3f open, past %.3f -- remeshing at seed %d",
                    job_id, worst, self.config.mesh_hole_max, mesh_seed,
                )
                # The phase, not just a label. request_cancel decides whether to
                # kill trellis-server by reading it, and killing the subprocess is
                # the only abort trellis has -- left at "audit" (what _audit_mesh
                # set), a cancel would set the event, skip the kill and then block
                # here for a whole reconstruction. Mirrors the first generate's
                # update above, for the same reason.
                self.progress.update(
                    job_id,
                    phase="trellis",
                    label="Remeshing" if self.trellis.running else "Starting 3D engine",
                    inner=0.0,
                    inner_next=0.02,
                    nominal=6.0,
                    detail=f"attempt {len(attempts) + 1}",
                )
                on_disk_seed = None
                try:
                    await self.trellis.generate(
                        trellis_input,
                        source_glb,
                        seed=mesh_seed,
                        resolution=resolution,
                        bg_removal=str(params.get("bg_removal") or guidance.FALLBACK_BG_REMOVAL),
                    )
                except Exception:
                    # The best reconstruction so far is already on disk and the
                    # retry was this code's own idea, so a retry that blows up must
                    # not retroactively fail a job that had a mesh -- the same rule
                    # the audit, the report and the grounding follow. source.glb is
                    # left disowned rather than trusted (a half-written file is
                    # exactly what a failed run leaves), and the restore below puts
                    # the kept attempt's pair back.
                    if self._cancel.event.is_set():
                        # Not a failure at all: request_cancel killed the server and
                        # the in-flight post died with it, exactly as designed. A
                        # traceback here would be noise on a routine cancel.
                        log.info("job %s: remesh cancelled", job_id)
                    else:
                        log.exception(
                            "remesh failed for job %s; keeping the best mesh so far", job_id
                        )
                    break
                on_disk_seed = mesh_seed
            # Whether what is on disk is what was chosen -- which is not the same
            # question as "did an earlier attempt win", because a retry that failed
            # part-way leaves neither attempt's reconstruction there.
            restored = best is not None and staged and on_disk_seed != best["seed"]
            if restored and best is not None:
                # Put the kept attempt's GLBs and its measurements back, so what is
                # on disk and what params claims about it describe the same mesh.
                #
                # source.glb first, and the order is the whole point: these are two
                # files and no filesystem makes the pair atomic, so the question is
                # which half-done state is survivable. Source-then-model leaves the
                # *source* as the kept attempt's, and service.jobs.optimize_job
                # re-derives model.glb from source.glb -- so the next retarget
                # converges on the mesh that was chosen. The other order strands the
                # winner's model.glb over the loser's source, and a retarget
                # silently swaps the rejected reconstruction back in.
                #
                # params is restored wholesale rather than key by key: the snapshot
                # is a copy of the mapping taken between two trellis runs, and only
                # _optimize, _apply_scale and _audit_mesh write between then and
                # here, so replacing it is exact -- and stays exact when one of
                # those three grows a new key, which a hand-written strip list
                # would not.
                try:
                    # source first (the ordering argument above), linked back for
                    # the reason the staging linked out (C37); _stage_link's
                    # os.replace keeps each served name whole for concurrent
                    # readers, which is what staged_copy bought the copy path.
                    def _restore_pair() -> None:
                        queue_mod._stage_link(keep_source, source_glb)
                        queue_mod._stage_link(keep, glb_path)

                    await asyncio.to_thread(_restore_pair)
                except OSError:
                    # Same rule as everywhere else here: the last attempt's mesh is
                    # on disk and usable, so a failed restore costs the user the
                    # better of two meshes, never the job.
                    log.exception(
                        "job %s: could not restore the best mesh; keeping the last attempt", job_id
                    )
                    restored = False
                else:
                    params.clear()
                    params.update(best["params"])
            if best is not None and (restored or len(attempts) > 1):
                if len(attempts) > 1:
                    params["mesh_attempts"] = attempts
                # The seed that reproduces the mesh that shipped, which is the best
                # attempt's and not necessarily the last one tried -- and the recipe
                # carries the same seed, or the provenance and the row would name
                # different reconstructions. If the restore itself failed then what
                # shipped is the last *measured* attempt, and saying so beats
                # claiming a seed whose mesh is not there.
                kept_seed = best["seed"] if restored else attempts[-1]["seed"]
                params["mesh_seed"] = kept_seed
                params.setdefault("recipe", {})["trellis"] = await asyncio.to_thread(
                    functools.partial(
                        provenance.trellis_recipe, self.config, params, mesh_seed=kept_seed
                    )
                )
                await asyncio.to_thread(self.store.set_params, job_id, params)
        except BaseException:
            # The restore above runs only on the clean fall-through. An
            # exception from a retry's _optimize/_apply_scale/_audit_mesh or
            # the params write (and a cancel that propagates) would skip it,
            # and the finally below would then delete the staged pair --
            # erasing the measured-good mesh while a half-processed attempt
            # sits on the served names. Best-effort and synchronous on purpose:
            # two link+replace ops, so a second cancel cannot skip it either.
            if staged and best is not None and on_disk_seed != best["seed"]:
                with contextlib.suppress(OSError):
                    queue_mod._stage_link(keep_source, source_glb)
                    queue_mod._stage_link(keep, glb_path)
            raise
        finally:
            for scratch in (keep, keep_source):
                with contextlib.suppress(OSError):
                    scratch.unlink()

    async def _expand_prompt(
        self: Worker, job_id: str, composed: str, params: dict[str, Any], seed: int
    ) -> str:
        """The composed subject through the local expander, or unchanged.

        Advisory in the way the rank and the seam report are: the door
        (``validation.check_weights``) already refused a submit whose expander
        is missing, so a failure here is a stored row that predates the
        weights or a broken load -- and neither must fail a job whose prompt
        is perfectly generatable as written. What *did* run is recorded as
        ``expanded_prompt`` (DERIVED_PARAMS: a reroll re-records it).
        """
        spec = models.EXPANDER_MODELS[models.DEFAULT_EXPANDER]
        model_dir = self.config.t2i_model_root / spec.dir_name
        try:
            from .pipelines import expand as expand_mod

            expanded = await asyncio.to_thread(
                functools.partial(expand_mod.expand, composed, model_dir, seed=seed)
            )
        except Exception:
            log.exception(
                "prompt expansion failed for job %s; using the prompt as written",
                job_id,
            )
            return composed
        if expanded is None:
            # The gate said this prompt is already detailed; keeping the
            # param absent is what says "nothing was appended".
            return composed
        params["expanded_prompt"] = expanded
        return expanded

    async def _get_text2image(self: Worker, base_key: str):
        """The resident image pipeline, swapped if the job wants a different base.

        One base model at a time, not one per key: a 32 GB card holds
        trellis-server (~16 GB) plus a single SDXL-class pipe (~7 GB) and not
        two of the latter, so a switch has to free the old one first. Style
        LoRAs are the opposite -- they are adapters on the resident pipe and
        switch per job with no reload at all (text2image._apply_adapters).

        The unload goes through a thread for the same reason every other
        unload() call site does: gc.collect() plus empty_cache() on the event
        loop stalls every other job on the queue.
        """
        # Backstop -- normally a no-op: _acquire_t2i (the only current caller)
        # already evicted a stale pipe before its commit-headroom check. Kept
        # so the staleness predicate has one owner and no future caller is
        # handed a pipe with the wrong base or generation.
        await self._evict_stale_t2i(base_key)
        if self._text2image is None:
            try:
                from .pipelines.text2image import Text2Image
            except ImportError as exc:
                raise RuntimeError(
                    "text-to-3D requires the text2image extra: uv sync --extra text2image"
                ) from exc
            spec = models.BASE_MODELS[base_key]
            self._text2image = Text2Image(
                spec,
                self.config.t2i_model_root,
                # Honoured for turbo only -- see Config.t2i_turbo_dir.
                self.config.t2i_turbo_dir if base_key == models.T2I_DIR_MODEL else None,
            )
            self._t2i_key = base_key
            # Read *after* the object is built, so a mutation that lands during
            # construction is seen as stale on the next job rather than missed.
            self._t2i_generation = fetch.store_generation()
        return self._text2image

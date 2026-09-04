"""``Worker``'s Muse stage: a generated piece of music.

One job kind, ``music``, and it is the shortest stage in the queue -- one model
call that writes one file. There is no source/derived split and no follow-up:
unlike ``source.glb``/``model.glb`` there is no second, differently-useful
version of what the model produced, so ``track.wav`` is the whole artifact, for
the same reason a reference job has only ``input.png``.

**Why it is not a branch of ``_generate``'s text stage.** That stage is
SDXL-and-trellis shaped end to end -- conditioning, the reroll budget, the
composition verdict, the promote gate, the handoff choreography with a running
trellis-server. None of it has a meaning here: there is no subject to be
badly composed and no mesh downstream to protect. What the two share is the
resident-child lifecycle, and that is shared for real, through
``_acquire_music``/``_release_music`` mirroring their image siblings.

**Handoff is one term, not three.** ``queue._needs_handoff`` is typed on
``BaseModel`` and its three reasons are SDXL-and-trellis reasons: the
conditioning term is about a ControlNet plus a CLIP encoder, and the residency
term is about ``enable_model_cpu_offload``. A music job has no conditioning and
ACE-Step has no OFFLOAD residency, so only ``vram_exclusive`` survives -- and at
~8.3 GiB the pipe can coexist with a resident trellis on a large card exactly as
the ~7.5 GiB image pipe does today. ``_music_needs_handoff`` says that in one
line rather than making the shared predicate take a union type in order to
answer False to two thirds of itself.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING, Any

from . import models

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


def _music_needs_handoff(*, exclusive: bool) -> bool:
    """Must trellis be stopped before the music model loads?

    See this module's docstring for why this is one term and
    ``queue._needs_handoff`` is three. A function rather than the expression
    inlined at its one call site, so that the *reasoning* has somewhere to live
    and a second term has somewhere to be added.
    """
    return exclusive


class MusicOps:
    """The music generation stage, mixed into :class:`~.queue.Worker`."""

    async def _music(self: Worker, job: dict[str, Any]) -> None:
        job_id = job["id"]
        params = job["params"]
        job_dir = self.config.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        output = job_dir / "track.wav"

        model_key = str(params.get("music_model") or models.DEFAULT_MUSIC_MODEL)
        if model_key not in models.MUSIC_MODELS:
            # Refused rather than defaulted: service.validation already checked
            # this at the door, so reaching here means a stored row names a
            # model this build no longer has -- and silently substituting one
            # would record a recipe that never ran.
            raise RuntimeError(f"unknown music model: {model_key!r}")
        spec = models.MUSIC_MODELS[model_key]

        assert self._cancel is not None
        client, handoff = await self._acquire_music(spec)
        try:
            await asyncio.to_thread(
                functools.partial(
                    client.generate,
                    str(job["prompt"] or ""),
                    output,
                    lyrics=str(params.get("lyrics") or ""),
                    audio_duration=float(params.get("duration", 60.0)),
                    infer_step=int(params.get("infer_step", 60)),
                    guidance_scale=float(params.get("guidance_scale", 15.0)),
                    scheduler_type=str(params.get("scheduler_type", "euler")),
                    cfg_type=str(params.get("cfg_type", "apg")),
                    omega_scale=float(params.get("omega_scale", 10.0)),
                    seed=int(params["seed"]) if params.get("seed") is not None else None,
                    on_state=lambda s: self._music_state(job_id, s),
                    on_step=lambda i, n: self._music_step(job_id, i, n),
                    cancel_event=self._cancel.event,
                )
            )
            # The moment the WAV is on disk and nothing else can undo it. A
            # cancel that arrives after this point would leave a finished
            # artifact under a cancelled row, which the library has no way to
            # draw honestly.
            self._cancel.commit()
            if client.last_recipe:
                params.setdefault("recipe", {})["music"] = client.last_recipe
            # What the worker *observed*, as opposed to what was asked for: the
            # duration the model actually rendered. In DERIVED_PARAMS, so a
            # rerun at a different duration does not inherit this one's.
            params["actual_duration"] = float(params.get("duration", 60.0))
            await asyncio.to_thread(self.store.set_params, job_id, params)
        finally:
            # Every path out, including a cancel and a raised generate: an
            # 8.3 GiB pipe left resident by a failed job is the failure
            # ``_release_t2i``'s docstring argues about at length.
            await self._release_music(client, spec, handoff=handoff)

    # --- the resident child ---------------------------------------------------

    async def _acquire_music(self: Worker, spec: models.MusicModel):
        """The VRAM handoff a music stage makes, asked once. -> (client, handoff)

        ``_acquire_t2i``'s shape and its ordering -- stop trellis if the flag
        demands it, then check host commit immediately before the load, so the
        answer is about the allocation that is actually about to happen. What it
        does *not* have is a stale-pipe eviction: there is one music model, so
        there is no key for a resident pipe to be wrong about, and the
        base-model cache-generation logic has nothing to key on either.
        """
        from . import queue as queue_mod

        handoff = _music_needs_handoff(exclusive=bool(self.config.vram_exclusive))
        if handoff:
            await asyncio.to_thread(self.trellis.stop)
            queue_mod._log_mem("after trellis stop")
        if self._music_client is None or not self._music_client.loaded:
            await queue_mod._require_commit_headroom_settled(
                f" before loading {spec.label}",
                "Close other applications, or generate a shorter track.",
                need_gib=spec.host_peak_gib or spec.vram_gib,
            )
        return await self._get_music_client(spec), handoff

    async def _get_music_client(self: Worker, spec: models.MusicModel):
        """The resident music child, constructed on first use.

        The import is guarded and names the extra for ``_get_text2image``'s
        reason: the client is import-light, but the child it spawns is not, and
        a host without the extra should hear one clear sentence rather than a
        subprocess failing to start.
        """
        if self._music_client is None:
            try:
                from .pipelines.music_client import MusicClient
            except ImportError as exc:
                raise RuntimeError(
                    "Muse requires the music extra: uv sync --extra music"
                ) from exc
            self._music_client = MusicClient(
                spec, self.config.t2i_model_root / spec.dir_name
            )
        return self._music_client

    async def _release_music(
        self: Worker, client: Any, spec: models.MusicModel, *, handoff: bool
    ) -> None:
        """Give the weights back at the end of the stage that loaded them.

        Unconditional, for ``_release_t2i``'s measured reason: a resident pipe
        is VRAM whose WDDM backing, plus the child's own arenas, is charged
        against a system commit limit that admission then refuses the *next*
        job on. A back-to-back take pays one reload rather than the queue
        holding 8.3 GiB against a job that may never come.

        The client object is unloaded and kept, not forgotten: ``.loaded`` is
        what every reader asks, and an unloaded client rebuilds its child on the
        next request, so keeping it is reuse rather than staleness.

        ``trim`` first and unconditionally -- the op exists on the worker for
        exactly this, so this path never has to branch on which kind of pipe it
        holds. ``handoff`` and ``spec`` stay in the signature because the
        argument above is written in terms of them, and a second term in
        ``_music_needs_handoff`` would need them here.
        """
        await asyncio.to_thread(client.trim)
        await asyncio.to_thread(client.unload)

    # --- progress -------------------------------------------------------------

    def _music_state(self: Worker, job_id: str, state: str) -> None:
        from . import queue as queue_mod

        phase, label = queue_mod.MUSIC_PHASES.get(state, ("music_load", "Preparing"))
        # Loading has no measurable inner progress; creep across the whole
        # phase so the bar still moves.
        self.progress.update(
            job_id,
            phase=phase,
            label=label,
            inner=0.0,
            inner_next=1.0,
            nominal=30.0,
            detail="",
        )

    def _music_step(self: Worker, job_id: str, step: int, total: int) -> None:
        self._step_progress(job_id, "music_sample", "Composing", step, total)

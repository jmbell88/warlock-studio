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


#: Which ACE-Step task each Muse task is spelled as on the wire.
#:
#: Two of the six are renames rather than passthroughs. ``loop`` is **Muse's
#: own name** for a repaint across a rolled joint -- upstream has no cyclic
#: objective and no loop task, and calling it ``repaint`` in the UI would name
#: the mechanism instead of the intent. ``audio2audio`` is a flag rather than a
#: task upstream, so it maps to plain ``text2music`` and turns the flag on.
_UPSTREAM_TASK = {
    "retake": "retake",
    "extend": "extend",
    "repaint": "repaint",
    "edit": "edit",
    "loop": "repaint",
    "audio2audio": "text2music",
}


def _task_kwargs(params: dict[str, Any], job_dir: Any) -> dict[str, Any]:
    """The extras a derived music job sends, or ``{}`` for an ordinary one.

    Module level and pure for ``_music_needs_handoff``'s stated reason: so the
    *reasoning* has somewhere to live, and so it can be tested without a
    client, a card or a queue.

    ``{}`` for a row with no ``task`` is not a convenience -- it is the
    guarantee that every job minted before this existed takes a byte-identical
    path through ``client.generate``.
    """
    task = str(params.get("task") or "")
    if not task:
        return {}
    upstream = _UPSTREAM_TASK.get(task)
    if upstream is None:
        # A stored row naming a task this build no longer has. Refused rather
        # than silently downgraded to text2music, which would record a recipe
        # that never ran -- ``_music``'s own rule for an unknown model.
        raise RuntimeError(f"unknown music task: {task!r}")

    out: dict[str, Any] = {"task": upstream}
    source = job_dir / "source.wav"

    if task == "audio2audio":
        # **Never ``src_audio_path``.** ``__call__`` asserts that that path
        # implies repaint/edit/extend, so sending both trips an assertion
        # inside the child rather than refusing at the door. The reference
        # travels as ``ref_audio_input``, which is a different argument
        # entirely and the reason this function exists rather than a dict
        # comprehension over params.
        out["audio2audio_enable"] = True
        out["ref_audio_input"] = str(source)
        out["ref_audio_strength"] = float(params.get("ref_audio_strength", 0.5))
        return out

    if task == "retake":
        # No source at all: a retake is a re-run from the parent's *noise
        # draw*, blended toward a fresh one. ``__call__`` sets the repaint
        # window to the whole duration itself for this task, so sending one
        # here would be a second, disagreeing spelling of the same thing.
        out["retake_variance"] = float(params.get("retake_variance", 0.5))
        return out

    out["src_audio_path"] = str(source)

    if task == "extend":
        # Upstream spells the pads as a *negative* repaint window: the head pad
        # runs from -left to 0 and the tail from duration to duration+right.
        # The door has already refused a pad longer than the parent (the pads
        # are sliced out of a tensor allocated at the source's frame length, so
        # a longer one is silently zero-filled -- silence, not music).
        left = float(params.get("extend_left", 0.0))
        right = float(params.get("extend_right", 0.0))
        out["repaint_start"] = -left
        out["repaint_end"] = float(params.get("parent_duration", 0.0)) + right
        return out

    if task == "edit":
        # The *target* conditioning is the new brief; the source conditioning
        # stays the parent's, which is what makes FlowEdit keep the take's
        # identity rather than composing a new piece to the new words.
        out["edit_target_prompt"] = str(params.get("edit_prompt") or "")
        out["edit_target_lyrics"] = str(params.get("edit_lyrics") or "")
        out["edit_n_min"] = float(params.get("edit_n_min", 0.0))
        out["edit_n_max"] = float(params.get("edit_n_max", 1.0))
        return out

    # repaint and loop. The window is the same argument pair; what differs is
    # that ``derive_music_job`` has already written a *rolled* source for a
    # loop and centred the window on the joint. See ``_roll_wav``.
    out["repaint_start"] = float(params.get("repaint_start", 0.0))
    out["repaint_end"] = float(params.get("repaint_end", 0.0))
    return out


def _roll_wav(data: bytes, seconds: float) -> bytes:
    """Rotate a WAV's frames by ``seconds``. -> the rolled file's bytes.

    **Why a loop is a rolled repaint.** ACE-Step has no cyclic objective, and a
    plain repaint of the tail does not condition on the head, because the mask
    is positional: the model sees "regenerate the last four seconds" and has no
    idea the first four are what they must join onto. Rolling the take by half
    its length puts the head/tail joint in the *middle* of the file, where a
    repaint across it has the music on both sides as context. Rolling back
    afterwards is exact -- a rotation loses nothing -- so the roll is the entire
    difference between a joint the model wrote and a cut.

    It still does not make the first and last samples equal. That is what the
    player's loop point and its crossfade are for, and the manual has to say
    both halves or it promises something the model cannot do.
    """
    import io
    import wave as wave_mod

    import numpy as np

    with wave_mod.open(io.BytesIO(data)) as handle:
        params = handle.getparams()
        raw = handle.readframes(handle.getnframes())

    # Stdlib ``wave`` on both sides, and deliberately not ``sirens.wavout``:
    # the queue may not import ``studio`` (``test_queue`` enforces it), and
    # ``read_wav`` would be the wrong tool anyway -- it downmixes to mono and
    # resamples to a target rate, and a roll has to be exact and reversible.
    # Round-tripping the same ``getparams`` is what makes it lossless: no
    # float conversion, no re-quantisation, the identical frames in a new
    # order. ``WARLOCK 5/5`` makes the width 16-bit by construction.
    if params.sampwidth != 2 or params.nchannels < 1:
        raise RuntimeError("source.wav is not the 16-bit PCM this build writes")
    frames = np.frombuffer(raw, dtype="<i2").reshape(-1, params.nchannels)
    shift = int(round(seconds * params.framerate)) % max(len(frames), 1)
    rolled = np.roll(frames, -shift, axis=0)

    out = io.BytesIO()
    with wave_mod.open(out, "wb") as handle:
        handle.setparams(params)
        handle.writeframes(rolled.astype("<i2").tobytes())
    return out.getvalue()


def _stage_rolled_wav(output: Any, seconds: float) -> None:
    """Roll ``track.wav`` back, staged, so a kill mid-roll cannot corrupt it.

    The 2026-09-05 audit (muse-04) found this read-then-write-in-place onto
    ``track.wav`` -- the job's served artifact name -- while every other
    writer onto a served name in this module stages to a temp sibling and
    ``replace``s it: ``_write_stems_sidecar`` below, and ``separation_worker``'s
    per-stem ``.tmp``. A process killed between the read and the write left a
    truncated or corrupt ``track.wav`` on disk with no temp file ever having
    existed to prove it. The temp name matches ``_write_stems_sidecar``'s
    ``.<name>.tmp`` convention -- a bare ``track.wav.tmp`` could collide with
    a concurrent writer using the same convention on a different file, which
    is its own recorded incident (M03).
    """
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_bytes(_roll_wav(output.read_bytes(), seconds))
    tmp.replace(output)


def _write_stems_sidecar(out_dir: Any, spec: Any, result: dict[str, Any], job_id: str) -> None:
    """``stems.json``, written **last**, as the completion gate.

    ``rig.json``'s rule and ``sheet.json``'s: the four WAVs land one at a time,
    so their existence cannot say the set is finished, and a reader that took
    it that way would offer a take with three stems as a take with four.

    Staged and renamed like every other write onto a name something else reads.
    What it records is what a reader of the directory could not otherwise
    reconstruct: which job produced these, and with which model.
    """
    import json

    payload = {
        "job": job_id,
        "model": spec.key,
        "sources": list(spec.sources),
        "files": list(result.get("files") or []),
        "rate": result.get("rate"),
    }
    path = out_dir / "stems.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


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

        # Raised before the acquire, so a row naming an unknown task fails
        # without having loaded 8.3 GiB to find out.
        extra = _task_kwargs(params, job_dir)

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
                    # ``{}`` for every row minted before tasks existed, which
                    # is what keeps their path through here byte-identical.
                    # ``MusicClient.generate`` takes these as ``**extra`` and
                    # deliberately names none of them -- see its docstring.
                    **extra,
                )
            )
            if params.get("task") == "loop":
                # Roll the finished take back. The source was rolled by half
                # its length at the door so that the joint sat in the middle,
                # where the repaint could see the music on both sides of it;
                # rolling back by the same amount is exact and puts the joint
                # where the user asked for it -- at the ends.
                #
                # Before the commit, because the commit is the point at which
                # the artifact becomes final: a take committed still rolled is
                # a take whose bars start in the wrong place.
                await asyncio.to_thread(
                    _stage_rolled_wav, output, -float(params.get("roll", 0.0))
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
                need_gib=queue_mod._host_peak_gib(spec),
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

        No ``trim`` first: unlike ``_release_t2i``'s in-process pipe, this
        client's ``unload`` kills the whole child (``MusicClient._stop_child``)
        rather than freeing a cache inside a process that keeps running, so it
        already gives back everything a trim would have -- both the device
        VRAM and the child's own host-commit arenas. A trim first would be a
        second IPC round-trip to a process about to be killed, one that can
        itself log a failure for work the kill makes moot. ``handoff`` and
        ``spec`` stay in the signature because the argument above is written in
        terms of them, and a second term in ``_music_needs_handoff`` would need
        them here.
        """
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

    async def _separate(self: Worker, job: dict[str, Any]) -> None:
        """Split a finished take into stems, in a child that then exits.

        **The one stage in this file that acquires nothing.** ``_music`` takes
        the resident pipe, hands off against trellis and gives it all back in a
        ``finally``; this spawns a ~300 MB child, waits, and the child dies. So
        there is no ``_acquire``/``_release`` pair to mirror, and adding one
        would be ceremony around a process that holds nothing between jobs --
        see ``pipelines/separation_worker``'s docstring for why it is one-shot.

        Its artifacts land in the **source take's** directory, not this job's,
        which is what makes it a follow-up in ``asset_open``'s sense -- the rig
        and the sheets already work this way, and ``dependent_jobs`` is built on
        the same fact.
        """
        from . import rigging

        params = job["params"]
        source = str(params.get("source_job") or "")
        if not rigging.is_valid_id(source):
            # Validated, not merely non-empty: this becomes a path, and an
            # empty one makes ``job_dir`` return the assets root.
            raise RuntimeError(f"separate job has no usable source_job: {source!r}")

        spec_model = models.SEPARATION_MODELS.get(
            str(params.get("separation_model") or models.DEFAULT_SEPARATION)
        )
        if spec_model is None:
            # ``_music``'s rule for an unknown model: refused rather than
            # substituted, because a substitution records a run that never was.
            raise RuntimeError(
                f"unknown separation model: {params.get('separation_model')!r}"
            )

        source_dir = self.config.job_dir(source)
        # ``service.files.STEMS_DIR``, restated because **the queue may not
        # import the service** (``test_queue`` enforces it) -- the same reason
        # ``VECTOR_PARAMS`` lives in ``warlock/vectors.py``. One literal, and
        # ``tests/test_separation.py`` asserts the two agree.
        out_dir = source_dir / "stems"
        job_dir = self.config.job_dir(job["id"])
        job_dir.mkdir(parents=True, exist_ok=True)

        self.progress.update(
            job["id"],
            phase="separate",
            label="Splitting into stems",
            inner=0.0,
            inner_next=1.0,
            nominal=60.0,
            detail="",
        )

        spec = {
            "source": str(source_dir / "track.wav"),
            "out_dir": str(out_dir),
            "model_dir": str(self.config.t2i_model_root / spec_model.dir_name),
            "sources": list(spec_model.sources),
            "segment_seconds": spec_model.segment_seconds,
            "result_path": str(job_dir / "separate.json"),
        }

        def on_progress(fraction: float, label: str) -> None:
            self.progress.update(
                job["id"],
                phase="separate",
                label=label or "Splitting into stems",
                inner=fraction,
                inner_next=1.0,
                nominal=60.0,
                detail="",
            )

        result = await asyncio.to_thread(
            functools.partial(
                rigging.run_worker,
                spec,
                on_progress=on_progress,
                on_start=self._note_blender,
                timeout=self.config.pose_timeout,
                module="warlock.pipelines.separation_worker",
                marker="separate",
                name="Stem separation",
            )
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "separation failed")

        # ``stems.json`` **last**, as the completion gate -- ``rig.json``'s rule
        # and ``sheet.json``'s, stated identically: the four WAVs appear one at
        # a time, so their existence cannot say the set is finished.
        self._cancel.commit()
        await asyncio.to_thread(
            _write_stems_sidecar, out_dir, spec_model, result, job["id"]
        )

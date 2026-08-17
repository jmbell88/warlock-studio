"""``Worker``'s job-level bookkeeping: progress, observations, cleanup.

The methods that are about a *job* rather than about what a job makes -- what
the bar says while it runs, what the corpus learns when it finishes, what is
queued after it, and what is deleted when it is cancelled or discarded. Split
out of ``queue.py`` for the reason every mixin here is: the class had 35
methods and five unrelated subjects, and the loop core -- which is the part
with the invariants -- was buried among them.

``_discard_artifacts`` is the one with teeth. It deletes files in *another*
job's directory (a rig, a sheet and a re-texture all write into the mesh's
row), so every id it takes from params is validated before it is joined onto a
path, and the deletion list is stated per kind rather than derived from a
glob.

Two queue.py module-level names are reached through ``queue_mod`` rather than
imported: a call-time lookup keeps ``monkeypatch.setattr(queue_mod, ...)``
landing, and a module-scope import of ``.queue`` here would be circular.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from typing import TYPE_CHECKING, Any

from . import rigging

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class JobOps:
    """Job-level bookkeeping, mixed into :class:`~.queue.Worker`."""

    def _emit_progress(
        self: Worker,
        phase: str,
        label: str,
        inner: float,
        inner_next: float | None,
        nominal: float,
        fields: dict[str, Any],
    ) -> None:
        """Sink for the trellis parser. Runs on the stdout reader thread."""
        job_id = self.current_job_id
        if job_id is None:
            return  # server chatter outside a job (startup banner, idle logs)
        self.progress.update(
            job_id,
            phase=phase,
            label=label,
            inner=inner,
            inner_next=inner_next,
            nominal=nominal,
            **fields,
        )

    def _t2i_state(self: Worker, job_id: str, state: str) -> None:
        from . import queue as queue_mod

        phase, label = queue_mod.T2I_PHASES.get(state, ("t2i_load", "Preparing"))
        # Loading has no measurable inner progress; creep across the whole
        # phase so the bar still moves.
        self.progress.update(
            job_id, phase=phase, label=label, inner=0.0, inner_next=1.0,
            nominal=20.0, detail="",
        )

    def _t2i_step(self: Worker, job_id: str, step: int, total: int) -> None:
        self._step_progress(job_id, "t2i_sample", "Drawing reference image", step, total)

    async def _record_observation(self: Worker, job_id: str) -> None:
        """Machine evidence for the findings corpus, recorded at completion.

        Same rule as ``_audit_mesh``: a diagnostic must never fail a job whose
        mesh is already on disk, so any failure is logged and swallowed. The
        "did it write a row" answer is ignored here on purpose -- a job with
        neither measurement is a job there is nothing to record about, which is
        an outcome rather than an event; the tests are what read it.
        """
        from . import queue as queue_mod

        try:
            await asyncio.to_thread(queue_mod._observe_finished, self.store, job_id)
        except Exception:
            log.exception("could not record observation for job %s", job_id)

    async def _maybe_queue_rig(self: Worker, job: dict[str, Any]) -> None:
        """Honour the generate form's "rig this" checkbox, once the mesh exists.

        Queued as an ordinary follow-up job rather than run inline: rigging is
        minutes of CPU that the user should be able to cancel independently,
        and chaining it inside the generate job would hide it from the history
        and make one cancel ambiguous between two very different operations.
        """
        params = job["params"]
        if job["kind"] == "rig" or not params.get("rig"):
            return
        if not (self.config.job_dir(job["id"]) / "model.glb").exists():
            return
        rig_params = {
            "source_job": job["id"],
            "template": params.get("rig_template") or self.config.rig_template,
            "auto": True,
        }
        try:
            rig_id = await asyncio.to_thread(
                self.store.create, "rig", job["prompt"], rig_params, None
            )
        except Exception:
            # The mesh is finished and on disk. A failure to queue the optional
            # follow-up must not retroactively fail the job that produced it.
            log.exception("could not queue follow-up rig for job %s", job["id"])
            return
        log.info("queued follow-up rig %s for job %s", rig_id, job["id"])

    def _discard_artifacts(self: Worker, job: dict[str, Any]) -> None:
        """Remove what a cancelled job half-wrote -- and only that.

        Keyed on the job's kind rather than a fixed filename: a rig job's
        output is rig.glb sitting *next to* the model.glb it read, and deleting
        model.glb there would destroy the finished mesh of a different,
        successful job because the user cancelled a rig.
        """
        params = job["params"]
        if job["kind"] in (
            "rig", "sheet", "pixel_sheet", "retexture", "sprite_synthesis"
        ):
            # Both write into the *source* job's directory, not their own --
            # see _rig and _sheet. Without a source_job there is nothing they
            # could have written, so there is nothing to undo.
            # Validated, not merely non-empty: this string becomes a path, and
            # the sibling sheet branch below has always checked its own id the
            # same way. An empty or malformed source_job makes job_dir() return
            # the assets root, so the cleanup would go looking for temps in the
            # directory that holds every job.
            source = str(params.get("source_job") or "")
            if not rigging.is_valid_id(source):
                return
            job_dir = self.config.job_dir(source)
            if job["kind"] == "rig":
                # Only the temps: the worker writes those, and finalize_rig
                # only renames them into rig.glb/rig.json on success. The
                # served names may belong to an earlier, successful rig job
                # (a cancelled re-rig must not destroy the rig it corrects).
                paths = [job_dir / rigging.RIG_GLB_TMP, job_dir / rigging.RIG_JSON_TMP]
            elif job["kind"] == "retexture":
                # Only the temp, for the rig's reason exactly: the served
                # model.glb in that directory is a different, successful job's
                # mesh -- either still its original skin, or one an *earlier*
                # re-texture published. A cancel must not destroy either.
                # This job's own renders and bakes go with it below.
                paths = [job_dir / rigging.RETEXTURE_GLB_TMP]
                with contextlib.suppress(OSError):
                    shutil.rmtree(self.config.job_dir(job["id"]) / "views")
            elif job["kind"] == "sprite_synthesis":
                # Only *this* job's trio, named by the draft id it minted at
                # the door. The directory holds every earlier draft of the same
                # reference, each from a different, successful job -- a cancel
                # must not take a stranger's work with it. Nothing is normally
                # here to delete at all: the trio is written in one go at the
                # very end, after the last cancel check.
                draft_id = str(params.get("draft_id") or "")
                if not rigging.is_valid_id(draft_id):
                    return
                paths = [rigging.sprite_draft_path(job_dir, draft_id)] + [
                    rigging.sprite_draft_png_path(job_dir, draft_id, c)
                    for c in rigging.SPRITE_CANDIDATES
                ]
            else:
                sheet_id = str(params.get("sheet_id") or "")
                if not rigging.is_valid_id(sheet_id):
                    return
                if job["kind"] == "pixel_sheet":
                    # Only the restyle's own pair. The render it was derived
                    # from belongs to a different, successful job -- deleting
                    # it because a restyle was cancelled would destroy minutes
                    # of Blender for a job the user did not cancel.
                    paths = [
                        rigging.sheet_pixel_path(job_dir, sheet_id),
                        rigging.sheet_pixel_png_path(job_dir, sheet_id),
                    ]
                else:
                    paths = [
                        rigging.sheet_path(job_dir, sheet_id),
                        rigging.sheet_png_path(job_dir, sheet_id),
                    ]
        elif job["kind"] == "ground_set":
            # This job's own directory, unlike the five above -- a ground set is
            # not a derivation of anything on disk. All of it goes: ``input.png``
            # here *is* the finished 47-column atlas, and a half-published one is
            # a sheet ``use_as_tileset`` would happily slice into forty-seven
            # tiles of nothing. The staging name goes too, for the reason the
            # rig's temps do: a cancel that lands mid-rename must not leave one.
            job_dir = self.config.job_dir(job["id"])
            with contextlib.suppress(OSError):
                shutil.rmtree(job_dir / "textures")
            # Both staging names, not one. ``_publish_text`` already unlinks its
            # own temp in a ``finally``, so ``.ground.json.tmp`` is belt and
            # braces -- but the comment above promises "the staging name goes
            # too" for a step that stages *twice*, and a list that covers one of
            # them is the kind of asymmetry a later reader trusts.
            paths = [
                job_dir / "input.png",
                job_dir / ".input.png.tmp",
                job_dir / "ground.json",
                job_dir / ".ground.json.tmp",
            ]
        else:
            # Both halves of the contract: model.glb is what the user would
            # see, source.glb is what it was derived from. Leaving the source
            # behind would let a cancelled job be re-optimized back into
            # existence.
            job_dir = self.config.job_dir(job["id"])
            paths = [
                job_dir / "model.glb",
                job_dir / "source.glb",
                # The remesh loop's scratch copies of both, held only while a
                # retry is in flight. A cancel lands mid-loop, so they are as
                # much this run's leftovers as the two above.
                job_dir / "best.glb",
                job_dir / "best.source.glb",
                # Both derived from ref.png/input.png by this run. ref.png
                # itself stays: it is a user-supplied input, and keeping it is
                # what makes "cancel, tweak, resubmit" work.
                job_dir / "reference.png",
                job_dir / "control.png",
            ]
        for path in paths:
            with contextlib.suppress(OSError):
                path.unlink()

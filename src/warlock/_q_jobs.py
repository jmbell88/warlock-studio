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

from . import followups, rigging

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

    async def _record_followup_failure(
        self: Worker,
        job_id: str,
        followup: str,
        error: BaseException | str,
    ) -> None:
        """Persist an optional follow-up failure without failing its parent."""
        try:
            saved = await asyncio.to_thread(
                followups.persist, self.store, job_id, followup, error
            )
            if not saved:
                log.warning(
                    "could not record %s follow-up failure; parent %s is gone",
                    followup,
                    job_id,
                )
        except Exception:
            # This is the fallback for the fallback. The parent artifact is
            # complete, so a second database failure remains diagnostic only.
            log.exception(
                "could not record %s follow-up failure for job %s",
                followup,
                job_id,
            )

    async def _maybe_queue_rig(self: Worker, job: dict[str, Any]) -> None:
        """Honour the generate form's "rig this" checkbox, once the mesh exists.

        Queued as an ordinary follow-up job rather than run inline: rigging is
        minutes of CPU that the user should be able to cancel independently,
        and chaining it inside the generate job would hide it from the history
        and make one cancel ambiguous between two very different operations.
        """
        params = job["params"]
        # The stage guard is not defensive; it is the only thing keeping this
        # off a reference. Troupe's door sets ``rig`` on the *reference* row --
        # it describes the mesh the gate will promote that reference into, not
        # the reference itself -- so without it this runs on every Troupe
        # reference completion, where a ``model.glb`` cannot exist because the
        # mesh does not exist until the user approves the drawing. That was a
        # silent no-op until the failure record below was added, and then it
        # became a permanent, false "the generated mesh artifact is missing" on
        # every character anyone started. ``_maybe_queue_sprite_sheet`` and
        # ``_maybe_queue_charsheet`` have carried this guard from the start;
        # this is the third of the three.
        if job["kind"] == "rig" or job["stage"] != "model" or not params.get("rig"):
            return
        if not (self.config.job_dir(job["id"]) / "model.glb").exists():
            await self._record_followup_failure(
                job["id"], "rig", "The generated mesh artifact is missing."
            )
            return
        rig_params = {
            "source_job": job["id"],
            "template": params.get("rig_template") or self.config.rig_template,
            "auto": True,
        }
        if params.get("rig_joints"):
            # Only where the job asked for it, so an ordinary auto-rig is the
            # byte-identical rig it always was. Troupe asks: its reference is a
            # constrained T-pose, and measuring joints off the vertex cloud is
            # what stops the A-pose template mis-fitting one.
            rig_params["joints"] = params["rig_joints"]
        try:
            rig_id = await asyncio.to_thread(
                self.store.create, "rig", job["prompt"], rig_params, None
            )
        except Exception as exc:
            # The mesh is finished and on disk. A failure to queue the optional
            # follow-up must not retroactively fail the job that produced it.
            log.exception("could not queue follow-up rig for job %s", job["id"])
            await self._record_followup_failure(job["id"], "rig", exc)
            return
        log.info("queued follow-up rig %s for job %s", rig_id, job["id"])

    async def _maybe_queue_sprite_sheet(self: Worker, job: dict[str, Any]) -> None:
        """Honour the Create form's Sheet/Sprite output, once the character
        exists.

        ``_maybe_queue_rig``'s shape and its reasons. Queued as an ordinary
        follow-up rather than run inline because it is two more generations the
        user should be able to cancel on their own, and because chaining it
        inside the reference job would hide it from the history and make one
        cancel ambiguous between "stop drawing my character" and "stop
        imagining its other sides".

        The character is a row in its own right either way. That is the point
        of the two-step shape: a sheet the user hates still leaves them the
        drawing it was made from, to reroll or edit or synthesise again with
        different options.
        """
        from . import rigging
        from .pipelines import spritesynth

        params = job["params"]
        block = params.get("sprite_sheet")
        if not block or job["stage"] != "reference":
            return
        if not (self.config.job_dir(job["id"]) / "input.png").exists():
            await self._record_followup_failure(
                job["id"],
                "sprite_synthesis",
                "The generated reference image is missing.",
            )
            return
        # Derived from the character's own seed rather than drawn: this worker
        # has no RNG and should not grow one -- every seed in this codebase
        # comes from a door or from arithmetic over one that did. It also makes
        # the whole chain reproducible from the single seed the user can lock.
        # ``is None`` rather than ``or``: seed 0 is a legal pinned seed, and
        # the truthiness spelling skipped it -- breaking the one-seed
        # reproducibility this comment promises for exactly that corner.
        raw = params.get("reference_seed")
        if raw is None:
            raw = params.get("seed")
        base = int(raw) if raw is not None else 0
        seed_a = spritesynth.candidate_seed(base, "a")
        seed_b = spritesynth.candidate_seed(base, "b")
        sheet_params = {
            "source_job": job["id"],
            "sheet_type": block.get("sheet_type"),
            "logical_size": block.get("logical_size"),
            "colors": block.get("colors"),
            "seed_a": seed_a,
            "seed_b": seed_b,
            # Minted here because the door that normally mints it is not on this
            # path. ``_discard_artifacts`` names this job's drafts by it, so a
            # cancel deletes exactly its own trio and no earlier draft of the
            # same character. Through ``rigging.new_id`` rather than a second
            # copy of its two-line body: the format is that module's, and
            # ``check_sprite_draft_id`` validates against it.
            "draft_id": rigging.new_id(),
            # ``service.sprites.SPRITE_BASE_MODEL``, restated because the queue
            # may not import the service -- and *pinned*, deliberately not
            # inherited from the character's row. A synthesis is a pixel-art
            # restyle: its identity is the SDXL-fitted pixel-art LoRA, which is
            # why the door writes this constant rather than a choice. Carrying
            # the reference's ``base_model`` across looks like keeping the
            # character's style and is the opposite -- a character drawn on
            # klein would queue a sheet whose base cannot take the LoRA, so the
            # worker's tolerance would draw both candidates bare and say so only
            # in the log, from a button that offered two pixel-art sheets. It
            # would also mean the door admitted the follow-up against one base's
            # weights and VRAM and the job ran on another's.
            "base_model": "sdxl_cfg",
        }
        for identity_key in ("asset_type", "asset_intent"):
            if params.get(identity_key):
                sheet_params[identity_key] = params[identity_key]
        try:
            sheet_id = await asyncio.to_thread(
                self.store.create, "sprite_synthesis", job["prompt"], sheet_params, None
            )
        except Exception as exc:
            # The character is finished and on disk. A failure to queue the
            # optional follow-up must not retroactively fail the job that
            # produced it -- the rig's rule, verbatim.
            log.exception("could not queue follow-up sprite sheet for job %s", job["id"])
            await self._record_followup_failure(
                job["id"], "sprite_synthesis", exc
            )
            return
        log.info("queued follow-up sprite sheet %s for job %s", sheet_id, job["id"])

    async def _maybe_queue_charsheet(self: Worker, job: dict[str, Any]) -> None:
        """Honour a Troupe request, once the character's mesh exists.

        ``_maybe_queue_rig``'s shape and its reasons, one link further down the
        chain. The block rides on the *reference* job and survives the promote
        gate (``promote_to_model`` copies everything that is not derived or
        conditioning), so this fires on the model-stage job the gate produced
        -- which is the job that has a ``model.glb`` for the sheet to depict.

        **Queued behind the rig, not instead of it.** ``_maybe_queue_rig`` runs
        immediately before this one on the same finished job, and the queue is
        serial and FIFO, so the rig row is claimed and finished before the
        charsheet row is claimed. That ordering is the whole reason this method
        does not rig anything itself: a character sheet needs a rig, the app
        already has a rig job with its own progress and its own cancel, and
        chaining a second rigger inside here would be a second implementation
        of the thing that failed.

        A missing rig is still refused in ``_charsheet`` rather than assumed:
        the user can cancel the rig row on its own, which is exactly what the
        two-row shape is for.
        """
        params = job["params"]
        block = params.get("troupe")
        if not block or job["kind"] in ("charsheet", "rig") or job["stage"] != "model":
            return
        if not (self.config.job_dir(job["id"]) / "model.glb").exists():
            await self._record_followup_failure(
                job["id"], "charsheet", "The generated mesh artifact is missing."
            )
            return
        if not params.get("rig"):
            # The door sets it, so this is a row whose rig flag was edited
            # away between the reference and the promotion. Saying so beats
            # queueing a sheet that will refuse an hour later.
            log.warning(
                "job %s asks for a character sheet but not a rig; skipping the sheet",
                job["id"],
            )
            await self._record_followup_failure(
                job["id"], "charsheet", "The character sheet requires an automatic rig."
            )
            return
        sheet_params = {
            "source_job": job["id"],
            # Minted here because the door that normally mints it is not on
            # this path -- ``_maybe_queue_sprite_sheet``'s draft id, exactly.
            # ``_discard_artifacts`` names this job's atlas by it, so a cancel
            # deletes its own sheet and no earlier one of the same character.
            "sheet_id": rigging.new_id(),
            "template": params.get("rig_template") or self.config.rig_template,
            "logical_size": block.get("logical_size"),
            "colors": block.get("colors"),
            "outline": block.get("outline"),
            "reduce_mode": block.get("reduce_mode"),
            "dither": bool(block.get("dither")),
            "palette": block.get("palette") or "",
            "name": "",
            # Resolved at the reference door and snapshotted there. Copy the
            # value, not today's defaults, so approval/rerun cannot silently
            # change what an older request means.
            "layout": block.get("layout"),
        }
        try:
            sheet_id = await asyncio.to_thread(
                self.store.create, "charsheet", job["prompt"], sheet_params, None
            )
        except Exception as exc:
            # The mesh is finished and on disk. A failure to queue the optional
            # follow-up must not retroactively fail the job that produced it --
            # the rig's rule, verbatim.
            log.exception("could not queue follow-up character sheet for job %s", job["id"])
            await self._record_followup_failure(job["id"], "charsheet", exc)
            return
        log.info("queued follow-up character sheet %s for job %s", sheet_id, job["id"])

    async def _maybe_queue_sheet_after_rig(self: Worker, job: dict[str, Any]) -> None:
        """The second half of "send this mesh to Troupe".

        ``service.troupe.send_to_troupe`` mints one rig row carrying a nested
        ``troupe_sheet`` block when the mesh it was handed is not rigged yet;
        this is what mints the sheet on the finished rig. The fourth
        ``_maybe_queue_*`` and the same shape as the three above -- one press
        mints one row, and the follow-up is minted by the mechanism that
        already mints rigs and sheets, so both rows cancel independently and
        the worker stays four ordinary jobs rather than an orchestrator.

        Guarded on ``kind == "rig"``, which is what keeps it off
        ``_maybe_queue_charsheet``'s path: that one fires on a *model*-stage
        row carrying ``troupe``, and this on a rig row carrying
        ``troupe_sheet``. Two markers because they are two claims -- see
        ``send_to_troupe``.

        The rig is re-checked rather than assumed: a user can cancel the rig
        row on its own, which is exactly what the two-row shape is for.
        Everything else was refused at the door, and the block is a snapshot
        of the settings that were on screen when the button was pressed.
        """
        if job["kind"] != "rig":
            return
        params = job["params"]
        block = params.get("troupe_sheet")
        if not block:
            return
        source_job = str(params.get("source_job") or "")
        if not source_job:
            return
        source_dir = self.config.job_dir(source_job)
        if not (source_dir / "rig.glb").exists():
            await self._record_followup_failure(
                source_job, "charsheet", "The rig artifact is missing."
            )
            return
        sheet_params = {
            "source_job": source_job,
            # Minted here for ``_maybe_queue_charsheet``'s reason: the door
            # that normally mints it is not on this path, and
            # ``_discard_artifacts`` names this job's atlas by it, so a cancel
            # deletes its own sheet and no earlier one of the same character.
            "sheet_id": rigging.new_id(),
            **{key: value for key, value in block.items() if key != "sheet_id"},
        }
        try:
            sheet_id = await asyncio.to_thread(
                self.store.create, "charsheet", job["prompt"], sheet_params, None
            )
        except Exception as exc:
            # The rig is finished and on disk. A failure to queue the optional
            # follow-up must not retroactively fail the job that produced it --
            # the rule the three above share, verbatim.
            log.exception("could not queue character sheet after rig %s", job["id"])
            await self._record_followup_failure(source_job, "charsheet", exc)
            return
        log.info("queued character sheet %s after rig %s", sheet_id, job["id"])

    def _discard_artifacts(self: Worker, job: dict[str, Any]) -> None:
        """Remove what a cancelled job half-wrote -- and only that.

        Keyed on the job's kind rather than a fixed filename: a rig job's
        output is rig.glb sitting *next to* the model.glb it read, and deleting
        model.glb there would destroy the finished mesh of a different,
        successful job because the user cancelled a rig.
        """
        params = job["params"]
        if job["kind"] in (
            "rig", "sheet", "charsheet", "pixel_sheet", "retexture", "sprite_synthesis"
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
                    if job["kind"] == "charsheet":
                        # The un-quantised render the pixel-art pass reads
                        # between the pack and the publish. A cancel lands
                        # between them often -- it is the longest gap in the
                        # job -- and it is this run's leftover by construction,
                        # named off the sheet id it minted at the door.
                        png = rigging.sheet_png_path(job_dir, sheet_id)
                        paths.append(png.with_name(f".{png.name}.render"))
                        paths.append(png.with_name(f".{png.name}.tmp"))
        elif job["kind"] == "tile_sheet":
            # This job's own directory, unlike the five above -- a tile sheet is
            # not a derivation of anything on disk, and ``input.png`` here *is*
            # the finished sheet. A half-published one is sixty-four tiles of
            # nothing that ``use_as_tileset`` would happily slice. Both staging
            # names go too, for the reason the rig's temps do: a cancel that
            # lands mid-rename must not leave one.
            #
            # ``ref.png`` is deliberately **not** on the list. The door wrote it
            # before the row existed, so it is an *input* the user supplied
            # rather than something this run produced -- and a cancelled sheet
            # whose reference survived is a row that can be re-run without the
            # user finding the file again. It goes with the directory when the
            # job is pruned, which is where an input belongs.
            job_dir = self.config.job_dir(job["id"])
            paths = [
                job_dir / "input.png",
                job_dir / ".input.png.tmp",
                job_dir / "sheet.png",
                job_dir / "sheet.json",
                job_dir / ".sheet.json.tmp",
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

"""``Worker``'s Blender stages: rig, deformation QA, sprite sheet.

The three jobs that run bpy out-of-process, plus the render-and-pack step two
of them share. Split out of ``queue.py`` for the reason every mixin here is:
the class had 35 methods and five unrelated subjects, and the loop core --
which is the part with the invariants -- was buried among them.

What these share, and why they are one module: none of them touches a resident
model or the VRAM handoff at all. They are CPU and an EEVEE render, and the
serial queue is the only thing keeping them from overlapping a trellis run.
They also all write into *another* job's directory -- a rig, its QA sheet and
its sprite sheets belong to the mesh they depict, not to the job that asked for
them -- which is what makes the staged-write and completion-marker rules in
here load-bearing rather than tidy.

``bpy`` never runs in this process: ``rigging.run_worker`` spawns
``pipelines/blender_worker.py``, and ``_note_blender`` (which stays on
``Worker``) holds the live ``Popen`` so a cancel can kill it.

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

from . import rigging
from .pipelines import pose2d

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class RigOps:
    """The Blender stages, mixed into :class:`~.queue.Worker`."""

    def _wants_landmarks(
        self: Worker, source_dir: Path, template: str, params: dict[str, Any]
    ) -> bool:
        """Whether reading joints off the reference image is worth attempting.

        Every one of these is a case where the answer could not be used, not a
        case where it might be poor -- how *good* a detection is is decided by
        pose2d's own sanity gates, which see the landmarks. Cheap checks first;
        ``available`` stats a directory and goes last because it is the only
        one that touches the disk outside the job's own.
        """
        return (
            template in pose2d.POSE_FIT_TEMPLATES
            and self.config.pose_fit
            # Adjust-joints. The user has already overruled one fit; running a
            # detector to produce a second one they would also overrule is pure
            # cost.
            and not params.get("bones")
            # An imported or hand-modelled mesh has no reference to read.
            and (source_dir / "input.png").exists()
            and pose2d.available(self.config)
        )

    async def _rig(self: Worker, job: dict[str, Any]) -> None:
        """Fit a skeleton to a finished job's mesh, out-of-process in Blender.

        A rig job is a job in its own right, but its artifacts land in the
        *source* job's directory alongside model.glb -- the rig belongs to the
        mesh, not to the request that produced it, and a UI that had to chase a
        second job id to find rig.glb would be the wrong shape. params carries
        source_job so the history list can attach it to the parent card.

        Nothing here touches the GPU or either resident model, so the VRAM
        handoff in _generate is not involved. The queue being serial is what
        keeps a multi-minute weighting solve from overlapping a trellis run.
        """
        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        # Validated before it becomes a path, the same way sheet_id already is.
        # This is the directory the Blender worker is pointed at and writes its
        # temps into, and job_dir("") is the assets root -- the whole reason
        # check_job_id and pose_path exist is to keep params off the filesystem
        # unchecked.
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        source_dir = self.config.job_dir(source_id)
        template = str(params.get("template") or self.config.rig_template)

        self.progress.update(
            job_id, phase="rig", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=12.0, detail="",
        )

        def on_progress(frac: float, label: str) -> None:
            # The worker's fractions are already the whole bar (PHASES_RIG),
            # and it emits them from its own thread-free subprocess; this
            # runs on the to_thread worker, which ProgressBus locks for.
            self.progress.update(
                job_id, phase="rig", label=label, inner=frac,
                inner_next=min(frac + 0.1, 1.0), nominal=30.0, detail="",
            )

        # Blocking (a PIL decode, a silhouette measurement and a torch forward
        # on the CPU), so it goes through to_thread exactly as run_worker below
        # does. Skipped entirely -- not merely ignored -- when any gate is
        # closed, because the point of the gates is that the model is never
        # loaded on a job that could not use its answer.
        from . import queue as queue_mod

        landmarks, fit = None, None
        if self._wants_landmarks(source_dir, template, params):
            landmarks, fit = await asyncio.to_thread(
                queue_mod._landmark_bones, self.config, source_dir, template
            )

        spec = rigging.rig_spec(
            source_dir,
            template,
            params.get("bones"),
            template_bones=landmarks,
            fit=fit,
            # Where the job asked for it -- Troupe does, because its reference
            # is a constrained T-pose and the shipped humanoid template is an
            # A-pose, which fits a T-pose mesh badly enough to skin the arms to
            # the chest. ``rig_spec`` documents where this sits in the order of
            # preference: below a user correction, above the template.
            joints=params.get("joints") or None,
        )
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    rigging.run_worker,
                    spec,
                    on_progress=on_progress,
                    on_start=self._note_blender,
                    timeout=self.config.rig_timeout,
                )
            )
            # A cancel that landed after the solve finished must not publish
            # the artifacts of a job about to be recorded as cancelled -- the
            # finally below throws the temps away instead.
            if self._cancel is not None and self._cancel.event.is_set():
                # Nothing was published, so nothing gets recorded either: the
                # weighting/bone_count of a discarded rig must not end up in
                # the params of a job recorded as cancelled.
                return
            await asyncio.to_thread(rigging.finalize_rig, source_dir)
        finally:
            # No-op on success (finalize renamed them away); on failure or
            # cancel it removes the half-written temps and never touches the
            # served rig.glb/rig.json, which may belong to an earlier,
            # successful rig job.
            await asyncio.to_thread(rigging.discard_rig_temps, source_dir)
        # Recorded on the rig job so the history row can say "envelope
        # weights" without the UI having to fetch rig.json for every card.
        params["weighting"] = result.get("weighting")
        # Beside it rather than only in rig.json: "envelope" is a degraded
        # outcome and the question it raises is immediately "why", which the
        # inspector cannot answer without opening a file per card.
        params["weighting_reason"] = result.get("weighting_reason")
        # How many joints the fitted skeleton ended up with (an int -- the
        # worker returns len(bones)), for the same reason as the two above: it
        # is in rig.json, and a card that wants to say "17 bones" should not
        # have to open a file to do it. None only from a worker whose result
        # predates the key.
        params["bone_count"] = result.get("bones")
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info(
            "rigged job %s from %s: %s weights, %s bones",
            job_id, source_id, result.get("weighting"), result.get("bones"),
        )

        # Log-and-swallow, the ``_audit_mesh`` rule: the rig is already
        # published, and a review sheet that could not be rendered must not
        # fail the job whose artifact is on disk. Recorded with merge_params
        # rather than a second set_params off the copy above, because the two
        # writes are a read-modify-write sequence the lock does not cover.
        try:
            qa = await self._deform_qa(job_id, source_id, source_dir, template)
        except Exception:
            log.exception("deformation QA sheet failed for job %s", job_id)
            qa = None
        if qa is not None:
            await asyncio.to_thread(self.store.merge_params, job_id, {"deform_qa": qa})

    async def _deform_qa(
        self: Worker, job_id: str, source_id: str, source_dir: Path, template: str
    ) -> dict[str, Any] | None:
        """Render the deformation battery for a freshly rigged mesh, or None.

        Human-reviewable and nothing more: this scores nothing and gates
        nothing. It exists because a weighting method that *reports* success
        says only that the solve produced numbers, and the way to see whether
        those numbers deform the mesh sensibly is to look at it bent.

        The poses are template data (``rigging.deform_battery``) and the render
        is the ordinary sheet path -- ``sheetlib.plan``/``pack``/``sidecar``
        around ``op_sheet`` -- because a second renderer is a second set of
        camera conventions to keep in agreement with the first. The output is
        not a sheet in ``sheets/``: it belongs to the rig, so it must not join
        the user's sheet list, count against MAX_SHEETS, or be deleted by a
        sheet delete.
        """
        from . import queue as queue_mod
        from .pipelines import sheet as sheetlib

        poses = rigging.deform_battery(template)
        rig_glb = source_dir / "rig.glb"
        if not poses or not self.config.deform_qa or not rig_glb.exists():
            return None

        layout = sheetlib.plan(
            poses,
            frame_size=queue_mod.DEFORM_QA_FRAME_SIZE,
            elevation=sheetlib.DEFAULT_ELEVATION,
            # Lit, where a sprite sheet defaults to flat: a collapsed elbow or
            # a candy-wrapper twist is a shading artefact, and unlit albedo is
            # exactly the render that hides it.
            lighting="lit",
            yaws=queue_mod.DEFORM_QA_YAWS,
        )
        bones = {(p["id"], 0): p["bones"] for p in poses}
        cells = [
            {
                "index": c.index,
                "yaw": c.yaw,
                "pose": c.pose,
                "frame": c.frame,
                "bones": bones.get((c.pose, c.frame)) or {},
            }
            for c in layout.cells
        ]

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="rig", label=f"Deformation QA: {label}", inner=frac,
                inner_next=min(frac + 0.1, 1.0), nominal=20.0, detail="",
            )

        png = rigging.rig_qa_png_path(source_dir)
        # Both halves of the QA sheet are staged and renamed in. The sidecar is
        # the completion marker (files.ready keys on the .json, not the .png),
        # so on a *re-rig* the previous run's sidecar already says "ready" while
        # this run's atlas is being packed over the served PNG -- a reader in
        # that window gets a torn sheet that nothing marks as suspect. Renaming
        # both keeps the marker's promise true at every instant.
        png_tmp = png.with_name(f".{png.name}.tmp")
        qa_json = rigging.rig_qa_path(source_dir)
        json_tmp = qa_json.with_name(f".{qa_json.name}.tmp")
        try:
            result, _trims = await self._render_sheet_atlas(
                rig_glb,
                layout,
                cells,
                prefix="warlock-rigqa-",
                on_progress=on_progress,
                pack_target=png_tmp,
            )
            await asyncio.to_thread(os.replace, png_tmp, png)
        finally:
            with contextlib.suppress(OSError):
                png_tmp.unlink(missing_ok=True)

        pivot = result.get("pivot") if isinstance(result, dict) else None
        meta = sheetlib.sidecar(
            layout,
            sheet_id="rig_qa",
            source_job=source_id,
            image=png.name,
            created=time.time(),
            name="deformation QA",
            pivot=(float(pivot[0]), float(pivot[1])) if pivot else None,
        )
        # Written last, so it is the completion marker the file rules key on --
        # the same ordering rig.json has for rig.glb.
        try:
            await asyncio.to_thread(
                json_tmp.write_text,
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )
            await asyncio.to_thread(os.replace, json_tmp, qa_json)
        finally:
            with contextlib.suppress(OSError):
                json_tmp.unlink(missing_ok=True)
        log.info(
            "rendered deformation QA for job %s: %d pose(s) x %d view(s)",
            source_id, layout.rows, layout.columns,
        )
        return {"poses": [str(p["name"]) for p in poses], "cells": len(cells)}

    async def _sheet(self: Worker, job: dict[str, Any]) -> None:
        """Render a pose x direction sprite sheet for a finished job's mesh.

        Like a rig, the output belongs to the mesh and lands in the *source*
        job's directory (``sheets/<sheet_id>.png`` plus its sidecar). The grid
        and the packing are pure host-side code in ``pipelines/sheet.py``;
        Blender only ever renders one square transparent frame per cell into a
        scratch directory that goes away either way.

        CPU and an EEVEE render, so nothing here touches the resident models or
        the VRAM handoff -- the serial queue is what keeps it from overlapping
        a trellis run.
        """
        from . import queue as queue_mod
        from .pipelines import sheet as sheetlib

        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        # Validated before it becomes a path, the same way sheet_id already is.
        # This is the directory the Blender worker is pointed at and writes its
        # temps into, and job_dir("") is the assets root -- the whole reason
        # check_job_id and pose_path exist is to keep params off the filesystem
        # unchecked.
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        source_dir = self.config.job_dir(source_id)
        sheet_id = str(params.get("sheet_id") or rigging.new_id())

        records = []
        for pose_id in params.get("poses") or []:
            record = await asyncio.to_thread(rigging.read_pose, source_dir, str(pose_id))
            if record is None:
                # Deleted between queueing and running. Failing beats quietly
                # rendering a sheet with a row missing that the user asked for.
                raise RuntimeError(f"pose {pose_id} no longer exists")
            records.append(record)

        clip = params.get("clip")
        if clip:
            # Rebuilt from the same two poses rather than shipped in params: the
            # host is the single place a grid or a clip is decided (see
            # pipelines/sheet.py), and storing the expanded frames would be a
            # second copy that could disagree with it.
            ends = [
                await asyncio.to_thread(rigging.read_pose, source_dir, str(clip[k]))
                for k in ("from", "to")
            ]
            if any(e is None for e in ends):
                raise RuntimeError("a pose in this clip no longer exists")
            records = sheetlib.interpolate(ends[0], ends[1], int(clip["frames"]))

        layout = sheetlib.plan(
            records,
            frame_size=int(params.get("frame_size", sheetlib.DEFAULT_FRAME_SIZE)),
            elevation=float(params.get("elevation", sheetlib.DEFAULT_ELEVATION)),
            lighting=str(params.get("lighting", "flat")),
            yaws=int(params.get("yaws", sheetlib.DEFAULT_YAWS)),
        )
        # A rig if there is one, so poses can apply; otherwise the plain mesh,
        # which is all an unrigged prop's turnaround needs.
        source_glb = source_dir / "rig.glb"
        if not source_glb.exists():
            source_glb = source_dir / "model.glb"
        if not source_glb.exists():
            raise RuntimeError("source job has no mesh to render")

        # Keyed by (pose id, frame) rather than pose id: every frame of a clip
        # shares an id by construction, and keying on the id alone would render
        # frame 0 in every row of the clip.
        bones = {(r.get("id"), r.get("frame", 0)): r["bones"] for r in records}
        # A clip authored as deltas from rest rather than in the pose editor's
        # node frame says so per record; see ``blender_worker.POSE_SPACES``.
        spaces = {
            (r.get("id"), r.get("frame", 0)): r["space"] for r in records if r.get("space")
        }
        # The why lives on _sheet_root_offsets, which is module-level so a test
        # can drive the map-building without a Worker or a Blender fake. The
        # read stays here: it is I/O, and only worth doing when some record
        # actually carries an offset.
        roots: dict[tuple[Any, int], list[float]] = {}
        root_bone: Any = None
        if any(float(v) for r in records for v in (r.get("root_translation") or ())):
            rig_meta = await asyncio.to_thread(rigging.read_rig, source_dir)
            roots, root_bone = queue_mod._sheet_root_offsets(records, rig_meta)
        cells = []
        for c in layout.cells:
            cell: dict[str, Any] = {
                "index": c.index,
                "yaw": c.yaw,
                "pose": c.pose,
                "frame": c.frame,
                "bones": bones.get((c.pose, c.frame)) or {},
            }
            # Only where the record says so, so an ordinary pose row is the
            # byte-identical cell it always was and the worker's default holds.
            space = spaces.get((c.pose, c.frame))
            if space:
                cell["pose_space"] = space
            offset = roots.get((c.pose, c.frame))
            if offset:
                cell["root_bone"] = root_bone
                cell["root_offset"] = offset
            cells.append(cell)

        self.progress.update(
            job_id, phase="sheet", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=12.0, detail=f"{len(cells)} frames",
        )

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="sheet", label=label, inner=frac,
                inner_next=min(frac + 0.05, 1.0), nominal=20.0, detail="",
            )

        png = rigging.sheet_png_path(source_dir, sheet_id)

        def before_pack() -> None:
            self.progress.update(
                job_id, phase="pack", label="Packing sheet", inner=0.0,
                inner_next=1.0, nominal=3.0, detail="",
            )

        result, trims = await self._render_sheet_atlas(
            source_glb,
            layout,
            cells,
            prefix="warlock-sheet-",
            on_progress=on_progress,
            pack_target=png,
            before_pack=before_pack,
        )

        # The sidecar is written last and is what list_sheets treats as the
        # completion marker, the same way rig.json is for a rig.
        #
        # A worker that reported no pivot (an older result, or a fake in a test)
        # falls back to the cell's centre-bottom rather than failing the sheet:
        # the atlas is already on disk and correct, and the pivot is metadata.
        pivot = result.get("pivot") if isinstance(result, dict) else None
        meta = sheetlib.sidecar(
            layout,
            sheet_id=sheet_id,
            source_job=source_id,
            image=png.name,
            created=time.time(),
            name=str(params.get("name") or ""),
            pivot=(float(pivot[0]), float(pivot[1])) if pivot else None,
            trims=trims,
        )
        await asyncio.to_thread(
            queue_mod._publish_text,
            rigging.sheet_path(source_dir, sheet_id),
            json.dumps(meta, indent=2),
        )
        params["sheet_id"] = sheet_id
        params["cells"] = len(cells)
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info(
            "rendered sheet %s for job %s: %dx%d cells at %dpx",
            sheet_id, source_id, layout.columns, layout.rows, layout.frame_size,
        )

    async def _render_sheet_atlas(
        self: Worker,
        glb: Path,
        layout: Any,
        cells: list[dict[str, Any]],
        *,
        prefix: str,
        on_progress: Any,
        pack_target: Path,
        before_pack: Any = None,
    ) -> tuple[Any, Any]:
        """Render one cell per frame into a scratch directory and pack them.

        The whole of what ``_sheet`` and ``_deform_qa`` share: a temporary
        directory, ``rigging.sheet_spec``, one ``run_worker`` under the sheet
        timeout, and ``sheetlib.pack`` over the frames it wrote. Blender only
        ever renders one square transparent frame per cell into a scratch
        directory that goes away either way; the grid and the packing are pure
        host-side code.

        The two callers differ only in what they render, where they pack it and
        what they say while it happens, so those are the parameters.
        ``pack_target`` is a staging name for ``_deform_qa`` (which renames it
        onto the served PNG) and the served name itself for ``_sheet``, whose
        atlas nothing is serving until its sidecar exists.
        """
        from .pipelines import sheet as sheetlib

        with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
            frames_dir = Path(tmp)
            spec = rigging.sheet_spec(
                glb,
                frames_dir,
                cells,
                frame_size=layout.frame_size,
                elevation=layout.elevation,
                lighting=layout.lighting,
            )
            result = await asyncio.to_thread(
                functools.partial(
                    rigging.run_worker,
                    spec,
                    on_progress=on_progress,
                    on_start=self._note_blender,
                    timeout=self.config.sheet_timeout,
                )
            )
            if before_pack is not None:
                before_pack()
            frames = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
            trims = await asyncio.to_thread(sheetlib.pack, layout, frames, pack_target)
        return result, trims

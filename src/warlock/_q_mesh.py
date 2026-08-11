"""``Worker``'s mesh post-processing: retarget, ground, rank, measure.

Everything that happens to a mesh *after* the reconstruction is on disk, and
nothing that happens before it. Split out of ``queue.py`` for the reason every
mixin here is: the class had 35 methods and five unrelated subjects, and the
worker's loop core -- which is the part with the invariants -- was buried in
the middle of them.

These five share a rule the rest of the worker does not: **none of them may
fail the job**. The reconstruction is on disk and usable by the time any of
them runs, so a lost triangle budget costs file size, a lost transform costs a
manual fixup, and a lost measurement costs a badge -- while raising would cost
the user the mesh. Every one of them logs and returns instead.

No module-scope import of ``.queue`` (that would be circular) and no queue
module-level names are referenced, so there is no ``queue_mod`` indirection
here -- the logger is this module's own.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class MeshPostOps:
    """Mesh post-processing, mixed into :class:`~.queue.Worker`."""

    def _drop_surface_artifacts(self: Worker, source_dir: Path) -> None:
        """Delete the exports that carry the old skin, and only those.

        ``model.stl`` and ``collision.glb`` are geometry with no texture in
        them at all, and a re-texture changes no geometry -- deleting them would
        cost the user a re-export to produce a byte-identical file. The rig, its
        poses and its sheets are the same argument one level up and are the
        subject of a written assertion elsewhere: a rig references geometry, not
        pixels.

        Under each artifact's own lock where one is available. The worker holds
        no service, so ``artifact_lock`` is injected by ``studio.runtime`` and
        falls back to no lock at all -- which is the pre-existing behaviour of
        every other write the worker makes to a model.glb, and is why
        ``optimize_job`` refuses a job that is queued or running.
        """
        from .pipelines import retexture

        for name in retexture.SURFACE_DERIVED:
            with self.artifact_lock(source_dir.name, name), contextlib.suppress(OSError):
                (source_dir / name).unlink()

    async def _optimize(
        self: Worker, job_id: str, source: Path, dest: Path, params: dict[str, Any]
    ) -> None:
        """Retarget the reconstruction to the job's triangle budget.

        Before the transform, not after: gltfpack rewrites the node graph, and
        running it over an already-grounded model would discard the transform
        node normalize_glb inserted. Optimizing first and transforming second is
        the only ordering where both survive.

        A failure here is not fatal. The reconstruction is on disk and usable;
        losing the budget costs the user file size, and failing the job would
        cost them the mesh.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        from .pipelines import optimize

        try:
            budget = optimize.resolve(
                str(params.get("profile") or self.config.mesh_profile),
                params.get("custom_triangles"),
            )
        except ValueError:
            log.warning("job %s has an unusable profile; shipping raw", job_id)
            budget = None
        self.progress.update(
            job_id, phase="optimize", label="Optimizing mesh", inner=0.0,
            inner_next=1.0, nominal=4.0, detail=f"{budget:,} tris" if budget else "raw",
        )
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    optimize.run,
                    source,
                    dest,
                    target_triangles=budget,
                    exe=self.config.gltfpack_exe,
                )
            )
        except Exception:
            log.exception("optimize failed for job %s; shipping the reconstruction", job_id)
            # Staged, not copyfile: on the /optimize route dest is a done
            # job's model.glb that a concurrent GET may be serving.
            await asyncio.to_thread(optimize.staged_copy, source, dest)
            return
        params["optimize"] = result
        await asyncio.to_thread(self.store.set_params, job_id, params)

    async def _apply_scale(
        self: Worker, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> None:
        """Resize the finished mesh, centre it in X/Z and sit it on the floor.

        The grounding half runs even when no ``size_m`` was requested -- a
        pivot at the reconstruction volume's centre is a manual fixup on every
        import regardless of whether the asset also needed rescaling.

        Runs after the GLB is on disk and holds no GPU memory, so it sits
        outside the VRAM handoff entirely. A cancel that landed during the
        trellis stage skips it -- the artifact is about to be deleted anyway.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        size_m = params.get("size_m")
        self.progress.update(
            job_id,
            phase="scale",
            label="Scaling and grounding",
            inner=0.0,
            inner_next=1.0,
            nominal=2.0,
            detail=f"{size_m} m" if size_m else "grounding",
        )
        # Imported here, not at module scope: postprocess pulls in trimesh,
        # which app startup should not pay for.
        from .pipelines import postprocess

        try:
            transform = await asyncio.to_thread(
                postprocess.normalize_glb, glb_path, float(size_m) if size_m else None
            )
        except Exception:
            # Grounding runs on every job now, not just sized ones, so a mesh
            # trimesh cannot parse must not fail a job whose GLB is already on
            # disk -- the same rule _audit_mesh and the report follow. The
            # report's achieved_size_m is what tells the user the size did not
            # land, rather than a job that errored after producing a model.
            log.exception("normalize failed for job %s; leaving the mesh as-is", job_id)
            return
        params["scale_factor"] = transform["scale"]
        params["transform"] = transform
        await asyncio.to_thread(self.store.set_params, job_id, params)

    def _rank_reference(self: Worker, image_path: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Score a finished reference. Blocking -- called through to_thread.

        ``image_path`` is the same input.png the report was measured from and is
        passed in rather than re-derived, so the two halves of the score can
        never end up describing different files.

        The anchor half is opportunistic in three separate ways, and every one
        of them is a "leave the number out", never a failure: ranking can be
        switched off, ref.png -- the conditioning reference, whether the
        profile's style anchor or an image the user attached themselves -- is
        only there on a run that had one, and DINOv2 is an optional download.
        What is left is the composition score, which is free: the report was
        measured either way.
        """
        from .bench import metrics
        from .pipelines import rank

        report = params.get("reference_report")
        cosine = None
        anchor = image_path.parent / "ref.png"
        if self.config.rank_candidates and anchor.exists():
            try:
                if metrics.dino_available(self.config):
                    # CPU deliberately: this runs on the job queue beside a
                    # resident trellis and a resident SDXL pipe, and a metric
                    # must not take VRAM from the models making the asset.
                    cosine = metrics.reference_cosine(
                        anchor, image_path, self.config, device="cpu"
                    )
            except Exception:
                log.exception("anchor similarity failed; ranking on composition alone")
        return rank.score(report, cosine)

    async def _audit_mesh(
        self: Worker, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Measure how see-through the finished mesh is and record it.

        trellis-server's narrow-band remesh can emit a crust of disconnected
        plates that passes every integrity check while being visibly
        perforated, and the user currently finds that out in Blender. Measuring
        it here turns it into something they see on the job the moment it
        finishes. Runs after _apply_scale so it measures the mesh that will
        actually be downloaded; like scaling it is CPU-only and holds no GPU
        memory, so it sits outside the VRAM handoff.

        Returns the summary it stored, or None whenever it stored nothing --
        because it was cancelled, or because the measurement itself broke.
        The remesh loop in _generate reads that to decide whether the mesh is
        worth redoing, and None is deliberately not a bad verdict: no
        measurement means no retry, exactly as with the retry switched off.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return None
        self.progress.update(
            job_id,
            phase="audit",
            label="Checking mesh",
            inner=0.0,
            inner_next=1.0,
            nominal=6.0,
            detail="",
        )
        try:
            # Imported here for the same reason as postprocess above: numpy and
            # trimesh at module scope would land in app startup.
            from . import meshaudit

            report = await asyncio.to_thread(
                meshaudit.hole_fraction,
                glb_path,
                meshaudit.DEFAULT_VIEWS,
                meshaudit.REQUEST_PATH_RESOLUTION,
            )
        except Exception:
            # A diagnostic must never be able to fail a job whose mesh is
            # already on disk and fine. Log it and leave mesh_audit unset --
            # the UI renders no badge rather than a wrong one.
            log.exception("mesh audit failed for job %s", job_id)
            return None
        # Only the summary is stored: the per-view detail would ride along on
        # every row of the 100-job list for no one to read.
        params["mesh_audit"] = {
            "worst": report["worst"],
            "mean": report["mean"],
            "faces": report["faces"],
            "resolution": report["resolution"],
        }
        # The silhouette number stays exactly as it was -- it is the only thing
        # that catches trellis's disconnected-plate crust. The report adds what
        # the silhouette cannot see: topology, materials, budget, and whether
        # the thing will sit on an engine's floor.
        try:
            from . import meshreport

            params["mesh_report"] = await asyncio.to_thread(
                functools.partial(
                    meshreport.build,
                    glb_path,
                    target_size_m=params.get("size_m"),
                    silhouette=params["mesh_audit"],
                )
            )
        except Exception:
            log.exception("mesh report failed for job %s", job_id)
        log.info(
            "job %s mesh audit: worst %.3f, mean %.3f over %d faces",
            job_id, report["worst"], report["mean"], report["faces"],
        )
        await asyncio.to_thread(self.store.set_params, job_id, params)
        return params["mesh_audit"]

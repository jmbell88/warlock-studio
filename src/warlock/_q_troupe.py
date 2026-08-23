"""``Worker``'s Troupe stage: a configured character sheet.

One job kind, ``charsheet``, and it is the last link of the chain the program
spec calls Phase 4: reference -> gate -> mesh -> rig -> **sheet**. The first
four links are machinery this repo already had -- an ordinary reference job
carrying a T-pose guide, the promote gate, trellis, and the auto-rig follow-up
-- so what is new here is only the part that had no precedent.

**The render size and the atlas size are different numbers, and that is
forced.** ``pipelines.pixelize.reduce_frames`` carries the argument: a 256-cell
sheet rendered at the 512px the program supersamples from would pack to
4096x16384, which ``sheet.check_atlas_size`` refuses at 8192 and no engine
would load. So the plan is built at the *logical* size the atlas will exist at,
while the Blender spec renders at ``charsheet.RENDER_SIZE``, and the reduction
between them is not an optimisation but the only route. That is also why this
does not reuse ``RigOps._render_sheet_atlas``: that helper renders and packs at
one size, correctly, for every sheet that is not this one.

Output is an ordinary sheet -- ``sheets/<id>.png`` plus its sidecar in the
*source* job's directory -- because "Open in Inker", the library, the exporters
and the Aseprite writer all already read that pair. What makes it a character
sheet is the frame table it was laid out on and the ``animation`` block in the
sidecar, and both of those are ``pipelines.charsheet``'s.

CPU and an EEVEE render throughout: nothing here touches the resident models or
the VRAM handoff, and the serial queue is what keeps it from overlapping a
trellis run.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import clips, rigging

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .queue import Worker

log = logging.getLogger(__name__)


class TroupeOps:
    """The character sheet stage, mixed into :class:`~.queue.Worker`."""

    async def _charsheet(self: Worker, job: dict[str, Any]) -> None:
        from . import queue as queue_mod
        from .pipelines import charsheet, pixel, pixelize, pixelsheet
        from .pipelines import sheet as sheetlib

        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        # Validated before it becomes a path, the rule every id that reaches
        # the filesystem from params follows: ``job_dir("")`` is the assets
        # root, and this directory is what the Blender worker is pointed at.
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
        source_dir = self.config.job_dir(source_id)
        sheet_id = str(params.get("sheet_id") or rigging.new_id())

        rig_glb = source_dir / "rig.glb"
        if not rig_glb.exists():
            # Deleted between queueing and running, or the auto-rig failed and
            # this follow-up was queued anyway. Every cell is a posed frame, so
            # the alternative is 256 copies of one T-pose.
            raise RuntimeError("that mesh is no longer rigged")

        template = str(params.get("template") or "humanoid")
        troupe_layout = charsheet.resolve_layout(params.get("layout"))
        records = await asyncio.to_thread(clips.expand_clips, template, troupe_layout)
        logical = int(params.get("logical_size", 32))
        layout = charsheet.plan(
            records,
            frame_size=logical,
            elevation=float(params.get("elevation", sheetlib.DEFAULT_ELEVATION)),
            lighting=str(params.get("lighting", "flat")),
            layout=troupe_layout,
        )

        # Keyed by ``(pose id, frame)`` rather than by id: every frame of a
        # clip shares an id by construction, and keying on the id alone would
        # render frame 0 in every cell of the animation. ``_sheet``'s rule.
        by_key = {
            (r.get("id"), r.get("frame", 0)): r
            for rows in records.values()
            for r in rows
        }
        roots, root_bone = await self._charsheet_roots(source_dir, records)
        cells: list[dict[str, Any]] = []
        for c in layout.cells:
            record = by_key.get((c.pose, c.frame)) or {}
            cell: dict[str, Any] = {
                "index": c.index,
                "yaw": c.yaw,
                "pose": c.pose,
                "frame": c.frame,
                "bones": record.get("bones") or {},
            }
            space = record.get("space")
            if space:
                # Only where the record says so, so a node-frame pose stays the
                # byte-identical cell it always was and the worker's default
                # holds. The clip library is authored in delta space; see
                # ``blender_worker.POSE_SPACES`` for why that is the only frame
                # that survives a re-fit.
                cell["pose_space"] = space
            offset = roots.get((c.pose, c.frame))
            if offset:
                cell["root_bone"] = root_bone
                cell["root_offset"] = offset
            cells.append(cell)

        self.progress.update(
            job_id, phase="sheet", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=40.0, detail=f"{len(cells)} frames",
        )

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="sheet", label=label, inner=frac,
                inner_next=min(frac + 0.05, 1.0), nominal=60.0, detail="",
            )

        png = rigging.sheet_png_path(source_dir, sheet_id)
        # The staging name ``_publish_text`` and every other served write use.
        atlas_path = png.with_name(f".{png.name}.render")
        reduce_mode = str(params.get("reduce_mode", "box"))
        result, trims = await self._render_charsheet(
            rig_glb, layout, cells, on_progress=on_progress, job_id=job_id,
            pack_target=atlas_path, reduce_mode=reduce_mode,
        )

        # --- the pixel-art pass -------------------------------------------
        #
        # Over the packed atlas rather than per cell: one nearest search
        # instead of 256 of them, and -- the part that matters -- **one palette
        # across every cell**, which is what stops the same shirt coming out
        # two shades in two directions. ``pixelize_atlas`` still runs the
        # neighbourhood passes per cell, because a dense atlas has no gutter
        # and a sprite touching its edge would otherwise be outlined against
        # the sprite beside it.
        # The Blender subprocess has exited, so from here a cancel has no
        # leverage on anything running -- and this is the expensive tail: a
        # whole-atlas nearest search over 256 cells. Every sibling stage checks
        # between its phases; without it a cancel clicked here does nothing,
        # visibly, for a long time.
        if self._cancel is not None and self._cancel.event.is_set():
            return
        self.progress.update(
            job_id, phase="pixel", label="Quantising", inner=0.0,
            inner_next=1.0, nominal=8.0, detail="",
        )
        palette_name = str(params.get("palette") or "")
        colors = int(params.get("colors", 64))

        def _quantise() -> tuple[dict[str, Any], dict[int, dict[str, int] | None]]:
            from PIL import Image

            with Image.open(atlas_path) as opened:
                opened.load()
                atlas = opened.convert("RGBA")
            if palette_name:
                entries = pixel.parse_palette(
                    _palette_path(self.config, palette_name).read_text("utf-8"),
                    _palette_path(self.config, palette_name).suffix,
                )
                chosen = "designed"
            else:
                # Median cut over the whole atlas, then handed to
                # ``pixelize_atlas`` as an ordinary palette -- so the outline
                # and orphan passes run identically whether the colours were
                # designed or derived, and a derived sheet is not a second code
                # path with its own bugs.
                _mapped, hexes = pixelsheet.quantize_shared(atlas, colors)
                entries = tuple(
                    (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hexes
                )
                chosen = "derived"
            out, report = pixelize.pixelize_atlas(
                atlas,
                columns=layout.columns,
                rows=layout.rows,
                cell=logical,
                palette=entries,
                dither=bool(params.get("dither")),
                # Still passed, and still a no-op on this path -- the atlas
                # arrives already reduced. Kept rather than dropped because
                # ``pixelize_atlas`` is shared with the restyle door, where the
                # atlas is *not* pre-reduced and the mode is live; the report
                # below is corrected instead.
                reduce_mode=reduce_mode,
                outline_mode=str(params.get("outline", "none")),
            )
            # About the reduction that actually happened, which is
            # ``reduce_frames``' and not ``pixelize_atlas``'. Computed here
            # rather than trusted from the report: on this path the atlas is
            # already at the target, so ``pixelize_atlas`` measured a stride of
            # 1 and answered ``True`` unconditionally -- including at 24, 48 and
            # 96px, where 512 does not divide and the real reduction fell back
            # to a NEAREST resize.
            report["exact_stride"] = charsheet.RENDER_SIZE % logical == 0
            report["palette"] = chosen
            report["palette_name"] = palette_name
            report["palette_size"] = len(entries)
            # Re-measured off the atlas that is actually published, not the one
            # that was packed. ``pack`` measures each frame as it composites --
            # correct there, and stale by the time this pass has finished with
            # it: ``snap_alpha`` zeroes alpha below 128 and shrinks the
            # silhouette, and ``outline`` in the default ``"outer"`` mode grows
            # it by a pixel on every side. The packed trims were written into
            # the sidecar unchanged, so every cell's rectangle was a pixel
            # short all round and a packer that honoured it -- which is the
            # field's whole purpose -- clipped the outline off every sprite.
            #
            # Cell-local, like ``measure_trim``'s own answer: the crop is taken
            # at the cell's place in the atlas and the box comes back relative
            # to the crop, so nothing downstream has to know where the cell sat.
            trimmed = {
                cell.index: sheetlib.measure_trim(
                    out.crop(
                        (
                            cell.x,
                            cell.y,
                            cell.x + layout.cell_w,
                            cell.y + layout.cell_h,
                        )
                    )
                )
                for cell in layout.cells
            }
            # Onto the served name last, staged: the sidecar is what marks the
            # sheet complete, so nothing is reading this yet -- but the rule
            # that a write onto a served path is staged does not have an
            # exception for "nothing is reading it yet".
            tmp = png.with_name(f".{png.name}.tmp")
            out.save(tmp, format="PNG")
            tmp.replace(png)
            # The un-quantised render has no readers once the pixel-art atlas
            # is on disk, and leaving it behind would double every character
            # sheet's footprint in a directory the user never cleans.
            atlas_path.unlink(missing_ok=True)
            return report, trimmed

        pixel_report, trims = await asyncio.to_thread(_quantise)

        # The sidecar is written last and is what ``list_sheets`` treats as the
        # completion marker -- ``_sheet``'s rule, and ``rig.json``'s before it.
        pivot = result.get("pivot") if isinstance(result, dict) else None
        # Into *cell* pixels: the worker projected it at ``RENDER_SIZE`` and
        # the sidecar documents cell-relative. ``_q_rig._sheet`` hands the same
        # value straight through and is right to -- there, render size *is*
        # cell size.
        pivot = charsheet.pivot_in_cell(pivot, layout.frame_size)
        meta = sheetlib.sidecar(
            layout,
            sheet_id=sheet_id,
            source_job=source_id,
            image=png.name,
            created=time.time(),
            name=str(params.get("name") or ""),
            pivot=pivot,
            trims=trims,
            animation=charsheet.animation_block(troupe_layout),
        )
        # Additive metadata on the ordinary sheet v1 format. Inker continues to
        # consume ``animation``; Troupe uses this immutable snapshot to drive
        # its per-sheet preview controls.
        meta["troupe"] = troupe_layout.as_dict()
        await asyncio.to_thread(
            queue_mod._publish_text,
            rigging.sheet_path(source_dir, sheet_id),
            json.dumps(meta, indent=2),
        )
        params["sheet_id"] = sheet_id
        params["cells"] = len(cells)
        # Deliberately **not** in ``DERIVED_PARAMS``, though the worker writes
        # it. What goes in that set is something the worker learned about the
        # *output* -- a value a reroll must not inherit or it wears a stale
        # verdict. This is the opposite: it is the request, normalised. A layout
        # the user supplied is carried forward because it is what they asked
        # for, and an absent one resolves to the immutable legacy layout
        # (``charsheet.resolve_layout``'s documented no-payload answer), so
        # writing the resolved form back cannot pin a default that later moves.
        params["layout"] = troupe_layout.as_dict()
        params["pixel_report"] = pixel_report
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info(
            "rendered character sheet %s for job %s: %dx%d cells at %dpx, %s palette",
            sheet_id, source_id, layout.columns, layout.rows, logical,
            pixel_report.get("palette"),
        )

    async def _charsheet_roots(
        self: Worker,
        source_dir: Path,
        records: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[tuple[Any, int], list[float]], Any]:
        """Per-cell root offsets, read only if some frame actually carries one.

        ``_sheet``'s arrangement: the map-building lives in ``queue`` at module
        scope so a test can drive it without a Worker or a Blender fake, and
        the read stays here because it is I/O and only worth doing when there
        is something to read it for.
        """
        from . import queue as queue_mod

        flat = [r for rows in records.values() for r in rows]
        if not any(float(v) for r in flat for v in (r.get("root_translation") or ())):
            return {}, None
        rig_meta = await asyncio.to_thread(rigging.read_rig, source_dir)
        return queue_mod._sheet_root_offsets(flat, rig_meta)

    async def _render_charsheet(
        self: Worker,
        glb: Path,
        layout: Any,
        cells: list[dict[str, Any]],
        *,
        on_progress: Any,
        job_id: str,
        pack_target: Path,
        reduce_mode: str = "box",
    ) -> tuple[Any, Any]:
        """Render at ``RENDER_SIZE``, reduce to the layout's size, then pack.

        The three-step shape the module docstring argues for, and the reason
        this is not ``_render_sheet_atlas`` with an extra parameter: that
        helper's contract is that the frames it renders are the pixels it
        packs, which every other sheet in the app relies on. Here they are not,
        and the step between them is the pixel-art reduction -- which also
        keeps ``sheet.pack``'s LANCZOS resize off this path, a filtered
        downscale being precisely the soft, fringed result the alpha snap would
        then have to guess about.

        Returns the worker's result and the pack's trims; the atlas itself is
        left at ``pack_target``.
        """
        from .pipelines import charsheet, pixelize
        from .pipelines import sheet as sheetlib

        with tempfile.TemporaryDirectory(prefix="warlock-charsheet-") as tmp:
            scratch = Path(tmp)
            frames_dir = scratch / "render"
            frames_dir.mkdir()
            spec = rigging.sheet_spec(
                glb,
                frames_dir,
                cells,
                frame_size=charsheet.RENDER_SIZE,
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
            self.progress.update(
                job_id, phase="reduce", label="Reducing frames", inner=0.0,
                inner_next=1.0, nominal=6.0, detail="",
            )
            rendered = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
            reduced = await asyncio.to_thread(
                functools.partial(
                    pixelize.reduce_frames,
                    rendered,
                    layout.frame_size,
                    scratch / "reduced",
                    # The user's choice belongs *here*, on the only reduction
                    # this path performs: 512 down to the logical size, per
                    # frame. It was passed to ``pixelize_atlas`` instead, where
                    # the atlas is already at the logical size -- so ``reduce``
                    # took its ``rgba.size == (w, h)`` early return, the mode
                    # was never consulted and the setting was dead while being
                    # validated at the door and reported back in the record.
                    mode=reduce_mode,
                )
            )
            self.progress.update(
                job_id, phase="pack", label="Packing sheet", inner=0.0,
                inner_next=1.0, nominal=3.0, detail="",
            )
            # Packed onto a staging name beside the served one rather than
            # onto the served name itself: the atlas the user is shown is the
            # *pixelised* one, and a served file that is briefly the
            # un-quantised render is a sheet the library can thumbnail in the
            # state nobody asked for. It has to leave the scratch directory
            # here because that directory is about to go away and the quantise
            # pass runs after it.
            pack_target.parent.mkdir(parents=True, exist_ok=True)
            trims = await asyncio.to_thread(sheetlib.pack, layout, reduced, pack_target)
        return result, trims


def _palette_path(config: Any, name: str) -> Path:
    """The palette file, re-resolved in the worker.

    ``service.palettes`` owns the lookup and the traversal check, and the queue
    may not import the service -- so this restates the *narrow* half: the name
    reached the door, was validated there, and is re-joined here under the same
    directory with the same containment check, because params outlive the door
    that wrote them.
    """
    directory = Path(config.palette_dir)
    for suffix in (".hex", ".gpl"):
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file() and candidate.resolve().parent == directory.resolve():
            return candidate
    raise RuntimeError(f"palette {name!r} is no longer installed")

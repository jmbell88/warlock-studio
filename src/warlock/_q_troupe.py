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

**Themed effects composite between the reduction and the pack**, and that
ordering is the whole point of where the phase sits. The trims, the structural
check and the outline pass all run afterwards, so they see the flames rather
than the body they hang off; and the quantisation stays **one shared pass**, so
a flame's oranges enter the same 32-colour cut as the character's skin.
Compositing *after* the quantise would give the flames a palette of their own
and a sheet two colour sets wide -- the same "same shirt, two shades" failure
the whole-atlas pass exists to prevent, reached from the other side.

A sheet only asks for sockets when the source directory holds a
``character.json``, and only that request makes the worker project them. Every
sheet rendered before the character registry existed is therefore byte-identical
to the one it was.

**A four-frame idle flame does not loop seamlessly**: Flourish's flame erodes
its silhouette with fbm scrolled along the rise, and fbm is not periodic, so the
last frame of a short loop does not hand back to the first. At 16-64px and 32
colours it reads as a flicker in the tongues rather than a pop, which is why it
ships; ``characters.effects`` and ``docs/manual/33-troupe.md`` both say so.

CPU and an EEVEE render throughout: nothing here touches the resident models or
the VRAM handoff, and the serial queue is what keeps it from overlapping a
trellis run.
"""

from __future__ import annotations

import asyncio
import contextlib
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
        from .pipelines import charsheet, pixelize, pixelsheet, sheetcheck
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
        # Refused rather than minted, for the reason ``_q_rig._sheet`` states
        # in full: ``_discard_artifacts`` deletes this kind's *served* pair by
        # this id, which is only safe while every door mints a fresh one.
        sheet_id = str(params.get("sheet_id") or "")
        if not rigging.is_valid_id(sheet_id):
            raise ValueError(f"sheet_id is not a sheet id: {sheet_id!r}")

        rig_glb = source_dir / "rig.glb"
        if not rig_glb.exists():
            # Deleted between queueing and running, or the auto-rig failed and
            # this follow-up was queued anyway. Every cell is a posed frame, so
            # the alternative is 256 copies of one T-pose.
            raise RuntimeError("that mesh is no longer rigged")

        template = str(params.get("template") or "humanoid")
        troupe_layout = charsheet.resolve_layout(params.get("layout"))

        # **A re-render of some of the runs.** Both keys or neither: a subset
        # with nothing to copy from is a sheet with holes in it, and a base
        # with nothing to re-render is a copy of a sheet that already exists.
        subset = params.get("subset")
        base_sheet = str(params.get("base_sheet") or "")
        if bool(subset) != bool(base_sheet):
            raise ValueError("a re-render needs both the runs and the sheet to copy")
        wanted: set[int] = set()
        base_png: Path | None = None
        base_margin: float | None = None
        if subset:
            if not rigging.is_valid_id(base_sheet):
                raise ValueError(f"base_sheet is not a sheet id: {base_sheet!r}")
            wanted = set(charsheet.subset_indices(subset, troupe_layout))
            base_png = rigging.sheet_png_path(source_dir, base_sheet)
            base_record = rigging.read_sheet(source_dir, base_sheet)
            if not base_png.exists() or not base_record:
                raise RuntimeError(
                    "the sheet this re-render copies from is no longer on disk"
                )
            # **Framed the way the sheet it is composited onto was framed.**
            # The re-render's whole geometry claim is that a cell keeps the
            # rectangle it has always had; a cell drawn at a different ortho
            # scale keeps the rectangle and changes the character's size inside
            # it, which is worse than either error alone. The base sidecar
            # records what it was actually rendered at, so a base that took the
            # reframe retry hands its wider window on. Only when it differs
            # from the constant, so an ordinary re-render's spec is the spec it
            # has always been.
            recorded = float((base_record.get("camera") or {}).get("frame_margin") or 0.0)
            if recorded and recorded != sheetlib.FRAME_MARGIN:
                base_margin = recorded
        # **Read before anything is rendered, so the socket list reaches the
        # spec.** Its presence is what makes the worker project sockets at all,
        # and a sheet whose source has no ``character.json`` sends the spec it
        # has always sent -- byte for byte.
        character = await asyncio.to_thread(_read_character, source_dir)
        socket_specs = _socket_specs(character)

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
            # **The plan is built unfiltered and only the spec list is cut.**
            # ``cell.index``, ``row``, ``column``, ``x`` and ``y`` are therefore
            # byte-identical to a full render's, which is the geometry claim the
            # whole re-render rests on: a re-rendered cell lands on exactly the
            # rectangle it has always had.
            if wanted and c.index not in wanted:
                continue
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
        # **The render and the quantise share one ``finally``.** Both write
        # ``atlas_path``, and it lands in ``source_dir`` -- the *mesh* job's
        # directory, not this job's -- so an exception in either used to orphan
        # up to an 8192-square PNG somewhere that deleting the failed sheet
        # would never reclaim, and nothing sweeps. ``_discard_artifacts`` is
        # not the answer: the queue calls that on a cancel, not on an error.
        try:
            result, trims, sockets_px = await self._render_charsheet(
                rig_glb, layout, cells, on_progress=on_progress, job_id=job_id,
                pack_target=atlas_path, reduce_mode=reduce_mode,
                only=wanted or None,
                margin=base_margin,
                sockets=socket_specs,
                character=character,
                troupe_layout=troupe_layout,
            )

            # **The reframe retry, and it runs exactly once.**
            #
            # A pose whose apex leaves the rest bounding box used to render
            # clipped on every cell of its run; ``op_sheet`` now frames from
            # the union of every pose, which fixes the cause. This is the
            # backstop for what the union cannot know: a silhouette wider than
            # its bounding volume once the outline and the alpha snap have had
            # it, or a socket nobody declared. One wider window, then publish
            # whatever comes back -- looping would let a subject that touches
            # the edge for any other reason re-render the whole sheet forever,
            # at a minute a go.
            #
            # **Measured on the packed, pre-quantisation trims.** ``_quantise``
            # produces the published ones, and it runs after this: it is the
            # pass that grows every silhouette by an outline pixel and snaps
            # the alpha, so judging the *framing* by its output would call a
            # correctly framed sheet clipped and reframe it for nothing. The
            # framing question is about what the camera drew, and these are
            # the trims of exactly that.
            #
            # **Never on a subset re-render.** Those cells are composited onto
            # a sheet rendered at another window, and reframing them would put
            # a differently sized character on the same rectangle -- see
            # ``base_margin`` above. A clipped subset stays clipped and is
            # flagged, which is the smaller of the two wrongs.
            reframed = False
            if base_png is None and sheetcheck.clipped_cells(layout, trims):
                reframed = True
                log.info(
                    "character sheet %s clipped at margin %.2f; re-rendering wider",
                    sheet_id, sheetlib.FRAME_MARGIN,
                )
                # Discarded rather than left to be overwritten: the second pack
                # writes the same name, and a half-written atlas from a failed
                # retry must not be mistaken for the first render's.
                with contextlib.suppress(OSError):
                    atlas_path.unlink(missing_ok=True)
                result, trims, sockets_px = await self._render_charsheet(
                    rig_glb, layout, cells, on_progress=on_progress, job_id=job_id,
                    pack_target=atlas_path, reduce_mode=reduce_mode,
                    only=wanted or None,
                    margin=sheetlib.FRAME_MARGIN * 1.25,
                    sockets=socket_specs,
                    character=character,
                    troupe_layout=troupe_layout,
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
                # Designed or median-cut, the colours are then handed to
                # ``pixelize_atlas`` as one ordinary palette -- so the outline and
                # orphan passes run identically whichever branch produced them, and
                # a derived sheet is not a second code path with its own bugs.
                # ``resolve_palette`` is that branch, and its docstring carries the
                # reason a palette file does not cost the shared-across-cells
                # property.
                designed = queue_mod._palette_entries(self.config, palette_name)
                if base_png is not None and designed is None:
                    # **Pinned from the sheet being re-rendered.** With no
                    # designed palette ``resolve_palette`` median-cuts the atlas
                    # it is given -- and a subset atlas is a handful of cells, so
                    # it would derive its own colours and the re-rendered runs
                    # would come back a different shade from the ones beside
                    # them. That is precisely the "same shirt, two shades"
                    # failure the whole-atlas pass exists to prevent, reached by
                    # a different road. The base atlas is already mapped, so its
                    # own colour set *is* the answer, exactly.
                    designed = _atlas_entries(base_png, colors)
                entries, chosen = pixelsheet.resolve_palette(
                    atlas, colors=colors, entries=designed or None
                )
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
                if base_png is not None:
                    # **Composed last, and that ordering is the whole point.**
                    # ``pixelize_atlas`` runs ``outline`` per cell, which in the
                    # shipped ``outer`` mode grows a silhouette by a pixel on
                    # every side -- so composing first and quantising the result
                    # would fatten every *copied* cell once per re-render, and
                    # the sheet would drift thinner in the runs nobody touched.
                    # Render, reduce, pack the subset, quantise the subset, then
                    # compose. Quantise once.
                    composed = png.with_name(f".{png.name}.composed")
                    out.save(composed, format="PNG")
                    try:
                        sheetlib.compose_cells(
                            base_png, composed, layout, wanted, composed
                        )
                        with Image.open(composed) as opened:
                            opened.load()
                            out = opened.convert("RGBA")
                    finally:
                        with contextlib.suppress(OSError):
                            composed.unlink(missing_ok=True)
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
                tmp = png.with_name(f".{png.name}.tmp")
                out.save(tmp, format="PNG")
                tmp.replace(png)
                return report, trimmed

            pixel_report, trims = await asyncio.to_thread(_quantise)
        finally:
            with contextlib.suppress(OSError):
                atlas_path.unlink(missing_ok=True)

        # The sidecar is written last and is what ``list_sheets`` treats as the
        # completion marker -- ``_sheet``'s rule, and ``rig.json``'s before it.
        pivot = result.get("pivot") if isinstance(result, dict) else None
        framing = (result.get("framing") if isinstance(result, dict) else None) or {}
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
        if subset:
            # Additive on the ordinary sheet v1 format, ``meta["troupe"]``'s
            # neighbour and its rule: a reader that does not know these keys
            # sees the sheet it would have seen anyway.
            meta["base_sheet"] = base_sheet
            meta["subset"] = [
                {"animation": animation, "direction": direction}
                for animation, direction in charsheet.check_subset(subset, troupe_layout)
            ]
        # Additive metadata on the ordinary sheet v1 format. Inker continues to
        # consume ``animation``; Troupe uses this immutable snapshot to drive
        # its per-sheet preview controls.
        meta["troupe"] = troupe_layout.as_dict()
        # Additive on the ordinary sheet v1 format too -- ``meta["troupe"]``'s
        # neighbour and its rule, sidecar version unchanged: a reader that does
        # not know this key sees the sheet it would have seen anyway.
        #
        # Written on **every** sheet and not only on the ones a preset named,
        # because "what was this framed from" is a question about the sheet
        # rather than about the request: an importer placing a sprite in a
        # scene needs the projection and the elevation whether or not the user
        # picked them off a list.
        meta["camera"] = _camera_meta(
            layout.elevation,
            pixel_size=logical,
            # **What was rendered, not what was requested.** The worker reports
            # the margin it actually framed with, which after a reframe retry
            # is not the one this row asked for -- and the field's only reader
            # is a subset re-render deciding how to frame itself to match, so a
            # requested figure there would put a differently sized character
            # onto the sheet's own rectangles. The param remains the fallback
            # for a result from before ``framing`` existed.
            margin=float(
                framing.get("margin") or params.get("margin") or sheetlib.FRAME_MARGIN
            ),
        )
        # **Flags, never fails.** Every finding here is about a sheet that is
        # already packed, quantised and on disk: it opens, it exports, and the
        # user can look at it. Raising would throw away a minute of rendering
        # over a verdict they may not care about -- an intentionally
        # edge-to-edge portrait sheet is "clipped" by this measure and fine by
        # theirs. Recorded on the sidecar so Troupe can say what it found.
        #
        # Off the *published* trims, unlike the reframe decision above: this
        # answer is about the pixels the user gets, outline and alpha snap
        # included, which is what an importer will see.
        meta["validation"] = sheetcheck.validate(
            layout, trims, meta, reframed=reframed
        )
        # Additive on sheet v1, ``meta["troupe"]``'s rule and its version --
        # a reader that has never heard of either key sees the sheet it would
        # have seen anyway.
        #
        # **Which character, not which recipe would rebuild it.** The recipe is
        # carried whole because it is the only thing that can reproduce this
        # subject, and the family plus its version because a species row that
        # moves is a sheet that can no longer be re-rendered to match -- the
        # same reason ``character.json`` carries both.
        if character:
            meta["character"] = {
                "family": character.get("family"),
                "family_version": character.get("family_version"),
                "recipe": character.get("recipe"),
            }
        if sockets_px:
            # Per cell and **only where the worker actually projected one**, in
            # cell pixels like ``pivot_x``/``pivot_y`` beside them. Absent on a
            # cell whose socket named a bone the rig does not have, and absent
            # on every cell of a subset re-render that was copied rather than
            # rendered -- an attachment point invented for a cell nobody
            # rendered would be a claim about geometry nothing measured.
            for entry in meta["cells"]:
                here = sockets_px.get(int(entry["index"]))
                if here:
                    entry["sockets"] = here
        await asyncio.to_thread(
            queue_mod._publish_text,
            rigging.sheet_path(source_dir, sheet_id),
            json.dumps(meta, indent=2),
        )
        # The sidecar is the completion marker, so the sheet is *visible* from
        # this line on -- ``list_sheets`` returns it and Troupe draws it. A
        # cancel arriving in the tail below would otherwise record the row
        # "cancelled" and send ``_discard_artifacts`` at
        # ``sheet_path``/``sheet_png_path``, which for this kind are the served
        # pair rather than temps: it deletes a sheet the user can already see.
        # ``_rig`` and ``_pixel_sheet`` commit at the same instant for the same
        # reason.
        #
        # Deliberately not earlier. The window between the atlas landing on
        # ``png`` and this publish is still cancellable, and should be: with no
        # sidecar nothing lists that atlas, so discarding it is the right
        # answer rather than a loss.
        if self._cancel is not None:
            self._cancel.commit()
        params["sheet_id"] = sheet_id
        # **The sheet's cell count, not the subset's.** ``cells`` is now the
        # filtered *spec* list on a re-render, and letting this follow it would
        # quietly change what a ``DERIVED_PARAMS`` field means -- a 12-cell
        # re-render of a 256-cell sheet reporting twelve. The plan is the sheet.
        params["cells"] = len(layout.cells)
        if subset:
            # Derived: something the worker learned about the output, ``cells``'
            # neighbour. ``subset`` and ``base_sheet`` are deliberately *not* --
            # they are the request, normalised, which is ``layout``'s case
            # below, so "run that again" re-renders those runs against that base.
            params["rendered_cells"] = len(wanted)
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
        # Derived: what the worker learned about *this* atlas, ``pixel_report``'s
        # neighbour, and in ``DERIVED_PARAMS`` for the same reason -- a reroll
        # inheriting it would wear a clipping verdict about frames it has not
        # rendered yet.
        params["validation"] = meta["validation"]
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
        only: set[int] | None = None,
        margin: float | None = None,
        sockets: list[dict[str, Any]] | None = None,
        character: dict[str, Any] | None = None,
        troupe_layout: Any = None,
    ) -> tuple[Any, Any, dict[int, dict[str, dict[str, Any]]]]:
        """Render at ``RENDER_SIZE``, reduce, composite effects, then pack.

        The three-step shape the module docstring argues for, and the reason
        this is not ``_render_sheet_atlas`` with an extra parameter: that
        helper's contract is that the frames it renders are the pixels it
        packs, which every other sheet in the app relies on. Here they are not,
        and the step between them is the pixel-art reduction -- which also
        keeps ``sheet.pack``'s LANCZOS resize off this path, a filtered
        downscale being precisely the soft, fringed result the alpha snap would
        then have to guess about.

        Returns the worker's result, the pack's trims, and the projected
        sockets in *cell* pixels; the atlas itself is left at ``pack_target``.
        ``margin`` widens the ortho window and has one caller, the reframe
        retry -- see ``_charsheet``.

        ``sockets`` is the attachment list to project, present only for a
        source that carries a ``character.json``; ``character`` is that sidecar,
        and its theme is what decides whether anything is composited at all.
        The effects pass sits between the reduce and the pack for the reason the
        module docstring gives: quantisation is one shared pass and the flames
        have to be in it.
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
                # Omitted unless the reframe retry asked for one, so the first
                # render's spec is byte-identical to the spec this stage has
                # always sent and the worker takes ``sheet.FRAME_MARGIN``.
                margin=margin,
                # Same rule, same reason: ``None`` writes no key at all, and no
                # key is what every sheet rendered before the character
                # registry existed was rendered with.
                sockets=sockets or None,
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
            rendered = {
                c.index: frames_dir / f"{c.index:04d}.png"
                for c in layout.cells
                if only is None or c.index in only
            }
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

            # --- the effects pass ---------------------------------------
            #
            # **Between the reduce and the pack, and that is the whole point.**
            # Everything after this line -- the trim measurement, the
            # structural check, the outline pass and the quantise -- then sees
            # the flames rather than the bare body, and the quantise stays one
            # shared pass, so a flame's oranges enter the same 32-colour cut as
            # the skin beside them. Compositing after it would hand the flames
            # a second palette.
            #
            # It works at the *logical* size because that is what
            # ``reduce_frames`` just produced: a composited cell is a drop-in
            # replacement for the reduced one, so ``pack`` cannot tell them
            # apart.
            sockets_px = _sockets_in_cells(result, layout.frame_size)
            if character and sockets_px:
                self.progress.update(
                    job_id, phase="effects", label="Drawing effects", inner=0.0,
                    inner_next=1.0, nominal=2.0, detail="",
                )
                composited = await asyncio.to_thread(
                    functools.partial(
                        _composite_effects,
                        character,
                        reduced,
                        sockets_px,
                        troupe_layout=troupe_layout,
                        logical=layout.frame_size,
                        out_dir=scratch / "effects",
                    )
                )
                # Merged rather than replaced: a cell whose socket the worker
                # did not project keeps the reduced frame it already had.
                reduced.update(composited)

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
            trims = await asyncio.to_thread(
                functools.partial(sheetlib.pack, layout, reduced, pack_target, only=only)
            )
        return result, trims, sockets_px


def _read_character(source_dir: Path) -> dict[str, Any] | None:
    """The mesh job's ``character.json``, or ``None``.

    ``None`` is the ordinary answer and not an error: every mesh this program
    reconstructed from a photograph has no character sidecar, and a sheet of
    one is a sheet with no sockets and no effects -- exactly the sheet it has
    always been.

    **A malformed one costs the effects and never the sheet.** A minute of
    Blender is already spent by the time anything here is read, and the failure
    mode of raising would be "the sheet you waited for is gone because a JSON
    file next to the mesh had a typo".
    """
    path = source_dir / "character.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("ignoring an unreadable %s: %s", path, exc)
        return None
    return raw if isinstance(raw, dict) else None


def _socket_specs(character: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The archetype's sockets in the shape ``rigging.sheet_spec`` wants.

    Off the **archetype** rather than off ``character.json``'s own ``sockets``
    block, and the difference matters: the sidecar records each socket's world
    *position* in the rest pose, and the worker needs the ``(along, lateral,
    up)`` offset in bone-length units so it can re-derive the point in every
    pose of every cell. A rest-pose position hung on a running character would
    leave the flame standing where the hand used to be.
    """
    if not character:
        return []
    from .characters import family

    try:
        arch = family.get_archetype(str(character.get("archetype") or ""))
    except Exception as exc:  # pragma: no cover - a sidecar naming no archetype
        log.warning("ignoring a character sidecar with no known archetype: %s", exc)
        return []
    return [
        {
            "name": s.name,
            "bone": s.bone,
            "offset": [float(v) for v in s.offset],
            "reach": float(s.reach),
        }
        for s in arch.sockets
    ]


def _sockets_in_cells(
    result: Any, frame_size: int
) -> dict[int, dict[str, dict[str, Any]]]:
    """The worker's per-cell socket projection, in **cell** pixels.

    The worker projects at ``charsheet.RENDER_SIZE`` and this sheet packs at the
    logical size, so the numbers need the same conversion the pivot needs --
    ``charsheet.point_in_cell``, which was generalised from ``pivot_in_cell``
    for exactly this. Without it a 32px cell would place a flame sixteen cells
    away from the hand.

    The keys arrive as JSON object keys, i.e. strings, because the result
    travels through ``result.json``; they come back as ints.
    """
    from .pipelines import charsheet

    if not isinstance(result, dict):
        return {}
    raw = result.get("sockets")
    if not isinstance(raw, dict):
        return {}
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for index, block in raw.items():
        if not isinstance(block, dict):
            continue
        here: dict[str, dict[str, Any]] = {}
        for name, point in block.items():
            if not isinstance(point, dict):
                continue
            converted = charsheet.point_in_cell(
                (point.get("x", 0.0), point.get("y", 0.0)), frame_size
            )
            here[str(name)] = {
                "x": converted[0],
                "y": converted[1],
                "depth": float(point.get("depth") or 0.0),
                "behind": bool(point.get("behind")),
            }
        if here:
            out[int(index)] = here
    return out


def _cells_by_index(troupe_layout: Any) -> dict[int, dict[str, Any]]:
    """``{cell index: {"frame", "frames", "fps"}}`` -- where each cell sits in
    its movement, how long that movement is, and how fast it plays.

    ``composite_effects`` needs all three and ``sheet.Cell`` carries only the
    first: a cell knows it is frame 5 and not that its walk is eight frames
    long. Built here because the frame table is ``charsheet``'s arithmetic and
    a second copy of it would be a second opinion about what cell 137 depicts.
    """
    from .pipelines import charsheet

    # ``frame_table``'s own guard, restated: the caller holds a resolved
    # ``LayoutSpec`` and ``resolve_layout`` takes a mapping.
    resolved = (
        troupe_layout
        if isinstance(troupe_layout, charsheet.LayoutSpec)
        else charsheet.resolve_layout(troupe_layout)
    )
    movements = {
        m.name: (int(m.frames), max(1, int(round(1000.0 / max(int(m.duration_ms), 1)))))
        for m in resolved.movements
    }
    out: dict[int, dict[str, Any]] = {}
    for cell in charsheet.frame_table(resolved):
        frames, fps = movements.get(cell.animation, (1, 12))
        out[int(cell.index)] = {"frame": int(cell.frame), "frames": frames, "fps": fps}
    return out


def _composite_effects(
    character: dict[str, Any],
    reduced: dict[int, Any],
    sockets_px: dict[int, dict[str, dict[str, Any]]],
    *,
    troupe_layout: Any,
    logical: int,
    out_dir: Path,
) -> dict[int, Any]:
    """``characters.effects.composite_effects``, with the registry lookups.

    Separated from the coroutine for ``_camera_meta``'s reason -- the
    interesting part is a pair of registry lookups, and a lookup buried in a
    300-line coroutine that needs Blender to reach is a lookup nothing can test.

    **The recipe seed is used exactly as it is stored**, on a subset re-render
    as much as on a full one, and that is the same argument as the pinned
    palette: a re-rendered run whose flames were seeded afresh would come back
    with different tongues from the cells beside it, which is worse than either
    a stale flame or no flame at all. The subset falls out of ``reduced``
    holding only the cells that were rendered -- nothing filters here.
    """
    from .characters import effects, family

    try:
        fam = family.get_family(str(character.get("family") or ""))
    except Exception as exc:
        log.warning("a character sidecar names no known species: %s", exc)
        return {}
    key = str(character.get("theme") or "")
    theme = next((t for t in fam.themes if t.key == key), None)
    if theme is None or not theme.effects:
        return {}
    recipe = character.get("recipe")
    seed = int((recipe or {}).get("seed", 0) or 0) if isinstance(recipe, dict) else 0
    return effects.composite_effects(
        reduced,
        _cells_by_index(troupe_layout),
        sockets_px,
        theme=theme,
        sockets=fam.arch.sockets,
        recipe_seed=seed,
        logical=int(logical),
        out_dir=out_dir,
    )


def _camera_meta(
    elevation: float, *, pixel_size: int, margin: float
) -> dict[str, Any]:
    """The ``camera`` block of a character sheet's sidecar.

    A named helper rather than a dict literal inside ``_charsheet`` for
    ``_atlas_entries``' reason: the interesting part is the preset lookup, and
    a lookup buried in a 300-line coroutine that needs Blender to reach is a
    lookup nothing can test.

    **The preset is matched, never trusted.** The row carries an elevation and
    not a preset name -- that is deliberate, so a preset table that gains a row
    is not a migration of every queued sheet -- so the key here is whichever
    preset holds exactly this elevation, and ``None`` when a caller set an
    angle of its own. Recording a *nearest* preset would be the sidecar
    claiming a framing the sheet was not rendered at.
    """
    from .pipelines import charsheet

    elevation = float(elevation)
    preset = next(
        (key for key, _label, angle in charsheet.CAMERA_PRESETS if angle == elevation),
        None,
    )
    return {
        "preset": preset,
        "elevation": elevation,
        # Orthographic throughout: ``rigging.sheet_spec`` frames every cell with
        # an ortho camera, which is what makes a sprite the same size wherever
        # it sits on the atlas.
        "projection": "orthographic",
        "pixel_size": int(pixel_size),
        "render_size": charsheet.RENDER_SIZE,
        "frame_margin": float(margin),
    }


def _atlas_entries(png: Path, colors: int) -> list[tuple[int, int, int]]:
    """The exact colour set a published sheet is already mapped to.

    Used to **pin** the palette of a subset re-render, so the cells that come
    back land in the same colours as the ones beside them. Exact rather than
    approximate: the base atlas went through ``pixelize_atlas`` and its opaque
    pixels *are* the entries, so reading them back is a lookup and not a second
    median cut that might land somewhere else.

    Sorted, because ``map_palette`` searches by nearest and the answer must not
    depend on dictionary order; opaque only, because a transparent pixel has no
    colour to contribute and ``outline`` writes its ink at full alpha.

    **Refused when the set is implausibly large.** A pin is only sound while the
    base really is palette-mapped; an unquantised atlas would yield thousands of
    "entries" and the re-render would be mapped to noise. Answering empty hands
    the caller back to the ordinary median cut, which is wrong in a smaller and
    much more visible way than being wrong by four thousand colours.
    """
    import numpy as np
    from PIL import Image

    with Image.open(png) as opened:
        opened.load()
        pixels = np.asarray(opened.convert("RGBA"))
    opaque = pixels[pixels[..., 3] > 0]
    if opaque.size == 0:
        return []
    unique = np.unique(opaque[:, :3].reshape(-1, 3), axis=0)
    if len(unique) > max(int(colors), 1) * 4:
        log.warning(
            "the sheet being re-rendered carries %d colours, which is not a "
            "palette -- deriving one instead of pinning",
            len(unique),
        )
        return []
    return [tuple(int(v) for v in row) for row in unique]

# ``_palette_path`` was lifted to ``queue._palette_entries`` on 2026-08-29. It
# was this file's own restatement of ``service.palettes._path``'s traversal
# check, and two more stages are about to want the same lookup -- three copies
# of a containment check being two chances to fix only one of them.
# ``_sprite_source``'s rule: a helper the ``_q_*`` modules share lives at
# ``queue`` module scope and is reached through ``queue_mod``.

"""An Inker animation as a sprite sheet: one cell per frame.

This is the first module under ``studio/inker/`` that imports outside the
package, and the exception is deliberate and bounded. ``pipelines.sheet`` is the
*authority* on the sheet format -- the `Plan`/`Cell` types, the sidecar keys,
the atlas ceiling -- and a second writer of a versioned public format is how
``version: 1`` comes to mean two subtly different documents. The invariant the
package's docstring states is about imgui, moderngl, pygame and the service
layer, i.e. about staying assertable headlessly; ``sheet`` is stdlib plus a lazy
Pillow import, so the *purpose* of the rule survives, not only its letter. The
inverse placement would be worse: putting this in ``pipelines/`` makes
``pipelines`` depend on ``studio``, and ``pipelines`` modules run inside worker
and Blender processes.

It does not go through ``sheet.plan``. That function's job is poses by yaws, and
its ``FRAME_SIZES`` check would refuse a 300x180 canvas outright -- so the cells
are built here and the *format* is still emitted there.

The core takes plain arrays rather than a ``Document`` so every rule about the
grid is assertable without one; ``from_document`` is the adapter.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ...pipelines import sheet as sheetlib
from .animation import DIRECTION_ORDER, DirectionalLayout

__all__ = [
    "ARRANGES",
    "COUNTED_ARRANGES",
    "DEFAULT_FRAME_TEMPLATE",
    "DEFAULT_LAYER_TEMPLATE",
    "DEFAULT_TAG_TEMPLATE",
    "FILENAME_KEYS",
    "build",
    "compose",
    "filename_for",
    "flatten_one",
    "flatten_subset",
    "frame_uids",
    "from_document",
    "layer_splits",
    "plan_frames",
    "rebase_tags",
    "require_distinct_names",
    "sanitize_stem",
    "scale_slices",
    "slices_block",
    "slices_snapshot",
    "snapshot",
    "tag_span",
    "timing",
]


# --- filename templates --------------------------------------------------------
#
# One small vocabulary, shared by every place Inker turns a document into a
# filename: the split-stem a per-tag or per-layer export gives each output
# (``inker_mode._split_stems``), and the per-frame name a PNG sequence gives
# each file inside one output. Both used to be one hard-coded f-string apiece;
# both are now one call into :func:`filename_for` with a template that -- left
# at its default -- produces the exact same bytes, which is what the byte pins
# beside each call site prove.

#: The whole vocabulary a template may reference. Checked by *name* against
#: this tuple rather than left to Python's own ``KeyError`` -- the same
#: doctrine :func:`plan_frames` applies to ``arrange``: a typo in a template a
#: user typed by hand should say what it does not recognise, not surface an
#: interpreter's own exception.
FILENAME_KEYS = ("title", "tag", "frame", "layer")

#: :func:`inker_mode.run_pngs`'s own template, unedited -- ``"{title}_{index:04d}"``
#: before templates existed, and this is the same bytes with ``index`` renamed.
#: A byte pin over the whole default PNG-sequence export exists specifically to
#: keep this constant honest.
DEFAULT_FRAME_TEMPLATE = "{title}_{frame}"

#: A tag split's default stem -- ``_split_stems``'s ``f"{stem}_{safe}"`` before
#: templates existed, spelled as a template. Pinned the same way.
DEFAULT_TAG_TEMPLATE = "{title}_{tag}"

#: A layer split's default stem. Same shape, same reason, ``layer`` instead of
#: ``tag`` only so a template written for one kind of split reads honestly
#: when it is reused for the other.
DEFAULT_LAYER_TEMPLATE = "{title}_{layer}"

#: Characters a filename's dynamic part may keep; everything else becomes a
#: dash. ``inker_mode._split_stems`` always applied this to a split's label
#: before templates existed; it is exported so :func:`filename_for` -- which
#: lives here, one level below :mod:`inker_mode` -- can share the one rule
#: rather than a second copy of it drifting.
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_stem(text: str) -> str:
    """``text``, with everything but letters, digits, dot, dash and underscore
    turned into a dash and trimmed off the ends. What a tag or a layer name
    becomes before it can be part of a filename."""
    return _SAFE_STEM.sub("-", text).strip("-")


def _template_keys(template: str) -> set[str]:
    """Every ``{name}`` :func:`str.format` would read out of ``template``.

    Through :class:`string.Formatter` rather than a regex: it is the same
    parser ``.format`` itself uses, so ``{{literal}}`` braces and a stray ``{``
    are read exactly as ``.format`` would read them instead of a second,
    slightly different opinion about the syntax.
    """
    return {
        field
        for _literal, field, _spec, _conv in string.Formatter().parse(template)
        if field
    }


def filename_for(
    template: str,
    *,
    title: str,
    tag: str | None = None,
    frame: int | None = None,
    layer: str | None = None,
) -> str:
    """One output's filename, built from a user-chosen template. Pure.

    :data:`FILENAME_KEYS` is the whole vocabulary -- ``{title}``, ``{tag}``,
    ``{frame}`` (always zero-padded to 4 digits; a template does not get to
    ask for a different width, the one thing every consumer of a numbered
    sequence already assumes) and ``{layer}``. A key outside that set is
    refused **by name**, listing the four this function actually knows,
    rather than surfacing ``str.format``'s own ``KeyError`` -- the same
    doctrine :func:`plan_frames` applies to a mistyped ``arrange``.

    A key that names a value *this particular call* was not given -- a plain
    PNG sequence's template asking for ``{tag}``, a split's asking for
    ``{frame}`` -- is refused the same way: a template is authored once and
    reused for whatever export it is next attached to, and a silently
    rendered ``"None"`` would be a corrupt filename nobody asked to write.

    ``tag`` and ``layer`` are sanitised through :func:`sanitize_stem` --
    ``inker_mode._split_stems``'s own rule before this function existed --
    whether or not the template actually reads them, so a tag or a layer that
    cannot be part of a filename is refused up front rather than only when a
    template happens to mention it. ``title`` is never touched: it is already
    a stem a save dialog handed back (or another call to this function
    already produced), already a legal filename.
    """
    keys = _template_keys(template)
    unknown = keys - set(FILENAME_KEYS)
    if unknown:
        bad = sorted(unknown)[0]
        known = ", ".join("{" + key + "}" for key in FILENAME_KEYS)
        raise ValueError(f"unknown filename key {{{bad}}}; this export knows {known}")

    values: dict[str, str] = {"title": title}
    for key, raw in (("tag", tag), ("layer", layer)):
        if raw is None:
            if key in keys:
                raise ValueError(
                    f"this export has no {key}; remove {{{key}}} from the template"
                )
            continue
        safe = sanitize_stem(raw)
        if not safe:
            raise ValueError(f"{raw!r} is not a name a file can be given")
        values[key] = safe
    if "frame" in keys:
        if frame is None:
            raise ValueError(
                "this export has no frame; remove {frame} from the template"
            )
        values["frame"] = f"{int(frame):04d}"

    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"filename template {template!r} could not be filled in") from exc


def require_distinct_names(names: Sequence[str]) -> None:
    """Refuse if any of ``names`` is empty, or two of them are the same.

    The one collision rule every multi-output naming step shares: a split
    template that sanitises two labels to one name, or a PNG-sequence template
    that drops ``{frame}`` and so gives every frame of a clip the same name,
    fail exactly the same way here -- one output silently overwriting another
    on disk is the one outcome a refusal exists to prevent.  ``_split_stems``
    refused a split's collision this way before templates existed; this is
    that same rule, generalised to the frame-numbering step that gained a
    template beside it.
    """
    seen: set[str] = set()
    for name in names:
        if not name:
            raise ValueError("a filename template produced an empty name")
        if name in seen:
            raise ValueError(f"two outputs would both be called {name!r}")
        seen.add(name)


def rebase_tags(tags: Sequence[Any], f0: int, f1: int) -> list[Any]:
    """The tags of frames ``f0..f1``, renumbered as if that span were the file.

    Pure, and it is three rules rather than one. A tag wholly outside the span
    is **dropped** -- an export of frames 10-19 that carried "intro, frames
    0-3" describes cells that are not in the file. A tag that straddles an end
    is **clamped**, because the part of it that made the cut is genuinely in
    there and refusing to name it at all would lose more than it protects. And
    every surviving tag is **shifted** by ``f0``, because the exported sheet's
    first cell is index 0 whatever the document called it.
    """
    out = []
    for tag in tags:
        start, end = max(int(tag.start), f0), min(int(tag.end), f1)
        if start > end:
            continue
        out.append(replace(tag, start=start - f0, end=end - f0))
    return out


#: The named packings ``arrange`` accepts, beside the plain row-wrap (``None``).
#: ``"rows"`` and ``"columns"`` are the *counted* forms -- the two that take a
#: ``wrap`` -- and the set is checked against by name rather than inferred, so
#: a typo refuses loudly instead of silently falling through to the row-wrap.
ARRANGES = (None, "horizontal", "vertical", "rows", "columns")
#: Public (not ``_``-prefixed) because ``panes/inker_timeline.py`` needs the
#: exact same set to decide when to draw the wrap-count field beside the
#: Arrange combo -- one tuple, imported, rather than two copies that could
#: drift the moment a third counted arrange is ever added.
COUNTED_ARRANGES = ("rows", "columns")


def plan_frames(
    count: int,
    frame_w: int,
    frame_h: int,
    *,
    name: str = "",
    layout: DirectionalLayout | None = None,
    arrange: str | None = None,
    wrap: int | None = None,
) -> sheetlib.Plan:
    """The grid for ``count`` frames of a ``frame_w`` x ``frame_h`` canvas.

    Wrapping into rows is required rather than a nicety: 32 frames of a 320px
    canvas is 10240px across, past what an engine will accept as a texture at
    all. So the row length is whatever fits, and ``check_atlas_size`` -- the same
    guard ``plan`` uses, which is why it was worth extracting -- has the last
    word on the result.

    ``layout`` replaces that wrap with the sheet's own fixed grid: a walk cycle
    is four rows of four *because a row is a direction*, and a wrap that
    happened to fit five across would produce an atlas whose rows mean nothing.
    A count that no longer fills the grid is refused rather than padded --
    silently emitting a sheet with a hole where "back, frame 3" should be is the
    one outcome a game would not notice until it played the animation.

    ``arrange`` is a third, user-chosen way to pick ``columns`` (and hence
    ``rows``), which is otherwise the row-wrap's job: ``"horizontal"`` is one
    row, ``"vertical"`` is one column, and the counted forms fix a side from
    ``wrap`` -- ``"rows"`` picks how many rows there should be (``columns`` is
    derived from it), ``"columns"`` picks ``columns`` directly. Every one of
    them still ends by deriving ``rows`` from ``columns`` the same way the
    plain row-wrap always did, so none of them can leave a dead trailing row
    the way asking for the count verbatim could. ``None`` is the row-wrap
    unchanged -- the byte pin above this module's tests proves it -- and it is
    refused together with ``layout``: a directional grid already fixes the
    columns for its own reason, and the two disagreeing about which reason
    wins would be a silent coin flip.
    """
    if count < 1:
        raise ValueError("a sheet needs at least one frame")
    if frame_w < 1 or frame_h < 1:
        raise ValueError("a frame has a positive size")
    if frame_w > sheetlib.MAX_ATLAS_PX or frame_h > sheetlib.MAX_ATLAS_PX:
        raise ValueError(
            f"a {frame_w}x{frame_h} frame does not fit in a "
            f"{sheetlib.MAX_ATLAS_PX}px atlas at all"
        )
    if arrange not in ARRANGES:
        raise ValueError(
            f"unknown arrange {arrange!r}; must be one of "
            f"{[a for a in ARRANGES if a is not None]}"
        )
    if arrange is not None and layout is not None:
        raise ValueError("a directional layout and an arrange cannot both be set")
    if wrap is not None and arrange not in COUNTED_ARRANGES:
        raise ValueError('wrap only applies to arrange="rows" or arrange="columns"')
    if arrange in COUNTED_ARRANGES:
        if wrap is None:
            raise ValueError(f'arrange="{arrange}" needs a wrap count')
        if wrap < 1:
            raise ValueError("wrap must be at least 1")
    if layout is not None:
        if count != layout.frame_count:
            raise ValueError(
                f"a {layout.kind} sheet is {layout.frame_count} frames and this "
                f"document has {count}"
            )
        columns, rows = layout.columns, layout.rows
    else:
        if arrange is None:
            columns = max(1, min(count, sheetlib.MAX_ATLAS_PX // frame_w))
        elif arrange == "horizontal":
            columns = count
        elif arrange == "vertical":
            columns = 1
        elif arrange == "rows":
            columns = -(-count // wrap)  # ceil
        else:  # "columns"
            columns = wrap
        rows = -(-count // columns)  # ceil
    sheetlib.check_atlas_size(columns * frame_w, rows * frame_h)

    pose_name = name or sheetlib.REST_POSE_NAME
    if layout is None:
        cells = tuple(
            sheetlib.Cell(
                index=i,
                row=i // columns,
                column=i % columns,
                x=(i % columns) * frame_w,
                y=(i // columns) * frame_h,
                # A drawn frame has no pose and no camera. ``frame`` is the
                # index, which is exactly what the field was reserved for.
                pose=None,
                pose_name=pose_name,
                yaw=0.0,
                frame=i,
            )
            for i in range(count)
        )
        poses: tuple[dict[str, Any], ...] = ({"id": None, "name": pose_name},)
        yaws: tuple[float, ...] = (0.0,)
    else:
        # ``pose_name`` carries the direction and ``frame`` restarts per row, so
        # the sidecar an engine reads says "this cell is left, frame 2" in
        # exactly the fields a rendered sheet says it in. Nothing new is
        # invented for the format's sake.
        placed = [layout.cell(i) for i in range(count)]
        cells = tuple(
            sheetlib.Cell(
                index=i,
                row=row,
                column=col,
                x=col * frame_w,
                y=row * frame_h,
                pose=None,
                pose_name=direction,
                yaw=float(yaw),
                frame=frame,
            )
            for i, (row, col, direction, yaw, frame) in enumerate(placed)
        )
        poses = tuple({"id": None, "name": d} for d in DIRECTION_ORDER)
        yaws = tuple(float(y) for y in dict.fromkeys(y for _, _, _, y, _ in placed))
    return sheetlib.Plan(
        # Zero, not ``frame_w``: see ``sidecar``. A square-only importer must
        # fail loudly rather than slice this correctly across and wrongly down.
        frame_size=0,
        columns=columns,
        rows=rows,
        yaws=yaws,
        elevation=0.0,
        lighting="flat",
        poses=poses,
        cells=cells,
        frame_w=frame_w,
        frame_h=frame_h,
    )


def animation_block(
    plan: sheetlib.Plan,
    durations_ms: Sequence[int],
    tags: Sequence[Any],
    *,
    layout: DirectionalLayout | None = None,
    arrange: str | None = None,
    wrap: int | None = None,
    frame_cells: Sequence[int | None] | None = None,
    trim: bool = False,
    padding: int = 0,
    extrude: int = 0,
) -> dict[str, Any]:
    """The ``animation`` key: which cell is which frame, and for how long.

    ``frame_cells`` is ``None`` for every caller before merge/skip-empty
    existed, and for a merge-off, skip-empty-off export today: ``plan.cells``
    is then one cell per frame in frame order, so ``cell.index`` doubles as
    the frame index into ``durations_ms`` -- the identity this function always
    assumed. When it is given, it is ``build``'s own record of which original
    frame landed on which surviving cell (``None`` for one ``skip_empty``
    dropped), and ``durations_ms`` is then indexed by *frame*, not by cell --
    so a 100ms and a 200ms frame sharing a cell still each get their own
    ``duration_ms`` entry: ``frames`` is a timeline, not a cell list, and two
    frames sharing a cell are still two frames.
    """
    if frame_cells is None:
        frames_out = [
            {"cell_index": cell.index, "duration_ms": int(durations_ms[cell.index])}
            for cell in plan.cells
        ]
        merged = False
        skipped: list[int] = []
    else:
        frames_out = []
        skipped = []
        seen: set[int] = set()
        merged = False
        for i, cell_index in enumerate(frame_cells):
            if cell_index is None:
                skipped.append(i)
                continue
            if cell_index in seen:
                merged = True
            else:
                seen.add(cell_index)
            frames_out.append(
                {"cell_index": cell_index, "duration_ms": int(durations_ms[i])}
            )
    block: dict[str, Any] = {
        "frames": frames_out,
        "tags": [
            {
                "name": tag.name,
                "start": int(tag.start),
                "end": int(tag.end),
                "loop": bool(tag.loop),
                # Additive, and the sidecar's version is unchanged for the same
                # reason ``animation.json``'s is: every existing key keeps its
                # value and its meaning, so a reader that has never heard of
                # this one is still correct about the file.
                "direction": str(getattr(tag, "direction", "forward")),
                # Written only when set, unlike ``direction`` beside it: 0 is
                # what every tag ever exported already meant, so a repeat-less
                # sidecar is byte-identical to the one this wrote before the
                # field existed -- which is what the square-sidecar pin needs.
                **(
                    {"repeat": int(tag.repeat)}
                    if int(getattr(tag, "repeat", 0) or 0)
                    else {}
                ),
            }
            for tag in tags
        ],
    }
    if merged:
        # True only when a cell actually got reused, not merely when the
        # option was asked for: a merge with no duplicates in it must read
        # exactly like an ordinary export, which is what keeps this key an
        # honest report of what happened rather than an echo of the request.
        block["merged"] = True
    if skipped:
        # Original frame indices, in ``skip_empty``'s own order -- the cell
        # list has already shrunk by the time this is called, so there is
        # nothing left in ``plan.cells`` to say which frame went missing.
        block["skipped"] = skipped
    if layout is not None:
        # Inside the ``animation`` mapping rather than beside it: this says how
        # to *play* the sheet, which is what that key is for. The sidecar's
        # pinned square path never reaches this branch, and the contents here
        # are unpinned, so a reader that has never heard of the key is still
        # correct about the file.
        block["layout"] = {"kind": layout.kind, "directions": list(DIRECTION_ORDER)}
    if arrange is not None:
        # Same placement as ``layout`` above and for the same reason: this
        # says how the sheet was packed, which is what the ``animation`` key is
        # for, and it is unreached by the pinned square path so it costs that
        # pin nothing. ``wrap`` only when set, so ``"horizontal"``/``"vertical"``
        # (which have none) do not grow a null nobody would read.
        block["arrange"] = arrange
        if wrap is not None:
            block["wrap"] = int(wrap)
    # Same placement and the same reason as ``arrange``/``wrap`` above: these
    # three describe how the sheet was *packed*, not what it depicts, so they
    # ride the ``animation`` mapping rather than growing ``sheet.py``'s
    # top-level, fixed-order sidecar -- which is also what keeps them costing
    # the byte pin nothing. A cell-driven importer needs none of them: every
    # cell already carries its own atlas ``x``/``y``/``w``/``h``, padding and
    # extrude baked in, which is the same reason ``sidecar()``'s own docstring
    # gives for why ``frame_size``/``columns``/``rows`` are a convenience and
    # not a requirement.
    if trim:
        # True rather than the trim rectangles themselves -- those already
        # live per cell in the sidecar's ``trims`` (the sibling of this dict),
        # unchanged in meaning: this is only the flag that says the cells were
        # packed to them instead of to the full frame.
        block["trimmed"] = True
    if padding:
        block["padding"] = int(padding)
    if extrude:
        block["extrude"] = int(extrude)
    return block


def build(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    tags: Sequence[Any] = (),
    *,
    name: str = "",
    layout: DirectionalLayout | None = None,
    arrange: str | None = None,
    wrap: int | None = None,
    merge: bool = False,
    skip_empty: bool = False,
    trim: bool = False,
    padding: int = 0,
    extrude: int = 0,
) -> tuple[Any, sheetlib.Plan, dict[int, dict[str, int] | None], list[int | None] | None]:
    """Composite the frames into one atlas.

    Returns ``(image, plan, trims, frame_cells)``. ``frame_cells`` is ``None``
    when neither option is on -- the identity ``animation_block`` already
    assumed, and the byte pin over the whole default export path never has to
    know this parameter exists. On, it is one entry per element of ``frames``:
    the cell index it landed on, or ``None`` where ``skip_empty`` dropped it.

    It composites here rather than through ``sheet.pack``, which reads frames
    from ``Path``s on disk -- these are already in memory, and writing sixteen
    PNGs to a scratch directory to read them straight back would be the slowest
    part of the export by a wide margin.

    ``trim``, ``padding`` and ``extrude`` are the three all off by default, so
    the untouched path below (``cell_w, cell_h = width, height``, no offset, no
    replicated border) is byte-for-byte the sheet this always packed -- which is
    what the default byte pin over the whole export proves. On, they compose in
    the order packwright's own grid mode already established: trim decides the
    *cell* size (the largest trimmed frame's, uniform across every surviving
    cell, each frame's own -- possibly smaller -- trimmed pixels placed flush
    at the cell's own corner rather than centred or stretched, so a frame with
    nothing in it stays a blank cell), padding then spaces that grid out
    (Tiled's own margin-and-spacing: one gutter between cells, one border round
    the outside), and extrude replicates each *placed rectangle's* own border
    -- the frame's trimmed size when trimming, the whole cell otherwise -- into
    whatever gutter padding left it. The same room guarantee packwright's
    ``PackSettings`` enforces at construction is enforced here at the door:
    ``padding`` has to be at least twice ``extrude``, or two neighbours
    extruding into one shared gutter would meet.
    """
    from PIL import Image

    if padding < 0 or extrude < 0:
        raise ValueError("padding and extrude cannot be negative")
    if padding < extrude * 2:
        raise ValueError(
            f"padding must be at least twice extrude "
            f"({extrude} x 2 = {extrude * 2}, padding is {padding}) "
            "-- two neighbours extrude into one gutter"
        )
    if len(frames) != len(durations_ms):
        raise ValueError("every frame needs a duration")
    if (merge or skip_empty) and layout is not None:
        # A directional layout is a claim about the *whole* timeline -- these
        # frames are four directions of a walk, in this fixed grid -- and a
        # merge or an omission changes which cells exist, which is exactly the
        # claim the grid is fixed on. Refused by name, the same shape as the
        # arrange/layout conflict ``plan_frames`` already refuses: a directional
        # grid already fixes the cells for its own reason, and a second reason
        # disagreeing about which cells there are would be a silent coin flip.
        raise ValueError(
            "a directional layout keeps its own fixed grid; merge and "
            "skip-empty cannot be combined with one"
        )
    height, width = frames[0].shape[:2]
    for pixels in frames:
        if pixels.shape[:2] != (height, width):
            raise ValueError("every frame of a sheet is the same size")

    frame_cells: list[int | None] | None = None
    cell_frames: Sequence[np.ndarray] = frames
    if merge or skip_empty:
        assigned: list[int | None] = []
        representative: list[np.ndarray] = []
        seen: dict[bytes, int] = {}
        for pixels in frames:
            if skip_empty:
                tile = Image.fromarray(pixels, "RGBA")
                empty = sheetlib.measure_trim(tile) is None
                tile.close()
                if empty:
                    assigned.append(None)
                    continue
            if merge:
                key = pixels.tobytes()
                existing = seen.get(key)
                if existing is not None:
                    assigned.append(existing)
                    continue
                index = len(representative)
                seen[key] = index
                representative.append(pixels)
                assigned.append(index)
            else:
                assigned.append(len(representative))
                representative.append(pixels)
        if not representative:
            raise ValueError("every frame is empty; there is nothing to export")
        frame_cells = assigned
        cell_frames = representative

    # Trim decides the cell size before ``plan_frames`` ever runs, the same
    # order merge/skip_empty already established: the grid is packed from
    # whatever the surviving cells actually need, not from the source frame
    # size. Measured on ``cell_frames`` -- already deduplicated and filtered
    # above -- so a merge's hash and a trim's bounding box never disagree about
    # which pixels a cell holds: the hash saw the full, untrimmed frame.
    trim_rects: dict[int, dict[str, int] | None] | None = None
    if trim:
        trim_rects = {}
        max_w = max_h = 0
        for index, pixels in enumerate(cell_frames):
            tile = Image.fromarray(pixels, "RGBA")
            rect = sheetlib.measure_trim(tile)
            tile.close()
            trim_rects[index] = rect
            if rect is not None:
                max_w = max(max_w, rect["w"])
                max_h = max(max_h, rect["h"])
        # A wholly transparent clip (every rect None) falls back to a 1x1 cell
        # rather than the full frame: a cell sized to content that does not
        # exist would be reporting a size nothing here ever measured.
        cell_w, cell_h = max(1, max_w), max(1, max_h)
    else:
        cell_w, cell_h = int(width), int(height)

    plan = plan_frames(
        len(cell_frames),
        cell_w,
        cell_h,
        name=name,
        layout=layout,
        arrange=arrange,
        wrap=wrap,
    )

    if padding:
        # Every cell moves out to the packwright grid's own pitch --
        # ``padding + column * (cell + padding)`` -- which is exactly what
        # ``Plan.width``/``.height`` now compute from ``columns``/``cell_w``
        # and this same ``padding`` field, so setting it here is the whole of
        # the atlas-size change; nothing below has to compute a size by hand.
        step_w, step_h = plan.cell_w + padding, plan.cell_h + padding
        plan = replace(
            plan,
            padding=padding,
            cells=tuple(
                replace(
                    cell,
                    x=padding + cell.column * step_w,
                    y=padding + cell.row * step_h,
                )
                for cell in plan.cells
            ),
        )
    # Re-checked here regardless of ``padding``: ``plan_frames`` already
    # checked the *unpadded* size against the same ceiling, but padding grows
    # the atlas past what that call could have known about.
    sheetlib.check_atlas_size(plan.width, plan.height)

    atlas = Image.new("RGBA", (plan.width, plan.height), (0, 0, 0, 0))
    trims: dict[int, dict[str, int] | None] = {}
    # Each cell's own placed rectangle -- the whole cell ordinarily, or (when
    # trimming) that frame's own possibly-smaller box -- fed to extrude after
    # every cell is painted, so a neighbour column already on the atlas cannot
    # be read as the source of a border that has not been drawn yet.
    placed: list[tuple[int, int, int, int]] = []
    for cell in plan.cells:
        pixels = cell_frames[cell.index]
        if trim:
            rect = trim_rects[cell.index]  # type: ignore[index]
            trims[cell.index] = rect
            if rect is not None:
                x0, y0, w0, h0 = rect["x"], rect["y"], rect["w"], rect["h"]
                cropped = np.ascontiguousarray(pixels[y0 : y0 + h0, x0 : x0 + w0])
                tile = Image.fromarray(cropped, "RGBA")
                atlas.paste(tile, (cell.x, cell.y))
                tile.close()
                placed.append((cell.x, cell.y, w0, h0))
            # ``rect is None``: nothing to paste -- the cell stays the blank,
            # fully transparent square ``Image.new`` already made it.
        else:
            tile = Image.fromarray(pixels, "RGBA")
            trims[cell.index] = sheetlib.measure_trim(tile)
            atlas.paste(tile, (cell.x, cell.y))
            tile.close()
            placed.append((cell.x, cell.y, plan.cell_w, plan.cell_h))

    if extrude:
        pixels_arr = np.array(atlas)
        for x, y, w, h in placed:
            sheetlib.extrude_edges(pixels_arr, x, y, w, h, extrude)
        atlas.close()
        atlas = Image.fromarray(pixels_arr, "RGBA")

    return atlas, plan, trims, frame_cells


def slices_snapshot(
    doc: Any, span: tuple[int, int] | None = None
) -> list[dict[str, Any]]:
    """Each frame's slices, resolved and in canvas coordinates. Plain data.

    ``Document.sprite_meta_for_frame`` does the resolving, which is the whole
    point of it living there: keys, uids and the pivot-source rule are the
    model's business, and what comes back here is dicts and tuples. Read on the
    **frame thread**, with the frames -- it walks the document, and a task
    thread doing that races the canvas.

    ``span`` is the same inclusive index pair :func:`frame_uids` takes, clamped
    by the same :func:`_clamped_span`, and it is not optional decoration: what
    comes back is consumed by :func:`slices_block`, which is keyed by
    ``cell.index`` -- the position in the *exported* sheet. A snapshot of the
    whole timeline handed to a three-cell span export would put frame 0's
    slices on the cell holding frame 1.

    A still document raises, like :func:`timing`: a sheet is an animation.
    """
    anim = getattr(doc, "anim", None)
    if anim is None or not anim.frames:
        raise ValueError("this document is not animated")
    f0, f1 = _clamped_span(len(anim.frames), span)
    return [
        doc.sprite_meta_for_frame(frame.uid) for frame in anim.frames[f0 : f1 + 1]
    ]


def scale_slices(
    snap: Sequence[dict[str, Any]], factor: int
) -> list[dict[str, Any]]:
    """A snapshot magnified by the same whole number the frames were.

    The union of two features that were built apart: an export can now be
    written at ``n`` x (``transform.upscale``), and an export can now carry
    slices -- and a sidecar whose rectangles describe the canvas while the
    atlas beside it is eight times bigger names the wrong pixels. ``pixelsheet``
    already rescales the block the other way when it reduces a sheet, so
    following the image is the convention here rather than a new rule.

    A pure multiply, on the snapshot's own spelling and *before*
    :func:`_slice_entry`, which is what makes it exact: everything here is an
    integer count of canvas pixels times an integer factor, so there is no
    rounding to have an opinion about. A factor of 1 hands the list straight
    back, so the unscaled path is the list it always was.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return list(snap)
    out = []
    for meta in snap:
        pivot = (meta or {}).get("pivot")
        out.append(
            {
                "pivot": None if pivot is None else tuple(v * factor for v in pivot),
                "slices": [
                    {
                        **entry,
                        "x": entry["x"] * factor,
                        "y": entry["y"] * factor,
                        "w": entry["w"] * factor,
                        "h": entry["h"] * factor,
                        "pivot": (
                            None
                            if entry.get("pivot") is None
                            else tuple(v * factor for v in entry["pivot"])
                        ),
                        "center": (
                            None
                            if entry.get("center") is None
                            else tuple(v * factor for v in entry["center"])
                        ),
                    }
                    for entry in (meta or {}).get("slices") or ()
                ],
            }
        )
    return out


def _rect_json(x: float, y: float, w: float, h: float) -> dict[str, Any]:
    return {"x": x, "y": y, "w": w, "h": h}


def _slice_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """One slice in the sidecar's own spelling: ``{x, y, w, h}``, absolute.

    Absolute -- source-image space, never relative to the slice's own origin --
    because that is the one convention every consumer of this file already has,
    and because it makes ``pixelsheet``'s rescale a single division rather than
    a case per field. The model stores the pivot and the centre relative to the
    bounds; ``sprite_meta_for_frame`` is the boundary that converts.

    ``pivot`` and ``center`` are written as ``null`` when absent rather than
    omitted, which is what ``trim`` beside them already does: a reader that
    branches on presence and a reader that branches on null are two readers, and
    this format has picked one.
    """
    center = entry.get("center")
    pivot = entry.get("pivot")
    return {
        "name": entry.get("name", ""),
        "bounds": _rect_json(
            int(entry["x"]), int(entry["y"]), int(entry["w"]), int(entry["h"])
        ),
        "pivot": None if pivot is None else {"x": float(pivot[0]), "y": float(pivot[1])},
        "center": (
            None if center is None else _rect_json(*(int(v) for v in center))
        ),
    }


def _trim_shift_slices(
    cell_slices: Sequence[dict[str, Any] | None],
    trims: Mapping[int, dict[str, int] | None],
) -> list[dict[str, Any] | None]:
    """``cell_slices`` re-expressed in a trimmed cell's own coordinates.

    ``build`` crops each surviving frame to its own trim rectangle and pastes
    the result flush at its cell's corner -- so a pivot or a slice authored in
    the *original* (untrimmed) canvas has to move by that same ``(trim.x,
    trim.y)`` the pixels moved by, or it goes on naming a point the art is no
    longer under. Precedent: ``scale_slices`` already keeps the sidecar
    describing "the atlas beside it" rather than the document it came from,
    for the same reason wearing a different transform.

    A slice's own bounds are also re-clipped to the frame's own placed box
    (its trimmed ``w``/``h``, not the uniform cell's, since a smaller frame's
    content does not fill the whole cell): whatever a slice covered outside
    the trim rectangle was cropped out of the pixels and is not in the atlas
    to describe. Only the bounds are clipped -- a pivot or a slice's centre is
    a point, translated like the rest of that slice's geometry and otherwise
    left alone, the same restraint ``_slices_offset`` already takes on a crop.

    ``trims`` is keyed by cell index, the same order ``cell_slices`` is in.
    A cell with nothing measured for it (an empty trim, or no entry at all)
    passes its meta through untouched -- there is no offset to know, because
    nothing of it was pasted.
    """
    out: list[dict[str, Any] | None] = []
    for index, meta in enumerate(cell_slices):
        rect = trims.get(index)
        if meta is None or rect is None:
            out.append(meta)
            continue
        ox, oy, ow, oh = rect["x"], rect["y"], rect["w"], rect["h"]
        pivot = meta.get("pivot")
        shifted: list[dict[str, Any]] = []
        for entry in meta.get("slices") or ():
            x, y = entry["x"] - ox, entry["y"] - oy
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(ow, x + entry["w"]), min(oh, y + entry["h"])
            e_pivot = entry.get("pivot")
            e_center = entry.get("center")
            shifted.append(
                {
                    **entry,
                    "x": x0,
                    "y": y0,
                    "w": max(0, x1 - x0),
                    "h": max(0, y1 - y0),
                    "pivot": (
                        None if e_pivot is None else (e_pivot[0] - ox, e_pivot[1] - oy)
                    ),
                    "center": (
                        None
                        if e_center is None
                        else (e_center[0] - ox, e_center[1] - oy, e_center[2], e_center[3])
                    ),
                }
            )
        out.append(
            {
                "pivot": None if pivot is None else (pivot[0] - ox, pivot[1] - oy),
                "slices": shifted,
            }
        )
    return out


def slices_block(
    plan: sheetlib.Plan, snap: Sequence[dict[str, Any]]
) -> dict[str, dict[int, Any]]:
    """``{"pivots": ..., "slices": ...}``, both keyed by **cell index**.

    ``cell.index`` and not ``cell.frame``: a directional sheet restarts
    ``frame`` per row, so it is the row's own numbering, while ``index`` is the
    timeline position in both branches of :func:`plan_frames` -- which is what
    the snapshot is in order of.

    Empty entries are left out rather than written as empties, so a document
    with no slices contributes nothing at all and the sidecar is what it was.
    """
    pivots: dict[int, tuple[float, float]] = {}
    blocks: dict[int, list[dict[str, Any]]] = {}
    for cell in plan.cells:
        if not 0 <= cell.index < len(snap):
            continue
        meta = snap[cell.index] or {}
        pivot = meta.get("pivot")
        if pivot is not None:
            pivots[cell.index] = (float(pivot[0]), float(pivot[1]))
        entries = [_slice_entry(entry) for entry in meta.get("slices") or ()]
        if entries:
            blocks[cell.index] = entries
    return {"pivots": pivots, "slices": blocks}


def snapshot(
    doc: Any, span: tuple[int, int] | None = None, *, track_uids: set[int] | None = None
) -> tuple[
    list[np.ndarray], list[int], list[Any], DirectionalLayout | None, list[dict[str, Any]]
]:
    """Everything an export needs, read off the document. **Frame thread only.**

    This is the whole of the export that touches the document, and it is split
    out so that it can be the whole of the export that runs on the frame thread.
    ``frame_flat`` is not a pure read: it fills and evicts the document's flatten
    cache, and ``layers_for`` copies each track's properties down onto its cels
    -- so running it on a task thread races the onion-skin draw doing the same
    thing on the same dicts. What comes back is a list of arrays the encoder can
    take away: the cache replaces entries rather than writing into them, so a
    later recomposite on the frame thread leaves these untouched.

    A still document raises rather than exporting a one-cell sheet: ``Export
    PNG`` already covers that case, and a "sprite sheet" with one frame in it is
    the kind of output a user has to go and check.

    Every frame goes through ``Document.frame_flat``, so a linked cel appearing
    in three frames yields three identical cells -- which is right. A sheet is
    played back by an engine that knows nothing about links, so the frames have
    to be there.

    ``Document.matte`` is deliberately not applied, where ``write_ora`` does
    apply it. The matte is what a *flattened* export puts behind transparency,
    and an atlas is not a flattened export: it is composited over whatever is
    behind it in the game, so baking white in would put a white square around
    every cell of a sprite opened from a photo.

    ``track_uids`` is the split-by-layer read: the same frames, composited from
    part of the stack, through :func:`flatten_subset` -- which deliberately does
    *not* touch the flatten cache. ``None`` is the whole stack and the call this
    always was.
    """
    uids = frame_uids(doc, span)
    if track_uids is None:
        frames = [flatten_one(doc, uid) for uid in uids]
    else:
        frames = [flatten_subset(doc, uid, track_uids) for uid in uids]
    return (frames, *timing(doc, span), slices_snapshot(doc, span))


# The same read, one frame at a time. **Frame thread only**, for exactly the
# reasons :func:`snapshot` gives -- these are its three halves, split out rather
# than reimplemented, so a caller that spreads the work across frames and a
# caller that wants it all at once cannot produce different sheets.
#
# The split exists because ``snapshot`` is the whole cost of an export and it
# was paid in one frame, at click time: a sixty-frame clip is sixty flattens
# and a visible freeze on the frame the user pressed the button. It cannot move
# to a task thread -- ``frame_flat`` fills and evicts the very cache the
# onion-skin draw is walking -- so the only place left to spend it is *later
# frames*, which is ``viewer/sheet.StripRender``'s answer to the same problem.


def _clamped_span(count: int, span: tuple[int, int] | None) -> tuple[int, int]:
    """``span`` trimmed onto ``count`` frames, or the whole timeline for None.

    One place, called by both halves, because ``frame_uids`` and ``timing`` are
    read a spread of frames apart and a span that clamped differently in the
    two would put one clip's cells under another clip's durations.
    """
    if span is None:
        return (0, count - 1)
    f0, f1 = (int(span[0]), int(span[1]))
    if f0 > f1:
        f0, f1 = f1, f0
    return (max(0, f0), min(f1, count - 1))


def frame_uids(doc: Any, span: tuple[int, int] | None = None) -> list[str]:
    """The frames an export will read, in order. Raises for a still document.

    Read up front so the stepper has a fixed work list: the document is locked
    against edits for the duration (``tab.saving``), but reading the list once
    is what makes "frame 3 of 60" a true statement rather than a guess.

    ``span`` is an inclusive index pair -- a tag, or a timeline range -- and
    None is the whole timeline, which is what every caller before spans meant
    and what keeps a whole-clip export byte-identical to the one this wrote
    before the parameter existed.
    """
    anim = getattr(doc, "anim", None)
    if anim is None or not anim.frames:
        raise ValueError("this document is not animated")
    f0, f1 = _clamped_span(len(anim.frames), span)
    if f0 > f1:
        raise ValueError("that range holds no frames")
    return [frame.uid for frame in anim.frames[f0 : f1 + 1]]


def flatten_one(doc: Any, uid: str) -> np.ndarray:
    """One frame, composited. The unit of work a stepper spends per frame."""
    plane = doc.frame_flat(uid)
    if plane is None:
        raise ValueError("a frame could not be flattened")
    return plane


def flatten_subset(doc: Any, uid: str, track_uids: Any) -> np.ndarray:
    """One frame, composited from **part** of the stack. Frame thread only.

    The unit of work a split-by-layer export spends per frame, and the one
    deliberate difference from :func:`flatten_one` beside it: it composites
    through ``Document.frame_stack`` directly rather than through
    ``frame_flat``, so it **never enters the flatten cache**. That cache is
    keyed on the frame uid and nothing else -- a subset stored under that key
    would be handed straight back to the onion skin, to playback and to the next
    whole-document export as if it were the whole frame.

    Nor does it read the cache: an entry that happened to be there is the whole
    frame, which is not what was asked for.

    A subset naming no live track raises rather than returning an empty plane
    (``LayerStack`` refuses to be empty) -- a sheet of nothing is a file the
    user has to go and check.
    """
    anim = getattr(doc, "anim", None)
    if anim is None:
        raise ValueError("this document is not animated")
    try:
        frame = anim.frames[anim.frame_index(uid)]
    except (KeyError, IndexError):
        raise ValueError("a frame could not be flattened") from None
    return doc.frame_stack(frame, track_uids=set(track_uids)).flatten()


def tag_span(anim: Any, tag: Any) -> tuple[int, int]:
    """A tag's ``(start, end)`` clamped onto the frames that exist.

    One spelling, because a tag exported on its own and the same tag exported as
    one of a batch must cover the same frames -- and a tag whose end outran the
    timeline (a delete leaves one behind) would otherwise clamp in one path and
    not in the other.
    """
    last = len(anim.frames) - 1
    return (
        max(0, min(int(tag.start), last)),
        max(0, min(int(tag.end), last)),
    )


def layer_splits(doc: Any) -> list[tuple[str, tuple[int, ...]]]:
    """``(name, track uids)`` per output of a split-by-layer export, bottom-first.

    **One output per top-level row, not per track**, and that is the whole
    decision here. A group is what the layers panel shows as a single row, its
    leaves are contiguous in stack order by invariant, and its members carry the
    group's folded visibility and opacity -- so "the fx folder" is a thing a
    user can point at and a thing this can composite, where "layer 3 of the fx
    folder" is neither. It also keeps :func:`flatten_subset` inside the one case
    the pass-through fold is exactly right for: a subset never cuts a group in
    half, so it never has to answer what half a group composites like.

    Hidden rows are left out entirely rather than exported as empty sheets: a
    hidden layer contributes no pixels to any export, and a file full of nothing
    named after it is worse than no file.

    A still document has no splits at all -- there are no tracks to split.
    """
    anim = getattr(doc, "anim", None)
    if anim is None or not anim.frames:
        return []
    from . import groups as gp

    group_of = getattr(doc, "group_of", {}) or {}
    nodes = getattr(doc, "groups", {}) or {}
    out: list[tuple[str, tuple[int, ...]]] = []
    seen: set[int] = set()
    order = [track.uid for track in anim.tracks]
    for track in anim.tracks:
        chain = gp.ancestry(group_of, track.uid)
        root = chain[-1] if chain else track.uid
        if root in seen:
            continue
        seen.add(root)
        if root == track.uid:
            if not track.visible or track.opacity <= 0.0:
                continue
            out.append((track.name, (track.uid,)))
            continue
        node = nodes.get(root)
        if node is None or not node.visible or node.opacity <= 0.0:
            continue
        leaves = tuple(gp.leaves_of(group_of, order, root))
        if leaves:
            out.append((node.name, leaves))
    return out


def index_plane_one(doc: Any, uid: str) -> np.ndarray | None:
    """One frame's own index plane, when the flatten *is* a cel's materialisation.

    An indexed clip whose frames are one visible layer at full opacity in normal
    blend -- which is most pixel-art animation -- already holds the exact answer
    a GIF wants, slot for slot. Handing it over means the export writes the slots
    the user painted rather than re-deriving them from the colours, which is the
    one place a duplicate swatch would otherwise collapse: two slots holding the
    same brown are one brown in a flattened plane, so a colour-keyed lookup has
    to pick one of them and the export stops being slot-stable.

    None whenever anything makes that untrue -- more than one contributing
    layer, a blend mode, an opacity, a matte -- and **the test is equality, not
    structure**. The structural walk only picks the candidate; what decides is
    whether materialising the candidate reproduces the flatten byte for byte.
    A condition list would have to be complete to be safe, and this is safe
    without being complete.
    """
    if getattr(doc, "color_mode", "rgb") != "indexed":
        return None
    anim = getattr(doc, "anim", None)
    if anim is None:
        return None
    frame = next((entry for entry in anim.frames if str(entry.uid) == str(uid)), None)
    if frame is None:
        return None
    stack = doc.frame_stack(frame)
    live = [layer for layer in stack if layer.visible and layer.opacity > 0.0]
    if len(live) != 1 or live[0].indices is None:
        return None
    flat = doc.frame_flat(uid)
    if flat is None or not np.array_equal(flat, live[0].pixels):
        return None
    return live[0].indices


def timing(
    doc: Any, span: tuple[int, int] | None = None
) -> tuple[list[int], list[Any], DirectionalLayout | None]:
    """Durations, tags and layout: the cheap half, read once at the end.

    Pure reads of small structures, unlike ``flatten_one`` -- so there is no
    reason to spread them, and taking them at the end means they describe the
    document as it was when the last frame was flattened rather than as it was
    sixty frames earlier.

    A **partial** span forces the layout to None, and that is a refusal rather
    than an omission: a directional layout says "these frames are four
    directions of a walk, in this fixed grid", which is a claim about the whole
    timeline. Half of it is not a smaller sheet of the same kind -- it is a
    clip, and the row-wrapped grid is the honest answer for one.
    """
    anim = getattr(doc, "anim", None)
    if anim is None or not anim.frames:
        raise ValueError("this document is not animated")
    f0, f1 = _clamped_span(len(anim.frames), span)
    whole = (f0, f1) == (0, len(anim.frames) - 1)
    return (
        [frame.duration_ms for frame in anim.frames[f0 : f1 + 1]],
        list(anim.tags) if whole else rebase_tags(anim.tags, f0, f1),
        anim.layout if whole else None,
    )


def _cell_representatives(frame_cells: Sequence[int | None]) -> list[int]:
    """For each surviving cell, the first original frame index that landed on
    it -- in cell order.

    Slices are authored per frame, and a merged cell has no single frame that
    owns it once the pixels agree; the first occurrence is the same tie-break
    ``build`` itself already makes (it is the frame that *becomes* the cell's
    representative pixels), so the slice geometry and the pixels this function
    is called beside always describe the same frame.
    """
    reps: dict[int, int] = {}
    for i, cell in enumerate(frame_cells):
        if cell is not None and cell not in reps:
            reps[cell] = i
    return [reps[i] for i in range(len(reps))]


def compose(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    tags: Sequence[Any] = (),
    layout: DirectionalLayout | None = None,
    slices: Sequence[dict[str, Any]] | None = None,
    *,
    name: str = "",
    arrange: str | None = None,
    wrap: int | None = None,
    merge: bool = False,
    skip_empty: bool = False,
    trim: bool = False,
    padding: int = 0,
    extrude: int = 0,
) -> tuple[Any, sheetlib.Plan, dict[str, Any]]:
    """``(image, plan, sidecar-without-identity)`` from a snapshot. Off-thread.

    ``layout`` and ``slices`` are positional rather than keyword-only so
    ``compose(*snapshot(doc))`` still holds: each is paired with the frames it
    describes, and a keyword-only element would have made that spelling drop it
    silently. ``arrange``/``wrap``/``merge``/``skip_empty``/``trim``/
    ``padding``/``extrude`` are keyword-only and default to the export this
    always was -- there is nothing in ``snapshot``'s five-tuple for them to
    collide with, because they are an export-time choice, not a document fact.
    """
    image, plan, trims, frame_cells = build(
        frames,
        durations_ms,
        tags,
        name=name,
        layout=layout,
        arrange=arrange,
        wrap=wrap,
        merge=merge,
        skip_empty=skip_empty,
        trim=trim,
        padding=padding,
        extrude=extrude,
    )
    # ``slices_block`` keys by cell index and reads ``snap[cell.index]`` --
    # true by construction when merge/skip_empty are off, since a cell *is* a
    # frame then. Once cells and frames diverge, the cell's slices are its
    # representative frame's, found the same way ``build`` picked the pixels.
    if frame_cells is None:
        cell_slices = slices or ()
    else:
        src = list(slices or ())
        cell_slices = [
            src[i] if 0 <= i < len(src) else None
            for i in _cell_representatives(frame_cells)
        ]
    if trim:
        # The pixels moved -- cropped to each frame's own trim rectangle and
        # pasted flush at its cell's corner -- so the pivots and slices
        # describing those pixels have to move with them, or the sidecar goes
        # on naming a point in a canvas the atlas no longer has.
        cell_slices = _trim_shift_slices(cell_slices, trims)
    block = slices_block(plan, cell_slices)
    return (
        image,
        plan,
        {
            "trims": trims,
            "animation": animation_block(
                plan,
                durations_ms,
                tags,
                layout=layout,
                arrange=arrange,
                wrap=wrap,
                frame_cells=frame_cells,
                trim=trim,
                padding=padding,
                extrude=extrude,
            ),
            "pivots": block["pivots"],
            "slices": block["slices"],
        },
    )


def from_document(
    doc: Any,
    *,
    name: str = "",
    arrange: str | None = None,
    wrap: int | None = None,
    merge: bool = False,
    skip_empty: bool = False,
    trim: bool = False,
    padding: int = 0,
    extrude: int = 0,
) -> tuple[Any, sheetlib.Plan, dict[str, Any]]:
    """The two halves back to back, for a caller that is already on one thread.

    The app is not such a caller -- ``inker_mode.export_sheet`` snapshots on the
    frame thread and composes on a task thread, which is the point of the split.
    """
    return compose(
        *snapshot(doc),
        name=name,
        arrange=arrange,
        wrap=wrap,
        merge=merge,
        skip_empty=skip_empty,
        trim=trim,
        padding=padding,
        extrude=extrude,
    )

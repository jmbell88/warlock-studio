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

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ...pipelines import sheet as sheetlib
from .animation import DIRECTION_ORDER, DirectionalLayout

__all__ = [
    "build",
    "compose",
    "flatten_one",
    "frame_uids",
    "from_document",
    "plan_frames",
    "rebase_tags",
    "scale_slices",
    "slices_block",
    "slices_snapshot",
    "snapshot",
    "timing",
]


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


def plan_frames(
    count: int,
    frame_w: int,
    frame_h: int,
    *,
    name: str = "",
    layout: DirectionalLayout | None = None,
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
    if layout is None:
        columns = max(1, min(count, sheetlib.MAX_ATLAS_PX // frame_w))
        rows = -(-count // columns)  # ceil
    else:
        if count != layout.frame_count:
            raise ValueError(
                f"a {layout.kind} sheet is {layout.frame_count} frames and this "
                f"document has {count}"
            )
        columns, rows = layout.columns, layout.rows
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
) -> dict[str, Any]:
    """The ``animation`` key: which cell is which frame, and for how long."""
    block: dict[str, Any] = {
        "frames": [
            {"cell_index": cell.index, "duration_ms": int(durations_ms[cell.index])}
            for cell in plan.cells
        ],
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
    if layout is not None:
        # Inside the ``animation`` mapping rather than beside it: this says how
        # to *play* the sheet, which is what that key is for. The sidecar's
        # pinned square path never reaches this branch, and the contents here
        # are unpinned, so a reader that has never heard of the key is still
        # correct about the file.
        block["layout"] = {"kind": layout.kind, "directions": list(DIRECTION_ORDER)}
    return block


def build(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    tags: Sequence[Any] = (),
    *,
    name: str = "",
    layout: DirectionalLayout | None = None,
) -> tuple[Any, sheetlib.Plan, dict[int, dict[str, int] | None]]:
    """Composite the frames into one atlas. Returns ``(image, plan, trims)``.

    It composites here rather than through ``sheet.pack``, which reads frames
    from ``Path``s on disk -- these are already in memory, and writing sixteen
    PNGs to a scratch directory to read them straight back would be the slowest
    part of the export by a wide margin.
    """
    from PIL import Image

    if len(frames) != len(durations_ms):
        raise ValueError("every frame needs a duration")
    height, width = frames[0].shape[:2]
    plan = plan_frames(
        len(frames), int(width), int(height), name=name, layout=layout
    )

    atlas = Image.new("RGBA", (plan.width, plan.height), (0, 0, 0, 0))
    trims: dict[int, dict[str, int] | None] = {}
    for cell in plan.cells:
        pixels = frames[cell.index]
        if pixels.shape[:2] != (height, width):
            raise ValueError("every frame of a sheet is the same size")
        tile = Image.fromarray(pixels, "RGBA")
        trims[cell.index] = sheetlib.measure_trim(tile)
        atlas.paste(tile, (cell.x, cell.y))
        tile.close()
    return atlas, plan, trims


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
    doc: Any, span: tuple[int, int] | None = None
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
    """
    uids = frame_uids(doc, span)
    frames = [flatten_one(doc, uid) for uid in uids]
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


def compose(
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    tags: Sequence[Any] = (),
    layout: DirectionalLayout | None = None,
    slices: Sequence[dict[str, Any]] | None = None,
    *,
    name: str = "",
) -> tuple[Any, sheetlib.Plan, dict[str, Any]]:
    """``(image, plan, sidecar-without-identity)`` from a snapshot. Off-thread.

    ``layout`` and ``slices`` are positional rather than keyword-only so
    ``compose(*snapshot(doc))`` still holds: each is paired with the frames it
    describes, and a keyword-only element would have made that spelling drop it
    silently.
    """
    image, plan, trims = build(frames, durations_ms, tags, name=name, layout=layout)
    block = slices_block(plan, slices or ())
    return (
        image,
        plan,
        {
            "trims": trims,
            "animation": animation_block(plan, durations_ms, tags, layout=layout),
            "pivots": block["pivots"],
            "slices": block["slices"],
        },
    )


def from_document(doc: Any, *, name: str = "") -> tuple[Any, sheetlib.Plan, dict[str, Any]]:
    """The two halves back to back, for a caller that is already on one thread.

    The app is not such a caller -- ``inker_mode.export_sheet`` snapshots on the
    frame thread and composes on a task thread, which is the point of the split.
    """
    return compose(*snapshot(doc), name=name)

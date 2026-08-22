"""Sprite-sheet layout, atlas packing and the engine-neutral sidecar.

Deliberately pure: no Blender, no three.js, no filesystem beyond reading the
rendered frames and writing the two outputs. Everything about *where a cell
lands and what it depicts* is decided here, so the browser preview, the Blender
renderer and the sidecar can never disagree about the grid -- and so the grid
is testable without a GPU.

The layout is poses down, view directions across: one row per pose, one column
per yaw. The sidecar records the grid twice over -- as ``columns``/``rows`` for
an importer that just wants to slice a regular atlas, and as a flat ``cells``
list that names what each cell actually contains. The flat list is the
extension point: an animated clip becomes more cells with a ``frame`` above
zero, which costs importers nothing and needs no new format version.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHEET_VERSION = 1

# Eight compass directions, the near-universal choice for 2D-from-3D: enough to
# read a turn, cheap enough to render, and it divides cleanly into the 4- and
# 8-direction conventions every 2D engine already has.
DEFAULT_YAWS = 8

DEFAULT_FRAME_SIZE = 128
FRAME_SIZES = (64, 128, 256, 512)
DEFAULT_ELEVATION = 30.0
LIGHTING = ("flat", "lit")

# Above this, engines start refusing the texture outright rather than scaling
# it. Checked at plan time so the failure is a rejected request, not a job that
# renders for two minutes and then cannot be loaded.
MAX_ATLAS_PX = 8192

# How much room is left around the subject when a cell is framed. One owner,
# here, because two modules have to agree about it and neither may import the
# other: ``blender_worker.op_sheet`` frames what is actually rendered, and
# ``studio.viewer.sheet`` frames the in-app direction preview of it. They were
# two hand-written 1.12s, so a preview that depicted a sheet nobody would get
# was one edit away. This module is where they already meet.
FRAME_MARGIN = 1.12

REST_POSE_NAME = "rest"


@dataclass(frozen=True, slots=True)
class Cell:
    """One sprite in the atlas, and what it depicts."""

    index: int
    row: int
    column: int
    x: int
    y: int
    pose: str | None      # pose id, or None for the unposed rest row
    pose_name: str
    yaw: float            # degrees clockwise from front (-Y in Blender axes)
    frame: int            # always 0 today; the seam animated clips arrive on

    def as_dict(self, w: int, h: int | None = None) -> dict[str, Any]:
        """``h`` defaults to ``w``: a 3D sheet's cells are square by
        construction, and the one caller that is not (an Inker clip, whose
        cells are the canvas) passes both."""
        return {
            "index": self.index,
            "row": self.row,
            "column": self.column,
            "x": self.x,
            "y": self.y,
            "w": w,
            "h": w if h is None else h,
            "pose": self.pose,
            "pose_name": self.pose_name,
            "yaw": self.yaw,
            "frame": self.frame,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    frame_size: int
    columns: int
    rows: int
    yaws: tuple[float, ...]
    elevation: float
    lighting: str
    poses: tuple[dict[str, Any], ...]
    cells: tuple[Cell, ...]
    # Non-square cells, for a source whose frame is a canvas rather than a
    # camera: an Inker animation is 300x180 because that is what the user drew
    # on. Zero means "square, i.e. ``frame_size``", and ``plan()`` never sets
    # them -- so every 3D sheet takes the same arithmetic it always did, to the
    # character. Trailing and defaulted because ``Plan`` is a frozen slots
    # dataclass whose earlier fields have no defaults.
    frame_w: int = 0
    frame_h: int = 0
    # A uniform border around the atlas and gutter between cells, in pixels --
    # Tiled's own margin-and-spacing geometry, restated: ``margin == spacing ==
    # padding``. Zero for every 3D sheet and every Inker export before this
    # field existed, which is what keeps ``width``/``height`` below computing
    # the exact number they always did (the formula collapses to the old one
    # at ``padding == 0``). Trailing and defaulted for the same reason
    # ``frame_w``/``frame_h`` are.
    padding: int = 0

    @property
    def cell_w(self) -> int:
        return self.frame_w or self.frame_size

    @property
    def cell_h(self) -> int:
        return self.frame_h or self.frame_size

    @property
    def width(self) -> int:
        return self.padding + self.columns * (self.cell_w + self.padding)

    @property
    def height(self) -> int:
        return self.padding + self.rows * (self.cell_h + self.padding)


def yaw_angles(count: int = DEFAULT_YAWS) -> tuple[float, ...]:
    """Evenly spaced yaws starting at the front view."""
    return tuple(round(i * 360.0 / count, 4) for i in range(count))


# --- animation ---------------------------------------------------------------
#
# An animated clip is not a new format: it is more cells whose ``frame`` is above
# zero, which is exactly what the flat cells list in the sidecar was built for.
# Interpolation happens here, on the host, for the same reason the grid does --
# it is pure arithmetic, it must be testable without a GPU, and the browser
# preview and the Blender renderer must agree about what frame 3 of a clip is.

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)   # XYZW, three.js order

# Below this the two rotations are parallel and the slerp denominator collapses;
# a straight lerp is then both correct and stable.
SLERP_LINEAR_THRESHOLD = 0.9995

MAX_CLIP_FRAMES = 32


def slerp(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    """Shortest-arc spherical interpolation between two XYZW quaternions."""
    import math

    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:
        # q and -q are the same rotation but interpolate the long way round.
        # Without this a walk cycle counter-rotates through its own midpoint.
        bx, by, bz, bw, dot = -bx, -by, -bz, -bw, -dot
    if dot > SLERP_LINEAR_THRESHOLD:
        out = [
            ax + (bx - ax) * t,
            ay + (by - ay) * t,
            az + (bz - az) * t,
            aw + (bw - aw) * t,
        ]
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        wa = math.sin((1.0 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta
        out = [ax * wa + bx * wb, ay * wa + by * wb, az * wa + bz * wb, aw * wa + bw * wb]
    norm = sum(v * v for v in out) ** 0.5 or 1.0
    return [v / norm for v in out]


EASINGS = ("linear", "ease", "ease_in", "ease_out")


def _ease(t: float, kind: str) -> float:
    """Reshape a 0..1 segment parameter. Pure arithmetic, no table.

    Easing is a *clip* property, not a renderer one: a jump's rise and fall are
    the same two poses either way, and what separates a readable jump from a
    stiff one is how the frames are spaced between them. Doing it here keeps
    that decision on the host with the rest of the clip, where the preview and
    the Blender renderer both read it.
    """
    if kind == "linear":
        return t
    if kind == "ease":            # smoothstep: slow out of A, slow into B
        return t * t * (3.0 - 2.0 * t)
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return t * (2.0 - t)
    raise ValueError(f"easing must be one of {list(EASINGS)}")


def _root(pose: Mapping[str, Any]) -> tuple[float, float, float]:
    values = tuple(float(v) for v in (pose.get("root_translation") or ()))
    return (values + (0.0, 0.0, 0.0))[:3]


def _blend(
    pose_a: Mapping[str, Any],
    pose_b: Mapping[str, Any],
    t: float,
) -> tuple[dict[str, list[float]], tuple[float, float, float]]:
    """One frame's bones and root offset, ``t`` of the way from A to B.

    A bone posed in only one of the two ends interpolates from rest, which is
    what the worker's ``_reset_pose`` already means by an omitted bone.
    """
    bones_a = pose_a.get("bones") or {}
    bones_b = pose_b.get("bones") or {}
    bones = {
        bone: slerp(
            bones_a.get(bone, IDENTITY_QUAT), bones_b.get(bone, IDENTITY_QUAT), t
        )
        for bone in sorted(set(bones_a) | set(bones_b))
    }
    ra, rb = _root(pose_a), _root(pose_b)
    root = tuple(a + (b - a) * t for a, b in zip(ra, rb, strict=True))
    return bones, root  # type: ignore[return-value]


def _record(
    clip_id: str,
    name: str,
    index: int,
    bones: dict[str, list[float]],
    root: tuple[float, float, float],
    *,
    with_root: bool,
    space: str = "node",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        # The row's identity is the clip, not the frame: the worker re-poses
        # per cell within a clip row, which is the one place the group-by-row
        # optimisation does not apply.
        "id": clip_id,
        "name": f"{name} #{index}",
        "frame": index,
        "bones": bones,
    }
    # Emitted only for a clip that is not in the default frame, so a record
    # from the two-key door is byte-identical to the one it always was.
    if space != "node":
        out["space"] = space
    # Emitted for every frame of a clip whose *keys* carry an offset, and for
    # no frame of one whose keys do not. Per clip and not per frame, because a
    # bob passes through zero at its midpoint and a frame that dropped the key
    # there would read as "this frame has no opinion" rather than "this frame
    # is at rest height" -- and because a clip that carried no offset at all
    # then produces byte-identical records to the ones it always did, which is
    # what keeps ``_sheet_root_offsets`` short-circuiting on them.
    if with_root:
        out["root_translation"] = [float(v) for v in root]
    return out


def interpolate(
    pose_a: Mapping[str, Any], pose_b: Mapping[str, Any], frames: int
) -> list[dict[str, Any]]:
    """``frames`` pose records stepping from A toward B.

    The last frame stops *short* of B rather than landing on it, so a clip that
    loops back to A does not hold a duplicate frame at the seam.

    **Root translation is interpolated, not refused.** It used to be refused by
    name, on the correctness argument that a snapshotted library pose carrying
    a root offset would render a clip disagreeing with the pose's own bake. The
    machinery to honour it instead was already everywhere else --
    ``rigging.root_offset_world``, ``queue._sheet_root_offsets`` (keyed by
    ``(pose id, frame)``, which is per *frame* and so already clip-shaped), and
    ``op_sheet``'s per-cell ``root_offset`` -- so the guard was costing the one
    thing a walk cycle needs most, a vertical bob, for a disagreement that no
    longer exists.
    """
    if not 1 <= int(frames) <= MAX_CLIP_FRAMES:
        raise ValueError(f"a clip must be 1-{MAX_CLIP_FRAMES} frames")
    return _expand([pose_a, pose_b], [int(frames)], "linear", land=False)


#: Which frame a stored rotation is in. ``node`` is the pose editor's and every
#: shipped *pose*'s: absolute, relative to the parent joint. ``delta`` is a
#: rotation from the bone's own rest, which is what a clip is authored in --
#: see ``blender_worker.POSE_SPACES`` for why the difference is load-bearing.
POSE_SPACES = ("node", "delta")


def interpolate_clip(
    keys: Sequence[Mapping[str, Any]],
    segments: Sequence[int],
    *,
    closed: bool = False,
    easing: str = "linear",
    space: str = "node",
    clip_id: str | None = None,
) -> list[dict[str, Any]]:
    """An ordered keyframe list expanded into pose records.

    ``segments[i]`` is how many frames the step from ``keys[i]`` onward holds.

    ``closed`` is a cycle -- Idle, Walk, Run: there is one segment per key and
    the last returns to the first, and no segment lands on its far key because
    that key is the next segment's frame 0. That is the seam rule ``interpolate``
    already had, generalised.

    Open is a one-shot -- Attack, Jump: one segment per *gap*, and one extra
    frame at the end that lands exactly on the last key, because a one-shot's
    final frame is a pose the animation holds rather than a seam it hides.

    Two keys and one segment, closed, is exactly the old two-key behaviour,
    which is why ``interpolate`` is now three characters of delegation. Eight
    frames from contact A to contact B was half a stride played straight; four
    keys -- contact A, passing A, contact B, passing B -- is a stride.
    """
    keys = list(keys)
    counts = [int(n) for n in segments]
    if len(keys) < 2:
        raise ValueError("a clip needs at least two keyframes")
    wanted = len(keys) if closed else len(keys) - 1
    if len(counts) != wanted:
        raise ValueError(
            f"a {'cyclic' if closed else 'one-shot'} clip over {len(keys)} keys "
            f"takes {wanted} segment lengths, not {len(counts)}"
        )
    if any(n < 1 for n in counts):
        raise ValueError("every segment holds at least one frame")
    total = sum(counts) + (0 if closed else 1)
    if not 1 <= total <= MAX_CLIP_FRAMES:
        raise ValueError(f"a clip must be 1-{MAX_CLIP_FRAMES} frames")

    if space not in POSE_SPACES:
        raise ValueError(f"space must be one of {list(POSE_SPACES)}")
    return _expand(
        keys, counts, easing, land=not closed, space=space, clip_id=clip_id
    )


def resample_clip(
    keys: Sequence[Mapping[str, Any]],
    segments: Sequence[int],
    frames: int,
    *,
    closed: bool = False,
    easing: str = "linear",
    space: str = "node",
    clip_id: str | None = None,
) -> list[dict[str, Any]]:
    """Sample an authored clip at exactly ``frames`` normalized times.

    Cycles sample ``i / frames`` and therefore never duplicate the seam.
    One-shots sample ``i / (frames - 1)`` and include both endpoints; their
    one-frame form is the first authored pose. Original segment lengths remain
    the timing weights, but a small requested frame count need not allocate a
    frame to every authored segment.
    """
    keys = list(keys)
    counts = [int(n) for n in segments]
    frames = int(frames)
    if len(keys) < 2:
        raise ValueError("a clip needs at least two keyframes")
    wanted = len(keys) if closed else len(keys) - 1
    if len(counts) != wanted or any(n < 1 for n in counts):
        raise ValueError("the authored segment table does not match the clip")
    if not 1 <= frames <= MAX_CLIP_FRAMES:
        raise ValueError(f"a clip must be 1-{MAX_CLIP_FRAMES} frames")
    if space not in POSE_SPACES:
        raise ValueError(f"space must be one of {list(POSE_SPACES)}")
    if easing not in EASINGS:
        raise ValueError(f"easing must be one of {list(EASINGS)}")

    identity = clip_id or ":".join(str(k.get("id") or "") for k in keys)
    name = " -> ".join(str(k.get("name") or "?") for k in keys)
    with_root = any(any(_root(k)) for k in keys)
    total = float(sum(counts))
    out: list[dict[str, Any]] = []
    for index in range(frames):
        phase = index / frames if closed else (index / (frames - 1) if frames > 1 else 0.0)
        position = phase * total
        if not closed and phase >= 1.0:
            a = b = keys[-1]
            local = 0.0
        else:
            offset = 0.0
            segment = len(counts) - 1
            for candidate, length in enumerate(counts):
                if position < offset + length:
                    segment = candidate
                    break
                offset += length
            a = keys[segment]
            b = keys[(segment + 1) % len(keys)]
            local = (position - offset) / counts[segment]
        bones, root = _blend(a, b, _ease(local, easing))
        out.append(
            _record(
                identity,
                name,
                index,
                bones,
                root,
                with_root=with_root,
                space=space,
            )
        )
    return out


def _expand(
    keys: Sequence[Mapping[str, Any]],
    counts: Sequence[int],
    easing: str,
    *,
    land: bool,
    space: str = "node",
    clip_id: str | None = None,
) -> list[dict[str, Any]]:
    """The expansion itself, shared by the two-key door and the multi-key one.

    ``counts[i]`` steps from ``keys[i]`` toward ``keys[i + 1]``, wrapping, and
    never lands on that far key. ``land`` appends the one extra frame that
    finishes a one-shot on its final pose.

    ``clip_id`` names the row outright. Derived from the keys' own ids when it
    is not given, which is right for :func:`interpolate`: its two poses are
    *pose library* rows, and those carry an ``id``. It is wrong for a clip
    library, whose key poses are built by ``rigging._load_clip_library`` as
    ``{"name", "bones"[, "root_translation"]}`` with no ``id`` at all -- so the
    join collapsed to ``":" * (len(keys) - 1)`` and every clip with the same
    number of keys shared one identity. ``walk`` and ``run`` both have four,
    and ``_q_troupe._charsheet`` keys its lookup on ``(id, frame)``: run
    overwrote walk, and all 64 walk cells of every character sheet rendered the
    run cycle. See :func:`warlock.clips.expand_clips`, which passes the name.
    """
    if clip_id is None:
        clip_id = ":".join(str(k.get("id") or "") for k in keys)
    name = " -> ".join(str(k.get("name") or "?") for k in keys)
    with_root = any(any(_root(k)) for k in keys)
    out: list[dict[str, Any]] = []
    for i, count in enumerate(counts):
        a, b = keys[i], keys[(i + 1) % len(keys)]
        for j in range(count):
            bones, root = _blend(a, b, _ease(j / count, easing))
            out.append(
                _record(
                    clip_id, name, len(out), bones, root,
                    with_root=with_root, space=space,
                )
            )
    if land:
        bones, root = _blend(keys[-1], keys[-1], 0.0)
        out.append(
            _record(
                clip_id, name, len(out), bones, root, with_root=with_root, space=space
            )
        )
    return out


def check_atlas_size(width: int, height: int) -> None:
    """Refuse an atlas an engine would refuse, before anything renders.

    Extracted from ``plan`` so the other builder of a ``Plan`` -- the Inker
    exporter, which cannot use ``plan`` because its job is poses by yaws --
    applies the same limit rather than its own approximation of it. The
    original message's shape is unchanged, deliberately: it is what the
    existing test matches on and what a user has seen before -- padding is
    named alongside the other two causes rather than replacing either of
    them, since Inker's sheet export can hit this ceiling three ways now.
    """
    if max(width, height) > MAX_ATLAS_PX:
        raise ValueError(
            f"that sheet would be {width}x{height}px; the limit is {MAX_ATLAS_PX}px "
            "-- use a smaller frame size, fewer poses, or less padding"
        )


def plan(
    poses: Sequence[Mapping[str, Any]],
    *,
    frame_size: int = DEFAULT_FRAME_SIZE,
    elevation: float = DEFAULT_ELEVATION,
    lighting: str = "flat",
    yaws: int = DEFAULT_YAWS,
) -> Plan:
    """Work out the grid, or raise ValueError.

    ``poses`` is the saved-pose records to render, in row order. An empty
    sequence is not an error: it means the unposed mesh, which is the whole
    point for a prop that has no rig.
    """
    if frame_size not in FRAME_SIZES:
        raise ValueError(f"frame_size must be one of {list(FRAME_SIZES)}")
    if lighting not in LIGHTING:
        raise ValueError(f"lighting must be one of {list(LIGHTING)}")
    if not -89.0 <= elevation <= 89.0:
        raise ValueError("elevation must be between -89 and 89 degrees")
    if yaws < 1:
        raise ValueError("a sheet needs at least one view direction")

    rows_in = list(poses) or [{"id": None, "name": REST_POSE_NAME}]
    check_atlas_size(yaws * frame_size, len(rows_in) * frame_size)

    angles = yaw_angles(yaws)
    cells: list[Cell] = []
    for row, pose in enumerate(rows_in):
        for column, yaw in enumerate(angles):
            cells.append(
                Cell(
                    index=len(cells),
                    row=row,
                    column=column,
                    x=column * frame_size,
                    y=row * frame_size,
                    pose=pose.get("id"),
                    pose_name=str(pose.get("name") or REST_POSE_NAME),
                    yaw=yaw,
                    # A record produced by interpolate() carries its own frame;
                    # an ordinary saved pose has none and defaults to 0.
                    frame=int(pose.get("frame", 0)),
                )
            )
    return Plan(
        frame_size=frame_size,
        columns=yaws,
        rows=len(rows_in),
        yaws=angles,
        elevation=float(elevation),
        lighting=lighting,
        poses=tuple(dict(p) for p in rows_in),
        cells=tuple(cells),
    )


def measure_trim(image: Any) -> dict[str, int] | None:
    """The alpha bounding box within one rendered frame, or None if it is empty.

    Handed to importers that pack tightly: a 128px cell whose subject occupies
    40x90 of it wastes most of its texture, and the trim rectangle is what lets
    a packer reclaim that without re-rendering. Measured rather than computed
    from the bbox because the silhouette, not the bounding volume, is what a
    packer cares about.
    """
    alpha = image.convert("RGBA").getchannel("A")
    box = alpha.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def extrude_edges(atlas: Any, x: int, y: int, w: int, h: int, margin: int) -> None:
    """Replicate one placed rectangle's border pixels outward into its gutter,
    in place -- so a filtered texture sampling just past the sprite's edge
    finds that sprite's own colour rather than its neighbour's.

    Ported byte-for-byte, ordering included, from
    ``studio.packwright.compose._extrude``: sides first, one-pixel-wide slices
    broadcast across the gutter's width, then top and bottom across the
    *widened* span -- which is what carries the columns just written into the
    four corners with no separate corner case, and is why the order here
    matters rather than being tidiness.

    Written against plain slice assignment rather than importing numpy, so
    this module's only import stays Pillow, lazily -- ``atlas`` need only
    support numpy-style ``__getitem__``/``__setitem__``, which is what every
    caller already hands it.

    The room this needs -- ``margin`` no more than half of whatever gutter
    surrounds ``(x, y, w, h)`` -- is guaranteed upstream, not here: the same
    division of labour as ``packwright.layout.PackSettings``, which refuses
    ``padding < extrude * 2`` at construction rather than clamping silently.
    """
    if margin <= 0:
        return
    atlas[y : y + h, x - margin : x] = atlas[y : y + h, x : x + 1]
    atlas[y : y + h, x + w : x + w + margin] = atlas[y : y + h, x + w - 1 : x + w]
    span = slice(x - margin, x + w + margin)
    atlas[y - margin : y, span] = atlas[y : y + 1, span]
    atlas[y + h : y + h + margin, span] = atlas[y + h - 1 : y + h, span]


def pack(
    sheet: Plan, frames: Mapping[int, Path], out_png: Path
) -> dict[int, dict[str, int] | None]:
    """Composite the rendered frames into one RGBA atlas.

    Returns each cell's alpha bounding box, measured here because every frame
    is already open and decoded at this point.

    A missing or wrong-sized frame raises rather than silently leaving a hole:
    a sheet with an invisible gap in it looks like a modelling problem, and the
    user would go looking in the wrong place.
    """
    from PIL import Image

    size = (sheet.cell_w, sheet.cell_h)
    atlas = Image.new("RGBA", (sheet.width, sheet.height), (0, 0, 0, 0))
    trims: dict[int, dict[str, int] | None] = {}
    try:
        for cell in sheet.cells:
            path = frames.get(cell.index)
            if path is None or not path.exists():
                raise ValueError(f"no rendered frame for cell {cell.index}")
            with Image.open(path) as frame:
                frame = frame.convert("RGBA")
                if frame.size != size:
                    frame = frame.resize(size, Image.LANCZOS)
                trims[cell.index] = measure_trim(frame)
                atlas.paste(frame, (cell.x, cell.y))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(out_png, "PNG")
    finally:
        atlas.close()
    return trims


def sidecar(
    sheet: Plan,
    *,
    sheet_id: str,
    source_job: str,
    image: str,
    created: float,
    name: str = "",
    pivot: tuple[float, float] | None = None,
    trims: Mapping[int, dict[str, int] | None] | None = None,
    animation: Mapping[str, Any] | None = None,
    pivots: Mapping[int, tuple[float, float]] | None = None,
    slices: Mapping[int, list[dict[str, Any]]] | None = None,
    slices_conflict: Mapping[int, list[int]] | None = None,
) -> dict[str, Any]:
    """The engine-neutral description of the atlas next to it.

    Engine-neutral means no Godot ``AtlasTexture``, no Unity ``SpriteMetaData``
    -- just pixel rectangles and what each one shows, in the plainest JSON that
    can carry it. Anything more opinionated would have to be rewritten for the
    second engine anyone tries.

    ``animation``, when given, adds one ``"animation"`` key: ``frames``, each a
    ``{cell_index, duration_ms}``, and ``tags``. It has **two** builders now --
    the Inker exporter, for a drawn clip, and ``charsheet.animation_block``,
    for a rendered character sheet, which is what closed the gap where a
    rendered sheet reached an engine as frame indices with no fps and no loop
    tags. Two builders, one format: ``charsheet`` deliberately emits the same
    keys with the same spellings (``repeat: 1`` for a play-once tag, omitted
    otherwise) so ``version: 1`` cannot come to mean two subtly different
    documents. A third writer should extend one of those rather than appear.

    ``pivots`` and ``slices`` are that exporter's other two, keyed by cell index
    and both **additive with no version bump**. A pivot overrides the constant
    below for one cell, so a drawn clip whose pivot moves with the animation can
    say so where a rendered sheet has one answer for every cell. ``slices``
    lands as a per-cell ``"slices"`` list, emitted only where there is one --
    which is what keeps every sheet this build wrote before them byte-identical,
    and is pinned by the square-sidecar equality test.

    ``slices_conflict`` is the fourth, and the same additive-with-no-version-bump
    rule governs it: a top-level ``{cell index: [dropped frame indices]}`` map,
    written **only when it is non-empty**. Merge and skip-empty collapse frames
    by pixels while slices are authored per frame, so a merged-away frame's own
    rectangles are dropped in favour of its representative's -- correctly, since
    a cell has one geometry, but until now silently. A reader that has never
    heard of the key sees exactly the file it saw before, because an export with
    nothing to report does not write it at all.

    On a non-square plan ``frame_size`` is emitted as **0** and ``frame_w`` /
    ``frame_h`` carry the truth. Zero is a loud wrong answer rather than a quiet
    one: putting the width in ``frame_size`` would let an importer that reads
    only that key slice the atlas correctly across and wrongly down. Every cell
    also carries its own ``w``/``h``, so a cell-driven importer needs none of
    this.
    """
    # The projected ground origin, in pixels within a cell. Identical for every
    # cell by construction: the camera is framed once from the rest bbox and
    # only spins, so the subject's origin lands in the same place in every
    # direction. That stability is the property an engine needs to place a
    # sprite without it drifting as the character turns.
    px, py = pivot if pivot is not None else (sheet.cell_w / 2.0, float(sheet.cell_h))
    cells = []
    for c in sheet.cells:
        entry = c.as_dict(sheet.cell_w, sheet.cell_h)
        # Per cell where the caller has an answer for that cell, the constant
        # otherwise -- so a sheet that carries none is the sheet it always was.
        where = (pivots or {}).get(c.index)
        entry["pivot_x"] = px if where is None else float(where[0])
        entry["pivot_y"] = py if where is None else float(where[1])
        entry["trim"] = (trims or {}).get(c.index)
        block = (slices or {}).get(c.index)
        if block:
            entry["slices"] = list(block)
        cells.append(entry)
    payload = {
        "version": SHEET_VERSION,
        "id": sheet_id,
        "name": name or sheet_id,
        "source_job": source_job,
        "created": created,
        "image": image,
        "frame_size": sheet.frame_size,
        "columns": sheet.columns,
        "rows": sheet.rows,
        "width": sheet.width,
        "height": sheet.height,
        "elevation": sheet.elevation,
        "lighting": sheet.lighting,
        # Degrees clockwise from the front view, which is -Y in the Blender
        # axes the templates and the rig are expressed in.
        "yaws": list(sheet.yaws),
        "poses": [
            {"id": p.get("id"), "name": str(p.get("name") or REST_POSE_NAME)}
            for p in sheet.poses
        ],
        "cells": cells,
    }
    if sheet.frame_w or sheet.frame_h:
        payload["frame_w"], payload["frame_h"] = sheet.cell_w, sheet.cell_h
    if animation is not None:
        payload["animation"] = dict(animation)
    # Only when there is something to say -- see the docstring: an empty map
    # writes no key, which is what makes this additive rather than a format
    # change every existing sheet would have to be re-checked against.
    if slices_conflict:
        payload["slices_conflict"] = {
            int(cell): [int(frame) for frame in frames]
            for cell, frames in sorted(slices_conflict.items())
        }
    return payload

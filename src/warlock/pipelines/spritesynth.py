"""Turning one finished drawing into a candidate sprite-sheet atlas.

The counterpart to ``pipelines/pixelsheet``, and deliberately the opposite
trade. There, eight directions are exact orthographic renders of one mesh, so
the geometry agrees by construction and only the *style* is generated. Here
there is no mesh -- there is a single 2D reference -- so the geometry has to be
generated too, and everything in this module exists to stop four or sixteen
independently-imagined poses from disagreeing with each other:

* **One generation per candidate.** The whole atlas is denoised in one 1024px
  latent under one seed, exactly as a band is there. Four separate 512px
  generations would give four characters wearing four shirts.
* **A pose guide, not a hope.** Each cell's rectangle carries a stick figure
  fed to the canny ControlNet, so where the limbs go is imposed rather than
  requested. The guide is already line art in canny space, so it is handed to
  the ControlNet *directly* -- running ``cv2.Canny`` over it would outline every
  stroke and double it.
* **One palette across the atlas**, via ``pixelsheet.quantize_shared``, for the
  reason that function's docstring gives.
* **A shared baseline**, because the thing that reads as a broken animation is
  not a slightly wrong arm, it is a character whose feet move between frames.

Pure: Pillow and NumPy inside the functions, no torch, no service/queue/studio
imports, and no decisions about where a file goes. The queue owns the model
calls and the paths; this module owns the geometry, the mattes and the
arithmetic, and is therefore testable with a Pillow-drawn fixture and no GPU.

Geometry note. The atlas is 1024px square because that is one SDXL frame -- the
same pin every other generation path here obeys. 2x2 turnaround cells are
therefore 512px and 4x4 walk cells are 256px, both exact. The logical sizes the
user may reduce to (32/48/64) are *not* required to divide those, which is why
``reduce_atlas`` exists instead of ``pixelsheet.downscale``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

log = logging.getLogger(__name__)

SPRITE_DRAFT_VERSION = 1

# One SDXL frame, and the reason the cell sizes below are what they are.
ATLAS_PX = 1024

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "sprite_guides"

# The four directions, in the order a cell index means, and the yaw each one
# stands for. These are the *studio* side's numbers too -- ``inker.animation``
# repeats them because a headless package may not import ``pipelines``, and
# ``tests/test_sprite_geometry_agreement.py`` is the single owner of the
# agreement between the two copies.
DIRECTION_ORDER = ("front", "left", "right", "back")
DIRECTION_YAWS = {"front": 0, "left": 90, "right": 270, "back": 180}

SHEET_TYPES = ("turnaround", "walk")

# The two grids, as literal tables rather than a loop, so that the order a cell
# index means is readable in one glance and cannot be changed by a clever
# arithmetic edit. ``(direction, frame, row, col)``.
_TURNAROUND_TABLE = (
    ("front", 0, 0, 0),
    ("left", 0, 0, 1),
    ("right", 0, 1, 0),
    ("back", 0, 1, 1),
)

_WALK_TABLE = tuple(
    (direction, frame, row, frame)
    for row, direction in enumerate(DIRECTION_ORDER)
    for frame in range(4)
)

# Below this share of a cell, the corner flood fill escaped through the
# subject's rim and ate it rather than measuring it. Same judgement as
# ``reference._LEAK_FLOOR`` and the same number: a cell that is 1% subject is
# very nearly always a leak, and the honest response is to leave the cell
# opaque and say so, not to publish a hole.
MIN_MATTE_FRACTION = 0.01

# How far a cell's subject area may sit from the atlas median before it is
# worth a sentence. Wide on purpose -- a crouching walk frame is legitimately
# smaller than a standing one, and this is a warning, never a refusal.
OCCUPANCY_LOW = 0.5
OCCUPANCY_HIGH = 2.0

# The front cell may be replaced by the user's own drawing only if the two
# agree about the subject's proportions this closely. A first guess; costs a
# warning either way, never a draft.
FRONT_ASPECT_LOW = 0.75
FRONT_ASPECT_HIGH = 1.33

# Machine keys for the per-cell warnings, beside the sentences, for the reason
# ``reference.REFUSAL_CODES`` gives: the sentences are rewritten whenever the
# wording improves, and anything that counts them across a corpus needs a key
# that is not allowed to drift.
WARNING_CODES = ("empty", "clipped", "occupancy", "unmatted")


@dataclass(frozen=True, slots=True)
class Cell:
    """One cell's rectangle and what it depicts. Atlas pixels."""

    name: str
    frame: int
    yaw: int
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass(frozen=True, slots=True)
class SheetGeometry:
    """The fixed grid one sheet type is generated on."""

    kind: str
    columns: int
    rows: int
    cell_w: int
    cell_h: int
    cells: tuple[Cell, ...]

    @property
    def frames_per_direction(self) -> int:
        return len(self.cells) // len(DIRECTION_ORDER)


def _build(kind: str, table: tuple[tuple[str, int, int, int], ...]) -> SheetGeometry:
    columns = max(col for _, _, _, col in table) + 1
    rows = max(row for _, _, row, _ in table) + 1
    cell_w = ATLAS_PX // columns
    cell_h = ATLAS_PX // rows
    cells = tuple(
        Cell(
            name=name,
            frame=frame,
            yaw=DIRECTION_YAWS[name],
            row=row,
            col=col,
            x=col * cell_w,
            y=row * cell_h,
            w=cell_w,
            h=cell_h,
        )
        for name, frame, row, col in table
    )
    return SheetGeometry(
        kind=kind, columns=columns, rows=rows, cell_w=cell_w, cell_h=cell_h, cells=cells
    )


GEOMETRY: dict[str, SheetGeometry] = {
    "turnaround": _build("turnaround", _TURNAROUND_TABLE),
    "walk": _build("walk", _WALK_TABLE),
}


def geometry(sheet_type: str) -> SheetGeometry:
    """The grid for ``sheet_type``. Unknown raises rather than defaulting.

    Defaulting here would generate a turnaround for a typo'd walk request and
    then publish it under the caller's name, which is a wrong sheet nobody
    asked for rather than an error somebody can read.
    """
    try:
        return GEOMETRY[sheet_type]
    except KeyError:
        raise ValueError(
            f"unknown sprite sheet type {sheet_type!r}; "
            f"this module handles {', '.join(SHEET_TYPES)}"
        ) from None


#: ``service.validation.MAX_SEED``, restated rather than imported: a pipeline
#: may not import the service layer. ``tests/test_sprite_followup.py`` pins the
#: two together, so the copy cannot drift.
MAX_SEED = 2**31 - 1

#: The candidate letters, in the order :func:`candidate_seed` steps through
#: them. ``rigging.SPRITE_CANDIDATES`` is the same tuple and is deliberately not
#: imported: that module knows about paths and this one may not.
CANDIDATES: tuple[str, ...] = ("a", "b")


def candidate_seed(base_seed: int, letter: str) -> int:
    """One candidate's seed, derived from the source job's own stored seed.

    Derived rather than drawn: the prompt-driven path mints its follow-up *in
    the worker*, which has no random seed of its own and should not grow one --
    every seed in this codebase comes from a door or from arithmetic over one
    that did. Deriving them also
    makes the whole two-step chain reproducible from the one seed the user can
    see and lock: the same character, and the same pair of sheets from it.

    The two must differ, which is ``create_sprite_synthesis``' own rule and its
    reason: the deliverable is a *pair* to pick between, and two candidates one
    apart in seed space come back looking like the same picture twice. A large
    odd multiplier per letter is what keeps them unrelated rather than adjacent.
    """
    if letter not in CANDIDATES:
        raise ValueError(f"unknown sprite candidate {letter!r}")
    # Masked, not modulo'd by a prime: the caller's contract is only "a legal
    # seed", and a mask is exactly that. Knuth's 32-bit golden-ratio multiplier,
    # for the property this needs -- neither step is a small offset in seed
    # space, so no two candidates land next to each other.
    mixed = int(base_seed) + (CANDIDATES.index(letter) + 1) * 2_654_435_761
    return mixed & MAX_SEED


# --- guide templates --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GuidePose:
    """One cell's stick figure, in normalised cell space (0..1, y down)."""

    name: str
    frame: int
    points: dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class GuideTemplate:
    kind: str
    head_radius: float
    head_point: str
    segments: tuple[tuple[str, str], ...]
    poses: tuple[GuidePose, ...]


def _parse_template(raw: dict[str, Any], geom: SheetGeometry) -> GuideTemplate:
    kind = str(raw["kind"])
    if kind != geom.kind:
        raise ValueError(f"guide template says {kind!r} but was loaded for {geom.kind!r}")
    head_point = str(raw.get("head_point") or "head")
    segments = tuple((str(a), str(b)) for a, b in raw["segments"])

    poses: list[GuidePose] = []
    for entry in raw["poses"]:
        points = {
            str(name): (float(xy[0]), float(xy[1])) for name, xy in entry["points"].items()
        }
        if head_point not in points:
            raise ValueError(
                f"guide pose {entry.get('name')!r} has no {head_point!r} point"
            )
        for a, b in segments:
            # Validated at load, not at draw: an unknown joint name is a typo
            # in data, and the only useful moment to hear about it is before a
            # 20-second generation runs against a guide missing a leg.
            if a not in points or b not in points:
                raise ValueError(
                    f"guide pose {entry.get('name')!r} is missing a point for "
                    f"segment {a!r}-{b!r}"
                )
        for name, (px, py) in points.items():
            if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
                raise ValueError(
                    f"guide point {name!r} of pose {entry.get('name')!r} is "
                    "outside its cell"
                )
        poses.append(
            GuidePose(name=str(entry["name"]), frame=int(entry["frame"]), points=points)
        )

    if len(poses) != len(geom.cells):
        raise ValueError(
            f"guide template for {kind!r} has {len(poses)} poses but the grid "
            f"has {len(geom.cells)} cells"
        )
    for pose, cell in zip(poses, geom.cells, strict=True):
        if (pose.name, pose.frame) != (cell.name, cell.frame):
            raise ValueError(
                f"guide pose {pose.name!r}/{pose.frame} does not match cell "
                f"{cell.name!r}/{cell.frame} -- the template is in the wrong order"
            )
    return GuideTemplate(
        kind=kind,
        head_radius=float(raw["head_radius"]),
        head_point=head_point,
        segments=segments,
        poses=tuple(poses),
    )


def load_guide_template(sheet_type: str) -> GuideTemplate:
    """Read and validate ``templates/sprite_guides/<type>.json``.

    Raises rather than degrading, unlike ``rigging._load_templates``: there is
    exactly one template per sheet type, so skipping a bad one would leave the
    feature with no guide at all and generate four unposed characters -- an
    error is the useful outcome, and the templates ship with the package.
    """
    geom = geometry(sheet_type)
    path = TEMPLATE_DIR / f"{sheet_type}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _parse_template(raw, geom)


def render_guide(geom: SheetGeometry, template: GuideTemplate) -> PILImage:
    """The atlas-sized pose guide: white stick figures on black.

    Handed to the ControlNet as the hint *directly*. It is already line art in
    canny space -- white strokes on black -- and running the detector over it
    would return the outline of each stroke, i.e. two lines where the guide
    means one, which is exactly the "why does my character have four legs"
    failure. ``control.edge_fraction`` is still recorded against it, so a guide
    that drew nothing is an answerable question.
    """
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (ATLAS_PX, ATLAS_PX), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for cell, pose in zip(geom.cells, template.poses, strict=True):
        width = max(2, cell.w // 128)

        def at(name: str, _cell: Cell = cell, _pose: GuidePose = pose) -> tuple[float, float]:
            px, py = _pose.points[name]
            return (_cell.x + px * _cell.w, _cell.y + py * _cell.h)

        for a, b in template.segments:
            draw.line([at(a), at(b)], fill=(255, 255, 255), width=width)
        hx, hy = at(template.head_point)
        r = template.head_radius * cell.w
        draw.ellipse(
            [hx - r, hy - r, hx + r, hy + r], outline=(255, 255, 255), width=width
        )
    return canvas


# --- slicing, matting, alignment --------------------------------------------


def split(atlas: PILImage, geom: SheetGeometry) -> list[PILImage]:
    """The atlas cut on its predetermined rectangles.

    The grid is never re-detected from pixels. A generated atlas whose content
    is a few pixels off the grid still splits identically -- the alternative,
    finding the seams, turns one mis-registered generation into a cascade of
    differently-sized cells that nothing downstream can lay out.
    """
    return [atlas.crop(cell.box) for cell in geom.cells]


def matte_cells(atlas: PILImage, geom: SheetGeometry) -> tuple[PILImage, tuple[bool, ...]]:
    """Per-cell corner flood fill, returned as one RGBA atlas plus which cells took.

    Per cell rather than whole-atlas because each cell has its own four corners
    of background, and a single fill seeded at the atlas corners would stop at
    the first subject it met. The heavy matting model is not used here: it is
    ~11.5s an image and the prompt asks for a plain background, so it is spent
    once on the *source* reference and not sixteen times on a draft.

    A cell whose fill leaked through the subject's rim is left fully opaque and
    reported as ``unmatted`` rather than published with a hole -- and that also
    keeps ``quantize_shared`` safe, which refuses an atlas with no opaque pixels
    at all.
    """
    import numpy as np
    from PIL import Image

    from . import reference

    rgb = atlas.convert("RGB")
    out = np.dstack(
        [np.asarray(rgb), np.full(rgb.size[::-1], 255, dtype=np.uint8)]
    ).copy()
    matted: list[bool] = []
    for cell in geom.cells:
        crop = rgb.crop(cell.box)
        try:
            mask = reference.subject_mask(crop)
        except Exception:
            # A cell that cannot be measured costs itself its matte, not the
            # draft: the same rule every other per-item loop here follows.
            log.exception("could not matte sprite cell %s/%s", cell.name, cell.frame)
            matted.append(False)
            continue
        if float(np.asarray(mask).mean()) < MIN_MATTE_FRACTION:
            matted.append(False)
            continue
        alpha = np.where(np.asarray(mask), 255, 0).astype(np.uint8)
        out[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w, 3] = alpha
        matted.append(True)
    return (Image.fromarray(out, "RGBA"), tuple(matted))


def _cell_bounds(alpha, cell: Cell) -> tuple[int, int] | None:
    """``(top, bottom)`` rows of the subject inside ``cell``, cell-relative."""
    import numpy as np

    sub = alpha[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
    rows = np.nonzero((sub > 0).any(axis=1))[0]
    if rows.size == 0:
        return None
    return (int(rows[0]), int(rows[-1]))


def baseline_align(atlas_rgba: PILImage, geom: SheetGeometry) -> PILImage:
    """Put every occupied cell's subject on one shared baseline.

    Vertical only, and horizontal is deliberately untouched: a walk cycle's
    horizontal padding is stride, and centring it would remove the very motion
    the frames exist to show. The target is the *median* of the cells' bottom
    edges rather than a fixed fraction of the cell, so a sheet the model drew
    consistently low is not dragged upward for no reason -- and each shift is
    clamped so no cell is ever pushed off its own rectangle. What reads as a
    broken animation is feet that move between frames.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(atlas_rgba.convert("RGBA")).copy()
    alpha = arr[:, :, 3]
    bounds = {i: _cell_bounds(alpha, c) for i, c in enumerate(geom.cells)}
    bottoms = [b[1] for b in bounds.values() if b is not None]
    if not bottoms:
        return Image.fromarray(arr, "RGBA")
    target = int(np.median(bottoms))

    for index, cell in enumerate(geom.cells):
        bound = bounds[index]
        if bound is None:
            continue
        top, bottom = bound
        shift = target - bottom
        shift = max(-top, min(cell.h - 1 - bottom, shift))
        if shift == 0:
            continue
        block = arr[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w].copy()
        block[:, :, 3] = 0
        moved = np.zeros_like(block)
        src = arr[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
        if shift > 0:
            moved[shift:] = src[: cell.h - shift]
            moved[:shift] = block[:shift]
        else:
            moved[: cell.h + shift] = src[-shift:]
            moved[cell.h + shift :] = block[cell.h + shift :]
        arr[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w] = moved
    return Image.fromarray(arr, "RGBA")


def reduce_atlas(atlas: PILImage, geom: SheetGeometry, logical_size: int) -> PILImage:
    """One NEAREST resize of the whole atlas to ``logical_size`` cells.

    Whole-atlas and not per cell, so a cell boundary in the output is at the
    same place whichever side of it you compute from. ``pixelsheet.downscale``
    cannot serve here: it takes an integer stride, and 48 does not divide 512
    or 256. The boundaries stay exact anyway, because the *columns* and *rows*
    divide: an output cell edge lands at ``col * logical_size``, which is an
    integer for every column, so no output pixel ever straddles two cells.
    """
    from PIL import Image

    if logical_size < 1:
        raise ValueError("the logical cell size must be at least 1")
    return atlas.convert("RGBA").resize(
        (geom.columns * logical_size, geom.rows * logical_size), Image.NEAREST
    )


# --- warnings ---------------------------------------------------------------


def _warning(cell: Cell, code: str, detail: str) -> dict[str, Any]:
    if code not in WARNING_CODES:  # pragma: no cover - guard against a typo
        raise ValueError(f"unknown sprite warning code {code!r}")
    return {"cell": cell.name, "frame": cell.frame, "code": code, "detail": detail}


def structural_warnings(
    atlas_rgba: PILImage, geom: SheetGeometry, matted: tuple[bool, ...] | list[bool]
) -> list[dict[str, Any]]:
    """What is wrong with this candidate, per cell, as codes and sentences.

    Warnings and never refusals. A candidate is two of a pair the user is about
    to look at: throwing one away because a foot touches an edge would leave
    them comparing one draft against nothing, which is worse than a sentence
    under a thumbnail saying which foot.
    """
    from . import reference

    reports = []
    for cell in geom.cells:
        crop = atlas_rgba.convert("RGBA").crop(cell.box)
        reports.append(reference.measure(crop))

    occupied = [r.occupancy for r in reports if r.bbox is not None and r.occupancy > 0]
    median = sorted(occupied)[len(occupied) // 2] if occupied else 0.0

    out: list[dict[str, Any]] = []
    for cell, report, took in zip(geom.cells, reports, matted, strict=True):
        if not took:
            out.append(
                _warning(
                    cell,
                    "unmatted",
                    "The background of this cell could not be separated, so it "
                    "was left opaque.",
                )
            )
        if report.bbox is None or report.occupancy <= 0:
            out.append(_warning(cell, "empty", "This cell came out empty."))
            continue
        if report.touches:
            out.append(
                _warning(
                    cell,
                    "clipped",
                    "The subject runs off the "
                    + ", ".join(report.touches)
                    + " of this cell.",
                )
            )
        if median > 0 and not (
            median * OCCUPANCY_LOW <= report.occupancy <= median * OCCUPANCY_HIGH
        ):
            out.append(
                _warning(
                    cell,
                    "occupancy",
                    f"The subject fills {report.occupancy * 100:.0f}% of this "
                    f"cell against {median * 100:.0f}% across the sheet.",
                )
            )
    return out


# --- front-cell preservation ------------------------------------------------


def front_fits(source_report, front_report) -> tuple[bool, str]:
    """Whether the user's own drawing may stand in for the generated front cell.

    A turnaround's front view is the one cell where the reference *is* the
    answer -- the model was only ever asked to imagine the other three. Pasting
    it in makes the sheet's identity exact rather than persuaded. The gate is
    about whether the two agree on the subject's proportions: a source that is
    cropped, is two objects, or is a different shape from what the model drew
    would land as a foreign body in the middle of the sheet instead.

    Returns the verdict and the sentence for it either way, because the sidecar
    records why *not* as well as why.
    """
    if not source_report.ok:
        return (False, "The reference itself was refused, so it was not pasted in.")
    if source_report.touches:
        return (
            False,
            "The reference touches the edge of its frame, so it was not pasted in.",
        )
    if source_report.components > 1:
        return (
            False,
            "The reference is more than one object, so it was not pasted in.",
        )
    if source_report.bbox is None or front_report.bbox is None:
        return (False, "The front cell could not be measured, so nothing was pasted in.")

    def aspect(bbox: tuple[int, int, int, int]) -> float:
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        return w / h

    ratio = aspect(source_report.bbox) / aspect(front_report.bbox)
    if not (FRONT_ASPECT_LOW <= ratio <= FRONT_ASPECT_HIGH):
        return (
            False,
            f"The reference is {ratio:.2f}x the generated front view's "
            "proportions, so it was not pasted in.",
        )
    return (True, "The front view is the reference drawing itself.")


def preserve_front(
    atlas_rgba: PILImage, geom: SheetGeometry, subject: PILImage
) -> PILImage:
    """Paste the matted reference into the front cell, on the shared baseline.

    Before reduction and quantization, never after: the paste has to go through
    the same NEAREST reduction and share the same median-cut palette as the
    other cells, or the one cell that is definitely the right character is also
    the one cell that does not match the sheet.
    """
    import numpy as np
    from PIL import Image

    if geom.kind != "turnaround":
        raise ValueError("front preservation is a turnaround-only step")
    front = next(c for c in geom.cells if c.name == "front" and c.frame == 0)

    arr = np.asarray(atlas_rgba.convert("RGBA")).copy()
    alpha = arr[:, :, 3]

    heights = []
    baselines = []
    for cell in geom.cells:
        bound = _cell_bounds(alpha, cell)
        if bound is None:
            continue
        heights.append(bound[1] - bound[0] + 1)
        baselines.append(bound[1])
    if not heights:
        return Image.fromarray(arr, "RGBA")
    target_h = int(np.median(heights))
    baseline = int(np.median(baselines))

    src = subject.convert("RGBA")
    box = src.getbbox()
    if box is None:
        return Image.fromarray(arr, "RGBA")
    src = src.crop(box)
    scale = target_h / max(1, src.height)
    nw = max(1, round(src.width * scale))
    nh = max(1, round(src.height * scale))
    # NEAREST like everything else on this path: the reduction that follows is
    # NEAREST, and a LANCZOS pass here would put a soft rim into the one cell
    # that is meant to be the crispest.
    src = src.resize((nw, nh), Image.NEAREST)

    left = front.x + max(0, (front.w - nw) // 2)
    top = front.y + max(0, min(front.h - nh, baseline - nh + 1))
    arr[front.y : front.y + front.h, front.x : front.x + front.w] = 0
    out = Image.fromarray(arr, "RGBA")
    out.paste(src, (left, top), src)
    return out


# --- the sidecar ------------------------------------------------------------


def draft_sidecar(
    *,
    draft_id: str,
    source_job: str,
    created: float,
    geom: SheetGeometry,
    logical_size: int,
    colors: int,
    candidates: list[dict[str, Any]],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    """The draft record: the one file that says a draft finished.

    ``cells`` is in timeline order -- index *i* is Inker frame *i* -- and in
    logical pixels, because the PNGs beside it are the reduced ones. Adoption
    slices on these rectangles and never re-detects the grid from pixels, which
    is what makes a slightly mis-registered generation an editable sheet rather
    than an unopenable one.
    """
    cells = [
        {
            "name": cell.name,
            "frame": cell.frame,
            "yaw": cell.yaw,
            "row": cell.row,
            "col": cell.col,
            "x": cell.col * int(logical_size),
            "y": cell.row * int(logical_size),
            "w": int(logical_size),
            "h": int(logical_size),
        }
        for cell in geom.cells
    ]
    return {
        "version": SPRITE_DRAFT_VERSION,
        "id": draft_id,
        "source_job": source_job,
        "created": created,
        "sheet_type": geom.kind,
        "logical_size": int(logical_size),
        "colors": int(colors),
        "columns": geom.columns,
        "rows": geom.rows,
        "cell_w": int(logical_size),
        "cell_h": int(logical_size),
        "cells": cells,
        "candidates": list(candidates),
        # What actually ran, not what was asked for: a dropped LoRA or a
        # fallback base model is recorded here, so "why do these two look
        # different" is answerable from the file rather than from the log.
        "recipe": dict(recipe),
    }

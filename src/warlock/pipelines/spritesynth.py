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
* **One palette across the atlas**, via ``pixelsheet.resolve_palette``, whether
  the user named one or the median cut picked it -- for the reason that
  function's docstring gives: what makes a palette shared was never the median
  cut, it is that one entry set is applied over the whole atlas in one pass.
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
the reduction is ``pixelize.reduce`` -- and, before it, :func:`reduce_atlas` --
rather than ``pixelsheet.downscale``, whose stride has to be an integer.

**Generation layout and published layout are two different grids**, and
conflating them is what capped this path at four directions and two sheet
types. ``tilesheet``'s measurement is the reason: the pixel-art LoRA draws
about :data:`PX_PER_ART_PIXEL` generation pixels per authored pixel at 1024, so
an honest cell for a 32px sprite is 256px and one for a 64px sprite is 512px --
sixteen and four to a 1024 frame respectively. Eight directions of an eight
frame walk is sixty-four cells at 256px, 4.2 megapixels, four SDXL frames. It
cannot be one generation, so :data:`ATLAS_PX` stops being *the atlas* and
becomes the ceiling on a :class:`Band`.

**One band is one whole direction, all of its frames.** That is
``pixelsheet.bands``' own argument transposed one axis -- "Whole rows, never
part of one ... splitting it across two denoises is exactly the flicker".
Drift between two frames of one direction plays at 10fps and reads as flicker,
so those frames must share a latent. Drift between two *directions* reads as
the character turning, and what holds identity across that seam is the
IP-Adapter, the shared source reference and the shared seed rather than the
denoise. Two directions to a band would be cheaper and is refused for a
different reason: it produces an **uneven** sheet, where two directions agree
perfectly and six drift, and uneven is read as a bug where uniform is read as
style.

The two legacy kinds -- ``turnaround`` and ``walk`` -- keep their literal
tables and their generation-pixel cell rectangles verbatim, because those
rectangles are on disk in every draft ever made here and :func:`split`,
:func:`matte_cells` and :func:`preserve_front` all address the atlas through
them. :func:`plan_sheet` builds everything else, and its ``SheetGeometry``
carries *logical* cell rectangles with the generation rectangles in
:attr:`SheetGeometry.bands`. :attr:`SheetGeometry.bands` being empty is
therefore the honest signal for "this kind is generated as one atlas, the
legacy way".
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

from . import charsheet

log = logging.getLogger(__name__)

#: 2 since 2026-08-29: the reduction changed. Every draft up to version 1 was
#: cut by :func:`reduce_atlas`, a single ``Image.NEAREST`` resize of the whole
#: atlas; from 2 the queue reduces with ``pixelize.reduce``'s alpha-weighted box
#: supersample, which at this path's integral strides (1024 / (4 * 32) = 8)
#: averages all 64 samples under an output pixel instead of keeping one and
#: discarding 63 -- and, being alpha-weighted, stops the transparent background
#: bleeding into the silhouette. The bytes move, so the sidecar has to say which
#: compiler drew the sheet: a bump that changes nothing is waste, and a change
#: that moves bytes without one leaves two different sheets claiming one format.
SPRITE_DRAFT_VERSION = 2

#: ``inner`` and never ``outer`` on this path, which is the opposite of Troupe's
#: default and is forced by the geometry rather than by taste.
#: :func:`structural_warnings` emits ``clipped`` routinely here -- a synthesised
#: cell is 256 or 512px of a 1024px atlas the model filled as it liked, and the
#: subject runs off its cell edge often enough to be a standing warning code.
#: ``pixelize.OUTLINE_MODES`` says ``outer`` "grows the silhouette by a pixel ...
#: and can clip at a cell edge -- so it is opt-in and never the default". Troupe
#: can afford it because a 512px orthographic render leaves margin by
#: construction; nothing here does.
DEFAULT_SPRITE_OUTLINE = "inner"

# One SDXL frame. Two jobs, and they used to be one: it is the side of a legacy
# atlas, and it is the ceiling on either axis of a :class:`Band`. Both are the
# same pin -- one SDXL frame is what this repo generates in -- which is why one
# constant serves them and why a band larger than this is refused rather than
# rescaled.
ATLAS_PX = 1024

#: Generation pixels the pixel-art LoRA spends on one authored pixel at 1024.
#: Not chosen here: ``tilesheet.COLS``' comment measures it ("1024/8 = 128 is
#: the true art resolution of one cell (the pixel-art LoRA draws ~8px 'art
#: pixels' at 1024)"), and this is the same model with the same LoRA drawing
#: the same kind of small figure. It is stated as a constant rather than left in
#: prose because :func:`plan_sheet` divides by it, and a cell below ``8 *
#: logical`` asks the model for detail it cannot resolve -- a 64px sprite drawn
#: in a 256px cell comes back as mush, not as a small sprite.
PX_PER_ART_PIXEL = 8

#: What a sprite is reduced to when nothing says otherwise. 32 is the middle of
#: ``generation.TARGET_CELL_PRESETS`` and the only rung at which every action in
#: :data:`ACTIONS` fits a band (see :func:`plan_sheet`), which makes it the
#: right default for a *planning* function whose caller may not have asked.
DEFAULT_LOGICAL_PX = 32

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "sprite_guides"

# The four directions, in the order a cell index means, and the yaw each one
# stands for. These are the *studio* side's numbers too -- ``inker.animation``
# repeats them because a headless package may not import ``pipelines``, and
# ``tests/test_sprite_geometry_agreement.py`` is the single owner of the
# agreement between the two copies.
DIRECTION_ORDER = ("front", "left", "right", "back")
DIRECTION_YAWS = {"front": 0, "left": 90, "right": 270, "back": 180}

#: Every direction name this repo has, with its yaw in degrees clockwise from
#: the front view. **Imported from ``charsheet``, not copied.** That module
#: already owns the canonical eight names, their yaws and the 1/4/8/16 presets
#: for the mesh path, it is a ``pipelines`` module importing only ``sheet``, so
#: there is nothing circular and nothing heavy about depending on it -- and a
#: third copy of eight names and eight angles is a third thing to drift. The
#: yaws are narrowed to ``int`` because that is what a ``Cell`` and the sidecar
#: have always carried here; every eight-direction yaw is whole.
DIRECTION_YAWS_8: dict[str, int] = {
    name: int(yaw) for name, yaw in charsheet.DIRECTIONS
}

#: How many directions a sprite sheet may carry. ``SpriteSettings.directions``
#: has intended these two since it was written; 1 and 16 are Troupe's
#: (``charsheet.DIRECTION_PRESETS``) and are deliberately not offered here,
#: because a single-direction *sprite* is a still and sixteen bands of one
#: character is sixteen generations for a difference of 22.5 degrees.
DIRECTION_COUNTS: tuple[int, ...] = (4, 8)

#: The order a row index means, per direction count.
#:
#: **Four is the legacy order and deliberately not
#: ``charsheet.DIRECTION_PRESETS[4]``.** The preset sweeps clockwise -- front,
#: left, back, right -- and every sprite draft this path has ever written is
#: front, left, right, back, with that order baked into each one's sidecar and
#: into ``inker.sheetin.walk_tags``. Re-ordering it would silently relabel the
#: back and right rows of every stored draft, which is the one failure a grid
#: table exists to prevent. Eight has no legacy to protect, so it takes the
#: preset's clockwise sweep verbatim -- and the two are the same *set*, which
#: ``tests/test_spritesynth.py`` asserts so the divergence stays a re-ordering
#: rather than a second vocabulary.
SPRITE_DIRECTIONS: dict[int, tuple[str, ...]] = {
    4: DIRECTION_ORDER,
    8: tuple(name for name, _yaw in charsheet.DIRECTION_PRESETS[8]),
}

#: ``(action, frames)``, as a literal table for ``charsheet.ANIMATIONS``' reason
#: -- the frame count of every action readable in one glance.
#:
#: **A second table, not an extension of ``charsheet.ANIMATIONS``**, and the
#: reason is a failure that would land an hour into a job. Troupe's table drives
#: Blender clip expansion through ``clips.expand_clips``, which looks each
#: animation up in ``templates/clips/`` -- and ``humanoid.json`` is the only
#: clip file that ships, so a name it does not carry raises ``KeyError`` in the
#: Blender stage rather than at the door. ``hurt`` and ``cast`` have no clips,
#: so they live here and nowhere else. The five names the two tables *share*
#: must agree on frame counts, and ``tests/test_spritesynth.py`` owns that
#: overlap: a walk that is eight frames here and six frames there is a sheet
#: whose two halves of the program disagree about what a cycle is.
ACTIONS: tuple[tuple[str, int], ...] = (
    ("idle", 4),
    ("walk", 8),
    ("run", 8),
    ("attack", 6),
    ("cast", 6),
    ("hurt", 4),
    ("jump", 6),
)
ACTION_FRAMES: dict[str, int] = dict(ACTIONS)

SHEET_TYPES = ("turnaround", "walk")

#: Every kind :func:`plan_sheet` can name, as ``f"{action}{directions}"``.
#:
#: **``walk4`` is not the legacy ``walk``**, and the near-collision is worth
#: stating because it looks like one. Legacy ``walk`` is a *four* frame cycle
#: over four directions, 4x4; ``walk4`` is this table's eight frame cycle over
#: the same four directions, 8x4. The frame count belongs to the action and not
#: to the direction count, so the two are different sheets that happen to share
#: a prefix, and aliasing one onto the other would silently halve or double a
#: user's cycle. They coexist: ``walk`` stays the name of every draft already on
#: disk, ``walk4`` is what this path plans now.
PLANNED_KINDS: tuple[str, ...] = tuple(
    f"{action}{count}" for action, _frames in ACTIONS for count in DIRECTION_COUNTS
)

_KIND_SPEC: dict[str, tuple[str, int]] = {
    f"{action}{count}": (action, count)
    for action, _frames in ACTIONS
    for count in DIRECTION_COUNTS
}

#: Rows in a band. Two, always, and the argument is aspect rather than area: a
#: direction's frames laid in one row is a 4:1 strip at eight frames, which is
#: further from anything SDXL was trained on than the same cells in two rows,
#: and stacking more than two rows would put the tallest band over the ceiling
#: before the widest one got there. Four frames is 2x2, six is 3x2, eight is
#: 4x2 -- and a one-frame direction has no second row to fill.
BAND_ROWS = 2

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
class Band:
    """One direction's whole run of frames, as one generation.

    The unit the module docstring argues for: never part of a direction, never
    more than one. ``cells`` are this band's own rectangles in *band* pixels --
    a band is handed to the model on its own, so its top-left is (0, 0) and the
    sheet's published grid has nothing to say about where a frame sits inside
    it.

    ``index`` is the direction's row in the published sheet, so a band and the
    row it fills carry the same number and no mapping has to be kept.
    """

    index: int
    direction: str
    columns: int
    rows: int
    cell_px: int
    cells: tuple[Cell, ...]

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` for ``t2i.generate(size=...)``."""
        return (self.columns * self.cell_px, self.rows * self.cell_px)


@dataclass(frozen=True, slots=True)
class SheetGeometry:
    """The grid one sheet type is *published* on, and the bands it is drawn in.

    ``cells`` is the published layout in timeline order, and for a
    :func:`plan_sheet` result ``cell_w``/``cell_h`` are the **logical** cell
    size -- the pixels the user asked for, the pixels the PNG beside the sidecar
    is cut to. The generation rectangles live in :attr:`bands` instead.

    ``bands`` is empty for the two legacy kinds, and that emptiness is load
    bearing rather than a gap: ``turnaround`` and ``walk`` are generated as one
    1024px atlas, their ``cell_w``/``cell_h`` are that atlas's own generation
    pixels, and :func:`split`, :func:`matte_cells`, :func:`baseline_align` and
    :func:`preserve_front` all address the atlas through those rectangles. An
    empty ``bands`` is therefore exactly the statement "this kind is one
    generation, laid out the way every draft on disk already is".
    """

    kind: str
    columns: int
    rows: int
    cell_w: int
    cell_h: int
    cells: tuple[Cell, ...]
    bands: tuple[Band, ...] = ()

    @property
    def directions(self) -> tuple[str, ...]:
        """The direction names, in the order a row index means.

        Derived from ``cells`` rather than stored, so the order a sheet is laid
        out in and the order it says it is laid out in cannot disagree.
        """
        out: list[str] = []
        seen: set[str] = set()
        for cell in self.cells:
            if cell.name not in seen:
                seen.add(cell.name)
                out.append(cell.name)
        return tuple(out)

    @property
    def frames_per_direction(self) -> int:
        """How many frames one direction carries.

        Over the number of *directions*, not the number of rows. Those are the
        same number for every :func:`plan_sheet` layout -- one direction per row
        is what makes ``DirectionalLayout.cell``'s ``row = index // columns``
        generalise -- and they are emphatically not the same number for the
        legacy ``turnaround``, which folds four directions into a 2x2 grid of
        two rows. Dividing by ``rows`` there would answer 2 for a sheet whose
        every cell is frame 0.
        """
        return len(self.cells) // len(self.directions)


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
            f"unknown sprite sheet type {sheet_type!r}; this module lays "
            f"{', '.join(SHEET_TYPES)} out as fixed atlases and everything else "
            "through plan_sheet"
        ) from None


# --- planned sheets: bands in, logical cells out ------------------------------


def band_grid(frames: int) -> tuple[int, int]:
    """``(columns, rows)`` for a band of ``frames``. See :data:`BAND_ROWS`."""
    if frames < 1:
        raise ValueError("a direction needs at least one frame")
    rows = 1 if frames == 1 else BAND_ROWS
    return (-(-frames // rows), rows)


def plan_sheet(
    action: str,
    directions: int = 8,
    frames: int | None = None,
    logical: int = DEFAULT_LOGICAL_PX,
) -> SheetGeometry:
    """The published grid and the bands for one action sheet.

    Published layout is ``columns = frames``, ``rows = directions``, in
    direction-major frame-minor order. That is not a taste: it is what makes
    ``inker.animation.DirectionalLayout.cell``'s ``row = index // columns``
    arithmetic true for every count at once, so the studio's copy of the grid
    generalises without gaining a second formula to keep in step with this one.

    Refuses rather than defaulting, for :func:`geometry`'s reason, and refuses
    one case that looks like a limitation and is arithmetic: a direction whose
    frames do not fit one band. At ``logical`` 64 an honest cell is 512px
    (:data:`PX_PER_ART_PIXEL`) and four of them fill a 1024 band exactly, so a
    six- or eight-frame action at that size would have to either split a
    direction across two denoises -- the flicker the module docstring refuses --
    or draw the sprite at four generation pixels per authored pixel, which
    returns mush. Naming both numbers and stopping is the only honest third
    option.
    """
    if action not in ACTION_FRAMES:
        raise ValueError(
            f"unknown sprite action {action!r}; "
            f"this module draws {', '.join(ACTION_FRAMES)}"
        )
    if directions not in SPRITE_DIRECTIONS:
        raise ValueError(
            f"a sprite sheet carries {' or '.join(str(n) for n in DIRECTION_COUNTS)} "
            f"directions, not {directions}"
        )
    count = ACTION_FRAMES[action] if frames is None else int(frames)
    if count < 1:
        raise ValueError(f"a {action} sheet needs at least one frame")
    size = int(logical)
    if size < 1:
        raise ValueError("the logical cell size must be at least 1")

    cell_px = PX_PER_ART_PIXEL * size
    band_columns, band_rows = band_grid(count)
    if band_columns * cell_px > ATLAS_PX or band_rows * cell_px > ATLAS_PX:
        raise ValueError(
            f"one direction of a {size}px {action} is {count} frames of "
            f"{cell_px}px, which needs a {band_columns * cell_px}x"
            f"{band_rows * cell_px} band, and one SDXL frame is "
            f"{ATLAS_PX}x{ATLAS_PX}; a direction is never split across two "
            "generations, so ask for fewer frames or a smaller sprite"
        )

    names = SPRITE_DIRECTIONS[directions]
    cells = tuple(
        Cell(
            name=name,
            frame=frame,
            yaw=DIRECTION_YAWS_8[name],
            row=row,
            col=frame,
            x=frame * size,
            y=row * size,
            w=size,
            h=size,
        )
        for row, name in enumerate(names)
        for frame in range(count)
    )
    bands = tuple(
        Band(
            index=row,
            direction=name,
            columns=band_columns,
            rows=band_rows,
            cell_px=cell_px,
            cells=tuple(
                Cell(
                    name=name,
                    frame=frame,
                    yaw=DIRECTION_YAWS_8[name],
                    row=frame // band_columns,
                    col=frame % band_columns,
                    x=(frame % band_columns) * cell_px,
                    y=(frame // band_columns) * cell_px,
                    w=cell_px,
                    h=cell_px,
                )
                for frame in range(count)
            ),
        )
        for row, name in enumerate(names)
    )
    return SheetGeometry(
        kind=f"{action}{directions}",
        columns=count,
        rows=directions,
        cell_w=size,
        cell_h=size,
        cells=cells,
        bands=bands,
    )


def plan_kind(kind: str, logical: int = DEFAULT_LOGICAL_PX) -> SheetGeometry:
    """:func:`plan_sheet` addressed by the name a sidecar carries.

    A lookup table rather than a parse of the trailing digits: an action is free
    to be named ``dash2`` one day, and a parser would quietly read that as a
    two-direction ``dash``.
    """
    try:
        action, directions = _KIND_SPEC[kind]
    except KeyError:
        raise ValueError(
            f"unknown sprite sheet kind {kind!r}; this module plans "
            f"{', '.join(PLANNED_KINDS)}"
        ) from None
    return plan_sheet(action, directions, None, logical)


def sheet_geometry(kind: str, logical: int = DEFAULT_LOGICAL_PX) -> SheetGeometry:
    """The grid for any kind this module names, legacy atlas or planned sheet.

    The one door for callers that hold a stored ``sheet_type`` and do not care
    which era it is from. :func:`geometry` and :func:`plan_kind` stay separate
    underneath because they answer different questions -- one hands back a fixed
    atlas whose cells are generation pixels, the other builds a published grid
    whose cells are logical pixels -- and a caller that *does* care must not
    have to guess which it got.
    """
    if kind in GEOMETRY:
        return GEOMETRY[kind]
    return plan_kind(kind, logical)


# --- prompt subjects ----------------------------------------------------------

#: What each action *is*, as words. Under :data:`SPRITE_DRAFT_VERSION` rather
#: than ``PROMPT_VERSION`` for ``tilesheet._VIEW_CLAUSE``'s reason and the same
#: split: ``prompt.SHEET_TEMPLATE`` serves the prompt preview as well as this
#: path and is unchanged by anything here, while these clauses are this module's
#: own and only this path can reach them.
_ACTION_CLAUSE: dict[str, str] = {
    "idle": "standing at rest, weight settled on both feet, a slight breathing sway",
    "walk": "walking at an even pace, one full stride cycle, arms swinging "
    "against the legs",
    "run": "running at speed, long stride, body pitched forward, arms driving",
    "attack": "swinging a melee attack, wind-up through strike to follow-through",
    "cast": "casting a spell, arms raised, hands gathering light at the peak of "
    "the gesture",
    "hurt": "recoiling from a hit, head snapped back, guard broken",
    "jump": "jumping, crouch through launch to apex and landing",
}

#: Where the camera is, per direction, in the vocabulary a sprite sheet means by
#: these names: "left" is the character walking towards the left of the screen,
#: not the camera standing on the character's left. Eight entries because
#: :data:`SPRITE_DIRECTIONS` has eight names and a missing one would be a
#: refusal at generation time, which is the point of the refusal below.
_DIRECTION_CLAUSE: dict[str, str] = {
    "front": "facing the viewer, seen from the front",
    "front_left": "facing the viewer and turned to the left, seen from a front "
    "three-quarter angle",
    "left": "facing to the left, seen in full side profile",
    "back_left": "facing away and turned to the left, seen from a rear "
    "three-quarter angle",
    "back": "facing away from the viewer, seen from behind",
    "back_right": "facing away and turned to the right, seen from a rear "
    "three-quarter angle",
    "right": "facing to the right, seen in full side profile",
    "front_right": "facing the viewer and turned to the right, seen from a front "
    "three-quarter angle",
}


def action_subject(prompt: str, action: str, direction: str) -> str:
    """The subject clause for one band. A *subject*, not a finished prompt.

    The caller runs this through ``guidance.compose_prompt`` and
    ``prompt.SHEET_TEMPLATE``, which is what adds the sheet, the pixel-art and
    the no-text clauses -- and which is untouched by this function, so nothing
    here bumps ``PROMPT_VERSION``.

    An unknown action **or** an unknown direction raises rather than falling
    back to a neutral clause, which is ``tilesheet.sheet_subject``'s rule and
    its reason verbatim: a fallback here is invisible by construction, because
    it produces a *plausible* sheet described by the wrong sentence. The moment
    an action can join :data:`ACTIONS` without a clause beside it, that is eight
    bands of the wrong picture and nobody is told.
    """
    text = str(prompt).strip()
    clause = _ACTION_CLAUSE.get(action)
    if clause is None:
        raise ValueError(
            f"unknown sprite action {action!r}; "
            f"this module draws {', '.join(_ACTION_CLAUSE)}"
        )
    view = _DIRECTION_CLAUSE.get(direction)
    if view is None:
        raise ValueError(
            f"unknown sprite direction {direction!r}; "
            f"this module draws {', '.join(_DIRECTION_CLAUSE)}"
        )
    return ", ".join(part for part in (text, clause, view) if part)


# --- the T-pose reference guide ---------------------------------------------
#
# Troupe's reference stage wants an orthographic T-pose character reference
# rather than a dynamic illustration, because both the reconstruction and the
# bbox-proportional ``rigging.fit_template`` are far more reliable against one:
# limb separation and silhouette are what a single-view reconstruction has to
# get right, and a folded arm is the failure it cannot recover from.
#
# It reuses this module's guide machinery -- ``_parse_template`` for the
# validation and ``render_guide`` for the drawing -- against a one-cell grid,
# and that grid is deliberately **not** in ``GEOMETRY``:
# ``tests/test_sprite_geometry_agreement.py`` owns the claim that ``GEOMETRY``,
# ``SHEET_TYPES`` and ``inker.animation.SHEET_KINDS`` name the same set of
# sprite sheet kinds, and a T-pose reference is not a sprite sheet. Registering
# it there to save a name would make that test say something weaker.

_TPOSE_TABLE = (("front", 0, 0, 0),)

#: One full 1024px SDXL frame. ``kind="tpose"``, which is what the two template
#: files declare and what ``_parse_template`` checks them against.
TPOSE_GEOMETRY = _build("tpose", _TPOSE_TABLE)

#: Male and female get their own guides, per the program spec: the proportions
#: differ enough (shoulder width, arm length, stance) that one compromise
#: figure conditions both badly.
TPOSE_VARIANTS: tuple[str, ...] = ("male", "female")


#: The reference poses on offer, and the ``kind`` each one's templates declare.
#: A *second* axis, crossed with :data:`TPOSE_VARIANTS` rather than folded into
#: it: sex and pose are independent choices, and four variant names spelling
#: out a two-by-two grid is a table pretending to be a list.
#:
#: The T-pose is first because it is what a single-view reconstruction most
#: wants -- limb separation, unambiguous silhouette. The A-pose is here because
#: the shipped humanoid rig template *is* an A-pose, so a mesh built from it is
#: fitted directly rather than needing joints measured off its own vertices.
#: Neither dominates; both ship.
REFERENCE_POSES: tuple[str, ...] = ("tpose", "apose")

#: One geometry per pose. Same single 1024px cell either way -- they differ only
#: in the ``kind`` each checks its template files against, which is what stops
#: an A-pose file being loaded as a T-pose and drawn without complaint.
REFERENCE_GEOMETRY = {pose: _build(pose, _TPOSE_TABLE) for pose in REFERENCE_POSES}


def load_reference_guide(variant: str, pose: str = "tpose") -> GuideTemplate:
    """Read and validate ``templates/sprite_guides/<pose>_<variant>.json``.

    Raises on an unknown variant *or* pose rather than defaulting, for the
    reason :func:`geometry` gives: silently conditioning on the other sex's
    guide -- or on a pose the caller did not ask for -- is a wrong character
    published under their name.
    """
    if variant not in TPOSE_VARIANTS:
        raise ValueError(
            f"unknown reference variant {variant!r}; "
            f"this module has {', '.join(TPOSE_VARIANTS)}"
        )
    if pose not in REFERENCE_POSES:
        raise ValueError(
            f"unknown reference pose {pose!r}; "
            f"this module has {', '.join(REFERENCE_POSES)}"
        )
    raw = json.loads((TEMPLATE_DIR / f"{pose}_{variant}.json").read_text(encoding="utf-8"))
    return _parse_template(raw, REFERENCE_GEOMETRY[pose])


def render_reference_guide(variant: str, pose: str = "tpose") -> PILImage:
    """The 1024px white-on-black reference stick figure for ``variant``/``pose``.

    Handed to the ControlNet as the hint *directly*, exactly as
    :func:`render_guide`'s docstring argues: it is already line art in canny
    space, and running the detector over it would return two lines where the
    guide means one.
    """
    return render_guide(REFERENCE_GEOMETRY[pose], load_reference_guide(variant, pose))


def load_tpose_guide(variant: str) -> GuideTemplate:
    """:func:`load_reference_guide` at the T-pose. Kept for its callers."""
    return load_reference_guide(variant, "tpose")


def render_tpose_guide(variant: str) -> PILImage:
    """:func:`render_reference_guide` at the T-pose. Kept for its callers."""
    return render_reference_guide(variant, "tpose")


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


#: Joints that sit on the character's midline and therefore mirror onto
#: themselves. Everything else in a guide must be one half of a ``.L``/``.R``
#: pair, because a mirror needs an answer for every point and "leave it where it
#: is" is the wrong answer for a hand.
CENTRAL_JOINTS: frozenset[str] = frozenset({"head", "neck", "hip"})

#: Which direction each derivable one is the mirror of. Derived from the
#: direction names by swapping the word rather than tabulated, so a ninth
#: direction cannot be added on one side of the sheet only.
_MIRROR_OF: dict[str, str] = {
    name: name.replace("right", "left")
    for name in SPRITE_DIRECTIONS[8]
    if "right" in name
}

#: The half of the sheet a mirrored template has to author: everything that is
#: not derived. Derived rather than listed, so it cannot disagree with
#: :data:`_MIRROR_OF`.
AUTHORED_DIRECTIONS: tuple[str, ...] = tuple(
    name for name in SPRITE_DIRECTIONS[8] if name not in _MIRROR_OF
)


def _mirror_joint(name: str) -> str:
    """A joint's name on the other side of the body."""
    if name.endswith(".L"):
        return f"{name[:-2]}.R"
    if name.endswith(".R"):
        return f"{name[:-2]}.L"
    return name


def _mirrored_points(
    points: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    """One pose reflected across the cell's vertical centre line.

    ``x -> 1 - x`` with ``.L`` and ``.R`` swapped, y untouched. **This flips
    handedness**, and that is accepted rather than corrected: a right-handed
    sword becomes a left-handed one in the derived half of the sheet. At the
    cell sizes this path draws -- 256px of generation for a 32px sprite -- the
    stick figure decides where the limbs are and the IP-Adapter decides who the
    character is, so what a viewer reads off the mirrored rows is the pose, not
    which hand the prop is in. The escape hatch for an action where it *does*
    matter is to author that direction: an authored pose always beats a derived
    one.
    """
    return {
        _mirror_joint(str(name)): (1.0 - float(xy[0]), float(xy[1]))
        for name, xy in points.items()
    }


def _check_mirrorable(points: dict[str, Any], where: str) -> None:
    for name in points:
        if name in CENTRAL_JOINTS:
            continue
        if str(name).endswith((".L", ".R")):
            partner = _mirror_joint(str(name))
            if partner not in points:
                raise ValueError(
                    f"guide pose {where} has {name!r} but no {partner!r}, so it "
                    "cannot be mirrored"
                )
            continue
        raise ValueError(
            f"guide pose {where} has {name!r}, which is neither a '.L'/'.R' pair "
            f"nor one of the central joints ({', '.join(sorted(CENTRAL_JOINTS))}), "
            "so a mirror has no answer for it"
        )


def _expand_mirrored(raw: dict[str, Any], geom: SheetGeometry) -> list[Any]:
    """``raw['poses']`` with every mirror-derivable direction filled in.

    Runs **before** the rest of :func:`_parse_template`, so the derived poses go
    through every check the authored ones do -- unknown joint, out-of-cell,
    missing head, count. A typo mirrored is still a typo, and a guide missing a
    leg is worth catching before a twenty-second generation rather than after.

    The mirror exists because the alternative is not "more work", it is *worse
    poses*: seven actions by eight directions by up to eight frames is about 450
    poses of hand JSON, and a left-facing and a right-facing walk authored
    separately disagree by hundredths and read as two different characters.
    Five directions are authored -- ``front``, ``front_left``, ``left``,
    ``back_left``, ``back`` -- and the three right-hand ones fall out exactly.

    A template with no ``mirror`` key is returned untouched, which is what keeps
    ``turnaround.json`` and ``walk.json`` byte-for-byte what they were.

    One check does change shape for a mirrored template. The positional order
    check below compares the expanded list against the grid cell for cell, and
    this function builds that list *by* the grid, so it can no longer fail --
    so the order refusal is made here instead, against the authored poses only,
    and joined by two the positional check never had: an authored pose naming a
    cell the grid does not have, and two poses claiming one cell.
    """
    entries = list(raw["poses"])
    spec = raw.get("mirror")
    if spec is None:
        return entries

    axis = str(spec.get("axis", "x"))
    if axis != "x":
        raise ValueError(
            f"guide template mirrors on {axis!r}; a standing figure has one "
            "mirror and it is 'x'"
        )
    pairs = str(spec.get("pairs", "suffix"))
    if pairs != "suffix":
        raise ValueError(
            f"guide template pairs joints by {pairs!r}; the '.L'/'.R' suffix "
            "scheme is the one that ships"
        )

    authored: dict[tuple[str, int], Any] = {}
    for entry in entries:
        key = (str(entry["name"]), int(entry["frame"]))
        if key in authored:
            raise ValueError(
                f"guide template for {geom.kind!r} has two poses for "
                f"{key[0]!r}/{key[1]}"
            )
        _check_mirrorable(entry["points"], f"{key[0]!r}/{key[1]}")
        authored[key] = entry

    grid = [(cell.name, cell.frame) for cell in geom.cells]
    unknown = [key for key in authored if key not in set(grid)]
    if unknown:
        name, frame = unknown[0]
        raise ValueError(
            f"guide pose {name!r}/{frame} is not a cell of the {geom.kind!r} grid"
        )
    if [key for key in grid if key in authored] != list(authored):
        raise ValueError(
            f"the authored poses of the {geom.kind!r} template are in the wrong "
            "order"
        )

    out: list[Any] = []
    for name, frame in grid:
        entry = authored.get((name, frame))
        if entry is not None:
            out.append(entry)
            continue
        source = _MIRROR_OF.get(name)
        mirror = authored.get((source, frame)) if source is not None else None
        if mirror is None:
            # One message and not two, because the two cases cannot be told
            # apart from the outside: this walks the grid in order and every
            # mirror source precedes the direction derived from it, so a missing
            # source is always reported as the source's own cell. A separate
            # "nothing to mirror" refusal would be a branch nothing can reach.
            raise ValueError(
                f"guide template for {geom.kind!r} has no pose for "
                f"{name!r}/{frame}; a mirrored template authors "
                f"{', '.join(AUTHORED_DIRECTIONS)} and derives the rest"
            )
        out.append(
            {
                "name": name,
                "frame": frame,
                "points": _mirrored_points(mirror["points"]),
            }
        )
    return out


def _parse_template(raw: dict[str, Any], geom: SheetGeometry) -> GuideTemplate:
    kind = str(raw["kind"])
    if kind != geom.kind:
        raise ValueError(f"guide template says {kind!r} but was loaded for {geom.kind!r}")
    head_point = str(raw.get("head_point") or "head")
    segments = tuple((str(a), str(b)) for a, b in raw["segments"])

    poses: list[GuidePose] = []
    for entry in _expand_mirrored(raw, geom):
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

    Serves the planned kinds through :func:`sheet_geometry` as well as the two
    legacy ones. The geometry is only ever asked which cells exist and in what
    order -- guide points are in normalised cell space -- so the logical size a
    planned grid is built at cannot reach a template.
    """
    geom = sheet_geometry(sheet_type)
    path = TEMPLATE_DIR / f"{sheet_type}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _parse_template(raw, geom)


def _draw_pose(draw: Any, cell: Cell, pose: GuidePose, template: GuideTemplate) -> None:
    """One stick figure, in one cell's rectangle of whatever canvas ``draw`` is."""
    width = max(2, cell.w // 128)

    def at(name: str) -> tuple[float, float]:
        px, py = pose.points[name]
        return (cell.x + px * cell.w, cell.y + py * cell.h)

    for a, b in template.segments:
        draw.line([at(a), at(b)], fill=(255, 255, 255), width=width)
    hx, hy = at(template.head_point)
    r = template.head_radius * cell.w
    draw.ellipse([hx - r, hy - r, hx + r, hy + r], outline=(255, 255, 255), width=width)


def render_guide(geom: SheetGeometry, template: GuideTemplate) -> PILImage:
    """The atlas-sized pose guide: white stick figures on black.

    Handed to the ControlNet as the hint *directly*. It is already line art in
    canny space -- white strokes on black -- and running the detector over it
    would return the outline of each stroke, i.e. two lines where the guide
    means one, which is exactly the "why does my character have four legs"
    failure. ``control.edge_fraction`` is still recorded against it, so a guide
    that drew nothing is an answerable question.

    For the legacy kinds only, because it draws on an ``ATLAS_PX`` canvas at the
    geometry's own cell rectangles -- which for a planned sheet are *logical*
    pixels, a 32px figure in the corner of a 1024px black frame. Planned sheets
    are drawn one band at a time by :func:`render_band_guide`, which is the same
    argument as the generation itself.
    """
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (ATLAS_PX, ATLAS_PX), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for cell, pose in zip(geom.cells, template.poses, strict=True):
        _draw_pose(draw, cell, pose, template)
    return canvas


def render_band_guide(band: Band, template: GuideTemplate) -> PILImage:
    """:func:`render_guide` for one band: this direction's frames, band-sized.

    Keyed on ``(direction, frame)`` rather than sliced positionally out of
    ``template.poses``. The template is validated against the *published* grid,
    which is direction-major, so a band's poses are in fact a contiguous run --
    but computing that run here would put the published layout's arithmetic in a
    second place, and the lookup costs nothing and cannot drift.
    """
    from PIL import Image, ImageDraw

    poses = {(pose.name, pose.frame): pose for pose in template.poses}
    canvas = Image.new("RGB", band.size, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for cell in band.cells:
        pose = poses.get((cell.name, cell.frame))
        if pose is None:
            raise ValueError(
                f"the {template.kind!r} guide has no pose for "
                f"{cell.name!r}/{cell.frame}"
            )
        _draw_pose(draw, cell, pose, template)
    return canvas


# --- slicing, matting, alignment --------------------------------------------


def split(atlas: PILImage, geom: SheetGeometry | Band) -> list[PILImage]:
    """The atlas -- or one band -- cut on its predetermined rectangles.

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

    **Nothing in the program calls this.** Since ``SPRITE_DRAFT_VERSION`` 2 the
    synthesis path reduces through ``pixelize.reduce``'s alpha-weighted box, for
    the reason that constant states, and no other caller was left.

    Kept rather than deleted, and the reason is prose rather than code: this is
    the named home of the whole-atlas boundary argument below, which
    ``pixelize``'s module docstring and :func:`pixelize.reduce_frames` both cite
    by this name when they explain why reducing per frame gives nothing up.
    :func:`pixelize.reduce` cites it for the other half -- its fallback for a
    target that does not divide (48 out of 1024) *is* this function's single
    NEAREST resize, which is why those rungs came through the version bump byte
    for byte. Delete it and two live explanations lose the thing they point at;
    what must not happen is this drifting into something the citations no longer
    describe, so it takes no new behaviour and gains no callers.

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
    the same reduction and share the same palette as the other cells, or the one
    cell that is definitely the right character is also the one cell that does
    not match the sheet.
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
    # NEAREST, and deliberately still NEAREST now that the reduction which
    # follows is an alpha-weighted box: this is a resize at an arbitrary ratio
    # (the scale is a median height over the source's own), so there is no
    # supersample to be had, and a LANCZOS pass here would put a soft rim into
    # the one cell that is meant to be the crispest -- which the box mean would
    # then average outward rather than remove.
    src = src.resize((nw, nh), Image.NEAREST)
    if nw > front.w:
        # front_fits admits sources up to a third wider than the generated
        # front's proportions, so a height-matched paste can come out wider
        # than the cell. Trim the excess evenly from both sides rather than
        # letting the paste spill: the cell to the right is the "left" view,
        # and an opaque stripe of the reference along its edge survives the
        # reduction into every export.
        lost = nw - front.w
        src = src.crop((lost // 2, 0, lost // 2 + front.w, nh))
        nw = front.w

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
    palette: str = "",
    palette_source: str = "derived",
    palette_hash: str = "",
    dither: bool = False,
    outline: str = DEFAULT_SPRITE_OUTLINE,
) -> dict[str, Any]:
    """The draft record: the one file that says a draft finished.

    ``cells`` is in timeline order -- index *i* is Inker frame *i* -- and in
    logical pixels, because the PNGs beside it are the reduced ones. Adoption
    slices on these rectangles and never re-detects the grid from pixels, which
    is what makes a slightly mis-registered generation an editable sheet rather
    than an unopenable one.

    Each ``candidates`` record carries a ``grid`` block -- ``pixel.lattice``'s
    two numbers for that candidate's own generation, measured once on the whole
    frame the model returned. Per candidate rather than per draft because each
    is a separate generation; additive, and it does **not** bump
    ``SPRITE_DRAFT_VERSION``, since a new optional key readers may ignore is not
    a new format and a bump that changes nothing would invalidate every stored
    benchmark comparison for free. Recorded and acted on by nothing; see
    ``pixel.lattice``.

    The five palette keys are draft-level rather than per candidate, because
    every one of them is a *request* the two candidates share -- and the colours
    each candidate actually ended up with are already in its own ``palette``
    list. ``palette_source`` says which branch of ``pixelsheet.resolve_palette``
    ran; ``palette``/``palette_hash`` name the authored file and fingerprint its
    colours, and are empty on the derived branch because there is no file to
    name. **The hash is not decoration**: a palette edited in place keeps its
    name, which is exactly why ``derive._pixel_current`` compares digests rather
    than names to decide whether an artifact on disk was cut the way the caller
    now wants.
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
        "palette": str(palette),
        "palette_source": str(palette_source),
        "palette_hash": str(palette_hash),
        "dither": bool(dither),
        "outline": str(outline),
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

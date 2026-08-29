"""Two ways to build a tileset out of *seamless materials*, and the words for both.

``docs/measurements/2026-08-18-tile-sheet-grid.md`` took one 1024px generation,
imposed an 8x8 grid on it with a canny guide, and asked SDXL for sixty-four
different tiles. The guide was obeyed and the tiles were not different: every
cell of the guide is identical, so there is no per-cell signal for variety, and
the model resolves that ambiguity either by painting one continuous scene
through the grid or by painting one tile sixty-four times. That document names
three candidates and ranks *N materials, one grid* first, for the reason the
retired ground path had already demonstrated: **variety is a property of the
request, not of the model's composition.** This module is that candidate, plus
the terrain case it makes reachable.

Two modes, one vocabulary:

``materials``
    N material descriptions, each generated on its own as a *seamless* 1024px
    tile through ``text2image``'s circular-padding path, each reduced to the
    tile size and laid out in a plain grid. N prompts, N generations, N
    genuinely different tiles.

``terrain``
    Two seamless materials -- an inner and an outer -- composited into a
    blob-47 autotile set by :mod:`.tilemask`. The boundary between them is a
    scalar field rather than a drawing, so the model is never asked to draw an
    edge and never gets the chance to draw it wrong.

**Both modes generate the same thing and differ only in what is done with it.**
That is what makes them one feature: a seamless material is a seamless
material, and ``terrain`` is ``materials`` with two cells and a compositor.

**A seamless material has exactly one view, and both modes accept only it.**
``prompt.TILE_TEMPLATE`` -- which is what ``text2image.generate(tile=True)``
selects -- hardcodes "flat top-down orthographic view", so there is no subject
clause here that could ask for another one without contradicting the template
that will wrap it. The other two views of ``tilesheet.VIEWS`` are refused by
name, each for a reason about *tiling* rather than about taste:

* **isometric** -- an isometric tile is a 2:1 diamond,
  ``text2image.circular_padding`` wraps a *rectangle*, and a diamond is not one;
* **3/4** -- ``tilesheet._VIEW_CLAUSE``'s 3/4 is "camera tilted about thirty
  degrees, square tiles with a shallow visible front face", and a visible front
  face does not tile vertically: the row below would occlude it. A seamless 3/4
  material is not redundant, it is incoherent.

Nothing is lost by either refusal, because the grid path keeps both --
``docs/measurements/2026-08-21-three-quarter-guide.md`` measured 3/4 for the
tile-sheet *guide*, which still ships. So ``view`` survives here as one value
with one job: the record. ``plotter_tilesets._VIEW_LATTICE`` reads it to know
which lattice a set was drawn for, and no clause anywhere below is a function of
it.

Pure, in this package's sense: stdlib and its two pure siblings at module scope,
numpy inside the functions that need it, no torch, no ``service``, no ``queue``,
no ``studio``. This module owns the arithmetic and the words; the queue owns the
model calls and every path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import tilemask, tilesheet

#: Bumped when anything below changes what a given form produces: the two
#: layouts, the subject clauses, the seed derivation, the reduction or the
#: sidecar's shape. Recorded in the sidecar so a set can say which compiler
#: built it, the way :data:`tilesheet.TILE_SHEET_VERSION` does.
#:
#: **It is deliberately not ``prompt.PROMPT_VERSION``.** ``TILE_TEMPLATE`` is
#: already exactly right for one seamless material and nothing here touches it;
#: bumping that counter would re-key the whole findings corpus through
#: ``provenance.versions()`` for a clause this module owns. That split is the
#: rule ``tilesheet.TILE_SHEET_VERSION`` and ``tilesheet.sheet_subject`` both
#: state, applied a second time.
TILE_ATLAS_VERSION = 1

MODE_MATERIALS = "materials"
MODE_TERRAIN = "terrain"

#: The two shapes a request can take. Written out rather than derived, for
#: ``tileset.OBJECT_ALIGNMENTS``' reason: one list, so a third mode cannot be
#: accepted by the geometry and unknown to the sidecar.
MODES: tuple[str, ...] = (MODE_MATERIALS, MODE_TERRAIN)

#: One SDXL frame, and the size the seamless generation is already done at.
#: Not a control: the circular-padded UNet wraps at whatever size it is given,
#: but SDXL 1.0 composes at 1024 and a 512px material comes back as half a
#: picture of something rather than as a surface. It is also the numerator of
#: every reduction below, and the reason the exact-divisor rule in
#: :func:`reduce_material` is livable -- 1024 is a power of two, so every tile
#: size that divides it divides it cleanly all the way down.
MATERIAL_PX = 1024

#: How many distinct *prompt lines* one request may name.
#: ``asset_workflows.collection_cells``' own 1-16, restated here so this module
#: can state the pair of limits it is one half of -- but **enforced at the
#: door**, not here, because that is where the lines still exist as lines. By
#: the time a geometry is asked for they have been expanded into cells and the
#: line count is no longer recoverable from the number.
MAX_MATERIALS = 16

#: The most cells one atlas may hold, and the limit this module actually
#: enforces. A cell is one ``(prompt line, variant)`` pair --
#: ``asset_workflows.collection_cells`` compiles at most sixteen lines by at
#: most four variants -- so sixty-four is exactly ``16 * 4`` rather than a round
#: number that happens to be nearby, and it is also where the eight-column
#: layout's 8x8 ceiling comes from.
#:
#: A ceiling on *generations*, not on pixels: each cell is its own full SDXL
#: sample. Raise it when somebody is willing to wait for the sixty-fifth.
MAX_CELLS = 64

#: How wide a materials sheet is laid out. **Deliberately not "roughly square"**
#: -- ``tileset.compose_collection`` packs a collection squarely because nothing
#: reads meaning off its order, and the opposite is true here. A materials sheet
#: *is the list the user typed*, in the order they typed it, and eight across
#: keeps a row of a full request readable while staying inside the width a
#: palette pane shows without scrolling.
MATERIAL_COLUMNS = 8

#: ``service.validation.MAX_SEED``, restated rather than imported -- a pipeline
#: may not import the service layer. ``tilesheet.MAX_SEED`` is the same value
#: restated for the same reason, and this is deliberately a third *copy* rather
#: than an alias of it: an alias could not drift, but it also could not be
#: pinned, and ``tests/test_tileatlas.py`` imports all three and asserts them
#: equal. Three copies with a test over them is the repo's pattern (``tilemask``
#: against ``studio.tilegrid.blob``, ``service.tilesheets.TILE_SIZES`` against
#: ``tilesheet.TILE_SIZES``); a chain of aliases is one edit away from being a
#: chain of aliases to the wrong number.
MAX_SEED = 2**31 - 1

#: Appended to every material subject in one request, so N separate generations
#: read as one sheet. :data:`tilesheet.DETAIL_CLAUSE`'s sibling and its argument
#: restated at a different size: a finished cell keeps at most ``tile_w`` px of
#: true detail, a material whose elements are smaller than that reduces to
#: noise, and asking for oversized high-contrast elements is what survives the
#: trip down from 1024.
#:
#: Shorter than ``DETAIL_CLAUSE`` and not a copy of it, because the two do
#: different jobs. That one has to fight a grid of sixty-four cells for
#: consistent scale; this one is appended to a texture template that already
#: says "uniform scale, repeating pattern", so the only thing left to ask for is
#: element size, contrast and flat light.
MATERIAL_STYLE_CLAUSE = (
    "chunky oversized elements, strong colour contrast, flat even lighting"
)

#: The one view a seamless material can be generated for. A tuple of one rather
#: than a bare constant, so the refusal in :func:`_resolve_view` reads the same
#: list every other ``VIEWS`` in this package does and a second entry is an edit
#: in one place -- but it is one, and the two absentees are refused by name
#: rather than merely missing. See the module docstring for why each.
VIEWS: tuple[str, ...] = (tilesheet.TOP_DOWN,)

#: The layout each mode publishes, as the sidecar spells it. ``"blob47"`` is a
#: promise about *positions*: forty-seven columns in ascending
#: ``tilemask.BLOB_MASKS`` order, which is what ``Tileset.local_for`` indexes and
#: what ``Tileset.__post_init__`` enforces.
_LAYOUTS: dict[str, str] = {MODE_MATERIALS: "grid", MODE_TERRAIN: "blob47"}


def _resolve_view(view: Any, mode: str) -> str:
    """A stored view value as this module spells it, or a refusal.

    One door for both geometries, so each refusal is one sentence in one place
    rather than two that drift apart. ``tilesheet.normalize_view`` first, so a
    row written before the vocabulary widened still reads as top-down.

    **Both of the other two views are named rather than merely absent.** They
    are the two a caller has an actual reason to ask for -- the tile-sheet form
    next door offers both -- so the refusal has to say what to do instead, and
    each says the *tiling* fact that makes it impossible rather than "not
    supported".
    """
    text = tilesheet.normalize_view(view)
    if text == tilesheet.ISOMETRIC:
        raise ValueError(
            "an isometric tile is a 2:1 diamond and cannot be a seamless "
            "square; use the grid layout"
        )
    if text == tilesheet.THREE_QUARTER:
        raise ValueError(
            "a 3/4 tile has a visible front face and cannot tile vertically; "
            "use the grid layout"
        )
    if text not in VIEWS:
        raise ValueError(
            f"unknown view {text!r} for a {mode} atlas; this module draws "
            f"{', '.join(VIEWS)}"
        )
    return text


@dataclass(frozen=True, slots=True)
class MaterialCell:
    """One cell of an atlas: which material, in which words, and where it lands.

    ``index`` is the cell's position in reading order and the identity every
    other field hangs off -- it is what a future "reroll material 3 only" names,
    and why :func:`material_seeds` derives a seed per cell rather than drawing
    one.

    ``variant`` is which draw of that prompt line this cell is.
    ``asset_workflows.collection_cells`` is what produces the pair: it expands N
    prompt lines by V variants into one cell each, numbering variants from 1, and
    a *cell* is therefore one ``(line, variant)`` pair rather than one material.
    The field exists because nothing else in the record could tell two cells of
    one line apart -- their prompts are identical by construction, and only the
    seed differs.

    **:func:`material_geometry` fills the geometry and leaves the words empty.**
    The layout is arithmetic and knows nothing about the request; the queue binds
    ``prompt``, ``variant`` and ``seed`` with :func:`dataclasses.replace` once
    the subjects are compiled. :func:`atlas_sidecar` refuses an unbound cell, so
    the two halves cannot be published half-joined.
    """

    index: int
    prompt: str
    variant: int
    seed: int
    row: int
    col: int


@dataclass(frozen=True, slots=True)
class AtlasGeometry:
    """The grid one atlas is assembled on."""

    mode: str
    view: str
    columns: int
    rows: int
    tile_w: int
    tile_h: int
    cells: tuple[MaterialCell, ...]

    @property
    def layout(self) -> str:
        """``"grid"`` or ``"blob47"`` -- what the positions mean."""
        return _LAYOUTS[self.mode]

    @property
    def atlas_size(self) -> tuple[int, int]:
        """What is published, in pixels. ``(w, h)``."""
        return (self.columns * self.tile_w, self.rows * self.tile_h)

    @property
    def source_size(self) -> tuple[int, int]:
        """What each *generation* is asked for, in pixels. ``(w, h)``.

        One material, not the atlas: the whole point of this path is that the
        cells are separate samples, so there is no single frame the sheet is cut
        out of and this is the size of each of ``tiles`` of them.
        """
        return (MATERIAL_PX, MATERIAL_PX)

    @property
    def tiles(self) -> int:
        return len(self.cells)


def material_geometry(tile_w: int, view: str, count: int) -> AtlasGeometry:
    """The grid for N material cells. Unknown or impossible asks raise.

    ``count`` is **cells, not prompt lines**: a cell is one
    ``(prompt line, variant)`` pair, already expanded by
    ``asset_workflows.collection_cells``. The two limits live at the two places
    that can see the thing they are about -- the door caps the lines at
    :data:`MAX_MATERIALS` and the variants at four, because that is where lines
    are still lines; this caps their product at :data:`MAX_CELLS`, because by
    here the count is all there is.

    ``count`` cells at :data:`MATERIAL_COLUMNS` across, ``ceil`` rows down, in
    reading order. **Not roughly square**, which is what a packer would do:
    a materials sheet is the list the user typed, expanded in place, and its
    order is the thing they chose -- so the layout preserves it rather than
    optimising the texture's aspect ratio. One material is one cell in one row,
    not a 1x1 square dressed up as a sheet.

    Cells come back with their geometry bound and their words empty -- see
    :class:`MaterialCell`.
    """
    width = int(tile_w)
    total = int(count)
    text = _resolve_view(view, MODE_MATERIALS)
    if total < 1:
        raise ValueError(f"a materials atlas needs at least one cell; got {total}")
    if total > MAX_CELLS:
        raise ValueError(
            f"{total} cells is past the {MAX_CELLS} ceiling ({MAX_MATERIALS} "
            f"prompt lines by four variants); each one is its own full generation"
        )
    _check_tile(width)
    columns = min(MATERIAL_COLUMNS, total)
    rows = math.ceil(total / columns)
    cells = tuple(
        MaterialCell(
            index=index,
            prompt="",
            variant=0,
            seed=0,
            row=index // columns,
            col=index % columns,
        )
        for index in range(total)
    )
    return AtlasGeometry(
        mode=MODE_MATERIALS,
        view=text,
        columns=columns,
        rows=rows,
        tile_w=width,
        tile_h=tilesheet.tile_height(width, text),
        cells=cells,
    )


def terrain_geometry(tile_w: int, view: str) -> AtlasGeometry:
    """The grid for one blob-47 terrain: forty-seven columns by one row.

    That is not a choice this module makes. ``Tileset.__post_init__`` refuses a
    terrain set that is not ``blob.TILE_COUNT`` columns wide, ``local_for``
    indexes a case by its column, and :func:`tilemask.blob_atlas` already emits
    its columns in ascending ``BLOB_MASKS`` order -- so the row this describes
    *is* a terrain set's atlas and nothing downstream reorders anything.

    One row, so one terrain and one phase. ``Tileset.phases`` is the door to
    stacked phase variants and ``tilemask.wrap_noise``'s ``period_tiles`` is the
    field half of it; neither ships, and a second row here would be the third
    piece of a feature whose first two are parked.

    The 16px floor on a terrain tile is :data:`tilemask.MIN_TILE`'s and is raised
    there, where the inset it is about lives.
    """
    width = int(tile_w)
    text = _resolve_view(view, MODE_TERRAIN)
    _check_tile(width)
    cells = tuple(
        MaterialCell(index=index, prompt="", variant=0, seed=0, row=0, col=index)
        for index in range(tilemask.TILE_COUNT)
    )
    return AtlasGeometry(
        mode=MODE_TERRAIN,
        view=text,
        columns=tilemask.TILE_COUNT,
        rows=1,
        tile_w=width,
        tile_h=tilesheet.tile_height(width, text),
        cells=cells,
    )


def _check_tile(width: int) -> None:
    """The one size rule both geometries share, raised with its numbers.

    A tile larger than the frame it is reduced from is an upscale nobody asked
    for, and a tile that does not divide the frame is the seam defect
    :func:`reduce_material` exists to refuse -- checked here as well so a form
    is told at the geometry rather than after N generations have run.
    """
    if width < 1:
        raise ValueError("a tile is at least one pixel across")
    if width > MATERIAL_PX:
        raise ValueError(
            f"a {width}px tile is larger than the {MATERIAL_PX}px material it is "
            f"reduced from; there is nothing to reduce"
        )
    if MATERIAL_PX % width:
        raise ValueError(
            f"a {width}px tile does not divide the {MATERIAL_PX}px material "
            f"({MATERIAL_PX}/{width} = {MATERIAL_PX / width:.4g}); a seamless "
            f"material must reduce on an exact partition or its wrap seam moves"
        )


# -- the words ----------------------------------------------------------------


def material_subject(prompt: str, *, index: int, total: int) -> str:
    """The subject clause for one material. A *subject*, not a finished prompt.

    The caller runs this through ``guidance.compose_prompt`` and
    ``prompt.TILE_TEMPLATE``, which is what adds "seamless tileable texture,
    flat top-down orthographic view, even diffuse lighting, no shadows, uniform
    scale, repeating pattern, no single focal object". That template is already
    exactly right for one seamless material, which is why nothing here repeats
    any of it and why this path bumps no ``PROMPT_VERSION``:
    :data:`MATERIAL_STYLE_CLAUSE` is the only thing this module adds, and it sits
    under :data:`TILE_ATLAS_VERSION` where only this path can reach it.

    **There is no view clause and there is no ``view`` parameter.** A seamless
    material is flat top-down by construction -- the template says so, and
    :func:`_resolve_view` refuses every other view before a subject is ever
    compiled -- so a framing clause here would have nothing to vary and one
    thing to contradict. ``view`` is recorded and never spoken.

    ``index`` and ``total`` are for the *refusal* and deliberately never enter
    the text. "material 3 of 8" in a subject is a phrase SDXL draws, which is
    what every template's "no text, no watermark" is there to prevent -- but
    "this material has no words" is a useless thing to say about a request that
    named twelve of them.
    """
    text = str(prompt).strip()
    position, count = int(index), int(total)
    if count < 1:
        raise ValueError(f"a request names at least one material; got {count}")
    if not 0 <= position < count:
        raise ValueError(
            f"material {position} is outside a request of {count} (0..{count - 1})"
        )
    if not text:
        raise ValueError(
            f"material {position + 1} of {count} has no words; a material is "
            f"described or it is not generated"
        )
    return ", ".join((text, MATERIAL_STYLE_CLAUSE))


def terrain_subjects(inner: str, outer: str, boundary: str = "") -> tuple[str, str]:
    """The two subjects a terrain set is generated from. ``(inner, outer)``.

    Two ordinary material subjects, which is the whole claim of the terrain
    mode: the model draws two surfaces and never sees an edge.
    :mod:`.tilemask` composites them through a distance field, so nothing here
    asks for a shoreline, a transition strip or a blend -- asking for one would
    put a *drawn* edge inside a tile that the field then cuts across, which is
    the one defect the composited path exists to make impossible.

    ``boundary`` is therefore **context, not an instruction**. It is appended to
    both subjects so the two materials come back from two independent samples
    sharing a world and a palette ("a temperate coastline" gives grass and water
    that belong to the same map). It is optional and empty is the ordinary case;
    a caller that puts an edge in it is asking for the defect above, which is why
    the parameter is named for the *place* rather than for the seam.

    ``inner`` is the terrain that gets the blob cases -- the one that appears as
    islands, coastlines and peninsulas -- and ``outer`` is what surrounds it.
    ``tilemask.blob_rects`` makes the centre cell a member always, so the order
    is not a convention: it is which of the two the forty-seven pictures are of.
    """
    context = str(boundary).strip()
    subjects = []
    for name, prompt in (("inner", inner), ("outer", outer)):
        text = str(prompt).strip()
        if not text:
            raise ValueError(
                f"a terrain set's {name} material has no words; both halves are "
                f"generated and both have to be described"
            )
        subjects.append(
            ", ".join(part for part in (text, context, MATERIAL_STYLE_CLAUSE) if part)
        )
    return (subjects[0], subjects[1])


def material_seeds(seed: int, count: int) -> tuple[int, ...]:
    """``count`` distinct seeds derived from one. ``seed + i``, wrapped.

    **Derived rather than drawn**, and that is the whole design: a request
    records one seed, so re-running it reproduces every material, and material
    ``i`` is reproducible *on its own* from the pair ``(seed, i)``. That is what
    makes a future "reroll material 3 only" a one-line change rather than a
    schema change -- the alternative, N seeds from one RNG, cannot say what
    material 3's seed was without replaying the draw.

    Wrapped modulo ``MAX_SEED + 1`` rather than clamped. Clamping is what turns a
    bound into a collision: ``MAX_SEED`` and ``MAX_SEED + 1`` would clamp to the
    same value, and two cells of one sheet would silently be the same picture.
    The wrap is distinct for any ``count`` this module allows, by a margin of
    about eight orders of magnitude.
    """
    base, total = int(seed), int(count)
    if not 0 <= base <= MAX_SEED:
        raise ValueError(f"a seed is between 0 and {MAX_SEED}; got {base}")
    if total < 1:
        raise ValueError(f"a request needs at least one seed; got {total}")
    if total > MAX_CELLS:
        raise ValueError(
            f"{total} seeds is past the {MAX_CELLS} cells one atlas can hold"
        )
    return tuple((base + offset) % (MAX_SEED + 1) for offset in range(total))


# -- the pixels ---------------------------------------------------------------


def reduce_material(pixels: Any, out_w: int, out_h: int) -> Any:
    """One seamless material at exactly the tile size, on an exact partition.

    :func:`tilesheet.reduce_cell` does the work -- the two-stage reducer
    measured in ``docs/measurements/2026-08-17-ground-reduction.md``, a box mean
    down to the art resolution and then a centre sample of each remaining group.
    What this adds is a refusal, and the refusal is the reason this function
    exists at all rather than the call being made directly.

    **A factor that does not divide exactly is fatal here and invisible there.**
    ``tilesheet._box_reduce``'s ``starts()`` lets blocks differ in size by one
    when the target does not divide the source, and its docstring calls that
    invisible "because the neighbouring blocks are still adjacent" -- which is
    true in the middle of a grid cell, and ``tilesheet``'s own module docstring
    draws exactly this line at lines 22-24: *a cell in a grid is not a torus*.
    A material is a torus. Its first and last block are neighbours, so a block
    that is one pixel wider than its opposite number puts a one-pixel step at the
    wrap seam -- and the periodicity that the circular-padded generation, the
    colour continuity in :mod:`.tilemask` and the whole seamless path exist for
    is gone, in the one place nobody looks.

    1024 divides by 16, 32, 64, 128 and 256. It does not divide by 48, which is
    in ``tilesheet.TILE_SIZES`` and is exactly the trap this refuses.
    """
    import numpy as np

    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError("a material is (h, w, channels)")
    height, width = int(array.shape[0]), int(array.shape[1])
    target_w, target_h = int(out_w), int(out_h)
    if target_w < 1 or target_h < 1:
        raise ValueError("a reduced tile is at least one pixel across")
    if target_w > width or target_h > height:
        raise ValueError(
            f"this material is {width}x{height} and cannot be reduced to "
            f"{target_w}x{target_h}; generate it larger"
        )
    for source, target, axis in ((width, target_w, "wide"), (height, target_h, "tall")):
        if source % target:
            raise ValueError(
                f"a {source}px seamless material does not reduce to {target}px "
                f"{axis} ({source}/{target} = {source / target:.4g}); the blocks "
                f"would differ by one pixel and the wrap seam of a torus is "
                f"exactly where that shows"
            )
    # ``reduce_cell``'s own second stage has to partition too. It reduces to
    # ``target * m`` before sampling, so ``m`` must divide the factor as well --
    # which is free at 1024 (a power of two, so every factor is) and is not free
    # for a caller reducing, say, 96px to 16px.
    factor = min(width // target_w, height // target_h)
    m = max(1, min(4, factor))
    if (width // target_w) % m or (height // target_h) % m:
        raise ValueError(
            f"reducing {width}x{height} to {target_w}x{target_h} prefilters at "
            f"{target_w * m}x{target_h * m}, which does not partition the source "
            f"either; a seamless material reduces by a whole power of the "
            f"prefilter step"
        )
    return tilesheet.reduce_cell(array, target_w, target_h)


def assemble(tiles: Any, geom: AtlasGeometry) -> Any:
    """The finished atlas. -> uint8 ``(rows*tile_h, columns*tile_w, 4)``.

    Cell ``i`` goes at ``(i // columns, i % columns)`` -- reading order, which
    for ``materials`` is the order the user typed and for ``terrain`` is
    ascending ``tilemask.BLOB_MASKS``. The same arithmetic serves both, which is
    what lets ``tests/test_tileatlas.py`` pin one row of this against
    :func:`tilemask.blob_atlas` byte for byte instead of trusting that two
    layouts agree.

    Three-channel tiles are promoted to opaque RGBA rather than refused.
    ``tileset.frozen_rgba`` requires four channels of anything that becomes a
    tileset and SDXL returns three, so the alternative is the same three lines at
    every call site -- and a promotion to alpha 255 is the only answer an opaque
    material has.
    """
    import numpy as np

    frames = list(tiles)
    if len(frames) != len(geom.cells):
        raise ValueError(
            f"this atlas has {len(geom.cells)} cells and {len(frames)} tiles were "
            f"given"
        )
    if not frames:
        raise ValueError("an atlas is assembled from at least one tile")
    if len(frames) > MAX_CELLS:
        raise ValueError(
            f"{len(frames)} tiles is past the {MAX_CELLS} cells one atlas can hold"
        )
    out = np.zeros(
        (geom.rows * geom.tile_h, geom.columns * geom.tile_w, 4), dtype=np.uint8
    )
    for cell, frame in zip(geom.cells, frames, strict=True):
        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                f"tile {cell.index} is {array.shape}; a tile is (h, w, 3) or "
                f"(h, w, 4)"
            )
        if array.shape[:2] != (geom.tile_h, geom.tile_w):
            raise ValueError(
                f"tile {cell.index} is {array.shape[1]}x{array.shape[0]} and this "
                f"atlas is laid out for {geom.tile_w}x{geom.tile_h} tiles"
            )
        top, left = cell.row * geom.tile_h, cell.col * geom.tile_w
        out[top : top + geom.tile_h, left : left + geom.tile_w, : array.shape[2]] = array
        if array.shape[2] == 3:
            out[top : top + geom.tile_h, left : left + geom.tile_w, 3] = 255
    return out


# -- the record ---------------------------------------------------------------


def _material_record(cell: MaterialCell) -> dict[str, Any]:
    """One cell as the sidecar carries it, refusing a cell nobody bound.

    :func:`material_geometry` leaves the words empty on purpose, and the failure
    this catches is the one that shape invites: a queue that laid out the grid,
    generated the materials and forgot to bind the two together would publish an
    atlas whose record says nothing about what is in it -- which is precisely the
    inference-instead-of-record failure this whole sidecar exists to prevent.
    """
    if not str(cell.prompt).strip():
        raise ValueError(
            f"cell {cell.index} has no prompt; a geometry's cells carry their "
            f"words only after the queue binds them"
        )
    return {
        "index": int(cell.index),
        "prompt": str(cell.prompt),
        "variant": int(cell.variant),
        "seed": int(cell.seed),
        "row": int(cell.row),
        "col": int(cell.col),
    }


def _terrain_record(entry: Any) -> dict[str, Any]:
    """One terrain as the sidecar carries it: a name and two colours.

    A **mapping** rather than a ``TerrainSpec``, because the writer is a worker
    process where ``studio`` is not importable at all. The colours are validated
    to ``tileset.rgba_colour``'s rule here rather than at the import door, for
    that function's own reason: a record that reaches the door malformed is a
    record that was written malformed, and the useful moment to say so is while
    the job that wrote it is still running.
    """
    if not isinstance(entry, dict):
        raise ValueError(
            f"a terrain record is a mapping of name/fill/outline; got {entry!r}"
        )
    name = str(entry.get("name", "")).strip()
    if not name:
        raise ValueError("a terrain has a name; it is what the map's palette shows")
    colours: dict[str, list[int]] = {}
    for key in ("fill", "outline"):
        try:
            channels = [int(part) for part in entry[key]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"a terrain {key} is four channels of 0..255; got {entry.get(key)!r}"
            ) from exc
        if len(channels) != 4 or any(part < 0 or part > 255 for part in channels):
            raise ValueError(
                f"a terrain {key} is four channels of 0..255; got {channels!r}"
            )
        colours[key] = channels
    return {"name": name, "fill": colours["fill"], "outline": colours["outline"]}


def _mask_record(mask: dict[str, Any]) -> dict[str, Any]:
    """The field that drew a terrain set, with :data:`tilemask.MASK_VERSION`
    stamped on it here rather than passed in.

    The version is the one value in the block the caller must not be able to get
    wrong: it says which field implementation produced these pixels, so a caller
    that supplied it could claim a version it did not run.
    """
    if not isinstance(mask, dict):
        raise ValueError(
            "a mask record is a mapping of seed/inset/amplitude/feather; got "
            f"{mask!r}"
        )
    try:
        seed = int(mask["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("a mask record names the seed its noise was drawn from") from exc
    record: dict[str, Any] = {"version": int(tilemask.MASK_VERSION), "seed": seed}
    for key in ("inset", "amplitude", "feather"):
        value = mask.get(key)
        record[key] = None if value is None else float(value)
    return record


def atlas_sidecar(
    geom: AtlasGeometry,
    *,
    created: float,
    materials: Any = (),
    terrains: Any = (),
    mask: dict[str, Any] | None = None,
    recipe: Any = None,
    grids: Any = (),
) -> dict[str, Any]:
    """The atlas's own record: what it is, not what it looks like.

    **Load-bearing, and not a convenience.** ``roles.infer_roles`` -- the thing
    that recognises a blob-47 set at the import door -- works by finding a
    background from transparency or from one dominant ring colour and matching
    every cell's silhouette against the forty-seven masks. A *generated* terrain
    set is two opaque textures composited edge to edge: it has no transparency
    and no ring colour, so the inference returns ``None`` on a set that is
    perfectly formed. The precedent is ``plotter_tilesets.SheetMismatch``, whose
    docstring says it outright -- **what catches it is not a measurement but a
    record** -- and this is that record: ``layout``, ``terrains`` and ``mask``
    are how a set is landed by what it *is* rather than by what a detector can
    still see in it.

    Written last, after the PNG, because it is the job's completion marker --
    ``sheet_sidecar``'s rule and ``_pixel_sheet``'s before it.

    ``grids`` is ``pixel.lattice``'s two numbers per generated material,
    measured once on each whole frame and never per cell. Additive, written
    only when there is one, and it does **not** bump ``TILE_ATLAS_VERSION``:
    a new optional key a reader has never heard of leaves it seeing exactly
    the file it saw before (``sheet.py``'s sidecar rule), and a bump that
    changes nothing would invalidate every stored benchmark comparison for
    free. Recorded and acted on by nothing; see ``pixel.lattice`` for why
    acting on it waits for a calibration run.

    Every value is a plain builtin, coerced here rather than at the call site:
    this is ``json.dumps``-ed *after* the atlas is on disk, and a numpy scalar
    that survived would fail the write with the artifact already published and no
    marker to say so. ``tests/test_tileatlas.py`` round-trips it.
    """
    if geom.mode not in MODES:
        raise ValueError(f"unknown mode {geom.mode!r}; this module builds {', '.join(MODES)}")
    cells = tuple(materials)
    terrain_specs = tuple(terrains)
    if len(cells) != len(geom.cells):
        raise ValueError(
            f"this atlas has {len(geom.cells)} cells and {len(cells)} material "
            f"records were given"
        )
    if geom.mode == MODE_TERRAIN:
        # ``Tileset.__post_init__`` needs one terrain per row at phase 1, and a
        # set that declares none is an ordinary atlas wearing a terrain layout --
        # every gid in it would be valid and every role wrong.
        if len(terrain_specs) != geom.rows:
            raise ValueError(
                f"a blob47 atlas of {geom.rows} row(s) declares {geom.rows} "
                f"terrain(s); {len(terrain_specs)} were given"
            )
        if mask is None:
            raise ValueError(
                "a terrain set records the mask field that drew it; without it "
                "nothing can say which noise produced these boundaries"
            )
    else:
        if terrain_specs:
            raise ValueError(
                "a materials atlas is a plain grid and declares no terrains; its "
                "cells have no blob roles to name"
            )
        if mask is not None:
            raise ValueError("a materials atlas is not composited and has no mask field")
    return {
        "version": int(TILE_ATLAS_VERSION),
        "created": float(created),
        "mode": str(geom.mode),
        "layout": str(geom.layout),
        "view": str(geom.view),
        "tile_w": int(geom.tile_w),
        "tile_h": int(geom.tile_h),
        "columns": int(geom.columns),
        "rows": int(geom.rows),
        "tiles": int(geom.tiles),
        "material_px": int(MATERIAL_PX),
        "materials": [_material_record(cell) for cell in cells],
        # Ordered, and the order is meaning: a terrain's position in this list is
        # its precedence, which is ``TerrainSpec``'s own rule and the whole of how
        # a cell with three terrains around it picks one picture.
        "terrains": [_terrain_record(entry) for entry in terrain_specs],
        "mask": None if mask is None else _mask_record(mask),
        "recipe": dict(recipe) if recipe is not None else None,
        # One per *material*, not per cell: a material is one generation and a
        # cell is a crop out of one, and ``pixel.lattice`` measures generations.
        # A terrain set is forty-seven cells composited from two of them, which
        # is exactly why this cannot ride on the ``materials`` list. Written
        # only when there is one, so a sidecar from before it existed and a
        # sidecar that measured nothing are the same file.
        **({"grids": [dict(entry) for entry in grids]} if grids else {}),
    }

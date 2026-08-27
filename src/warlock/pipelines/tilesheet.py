"""The half of an AI tile sheet that is arithmetic and words.

A tile sheet is one SDXL generation laid out as an 8x8 grid of *different*
tiles -- grass, path, water, cliff, prop -- sliced on that grid and reduced to
the user's tile size. It is the opposite trade from the seamless tile in
``prompt.TILE_TEMPLATE``: there, one surface has to survive being repeated;
here, sixty-four small pictures have to stay inside their own rectangles and
share one palette.

**The grid is imposed, not requested.** Each cell boundary is drawn into a
black-and-white guide that goes to the canny ControlNet, exactly as
``spritesynth.render_guide`` imposes a sprite sheet's cell layout. The slicing
then uses the rectangles the guide was drawn from, never rectangles re-detected
from the returned pixels -- a generation whose art drifted a few pixels still
slices identically, where seam-finding would turn one mis-registered sheet into
sixty-four differently-sized tiles.

**The reduction was measured for the ground path this replaced, and carried
over with it.** :func:`reduce_cell` is two stages -- a box mean down to the art
resolution, then a centre sample of each remaining group. The argument in
``docs/measurements/2026-08-17-ground-reduction.md`` is about a 128px block of
pixel-art-LoRA output reduced to a 32px tile, which is exactly what happens
here; only the seamless-torus half of that argument fell away with the ground
set (deleted 2026-08-18), because a cell in a grid is not a torus.

**The grid mechanism is measured; the art direction is not settled.**
``docs/measurements/2026-08-18-tile-sheet-grid.md`` records a four-arm run: the
guide *is* obeyed -- the model's discontinuities land on the rectangles the
slicer cuts on -- but SDXL 1.0 with the pixel-art LoRA resolves an 8x8 grid of
*identical* guide cells either by painting one continuous scene through it or by
painting one tile sixty-four times, depending on how hard the ControlNet is
pushed. ``spritesynth`` gets away with the same mechanism because each of its
cells carries a *different* guide, and a terrain tile has no canonical per-cell
silhouette to put there. So :data:`_PROJECTION_CLAUSE`, the guide's line width
and ``prompt.TILESHEET_TEMPLATE`` ship **provisional**, and that document is what
the next change to them argues against.

Pure, in this package's sense: stdlib at module scope, numpy and Pillow inside
the functions that need them, no torch, no service, no decision about where a
file goes. The queue owns the model calls and the paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

#: Bumped when anything below changes what a given form produces: the grid, the
#: guide, the subject clauses or the reduction. Recorded in the sidecar so a
#: sheet can say which compiler drew it, the way ``PIXEL_SHEET_VERSION`` does.
#: The *layout* clauses live in ``prompt.TILESHEET_TEMPLATE`` under
#: ``PROMPT_VERSION`` instead -- that template serves the prompt preview as well
#: as this path, and the two numbers answer different questions.
#:
#: 2: the third view. ``orthogonal`` became ``top_down`` (the old spelling still
#: reads, :func:`normalize_view`), ``three_quarter`` joined it, and every
#: subject clause is now this module's own rather than a lookup that fell back
#: to top-down when it did not recognise the key.
TILE_SHEET_VERSION = 2

#: One SDXL frame's long edge, and the reason the cell size below is what it is.
SHEET_PX = 1024

#: Eight by eight, fixed. Not a control: 1024/8 = 128 is the true art
#: resolution of one cell (the pixel-art LoRA draws ~8px "art pixels" at 1024),
#: so a denser grid would ask the model for detail it cannot draw and a sparser
#: one would waste the frame. Sixty-four tiles is also about what the reference
#: sheets in ``examples/`` carry.
COLS = 8
ROWS = 8

#: The stored spelling of :data:`TOP_DOWN`, and the only reason this constant
#: still exists. Rows and sidecars written before the vocabulary widened carry
#: ``"orthogonal"``; the house rule is that a stale value is *ignored, not
#: stripped*, so :func:`normalize_view` reads it and every one of them still
#: opens. Nothing writes it any more.
ORTHOGONAL = "orthogonal"

TOP_DOWN = "top_down"
THREE_QUARTER = "three_quarter"
ISOMETRIC = "isometric"

#: The three the sheet generator serves, and the one axis it has. A *view* is
#: where the camera is; the lattice is derived from it (:func:`tile_height`)
#: rather than chosen beside it, because two of the four camera-by-lattice
#: combinations name nothing that renders.
#:
#: ``oblique`` is still deliberately absent, and the reason it was refused is
#: worth restating because :data:`THREE_QUARTER` looks like the same case and is
#: not. Oblique uses the rectangle exactly as top-down does *and asks the model
#: for the same picture*, so it would have been a third button that changed
#: nothing. 3/4 shares the rectangle too -- but what SDXL is handed is the
#: subject clause and the guide, and both of those differ, which is the whole
#: measured argument (``docs/measurements/2026-08-21-three-quarter-guide.md``).
VIEWS: tuple[str, ...] = (TOP_DOWN, THREE_QUARTER, ISOMETRIC)

#: Stored spellings that are not the canonical one. Read-only: one entry, and a
#: second would want a reason of its own.
_VIEW_ALIASES: dict[str, str] = {ORTHOGONAL: TOP_DOWN}


def normalize_view(view: Any) -> str:
    """A stored view value as this module spells it today.

    The single door every entry point takes, so "orthogonal reads as top-down"
    is one fact in one place rather than a `` == `` scattered across the door,
    the worker and the pane. Membership is *not* checked here: :func:`geometry`
    and :func:`sheet_subject` each refuse an unknown view in their own words,
    and a third refusal site would be a third message about one mistake.
    """
    text = str(view)
    return _VIEW_ALIASES.get(text, text)

#: What the form offers. Every one divides evenly enough to leave the two-stage
#: reduction a factor to work with (128 // 16 = 8, 128 // 64 = 2), and every one
#: is even, which isometric requires.
TILE_SIZES: tuple[int, ...] = (16, 32, 48, 64)

#: ``service.validation.MAX_SEED``, restated rather than imported: a pipeline
#: may not import the service layer. ``tests/test_tilesheet_pipeline.py`` pins
#: the two together, so the copy cannot drift.
MAX_SEED = 2**31 - 1

#: Appended to every sheet subject, and measured for the ground path this
#: replaced (see the module docstring): a cell keeps at most ~128px of true
#: detail, and a material whose elements are smaller than that reduces to
#: near-noise. Asking for oversized, high-contrast elements is what survives.
DETAIL_CLAUSE = (
    "chunky oversized texture details, strong colour contrast, "
    "large distinct repeating elements"
)

#: What the service door uses when the form left the negative prompt empty.
#: The anti-photo, anti-single-object list the ground run settled, plus the
#: two failure modes specific to a grid: cells bleeding into one another, and
#: the model drawing one big picture across the whole frame instead of
#: sixty-four small ones. ``sdxl_cfg`` runs at CFG 7.0, where a negative
#: prompt actually steers.
SHEET_NEGATIVE_PROMPT = (
    "blurry, smooth gradients, photorealistic, photo, 3d render, "
    "film grain, soft focus, depth of field, single centered object, "
    "text, watermark, signature, vignette, picture frame, "
    "one large scene, tiles bleeding together, inconsistent scale"
)

#: How each view frames a cell, as words. Kept here rather than in the
#: template, and the split is the point: the template serves the prompt preview
#: as well as this path, and a change to it bumps ``PROMPT_VERSION`` -- whereas
#: these clauses are this module's own and sit under
#: :data:`TILE_SHEET_VERSION`.
#:
#: The 3/4 clause is deliberately *not* ``prompt.PROMPT_TEMPLATE``'s "3/4
#: perspective view" verbatim. That literal frames a single object for trellis
#: and is pinned byte-for-byte by ``tests/test_prompt.py``; this one frames a
#: ground tile, and two clauses that happened to share a spelling would be one
#: string with two owners.
_VIEW_CLAUSE: dict[str, str] = {
    TOP_DOWN: "flat top-down orthographic view, square tiles",
    THREE_QUARTER: (
        "3/4 top-down game view, camera tilted about thirty degrees, "
        "square tiles with a shallow visible front face"
    ),
    ISOMETRIC: "2:1 isometric projection, diamond tiles on a dimetric grid",
}


@dataclass(frozen=True, slots=True)
class Cell:
    """One cell's rectangle in generation pixels, and where it lands on the
    finished sheet."""

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
    """The fixed grid one request is generated and sliced on."""

    view: str
    columns: int
    rows: int
    cell_w: int
    cell_h: int
    tile_w: int
    tile_h: int
    cells: tuple[Cell, ...]

    @property
    def source_size(self) -> tuple[int, int]:
        """What the model is asked for, in pixels. ``(w, h)``."""
        return (self.columns * self.cell_w, self.rows * self.cell_h)

    @property
    def sheet_size(self) -> tuple[int, int]:
        """What is published, in pixels. ``(w, h)``."""
        return (self.columns * self.tile_w, self.rows * self.tile_h)

    @property
    def tiles(self) -> int:
        return len(self.cells)


def tile_height(tile_w: int, view: str) -> int:
    """A tile's height, which is a function of its width and the view.

    Not a second control. An isometric tile is 2:1 by definition of the view,
    and a form that let the two disagree would let a user ask for a diamond
    lattice that does not tesselate.

    Spelled out per view rather than "isometric halves it, everything else
    falls through". The fallthrough gave a *correct* answer for a view this
    module had never heard of, which is the silent defaulting :func:`geometry`
    exists to refuse -- and a third view is what turns that from a tidiness
    argument into a live one.
    """
    width = int(tile_w)
    resolved = normalize_view(view)
    if resolved == ISOMETRIC:
        return width // 2
    if resolved in (TOP_DOWN, THREE_QUARTER):
        # A 3/4 tile is square: the tilt is in the art, not the lattice. That
        # is exactly why the front face is the guide's business.
        return width
    raise ValueError(f"unknown view {resolved!r}; this module draws {', '.join(VIEWS)}")


def geometry(tile_w: int, view: str) -> SheetGeometry:
    """The grid for one request. Unknown or impossible asks raise.

    Defaulting on an unknown view would generate a top-down sheet for a typo'd
    isometric request and publish it under the caller's name -- a wrong sheet
    nobody asked for, rather than an error somebody can read.
    ``spritesynth.geometry``'s rule, and its reason.
    """
    width = int(tile_w)
    if width < 1:
        raise ValueError("a tile is at least one pixel across")
    text = normalize_view(view)
    if text not in VIEWS:
        raise ValueError(
            f"unknown view {text!r}; this module draws {', '.join(VIEWS)}"
        )
    if text == ISOMETRIC and width % 2:
        # ``tile_h`` is ``tile_w // 2``, so an odd width silently loses half a
        # pixel and puts every diamond a half-pixel off its own lattice.
        raise ValueError(f"an isometric tile needs an even width; {width} is odd")

    cell_w = SHEET_PX // COLS
    # Only isometric halves the cell: 3/4 is generated on the same square frame
    # as top-down, because the tilt is in what is drawn rather than in the
    # rectangle it is drawn in.
    cell_h = cell_w // 2 if text == ISOMETRIC else cell_w
    height = tile_height(width, text)
    if width > cell_w or height > cell_h:
        raise ValueError(
            f"a {width}x{height} tile cannot be larger than the {cell_w}x{cell_h} "
            f"cell it is reduced from"
        )
    cells = tuple(
        Cell(row=row, col=col, x=col * cell_w, y=row * cell_h, w=cell_w, h=cell_h)
        for row in range(ROWS)
        for col in range(COLS)
    )
    return SheetGeometry(
        view=text,
        columns=COLS,
        rows=ROWS,
        cell_w=cell_w,
        cell_h=cell_h,
        tile_w=width,
        tile_h=height,
        cells=cells,
    )


def render_guide(geom: SheetGeometry) -> PILImage:
    """The generation-sized grid guide: white lines on black.

    Handed to the ControlNet as the hint *directly*. It is already line art in
    canny space, and running the detector over it would return the outline of
    each stroke -- two lines where the guide means one, which for a grid is a
    doubled seam in every cell.

    The isometric guide inscribes a diamond in each cell as well as drawing the
    rectangle. Without it the model fills the rectangle and every tile comes
    back as a square seen from above; with it, the cell still says where the
    storage rectangle is (which is what the slicer needs) and the diamond says
    where the tile's top face goes.

    **A 3/4 cell gets the plain rectangle, exactly as a top-down one does, and
    that was measured rather than assumed** --
    ``docs/measurements/2026-08-21-three-quarter-guide.md``. Two interior marks
    were tried (a horizon line at 2/3 height, and that plus a hatched front
    band) and both were *obeyed* and both made the sheet worse: the model drew
    the mark as a dark stripe rather than a change of plane, and flattened every
    cell towards the same tile doing it -- the failure mode the grid measurement
    already named. What carries the view is the subject clause, so there are
    still exactly two guide shapes here and the third view adds none.
    """
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", geom.source_size, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    white = (255, 255, 255)
    # Two pixels at the house 128px cell. Thin on purpose: the guide is a
    # boundary, and a fat line is a wall the model draws furniture against.
    width = max(2, geom.cell_w // 128)
    for cell in geom.cells:
        right, bottom = cell.x + cell.w - 1, cell.y + cell.h - 1
        draw.rectangle([cell.x, cell.y, right, bottom], outline=white, width=width)
        # A per-view dispatch rather than "everything that is not isometric
        # continues". Two views share the plain rectangle today and the third
        # does not, so which of them a new view joins has to be a decision
        # somebody wrote down.
        if geom.view == ISOMETRIC:
            mid_x = cell.x + cell.w // 2
            mid_y = cell.y + cell.h // 2
            # Four explicit lines rather than ``polygon(outline=...)``: the
            # width keyword on a polygon outline is newer than the Pillow floor
            # this package builds against, and an antialiased fallback would
            # break the "only 0 and 255" promise the canny hint depends on.
            points = [(mid_x, cell.y), (right, mid_y), (mid_x, bottom), (cell.x, mid_y)]
            for start, end in zip(points, points[1:] + points[:1], strict=True):
                draw.line([start, end], fill=white, width=width)
    return canvas


def sheet_subject(prompt: str, view: str) -> str:
    """The subject clause for one sheet. A *subject*, not a finished prompt.

    The caller runs this through ``guidance.compose_prompt`` and
    ``prompt.TILESHEET_TEMPLATE``, which is what adds the grid, the
    separate-cells and the no-text clauses. Putting those here would mean a
    sheet prompt assembled two ways -- and the two halves version separately,
    which is the reason the split is drawn here rather than anywhere else:
    the template sits under ``PROMPT_VERSION`` because the prompt preview
    serves it too, and the clauses below sit under :data:`TILE_SHEET_VERSION`
    because only this path can reach them.

    An unknown view raises rather than falling back to the top-down clause.
    This was the one function that defaulted where :func:`geometry` refuses, and
    the fallback was invisible by construction: it produced a *plausible* sheet
    described by the wrong sentence. The moment a view can be added to
    :data:`VIEWS` without a clause beside it, that is a wrong picture nobody is
    told about.
    """
    text = str(prompt).strip()
    resolved = normalize_view(view)
    clause = _VIEW_CLAUSE.get(resolved)
    if clause is None:
        raise ValueError(f"unknown view {resolved!r}; this module draws {', '.join(VIEWS)}")
    return ", ".join(part for part in (text, clause, DETAIL_CLAUSE) if part)


def reduce_cell(pixels: Any, out_w: int, out_h: int) -> Any:
    """One cell at exactly the tile size.

    Two stages, both partitions of the cell: a box mean down to ``out * m`` (a
    light prefilter to the art resolution -- at 128 -> 32 the cap of 4 lands on
    128, one mid pixel per pixel-art-LoRA "art pixel"), then the centre sample
    of each remaining ``m x m`` group. A pure box mean all the way down
    averaged uncorrelated art pixels into every output pixel and regressed each
    material to its mean colour; a pure centre sample keeps single-pixel noise.
    Measured in ``docs/measurements/2026-08-17-ground-reduction.md``.

    Identity when the target equals the source, and ``m`` degrades to 1 for
    tiles near the cell size, where the stages collapse back to the plain box
    mean. Integer-only throughout, so the answer does not depend on a float
    rounding mode.
    """
    import numpy as np

    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError("a cell is (h, w, channels)")
    height, width = int(array.shape[0]), int(array.shape[1])
    out_w, out_h = int(out_w), int(out_h)
    if out_w < 1 or out_h < 1:
        raise ValueError("a reduced tile is at least one pixel across")
    if out_w > width or out_h > height:
        raise ValueError(
            f"this cell is {width}x{height} and cannot be reduced to "
            f"{out_w}x{out_h}; generate it larger"
        )
    m = max(1, min(4, height // out_h, width // out_w))
    mid = _box_reduce(array, out_w * m, out_h * m)
    return np.ascontiguousarray(mid[m // 2 :: m, m // 2 :: m])


def _box_reduce(pixels: Any, out_w: int, out_h: int) -> Any:
    """The integer box mean over an exact partition of the source.

    Every output pixel is the integer mean of one *block* of input pixels, and
    the blocks partition the source exactly. Blocks are allowed to differ in
    size by one when the target does not divide the source; that is the price
    of not restricting tile sizes to divisors of 128, and it is invisible
    because the neighbouring blocks are still adjacent.

    Integer accumulation in ``int64`` with round-half-up, so the answer does not
    depend on a float rounding mode.
    """
    import numpy as np

    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError("a cell is (h, w, channels)")
    height, width = int(array.shape[0]), int(array.shape[1])
    out_w, out_h = int(out_w), int(out_h)
    if out_w < 1 or out_h < 1:
        raise ValueError("a reduced tile is at least one pixel across")
    if out_w > width or out_h > height:
        raise ValueError(
            f"this cell is {width}x{height} and cannot be reduced to "
            f"{out_w}x{out_h}; generate it larger"
        )

    def starts(total: int, groups: int) -> Any:
        # Input index ``i`` belongs to output group ``i * groups // total``, so a
        # group begins at ``ceil(g * total / groups)``. Ceil by negation, which
        # keeps it integer.
        return np.array([-(-g * total // groups) for g in range(groups)], dtype=np.int64)

    rows = starts(height, out_h)
    columns = starts(width, out_w)
    row_counts = np.diff(np.append(rows, height))
    column_counts = np.diff(np.append(columns, width))

    summed = np.add.reduceat(array.astype(np.int64), rows, axis=0)
    summed = np.add.reduceat(summed, columns, axis=1)
    counts = (row_counts[:, None] * column_counts[None, :])[:, :, None]
    return ((summed + counts // 2) // counts).astype(np.uint8)


def reduce_sheet(pixels: Any, geom: SheetGeometry) -> Any:
    """The whole generation reduced onto the finished sheet.

    Cell by cell through :func:`reduce_cell`, rather than one resize of the
    whole image: the tile size need not divide the cell, and a whole-image
    resample would then let one output pixel straddle two cells -- a tile
    carrying a sliver of its neighbour, which on a tile sheet is the one defect
    that is visible in every map painted with it.
    """
    import numpy as np

    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError("a sheet is (h, w, channels)")
    height, width = int(array.shape[0]), int(array.shape[1])
    want_w, want_h = geom.source_size
    if (width, height) != (want_w, want_h):
        raise ValueError(
            f"this grid is generated at {want_w}x{want_h} and the image is "
            f"{width}x{height}"
        )
    out_w, out_h = geom.sheet_size
    out = np.zeros((out_h, out_w, array.shape[2]), dtype=np.uint8)
    for cell in geom.cells:
        crop = array[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
        tile = reduce_cell(crop, geom.tile_w, geom.tile_h)
        top, left = cell.row * geom.tile_h, cell.col * geom.tile_w
        out[top : top + geom.tile_h, left : left + geom.tile_w] = tile
    return out


def sheet_sidecar(
    *,
    prompt: str,
    tile_w: int,
    tile_h: int,
    view: str,
    colors: int,
    palette: Any,
    recipe: Any,
    created: float,
    working_cell_px: tuple[int, int] | None = None,
    target_cell_px: int | None = None,
    reduction: str | None = None,
    source_seed: int | None = None,
    workflow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The sheet's own record: what was asked for, and what actually ran.

    Written last, after the PNG, because it is this job's completion marker --
    the rule ``_pixel_sheet`` follows about its own sidecar. The
    recipe records what ran rather than what was requested, so a style the base
    could not take is simply absent instead of being claimed.

    Every value is coerced to a plain builtin here rather than at the call
    site: this is written with ``json.dumps`` *after* the sheet is on disk, and
    a numpy scalar that survived would fail the write with the artifact already
    published and no marker to say so.

    **The key on disk is still ``"projection"``** while the value is a *view*.
    The rename is a Python one: every sidecar already written carries that key,
    ``plotter_tilesets`` reads sheets by it, and renaming it would mean two
    spellings on disk to buy one tidier word. The vocabulary widened; the key
    did not move.
    """
    return {
        "version": TILE_SHEET_VERSION,
        "created": float(created),
        "prompt": str(prompt),
        "image": "input.png",
        "columns": int(COLS),
        "rows": int(ROWS),
        "tiles": int(COLS * ROWS),
        "tile_w": int(tile_w),
        "tile_h": int(tile_h),
        "projection": normalize_view(view),
        "colors": int(colors),
        "palette": [str(entry) for entry in palette],
        "recipe": dict(recipe),
        # Added fields are optional so sidecars written by older workers keep
        # their original shape and remain readable.
        "working_cell_px": list(
            working_cell_px or (128, 64 if normalize_view(view) == ISOMETRIC else 128)
        ),
        "target_cell_px": target_cell_px,
        "reduction": reduction if target_cell_px is not None else None,
        "source_seed": int(source_seed) if source_seed is not None else None,
        **({"workflow": dict(workflow)} if workflow is not None else {}),
    }

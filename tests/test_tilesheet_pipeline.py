"""The tile sheet's pure half: the grid, the guide, and the reduction.

Two claims carry this module.

The first is that **the grid is arithmetic, never detection**. A generated sheet
whose art drifted a few pixels off the guide still slices on the same
rectangles, because the rectangles were decided before the generation ran. The
alternative -- finding the seams in the returned pixels -- turns one
mis-registered generation into sixty-four differently-sized tiles that nothing
downstream can lay out.

The second is that **the reduction partitions the cell**. It was measured for
the ground path this replaced -- deleted 2026-08-18, measurement kept at
``docs/measurements/2026-08-17-ground-reduction.md`` -- and the argument
survives the move unchanged: a 128px cell is one pixel-art-LoRA "art pixel" per
mid pixel, so a plain box mean all the way down to 32 averages 4x4 uncorrelated
art pixels and regresses every tile to its mean colour.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.pipelines import tilesheet
from warlock.service.validation import MAX_SEED


def _noise(width: int, height: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)


def _blocks(width: int, height: int, block: int) -> np.ndarray:
    """Flat blocks of one colour each, so a partitioning reduction is exact."""
    rng = np.random.default_rng(3)
    tall, wide = height // block, width // block
    small = rng.integers(0, 256, size=(tall, wide, 4), dtype=np.uint8)
    return np.repeat(np.repeat(small, block, axis=0), block, axis=1)


# -- the grid ----------------------------------------------------------------


def test_an_orthogonal_sheet_is_eight_by_eight_square_cells():
    """128px cells are not arbitrary: the pixel-art LoRA draws ~8px art pixels
    at 1024, so 1024/8 is the true art resolution of one cell."""
    geom = tilesheet.geometry(32, tilesheet.ORTHOGONAL)
    assert (geom.columns, geom.rows) == (8, 8)
    assert (geom.cell_w, geom.cell_h) == (128, 128)
    assert geom.source_size == (1024, 1024)
    assert len(geom.cells) == 64


def test_an_isometric_sheet_keeps_eight_by_eight_but_halves_the_cell():
    """An isometric tile is 2:1, so the cell is -- and the generation gets
    shorter rather than the grid getting denser. 8x8 either way is what makes
    the two projections one feature and not two."""
    geom = tilesheet.geometry(64, tilesheet.ISOMETRIC)
    assert (geom.columns, geom.rows) == (8, 8)
    assert (geom.cell_w, geom.cell_h) == (128, 64)
    assert geom.source_size == (1024, 512)
    assert len(geom.cells) == 64


@pytest.mark.parametrize("tile_w", tilesheet.TILE_SIZES)
def test_an_orthogonal_tile_is_square_and_an_isometric_one_is_two_to_one(tile_w):
    assert tilesheet.tile_height(tile_w, tilesheet.ORTHOGONAL) == tile_w
    assert tilesheet.tile_height(tile_w, tilesheet.ISOMETRIC) == tile_w // 2


@pytest.mark.parametrize("tile_w", tilesheet.TILE_SIZES)
@pytest.mark.parametrize("projection", tilesheet.PROJECTIONS)
def test_the_sheet_is_exactly_the_grid_times_the_tile(tile_w, projection):
    geom = tilesheet.geometry(tile_w, projection)
    tile_h = tilesheet.tile_height(tile_w, projection)
    assert geom.tile_w == tile_w
    assert geom.tile_h == tile_h
    assert geom.sheet_size == (8 * tile_w, 8 * tile_h)


@pytest.mark.parametrize("projection", tilesheet.PROJECTIONS)
def test_the_cells_partition_the_generation_exactly(projection):
    """No gap and no overlap. A gap is a strip of art nothing ever reads; an
    overlap is one strip published in two tiles."""
    geom = tilesheet.geometry(32, projection)
    width, height = geom.source_size
    covered = np.zeros((height, width), dtype=np.int32)
    for cell in geom.cells:
        covered[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w] += 1
    assert (covered == 1).all()


@pytest.mark.parametrize("projection", tilesheet.PROJECTIONS)
def test_every_cell_knows_where_it_sits(projection):
    geom = tilesheet.geometry(32, projection)
    seen = {(cell.row, cell.col) for cell in geom.cells}
    assert seen == {(row, col) for row in range(8) for col in range(8)}
    for cell in geom.cells:
        assert cell.box == (cell.x, cell.y, cell.x + cell.w, cell.y + cell.h)


def test_an_unknown_projection_raises_rather_than_defaulting():
    """Defaulting would generate an orthogonal sheet for a typo'd isometric
    request and publish it under the caller's name -- a wrong sheet nobody
    asked for, rather than an error somebody can read."""
    with pytest.raises(ValueError, match="hexagonal"):
        tilesheet.geometry(32, "hexagonal")


def test_a_tile_larger_than_its_cell_is_refused():
    """The reduction may only ever reduce. A tile bigger than the 128px cell
    would have to invent detail the generation never drew."""
    with pytest.raises(ValueError, match="cannot be larger"):
        tilesheet.geometry(256, tilesheet.ORTHOGONAL)


@pytest.mark.parametrize("tile_w", [0, -8])
def test_a_tile_must_have_a_positive_edge(tile_w):
    with pytest.raises(ValueError):
        tilesheet.geometry(tile_w, tilesheet.ORTHOGONAL)


def test_an_odd_tile_cannot_be_isometric():
    """``tile_h`` is ``tile_w // 2``, so an odd width would silently lose half a
    pixel and put every diamond a half-pixel off its own lattice."""
    with pytest.raises(ValueError, match="even"):
        tilesheet.geometry(33, tilesheet.ISOMETRIC)


# -- the guide ---------------------------------------------------------------


@pytest.mark.parametrize("projection", tilesheet.PROJECTIONS)
def test_the_guide_is_the_size_of_the_generation(projection):
    geom = tilesheet.geometry(32, projection)
    guide = tilesheet.render_guide(geom)
    assert guide.size == geom.source_size
    assert guide.mode == "RGB"


@pytest.mark.parametrize("projection", tilesheet.PROJECTIONS)
def test_the_guide_is_line_art_in_canny_space(projection):
    """White strokes on black, handed to the ControlNet directly. Running the
    detector over it would outline every stroke and draw two lines where the
    guide means one -- ``spritesynth.render_guide``'s reason, verbatim."""
    geom = tilesheet.geometry(32, projection)
    arr = np.asarray(tilesheet.render_guide(geom))
    assert (arr == 0).any()
    assert (arr == 255).any()
    assert set(np.unique(arr)) <= {0, 255}


def test_the_guide_draws_every_interior_cell_boundary():
    """The whole point: the model is told where the seams go rather than asked.
    Every interior column and row boundary must carry ink."""
    geom = tilesheet.geometry(32, tilesheet.ORTHOGONAL)
    arr = np.asarray(tilesheet.render_guide(geom)).max(axis=2)
    for col in range(1, geom.columns):
        assert arr[:, col * geom.cell_w].any(), f"column boundary {col} is blank"
    for row in range(1, geom.rows):
        assert arr[row * geom.cell_h, :].any(), f"row boundary {row} is blank"


def test_the_isometric_guide_inscribes_a_diamond_the_orthogonal_one_does_not():
    """An isometric cell stores a diamond with transparent corners, so the
    guide has to say where the top face is. Without it the model fills the
    rectangle and every tile comes back as a square seen from above."""
    iso = np.asarray(tilesheet.render_guide(tilesheet.geometry(32, tilesheet.ISOMETRIC)))
    ortho = np.asarray(tilesheet.render_guide(tilesheet.geometry(32, tilesheet.ORTHOGONAL)))
    # The centre of the first cell's left edge is on the diamond for iso and
    # empty interior for orthogonal.
    assert iso[32, 0:4].max() > 0
    assert ortho[64, 4:120].max() == 0


def test_the_guide_is_the_same_picture_every_time():
    geom = tilesheet.geometry(48, tilesheet.ORTHOGONAL)
    first = np.asarray(tilesheet.render_guide(geom))
    second = np.asarray(tilesheet.render_guide(geom))
    assert np.array_equal(first, second)


# -- the subject -------------------------------------------------------------


def test_the_subject_carries_the_detail_clause():
    """A material whose elements are smaller than the art resolution reduces to
    near-noise, so oversized high-contrast elements are asked for explicitly."""
    text = tilesheet.sheet_subject("mossy dungeon floor", tilesheet.ORTHOGONAL)
    assert tilesheet.DETAIL_CLAUSE in text
    assert "mossy dungeon floor" in text


@pytest.mark.parametrize(
    "projection,word",
    [(tilesheet.ORTHOGONAL, "top-down"), (tilesheet.ISOMETRIC, "isometric")],
)
def test_the_subject_names_its_projection(projection, word):
    assert word in tilesheet.sheet_subject("stone", projection)


def test_an_empty_subject_leaves_no_dangling_comma():
    """``guidance.compose_prompt`` joins on comma boundaries, so an empty part
    that survived would reach CLIP as a stray separator."""
    text = tilesheet.sheet_subject("   ", tilesheet.ORTHOGONAL)
    assert ", ," not in text
    assert not text.startswith(",")
    assert not text.endswith(",")


def test_the_subject_is_a_subject_and_not_a_finished_prompt():
    """The grid clauses live in ``prompt.TILESHEET_TEMPLATE`` under
    PROMPT_VERSION; this module's words sit under TILE_SHEET_VERSION. Putting
    the template here would mean a sheet prompt assembled two ways."""
    text = tilesheet.sheet_subject("stone", tilesheet.ORTHOGONAL)
    assert "no watermark" not in text
    assert "tile sheet" not in text


# -- the reduction -----------------------------------------------------------


def test_a_reduction_to_its_own_size_changes_nothing():
    source = _noise(128, 128)
    assert np.array_equal(tilesheet.reduce_cell(source, 128, 128), source)


def test_a_reduction_may_never_enlarge():
    """It would have to invent detail the generation never drew."""
    with pytest.raises(ValueError, match="cannot be reduced"):
        tilesheet.reduce_cell(_noise(32, 32), 128, 128)


def test_a_reduction_lands_on_exactly_the_asked_for_size():
    out = tilesheet.reduce_cell(_noise(128, 128), 48, 24)
    assert out.shape == (24, 48, 4)


def test_a_reduction_is_the_same_answer_every_time():
    source = _noise(128, 128)
    assert np.array_equal(
        tilesheet.reduce_cell(source, 32, 32), tilesheet.reduce_cell(source, 32, 32)
    )


def test_a_reduction_is_integer_only():
    """``uint8`` in, ``uint8`` out, with no float rounding mode in the middle --
    the sheet's bytes must not depend on the host's libm."""
    out = tilesheet.reduce_cell(_noise(128, 128), 32, 32)
    assert out.dtype == np.uint8


def test_a_flat_block_reduces_to_its_own_colours():
    """The partition promise, tested where the answer is knowable: blocks of one
    colour each, reduced onto their own lattice, must come back unchanged."""
    source = _blocks(128, 128, 4)
    out = tilesheet.reduce_cell(source, 32, 32)
    assert np.array_equal(out, source[::4, ::4])


def test_two_stage_beats_a_plain_box_mean_on_contrast():
    """The measured reason the sampler is two-stage
    (docs/measurements/2026-08-17-ground-reduction.md): averaging 4x4
    uncorrelated art pixels regresses every material to its mean colour."""
    source = _noise(128, 128)
    two_stage = tilesheet.reduce_cell(source, 32, 32).astype(np.int64)
    box = tilesheet._box_reduce(source, 32, 32).astype(np.int64)
    assert two_stage.std() > box.std()


# -- the whole sheet ---------------------------------------------------------


@pytest.mark.parametrize("projection", tilesheet.PROJECTIONS)
def test_a_reduced_sheet_is_exactly_the_output_size(projection):
    geom = tilesheet.geometry(32, projection)
    width, height = geom.source_size
    out = tilesheet.reduce_sheet(_noise(width, height), geom)
    assert out.shape == (geom.sheet_size[1], geom.sheet_size[0], 4)


@pytest.mark.parametrize("projection", tilesheet.PROJECTIONS)
def test_every_tile_is_its_own_cell_reduced(projection):
    """The sheet is assembled from the same function the cell test pins, so a
    tile can never be a blend of two cells -- which is what a whole-atlas
    resize would produce wherever the tile size did not divide the cell."""
    geom = tilesheet.geometry(48, projection)
    width, height = geom.source_size
    source = _noise(width, height)
    out = tilesheet.reduce_sheet(source, geom)
    for cell in geom.cells:
        crop = source[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
        expected = tilesheet.reduce_cell(crop, geom.tile_w, geom.tile_h)
        got = out[
            cell.row * geom.tile_h : (cell.row + 1) * geom.tile_h,
            cell.col * geom.tile_w : (cell.col + 1) * geom.tile_w,
        ]
        assert np.array_equal(got, expected), f"tile r{cell.row}c{cell.col} differs"


def test_a_sheet_of_the_wrong_size_is_refused():
    """The worker hands over whatever the model returned; a generation that came
    back at another size is a bug worth a sentence, not a silent crop."""
    geom = tilesheet.geometry(32, tilesheet.ORTHOGONAL)
    with pytest.raises(ValueError, match="1024x1024"):
        tilesheet.reduce_sheet(_noise(512, 512), geom)


# -- the sidecar -------------------------------------------------------------


def test_the_sidecar_records_what_was_asked_for_and_what_ran():
    doc = tilesheet.sheet_sidecar(
        prompt="mossy dungeon",
        tile_w=32,
        tile_h=32,
        projection=tilesheet.ORTHOGONAL,
        colors=64,
        palette=["#000000"],
        recipe={"base_model": "sdxl_cfg"},
        created=1.0,
    )
    assert doc["version"] == tilesheet.TILE_SHEET_VERSION
    assert doc["image"] == "input.png"
    assert doc["columns"] == 8
    assert doc["rows"] == 8
    assert doc["tiles"] == 64
    assert doc["tile_w"] == 32
    assert doc["projection"] == tilesheet.ORTHOGONAL
    assert doc["recipe"] == {"base_model": "sdxl_cfg"}


def test_the_sidecar_is_plain_json_values():
    """It is written with ``json.dumps`` as the completion marker, so a numpy
    scalar in it would fail the write *after* the sheet had been published."""
    import json

    doc = tilesheet.sheet_sidecar(
        prompt="stone",
        tile_w=np.int64(32),
        tile_h=np.int64(32),
        projection=tilesheet.ISOMETRIC,
        colors=np.int64(64),
        palette=["#ffffff"],
        recipe={},
        created=2.0,
    )
    assert json.loads(json.dumps(doc)) == doc


# -- the restated constant ---------------------------------------------------


def test_the_seed_ceiling_matches_the_service_door():
    """A pipeline may not import the service layer, so the ceiling is restated
    -- and pinned here, so the copy cannot drift."""
    assert tilesheet.MAX_SEED == MAX_SEED

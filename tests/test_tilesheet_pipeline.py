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


def test_a_top_down_sheet_is_eight_by_eight_square_cells():
    """128px cells are not arbitrary: the pixel-art LoRA draws ~8px art pixels
    at 1024, so 1024/8 is the true art resolution of one cell."""
    geom = tilesheet.geometry(32, tilesheet.TOP_DOWN)
    assert (geom.columns, geom.rows) == (8, 8)
    assert (geom.cell_w, geom.cell_h) == (128, 128)
    assert geom.source_size == (1024, 1024)
    assert len(geom.cells) == 64


def test_an_isometric_sheet_keeps_eight_by_eight_but_halves_the_cell():
    """An isometric tile is 2:1, so the cell is -- and the generation gets
    shorter rather than the grid getting denser. 8x8 whichever view is asked
    for is what makes the three one feature and not three."""
    geom = tilesheet.geometry(64, tilesheet.ISOMETRIC)
    assert (geom.columns, geom.rows) == (8, 8)
    assert (geom.cell_w, geom.cell_h) == (128, 64)
    assert geom.source_size == (1024, 512)
    assert len(geom.cells) == 64


@pytest.mark.parametrize("tile_w", tilesheet.TILE_SIZES)
def test_only_isometric_halves_the_tile(tile_w):
    """3/4 is square like top-down: the tilt is in the art, not the lattice.
    That is the whole reason the front face is the guide's business."""
    assert tilesheet.tile_height(tile_w, tilesheet.TOP_DOWN) == tile_w
    assert tilesheet.tile_height(tile_w, tilesheet.THREE_QUARTER) == tile_w
    assert tilesheet.tile_height(tile_w, tilesheet.ISOMETRIC) == tile_w // 2


def test_tile_height_refuses_a_view_it_does_not_know():
    """It used to fall through to square, which is a *correct* answer given by
    accident -- the silent defaulting ``geometry`` exists to refuse."""
    with pytest.raises(ValueError, match="hexagonal"):
        tilesheet.tile_height(32, "hexagonal")


def test_the_old_orthogonal_spelling_still_reads():
    """Rows and sidecars written before the vocabulary widened carry it, and a
    stale value is ignored rather than stripped -- so every one still opens."""
    assert tilesheet.normalize_view("orthogonal") == tilesheet.TOP_DOWN
    assert tilesheet.geometry(32, "orthogonal").view == tilesheet.TOP_DOWN
    assert tilesheet.tile_height(32, "orthogonal") == 32
    assert "top-down" in tilesheet.sheet_subject("moss", "orthogonal")


@pytest.mark.parametrize("tile_w", tilesheet.TILE_SIZES)
@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_the_sheet_is_exactly_the_grid_times_the_tile(tile_w, view):
    geom = tilesheet.geometry(tile_w, view)
    tile_h = tilesheet.tile_height(tile_w, view)
    assert geom.tile_w == tile_w
    assert geom.tile_h == tile_h
    assert geom.sheet_size == (8 * tile_w, 8 * tile_h)


@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_the_cells_partition_the_generation_exactly(view):
    """No gap and no overlap. A gap is a strip of art nothing ever reads; an
    overlap is one strip published in two tiles."""
    geom = tilesheet.geometry(32, view)
    width, height = geom.source_size
    covered = np.zeros((height, width), dtype=np.int32)
    for cell in geom.cells:
        covered[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w] += 1
    assert (covered == 1).all()


@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_every_cell_knows_where_it_sits(view):
    geom = tilesheet.geometry(32, view)
    seen = {(cell.row, cell.col) for cell in geom.cells}
    assert seen == {(row, col) for row in range(8) for col in range(8)}
    for cell in geom.cells:
        assert cell.box == (cell.x, cell.y, cell.x + cell.w, cell.y + cell.h)


def test_an_unknown_view_raises_rather_than_defaulting():
    """Defaulting would generate a top-down sheet for a typo'd isometric
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


@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_the_guide_is_the_size_of_the_generation(view):
    geom = tilesheet.geometry(32, view)
    guide = tilesheet.render_guide(geom)
    assert guide.size == geom.source_size
    assert guide.mode == "RGB"


@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_the_guide_is_line_art_in_canny_space(view):
    """White strokes on black, handed to the ControlNet directly. Running the
    detector over it would outline every stroke and draw two lines where the
    guide means one -- ``spritesynth.render_guide``'s reason, verbatim."""
    geom = tilesheet.geometry(32, view)
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


def test_the_isometric_guide_inscribes_a_diamond_the_top_down_one_does_not():
    """An isometric cell stores a diamond with transparent corners, so the
    guide has to say where the top face is. Without it the model fills the
    rectangle and every tile comes back as a square seen from above."""
    iso = np.asarray(tilesheet.render_guide(tilesheet.geometry(32, tilesheet.ISOMETRIC)))
    flat = np.asarray(tilesheet.render_guide(tilesheet.geometry(32, tilesheet.TOP_DOWN)))
    # The centre of the first cell's left edge is on the diamond for iso and
    # empty interior for top-down.
    assert iso[32, 0:4].max() > 0
    assert flat[64, 4:120].max() == 0


def test_the_three_quarter_guide_is_the_top_down_one():
    """Measured, not assumed
    (``docs/measurements/2026-08-21-three-quarter-guide.md``): two interior
    marks were tried, both were obeyed, and both drew a dark stripe rather than
    a change of plane while flattening every cell towards the same tile. The
    subject clause is what carries the view, so 3/4 adds no guide shape -- and
    this is the assertion that says a future mark has to argue against a run."""
    flat = np.asarray(tilesheet.render_guide(tilesheet.geometry(32, tilesheet.TOP_DOWN)))
    three = np.asarray(tilesheet.render_guide(tilesheet.geometry(32, tilesheet.THREE_QUARTER)))
    assert np.array_equal(flat, three)


def test_the_guide_is_the_same_picture_every_time():
    geom = tilesheet.geometry(48, tilesheet.TOP_DOWN)
    first = np.asarray(tilesheet.render_guide(geom))
    second = np.asarray(tilesheet.render_guide(geom))
    assert np.array_equal(first, second)


# -- the subject -------------------------------------------------------------


def test_the_subject_carries_the_detail_clause():
    """A material whose elements are smaller than the art resolution reduces to
    near-noise, so oversized high-contrast elements are asked for explicitly."""
    text = tilesheet.sheet_subject("mossy dungeon floor", tilesheet.TOP_DOWN)
    assert tilesheet.DETAIL_CLAUSE in text
    assert "mossy dungeon floor" in text


#: One word per view that must survive into the subject. A hand-written list
#: rather than a parametrize over ``VIEWS``, because the point is that each view
#: says something *different* -- but it is checked against ``VIEWS`` below, so a
#: view added without an entry here fails rather than going quietly uncovered,
#: which is what the old two-entry version did.
_VIEW_WORDS = {
    tilesheet.TOP_DOWN: "top-down",
    tilesheet.THREE_QUARTER: "3/4",
    tilesheet.ISOMETRIC: "isometric",
}


def test_every_view_has_a_word_of_its_own_to_say():
    assert set(_VIEW_WORDS) == set(tilesheet.VIEWS)
    assert len(set(_VIEW_WORDS.values())) == len(_VIEW_WORDS)


@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_the_subject_names_its_view(view):
    assert _VIEW_WORDS[view] in tilesheet.sheet_subject("stone", view)


def test_the_subject_refuses_a_view_it_has_no_clause_for():
    """The one function that defaulted where ``geometry`` refuses. The fallback
    produced a *plausible* sheet described by the wrong sentence, which is
    invisible by construction."""
    with pytest.raises(ValueError, match="hexagonal"):
        tilesheet.sheet_subject("stone", "hexagonal")


def test_an_empty_subject_leaves_no_dangling_comma():
    """``guidance.compose_prompt`` joins on comma boundaries, so an empty part
    that survived would reach CLIP as a stray separator."""
    text = tilesheet.sheet_subject("   ", tilesheet.TOP_DOWN)
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


@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_a_reduced_sheet_is_exactly_the_output_size(view):
    geom = tilesheet.geometry(32, view)
    width, height = geom.source_size
    out = tilesheet.reduce_sheet(_noise(width, height), geom)
    assert out.shape == (geom.sheet_size[1], geom.sheet_size[0], 4)


@pytest.mark.parametrize("view", tilesheet.VIEWS)
def test_every_tile_is_its_own_cell_reduced(view):
    """The sheet is assembled from the same function the cell test pins, so a
    tile can never be a blend of two cells -- which is what a whole-atlas
    resize would produce wherever the tile size did not divide the cell."""
    geom = tilesheet.geometry(48, view)
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
        view=tilesheet.TOP_DOWN,
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
    # The key on disk is still ``projection`` while the value is a view: every
    # sidecar already written carries that key and ``plotter_tilesets`` reads
    # sheets by it. The vocabulary widened; the key did not move.
    assert doc["projection"] == tilesheet.TOP_DOWN
    assert doc["recipe"] == {"base_model": "sdxl_cfg"}
    # Additive, and absent altogether from a request that measured nothing --
    # so a reader that never heard of the key sees the file it saw before.
    assert "grid" not in doc


def test_the_sidecar_carries_the_lattice_the_generation_was_drawn_on():
    """Measurement only. Nothing reduces on this number -- see
    ``pixel.lattice`` -- and recording it does **not** bump
    ``TILE_SHEET_VERSION``, because a new optional key readers may ignore is
    not a new format."""
    import json

    from PIL import Image

    from warlock.pipelines import pixel

    doc = tilesheet.sheet_sidecar(
        prompt="brick",
        tile_w=32,
        tile_h=32,
        view=tilesheet.TOP_DOWN,
        colors=64,
        palette=["#000000"],
        recipe={},
        created=1.0,
        grid=pixel.lattice(Image.fromarray(_noise(256, 256, seed=3), "RGBA")),
    )
    assert set(doc["grid"]) == {"scale", "residual"}
    assert json.loads(json.dumps(doc)) == doc


def test_the_sidecar_is_plain_json_values():
    """It is written with ``json.dumps`` as the completion marker, so a numpy
    scalar in it would fail the write *after* the sheet had been published."""
    import json

    doc = tilesheet.sheet_sidecar(
        prompt="stone",
        tile_w=np.int64(32),
        tile_h=np.int64(32),
        view=tilesheet.ISOMETRIC,
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


# -- the palette tail, shared by all three tile modes -------------------------
#
# ``quantize_tiles`` is the one place the grid path and the two seamless modes
# meet, so the byte-identity claim is pinned once here on the function and again
# per mode in the two worker files, where it also pins that the *reduction* in
# front of it is still the measured two-stage one.


def _atlas(width: int = 64, height: int = 64, seed: int = 11):
    from PIL import Image

    array = _noise(width, height, seed=seed).copy()
    array[:, :, 3] = 255
    return Image.fromarray(array, "RGBA")


def test_no_palette_and_no_dither_is_quantize_shared_byte_for_byte():
    """The whole compatibility claim. ``resolve_palette`` + ``map_palette`` is
    *not* the identity on this branch -- PIL's median cut publishes its own
    assignment and a nearest-in-Oklab remap of the same table lands a pixel near
    a box boundary on a different entry -- so a default sheet has to go through
    the old call itself, not through the new pair."""
    from warlock.pipelines.pixelsheet import quantize_shared

    atlas = _atlas()
    want, want_palette = quantize_shared(atlas, 16)
    got, palette, source = tilesheet.quantize_tiles(atlas, colors=16)

    assert np.array_equal(np.asarray(got), np.asarray(want))
    assert palette == want_palette
    assert source == "derived"


def test_a_designed_palette_puts_nothing_else_in_the_atlas():
    entries = ((26, 28, 44), (244, 244, 244), (180, 60, 60))
    got, palette, source = tilesheet.quantize_tiles(
        _atlas(), colors=16, entries=entries
    )

    assert source == "designed"
    used = {tuple(int(c) for c in row) for row in np.asarray(got)[:, :, :3].reshape(-1, 3)}
    assert used <= set(entries)
    # And the *reported* palette is what the sheet contains, not what was
    # offered: an entry this atlas never reached is not a colour of this file.
    assert set(palette) <= {f"#{r:02x}{g:02x}{b:02x}" for r, g, b in entries}


def test_the_reported_palette_is_the_colours_present_not_the_entries_offered():
    """A one-colour atlas on a three-colour palette is a one-colour file, and
    ``len(palette)`` is what the row report and the log line both call
    "colours"."""
    from PIL import Image

    flat = Image.new("RGBA", (16, 16), (250, 250, 250, 255))
    _got, palette, _source = tilesheet.quantize_tiles(
        flat, colors=16, entries=((0, 0, 0), (250, 250, 250), (120, 10, 10))
    )
    assert palette == ["#fafafa"]


def test_dither_with_no_palette_still_derives_one_and_says_so():
    """The two settings are independent: a dithered sheet that named no palette
    median-cuts its own table and then dithers against it."""
    atlas = _atlas()
    plain, _p, _s = tilesheet.quantize_tiles(atlas, colors=8)
    dithered, _palette, source = tilesheet.quantize_tiles(atlas, colors=8, dither=True)

    assert source == "derived"
    # Dither has to actually reach the mapping. It is an offset added before the
    # nearest search, so on a 8-entry table over noise it must move pixels.
    assert not np.array_equal(np.asarray(dithered), np.asarray(plain))


def test_dither_changes_the_bytes_on_a_designed_palette_too():
    entries = ((0, 0, 0), (255, 255, 255))
    atlas = _atlas()
    plain, _p, _s = tilesheet.quantize_tiles(atlas, colors=8, entries=entries)
    dithered, _q, _t = tilesheet.quantize_tiles(
        atlas, colors=8, entries=entries, dither=True
    )
    assert not np.array_equal(np.asarray(dithered), np.asarray(plain))


def test_an_atlas_with_no_opaque_pixels_is_refused_on_either_branch():
    """Asked on the input rather than left to ``quantize_shared``, so which
    answer an empty atlas gets does not depend on whether a palette was named --
    a setting that has nothing to do with the question."""
    from PIL import Image

    empty = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="cells are empty"):
        tilesheet.quantize_tiles(empty, colors=8)
    with pytest.raises(ValueError, match="cells are empty"):
        tilesheet.quantize_tiles(empty, colors=8, entries=((1, 2, 3),))


def test_the_transparent_padding_of_a_partial_row_stays_transparent():
    """``tileatlas.assemble`` zero-fills the slots a partial last row leaves, and
    a mapped atlas must not turn them opaque -- ``map_palette`` carries alpha
    through untouched, which is what makes the snap ``quantize_shared`` does
    unnecessary here."""
    from PIL import Image

    array = _noise(32, 32, seed=5).copy()
    array[:, :, 3] = 255
    array[16:, :, 3] = 0
    padded = Image.fromarray(array, "RGBA")
    got, _palette, _source = tilesheet.quantize_tiles(
        padded, colors=8, entries=((10, 20, 30), (200, 210, 220))
    )
    assert np.array_equal(np.asarray(got)[:, :, 3], array[:, :, 3])


# -- what the recipe records about the palette --------------------------------


def test_a_sheet_that_asked_for_neither_records_neither():
    """The sidecar rule: a reader that never heard of these keys sees exactly
    the file it saw before, and the bytes beside it are unchanged too."""
    assert (
        tilesheet.palette_record(name="", entries=(), source="derived", dither=False)
        == {}
    )


def test_dither_alone_records_the_source_and_the_flag_and_no_file():
    record = tilesheet.palette_record(
        name="", entries=(), source="derived", dither=True
    )
    assert record == {"palette_source": "derived", "dither": True}


def test_a_named_palette_records_its_file_and_a_digest_of_its_colours():
    from warlock.pipelines import pixel

    entries = ((26, 28, 44), (244, 244, 244))
    record = tilesheet.palette_record(
        name="duo", entries=entries, source="designed", dither=False
    )
    assert record == {
        "palette_source": "designed",
        "palette_file": "duo",
        "palette_hash": pixel.palette_digest(entries),
    }
    # Of the *colours*, not of the file: reformatting a palette or converting it
    # from .hex to .gpl must not re-derive anything, and one changed channel
    # must.
    assert (
        tilesheet.palette_record(
            name="duo",
            entries=((26, 28, 44), (244, 244, 245)),
            source="designed",
            dither=False,
        )["palette_hash"]
        != record["palette_hash"]
    )


def test_the_palette_record_is_plain_json_values():
    import json

    record = tilesheet.palette_record(
        name="duo", entries=((1, 2, 3),), source="designed", dither=True
    )
    assert json.loads(json.dumps(record)) == record

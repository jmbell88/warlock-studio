"""The tile atlas's pure half: the two layouts, the words, and the reduction.

Three claims carry this module.

The first is the one ``docs/measurements/2026-08-18-tile-sheet-grid.md`` argues
for: **variety is a property of the request.** N materials are N generations
laid out in the order the user typed them, so the layout is a list rather than a
packing and the tests below pin reading order rather than aspect ratio.

The second is that **a material is a torus and a grid cell is not**.
``tilesheet._box_reduce`` lets blocks differ in size by one when the target does
not divide the source, and calls that invisible -- which is true between two
adjacent cells of a sheet and false at the wrap seam of a seamless material,
where the first and last block are neighbours. So ``reduce_material`` refuses
what ``reduce_cell`` accepts, and 48 is the size that separates them.

The third is that **a generated terrain set cannot be recognised by looking at
it**. ``roles.infer_roles`` needs transparency or one dominant ring colour and a
composited set of two opaque textures has neither, so the sidecar is not a
convenience: it is the only thing that can say what the atlas is.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from warlock.pipelines import tileatlas, tilemask, tilesheet
from warlock.service.validation import MAX_SEED


def _blocks(width: int, height: int, block: int) -> np.ndarray:
    """Flat blocks of one colour each, so a partitioning reduction is exact."""
    rng = np.random.default_rng(3)
    tall, wide = height // block, width // block
    small = rng.integers(0, 256, size=(tall, wide, 4), dtype=np.uint8)
    return np.repeat(np.repeat(small, block, axis=0), block, axis=1)


def _flat(size: int, colour: tuple[int, int, int, int]) -> np.ndarray:
    tile = np.zeros((size, size, 4), dtype=np.uint8)
    tile[:, :] = colour
    return tile


def _bind(geom: tileatlas.AtlasGeometry, prompts: list[str]) -> tuple:
    """The cells with their words and seeds attached, as the queue binds them."""
    seeds = tileatlas.material_seeds(7, len(geom.cells))
    return tuple(
        dataclasses.replace(cell, prompt=prompt, variant=0, seed=seed)
        for cell, prompt, seed in zip(geom.cells, prompts, seeds, strict=True)
    )


# -- the layouts -------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "columns", "rows"),
    [
        (1, 1, 1),
        (2, 2, 1),
        (7, 7, 1),
        (8, 8, 1),
        (9, 8, 2),
        (12, 8, 2),
        (16, 8, 2),
        (64, 8, 8),
    ],
)
def test_a_materials_sheet_is_a_list_not_a_square(count, columns, rows):
    """Eight across and ceil down, never "roughly square". The order is the
    order the user typed, expanded in place, and a packer would destroy it to
    save texture width."""
    geom = tileatlas.material_geometry(32, "top_down", count)
    assert (geom.columns, geom.rows) == (columns, rows)
    assert geom.tiles == count
    assert geom.layout == "grid"
    assert geom.atlas_size == (columns * 32, rows * 32)


def test_every_materials_count_lays_out_in_reading_order():
    """Cell ``i`` at ``(i // columns, i % columns)`` for every count this
    module accepts -- the one arithmetic ``assemble`` and the sidecar share."""
    for count in range(1, tileatlas.MAX_CELLS + 1):
        geom = tileatlas.material_geometry(16, "top_down", count)
        for cell in geom.cells:
            assert (cell.row, cell.col) == divmod(cell.index, geom.columns)


def test_a_terrain_atlas_is_forty_seven_columns_by_one_row():
    """Not this module's choice: ``Tileset.__post_init__`` refuses a terrain set
    that is not ``blob.TILE_COUNT`` columns wide and ``local_for`` indexes a
    case by its column."""
    geom = tileatlas.terrain_geometry(32, "top_down")
    assert (geom.columns, geom.rows) == (tilemask.TILE_COUNT, 1)
    assert geom.tiles == 47
    assert geom.layout == "blob47"
    assert geom.atlas_size == (47 * 32, 32)


def test_the_atlas_width_agrees_with_tilemask():
    """One row of this layout *is* a terrain set's atlas, so the two modules'
    idea of how wide that is has to be one fact."""
    geom = tileatlas.terrain_geometry(16, "top_down")
    assert geom.columns == tilemask.TILE_COUNT == 47
    assert geom.atlas_size[0] == tilemask.TILE_COUNT * 16


def test_a_seamless_material_has_exactly_one_view():
    """``prompt.TILE_TEMPLATE`` hardcodes "flat top-down orthographic view", so
    there is one view a seamless material can be and both modes take only it.
    The tile is square because that view's lattice is."""
    assert tileatlas.VIEWS == ("top_down",)
    for view in tileatlas.VIEWS:
        geom = tileatlas.material_geometry(64, view, 4)
        assert (geom.tile_w, geom.tile_h) == (64, 64)
        assert tileatlas.terrain_geometry(64, view).tile_h == 64


def test_a_sidecar_view_written_before_the_vocabulary_widened_still_reads():
    """``tilesheet.normalize_view``'s rule, taken through this module's door."""
    assert tileatlas.material_geometry(32, "orthogonal", 3).view == "top_down"
    assert tileatlas.terrain_geometry(32, "orthogonal").view == "top_down"


# -- the refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: tileatlas.material_geometry(32, "isometric", 4),
        lambda: tileatlas.terrain_geometry(32, "isometric"),
    ],
)
def test_isometric_is_refused_by_name_in_both_modes(call):
    """``circular_padding`` wraps a rectangle and a 2:1 diamond is not one, so
    an isometric seamless tile is not a thing that exists. Named rather than
    merely absent, because it is the one view a caller has a reason to ask
    for."""
    with pytest.raises(ValueError, match="isometric"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: tileatlas.material_geometry(32, "three_quarter", 4),
        lambda: tileatlas.terrain_geometry(32, "three_quarter"),
    ],
)
def test_three_quarter_is_refused_by_name_in_both_modes(call):
    """Not merely redundant -- incoherent. ``tilesheet._VIEW_CLAUSE``'s 3/4 asks
    for "a shallow visible front face", and a front face does not tile
    vertically: the row below would occlude it. Nothing is lost, because the
    grid path still offers 3/4 and its guide is measured
    (``docs/measurements/2026-08-21-three-quarter-guide.md``)."""
    with pytest.raises(ValueError, match="front face"):
        call()
    # The clause this is refusing really does say it, so the argument is not
    # about a string this test made up.
    assert "front face" in tilesheet._VIEW_CLAUSE[tilesheet.THREE_QUARTER]
    # And the grid path is untouched: it still lays 3/4 out.
    assert tilesheet.geometry(32, "three_quarter").view == "three_quarter"


@pytest.mark.parametrize(
    "call",
    [
        lambda: tileatlas.material_geometry(32, "hexagonal", 4),
        lambda: tileatlas.terrain_geometry(32, "hexagonal"),
    ],
)
def test_an_unknown_view_is_refused_with_its_own_spelling(call):
    with pytest.raises(ValueError, match="hexagonal"):
        call()


def test_a_materials_request_of_none_is_refused():
    with pytest.raises(ValueError, match="at least one cell"):
        tileatlas.material_geometry(32, "top_down", 0)


def test_more_cells_than_the_ceiling_is_refused_with_the_number():
    """A ceiling on generations rather than on pixels: each cell is its own full
    sample. ``count`` is cells, so sixteen prompt lines at four variants is the
    largest legal request and sixty-five is not one."""
    assert tileatlas.material_geometry(32, "top_down", 64).tiles == 64
    with pytest.raises(ValueError, match="65"):
        tileatlas.material_geometry(32, "top_down", tileatlas.MAX_CELLS + 1)


def test_the_cell_ceiling_is_the_doors_lines_times_its_variants():
    """The two limits live at the two places that can see the thing they are
    about: ``asset_workflows.collection_cells`` caps 1-16 lines by 1-4 variants,
    and this module caps their product."""
    assert tileatlas.MAX_CELLS == tileatlas.MAX_MATERIALS * 4 == 64


@pytest.mark.parametrize(
    "call",
    [
        lambda: tileatlas.material_geometry(48, "top_down", 4),
        lambda: tileatlas.terrain_geometry(48, "top_down"),
    ],
)
def test_a_tile_that_does_not_divide_the_material_is_refused_at_the_geometry(call):
    """48 is in ``tilesheet.TILE_SIZES`` and does not divide 1024. Refused here
    as well as in the reduction, so a form is told before N generations run."""
    with pytest.raises(ValueError, match="48"):
        call()


def test_a_tile_larger_than_the_material_is_refused():
    with pytest.raises(ValueError, match="2048"):
        tileatlas.material_geometry(2048, "top_down", 2)


def test_a_zero_sized_tile_is_refused():
    with pytest.raises(ValueError, match="at least one pixel"):
        tileatlas.terrain_geometry(0, "top_down")


# -- the words ---------------------------------------------------------------


def test_the_style_clause_is_on_every_material_subject():
    """N separate generations have to read as one sheet, and the only thing
    that can make them is a clause every one of them carries."""
    prompts = ["mossy cobblestone", "dry cracked earth", "still dark water"]
    for index, prompt in enumerate(prompts):
        subject = tileatlas.material_subject(prompt, index=index, total=len(prompts))
        assert subject.startswith(prompt)
        assert subject.endswith(tileatlas.MATERIAL_STYLE_CLAUSE)


def test_a_material_subject_carries_no_view_clause():
    """A seamless material has no view: it is flat top-down by construction and
    ``prompt.TILE_TEMPLATE`` already says so. A second framing clause would
    either repeat it or contradict it."""
    subject = tileatlas.material_subject("grass", index=0, total=1)
    assert subject == f"grass, {tileatlas.MATERIAL_STYLE_CLAUSE}"
    for clause in tilesheet._VIEW_CLAUSE.values():
        assert clause not in subject
    assert "isometric" not in subject
    assert "3/4" not in subject


def test_a_material_subject_never_names_its_own_index():
    """"material 3 of 8" is a phrase SDXL draws, which is what every template's
    "no text, no watermark" exists to prevent. The index is for the refusal."""
    subject = tileatlas.material_subject("gravel", index=2, total=8)
    assert "3" not in subject and "8" not in subject


def test_an_empty_material_names_which_one_it_was():
    """"a material has no words" is useless about a request that named
    twelve."""
    with pytest.raises(ValueError, match="material 3 of 12"):
        tileatlas.material_subject("   ", index=2, total=12)


def test_a_material_index_outside_its_request_is_refused():
    with pytest.raises(ValueError, match="outside a request of 4"):
        tileatlas.material_subject("sand", index=4, total=4)


def test_terrain_subjects_are_two_ordinary_materials():
    """The model draws two surfaces and never sees an edge -- that is the whole
    claim of the composited path."""
    inner, outer = tileatlas.terrain_subjects("green grass", "deep water")
    assert inner == f"green grass, {tileatlas.MATERIAL_STYLE_CLAUSE}"
    assert outer == f"deep water, {tileatlas.MATERIAL_STYLE_CLAUSE}"


def test_the_boundary_clause_is_context_carried_by_both_halves():
    """Two independent samples that have to share a world and a palette. It is
    not an instruction to draw a seam: the seam is a distance field."""
    inner, outer = tileatlas.terrain_subjects("grass", "water", "a temperate coastline")
    for subject in (inner, outer):
        assert "a temperate coastline" in subject
        assert subject.endswith(tileatlas.MATERIAL_STYLE_CLAUSE)


def test_an_empty_boundary_leaves_no_stray_comma():
    inner, _ = tileatlas.terrain_subjects("grass", "water", "  ")
    assert inner == f"grass, {tileatlas.MATERIAL_STYLE_CLAUSE}"


@pytest.mark.parametrize(("inner", "outer"), [("", "water"), ("grass", "  ")])
def test_a_terrain_half_with_no_words_is_refused_by_name(inner, outer):
    with pytest.raises(ValueError, match="inner|outer"):
        tileatlas.terrain_subjects(inner, outer)


# -- the seeds ---------------------------------------------------------------


def test_the_seed_ceiling_is_the_one_the_rest_of_the_tree_uses():
    """Restated rather than imported -- a pipeline may not import the service
    layer -- so a test that is not layered pins every copy together."""
    assert tileatlas.MAX_SEED == tilesheet.MAX_SEED == MAX_SEED == 2**31 - 1


@pytest.mark.parametrize("count", [1, 2, 8, 16, tileatlas.MAX_CELLS])
def test_material_seeds_are_distinct_and_in_range(count):
    seeds = tileatlas.material_seeds(12345, count)
    assert len(seeds) == count == len(set(seeds))
    assert all(0 <= seed <= tileatlas.MAX_SEED for seed in seeds)


def test_material_seeds_are_derived_so_one_material_can_be_rerolled():
    """``seed + i``, and nothing else: material 3 is reproducible from the pair
    ``(seed, 3)`` without replaying anybody's draw."""
    assert tileatlas.material_seeds(100, 4) == (100, 101, 102, 103)


def test_the_seed_bound_wraps_rather_than_clamps():
    """Clamping is what turns a bound into a collision: ``MAX_SEED`` and
    ``MAX_SEED + 1`` would clamp to the same value and two cells of one sheet
    would silently be the same picture."""
    seeds = tileatlas.material_seeds(tileatlas.MAX_SEED - 1, 4)
    assert seeds == (tileatlas.MAX_SEED - 1, tileatlas.MAX_SEED, 0, 1)
    assert len(set(seeds)) == 4


def test_a_seed_outside_the_range_is_refused_with_the_ceiling():
    with pytest.raises(ValueError, match=str(tileatlas.MAX_SEED)):
        tileatlas.material_seeds(tileatlas.MAX_SEED + 1, 2)


# -- the reduction -----------------------------------------------------------


@pytest.mark.parametrize("tile", [16, 32, 64])
def test_reduce_material_accepts_the_sizes_that_divide(tile):
    """1024 is a power of two, so every tile size that divides it divides it
    cleanly all the way down through the prefilter as well."""
    material = _blocks(tileatlas.MATERIAL_PX, tileatlas.MATERIAL_PX, 64)
    out = tileatlas.reduce_material(material, tile, tile)
    assert out.shape == (tile, tile, 4)
    assert out.dtype == np.uint8


def test_reduce_material_is_tilesheets_measured_reducer():
    """Delegation, pinned: the two-stage reducer of
    ``docs/measurements/2026-08-17-ground-reduction.md`` is not reimplemented
    here, only fenced."""
    material = _blocks(256, 256, 8)
    assert np.array_equal(
        tileatlas.reduce_material(material, 32, 32),
        tilesheet.reduce_cell(material, 32, 32),
    )


def test_reduce_material_refuses_forty_eight_by_name():
    """The size that separates a grid cell from a torus. ``reduce_cell`` takes
    it happily; a seamless material cannot."""
    material = _blocks(tileatlas.MATERIAL_PX, tileatlas.MATERIAL_PX, 64)
    with pytest.raises(ValueError, match="48"):
        tileatlas.reduce_material(material, 48, 48)
    # And the thing that is being refused really is accepted next door, which is
    # the whole reason this function exists rather than the call being direct.
    assert tilesheet.reduce_cell(material, 48, 48).shape == (48, 48, 4)


def test_forty_eight_puts_blocks_of_two_different_widths_on_the_wrap_seam():
    """``tilesheet._box_reduce``'s own ``starts`` formula, restated: at
    1024 -> 48 the prefilter runs at 192 and the blocks are 5 or 6 pixels wide.
    Invisible between two cells of a sheet; a one-pixel step at the seam of a
    torus."""
    groups = 48 * 4  # ``m`` caps at 4, so the prefilter target is 192.
    starts = [-(-g * 1024 // groups) for g in range(groups)]
    widths = {b - a for a, b in zip(starts, starts[1:] + [1024], strict=True)}
    assert widths == {5, 6}
    # The divisor case has one width, which is what the refusal is protecting.
    starts = [-(-g * 1024 // (32 * 4)) for g in range(32 * 4)]
    assert {b - a for a, b in zip(starts, starts[1:] + [1024], strict=True)} == {8}


def test_reduce_material_refuses_a_prefilter_that_does_not_partition_either():
    """96 divides by 16 six times, and six is not a multiple of the prefilter's
    step of four -- so the first stage is uneven even though the factor is
    whole. Free at 1024; not free for a caller who brings their own size."""
    material = _blocks(96, 96, 16)
    with pytest.raises(ValueError, match="prefilter"):
        tileatlas.reduce_material(material, 16, 16)


def test_reduce_material_refuses_an_upscale():
    with pytest.raises(ValueError, match="generate it larger"):
        tileatlas.reduce_material(_blocks(64, 64, 8), 128, 128)


# -- the assembly ------------------------------------------------------------


def test_assemble_places_cell_i_at_row_i_over_columns():
    geom = tileatlas.material_geometry(8, "top_down", 11)
    tiles = [_flat(8, (index * 10, 0, 0, 255)) for index in range(11)]
    atlas = tileatlas.assemble(tiles, geom)
    assert atlas.shape == (2 * 8, 8 * 8, 4)
    for index in range(11):
        row, col = divmod(index, geom.columns)
        block = atlas[row * 8 : row * 8 + 8, col * 8 : col * 8 + 8]
        assert np.array_equal(block, tiles[index])


def test_assemble_returns_exactly_columns_by_rows_tiles():
    for count in (1, 5, 8, 9, 16):
        geom = tileatlas.material_geometry(16, "top_down", count)
        tiles = [_flat(16, (7, 7, 7, 255))] * count
        atlas = tileatlas.assemble(tiles, geom)
        assert atlas.shape == (geom.rows * 16, geom.columns * 16, 4)
        assert atlas.shape[:2][::-1] == geom.atlas_size


def test_a_short_last_row_is_transparent_rather_than_repeated():
    """Nine materials at eight across leaves seven empty cells. They are holes,
    not a repeat of cell zero -- a map painted with a duplicate would be painted
    with a tile nobody asked for."""
    geom = tileatlas.material_geometry(4, "top_down", 9)
    atlas = tileatlas.assemble([_flat(4, (9, 9, 9, 255))] * 9, geom)
    assert atlas[4:, 4:].max() == 0


def test_assemble_promotes_an_opaque_rgb_material():
    """``tileset.frozen_rgba`` needs four channels and SDXL returns three; the
    only answer an opaque material has is alpha 255."""
    geom = tileatlas.material_geometry(4, "top_down", 2)
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    atlas = tileatlas.assemble([rgb, rgb], geom)
    assert atlas.shape == (4, 8, 4)
    assert (atlas[:, :, 3] == 255).all()


def test_assemble_of_a_blob_atlas_columns_is_the_blob_atlas():
    """The layout claim, pinned byte for byte against the module that owns the
    47 cases rather than trusting that two arithmetics agree."""
    inner, outer = _flat(16, (20, 160, 40, 255)), _flat(16, (10, 40, 200, 255))
    reference = tilemask.blob_atlas(inner, outer, 16, seed=5)
    columns = [
        reference[:, index * 16 : (index + 1) * 16] for index in range(tilemask.TILE_COUNT)
    ]
    atlas = tileatlas.assemble(columns, tileatlas.terrain_geometry(16, "top_down"))
    assert np.array_equal(atlas, reference)


def test_assemble_refuses_a_tile_of_the_wrong_size_by_number():
    geom = tileatlas.material_geometry(16, "top_down", 2)
    with pytest.raises(ValueError, match="8x8"):
        tileatlas.assemble([_flat(16, (1, 1, 1, 255)), _flat(8, (1, 1, 1, 255))], geom)


def test_assemble_refuses_the_wrong_number_of_tiles():
    geom = tileatlas.material_geometry(16, "top_down", 3)
    with pytest.raises(ValueError, match="3 cells and 2 tiles"):
        tileatlas.assemble([_flat(16, (1, 1, 1, 255))] * 2, geom)


# -- the record --------------------------------------------------------------


def _mask() -> dict:
    return {"seed": 5, "inset": 6.0, "amplitude": 2.88, "feather": 0.96}


def _terrain() -> dict:
    return {"name": "grass", "fill": [20, 160, 40, 255], "outline": [10, 80, 20, 255]}


def test_a_materials_sidecar_records_the_grid_and_every_cell():
    geom = tileatlas.material_geometry(32, "top_down", 3)
    cells = _bind(geom, ["moss", "gravel", "water"])
    card = tileatlas.atlas_sidecar(geom, created=1.5, materials=cells)
    assert card["version"] == tileatlas.TILE_ATLAS_VERSION
    assert card["mode"] == "materials"
    assert card["layout"] == "grid"
    assert card["view"] == "top_down"
    assert (card["columns"], card["rows"], card["tiles"]) == (3, 1, 3)
    assert (card["tile_w"], card["tile_h"]) == (32, 32)
    assert [entry["prompt"] for entry in card["materials"]] == ["moss", "gravel", "water"]
    assert [entry["seed"] for entry in card["materials"]] == [7, 8, 9]
    assert card["terrains"] == []
    assert card["mask"] is None


def test_two_variants_of_one_line_are_two_cells_told_apart_by_variant():
    """A cell is a ``(prompt line, variant)`` pair, so two cells of one line
    carry identical prompts and nothing but ``variant`` and ``seed``
    distinguishes them in the record. ``collection_cells`` numbers variants from
    one."""
    geom = tileatlas.material_geometry(16, "top_down", 4)
    pairs = [("moss", 1), ("moss", 2), ("silt", 1), ("silt", 2)]
    seeds = tileatlas.material_seeds(40, 4)
    cells = tuple(
        dataclasses.replace(cell, prompt=prompt, variant=variant, seed=seed)
        for cell, (prompt, variant), seed in zip(geom.cells, pairs, seeds, strict=True)
    )
    card = tileatlas.atlas_sidecar(geom, created=0.0, materials=cells)
    assert [(e["prompt"], e["variant"]) for e in card["materials"]] == pairs
    assert len({e["seed"] for e in card["materials"]}) == 4


def test_a_terrain_sidecar_says_what_it_is_rather_than_leaving_it_to_be_seen():
    """``roles.infer_roles`` returns ``None`` on a perfectly formed generated
    set -- two opaque textures have neither transparency nor a dominant ring
    colour. ``SheetMismatch``'s rule: what catches it is not a measurement but a
    record."""
    geom = tileatlas.terrain_geometry(32, "top_down")
    cells = _bind(geom, ["grass"] * tilemask.TILE_COUNT)
    card = tileatlas.atlas_sidecar(
        geom, created=2.0, materials=cells, terrains=[_terrain()], mask=_mask()
    )
    assert card["layout"] == "blob47"
    assert card["columns"] == tilemask.TILE_COUNT
    assert card["terrains"] == [_terrain()]
    assert card["mask"] == {
        "version": tilemask.MASK_VERSION,
        "seed": 5,
        "inset": 6.0,
        "amplitude": 2.88,
        "feather": 0.96,
    }


def test_the_mask_version_is_stamped_not_taken_from_the_caller():
    """It says which field implementation drew these pixels, so a caller that
    supplied it could claim a version it did not run."""
    geom = tileatlas.terrain_geometry(16, "top_down")
    cells = _bind(geom, ["sand"] * tilemask.TILE_COUNT)
    card = tileatlas.atlas_sidecar(
        geom,
        created=0.0,
        materials=cells,
        terrains=[_terrain()],
        mask={**_mask(), "version": 999},
    )
    assert card["mask"]["version"] == tilemask.MASK_VERSION != 999


def test_every_sidecar_value_is_a_plain_builtin():
    """Written with ``json.dumps`` *after* the atlas is on disk: a numpy scalar
    that survived would fail the write with the artifact already published and
    no marker to say so."""
    geom = tileatlas.terrain_geometry(32, "top_down")
    seeds = tileatlas.material_seeds(11, tilemask.TILE_COUNT)
    cells = tuple(
        dataclasses.replace(cell, prompt="grass", variant=0, seed=int(np.int64(seed)))
        for cell, seed in zip(geom.cells, seeds, strict=True)
    )
    card = tileatlas.atlas_sidecar(
        geom,
        created=float(np.float32(3.0)),
        materials=cells,
        terrains=[{"name": "grass", "fill": np.array([1, 2, 3, 4]), "outline": (5, 6, 7, 8)}],
        mask={"seed": np.int64(4), "inset": np.float32(6.0), "amplitude": 2.0, "feather": 1.0},
        recipe={"base": "sdxl_cfg"},
    )
    assert json.loads(json.dumps(card)) == card


def test_a_sidecar_refuses_a_cell_the_queue_never_bound():
    """The geometry leaves the words empty on purpose, and this is the failure
    that shape invites: an atlas whose record says nothing about what is in
    it."""
    geom = tileatlas.material_geometry(32, "top_down", 2)
    with pytest.raises(ValueError, match="cell 0 has no prompt"):
        tileatlas.atlas_sidecar(geom, created=0.0, materials=geom.cells)


def test_a_terrain_sidecar_without_a_mask_is_refused():
    geom = tileatlas.terrain_geometry(16, "top_down")
    cells = _bind(geom, ["grass"] * tilemask.TILE_COUNT)
    with pytest.raises(ValueError, match="mask field"):
        tileatlas.atlas_sidecar(geom, created=0.0, materials=cells, terrains=[_terrain()])


def test_a_terrain_sidecar_that_declares_no_terrain_is_refused():
    """``Tileset.__post_init__`` needs one terrain per row: a set that declares
    none is an ordinary atlas wearing a terrain layout, every gid valid and
    every role wrong."""
    geom = tileatlas.terrain_geometry(16, "top_down")
    cells = _bind(geom, ["grass"] * tilemask.TILE_COUNT)
    with pytest.raises(ValueError, match="terrain"):
        tileatlas.atlas_sidecar(geom, created=0.0, materials=cells, mask=_mask())


def test_a_materials_sidecar_refuses_terrains_and_a_mask():
    geom = tileatlas.material_geometry(16, "top_down", 2)
    cells = _bind(geom, ["moss", "stone"])
    with pytest.raises(ValueError, match="declares no terrains"):
        tileatlas.atlas_sidecar(geom, created=0.0, materials=cells, terrains=[_terrain()])
    with pytest.raises(ValueError, match="not composited"):
        tileatlas.atlas_sidecar(geom, created=0.0, materials=cells, mask=_mask())


def test_a_terrain_colour_that_is_not_four_channels_is_refused():
    geom = tileatlas.terrain_geometry(16, "top_down")
    cells = _bind(geom, ["grass"] * tilemask.TILE_COUNT)
    with pytest.raises(ValueError, match="four channels"):
        tileatlas.atlas_sidecar(
            geom,
            created=0.0,
            materials=cells,
            terrains=[{"name": "grass", "fill": [1, 2, 3], "outline": [1, 2, 3, 4]}],
            mask=_mask(),
        )


def test_the_terrains_list_is_ordered_because_position_is_precedence():
    """``TerrainSpec``'s own rule: a terrain's position is how a cell with three
    terrains around it picks one picture."""
    geom = tileatlas.terrain_geometry(16, "top_down")
    cells = _bind(geom, ["grass"] * tilemask.TILE_COUNT)
    card = tileatlas.atlas_sidecar(
        geom, created=0.0, materials=cells, terrains=[_terrain()], mask=_mask()
    )
    assert isinstance(card["terrains"], list)
    assert card["terrains"][0]["name"] == "grass"


# -- the module's own promises -----------------------------------------------


def test_the_modes_table_and_the_layouts_table_cannot_disagree():
    assert set(tileatlas.MODES) == set(tileatlas._LAYOUTS)
    assert tileatlas.MODES == ("materials", "terrain")


def test_this_module_imports_no_studio_and_no_service():
    """A pipeline runs inside worker and Blender processes where ``studio`` is
    not importable at all -- ``tilemask``'s rule, and
    ``tests/tilegrid/test_tilegrid_imports.py``'s."""
    source = (
        __import__("pathlib").Path(tileatlas.__file__).read_text(encoding="utf-8")
    )
    for banned in ("import torch", "from warlock.studio", "from ..studio", "from ..service"):
        assert banned not in source

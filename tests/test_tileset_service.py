"""The seamless tile modes' door: every refusal, and what a good request writes.

``tests/test_tilesheet_service.py`` is the same door's *grid* half and stays
that way -- the two files split on the thing that actually differs, which is
what a mode loads and what it lays out, not on which function is called.

Three things this file exists to hold.

**A refusal names the control and the number.** A door whose whole job is to
turn a bad form into a sentence has to produce one a user can act on, so every
refusal below is asserted on its ``field`` *and* on the offending value being in
the text -- a message that says "variants must be 1-4" and does not say what was
asked for is a message the user reads twice.

**The seamless modes must not require a ControlNet.** They never open one. The
row lists are asserted to differ per mode, and there is a live test that deletes
the canny weights and builds a materials sheet on the host that is left -- which
is the case the old single-tuple ``_TILE_SHEET_REQUIRED`` got wrong.

**The restatements are pinned.** ``service/`` may not import ``studio/`` and
holds its own copies of the tile sizes, the views and the modes; the pins at the
bottom are what stop a copy from drifting silently, exactly as the sibling file
pins its two against ``pipelines.tilesheet``.
"""

from __future__ import annotations

import pytest

from warlock import asset_workflows, generation, models
from warlock.pipelines import tileatlas, tilemask
from warlock.service import tilesheets
from warlock.service.errors import Invalid

MATERIALS = ("mossy cobblestone", "cracked dry mud", "still dark water")


def _materials(svc, **overrides):
    kwargs = {
        "prompt": "a temperate ruin, muted palette",
        "tile_size": 32,
        "view": "top_down",
        "mode": tilesheets.MODE_MATERIALS,
        "prompt_items": MATERIALS,
    }
    kwargs.update(overrides)
    return tilesheets.create_tile_sheet(svc, **kwargs)


def _terrain(svc, **overrides):
    kwargs = {
        "prompt": "a temperate coastline",
        "tile_size": 32,
        "view": "top_down",
        "mode": tilesheets.MODE_TERRAIN,
        "inner_terrain": "short green grass",
        "outer_terrain": "shallow blue water",
    }
    kwargs.update(overrides)
    return tilesheets.create_tile_sheet(svc, **kwargs)


def _sheet(svc, made):
    return svc.store.get(made["id"])["params"]["sheet"]


# -- the mode ----------------------------------------------------------------


def test_an_unstated_mode_is_the_materials_one(svc):
    """The default is the mode that produces N different tiles, not the one the
    measurement says produces one tile sixty-four times."""
    made = tilesheets.create_tile_sheet(
        svc, prompt="a temperate ruin", prompt_items=MATERIALS
    )
    assert made["mode"] == tilesheets.MODE_MATERIALS
    assert _sheet(svc, made)["mode"] == tilesheets.MODE_MATERIALS


@pytest.mark.parametrize("mode", ["wang", "path", "", None, "Materials"])
def test_a_mode_off_the_menu_is_refused_and_says_what_it_was(svc, mode):
    with pytest.raises(Invalid) as excinfo:
        _materials(svc, mode=mode)
    assert excinfo.value.field == "mode"
    assert repr(mode) in str(excinfo.value) or str(mode) in str(excinfo.value)


def test_a_new_grid_request_is_refused_and_named_its_two_replacements(svc):
    """The grid mode was measured, not merely disliked: every cell of its guide
    is identical, so there is no per-cell signal for variety."""
    with pytest.raises(Invalid) as excinfo:
        tilesheets.create_tile_sheet(
            svc, prompt="a mossy dungeon", mode=tilesheets.MODE_GRID
        )
    assert excinfo.value.field == "mode"
    message = str(excinfo.value)
    assert "materials" in message
    assert "terrain" in message
    assert "2026-08-18-tile-sheet-grid" in message


def test_the_grid_still_builds_for_anything_that_asks_for_it_by_name(svc):
    """Rerunning a sheet made last week is not an error, so the refusal above is
    about a *new* request and nothing else."""
    made = tilesheets.create_tile_sheet(
        svc, prompt="a mossy dungeon", mode=tilesheets.MODE_GRID, allow_grid=True
    )
    assert made["tiles"] == 64
    assert svc.store.get(made["id"])["params"]["control"] == "canny"


# -- what a materials request writes -----------------------------------------


def test_the_materials_block_is_the_list_that_was_typed_in_order(svc):
    block = _sheet(svc, _materials(svc, seed=7))
    assert block["version"] == 3
    assert block["mode"] == "materials"
    assert block["layout"] == "grid"
    assert block["projection"] == "top_down"
    assert block["tile_w"] == 32
    assert block["tile_h"] == 32
    assert block["columns"] == 3
    assert block["rows"] == 1
    assert [cell["prompt"] for cell in block["materials"]] == list(MATERIALS)
    assert [cell["index"] for cell in block["materials"]] == [0, 1, 2]
    assert [cell["variant"] for cell in block["materials"]] == [1, 1, 1]
    # Derived from the one request seed rather than drawn, so re-running the
    # request reproduces every cell and cell i is reproducible on its own.
    assert [cell["seed"] for cell in block["materials"]] == list(
        tileatlas.material_seeds(7, 3)
    )


def test_the_materials_block_declares_no_terrains_and_no_mask(svc):
    """A materials sheet is a plain grid: its cells have no blob roles to name
    and nothing composited them, which is exactly what
    ``tileatlas.atlas_sidecar`` refuses a materials atlas for claiming."""
    block = _sheet(svc, _materials(svc))
    assert block["terrains"] == []
    assert block["mask"] is None
    assert block["boundary"] == ""


def test_variants_expand_in_place_and_are_recorded(svc):
    block = _sheet(svc, _materials(svc, variants=2))
    assert [(c["prompt"], c["variant"]) for c in block["materials"]] == [
        (MATERIALS[0], 1),
        (MATERIALS[0], 2),
        (MATERIALS[1], 1),
        (MATERIALS[1], 2),
        (MATERIALS[2], 1),
        (MATERIALS[2], 2),
    ]
    assert block["variants"] == 2
    assert len({c["seed"] for c in block["materials"]}) == 6


def test_the_full_request_is_sixteen_lines_by_four_draws(svc):
    """The two ceilings meet exactly at :data:`tileatlas.MAX_CELLS`, so the
    largest legal request is the one that lands on it rather than one the
    product guard has to catch."""
    lines = tuple(f"material {n}" for n in range(tilesheets.MAX_MATERIALS))
    made = _materials(svc, prompt_items=lines, variants=tilesheets.MAX_VARIANTS)
    assert made["tiles"] == tileatlas.MAX_CELLS
    assert _sheet(svc, made)["columns"] == 8
    assert _sheet(svc, made)["rows"] == 8


def test_a_one_material_sheet_is_one_cell_in_one_row(svc):
    """Not a 1x1 square dressed up as a sheet: a materials sheet is the list
    the user typed, laid out in the order they typed it."""
    made = _materials(svc, prompt_items=("packed sand",))
    block = _sheet(svc, made)
    assert (block["columns"], block["rows"]) == (1, 1)
    assert made["tiles"] == 1


# -- what a terrain request writes -------------------------------------------


def test_the_terrain_block_is_a_blob47_row_of_two_materials(svc):
    block = _sheet(svc, _terrain(svc, seed=11))
    assert block["version"] == 3
    assert block["mode"] == "terrain"
    assert block["layout"] == "blob47"
    assert block["columns"] == tilemask.TILE_COUNT
    assert block["rows"] == 1
    # The two *sources*, not the forty-seven composites: the cells of a terrain
    # atlas are made from these two, not generated on their own. So the
    # generation count and the cell count are different numbers here, and the
    # worker refuses anything but exactly two.
    assert len(block["materials"]) == 2
    assert [cell["index"] for cell in block["materials"]] == [0, 1]
    # Order is meaning, not presentation: ``tilemask.blob_rects`` makes a tile's
    # own centre cell a member, so entry 0 is the material that occupies the
    # covered region and entry 1 is what surrounds it.
    assert [cell["prompt"] for cell in block["materials"]] == [
        "short green grass",
        "shallow blue water",
    ]
    assert [cell["seed"] for cell in block["materials"]] == list(
        tileatlas.material_seeds(11, 2)
    )


def test_the_terrain_block_declares_its_one_terrain(svc):
    """One entry, naming ``inner`` -- the forty-seven blob cases are pictures
    of it. ``outer`` is the background composited against it, not a terrain
    of its own; it stays in ``materials`` and ``recipe.terrain.outer``.
    ``atlas_sidecar`` refuses a terrain count that does not match the atlas's
    one row, so a second entry here would fail every terrain job at the end
    of generation."""
    block = _sheet(svc, _terrain(svc))
    assert [entry["name"] for entry in block["terrains"]] == ["short green grass"]
    entry = block["terrains"][0]
    assert len(entry["fill"]) == 4
    assert len(entry["outline"]) == 4
    assert all(0 <= part <= 255 for part in entry["fill"] + entry["outline"])


def test_the_terrain_block_is_what_atlas_sidecar_will_actually_accept(svc):
    """The gap this closes: this file only ever checked the params dict, and
    ``test_tileset_worker.py`` only ever hand-writes params already shaped to
    fit ``atlas_sidecar`` -- so a door that declared the wrong terrain count
    could pass both suites and still fail every real job at the sidecar
    write, which is what a ``terrains=(inner, outer)`` door once did. This
    calls the door and then the sidecar it feeds, on the same block."""
    from warlock import _q_tileset

    block = _sheet(svc, _terrain(svc))
    geom = tileatlas.terrain_geometry(block["tile_w"], block["projection"])
    seeds = [int(entry["seed"]) for entry in block["materials"]]
    tileatlas.atlas_sidecar(
        geom,
        created=0.0,
        materials=_q_tileset._bind(geom, block["materials"], seeds, tileatlas.MODE_TERRAIN),
        terrains=block["terrains"],
        mask=block["mask"],
    )


def test_a_terrain_name_is_the_descriptions_first_words_not_the_description(svc):
    """A name is what a palette list shows beside a swatch; a description is
    capped at MAX_PROMPT, which is a thousand characters of it."""
    long = "a very " * 40 + "green field"
    block = _sheet(svc, _terrain(svc, inner_terrain=long))
    assert len(block["terrains"][0]["name"]) <= tilesheets.MAX_TERRAIN_NAME
    assert block["terrains"][0]["name"]
    # The description itself is unclipped: the words the model is given are not
    # the words the palette shows.
    assert block["materials"][0]["prompt"] == long


def test_the_terrain_block_records_the_field_that_will_draw_it(svc):
    """``roles.infer_roles`` returns None on a perfectly formed generated
    terrain set -- two opaque textures composited edge to edge have neither
    transparency nor a ring colour -- so what lands the set is this record."""
    block = _sheet(svc, _terrain(svc, seed=5))
    assert block["mask"] == {
        "version": tilemask.MASK_VERSION,
        "seed": 5,
        # None, not numbers: ``tilemask`` reads absent as "the ratio", which is
        # what keeps the boundary the same shape at every tile size.
        "inset": None,
        "amplitude": None,
        "feather": None,
    }


def test_the_boundary_is_carried_as_context_for_both_materials(svc):
    block = _sheet(svc, _terrain(svc, boundary="a temperate coastline"))
    assert block["boundary"] == "a temperate coastline"


def test_a_materials_request_never_carries_a_boundary(svc):
    block = _sheet(svc, _materials(svc, boundary="a temperate coastline"))
    assert block["boundary"] == ""


# -- the materials refusals --------------------------------------------------


def test_a_materials_sheet_with_no_lines_is_refused(svc):
    with pytest.raises(Invalid) as excinfo:
        _materials(svc, prompt_items=())
    assert excinfo.value.field == "prompt_items"


def test_blank_lines_are_not_materials(svc):
    """A trailing newline is what a text box has, not a material somebody asked
    for and forgot to describe."""
    with pytest.raises(Invalid) as excinfo:
        _materials(svc, prompt_items=("", "   ", "\n"))
    assert excinfo.value.field == "prompt_items"


def test_too_many_material_lines_is_refused_with_both_numbers(svc):
    lines = tuple(f"material {n}" for n in range(tilesheets.MAX_MATERIALS + 1))
    with pytest.raises(Invalid) as excinfo:
        _materials(svc, prompt_items=lines)
    assert excinfo.value.field == "prompt_items"
    assert str(len(lines)) in str(excinfo.value)
    assert str(tilesheets.MAX_MATERIALS) in str(excinfo.value)


def test_an_overlong_material_line_is_refused(svc):
    with pytest.raises(Invalid) as excinfo:
        _materials(svc, prompt_items=("moss", "x" * 5000))
    assert excinfo.value.field == "prompt_items"


@pytest.mark.parametrize("variants", [0, -1, 5, 64])
def test_a_variant_count_off_the_menu_is_refused_and_says_what_it_was(svc, variants):
    with pytest.raises(Invalid) as excinfo:
        _materials(svc, variants=variants)
    assert excinfo.value.field == "variants"
    assert str(variants) in str(excinfo.value)


def test_a_variant_count_that_is_not_a_number_is_a_refusal_not_a_crash(svc):
    with pytest.raises(Invalid) as excinfo:
        _materials(svc, variants="two")
    assert excinfo.value.field == "variants"


# -- the terrain refusals ----------------------------------------------------


@pytest.mark.parametrize("field", ["inner_terrain", "outer_terrain"])
@pytest.mark.parametrize("value", ["", "   "])
def test_a_terrain_set_needs_both_halves_described(svc, field, value):
    """Both are generated: a request describing one has nothing to put on the
    other side of every boundary."""
    with pytest.raises(Invalid) as excinfo:
        _terrain(svc, **{field: value})
    assert excinfo.value.field == field


@pytest.mark.parametrize("field", ["inner_terrain", "outer_terrain", "boundary"])
def test_an_overlong_terrain_description_is_refused(svc, field):
    with pytest.raises(Invalid) as excinfo:
        _terrain(svc, **{field: "x" * 5000})
    assert excinfo.value.field == field


@pytest.mark.parametrize("layout", ["wang16", "grid", ""])
def test_a_terrain_layout_off_the_menu_is_refused_and_says_what_it_was(svc, layout):
    with pytest.raises(Invalid) as excinfo:
        _terrain(svc, terrain_layout=layout)
    assert excinfo.value.field == "terrain_layout"
    assert repr(layout) in str(excinfo.value)


# -- the geometry the seamless modes delegate --------------------------------


@pytest.mark.parametrize("maker", [_materials, _terrain])
def test_a_tile_size_that_does_not_divide_a_material_is_refused(svc, maker):
    """48 is on the tile-size menu and does not divide 1024. Reducing on an
    inexact partition puts a one-pixel step at the wrap seam, which is the one
    place on a torus nobody looks."""
    with pytest.raises(Invalid) as excinfo:
        maker(svc, tile_size=48)
    assert excinfo.value.field == "tile_size"
    assert "48" in str(excinfo.value)
    assert str(tileatlas.MATERIAL_PX) in str(excinfo.value)


def test_the_grid_still_accepts_the_size_the_seamless_modes_refuse(svc):
    """Nothing is lost by the refusal above -- 48px tiles are still reachable,
    through the path that cuts them out of one frame rather than wrapping
    them."""
    made = tilesheets.create_tile_sheet(
        svc, prompt="a mossy dungeon", tile_size=48, mode=tilesheets.MODE_GRID,
        allow_grid=True,
    )
    assert made["tile_w"] == 48


@pytest.mark.parametrize("maker", [_materials, _terrain])
@pytest.mark.parametrize("view", ["three_quarter", "isometric"])
def test_a_view_that_cannot_tile_is_refused_by_name(svc, maker, view):
    """Each for a reason about tiling rather than about taste: a 3/4 tile has a
    visible front face that the row below would occlude, and an isometric tile
    is a 2:1 diamond rather than a rectangle to wrap."""
    with pytest.raises(Invalid) as excinfo:
        maker(svc, view=view)
    assert excinfo.value.field == "projection"
    assert "grid" in str(excinfo.value)


@pytest.mark.parametrize("maker", [_materials, _terrain])
def test_the_old_orthogonal_spelling_still_reads(svc, maker):
    assert _sheet(svc, maker(svc, view="orthogonal"))["projection"] == "top_down"


@pytest.mark.parametrize("maker", [_materials, _terrain])
def test_a_view_off_every_menu_names_the_control_the_user_sees(svc, maker):
    with pytest.raises(Invalid) as excinfo:
        maker(svc, view="hexagonal")
    assert excinfo.value.field == "projection"


# -- the ControlNet that is a property of the mode ---------------------------


@pytest.mark.parametrize("maker", [_materials, _terrain])
def test_a_seamless_request_never_names_a_control(svc, maker):
    """A key in params is a live setting: a seamless job carrying ``control``
    would charge a ControlNet it never opens against admission and write a
    sidecar naming a guide that never loaded."""
    params = svc.store.get(maker(svc)["id"])["params"]
    assert "control" not in params


def test_a_host_with_no_canny_weights_can_still_build_a_seamless_sheet(svc):
    """The bug this is about: ``control:canny`` was a hard requirement of the
    *kind*, so a materials sheet -- which never touches a ControlNet -- was
    refused at the door for a download it would never have opened."""
    from warlock import fetch

    cn = models.CONTROLNETS["canny"]
    variant = f".{cn.variant}" if cn.variant else ""
    root = svc.config.t2i_model_root / cn.dir_name
    (root / f"diffusion_pytorch_model{variant}.safetensors").unlink()
    (root / "config.json").unlink()
    assert not fetch.present(svc.config, "control", cn)

    assert _materials(svc)["tiles"] == 3
    assert _terrain(svc)["tiles"] == tilemask.TILE_COUNT
    # And the grid, which genuinely needs it, is still refused by name.
    with pytest.raises(Invalid) as excinfo:
        tilesheets.create_tile_sheet(
            svc, prompt="a mossy dungeon", mode=tilesheets.MODE_GRID, allow_grid=True
        )
    assert excinfo.value.field == "control"


def test_the_pixel_lora_is_required_by_every_mode(svc):
    """The art style *is* the LoRA in all three, so this one is not a property
    of the mode: missing, the worker logs and paints bare."""
    lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    (svc.config.t2i_model_root / "loras" / lora.filename).unlink()
    for maker in (_materials, _terrain):
        with pytest.raises(Invalid) as excinfo:
            maker(svc)
        assert excinfo.value.field == "style_lora"


# -- the rows the pane offers to install -------------------------------------


@pytest.mark.parametrize("mode", [tilesheets.MODE_MATERIALS, tilesheets.MODE_TERRAIN])
def test_a_seamless_mode_asks_for_no_controlnet(svc, mode):
    rows = tilesheets.rows_needed(mode)
    assert not any("control" in row for row in rows)
    assert f"lora:{models.PIXEL_SHEET_LORA}" in rows
    assert f"base:{tilesheets.TILE_SHEET_BASE_MODEL}" in rows


def test_the_grid_mode_still_asks_for_its_guide(svc):
    assert "control:canny" in tilesheets.rows_needed(tilesheets.MODE_GRID)


@pytest.mark.parametrize("mode", tilesheets.TILE_MODES)
def test_a_reference_adds_the_adapter_to_every_mode(svc, mode):
    plain = tilesheets.rows_needed(mode)
    conditioned = tilesheets.rows_needed(mode, True)
    assert conditioned[: len(plain)] == plain
    assert any("adapter" in row for row in conditioned)


def test_the_options_reply_publishes_the_row_list_of_every_mode(svc):
    """The pane must hardcode no ceiling this module enforces, which includes
    "what does this mode need downloaded"."""
    options = tilesheets.tile_sheet_options()
    assert options["modes"] == list(tilesheets.TILE_MODES)
    assert set(options["mode_labels"]) == set(tilesheets.TILE_MODES)
    for mode in tilesheets.TILE_MODES:
        assert options["mode_rows_needed"][mode] == list(tilesheets.rows_needed(mode))
        assert options["mode_reference_rows_needed"][mode] == list(
            tilesheets.rows_needed(mode, True)
        )
    for mode in (tilesheets.MODE_MATERIALS, tilesheets.MODE_TERRAIN):
        assert not any("canny" in row for row in options["mode_rows_needed"][mode])
        assert not any(
            "canny" in row for row in options["mode_reference_rows_needed"][mode]
        )
    assert any("canny" in row for row in options["mode_rows_needed"]["grid"])


def test_the_options_reply_says_which_sizes_and_views_the_seamless_modes_take(svc):
    options = tilesheets.tile_sheet_options()
    assert options["seamless_views"] == list(tileatlas.VIEWS)
    assert options["seamless_tile_sizes"] == [16, 32, 64]
    assert 48 in options["tile_sizes"]
    assert options["max_materials"] == tilesheets.MAX_MATERIALS
    assert options["max_variants"] == tilesheets.MAX_VARIANTS
    assert options["max_cells"] == tilesheets.MAX_CELLS
    assert options["terrain_layouts"] == list(tilesheets.TERRAIN_LAYOUTS)
    assert options["defaults"]["mode"] == tilesheets.DEFAULT_MODE


def test_style_lock_is_always_a_real_bool_in_every_mode(svc):
    """Present, never absent, and never a truthy string. ``vram`` gates the
    IP-encoder term on it: under style lock there is no user reference for the
    door to write ``ip_adapter`` from -- the anchor image does not exist until
    the job is half over -- so an absent key silently under-charges admission by
    about a gibibyte, which is an OOM reproducible only on a full card."""
    for made in (
        _materials(svc),
        _materials(svc, style_lock=True),
        _terrain(svc),
        _terrain(svc, style_lock=True),
        tilesheets.create_tile_sheet(
            svc, prompt="a mossy dungeon", mode=tilesheets.MODE_GRID, allow_grid=True
        ),
    ):
        block = _sheet(svc, made)
        assert "style_lock" in block
        assert block["style_lock"] is True or block["style_lock"] is False


def test_style_lock_is_carried_rather_than_interpreted(svc):
    assert _sheet(svc, _materials(svc, style_lock=True))["style_lock"] is True
    assert _sheet(svc, _materials(svc))["style_lock"] is False


# -- the palette ------------------------------------------------------------


@pytest.mark.parametrize(
    "cells,expected",
    [
        # One and two materials sit on the floor, which is the grid mode's
        # measured value -- and two materials is exactly what the ground run
        # measured, so this range is byte-identical to what shipped.
        (1, 64),
        (2, 64),
        # Sixteen and sixty-four saturate the ceiling: an indexed PNG stops
        # being indexed past 256.
        (16, 256),
        (64, 256),
    ],
)
def test_the_palette_grows_with_the_material_count_and_stops_at_a_byte(cells, expected):
    assert tilesheets.sheet_colors(cells) == expected


def test_the_palette_floor_is_the_grid_modes_measured_value(svc):
    """Not a coincidence and not a copy: 64 is what the ground run measured for
    one generation of one subject, and a sheet of one or two materials is that
    case."""
    assert tilesheets.sheet_colors(1) == tilesheets.SHEET_COLORS


def test_a_sixteen_material_sheet_is_not_quantized_to_four_colours_each(svc):
    """The bug the function is about: sixteen unrelated materials sharing 64
    entries is four colours apiece, which is a posterisation rather than a
    shared palette."""
    lines = tuple(f"material {n}" for n in range(16))
    params = svc.store.get(_materials(svc, prompt_items=lines)["id"])["params"]
    assert params["colors"] == 256


def test_a_terrain_sheet_is_quantized_for_its_two_materials(svc):
    """Forty-seven cells, two materials: what the shared table has to hold is
    two surfaces, which is the case the measurement covers."""
    params = svc.store.get(_terrain(svc)["id"])["params"]
    assert params["colors"] == 64


def test_the_grid_mode_keeps_the_constant(svc):
    params = svc.store.get(
        tilesheets.create_tile_sheet(
            svc, prompt="a mossy dungeon", mode=tilesheets.MODE_GRID, allow_grid=True
        )["id"]
    )["params"]
    assert params["colors"] == tilesheets.SHEET_COLORS


# -- nothing is written until every check has passed -------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt_items": ()},
        {"variants": 9},
        {"tile_size": 48},
        {"view": "isometric"},
        {"mode": "wang"},
    ],
)
def test_a_refused_request_leaves_no_row(svc, overrides):
    before = len(svc.store.list())
    with pytest.raises(Invalid):
        _materials(svc, **overrides)
    assert len(svc.store.list()) == before


def test_a_refused_terrain_request_leaves_no_row(svc):
    before = len(svc.store.list())
    with pytest.raises(Invalid):
        _terrain(svc, inner_terrain="")
    assert len(svc.store.list()) == before


# -- the planner, which is now load-bearing ----------------------------------
#
# ``create_generation_request`` used to flatten the structured tile document
# into one prompt string and hand it to one generation, so the mode, the
# material list and the two terrain descriptions were compiled into a sentence
# and the door had no way to tell a materials sheet from a terrain set. Nothing
# in the tree covered that arm; these are the tests that would have caught it.


def _request(**tile):
    return generation.GenerationRequest(
        generation_type="tileset",
        prompt="a temperate ruin, muted palette",
        tile=generation.TileSettings(**tile),
    )


def test_the_planner_hands_the_material_list_through_rather_than_joining_it(svc):
    from warlock.service import jobs as svc_jobs

    made = svc_jobs.create_generation_request(
        svc, _request(mode="collection", prompt_items=MATERIALS)
    )
    block = _sheet(svc, made)
    assert block["mode"] == "materials"
    assert [cell["prompt"] for cell in block["materials"]] == list(MATERIALS)
    # The sheet-level prompt is the style sentence, not the joined list.
    assert svc.store.get(made["id"])["prompt"] == "a temperate ruin, muted palette"


def test_the_planner_hands_the_two_terrains_through_rather_than_describing_them(svc):
    """This is the sentence that used to be built here: "inner terrain: X;
    outer terrain: Y; Z", handed to one generation."""
    from warlock.service import jobs as svc_jobs

    made = svc_jobs.create_generation_request(
        svc,
        _request(
            mode="terrain_transition",
            inner_terrain="short green grass",
            outer_terrain="shallow blue water",
            boundary="a temperate coastline",
        ),
    )
    block = _sheet(svc, made)
    assert block["mode"] == "terrain"
    assert [cell["prompt"] for cell in block["materials"]] == [
        "short green grass",
        "shallow blue water",
    ]
    assert block["boundary"] == "a temperate coastline"
    assert "inner terrain:" not in svc.store.get(made["id"])["prompt"]


def test_a_path_set_is_a_terrain_transition_with_the_two_surfaces_renamed(svc):
    """The path is what appears as the blob shapes and the ground is what
    surrounds it, which is exactly inner/outer."""
    from warlock.service import jobs as svc_jobs

    made = svc_jobs.create_generation_request(
        svc,
        _request(
            mode="path", ground="packed dirt", path="worn cobblestones", edge="a village"
        ),
    )
    block = _sheet(svc, made)
    assert block["mode"] == "terrain"
    assert [cell["prompt"] for cell in block["materials"]] == [
        "worn cobblestones",
        "packed dirt",
    ]
    assert block["boundary"] == "a village"


def test_an_unbuildable_target_cell_is_refused_rather_than_quietly_replaced(svc):
    """It used to read ``target if target in TILE_SIZES else DEFAULT_TILE_SIZE``,
    which answered "make me 96px tiles" with 32px tiles and told nobody -- while
    the request document, the row and the sidecar all went on saying 96."""
    from warlock.service import jobs as svc_jobs

    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_generation_request(
            svc, _request(mode="collection", prompt_items=MATERIALS, target_cell_px=96)
        )
    assert excinfo.value.field == "tile.target_cell_px"
    assert "96" in str(excinfo.value)


def test_a_buildable_target_cell_is_honoured(svc):
    from warlock.service import jobs as svc_jobs

    made = svc_jobs.create_generation_request(
        svc, _request(mode="collection", prompt_items=MATERIALS, target_cell_px=64)
    )
    assert _sheet(svc, made)["tile_w"] == 64


def test_an_unstated_target_cell_takes_the_doors_default(svc):
    from warlock.service import jobs as svc_jobs

    made = svc_jobs.create_generation_request(
        svc, _request(mode="collection", prompt_items=MATERIALS)
    )
    assert _sheet(svc, made)["tile_w"] == tilesheets.DEFAULT_TILE_SIZE


def test_the_legacy_form_adapter_carries_the_whole_tile_document():
    """``request_from_legacy`` built ``TileSettings(view=..., target_cell_px=...)``
    and nothing else, which is why ``mode`` has been unreachable from the UI
    since it was written."""
    request = generation.request_from_legacy(
        {
            "output": "sheet",
            "prompt": "a temperate ruin",
            "tile_mode": "terrain",
            "projection": "orthogonal",
            # A multi-line text control gives one string; a list control gives a
            # list. Both spellings reach this adapter from stored profiles.
            "prompt_items": "moss\n\ncracked mud\n",
            "inner_terrain": "short green grass",
            "outer_terrain": "shallow blue water",
            "boundary": "a temperate coastline",
            "variants": 3,
            "style_lock": True,
        }
    )
    assert request.generation_type == "tileset"
    assert request.tile.mode == "terrain"
    assert request.tile.view == "top_down"
    assert request.tile.prompt_items == ("moss", "cracked mud")
    assert request.tile.inner_terrain == "short green grass"
    assert request.tile.outer_terrain == "shallow blue water"
    assert request.tile.boundary == "a temperate coastline"
    assert request.tile.variants == 3
    assert request.tile.style_lock is True
    assert request.tile.terrain_layout == "blob47"


def test_the_structured_request_can_name_a_palette(svc, tmp_path):
    """``TileSettings`` had no palette or dither field, so a tileset submitted
    through ``create_generation_request`` could not use a capability the pane
    path could -- not because the door refused it, but because there was nowhere
    on the request to say it."""
    from warlock.service import jobs as svc_jobs

    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    (directory / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    made = svc_jobs.create_generation_request(
        svc,
        _request(
            mode="collection", prompt_items=MATERIALS, palette="duo", dither=True
        ),
    )
    params = svc.store.get(made["id"])["params"]
    assert params["palette"] == "duo"
    assert params["dither"] is True


def test_a_palette_the_structured_request_names_is_still_checked_at_the_door(svc):
    from warlock.service import jobs as svc_jobs

    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_generation_request(
            svc, _request(mode="collection", prompt_items=MATERIALS, palette="gone")
        )
    assert excinfo.value.field == "palette"


def test_the_legacy_form_adapter_carries_the_pixel_look_to_both_blocks():
    """One form serves both sheet arms, so both settings blocks read the same
    two keys -- the sprite block through ``_check_sprite_sheet``, the tile block
    through ``create_tile_sheet``."""
    request = generation.request_from_legacy(
        {"output": "sheet", "prompt": "x", "palette": "nord", "dither": True}
    )
    assert (request.tile.palette, request.tile.dither) == ("nord", True)
    assert (request.sprite.palette, request.sprite.dither) == ("nord", True)


def test_a_form_that_says_nothing_about_tiles_still_reads_as_it_always_did():
    """Every new field has to have a defaulted answer, or the adapter starts
    refusing forms saved before it grew them."""
    request = generation.request_from_legacy({"output": "sheet", "prompt": "x"})
    assert request.tile == generation.TileSettings(view="top_down")


def test_a_stored_row_from_before_the_new_fields_still_opens():
    """``from_dict`` fills the defaults, and ``mode`` keeps the spelling stored
    rows carry."""
    tile = generation.GenerationRequest.from_dict(
        {"generation_type": "tileset", "prompt": "x", "tile": {"mode": "collection"}}
    ).tile
    assert tile.mode == "collection"
    assert tile.terrain_layout == "blob47"
    assert tile.style_lock is False


def test_the_terrain_case_is_validated_before_anything_is_queued():
    request = _request(mode="terrain", inner_terrain="grass")
    resolved = generation.resolve_recipe(request, None)
    issues = generation.validate_request(request, resolved)
    assert any(issue.field == "tile.terrain" for issue in issues)


def test_a_terrain_layout_nothing_ships_is_refused_by_the_request_document():
    request = _request(
        mode="terrain",
        inner_terrain="grass",
        outer_terrain="water",
        terrain_layout="wang16",
    )
    issues = generation.validate_request(request, generation.resolve_recipe(request, None))
    assert any(issue.field == "tile.terrain_layout" for issue in issues)


def test_the_tileset_recipe_does_not_require_a_controlnet():
    """A grid-mode requirement, not a requirement of the asset type: a host with
    no canny weights can build materials and terrain, so a recipe that demanded
    one would disqualify the whole tileset type on it."""
    recipe = next(r for r in generation.RECIPES if r.key == "tileset_sdxl")
    assert not any("control" in row for row in recipe.required_downloads)


# -- the drift pins ----------------------------------------------------------


def test_the_modes_match_the_pipeline():
    """``tileatlas`` builds the two seamless ones; this door offers those two
    and the legacy grid, which the pipeline knows nothing about."""
    assert tileatlas.MODES == (tilesheets.MODE_MATERIALS, tilesheets.MODE_TERRAIN)
    assert set(tileatlas.MODES) < set(tilesheets.TILE_MODES)
    assert tilesheets.DEFAULT_MODE in tilesheets.TILE_MODES


def test_the_seamless_view_list_matches_the_pipeline():
    assert tileatlas.VIEWS == ("top_down",)
    assert set(tileatlas.VIEWS) <= set(tilesheets.VIEWS)


def test_the_seamless_sizes_are_exactly_the_menu_entries_that_divide_a_material():
    """A door offering a size the pipeline refuses is a form that costs the
    request rather than refusing it at the control."""
    for size in tilesheets.TILE_SIZES:
        divides = not tileatlas.MATERIAL_PX % size
        assert divides == (size in (16, 32, 64))


def test_the_cell_bounds_are_one_rule_with_three_enforcement_points():
    """The door caps the lines and the variants, ``tileatlas`` caps their
    product, and ``asset_workflows`` does the expansion. Three places, one
    rule."""
    assert tilesheets.MAX_MATERIALS == asset_workflows.MAX_COLLECTION_LINES
    assert tilesheets.MAX_VARIANTS == asset_workflows.MAX_COLLECTION_VARIANTS
    assert tilesheets.MAX_CELLS == asset_workflows.MAX_COLLECTION_CELLS
    assert tilesheets.MAX_MATERIALS == tileatlas.MAX_MATERIALS
    assert tilesheets.MAX_CELLS == tileatlas.MAX_CELLS
    # And the two ceilings meet exactly, which is why the largest legal request
    # lands on the product cap rather than being caught by it.
    assert tilesheets.MAX_MATERIALS * tilesheets.MAX_VARIANTS == tilesheets.MAX_CELLS


def test_the_grid_planner_is_gone_and_nothing_reaches_for_it():
    """``tile_plan`` and the Wang/path role tables it was the only caller of
    were deleted on 2026-08-29.

    Not tidying: the plan was compiled from a stored request and then written
    beside a *grid* sheet -- one guided generation sliced on a fixed lattice --
    as a description of per-cell prompts or sixteen corner roles that the
    picture does not contain. A planner nothing plans with is worse than no
    planner, so this is the same scan the six functions before it got: the names
    exist nowhere in ``src/`` but the module docstring that says why they went.
    """
    from pathlib import Path

    import warlock

    for name in ("tile_plan(", "wang_roles(", "path_roles(", "TileRole("):
        for path in Path(warlock.__file__).parent.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert name not in source, f"{path.name} still reaches for {name}"
    assert not hasattr(asset_workflows, "tile_plan")
    # What survived, and why: this one has a caller that generates from it.
    assert asset_workflows.collection_cells(("moss",), 1)[0]["prompt"] == "moss"


def test_the_layout_words_match_the_pipeline():
    assert tilesheets.TERRAIN_LAYOUTS == ("blob47",)
    assert tileatlas.terrain_geometry(32, "top_down").layout == "blob47"
    assert tileatlas.material_geometry(32, "top_down", 4).layout == "grid"


def test_the_request_documents_mode_words_reach_this_door():
    """``generation.TILE_MODES`` is what a stored request may say and
    ``asset_workflows.TILE_MODE_ALIASES`` is what it means; every one of them
    has to land on a mode this door builds."""
    assert set(generation.TILE_MODES) == set(asset_workflows.TILE_MODE_ALIASES)
    for word in generation.TILE_MODES:
        assert asset_workflows.TILE_MODE_ALIASES[word] in tilesheets.TILE_MODES
    # And the stored default still reads as the materials shape.
    assert (
        asset_workflows.TILE_MODE_ALIASES[generation.TileSettings().mode]
        == tilesheets.MODE_MATERIALS
    )


# -- the authored palette, on the two seamless modes --------------------------
#
# One door, so the refusals themselves are pinned in the grid file. What is
# worth stating here is that the seamless modes reach the same block -- and
# that a named palette meets ``sheet_colors``' provisional budget rather than
# combining with it.


@pytest.fixture
def paldir(svc, tmp_path):
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    return directory


def test_a_materials_request_carries_the_palette_and_the_dither(svc, paldir):
    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    params = svc.store.get(_materials(svc, palette="duo", dither=True)["id"])["params"]
    assert params["palette"] == "duo"
    assert params["dither"] is True


def test_a_terrain_request_carries_them_too(svc, paldir):
    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    params = svc.store.get(_terrain(svc, palette="duo")["id"])["params"]
    assert params["palette"] == "duo"


def test_a_named_palette_does_not_change_the_stored_colour_budget(svc, paldir):
    """They are not two halves of one setting. ``colors`` is still what the row
    asked for -- a reroll that later drops the palette needs an answer -- and
    ``palette_source`` in the sidecar is what says the median cut never ran."""
    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    plain = svc.store.get(_materials(svc)["id"])["params"]["colors"]
    with_palette = svc.store.get(_materials(svc, palette="duo")["id"])["params"]["colors"]
    assert plain == with_palette == tilesheets.sheet_colors(len(MATERIALS))


def test_the_seamless_modes_refuse_an_outline_by_name_as_well(svc):
    with pytest.raises(Invalid) as excinfo:
        _terrain(svc, outline="inner")
    assert excinfo.value.field == "outline"

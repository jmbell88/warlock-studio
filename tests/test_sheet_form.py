"""The Sheet output: the switch, the arms, and which door each opens.

Most of what is asserted here is a pure function of the form dict, because the
pane draws what these return -- the split ``test_tile_form`` already uses.

The claim the whole file is about: **one Output switch, two doors**. A tile set
is its own job kind and bypasses ``create_job`` entirely; a sprite sheet is an
ordinary reference job carrying a follow-up request, the shape the rig checkbox
already has. Getting that backwards in either direction is silent -- a tile set
through ``create_job`` would be refused for an unknown output, and a sprite
sheet through the tile door would draw a grid of sixty-four characters.

**And the submit closure is exercised**, which is the hole this file used to
have. ``validate``, ``submit_kwargs`` and ``sheet_rows`` were all covered and the
one thing that actually reaches ``create_tile_sheet`` was not, so when the door
grew a ``mode`` the submit went on not sending one -- compiling a materials
request with no materials in it, refused at ``field="prompt_items"``, on every
press of a button the whole rest of this file said was fine.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from warlock.pipelines import tileatlas, tilesheet
from warlock.service import sprites as svc_sprites
from warlock.service import tilesheets as svc_tilesheets
from warlock.studio import create_assets, settings
from warlock.studio.panes import settings_2d
from warlock.studio.state import default_form_2d

#: The three layouts, under the names the form field carries.
MATERIALS = svc_tilesheets.MODE_MATERIALS
TERRAIN = svc_tilesheets.MODE_TERRAIN
GRID = svc_tilesheets.MODE_GRID


def _sheet_form(**overrides):
    """A tile-set form as the pane would have it, plus overrides.

    Built through ``create_assets.sync_legacy_fields`` rather than by setting
    ``output`` by hand: those fields are derived from the asset type on every
    frame, and a fixture that set one of them alone tests a form the app never
    produces -- in particular one whose ``asset_type`` the tile-set door refuses.

    A material line is typed, because the default layout is ``materials`` and a
    materials sheet is the list of surfaces you type. The empty case is a test of
    its own below rather than the shape every other test starts from.
    """
    form = default_form_2d()
    form["prompt"] = "mossy dungeon"
    form["asset_type"] = "tileset"
    create_assets.sync_legacy_fields(form)
    form["materials"] = "mossy stone"
    form.update(overrides)
    return form


def _terrain_form(**overrides):
    fields = {
        "tile_mode": TERRAIN,
        "inner_terrain": "wet grass",
        "outer_terrain": "dark water",
        **overrides,
    }
    return _sheet_form(**fields)


# -- the switch --------------------------------------------------------------


def test_a_new_form_is_not_a_sheet():
    assert default_form_2d()["output"] == "reference"


def test_the_output_kind_comes_from_the_asset_registry_not_a_control():
    """``OUTPUTS`` and ``OUTPUT_NOTES`` were the pre-registry segmented control
    and its prose. ``_asset_type`` is the shipped control now and
    ``create_assets.sync_legacy_fields`` sets ``form["output"]`` from the chosen
    spec, so the old table had no caller left -- and this test was guarding the
    *order of a dead control*, which is the failure it exists to catch now.
    """
    assert not hasattr(settings_2d, "OUTPUTS")
    assert not hasattr(settings_2d, "OUTPUT_NOTES")
    assert not hasattr(settings_2d, "_output")
    # And the three kinds are still the three kinds, read where they live.
    from warlock.studio import create_assets

    outputs = {spec.output for spec in create_assets.ASSET_TYPES.values()}
    assert outputs == {"reference", "tile", "sheet"}


def test_a_new_form_defaults_to_a_tile_grid():
    form = default_form_2d()
    assert form["sheet_type"] == "tile"
    assert form["tile_size"] == "32"
    # The key is still ``projection``; only its vocabulary widened.
    assert form["projection"] == "top_down"


def test_the_sheet_fields_are_strings_so_they_restore():
    """``settings.restore_form`` gates on ``type(value) is type(default)``, so
    an int default here would make every persisted size fail to restore --
    silently, which is the whole hazard."""
    form = default_form_2d()
    for key in ("sheet_type", "tile_size", "projection", "sheet_layout", "cell_size"):
        assert isinstance(form[key], str), key


# -- the tile arm ------------------------------------------------------------


def test_a_tile_grid_does_not_go_through_create_job():
    """It is its own job kind with its own door. ``submit_kwargs`` is only ever
    asked about the paths that *do* reach ``create_job``, and ``generate``
    branches before it -- this pins that the branch exists by pinning that the
    kwargs would be wrong if it did not."""
    form = _sheet_form()
    assert settings_2d.submit_kwargs(form)["output"] == "reference"


def test_a_tile_grid_still_needs_a_prompt():
    form = _sheet_form(prompt="")
    assert any(p.field == "prompt" for p in settings_2d.validate(form))


@pytest.mark.parametrize("size", svc_tilesheets.TILE_SIZES)
def test_every_size_the_grid_layout_offers_validates(size):
    assert not settings_2d.validate(_sheet_form(tile_mode=GRID, tile_size=str(size)))


@pytest.mark.parametrize("mode", [MATERIALS, TERRAIN])
def test_every_size_a_seamless_layout_offers_validates(mode):
    form = _terrain_form() if mode == TERRAIN else _sheet_form()
    for size in settings_2d.tile_sizes_for(form):
        assert not settings_2d.validate({**form, "tile_size": str(size)}), size


def test_a_tile_size_off_the_menu_is_caught_before_the_door():
    """Reachable from a *restored* form rather than from this frame's control:
    the value is persisted and the menu can change between releases."""
    problems = settings_2d.validate(_sheet_form(tile_size="24"))
    assert any(p.field == "tile_size" for p in problems)


def test_a_view_off_the_menu_is_caught_before_the_door():
    problems = settings_2d.validate(_sheet_form(tile_mode=GRID, projection="hexagonal"))
    assert any(p.field == "projection" for p in problems)


def test_the_sprite_arm_is_not_judged_by_the_tile_arms_rules():
    """The two arms share a form dict, so a tile size left over from the other
    arm must not refuse a sprite sheet that never reads it."""
    form = _sheet_form(sheet_type="sprite", tile_size="24", projection="hexagonal")
    assert not settings_2d.validate(form)


@pytest.mark.parametrize(
    "leftover",
    [
        {"control": "canny"},
        {"ip_adapter": "plus"},
        {"control": "canny", "base_model": "sdxl_turbo"},
        {"style_lora": "pixel-art-xl", "base_model": "klein"},
    ],
)
def test_a_tile_grid_is_not_refused_over_fields_its_door_never_reads(leftover):
    """``create_tile_sheet`` pins its own base, LoRA and ControlNet and reads
    none of these -- so a refusal naming one of them disables Generate over
    somebody else's job.

    Reachable rather than theoretical: ``control`` and ``ip_adapter`` are
    persisted and ``ref_path`` is VOLATILE, so any session that once
    conditioned an Object reopens with the pair already split -- and the Sheet
    output then came up with a dead Generate button reading "Conditioning needs
    a reference image".
    """
    assert not settings_2d.validate(_sheet_form(**leftover))


@pytest.mark.parametrize(
    "leftover,field",
    [
        ({"control": "canny"}, "ref_path"),
        ({"style_lora": "pixel-art-xl", "base_model": "klein"}, "style_lora"),
    ],
)
def test_the_sprite_arm_keeps_every_one_of_them(leftover, field):
    """The other half of the same rule, and the reason it is a per-arm gate
    rather than a per-output one: the sprite arm's first step *is* an ordinary
    reference job, so it reads all four and every refusal above still applies
    to it."""
    form = _sheet_form(sheet_type="sprite", **leftover)
    assert any(p.field == field for p in settings_2d.validate(form)), leftover


# -- the sprite arm ----------------------------------------------------------


def test_a_sprite_sheet_is_a_reference_job_carrying_a_follow_up():
    """The rig checkbox's shape, and for its reason: the character is a row in
    its own right, so a sheet the user hates still leaves them the drawing."""
    kwargs = settings_2d.submit_kwargs(_sheet_form(sheet_type="sprite"))
    assert kwargs["output"] == "reference"
    assert kwargs["sprite_sheet"] == {
        "sheet_type": "turnaround",
        "logical_size": 64,
        "colors": svc_sprites.DEFAULT_SPRITE_COLORS,
        # Blank by default: the optional final reduction, which never upscales
        # and so means "keep the working cell" when nothing asked for one.
        "target_cell_px": None,
    }


def test_the_sprite_block_carries_what_the_form_chose():
    kwargs = settings_2d.submit_kwargs(
        _sheet_form(sheet_type="sprite", sheet_layout="walk", cell_size="32")
    )
    assert kwargs["sprite_sheet"]["sheet_type"] == "walk"
    assert kwargs["sprite_sheet"]["logical_size"] == 32


def test_no_other_output_carries_a_sprite_block():
    """The key's presence is what the door branches on, so an object job that
    grew one would queue two generations nobody asked for."""
    for form in (default_form_2d(), _sheet_form(), _sheet_form(output="tile")):
        form["prompt"] = "a barrel"
        assert "sprite_sheet" not in settings_2d.submit_kwargs(form)


# -- what a sheet needs downloaded -------------------------------------------


def test_the_tile_arm_asks_for_no_adapter_without_a_reference():
    """Its IP-Adapter is optional, and a gate that demanded one would tell a
    user who has everything the common request uses that they are missing a
    download."""
    assert settings_2d.sheet_rows(_sheet_form(tile_mode=GRID)) == (
        svc_tilesheets.TILE_SHEET_ROWS
    )


def test_attaching_a_reference_adds_the_adapter_to_what_is_needed():
    rows = settings_2d.sheet_rows(
        _sheet_form(tile_mode=GRID, ref_path="C:/somewhere/style.png")
    )
    assert rows == svc_tilesheets.TILE_SHEET_REFERENCE_ROWS
    assert any("adapter" in row for row in rows)


@pytest.mark.parametrize("mode", [MATERIALS, TERRAIN])
def test_a_seamless_layout_asks_for_no_controlnet(mode):
    """The grid guide *is* the ControlNet and these two never open one, so
    asking for canny here tells a user who has everything this request uses that
    they are missing a download."""
    rows = settings_2d.sheet_rows(_sheet_form(tile_mode=mode))
    assert rows == tuple(svc_tilesheets.rows_needed(mode))
    assert not any("control" in row for row in rows)
    assert any("control" in row for row in settings_2d.sheet_rows(_sheet_form(tile_mode=GRID)))


def test_the_sprite_arm_asks_for_the_sprite_rows():
    """Both adapters are mandatory there -- the pose guide *is* the ControlNet
    and the identity *is* the IP-Adapter."""
    assert settings_2d.sheet_rows(_sheet_form(sheet_type="sprite")) == svc_sprites.SPRITE_ROWS


def test_a_missing_sheet_row_names_the_output_not_a_model_select():
    """A sheet's doors pin their own base and LoRA and ignore the form's, so a
    refusal pointing at the Model select would be about a choice the run never
    reads."""
    ctx = SimpleNamespace(
        model_rows=[
            {"row_key": key, "present": key != "control:canny", "label": key}
            for key in svc_tilesheets.TILE_SHEET_ROWS
        ]
    )
    problem = settings_2d.weights_problem(ctx, _sheet_form(tile_mode=GRID))
    assert problem is not None
    assert problem.field == "output"


def test_a_seamless_layout_is_not_stopped_by_the_grids_missing_controlnet():
    """The other half of the per-layout rows: a host with no canny weights can
    build materials and terrain, and used to be told at the gate to download one
    for a request that never opens it."""
    ctx = SimpleNamespace(
        model_rows=[
            {"row_key": key, "present": key != "control:canny", "label": key}
            for key in svc_tilesheets.TILE_SHEET_ROWS
        ]
    )
    assert settings_2d.weights_problem(ctx, _sheet_form()) is None


def test_a_fully_installed_host_has_no_sheet_problem():
    ctx = SimpleNamespace(
        model_rows=[
            {"row_key": key, "present": True, "label": key}
            for key in svc_tilesheets.TILE_SHEET_REFERENCE_ROWS
        ]
    )
    assert settings_2d.weights_problem(ctx, _sheet_form()) is None


def test_a_sprite_checks_its_selected_reference_model_before_the_locked_recipe():
    form = _sheet_form(sheet_type="sprite")
    form["base_model"] = "turbo"
    rows = [
        {"row_key": key, "present": True, "label": key}
        for key in svc_sprites.SPRITE_ROWS
    ]
    rows.append({"row_key": "base:turbo", "present": False, "label": "Turbo"})
    problem = settings_2d.weights_problem(SimpleNamespace(model_rows=rows), form)
    assert problem is not None
    assert problem.field == "base_model"


def test_a_sprite_checks_its_locked_recipe_after_the_reference_recipe():
    form = _sheet_form(sheet_type="sprite")
    rows = [
        {"row_key": "base:sdxl_cfg", "present": True, "label": "SDXL"},
        *[
            {"row_key": key, "present": key != "control:canny", "label": key}
            for key in svc_sprites.SPRITE_ROWS
        ],
    ]
    problem = settings_2d.weights_problem(SimpleNamespace(model_rows=rows), form)
    assert problem is not None
    assert problem.field == "output"


# -- the preview the pane asks for -------------------------------------------


class _PreviewCtx:
    """Just enough of AppCtx for ``_preview``'s off-thread request."""

    def __init__(self, form):
        self.state = SimpleNamespace(form_2d=form, preview_dirty_at=1e-9, preview=None)
        self.svc = object()
        self.calls: list[tuple[tuple, dict]] = []

    def submit(self, _key, _fn, *args, **kwargs):
        self.calls.append((args, kwargs))
        return True


def test_the_preview_of_a_grid_asks_for_the_grid_template():
    ctx = _PreviewCtx(_sheet_form(tile_mode=GRID))
    settings_2d._preview(ctx)
    assert ctx.calls and ctx.calls[0][1]["tilesheet"] is True
    assert ctx.calls[0][1]["tile"] is False


def test_the_preview_of_a_grid_sends_the_pipelines_subject_not_the_bare_prompt():
    """The view and detail clauses are appended before the template, so a
    preview built from the raw prompt would be missing half the composition
    this output kind adds."""
    ctx = _PreviewCtx(_sheet_form(tile_mode=GRID, projection="isometric"))
    settings_2d._preview(ctx)
    subject = ctx.calls[0][0][2]
    assert "mossy dungeon" in subject
    assert "isometric" in subject
    assert tilesheet.DETAIL_CLAUSE in subject


def test_the_preview_of_a_seamless_layout_is_of_its_first_material():
    """What will actually run. These layouts compose one material at a time
    through ``prompt.TILE_TEMPLATE`` and never send the Description at all, so
    previewing the grid's sheet subject here would show a sentence no generation
    on this path ever sees."""
    ctx = _PreviewCtx(_sheet_form(materials="wet cobblestones\ndry sand"))
    settings_2d._preview(ctx)
    assert ctx.calls[0][1] == {"tile": True, "tilesheet": False}
    subject = ctx.calls[0][0][2]
    assert subject == tileatlas.material_subject("wet cobblestones", index=0, total=2)
    assert "mossy dungeon" not in subject


def test_the_preview_of_a_terrain_set_is_of_its_inner_surface():
    """The one the forty-seven cases are pictures of, carrying the shared
    setting -- which is what ``terrain_subjects`` appends to both halves."""
    ctx = _PreviewCtx(_terrain_form(boundary="a temperate coastline"))
    settings_2d._preview(ctx)
    subject = ctx.calls[0][0][2]
    assert subject.startswith("wet grass, a temperate coastline")
    assert "dark water" not in subject


def test_a_seamless_layout_with_nothing_described_previews_nothing():
    """A request with no material has no first material. Showing the
    Description instead would be a preview of a sentence this layout does not
    send, which is the whole failure the seamless preview exists to avoid."""
    ctx = _PreviewCtx(_sheet_form(materials=""))
    ctx.state.preview = {"prompt": "stale", settings_2d.CLEARED_KEY: ["kept"]}
    settings_2d._preview(ctx)
    assert ctx.calls == []
    assert ctx.state.preview == {settings_2d.CLEARED_KEY: ["kept"]}


def test_the_preview_of_a_sprite_sheet_is_an_ordinary_reference_preview():
    """Its first step *is* an ordinary reference of one character, so it takes
    the object template and stays eligible for expansion."""
    ctx = _PreviewCtx(_sheet_form(sheet_type="sprite"))
    settings_2d._preview(ctx)
    assert ctx.calls[0][1]["tilesheet"] is False
    assert ctx.calls[0][1]["tile"] is False
    assert ctx.calls[0][0][2] == "mossy dungeon"


def test_a_restored_form_with_the_old_projection_word_still_validates():
    """A profile saved before the vocabulary widened carries "orthogonal". The
    form reads it through the service's alias table rather than holding a second
    opinion, so an old profile opens on Top-down instead of being refused."""
    assert not [p for p in settings_2d.validate(_sheet_form(projection="orthogonal"))
                if p.field == "projection"]


def test_every_view_the_grid_layout_offers_validates():
    """The mirror of the tile-size sweep beside it. The old suite had no such
    sweep for views, which is how a third one could have been added and gone
    unexercised by the form."""
    for view in svc_tilesheets.VIEWS:
        problems = settings_2d.validate(_sheet_form(tile_mode=GRID, projection=view))
        assert not [p for p in problems if p.field == "projection"], view


# -- the three layouts -------------------------------------------------------


def test_a_new_form_asks_for_materials_and_never_for_the_grid():
    """The door refuses the grid unless the request explicitly asks for it, so
    the layout a user never touched must not be the one that needs the escape
    hatch."""
    assert default_form_2d()["tile_mode"] == svc_tilesheets.DEFAULT_MODE
    assert settings_2d.tile_mode_of(default_form_2d()) != GRID


def test_an_unrecognised_stored_layout_reads_as_the_default():
    """A persisted field whose menu moved between releases: resolving to nothing
    would disable Generate over a control whose value nobody can see."""
    assert settings_2d.tile_mode_of(_sheet_form(tile_mode="quilt")) == MATERIALS


def test_a_seamless_layout_offers_only_the_sizes_that_divide_a_material():
    """48 is on the grid's menu and is not on this one -- 1024/48 is not whole,
    and a block that differs by a pixel puts a step at the wrap seam of a torus.
    Sourced from the pipeline rather than listed here, so the day the frame size
    moves this test moves with it."""
    offered = settings_2d.tile_sizes_for(_sheet_form())
    assert 48 in settings_2d.tile_sizes_for(_sheet_form(tile_mode=GRID))
    assert 48 not in offered
    assert offered == [
        size for size in svc_tilesheets.TILE_SIZES if not tileatlas.MATERIAL_PX % size
    ]


@pytest.mark.parametrize("mode", [MATERIALS, TERRAIN])
def test_a_seamless_layout_offers_only_the_view_that_wraps(mode):
    form = _terrain_form() if mode == TERRAIN else _sheet_form()
    assert settings_2d.views_for(form) == list(tileatlas.VIEWS) == ["top_down"]


@pytest.mark.parametrize("size,view", [("48", "top_down"), ("32", "isometric")])
def test_a_seamless_layout_refuses_the_geometry_it_cannot_draw(size, view):
    """Reachable from a restored form: both fields are persisted, so a sheet
    composed under the grid layout and reopened under a seamless one arrives
    carrying values that layout has no way to draw."""
    problems = settings_2d.validate(_sheet_form(tile_size=size, projection=view))
    assert problems
    assert {p.field for p in problems} & {"tile_size", "projection"}


def test_a_materials_sheet_with_no_materials_is_refused_on_that_field():
    problems = settings_2d.validate(_sheet_form(materials="  \n\n"))
    assert [p.field for p in problems] == ["prompt_items"]


def test_a_materials_sheet_counts_lines_rather_than_newlines():
    """A trailing newline is not a seventeenth material, and the door drops
    blanks too -- a form that counted them would report a cell total the request
    will not produce."""
    form = _sheet_form(materials="grass\n\ndirt\n")
    assert settings_2d.material_lines(form) == ("grass", "dirt")


def test_more_materials_than_one_sheet_can_name_is_refused():
    lines = "\n".join(f"material {n}" for n in range(svc_tilesheets.MAX_MATERIALS + 1))
    problems = settings_2d.validate(_sheet_form(materials=lines))
    assert any(p.field == "prompt_items" for p in problems)


def test_lines_by_variants_past_the_cell_ceiling_is_refused_on_the_variants():
    """Each cell is its own full generation, so the ceiling is on the product --
    and the control that can be turned down is the one the refusal names."""
    lines = "\n".join(f"material {n}" for n in range(svc_tilesheets.MAX_MATERIALS))
    problems = settings_2d.validate(_sheet_form(materials=lines, variants="4"))
    assert not problems  # 16 x 4 is exactly the ceiling
    problems = settings_2d.validate(
        _sheet_form(materials=lines, variants=str(svc_tilesheets.MAX_VARIANTS + 1))
    )
    assert any(p.field == "variants" for p in problems)


@pytest.mark.parametrize("missing", ["inner_terrain", "outer_terrain"])
def test_a_terrain_set_needs_both_surfaces_described(missing):
    """Both halves are generated: a request describing one has nothing to put on
    the other side of every boundary."""
    problems = settings_2d.validate(_terrain_form(**{missing: ""}))
    assert [p.field for p in problems] == [missing]


def test_a_terrain_sets_boundary_is_optional():
    assert not settings_2d.validate(_terrain_form(boundary=""))
    assert not settings_2d.validate(_terrain_form(boundary="a temperate coastline"))


def test_a_grid_sheet_needs_no_materials_and_no_terrains():
    """The layouts do not share their fields, and a leftover from another one
    must not refuse the request in front of the user."""
    assert not settings_2d.validate(
        _sheet_form(tile_mode=GRID, materials="", projection="isometric")
    )


def test_switching_to_a_seamless_layout_drops_the_geometry_it_cannot_draw():
    """``clear_unusable``'s rule applied to the tile arm: a control that
    ``validate`` refuses while offering only legal values is a dead end, and
    both of these are persisted, so both survive the switch."""
    form = _sheet_form(tile_mode=GRID, tile_size="48", projection="isometric")
    form["tile_mode"] = MATERIALS
    cleared = settings_2d.clear_for_layout(form)
    assert len(cleared) == 2
    assert form["tile_size"] == "32"
    assert form["projection"] == "top_down"
    # And the dead end is gone rather than merely explained.
    assert not settings_2d.validate(form)


def test_switching_to_the_grid_takes_nothing_away():
    """It draws every size and every view, so there is nothing it cannot keep --
    and a clear that fired here would throw away a choice the user just made."""
    form = _sheet_form(tile_mode=GRID, tile_size="48", projection="isometric")
    assert settings_2d.clear_for_layout(form) == []
    assert form["tile_size"] == "48"


def test_a_seamless_layout_keeps_a_geometry_it_can_draw():
    form = _sheet_form(tile_size="64")
    assert settings_2d.clear_for_layout(form) == []
    assert form["tile_size"] == "64"


# -- what the submit actually sends ------------------------------------------


class _SubmitCtx:
    """Just enough of AppCtx for the tile-set submit, run inline."""

    def __init__(self) -> None:
        self.svc = object()
        self.state = _SubmitState()
        self.keys: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.result: dict | None = None
        self.model_rows: list[dict] = []

    def submit(self, key, run, *args, **kwargs):
        self.keys.append(key)
        self.result = run(*args, **kwargs)
        return True

    def toast(self, message, kind="info"):
        self.toasts.append((message, kind))


class _SubmitState:
    def __init__(self) -> None:
        self.field_errors: dict[str, str] = {}
        self.history: list[str] = []
        self.preview: dict = {}
        self.preview_dirty_at = 0.0

    def clear_field_errors(self):
        self.field_errors.clear()

    def clear_field_error(self, field):
        self.field_errors.pop(field, None)

    def note_field_error(self, field, message):
        self.field_errors[field] = message

    def remember_prompt(self, prompt):
        self.history.insert(0, prompt)


def _sent(monkeypatch, form, *, through_generate: bool = False):
    """The kwargs that reach ``create_tile_sheet``, and the ctx that sent them.

    Monkeypatched on the module the pane calls it through, which is the call the
    press makes -- the whole point of this helper is that nothing here is a
    re-derivation of what the closure would do.
    """
    seen: dict = {}

    def fake_create_tile_sheet(svc, **kwargs):
        seen.update(kwargs)
        seen["svc"] = svc
        return {"id": "abc123", "mode": kwargs.get("mode"), "tiles": 1}

    monkeypatch.setattr(
        settings_2d.svc_tilesheets, "create_tile_sheet", fake_create_tile_sheet
    )
    ctx = _SubmitCtx()
    if through_generate:
        settings_2d.generate(ctx, form)
    else:
        settings_2d._generate_tile_sheet(ctx, form)
    return seen, ctx


def test_a_materials_sheet_submits_its_lines_and_its_variants(monkeypatch):
    sent, ctx = _sent(monkeypatch, _sheet_form(materials="grass\ndirt", variants="2"))
    assert ctx.keys == ["submit"]
    assert sent["mode"] == MATERIALS
    assert sent["prompt_items"] == ["grass", "dirt"]
    assert sent["variants"] == 2
    assert sent["prompt"] == "mossy dungeon"
    assert sent["tile_size"] == 32
    assert sent["view"] == "top_down"
    assert sent["asset_type"] == "tileset"
    assert sent["asset_intent"] == "tileset"
    assert "allow_grid" not in sent


def test_a_terrain_set_submits_both_surfaces_and_its_shared_setting(monkeypatch):
    sent, _ctx = _sent(
        monkeypatch, _terrain_form(boundary="a temperate coastline", tile_size="16")
    )
    assert sent["mode"] == TERRAIN
    assert sent["inner_terrain"] == "wet grass"
    assert sent["outer_terrain"] == "dark water"
    assert sent["boundary"] == "a temperate coastline"
    assert sent["tile_size"] == 16
    assert "prompt_items" not in sent
    assert "variants" not in sent
    assert "allow_grid" not in sent


def test_only_an_explicit_grid_carries_the_escape_hatch(monkeypatch):
    """``allow_grid`` is the door's escape hatch on a refusal about a
    *measurement*, so an explicit choice is exactly what it is for -- and a
    default that carried it would put every unconsidered press back on the
    layout the measurement is about."""
    sent, _ctx = _sent(monkeypatch, _sheet_form(tile_mode=GRID, projection="isometric"))
    assert sent["mode"] == GRID
    assert sent["allow_grid"] is True
    assert sent["view"] == "isometric"
    for form in (_sheet_form(), _terrain_form()):
        sent, _ctx = _sent(monkeypatch, form)
        assert sent.get("allow_grid") is not True


def test_the_generate_button_reaches_the_tile_door(monkeypatch):
    """The regression this file was missing: every assertion above about the
    form was true while the press itself sent a request the door refused."""
    sent, ctx = _sent(monkeypatch, _sheet_form(), through_generate=True)
    assert sent["mode"] == MATERIALS
    assert ctx.result["id"] == "abc123"
    assert ctx.state.field_errors == {}


def test_a_refused_form_never_reaches_the_door(monkeypatch):
    sent, ctx = _sent(monkeypatch, _sheet_form(materials=""), through_generate=True)
    assert sent == {}
    assert ctx.state.field_errors == {
        "prompt_items": (
            "A materials sheet is the list of surfaces you type; describe at "
            "least one, one per line."
        )
    }


@pytest.mark.parametrize("mode", [MATERIALS, TERRAIN, GRID])
def test_every_layout_the_form_offers_is_queued_by_the_real_door(svc, mode):
    """The end of the wire, and the test that would have caught the break.

    Everything above is a fake ``create_tile_sheet``; this one is the door
    itself, with weights on the host and a row at the end of it. A submit that
    compiles a request the door refuses is not a wrong value in a dict -- it is a
    button that does nothing -- and only a call that reaches the door can tell
    the difference.
    """
    form = _terrain_form() if mode == TERRAIN else _sheet_form(tile_mode=mode)
    made = svc_tilesheets.create_tile_sheet(
        svc, **settings_2d.tile_sheet_kwargs(form)
    )
    assert made["mode"] == mode
    block = svc.store.get(made["id"])["params"]["sheet"]
    assert block["mode"] == mode
    assert block["tile_w"] == 32
    if mode == MATERIALS:
        assert [cell["prompt"] for cell in block["materials"]] == ["mossy stone"]
    elif mode == TERRAIN:
        assert [t["name"] for t in block["terrains"]] == ["wet grass", "dark water"]
        assert block["layout"] == "blob47"


@pytest.mark.parametrize("mode", [MATERIALS, TERRAIN, GRID])
def test_every_layout_sends_only_arguments_the_door_accepts(mode):
    """A contract check rather than a spelling check: the pane names its kwargs
    and the door names its parameters, and the failure when those drift is a
    ``TypeError`` on a background thread that the button reports as a toast with
    no field in it."""
    form = _terrain_form() if mode == TERRAIN else _sheet_form(tile_mode=mode)
    kwargs = settings_2d.tile_sheet_kwargs(form)
    signature = inspect.signature(svc_tilesheets.create_tile_sheet)
    signature.bind(object(), reference=None, **kwargs)


# -- the form an upgrading user reopens --------------------------------------


def _upgraded_form(tmp_path, **overrides):
    """A 2D form saved before the layout control existed, read back as the app
    reads it.

    Through a real settings file and ``Settings.load`` rather than by building
    the dict here: the question these tests ask is what the *migration* does,
    and a hand-built form would answer it by assumption. The keys popped are
    exactly the ones the seamless layouts brought with them -- a file written
    before them has none of them, and a materials list least of all.
    """
    stored = settings.sanitise_form(_sheet_form(**overrides))
    for key in (
        "tile_mode",
        "materials",
        "variants",
        "inner_terrain",
        "outer_terrain",
        "boundary",
    ):
        stored.pop(key, None)
    (tmp_path / settings.FILENAME).write_text(
        json.dumps({"version": settings.VERSION, "data": {"form_2d": stored}}),
        encoding="utf-8",
    )
    loaded = settings.Settings.load(tmp_path)
    assert loaded.take_notice() is None, "the reset net must not be what ran"
    return settings.restore_form(default_form_2d(), loaded.get("form_2d"))


def test_a_form_saved_before_the_layout_control_reopens_on_the_grid(tmp_path):
    """It described an 8x8 grid, because that was the only sheet the tile arm
    drew. Grid is the layout that still draws exactly that."""
    form = _upgraded_form(tmp_path)
    assert settings_2d.tile_mode_of(form) == GRID
    assert not settings_2d.validate(form)


def test_that_form_generates_the_grid_it_described(tmp_path, monkeypatch):
    """The whole point of the migration, at the press. On Materials this form
    is a sheet with nothing in it: refused at ``field="prompt_items"``, on the
    first Generate after an upgrade, over a layout nobody picked."""
    form = _upgraded_form(tmp_path)
    sent, ctx = _sent(monkeypatch, form, through_generate=True)
    assert sent["mode"] == GRID
    assert sent["allow_grid"] is True
    assert ctx.state.field_errors == {}
    assert ctx.result["id"] == "abc123"


def test_that_form_is_queued_by_the_real_door(svc, tmp_path):
    """And the end of the wire, since a request the door refuses is a button
    that does nothing rather than a wrong value in a dict."""
    form = _upgraded_form(tmp_path)
    made = svc_tilesheets.create_tile_sheet(svc, **settings_2d.tile_sheet_kwargs(form))
    assert made["mode"] == GRID
    assert svc.store.get(made["id"])["params"]["sheet"]["mode"] == GRID


def test_a_genuinely_fresh_form_opens_on_materials(tmp_path):
    """No history to read, so it starts on the better path -- the door's own
    default, not this file's opinion of it."""
    assert default_form_2d()["tile_mode"] == MATERIALS
    assert settings.restore_form(default_form_2d(), None)["tile_mode"] == MATERIALS
    assert settings.Settings.load(tmp_path).get("form_2d") is None

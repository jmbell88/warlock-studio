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

from warlock.pipelines import tileatlas
from warlock.service import jobs as svc_jobs
from warlock.service import sprites as svc_sprites
from warlock.service import tilesheets as svc_tilesheets
from warlock.service.errors import Invalid
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
    # And the kinds are still read where they live. ``character`` is a fourth
    # word and deliberately *not* a fourth ``create_job`` output: it is what the
    # Character type derives, and its door is ``service.characters`` -- so
    # ``create_job``'s own three are the first three and nothing else may join
    # them without a branch at that door.
    from warlock.service import _jobs_create
    from warlock.studio import create_assets

    outputs = {spec.output for spec in create_assets.ASSET_TYPES.values()}
    assert outputs == {"reference", "tile", "sheet", "character"}
    door = inspect.getsource(_jobs_create.create_job)
    assert '("reference", "model", "tile")' in door
    assert '"character"' not in door, "the character type has its own door"


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
        # The number the form's own summary line promised, not a default the
        # worker would pick separately: see ``sprite_sheet_kwargs``.
        "candidates": 2,
        "logical_size": 64,
        "colors": svc_sprites.DEFAULT_SPRITE_COLORS,
        # Blank by default: the optional final reduction, which never upscales
        # and so means "keep the working cell" when nothing asked for one.
        "target_cell_px": None,
        # The pixel look. Sent always rather than only when set, so "no palette"
        # and "the form was never asked" are different requests -- and the
        # outline is the pipeline's own forced default rather than "none",
        # because a synthesised cell has no guaranteed margin.
        "palette": "",
        "dither": False,
        "outline": svc_sprites.DEFAULT_SPRITE_OUTLINE,
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
        # One entry: the inner terrain, the row's blob cases are pictures of
        # it. ``dark water`` (outer) stays in ``materials``, not here --
        # see ``_terrain_record``.
        assert [t["name"] for t in block["terrains"]] == ["wet grass"]
        assert [m["prompt"] for m in block["materials"]] == ["wet grass", "dark water"]
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


# -- the pixel look, on both arms --------------------------------------------
#
# The three settings both sheet doors have taken since they started sharing
# ``service.pixelopts`` and that no pane offered. Every assertion here is on the
# submit path rather than on a widget, because a control that sets a form key
# nothing sends is the same unreachable capability with an extra step.


@pytest.fixture
def paldir(svc, tmp_path):
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    return directory


def test_the_tile_submit_carries_the_palette_and_the_dither(monkeypatch):
    sent, _ctx = _sent(monkeypatch, _sheet_form(palette="nord", dither=True))
    assert sent["palette"] == "nord"
    assert sent["dither"] is True


def test_the_tile_submit_says_no_palette_rather_than_saying_nothing(monkeypatch):
    """Sent always, so "derive one" and "the form was never asked" are the same
    request only because they mean the same thing -- and the door's own defaults
    are these values."""
    sent, _ctx = _sent(monkeypatch, _sheet_form())
    assert sent["palette"] == ""
    assert sent["dither"] is False


def test_the_tile_submit_never_names_an_outline(monkeypatch):
    """The door refuses one by name: ``pixelize._edge_mask`` pads
    ``constant_values=False``, so on a cell that is opaque edge to edge --
    which every tile is -- ``inner`` returns the outer ring of *each* cell, a
    grid line around all sixty-four tiles. The form must not offer what the door
    refuses, and must not send it even as "none"."""
    for form in (_sheet_form(), _terrain_form(), _sheet_form(tile_mode=GRID)):
        form["outline"] = "inner"
        sent, _ctx = _sent(monkeypatch, form)
        assert "outline" not in sent


def test_a_tile_sheet_with_a_palette_is_queued_by_the_real_door(svc, paldir):
    """The end of the wire. Everything above is a fake ``create_tile_sheet``;
    only a call that reaches the door can tell a carried setting from a button
    that does nothing."""
    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    form = _sheet_form(palette="duo", dither=True)
    made = svc_tilesheets.create_tile_sheet(svc, **settings_2d.tile_sheet_kwargs(form))
    params = svc.store.get(made["id"])["params"]
    assert params["palette"] == "duo"
    assert params["dither"] is True


def test_a_palette_deleted_since_the_form_listed_it_costs_the_request(svc, paldir):
    """Which is what loading it at the door is for: the alternative is N full
    generations and a sheet that merely came back the wrong colours."""
    form = _sheet_form(palette="gone")
    with pytest.raises(Invalid) as excinfo:
        svc_tilesheets.create_tile_sheet(svc, **settings_2d.tile_sheet_kwargs(form))
    assert excinfo.value.field == "palette"


def test_the_sprite_block_carries_the_whole_pixel_look():
    block = settings_2d.sprite_sheet_kwargs(
        _sheet_form(sheet_type="sprite", palette="nord", dither=True, outline="outer")
    )
    assert block["palette"] == "nord"
    assert block["dither"] is True
    assert block["outline"] == "outer"


def test_the_sprite_block_defaults_to_the_outline_its_geometry_forces():
    """``inner`` and never ``outer``: a synthesised cell is 256 or 512px of a
    1024px atlas the model filled as it liked, so the subject runs off its cell
    edge often enough that growing the silhouette would clip."""
    block = settings_2d.sprite_sheet_kwargs(_sheet_form(sheet_type="sprite"))
    assert block["outline"] == svc_sprites.DEFAULT_SPRITE_OUTLINE == "inner"


def test_the_sprite_block_is_refused_by_the_real_checker_for_a_bad_palette(svc, paldir):
    """``_check_sprite_sheet`` validates the follow-up at the *reference* door,
    so a palette that has gone missing costs the request rather than an SDXL
    generation and an hour."""
    from warlock.service import sprites as sprites_door

    block = settings_2d.sprite_sheet_kwargs(_sheet_form(sheet_type="sprite", palette="gone"))
    with pytest.raises(Invalid) as excinfo:
        sprites_door._check_options(svc, block)
    assert excinfo.value.field == "palette"


def test_every_outline_the_sprite_arm_offers_survives_the_sprite_checker(svc):
    """The form's menu against the door's ladder: a segmented control offering a
    mode the assembler refuses is a control that fails at the door it was drawn
    from."""
    from warlock.pipelines import pixelize
    from warlock.service import sprites as sprites_door

    for mode in pixelize.OUTLINE_MODES:
        block = settings_2d.sprite_sheet_kwargs(
            _sheet_form(sheet_type="sprite", outline=mode)
        )
        assert sprites_door._check_options(svc, block)["outline"] == mode


def test_the_palette_list_is_not_the_cached_options_blob():
    """``tile_sheet_options`` is cached in a one-slot list for the life of the
    process on the stated ground that nothing in it reads disk. A palette is a
    file the user drops in a directory, so a listing folded in there would never
    show one installed after launch."""
    assert "palettes" not in svc_tilesheets.tile_sheet_options()
    assert "palettes" not in svc_sprites.sprite_options()


def test_a_palette_that_is_no_longer_installed_stays_on_the_menu_marked():
    """``lora_options``' rule, and for its reason: a palette is a file, so a
    stem the form holds can stop existing between two launches -- and the value
    keeping Generate off must not be the one thing the user cannot see."""
    assert settings_2d.palette_options(["duo"], "") == (
        ("", "Derived from the render"),
        ("duo", "duo"),
    )
    assert settings_2d.palette_options(["duo"], "duo")[-1] == ("duo", "duo")
    missing = settings_2d.palette_options(["duo"], "nord")[-1]
    assert missing[0] == "nord"
    assert "not in the palette folder" in missing[1]
    # And with nothing installed at all, the named one is still listed: it is
    # what the door is about to refuse.
    assert settings_2d.palette_options([], "nord")[-1][0] == "nord"


def test_each_arm_lists_palettes_through_its_own_door(svc, paldir):
    """Uncached, and asked of the door rather than of the options blob."""
    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    assert svc_tilesheets.tile_sheet_palettes(svc) == ["duo"]
    assert svc_sprites.sprite_palettes(svc) == ["duo"]
    (paldir / "nes.hex").write_text("#000000\n#ffffff\n")
    assert svc_sprites.sprite_palettes(svc) == ["duo", "nes"]


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


# -- the sprite arm's action and directions ----------------------------------
#
# Every assertion here is on the submit path or on the pure function the pane
# draws its line from, rather than on a widget: a control that sets a form key
# nothing sends is an unreachable capability with an extra step.


@pytest.fixture
def sprite_weights(monkeypatch):
    """Every weight a synthesis loads, present. The Create door checks all of
    them for the *follow-up* before it will take the reference."""
    from warlock import fetch

    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (True, None))
    monkeypatch.setattr(fetch, "present", lambda *a, **k: True)


def _sprite_form(**overrides):
    """A sprite-sheet form as the pane would have it. ``_sheet_form``'s rule:
    through ``sync_legacy_fields``, because those keys are derived every frame.
    """
    form = default_form_2d()
    form["prompt"] = "a hooded ranger"
    form["asset_type"] = "sprite_sheet"
    create_assets.sync_legacy_fields(form)
    form.update(overrides)
    return form


def test_a_new_sprite_form_is_the_turnaround_it_has_always_been():
    """The default is not moved by the actions arriving: a stored form and a new
    one have to mean the same sheet."""
    plan = settings_2d.sprite_plan(_sprite_form())

    assert plan["layout"] == "turnaround"
    assert plan["cells"] == 4
    assert plan["candidates"] == 2


def test_the_action_combo_offers_the_turnaround_and_what_has_a_guide():
    options = svc_sprites.sprite_options()
    entries = settings_2d.sprite_action_options(options, "legacy:turnaround")

    keys = [key for key, _label in entries]
    assert keys[0] == "legacy:turnaround"
    assert keys[1:] == [a["key"] for a in options["actions"]]
    # Actions, never kinds: the direction count is the *other* control.
    assert not any(key[-1].isdigit() for key in keys)


def test_a_legacy_kind_and_an_action_of_the_same_name_are_different_entries():
    """The trap in this combo. Legacy ``walk`` is a four-frame cycle and the
    action ``walk`` is eight, so one key meaning both would show a stored legacy
    walk as the action -- and selecting it would silently double the cycle."""
    options = svc_sprites.sprite_options()

    assert settings_2d.sprite_action_key("walk") == "legacy:walk"
    assert settings_2d.sprite_action_key("walk8") == "walk"
    assert settings_2d.sprite_action_key("turnaround") == "legacy:turnaround"

    entries = dict(settings_2d.sprite_action_options(options, "legacy:walk"))
    assert entries["legacy:walk"] == "Walk (legacy, 4 frames)"
    assert entries["walk"] == "Walk"
    # And each resolves back to the sheet it names.
    assert settings_2d.sprite_layout_for(options, "legacy:walk", 8) == "walk"
    assert settings_2d.sprite_layout_for(options, "walk", 8) == "walk8"


def test_a_stored_layout_the_menu_cannot_offer_is_named_rather_than_dropped():
    """``palette_options``' rule: silently moving a form off the thing it says
    it is set to is how a user comes to submit something they did not choose."""
    options = svc_sprites.sprite_options()

    unknown = dict(settings_2d.sprite_action_options(options, "dance4"))
    assert unknown["dance4"] == "dance4 (unavailable)"


def test_picking_an_action_composes_the_kind_from_the_pair():
    options = svc_sprites.sprite_options()

    assert settings_2d.sprite_layout_for(options, "idle", 8) == "idle8"
    # An action with no four-direction guide falls back to a count it *has*,
    # because the two controls move independently: picking an action while the
    # Directions control still says four must land on a sheet that exists.
    assert settings_2d.sprite_layout_for(options, "walk", 4) == "walk8"
    assert settings_2d.sprite_layout_for(options, "legacy:turnaround", 8) == "turnaround"


def test_the_plan_is_the_arithmetic_the_line_and_the_block_both_read():
    form = _sprite_form(sheet_layout="walk8")
    plan = settings_2d.sprite_plan(form)

    assert plan == {
        "layout": "walk8",
        "action": "walk",
        "directions": 8,
        "frames": 8,
        "cells": 64,
        "bands": 8,
        "candidates": 1,
        "generations": 8,
        "sizes": (32,),
        "logical_size": 32,
    }


def test_the_cost_line_names_the_grid_the_generations_and_the_wait():
    line = settings_2d._sprite_cost(settings_2d.sprite_plan(_sprite_form(sheet_layout="idle8")))

    assert "8 directions x 4 frames = 32 cells" in line
    assert "8 generations for one draft" in line
    assert "about 3 minutes" in line


def test_the_turnaround_line_still_describes_a_pair():
    line = settings_2d._sprite_cost(settings_2d.sprite_plan(_sprite_form()))

    assert "4 directions x 1 frames = 4 cells" in line
    assert "2 generations for 2 drafts" in line


def test_the_submit_block_carries_the_action_the_form_chose():
    kwargs = settings_2d.submit_kwargs(_sprite_form(sheet_layout="idle8", cell_size="32"))

    assert kwargs["sprite_sheet"]["sheet_type"] == "idle8"
    assert kwargs["sprite_sheet"]["logical_size"] == 32
    # The number the cost line promised, not one the worker would pick
    # separately -- a form that says eight and spends sixteen is the defect.
    assert kwargs["sprite_sheet"]["candidates"] == 1


def test_the_size_picker_moves_off_a_size_the_chosen_action_cannot_take():
    """The gate. A 64px cell is the form's default and an eight-frame walk does
    not fit a band at it, so the picker draws the ladder that is left and the
    form is moved onto it -- rather than composing a request refused after the
    press."""
    form = _sprite_form(sheet_layout="walk8", cell_size="64")

    assert settings_2d.sprite_plan(form)["sizes"] == (32,)
    assert settings_2d.sprite_plan(form)["logical_size"] == 32
    # And the *submit* is what carries it, not the picker: the Action control is
    # always on screen and the size picker lives inside Advanced, so a clamp
    # that only ran while that section was drawn would not run at all for a user
    # who never opened it.
    assert settings_2d.submit_kwargs(form)["sprite_sheet"]["logical_size"] == 32


def test_a_size_the_chosen_action_does_allow_is_left_alone():
    for size in ("32", "48", "64"):
        form = _sprite_form(sheet_layout="idle8", cell_size=size)
        assert settings_2d.sprite_plan(form)["logical_size"] == int(size)


def test_an_action_sheet_press_is_admitted_by_the_real_door(svc, sprite_weights):
    """The end of the wire, ``test_every_layout_the_form_offers_is_queued_by_
    the_real_door``'s reason: everything above is a dict, and only a call that
    reaches ``create_job`` can tell a carried setting from a button that does
    nothing. The follow-up block is validated *here*, at the reference door."""
    form = _sprite_form(sheet_layout="idle8", cell_size="32")
    made = svc_jobs.create_job(svc, **settings_2d.submit_kwargs(form))

    block = svc.store.get(made["id"])["params"]["sprite_sheet"]
    assert block["sheet_type"] == "idle8"
    assert block["logical_size"] == 32
    assert block["candidates"] == 1


def test_a_press_the_size_gate_would_have_stopped_is_refused_at_that_door(
    svc, sprite_weights
):
    """The same refusal from the other side: if the picker ever offered a size
    the action cannot take, this is what the press would meet -- a sentence
    naming both numbers, on the field the control is drawn under."""
    form = _sprite_form(sheet_layout="walk8", cell_size="64")
    kwargs = settings_2d.submit_kwargs(form)
    kwargs["sprite_sheet"]["logical_size"] = 64

    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(svc, **kwargs)

    assert "8 frames of 512px" in str(caught.value)
    assert caught.value.field == "logical_size"


def test_a_sprite_form_sends_only_arguments_the_door_accepts():
    """The contract check the tile arm has, on the arm that grew two controls:
    the failure when the pane's kwargs and the door's parameters drift is a
    ``TypeError`` on a background thread, reported as a toast with no field."""
    kwargs = settings_2d.submit_kwargs(_sprite_form(sheet_layout="idle8"))
    signature = inspect.signature(svc_jobs.create_job)
    signature.bind(object(), **kwargs)

    block = kwargs["sprite_sheet"]
    checker = inspect.signature(svc_sprites.create_sprite_synthesis)
    # Every key of the block is a name the *direct* door takes too, or the two
    # ways of making a synthesis row would admit different requests.
    for key in block:
        if key == "target_cell_px":
            continue  # the optional final reduction, which is not a sheet option
        assert key in checker.parameters, key


def test_a_restored_layout_from_settings_survives_the_round_trip():
    """The persisted field carries a kind now, so the boundary check has to know
    the planned ones -- an unrecognised value is dropped, which would silently
    reopen an action sheet as a turnaround."""
    for layout in ("turnaround", "walk", "idle8", "walk8"):
        assert settings._safe_form_value("sheet_layout", layout) is True
    assert settings._safe_form_value("sheet_layout", "dance9") is False


def test_the_request_document_and_the_form_field_are_the_same_choice():
    """``sheet_layout`` is a kind; ``SpriteSettings`` is a mode plus a pair. The
    two legacy kinds name themselves rather than pretending to be an action
    whose frame count they do not have."""
    from warlock import generation as gen

    assert gen.sprite_from_layout("idle8") == ("action", "idle", 8)
    assert gen.sprite_from_layout("walk") == ("walk", "idle", 4)
    assert gen.sprite_from_layout("turnaround") == ("turnaround", "idle", 4)
    assert gen.sprite_from_layout("dance9") == ("turnaround", "idle", 4)
    for layout in ("turnaround", "walk", "idle8", "walk4"):
        mode, action, directions = gen.sprite_from_layout(layout)
        assert gen.sprite_layout_of(
            gen.SpriteSettings(mode=mode, action=action, directions=directions)
        ) == layout


def test_a_structured_request_no_longer_collapses_every_action_onto_walk(
    svc, sprite_weights
):
    """It read ``"turnaround" if mode == "turnaround" else "walk"``, which
    answered all seven actions and both direction counts with the legacy 4x4
    walk -- admitted, queued and published as a sheet nobody asked for, with the
    request document still saying "idle" beside it."""
    from warlock import generation as gen

    request = gen.GenerationRequest(
        generation_type="sprite_sheet",
        prompt="a hooded ranger",
        sprite=gen.SpriteSettings(mode="action", action="idle", directions=8),
    )
    made = svc_jobs.create_generation_request(svc, request)

    block = svc.store.get(made["id"])["params"]["sprite_sheet"]
    assert block["sheet_type"] == "idle8"


def test_provenance_is_on_the_row_before_the_row_exists(svc, tmp_path, monkeypatch):
    """It used to be merged on *after* ``create_job`` returned, which lost two
    races: the worker's ``next_queued`` poll can claim the row in the gap, and
    ``_q_generate`` writes its claim-time snapshot of ``params`` back whole --
    deleting every key merged in since. So the check is not "the row ends up
    with it" but "the row was never without it"."""
    from PIL import Image

    from warlock import generation as gen

    ref = tmp_path / "ref.png"
    Image.new("RGB", (8, 8), (30, 30, 30)).save(ref)

    seen: list[dict] = []
    real = svc.store.create

    def spy(kind, prompt, params, job_id, **kw):
        # What the insert was *handed*, and what was on disk at that moment.
        seen.append(
            {
                "params": dict(params),
                "files": sorted(p.name for p in svc.config.job_dir(job_id).glob("*")),
            }
        )
        return real(kind, prompt, params, job_id, **kw)

    monkeypatch.setattr(svc.store, "create", spy)
    request = gen.GenerationRequest(
        generation_type="image", prompt="a hooded ranger", references=[str(ref)]
    )
    svc_jobs.create_generation_request(svc, request)

    assert len(seen) == 1
    at_insert = seen[0]["params"]
    assert at_insert["generation_request"]["prompt"] == "a hooded ranger"
    assert at_insert["resolved_recipe"]["version"] == gen.RECIPE_REGISTRY_VERSION
    named = at_insert["native_reference_files"]
    assert named == ["native_reference_0.png"]
    # Named *and* already written: an empty list and a list naming a file that
    # is not there yet are the same bug one step apart.
    assert set(named) <= set(seen[0]["files"])


def test_a_structured_request_can_still_name_the_two_legacy_atlases(
    svc, sprite_weights
):
    from warlock import generation as gen

    for mode in gen.SPRITE_LEGACY_MODES:
        request = gen.GenerationRequest(
            generation_type="sprite_sheet",
            prompt="a hooded ranger",
            sprite=gen.SpriteSettings(mode=mode),
        )
        made = svc_jobs.create_generation_request(svc, request)
        block = svc.store.get(made["id"])["params"]["sprite_sheet"]
        assert block["sheet_type"] == mode

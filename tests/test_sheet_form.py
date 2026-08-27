"""The Sheet output: the switch, the two arms, and which door each opens.

Everything asserted here is a pure function of the form dict, because the pane
draws what these return -- the split ``test_tile_form`` already uses.

The claim the whole file is about: **one Output switch, two doors**. A tile grid
is its own job kind and bypasses ``create_job`` entirely; a sprite sheet is an
ordinary reference job carrying a follow-up request, the shape the rig checkbox
already has. Getting that backwards in either direction is silent -- a tile grid
through ``create_job`` would be refused for an unknown output, and a sprite
sheet through the tile door would draw a grid of sixty-four characters.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.pipelines import tilesheet
from warlock.service import sprites as svc_sprites
from warlock.service import tilesheets as svc_tilesheets
from warlock.studio.panes import settings_2d
from warlock.studio.state import default_form_2d


def _sheet_form(**overrides):
    form = default_form_2d()
    form["prompt"] = "mossy dungeon"
    form["output"] = "sheet"
    form.update(overrides)
    return form


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
def test_every_offered_tile_size_validates(size):
    assert not settings_2d.validate(_sheet_form(tile_size=str(size)))


def test_a_tile_size_off_the_menu_is_caught_before_the_door():
    """Reachable from a *restored* form rather than from this frame's control:
    the value is persisted and the menu can change between releases."""
    problems = settings_2d.validate(_sheet_form(tile_size="24"))
    assert any(p.field == "tile_size" for p in problems)


def test_a_view_off_the_menu_is_caught_before_the_door():
    problems = settings_2d.validate(_sheet_form(projection="hexagonal"))
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
    assert settings_2d.sheet_rows(_sheet_form()) == svc_tilesheets.TILE_SHEET_ROWS


def test_attaching_a_reference_adds_the_adapter_to_what_is_needed():
    rows = settings_2d.sheet_rows(_sheet_form(ref_path="C:/somewhere/style.png"))
    assert rows == svc_tilesheets.TILE_SHEET_REFERENCE_ROWS
    assert any("adapter" in row for row in rows)


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
    problem = settings_2d.weights_problem(ctx, _sheet_form())
    assert problem is not None
    assert problem.field == "output"


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
    ctx = _PreviewCtx(_sheet_form())
    settings_2d._preview(ctx)
    assert ctx.calls and ctx.calls[0][1]["tilesheet"] is True
    assert ctx.calls[0][1]["tile"] is False


def test_the_preview_of_a_grid_sends_the_pipelines_subject_not_the_bare_prompt():
    """The view and detail clauses are appended before the template, so a
    preview built from the raw prompt would be missing half the composition
    this output kind adds."""
    ctx = _PreviewCtx(_sheet_form(projection="isometric"))
    settings_2d._preview(ctx)
    subject = ctx.calls[0][0][2]
    assert "mossy dungeon" in subject
    assert "isometric" in subject
    assert tilesheet.DETAIL_CLAUSE in subject


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


def test_every_offered_view_validates():
    """The mirror of the tile-size sweep beside it. The old suite had no such
    sweep for views, which is how a third one could have been added and gone
    unexercised by the form."""
    from warlock.service import tilesheets as svc_tilesheets

    for view in svc_tilesheets.VIEWS:
        problems = settings_2d.validate(_sheet_form(projection=view))
        assert not [p for p in problems if p.field == "projection"], view

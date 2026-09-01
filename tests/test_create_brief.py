"""Create's command bar: what it draws, where, and what it refuses to draw.

The bar is drawn from ``main._build_ui`` rather than from ``panes/``, so the
pane smoke sweep does not reach it. These are its own gates.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from warlock.studio import create_brief, create_stages
from warlock.studio.state import default_form_2d


def _body(fn) -> str:
    """A function's source with its docstring taken off.

    These assertions are about the *code*: a prose mention of ``TYPE_W`` in the
    docstring explaining why it is absent would otherwise fail the test that
    checks it is absent.
    """
    source = inspect.getsource(fn)
    doc = inspect.getdoc(fn)
    if doc:
        for line in doc.splitlines():
            source = source.replace(line, "")
    return source


def _state(stage="reference", mode="create", **kw):
    return SimpleNamespace(
        mode=mode,
        create_stage=stage,
        form_2d=default_form_2d(),
        field_errors={},
        clear_field_error=lambda _f: None,
        **kw,
    )


# --- where it draws ---------------------------------------------------------


def test_the_bar_draws_on_the_reference_stage_only():
    """Mesh, Rig, Pose and Export draw no bar and start their columns higher.

    ``create_stages``' own rule about the rail, applied to the bar under it: a
    row with one live control and three dead ones is not honest, and an inert
    bar is worse than an absent one.
    """
    for stage in create_stages.STAGES:
        ctx = SimpleNamespace(state=_state(stage))
        assert create_brief.shows(ctx) is (stage == "reference"), stage


def test_the_bar_draws_in_no_other_mode():
    """``create_stage`` is not cleared on a mode switch -- coming back from
    Inker lands where you left -- so asking the stage alone would put Create's
    brief across the top of another workspace."""
    for mode in ("inker", "clay", "plotter", "home", "library"):
        ctx = SimpleNamespace(state=_state("reference", mode=mode))
        assert create_brief.shows(ctx) is False, mode


# --- what it holds ----------------------------------------------------------


def test_the_bar_holds_exactly_the_four_controls_of_a_brief():
    source = inspect.getsource(create_brief.draw)
    for call in ("_type(ctx", "_prompt(ctx", "_count(ctx", "_generate(ctx"):
        assert call in source, call


def test_the_recipe_is_not_in_the_bar():
    """The split is the design: the bar is *what to make*, the column is *how*.

    A control belongs to exactly one of them, which is the one-owner rule the
    two generation panes already keep.
    """
    source = inspect.getsource(create_brief)
    for absent in ("base_model", "style_lora", "lora_weight", "negative_prompt", "ref_path"):
        assert absent not in source, absent


def test_the_count_is_the_service_capped_row():
    from warlock.service.validation import MAX_REFERENCE_COUNT

    assert create_brief._COUNTS == (1, 2, 4, 8)
    assert max(create_brief._COUNTS) == MAX_REFERENCE_COUNT


def test_every_generation_type_has_a_hint():
    from warlock.studio import create_assets

    offered = {key for key, _label in create_assets.ASSET_TYPE_OPTIONS}
    assert set(create_brief._TYPE_HINTS) == offered


# --- the count is hidden where the door refuses a batch ---------------------


@pytest.mark.parametrize("asset_type", ["tileset", "sprite_sheet"])
def test_a_sheet_hides_the_count_rather_than_offering_refusals(asset_type):
    """Both sheet doors refuse a batch and say why, so four radios of which
    three are refusals would be a control offering what the thing behind it
    will not do."""
    from warlock.studio import create_assets

    form = default_form_2d()
    form["asset_type"] = asset_type
    create_assets.sync_legacy_fields(form)
    assert form["output"] == "sheet"
    assert form["count"] == 1
    # The width calculation is the observable half of "the control is skipped".
    source = inspect.getsource(create_brief.draw)
    assert "if show_count:" in source
    assert "_count(ctx, form)" in source
    # And _row_widths gives the count no width at all for a sheet.
    assert "0.0 if sheet else" in inspect.getsource(create_brief._row_widths)


def test_the_row_gives_way_in_a_stated_order():
    """The prompt shrinks, then the count is dropped, and the type and Generate
    never give way -- because Generate is the control the bar exists to keep
    visible, and ``same_line`` past the pane edge draws a control nowhere.

    The width is measured after the type combo's ``same_line``, so
    ``get_content_region_avail`` has already taken the combo off; subtracting
    ``TYPE_W`` again double-counted it and the floor underneath turned the
    shortfall into a clipped Generate at the resize floor.
    """
    body = _body(create_brief._row_widths)
    assert "TYPE_W" not in body, "the combo is already off the avail"
    assert "GENERATE_W" in body and "COUNT_W" in body
    assert "PROMPT_MIN_W" in body
    # The count is the one that goes, and only after the prompt has bottomed.
    assert "count = 0.0" in body


# --- the anchors the guided tour points at ----------------------------------


def test_the_tour_anchors_moved_with_the_controls():
    """``studio/tour/scripts.py`` binds steps to ``create/prompt`` and
    ``create/generate``. The tour reads positions and never computes them, so
    the anchors keep working precisely because they are marked at the controls'
    new home rather than left behind at the old one."""
    source = inspect.getsource(create_brief)
    assert 'anchors.mark("create/prompt")' in source
    assert 'anchors.mark("create/generate")' in source

    from warlock.studio.panes import settings_2d

    pane = inspect.getsource(settings_2d)
    assert 'anchors.mark("create/prompt")' not in pane
    assert 'anchors.mark("create/generate")' not in pane


def test_the_tours_still_name_anchors_something_marks():
    """Both directions, so a rename on either side fails here rather than as a
    tour step pointing at nothing."""
    from warlock.studio import anchors as anchors_mod  # noqa: F401
    from warlock.studio.tour import scripts

    wanted = {
        step.anchor
        for tour in scripts.TOURS
        for step in tour.steps
        if step.anchor and step.anchor.startswith("create/")
    }
    marked = set()
    for mod in (create_brief,):
        for line in inspect.getsource(mod).splitlines():
            if "anchors.mark(" in line:
                marked.add(line.split('"')[1])
    # Every create/ anchor the tours name is either marked here or elsewhere in
    # Create; the two this move touched must be here.
    assert {"create/prompt", "create/generate"} <= marked
    assert {"create/prompt", "create/generate"} <= wanted


# --- the pane registration --------------------------------------------------


def test_the_bar_is_a_registered_pane_not_a_bare_row():
    """``layout.pane`` is what puts it in ``FRAME_PANES``, which is what gives
    it the role fill, the divider, ``guard``'s isolation and -- the reason that
    matters downstream -- a slot for ``probe._pane_at``. Drawn bare, the bar's
    four controls are censused against the empty-string pane, which reads as
    four controls nobody owns."""
    from warlock.studio import main

    source = inspect.getsource(main.App._build_ui)
    assert 'layout_mod.pane(\n                            "brief",' in source
    assert "create_brief.shows(ctx)" in source
    assert "create_brief.draw(ctx)" in source

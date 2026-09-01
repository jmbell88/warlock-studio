"""The Create plan is a view of the existing request, not a second planner."""

from __future__ import annotations

import pytest

from warlock.studio import create_assets, generation_workspace
from warlock.studio.state import default_form_2d


@pytest.mark.parametrize(
    ("asset_type", "minimum_generations"),
    (
        ("image", 1),
        ("model_3d", 1),
        ("seamless_material", 1),
        ("tileset", 1),
        # A sheet always includes its character reference plus its sheet work.
        ("sprite_sheet", 2),
    ),
)
def test_generation_plan_covers_every_create_outcome(asset_type, minimum_generations):
    form = default_form_2d()
    form["asset_type"] = asset_type
    form["generation_type"] = asset_type
    create_assets.sync_legacy_fields(form)

    plan = generation_workspace.plan_for(form)

    assert plan.candidates >= 1
    assert plan.generations >= minimum_generations
    assert plan.duration
    assert plan.stages


def test_sprite_plan_states_the_compound_work_plainly():
    form = default_form_2d()
    form["asset_type"] = form["generation_type"] = "sprite_sheet"
    create_assets.sync_legacy_fields(form)

    plan = generation_workspace.plan_for(form)

    assert "character reference" in plan.stages
    assert "sheet generation" in plan.stages


# --- nothing in the tray is drawn where it cannot be pressed -----------------


def test_the_tray_shows_one_whole_row_rather_than_two_half_rows():
    """The tray is a fixed-height strip. Six results filled its three columns
    twice over, and the second row's cards were drawn with their actions below
    the fold, where nothing can press them.

    ``/exercise-mode create`` reported fifteen clipped controls, every one of
    them a result-card action. No test could: a clipped button is still drawn,
    and the smoke suite only asks whether a pane builds.
    """
    import inspect

    from warlock.studio import generation_workspace as gw

    assert gw._RESULT_COLUMNS == 3
    source = inspect.getsource(gw._recent_results)
    assert "_RESULT_COLUMNS" in source, "the cap and the grid width are one fact"
    # And the grid is built from the same number, so they cannot drift.
    assert "_RESULT_COLUMNS" in inspect.getsource(gw._result_grid)


def test_the_result_actions_are_two_per_row():
    """Four full-width buttons under a 72 dp thumbnail make a card taller than
    the strip that holds it."""
    import inspect

    from warlock.studio import generation_workspace as gw

    source = inspect.getsource(gw._result_card)
    assert "_half_width()" in source
    assert "(-1, 0)" not in source, "a full-width action is a row of its own"
    assert source.count("imgui.same_line()") >= 2


def test_the_candidate_grid_scrolls_rather_than_truncating():
    """A count of 8 means eight candidates and choosing between them is the
    whole purpose, so this grid cannot be trimmed to a row the way the results
    grid is."""
    import inspect

    from warlock.studio import generation_workspace as gw

    source = inspect.getsource(gw._candidate_grid)
    assert 'begin_child("generation-candidate-scroll"' in source
    assert "end_child()" in source

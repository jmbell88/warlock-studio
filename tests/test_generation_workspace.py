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

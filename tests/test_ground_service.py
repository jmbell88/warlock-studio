"""The AI ground set's door: every refusal, and what a good request writes.

The geometry rules here are *copies* -- ``service/`` may not import
``studio/`` -- so the drift pins at the bottom of this file are the point of it:
if somebody raises ``tilegen.MAX_TERRAINS`` and forgets this module, the copy
goes red rather than silently refusing a set the compositor would have built.
"""

from __future__ import annotations

import pytest

from warlock import models
from warlock.service import Invalid, grounds
from warlock.service._jobs_resubmit import rerun_job
from warlock.service.validation import DERIVED_PARAMS
from warlock.studio.plotter import project, tilegen


def _terrains(count: int = 2):
    names = ("stone", "water", "lava", "moss", "sand", "ice", "mud", "ash", "bone")
    return [
        {
            "name": names[i],
            "material": f"{names[i]} surface",
            "edge": "",
            "color": [40 + i, 80, 120, 255],
        }
        for i in range(count)
    ]


def _create(svc, **overrides):
    kwargs = {
        "theme": "dungeon",
        "terrains": _terrains(2),
        "tile_w": 32,
        "tile_h": 32,
        "projection": "orthogonal",
    }
    kwargs.update(overrides)
    return grounds.create_ground_set(svc, **kwargs)


# -- the good path -----------------------------------------------------------


def test_a_good_request_writes_a_queued_row_and_no_files(svc):
    made = _create(svc, seed=42)
    row = svc.store.get(made["id"])
    assert row["kind"] == "ground_set"
    assert row["stage"] == "ground"
    assert row["status"] == "queued"
    assert row["params"]["seed"] == 42
    assert row["params"]["base_model"] == grounds.GROUND_BASE_MODEL
    assert row["params"]["style_lora"] == models.PIXEL_SHEET_LORA
    # Nothing is on disk: the worker makes the job directory, not the door.
    assert not (svc.job_dir(made["id"]) / "input.png").exists()


def test_the_ground_block_carries_the_geometry_and_the_rows(svc):
    made = _create(svc, tile_w=16, tile_h=8, projection="isometric", border=3)
    block = svc.store.get(made["id"])["params"]["ground"]
    assert block["tile_w"] == 16
    assert block["tile_h"] == 8
    assert block["projection"] == "isometric"
    assert block["border"] == 3
    assert [entry["name"] for entry in block["terrains"]] == ["stone", "water"]


def test_the_row_is_named_so_a_library_list_says_what_it_is(svc):
    made = _create(svc)
    assert svc.store.get(made["id"])["name"] == "dungeon ground set"


def test_the_reply_says_what_it_will_cost(svc):
    made = _create(svc, terrains=_terrains(3))
    assert made["terrains"] == 3
    assert made["textures"] == 6


def test_an_absent_seed_is_filled_in(svc):
    row = svc.store.get(_create(svc)["id"])
    assert isinstance(row["params"]["seed"], int)


def test_an_absent_border_is_left_for_the_compositor(svc):
    """Zero means "whatever suits this tile" -- the pixel geometry is the
    plotter package's rule, and restating it here would be a second copy."""
    assert svc.store.get(_create(svc)["id"])["params"]["ground"]["border"] == 0


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,field",
    [
        ({"theme": "  "}, "theme"),
        ({"theme": "x" * 2000}, "theme"),
        ({"negative_prompt": "x" * 2000}, "negative_prompt"),
        ({"terrains": []}, "terrains"),
        ({"terrains": _terrains(9)}, "terrains"),
        ({"tile_w": 3}, "tile_w"),
        ({"tile_h": 3}, "tile_h"),
        ({"tile_w": 4096}, "tile_w"),
        ({"projection": "hexagonal"}, "projection"),
        ({"colors": 7}, "colors"),
        ({"border": 16}, "border"),
        ({"border": -1}, "border"),
        ({"seed": -1}, "seed"),
    ],
)
def test_every_bad_value_is_refused_with_the_control_it_names(svc, overrides, field):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, **overrides)
    assert excinfo.value.field == field


def test_a_nameless_terrain_is_refused(svc):
    rows = _terrains(2)
    rows[1]["name"] = "   "
    with pytest.raises(Invalid) as excinfo:
        _create(svc, terrains=rows)
    assert excinfo.value.field == "terrains"
    assert "2" in str(excinfo.value)


def test_two_terrains_cannot_share_a_name(svc):
    rows = _terrains(2)
    rows[1]["name"] = "Stone"
    with pytest.raises(Invalid) as excinfo:
        _create(svc, terrains=rows)
    assert "Stone" in str(excinfo.value)


def test_a_paragraph_of_material_is_refused(svc):
    rows = _terrains(1)
    rows[0]["material"] = "x" * (grounds.MAX_MATERIAL + 1)
    with pytest.raises(Invalid) as excinfo:
        _create(svc, terrains=rows)
    assert excinfo.value.field == "terrains"


def test_a_colour_outside_the_byte_range_is_refused(svc):
    rows = _terrains(1)
    rows[0]["color"] = [0, 0, 0, 999]
    with pytest.raises(Invalid):
        _create(svc, terrains=rows)


def test_a_refusal_costs_no_row(svc):
    with pytest.raises(Invalid):
        _create(svc, terrains=[])
    assert svc.store.list(limit=10) == []


def test_missing_weights_are_refused_with_an_install_remedy(svc):
    lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    (svc.config.t2i_model_root / "loras" / lora.filename).unlink()
    with pytest.raises(Invalid) as excinfo:
        _create(svc)
    assert "Settings" in str(excinfo.value)


# -- what a ground set is not ------------------------------------------------


def test_a_ground_set_cannot_be_rerolled_from_the_library(svc):
    """A reroll would copy the whole ``ground`` block while changing the one
    seed every texture derives from, and would skip this door's weights
    admission entirely."""
    made = _create(svc)
    svc.store.finish(made["id"], "done")
    with pytest.raises(Invalid) as excinfo:
        rerun_job(svc, made["id"])
    assert "Plotter" in str(excinfo.value)


def test_a_ground_set_cannot_be_promoted_to_a_mesh(svc):
    from warlock.service._jobs_resubmit import promote_to_model

    made = _create(svc)
    svc.store.finish(made["id"], "done")
    with pytest.raises(Invalid) as excinfo:
        promote_to_model(svc, made["id"])
    # The sentence, not just the type. ``create_stages.available("mesh", ...)``
    # predicts this refusal *verbatim* on the disabled Mesh rail segment, so a
    # reworded refusal has to fail here rather than quietly leave a tooltip
    # saying something the toast no longer says.
    assert "not a reference" in str(excinfo.value)


def test_a_ground_set_is_refused_by_both_rerun_modes(svc):
    """Not just ``reroll``. The library's retry ladder picks the mode from
    ``_remeshable``, so a kind refused in one mode and accepted in the other
    would be a Try again button that worked or failed depending on whether the
    row happened to carry an ``input.png``."""
    made = _create(svc)
    svc.store.finish(made["id"], "done")
    for mode in ("reroll", "remesh"):
        with pytest.raises(Invalid) as excinfo:
            rerun_job(svc, made["id"], mode=mode)
        assert "Plotter" in str(excinfo.value), mode


def test_the_studio_never_offers_an_action_the_service_refuses_for_a_ground_set():
    """The three predicates that decide what a ground-set card offers.

    ``test_the_worker_report_is_stripped_on_a_rerun`` used to sit here asserting
    only ``"ground_report" in DERIVED_PARAMS`` -- true by inspection of the
    constant, and about a rerun path this kind is refused from anyway. What
    actually needed pinning is the other side: every one of these fell through
    a stage-keyed table when the kind landed, and each fell through *silently*.
    """
    from warlock.studio import create_stages, state
    from warlock.studio.panes import library

    row = {
        "id": "g1",
        "kind": "ground_set",
        "stage": "ground",
        "status": "done",
        "files": ["input.png"],
    }
    assert not library._remeshable(row), "Remesh reaches rerun_job, which refuses it"
    assert state.primary_action(row) == "open"
    # And an errored one offers nothing: the ladder's default is "retry", which
    # routes into the same refused call.
    assert state.primary_action({**row, "status": "error"}) is None
    assert create_stages.available("mesh", row) == "this job is not a reference"


def test_the_worker_report_is_a_derived_param():
    """``ground_report`` is what the worker records about the artifact, so it
    joins ``DERIVED_PARAMS`` by the standing rule. Kept as an inspection
    assertion because that is honestly all it is -- the rerun path it would
    matter on refuses this kind outright."""
    assert "ground_report" in DERIVED_PARAMS


# -- the copies, pinned to what they copy ------------------------------------


def test_the_terrain_ceiling_matches_the_generator():
    assert grounds.MAX_TERRAINS == tilegen.MAX_TERRAINS


def test_the_tile_edge_range_matches_the_generator():
    assert grounds.MAX_TILE_EDGE == tilegen.MAX_TILE_EDGE
    # Four is ``GenSpec``'s floor, stated in its own refusal rather than a
    # constant, so this is the pin for it.
    with pytest.raises(ValueError):
        tilegen.GenSpec(tile_w=grounds.MIN_TILE_EDGE - 1)
    tilegen.GenSpec(tile_w=grounds.MIN_TILE_EDGE, tile_h=grounds.MIN_TILE_EDGE)


def test_the_projection_list_matches_the_lattice():
    assert grounds.PROJECTIONS == project.PROJECTIONS


def test_the_options_block_reports_the_same_ceilings():
    options = grounds.ground_options()
    assert options["max_terrains"] == grounds.MAX_TERRAINS
    assert options["projections"] == list(grounds.PROJECTIONS)
    assert options["colors"] == list(grounds.COLOR_CHOICES)

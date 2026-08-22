"""The flat Create asset registry and its compatibility boundary."""

from __future__ import annotations

import json

import pytest

from warlock import models
from warlock.studio import create_assets, settings
from warlock.studio.panes import settings_2d
from warlock.studio.state import default_form_2d, primary_action


def test_the_asset_registry_is_the_approved_flat_list():
    assert create_assets.ASSET_TYPE_OPTIONS == (
        ("image_2d", "2D Image"),
        ("model_3d", "3D Model"),
        ("seamless_tile", "Seamless Tile"),
        ("tileset_top_down", "Top-Down Tileset"),
        ("tileset_three_quarter", "3/4 Tileset"),
        ("tileset_isometric", "Isometric Tileset"),
        ("sprite_turnaround", "Sprite Turnaround"),
        ("sprite_walk", "Sprite Walk Cycle"),
    )


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ({"output": "tile"}, "seamless_tile"),
        ({"output": "sheet", "projection": "orthogonal"}, "tileset_top_down"),
        ({"output": "sheet", "projection": "three_quarter"}, "tileset_three_quarter"),
        ({"output": "sheet", "projection": "isometric"}, "tileset_isometric"),
        (
            {"output": "sheet", "sheet_type": "sprite", "sheet_layout": "walk"},
            "sprite_walk",
        ),
    ],
)
def test_legacy_switches_migrate_without_guessing(legacy, expected):
    assert create_assets.legacy_asset_type(legacy) == expected


def test_a_new_asset_type_overrides_contradictory_legacy_switches():
    form = default_form_2d()
    form.update(asset_type="tileset_isometric", output="reference", projection="top_down")
    create_assets.sync_legacy_fields(form)
    assert (form["output"], form["projection"], form["count"]) == (
        "sheet", "isometric", 1,
    )


def test_a_corrupt_new_asset_type_does_not_resurrect_legacy_switches(tmp_path):
    payload = {
        "version": settings.VERSION,
        "data": {
            "form_2d": {
                "asset_type": "not-a-type",
                "output": "sheet",
                "sheet_type": "sprite",
                "sheet_layout": "walk",
            }
        },
    }
    (tmp_path / settings.FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = settings.Settings.load(tmp_path)
    restored = settings.restore_form(default_form_2d(), loaded.get("form_2d"))
    assert restored["asset_type"] == create_assets.DEFAULT_ASSET_TYPE
    assert restored["output"] == "reference"


def test_a_corrupt_job_asset_type_is_also_authoritative_over_legacy_shape():
    assert create_assets.asset_type_from_params(
        {
            "asset_type": "not-a-type",
            "sprite_sheet": {"sheet_type": "walk"},
        }
    ) == create_assets.DEFAULT_ASSET_TYPE


def test_the_visible_model_default_is_the_model_that_will_run():
    assert default_form_2d()["base_model"] == models.DEFAULT_BASE_MODEL


def test_submit_persists_type_and_intent():
    form = default_form_2d()
    form.update(prompt="a barrel", asset_type="image_2d")
    kwargs = settings_2d.submit_kwargs(form)
    assert kwargs["asset_type"] == "image_2d"
    assert kwargs["asset_intent"] == "refine_2d"


def test_result_actions_follow_the_persisted_intent():
    def job(intent, *, stage="reference", files=("input.png",), kind="text"):
        return {
            "status": "done", "stage": stage, "files": list(files), "kind": kind,
            "params": {"asset_intent": intent},
        }

    assert primary_action(job("refine_2d")) == "inker"
    assert primary_action(job("reconstruct_3d")) == "inker"
    assert primary_action(job("tileset", stage="tilesheet", kind="tile_sheet")) == "plotter"
    assert primary_action(job("sprite")) == "inker"
    assert primary_action(job("reconstruct_3d", stage="model", files=("model.glb",))) == "clay"


def test_corrupt_persisted_form_values_fall_back_safely(tmp_path):
    payload = {
        "version": settings.VERSION,
        "data": {
            "form_2d": {
                "asset_type": "not-a-type",
                "base_model": "missing",
                "count": 999999,
                "lora_weight": float("nan"),
            }
        },
    }
    (tmp_path / settings.FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = settings.Settings.load(tmp_path)
    restored = settings.restore_form(default_form_2d(), loaded.get("form_2d"))
    defaults = default_form_2d()
    assert restored["asset_type"] == create_assets.DEFAULT_ASSET_TYPE
    assert restored["base_model"] == defaults["base_model"]
    assert restored["count"] == defaults["count"]
    assert restored["lora_weight"] == defaults["lora_weight"]

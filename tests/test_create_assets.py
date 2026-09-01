"""The flat Create asset registry and its compatibility boundary."""

from __future__ import annotations

import json

import pytest

from warlock import models
from warlock.studio import create_assets, settings
from warlock.studio.panes import settings_2d
from warlock.studio.state import default_form_2d, primary_action


def test_the_asset_registry_is_the_approved_flat_list():
    """Five offered types, not eight.

    The three tilesets differed only by projection and the two sprite sheets
    only by layout, and both of those are fields on the form already -- so the
    selector was asking the same question twice and the answers could
    contradict each other. Projection and layout stayed; the types folded.
    """
    assert create_assets.ASSET_TYPE_OPTIONS == (
        ("image", "Image"),
        ("3d_model", "3D Model"),
        ("seamless_material", "Seamless Material"),
        ("tileset", "Tileset"),
        ("sprite_sheet", "Sprite Sheet"),
    )


def test_the_retired_keys_still_resolve():
    """A saved job or a restored form naming an old key must still open. They
    are aliases, not options: readable, never offered."""
    for old, new in {
        "image_2d": "image",
        "seamless_tile": "seamless_material",
        "tileset_top_down": "tileset",
        "tileset_three_quarter": "tileset",
        "tileset_isometric": "tileset",
        "sprite_turnaround": "sprite_sheet",
        "sprite_walk": "sprite_sheet",
    }.items():
        assert create_assets.ASSET_TYPES[old].key == new
        assert old not in dict(create_assets.ASSET_TYPE_OPTIONS)


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ({"output": "tile"}, "seamless_material"),
        # Projection no longer picks the type -- it is its own field, and the
        # three tileset keys it used to choose between are one key now.
        ({"output": "sheet", "projection": "orthogonal"}, "tileset"),
        ({"output": "sheet", "projection": "three_quarter"}, "tileset"),
        ({"output": "sheet", "projection": "isometric"}, "tileset"),
        (
            {"output": "sheet", "sheet_type": "sprite", "sheet_layout": "walk"},
            "sprite_sheet",
        ),
        # Old "Object" meant a reconstruction reference, not a standalone image.
        ({"output": "reference"}, "3d_model"),
    ],
)
def test_legacy_switches_migrate_without_guessing(legacy, expected):
    assert create_assets.legacy_asset_type(legacy) == expected


def test_a_new_asset_type_overrides_contradictory_legacy_switches():
    """The type owns the door it opens; projection is no longer part of it.

    ``output`` is derived from the type and is corrected. ``projection`` is a
    field the user sets, so a retired key that used to carry one no longer
    speaks for it -- the form's own value stands.
    """
    form = default_form_2d()
    form.update(asset_type="tileset_isometric", output="reference", projection="top_down")
    create_assets.sync_legacy_fields(form)
    assert (form["output"], form["projection"], form["count"]) == (
        "sheet", "top_down", 1,
    )
    assert form["asset_type"] == "tileset"


def test_a_retired_tileset_key_does_not_overwrite_a_real_projection():
    """The other half of the same rule: folding the key must not cost a job
    the projection it was actually saved with."""
    form = default_form_2d()
    form.update(asset_type="tileset_isometric", projection="isometric")
    create_assets.sync_legacy_fields(form)
    assert (form["asset_type"], form["projection"]) == ("tileset", "isometric")


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
    # Persisted under today's key, not the retired one it was written with.
    assert kwargs["asset_type"] == "image"
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


def test_the_two_registries_agree_on_every_key():
    """``create_assets`` and ``generation`` are two spellings of one choice, and
    they disagreed about exactly one of them: this registry said ``model_3d``
    where ``generation.GENERATION_TYPES`` said ``3d_model``. Harmless only
    because every reader accepted both -- ``settings_2d.draw`` ran a membership
    test that failed on the default type every frame and wrote it straight
    back. Unified onto ``generation``'s spelling, with the old one kept as an
    alias so a persisted form and a stored job row still resolve.
    """
    from warlock import generation

    assert tuple(key for key, _ in create_assets.ASSET_TYPE_OPTIONS) == generation.GENERATION_TYPES
    assert create_assets.ASSET_TYPE_OPTIONS == generation.GENERATION_TYPE_OPTIONS
    assert create_assets.DEFAULT_ASSET_TYPE in generation.GENERATION_TYPES
    # The old spelling reads, and reads as the new one.
    assert create_assets.ASSET_TYPES["model_3d"].key == "3d_model"


def test_a_form_persisted_under_the_old_spelling_still_restores():
    """The settings-reset class of bug, guarded at the one boundary that can
    cause it. ``_safe_form_value`` is a *boundary* check over untrusted JSON,
    and it used to test ``generation_type`` against a literal set -- so
    unifying the registries would have made every upgrading user's remembered
    type fail validation and silently revert to the default.
    """
    from warlock.studio import settings

    for key in ("asset_type", "generation_type"):
        assert settings._safe_form_value(key, "model_3d") is True, key
        assert settings._safe_form_value(key, "3d_model") is True, key
        assert settings._safe_form_value(key, "sprite_turnaround") is True, key
        assert settings._safe_form_value(key, "not_a_type") is False, key

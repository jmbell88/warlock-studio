"""Saved config vectors: what a preset carries, and which form each key goes to."""

from __future__ import annotations

from typing import Any

from warlock.studio import vector_presets
from warlock.studio.state import AppState


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def test_applying_a_vector_fills_the_form_that_owns_each_key():
    state = AppState()
    vector_presets.apply(
        state,
        {
            "genre": "fantasy",
            "lora_weight": 1,          # an int where the form holds a float
            "platform": "pc",          # the 3D pane's, not the 2D pane's
            "reference_prep": True,
            "size_m": 0.4,
            "stage": "model",          # not a setting
            "resolution": 1024,        # derived from platform
        },
    )
    assert state.form_2d["genre"] == "fantasy"
    assert state.form_2d["lora_weight"] == 1.0
    # platform is two controls; a vector's value is the geometry one.
    assert state.form_3d["platform"] == "pc"
    assert state.form_2d["platform"] == ""
    assert state.form_3d["reference_prep"] is True
    assert state.form_3d["size_m"] == 0.4


def test_a_key_no_form_has_is_ignored_rather_than_added():
    state = AppState()
    before_2d = dict(state.form_2d)
    vector_presets.apply(state, {"nonsense": 1})
    assert state.form_2d == before_2d


def test_an_uncoercible_value_leaves_the_form_alone():
    state = AppState()
    vector_presets.apply(state, {"lora_weight": "not a number"})
    assert isinstance(state.form_2d["lora_weight"], float)


def test_saving_listing_and_deleting():
    settings = _Settings()
    assert vector_presets.save_preset(settings, "  ", {"genre": "fantasy"}) is False
    assert vector_presets.list_presets(settings) == {}

    assert vector_presets.save_preset(
        settings, "chests", {"genre": "fantasy", "stage": "model"}
    ) is True
    # ``stage`` is not a setting and is not stored.
    assert vector_presets.list_presets(settings) == {"chests": {"genre": "fantasy"}}

    vector_presets.delete_preset(settings, "chests")
    assert vector_presets.list_presets(settings) == {}
    # Deleting one that is gone is not an error.
    vector_presets.delete_preset(settings, "chests")


def test_a_vector_describes_itself_without_the_non_settings():
    line = vector_presets.describe({"genre": "fantasy", "stage": "model"})
    assert line == "genre=fantasy"
    assert vector_presets.describe({"stage": "model"}) == "the defaults"

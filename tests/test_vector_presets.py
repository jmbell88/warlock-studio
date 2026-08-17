"""Applying a found config vector to the generate forms.

The save/list/delete mechanism (``studio/vector_presets.py``) retired with the
taxonomy on 2026-08-17; what survives is Review's "Apply to forms", whose
helpers moved into ``studio/review_mode.py``.
"""

from __future__ import annotations

from warlock.studio import review_mode
from warlock.studio.state import AppState


def test_applying_a_vector_fills_the_form_that_owns_each_key():
    state = AppState()
    review_mode.apply_vector(
        state,
        {
            "negative_prompt": "smooth",
            "lora_weight": 1,          # an int where the form holds a float
            "platform": "pc",          # the 3D pane's select
            "reference_prep": True,
            "size_m": 0.4,
            "stage": "model",          # not a setting
            "resolution": 1024,        # derived from platform
        },
    )
    assert state.form_2d["negative_prompt"] == "smooth"
    assert state.form_2d["lora_weight"] == 1.0
    assert state.form_3d["platform"] == "pc"
    # The 2D form has no platform key since the taxonomy retirement.
    assert "platform" not in state.form_2d
    assert state.form_3d["reference_prep"] is True
    assert state.form_3d["size_m"] == 0.4


def test_a_key_no_form_has_is_ignored_rather_than_added():
    state = AppState()
    before_2d = dict(state.form_2d)
    review_mode.apply_vector(state, {"nonsense": 1})
    assert state.form_2d == before_2d


def test_a_stale_taxonomy_key_in_an_old_findings_vector_is_ignored():
    """Old ``findings.json`` vectors carry retired keys (art_style, framing,
    genre...). They match no form key any more, so Apply must skip them
    silently rather than error or resurrect a control."""
    state = AppState()
    before_2d = dict(state.form_2d)
    before_3d = dict(state.form_3d)
    review_mode.apply_vector(
        state,
        {"art_style": "nes", "framing": "front_ortho", "genre": "fantasy",
         "material": "steel", "platform": "3d"},
    )
    assert state.form_2d == before_2d
    assert state.form_3d == {**before_3d, "platform": "3d"}


def test_an_uncoercible_value_leaves_the_form_alone():
    state = AppState()
    review_mode.apply_vector(state, {"lora_weight": "not a number"})
    assert isinstance(state.form_2d["lora_weight"], float)


def test_a_vector_describes_itself_without_the_non_settings():
    line = review_mode.describe_vector({"base_model": "sdxl", "stage": "model"})
    assert line == "base_model=sdxl"
    assert review_mode.describe_vector({"stage": "model"}) == "the defaults"

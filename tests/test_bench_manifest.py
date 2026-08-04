"""A run half-measured against two checkpoints is worse than no run: this is
what stands between that and a plausible-looking scores.json."""

from __future__ import annotations

import json

import pytest

from warlock.bench import manifest as manifest_mod
from warlock.bench import recipe as recipe_mod
from warlock.bench import suite as suite_mod
from warlock.config import Config


def _doc(tmp_path, **overrides):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i",
        **overrides,
    )
    return manifest_mod.build_manifest(
        config,
        suite_mod.load(),
        recipe_mod.load("baseline-turbo-raw"),
        stage="reference",
        items=["prop-01"],
        seeds=[42],
        started="20260803-120000",
    )


def test_a_manifest_is_json_safe(tmp_path):
    json.dumps(_doc(tmp_path))


def test_a_manifest_records_what_can_move_a_pixel(tmp_path):
    doc = _doc(tmp_path)
    assert doc["suite"]["key"] == "core-v1"
    assert doc["recipe"]["key"] == "baseline-turbo-raw"
    assert set(doc["config"]) == set(manifest_mod.CONFIG_FIELDS)
    assert "trellis_server_exe" in doc["models"]
    assert doc["versions"]["recipe"]


def test_a_manifest_round_trips_through_disk(tmp_path):
    run_dir = tmp_path / "run"
    manifest_mod.write_manifest(run_dir, _doc(tmp_path))
    assert manifest_mod.read_manifest(run_dir)["suite"]["key"] == "core-v1"


def test_an_identical_setup_is_resumable(tmp_path):
    a, b = _doc(tmp_path), _doc(tmp_path)
    assert manifest_mod.compare_manifests(a, b) == []
    manifest_mod.assert_resumable(a, b)


def test_a_started_timestamp_alone_does_not_block_a_resume(tmp_path):
    a = _doc(tmp_path)
    b = {**_doc(tmp_path), "started": "20261225-000000"}
    manifest_mod.assert_resumable(a, b)


def test_a_narrowed_item_list_does_not_block_a_resume(tmp_path):
    """A resume legitimately covers a subset of what the run planned."""
    a = _doc(tmp_path)
    b = {**_doc(tmp_path), "items": ["prop-01", "prop-02"]}
    manifest_mod.assert_resumable(a, b)


@pytest.mark.parametrize(
    "patch,needle",
    [
        ({"stage": "model"}, "stage"),
        ({"suite": {"key": "core-v2", "file": "x", "seeds": []}}, "suite.key"),
        ({"recipe": {"key": "other", "file": "x"}}, "recipe.key"),
    ],
)
def test_a_changed_setup_refuses_a_resume(tmp_path, patch, needle):
    a = _doc(tmp_path)
    with pytest.raises(manifest_mod.NotResumable, match=needle):
        manifest_mod.assert_resumable(a, {**a, **patch})


def test_a_changed_config_field_refuses_a_resume(tmp_path):
    a = _doc(tmp_path)
    b = {**a, "config": {**a["config"], "trellis_tex_res": 1024}}
    with pytest.raises(manifest_mod.NotResumable, match="trellis_tex_res"):
        manifest_mod.assert_resumable(a, b)


def test_a_swapped_checkpoint_refuses_a_resume(tmp_path):
    a = _doc(tmp_path)
    b = {**a, "models": {**a["models"], "trellis_server_exe": "s1:deadbeef"}}
    with pytest.raises(manifest_mod.NotResumable, match="trellis_server_exe"):
        manifest_mod.assert_resumable(a, b)


def test_a_prompt_compiler_change_refuses_a_resume(tmp_path):
    """The whole reason PROMPT_VERSION exists: no dependency version moves
    when PROMPT_TEMPLATE does."""
    a = _doc(tmp_path)
    b = {**a, "versions": {**a["versions"], "prompt": "2"}}
    with pytest.raises(manifest_mod.NotResumable, match="prompt"):
        manifest_mod.assert_resumable(a, b)


def test_an_unrelated_library_bump_does_not_refuse_a_resume(tmp_path):
    a = _doc(tmp_path)
    b = {**a, "versions": {**a["versions"], "trimesh": "99.0"}}
    manifest_mod.assert_resumable(a, b)

"""The PickScore preference term in candidate ranking.

The invariant everything here defends: **absence changes nothing**. A
candidate ranked before this metric existed -- or on a host that never
downloads it -- scores byte-for-byte what it always did; the preference term
only ever refines the ordering of candidates that were actually measured.
"""

from __future__ import annotations

import pytest

from warlock import fetch, models
from warlock.bench import metrics
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import rank
from warlock.queue import Worker

GOOD_REPORT = {"ok": True, "occupancy": 0.78, "warnings": [], "components": 1}


def test_absence_changes_nothing():
    assert rank.score(GOOD_REPORT) == rank.score(GOOD_REPORT, None, None)
    with_anchor = rank.score(GOOD_REPORT, anchor_cosine=0.5)
    assert "preference" not in with_anchor
    assert with_anchor["score"] == pytest.approx(
        rank.COMPOSITION_WEIGHT * 1.0 + rank.ANCHOR_WEIGHT * 0.75
    )


def test_preference_component_is_a_clamped_window():
    assert rank.preference_component(rank.PREFERENCE_LOW - 5) == 0.0
    assert rank.preference_component(rank.PREFERENCE_HIGH + 5) == 1.0
    assert rank.preference_component(
        (rank.PREFERENCE_LOW + rank.PREFERENCE_HIGH) / 2
    ) == pytest.approx(0.5)


def test_a_measured_preference_is_recorded_raw_and_moves_the_score():
    liked = rank.score(GOOD_REPORT, None, rank.PREFERENCE_HIGH)
    disliked = rank.score(GOOD_REPORT, None, rank.PREFERENCE_LOW)
    assert liked["preference"] == rank.PREFERENCE_HIGH
    assert liked["score"] > disliked["score"]
    # The blend, exactly: base 1.0 pulled toward the preference component.
    assert disliked["score"] == pytest.approx(1.0 - rank.PREFERENCE_WEIGHT)
    assert liked["score"] == pytest.approx(1.0)


def test_the_blend_composes_with_the_anchor():
    out = rank.score(GOOD_REPORT, anchor_cosine=0.0, preference=20.0)
    base = rank.COMPOSITION_WEIGHT * 1.0 + rank.ANCHOR_WEIGHT * 0.5
    expected = (1 - rank.PREFERENCE_WEIGHT) * base + (
        rank.PREFERENCE_WEIGHT * rank.preference_component(20.0)
    )
    assert out["score"] == pytest.approx(expected)
    assert out["anchor"] == 0.0 and out["preference"] == 20.0


def test_the_score_stays_bounded():
    for pref in (-100.0, 0.0, 18.5, 21.0, 100.0):
        for cosine in (None, -1.0, 1.0):
            out = rank.score(GOOD_REPORT, cosine, pref)
            assert 0.0 <= out["score"] <= 1.0


def test_pickscore_is_a_pinned_registry_row():
    entry = fetch.find("metric:pickscore")
    assert entry is not None
    assert entry.fetch[0].revision


def test_availability_is_the_weights_not_the_directory(tmp_path):
    cfg = Config(t2i_model_root=tmp_path)
    assert not metrics.pickscore_available(cfg)
    d = tmp_path / models.METRIC_MODELS["pickscore"].dir_name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    assert not metrics.pickscore_available(cfg)  # config alone is not a model
    (d / "model.safetensors").write_bytes(b"x")
    assert metrics.pickscore_available(cfg)


# --- the worker's wiring ------------------------------------------------------


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


def test_rank_reference_scores_against_the_composed_prompt(worker, tmp_path, monkeypatch):
    from PIL import Image

    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(image)

    asked: list[str] = []

    monkeypatch.setattr(metrics, "pickscore_available", lambda cfg: True)

    def fake_pref(prompt, path, cfg, device):
        asked.append(prompt)
        assert device == "cpu"
        return 21.5

    monkeypatch.setattr(metrics, "preference_score", fake_pref)
    monkeypatch.setattr(worker.config, "rank_candidates", True)

    out = worker._rank_reference(
        image, {"reference_report": GOOD_REPORT, "composed_prompt": "a barrel, ornate"}
    )
    # The composed prompt -- expansion and template included -- not the bare one.
    assert asked == ["a barrel, ornate"]
    assert out["preference"] == 21.5


def test_a_broken_preference_model_costs_the_number_not_the_rank(worker, tmp_path, monkeypatch):
    from PIL import Image

    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(image)

    monkeypatch.setattr(metrics, "pickscore_available", lambda cfg: True)

    def boom(prompt, path, cfg, device):
        raise RuntimeError("corrupt weights")

    monkeypatch.setattr(metrics, "preference_score", boom)
    monkeypatch.setattr(worker.config, "rank_candidates", True)

    out = worker._rank_reference(image, {"reference_report": GOOD_REPORT})
    assert "preference" not in out
    assert out["score"] == rank.score(GOOD_REPORT)["score"]


# --- real weights, gpu lane only ----------------------------------------------


@pytest.mark.gpu
def test_real_pickscore_prefers_the_matching_prompt():
    """CPU inference in the gpu lane, the expander's rule: that lane sees the
    real ~/.warlock. A minimal sanity: the same image scores higher for a
    prompt that describes it than for one that does not."""
    config = Config()
    if not metrics.pickscore_available(config):
        pytest.skip("PickScore weights not downloaded")
    import tempfile
    from pathlib import Path

    from PIL import Image, ImageDraw

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "red-circle.png"
        im = Image.new("RGB", (256, 256), (255, 255, 255))
        ImageDraw.Draw(im).ellipse((64, 64, 192, 192), fill=(220, 30, 30))
        im.save(path)
        matching = metrics.preference_score(
            "a red circle on a white background", path, config, device="cpu"
        )
        mismatched = metrics.preference_score(
            "a photograph of a city street at night", path, config, device="cpu"
        )
    assert matching is not None and mismatched is not None
    assert matching > mismatched

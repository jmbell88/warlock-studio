"""Scoring a run, and the drift guard that decides two runs are comparable.

The half the package always claimed: ``--render`` produced eight views per
mesh "for measurement" and nothing read them, ``score_view`` was called only by
``calibrate``, and the ``scores.json`` the manifest docstring names was written
by nothing -- so an A/B could be generated but not decided.

The metrics themselves are covered by test_bench_metrics; here the metric is
stubbed, because what is under test is the aggregation, the skip reasons and
the comparison.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from warlock import provenance
from warlock.bench import manifest as manifest_mod
from warlock.bench import runner as runner_mod
from warlock.bench import score as score_mod


@pytest.fixture
def scored(monkeypatch):
    """A deterministic stand-in for the torch-backed metrics."""
    scores = {}

    def fake_available(config=None):
        return ("silhouette_iou",)

    def fake_score_view(reference, render, config=None):
        return {"silhouette_iou": scores.get(render.parent.parent.name, 0.5)}

    from warlock.bench import metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "available", fake_available)
    monkeypatch.setattr(metrics_mod, "score_view", fake_score_view)
    return scores


def _run_dir(tmp_path, records, *, views_for=(), manifest=None):
    run_dir = tmp_path / "20260101-000000-r-s"
    (run_dir / "items").mkdir(parents=True)
    (run_dir / runner_mod.ITEMS_FILE).write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    for record in records:
        item_dir = run_dir / "items" / record["key"]
        item_dir.mkdir(parents=True, exist_ok=True)
        (item_dir / "input.png").write_bytes(b"png")
        if record["key"] in views_for:
            (item_dir / "views").mkdir(exist_ok=True)
            (item_dir / "views" / "0000.png").write_bytes(b"png")
    if manifest is not None:
        manifest_mod.write_manifest(run_dir, manifest)
    return run_dir


def _record(key, status="done", category="prop"):
    return {"key": key, "item": key.split("--")[0], "category": category,
            "seed": 1, "status": status}


# --- what gets scored ---------------------------------------------------------


def test_a_run_with_rendered_views_is_scored(tmp_path, scored):
    run_dir = _run_dir(tmp_path, [_record("prop-01--s1")], views_for=("prop-01--s1",))

    doc = score_mod.score_run(run_dir)

    assert doc["summary"]["scored"] == 1
    assert doc["units"][0]["scores"]["silhouette_iou"] == 0.5
    assert (run_dir / score_mod.FILENAME).exists()


def test_scores_json_is_written_where_the_docs_say(tmp_path, scored):
    run_dir = _run_dir(tmp_path, [_record("prop-01--s1")], views_for=("prop-01--s1",))
    score_mod.score_run(run_dir)
    assert score_mod.read_scores(run_dir)["run"] == run_dir.name


def test_a_unit_with_no_views_says_so_rather_than_failing(tmp_path, scored):
    """A reference-stage run has no mesh, and a model run without --render has
    no views. Naming which is the difference between "rerun with --render" and
    "this is fine"."""
    run_dir = _run_dir(tmp_path, [_record("prop-01--s1")])

    doc = score_mod.score_run(run_dir)

    assert doc["units"][0]["skipped"] == "no rendered views"
    assert doc["summary"]["scored"] == 0


def test_an_errored_unit_is_not_scored(tmp_path, scored):
    run_dir = _run_dir(
        tmp_path, [_record("prop-01--s1", status="error")], views_for=("prop-01--s1",)
    )
    doc = score_mod.score_run(run_dir)
    assert doc["units"][0]["skipped"] == "status error"


def test_a_run_with_no_items_is_refused(tmp_path, scored):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="no items"):
        score_mod.score_run(run_dir)


def test_a_retried_unit_is_scored_once_at_its_latest_status(tmp_path, scored):
    run_dir = _run_dir(
        tmp_path,
        [_record("prop-01--s1", status="error"), _record("prop-01--s1")],
        views_for=("prop-01--s1",),
    )
    doc = score_mod.score_run(run_dir)
    assert len(doc["units"]) == 1
    assert doc["units"][0]["scores"]


# --- the aggregate ------------------------------------------------------------


def test_the_summary_reports_the_count_beside_every_figure(tmp_path, scored):
    """A mean over three of a hundred units is not a run's score, and without
    the count the two are indistinguishable."""
    records = [_record(f"prop-{i:02d}--s1") for i in range(1, 4)]
    run_dir = _run_dir(tmp_path, records, views_for=("prop-01--s1",))

    doc = score_mod.score_run(run_dir)

    stats = doc["summary"]["silhouette_iou"]
    assert stats["n"] == 1
    assert doc["summary"]["units"] == 3


def test_scores_are_broken_out_by_category(tmp_path, scored):
    records = [
        _record("prop-01--s1", category="prop"),
        _record("weapon-01--s1", category="weapon"),
    ]
    run_dir = _run_dir(tmp_path, records, views_for=[r["key"] for r in records])
    scored["prop-01--s1"] = 0.9
    scored["weapon-01--s1"] = 0.1

    doc = score_mod.score_run(run_dir)

    assert doc["by_category"]["prop"]["silhouette_iou"]["mean"] == 0.9
    assert doc["by_category"]["weapon"]["silhouette_iou"]["mean"] == 0.1


def test_summary_lines_mention_an_unscorable_run(tmp_path, scored):
    run_dir = _run_dir(tmp_path, [_record("prop-01--s1")])
    lines = score_mod.summary_lines(score_mod.score_run(run_dir))
    assert any("nothing to score" in line for line in lines)


# --- the A/B ------------------------------------------------------------------


def test_two_runs_compare_metric_by_metric():
    old = {"metrics": ["silhouette_iou"], "summary": {"silhouette_iou": {"mean": 0.40, "n": 5}}}
    new = {"metrics": ["silhouette_iou"], "summary": {"silhouette_iou": {"mean": 0.55, "n": 5}}}

    lines = score_mod.compare_lines(old, new)

    assert any("0.4 -> 0.55" in line and "+0.1500" in line for line in lines)


def test_a_metric_missing_from_one_side_is_named_rather_than_averaged():
    old = {"metrics": ["dino_cosine"], "summary": {"dino_cosine": {"mean": 0.8, "n": 2}}}
    new = {"metrics": ["dino_cosine"], "summary": {"dino_cosine": {"mean": None, "n": 0}}}
    assert any("not measured on both" in line for line in score_mod.compare_lines(old, new))


def test_a_unit_whose_metrics_all_came_back_none_is_not_counted_as_scored(
    tmp_path, monkeypatch
):
    """``score_view`` always returns a dict -- ``{"silhouette_iou": None}`` when
    the render has no subject -- and a non-empty dict is truthy. So a run where
    every mesh rendered empty reported "N of N scored" with every mean None,
    and __main__'s "rerun with --render" guard never fired."""
    from warlock.bench import metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "available", lambda config=None: ("silhouette_iou",))
    monkeypatch.setattr(
        metrics_mod, "score_view", lambda r, v, config=None: {"silhouette_iou": None}
    )

    run_dir = _run_dir(tmp_path, [_record("prop-01--s1")], views_for=("prop-01--s1",))
    doc = score_mod.score_run(run_dir)

    assert doc["summary"]["units"] == 1
    assert doc["summary"]["scored"] == 0
    assert doc["summary"]["silhouette_iou"]["n"] == 0


def test_one_unscorable_unit_does_not_discard_the_whole_pass(tmp_path, monkeypatch):
    """A CUDA OOM on unit 900 of 1280 threw out 900 units of GPU work and wrote
    no scores.json at all -- the opposite of the runner's own rule that a
    render failure costs *that item* its score, not the run."""
    from warlock.bench import metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "available", lambda config=None: ("silhouette_iou",))

    def explode_on_the_second(reference, render, config=None):
        if "02" in render.parent.parent.name:
            raise RuntimeError("CUDA out of memory")
        return {"silhouette_iou": 0.5}

    monkeypatch.setattr(metrics_mod, "score_view", explode_on_the_second)

    keys = ("prop-01--s1", "prop-02--s1", "prop-03--s1")
    run_dir = _run_dir(tmp_path, [_record(k) for k in keys], views_for=keys)

    doc = score_mod.score_run(run_dir)

    assert (run_dir / score_mod.FILENAME).exists()
    assert doc["summary"]["scored"] == 2
    bad = next(u for u in doc["units"] if u["key"] == "prop-02--s1")
    assert "CUDA out of memory" in bad["error"]


def test_a_unit_whose_render_did_not_run_is_not_scored_on_an_older_meshs_frames(
    tmp_path, scored
):
    """views/ is never cleared, so a resumed unit whose second render wrote
    nothing still had the first mesh's 0000.png sitting there -- and _pair
    consulted only the directory, never the record."""
    record = _record("prop-01--s1")
    record["views"] = False  # the re-run render timed out having written nothing
    run_dir = _run_dir(tmp_path, [record], views_for=("prop-01--s1",))

    doc = score_mod.score_run(run_dir)

    assert doc["units"][0]["skipped"] == "no rendered views"
    assert doc["summary"]["scored"] == 0


# --- the drift guard ----------------------------------------------------------


def test_the_scores_carry_every_field_the_drift_check_reads(tmp_path, scored):
    """``_provenance``'s projection and ``compare_manifests``' reads have to be
    the same set of keys. They were not, and both sides missing a key compare
    equal -- which is why the two that mattered went unnoticed."""
    manifest = {key: {"x": 1} for key in score_mod.PROVENANCE_KEYS}
    run_dir = _run_dir(
        tmp_path, [_record("prop-01--s1")], views_for=("prop-01--s1",), manifest=manifest
    )

    measured = score_mod.score_run(run_dir)["measured"]

    assert set(measured) == set(score_mod.PROVENANCE_KEYS)
    assert {"config", "models"} <= set(measured)


def test_an_unreadable_manifest_does_not_stop_a_run_being_scored(tmp_path, scored):
    """The views are on disk and the numbers are still worth having; only the
    self-describing header is lost."""
    run_dir = _run_dir(tmp_path, [_record("prop-01--s1")], views_for=("prop-01--s1",))
    (run_dir / manifest_mod.FILENAME).write_text("{not json", encoding="utf-8")

    doc = score_mod.score_run(run_dir)

    assert doc["measured"] == {}
    assert doc["summary"]["scored"] == 1


def test_a_config_change_between_two_scored_runs_is_reported_as_drift(tmp_path, scored):
    """``_provenance`` kept only stage/suite/recipe/versions, and
    ``compare_manifests`` reads ``config`` and ``models`` -- absent on both
    sides, so they always compared equal. Change the band and the checkpoint
    between A and B and the A/B reported no drift at all, which is exactly the
    plausible-looking scores.json the manifest exists to prevent."""
    docs = []
    for band, digest in ((0.5, "sha:aaa"), (0.9, "sha:bbb")):
        manifest = {
            "stage": "model",
            "suite": {"key": "s", "file": "f"},
            "recipe": {"key": "r", "file": "f"},
            "versions": {"torch": "2.0"},
            "config": {"trellis_band": band},
            "models": {"base_model": digest},
        }
        run_dir = _run_dir(
            tmp_path / str(band),
            [_record("prop-01--s1")],
            views_for=("prop-01--s1",),
            manifest=manifest,
        )
        docs.append(score_mod.score_run(run_dir))

    lines = score_mod.compare_lines(*docs)

    assert any("config.trellis_band" in line for line in lines)
    assert any("model base_model changed" in line for line in lines)


# torch and diffusers live in the `text2image` extra, and provenance.py is
# explicitly built to import without it. Asserting on them unconditionally
# would make a dev-only environment fail the very contract it demonstrates.
needs_text2image = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("diffusers") is None,
    reason="needs the text2image extra",
)


def test_the_manifest_records_the_versions_it_compares_on():
    """The guard was vacuous: the manifest is built before torch or diffusers
    have ever been imported, so reading only sys.modules recorded neither --
    and compare_manifests then compared None to None.

    ``prompt`` is asserted unconditionally because it is core; the two heavy
    libraries only when the extra that ships them is installed.
    """
    versions = provenance.versions()
    assert versions.get("prompt")
    for name in ("torch", "diffusers"):
        if importlib.util.find_spec(name) is not None:
            assert versions.get(name), f"{name} is installed but not recorded"


@needs_text2image
def test_reading_a_version_does_not_import_the_library():
    """The manifest is written on the bench CLI's cold path, and importing
    torch to ask its version costs seconds and gigabytes."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from warlock import provenance; v = provenance.versions();"
            " print(bool(v.get('torch')), 'torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "True False"


@pytest.mark.parametrize("name", ["torch", "diffusers", "prompt"])
def test_a_version_change_actually_refuses_a_resume(name):
    a = {"versions": provenance.versions()}
    b = {"versions": {**a["versions"], name: "0.0.1"}}
    assert any(name in line for line in manifest_mod.compare_manifests(a, b))


@pytest.mark.parametrize("field", manifest_mod.CONFIG_FIELDS)
def test_a_config_change_actually_refuses_a_resume(field):
    """Every field in CONFIG_FIELDS is there because it moves a pixel; a field
    listed but not compared is a silent hole in the guard."""
    a = {"config": {name: 1 for name in manifest_mod.CONFIG_FIELDS}}
    b = {"config": {**a["config"], field: 2}}
    assert any(f"config.{field}" in line for line in manifest_mod.compare_manifests(a, b))


def test_a_checkpoint_change_actually_refuses_a_resume():
    a = {"models": {"base_model": "sha:aaa", "trellis_models_dir": "sha:bbb"}}
    b = {"models": {**a["models"], "base_model": "sha:ccc"}}
    assert manifest_mod.compare_manifests(a, b) == ["model base_model changed"]


def test_two_manifests_from_this_process_are_comparable():
    """The other direction, so the guard is not simply "always refuse"."""
    a = {"versions": provenance.versions()}
    b = {"versions": provenance.versions()}
    assert manifest_mod.compare_manifests(a, b) == []

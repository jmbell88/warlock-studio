"""Turning a sweep's verdicts into a findings table.

Grouped by (param, str(value)) as ``items.jsonl`` recorded it -- the join
target is the item record, not the verdict's own copy of param/value -- with
baseline units (param None) landing under the literal "baseline"/"baseline"
bucket. Scores join is best-effort: a run with no scores.json, or a metric
silently missing from one (the dino_cosine/torchvision caveat), must not
crash and must not corrupt another metric's mean.
"""

from __future__ import annotations

import json

import pytest

from warlock.bench import manifest as manifest_mod
from warlock.bench import report as report_mod
from warlock.bench import runner as runner_mod
from warlock.bench import score as score_mod
from warlock.bench import verdicts as verdicts_mod
from warlock.config import Config


@pytest.fixture
def config(tmp_path):
    return Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i",
        bench_dir=tmp_path / "bench",
    )


def _write_items(run_dir, records):
    (run_dir / runner_mod.ITEMS_FILE).write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def _write_scores(run_dir, units):
    score_mod.write_scores(
        run_dir,
        {"version": 1, "run": run_dir.name, "metrics": ["silhouette_iou"], "units": units},
    )


def _item(key, param=None, value=None, status="done"):
    return {"key": key, "param": param, "value": value, "status": status}


def _sweep_run(tmp_path, name, *, axis_param="lora_weight", axis_values=(0.6, 1.2)):
    run_dir = tmp_path / "bench" / "runs" / name
    run_dir.mkdir(parents=True)
    manifest_mod.write_manifest(
        run_dir,
        {
            "version": 1,
            "sweep": {"key": "x", "axes": [{"param": axis_param, "values": list(axis_values)}]},
        },
    )
    return run_dir


# --- sweep_runs ----------------------------------------------------------------


def test_sweep_runs_finds_only_manifests_with_a_sweep_key(tmp_path, config):
    sweep_dir = _sweep_run(tmp_path, "20260101-sweep-x")

    plain_dir = tmp_path / "bench" / "runs" / "20260102-r-s"
    plain_dir.mkdir(parents=True)
    manifest_mod.write_manifest(plain_dir, {"version": 1, "suite": {"key": "s"}})

    found = report_mod.sweep_runs(config)

    assert found == [sweep_dir]


def test_sweep_runs_on_a_missing_runs_dir_is_empty(tmp_path, config):
    assert report_mod.sweep_runs(config) == []


def test_sweep_runs_tolerates_an_unreadable_manifest(tmp_path, config):
    sweep_dir = _sweep_run(tmp_path, "20260101-sweep-x")
    broken = tmp_path / "bench" / "runs" / "20260103-broken"
    broken.mkdir(parents=True)
    (broken / manifest_mod.FILENAME).write_text("{not json", encoding="utf-8")

    assert report_mod.sweep_runs(config) == [sweep_dir]


# --- aggregate: accept counting / grouping --------------------------------------


def test_accept_counting_and_rate(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(
        run_dir,
        [
            _item("lora_weight=0.6--s1", "lora_weight", 0.6),
            _item("lora_weight=0.6--s2", "lora_weight", 0.6),
            _item("lora_weight=0.6--s3", "lora_weight", 0.6),
        ],
    )
    for unit, verdict in (
        ("lora_weight=0.6--s1", "accept"),
        ("lora_weight=0.6--s2", "accept"),
        ("lora_weight=0.6--s3", "reject"),
    ):
        verdicts_mod.append_verdict(
            run_dir, unit=unit, source="human", verdict=verdict,
            param="lora_weight", value=0.6,
        )

    doc = report_mod.aggregate([run_dir])

    entry = doc["lora_weight"]["0.6"]
    assert entry["n"] == 3
    assert entry["accepts"] == 2
    assert entry["accept_rate"] == 0.667


def test_baseline_units_group_under_the_baseline_bucket(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("baseline--s1", None, None)])
    verdicts_mod.append_verdict(
        run_dir, unit="baseline--s1", source="human", verdict="accept",
        param=None, value=None,
    )

    doc = report_mod.aggregate([run_dir])

    assert doc["baseline"]["baseline"]["n"] == 1
    assert doc["baseline"]["baseline"]["accepts"] == 1


def test_a_verdict_for_a_unit_absent_from_items_is_skipped(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [])  # nothing planned/recorded
    verdicts_mod.append_verdict(
        run_dir, unit="ghost--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )

    doc = report_mod.aggregate([run_dir])

    assert doc == {}


def test_non_sweep_runs_are_simply_not_passed_in_and_produce_nothing(tmp_path):
    # aggregate itself does not filter -- sweep_runs does that -- but an
    # empty run list must still produce an empty, valid aggregate.
    assert report_mod.aggregate([]) == {}


# --- aggregate: latest-verdict-wins ---------------------------------------------


def test_latest_verdict_wins_in_aggregate(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="reject",
        reasons=("holes",), param="lora_weight", value=0.6,
    )
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )

    doc = report_mod.aggregate([run_dir])

    entry = doc["lora_weight"]["0.6"]
    assert entry["n"] == 1
    assert entry["accepts"] == 1
    assert entry["top_reasons"] == []  # the reject (and its reason) was superseded


def test_a_reviewers_change_of_mind_does_not_double_count_sources(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="reject",
        reasons=("holes",), param="lora_weight", value=0.6,
    )
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )

    doc = report_mod.aggregate([run_dir])

    sources = doc["lora_weight"]["0.6"]["sources"]
    assert sources == {"human": {"accept": 1, "reject": 0}}


# --- aggregate: sources / top_reasons --------------------------------------------


def test_sources_are_counted_per_source(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(
        run_dir,
        [
            _item("lora_weight=0.6--s1", "lora_weight", 0.6),
            _item("lora_weight=0.6--s2", "lora_weight", 0.6),
        ],
    )
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s2", source="ai:vision", verdict="reject",
        reasons=("bad-shape",), param="lora_weight", value=0.6,
    )

    doc = report_mod.aggregate([run_dir])

    sources = doc["lora_weight"]["0.6"]["sources"]
    assert sources == {
        "human": {"accept": 1, "reject": 0},
        "ai:vision": {"accept": 0, "reject": 1},
    }


def test_top_reasons_are_ranked_by_count(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    keys = [f"lora_weight=0.6--s{i}" for i in range(1, 4)]
    _write_items(run_dir, [_item(k, "lora_weight", 0.6) for k in keys])
    for key, reason in zip(keys, ("holes", "holes", "broken"), strict=True):
        verdicts_mod.append_verdict(
            run_dir, unit=key, source="human", verdict="reject",
            reasons=(reason,), param="lora_weight", value=0.6,
        )

    doc = report_mod.aggregate([run_dir])

    assert doc["lora_weight"]["0.6"]["top_reasons"] == [["holes", 2], ["broken", 1]]


# --- aggregate: scores join / null tolerance -------------------------------------


def test_scores_join_computes_a_mean(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    keys = ["lora_weight=0.6--s1", "lora_weight=0.6--s2"]
    _write_items(run_dir, [_item(k, "lora_weight", 0.6) for k in keys])
    for key in keys:
        verdicts_mod.append_verdict(
            run_dir, unit=key, source="human", verdict="accept",
            param="lora_weight", value=0.6,
        )
    _write_scores(
        run_dir,
        [
            {"key": keys[0], "scores": {"silhouette_iou": 0.8}},
            {"key": keys[1], "scores": {"silhouette_iou": 0.6}},
        ],
    )

    doc = report_mod.aggregate([run_dir])

    assert doc["lora_weight"]["0.6"]["mean_silhouette_iou"] == 0.7
    assert doc["lora_weight"]["0.6"]["mean_dino_cosine"] is None


def test_a_missing_scores_json_leaves_means_null_without_crashing(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )

    doc = report_mod.aggregate([run_dir])

    entry = doc["lora_weight"]["0.6"]
    assert entry["mean_silhouette_iou"] is None
    assert entry["mean_dino_cosine"] is None


def test_a_null_score_for_one_unit_is_skipped_not_averaged_as_zero(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    keys = ["lora_weight=0.6--s1", "lora_weight=0.6--s2"]
    _write_items(run_dir, [_item(k, "lora_weight", 0.6) for k in keys])
    for key in keys:
        verdicts_mod.append_verdict(
            run_dir, unit=key, source="human", verdict="accept",
            param="lora_weight", value=0.6,
        )
    _write_scores(
        run_dir,
        [
            {"key": keys[0], "scores": {"silhouette_iou": 0.8}},
            {"key": keys[1], "scores": {"silhouette_iou": None}},
        ],
    )

    doc = report_mod.aggregate([run_dir])

    assert doc["lora_weight"]["0.6"]["mean_silhouette_iou"] == 0.8


# --- write_findings: schema stability --------------------------------------------


def test_findings_json_has_the_exact_top_level_keys(tmp_path, config):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )
    doc = report_mod.aggregate([run_dir])

    json_path, md_path = report_mod.write_findings(config, doc)

    assert json_path == config.bench_dir / "findings.json"
    assert md_path == config.bench_dir / "findings.md"
    written = json.loads(json_path.read_text("utf-8"))
    assert set(written) == {"version", "generated", "params"}
    assert written["version"] == 1
    assert written["generated"]

    entry = written["params"]["lora_weight"]["0.6"]
    assert set(entry) == {
        "n", "accepts", "accept_rate", "sources",
        "mean_silhouette_iou", "mean_dino_cosine", "top_reasons",
    }


def test_findings_md_is_written_and_mentions_the_param(tmp_path, config):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )
    doc = report_mod.aggregate([run_dir])

    _, md_path = report_mod.write_findings(config, doc)

    text = md_path.read_text("utf-8")
    assert "lora_weight" in text
    assert "0.6" in text


def test_write_findings_with_no_verdicts_still_writes_both_files(tmp_path, config):
    json_path, md_path = report_mod.write_findings(config, {})

    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text("utf-8"))["params"] == {}
    assert "no verdicts" in md_path.read_text("utf-8")


# --- summary_lines ---------------------------------------------------------------


def test_summary_lines_reports_accept_rate_and_nothing_to_score(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )
    doc = report_mod.aggregate([run_dir])
    full = {"version": 1, "generated": "x", "params": doc}

    lines = report_mod.summary_lines(full)

    assert any("lora_weight" in line for line in lines)
    assert any("1/1 accepted" in line for line in lines)
    assert any("nothing to score" in line for line in lines)


def test_summary_lines_min_n_hides_a_thin_bucket(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )
    doc = report_mod.aggregate([run_dir])
    full = {"version": 1, "generated": "x", "params": doc}

    lines = report_mod.summary_lines(full, min_n=5)

    assert lines == ["no verdicts recorded yet"]


def test_summary_lines_min_n_defaults_to_showing_everything(tmp_path):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )
    doc = report_mod.aggregate([run_dir])
    full = {"version": 1, "generated": "x", "params": doc}

    assert report_mod.summary_lines(full) == report_mod.summary_lines(full, min_n=0)


# --- CLI: `report` subcommand ----------------------------------------------------


from warlock.bench import __main__ as bench_main  # noqa: E402


def test_report_cli_with_no_sweep_runs_says_so_and_exits_zero(tmp_path, config, capsys):
    args = bench_main.build_parser().parse_args(["report"])
    rc = bench_main._report(config, args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "no verdicts recorded yet" in out
    assert (config.bench_dir / "findings.json").exists()
    assert (config.bench_dir / "findings.md").exists()


def test_report_cli_prints_summary_and_both_paths(tmp_path, config, capsys):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )

    args = bench_main.build_parser().parse_args(["report", "--threshold", "0"])
    rc = bench_main._report(config, args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "lora_weight" in out
    json_path = config.bench_dir / "findings.json"
    md_path = config.bench_dir / "findings.md"
    assert str(json_path) in out
    assert str(md_path) in out


def test_report_cli_threshold_hides_a_thin_bucket_but_findings_json_keeps_it(
    tmp_path, config, capsys
):
    run_dir = _sweep_run(tmp_path, "r1")
    _write_items(run_dir, [_item("lora_weight=0.6--s1", "lora_weight", 0.6)])
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.6--s1", source="human", verdict="accept",
        param="lora_weight", value=0.6,
    )

    args = bench_main.build_parser().parse_args(["report", "--threshold", "5"])
    rc = bench_main._report(config, args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "lora_weight" not in out
    written = json.loads((config.bench_dir / "findings.json").read_text("utf-8"))
    assert written["params"]["lora_weight"]["0.6"]["n"] == 1

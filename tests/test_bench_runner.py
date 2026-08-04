"""The runner drives the real service against a real Runtime.

The fakes here are the GPU pipelines (the same ones conftest patches for the
queue tests), never the service or the store: a benchmark that submitted
differently from the app would measure a code path no user can reach.
"""

from __future__ import annotations

import json

import pytest

from warlock.bench import manifest as manifest_mod
from warlock.bench import recipe as recipe_mod
from warlock.bench import runner as runner_mod
from warlock.bench import suite as suite_mod
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


# --- planning (no GPU, no runtime) -------------------------------------------


def test_a_plan_is_items_times_seeds(config):
    s, r = suite_mod.load(), recipe_mod.load("baseline-turbo-raw")
    _dir, todo, doc = runner_mod.plan_run(
        config, s, r, stage="reference", started="t", limit=3, seeds=(1, 2)
    )
    assert len(todo) == 6
    assert doc["seeds"] == [1, 2]
    assert len(doc["items"]) == 3


def test_the_full_default_plan_is_a_hundred_and_sixty_jobs(config):
    _dir, todo, _doc = runner_mod.plan_run(
        config, suite_mod.load(), recipe_mod.load("baseline-turbo-raw"),
        stage="reference", started="t",
    )
    assert len(todo) == 160


def test_a_unit_key_names_its_item_and_seed(config):
    _dir, todo, _doc = runner_mod.plan_run(
        config, suite_mod.load(), recipe_mod.load("baseline-turbo-raw"),
        stage="reference", started="t", ids=("prop-01",), seeds=(42,),
    )
    assert todo[0].key == "prop-01--s42"


def test_the_run_directory_leads_with_its_timestamp(config):
    """prune sorts by name, so the timestamp has to come first."""
    directory, _todo, _doc = runner_mod.plan_run(
        config, suite_mod.load(), recipe_mod.load("baseline-turbo-raw"),
        stage="reference", started="20260803-120000",
    )
    assert directory.name.startswith("20260803-120000-")
    assert directory.parent.name == "runs"


# --- execution ---------------------------------------------------------------


def _run(config, **kwargs):
    return runner_mod.run(
        config,
        recipe_key="baseline-turbo-raw",
        stage="reference",
        started="20260803-120000",
        ids=("prop-01",),
        seeds=(42, 1337),
        **kwargs,
    )


def test_a_run_writes_a_manifest_a_jsonl_and_one_dir_per_unit(config, fake_pipelines):
    run_dir = _run(config)

    assert (run_dir / "manifest.json").exists()
    records = [
        json.loads(line)
        for line in (run_dir / runner_mod.ITEMS_FILE).read_text("utf-8").splitlines()
    ]
    assert [r["key"] for r in records] == ["prop-01--s42", "prop-01--s1337"]
    assert all(r["status"] == "done" for r in records)
    assert all(r["seconds"] >= 0 for r in records)
    for record in records:
        item_dir = run_dir / "items" / record["key"]
        assert (item_dir / "job.json").exists()
        assert (item_dir / "input.png").exists()


def test_artifacts_are_copied_so_a_run_survives_a_prune(config, fake_pipelines):
    from warlock.service import jobs as svc_jobs
    from warlock.studio.runtime import Runtime

    run_dir = _run(config)
    copied = run_dir / "items" / "prop-01--s42" / "input.png"
    assert copied.exists()

    with Runtime(config) as svc:
        svc_jobs.prune_jobs(svc, keep=0)
    assert copied.exists(), "the run must not reference the job directory"


def test_a_run_never_leaves_more_than_one_job_in_flight(config, fake_pipelines, monkeypatch):
    """The queue is serial, so batching buys nothing -- and a Ctrl-C mid-run
    would otherwise leave 159 queued rows for the app to chew through at next
    launch."""
    seen = []
    real = runner_mod._await_job

    def watching(svc, job_id, **kwargs):
        live = [j for j in svc.store.list(100) if j["status"] in ("queued", "running")]
        seen.append(len(live))
        return real(svc, job_id, **kwargs)

    monkeypatch.setattr(runner_mod, "_await_job", watching)
    _run(config)
    assert seen and max(seen) <= 1


def test_the_recipe_reaches_the_stored_params(config, fake_pipelines):
    run_dir = _run(config)
    job = json.loads((run_dir / "items" / "prop-01--s42" / "job.json").read_text("utf-8"))
    assert job["params"]["base_model"] == "turbo"
    assert job["params"]["seed"] == 42
    assert job["stage"] == "reference"


def test_a_resume_creates_zero_jobs_for_what_is_done(config, fake_pipelines):
    run_dir = _run(config)
    before = len(runner_mod.completed(run_dir))
    assert before == 2

    _run(config, resume=run_dir)
    assert len(runner_mod.completed(run_dir)) == before


def test_a_resume_against_a_changed_setup_is_refused(config, fake_pipelines):
    run_dir = _run(config)
    doc = manifest_mod.read_manifest(run_dir)
    doc["config"]["trellis_tex_res"] = 2048
    manifest_mod.write_manifest(run_dir, doc)

    with pytest.raises(manifest_mod.NotResumable):
        _run(config, resume=run_dir)


def test_a_torn_last_line_is_simply_re_run(config):
    run_dir = config.bench_dir / "runs" / "x"
    run_dir.mkdir(parents=True)
    (run_dir / runner_mod.ITEMS_FILE).write_text(
        json.dumps({"key": "prop-01--s42"}) + '\n{"key": "prop-0',
        encoding="utf-8",
    )
    assert runner_mod.completed(run_dir) == {"prop-01--s42"}


def test_prune_keeps_the_newest_runs(config):
    root = config.bench_dir / "runs"
    for name in ("20260101-000000-a-b", "20260201-000000-a-b", "20260301-000000-a-b"):
        (root / name).mkdir(parents=True)

    removed = runner_mod.prune_runs(config, keep=1)

    assert removed == ["20260201-000000-a-b", "20260101-000000-a-b"]
    assert [p.name for p in root.iterdir()] == ["20260301-000000-a-b"]


def test_prune_on_an_empty_bench_dir_is_not_an_error(config):
    assert runner_mod.prune_runs(config, keep=3) == []

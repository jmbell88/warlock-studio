"""The download machinery, headlessly: the record, the path rule, the dedupe,
the disk refusal, and the promise that the app process stays offline.

Everything here is pure or stubbed. Nothing in this file may download anything
-- a real fetch is the user's call, and a test that made one would be the exact
thing the offline invariant exists to prevent.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from warlock import fetch, models
from warlock.config import Config

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock"


# --- the record and the string it derives ------------------------------------


def test_a_fetch_renders_the_command_it_stands_for():
    one = models.Fetch(
        "acme/thing",
        "thing",
        allow_patterns=("*.json", "*fp16.safetensors"),
        ignore_patterns=("big.safetensors",),
    )
    assert one.command() == (
        "uvx hf download acme/thing "
        '--include "*.json" --include "*fp16.safetensors" '
        '--exclude "big.safetensors" '
        "--local-dir models/thing"
    )


def test_the_download_text_is_what_doctor_prints():
    """Verbatim, for the entries that carry every shape there is.

    Pinned rather than described because the whole point of deriving the string
    is that doctor's text and the README cannot drift from what the button
    does; a test that only asserted "starts with uvx" would let the derivation
    quietly change the words a person is told to paste.
    """
    assert models.BASE_MODELS["turbo"].download == (
        "uvx hf download stabilityai/sdxl-turbo "
        '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
        '--exclude "sd_xl_turbo_1.0*" --local-dir models/sdxl-turbo'
    )
    # Two commands, joined by the two-space continuation doctor's
    # "download with:\n  " prefix lines up with.
    assert models.BASE_MODELS["sdxl"].download == (
        "uvx hf download stabilityai/stable-diffusion-xl-base-1.0 "
        '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
        "--local-dir models/sdxl-base-1.0\n"
        "  uvx hf download ByteDance/Hyper-SD Hyper-SDXL-4steps-lora.safetensors "
        "--local-dir models/loras"
    )
    # The rename, which is why the string could never simply be executed.
    assert models.BASE_MODELS["pixel"].download.endswith(
        "  then rename models/loras/pytorch_lora_weights.safetensors "
        "to lcm-lora-sdxl.safetensors"
    )
    # The non-shell step, which the button deliberately does not run.
    assert models.MATTING_MODELS["birefnet"].download.endswith(
        "  then: uv sync --extra text2image "
        "-- BiRefNet's modelling code imports einops, kornia, timm and "
        "torchvision, and that extra is what supplies them"
    )


def test_every_registry_entry_has_a_fetch_and_a_download_line():
    for entry in fetch.entries():
        assert entry.fetch, f"{entry.row_key} has nothing to download"
        assert entry.spec.download.startswith("uvx hf download "), entry.row_key
        for one in entry.fetch:
            assert one.repo_id and one.local_dir, entry.row_key
            assert one.size_gib > 0, f"{entry.row_key} declares no size"


def test_the_readme_names_every_repository_the_registry_does():
    """The drift this package closed, kept closed.

    Four repo ids -- the IP-Adapter, the Canny ControlNet, DINOv2 and BiRefNet
    -- existed only in models.py, so the README's list of "the only network use
    there is" was not one. Both halves come off ``Fetch`` now; this is what
    notices when a new entry is added to one and not the other.
    """
    readme = (SRC.parents[1] / "README.md").read_text(encoding="utf-8")
    repos = {one.repo_id for entry in fetch.entries() for one in entry.fetch}
    missing = sorted(repo for repo in repos if repo not in readme)
    assert not missing, f"in models.py but not in the README: {missing}"


def test_a_note_is_not_a_fetch():
    """BiRefNet's `uv sync` is prose in the string and absent from the plan."""
    entry = fetch.find("matting:birefnet")
    assert entry is not None
    jobs = fetch.plan(Config(), [entry])
    assert [j.repo_id for j in jobs] == ["ZhengPeng7/BiRefNet"]


# --- where things land -------------------------------------------------------


def _config(tmp_path: Path, turbo: Path | None = None) -> Config:
    cfg = Config()
    cfg.t2i_model_root = tmp_path / "models"
    cfg.t2i_turbo_dir = turbo
    return cfg


def test_only_turbo_honours_the_legacy_override(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    cfg = _config(tmp_path, turbo=elsewhere)
    assert fetch.base_model_dir(cfg, models.BASE_MODELS["turbo"]) == elsewhere
    assert (
        fetch.base_model_dir(cfg, models.BASE_MODELS["sdxl"])
        == cfg.t2i_model_root / "sdxl-base-1.0"
    )


def test_a_plan_resolves_under_the_model_root_not_the_literal_models_dir(tmp_path):
    """The hardcoded ``--local-dir models/...`` is the README's spelling of the
    default root; WARLOCK_T2I_ROOT has to relocate a fetch the way it already
    relocates a load."""
    cfg = _config(tmp_path)
    entry = fetch.find("base:sdxl")
    jobs = fetch.plan(cfg, [entry])
    dests = {job.repo_id: job.dest for job in jobs}
    assert dests["stabilityai/stable-diffusion-xl-base-1.0"] == (
        cfg.t2i_model_root / "sdxl-base-1.0"
    )
    assert dests["ByteDance/Hyper-SD"] == cfg.t2i_model_root / "loras"


def test_the_override_moves_the_turbo_download_too(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    cfg = _config(tmp_path, turbo=elsewhere)
    jobs = fetch.plan(cfg, [fetch.find("base:turbo")])
    assert [job.dest for job in jobs] == [elsewhere]


# --- dedupe ------------------------------------------------------------------


def test_four_recipes_over_one_checkpoint_are_one_download(tmp_path):
    """dir_name is not unique, so a key-keyed plan would fetch 7 GB four times."""
    cfg = _config(tmp_path)
    chosen = [fetch.find(k) for k in ("base:sdxl", "base:sdxl_cfg", "base:pixel", "base:lightning")]
    jobs = fetch.plan(cfg, chosen)
    repos = [job.repo_id for job in jobs]
    assert repos.count("stabilityai/stable-diffusion-xl-base-1.0") == 1
    # ...and the size is counted once, which is the difference between
    # "this needs 7 GB" and "this needs 28 GB".
    base = next(j for j in jobs if j.repo_id.endswith("stable-diffusion-xl-base-1.0"))
    assert base.size_gib == pytest.approx(7.0)
    # The three distinct LoRA repos survive.
    assert len(jobs) == 4


def test_two_records_against_one_repository_merge_their_patterns(tmp_path):
    """The IP-Adapter's weights and CLIP vision encoder are two doctor lines and
    one fetch; dropping either half is the load-fine-then-fail-at-first-call
    directory its two-part probe exists to catch."""
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("adapter:plus")])
    assert len(jobs) == 1
    job = jobs[0]
    assert job.filenames == ("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",)
    assert job.allow_patterns == ("models/image_encoder/*",)


def test_dedupe_is_on_the_destination_not_the_repo_alone(tmp_path):
    """Same repo into two directories is two fetches."""
    cfg = _config(tmp_path)
    entry = fetch.Entry(
        "metric",
        "x",
        "X",
        models.MetricModel(
            "x",
            "X",
            "x",
            fetch=(
                models.Fetch("acme/x", "one", size_gib=1.0),
                models.Fetch("acme/x", "two", size_gib=1.0),
            ),
        ),
    )
    assert len(fetch.plan(cfg, [entry])) == 2


# --- refusing rather than half-downloading -----------------------------------


def test_a_plan_that_does_not_fit_is_refused(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("base:flux_klein")])
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 3.0)
    refusal = fetch.disk_refusal(jobs)
    assert refusal is not None and "16" not in refusal.split("needs")[0]
    assert "3.0 GB is free" in refusal


def test_a_plan_that_fits_is_not_refused(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("lora:pixelxl")])
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 500.0)
    assert fetch.disk_refusal(jobs) is None


def test_free_space_that_cannot_be_read_is_not_a_refusal(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("base:flux_klein")])
    monkeypatch.setattr(fetch, "free_gib", lambda _path: None)
    assert fetch.disk_refusal(jobs) is None


def test_free_space_answers_for_a_directory_that_does_not_exist_yet(tmp_path):
    """A download's destination routinely does not exist; disk_usage raises on
    a missing path rather than answering about its volume."""
    assert fetch.free_gib(tmp_path / "not" / "here" / "yet") is not None


# --- presence ----------------------------------------------------------------


def test_a_partial_directory_is_absent(tmp_path):
    cfg = _config(tmp_path)
    entry = fetch.find("base:playground")
    root = cfg.t2i_model_root / "playground-v2.5"
    root.mkdir(parents=True)
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    assert not entry.is_present(cfg)
    unet = root / "unet"
    unet.mkdir()
    (unet / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    assert entry.is_present(cfg)


def test_a_checkpoint_without_its_step_distillation_lora_is_absent(tmp_path):
    cfg = _config(tmp_path)
    root = cfg.t2i_model_root / "sdxl-base-1.0"
    (root / "unet").mkdir(parents=True)
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    (root / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    ok, missing = fetch.base_model_state(cfg, models.BASE_MODELS["sdxl"])
    assert not ok and missing is not None and missing.name.startswith("Hyper-SDXL")
    # sdxl_cfg is the same weights with no distillation LoRA, so it is ready.
    assert fetch.base_model_state(cfg, models.BASE_MODELS["sdxl_cfg"])[0]


def test_doctor_and_the_planner_share_one_probe():
    from warlock import doctor

    assert doctor._base_model_dir is fetch.base_model_dir


# --- the app process never becomes online-capable ----------------------------


def test_only_the_fetch_worker_ever_clears_hf_hub_offline():
    """The whole exception to the offline invariant, expressed as a scan.

    ``HF_HUB_OFFLINE`` is read by huggingface_hub at import time, so a value of
    "0" anywhere the app imports would make this process online-capable for the
    rest of its life. Exactly one module may write it, and it is the one that
    is spawned, does one thing and dies.

    Written as a scan for a *use* rather than for the string, because several
    modules explain the invariant in a comment and prose is not a hazard --
    ``os.environ[...]`` and ``setdefault`` are.
    """
    use = re.compile(r"""(?:environ\[|setdefault\(\s*)["']HF_HUB_OFFLINE""")
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if not use.search(text):
            continue
        if path.name in ("__init__.py", "fetch_worker.py"):
            continue
        offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "only warlock/__init__.py (which sets it to 1) and "
        f"pipelines/fetch_worker.py (a child process) may touch it: {offenders}"
    )


def test_the_package_still_sets_hf_hub_offline_to_one():
    import warlock  # noqa: F401  -- the import is what sets it

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_fetch_leaves_no_half_populated_directory(tmp_path):
    """With the network unavailable, the worker must fail and clean up.

    Run for real, as a subprocess, with snapshot_download replaced by one that
    writes a file and then raises -- which is the shape of every real failure
    (a partial download, then an error) and the only one that can leave a
    directory the presence probes would call finished.
    """
    dest = tmp_path / "models" / "thing"
    result = tmp_path / "result.json"
    spec = {
        "repo_id": "acme/thing",
        "dest": str(dest),
        "filenames": [],
        "allow_patterns": [],
        "ignore_patterns": [],
        "rename": None,
        "size_gib": 0.1,
        "result_path": str(result),
    }
    stub = (
        "import sys, types, pathlib\n"
        "mod = types.ModuleType('huggingface_hub')\n"
        "def snapshot_download(**kw):\n"
        "    root = pathlib.Path(kw['local_dir'])\n"
        "    (root / 'config.json').write_text('{}')\n"
        "    raise OSError('no network')\n"
        "mod.snapshot_download = snapshot_download\n"
        "sys.modules['huggingface_hub'] = mod\n"
        "from warlock.pipelines import fetch_worker\n"
        "raise SystemExit(fetch_worker.main())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", stub],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["ok"] is False and "no network" in payload["error"]
    assert not dest.exists(), "a failed fetch left a model directory behind"
    assert not list((tmp_path / "models").glob("*.part")), "staging tree survived"


def test_a_successful_fetch_moves_the_files_in_and_renames(tmp_path):
    dest = tmp_path / "models" / "loras"
    dest.mkdir(parents=True)
    (dest / "already.safetensors").write_bytes(b"keep")
    result = tmp_path / "result.json"
    spec = {
        "repo_id": "acme/lora",
        "dest": str(dest),
        "filenames": ["pytorch_lora_weights.safetensors"],
        "allow_patterns": [],
        "ignore_patterns": [],
        "rename": ["pytorch_lora_weights.safetensors", "lcm-lora-sdxl.safetensors"],
        "size_gib": 0.1,
        "result_path": str(result),
    }
    stub = (
        "import sys, types, pathlib, os\n"
        "mod = types.ModuleType('huggingface_hub')\n"
        "def snapshot_download(**kw):\n"
        "    root = pathlib.Path(kw['local_dir'])\n"
        "    (root / 'pytorch_lora_weights.safetensors').write_bytes(b'w')\n"
        "    (root / '.cache').mkdir()\n"
        "    (root / '.cache' / 'junk').write_text('x')\n"
        "mod.snapshot_download = snapshot_download\n"
        "sys.modules['huggingface_hub'] = mod\n"
        "from warlock.pipelines import fetch_worker\n"
        "raise SystemExit(fetch_worker.main())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", stub],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (dest / "lcm-lora-sdxl.safetensors").read_bytes() == b"w"
    # The rename happened inside staging, before the move: the generic name
    # never lands in loras/ at all -- which is the whole point of the rename,
    # since two repos ship the same generic filename into one flat directory.
    assert not (dest / "pytorch_lora_weights.safetensors").exists()
    # A shared destination gains files rather than being replaced.
    assert (dest / "already.safetensors").read_bytes() == b"keep"
    # huggingface_hub's resume bookkeeping does not follow the weights in.
    assert not (dest / ".cache").exists()


def test_a_rename_whose_source_never_arrived_fails_the_fetch(tmp_path):
    """fetch_one applies a rename unconditionally, and loudly. A repo that
    stopped shipping the declared filename must fail the whole fetch rather
    than move in a directory missing the file the presence probes key on --
    ``pixelklein`` (the first ``filenames=`` plus ``rename=`` entry) would
    otherwise read as downloaded forever while loading nothing."""
    dest = tmp_path / "models" / "loras"
    result = tmp_path / "result.json"
    spec = {
        "repo_id": "acme/lora",
        "dest": str(dest),
        "filenames": ["pytorch_lora_weights.safetensors"],
        "allow_patterns": [],
        "ignore_patterns": [],
        "rename": ["pytorch_lora_weights.safetensors", "pixel-art-klein.safetensors"],
        "size_gib": 0.1,
        "result_path": str(result),
    }
    stub = (
        "import sys, types, pathlib\n"
        "mod = types.ModuleType('huggingface_hub')\n"
        "def snapshot_download(**kw):\n"
        "    root = pathlib.Path(kw['local_dir'])\n"
        "    (root / 'something_else.safetensors').write_bytes(b'w')\n"
        "mod.snapshot_download = snapshot_download\n"
        "sys.modules['huggingface_hub'] = mod\n"
        "from warlock.pipelines import fetch_worker\n"
        "raise SystemExit(fetch_worker.main())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", stub],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["ok"] is False and "did not provide" in payload["error"]
    # The staging promise holds on this path too: nothing lands, nothing
    # lingers, and neither spelling of the filename reaches loras/.
    assert not dest.exists()
    assert not list((tmp_path / "models").glob("*.part"))


def test_the_fetch_worker_spawn_is_in_the_kill_on_close_job():
    """Named here as well as in the package-wide scan, because this child is the
    one that holds a socket and writes gigabytes: one that outlived a hard kill
    would go on filling the disk with nothing on screen to say so."""
    text = (SRC / "service" / "downloads.py").read_text(encoding="utf-8")
    spawn = text.index("subprocess.Popen(")
    assert "winjob.assign" in text[spawn : spawn + 1200]


# --- progress ----------------------------------------------------------------


def test_progress_is_only_recorded_while_the_key_is_in_flight():
    """A task thread's last report must not resurrect a collected entry.

    ``Done`` arrives at completion and the frame that collects it stops drawing
    the bar; a report landing after that would leave a bar on screen for a task
    that has finished, with nothing to clear it.
    """
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        assert runner.submit("download:x", lambda: 1)
        runner.set_progress("download:x", 40.0, "half")
        assert runner.progress("download:x") == {"percent": 40.0, "label": "half"}
        while not runner.poll():
            pass
        assert runner.progress("download:x") is None
        runner.set_progress("download:x", 90.0, "late")
        assert runner.progress("download:x") is None
    finally:
        runner.shutdown(wait=False)


def test_resubmitting_a_key_starts_from_no_progress():
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        runner.submit("download:x", lambda: 1)
        runner.set_progress("download:x", 80.0, "nearly")
        while not runner.poll():
            pass
        runner.submit("download:x", lambda: 1)
        assert runner.progress("download:x") is None
    finally:
        runner.shutdown(wait=False)


def test_a_percentage_is_clamped():
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        runner.submit("download:x", lambda: 1)
        runner.set_progress("download:x", 140.0)
        assert runner.progress("download:x")["percent"] == 100.0
        runner.set_progress("download:x", -3.0)
        assert runner.progress("download:x")["percent"] == 0.0
    finally:
        runner.shutdown(wait=False)


# --- the service door --------------------------------------------------------


class _Svc:
    def __init__(self, config):
        self.config = config


def test_a_download_that_does_not_fit_never_spawns_anything(tmp_path, monkeypatch):
    from warlock.service import downloads
    from warlock.service.errors import Invalid

    monkeypatch.setattr(fetch, "free_gib", lambda _path: 1.0)
    monkeypatch.setattr(
        downloads,
        "_run_worker",
        lambda *a, **k: pytest.fail("spawned a worker for a refused plan"),
    )
    with pytest.raises(Invalid) as caught:
        downloads.download(_Svc(_config(tmp_path)), ["base:flux_klein"])
    assert "disk space" in caught.value.message


def test_an_unknown_row_is_refused_by_name(tmp_path):
    from warlock.service import downloads
    from warlock.service.errors import NotFound

    with pytest.raises(NotFound):
        downloads.plan_for(_Svc(_config(tmp_path)), ["base:nope"])


def test_a_download_reports_overall_progress_across_its_fetches(tmp_path, monkeypatch):
    from warlock.service import downloads

    monkeypatch.setattr(fetch, "free_gib", lambda _path: 5000.0)
    seen: list[float] = []

    def fake(job, *, on_progress, timeout):
        on_progress(50.0, job.repo_id)
        on_progress(100.0, job.repo_id)
        return {"ok": True}

    monkeypatch.setattr(downloads, "_run_worker", fake)
    downloads.download(
        _Svc(_config(tmp_path)),
        ["base:sdxl"],
        on_progress=lambda percent, _label: seen.append(percent),
    )
    # Monotone, never past 100, and ending exactly at it.
    assert seen == sorted(seen)
    assert seen[-1] == 100.0
    assert max(seen[:-1]) < 100.0


_STUB_WORKER = """
import sys, types, pathlib
mod = types.ModuleType('huggingface_hub')
def snapshot_download(**kw):
    root = pathlib.Path(kw['local_dir'])
    (root / 'pixel-art-xl.safetensors').write_bytes(b'weights')
mod.snapshot_download = snapshot_download
sys.modules['huggingface_hub'] = mod
from warlock.pipelines import fetch_worker
raise SystemExit(fetch_worker.main())
"""


def test_a_download_runs_end_to_end_through_a_real_child(tmp_path, monkeypatch):
    """The host half against a real subprocess: spawn, stdin, progress, result.

    The child's ``snapshot_download`` is a stub that writes a file -- no test in
    this project may make a network call, and the machinery either side of that
    one function is the whole of what this package added.
    """
    from warlock.service import downloads

    cfg = _config(tmp_path)
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 5000.0)
    monkeypatch.setattr(downloads, "worker_argv", lambda: [sys.executable, "-c", _STUB_WORKER])
    seen: list[tuple[float, str]] = []
    downloads.download(
        _Svc(cfg),
        ["lora:pixelxl"],
        on_progress=lambda percent, label: seen.append((percent, label)),
    )
    assert (cfg.t2i_model_root / "loras" / "pixel-art-xl.safetensors").read_bytes() == b"weights"
    assert seen and seen[-1] == (100.0, "")
    entry = fetch.find("lora:pixelxl")
    assert entry.is_present(cfg)


def test_a_child_that_fails_is_reported_in_its_own_words(tmp_path, monkeypatch):
    from warlock.service import downloads
    from warlock.service.errors import Invalid

    stub = _STUB_WORKER.replace(
        "(root / 'pixel-art-xl.safetensors').write_bytes(b'weights')",
        "raise OSError('the network is down')",
    )
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 5000.0)
    monkeypatch.setattr(downloads, "worker_argv", lambda: [sys.executable, "-c", stub])
    cfg = _config(tmp_path)
    with pytest.raises(Invalid) as caught:
        downloads.download(_Svc(cfg), ["lora:pixelxl"])
    assert "the network is down" in caught.value.message
    assert not (cfg.t2i_model_root / "loras").exists() or not list(
        (cfg.t2i_model_root / "loras").iterdir()
    )


def test_rows_carry_a_flag_rather_than_the_word_missing(tmp_path):
    from warlock.service import downloads

    rows = downloads.rows(_Svc(_config(tmp_path)))
    assert rows and all("present" in row for row in rows)
    # Nothing is on disk under a fresh root, and no label leans on a substring.
    assert not any(row["present"] for row in rows)
    assert not any("missing" in row["label"].lower() for row in rows)
    # The style LoRAs are in the list at all, which is what they were not
    # before: the pane's old missing-ness came from a substring main.py only
    # ever put into base-model labels.
    assert any(row["kind"] == "lora" for row in rows)


def test_the_child_clears_the_flag_only_in_its_own_environment(tmp_path):
    """The parent's HF_HUB_OFFLINE is untouched by running a fetch."""
    before = os.environ["HF_HUB_OFFLINE"]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import warlock, os; "
            "from warlock.pipelines import fetch_worker; "
            "print(os.environ['HF_HUB_OFFLINE'])",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "0", proc.stderr
    assert os.environ["HF_HUB_OFFLINE"] == before == "1"

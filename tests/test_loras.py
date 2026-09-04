"""Style LoRAs the user brings: the import door, the trainer's plumbing, the
``style_lock`` control, and the generic child-runner contract.

The trainer child itself needs a card and is exercised in the gpu lane; here
``rigging.run_worker`` is faked and what is under test is that the queue frees
the card, spawns the right module, and registers what came back through the
one import path.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from warlock import generation, models, progress, rigging, vram
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import lora_train
from warlock.queue import Worker
from warlock.service import loras as svc_loras
from warlock.service.errors import Invalid
from warlock.service.validation import DERIVED_PARAMS
from warlock.studio.panes import app_settings

# --- the pure module -----------------------------------------------------------------


def test_check_steps_names_the_range():
    assert lora_train.check_steps(lora_train.DEFAULT_STEPS) == lora_train.DEFAULT_STEPS
    for bad in (lora_train.MIN_STEPS - 1, lora_train.MAX_STEPS + 1, "x", None):
        with pytest.raises(ValueError):
            lora_train.check_steps(bad)


def test_the_spec_is_strings_and_numbers_only(tmp_path):
    spec = lora_train.train_spec(
        tmp_path / "base", [tmp_path / "a.png"], tmp_path / "out", tmp_path,
        trigger="cosmos style", steps=300, seed=7,
    )
    assert spec["op"] == "train" and spec["steps"] == 300 and spec["seed"] == 7
    assert spec["images"] == [str(tmp_path / "a.png")]
    assert Path(spec["result_path"]).parent == tmp_path
    json.dumps(spec)


def test_the_report_line():
    assert lora_train.report_line({"steps": 800, "images": 24, "loss": 0.0812}) == (
        "800 steps over 24 images, final loss 0.081"
    )
    assert lora_train.report_line({"steps": 1}) is None
    assert lora_train.report_line(None) is None


def test_the_kind_is_in_every_stage_keyed_table():
    assert "lora_result" in DERIVED_PARAMS
    assert progress.phases_for("lora_train") is progress.PHASES_LORA_TRAIN
    need, image = vram.estimate_parts("lora_train", "reference", {}, exclusive=False)
    assert need == vram.LORA_TRAIN_GIB and image == 0.0


def test_the_trainer_module_imports_nothing_at_import_time():
    import importlib

    module = importlib.import_module("warlock.pipelines.lora_train_worker")
    assert module.MARKER == lora_train.MARKER


# --- the generic child runner ---------------------------------------------------------


def test_run_worker_spawns_the_named_module_and_reads_its_marker(monkeypatch, tmp_path):
    import subprocess

    seen = {}

    class _Proc:
        pid = 4242
        args = ["x"]

        def __init__(self, argv, **kw):
            seen["argv"] = argv
            import io

            self.stdin = io.StringIO()
            self.stdout = io.StringIO("[train] 0.500 Step 5/10\nnoise\n")

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def popen(argv, **kw):
        proc = _Proc(argv, **kw)
        (tmp_path / ".r.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(rigging.winjob, "assign", lambda pid: True)
    monkeypatch.setattr(rigging.winjob, "track", lambda pid, what: seen.__setitem__("track", what))
    monkeypatch.setattr(rigging.winjob, "untrack", lambda pid: None)
    progress_seen = []
    out = rigging.run_worker(
        {"result_path": str(tmp_path / ".r.json")},
        on_progress=lambda f, label: progress_seen.append((f, label)),
        module="warlock.pipelines.lora_train_worker",
        marker="train",
        name="LoRA trainer",
    )
    assert out == {"ok": True}
    assert seen["argv"][-2:] == ["-m", "warlock.pipelines.lora_train_worker"]
    assert seen["track"] == "lora trainer"
    assert progress_seen == [(0.5, "Step 5/10")]


# --- the import door --------------------------------------------------------------------


def _adapter(tmp_path: Path, name: str = "mystyle.safetensors") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00" * 64)
    return path


@pytest.fixture(autouse=True)
def _clean_registry():
    before = dict(models.STYLE_LORAS)
    yield
    models.STYLE_LORAS.clear()
    models.STYLE_LORAS.update(before)


def test_import_registers_the_adapter_and_remove_forgets_it(svc, tmp_path):
    out = svc_loras.import_lora(
        svc, _adapter(tmp_path), label="Cosmos", trigger_text="cosmos style",
        tuned_weight=0.8, commercial=True,
    )
    key = out["key"]
    assert key.startswith("imported_")
    assert key in models.STYLE_LORAS
    entry = models.STYLE_LORAS[key]
    assert entry.trigger == "cosmos style" and entry.default_weight == 0.8
    assert (Path(svc.config.t2i_model_root) / "loras" / entry.filename).exists()
    assert [m["key"] for m in svc_loras.imported(svc)] == [key]
    rows = svc_loras.catalog(svc)
    assert any(row["key"] == key and row["source"] == "local file" for row in rows)

    svc_loras.remove_lora(svc, key)
    assert key not in models.STYLE_LORAS
    assert svc_loras.imported(svc) == []
    assert not (Path(svc.config.t2i_model_root) / "loras" / entry.filename).exists()


def test_a_built_in_style_cannot_be_removed(svc):
    with pytest.raises(Invalid):
        svc_loras.remove_lora(svc, "render3d")
    assert "render3d" in models.STYLE_LORAS


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"label": ""}, "label"),
        ({"label": "x" * 65}, "label"),
        ({"family": "sd15"}, "family"),
        ({"tuned_weight": 0.0}, "tuned_weight"),
        ({"tuned_weight": models.LORA_WEIGHT_MAX + 0.1}, "tuned_weight"),
        ({"trigger_text": "t" * 65}, "trigger_text"),
    ],
)
def test_the_import_door_refuses_by_field(svc, tmp_path, kwargs, field):
    base = {"label": "ok"}
    base.update(kwargs)
    with pytest.raises(Invalid) as info:
        svc_loras.import_lora(svc, _adapter(tmp_path), **base)
    assert info.value.field == field


def test_a_non_safetensors_file_is_refused(svc, tmp_path):
    bad = tmp_path / "style.ckpt"
    bad.write_bytes(b"x")
    with pytest.raises(Invalid) as info:
        svc_loras.import_lora(svc, bad, label="x")
    assert info.value.field == "source"


def test_the_service_registers_imported_adapters_at_startup(svc, tmp_path):
    out = svc_loras.import_lora(svc, _adapter(tmp_path), label="Startup")
    models.STYLE_LORAS.pop(out["key"])
    from warlock.service import WarlockService

    WarlockService(svc.config, svc.store)
    assert out["key"] in models.STYLE_LORAS


def test_concurrent_register_and_iterate_does_not_raise(tmp_path):
    """models.STYLE_LORAS is mutated in place from a worker thread while the
    frame thread reads it every frame (settings_2d, main.ctx.style_loras,
    models.loras_by_base/catalog). Before the STYLE_LORAS_LOCK +
    style_loras_snapshot() fix, a reader iterating the live dict while a
    writer inserted/popped could raise ``RuntimeError: dictionary changed
    size during iteration`` -- rare under the GIL, but real, since a
    dict-comprehension's iterator yields control between elements.

    This drives register/remove on one thread and every full-table reader on
    another for a few hundred rounds each; the assertion is simply that
    nothing raised.
    """
    import threading

    root = tmp_path / "loras"
    root.mkdir()
    keys = [f"concurrent_lora_{i}" for i in range(12)]
    manifests = [
        {
            "key": key,
            "label": key,
            "family": models.FAMILY_SDXL,
            "trigger_text": "",
            "tuned_weight": models.DEFAULT_LORA_WEIGHT,
            "license": "",
            "commercial": True,
            "source": "test",
            "checksum": "",
            "filename": f"{key}.safetensors",
            "schema_version": 1,
        }
        for key in keys
    ]
    (root / "manifests.json").write_text(
        json.dumps({"version": 1, "manifests": manifests}), encoding="utf-8"
    )
    config = Config(t2i_model_root=root.parent)

    errors: list[BaseException] = []
    rounds = 300

    def _writer() -> None:
        try:
            for i in range(rounds):
                key = keys[i % len(keys)]
                if i % 2 == 0:
                    generation.remove_imported_lora(config, key)
                else:
                    # Restore the manifest row this pass removed, then
                    # re-register so insert and pop both fire under load.
                    (root / "manifests.json").write_text(
                        json.dumps({"version": 1, "manifests": manifests}), encoding="utf-8"
                    )
                    generation._forget_manifests(root / "manifests.json")
                    generation.register_imported_loras(config)
        except BaseException as exc:  # noqa: BLE001 -- captured for the assertion below
            errors.append(exc)

    def _reader() -> None:
        try:
            for _ in range(rounds):
                list(models.style_loras_snapshot().values())
                models.loras_by_base()
                models.catalog()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_writer), threading.Thread(target=_reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    for key in keys:
        models.STYLE_LORAS.pop(key, None)


# --- the training door -------------------------------------------------------------------


def _images(tmp_path: Path, count: int) -> list[Path]:
    folder = tmp_path / "art"
    folder.mkdir(exist_ok=True)
    out = []
    for index in range(count):
        path = folder / f"{index}.png"
        Image.new("RGB", (8, 8), (index * 10, 0, 0)).save(path)
        out.append(path)
    return out


@pytest.fixture
def base_present(monkeypatch):
    from warlock.service import loras as mod

    monkeypatch.setattr(mod, "check_base_model_weights", lambda *a, **k: None)


def test_the_training_door_copies_the_images_and_queues_a_row(svc, tmp_path, base_present):
    out = svc_loras.train_lora(
        svc, _images(tmp_path, 4), label="Cosmos", trigger="cosmos style", steps=200
    )
    row = svc.store.get(out["id"])
    assert row["kind"] == "lora_train" and row["status"] == "queued"
    assert row["params"]["trigger"] == "cosmos style"
    assert row["params"]["steps"] == 200 and row["params"]["images"] == 4
    assert row["params"]["base_model"] == models.DEFAULT_BASE_MODEL
    copied = sorted((svc.job_dir(out["id"]) / "train").glob("*.png"))
    assert len(copied) == 4


@pytest.mark.parametrize(
    ("count", "kwargs", "field"),
    [
        (2, {}, "images"),
        (4, {"label": ""}, "label"),
        (4, {"trigger": ""}, "trigger"),
        (4, {"steps": 5}, "steps"),
        (4, {"base_model": "turbo"}, "base_model"),
        (4, {"base_model": "sdxl"}, "base_model"),
    ],
)
def test_the_training_door_refuses_by_field(svc, tmp_path, base_present, count, kwargs, field):
    base = {"label": "Cosmos", "trigger": "cosmos style"}
    base.update(kwargs)
    with pytest.raises(Invalid) as info:
        svc_loras.train_lora(svc, _images(tmp_path, count), **base)
    assert info.value.field == field


def test_an_unreadable_image_is_refused_at_the_door(svc, tmp_path, base_present):
    paths = _images(tmp_path, 3)
    paths[1].write_bytes(b"not a png")
    with pytest.raises(Invalid) as info:
        svc_loras.train_lora(svc, paths, label="x", trigger="y")
    assert info.value.field == "images"


def test_the_training_door_needs_the_base_weights(svc, tmp_path):
    # The svc fixture materialises the default checkpoint's marker files so
    # text jobs are admitted; take the marker away and the door must refuse.
    from warlock import fetch

    spec = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    (fetch.base_model_dir(svc.config, spec) / "model_index.json").unlink()
    with pytest.raises(Invalid) as info:
        svc_loras.train_lora(svc, _images(tmp_path, 3), label="x", trigger="y")
    assert info.value.field == "base_model"


# --- the worker ------------------------------------------------------------------------------


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


async def test_a_training_job_frees_the_card_runs_the_trainer_and_registers(
    worker, monkeypatch
):
    calls = []
    stopped = []
    monkeypatch.setattr(worker.trellis, "stop", lambda: stopped.append(True))

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0, **kw):
        calls.append({"spec": spec, "timeout": timeout, **kw})
        on_progress(0.5, "Step 100/200")
        out = Path(spec["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / lora_train.WEIGHTS_NAME).write_bytes(b"\x00" * 32)
        return {"ok": True, "steps": 200, "images": 3, "rank": 16, "loss": 0.05,
                "weights": str(out / lora_train.WEIGHTS_NAME)}

    monkeypatch.setattr(rigging, "run_worker", fake)
    job_id = worker.store.create(
        "lora_train", "Cosmos",
        {"base_model": "sdxl_cfg", "label": "Cosmos", "trigger": "cosmos style", "steps": 200},
    )
    train_dir = worker.config.job_dir(job_id) / "train"
    train_dir.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (8, 8)).save(train_dir / f"{i:03d}.png")

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()
    row = worker.store.get(job_id)
    assert row["status"] == "done", row.get("error")
    assert stopped, "trellis was not stopped before the trainer took the card"
    call = calls[0]
    assert call["module"] == "warlock.pipelines.lora_train_worker"
    assert call["marker"] == lora_train.MARKER
    assert call["spec"]["trigger"] == "cosmos style" and call["spec"]["steps"] == 200
    assert len(call["spec"]["images"]) == 3
    assert Path(call["spec"]["base_dir"]) == worker.config.t2i_model_root / "sdxl-base-1.0"
    report = row["params"]["lora_result"]
    assert report["loss"] == 0.05
    key = report["lora"]["key"]
    assert key in models.STYLE_LORAS
    assert models.STYLE_LORAS[key].trigger == "cosmos style"
    assert models.STYLE_LORAS[key].default_weight == lora_train.TRAINED_WEIGHT
    assert generation.imported_lora(worker.config, key).source == f"trained:{job_id}"


async def test_a_trainer_that_wrote_nothing_fails_the_job(worker, monkeypatch):
    monkeypatch.setattr(worker.trellis, "stop", lambda: None)
    monkeypatch.setattr(
        rigging, "run_worker", lambda spec, **kw: {"ok": True, "steps": 1, "images": 1}
    )
    job_id = worker.store.create("lora_train", "x", {"trigger": "t", "steps": 100})
    train_dir = worker.config.job_dir(job_id) / "train"
    train_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(train_dir / "000.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()
    assert worker.store.get(job_id)["status"] == "error"


# --- style_lock reaches the door ---------------------------------------------------------


def test_style_lock_requires_the_encoder_weights_at_the_door(svc, monkeypatch):
    from warlock.service import tilesheets

    seen = {}

    def fake_required(mode, with_reference):
        seen["with_reference"] = with_reference
        return ()

    monkeypatch.setattr(tilesheets, "_required", fake_required)
    monkeypatch.setattr(tilesheets, "check_base_model_weights", lambda *a, **k: None)
    tilesheets._check_weights(svc, mode=tilesheets.MODE_MATERIALS, style_lock=True)
    assert seen["with_reference"] is True
    tilesheets._check_weights(svc, mode=tilesheets.MODE_MATERIALS, style_lock=False)
    assert seen["with_reference"] is False


def test_the_materials_form_carries_style_lock_into_the_request():
    src = Path(generation.__file__).read_text(encoding="utf-8")
    assert 'style_lock=bool(form.get("style_lock"))' in src
    pane = Path(app_settings.__file__).parent / "settings_2d.py"
    assert "Keep one style across the list" in pane.read_text(encoding="utf-8")


# --- the settings pane's helpers ------------------------------------------------------------


def test_the_import_form_maps_onto_the_doors_arguments():
    form = {
        "label": "Cosmos", "family": "sdxl", "trigger_text": "cosmos style",
        "tuned_weight": 0.7, "commercial": True,
    }
    assert app_settings.lora_import_kwargs(form) == {
        "label": "Cosmos", "family": "sdxl", "trigger_text": "cosmos style",
        "tuned_weight": 0.7, "commercial": True,
    }


def test_training_images_takes_only_image_files_in_the_folder(tmp_path):
    _images(tmp_path, 2)
    (tmp_path / "art" / "notes.txt").write_text("x")
    (tmp_path / "art" / "sub").mkdir()
    assert [p.name for p in app_settings.training_images(tmp_path / "art")] == ["0.png", "1.png"]
    assert app_settings.training_images(tmp_path / "missing") == []

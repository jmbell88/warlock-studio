"""The service surface: what a submit accepts, what it stores, what it refuses.

This was the HTTP suite. It is now the same behaviours asserted against the
service functions the desktop app calls, which is where they always lived --
the routes only ever translated arguments and exceptions.
"""

from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import warlock.config as config_mod
from warlock import guidance, models
from warlock.service import Conflict, Failed, Invalid, NotFound, NotReady, TooLarge
from warlock.service import derive as svc_derive
from warlock.service import export as svc_export
from warlock.service import files as svc_files
from warlock.service import jobs as svc_jobs
from warlock.service import system as svc_system


class FakeWorker:
    """The little of Worker the service actually reaches for."""

    def __init__(self) -> None:
        from warlock.progress import ProgressBus

        self.alive = True
        self.fatal = None
        self.current_job_id = None
        self.progress = ProgressBus()
        self.trellis = type("T", (), {"running": False})()
        self.wakes = 0
        self.cancelled: list[str] = []

    def wake(self) -> None:
        self.wakes += 1

    async def request_cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


@pytest.fixture
def worker(svc):
    svc.worker = FakeWorker()
    return svc.worker


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def _png_bytes(fmt: str = "PNG") -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (200, 50, 50, 255)).save(buf, fmt)
    return buf.getvalue()


def _job(svc, job_id):
    return svc_jobs.get_job(svc, job_id)


def _params(svc, job_id):
    return svc.store.get(job_id)["params"]


# --- creating jobs ----------------------------------------------------------


def test_create_text_job(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job = _job(svc, job_id)
    assert job["kind"] == "text"
    assert job["prompt"] == "a barrel"


def test_text_job_requires_prompt(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text")


def test_image_job_requires_upload(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image")


def test_create_image_job_stores_upload(svc):
    job_id = svc_jobs.create_job(svc, kind="image", image=_png_bytes())["id"]
    assert "input.png" in _job(svc, job_id)["files"]


def test_webp_upload_is_normalized_to_png(svc, assets):
    # trellis.cpp only decodes PNG/JPEG; any upload (webp, bmp, ...) must be
    # stored as a real PNG regardless of its original format.
    from PIL import Image

    job_id = svc_jobs.create_job(svc, kind="image", image=_png_bytes("WEBP"))["id"]
    assert Image.open(assets / job_id / "input.png").format == "PNG"


def test_opaque_jpeg_upload_preserves_rgb(svc, assets):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "JPEG")
    job_id = svc_jobs.create_job(svc, kind="image", image=buf.getvalue())["id"]
    assert Image.open(assets / job_id / "input.png").mode == "RGB"


def test_invalid_image_upload_rejected(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=b"not an image")


# --- input limits -----------------------------------------------------------


def test_oversized_upload_is_rejected_before_decode(svc):
    blob = b"\x89PNG\r\n\x1a\n" + b"\0" * (20 * 1024 * 1024)
    with pytest.raises(TooLarge):
        svc_jobs.create_job(svc, kind="image", image=blob)


def test_a_decompression_bomb_is_rejected_by_pixel_count(svc):
    # 5000x4000 = 20 MP: tiny on disk as a flat PNG, over the 16 MP cap decoded.
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (5000, 4000)).save(buf, "PNG")
    with pytest.raises(Invalid, match="pixel"):
        svc_jobs.create_job(svc, kind="image", image=buf.getvalue())


def test_an_overlong_prompt_is_rejected(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x" * 1001)


@pytest.mark.parametrize("bad", [-1, 2**31])
def test_an_out_of_range_seed_is_rejected(svc, bad):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", seed=bad)


@pytest.mark.parametrize("bad", [1.5, True, "7", 3.0])
def test_a_seed_that_is_not_a_whole_number_is_rejected(svc, bad):
    """The range check alone let floats and bools through: 0 <= 1.5 <= MAX_SEED
    and 0 <= True are both true."""
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", seed=bad)


def test_favorite_must_be_a_real_bool(svc):
    """bool("false") is True, so the string form favourited what it meant to
    unfavourite."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid):
        svc_jobs.update_job(svc, job_id, {"favorite": "false"})
    assert svc_jobs.update_job(svc, job_id, {"favorite": True})["favorite"] == 1


def test_a_failed_db_insert_leaves_no_orphan_job_dir(svc, assets, monkeypatch):
    """input.png is deliberately written before the row exists (the worker's
    next_queued poll must never claim an image job with no file on disk), so
    a failed insert has to clean up after itself or dirs accumulate forever."""
    before = {p.name for p in assets.iterdir() if p.is_dir()}

    def explode(*args, **kwargs):
        raise RuntimeError("db is on fire")

    monkeypatch.setattr(svc.store, "create", explode)
    with pytest.raises(RuntimeError):
        svc_jobs.create_job(svc, kind="image", image=_png_bytes())
    assert {p.name for p in assets.iterdir() if p.is_dir()} == before


def test_invalid_resolution_rejected(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", resolution=999)


def test_input_png_written_before_the_db_row_is_created(svc, assets, monkeypatch):
    original = svc.store.create
    seen = {}

    def spying_create(kind, prompt, params, job_id=None, **kwargs):
        if job_id is not None:
            seen["exists"] = (assets / job_id / "input.png").exists()
        return original(kind, prompt, params, job_id=job_id, **kwargs)

    monkeypatch.setattr(svc.store, "create", spying_create)
    svc_jobs.create_job(svc, kind="image", image=_png_bytes())
    assert seen.get("exists") is True


# --- guidance ---------------------------------------------------------------


def test_the_guidance_catalog_is_served(svc):
    body = svc_system.guidance_catalog(svc)
    assert set(body["fields"]) == {
        "genre", "art_style", "category", "platform", "framing",
        "base_model", "style_lora", "ip_adapter", "control",
        "material", "condition", "setting", "palette", "emissive", "rarity",
        "silhouette", "mood",
    }
    assert body["defaults"]["platform"] in {o["key"] for o in body["fields"]["platform"]}
    assert body["defaults"]["base_model"] in {o["key"] for o in body["fields"]["base_model"]}


def test_guidance_is_stored_on_the_job(svc):
    job_id = svc_jobs.create_job(
        svc,
        kind="text",
        prompt="a plasma rifle",
        guidance_fields={
            "genre": "scifi",
            "art_style": "ps1",
            "category": "weapon",
            "platform": "2d",
        },
    )["id"]
    params = _params(svc, job_id)
    assert params["genre"] == "scifi"
    assert params["art_style"] == "ps1"
    assert params["category"] == "weapon"
    assert params["resolution"] == 512  # derived from the 2D platform preset
    assert params["size_m"] == 1.0  # derived from the weapon category default


def test_explicit_size_overrides_the_category_default(svc):
    job_id = svc_jobs.create_job(
        svc, kind="text", prompt="x", size_m=0.25, guidance_fields={"category": "weapon"}
    )["id"]
    assert _params(svc, job_id)["size_m"] == 0.25


def test_a_job_without_guidance_still_gets_usable_defaults(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    params = _params(svc, job_id)
    assert params["resolution"] in (512, 1024, 1536)
    assert params["size_m"] > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("genre", "nonsense"), ("category", "nonsense"), ("platform", "nonsense")],
)
def test_invalid_guidance_rejected(svc, field, value):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", guidance_fields={field: value})


def test_an_out_of_range_size_is_rejected(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", size_m=500)


def test_rejected_guidance_leaves_no_input_png_behind(svc, assets):
    with pytest.raises(Invalid):
        svc_jobs.create_job(
            svc, kind="image", image=_png_bytes(), guidance_fields={"genre": "nonsense"}
        )
    assert not list(assets.glob("*/input.png"))


def test_the_newer_guidance_fields_are_accepted(svc):
    job_id = svc_jobs.create_job(
        svc,
        kind="text",
        prompt="a rifle",
        guidance_fields={"material": "steel", "condition": "pristine", "rarity": "epic"},
    )["id"]
    params = _params(svc, job_id)
    assert params["material"] == "steel"
    assert params["condition"] == "pristine"
    assert params["rarity"] == "epic"


def test_an_unknown_guidance_value_is_refused(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(
            svc, kind="text", prompt="x", guidance_fields={"material": "unobtainium"}
        )


# --- prompt preview ---------------------------------------------------------


def test_the_prompt_preview_returns_the_composed_prompt(svc):
    body = svc_system.prompt_preview(svc, {"genre": "scifi"}, "a barrel")
    assert body["prompt"].startswith("a barrel, ")
    assert "science fiction" in body["prompt"]
    assert body["negative_prompt"]  # falls back to the guidance default


def test_the_prompt_preview_rejects_unknown_guidance(svc):
    with pytest.raises(Invalid):
        svc_system.prompt_preview(svc, {"material": "unobtainium"}, "x")


def test_the_prompt_preview_degrades_to_null_tokens_without_a_tokenizer(svc, monkeypatch):
    from warlock.pipelines import prompt as prompt_pipeline

    def _raise(_model_dir, _family="sdxl"):
        raise OSError("no tokenizer on disk")

    monkeypatch.setattr(prompt_pipeline, "load_tokenizers", _raise)
    body = svc_system.prompt_preview(svc, {}, "a barrel")
    assert body["tokens"] is None
    assert body["chunks"] is None


# --- cancelling and deleting ------------------------------------------------


def test_cancel_and_delete(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc_jobs.cancel_job(svc, job_id)
    assert _job(svc, job_id)["status"] == "cancelled"
    svc_jobs.delete_job(svc, job_id)
    with pytest.raises(NotFound):
        _job(svc, job_id)


def test_cancelling_a_running_job_reaches_the_worker(svc, worker):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "running")
    worker.current_job_id = job_id
    svc_jobs.cancel_job(svc, job_id)
    assert worker.cancelled == [job_id]
    assert _job(svc, job_id)["status"] == "cancelled"


def test_cancel_is_idempotent_when_already_cancelled(svc, worker):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "running")
    svc.store.cancel(job_id)  # already cancelled by the time this call lands
    assert svc_jobs.cancel_job(svc, job_id) == {"ok": True}


# --- health and progress ----------------------------------------------------


def test_health_reports_the_worker_and_the_doctor_checks(svc, worker):
    body = svc_system.health(svc)
    assert body["ok"] is True
    assert body["worker_alive"] is True
    assert body["fatal"] is None
    assert len(body["checks"]) == (
        11
        + len(models.BASE_MODELS)
        + len(models.STYLE_LORAS)
        + len(models.IP_ADAPTERS)
        + len(models.CONTROLNETS)
        + len(models.METRIC_MODELS)
        + len(models.MATTING_MODELS)
        + len(models.POSE_MODELS)
    )


def test_current_checks_answers_without_a_worker_and_through_the_cache(svc):
    # What the header health dot polls: no worker at all (the studio before
    # the runtime finishes, or a headless caller) must still get rows, and a
    # second call inside the TTL must be the cached list, not a re-probe.
    first = svc_system.current_checks(svc)
    assert first and all(hasattr(c, "ok") for c in first)
    assert svc_system.current_checks(svc) is first


def test_health_does_not_rerun_the_doctor_suite_on_every_call(svc, worker, monkeypatch):
    """The suite binds a socket, stats the disk and probes a dozen paths, and
    the UI asks continuously. None of those answers changes second to second."""
    from warlock import doctor

    calls = []
    real = doctor.run_checks
    monkeypatch.setattr(
        doctor, "run_checks", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    svc_system.health(svc)
    svc_system.health(svc)
    assert len(calls) == 1


def test_progress_is_absent_when_idle(svc, worker):
    assert worker.progress.snapshot() is None


def test_a_job_carries_progress_only_while_it_runs(svc, worker):
    a = svc_jobs.create_job(svc, kind="text", prompt="a")["id"]
    b = svc_jobs.create_job(svc, kind="text", prompt="b")["id"]
    worker.progress.begin(a, "text")
    worker.progress.update(a, phase="trellis", label="Texture", inner=0.3)
    try:
        jobs = {j["id"]: j for j in svc_jobs.list_jobs(svc)}
        assert jobs[a]["progress"]["label"] == "Texture"
        assert jobs[b]["progress"] is None
    finally:
        worker.progress.end(a)


def test_progress_clears_when_the_job_ends(svc, worker):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    worker.progress.begin(job_id, "text")
    worker.progress.end(job_id)
    assert _job(svc, job_id)["progress"] is None


def test_submitting_a_job_wakes_the_worker(svc, worker):
    """Dispatch polls on an interval; without this a submit waits out up to a
    second of it before anything starts."""
    svc_jobs.create_job(svc, kind="text", prompt="a barrel")
    assert worker.wakes == 1


# --- file attachment --------------------------------------------------------


def test_model_glb_hidden_until_the_job_is_done(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "model.glb").write_bytes(b"placeholder")

    running = {"status": "running"}
    svc_files.attach_files(running, job_dir)
    assert "model.glb" not in running["files"]

    done = {"status": "done"}
    svc_files.attach_files(done, job_dir)
    assert "model.glb" in done["files"]


def test_input_png_stays_existence_based_regardless_of_status(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "input.png").write_bytes(b"placeholder")

    job = {"status": "running"}
    svc_files.attach_files(job, job_dir)
    assert "input.png" in job["files"]


# --- job id validation ------------------------------------------------------
#
# config.job_dir() is a bare data_dir / job_id join. Only get_file could
# actually be steered (the others gate on a DB lookup first), but the check
# belongs at the boundary on all of them.


@pytest.mark.parametrize(
    "bad",
    [
        "not-hex-here",
        "abcdef",            # right alphabet, wrong length
        "abcdef0123456",     # one too many
        "ABCDEF012345",      # uuid4().hex is lowercase
        "....",
        "..",
        "../../etc",
        "0123456789ab.",
        "",
    ],
)
def test_a_malformed_job_id_is_rejected_by_every_entry_point(svc, bad):
    with pytest.raises(NotFound):
        svc_jobs.get_job(svc, bad)
    with pytest.raises(NotFound):
        svc_derive.get_file(svc, bad, "model.glb")
    with pytest.raises(NotFound):
        svc_jobs.cancel_job(svc, bad)
    with pytest.raises(NotFound):
        svc_jobs.rerun_job(svc, bad, mode="reroll")
    with pytest.raises(NotFound):
        svc_jobs.delete_job(svc, bad)


# --- rerun ------------------------------------------------------------------


def test_reroll_copies_the_prompt_and_guidance_with_a_new_seed(svc):
    src = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", seed=42, guidance_fields={"genre": "fantasy"}
    )["id"]
    # Pretend the worker finished it and recorded its derived values.
    svc.store.merge_params(
        src,
        {
            "composed_prompt": "a barrel, fantasy",
            "scale_factor": 2.0,
            "mesh_audit": {"worst": 0.3, "mean": 0.2, "faces": 9, "resolution": 512},
        },
    )

    new_id = svc_jobs.rerun_job(svc, src, mode="reroll")["id"]
    new = svc.store.get(new_id)
    assert new["kind"] == "text"
    assert new["prompt"] == "a barrel"
    assert new["params"]["genre"] == "fantasy"
    assert new["params"]["seed"] != 42
    assert new["params"]["rerun_of"] == src
    # Derived values describe the source run, not this one.
    for key in ("composed_prompt", "scale_factor", "mesh_audit"):
        assert key not in new["params"]


def test_reroll_accepts_an_explicit_seed(svc):
    src = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    out = svc_jobs.rerun_job(svc, src, mode="reroll", seed=7)
    assert out["seed"] == 7
    assert _params(svc, out["id"])["seed"] == 7


def test_remesh_reuses_the_reference_image_as_an_image_job(svc, assets):
    """The point of remesh: when SDXL drew a good reference but trellis made a
    poor mesh, retry the 3D stage without paying for SDXL again."""
    src = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    src_dir = assets / src
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "input.png").write_bytes(_png_bytes())

    new_id = svc_jobs.rerun_job(svc, src, mode="remesh")["id"]
    new = svc.store.get(new_id)
    assert new["kind"] == "image"
    assert (assets / new_id / "input.png").read_bytes() == _png_bytes()
    # The prompt rides along for display even though it won't be used again.
    assert new["prompt"] == "a barrel"


def test_remesh_without_a_reference_image_is_rejected(svc):
    src = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    with pytest.raises(Invalid):
        svc_jobs.rerun_job(svc, src, mode="remesh")


def test_reroll_of_an_image_job_carries_the_upload_across(svc, assets):
    src = svc_jobs.create_job(svc, kind="image", image=_png_bytes())["id"]
    new_id = svc_jobs.rerun_job(svc, src, mode="reroll")["id"]
    assert (assets / new_id / "input.png").exists()


def test_a_failed_rerun_insert_leaves_no_orphan_job_dir(svc, assets, monkeypatch):
    """Same ordering as create_job -- the dir is written first so the worker
    can never claim a job with no input.png -- so it needs the same cleanup."""
    src = svc_jobs.create_job(svc, kind="image", image=_png_bytes())["id"]
    before = {p.name for p in assets.iterdir() if p.is_dir()}

    def explode(*args, **kwargs):
        raise RuntimeError("db is on fire")

    monkeypatch.setattr(svc.store, "create", explode)
    with pytest.raises(RuntimeError):
        svc_jobs.rerun_job(svc, src, mode="reroll")
    assert {p.name for p in assets.iterdir() if p.is_dir()} == before


def test_a_failed_promotion_insert_leaves_no_orphan_job_dir(svc, assets, monkeypatch):
    src = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference")["id"]
    src_dir = assets / src
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(src, "done")
    before = {p.name for p in assets.iterdir() if p.is_dir()}

    def explode(*args, **kwargs):
        raise RuntimeError("db is on fire")

    monkeypatch.setattr(svc.store, "create", explode)
    with pytest.raises(RuntimeError):
        svc_jobs.promote_to_model(svc, src)
    assert {p.name for p in assets.iterdir() if p.is_dir()} == before


def test_rerun_rejects_an_unknown_mode(svc):
    src = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid):
        svc_jobs.rerun_job(svc, src, mode="sideways")


def test_rerun_of_a_missing_job_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_jobs.rerun_job(svc, "0123456789ab", mode="reroll")


def test_remesh_inherits_a_legacy_seed_as_the_reference_and_rerolls_only_the_mesh(svc, assets):
    # A source job that predates the split only has "seed" -- no reference_seed
    # or mesh_seed keys at all. remesh must still treat that lone seed as the
    # reference to inherit, and must not also reuse it for the mesh stage.
    src = svc_jobs.create_job(svc, kind="text", prompt="a barrel", seed=42)["id"]
    (assets / src).mkdir(parents=True, exist_ok=True)
    (assets / src / "input.png").write_bytes(_png_bytes())

    params = _params(svc, svc_jobs.rerun_job(svc, src, mode="remesh")["id"])
    assert params["reference_seed"] == 42
    assert params["mesh_seed"] != 42


def test_remesh_inherits_the_explicit_reference_seed_not_the_legacy_one(svc, assets):
    # This source already went through the split and has both keys, with
    # different values -- if remesh fell back to plain "seed" instead, matching
    # fixture values would hide it, so the two are deliberately made to differ.
    src = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", seed=42, reference_seed=99, mesh_seed=42
    )["id"]
    (assets / src).mkdir(parents=True, exist_ok=True)
    (assets / src / "input.png").write_bytes(_png_bytes())

    params = _params(svc, svc_jobs.rerun_job(svc, src, mode="remesh")["id"])
    assert params["reference_seed"] == 99
    assert params["mesh_seed"] != 42
    assert params["mesh_seed"] != 99


def test_reroll_gives_both_seeds_the_same_fresh_value(svc):
    src = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", seed=42, reference_seed=11, mesh_seed=22
    )["id"]
    params = _params(svc, svc_jobs.rerun_job(svc, src, mode="reroll")["id"])
    assert params["reference_seed"] == params["mesh_seed"]
    assert params["reference_seed"] != 11
    assert params["mesh_seed"] != 22


def test_rerun_carries_the_model_selection_forward(svc):
    """base_model/style_lora are inputs, not derived from the source run, so
    unlike composed_prompt they must survive the derived-key strip."""
    first = svc_jobs.create_job(
        svc,
        kind="text",
        prompt="a barrel",
        lora_weight=0.6,
        guidance_fields={"base_model": "sdxl", "style_lora": "ps1"},
    )["id"]
    params = _params(svc, svc_jobs.rerun_job(svc, first)["id"])
    assert params["base_model"] == "sdxl"
    assert params["style_lora"] == "ps1"
    assert params["lora_weight"] == 0.6


def test_rerun_strips_every_derived_param_not_just_the_original_three(svc):
    """The strip list has to keep up with what the worker records: a reroll
    wearing the source's mesh_report claims a quality verdict about a mesh
    that doesn't exist yet."""
    src = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    svc.store.merge_params(
        src,
        {
            "mesh_report": {"status": "ready", "reasons": []},
            "optimize": {"achieved": 5},
            "transform": {"scale": 1.0},
            "weighting": "automatic",
            "bone_count": 19,
        },
    )
    params = _params(svc, svc_jobs.rerun_job(svc, src, mode="reroll")["id"])
    for key in ("mesh_report", "optimize", "transform", "weighting", "bone_count"):
        assert key not in params, key


# --- models and seeds -------------------------------------------------------


def test_the_model_selection_is_stored_on_the_job(svc):
    job_id = svc_jobs.create_job(
        svc,
        kind="text",
        prompt="a plasma rifle",
        lora_weight=1.1,
        guidance_fields={"base_model": "sdxl", "style_lora": "render3d"},
    )["id"]
    params = _params(svc, job_id)
    assert params["base_model"] == "sdxl"
    assert params["style_lora"] == "render3d"
    assert params["lora_weight"] == 1.1


def test_an_unknown_base_model_is_refused(svc):
    with pytest.raises(Invalid, match="base_model"):
        svc_jobs.create_job(
            svc, kind="text", prompt="x", guidance_fields={"base_model": "dall-e"}
        )


def test_an_out_of_range_lora_weight_is_refused(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(
            svc, kind="text", prompt="x", lora_weight=9, guidance_fields={"style_lora": "ps1"}
        )


def test_the_seeds_split_and_a_legacy_seed_is_still_read(svc):
    job_id = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", reference_seed=11, mesh_seed=22
    )["id"]
    params = _params(svc, job_id)
    assert params["reference_seed"] == 11
    assert params["mesh_seed"] == 22

    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", seed=5)["id"]
    params = _params(svc, job_id)
    assert params["reference_seed"] == 5
    assert params["mesh_seed"] == 5


# --- the two stages ---------------------------------------------------------


def test_a_reference_output_stops_before_the_mesh(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="reference")["id"]
    assert svc.store.get(job_id)["stage"] == "reference"


def test_the_default_output_is_still_a_model(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    assert svc.store.get(job_id)["stage"] == "model"


def test_an_image_job_cannot_request_a_reference_output(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=_png_bytes(), output="reference")


def _done_reference(svc, **extra):
    """A finished reference-stage job, standing in for a real pipeline run."""
    ref_id = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference", **extra
    )["id"]
    job_dir = svc.job_dir(ref_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"png")
    svc.store.set_status(ref_id, "done")
    return ref_id


def test_promote_creates_a_child_model_job(svc):
    ref_id = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference"
    )["id"]
    reference_bytes = b"png-distinctive-parent-bytes-\x00\x01\x02"
    job_dir = svc.job_dir(ref_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(reference_bytes)
    svc.store.set_status(ref_id, "done")

    child_id = svc_jobs.promote_to_model(svc, ref_id)["id"]
    child = svc.store.get(child_id)
    assert child["parent_id"] == ref_id
    assert child["kind"] == "image"
    assert child["stage"] == "model"
    # The load-bearing half of promote: the child's input.png must actually be
    # the parent's bytes, not just exist. A worker picking this job up minutes
    # later fails without it.
    assert (svc.job_dir(child_id) / "input.png").read_bytes() == reference_bytes


def test_promote_reuses_the_parents_reference_seed(svc):
    ref_id = _done_reference(svc, reference_seed=11)
    out = svc_jobs.promote_to_model(svc, ref_id)
    params = _params(svc, out["id"])
    # The reference is reused verbatim -- promotion must not redraw it.
    assert params["reference_seed"] == 11
    assert params["mesh_seed"] == out["mesh_seed"]


def test_promote_records_the_mesh_overrides_it_was_given(svc):
    # The 3D pane owns the mesh-side decisions, and the reference was made
    # without any of them being asked -- so promotion has to be able to say
    # what they are, not just inherit whatever the reference happened to store.
    ref_id = _done_reference(svc, size_m=0.4, guidance_fields={"platform": "2d"})
    out = svc_jobs.promote_to_model(
        svc,
        ref_id,
        platform="3d",
        size_m=2.5,
        bg_removal="threshold",
        profile="raw",
        rig=True,
        rig_template="quadruped",
    )
    params = _params(svc, out["id"])
    assert params["platform"] == "3d"
    assert params["size_m"] == 2.5
    assert params["bg_removal"] == "threshold"
    assert params["profile"] == "raw"
    assert params["rig"] is True
    assert params["rig_template"] == "quadruped"
    # The resolution the worker actually sends trellis follows the new
    # platform; the reference's stored one must not survive to contradict it.
    assert params["resolution"] == guidance.PLATFORMS["3d"].resolution


def test_promote_without_overrides_inherits_the_reference(svc):
    # The whole point of the overrides being optional: omitting them all must
    # leave the reference's own settings entirely alone.
    ref_id = _done_reference(
        svc, size_m=0.4, bg_removal="birefnet", guidance_fields={"platform": "2d"}
    )
    params = _params(svc, svc_jobs.promote_to_model(svc, ref_id)["id"])
    assert params["platform"] == "2d"
    assert params["size_m"] == 0.4
    assert params["bg_removal"] == "birefnet"


def test_a_job_gets_the_learned_matte_when_its_weights_are_on_disk(svc):
    """The default-matte decision, end to end: the gate is applied where a job is
    actually
    created, not only in ``guidance``. ``auto`` went 0 for 80 in the 2026-08-07
    review and every accept was birefnet, so this is the default the queue must
    see -- there is no point in the constant moving if ``create_job`` does not
    read it."""
    from warlock import guidance as guidance_mod

    svc.config.trellis_models_dir.mkdir(parents=True, exist_ok=True)
    (svc.config.trellis_models_dir / guidance_mod.BIREFNET_WEIGHTS).write_bytes(b"")
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a rogue")["id"]
    assert _params(svc, job_id)["bg_removal"] == "birefnet"


def test_a_job_falls_back_to_auto_when_the_matte_weights_are_absent(svc):
    # The gate's other half. The svc fixture points trellis_models_dir at an
    # empty tmp dir, which is this case.
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a rogue")["id"]
    assert _params(svc, job_id)["bg_removal"] == "auto"


def test_the_form_default_matches_what_a_submit_would_pick(svc):
    """The catalog's ``defaults`` is what the generate pane initialises from,
    so an ungated one would show a matte the job then refuses to run at."""
    submitted = _params(svc, svc_jobs.create_job(svc, kind="text", prompt="x")["id"])
    assert svc_system.guidance_catalog(svc)["defaults"]["bg_removal"] == (
        submitted["bg_removal"]
    )


def test_promote_migrates_a_reference_carrying_legacy_guidance(svc):
    # Every reference on disk from before the platform and art-style tables
    # were renamed carries the old keys, and an override re-normalizes the
    # whole stored dict -- so without the alias table this is a 400 on an
    # otherwise valid asset, and the user has no way to edit it.
    ref_id = _done_reference(svc)
    svc.store.merge_params(ref_id, {"platform": "hero", "art_style": "handpainted"})
    params = _params(svc, svc_jobs.promote_to_model(svc, ref_id, size_m=2.0)["id"])
    assert params["platform"] == "3d"
    assert params["art_style"] == "ps2"


def test_promote_does_not_claim_to_be_a_rerun(svc):
    # A rerolled reference carries rerun_of; the promotion's lineage is
    # parent_id, so carrying rerun_of onto the child would claim the mesh job
    # is a rerun of a *reference* it never was.
    ref_id = _done_reference(svc)
    svc.store.merge_params(ref_id, {"rerun_of": "aaaaaaaaaaaa"})
    params = _params(svc, svc_jobs.promote_to_model(svc, ref_id)["id"])
    assert "rerun_of" not in params


def test_promote_can_clear_an_inherited_rig_request(svc):
    ref_id = _done_reference(svc, rig=True)
    params = _params(svc, svc_jobs.promote_to_model(svc, ref_id, rig=False)["id"])
    assert "rig" not in params
    assert "rig_template" not in params


@pytest.mark.parametrize(
    "bad",
    [
        {"platform": "playstation-9"},
        {"size_m": 9999},
        {"bg_removal": "magic"},
        {"profile": "not-a-tier"},
        {"rig": True, "rig_template": "octopus"},
    ],
)
def test_promote_with_an_unusable_override_creates_nothing(svc, bad):
    # Rejected at the door, for the same reason create_job validates up front:
    # an unusable value should cost the request, not the two minutes of GPU
    # that would precede the step that finally notices it.
    ref_id = _done_reference(svc)
    before = len(svc_jobs.list_jobs(svc))
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, ref_id, **bad)
    assert len(svc_jobs.list_jobs(svc)) == before


def test_promote_without_a_reference_is_refused(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, job_id)


def test_reroll_of_a_reference_stays_a_reference(svc):
    # "try another" delegates to reroll -- it must not silently fall through to
    # the default "model" stage and pay for a trellis run.
    ref_id = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference"
    )["id"]
    new_id = svc_jobs.rerun_job(svc, ref_id)["id"]
    assert svc.store.get(new_id)["stage"] == "reference"


def test_promoting_a_model_stage_job_is_refused(svc):
    # A plain job's stage is already "model" -- there is no reference to
    # promote, and it must not be treated as one.
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, job_id)


def test_promoting_a_not_yet_done_reference_is_refused(svc):
    ref_id = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference"
    )["id"]
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, ref_id)


def test_promotion_strips_every_derived_param(svc, assets):
    src = svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="reference")["id"]
    (assets / src).mkdir(parents=True, exist_ok=True)
    (assets / src / "input.png").write_bytes(_png_bytes())
    svc.store.merge_params(src, {"mesh_report": {"status": "ready"}, "transform": {"scale": 1.0}})
    svc.store.set_status(src, "done")

    params = _params(svc, svc_jobs.promote_to_model(svc, src)["id"])
    for key in ("mesh_report", "transform"):
        assert key not in params, key


# --- candidate batches ------------------------------------------------------


def test_count_creates_n_reference_jobs_with_distinct_seeds(svc):
    body = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference", count=4
    )
    assert len(body["ids"]) == 4
    assert body["id"] == body["ids"][0]
    assert len({_params(svc, i)["reference_seed"] for i in body["ids"]}) == 4
    # If stage regressed to the default "model", candidates 1..N-1 would each
    # silently pay a full multi-minute trellis run instead of stopping after
    # the reference image.
    for job_id in body["ids"]:
        assert svc.store.get(job_id)["stage"] == "reference"


def test_count_above_one_requires_a_reference_output(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", count=3)


def test_count_is_bounded(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", output="reference", count=99)


def test_candidate_zero_keeps_the_requested_seed(svc):
    # A pinned seed must still land on candidate 0 verbatim -- only 1..N-1 fan
    # out to fresh random seeds.
    body = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference", count=3, seed=424242
    )
    params = _params(svc, body["ids"][0])
    assert params["reference_seed"] == 424242
    assert params["seed"] == 424242


def test_a_single_text_job_behaves_exactly_as_before(svc):
    body = svc_jobs.create_job(svc, kind="text", prompt="a barrel", count=1)
    assert body["ids"] == [body["id"]]
    job = svc.store.get(body["id"])
    assert job["kind"] == "text"
    assert job["prompt"] == "a barrel"


def test_a_single_image_job_still_writes_input_png(svc, assets):
    body = svc_jobs.create_job(svc, kind="image", image=_png_bytes(), count=1)
    assert body["ids"] == [body["id"]]
    assert (assets / body["id"] / "input.png").exists()


def test_candidate_ids_are_distinct_and_all_exist(svc):
    ids = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference", count=4
    )["ids"]
    assert len(set(ids)) == 4
    for job_id in ids:
        assert svc.store.get(job_id) is not None


# --- storage and pruning ----------------------------------------------------


def test_storage_reports_zero_before_any_job(svc):
    assert svc_jobs.storage(svc) == {"job_dirs": 0, "bytes": 0}


def test_storage_counts_job_directories_and_bytes(svc):
    svc_jobs.create_job(svc, kind="image", image=_png_bytes())
    body = svc_jobs.storage(svc)
    assert body["job_dirs"] == 1
    assert body["bytes"] > 0


def test_prune_keeps_the_newest_and_deletes_their_files(svc, assets):
    ids = [svc_jobs.create_job(svc, kind="text", prompt=f"j{i}")["id"] for i in range(5)]
    # Give the oldest a job dir so the file cleanup is observable.
    oldest_dir = assets / ids[0]
    oldest_dir.mkdir(parents=True, exist_ok=True)
    (oldest_dir / "model.glb").write_bytes(b"x" * 100)

    assert svc_jobs.prune_jobs(svc, keep=2)["deleted"] == 3
    assert {j["id"] for j in svc_jobs.list_jobs(svc)} == set(ids[-2:])
    assert not oldest_dir.exists()


def test_prune_refuses_to_delete_a_running_job(svc):
    ids = [svc_jobs.create_job(svc, kind="text", prompt=f"j{i}")["id"] for i in range(3)]
    svc.store.set_status(ids[0], "running")
    assert svc_jobs.prune_jobs(svc, keep=0)["deleted"] == 2
    assert svc.store.get(ids[0]) is not None


def test_prune_walks_past_a_single_page(svc, monkeypatch):
    """A history longer than one page used to be un-prunable past its first
    MAX_LIST_LIMIT rows -- which is exactly the history that needs pruning."""
    monkeypatch.setattr(svc_jobs, "MAX_LIST_LIMIT", 10)
    ids = [svc_jobs.create_job(svc, kind="text", prompt=f"j{i}")["id"] for i in range(25)]
    svc.store.set_status(ids[0], "running")

    # 25 jobs, keep the newest 3, and the one running job survives too.
    assert svc_jobs.prune_jobs(svc, keep=3)["deleted"] == 21
    surviving = {j["id"] for j in svc.store.list(100)}
    assert surviving == {ids[0], *ids[-3:]}


def test_prune_rejects_a_negative_keep(svc):
    with pytest.raises(Invalid):
        svc_jobs.prune_jobs(svc, keep=-1)


def test_the_list_limit_is_honoured_and_capped(svc):
    for i in range(4):
        svc_jobs.create_job(svc, kind="text", prompt=f"j{i}")
    assert len(svc_jobs.list_jobs(svc, 2)) == 2
    # Out-of-range values clamp rather than erroring or reaching the DB raw.
    assert len(svc_jobs.list_jobs(svc, 0)) == 1
    assert len(svc_jobs.list_jobs(svc, 999999)) == 4


# --- optimize ---------------------------------------------------------------


def test_optimize_reports_the_rig_artifacts_it_made_stale(svc, assets):
    """Re-optimizing swaps model.glb for a different mesh; rig.glb and
    everything baked from it still describe the old one. It must say so rather
    than silently leaving a rig that no longer matches."""
    trimesh = pytest.importorskip("trimesh")

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=1).export(job_dir / "source.glb")
    (job_dir / "rig.glb").write_bytes(b"rig")
    (job_dir / "rig.json").write_text("{}")
    (job_dir / "poses").mkdir()
    (job_dir / "poses" / "0123456789ab.glb").write_bytes(b"baked")

    stale = svc_jobs.optimize_job(svc, job_id, profile="raw")["stale"]
    assert "rig.glb" in stale
    assert "rig.json" in stale
    assert "poses/0123456789ab.glb" in stale


def test_optimize_reports_nothing_stale_for_an_unrigged_job(svc, assets):
    trimesh = pytest.importorskip("trimesh")

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=1).export(job_dir / "source.glb")

    assert svc_jobs.optimize_job(svc, job_id, profile="raw")["stale"] == []


def test_optimize_requires_a_source(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid):
        svc_jobs.optimize_job(svc, job_id)


@pytest.mark.parametrize("status", ["queued", "running"])
def test_optimize_refuses_a_job_the_worker_may_still_be_writing(svc, assets, status):
    """It rewrites model.glb, and the worker's own optimize/scale/audit steps
    write the same file and the same params blob. Nothing can serialise the two
    -- the worker half takes no lock -- so a job that hasn't reached a terminal
    status is refused rather than raced."""
    trimesh = pytest.importorskip("trimesh")

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, status)
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=1).export(job_dir / "source.glb")

    with pytest.raises(Conflict):
        svc_jobs.optimize_job(svc, job_id, profile="raw")


def test_optimize_merges_params_instead_of_writing_a_stale_copy(svc, assets):
    """params is one JSON blob. This used to read the job at the top and write
    that whole copy back at the end, so anything committed in between -- a
    rename, a tag, another derived value -- was silently discarded."""
    trimesh = pytest.importorskip("trimesh")

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=1).export(job_dir / "source.glb")
    svc.store.merge_params(job_id, {"written_meanwhile": "keep me"})

    svc_jobs.optimize_job(svc, job_id, profile="raw")
    params = _params(svc, job_id)
    assert params["written_meanwhile"] == "keep me"
    assert params["profile"] == "raw"


def test_optimize_rejects_an_unknown_profile(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid):
        svc_jobs.optimize_job(svc, job_id, profile="nonsense")


def test_a_submit_rejects_an_unknown_profile(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", profile="nonsense")


def test_a_submit_stores_a_valid_profile(svc, tmp_path):
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    svc.config.gltfpack_exe = exe
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", profile="draft")["id"]
    assert _params(svc, job_id)["profile"] == "draft"


def test_a_named_tier_is_refused_while_gltfpack_is_absent(svc):
    # Without the binary the worker falls back to the raw copy silently, so a
    # job finishing "done" wearing profile="standard" would file every verdict
    # against a tier that never ran. Refusing at the door is the fix.
    with pytest.raises(Invalid, match="gltfpack"):
        svc_jobs.create_job(svc, kind="text", prompt="x", profile="standard")


def test_the_raw_profile_needs_no_binary(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", profile="raw")["id"]
    assert _params(svc, job_id)["profile"] == "raw"


# --- derived artifacts ------------------------------------------------------


def test_concurrent_stl_requests_convert_only_once(svc, assets, monkeypatch):
    """A double-clicked download used to start two exports over the same
    destination path, either racing each other or serving a torn file."""
    import warlock.pipelines.postprocess as pp

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh = pytest.importorskip("trimesh")
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(job_dir / "model.glb")
    svc.store.set_status(job_id, "done")

    calls = []
    real = pp.glb_to_stl

    def counting(glb, stl):
        calls.append(glb)
        time.sleep(0.05)  # widen the window a second caller could enter
        return real(glb, stl)

    monkeypatch.setattr(pp, "glb_to_stl", counting)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = [
            f.result()
            for f in [
                pool.submit(svc_derive.get_file, svc, job_id, "model.stl") for _ in range(4)
            ]
        ]

    assert all(p.exists() and p.stat().st_size for p in paths)
    assert len(calls) == 1, "the second caller should get the finished file"


def test_collision_glb_is_derived_on_demand(svc, assets):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")  # trimesh delegates the convex hull to QHull

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=2).export(job_dir / "model.glb")
    svc.store.set_status(job_id, "done")

    assert not (job_dir / "collision.glb").exists()
    path = svc_derive.get_file(svc, job_id, "collision.glb")
    assert path.stat().st_size
    # Cached beside the mesh, like the STL and OBJ exports.
    assert (job_dir / "collision.glb").exists()


def test_collision_glb_is_unavailable_without_a_mesh(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "collision.glb")


def test_fbx_is_derived_on_demand_through_the_blender_worker(svc, assets):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("bpy")

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(job_dir / "model.glb")
    svc.store.set_status(job_id, "done")

    path = svc_derive.get_file(svc, job_id, "model.fbx")
    assert path.read_bytes()[:18] == b"Kaydara FBX Binary"
    assert (job_dir / "model.fbx").exists()


def test_a_failed_fbx_export_leaves_no_partial_file_to_serve(svc, assets, monkeypatch):
    """Blender writing part of an FBX and then dying must not poison the name.

    Existence is this artifact's freshness test -- there is no sidecar and no
    mtime comparison -- so an unstaged export that timed out or was killed at
    shutdown would leave a truncated FBX that every later request serves,
    with no way for a retry to get past it.
    """
    from warlock import rigging

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")
    svc.store.set_status(job_id, "done")

    def run_worker(spec, **kwargs):
        Path(spec["out_fbx"]).write_bytes(b"Kaydara FB")
        raise rigging.BlenderError("killed")

    monkeypatch.setattr(rigging, "run_worker", run_worker)
    with pytest.raises(Failed):
        svc_derive.get_file(svc, job_id, "model.fbx")

    assert not (job_dir / "model.fbx").exists()
    assert [p.name for p in job_dir.iterdir() if p.name.startswith(".model.fbx")] == []


def test_fbx_is_unavailable_without_a_mesh(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "model.fbx")


def test_textures_zip_is_unavailable_on_an_untextured_mesh(svc, assets):
    # A mesh with no maps is a fact about the model, not a fault.
    trimesh = pytest.importorskip("trimesh")

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(job_dir / "model.glb")
    svc.store.set_status(job_id, "done")

    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "textures.zip")
    # No empty zip left behind to be served as a success next time.
    assert not (job_dir / "textures.zip").exists()


def test_textures_zip_is_derived_on_demand(svc, assets):
    import zipfile

    np = pytest.importorskip("numpy")
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (0, 255, 0))
        ),
    )
    trimesh.Scene(mesh).export(job_dir / "model.glb")
    svc.store.set_status(job_id, "done")

    path = svc_derive.get_file(svc, job_id, "textures.zip")
    with zipfile.ZipFile(path) as zf:
        assert "base_color.png" in zf.namelist()


def test_an_unknown_artifact_name_is_refused(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(NotFound):
        svc_derive.get_file(svc, job_id, "../secrets")


# --- export -----------------------------------------------------------------


def test_bulk_export_zips_the_selected_jobs(svc, assets, tmp_path):
    import zipfile

    ids = []
    for _ in range(2):
        job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
        job_dir = assets / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "model.glb").write_bytes(b"glb")
        svc.store.set_status(job_id, "done")
        ids.append(job_id)

    dest = tmp_path / "out" / "export.zip"
    out = svc_export.bulk_export(svc, ids, None, dest)
    assert out["files"] == 2
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
    assert len(names) == 2
    assert all(n.endswith("model.glb") for n in names)


def test_bulk_export_skips_a_job_that_is_still_running(svc, assets, tmp_path):
    """Export used to include any model.glb that existed -- including one the
    worker was still writing (queue._apply_scale rewrites it in place)."""
    running = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    done = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    for job_id in (running, done):
        job_dir = assets / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "model.glb").write_bytes(b"glb")
    svc.store.set_status(running, "running")
    svc.store.set_status(done, "done")

    dest = tmp_path / "export.zip"
    out = svc_export.bulk_export(svc, [running, done], None, dest)
    assert out["files"] == 1
    with zipfile.ZipFile(dest) as zf:
        assert zf.namelist() == [f"{done}/model.glb"]


def test_export_skips_a_directory_with_no_job_row(svc, assets, tmp_path):
    """An orphaned assets/ directory is not an exportable job."""
    orphan = "a" * 12
    (assets / orphan).mkdir(parents=True, exist_ok=True)
    (assets / orphan / "model.glb").write_bytes(b"glb")
    with pytest.raises(NotFound):
        svc_export.bulk_export(svc, [orphan], None, tmp_path / "o.zip")


def test_a_file_is_not_served_before_the_job_is_done(svc, assets):
    """The file route used to serve on existence alone, disagreeing with the
    listing that gated the same artifact on status."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")
    (job_dir / "source.glb").write_bytes(b"glb")
    svc.store.set_status(job_id, "running")

    for name in ("model.glb", "source.glb"):
        with pytest.raises(NotReady):
            svc_derive.get_file(svc, job_id, name)

    svc.store.set_status(job_id, "done")
    assert svc_derive.get_file(svc, job_id, "model.glb").exists()
    assert svc_derive.get_file(svc, job_id, "source.glb").exists()


def test_a_file_from_a_job_that_does_not_exist_is_not_found(svc, assets):
    """An orphaned directory used to serve files for a deleted job."""
    orphan = "b" * 12
    (assets / orphan).mkdir(parents=True, exist_ok=True)
    (assets / orphan / "model.glb").write_bytes(b"glb")
    with pytest.raises(NotFound):
        svc_derive.get_file(svc, orphan, "model.glb")


def test_bulk_export_rejects_an_unknown_file_name(svc, tmp_path):
    with pytest.raises(Invalid):
        svc_export.bulk_export(svc, ["0" * 12], ["../secrets"], tmp_path / "o.zip")


def test_bulk_export_is_refused_when_nothing_exists(svc, tmp_path):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(NotFound):
        svc_export.bulk_export(svc, [job_id], None, tmp_path / "o.zip")


def test_export_to_folder_is_refused_when_unconfigured(svc):
    with pytest.raises(NotFound):
        svc_export.export_to_folder(svc, ["0" * 12], None)


def test_export_to_folder_copies_into_the_configured_dir(tmp_path, monkeypatch):
    # Its own service: export_dir is read once at Config construction.
    from warlock.config import get_config
    from warlock.db import JobStore
    from warlock.service import WarlockService

    target = tmp_path / "godot_project" / "assets"
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("WARLOCK_DB", str(tmp_path / "assets" / "jobs.sqlite"))
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    monkeypatch.setenv("WARLOCK_EXPORT_DIR", str(target))
    monkeypatch.setattr(config_mod, "_config", None)

    config = get_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(config.db_path)
    try:
        svc = WarlockService(config, store)
        job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
        job_dir = config.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "model.glb").write_bytes(b"glb")
        svc.store.set_status(job_id, "done")

        out = svc_export.export_to_folder(svc, [job_id], None)
        assert out["copied"] == 1
        assert (target / job_id / "model.glb").read_bytes() == b"glb"
    finally:
        store.close()


def test_health_reports_the_export_dir(svc, worker):
    # The UI reveals its "save to project" action off this field alone.
    assert svc_system.health(svc)["export_dir"] is None


# --- metadata ---------------------------------------------------------------


def test_updating_job_metadata(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    body = svc_jobs.update_job(
        svc, job_id, {"name": "Oak barrel", "tags": ["Prop", " Fantasy "], "favorite": True}
    )
    assert body["name"] == "Oak barrel"
    assert body["tags"] == "fantasy,prop"  # normalized and sorted
    assert body["favorite"] == 1


def test_updating_rejects_an_overlong_name(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid):
        svc_jobs.update_job(svc, job_id, {"name": "x" * 200})


def test_updating_a_missing_job_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_jobs.update_job(svc, "0" * 12, {"name": "x"})


# --- thumbnails -------------------------------------------------------------


def test_a_thumbnail_is_saved_and_then_listed(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    assert svc_files.save_thumbnail(svc, job_id, _png_bytes()) == {"ok": True}
    assert svc_derive.get_file(svc, job_id, "thumb.png").exists()
    assert "thumb.png" in _job(svc, job_id)["files"]


def test_a_thumbnail_that_is_not_a_png_is_refused(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid):
        svc_files.save_thumbnail(svc, job_id, b"not a png")


def test_an_oversized_thumbnail_is_refused(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(TooLarge):
        svc_files.save_thumbnail(svc, job_id, b"\x89PNG\r\n\x1a\n" + b"0" * 600_000)


# --- logs -------------------------------------------------------------------


def test_a_failed_jobs_traceback_is_retrievable(svc, assets):
    """The DB only ever holds the one-line friendly sentence. The traceback has
    always been written to the job dir and was unreachable from the UI, which
    made every failure a one-line dead end."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "error.log").write_text("Traceback (most recent call last):\n  boom\n")
    svc.store.set_status(job_id, "error", "it broke")

    assert "error.log" in _job(svc, job_id)["files"]
    assert "boom" in svc_derive.get_file(svc, job_id, "error.log").read_text()


def test_the_trellis_log_tail_is_readable(svc, assets):
    with pytest.raises(NotFound):
        svc_system.trellis_log(svc)
    (assets / "trellis.log").write_text("stage 1\nstage 2\n")
    assert "stage 2" in svc_system.trellis_log(svc)["text"]


def test_a_service_error_carries_the_status_its_route_used_to_return():
    """The hierarchy is small so this mapping stays mechanical: an extraction
    that changed a status code should be a visible edit, not a silent one."""
    assert NotFound("x").status == 404
    assert NotReady("x").status == 404
    assert Invalid("x").status == 400
    assert Conflict("x").status == 409
    assert TooLarge("x").status == 413
    assert Failed("x").status == 500

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import warlock.config as config_mod
from warlock import models
from warlock.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("WARLOCK_DB", str(tmp_path / "assets" / "jobs.sqlite"))
    # Point at a nonexistent exe; the worker only touches it when a job runs.
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(config_mod, "_config", None)
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_create_text_job(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"})
    assert r.status_code == 200
    job_id = r.json()["id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["kind"] == "text"
    assert job["prompt"] == "a barrel"


def test_text_job_requires_prompt(client):
    r = client.post("/api/jobs", data={"kind": "text"})
    assert r.status_code == 400


def test_image_job_requires_upload(client):
    r = client.post("/api/jobs", data={"kind": "image"})
    assert r.status_code == 400


def _png_bytes(fmt: str = "PNG") -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (200, 50, 50, 255)).save(buf, fmt)
    return buf.getvalue()


def test_create_image_job_stores_upload(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 200
    job = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert "input.png" in job["files"]


def test_webp_upload_is_normalized_to_png(client, tmp_path):
    # trellis.cpp only decodes PNG/JPEG; any upload (webp, bmp, ...) must be
    # stored as a real PNG regardless of its original format.
    from PIL import Image

    r = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.webp", io.BytesIO(_png_bytes("WEBP")), "image/webp")},
    )
    assert r.status_code == 200
    stored = tmp_path / "assets" / r.json()["id"] / "input.png"
    assert Image.open(stored).format == "PNG"


def test_invalid_image_upload_rejected(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.png", io.BytesIO(b"not an image"), "image/png")},
    )
    assert r.status_code == 400


def test_invalid_resolution_rejected(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "x", "resolution": "999"})
    assert r.status_code == 400


def test_guidance_catalog_is_served(client):
    body = client.get("/api/guidance").json()
    assert set(body["fields"]) == {
        "genre", "art_style", "category", "platform", "base_model", "style_lora",
    }
    assert body["defaults"]["platform"] in {o["key"] for o in body["fields"]["platform"]}
    assert body["defaults"]["base_model"] in {o["key"] for o in body["fields"]["base_model"]}


def test_guidance_is_stored_on_the_job(client):
    r = client.post(
        "/api/jobs",
        data={
            "kind": "text",
            "prompt": "a plasma rifle",
            "genre": "scifi",
            "art_style": "lowpoly",
            "category": "weapon",
            "platform": "mobile",
        },
    )
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["genre"] == "scifi"
    assert params["art_style"] == "lowpoly"
    assert params["category"] == "weapon"
    assert params["resolution"] == 512  # derived from the mobile platform preset
    assert params["size_m"] == 1.0  # derived from the weapon category default


def test_explicit_size_overrides_the_category_default(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "x", "category": "weapon", "size_m": "0.25"},
    )
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["size_m"] == 0.25


def test_job_without_guidance_still_gets_usable_defaults(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"})
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["resolution"] in (512, 1024, 1536)
    assert params["size_m"] > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("genre", "nonsense"), ("category", "nonsense"), ("platform", "nonsense"), ("size_m", "500")],
)
def test_invalid_guidance_rejected(client, field, value):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "x", field: value})
    assert r.status_code == 400


def test_rejected_guidance_leaves_no_input_png_behind(client, tmp_path):
    r = client.post(
        "/api/jobs",
        data={"kind": "image", "genre": "nonsense"},
        files={"image": ("ref.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 400
    assert not list((tmp_path / "assets").glob("*/input.png"))


def test_cancel_and_delete(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_cancel_running_job_calls_request_cancel(client, monkeypatch):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    worker = client.app.state.worker
    worker.store.set_status(job_id, "running")
    worker.current_job_id = job_id
    called = []

    async def fake_request_cancel(jid):
        called.append(jid)

    monkeypatch.setattr(worker, "request_cancel", fake_request_cancel)
    try:
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200
        assert called == [job_id]
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    finally:
        worker.current_job_id = None


def test_cancel_is_idempotent_when_already_cancelled(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    worker = client.app.state.worker
    worker.store.set_status(job_id, "running")
    worker.store.cancel(job_id)  # simulate: already cancelled by the time this request lands
    worker.current_job_id = None  # not the currently-running job from the route's perspective
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200


def test_input_png_written_before_db_row_is_created(client, tmp_path, monkeypatch):
    from warlock.db import JobStore

    original_create = JobStore.create
    seen = {}

    def spying_create(self, kind, prompt, params, job_id=None, **kwargs):
        if job_id is not None:
            path = tmp_path / "assets" / job_id / "input.png"
            seen["exists"] = path.exists()
        return original_create(self, kind, prompt, params, job_id=job_id, **kwargs)

    monkeypatch.setattr(JobStore, "create", spying_create)

    r = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 200
    assert seen.get("exists") is True


def test_opaque_jpeg_upload_preserves_rgb(client, tmp_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "JPEG")
    r = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")},
    )
    assert r.status_code == 200
    stored = tmp_path / "assets" / r.json()["id"] / "input.png"
    assert Image.open(stored).mode == "RGB"


def test_health_reports_worker_alive_and_doctor_checks(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["worker_alive"] is True
    assert body["fatal"] is None
    assert isinstance(body["checks"], list)
    assert len(body["checks"]) == 7 + len(models.BASE_MODELS) + len(models.STYLE_LORAS)


# --- progress ---


def test_progress_is_empty_when_idle(client):
    r = client.get("/api/progress")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] is None
    assert body["progress"] is None
    assert body["server_time"] > 0


def test_progress_reports_the_running_job(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    worker = client.app.state.worker
    worker.current_job_id = job_id
    worker.progress.begin(job_id, "text")
    worker.progress.update(
        job_id, phase="trellis", label="Mesh decode", inner=0.5, detail="building mesh"
    )
    try:
        body = client.get("/api/progress").json()
        assert body["job_id"] == job_id
        assert body["progress"]["label"] == "Mesh decode"
        assert body["progress"]["percent"] > 0
    finally:
        worker.current_job_id = None
        worker.progress.end(job_id)


def test_job_list_carries_progress_only_for_the_running_job(client):
    a = client.post("/api/jobs", data={"kind": "text", "prompt": "a"}).json()["id"]
    b = client.post("/api/jobs", data={"kind": "text", "prompt": "b"}).json()["id"]
    worker = client.app.state.worker
    worker.progress.begin(a, "text")
    worker.progress.update(a, phase="trellis", label="Texture", inner=0.3)
    try:
        jobs = {j["id"]: j for j in client.get("/api/jobs").json()}
        assert jobs[a]["progress"]["label"] == "Texture"
        assert jobs[b]["progress"] is None
        assert client.get(f"/api/jobs/{b}").json()["progress"] is None
    finally:
        worker.progress.end(a)


def test_progress_clears_when_the_job_ends(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    worker = client.app.state.worker
    worker.progress.begin(job_id, "text")
    worker.progress.end(job_id)
    assert client.get(f"/api/jobs/{job_id}").json()["progress"] is None


# --- file attachment ---


def test_model_glb_hidden_until_job_is_done(tmp_path):
    from warlock.app import _attach_files

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "model.glb").write_bytes(b"placeholder")

    running = {"status": "running"}
    _attach_files(running, job_dir)
    assert "model.glb" not in running["files"]

    done = {"status": "done"}
    _attach_files(done, job_dir)
    assert "model.glb" in done["files"]


def test_input_png_stays_existence_based_regardless_of_status(tmp_path):
    from warlock.app import _attach_files

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "input.png").write_bytes(b"placeholder")

    job = {"status": "running"}
    _attach_files(job, job_dir)
    assert "input.png" in job["files"]


# --- job id validation ---
#
# config.job_dir() is a bare data_dir / job_id join. Only the file route could
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
        "%2e%2e",
        "0123456789ab.",
    ],
)
def test_malformed_job_id_is_rejected_on_every_route(client, bad):
    """Ids that reach the handler must 404 before any path is built from them."""
    assert client.get(f"/api/jobs/{bad}").status_code == 404
    assert client.get(f"/api/jobs/{bad}/files/model.glb").status_code == 404
    assert client.post(f"/api/jobs/{bad}/cancel").status_code == 404
    assert client.post(f"/api/jobs/{bad}/rerun", data={"mode": "reroll"}).status_code == 404
    assert client.delete(f"/api/jobs/{bad}").status_code == 404


@pytest.mark.parametrize("bad", ["..", "../../etc", "..%2f..%2fwindows", ""])
def test_traversal_shaped_job_ids_never_succeed(client, bad):
    """These are absorbed by URL normalisation or routing before the handler
    sees them (an empty segment can't even match the route). Asserting a
    specific status would be testing Starlette; what matters is that none of
    them reads or writes anything."""
    for response in (
        client.get(f"/api/jobs/{bad}"),
        client.get(f"/api/jobs/{bad}/files/model.glb"),
        client.post(f"/api/jobs/{bad}/cancel"),
        client.delete(f"/api/jobs/{bad}"),
    ):
        assert not response.is_success


# --- rerun ---


def test_reroll_copies_the_prompt_and_guidance_with_a_new_seed(client):
    src = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "a barrel", "seed": 42, "genre": "fantasy"},
    ).json()["id"]
    # Pretend the worker finished it and recorded its derived values.
    store = client.app.state.store
    params = store.get(src)["params"]
    params["composed_prompt"] = "a barrel, fantasy"
    params["scale_factor"] = 2.0
    params["mesh_audit"] = {"worst": 0.3, "mean": 0.2, "faces": 9, "resolution": 512}
    store.set_params(src, params)

    r = client.post(f"/api/jobs/{src}/rerun", data={"mode": "reroll"})
    assert r.status_code == 200
    new = client.get(f"/api/jobs/{r.json()['id']}").json()

    assert new["kind"] == "text"
    assert new["prompt"] == "a barrel"
    assert new["params"]["genre"] == "fantasy"
    assert new["params"]["seed"] != 42
    assert new["params"]["rerun_of"] == src
    # Derived values describe the source run, not this one.
    for key in ("composed_prompt", "scale_factor", "mesh_audit"):
        assert key not in new["params"]


def test_reroll_accepts_an_explicit_seed(client):
    src = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"}).json()["id"]
    r = client.post(f"/api/jobs/{src}/rerun", data={"mode": "reroll", "seed": 7})
    assert r.json()["seed"] == 7
    assert client.get(f"/api/jobs/{r.json()['id']}").json()["params"]["seed"] == 7


def test_remesh_reuses_the_reference_image_as_an_image_job(client, tmp_path):
    """The point of remesh: when SDXL drew a good reference but trellis made a
    poor mesh, retry the 3D stage without paying for SDXL again."""
    src = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"}).json()["id"]
    # Stand in for the SDXL output the worker would have written.
    src_dir = tmp_path / "assets" / src
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "input.png").write_bytes(_png_bytes())

    r = client.post(f"/api/jobs/{src}/rerun", data={"mode": "remesh"})
    assert r.status_code == 200
    new_id = r.json()["id"]
    new = client.get(f"/api/jobs/{new_id}").json()

    assert new["kind"] == "image"
    assert (tmp_path / "assets" / new_id / "input.png").read_bytes() == _png_bytes()
    # The prompt rides along for display even though it won't be used again.
    assert new["prompt"] == "a barrel"


def test_remesh_without_a_reference_image_is_rejected(client):
    src = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"}).json()["id"]
    r = client.post(f"/api/jobs/{src}/rerun", data={"mode": "remesh"})
    assert r.status_code == 400


def test_reroll_of_an_image_job_carries_the_upload_across(client, tmp_path):
    src = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.png", io.BytesIO(_png_bytes()), "image/png")},
    ).json()["id"]
    new_id = client.post(f"/api/jobs/{src}/rerun", data={"mode": "reroll"}).json()["id"]
    assert (tmp_path / "assets" / new_id / "input.png").exists()


def test_rerun_rejects_an_unknown_mode(client):
    src = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    assert client.post(f"/api/jobs/{src}/rerun", data={"mode": "sideways"}).status_code == 400


def test_rerun_of_a_missing_job_is_404(client):
    assert client.post("/api/jobs/0123456789ab/rerun", data={"mode": "reroll"}).status_code == 404


def test_remesh_inherits_legacy_seed_as_reference_and_rerolls_only_mesh(client, tmp_path):
    # Source job predates the split, so it only has "seed" -- no reference_seed
    # / mesh_seed keys at all. remesh must still treat that lone seed as the
    # reference to inherit, and must not also reuse it for the mesh stage.
    src = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a barrel", "seed": 42}
    ).json()["id"]
    src_dir = tmp_path / "assets" / src
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "input.png").write_bytes(_png_bytes())

    r = client.post(f"/api/jobs/{src}/rerun", data={"mode": "remesh"})
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["reference_seed"] == 42
    assert params["mesh_seed"] != 42


def test_remesh_inherits_explicit_reference_seed_not_legacy_seed(client, tmp_path):
    # Source job already went through the split and has both keys, with
    # different values. remesh must inherit reference_seed specifically --
    # if it fell back to plain "seed" instead, this would still pass with
    # matching fixture values, so the two are deliberately made to differ.
    src = client.post(
        "/api/jobs",
        data={
            "kind": "text",
            "prompt": "a barrel",
            "seed": 42,
            "reference_seed": 99,
            "mesh_seed": 42,
        },
    ).json()["id"]
    src_dir = tmp_path / "assets" / src
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "input.png").write_bytes(_png_bytes())

    r = client.post(f"/api/jobs/{src}/rerun", data={"mode": "remesh"})
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["reference_seed"] == 99
    assert params["mesh_seed"] != 42
    assert params["mesh_seed"] != 99


def test_reroll_gives_both_seeds_the_same_fresh_value(client):
    src = client.post(
        "/api/jobs",
        data={
            "kind": "text",
            "prompt": "a barrel",
            "seed": 42,
            "reference_seed": 11,
            "mesh_seed": 22,
        },
    ).json()["id"]

    r = client.post(f"/api/jobs/{src}/rerun", data={"mode": "reroll"})
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["reference_seed"] == params["mesh_seed"]
    assert params["reference_seed"] != 11
    assert params["mesh_seed"] != 22


# --- on-demand conversion ---


def test_concurrent_stl_requests_convert_only_once(client, tmp_path, monkeypatch):
    """A double-clicked download link used to start two exports over the same
    destination path, either racing each other or serving a torn file."""
    import warlock.pipelines.postprocess as pp

    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    job_dir = tmp_path / "assets" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    trimesh = pytest.importorskip("trimesh")
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(job_dir / "model.glb")

    calls = []
    real = pp.glb_to_stl

    def counting(glb, stl):
        calls.append(glb)
        time.sleep(0.05)      # widen the window a second request could enter
        return real(glb, stl)

    monkeypatch.setattr(pp, "glb_to_stl", counting)

    url = f"/api/jobs/{job_id}/files/model.stl"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [f.result() for f in [pool.submit(client.get, url) for _ in range(4)]]

    assert all(r.status_code == 200 for r in results)
    assert all(r.content for r in results)
    assert len(calls) == 1, "the second request should serve the finished file"


# --- storage and pruning ---


def test_storage_reports_zero_before_any_job(client):
    body = client.get("/api/storage").json()
    assert body == {"job_dirs": 0, "bytes": 0}


def test_storage_counts_job_directories_and_bytes(client, tmp_path):
    client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    body = client.get("/api/storage").json()
    assert body["job_dirs"] == 1
    assert body["bytes"] > 0


def test_prune_keeps_the_newest_and_deletes_their_files(client, tmp_path):
    ids = [
        client.post("/api/jobs", data={"kind": "text", "prompt": f"j{i}"}).json()["id"]
        for i in range(5)
    ]
    # Give the oldest a job dir so the file cleanup is observable.
    oldest_dir = tmp_path / "assets" / ids[0]
    oldest_dir.mkdir(parents=True, exist_ok=True)
    (oldest_dir / "model.glb").write_bytes(b"x" * 100)

    r = client.post("/api/jobs/prune", data={"keep": 2})
    assert r.status_code == 200
    assert r.json()["deleted"] == 3

    remaining = {j["id"] for j in client.get("/api/jobs").json()}
    assert remaining == set(ids[-2:])
    assert not oldest_dir.exists()


def test_prune_refuses_to_delete_a_running_job(client):
    ids = [
        client.post("/api/jobs", data={"kind": "text", "prompt": f"j{i}"}).json()["id"]
        for i in range(3)
    ]
    client.app.state.store.set_status(ids[0], "running")

    assert client.post("/api/jobs/prune", data={"keep": 0}).json()["deleted"] == 2
    assert client.get(f"/api/jobs/{ids[0]}").status_code == 200


def test_prune_rejects_a_negative_keep(client):
    assert client.post("/api/jobs/prune", data={"keep": -1}).status_code == 400


def test_job_list_limit_is_honoured_and_capped(client):
    for i in range(4):
        client.post("/api/jobs", data={"kind": "text", "prompt": f"j{i}"})
    assert len(client.get("/api/jobs?limit=2").json()) == 2
    # Out-of-range values clamp rather than erroring or reaching the DB raw.
    assert len(client.get("/api/jobs?limit=0").json()) == 1
    assert len(client.get("/api/jobs?limit=999999").json()) == 4


def test_model_selection_is_stored_on_the_job(client):
    r = client.post(
        "/api/jobs",
        data={
            "kind": "text",
            "prompt": "a plasma rifle",
            "base_model": "sdxl",
            "style_lora": "render3d",
            "lora_weight": "1.1",
        },
    )
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["base_model"] == "sdxl"
    assert params["style_lora"] == "render3d"
    assert params["lora_weight"] == 1.1


def test_unknown_base_model_is_a_400(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "x", "base_model": "dall-e"})
    assert r.status_code == 400
    assert "base_model" in r.json()["detail"]


def test_out_of_range_lora_weight_is_a_400(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "x", "style_lora": "ps1", "lora_weight": "9"},
    )
    assert r.status_code == 400


def test_rerun_carries_model_selection_forward(client):
    """base_model/style_lora are inputs, not derived from the source run, so
    unlike composed_prompt they must survive the derived-key strip."""
    first = client.post(
        "/api/jobs",
        data={
            "kind": "text",
            "prompt": "a barrel",
            "base_model": "sdxl",
            "style_lora": "ps1",
            "lora_weight": "0.6",
        },
    ).json()["id"]
    r = client.post(f"/api/jobs/{first}/rerun")
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["base_model"] == "sdxl"
    assert params["style_lora"] == "ps1"
    assert params["lora_weight"] == 0.6


def test_seeds_split_and_legacy_seed_still_read(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "a barrel", "reference_seed": 11, "mesh_seed": 22},
    )
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["reference_seed"] == 11
    assert params["mesh_seed"] == 22

    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel", "seed": 5})
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["reference_seed"] == 5
    assert params["mesh_seed"] == 5


def test_reference_output_stops_before_the_mesh(client):
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a barrel", "output": "reference"}
    )
    ref_id = r.json()["id"]
    assert client.get(f"/api/jobs/{ref_id}").json()["stage"] == "reference"


def test_default_output_is_still_model(client):
    # Backward compatibility: every existing client omits `output` and must
    # keep getting the old one-job behaviour.
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"})
    assert client.get(f"/api/jobs/{r.json()['id']}").json()["stage"] == "model"


def test_image_job_cannot_request_a_reference_output(client, tmp_path):
    r = client.post(
        "/api/jobs",
        data={"kind": "image", "output": "reference"},
        files={"image": ("in.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 400


def test_promote_creates_a_child_model_job(client, tmp_path):
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a barrel", "output": "reference"}
    )
    ref_id = r.json()["id"]
    # The worker is not running a real pipeline in this fixture; stand the
    # reference in by hand, which is exactly the state promotion requires.
    import warlock.config as config_mod

    reference_bytes = b"png-distinctive-parent-bytes-\x00\x01\x02"
    job_dir = config_mod.get_config().job_dir(ref_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(reference_bytes)
    # The fixture never runs a real pipeline, so nothing marks the row
    # "done" on its own -- stand that in too, exactly like input.png above.
    client.app.state.store.set_status(ref_id, "done")

    r = client.post(f"/api/jobs/{ref_id}/model")
    assert r.status_code == 200
    child = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert child["parent_id"] == ref_id
    assert child["kind"] == "image"
    assert child["stage"] == "model"
    # The load-bearing half of promote: the child's input.png must actually
    # be the parent's bytes, not just exist as an empty/wrong file. A worker
    # picking this job up minutes later fails without it.
    child_png = config_mod.get_config().job_dir(child["id"]) / "input.png"
    assert child_png.exists()
    assert child_png.read_bytes() == reference_bytes


def test_promote_reuses_the_parents_reference_seed(client, tmp_path):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "a barrel", "output": "reference", "reference_seed": 11},
    )
    ref_id = r.json()["id"]
    import warlock.config as config_mod

    job_dir = config_mod.get_config().job_dir(ref_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"png")
    client.app.state.store.set_status(ref_id, "done")

    r = client.post(f"/api/jobs/{ref_id}/model")
    child = client.get(f"/api/jobs/{r.json()['id']}").json()
    # The reference is reused verbatim -- promotion must not redraw it.
    assert child["params"]["reference_seed"] == 11
    assert child["params"]["mesh_seed"] == r.json()["mesh_seed"]


def test_promote_without_a_reference_is_a_400(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"})
    assert client.post(f"/api/jobs/{r.json()['id']}/model").status_code == 400


def test_reroll_of_a_reference_stays_a_reference(client):
    # "try another" delegates to reroll -- it must not silently fall through
    # to the default "model" stage and pay for a trellis run.
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a barrel", "output": "reference"}
    )
    ref_id = r.json()["id"]
    r = client.post(f"/api/jobs/{ref_id}/rerun")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{r.json()['id']}").json()["stage"] == "reference"


def test_promote_a_model_stage_job_is_a_400(client):
    # A plain job's stage is already "model" -- there is no reference to
    # promote, and it must not be treated as one.
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"})
    job_id = r.json()["id"]
    client.app.state.store.set_status(job_id, "done")
    assert client.post(f"/api/jobs/{job_id}/model").status_code == 400


def test_promote_a_not_yet_done_reference_is_a_400(client):
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a barrel", "output": "reference"}
    )
    ref_id = r.json()["id"]
    # No input.png written and status left at "queued" -- not done yet.
    assert client.post(f"/api/jobs/{ref_id}/model").status_code == 400


def test_count_creates_n_reference_jobs_with_distinct_seeds(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "a barrel", "output": "reference", "count": 4},
    )
    body = r.json()
    assert len(body["ids"]) == 4
    assert body["id"] == body["ids"][0]
    seeds = {
        client.get(f"/api/jobs/{i}").json()["params"]["reference_seed"]
        for i in body["ids"]
    }
    assert len(seeds) == 4
    # If stage regressed to the default "model", candidates 1..N-1 would each
    # silently pay a full multi-minute trellis run instead of stopping after
    # the reference image.
    for job_id in body["ids"]:
        assert client.get(f"/api/jobs/{job_id}").json()["stage"] == "reference"


def test_count_above_one_requires_reference_output(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "x", "count": 3})
    assert r.status_code == 400


def test_count_is_bounded(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "x", "output": "reference", "count": 99},
    )
    assert r.status_code == 400


def test_candidate_zero_keeps_the_requested_seed(client):
    # A pinned seed must still land on candidate 0 verbatim -- only 1..N-1 fan
    # out to fresh random seeds. Use a value nowhere near _random_seed()'s
    # 31-bit range collision odds, so an accidental match is not plausible.
    r = client.post(
        "/api/jobs",
        data={
            "kind": "text",
            "prompt": "a barrel",
            "output": "reference",
            "count": 3,
            "seed": 424242,
        },
    )
    body = r.json()
    first = client.get(f"/api/jobs/{body['ids'][0]}").json()
    assert first["params"]["reference_seed"] == 424242
    assert first["params"]["seed"] == 424242


def test_count_one_text_job_behaves_exactly_as_before(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel", "count": 1})
    body = r.json()
    assert body["ids"] == [body["id"]]
    job = client.get(f"/api/jobs/{body['id']}").json()
    assert job["kind"] == "text"
    assert job["prompt"] == "a barrel"


def test_count_one_image_job_still_writes_input_png(client, tmp_path):
    r = client.post(
        "/api/jobs",
        data={"kind": "image", "count": 1},
        files={"image": ("ref.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    body = r.json()
    assert body["ids"] == [body["id"]]
    stored = tmp_path / "assets" / body["id"] / "input.png"
    assert stored.exists()


def test_count_ids_are_distinct_and_all_exist(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "a barrel", "output": "reference", "count": 4},
    )
    ids = r.json()["ids"]
    assert len(set(ids)) == 4
    for job_id in ids:
        assert client.get(f"/api/jobs/{job_id}").status_code == 200

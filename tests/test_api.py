from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

import animancer3d.config as config_mod
from animancer3d.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMANCER3D_DATA_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("ANIMANCER3D_DB", str(tmp_path / "assets" / "jobs.sqlite"))
    # Point at a nonexistent exe; the worker only touches it when a job runs.
    monkeypatch.setenv("ANIMANCER3D_TRELLIS_EXE", str(tmp_path / "missing.exe"))
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


def test_cancel_and_delete(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code in (200, 409)
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404

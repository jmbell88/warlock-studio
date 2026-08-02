"""Sprite sheets: the pure layout/packing module, the HTTP surface, and the
queue's handling of the job kind.

Blender is faked for all of it except the last section, which is behind the
gpu marker: everything up to there is arithmetic and plumbing, and only the
final test needs bpy to put actual pixels in a frame.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import warlock.config as config_mod
from warlock import rigging
from warlock.app import create_app
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import sheet as sheetlib
from warlock.queue import Worker

IDENTITY = [0.0, 0.0, 0.0, 1.0]


# --- layout -----------------------------------------------------------------


def test_grid_is_poses_down_and_yaws_across():
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}, {"id": "b" * 12, "name": "run"}])
    assert (layout.columns, layout.rows) == (8, 2)
    assert (layout.width, layout.height) == (8 * 128, 2 * 128)
    assert len(layout.cells) == 16


def test_cells_are_indexed_row_major_and_placed_on_the_grid():
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}], frame_size=64)
    for i, cell in enumerate(layout.cells):
        assert cell.index == i
        assert (cell.x, cell.y) == (cell.column * 64, cell.row * 64)
    assert [c.yaw for c in layout.cells] == [0, 45, 90, 135, 180, 225, 270, 315]


def test_no_poses_means_one_rest_row_not_an_empty_sheet():
    """An unrigged prop still wants a turnaround."""
    layout = sheetlib.plan([])
    assert layout.rows == 1
    assert {c.pose for c in layout.cells} == {None}
    assert {c.pose_name for c in layout.cells} == {sheetlib.REST_POSE_NAME}


def test_every_cell_carries_a_frame_number():
    """The seam animated clips arrive on: more cells, same format."""
    layout = sheetlib.plan([])
    assert {c.frame for c in layout.cells} == {0}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_size": 100},
        {"lighting": "toon"},
        {"elevation": 90.0},
        {"elevation": -95.0},
        {"yaws": 0},
    ],
)
def test_unrenderable_requests_are_rejected_at_plan_time(kwargs):
    with pytest.raises(ValueError):
        sheetlib.plan([], **kwargs)


def test_an_atlas_over_the_texture_limit_is_refused():
    poses = [{"id": f"{i:012x}", "name": str(i)} for i in range(20)]
    with pytest.raises(ValueError, match="8192"):
        sheetlib.plan(poses, frame_size=512)


def test_yaw_angles_start_at_the_front_view():
    assert sheetlib.yaw_angles(4) == (0.0, 90.0, 180.0, 270.0)


# --- animation ----------------------------------------------------------------


def test_slerp_endpoints_and_midpoint():
    a = [0.0, 0.0, 0.0, 1.0]
    half = math.radians(90) / 2
    b = [0.0, 0.0, math.sin(half), math.cos(half)]
    assert sheetlib.slerp(a, b, 0.0) == pytest.approx(a)
    assert sheetlib.slerp(a, b, 1.0) == pytest.approx(b)
    quarter = math.radians(45) / 2
    assert sheetlib.slerp(a, b, 0.5) == pytest.approx(
        [0.0, 0.0, math.sin(quarter), math.cos(quarter)], abs=1e-6
    )


def test_slerp_takes_the_short_way_round():
    """Negating a quaternion is the same rotation; without the sign fix the
    interpolation spins the long way and the sprite counter-rotates."""
    a = [0.0, 0.0, 0.0, 1.0]
    b = [0.0, 0.0, 0.0, -1.0]
    mid = sheetlib.slerp(a, b, 0.5)
    assert abs(mid[3]) == pytest.approx(1.0, abs=1e-6)


def test_interpolate_produces_numbered_frames_between_two_poses():
    a = {"id": "a" * 12, "name": "contact A", "bones": {"thigh.L": [0.0, 0.0, 0.0, 1.0]}}
    half = math.radians(60) / 2
    b = {
        "id": "b" * 12,
        "name": "contact B",
        "bones": {"thigh.L": [math.sin(half), 0.0, 0.0, math.cos(half)]},
    }

    frames = sheetlib.interpolate(a, b, 4)

    assert len(frames) == 4
    assert [f["frame"] for f in frames] == [0, 1, 2, 3]
    assert frames[0]["bones"]["thigh.L"] == pytest.approx(a["bones"]["thigh.L"])
    # The last frame stops short of B, so a looping clip does not hold a
    # duplicate frame at the seam.
    assert frames[-1]["bones"]["thigh.L"] != pytest.approx(b["bones"]["thigh.L"])


def test_a_bone_posed_in_only_one_end_interpolates_from_rest():
    a = {"id": "a" * 12, "name": "A", "bones": {}}
    b = {"id": "b" * 12, "name": "B", "bones": {"head": [0.0, 0.0, 0.7071, 0.7071]}}
    frames = sheetlib.interpolate(a, b, 2)
    assert frames[0]["bones"]["head"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_interpolate_rejects_a_clip_length_it_cannot_render():
    a = {"id": "a" * 12, "name": "A", "bones": {}}
    with pytest.raises(ValueError):
        sheetlib.interpolate(a, a, 0)
    with pytest.raises(ValueError):
        sheetlib.interpolate(a, a, sheetlib.MAX_CLIP_FRAMES + 1)


def test_plan_honours_the_yaw_count():
    layout = sheetlib.plan([], yaws=4)
    assert layout.columns == 4
    assert len(layout.cells) == 4


def test_plan_carries_each_records_own_frame_index():
    """A clip's rows differ only by frame; dropping it renders row 0 six times."""
    a = {"id": "a" * 12, "name": "A", "bones": {}}
    b = {"id": "b" * 12, "name": "B", "bones": {}}
    layout = sheetlib.plan(sheetlib.interpolate(a, b, 3), yaws=2)
    assert [c.frame for c in layout.cells] == [0, 0, 1, 1, 2, 2]


# --- packing ----------------------------------------------------------------


def _frames(tmp_path, layout, colours=None):
    """One solid-colour PNG per cell, so a paste can be located by pixel."""
    out = {}
    for cell in layout.cells:
        colour = (colours or {}).get(cell.index, (cell.index * 8 % 256, 64, 128, 255))
        path = tmp_path / f"{cell.index:04d}.png"
        Image.new("RGBA", (layout.frame_size, layout.frame_size), colour).save(path)
        out[cell.index] = path
    return out


def test_pack_places_each_frame_at_its_own_cell(tmp_path):
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}], frame_size=64)
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, _frames(tmp_path, layout), out)
    with Image.open(out) as atlas:
        assert atlas.size == (layout.width, layout.height)
        for cell in layout.cells:
            probe = atlas.getpixel((cell.x + 32, cell.y + 32))
            assert probe == (cell.index * 8 % 256, 64, 128, 255)


def test_pack_resizes_a_frame_that_came_back_the_wrong_size(tmp_path):
    layout = sheetlib.plan([], frame_size=64)
    frames = _frames(tmp_path, layout)
    Image.new("RGBA", (32, 32), (10, 20, 30, 255)).save(frames[0])
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, frames, out)
    with Image.open(out) as atlas:
        assert atlas.size == (layout.width, layout.height)
        assert atlas.getpixel((32, 32)) == (10, 20, 30, 255)


def test_pack_refuses_to_leave_a_hole(tmp_path):
    """A gap in a sheet reads as a modelling bug and sends the user hunting in
    the wrong place."""
    layout = sheetlib.plan([], frame_size=64)
    frames = _frames(tmp_path, layout)
    frames[3].unlink()
    with pytest.raises(ValueError, match="cell 3"):
        sheetlib.pack(layout, frames, tmp_path / "sheet.png")


def test_untouched_cells_stay_transparent(tmp_path):
    layout = sheetlib.plan([], frame_size=64)
    frames = _frames(tmp_path, layout, colours={i: (0, 0, 0, 0) for i in range(8)})
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, frames, out)
    with Image.open(out) as atlas:
        assert atlas.getpixel((10, 10))[3] == 0


# --- sidecar ----------------------------------------------------------------


def test_sidecar_describes_the_grid_and_every_cell():
    layout = sheetlib.plan([{"id": "a" * 12, "name": "idle"}], frame_size=64, lighting="lit")
    meta = sheetlib.sidecar(
        layout, sheet_id="c" * 12, source_job="d" * 12, image="c.png", created=1.0
    )
    assert meta["version"] == sheetlib.SHEET_VERSION
    assert (meta["columns"], meta["rows"]) == (8, 1)
    assert (meta["width"], meta["height"]) == (512, 64)
    assert meta["lighting"] == "lit"
    assert meta["poses"] == [{"id": "a" * 12, "name": "idle"}]
    assert len(meta["cells"]) == 8
    first = meta["cells"][0]
    assert first == {
        "index": 0, "row": 0, "column": 0, "x": 0, "y": 0, "w": 64, "h": 64,
        "pose": "a" * 12, "pose_name": "idle", "yaw": 0.0, "frame": 0,
        # Additive: an importer that ignores these reads the sheet exactly as
        # it did before, which is why the version does not move.
        "pivot_x": 32.0, "pivot_y": 64.0, "trim": None,
    }


def test_sidecar_carries_a_pivot_per_cell():
    layout = sheetlib.plan([], frame_size=128, yaws=4)
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png",
        created=1.0, pivot=(64.0, 118.0),
    )
    assert all(c["pivot_x"] == 64.0 and c["pivot_y"] == 118.0 for c in meta["cells"])


def test_pivot_defaults_to_the_cell_centre_bottom_when_unmeasured():
    layout = sheetlib.plan([], frame_size=128, yaws=4)
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png", created=1.0
    )
    assert meta["cells"][0]["pivot_x"] == 64.0
    assert meta["cells"][0]["pivot_y"] == 128.0


def test_sidecar_carries_the_trim_it_was_measured(tmp_path):
    layout = sheetlib.plan([], frame_size=64, yaws=2)
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png", created=1.0,
        trims={0: {"x": 1, "y": 2, "w": 3, "h": 4}},
    )
    assert meta["cells"][0]["trim"] == {"x": 1, "y": 2, "w": 3, "h": 4}
    # A cell nothing was measured for says so, rather than claiming full bleed.
    assert meta["cells"][1]["trim"] is None


def test_trim_measures_the_alpha_bounding_box():
    frame = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    frame.paste((255, 0, 0, 255), (10, 20, 30, 50))
    assert sheetlib.measure_trim(frame) == {"x": 10, "y": 20, "w": 20, "h": 30}


def test_trim_of_an_empty_frame_is_none():
    assert sheetlib.measure_trim(Image.new("RGBA", (64, 64), (0, 0, 0, 0))) is None


def test_pack_returns_a_trim_for_every_cell(tmp_path):
    """Measured while packing because every frame is already open and decoded
    there; a second pass over the atlas would be the same pixels twice."""
    layout = sheetlib.plan([], frame_size=64, yaws=2)
    frames = {}
    for cell in layout.cells:
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        if cell.index == 0:
            img.paste((0, 255, 0, 255), (8, 16, 24, 40))
        path = tmp_path / f"{cell.index}.png"
        img.save(path)
        frames[cell.index] = path
    trims = sheetlib.pack(layout, frames, tmp_path / "sheet.png")
    assert trims[0] == {"x": 8, "y": 16, "w": 16, "h": 24}
    assert trims[1] is None


def test_sidecar_is_plain_json():
    layout = sheetlib.plan([])
    meta = sheetlib.sidecar(
        layout, sheet_id="c" * 12, source_job="d" * 12, image="c.png", created=1.0
    )
    assert json.loads(json.dumps(meta)) == meta


# --- storage ----------------------------------------------------------------


def _write_sheet(job_dir, sheet_id, *, png=True, created=0.0):
    if png:
        Image.new("RGBA", (8, 8)).save(rigging.sheet_png_path(job_dir, sheet_id))
    rigging.sheet_path(job_dir, sheet_id).write_text(
        json.dumps({"id": sheet_id, "created": created, "rows": 1, "columns": 8,
                    "frame_size": 64, "lighting": "flat", "elevation": 30.0})
    )


def test_a_sidecar_without_its_png_is_not_listed(tmp_path):
    """The PNG is written first and the sidecar last, so a sidecar alone means
    a half-cleaned directory, not a finished sheet."""
    rigging.sheet_dir(tmp_path).mkdir(parents=True)
    _write_sheet(tmp_path, "a" * 12, png=False)
    _write_sheet(tmp_path, "b" * 12, created=1.0)
    assert [s["id"] for s in rigging.list_sheets(tmp_path)] == ["b" * 12]


def test_delete_sheet_removes_both_files(tmp_path):
    rigging.sheet_dir(tmp_path).mkdir(parents=True)
    _write_sheet(tmp_path, "a" * 12)
    assert rigging.delete_sheet(tmp_path, "a" * 12) is True
    assert rigging.list_sheets(tmp_path) == []
    assert rigging.delete_sheet(tmp_path, "a" * 12) is False


@pytest.mark.parametrize("bad", ["..", "../x", "not-an-id", ""])
def test_sheet_paths_reject_ids_that_are_not_ours(tmp_path, bad):
    with pytest.raises(ValueError):
        rigging.sheet_path(tmp_path, bad)


# --- HTTP -------------------------------------------------------------------


@pytest.fixture
def sheet_client(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(assets))
    monkeypatch.setenv("WARLOCK_DB", str(assets / "jobs.sqlite"))
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(config_mod, "_config", None)
    monkeypatch.setattr(JobStore, "next_queued", lambda self: None)
    with TestClient(create_app()) as c:
        yield c, assets


def _mesh_job(client, assets, *, rigged=False) -> str:
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "a knight"}).json()["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if rigged:
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(json.dumps({"bones": [{"name": "hips"}]}))
    client.app.state.store.set_status(job_id, "done")
    return job_id


def test_options_route_is_the_single_source_for_the_form(sheet_client):
    client, _ = sheet_client
    body = client.get("/api/sheets/options").json()
    assert body["frame_sizes"] == list(sheetlib.FRAME_SIZES)
    assert body["lighting"] == list(sheetlib.LIGHTING)
    assert len(body["yaws"]) == 8
    assert body["defaults"]["frame_size"] in body["frame_sizes"]


def test_a_sheet_can_be_queued_for_an_unrigged_mesh(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets)
    r = client.post(f"/api/jobs/{job_id}/sheets", data={"frame_size": 64})
    assert r.status_code == 200
    job = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert job["kind"] == "sheet"
    assert job["params"]["source_job"] == job_id
    assert job["params"]["poses"] == []
    # Allocated up front so a cancelled job knows what to clean up.
    assert rigging.is_valid_id(job["params"]["sheet_id"])


def test_a_sheet_needs_a_finished_mesh(sheet_client):
    client, _ = sheet_client
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/sheets").status_code == 400


def test_a_posed_sheet_needs_a_rig(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets)
    pose = {"name": "idle", "bones": {"hips": IDENTITY}}
    rigging.save_pose(assets / job_id, pose, "a" * 12)
    r = client.post(f"/api/jobs/{job_id}/sheets", data={"poses": ["a" * 12]})
    assert r.status_code == 400
    assert "rigged" in r.json()["detail"]


def test_a_sheet_names_the_poses_it_will_render(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets, rigged=True)
    ids = [
        rigging.save_pose(assets / job_id, {"name": n, "bones": {"hips": IDENTITY}})["id"]
        for n in ("idle", "run")
    ]
    r = client.post(f"/api/jobs/{job_id}/sheets", data={"poses": ids, "lighting": "lit"})
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["poses"] == ids
    assert params["lighting"] == "lit"


def test_a_sheet_of_a_deleted_pose_is_404(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets, rigged=True)
    r = client.post(f"/api/jobs/{job_id}/sheets", data={"poses": ["a" * 12]})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "data",
    [
        {"frame_size": 100},
        {"lighting": "toon"},
        {"elevation": 91},
    ],
)
def test_an_unrenderable_sheet_is_refused_before_it_costs_a_queue_slot(sheet_client, data):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets)
    assert client.post(f"/api/jobs/{job_id}/sheets", data=data).status_code == 400


def test_finished_sheets_are_listed_downloaded_and_deleted(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets)
    rigging.sheet_dir(assets / job_id).mkdir(parents=True)
    _write_sheet(assets / job_id, "a" * 12)

    listed = client.get(f"/api/jobs/{job_id}/sheets").json()["sheets"]
    assert [s["id"] for s in listed] == ["a" * 12]
    assert client.get(f"/api/jobs/{job_id}/sheets/{'a' * 12}").json()["id"] == "a" * 12
    png = client.get(f"/api/jobs/{job_id}/sheets/{'a' * 12}/sheet.png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert client.delete(f"/api/jobs/{job_id}/sheets/{'a' * 12}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/sheets").json()["sheets"] == []


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", "0123456789abcd"])
def test_sheet_routes_reject_malformed_sheet_ids(sheet_client, bad):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets)
    assert client.get(f"/api/jobs/{job_id}/sheets/{bad}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/sheets/{bad}/sheet.png").status_code == 404
    assert client.delete(f"/api/jobs/{job_id}/sheets/{bad}").status_code == 404



def test_a_clip_sheet_records_its_two_ends_not_the_expanded_frames(sheet_client):
    """params carries the ends so the queue rebuilds the frames from the same
    interpolate() the route validated against -- one source of truth."""
    client, assets = sheet_client
    job_id = _mesh_job(client, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}})
    b = rigging.save_pose(
        assets / job_id, {"name": "B", "bones": {"hips": [0.0, 0.0, 0.7071068, 0.7071068]}}
    )

    r = client.post(
        f"/api/jobs/{job_id}/sheets",
        data={"clip_from": a["id"], "clip_to": b["id"], "clip_frames": 4, "yaws": 4},
    )
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["clip"] == {"from": a["id"], "to": b["id"], "frames": 4}
    assert params["yaws"] == 4


def test_a_clip_needs_both_ends(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}})
    r = client.post(f"/api/jobs/{job_id}/sheets", data={"clip_from": a["id"]})
    assert r.status_code == 400
    assert "both" in r.json()["detail"]


def test_a_clip_needs_a_rig(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}}, "a" * 12)
    b = rigging.save_pose(assets / job_id, {"name": "B", "bones": {"hips": IDENTITY}}, "b" * 12)
    r = client.post(
        f"/api/jobs/{job_id}/sheets", data={"clip_from": a["id"], "clip_to": b["id"]}
    )
    assert r.status_code == 400
    assert "rigged" in r.json()["detail"]


def test_a_clip_frame_count_over_the_limit_is_a_400(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}}, "a" * 12)
    b = rigging.save_pose(assets / job_id, {"name": "B", "bones": {"hips": IDENTITY}}, "b" * 12)
    r = client.post(
        f"/api/jobs/{job_id}/sheets",
        data={"clip_from": a["id"], "clip_to": b["id"], "clip_frames": 999},
    )
    assert r.status_code == 400


def test_a_clip_end_that_no_longer_exists_is_a_404(sheet_client):
    client, assets = sheet_client
    job_id = _mesh_job(client, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}}, "a" * 12)
    r = client.post(
        f"/api/jobs/{job_id}/sheets", data={"clip_from": a["id"], "clip_to": "b" * 12}
    )
    assert r.status_code == 404


# --- the queue --------------------------------------------------------------


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


def _fake_render(monkeypatch, *, side_effect=None, hold=None):
    """Stand in for the Blender render, writing one frame per requested cell."""
    calls: list[dict] = []

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0):
        calls.append({"spec": spec, "timeout": timeout})
        if on_progress is not None:
            on_progress(0.5, "Rendering 8/16")
        if hold is not None:
            hold.wait(timeout=10)
        if side_effect is not None:
            raise side_effect
        frames_dir = Path(spec["frames_dir"])
        for cell in spec["cells"]:
            Image.new(
                "RGBA", (spec["frame_size"],) * 2, (cell["index"], 0, 0, 255)
            ).save(frames_dir / f"{cell['index']:04d}.png")
        return {"ok": True, "frames": [c["index"] for c in spec["cells"]]}

    monkeypatch.setattr(rigging, "run_worker", fake)
    return calls


def _source_job(worker: Worker, *, rigged=False) -> str:
    job_id = worker.store.create("text", "a knight", {"seed": 1})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if rigged:
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(json.dumps({"bones": [{"name": "hips"}]}))
    worker.store.set_status(job_id, "done")
    return job_id


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


@pytest.mark.asyncio
async def test_a_sheet_job_writes_into_the_source_jobs_directory(worker, monkeypatch):
    """A sheet belongs to the mesh it depicts, like a rig does."""
    calls = _fake_render(monkeypatch)
    source = _source_job(worker, rigged=True)
    source_dir = worker.config.job_dir(source)
    pose = rigging.save_pose(source_dir, {"name": "idle", "bones": {"hips": IDENTITY}})
    sheet_id = rigging.new_id()
    job_id = worker.store.create(
        "sheet",
        None,
        {"source_job": source, "sheet_id": sheet_id, "poses": [pose["id"]],
         "frame_size": 64, "elevation": 20.0, "lighting": "flat"},
    )

    worker.start()
    try:
        await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    finally:
        await worker.shutdown()

    assert worker.store.get(job_id)["error"] is None
    png = rigging.sheet_png_path(source_dir, sheet_id)
    with Image.open(png) as atlas:
        assert atlas.size == (8 * 64, 64)
    meta = rigging.read_sheet(source_dir, sheet_id)
    assert meta["rows"] == 1 and meta["columns"] == 8
    assert meta["poses"] == [{"id": pose["id"], "name": "idle"}]
    # The pose's rotations ride along with the cell: the worker has no access
    # to the job directory.
    spec = calls[0]["spec"]
    assert spec["cells"][0]["bones"] == {"hips": IDENTITY}
    assert spec["source_glb"].endswith("rig.glb")
    assert calls[0]["timeout"] == worker.config.sheet_timeout


@pytest.mark.asyncio
async def test_an_unrigged_sheet_renders_the_plain_mesh(worker, monkeypatch):
    calls = _fake_render(monkeypatch)
    source = _source_job(worker)
    job_id = worker.store.create(
        "sheet", None,
        {"source_job": source, "sheet_id": rigging.new_id(), "poses": [], "frame_size": 64},
    )
    worker.start()
    try:
        await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    finally:
        await worker.shutdown()
    assert worker.store.get(job_id)["error"] is None
    assert calls[0]["spec"]["source_glb"].endswith("model.glb")
    assert len(calls[0]["spec"]["cells"]) == 8


@pytest.mark.asyncio
async def test_cancelling_a_sheet_never_deletes_the_source_mesh(worker, monkeypatch):
    """The sheet lives in someone else's directory; cleanup must not overreach."""
    import threading

    hold = threading.Event()
    _fake_render(monkeypatch, hold=hold)
    source = _source_job(worker)
    source_dir = worker.config.job_dir(source)
    sheet_id = rigging.new_id()
    job_id = worker.store.create(
        "sheet", None,
        {"source_job": source, "sheet_id": sheet_id, "poses": [], "frame_size": 64},
    )

    worker.start()
    try:
        await _wait_until(lambda: worker.current_job_id == job_id)
        await worker.request_cancel(job_id)
        worker.store.cancel(job_id)
        hold.set()
        await _wait_until(lambda: worker.current_job_id is None)
    finally:
        hold.set()
        await worker.shutdown()

    assert (source_dir / "model.glb").exists()
    assert not rigging.sheet_png_path(source_dir, sheet_id).exists()
    assert not rigging.sheet_path(source_dir, sheet_id).exists()


@pytest.mark.asyncio
async def test_a_sheet_of_a_vanished_pose_fails_the_job_not_the_worker(worker, monkeypatch):
    _fake_render(monkeypatch)
    source = _source_job(worker, rigged=True)
    job_id = worker.store.create(
        "sheet", None,
        {"source_job": source, "sheet_id": rigging.new_id(), "poses": ["a" * 12]},
    )
    worker.start()
    try:
        await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
        # Checked before shutdown: a failed job must not take the queue with it.
        assert worker.alive
    finally:
        await worker.shutdown()
    assert worker.store.get(job_id)["status"] == "error"


@pytest.mark.asyncio
async def test_a_clip_job_gives_every_row_its_own_frames_rotations(worker, monkeypatch):
    """The cache keys on the queue and in the worker both had to grow a frame
    component. Get either wrong and every row of the clip renders frame 0 --
    which looks like a posing bug, not a keying bug."""
    calls = _fake_render(monkeypatch)
    source = _source_job(worker, rigged=True)
    source_dir = worker.config.job_dir(source)
    a = rigging.save_pose(source_dir, {"name": "A", "bones": {"hips": IDENTITY}})
    b = rigging.save_pose(
        source_dir, {"name": "B", "bones": {"hips": [0.0, 0.0, 0.7071068, 0.7071068]}}
    )
    job_id = worker.store.create(
        "sheet",
        None,
        {
            "source_job": source,
            "sheet_id": rigging.new_id(),
            "frame_size": 64,
            "yaws": 2,
            "clip": {"from": a["id"], "to": b["id"], "frames": 4},
        },
    )

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")

    cells = calls[0]["spec"]["cells"]
    assert len(cells) == 8                      # 4 frames x 2 yaws
    assert [c["frame"] for c in cells] == [0, 0, 1, 1, 2, 2, 3, 3]
    # One distinct rotation per frame, and frame 0 is end A exactly.
    per_frame = {c["frame"]: tuple(c["bones"]["hips"]) for c in cells}
    assert len(set(per_frame.values())) == 4
    assert per_frame[0] == pytest.approx(tuple(IDENTITY))
    await worker.shutdown()


@pytest.mark.asyncio
async def test_a_clip_whose_pose_was_deleted_fails_the_job(worker, monkeypatch):
    _fake_render(monkeypatch)
    source = _source_job(worker, rigged=True)
    source_dir = worker.config.job_dir(source)
    a = rigging.save_pose(source_dir, {"name": "A", "bones": {"hips": IDENTITY}})
    job_id = worker.store.create(
        "sheet",
        None,
        {
            "source_job": source,
            "sheet_id": rigging.new_id(),
            "clip": {"from": a["id"], "to": rigging.new_id(), "frames": 2},
        },
    )

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "error")
    assert "no longer exists" in worker.store.get(job_id)["error"]
    await worker.shutdown()


# --- the real renderer ------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("lighting", ["flat", "lit"])
def test_a_rendered_sheet_actually_contains_eight_distinct_views(tmp_path, lighting):
    """The half only Blender can do: pixels in the frames, transparency around
    them, and a different silhouette in every column."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    # Deliberately lopsided, so a camera that failed to turn would show up as
    # eight identical cells rather than eight plausible ones.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)
    bpy.ops.mesh.primitive_cone_add(radius1=0.35, depth=0.6, location=(0, -0.5, 0.9))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")

    layout = sheetlib.plan([], frame_size=64, elevation=25.0, lighting=lighting)
    frames_dir = tmp_path / "frames"
    spec = rigging.sheet_spec(
        tmp_path / "model.glb",
        frames_dir,
        [{"index": c.index, "yaw": c.yaw, "pose": None, "bones": {}} for c in layout.cells],
        frame_size=64,
        elevation=25.0,
        lighting=lighting,
    )
    result = blender_worker.op_sheet(bpy, spec)
    assert result["frames"] == [c.index for c in layout.cells]

    frames = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
    out = tmp_path / "sheet.png"
    sheetlib.pack(layout, frames, out)
    with Image.open(out) as atlas:
        assert atlas.size == (512, 64)
        cells = [atlas.crop((c.x, c.y, c.x + 64, c.y + 64)) for c in layout.cells]
        # Every column shows something...
        for i, cell in enumerate(cells):
            opaque = sum(1 for v in cell.getchannel("A").get_flattened_data() if v > 8)
            assert 0.02 < opaque / (64 * 64) < 0.9, f"column {i} covers {opaque} px"
        # ...the background is not part of it...
        assert atlas.getpixel((1, 1))[3] == 0
        # ...and no two columns are the same picture.
        assert len({c.tobytes() for c in cells}) == len(cells)


@pytest.mark.gpu
def test_a_rigged_subject_is_framed_by_its_own_size(tmp_path):
    """Regression: Blender's glTF importer adds bone-shape widgets in a
    glTF_not_exported collection. They never render, but they were still in
    scene.objects -- a unit icosphere among them trebled the measured bounds
    and framed every rigged sheet's subject at a third of its proper size."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)   # 0.6 x 0.4 x 2.0
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")
    blender_worker.op_rig(bpy, rigging.rig_spec(tmp_path, "humanoid"))

    layout = sheetlib.plan([], frame_size=128, elevation=0.0)
    frames_dir = tmp_path / "frames"
    blender_worker.op_sheet(
        bpy,
        rigging.sheet_spec(
            tmp_path / "rig.glb",
            frames_dir,
            [{"index": c.index, "yaw": c.yaw, "pose": None, "bones": {}} for c in layout.cells],
            frame_size=128,
            elevation=0.0,
            lighting="flat",
        ),
    )
    with Image.open(frames_dir / "0000.png") as frame:
        mask = frame.convert("RGBA").getchannel("A").point(lambda v: 255 if v > 8 else 0)
        box = mask.getbbox()
    height = box[3] - box[1]
    # 2.0 tall inside a 2.24 ortho window is ~89% of the frame; anything near a
    # third of that means something else got into the bounds again.
    assert height / 128 > 0.75, f"subject only fills {height}/128 px"


@pytest.mark.gpu
def test_the_reported_pivot_sits_at_the_subjects_feet_in_every_direction(tmp_path):
    """The sidecar's pivot is only worth anything if an engine can place a
    sprite by it without the subject drifting as it turns.

    Rendered at elevation 0 with a lopsided subject, so a pivot computed from
    the wrong point or projected with the wrong sign would land somewhere the
    silhouette is not. Checked against each cell's own alpha bbox, which is the
    same rectangle a Pygame ``subsurface`` blit would honour.
    """
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)
    bpy.ops.mesh.primitive_cone_add(radius1=0.35, depth=0.6, location=(0, -0.5, 0.9))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")

    layout = sheetlib.plan([], frame_size=128, elevation=0.0)
    frames_dir = tmp_path / "frames"
    spec = rigging.sheet_spec(
        tmp_path / "model.glb",
        frames_dir,
        [{"index": c.index, "yaw": c.yaw, "pose": None, "bones": {}} for c in layout.cells],
        frame_size=128,
        elevation=0.0,
        lighting="flat",
    )
    result = blender_worker.op_sheet(bpy, spec)
    pivot = result["pivot"]

    frames = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
    trims = sheetlib.pack(layout, frames, tmp_path / "sheet.png")
    meta = sheetlib.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png", created=1.0,
        pivot=(pivot[0], pivot[1]), trims=trims,
    )

    px, py = meta["cells"][0]["pivot_x"], meta["cells"][0]["pivot_y"]
    # Horizontally centred, and on the floor: at elevation 0 the bottom of the
    # silhouette *is* the ground plane.
    assert px == pytest.approx(64.0, abs=1.5)
    for cell in meta["cells"]:
        trim = cell["trim"]
        assert trim is not None, f"column {cell['column']} rendered nothing"
        # Every cell carries the same pivot -- that stability is the point.
        assert (cell["pivot_x"], cell["pivot_y"]) == (px, py)
        bottom = trim["y"] + trim["h"]
        assert abs(py - bottom) <= 2, f"column {cell['column']}: pivot {py} vs floor {bottom}"
        assert trim["x"] <= px <= trim["x"] + trim["w"]

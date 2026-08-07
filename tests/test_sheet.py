"""Sprite sheets: the pure layout/packing module, the service surface, and the
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

import numpy as np
import pytest
from PIL import Image

from warlock import models, rigging
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import sheet as sheetlib
from warlock.queue import Worker
from warlock.service import Invalid, NotFound
from warlock.service import jobs as svc_jobs
from warlock.service import sheets as svc_sheets

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


# --- the service surface ----------------------------------------------------


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def _mesh_job(svc, assets, *, rigged=False) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if rigged:
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(json.dumps({"bones": [{"name": "hips"}]}))
    svc.store.set_status(job_id, "done")
    return job_id


def test_the_options_call_is_the_single_source_for_the_form():
    body = svc_sheets.sheet_options()
    assert body["frame_sizes"] == list(sheetlib.FRAME_SIZES)
    assert body["lighting"] == list(sheetlib.LIGHTING)
    assert len(body["yaws"]) == 8
    assert body["defaults"]["frame_size"] in body["frame_sizes"]


def test_a_sheet_can_be_queued_for_an_unrigged_mesh(svc, assets):
    job_id = _mesh_job(svc, assets)
    out = svc_sheets.create_sheet(svc, job_id, frame_size=64)
    job = svc.store.get(out["id"])
    assert job["kind"] == "sheet"
    assert job["params"]["source_job"] == job_id
    assert job["params"]["poses"] == []
    # Allocated up front so a cancelled job knows what to clean up.
    assert rigging.is_valid_id(job["params"]["sheet_id"])


def test_a_sheet_needs_a_finished_mesh(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid):
        svc_sheets.create_sheet(svc, job_id)


def test_a_posed_sheet_needs_a_rig(svc, assets):
    job_id = _mesh_job(svc, assets)
    rigging.save_pose(assets / job_id, {"name": "idle", "bones": {"hips": IDENTITY}}, "a" * 12)
    with pytest.raises(Invalid, match="rigged"):
        svc_sheets.create_sheet(svc, job_id, poses=["a" * 12])


def test_a_sheet_names_the_poses_it_will_render(svc, assets):
    job_id = _mesh_job(svc, assets, rigged=True)
    ids = [
        rigging.save_pose(assets / job_id, {"name": n, "bones": {"hips": IDENTITY}})["id"]
        for n in ("idle", "run")
    ]
    out = svc_sheets.create_sheet(svc, job_id, poses=ids, lighting="lit")
    params = svc.store.get(out["id"])["params"]
    assert params["poses"] == ids
    assert params["lighting"] == "lit"


def test_an_overlong_sheet_name_is_rejected(svc, assets):
    """Capped like a pose name; it is a label the UI has to render."""
    job_id = _mesh_job(svc, assets)
    with pytest.raises(Invalid, match="at most"):
        svc_sheets.create_sheet(svc, job_id, name="x" * (rigging.MAX_SHEET_NAME + 1))
    assert svc_sheets.create_sheet(svc, job_id, name="x" * rigging.MAX_SHEET_NAME)


def test_a_sheet_of_a_deleted_pose_is_not_found(svc, assets):
    job_id = _mesh_job(svc, assets, rigged=True)
    with pytest.raises(NotFound):
        svc_sheets.create_sheet(svc, job_id, poses=["a" * 12])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_size": 100},
        {"lighting": "toon"},
        {"elevation": 91},
    ],
)
def test_an_unrenderable_sheet_is_refused_before_it_costs_a_queue_slot(svc, assets, kwargs):
    job_id = _mesh_job(svc, assets)
    with pytest.raises(Invalid):
        svc_sheets.create_sheet(svc, job_id, **kwargs)


def test_finished_sheets_are_listed_read_and_deleted(svc, assets):
    job_id = _mesh_job(svc, assets)
    rigging.sheet_dir(assets / job_id).mkdir(parents=True)
    _write_sheet(assets / job_id, "a" * 12)

    listed = svc_sheets.list_sheets(svc, job_id)["sheets"]
    assert [s["id"] for s in listed] == ["a" * 12]
    assert svc_sheets.get_sheet(svc, job_id, "a" * 12)["id"] == "a" * 12
    assert svc_sheets.sheet_png(svc, job_id, "a" * 12).read_bytes().startswith(b"\x89PNG")
    assert svc_sheets.delete_sheet(svc, job_id, "a" * 12) == {"ok": True}
    assert svc_sheets.list_sheets(svc, job_id)["sheets"] == []


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", "0123456789abcd"])
def test_the_sheet_entry_points_reject_malformed_sheet_ids(svc, assets, bad):
    job_id = _mesh_job(svc, assets)
    for call in (svc_sheets.get_sheet, svc_sheets.sheet_png, svc_sheets.delete_sheet):
        with pytest.raises(NotFound):
            call(svc, job_id, bad)


def test_a_clip_sheet_records_its_two_ends_not_the_expanded_frames(svc, assets):
    """params carries the ends so the queue rebuilds the frames from the same
    interpolate() this validated against -- one source of truth."""
    job_id = _mesh_job(svc, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}})
    b = rigging.save_pose(
        assets / job_id, {"name": "B", "bones": {"hips": [0.0, 0.0, 0.7071068, 0.7071068]}}
    )
    out = svc_sheets.create_sheet(
        svc, job_id, clip_from=a["id"], clip_to=b["id"], clip_frames=4, yaws=4
    )
    params = svc.store.get(out["id"])["params"]
    assert params["clip"] == {"from": a["id"], "to": b["id"], "frames": 4}
    assert params["yaws"] == 4


def test_a_clip_needs_both_ends(svc, assets):
    job_id = _mesh_job(svc, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}})
    with pytest.raises(Invalid, match="both"):
        svc_sheets.create_sheet(svc, job_id, clip_from=a["id"])


def test_a_clip_needs_a_rig(svc, assets):
    job_id = _mesh_job(svc, assets)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}}, "a" * 12)
    b = rigging.save_pose(assets / job_id, {"name": "B", "bones": {"hips": IDENTITY}}, "b" * 12)
    with pytest.raises(Invalid, match="rigged"):
        svc_sheets.create_sheet(svc, job_id, clip_from=a["id"], clip_to=b["id"])


def test_a_clip_frame_count_over_the_limit_is_refused(svc, assets):
    job_id = _mesh_job(svc, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}}, "a" * 12)
    b = rigging.save_pose(assets / job_id, {"name": "B", "bones": {"hips": IDENTITY}}, "b" * 12)
    with pytest.raises(Invalid):
        svc_sheets.create_sheet(
            svc, job_id, clip_from=a["id"], clip_to=b["id"], clip_frames=999
        )


def test_a_clip_end_that_no_longer_exists_is_not_found(svc, assets):
    job_id = _mesh_job(svc, assets, rigged=True)
    a = rigging.save_pose(assets / job_id, {"name": "A", "bones": {"hips": IDENTITY}}, "a" * 12)
    with pytest.raises(NotFound):
        svc_sheets.create_sheet(svc, job_id, clip_from=a["id"], clip_to="b" * 12)


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
    rigging.finalize_rig(tmp_path)

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


# --- the pixel restyle -------------------------------------------------------


def _rendered_sheet(worker, source, *, frame_size=128, columns=8, rows=1):
    """A finished render on disk, with a subject in each cell."""
    source_dir = worker.config.job_dir(source)
    sheet_id = rigging.new_id()
    png = rigging.sheet_png_path(source_dir, sheet_id)
    png.parent.mkdir(parents=True, exist_ok=True)
    atlas = Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 0))
    for row in range(rows):
        for column in range(columns):
            block = Image.new("RGBA", (frame_size // 2,) * 2, (200, 60, 60, 255))
            atlas.paste(
                block,
                (column * frame_size + frame_size // 4, row * frame_size + frame_size // 4),
            )
    atlas.save(png)
    meta = {
        "version": 1, "id": sheet_id, "name": "turnaround", "source_job": source,
        "created": 1.0, "image": png.name, "frame_size": frame_size,
        "columns": columns, "rows": rows,
        "width": frame_size * columns, "height": frame_size * rows,
        "elevation": 30.0, "lighting": "flat",
        "yaws": [i * 360.0 / columns for i in range(columns)],
        "poses": [{"id": None, "name": "rest"}],
        "cells": [
            {
                "index": row * columns + column, "row": row, "column": column,
                "x": column * frame_size, "y": row * frame_size,
                "w": frame_size, "h": frame_size,
                "pose": None, "pose_name": "rest",
                "yaw": column * 360.0 / columns, "frame": 0,
                "pivot_x": frame_size / 2, "pivot_y": float(frame_size),
                "trim": None,
            }
            for row in range(rows)
            for column in range(columns)
        ],
    }
    rigging.sheet_path(source_dir, sheet_id).write_text(json.dumps(meta), encoding="utf-8")
    return sheet_id


async def _run(worker, job_id):
    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(job_id)["status"] in ("done", "error", "cancelled")
        )
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


@pytest.mark.asyncio
async def test_a_restyle_writes_its_pair_beside_the_render(worker):
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    source_dir = worker.config.job_dir(source)
    job_id = worker.store.create(
        "pixel_sheet", "a knight",
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 32,
         "colors": 8, "seed": 3},
    )

    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    png = rigging.sheet_pixel_png_path(source_dir, sheet_id)
    with Image.open(png) as out:
        # 8 columns x 128px, reduced by 128/32 = 4.
        assert out.size == (8 * 32, 32)
        arr = np.asarray(out.convert("RGBA"))
    doc = rigging.read_sheet_pixel(source_dir, sheet_id)
    assert doc["frame_size"] == 32 and doc["columns"] == 8
    assert doc["restyle"]["seed"] == 3
    assert len(doc["palette"]) <= 8

    # The silhouette is the render's, exactly -- and the colours are the
    # generation's, which is what makes those two separable claims.
    assert set(np.unique(arr[:, :, 3])) <= {0, 255}
    assert arr[0, 0, 3] == 0
    assert arr[16, 16, 3] == 255


@pytest.mark.asyncio
async def test_the_render_survives_a_cancelled_restyle(worker):
    """The restyle lives in someone else's directory: a cancel must take its
    own pair and nothing else."""
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    source_dir = worker.config.job_dir(source)
    job_id = worker.store.create(
        "pixel_sheet", "a knight",
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 32, "colors": 8},
    )

    worker.start()
    try:
        await _wait_until(lambda: worker.current_job_id == job_id)
        await worker.request_cancel(job_id)
        worker.store.cancel(job_id)
        await _wait_until(lambda: worker.current_job_id is None)
    finally:
        await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "cancelled"
    assert not rigging.sheet_pixel_path(source_dir, sheet_id).exists()
    assert not rigging.sheet_pixel_png_path(source_dir, sheet_id).exists()
    # The render it was derived from is minutes of Blender belonging to a
    # different, successful job.
    assert rigging.sheet_png_path(source_dir, sheet_id).exists()
    assert rigging.sheet_path(source_dir, sheet_id).exists()
    assert (source_dir / "model.glb").exists()


@pytest.mark.asyncio
async def test_a_restyle_of_a_deleted_sheet_fails_rather_than_inventing_one(worker):
    source = _source_job(worker)
    job_id = worker.store.create(
        "pixel_sheet", "a knight",
        {"source_job": source, "sheet_id": rigging.new_id(), "logical_size": 32},
    )
    row = await _run(worker, job_id)
    assert row["status"] == "error"


@pytest.mark.asyncio
async def test_every_band_is_generated_under_one_seed(worker):
    """Twelve rows is two bands, and one identity down the whole atlas is the
    reason the restyle is banded rather than per cell."""
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source, rows=12)
    job_id = worker.store.create(
        "pixel_sheet", "a knight",
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 32,
         "colors": 8, "seed": 11},
    )

    row = await _run(worker, job_id)

    assert row["error"] is None
    pipe = worker._text2image
    assert len(pipe.seeds) == 2
    assert set(pipe.seeds) == {11}
    # And it is a sheet generation, with the pixel LoRA -- not a reference one.
    assert all(pipe.sheets)
    assert {lora for lora, _weight in pipe.lora_calls} == {models.PIXEL_SHEET_LORA}
    assert all(c.uses_init for c in pipe.conditionings)

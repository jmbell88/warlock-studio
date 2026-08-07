"""Parity pins for the service layer.

These are the behaviors the HTTP routes used to own and that the desktop UI now
depends on directly. They are deliberately about *rules* -- what a seed fan-out
does to candidate 0, what a promotion inherits, which artifacts a retarget
invalidates -- rather than about wiring, because wiring is what the extraction
changed and rules are what it must not have.
"""

from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from warlock.service import Conflict, Invalid, NotFound, NotReady, TooLarge
from warlock.service import derive as svc_derive
from warlock.service import export as svc_export
from warlock.service import files as svc_files
from warlock.service import jobs as svc_jobs
from warlock.service import rig as svc_rig
from warlock.service import sheets as svc_sheets
from warlock.service.validation import DERIVED_PARAMS, MAX_SEED


def _png_bytes(size=(8, 8), fmt="PNG") -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, fmt)
    return buf.getvalue()


# --- seeds ------------------------------------------------------------------


def test_candidate_zero_keeps_the_requested_seed_and_the_rest_fan_out(svc):
    """A pinned seed must still reproduce even when asking for eight tries."""
    ids = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", seed=1234, output="reference", count=4
    )["ids"]
    params = [svc.store.get(i)["params"] for i in ids]

    assert params[0]["seed"] == 1234
    assert params[0]["reference_seed"] == 1234
    assert params[0]["mesh_seed"] == 1234
    for p in params[1:]:
        assert p["seed"] != 1234
        # All three move together: the settings panel shows them side by side,
        # and one of them silently not following would read as a bug.
        assert p["seed"] == p["reference_seed"] == p["mesh_seed"]
    assert len({p["seed"] for p in params}) == 4


def test_seeds_are_bounded_at_the_door(svc):
    for field in ("seed", "reference_seed", "mesh_seed"):
        with pytest.raises(Invalid):
            svc_jobs.create_job(svc, kind="text", prompt="x", **{field: MAX_SEED + 1})
        with pytest.raises(Invalid):
            svc_jobs.create_job(svc, kind="text", prompt="x", **{field: -1})


def test_a_reference_seed_defaults_to_the_legacy_single_seed(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", seed=7)["id"]
    params = svc.store.get(job_id)["params"]
    assert params["reference_seed"] == params["mesh_seed"] == 7


# --- VRAM admission ---------------------------------------------------------


def _small_card(svc, total=8.0):
    from warlock import vram

    svc.vram_plan = vram.plan(exclusive=None, total_gib=total)
    return svc


def test_a_job_the_card_cannot_hold_is_refused_at_the_door(svc):
    _small_card(svc)
    with pytest.raises(Invalid) as exc:
        svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="model")
    assert "GiB of VRAM" in str(exc.value)
    # No row, and -- the ordering that matters -- nothing on disk either.
    assert svc.store.list(limit=10) == []
    assert [p for p in svc.config.data_dir.iterdir() if p.is_dir()] == []


def test_a_refused_upload_leaves_no_input_png(svc):
    _small_card(svc)
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=_png_bytes(), output="model")
    assert not list(svc.config.data_dir.glob("*/input.png"))


def test_a_reference_still_fits_a_card_a_reconstruction_does_not(svc):
    """The gate is per stage, not per app: the cheap half must stay usable."""
    # 12 GiB: the 7 GiB pipe fits, the 16 GiB reconstruction never will.
    _small_card(svc, total=12.0)
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="reference")["id"]
    assert svc.store.get(job_id) is not None
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="model")


def test_a_promotion_is_gated_too(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="reference")["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png_bytes())
    svc.store.claim(job_id)
    svc.store.finish(job_id, "done", None)
    _small_card(svc)
    with pytest.raises(Invalid) as exc:
        svc_jobs.promote_to_model(svc, job_id)
    assert "GiB of VRAM" in str(exc.value)


def test_an_unmeasured_host_enforces_nothing(svc):
    """No plan and no torch: the gate is off, not accidentally strict."""
    assert svc.vram_plan is None
    assert svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="model")["id"]


# --- batching ---------------------------------------------------------------


def test_only_the_cheap_reference_stage_may_be_batched(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", output="model", count=2)


def test_only_a_text_job_can_stop_at_a_reference(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=_png_bytes(), output="reference")


@pytest.mark.parametrize("count", [0, 9])
def test_the_candidate_count_is_bounded(svc, count):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", output="reference", count=count)


# --- uploads ----------------------------------------------------------------


def test_an_oversized_upload_is_refused_before_it_is_decoded(svc):
    with pytest.raises(TooLarge):
        svc_jobs.create_job(svc, kind="image", image=b"\x00" * (20 * 1024 * 1024 + 1))


def test_a_decompression_bomb_is_refused_from_the_header(svc):
    """5000x4000 = 20 MP: tiny on disk as a flat PNG, over the cap decoded."""
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=_png_bytes((5000, 4000)))


def test_alpha_survives_only_when_the_source_had_it(svc):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(buf, "PNG")
    with Image.open(io.BytesIO(svc_files.to_png(buf.getvalue()))) as im:
        assert im.mode == "RGBA"
    with Image.open(io.BytesIO(svc_files.to_png(_png_bytes()))) as im:
        assert im.mode == "RGB"


def test_a_rejected_submit_leaves_no_job_directory_behind(svc):
    """input.png is written before the row exists, so a failed insert has to
    remove the directory it already wrote."""
    def explode(*a, **k):
        raise RuntimeError("db is gone")

    svc.store.create = explode
    before = set(svc.config.data_dir.iterdir())
    with pytest.raises(RuntimeError):
        svc_jobs.create_job(svc, kind="image", image=_png_bytes())
    assert set(svc.config.data_dir.iterdir()) == before


# --- derived params ---------------------------------------------------------


def _finished_job(svc, **params):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", **params)["id"]
    svc.store.merge_params(job_id, {k: "stale" for k in DERIVED_PARAMS})
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_reroll_inherits_no_verdict_about_the_old_mesh(svc):
    job_id = _finished_job(svc)
    new_id = svc_jobs.rerun_job(svc, job_id)["id"]
    params = svc.store.get(new_id)["params"]
    assert not [k for k in DERIVED_PARAMS if k in params]
    assert params["rerun_of"] == job_id


def test_a_reroll_of_a_reference_stops_at_a_reference_again(svc):
    """Otherwise "try another" silently pays for a trellis run nobody asked for."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference")["id"]
    svc.store.set_status(job_id, "done")
    new_id = svc_jobs.rerun_job(svc, job_id)["id"]
    assert svc.store.get(new_id)["stage"] == "reference"


def test_a_remesh_keeps_the_reference_seed_and_rerolls_only_the_mesh(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", seed=11)["id"]
    (svc.job_dir(job_id)).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(job_id, "done")

    new_id = svc_jobs.rerun_job(svc, job_id, mode="remesh", seed=99)["id"]
    params = svc.store.get(new_id)["params"]
    assert params["reference_seed"] == 11
    assert params["mesh_seed"] == 99
    assert svc.store.get(new_id)["kind"] == "image"
    assert svc.store.get(new_id)["stage"] == "model"


def test_a_remesh_needs_a_reference_image_to_reuse(svc):
    job_id = _finished_job(svc)
    with pytest.raises(Invalid):
        svc_jobs.rerun_job(svc, job_id, mode="remesh")


# --- promotion --------------------------------------------------------------


def _reference(svc, **params):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference", **params)["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_promotion_is_a_child_of_its_reference(svc):
    ref = _reference(svc)
    out = svc_jobs.promote_to_model(svc, ref)
    child = svc.store.get(out["id"])
    assert child["parent_id"] == ref
    assert child["kind"] == "image"
    assert child["stage"] == "model"
    assert (svc.job_dir(out["id"]) / "input.png").exists()


def test_a_platform_override_drops_the_resolution_it_implied(svc):
    ref = _reference(svc, resolution=512)
    assert svc.store.get(ref)["params"]["resolution"] == 512
    out = svc_jobs.promote_to_model(svc, ref, platform="3d")
    params = svc.store.get(out["id"])["params"]
    # Re-derived from the new platform, never left contradicting it.
    assert params["platform"] == "3d"
    assert params["resolution"] == 1024


def test_an_explicit_false_clears_an_inherited_rig_request(svc):
    ref = _reference(svc)
    svc.store.merge_params(ref, {"rig": True, "rig_template": "biped"})
    out = svc_jobs.promote_to_model(svc, ref, rig=False)
    params = svc.store.get(out["id"])["params"]
    assert "rig" not in params
    assert "rig_template" not in params


def test_an_omitted_override_keeps_what_the_reference_recorded(svc):
    ref = _reference(svc)
    svc.store.merge_params(ref, {"rig": True, "rig_template": "biped"})
    out = svc_jobs.promote_to_model(svc, ref)
    assert svc.store.get(out["id"])["params"]["rig"] is True


def test_only_a_finished_reference_can_be_promoted(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference")["id"]
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, job_id)
    model_job = _finished_job(svc)
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, model_job)


# --- id guards --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["../etc", "..", "a" * 13, "ABCDEF012345", "", "0123456789ab/x"]
)
def test_a_supplied_id_never_becomes_a_path_component(svc, bad):
    """config.job_dir is a bare join, so every entry point guards its id."""
    for fn in (svc_jobs.get_job, svc_jobs.delete_job, svc_jobs.cancel_job):
        with pytest.raises(NotFound):
            fn(svc, bad)
    with pytest.raises(NotFound):
        svc_derive.get_file(svc, bad, "model.glb")


def test_an_unknown_artifact_name_is_refused_before_any_path_join(svc):
    with pytest.raises(NotFound):
        svc_derive.get_file(svc, "0123456789ab", "../../secrets")


def test_bulk_export_refuses_a_name_outside_the_allowlist(svc, tmp_path):
    with pytest.raises(Invalid):
        svc_export.bulk_export(svc, ["0123456789ab"], ["../secrets"], tmp_path / "o.zip")


# --- file gating ------------------------------------------------------------


def test_model_glb_is_gated_on_status_not_existence(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"half-written")

    svc.store.set_status(job_id, "running")
    assert "model.glb" not in svc_jobs.get_job(svc, job_id)["files"]
    svc.store.set_status(job_id, "done")
    assert "model.glb" in svc_jobs.get_job(svc, job_id)["files"]


def test_rig_glb_is_gated_on_rig_json(svc):
    """The Blender export is not atomic and rig.json is written last."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "rig.glb").write_bytes(b"half-written")
    svc.store.set_status(job_id, "done")

    assert "rig.glb" not in svc_jobs.get_job(svc, job_id)["files"]
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "rig.glb")

    (job_dir / "rig.json").write_text("{}")
    assert "rig.glb" in svc_jobs.get_job(svc, job_id)["files"]
    assert svc_derive.get_file(svc, job_id, "rig.glb").name == "rig.glb"


def test_input_png_stays_existence_based_regardless_of_status(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(job_id, "running")
    assert "input.png" in svc_jobs.get_job(svc, job_id)["files"]


# --- cancellation -----------------------------------------------------------


def test_cancelling_twice_is_an_idempotent_success(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    assert svc_jobs.cancel_job(svc, job_id) == {"ok": True}
    assert svc_jobs.cancel_job(svc, job_id) == {"ok": True}


def test_cancelling_a_finished_job_is_a_conflict(svc):
    job_id = _finished_job(svc)
    with pytest.raises(Conflict):
        svc_jobs.cancel_job(svc, job_id)


def test_a_terminal_write_that_landed_first_stands(svc):
    """The worker's own done/error must not be retroactively overwritten."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "running")

    real_cancel = svc.store.cancel

    def racing_cancel(jid):
        svc.store.finish(jid, "done")
        return real_cancel(jid)

    svc.store.cancel = racing_cancel
    with pytest.raises(Conflict):
        svc_jobs.cancel_job(svc, job_id)
    assert svc.store.get(job_id)["status"] == "done"


def test_a_running_job_cannot_be_deleted_out_from_under_the_worker(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "running")
    with pytest.raises(Conflict):
        svc_jobs.delete_job(svc, job_id)


def test_prune_never_touches_a_running_job(svc):
    ids = [svc_jobs.create_job(svc, kind="text", prompt=str(i))["id"] for i in range(3)]
    svc.store.set_status(ids[0], "running")  # the oldest
    assert svc_jobs.prune_jobs(svc, keep=1)["deleted"] == 1
    assert svc.store.get(ids[0]) is not None


# --- deleting something another job is writing into --------------------------
#
# A rig or a sheet is a separate row whose artifacts land beside the model.glb
# they were fitted to -- the rig belongs to the mesh. So the mesh's own status
# says nothing: it is `done`, which is precisely why a rig could be queued for
# it. Deleting it mid-rig let finalize_rig rename into a directory that was no
# longer there and recreate it as an orphan.


def _rig_job_for(svc, source_id, status="running"):
    return svc.store.create(
        "rig", "", {"source_job": source_id}, stage="model", status=status
    )


def test_a_mesh_cannot_be_deleted_while_a_rig_for_it_is_running(svc):
    source = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(source, "done")
    _rig_job_for(svc, source)

    with pytest.raises(Conflict):
        svc_jobs.delete_job(svc, source)
    assert svc.store.get(source) is not None


def test_a_queued_rig_counts_too(svc):
    """It has not started writing yet, and it will: the worker picks it up on
    the next poll, into a directory this call would have removed."""
    source = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(source, "done")
    _rig_job_for(svc, source, status="queued")

    assert svc_jobs.dependent_jobs(svc, source)
    with pytest.raises(Conflict):
        svc_jobs.delete_job(svc, source)


def test_a_finished_rig_is_no_obstacle(svc):
    source = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(source, "done")
    _rig_job_for(svc, source, status="done")

    assert svc_jobs.dependent_jobs(svc, source) == []
    assert svc_jobs.delete_job(svc, source)["ok"] is True


def test_prune_skips_a_mesh_with_a_rig_in_flight_and_takes_the_rest(svc):
    """Skipped rather than refused: pruning is a bulk reclaim, and one asset
    with a rig running is no reason to keep the other two hundred."""
    ids = [svc_jobs.create_job(svc, kind="text", prompt=str(i))["id"] for i in range(3)]
    for job_id in ids:
        svc.store.set_status(job_id, "done")
    _rig_job_for(svc, ids[0])  # the oldest, which prune would otherwise take

    # keep=1 leaves the newest; the rig row is itself queued/running so it is
    # never a prune candidate either.
    svc_jobs.prune_jobs(svc, keep=1)

    assert svc.store.get(ids[0]) is not None
    assert svc.store.get(ids[1]) is None


def test_a_job_the_worker_is_still_inside_is_not_deleted_however_the_row_reads(svc):
    """``cancel_job`` writes ``cancelled`` and only *asks* the worker to stop,
    so between that write and the worker unwinding the status says the job is
    over while the reconstruction is still writing into its directory."""
    from types import SimpleNamespace

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "cancelled")
    svc.worker = SimpleNamespace(current_job_id=job_id)

    assert svc_jobs.worker_is_inside(svc, job_id) is True
    with pytest.raises(Conflict):
        svc_jobs.delete_job(svc, job_id)

    svc.worker = SimpleNamespace(current_job_id=None)
    assert svc_jobs.delete_job(svc, job_id)["ok"] is True


# --- optimize ---------------------------------------------------------------


def test_optimize_refuses_a_job_the_worker_may_still_be_writing(svc):
    """The worker's half of the write takes no lock, so refusing beats racing."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    for status in ("queued", "running"):
        svc.store.set_status(job_id, status)
        with pytest.raises(Conflict):
            svc_jobs.optimize_job(svc, job_id)


def test_optimize_needs_a_source_reconstruction(svc):
    job_id = _finished_job(svc)
    with pytest.raises(Invalid):
        svc_jobs.optimize_job(svc, job_id)


def test_a_retarget_reports_the_rig_artifacts_it_made_stale(svc):
    """Reported, never deleted: a rig and its poses are user work."""
    from warlock import rigging

    job_id = _finished_job(svc)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "rig.glb").write_bytes(b"x")
    (job_dir / "rig.json").write_text("{}")
    rigging.pose_dir(job_dir).mkdir(parents=True, exist_ok=True)
    (rigging.pose_dir(job_dir) / "abcdef012345.glb").write_bytes(b"x")

    stale = svc_jobs.stale_rig_artifacts(job_dir)
    assert "rig.glb" in stale and "rig.json" in stale
    assert f"{rigging.POSE_DIR_NAME}/abcdef012345.glb" in stale
    # Nothing was removed.
    assert (job_dir / "rig.glb").exists()


def test_derived_artifacts_outlive_the_normalize_that_finishes_the_new_mesh(svc, monkeypatch):
    """Deleted *after* grounding is reapplied, not before.

    ``derive.get_file`` takes only its own artifact's lock, never this one, so
    an STL or OBJ export landing between the unlink and the normalize rebuilt
    itself from the ungrounded mesh -- and cached that answer indefinitely.
    """
    from warlock.pipelines import optimize, postprocess
    from warlock.service import files as svc_files

    job_id = _finished_job(svc)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "source.glb").write_bytes(b"glb")
    (job_dir / "model.glb").write_bytes(b"glb")
    derived = next(iter(svc_files.DERIVED))
    (job_dir / derived).write_bytes(b"stale")

    order: list[str] = []

    def fake_run(source, out, **_kwargs):
        out.write_bytes(b"optimized")
        return {"ok": True}

    def fake_normalize(path, size_m=None):
        # The window: whatever a concurrent export sees here is what it caches.
        order.append(f"normalize:{(job_dir / derived).exists()}")
        return {"scale": 1.0}

    monkeypatch.setattr(optimize, "run", fake_run)
    monkeypatch.setattr(postprocess, "normalize_glb", fake_normalize)
    svc_jobs.optimize_job(svc, job_id, profile="raw")

    assert order == ["normalize:True"]
    assert not (job_dir / derived).exists()


# --- tags -------------------------------------------------------------------


def test_tags_are_folded_and_deduped_on_the_way_in(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job = svc_jobs.update_job(svc, job_id, {"tags": ["Prop", "prop ", "  ", "Weapon"]})
    assert job["tags"] == "prop,weapon"


def test_a_tag_list_is_bounded(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid):
        svc_jobs.update_job(svc, job_id, {"tags": [f"t{i}" for i in range(21)]})
    with pytest.raises(Invalid):
        svc_jobs.update_job(svc, job_id, {"tags": ["x" * 33]})


# --- convert locks ----------------------------------------------------------


def test_one_artifact_is_never_converted_twice_concurrently(svc):
    """The lock table is what makes a double-clicked download one export."""
    job_id = _finished_job(svc)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"not really a glb")

    running = []
    overlapped = []

    def slow_convert(glb, out):
        running.append(1)
        if len(running) > 1:
            overlapped.append(1)
        time.sleep(0.15)
        out.write_bytes(b"stl")
        running.pop()

    from warlock.pipelines import postprocess

    original = postprocess.glb_to_stl
    postprocess.glb_to_stl = slow_convert
    try:
        with ThreadPoolExecutor(4) as pool:
            paths = list(
                pool.map(lambda _: svc_derive.get_file(svc, job_id, "model.stl"), range(4))
            )
    finally:
        postprocess.glb_to_stl = original

    assert not overlapped
    assert all(p.name == "model.stl" for p in paths)


def test_the_lock_table_hands_out_one_lock_per_artifact(svc):
    """Get-or-create races: two locks for one artifact is the double
    conversion the table exists to prevent."""
    seen = []
    barrier = threading.Barrier(8)

    def grab(_):
        barrier.wait()
        seen.append(svc.convert_lock("0123456789ab", "model.stl"))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(grab, range(8)))
    assert len({id(lock) for lock in seen}) == 1


# --- rigs and sheets --------------------------------------------------------


def test_a_rig_job_cannot_be_rigged(svc):
    job_id = svc.store.create("rig", "x", {"source_job": "0123456789ab"}, "aaaaaaaaaaaa")
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid):
        svc_rig.create_rig(svc, job_id)


def test_rigging_needs_a_finished_mesh(svc):
    job_id = _finished_job(svc)
    with pytest.raises(Invalid):
        svc_rig.create_rig(svc, job_id)


def test_a_sheet_needs_a_finished_mesh(svc):
    job_id = _finished_job(svc)
    with pytest.raises(Invalid):
        svc_sheets.create_sheet(svc, job_id)


def test_a_clip_needs_both_ends(svc):
    job_id = _finished_job(svc)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"x")
    with pytest.raises(Invalid):
        svc_sheets.create_sheet(svc, job_id, clip_from="abcdef012345")


def test_a_pose_id_is_validated_before_it_names_a_file(svc):
    job_id = _finished_job(svc)
    with pytest.raises(NotFound):
        svc_rig.delete_pose(svc, job_id, "../../etc/passwd")
    with pytest.raises(NotFound):
        svc_rig.posed_model(svc, job_id, "NOTHEX")


# --- conditioning references ------------------------------------------------


def test_a_reference_is_written_before_the_row(svc):
    ids = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference",
        reference=_png_bytes(), guidance_fields={"ip_adapter": "plus"},
    )["ids"]
    job_dir = svc.job_dir(ids[0])
    assert (job_dir / "ref.png").exists()
    assert svc.store.get(ids[0])["params"]["ip_adapter"] == "plus"


def test_each_candidate_gets_its_own_reference(svc):
    ids = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference", count=3,
        reference=_png_bytes(), guidance_fields={"ip_adapter": "plus"},
    )["ids"]
    assert all((svc.job_dir(i) / "ref.png").exists() for i in ids)


def test_a_failed_insert_removes_the_reference_directory(svc, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db is down")

    monkeypatch.setattr(svc.store, "create", boom)
    with pytest.raises(RuntimeError):
        svc_jobs.create_job(
            svc, kind="text", prompt="a barrel", output="reference",
            reference=_png_bytes(),
        )
    # The other half of writing the dir before the row: no orphan directory.
    assert not [p for p in svc.config.data_dir.iterdir() if p.is_dir()]


def test_conditioning_without_a_reference_is_refused(svc):
    with pytest.raises(Invalid, match="reference"):
        svc_jobs.create_job(
            svc, kind="text", prompt="a barrel", output="reference",
            guidance_fields={"ip_adapter": "plus"},
        )


def test_only_a_text_job_takes_a_reference(svc):
    with pytest.raises(Invalid, match="text jobs"):
        svc_jobs.create_job(
            svc, kind="image", image=_png_bytes(), reference=_png_bytes()
        )


def test_an_oversized_reference_is_refused_before_decode(svc):
    with pytest.raises(TooLarge):
        svc_jobs.create_job(
            svc, kind="text", prompt="a barrel", output="reference",
            reference=b"\x00" * (20 * 1024 * 1024 + 1),
        )


def test_an_undecodable_reference_is_refused(svc):
    with pytest.raises(Invalid, match="reference"):
        svc_jobs.create_job(
            svc, kind="text", prompt="a barrel", output="reference",
            reference=b"not an image at all",
        )


def test_a_reroll_carries_the_reference_across(svc):
    """A reroll reruns SDXL, so the image its conditioning needs has to come
    with it -- the first case where a *text* rerun writes a directory before
    the row exists."""
    src = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference",
        reference=_png_bytes(), guidance_fields={"ip_adapter": "plus"},
    )["id"]
    svc.store.set_status(src, "done")

    new_id = svc_jobs.rerun_job(svc, src, mode="reroll")["id"]
    assert (svc.job_dir(new_id) / "ref.png").exists()
    assert svc.store.get(new_id)["params"]["ip_adapter"] == "plus"


def test_a_remesh_drops_the_conditioning_it_cannot_have_run(svc):
    src = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference",
        reference=_png_bytes(), guidance_fields={"ip_adapter": "plus"},
    )["id"]
    (svc.job_dir(src)).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(src) / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(src, "done")

    new_id = svc_jobs.rerun_job(svc, src, mode="remesh")["id"]
    params = svc.store.get(new_id)["params"]
    assert "ip_adapter" not in params
    assert "ip_scale" not in params
    assert not (svc.job_dir(new_id) / "ref.png").exists()


def test_a_promotion_drops_the_conditioning_and_copies_no_reference(svc):
    src = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference",
        reference=_png_bytes(), guidance_fields={"ip_adapter": "plus"},
    )["id"]
    (svc.job_dir(src) / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(src, "done")

    new_id = svc_jobs.promote_to_model(svc, src)["id"]
    params = svc.store.get(new_id)["params"]
    assert not any(k in params for k in ("ip_adapter", "ip_scale", "control"))
    assert not (svc.job_dir(new_id) / "ref.png").exists()


def test_a_promotion_refuses_a_reference_that_cannot_reconstruct(svc):
    """The mesh stage is where the GPU minutes are, so the check happens here
    rather than after."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="reference")["id"]
    (svc.job_dir(job_id)).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png_bytes())
    svc.store.merge_params(
        job_id,
        {"reference_report": {"ok": False, "reasons": ["There is more than one object."]}},
    )
    svc.store.set_status(job_id, "done")

    with pytest.raises(Invalid, match="more than one object"):
        svc_jobs.promote_to_model(svc, job_id)

    # Heuristics about composition, not facts -- so it is bypassable.
    assert svc_jobs.promote_to_model(svc, job_id, force=True)["parent"] == job_id


def test_a_rank_never_survives_into_a_new_job():
    assert "rank" in DERIVED_PARAMS


# --- tiles ------------------------------------------------------------------


def test_a_tile_job_is_accepted_and_lands_at_the_tile_stage(svc):
    out = svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile")
    assert svc.store.get(out["id"])["stage"] == "tile"


def test_only_text_jobs_can_be_tiles(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=_png_bytes(), output="tile")


def test_tiles_can_be_batched_like_references(svc):
    out = svc_jobs.create_job(
        svc, kind="text", prompt="cobblestone", output="tile", count=3
    )
    assert len(out["ids"]) == 3


def test_a_tile_on_a_non_sdxl_base_is_refused_at_the_door(svc):
    """Seamlessness is circular padding over Conv2d, and a DiT has none -- so
    patching what a Flux pipe does have (its VAE) would give back an image
    whose latent never wrapped: seamless in a thumbnail, seamed in a material.
    Refusing beats producing that."""
    with pytest.raises(Invalid, match="seamless tile"):
        svc_jobs.create_job(
            svc,
            kind="text",
            prompt="cobblestone",
            output="tile",
            guidance_fields={"base_model": "flux_klein"},
        )
    # The same base is fine for an ordinary reference -- the refusal is about
    # the pairing, not the checkpoint.
    assert svc_jobs.create_job(
        svc,
        kind="text",
        prompt="a barrel",
        output="reference",
        guidance_fields={"base_model": "flux_klein"},
    )["id"]


def test_a_tile_cannot_be_promoted_to_a_mesh(svc):
    # There is no subject to reconstruct. Refusing at the door beats two
    # minutes of trellis turning a texture into a lumpy plane.
    out = svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile")
    job_id = out["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid, match="no subject"):
        svc_jobs.promote_to_model(svc, job_id)


def test_a_tile_is_priced_as_the_image_stage_it_is(svc):
    # The admission check has to name the tile stage, not fall through to the
    # mesh branch: a card that can hold an image model but not trellis must
    # still be allowed to make a texture.
    _small_card(svc, total=10.0)
    assert svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile")["id"]
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="model")


def test_a_seam_report_never_survives_into_a_new_job():
    assert "seam_report" in DERIVED_PARAMS


def test_the_tile_flag_is_an_input_and_not_a_derived_value():
    # A reroll of a tile must stay a tile. The stage carries that, and the
    # top-level flag must never join the strip list -- only the copy that rides
    # inside params["recipe"], which is stripped with the rest of the recipe.
    assert "tile" not in DERIVED_PARAMS


def test_the_prompt_preview_mirrors_a_tile_rather_than_an_object(svc):
    from warlock.service import system as svc_system

    body = svc_system.prompt_preview(svc, {}, "cobblestone", tile=True)
    assert "seamless" in body["prompt"]
    assert "single object" not in body["prompt"]


def test_a_reroll_of_a_tile_stays_a_tile(svc):
    out = svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile")
    svc.store.set_status(out["id"], "done")
    new_id = svc_jobs.rerun_job(svc, out["id"], mode="reroll")["id"]
    assert svc.store.get(new_id)["stage"] == "tile"


def test_a_tile_cannot_be_remeshed_either(svc):
    # The other door onto a trellis run from an image that has no subject.
    # promote_to_model is the one the 3D pane offers; remesh is the one the
    # library's retry button reaches for whenever an input.png exists.
    out = svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile")
    job_id = out["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png_bytes())
    svc.store.set_status(job_id, "error")
    with pytest.raises(Invalid, match="no subject"):
        svc_jobs.rerun_job(svc, job_id, mode="remesh")


# --- the server-config axes -------------------------------------------------


def test_the_server_axes_land_in_params_only_when_asked_for(svc):
    pinned = svc_jobs.create_job(
        svc, kind="text", prompt="a chest", output="reference",
        trellis_band=8, trellis_tex_res=2048,
    )
    params = svc.store.get(pinned["id"])["params"]
    assert (params["trellis_band"], params["trellis_tex_res"]) == (8, 2048)

    plain = svc_jobs.create_job(svc, kind="text", prompt="a chest", output="reference")
    unset = svc.store.get(plain["id"])["params"]
    assert "trellis_band" not in unset and "trellis_tex_res" not in unset


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trellis_band": 0},
        {"trellis_band": 65},
        {"trellis_band": 8.0},
        {"trellis_tex_res": 64},
        {"trellis_tex_res": 8192},
        {"trellis_tex_res": True},
    ],
)
def test_an_out_of_range_server_axis_is_refused_before_any_write(svc, kwargs):
    with pytest.raises(Invalid):
        svc_jobs.create_job(
            svc, kind="image", image=_png_bytes(), output="model", **kwargs
        )
    assert list(svc.config.data_dir.glob("*/input.png")) == []


def test_the_server_axes_survive_a_reroll_because_they_are_inputs(svc):
    created = svc_jobs.create_job(
        svc, kind="text", prompt="a chest", output="reference", trellis_band=8
    )
    svc.store.set_status(created["id"], "done")
    rerun = svc_jobs.rerun_job(svc, created["id"], mode="reroll")
    assert svc.store.get(rerun["id"])["params"]["trellis_band"] == 8
    assert "trellis_band" not in DERIVED_PARAMS


# --- sweep membership -------------------------------------------------------


def test_sweep_membership_is_columns_and_never_leaks_onto_a_reroll(svc):
    sweep_id = svc.store.create_sweep("lora", "a chest", {})
    created = svc_jobs.create_job(
        svc, kind="text", prompt="a chest", output="reference",
        sweep_id=sweep_id, sweep_unit="lora_weight=0.6 s42",
    )
    row = svc.store.get(created["id"])
    assert row["sweep_id"] == sweep_id
    assert row["sweep_unit"] == "lora_weight=0.6 s42"
    # Not in params, so nothing that copies params can carry it.
    assert "sweep_id" not in row["params"]

    svc.store.set_status(created["id"], "done")
    rerun = svc.store.get(svc_jobs.rerun_job(svc, created["id"], mode="reroll")["id"])
    assert rerun["sweep_id"] is None
    assert rerun["sweep_unit"] == ""


def test_a_sweep_unit_is_a_single_job(svc):
    sweep_id = svc.store.create_sweep("lora", "a chest", {})
    with pytest.raises(Invalid):
        svc_jobs.create_job(
            svc, kind="text", prompt="a chest", output="reference",
            count=3, sweep_id=sweep_id,
        )

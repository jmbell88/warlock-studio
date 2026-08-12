"""A ``done`` job may be degraded, and it has to say so.

ART-01. Normalization (requested size, X/Z centring, floor grounding) and the
mesh report are wrapped in catch-everything handlers, and that decision is
right: the reconstruction is on disk and usable, so a mesh trimesh cannot parse
must not cost the user the asset. What was wrong is that the job then stayed
``done`` with the only evidence in a log line -- so a visibly successful asset
could be exported with the engine's pivot and scale, and nothing anywhere
explained why it landed sideways in a game project.

The fix is a structured health record on the job's own row, which means it must
also be *derived*: a reroll that inherited it would wear a degradation belonging
to the run it replaced.
"""

from __future__ import annotations

import pytest

from warlock import _q_mesh
from warlock.service import export as svc_export
from warlock.service.validation import ARTIFACT_HEALTH, DERIVED_PARAMS, note_degraded


def test_the_two_spellings_of_the_health_key_agree():
    """Two copies rather than one import: the queue layer may not import
    ``service`` and the service layer must not reach into a queue private --
    the same shape as ``VECTOR_PARAMS`` living in ``vectors.py``. So the pair
    is pinned here instead."""
    assert _q_mesh.ARTIFACT_HEALTH == ARTIFACT_HEALTH


def test_health_is_derived_so_a_reroll_starts_clean():
    """Without this, a reroll of a degraded job inherits the note and claims a
    failure it never had -- the exact trap ``DERIVED_PARAMS`` exists for, and
    the one CLAUDE.md calls out for any new worker-recorded param."""
    assert ARTIFACT_HEALTH in DERIVED_PARAMS


@pytest.mark.parametrize("note", [_q_mesh._note_degraded, note_degraded])
def test_steps_accumulate_rather_than_overwrite(note):
    """A mesh can fail more than one step, and the second failure is not a
    correction of the first."""
    params: dict = {}
    note(params, "normalize", "pivot is the engine's")
    note(params, "report", "not measured")
    assert set(params[ARTIFACT_HEALTH]) == {"normalize", "report"}
    # Re-noting one step replaces that step's detail and leaves the other.
    note(params, "normalize", "still not grounded")
    assert params[ARTIFACT_HEALTH]["normalize"] == "still not grounded"
    assert params[ARTIFACT_HEALTH]["report"] == "not measured"


@pytest.mark.parametrize("note", [_q_mesh._note_degraded, note_degraded])
def test_a_hand_edited_row_does_not_break_the_recorder(note):
    """``params`` is a JSON blob a user can edit; a non-dict under this key must
    not raise inside a handler whose whole purpose is not to raise."""
    params: dict = {ARTIFACT_HEALTH: "broken"}
    note(params, "normalize", "pivot is the engine's")
    assert params[ARTIFACT_HEALTH] == {"normalize": "pivot is the engine's"}


def test_an_export_names_the_degraded_assets_it_shipped(svc, tmp_path, monkeypatch):
    """The moment it stops being an internal detail: the file is on its way into
    a game project, where a wrong pivot is a manual fixup on every import."""
    good = svc.store.create("image", "a crate", {}, "aaaaaaaaaaaa", stage="model")
    bad_params: dict = {}
    note_degraded(bad_params, "normalize", "the mesh was not centred or grounded")
    bad = svc.store.create("image", "a barrel", bad_params, "bbbbbbbbbbbb", stage="model")
    for job_id in (good, bad):
        svc.store.set_status(job_id, "done")
        (svc.job_dir(job_id)).mkdir(parents=True, exist_ok=True)
        (svc.job_dir(job_id) / "model.glb").write_bytes(b"glb")

    assert svc_export.degraded_ids(svc, [good, bad]) == [bad]

    out = tmp_path / "project"
    monkeypatch.setattr(svc.config, "export_dir", out)
    result = svc_export.export_to_folder(svc, [good, bad], ["model.glb"])
    assert result["copied"] == 2
    assert result["degraded"] == [bad]


def test_an_export_never_truncates_the_file_it_is_replacing(svc, tmp_path, monkeypatch):
    """SVC-06: ``WARLOCK_EXPORT_DIR`` exists to be *watched* -- it is a game
    project's assets folder -- and ``copyfile`` truncates its target before
    writing a byte, so a hot-reloading engine could read a torn GLB."""
    job_id = svc.store.create("image", "a crate", {}, "cccccccccccc", stage="model")
    svc.store.set_status(job_id, "done")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "model.glb").write_bytes(b"new-bytes")

    out = tmp_path / "project"
    dest = out / job_id / "model.glb"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"the-previous-export")
    monkeypatch.setattr(svc.config, "export_dir", out)

    seen: list[bytes] = []
    real_copyfile = svc_export.shutil.copyfile

    def _watched(src, dst, **kw):
        # Whatever a watcher would see *while* the copy is in progress: with a
        # staged copy the served name still holds the old bytes intact.
        seen.append(dest.read_bytes())
        return real_copyfile(src, dst, **kw)

    monkeypatch.setattr(svc_export.shutil, "copyfile", _watched)
    svc_export.export_to_folder(svc, [job_id], ["model.glb"])

    assert seen == [b"the-previous-export"], "the destination was truncated mid-export"
    assert dest.read_bytes() == b"new-bytes"
    assert not list(out.glob("**/.*.tmp")), "the staging file was left behind"

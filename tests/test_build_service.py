"""A built asset is an ordinary asset: ``jobs.import_mesh``.

The payoff is disproportionate to the size of the function, and that is the
thing worth pinning. Everything downstream of a mesh -- rigging, posing, sprite
sheets, the triangle retarget, STL/OBJ/FBX/collision export -- is a pure
function of ``model.glb``, so a row minted here inherits all of it without any
of those paths learning that Build mode exists. What the tests below check is
that the row really is the same shape a worker run produces, and that the two
doors which would take a built asset somewhere meaningless are shut.
"""

from __future__ import annotations

import io

import pytest
import trimesh

from warlock.service import files, jobs
from warlock.service.errors import Invalid, TooLarge
from warlock.studio.build import document as bd
from warlock.studio.build import primitives as bp
from warlock.studio.viewer import glbwrite


def _glb(mesh=None, translation=(0.0, 0.0, 0.0)) -> bytes:
    """Authored bytes, straight out of the writer Build mode uses."""
    doc = bd.BuildDoc()
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box() if mesh is None else mesh,
            translation=translation,
        )
    )
    return glbwrite.write_glb(bd.to_model(doc))


def _bounds(path):
    loaded = trimesh.load(path, process=False)
    return loaded.bounds


# --- the row -----------------------------------------------------------------


def test_import_mesh_mints_a_finished_model_row(svc) -> None:
    out = jobs.import_mesh(svc, _glb(), prompt="a crate")
    job = svc.store.get(out["id"])

    assert job["status"] == "done"
    assert job["stage"] == "model"
    assert job["prompt"] == "a crate"
    job_dir = svc.job_dir(out["id"])
    assert (job_dir / "source.glb").exists()
    assert (job_dir / "model.glb").exists()
    assert job["params"]["mesh_report"]["status"] in ("ready", "review", "invalid")


def test_a_built_asset_says_so_in_its_params(svc) -> None:
    out = jobs.import_mesh(svc, _glb())
    assert svc.store.get(out["id"])["params"]["built"] is True


def test_source_glb_is_the_authored_bytes_and_model_glb_derives_from_it(svc) -> None:
    """The existing invariant, and what keeps ``optimize_job``'s retarget
    working on a built asset: nothing ever overwrites ``source.glb``."""
    data = _glb()
    out = jobs.import_mesh(svc, data)
    job_dir = svc.job_dir(out["id"])
    assert (job_dir / "source.glb").read_bytes() == data
    assert (job_dir / "model.glb").read_bytes() != data  # grounded


def test_a_name_is_recorded_when_one_is_given(svc) -> None:
    out = jobs.import_mesh(svc, _glb(), name="Crate")
    assert svc.store.get(out["id"])["name"] == "Crate"


# --- grounding ---------------------------------------------------------------


def test_an_imported_mesh_is_grounded_like_every_other_one(svc) -> None:
    """Grounding is not conditional on a size: a pivot at the mesh's own centre
    is a manual fixup on every Godot or Unity import, and a built asset is no
    more exempt than a reconstructed one."""
    out = jobs.import_mesh(svc, _glb(translation=(3.0, 7.0, -2.0)))
    lo, hi = _bounds(svc.job_dir(out["id"]) / "model.glb")

    assert lo[1] == pytest.approx(0.0, abs=1e-5)
    assert (lo[0] + hi[0]) == pytest.approx(0.0, abs=1e-5)
    assert (lo[2] + hi[2]) == pytest.approx(0.0, abs=1e-5)


def test_a_size_is_applied_when_one_is_asked_for(svc) -> None:
    out = jobs.import_mesh(svc, _glb(), size_m=4.0)
    lo, hi = _bounds(svc.job_dir(out["id"]) / "model.glb")
    assert max(hi - lo) == pytest.approx(4.0, rel=1e-3)


def test_a_normalize_failure_is_swallowed_rather_than_failing_the_job(svc, monkeypatch) -> None:
    """The GLB is already on disk by then. The same rule ``_audit_mesh`` and the
    mesh report follow: a job that produced a mesh does not fail afterwards."""
    from warlock.pipelines import postprocess

    monkeypatch.setattr(
        postprocess, "normalize_glb", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no"))
    )
    out = jobs.import_mesh(svc, _glb())
    job = svc.store.get(out["id"])
    assert job["status"] == "done"
    assert (svc.job_dir(out["id"]) / "model.glb").exists()


def test_a_mesh_report_failure_is_swallowed_too(svc, monkeypatch) -> None:
    from warlock import meshreport

    monkeypatch.setattr(
        meshreport, "build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no"))
    )
    out = jobs.import_mesh(svc, _glb())
    job = svc.store.get(out["id"])
    assert job["status"] == "done"
    assert "mesh_report" not in job["params"]


# --- the doors that must stay shut -------------------------------------------


@pytest.mark.parametrize("mode", ["reroll", "remesh"])
def test_a_built_asset_cannot_be_rerolled_or_remeshed(svc, mode: str) -> None:
    """``rerun_job`` already refuses a job with no ``input.png`` -- but with the
    reference message, which tells the user to go and hand-edit an image that
    does not exist. A built asset needs its own reason."""
    out = jobs.import_mesh(svc, _glb())
    with pytest.raises(Invalid, match="built"):
        jobs.rerun_job(svc, out["id"], mode=mode)


def test_promote_to_model_still_refuses_and_needed_no_change(svc) -> None:
    """It already requires ``stage == "reference"``, and a built asset is at
    stage ``model``. Asserted rather than assumed, because the refusal is what
    stops a finished mesh being fed to trellis as though it were a picture."""
    out = jobs.import_mesh(svc, _glb())
    with pytest.raises(Invalid):
        jobs.promote_to_model(svc, out["id"])


# --- the door itself ---------------------------------------------------------


def test_oversize_bytes_are_refused_before_anything_is_written(svc) -> None:
    from warlock.service import validation

    before = set(svc.config.data_dir.iterdir()) if svc.config.data_dir.exists() else set()
    with pytest.raises(TooLarge):
        jobs.import_mesh(svc, b"glTF" + b"\x00" * (validation.MAX_MESH_BYTES + 1))
    after = set(svc.config.data_dir.iterdir()) if svc.config.data_dir.exists() else set()
    assert after == before


def test_bytes_that_are_not_a_glb_are_refused(svc) -> None:
    with pytest.raises(Invalid, match="glTF"):
        jobs.import_mesh(svc, b"this is not a glb at all, not even close")


def test_a_glb_with_no_mesh_in_it_is_refused(svc) -> None:
    """Belt and braces -- we author these bytes -- but "inputs are bounded at
    the door" is the invariant, and this is the door. A GLB of nothing would
    mint a done row whose every downstream export produced an empty file."""
    from warlock.studio.viewer import gltf

    empty = glbwrite.write_glb(gltf.Model([gltf.Node(name="nothing")], [0], [], []))
    with pytest.raises(Invalid, match="no mesh"):
        jobs.import_mesh(svc, empty)


def test_a_prompt_over_the_limit_is_refused(svc) -> None:
    from warlock.service.validation import MAX_PROMPT

    with pytest.raises(Invalid, match="prompt"):
        jobs.import_mesh(svc, _glb(), prompt="x" * (MAX_PROMPT + 1))


def test_a_failed_insert_leaves_no_job_directory(svc, monkeypatch) -> None:
    """The files are written before the row exists, deliberately, so this
    cleanup is the other half of that ordering -- exactly as ``create_job``
    does with ``input.png``."""
    monkeypatch.setattr(
        svc.store, "create", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("insert failed"))
    )
    before = set(svc.config.data_dir.iterdir()) if svc.config.data_dir.exists() else set()
    with pytest.raises(RuntimeError):
        jobs.import_mesh(svc, _glb())
    after = set(svc.config.data_dir.iterdir()) if svc.config.data_dir.exists() else set()
    assert after == before


# --- the .wblk sidecar -------------------------------------------------------


def test_the_build_source_is_stored_beside_the_mesh(svc) -> None:
    out = jobs.import_mesh(svc, _glb())
    assert files.build_source_status(svc, out["id"]) == {"exists": False}

    files.save_build_source(svc, out["id"], b"PK\x03\x04" + b"stand-in wblk bytes")
    assert files.build_source_status(svc, out["id"]) == {"exists": True}
    assert (svc.job_dir(out["id"]) / files.BUILD_SOURCE).exists()


def test_the_build_source_is_never_served(svc) -> None:
    """The ``paint.ora`` precedent: internal working state, absent from MEDIA
    and LISTED, and it goes away with the job directory for free."""
    assert files.BUILD_SOURCE not in files.MEDIA
    assert files.BUILD_SOURCE not in files.LISTED


def test_a_build_source_that_is_not_a_zip_is_refused(svc) -> None:
    out = jobs.import_mesh(svc, _glb())
    with pytest.raises(Invalid):
        files.save_build_source(svc, out["id"], b"not a zip")


def test_an_oversize_build_source_is_refused(svc) -> None:
    out = jobs.import_mesh(svc, _glb())
    with pytest.raises(TooLarge):
        files.save_build_source(
            svc, out["id"], b"PK\x03\x04" + b"\x00" * files.MAX_BUILD_SOURCE_BYTES
        )


def test_the_sidecar_has_no_staleness_rule(svc) -> None:
    """Unlike ``paint.ora``. That rule exists because a revert can rewrite
    ``input.png`` behind the layers; a build export writes the GLB and the
    ``.wblk`` in one operation, and a later retarget does not make the authored
    source wrong -- it makes ``model.glb`` differ from it, which is the point."""
    out = jobs.import_mesh(svc, _glb())
    files.save_build_source(svc, out["id"], b"PK\x03\x04" + b"authored")
    # Touch model.glb so it is unambiguously newer than the sidecar.
    (svc.job_dir(out["id"]) / "model.glb").write_bytes(_glb())
    assert files.build_source_status(svc, out["id"])["exists"] is True
    assert "fresh" not in files.build_source_status(svc, out["id"])


def test_reading_back_a_stored_build_source_gives_the_same_bytes(svc) -> None:
    out = jobs.import_mesh(svc, _glb())
    data = io.BytesIO(b"PK\x03\x04zipish").getvalue()
    files.save_build_source(svc, out["id"], data)
    assert (svc.job_dir(out["id"]) / files.BUILD_SOURCE).read_bytes() == data

"""Saving and reverting a hand-edited reference.

The interesting property is not "the bytes landed" -- it is that the *other*
readers of input.png stay honest afterwards: promote_to_model's quality gate
was measured from the generated pixels, and an edit makes that a verdict about
an image that no longer exists.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from warlock.service import files as svc_files
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Conflict, Invalid, NotFound, TooLarge


def _png(size=(64, 64), colour=(200, 30, 30, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, colour).save(buf, "PNG")
    return buf.getvalue()


def _reference(svc, **params) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference", **params)["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png())
    svc.store.set_status(job_id, "done")
    return job_id


# --- the gates --------------------------------------------------------------


def test_only_a_finished_reference_may_be_edited(svc):
    job_id = _reference(svc)
    svc.store.set_status(job_id, "running")
    with pytest.raises(Invalid):
        svc_files.save_edited_image(svc, job_id, _png())


def test_a_mesh_job_is_not_an_editable_reference(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid):
        svc_files.save_edited_image(svc, job_id, _png())


def test_a_bad_job_id_never_reaches_the_filesystem(svc):
    with pytest.raises(NotFound):
        svc_files.save_edited_image(svc, "../../etc", _png())
    with pytest.raises(NotFound):
        svc_files.save_edited_image(svc, "0" * 12, _png())


def test_something_that_is_not_a_png_is_refused(svc):
    job_id = _reference(svc)
    with pytest.raises(Invalid):
        svc_files.save_edited_image(svc, job_id, b"GIF89a not really")


def test_an_oversized_body_is_refused_before_it_is_decoded(svc):
    job_id = _reference(svc)
    payload = svc_files.PNG_MAGIC + b"\0" * (24 * 1024 * 1024)
    with pytest.raises(TooLarge):
        svc_files.save_edited_image(svc, job_id, payload)


def test_the_pixel_cap_is_read_from_the_header(svc):
    """A flat 20 MP PNG is a few hundred KB on disk and enormous decoded."""
    job_id = _reference(svc)
    big = _png((5000, 5000), (0, 0, 0, 255))
    assert len(big) < svc_files.MAX_UPLOAD_BYTES
    with pytest.raises(TooLarge):
        svc_files.save_edited_image(svc, job_id, big)


# --- the backup -------------------------------------------------------------


def test_the_first_save_keeps_the_generated_image_and_later_ones_do_not(svc):
    job_id = _reference(svc)
    original = (svc.job_dir(job_id) / "input.png").read_bytes()

    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    backup = svc.job_dir(job_id) / svc_files.ORIGINAL
    assert backup.read_bytes() == original

    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 0, 255, 255)))
    assert backup.read_bytes() == original


def test_the_backup_is_never_listed_or_downloadable(svc):
    """It is an internal undo, not an artifact: listing it would offer a
    download of a file the user was never told about."""
    job_id = _reference(svc)
    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    job = svc.store.get(job_id)
    svc_files.attach_files(job, svc.job_dir(job_id))
    assert svc_files.ORIGINAL not in job["files"]
    assert svc_files.ORIGINAL not in svc_files.MEDIA


def test_a_revert_restores_the_bytes_and_consumes_the_backup(svc):
    job_id = _reference(svc)
    original = (svc.job_dir(job_id) / "input.png").read_bytes()
    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    svc_files.revert_reference(svc, job_id)
    assert (svc.job_dir(job_id) / "input.png").read_bytes() == original
    assert not (svc.job_dir(job_id) / svc_files.ORIGINAL).exists()


def test_a_revert_with_nothing_to_revert_to_is_a_conflict(svc):
    job_id = _reference(svc)
    with pytest.raises(Conflict):
        svc_files.revert_reference(svc, job_id)


def test_the_edit_status_probe_reports_both_answers(svc):
    job_id = _reference(svc)
    assert svc_files.reference_edit_status(svc, job_id) == {
        "editable": True,
        "has_original": False,
    }
    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    assert svc_files.reference_edit_status(svc, job_id)["has_original"] is True


def test_the_edit_status_probe_says_no_rather_than_raising_for_an_unknown_job(svc):
    assert svc_files.reference_edit_status(svc, "0" * 12)["editable"] is False


# --- the quality gate -------------------------------------------------------


def test_a_save_re_measures_the_reference_and_marks_it_hand_edited(svc):
    job_id = _reference(svc)
    svc.store.merge_params(job_id, {"reference_report": {"ok": False, "reasons": ["stale"]}})
    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    params = svc.store.get(job_id)["params"]
    assert params["hand_edited"] is True
    assert params["reference_report"]["reasons"] != ["stale"]


def test_a_revert_re_measures_and_drops_the_hand_edited_flag(svc):
    job_id = _reference(svc)
    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    svc_files.revert_reference(svc, job_id)
    params = svc.store.get(job_id)["params"]
    assert "hand_edited" not in params
    assert "reference_report" in params


def test_promotion_after_an_edit_carries_the_edited_pixels(svc):
    job_id = _reference(svc)
    edited = _png(colour=(0, 255, 0, 255))
    svc_files.save_edited_image(svc, job_id, edited)
    # force, because a flat test swatch fails the composition heuristics --
    # which is itself the point of the test above this one.
    out = svc_jobs.promote_to_model(svc, job_id, force=True)
    assert (svc.job_dir(out["id"]) / "input.png").read_bytes() == edited


def test_promotion_is_judged_on_the_edited_pixels_and_force_still_wins(svc, monkeypatch):
    """The gate used to read a report measured from the generated image, which
    after an edit is a verdict about pixels nobody will ever see again."""
    from warlock.pipelines import reference

    job_id = _reference(svc)

    def refuse(_src):
        return reference.unmeasured("nope")

    monkeypatch.setattr(reference, "measure_file", refuse)
    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    report = svc.store.get(job_id)["params"]["reference_report"]
    if report.get("ok") is False:
        with pytest.raises(Invalid):
            svc_jobs.promote_to_model(svc, job_id)
    assert svc_jobs.promote_to_model(svc, job_id, force=True)["parent"] == job_id


def test_a_save_leaves_no_temp_file_behind(svc):
    job_id = _reference(svc)
    svc_files.save_edited_image(svc, job_id, _png(colour=(0, 255, 0, 255)))
    names = {p.name for p in svc.job_dir(job_id).iterdir()}
    assert not [n for n in names if n.endswith(".tmp")]


# --- the layered working file -----------------------------------------------
#
# paint.ora sits beside input.png so that reopening an edited reference brings
# its layers back rather than a flattened image. It is internal working state,
# never served, and it is *stale* rather than authoritative the moment anything
# else rewrites the reference.


def _ora(size=(64, 64), layers=2) -> bytes:
    from warlock.studio import inker

    doc = inker.Document.blank(*size)
    doc.stack[0].pixels[:, :] = (200, 30, 30, 255)
    for _ in range(layers - 1):
        doc.add_layer()
    return inker.ora_bytes(doc)


def test_the_layered_source_saves_beside_the_reference_it_flattens_to(svc):
    job_id = _reference(svc)
    svc_files.save_paint_working(svc, job_id, _ora())
    assert svc_files.paint_working_path(svc, job_id).exists()
    # Internal: never listed, never downloadable.
    assert svc_files.PAINT_WORKING not in svc_files.LISTED
    assert svc_files.PAINT_WORKING not in svc_files.MEDIA


def test_a_layered_source_must_actually_be_one(svc):
    job_id = _reference(svc)
    with pytest.raises(Invalid):
        svc_files.save_paint_working(svc, job_id, _png())


def test_a_layered_source_is_bounded(svc):
    job_id = _reference(svc)
    with pytest.raises(TooLarge):
        svc_files.save_paint_working(svc, job_id, b"PK\x03\x04" + b"0" * svc_files.MAX_PAINT_BYTES)


def test_only_a_finished_reference_gets_a_layered_source(svc):
    job_id = _reference(svc)
    svc.store.set_status(job_id, "running")
    with pytest.raises(Invalid):
        svc_files.save_paint_working(svc, job_id, _ora())


def test_a_working_file_older_than_the_reference_is_treated_as_stale(svc):
    """A revert or a regenerate rewrites input.png without touching the
    layers, which would otherwise resurrect an edit of an image that is gone."""
    import os
    import time

    job_id = _reference(svc)
    svc_files.save_paint_working(svc, job_id, _ora())
    assert svc_files.paint_working_status(svc, job_id) == {"exists": True, "fresh": True}

    later = time.time() + 10
    os.utime(svc.job_dir(job_id) / "input.png", (later, later))
    status = svc_files.paint_working_status(svc, job_id)
    assert status["exists"] and not status["fresh"]


def test_no_working_file_is_neither_present_nor_fresh(svc):
    assert svc_files.paint_working_status(svc, _reference(svc)) == {
        "exists": False,
        "fresh": False,
    }


def test_discarding_the_working_file_is_forgiving_of_its_absence(svc):
    job_id = _reference(svc)
    svc_files.discard_paint_working(svc, job_id)
    svc_files.save_paint_working(svc, job_id, _ora())
    svc_files.discard_paint_working(svc, job_id)
    assert not svc_files.paint_working_path(svc, job_id).exists()


def test_a_working_save_leaves_no_temp_file_behind(svc):
    job_id = _reference(svc)
    svc_files.save_paint_working(svc, job_id, _ora())
    names = {p.name for p in svc.job_dir(job_id).iterdir()}
    assert not [n for n in names if n.endswith(".tmp")]


# --- importing painted pixels -----------------------------------------------


def test_an_imported_reference_is_finished_the_moment_it_exists(svc):
    """No worker run: the image already exists, so queueing one would spend two
    minutes of GPU reproducing what the user just drew."""
    job_id = svc_jobs.import_reference(svc, _png())["id"]
    job = svc.store.get(job_id)
    assert job["status"] == "done"
    assert job["stage"] == "reference"
    assert (svc.job_dir(job_id) / "input.png").exists()


def test_an_imported_reference_is_measured_so_the_quality_gate_has_data(svc):
    job_id = svc_jobs.import_reference(svc, _png())["id"]
    params = svc.store.get(job_id)["params"]
    assert "reference_report" in params
    assert params["hand_edited"] is True
    assert params["imported"] is True


def test_an_imported_reference_can_be_promoted_like_a_generated_one(svc):
    job_id = svc_jobs.import_reference(svc, _png())["id"]
    out = svc_jobs.promote_to_model(svc, job_id, force=True)
    assert out["parent"] == job_id
    assert (svc.job_dir(out["id"]) / "input.png").exists()


def test_an_import_carries_a_name_when_one_is_given(svc):
    job_id = svc_jobs.import_reference(svc, _png(), name="barrel")["id"]
    assert svc.store.get(job_id)["name"] == "barrel"


def test_an_import_re_encodes_whatever_it_was_given(svc):
    """to_png, same as an upload: trellis.cpp only decodes PNG and JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "BMP")
    job_id = svc_jobs.import_reference(svc, buf.getvalue())["id"]
    assert (svc.job_dir(job_id) / "input.png").read_bytes().startswith(svc_files.PNG_MAGIC)


def test_an_oversized_import_is_refused_and_leaves_nothing_behind(svc):
    def listing() -> set[str]:
        root = svc.config.data_dir
        return {p.name for p in root.iterdir()} if root.exists() else set()

    before = listing()
    with pytest.raises(TooLarge):
        svc_jobs.import_reference(svc, b"0" * (svc_files.MAX_UPLOAD_BYTES + 1))
    assert listing() == before


def test_an_undecodable_import_is_refused(svc):
    with pytest.raises(Invalid):
        svc_jobs.import_reference(svc, b"not an image")

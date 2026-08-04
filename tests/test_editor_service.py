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

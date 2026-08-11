"""Rerunning a job, and what a rerun is allowed to inherit.

The rules here are about *provenance*: a new row must not claim something that
was true of the row it came from. Two of them were wrong in opposite
directions -- one combination the service refuses at the front door could be
minted out the back, and one flag that genuinely should survive a remesh was
being asked to die with the derived values.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from warlock.service import files as svc_files
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Invalid


def _png(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, (200, 30, 30, 255)).save(buf, "PNG")
    return buf.getvalue()


def _reference(svc, **params) -> str:
    job_id = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", output="reference", **params
    )["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png())
    svc.store.set_status(job_id, "done")
    return job_id


# --- the combination create_job refuses ---------------------------------------


def test_create_job_refuses_an_image_job_that_stops_at_a_reference(svc):
    """The rule a reroll was going around."""
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=_png(), output="reference")


def test_rerolling_a_hand_made_reference_is_refused(svc):
    """An imported or painted reference has no generator behind it, so a new
    seed changes nothing -- and the row it minted (kind=image, stage=reference)
    fell past the worker's reference early-return into a full trellis run."""
    result = svc_jobs.import_reference(svc, _png(), name="drawn")
    with pytest.raises(Invalid) as exc:
        svc_jobs.rerun_job(svc, result["id"], mode="reroll")
    assert "remesh" in str(exc.value)


def test_a_hand_made_reference_can_still_be_remeshed(svc):
    """The useful half: it is exactly the pixels a mesh should be built from."""
    result = svc_jobs.import_reference(svc, _png(), name="drawn")
    new = svc_jobs.rerun_job(svc, result["id"], mode="remesh")
    row = svc.store.get(new["id"])
    assert row["kind"] == "image"
    assert row["stage"] == "model"


@pytest.mark.parametrize("mode", ["reroll", "remesh"])
def test_no_rerun_can_mint_an_image_job_at_the_reference_stage(svc, mode):
    """Stated as the invariant rather than as the two cases above.

    Both modes are expected to *succeed* on a text reference, so the rows are
    asserted directly rather than swallowing ``Invalid`` -- a ``rerun_job``
    that refused everything would satisfy a skip-on-refusal loop having checked
    nothing at all.
    """
    text_ref = _reference(svc)
    new = svc_jobs.rerun_job(svc, text_ref, mode=mode)
    row = svc.store.get(new["id"])
    assert not (row["kind"] == "image" and row["stage"] == "reference")


def test_a_text_reference_still_rerolls_to_a_reference(svc):
    """The case the stage-carrying rule exists for is unchanged."""
    job_id = _reference(svc)
    new = svc_jobs.rerun_job(svc, job_id, mode="reroll")
    row = svc.store.get(new["id"])
    assert row["kind"] == "text"
    assert row["stage"] == "reference"


# --- hand_edited follows the pixels, not the run ------------------------------


def test_a_text_reroll_drops_the_hand_edited_flag(svc):
    """It regenerates input.png from the prompt, so the flag would be a claim
    about pixels nobody has touched."""
    job_id = _reference(svc)
    svc_files.save_edited_image(svc, job_id, _png())
    assert svc.store.get(job_id)["params"]["hand_edited"] is True

    new = svc_jobs.rerun_job(svc, job_id, mode="reroll")
    assert "hand_edited" not in svc.store.get(new["id"])["params"]


def test_a_remesh_keeps_the_hand_edited_flag(svc):
    """A remesh *copies* input.png, so the edit is still there -- which is why
    this is not simply a derived param."""
    job_id = _reference(svc)
    svc_files.save_edited_image(svc, job_id, _png())

    new = svc_jobs.rerun_job(svc, job_id, mode="remesh")
    assert svc.store.get(new["id"])["params"]["hand_edited"] is True


def test_a_promotion_keeps_the_hand_edited_flag(svc):
    job_id = _reference(svc)
    svc_files.save_edited_image(svc, job_id, _png())

    # force: the fixture is a flat rectangle, which the composition heuristics
    # rightly refuse. What is under test here is what the child inherits.
    child = svc_jobs.promote_to_model(svc, job_id, force=True)
    assert svc.store.get(child["id"])["params"]["hand_edited"] is True


def test_hand_edited_is_not_in_the_derived_list(svc):
    """Deliberately: DERIVED_PARAMS is stripped by promote and remesh too, and
    both of them carry the edited file across."""
    from warlock.service.validation import DERIVED_PARAMS

    assert "hand_edited" not in DERIVED_PARAMS


# --- admission, at the back door as well as the front -------------------------


def test_a_reroll_whose_style_lora_has_gone_missing_is_refused(svc, monkeypatch):
    """The rerun door checked VRAM and not weights.

    A style LoRA is the selection that fails *silently* at load -- the loader
    skips a missing adapter -- so without this the reroll queues, runs, finishes
    looking nothing like the row it rerolled, and writes params claiming a style
    that never ran. ``style_lora`` is in VECTOR_PARAMS, so that row is then
    evidence in the findings corpus about a style it never wore.
    """
    from warlock import fetch

    job_id = _reference(svc, guidance_fields={"style_lora": "ps1"})

    # Admitted when it was submitted; the file has gone since.
    monkeypatch.setattr(fetch, "present", lambda *a, **k: False)
    with pytest.raises(Invalid) as exc:
        svc_jobs.rerun_job(svc, job_id, mode="reroll")
    assert exc.value.field == "style_lora"
    assert "hf download" in str(exc.value)


def test_a_reroll_whose_checkpoint_has_gone_missing_is_refused(svc, monkeypatch):
    """The loud half of the same door: a missing checkpoint would reach the
    worker as a diffusers traceback naming a directory, two minutes and a
    queue place later."""
    from warlock import fetch

    job_id = _reference(svc)
    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (False, None))
    with pytest.raises(Invalid) as exc:
        svc_jobs.rerun_job(svc, job_id, mode="reroll")
    assert exc.value.field == "base_model"


def test_a_remesh_is_not_refused_for_image_model_weights(svc, monkeypatch):
    """The check is unconditional, but check_weights is text-only by design: a
    remesh reruns trellis and loads no image model at all, so a pruned SDXL is
    none of its business."""
    from warlock import fetch

    job_id = _reference(svc)
    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (False, None))
    monkeypatch.setattr(fetch, "present", lambda *a, **k: False)

    new = svc_jobs.rerun_job(svc, job_id, mode="remesh")
    assert svc.store.get(new["id"])["kind"] == "image"

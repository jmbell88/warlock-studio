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


# --- the restyle kinds: what a reroll may and may not inherit -----------------


def _sprite_job(svc) -> str:
    """A finished ``sprite_synthesis`` row carrying the params its door writes."""
    from warlock import rigging
    from warlock.service import sprites as svc_sprites

    ref = _reference(svc)
    params = {
        "source_job": ref,
        "sheet_type": "turnaround",
        "logical_size": 64,
        "colors": 32,
        "seed_a": 11,
        "seed_b": 12,
        "draft_id": rigging.new_id(),
        "base_model": svc_sprites.SPRITE_BASE_MODEL,
    }
    job_id = svc.store.create("sprite_synthesis", "a barrel", params)
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_sprite_reroll_mints_a_fresh_draft_id(svc):
    """``draft_id`` names the trio this job publishes into the *source
    reference's* directory, and it is minted at the door rather than recorded
    by the worker -- so DERIVED_PARAMS never stripped it. Copied verbatim, a
    cancelled reroll made ``_discard_artifacts`` delete the original job's
    published trio, and a finished one silently overwrote it."""
    from warlock import rigging

    job_id = _sprite_job(svc)
    old = svc.store.get(job_id)["params"]["draft_id"]
    new = svc_jobs.rerun_job(svc, job_id, mode="reroll")
    minted = svc.store.get(new["id"])["params"]["draft_id"]
    assert rigging.is_valid_id(minted)
    assert minted != old


def test_a_sprite_reroll_rerolls_the_pair_the_worker_samples_from(svc):
    """The worker reads ``seed_a``/``seed_b``; the generic ``seed`` is never
    read by this kind, so copying the pair verbatim made "give me another"
    reproduce a byte-identical sheet."""
    job_id = _sprite_job(svc)
    new = svc_jobs.rerun_job(svc, job_id, mode="reroll", seed=999)
    params = svc.store.get(new["id"])["params"]
    assert params["seed_a"] == 999, "a pinned seed lands on the first candidate"
    assert params["seed_b"] != 999, "the pair must stay distinct"
    assert (params["seed_a"], params["seed_b"]) != (11, 12)


def test_a_sprite_reroll_readmits_the_weights(svc, monkeypatch):
    """``check_weights`` is text-only, and both adapters plus the pixel LoRA
    are mandatory for this kind: pruned since the original ran, the reroll must
    be refused at the door rather than dispatched into a runtime failure --
    the door-side check is ``sprites._check_weights``, held again here."""
    from warlock import fetch

    job_id = _sprite_job(svc)
    monkeypatch.setattr(fetch, "present", lambda *a, **k: False)
    with pytest.raises(Invalid) as exc:
        svc_jobs.rerun_job(svc, job_id, mode="reroll")
    assert exc.value.field in ("ip_adapter", "control", "style_lora")


def _pixel_sheet_job(svc) -> str:
    """A finished ``pixel_sheet`` row carrying the params its door writes."""
    from warlock import rigging

    render = _reference(svc)  # stands in for the sheet render's source job
    params = {
        "source_job": render,
        "sheet_id": rigging.new_id(),
        "logical_size": 32,
        "colors": 32,
        "strength": 0.5,
        "structure_lock": True,
        "seed": 7,
        "base_model": "sdxl_cfg",
    }
    job_id = svc.store.create("pixel_sheet", "a barrel", params)
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_pixel_sheet_reroll_keeps_its_sheet_id(svc):
    """``sheet_id`` is on DERIVED_PARAMS for the *sheet* render kind, where the
    worker records it about its artifact; on this kind it is the input naming
    which sheet the restyle depicts. Stripped, every reroll of a pixel sheet
    dispatched straight into ``sheet_id is not a sheet id: ''``."""
    from warlock import rigging

    job_id = _pixel_sheet_job(svc)
    wanted = svc.store.get(job_id)["params"]["sheet_id"]
    new = svc_jobs.rerun_job(svc, job_id, mode="reroll")
    kept = svc.store.get(new["id"])["params"]["sheet_id"]
    assert kept == wanted
    assert rigging.is_valid_id(kept)

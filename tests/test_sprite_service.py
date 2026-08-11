"""The sprite-synthesis door: every refusal, before the row exists.

The point of the layer, and the reason each of these is a test rather than a
comment: a request that cannot run should cost the request, not a place in the
queue and two SDXL generations behind whatever else is going.
"""

from __future__ import annotations

import json
import time

import pytest
from PIL import Image

from warlock import fetch, rigging
from warlock.pipelines import spritesynth as ss
from warlock.service import Conflict, Invalid, NotFound
from warlock.service import jobs as svc_jobs
from warlock.service import sprites as svc_sprites


@pytest.fixture
def weights(monkeypatch):
    """Every model this kind loads, pinned present.

    Pinned rather than left to the machine, for ``conftest``'s stated reason:
    whether a refusal fires would otherwise depend on which weights whoever ran
    the suite happens to have downloaded.
    """
    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (True, None))
    monkeypatch.setattr(fetch, "present", lambda *a, **k: True)


def _reference(svc, *, done=True, image=True):
    job_id = svc.store.create("text", "a knight", {"seed": 1}, stage="reference")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    if image:
        Image.new("RGB", (64, 64), (200, 200, 200)).save(job_dir / "input.png")
    if done:
        svc.store.set_status(job_id, "done")
    return job_id


def _draft_on_disk(svc, job_id, *, draft_id=None, created=1.0, sidecar=True):
    """A finished draft's trio, written in the worker's own order."""
    draft_id = draft_id or rigging.new_id()
    job_dir = svc.job_dir(job_id)
    geom = ss.geometry("turnaround")
    for letter in rigging.SPRITE_CANDIDATES:
        path = rigging.sprite_draft_png_path(job_dir, draft_id, letter)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (128, 128), (0, 0, 0, 0)).save(path)
    if sidecar:
        doc = ss.draft_sidecar(
            draft_id=draft_id,
            source_job=job_id,
            created=created,
            geom=geom,
            logical_size=64,
            colors=32,
            candidates=[{"image": f"{draft_id}.a.png", "seed": 1}],
            recipe={},
        )
        rigging.sprite_draft_path(job_dir, draft_id).write_text(
            json.dumps(doc), encoding="utf-8"
        )
    return draft_id


# --- the options ------------------------------------------------------------


def test_the_options_describe_both_grids_from_the_geometry():
    options = svc_sprites.sprite_options()
    keys = {entry["key"] for entry in options["sheet_types"]}
    assert keys == set(svc_sprites.SPRITE_SHEET_TYPES)
    for entry in options["sheet_types"]:
        geom = ss.geometry(entry["key"])
        assert (entry["columns"], entry["rows"]) == (geom.columns, geom.rows)
        assert entry["cells"] == len(geom.cells)


def test_the_defaults_are_offered_choices():
    options = svc_sprites.sprite_options()
    defaults = options["defaults"]
    assert defaults["sheet_type"] in svc_sprites.SPRITE_SHEET_TYPES
    assert defaults["logical_size"] in svc_sprites.SPRITE_LOGICAL_SIZES
    assert defaults["colors"] in svc_sprites.SPRITE_COLOR_CHOICES


# --- refusals ---------------------------------------------------------------


def test_an_unfinished_reference_is_refused(svc, weights):
    job_id = _reference(svc, done=False)
    with pytest.raises(Invalid, match="That reference"):
        svc_sprites.create_sprite_synthesis(svc, job_id)


def test_a_reference_with_no_image_is_refused(svc, weights):
    job_id = _reference(svc, image=False)
    with pytest.raises(Invalid, match="no image"):
        svc_sprites.create_sprite_synthesis(svc, job_id)


def test_a_mesh_job_is_refused(svc, weights):
    job_id = svc.store.create("text", "a knight", {"seed": 1}, stage="model")
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid, match="only a 2D reference"):
        svc_sprites.create_sprite_synthesis(svc, job_id)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    (
        ({"sheet_type": "isometric"}, "sheet_type"),
        ({"logical_size": 17}, "logical_size"),
        ({"colors": 7}, "colors"),
    ),
)
def test_a_value_off_the_menu_is_refused_naming_its_control(svc, weights, kwargs, field):
    job_id = _reference(svc)
    with pytest.raises(Invalid) as exc:
        svc_sprites.create_sprite_synthesis(svc, job_id, **kwargs)
    assert exc.value.field == field


def test_two_equal_seeds_are_refused(svc, weights):
    job_id = _reference(svc)
    with pytest.raises(Invalid) as exc:
        svc_sprites.create_sprite_synthesis(svc, job_id, seed_a=7, seed_b=7)
    assert exc.value.field == "seed_b"
    assert "same picture twice" in str(exc.value)


def test_an_out_of_range_seed_is_refused(svc, weights):
    job_id = _reference(svc)
    with pytest.raises(Invalid):
        svc_sprites.create_sprite_synthesis(svc, job_id, seed_a=-1)


def test_a_missing_checkpoint_is_refused_with_its_download_line(svc, monkeypatch):
    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (False, None))
    monkeypatch.setattr(fetch, "present", lambda *a, **k: True)
    job_id = _reference(svc)
    with pytest.raises(Invalid) as exc:
        svc_sprites.create_sprite_synthesis(svc, job_id)
    assert exc.value.field == "base_model"
    assert "hf download" in str(exc.value)


def test_a_missing_adapter_is_refused_with_its_download_line(svc, monkeypatch):
    """The three the feature cannot run without are refused by name -- a
    missing ControlNet is not a plainer picture here, it is no pose guide."""
    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (True, None))
    monkeypatch.setattr(fetch, "present", lambda *a, **k: False)
    job_id = _reference(svc)
    with pytest.raises(Invalid) as exc:
        svc_sprites.create_sprite_synthesis(svc, job_id)
    assert exc.value.field in ("ip_adapter", "control", "style_lora")
    assert "hf download" in str(exc.value)


def test_the_draft_cap_is_a_conflict_not_an_invalid(svc, weights, monkeypatch):
    monkeypatch.setattr(rigging, "MAX_SPRITE_DRAFTS", 1)
    job_id = _reference(svc)
    _draft_on_disk(svc, job_id)
    with pytest.raises(Conflict, match="delete one first"):
        svc_sprites.create_sprite_synthesis(svc, job_id)


# --- what a successful submit records ---------------------------------------


def test_a_submit_records_every_input_and_names_the_draft(svc, weights):
    job_id = _reference(svc)
    result = svc_sprites.create_sprite_synthesis(
        svc, job_id, sheet_type="walk", logical_size=48, colors=16, seed_a=3, seed_b=4
    )
    row = svc.store.get(result["id"])
    assert row["kind"] == "sprite_synthesis"
    assert row["params"] == {
        "source_job": job_id,
        "sheet_type": "walk",
        "logical_size": 48,
        "colors": 16,
        "seed_a": 3,
        "seed_b": 4,
        "draft_id": result["draft"],
        "base_model": svc_sprites.SPRITE_BASE_MODEL,
    }
    assert rigging.is_valid_id(result["draft"])


def test_unspecified_seeds_are_drawn_distinct(svc, weights):
    job_id = _reference(svc)
    for _ in range(5):
        result = svc_sprites.create_sprite_synthesis(svc, job_id)
        params = svc.store.get(result["id"])["params"]
        assert params["seed_a"] != params["seed_b"]


def test_the_card_is_checked_at_the_door(svc, weights, monkeypatch):
    """Not in the worker: on Windows an overcommit spills into host commit and
    the symptom is the machine dying, not the job erroring."""
    from warlock.service import validation

    seen: list[tuple] = []
    monkeypatch.setattr(
        validation, "check_vram", lambda svc, kind, stage, params: seen.append((kind, stage))
    )
    monkeypatch.setattr(svc_sprites, "check_vram", validation.check_vram)
    job_id = _reference(svc)
    svc_sprites.create_sprite_synthesis(svc, job_id)
    assert seen == [("sprite_synthesis", "model")]


def test_a_running_synthesis_blocks_deleting_its_reference(svc, weights):
    job_id = _reference(svc)
    result = svc_sprites.create_sprite_synthesis(svc, job_id)
    assert svc_jobs.dependent_jobs(svc, job_id) == [result["id"]]


# --- reading drafts back ----------------------------------------------------


def test_drafts_accumulate_oldest_first(svc):
    job_id = _reference(svc)
    first = _draft_on_disk(svc, job_id, created=1.0)
    second = _draft_on_disk(svc, job_id, created=2.0)
    listed = svc_sprites.list_sprite_drafts(svc, job_id)["drafts"]
    assert [d["id"] for d in listed] == [first, second]


def test_a_draft_with_no_sidecar_is_not_listed_or_served(svc):
    """The sidecar is the completion marker: both PNGs are written first, so a
    PNG that exists can still be half-written."""
    job_id = _reference(svc)
    draft_id = _draft_on_disk(svc, job_id, sidecar=False)
    assert svc_sprites.list_sprite_drafts(svc, job_id)["drafts"] == []
    with pytest.raises(NotFound):
        svc_sprites.sprite_draft_png(svc, job_id, draft_id, "a")


def test_a_finished_draft_serves_both_candidates(svc):
    job_id = _reference(svc)
    draft_id = _draft_on_disk(svc, job_id)
    for letter in rigging.SPRITE_CANDIDATES:
        assert svc_sprites.sprite_draft_png(svc, job_id, draft_id, letter).exists()


def test_an_unknown_candidate_letter_is_not_found(svc):
    job_id = _reference(svc)
    draft_id = _draft_on_disk(svc, job_id)
    with pytest.raises(NotFound):
        svc_sprites.sprite_draft_png(svc, job_id, draft_id, "c")


def test_a_malformed_draft_id_is_not_found(svc):
    job_id = _reference(svc)
    with pytest.raises(NotFound):
        svc_sprites.get_sprite_draft(svc, job_id, "../../etc")


def test_deleting_one_draft_leaves_the_others(svc):
    job_id = _reference(svc)
    first = _draft_on_disk(svc, job_id, created=1.0)
    second = _draft_on_disk(svc, job_id, created=2.0)
    svc_sprites.delete_sprite_draft(svc, job_id, first)
    assert [d["id"] for d in svc_sprites.list_sprite_drafts(svc, job_id)["drafts"]] == [
        second
    ]
    with pytest.raises(NotFound):
        svc_sprites.delete_sprite_draft(svc, job_id, first)


def test_deleting_a_draft_takes_both_of_its_candidates(svc):
    job_id = _reference(svc)
    draft_id = _draft_on_disk(svc, job_id)
    svc_sprites.delete_sprite_draft(svc, job_id, draft_id)
    job_dir = svc.job_dir(job_id)
    assert not any(
        rigging.sprite_draft_png_path(job_dir, draft_id, c).exists()
        for c in rigging.SPRITE_CANDIDATES
    )


def test_a_draft_is_never_rewritten_in_place(svc, weights):
    """What makes the pane's directory-mtime cache sound: every run mints a
    fresh id, so a cached record can go stale only by disappearing."""
    job_id = _reference(svc)
    ids = {
        svc_sprites.create_sprite_synthesis(svc, job_id)["draft"] for _ in range(5)
    }
    assert len(ids) == 5


def test_the_created_stamp_orders_a_listing_written_out_of_order(svc):
    job_id = _reference(svc)
    now = time.time()
    late = _draft_on_disk(svc, job_id, created=now)
    early = _draft_on_disk(svc, job_id, created=now - 1000)
    assert [d["id"] for d in rigging.list_sprite_drafts(svc.job_dir(job_id))] == [
        early,
        late,
    ]

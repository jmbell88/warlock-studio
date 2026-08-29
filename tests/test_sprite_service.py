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


def test_the_options_describe_every_grid_from_the_geometry():
    """Both legacy atlases, plus every planned kind whose pose guide is on
    disk -- which is the whole point of the menu being discovered rather than
    listed: an action with no guide behind it is not one this build can draw."""
    options = svc_sprites.sprite_options()
    keys = {entry["key"] for entry in options["sheet_types"]}
    assert keys == set(svc_sprites.sprite_sheet_types())
    assert set(svc_sprites.SPRITE_LEGACY_SHEET_TYPES) <= keys
    for entry in options["sheet_types"]:
        geom = ss.sheet_geometry(entry["key"], max(entry["logical_sizes"]))
        assert (entry["columns"], entry["rows"]) == (geom.columns, geom.rows)
        assert entry["cells"] == len(geom.cells)
        assert entry["directions"] == list(geom.directions)


def test_the_defaults_are_offered_choices():
    options = svc_sprites.sprite_options()
    defaults = options["defaults"]
    assert defaults["sheet_type"] in svc_sprites.sprite_sheet_types()
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
        # Sixteen cells, so the pair the feature was written around; the number
        # is recorded rather than left for the worker to decide later.
        "candidates": 2,
        "logical_size": 48,
        "colors": 16,
        # Normalised by ``_check_options`` and written whether or not the
        # caller named them: the worker reads params, not the door's defaults,
        # and a key that is present only sometimes is a second code path.
        "palette": "",
        "dither": False,
        "outline": svc_sprites.DEFAULT_SPRITE_OUTLINE,
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


# --- the two doors, held to one answer ----------------------------------------


@pytest.mark.parametrize(
    ("block", "field"),
    [
        ({"logical_size": 96}, "logical_size"),
        ({"colors": 7}, "colors"),
        ({"outline": "glow"}, "outline"),
        ({"palette": "never-installed"}, "palette"),
    ],
)
def test_the_two_doors_refuse_the_same_values_in_the_same_words(
    svc, weights, block, field
):
    """There are exactly two ways a ``sprite_synthesis`` row comes to exist --
    this door, and the Create form's follow-up block, which the worker mints
    itself and which therefore never passes through here. A value one takes and
    the other refuses is a request that succeeds or fails depending on which
    button made it.

    Both go through ``sprites._check_options`` since 2026-08-29, so this pins
    the *message* and the field rather than merely "both raised": a refusal is a
    sentence somebody reads and a field a form highlights, and two doors saying
    it differently is the drift the shared checker exists to stop.
    """
    job_id = _reference(svc)

    with pytest.raises(Invalid) as direct:
        svc_sprites.create_sprite_synthesis(svc, job_id, **block)
    with pytest.raises(Invalid) as followup:
        svc_jobs.create_job(
            svc, kind="text", prompt="a ranger", output="reference", sprite_sheet=block
        )

    assert direct.value.field == followup.value.field == field
    assert direct.value.message == followup.value.message


def test_a_palette_is_read_at_the_door_and_only_its_name_is_stored(
    svc, weights, tmp_path
):
    """The reason ``check_pixel_options`` loads a file it then throws away: an
    unreadable palette should cost the request, not two SDXL generations and a
    pair of sheets that came back the wrong colours.

    Only the name is carried, so the worker re-reads the file -- an edit between
    queueing and running is the user's edit rather than a stale snapshot.
    """
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    (directory / "ramp.hex").write_text("#1a1c2c\n#f4f4f4\n")
    (directory / "broken.hex").write_text("this is not a palette\n")
    job_id = _reference(svc)

    with pytest.raises(Invalid) as excinfo:
        svc_sprites.create_sprite_synthesis(svc, job_id, palette="broken")
    assert excinfo.value.field == "palette"

    result = svc_sprites.create_sprite_synthesis(
        svc, job_id, palette=" ramp ", dither=True, outline="outer"
    )
    params = svc.store.get(result["id"])["params"]
    assert params["palette"] == "ramp"
    assert params["dither"] is True
    assert params["outline"] == "outer"


def test_the_offered_outlines_are_the_ones_the_assembler_draws():
    """A form offering a mode the pixeliser refuses is a control that fails at
    the door it was drawn from."""
    from warlock.pipelines import pixelize

    options = svc_sprites.sprite_options()
    assert options["outlines"] == list(pixelize.OUTLINE_MODES)
    assert options["defaults"]["outline"] in options["outlines"]
    # ``inner`` and not ``outer``, which is this path's whole departure from
    # Troupe's: a synthesised cell has no guaranteed margin.
    assert options["defaults"]["outline"] == "inner"
    assert svc_sprites.DEFAULT_SPRITE_OUTLINE == ss.DEFAULT_SPRITE_OUTLINE


# --- the action menu, and the gate on the size picker ------------------------


def test_only_actions_with_a_pose_guide_on_disk_are_offered(monkeypatch, tmp_path):
    """The guide *is* the pose -- it is the ControlNet hint, and it is the only
    thing that decides where the limbs go. An action offered without one is a
    control whose result is eight bands of an unposed character."""
    monkeypatch.setattr(ss, "TEMPLATE_DIR", tmp_path)
    assert svc_sprites.sprite_options()["actions"] == []
    assert svc_sprites.sprite_sheet_types() == svc_sprites.SPRITE_LEGACY_SHEET_TYPES

    (tmp_path / "idle8.json").write_text("{}", encoding="utf-8")
    options = svc_sprites.sprite_options()
    assert [a["key"] for a in options["actions"]] == ["idle"]
    assert [d["count"] for d in options["actions"][0]["directions"]] == [8]
    assert "idle8" in svc_sprites.sprite_sheet_types()


def test_the_actions_carry_the_arithmetic_the_form_draws_its_line_from():
    """From the door and not recomputed by the pane: a second copy of a cell
    count is a label that goes stale the first time a frame count moves."""
    options = svc_sprites.sprite_options()
    walk = next(a for a in options["actions"] if a["key"] == "walk")
    eight = next(d for d in walk["directions"] if d["count"] == 8)

    assert walk["frames"] == 8
    assert eight["kind"] == "walk8"
    assert eight["cells"] == 64
    assert (eight["columns"], eight["rows"]) == (8, 8)
    # One band is one whole direction, so this is also how many generations one
    # candidate costs.
    assert eight["bands"] == 8
    assert eight["candidates"] == 1
    assert eight["seconds_per_candidate"] == 8 * svc_sprites.SECONDS_PER_GENERATION


def test_an_eight_frame_action_is_offered_only_at_the_size_it_fits():
    """One direction of an eight-frame walk at 64px would want a 2048px band and
    there is no such thing, so the menu says 32 and the door refuses the rest."""
    options = svc_sprites.sprite_options()
    by_key = {a["key"]: a for a in options["actions"]}

    assert by_key["walk"]["logical_sizes"] == [32]
    assert by_key["idle"]["logical_sizes"] == list(svc_sprites.SPRITE_LOGICAL_SIZES)
    # And the legacy atlases are unaffected: they are one 1024px generation
    # however small the published cell is.
    for entry in options["sheet_types"]:
        if entry["key"] in svc_sprites.SPRITE_LEGACY_SHEET_TYPES:
            assert entry["logical_sizes"] == list(svc_sprites.SPRITE_LOGICAL_SIZES)


@pytest.mark.parametrize("action", ["walk", "run", "attack"])
@pytest.mark.parametrize("size", [48, 64])
def test_a_multi_frame_action_at_a_big_cell_is_refused_naming_both_numbers(
    svc, weights, action, size
):
    """Refused at the door rather than after the press, and in a sentence with
    the frame count *and* the band size in it -- the pipeline's own words, so
    the two cannot come to refuse two different sets."""
    job_id = _reference(svc)
    with pytest.raises(Invalid) as caught:
        svc_sprites.create_sprite_synthesis(
            svc, job_id, action=action, directions=8, logical_size=size
        )

    message = str(caught.value)
    frames = ss.ACTION_FRAMES[action]
    assert f"is {frames} frames of {size * ss.PX_PER_ART_PIXEL}px" in message
    assert "1024x1024" in message
    assert caught.value.field == "logical_size"
    assert svc.store.active_jobs() == []


def test_the_same_action_at_the_size_it_fits_is_admitted(svc, weights):
    job_id = _reference(svc)
    result = svc_sprites.create_sprite_synthesis(
        svc, job_id, action="walk", directions=8, logical_size=32
    )

    params = svc.store.get(result["id"])["params"]
    assert params["sheet_type"] == "walk8"
    assert params["logical_size"] == 32
    # 64 cells, so one draft rather than the pair: two of these is sixteen
    # generations.
    assert params["candidates"] == 1


def test_an_action_with_no_guide_on_disk_is_refused_by_name(
    svc, weights, monkeypatch, tmp_path
):
    """The other half of "not offered": the two have to agree, or a request the
    menu would never make is one the door quietly accepts. Reached by emptying
    the guide directory, because every planned kind ships a guide today -- which
    is exactly why the refusal needs a test of its own rather than a kind that
    happens to be missing."""
    monkeypatch.setattr(ss, "TEMPLATE_DIR", tmp_path)
    job_id = _reference(svc)
    with pytest.raises(Invalid, match="no pose guide"):
        svc_sprites.create_sprite_synthesis(svc, job_id, action="idle", directions=8)


def test_an_unknown_action_is_refused_before_anything_is_queued(svc, weights):
    job_id = _reference(svc)
    with pytest.raises(Invalid, match="sheet_type must be one of"):
        svc_sprites.create_sprite_synthesis(svc, job_id, action="dance", directions=8)
    assert svc.store.active_jobs() == []


def test_the_action_pair_and_the_stored_kind_are_the_same_choice():
    """Two callers speak two languages -- a form has an Action combo, a params
    blob has a ``sheet_type`` -- and exactly one spelling is ever written down.
    """
    assert svc_sprites.resolve_sheet_kind("turnaround", None, None) == "turnaround"
    assert svc_sprites.resolve_sheet_kind(None, "walk", 8) == "walk8"
    assert svc_sprites.resolve_sheet_kind(None, "idle", 4) == "idle4"
    # An explicit action wins over a sheet_type sent beside it: that request came
    # from a form whose controls are the pair, and the sheet_type is the stale
    # half.
    assert svc_sprites.resolve_sheet_kind("turnaround", "idle", 8) == "idle8"
    # Nothing at all is the door's own default rather than a crash.
    assert (
        svc_sprites.resolve_sheet_kind(None, None, None)
        == svc_sprites.DEFAULT_SPRITE_SHEET_TYPE
    )


@pytest.mark.parametrize("asked", [0, 3, "two"])
def test_a_candidate_count_that_is_not_one_or_two_is_refused(svc, weights, asked):
    job_id = _reference(svc)
    with pytest.raises(Invalid, match="one candidate or two"):
        svc_sprites.create_sprite_synthesis(svc, job_id, candidates=asked)


def test_a_pinned_pair_is_honoured_even_on_a_big_sheet(svc, weights):
    """The default drops to one past the legacy sheet size; it is a default and
    not a cap, and a user who asks for the pair gets it."""
    job_id = _reference(svc)
    result = svc_sprites.create_sprite_synthesis(
        svc, job_id, action="walk", directions=8, logical_size=32, candidates=2
    )

    assert svc.store.get(result["id"])["params"]["candidates"] == 2

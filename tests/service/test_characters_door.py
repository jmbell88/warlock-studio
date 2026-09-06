"""``service.troupe``'s doors, seen from the character rather than the mesh.

What a character sheet actually needs is a rig with clips authored for it. The
door used to ask a narrower question -- "is this the humanoid template?" -- and
that answer was right only for as long as ``humanoid`` was the one template with
a clip library.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from warlock import doctor, rigging
from warlock.characters import family as family_mod
from warlock.service import characters as svc_characters
from warlock.service import export as svc_export
from warlock.service import jobs as svc_jobs
from warlock.service import troupe as svc_troupe
from warlock.service.errors import Invalid, NotFound


def _rigged_mesh(svc, template):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


def test_create_charsheet_accepts_any_template_with_a_clip_library_and_refuses_one_without(
    svc, monkeypatch
):
    """**The refusal is about clips, not about ``humanoid``.**

    A family that ships its own skeleton and its own walk cycle was turned away
    for not being the one template that happened to have a library first -- and
    the message told it so, which sent the reader to change the rig rather than
    to author the clips. Now the door asks ``rigging.clip_library`` (which
    answers "no clips" rather than failing, so somebody has to ask) and the
    sheet is expanded from the rig's *own* template.
    """
    # Nothing is authored for a fish, so the sheet is refused by what is
    # missing -- and the sentence names the rig rather than naming humanoid.
    #
    # ``fish`` and not ``quadruped``: the quadruped, bird and blob skeletons
    # gained authored clip libraries on 2026-09-05 for the character families,
    # so the example this test was written around stopped being an example of
    # anything. The claim is unchanged; only the template that still has no
    # clips is.
    fish = _rigged_mesh(svc, "fish")
    with pytest.raises(Invalid) as refused:
        svc_troupe.create_charsheet(svc, fish)
    assert "clip library" in str(refused.value)
    assert "fish" in str(refused.value)

    # The same rig, once clips exist for it, is accepted -- and the row is
    # minted on its own template, because that is the skeleton on disk.
    library = rigging.clip_library(svc_troupe.TROUPE_TEMPLATE)
    monkeypatch.setattr(
        rigging,
        "clip_library",
        lambda key: library if key == "fish" else {"poses": {}, "clips": []},
    )
    made = svc_troupe.create_charsheet(svc, fish)
    row = svc.store.get(made["id"])
    assert row["params"]["template"] == "fish"

    # And a rig whose template is not recorded at all is still refused, rather
    # than raising the ``ValueError`` ``get_template`` answers an unknown key
    # with -- from the user's side it is the same missing clip library.
    monkeypatch.undo()
    nameless = _rigged_mesh(svc, "")
    with pytest.raises(Invalid, match="clip library"):
        svc_troupe.create_charsheet(svc, nameless)


# --- the character door ------------------------------------------------------
#
# ``service.characters`` is one press that produces a body, a skeleton and a
# queued sheet. What these pin is the *order* it does that in and the fact that
# nothing in it is humanoid-shaped.

@pytest.fixture
def blender(monkeypatch):
    """Blender answered present. Every character needs a rig, so without this
    every test below would be testing the refusal instead of the door."""

    class _Ok:
        ok = True
        detail = ""

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _Ok())
    return _Ok()


#: One species per archetype, and the point of the list. A door that reads the
#: rig template off a constant passes the first row and fails the other three.
ONE_PER_ARCHETYPE = ["human", "wolf", "dragon", "slime"]


def _recipe(family_key, **changes):
    """The smallest recipe that renders: one animation, one direction.

    Deliberately not ``DEFAULT_RECIPE`` -- that is 144 cells, and the plan this
    door runs up front walks every one of them. The claims here are about rows
    and params, none of which change with the cell count.
    """
    fam = family_mod.get_family(family_key)
    return {
        "family": family_key,
        "theme": fam.themes[0].key,
        "animations": {"idle": 2},
        "directions": 1,
        "logical_size": 32,
        "colors": 16,
        **changes,
    }


@pytest.mark.parametrize("family_key", ONE_PER_ARCHETYPE)
def test_a_character_mints_a_built_done_model_row_and_one_rig_row(svc, blender, family_key):
    """**One press, two rows -- for every archetype, not just humanoids.**

    The mesh row is minted finished the way ``import_mesh`` mints one: there is
    no reconstruction behind a generated body, so queueing an image job for it
    would spend two minutes of GPU reproducing what this door just built.

    Parameterised across all four body plans because the first draft of a door
    like this reads its rig template and its clip library off
    ``TROUPE_TEMPLATE``, passes for a human and mints a wolf on a humanoid
    skeleton -- which fails an hour later in the worker as a frame-count error.
    """
    made = svc_characters.create_character(svc, _recipe(family_key))
    assert made["kind"] == "character"

    row = svc.store.get(made["id"])
    assert row["stage"] == "model"
    assert row["status"] == "done"
    assert row["params"]["built"] is True
    assert row["params"]["asset_intent"] == "character"
    assert row["params"]["family"] == family_key
    job_dir = svc.job_dir(made["id"])
    assert (job_dir / "model.glb").exists()
    assert (job_dir / "source.glb").exists()
    assert (job_dir / "character.json").exists()

    rig = svc.store.get(made["rig"])
    assert rig["kind"] == "rig"
    assert rig["params"]["source_job"] == made["id"]
    # The species' own skeleton and the species' own clips, read off the
    # archetype. This is the assertion the parameterisation exists for.
    arch = family_mod.get_family(family_key).arch
    assert rig["params"]["template"] == arch.template
    assert rig["params"]["troupe_sheet"]["template"] == arch.template

    # And exactly two rows: no third row minted eagerly. The sheet is the
    # worker's to mint on the finished rig, so it cancels on its own.
    assert len(svc.store.list(limit=50)) == 2


def test_the_rig_row_carries_the_exact_joints_and_never_measures_them(svc, blender):
    """A family states its skeleton exactly; measuring would guess it again.

    ``joints="measured"`` reads joint positions off a *reference image*, which
    this character never had -- there is no reference, only a recipe. Passing
    ``bones`` withholds that flag in ``send_to_troupe``, and the pin is here as
    well as there because this is the one door that always has an exact answer.
    """
    made = svc_characters.create_character(svc, _recipe("human"))
    params = svc.store.get(made["rig"])["params"]
    assert "joints" not in params, "the exact skeleton was thrown away and re-measured"
    stored = params["bones"]
    bones = stored["bones"] if isinstance(stored, dict) else stored
    names = [b["name"] for b in bones]
    template = rigging.get_template("humanoid")
    assert names == [b["name"] for b in template.bones]
    # Real coordinates, not a stub: a skeleton collapsed to the origin would
    # satisfy every structural assertion above and skin the whole mesh to one
    # point.
    assert any(abs(v) > 1e-6 for b in bones for v in list(b["head"]) + list(b["tail"]))


def test_a_character_refuses_a_rerun_and_reroll_character_is_the_door_instead(svc, blender):
    """``rerun_job`` has nothing to re-run: ``built`` says so.

    The compensation is :func:`reroll_character`, and it is a door that copies
    ``params`` -- which puts it under
    ``test_rerun_regressions::test_every_door_that_copies_params_rerolls_the_seeds``.
    A reroll that reproduced the recipe's seed would look like it ran, take the
    time, and hand back a byte-identical character.
    """
    made = svc_characters.create_character(svc, _recipe("human", seed=4242))
    with pytest.raises(Invalid, match="built"):
        svc_jobs.rerun_job(svc, made["id"], mode="reroll")

    again = svc_characters.reroll_character(svc, made["id"])
    assert again["id"] != made["id"]
    before = svc.store.get(made["id"])["params"]
    after = svc.store.get(again["id"])["params"]
    assert before["character"]["recipe"]["seed"] == 4242
    assert after["character"]["recipe"]["seed"] != 4242, "the recipe seed came through unchanged"
    assert after["mesh_seed"] != before["mesh_seed"], "the mesh seed came through unchanged"
    # Same character, different roll: everything else in the recipe survives.
    assert {k: v for k, v in after["character"]["recipe"].items() if k != "seed"} == {
        k: v for k, v in before["character"]["recipe"].items() if k != "seed"
    }


def test_a_re_render_of_a_subset_keeps_the_recipe_seed_byte_for_byte(svc, blender):
    """**The flames in twelve re-rendered cells must match the ones beside them.**

    ``rerender_charsheet`` copies the row's params wholesale and strips only
    what the *worker* recorded, so the nested character block -- and the recipe
    seed inside it that drives every procedural effect -- rides through
    untouched. Exactly the pinned-palette argument: a door that re-rolled it
    would produce a subset whose effects are a different draw from the cells it
    is landing among.
    """
    made = svc_characters.create_character(svc, _recipe("human"))
    block = svc.store.get(made["rig"])["params"]["troupe_sheet"]["character"]

    job_dir = svc.job_dir(made["id"])
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    sheet = svc_troupe.create_charsheet(svc, made["id"], character=block)
    sheet_params = svc.store.get(sheet["id"])["params"]
    # The sheet has to exist on disk for the re-render door to read its layout.
    rigging.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    rigging.sheet_path(job_dir, sheet["sheet_id"]).write_text(
        json.dumps({"troupe": sheet_params["layout"]}), "utf-8"
    )
    rigging.sheet_png_path(job_dir, sheet["sheet_id"]).write_bytes(b"png")

    again = svc_troupe.rerender_charsheet(
        svc,
        made["id"],
        sheet_id=sheet["sheet_id"],
        subset=[{"animation": "idle", "direction": "front"}],
    )
    copied = svc.store.get(again["id"])["params"]["character"]
    assert copied["recipe"]["seed"] == block["recipe"]["seed"]
    assert copied == block


def test_the_character_block_rides_every_row_of_the_chain(svc, blender):
    """Model, rig and sheet all say who this is -- and it is always *nested*.

    Flattened, any of ``seed``, ``palette`` or ``colors`` inside the recipe
    would be one entry away from being picked up by ``VECTOR_PARAMS``, which is
    an allowlist of flat settings, and quietly becoming a rerun vector for a row
    that never asked for one.
    """
    from warlock.vectors import VECTOR_PARAMS

    made = svc_characters.create_character(
        svc, _recipe("wolf", name="Fang"), prompt="a grey wolf"
    )

    model = svc.store.get(made["id"])["params"]
    rig = svc.store.get(made["rig"])["params"]
    block = model["character"]
    assert block["version"] == svc_characters.CHARACTER_BLOCK_VERSION
    assert block["recipe"]["family"] == "wolf"
    assert block["prompt"] == "a grey wolf"
    assert block["resolution"]["family"] is None  # nothing was resolved
    assert rig["troupe_sheet"]["character"] == block

    # And the third row, minted the way the worker mints it -- from the block
    # the rig row is carrying.
    job_dir = svc.job_dir(made["id"])
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": "quadruped"}), "utf-8")
    sheet = svc_troupe.create_charsheet(
        svc, made["id"], character=rig["troupe_sheet"]["character"]
    )
    assert svc.store.get(sheet["id"])["params"]["character"] == block

    # Nested, and provably so. ``VECTOR_PARAMS`` is an allowlist of *flat*
    # settings, and the block is not on it -- nor is any recipe setting a
    # top-level key of the rows that carry it, which is what would have to be
    # true for one of them to be picked up as a rerun vector.
    from warlock.service.validation import DERIVED_PARAMS

    assert "character" not in VECTOR_PARAMS
    assert not set(model) & set(VECTOR_PARAMS)
    flattened = {"palette", "colors", "logical_size", "outline", "animations", "appearance"}
    assert not flattened & set(model)
    assert not flattened & set(rig)
    # And it is the *request*, not something the worker learned: stripping it
    # on a rerun would leave a sheet that no longer knows who it depicts.
    assert "character" not in DERIVED_PARAMS


def test_a_bad_recipe_costs_the_request_and_leaves_no_directory(svc, blender):
    """The whole ordering argument, in one assertion.

    An unmakeable character must cost the request -- not a mesh, not a rig, not
    144 EEVEE frames. So every refusal that can be raised without building
    anything is raised first, and the one failure that cannot (the build itself)
    takes its directory with it on the way out.
    """
    before = sorted(p.name for p in svc.config.data_dir.iterdir())

    with pytest.raises(Invalid) as refused:
        svc_characters.create_character(svc, _recipe("human", colors=7))
    assert refused.value.field == "colors"
    with pytest.raises(Invalid) as unknown:
        svc_characters.create_character(svc, _recipe("human", theme="chartreuse"))
    assert unknown.value.field == "theme"

    assert svc.store.list(limit=50) == []
    assert sorted(p.name for p in svc.config.data_dir.iterdir()) == before


def test_a_build_that_fails_halfway_takes_its_directory_with_it(svc, blender, monkeypatch):
    """``import_mesh``'s cleanup, for the same reason: a disk-full mid-write
    must not leave a truncated orphan the library then lists."""
    from warlock.characters import instantiate as instantiate_mod

    real = instantiate_mod.instantiate
    seen = []

    def _explode(recipe, out_dir):
        real(recipe, out_dir)
        seen.append(Path(out_dir))
        raise OSError("no space left on device")

    monkeypatch.setattr(instantiate_mod, "instantiate", _explode)
    with pytest.raises(OSError):
        svc_characters.create_character(svc, _recipe("human"))
    assert seen and not seen[0].exists()
    assert svc.store.list(limit=50) == []


def test_without_blender_a_character_is_refused_in_the_rig_segments_words(svc, monkeypatch):
    """One wording for "this needs Blender", wherever the app meets it -- and
    the refusal lands *before* the mesh, because a body whose skeleton can never
    be built is the half-made asset this door's ordering exists to prevent."""

    class _Missing:
        ok = False
        detail = "no bpy"

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _Missing())
    with pytest.raises(Invalid) as refused:
        svc_characters.create_character(svc, _recipe("human"))
    assert str(refused.value) == "Rigging needs Blender, which is not installed."
    assert svc.store.list(limit=50) == []


# --- the exported package ----------------------------------------------------


def _sheet_on_disk(svc, job_id):
    sheet_id = rigging.new_id()
    job_dir = svc.job_dir(job_id)
    rigging.sheet_dir(job_dir).mkdir(parents=True, exist_ok=True)
    rigging.sheet_png_path(job_dir, sheet_id).write_bytes(b"png-bytes")
    rigging.sheet_path(job_dir, sheet_id).write_text(json.dumps({"cells": []}), "utf-8")
    return sheet_id


def test_an_exported_package_lands_as_a_pair_and_leaves_no_temp(svc, blender, tmp_path):
    """**Both files or neither.** The PNG is the atlas and the JSON is what says
    which cell is ``walk`` facing south-east; a folder holding one without the
    other holds an asset nothing can interpret. The copy is staged for
    ``export.staged_copy``'s reason -- the destination is a watched project
    folder, and a torn read there is a hot-reloading engine's crash."""
    made = svc_characters.create_character(svc, _recipe("human", name="Ranger"))
    sheet_id = _sheet_on_disk(svc, made["id"])

    dest = tmp_path / "project" / "assets"
    out = svc_characters.export_package(svc, made["id"], sheet_id, dest_dir=dest)

    assert Path(out["png"]).read_bytes() == b"png-bytes"
    assert json.loads(Path(out["json"]).read_text("utf-8")) == {"cells": []}
    assert Path(out["png"]).stem == Path(out["json"]).stem == "Ranger"
    assert sorted(p.name for p in dest.iterdir()) == ["Ranger.json", "Ranger.png"]

    # Neither, when the pair cannot be completed: the sidecar's copy fails and
    # the PNG must not be standing at the destination on its own.
    real_copy = shutil.copyfile

    def _half(src, dst, *a, **k):
        if str(src).endswith(".json"):
            raise OSError("no space left on device")
        return real_copy(src, dst, *a, **k)

    other = tmp_path / "second"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(svc_export.shutil, "copyfile", _half)
        with pytest.raises(OSError):
            svc_characters.export_package(svc, made["id"], sheet_id, dest_dir=other)
    assert list(other.iterdir()) == []


def test_export_refuses_without_an_export_folder_in_the_librarys_exact_words(svc, blender):
    """``export_to_folder``'s sentence verbatim. Two wordings for one condition
    is how a user comes to believe they are two different problems."""
    made = svc_characters.create_character(svc, _recipe("human"))
    sheet_id = _sheet_on_disk(svc, made["id"])
    assert svc.config.export_dir is None

    with pytest.raises(NotFound) as refused:
        svc_characters.export_package(svc, made["id"], sheet_id)
    assert str(refused.value) == "no export folder configured (set WARLOCK_EXPORT_DIR)"


def test_a_preview_is_a_temp_glb_and_never_a_row(svc):
    """A slider drag must not mint a library row per frame. Keyed by the whole
    recipe, so dragging back to where it was is a hit rather than a rebuild."""
    path = svc_characters.preview_character(svc, _recipe("slime"))
    assert path.exists() and path.suffix == ".glb"
    assert path.parent == svc.config.data_dir / "tmp"
    assert svc.store.list(limit=50) == []
    assert svc_characters.preview_character(svc, _recipe("slime")) == path
    assert svc_characters.preview_character(svc, _recipe("ooze")) != path
    # And no scratch directory left behind beside them.
    assert all(p.is_file() and p.suffix == ".glb" for p in path.parent.iterdir())


def test_the_estimate_grows_with_the_cells_and_is_never_zero(svc):
    """A form has to say something before the press. It is an estimate and
    labelled one -- but a *character* reported as instant is a user who thinks
    the button did nothing."""
    assert svc_characters.estimate_minutes(0) == pytest.approx(svc_characters.RIG_MINUTES)
    assert svc_characters.estimate_minutes(144) > svc_characters.estimate_minutes(16)
    assert svc_characters.estimate_minutes(-5) == svc_characters.estimate_minutes(0)


def test_character_options_answers_for_every_archetype_and_species(svc):
    """One source for the whole form. A picker built from a subset of the
    registry is a picker that cannot offer three quarters of what ships."""
    options = svc_characters.character_options(svc)
    assert {a["key"] for a in options["archetypes"]} == set(family_mod.archetypes())
    assert len(options["families"]) == len(family_mod.families())
    assert set(options["channels"]) == set(family_mod.families())
    assert options["default_recipe"]["family"]
    assert options["troupe"]["camera_presets"]


def test_a_finished_character_offers_troupe_once_it_is_rigged_and_clay_before(svc):
    """The card's one action, for the intent this door writes.

    Never "rig": the rig row was minted in the same press, and offering to make
    a second one is how a user spends Blender twice on one character. That is
    exactly what the generic ladder below the arm answers for the window
    between the mesh landing and the rig landing.
    """
    from warlock.studio import state

    body = {
        "status": "done",
        "stage": "model",
        "files": ["model.glb"],
        "params": {"asset_intent": "character"},
    }
    assert state.primary_action(body) == "clay"
    rigged = {**body, "files": ["model.glb", "rig.glb"]}
    assert state.primary_action(rigged) == "troupe"
    # And the label exists, because the library looks it up by name.
    assert state.ACTIONS["troupe"]

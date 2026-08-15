"""Create mode's stage model: the gates, the lineage walk, and the one switch.

Pure -- no imgui, no window, no service instance. That is the point of
``studio/create_stages.py`` being its own module: the rail is a picture of
these functions, and a picture is not where a wrong gate should first be
noticed.
"""

from __future__ import annotations

import pytest

from warlock.studio import create_stages
from warlock.studio.state import AppState


def job(**kwargs):
    base = {
        "id": "0123456789ab",
        "kind": "text",
        "stage": "model",
        "status": "done",
        "files": ["model.glb", "input.png"],
        "prompt": "a barrel",
        "parent_id": None,
        "created_at": "2026-01-01T00:00:00",
    }
    return {**base, **kwargs}


class FakeCache:
    def __init__(self, jobs=()):
        self.jobs = list(jobs)
        self.by_id = {j["id"]: j for j in self.jobs}

    def get(self, job_id):
        return None if job_id is None else self.by_id.get(job_id)


class FakeCtx:
    """Only what ``go`` touches: the state, the cache and ``job()``."""

    def __init__(self, jobs=(), selected=None):
        self.state = AppState()
        self.state.selected = selected
        self.cache = FakeCache(jobs)

    def job(self, job_id=None):
        return self.cache.get(job_id or self.state.selected)


# --- the stage list ---------------------------------------------------------


def test_the_stage_list_only_ever_grows():
    """The order is the pipeline's order and a released prefix of it is a
    promise. Wave 5 ships two segments and then four and then five; a step
    that *reordered* them would move a breadcrumb the user has already
    learned, and one that dropped a stage would strand its pane."""
    assert create_stages.STAGES[:2] == ("reference", "mesh")


def test_every_stage_has_a_label_and_a_reached_predicate():
    for stage in create_stages.STAGES:
        assert stage in create_stages.LABELS
        assert stage in create_stages._REACHED


# --- reached ----------------------------------------------------------------


def test_nothing_selected_has_reached_no_stage_at_all():
    """Not "reference": the rail ticks its segments off against this, and an
    empty Create mode used to show a check beside Reference on a screen where
    nothing had been generated."""
    assert create_stages.reached(None) is None


def test_a_reference_has_reached_the_reference_stage():
    assert create_stages.reached(job(stage="reference", files=["input.png"])) == "reference"


def test_a_tile_has_reached_the_reference_stage():
    assert create_stages.reached(job(stage="tile", files=["input.png"])) == "reference"


def test_a_model_job_has_reached_the_mesh_stage():
    assert create_stages.reached(job(stage="model")) == "mesh"


# --- shows ------------------------------------------------------------------


def test_reference_shows_a_mesh_because_the_mesh_carries_its_own_image():
    """The promotion copies ``input.png`` into the model job's directory, which
    is what lets the rail walk backwards without moving the selection."""
    assert create_stages.shows("reference", job(stage="model")) is True


def test_reference_cannot_show_a_job_with_no_image():
    assert create_stages.shows("reference", job(stage="model", files=["model.glb"])) is False


def test_only_a_model_job_is_shown_by_the_mesh_stage():
    assert create_stages.shows("mesh", job(stage="model")) is True
    assert create_stages.shows("mesh", job(stage="reference", files=["input.png"])) is False


def test_nothing_is_shown_by_nothing():
    for stage in create_stages.STAGES:
        assert create_stages.shows(stage, None) is False


# --- available --------------------------------------------------------------


def test_the_reference_stage_is_never_blocked():
    assert create_stages.available("reference", None) is None
    assert create_stages.available("reference", job(stage="tile")) is None


def test_the_mesh_stage_is_open_with_nothing_selected():
    """It takes an uploaded image, so an empty selection is not a blocker --
    it is the state the form is designed for."""
    assert create_stages.available("mesh", None) is None


def test_a_tile_blocks_the_mesh_stage_in_the_services_own_words():
    reason = create_stages.available("mesh", job(stage="tile", files=["input.png"]))
    assert reason == "a tile has no subject to reconstruct"


def test_an_unfinished_reference_blocks_the_mesh_stage_and_says_why():
    reason = create_stages.available(
        "mesh", job(stage="reference", status="running", files=["input.png"])
    )
    assert reason is not None
    assert "That reference" in reason


def test_a_finished_reference_does_not_block_the_mesh_stage():
    assert (
        create_stages.available("mesh", job(stage="reference", files=["input.png"])) is None
    )


def test_an_unknown_stage_is_a_programming_error():
    with pytest.raises(ValueError):
        create_stages.available("texture", job())


# --- go ---------------------------------------------------------------------


def test_the_stage_starts_at_the_front_of_the_pipeline():
    assert AppState().create_stage == create_stages.STAGES[0]


def test_go_is_the_only_thing_that_writes_the_stage():
    """A source scan, ``test_mode_writes``'s idiom and for its reason: the
    switch has obligations (move the selection, guard an unsaved pose) that a
    bare assignment silently skips."""
    import pathlib
    import re

    # An *assignment*, so a comparison (``!=``, ``==``) is not mistaken for one.
    write = re.compile(r"\.create_stage\s*=(?!=)")
    root = pathlib.Path(create_stages.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "create_stages.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if write.search(line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


def test_go_records_where_it_arrived():
    ctx = FakeCtx()
    create_stages.go(ctx, "mesh")
    assert ctx.state.create_stage == "mesh"


def test_go_switches_the_mode_and_leaves_escape_a_way_back():
    ctx = FakeCtx()
    ctx.state.mode = "home"
    create_stages.go(ctx, "reference")
    assert ctx.state.mode == create_stages.MODE
    assert ctx.state.previous_mode == "home"


def test_go_with_an_explicit_selection_takes_it_along():
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    ctx = FakeCtx([ref])
    create_stages.go(ctx, "reference", select="aaaaaaaaaaaa")
    assert ctx.state.selected == "aaaaaaaaaaaa"


def test_walking_back_to_reference_leaves_the_mesh_selected():
    """Reference can show a mesh, so there is nothing to move -- and moving it
    would strand the inspector on a different asset than the viewport."""
    mesh = job(id="bbbbbbbbbbbb")
    ctx = FakeCtx([mesh], selected="bbbbbbbbbbbb")
    create_stages.go(ctx, "reference")
    assert ctx.state.selected == "bbbbbbbbbbbb"


def test_walking_forward_follows_the_promotion_edge():
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    mesh = job(id="bbbbbbbbbbbb", parent_id="aaaaaaaaaaaa")
    ctx = FakeCtx([ref, mesh], selected="aaaaaaaaaaaa")
    create_stages.go(ctx, "mesh")
    assert ctx.state.selected == "bbbbbbbbbbbb"


def test_the_newest_promotion_wins():
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    old = job(id="bbbbbbbbbbbb", parent_id="aaaaaaaaaaaa", created_at="2026-01-01T00:00:00")
    new = job(id="cccccccccccc", parent_id="aaaaaaaaaaaa", created_at="2026-02-01T00:00:00")
    ctx = FakeCtx([ref, old, new], selected="aaaaaaaaaaaa")
    create_stages.go(ctx, "mesh")
    assert ctx.state.selected == "cccccccccccc"


def test_walking_forward_with_no_mesh_yet_leaves_the_selection_alone():
    """The mesh stage is a *form* -- arriving with the reference still selected
    is how you fill it in. Inventing a selection here would silently retarget
    the promotion at some unrelated asset."""
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    ctx = FakeCtx([ref], selected="aaaaaaaaaaaa")
    create_stages.go(ctx, "mesh")
    assert ctx.state.selected == "aaaaaaaaaaaa"


def test_go_refuses_a_stage_that_does_not_exist():
    with pytest.raises(ValueError):
        create_stages.go(FakeCtx(), "texture")


def test_a_stage_is_only_the_thing_on_screen_while_create_is_the_mode():
    """``create_stage`` is not cleared on the way out -- coming back to Create
    from Inker should land where you left. So the panes' gate has to ask both
    halves, or the inspector in Poser grows the Mesh stage's quality section."""
    state = AppState()
    state.mode = create_stages.MODE
    state.create_stage = "mesh"
    assert create_stages.at(state, "mesh") is True
    assert create_stages.in_create(state) is True

    state.mode = "poser"
    assert create_stages.at(state, "mesh") is False
    assert create_stages.in_create(state) is False


def test_follow_false_leaves_the_selection_where_it_was():
    """The two arrivals that are about to *make* something -- a dropped image
    and a promotion -- carry their own source. Walking onto a mesh this
    reference already has would describe the wrong asset."""
    ref = job(id="aaaaaaaaaaaa", stage="reference", files=["input.png"])
    mesh = job(id="bbbbbbbbbbbb", parent_id="aaaaaaaaaaaa")
    ctx = FakeCtx([ref, mesh], selected="aaaaaaaaaaaa")
    create_stages.go(ctx, "mesh", follow=False)
    assert ctx.state.selected == "aaaaaaaaaaaa"
    assert ctx.state.create_stage == "mesh"


def test_a_ctx_with_no_job_cache_can_still_switch_stage():
    """The palette and the profile sheet both call ``go`` with whatever ctx
    they were handed, and a stage switch is not the place to require a cache."""
    from types import SimpleNamespace

    state = AppState()
    create_stages.go(SimpleNamespace(state=state), "mesh")
    assert state.mode == create_stages.MODE
    assert state.create_stage == "mesh"


# --- the vocabulary boundary ------------------------------------------------


def test_no_service_module_imports_the_stage_names():
    """``mesh`` is a UI word and ``model`` is a corpus word, and the corpus is
    keyed on its own. A service module that imported this one would be one
    refactor away from writing ``"mesh"`` into a verdict row, where nothing
    would ever read it again."""
    import pathlib

    root = pathlib.Path(create_stages.__file__).resolve().parents[1] / "service"
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        if "create_stages" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []

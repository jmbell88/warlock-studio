"""Phase 5: the mode. What it lists, what it plays, and what it does not hold.

Three claims carry these:

* **the preview is a clock**, so a slow frame skips cells rather than falling
  behind, and two machines running at different frame rates play a run cycle at
  the same speed;
* **which cell is on screen comes from the frame table**, never from arithmetic
  over the animation lengths -- a third copy of that arithmetic is a third thing
  nobody owns;
* **there is no document**, which is why the mode registers no journal provider
  and no palette Save, and why entering it from Home creates nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pygame
import pytest

from warlock import rigging
from warlock.pipelines import charsheet
from warlock.studio import modes as modes_mod
from warlock.studio import troupe_mode
from warlock.studio.troupe import spec as troupe_spec


class _Ctx:
    """The narrow slice of the app context this mode's logic touches.

    Hand-rolled rather than the ``app_ctx`` fixture, for the reason the headless
    tests in this directory all take: none of what is asserted here needs a GL
    context, and a test that opened one to check a clock would be untrue about
    what it is testing.
    """

    def __init__(self, svc):
        self.svc = svc
        self.cache = svc.store
        self.state = SimpleNamespace(troupe=None, preview={}, mode="troupe")
        self.viewer = None
        self.toasts: list[tuple[str, str]] = []

    def job_dir(self, job_id):
        return self.svc.job_dir(job_id)

    def toast(self, text, level="info", *a, **k):
        self.toasts.append((text, level))

    def busy(self, key):
        return False


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)


def _character(svc, *, sheets=1, size=32):
    """A finished mesh with ``sheets`` character sheets and a row per sheet."""
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    made = []
    for index in range(sheets):
        sheet_id = rigging.new_id()
        path = rigging.sheet_path(job_dir, sheet_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "id": sheet_id,
                    "columns": charsheet.COLUMNS,
                    "rows": 32,
                    "frame_size": size + index * 16,
                    "created": 100.0 + index,
                    "animation": charsheet.animation_block(),
                }
            ),
            "utf-8",
        )
        rigging.sheet_png_path(job_dir, sheet_id).write_bytes(b"atlas")
        row = svc.store.create(
            "charsheet", "a hooded ranger", {"source_job": job_id, "sheet_id": sheet_id}
        )
        svc.store.set_status(row, "done")
        made.append(sheet_id)
    return job_id, made


def _v2_character(svc):
    job_id, made = _character(svc)
    path = rigging.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(path.read_text("utf-8"))
    record["troupe"] = charsheet.resolve_layout(
        {
            "version": 2,
            "movements": [
                {"key": "idle", "frames": 3, "directions": 1},
                {"key": "walk", "frames": 6, "directions": 4},
            ],
        }
    ).as_dict()
    path.write_text(json.dumps(record), "utf-8")
    return job_id, made


def test_v2_preview_uses_the_selected_sheets_movements_and_runs(ctx, svc):
    job_id, made = _v2_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    state = troupe_mode.ensure(ctx)
    state.animation, state.direction, state.frame = "walk", "back", 5
    assert troupe_mode.cell_index(ctx) == 20
    assert int(troupe_mode.preview_movement(ctx)["frames"]) == 6


def test_v2_selection_reconciles_a_movement_missing_from_the_sheet(ctx, svc):
    state = troupe_mode.ensure(ctx)
    state.animation, state.direction, state.frame = "jump", "back_right", 5
    job_id, made = _v2_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    assert (state.animation, state.direction, state.frame) == ("idle", "front", 0)


# -- the cast ----------------------------------------------------------------


def test_the_cast_is_the_meshes_that_have_a_character_sheet(ctx, svc):
    """Not the asset library: "what have I made" and "what can I animate" are
    different questions, and a hundred barrels are the right answer to one and
    noise in the other."""
    job_id, _sheets = _character(svc)
    svc.store.create("image", "a barrel", {}, stage="model")

    cast = troupe_mode.characters(ctx)
    assert [c["id"] for c in cast] == [job_id]


def test_a_character_appears_once_however_many_sheets_it_has(ctx, svc):
    job_id, sheets = _character(svc, sheets=3)
    assert len(sheets) == 3
    assert [c["id"] for c in troupe_mode.characters(ctx)] == [job_id]


def test_an_unfinished_sheet_puts_nobody_in_the_cast(ctx, svc):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    svc.store.create("charsheet", "a ranger", {"source_job": job_id})
    assert troupe_mode.characters(ctx) == []


def test_a_character_survives_its_source_falling_off_the_library_page(ctx, svc):
    """``ctx.cache`` answers about the library's *currently loaded page*, and
    this used to ``continue`` on a miss -- so a finished character silently
    vanished from the cast as soon as its mesh scrolled out of a list that has
    nothing to do with this mode. The charsheet row carries the prompt itself,
    because every charsheet door mints with ``source["prompt"]``."""
    job_id, _sheets = _character(svc)
    ctx.cache = SimpleNamespace(get=lambda _job_id: None, jobs=[])

    cast = troupe_mode.characters(ctx)
    assert [c["id"] for c in cast] == [job_id]
    assert cast[0]["prompt"], "the title comes off the charsheet row"


# -- characters that are still on their way ----------------------------------


def _troupe_reference(svc, *, status="done", prompt="a fire guardian"):
    # Status at creation: ``store.finish`` only transitions a *running* row.
    return svc.store.create(
        "text",
        prompt,
        {"troupe": {"logical_size": 32}, "rig": True},
        stage="reference",
        status=status,
    )


def test_a_submitted_character_shows_as_waiting_before_it_is_approved(ctx, svc):
    """The row that was missing. Submitting here hands off to Create, so a user
    who came back before approving the drawing was told "No character sheets
    yet" -- about the character they had just started."""
    job_id = _troupe_reference(svc)

    pending = troupe_mode.in_progress(ctx)
    assert [p["id"] for p in pending] == [job_id]
    assert pending[0]["waiting"] is True
    assert "Approve" in pending[0]["phase"]


def test_a_character_still_drawing_offers_nothing_to_press(ctx, svc):
    """The gate is the only link the *user* can be blocking on."""
    _troupe_reference(svc, status="queued")
    assert troupe_mode.in_progress(ctx)[0]["waiting"] is False


def test_an_approved_character_says_it_is_building(ctx, svc):
    job_id = _troupe_reference(svc)
    svc.store.create("image", "a fire guardian", {}, stage="model", parent_id=job_id)

    pending = troupe_mode.in_progress(ctx)
    assert pending[0]["waiting"] is False
    assert "mesh" in pending[0]["phase"]


def test_a_finished_character_is_not_listed_twice(ctx, svc):
    """Once the sheet exists the cast holds it, and saying it is on its way
    would contradict the preview about to play it."""
    ref_id = _troupe_reference(svc)
    mesh_id = svc.store.create(
        "image", "a fire guardian", {}, stage="model", parent_id=ref_id
    )
    svc.store.create(
        "charsheet", "a fire guardian", {"source_job": mesh_id}, status="done"
    )

    assert troupe_mode.in_progress(ctx) == []


def test_a_failed_reference_is_left_to_the_library(ctx, svc):
    """Repeating it here would be a second account of one failure, in a sidebar
    that cannot act on it."""
    _troupe_reference(svc, status="error")
    assert troupe_mode.in_progress(ctx) == []


def test_a_promoted_character_names_the_link_it_is_actually_on(ctx, svc):
    """The branch that used to be a constant. Once a mesh existed the sidebar
    said "Building the mesh, rig and sheet..." whatever had happened since, so
    the rig and the render -- the two long links -- were indistinguishable."""
    ref_id = _troupe_reference(svc)
    mesh_id = svc.store.create(
        "image", "a fire guardian", {}, stage="model", parent_id=ref_id, status="done"
    )

    # The window between the mesh landing and the follow-up being minted is
    # real, and is the one phase that still names the whole remainder.
    assert troupe_mode.in_progress(ctx)[0]["phase"] == troupe_mode._QUEUEING

    rig_id = svc.store.create("rig", "a fire guardian", {"source_job": mesh_id})
    assert troupe_mode.in_progress(ctx)[0]["phase"] == troupe_mode._RIGGING

    svc.store.set_status(rig_id, "done")
    svc.store.create("charsheet", "a fire guardian", {"source_job": mesh_id})
    assert troupe_mode.in_progress(ctx)[0]["phase"] == troupe_mode._RENDERING


def test_a_character_whose_mesh_is_still_running_says_so(ctx, svc):
    ref_id = _troupe_reference(svc)
    svc.store.create(
        "image", "a fire guardian", {}, stage="model", parent_id=ref_id, status="running"
    )
    assert troupe_mode.in_progress(ctx)[0]["phase"] == troupe_mode._BUILDING_MESH


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_a_broken_chain_stops_claiming_to_be_building(ctx, svc, status):
    """**The bug this closes.** ``in_progress`` read the reference's own status
    and whether it had children, so a failed mesh left the sidebar saying
    "Building the mesh, rig and sheet..." for ever. Dropped rather than
    relabelled, which is what the failed-reference path beside it already does:
    the library's failure card owns a failed row, with the error text and the
    reroll."""
    ref_id = _troupe_reference(svc)
    svc.store.create(
        "image", "a fire guardian", {}, stage="model", parent_id=ref_id, status=status
    )
    assert troupe_mode.in_progress(ctx) == []


@pytest.mark.parametrize("kind", ["rig", "charsheet"])
def test_a_failed_follow_up_stops_the_chain_too(ctx, svc, kind):
    """The mesh landed and the link after it did not. Same answer, one link
    further down, and it was the same silent "building" before."""
    ref_id = _troupe_reference(svc)
    mesh_id = svc.store.create(
        "image", "a fire guardian", {}, stage="model", parent_id=ref_id, status="done"
    )
    svc.store.create(
        kind, "a fire guardian", {"source_job": mesh_id}, status="error"
    )
    assert troupe_mode.in_progress(ctx) == []


def test_a_recorded_follow_up_failure_stops_the_chain(ctx, svc):
    """The follow-up was never queued at all, so there is no row to read. The
    stamp on the mesh is the only record that it is never coming."""
    from warlock import followups

    ref_id = _troupe_reference(svc)
    mesh_id = svc.store.create(
        "image", "a fire guardian", {}, stage="model", parent_id=ref_id, status="done"
    )
    followups.persist(svc.store, mesh_id, "rig", "no rig template for that mesh")

    assert troupe_mode.in_progress(ctx) == []


def test_a_live_reroll_beside_a_dead_one_is_still_building(ctx, svc):
    """A reroll leaves the dead row in place, so "any dead follow-up" would be
    the wrong test -- it would call a working chain broken."""
    ref_id = _troupe_reference(svc)
    mesh_id = svc.store.create(
        "image", "a fire guardian", {}, stage="model", parent_id=ref_id, status="done"
    )
    svc.store.create(
        "rig", "a fire guardian", {"source_job": mesh_id}, status="error"
    )
    svc.store.create("rig", "a fire guardian", {"source_job": mesh_id})

    assert troupe_mode.in_progress(ctx)[0]["phase"] == troupe_mode._RIGGING


def test_the_mesh_picker_does_not_walk_the_library_every_frame(ctx, svc):
    """``sendable_meshes``' predicate reads ``files``, which ``attach_files``
    fills at one stat per listed name per row -- its own docstring calls that
    the frame loop's single largest syscall cost and asks the caller to own a
    cache. The picker owned none and asked per frame, for as long as its header
    was open."""
    _plain_mesh(svc)
    reads = []
    original = svc.store.list

    def counted(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)

    svc.store.list = counted
    try:
        first = troupe_mode.sendable_meshes(ctx)
        for _ in range(30):
            troupe_mode.sendable_meshes(ctx)
    finally:
        svc.store.list = original

    assert len(reads) == 1
    assert troupe_mode.sendable_meshes(ctx) == first


def test_a_finished_job_drops_the_picker_cache_rather_than_waiting_it_out(ctx, svc):
    """The interval is there to stop idle polling, not to delay news the pane
    already has -- ``invalidate_cast``'s rule, for the list beside it."""
    _plain_mesh(svc)
    troupe_mode.sendable_meshes(ctx)
    assert troupe_mode.ensure(ctx).sendable_cache is not None

    troupe_mode.invalidate_sendable(ctx)
    assert troupe_mode.ensure(ctx).sendable_cache is None

def test_only_troupe_references_are_listed(ctx, svc):
    svc.store.create("text", "a barrel", {}, stage="reference")
    assert troupe_mode.in_progress(ctx) == []


def test_a_trashed_character_is_not_work_you_still_owe(ctx, svc):
    """``store.list`` does not filter the trash, so without this a character
    the user threw away came back as an unfinished one, forever."""
    job_id = _troupe_reference(svc)
    svc.store.set_deleted_if_not_running(job_id, 1.0)
    assert troupe_mode.in_progress(ctx) == []


def test_a_trashed_character_is_not_in_the_cast(ctx, svc):
    """Trashing a mesh does not cascade to the sheets made from it, so its
    charsheet rows stay live. The library page used to hide that by accident;
    dropping the page dependency dropped the accident with it."""
    job_id, _sheets = _character(svc)
    svc.store.set_deleted_if_not_running(job_id, 1.0)
    ctx.cache = SimpleNamespace(get=svc.store.get, jobs=[])

    assert troupe_mode.characters(ctx) == []


def test_the_pending_list_is_capped(ctx, svc):
    for index in range(troupe_mode.MAX_IN_PROGRESS + 3):
        _troupe_reference(svc, prompt=f"guardian {index}")
    assert len(troupe_mode.in_progress(ctx)) == troupe_mode.MAX_IN_PROGRESS


# -- the one door in ---------------------------------------------------------


def test_open_sheet_enters_the_mode_pointed_at_the_sheet(ctx, svc):
    job_id, sheets = _character(svc)
    ctx.state.mode = "library"

    assert troupe_mode.open_sheet(ctx, job_id, sheets[0]) is True
    assert ctx.state.mode == "troupe"
    assert troupe_mode.ensure(ctx).sheet_id == sheets[0]


def test_open_sheet_refuses_rather_than_switching_into_an_empty_room(ctx, svc):
    """Troupe would draw "No character on screen", which is the blank arrival
    the whole routing change exists to stop."""
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    ctx.state.mode = "library"

    assert troupe_mode.open_sheet(ctx, job_id) is False
    assert ctx.state.mode == "library"


def test_only_character_sheets_are_listed_under_a_character(ctx, svc):
    """A mesh can also hold ordinary pose sheets. They have no animation block,
    no direction runs and nothing this mode can play."""
    job_id, sheets = _character(svc)
    plain = rigging.new_id()
    rigging.sheet_path(svc.job_dir(job_id), plain).write_text(
        json.dumps({"id": plain, "columns": 8, "rows": 1, "frame_size": 64}), "utf-8"
    )

    listed = [r["id"] for r in troupe_mode.sheets(ctx, job_id)]
    assert listed == sheets
    assert plain not in listed


def test_the_sheet_directory_is_not_re_read_every_frame(ctx, svc, monkeypatch):
    """``sheets`` is a glob plus a stat plus a read plus a JSON parse per sheet
    and the cast pane calls it from its draw; ``active_sheet`` is another read
    and three panes ask for it. Between them Troupe hit the disk three or four
    times a frame for a directory that changes when a sheet is *built*."""
    from warlock import rigging as rigging_mod

    job_id, listed = _character(svc, sheets=2)
    troupe_mode.select(ctx, job_id)

    globs: list[int] = []
    reads: list[int] = []
    real_list, real_read = rigging_mod.list_sheets, rigging_mod.read_sheet
    monkeypatch.setattr(
        rigging_mod,
        "list_sheets",
        lambda d: (globs.append(1), real_list(d))[1],
    )
    monkeypatch.setattr(
        rigging_mod,
        "read_sheet",
        lambda d, i: (reads.append(1), real_read(d, i))[1],
    )

    for _ in range(10):
        troupe_mode.sheets(ctx, job_id)
        troupe_mode.active_sheet(ctx)
    assert globs == [] and reads == [], "the selection already read both"

    # And a selection is read at once rather than waited out, which is the half
    # a bare interval would get wrong.
    troupe_mode.select(ctx, job_id, listed[0])
    assert len(globs) == 1
    assert troupe_mode.active_sheet(ctx)["id"] == listed[0]
    settled = len(reads)
    assert settled, "a selection reads at once rather than waiting out the interval"
    for _ in range(10):
        troupe_mode.active_sheet(ctx)
    assert len(reads) == settled, "and then stops"


def test_selecting_a_character_takes_its_newest_sheet(ctx, svc):
    job_id, sheets = _character(svc, sheets=3)
    troupe_mode.select(ctx, job_id)
    state = troupe_mode.ensure(ctx)
    assert state.sheet_id == sheets[-1]  # newest by ``created``


def test_selecting_resets_the_clock(ctx, svc):
    """Carried across a selection it would show the new character mid-stride at
    whatever frame the old one was on, which reads as a rendering fault."""
    job_id, _sheets = _character(svc)
    state = troupe_mode.ensure(ctx)
    state.frame, state.clock = 5, 0.4
    troupe_mode.select(ctx, job_id)
    assert (state.frame, state.clock) == (0, 0.0)


# -- the clock ---------------------------------------------------------------


def _table():
    return troupe_spec.load()


def test_a_walk_advances_one_frame_per_its_own_duration(ctx):
    state = troupe_mode.ensure(ctx)
    state.playing = True
    state.animation = "walk"
    walk = _table().animation("walk")
    troupe_mode.advance(ctx, walk.duration_ms / 1000.0)
    assert state.frame == 1


def test_the_frame_rate_does_not_change_the_playback_speed(ctx):
    """The whole reason this is a clock. Sixty small steps and six big ones
    covering the same wall-clock time have to land on the same frame."""
    fast = troupe_mode.ensure(ctx)
    fast.animation = "walk"
    for _ in range(60):
        troupe_mode.advance(ctx, 1.0 / 60.0)
    quick = fast.frame

    other = _Ctx(ctx.svc)
    slow = troupe_mode.ensure(other)
    slow.animation = "walk"
    for _ in range(6):
        troupe_mode.advance(other, 1.0 / 6.0)
    assert slow.frame == quick


def test_a_stalled_frame_skips_cells_rather_than_falling_behind(ctx):
    state = troupe_mode.ensure(ctx)
    state.playing = True
    state.animation = "walk"
    walk = _table().animation("walk")
    troupe_mode.advance(ctx, walk.duration_ms / 1000.0 * 3)
    assert state.frame == 3


def test_a_cycle_loops_and_a_one_shot_holds_its_last_frame(ctx):
    """A cycle wraps and a one-shot holds, once something is playing at all.

    Playback is armed here rather than assumed: the preview opens paused (see
    ``TroupeState.playing``), so what this pins is the looping rule and not the
    default.
    """
    state = troupe_mode.ensure(ctx)
    state.playing = True
    for name in ("walk", "attack"):
        troupe_mode.set_animation(ctx, name)
        animation = _table().animation(name)
        troupe_mode.advance(ctx, animation.duration_ms / 1000.0 * (animation.frames + 2))
        if animation.loop:
            assert state.frame < animation.frames
        else:
            assert state.frame == animation.frames - 1


def test_a_paused_preview_does_not_move(ctx):
    state = troupe_mode.ensure(ctx)
    state.playing = False
    troupe_mode.advance(ctx, 10.0)
    assert state.frame == 0


def test_stepping_pauses(ctx):
    state = troupe_mode.ensure(ctx)
    state.playing = True
    troupe_mode.step(ctx, 1)
    assert not state.playing and state.frame == 1


def test_stepping_back_from_zero_wraps_to_the_last_frame(ctx):
    state = troupe_mode.ensure(ctx)
    state.animation = "walk"
    troupe_mode.step(ctx, -1)
    assert state.frame == _table().animation("walk").frames - 1


def test_changing_animation_restarts_and_changing_direction_does_not(ctx):
    """Turning a character mid-stride should show the same frame from the other
    side; changing what it is *doing* should not."""
    state = troupe_mode.ensure(ctx)
    state.frame = 3
    troupe_mode.set_direction(ctx, "left")
    assert state.frame == 3
    troupe_mode.set_animation(ctx, "run")
    assert state.frame == 0


# -- which cell --------------------------------------------------------------


def test_the_cell_comes_from_the_frame_table(ctx):
    """Against ``charsheet``'s copy rather than a number written here: the
    agreement between the two tables has exactly one owner, and a literal in
    this test would be a third."""
    state = troupe_mode.ensure(ctx)
    for cell in charsheet.frame_table():
        state.animation, state.direction, state.frame = (
            cell.animation,
            cell.direction,
            cell.frame,
        )
        assert troupe_mode.cell_index(ctx) == cell.index


def test_a_frame_past_the_end_of_a_clip_is_no_cell_rather_than_a_wrong_one(ctx):
    state = troupe_mode.ensure(ctx)
    state.animation, state.direction = "walk", "front"
    state.frame = 99
    assert troupe_mode.cell_index(ctx) is None


# -- registration ------------------------------------------------------------


def test_the_mode_is_registered_everywhere_the_dispatch_needs_it():
    """``_build_ui``'s dispatch ends in a bare ``else`` that draws Inker, so an
    unregistered mode silently draws the wrong workspace."""
    from warlock.studio.panes import overlay

    assert "troupe" in modes_mod.KEYS
    assert "troupe" in modes_mod.WORK_MODES
    assert "troupe" in modes_mod.WORKSPACE_MODES
    assert "troupe" in modes_mod.NAV_KEY_MODES
    assert any("troupe" in group for group in modes_mod.RAIL_GROUPS)
    assert "troupe" in overlay.PLACEHOLDERS


def test_the_rail_list_is_still_the_flattening_of_its_groups():
    flat = [key for group in modes_mod.RAIL_GROUPS for key in group]
    assert flat == list(modes_mod.KEYS)


def test_troupe_holds_no_document_and_says_so_by_omission():
    """Not an oversight, and asserted so that adding one is a deliberate act:
    Troupe is a selection over sheets a worker published, so a Save command
    would have nothing to write and a journal provider nothing to recover."""
    from warlock.studio import journal, palette

    assert "troupe" not in palette._DOC_MODES
    assert "troupe_mode" not in journal._PROVIDER_MODULES


def test_entering_from_home_creates_nothing(ctx, svc):
    """Unlike the four document modes: entering Plotter *was* the act of
    creating a map, silently and at whatever the default happened to be."""
    from warlock.studio.panes import landing

    before = len(svc.store.list())
    landing.start_troupe(ctx)
    assert ctx.state.mode == "troupe"
    assert len(svc.store.list()) == before


def _press(key):
    """One KEYDOWN. ``handle_key`` is presses-only by contract."""
    return SimpleNamespace(type=pygame.KEYDOWN, key=key, mod=0)


def test_every_reserved_nav_key_does_something_in_troupe():
    """**The assertion that catches the whole class.** ``NAV_KEY_MODES``
    membership withholds all nine of ``imgui_backend._NAV_KEYS`` from imgui
    while Troupe is up, and ``main._shortcut``'s Troupe arm returns whatever
    ``handle_key`` answered -- so a reserved key this mode does not bind is
    taken from one consumer and given to none. Six of the nine were dead that
    way: Up, Down, PageUp, PageDown, Home and End.

    Asserted over the reserved set rather than over a list written here, so a
    tenth key joining ``_NAV_KEYS`` fails until somebody decides what it means
    in Troupe."""
    import inspect

    from warlock.studio import imgui_backend, modes

    assert "troupe" in modes.NAV_KEY_MODES
    source = inspect.getsource(troupe_mode.handle_key)
    # By the constant's own name, resolved off ``pygame`` -- ``key.name`` gives
    # "page up" with a space, which no source line spells.
    names = {
        value: attr
        for attr in dir(pygame)
        if attr.startswith("K_") and isinstance(value := getattr(pygame, attr), int)
    }
    missing = sorted(
        names.get(key, str(key))
        for key in imgui_backend._NAV_KEYS
        if names.get(key, "") not in source
    )
    assert not missing, f"reserved but unbound in Troupe: {missing}"


def test_up_and_down_walk_the_directions_and_wrap(ctx, svc):
    """Through ``set_direction``, so the frame is held -- the manual promises
    you can turn the character mid-stride and see the same moment."""
    _v2_character(svc)
    state = troupe_mode.ensure(ctx)
    names = [
        d["key"] for d in (troupe_mode.preview_movement(ctx) or {})["directions"]
    ]
    state.direction, state.frame = names[0], 1

    for expected in names[1:] + names[:1]:
        assert troupe_mode.handle_key(ctx, _press(pygame.K_DOWN)) is True
        assert state.direction == expected
    assert state.frame == 1, "turning must not move the frame"

    troupe_mode.handle_key(ctx, _press(pygame.K_UP))
    assert state.direction == names[-1]


def test_the_page_keys_change_animation_and_restart_the_clip(ctx, svc):
    """The documented difference from a direction change, and the reason the
    two pairs are different keys."""
    _v2_character(svc)
    state = troupe_mode.ensure(ctx)
    names = [m["key"] for m in troupe_mode.preview_layout(ctx)["movements"]]
    state.animation, state.frame, state.clock = names[0], 3, 0.4

    assert troupe_mode.handle_key(ctx, _press(pygame.K_PAGEDOWN)) is True
    assert state.animation == names[1]
    assert (state.frame, state.clock) == (0, 0.0)


def test_home_and_end_land_on_the_ends_of_the_run_and_pause(ctx, svc):
    _v2_character(svc)
    state = troupe_mode.ensure(ctx)
    state.playing = True

    assert troupe_mode.handle_key(ctx, _press(pygame.K_END)) is True
    frames = int(troupe_mode.preview_movement(ctx)["frames"])
    assert state.frame == frames - 1
    assert state.playing is False

    assert troupe_mode.handle_key(ctx, _press(pygame.K_HOME)) is True
    assert state.frame == 0


def test_a_key_on_a_sheet_with_no_layout_is_consumed_and_does_nothing(ctx):
    """An invalid v2 snapshot resolves to a layout with no movements. A press
    must not invent a direction the sheet does not have."""
    state = troupe_mode.ensure(ctx)
    state.job_id, state.sheet_id = "", ""
    before = (state.animation, state.direction)

    for key in (pygame.K_UP, pygame.K_DOWN, pygame.K_PAGEUP, pygame.K_PAGEDOWN):
        assert troupe_mode.handle_key(ctx, _press(key)) is True
    assert (state.animation, state.direction) == before

def test_a_key_release_never_acts_twice(ctx):
    """``handle_key`` used to read ``event.key`` without looking at
    ``event.type``, so every binding ran on the press *and* on the release:
    Space toggled play and toggled it straight back, and a tap of Right stepped
    two frames."""
    import pygame

    state = troupe_mode.ensure(ctx)
    was = state.playing
    down = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_SPACE, mod=0)
    up = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_SPACE, mod=0)
    assert troupe_mode.handle_key(ctx, down) is True
    assert state.playing is not was
    assert troupe_mode.handle_key(ctx, up) is False
    assert state.playing is not was, "a release must not undo the press"

    state.frame = 0
    right = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    assert troupe_mode.handle_key(ctx, right) is True
    stepped = state.frame
    release = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_RIGHT, mod=0)
    assert troupe_mode.handle_key(ctx, release) is False
    assert state.frame == stepped, "a tap of Right stepped two frames"


# --- W0.3: two controls that were wired to nothing --------------------------


def _preview_source() -> str:
    import pathlib

    from warlock.studio.panes import troupe_preview

    return pathlib.Path(troupe_preview.__file__).read_text(encoding="utf-8")


def test_the_centre_pane_takes_the_wheel_and_now_gives_it_to_something() -> None:
    """``no_scroll_with_mouse`` said the wheel belonged to the zoom control.
    No Troupe pane read the wheel, so it belonged to nothing and turning it
    over the sprite did nothing at all."""
    source = _preview_source()
    assert "io.mouse_wheel" in source
    assert "state.zoom = max(1, min(int(state.zoom + io.mouse_wheel), 32))" in source


def test_playback_speed_has_a_control_at_last() -> None:
    """``advance`` has divided the frame interval by ``state.speed`` since the
    mode was written and nothing could ever change it, so every preview played
    at exactly 1x."""
    from warlock.studio.panes import troupe_preview

    assert "##troupe-speed" in _preview_source()
    keys = [float(key) for key, _ in troupe_preview._SPEEDS]
    assert 1.0 in keys and min(keys) < 1.0 < max(keys)
    # A stored value off the ladder resolves to its nearest rung rather than
    # showing a blank combo.
    assert troupe_preview._speed_key(0.9) == "1.0"
    assert troupe_preview._speed_key(0.3) == "0.25"


def test_the_transport_is_a_measured_row_not_a_same_line_chain() -> None:
    """``same_line`` clips rather than wraps, so on a narrow centre pane the
    zoom field went off the edge with no way to reach it."""
    source = _preview_source()
    assert 'toolbar.toolbar("troupe-transport"' in source
    # Play/back/forward are the row's reason for existing: a Play collapsed
    # into an overflow menu is not a transport.
    assert source.count("pinned=True") == 3



# --- sending a mesh in ------------------------------------------------------


def _plain_mesh(svc, *, done=True, files=("model.glb",)):
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        (job_dir / name).write_bytes(b"fake-glb")
    if done:
        svc.store.set_status(job_id, "done")
    return job_id


def _row(svc, job_id):
    row = dict(svc.store.get(job_id))
    row["files"] = [p.name for p in svc.job_dir(job_id).iterdir()]
    return row


def test_the_predicate_answers_from_the_row_and_asks_for_no_rig(ctx, svc):
    """An unrigged mesh is exactly what the door is for.

    And no filesystem call: a toolbar asks this every frame, which is
    ``inker_mode.can_edit_job``'s rule.
    """
    import inspect

    job_id = _plain_mesh(svc)
    assert troupe_mode.can_send_to_troupe(ctx, _row(svc, job_id))

    # The *code*, not the prose: this file's style is to name the rejected
    # alternative, so a raw scan would fail on the docstring explaining why
    # neither of these is read.
    body = [
        line
        for line in inspect.getsource(troupe_mode.can_send_to_troupe).splitlines()
        if not line.lstrip().startswith("#")
    ]
    body = chr(10).join(body).split(chr(34) * 3)[-1]
    assert "rig.glb" not in body, "an unrigged mesh is the case this exists for"
    assert "source.glb" not in body, "the reconstruction is not the asset"
    for banned in ("job_dir", "exists()", "read_rig"):
        assert banned not in body, banned


def test_the_predicate_refuses_what_has_no_mesh_to_render(ctx, svc):
    unfinished = _plain_mesh(svc, done=False)
    assert not troupe_mode.can_send_to_troupe(ctx, _row(svc, unfinished))

    no_mesh = _plain_mesh(svc, files=("input.png",))
    assert not troupe_mode.can_send_to_troupe(ctx, _row(svc, no_mesh))

    reference = svc.store.create("image", "a drawing", {}, stage="reference")
    svc.store.set_status(reference, "done")
    assert not troupe_mode.can_send_to_troupe(ctx, dict(svc.store.get(reference)))

    trashed = _row(svc, _plain_mesh(svc))
    trashed["deleted_at"] = "2026-08-23"
    assert not troupe_mode.can_send_to_troupe(ctx, trashed)


def test_sending_submits_under_its_own_key_and_does_not_switch_mode(ctx, svc):
    """``start_character`` switches to Create because the gate is there and
    the user has something to do. Here there is nothing to do, and pulling
    somebody out of the library to watch a spinner is the opposite of the
    affordance."""
    job_id = _plain_mesh(svc)
    submitted: list[str] = []
    ctx.submit = lambda key, run, *a: (submitted.append(key), True)[1]

    assert troupe_mode.send_to_troupe(ctx, _row(svc, job_id))
    assert submitted == [f"troupe-send:{job_id}"]
    assert ctx.state.mode == "troupe"


def test_a_sheet_can_be_named_from_the_form(ctx, svc):
    """**The whole path existed except the field.** The door validates
    ``name`` against ``rigging.MAX_SHEET_NAME``, the worker writes it into the
    sidecar and the chooser reads it back -- and ``build_sheet`` passed none,
    so every sheet a character had was "sheet - 32px" and two builds at one
    size were two identical rows."""
    import inspect

    source = inspect.getsource(troupe_mode.build_sheet)
    assert "name=" in source, "build_sheet must carry the form's name to the door"

    from warlock.studio.panes import troupe_settings

    pane = inspect.getsource(troupe_settings)
    assert '"name"' in pane, "the form needs a name field for build_sheet to carry"
    assert "MAX_SHEET_NAME" in pane, "the field must cap at what the door accepts"


def test_the_send_door_still_carries_every_parameter_it_validates(ctx, svc):
    """``elevation`` and ``lighting`` have no control yet. The read stays, so
    adding one is a pane change -- which is the state ``name`` was in until its
    field landed."""
    import inspect

    source = inspect.getsource(troupe_mode.send_to_troupe)
    for field in ("elevation", "lighting", "name"):
        assert f"{field}=" in source, field

def test_the_camera_preset_reaches_build_sheet_as_elevation(ctx, svc):
    """**The form holds a name and every door takes a number.**

    A preset is a name for an elevation, and the translation happens once, on
    the way out of the mode -- so a combo reading "Side" cannot reach a door
    that renders at 30 degrees because nobody converted it. An elevation set
    explicitly still wins, because an angle off the ladder has to stay
    expressible or the ladder becomes the only thing anyone can render.
    """
    captured: dict = {}
    ctx.submit = lambda key, fn, *a, **kw: (captured.update(kw), True)[1]
    ctx.state.clear_field_errors = lambda: None
    angles = {key: angle for key, _label, angle in charsheet.CAMERA_PRESETS}

    for key, angle in angles.items():
        captured.clear()
        assert troupe_mode.build_sheet(ctx, "a-job", {"camera": key})
        assert captured["elevation"] == angle, key

    captured.clear()
    troupe_mode.build_sheet(ctx, "a-job", {"camera": "side", "elevation": 42.0})
    assert captured["elevation"] == 42.0, "a custom angle must stay expressible"

    # A form from before the control existed names no camera, and the door's
    # own default is what answers -- not a number invented here.
    captured.clear()
    troupe_mode.build_sheet(ctx, "a-job", {})
    assert captured["elevation"] is None


def test_the_picker_never_points_the_mode_at_a_bare_mesh(ctx, svc):
    """The trap this is written around.

    ``select`` accepts any job id and ``sheets()`` returns [] for a mesh, so
    pointing the mode at one lands on the blank arrival ``open_sheet``'s False
    return exists to prevent. The picker holds a local choice instead, so
    ``TroupeState`` gains no field.
    """
    import inspect

    source = inspect.getsource(troupe_mode.send_to_troupe)
    assert "select(" not in source
    assert not hasattr(troupe_mode.ensure(ctx), "send_mesh")

    job_id = _plain_mesh(svc)
    assert troupe_mode.sendable_meshes(ctx)[0]["id"] == job_id
    assert troupe_mode.sheets(ctx, job_id) == []
    assert troupe_mode.open_sheet(ctx, job_id) is False


def test_a_sent_mesh_shows_its_chain_rather_than_the_empty_state(ctx, svc):
    """Without this the door is a multi-minute silent CPU spend, in front of
    the exact "No character sheets yet" state ``in_progress`` exists to
    eliminate."""
    job_id = _plain_mesh(svc)
    rig_id = svc.store.create(
        "rig", "a hooded ranger", {"source_job": job_id, "troupe_sheet": {}}
    )

    rows = troupe_mode.in_progress(ctx)
    assert [item["id"] for item in rows] == [job_id]
    assert rows[0]["phase"] == "Rigging the mesh..."
    # No human gate on this path, so nothing here is ever waiting on the user
    # -- which is what keeps the gate the only phase that offers a button.
    assert rows[0]["waiting"] is False

    svc.store.set_status(rig_id, "done")
    svc.store.create("charsheet", "a hooded ranger", {"source_job": job_id})
    assert troupe_mode.in_progress(ctx)[0]["phase"] == "Rendering the character sheet..."


def test_a_finished_or_failed_chain_leaves_the_in_progress_list(ctx, svc):
    job_id = _plain_mesh(svc)
    failed = svc.store.create("rig", "a ranger", {"source_job": job_id})
    svc.store.set_status(failed, "error")
    assert troupe_mode.in_progress(ctx) == []

    sheet = svc.store.create("charsheet", "a ranger", {"source_job": job_id})
    svc.store.set_status(sheet, "done")
    # ``characters`` holds it now; listing it here as well would say the thing
    # it is about to draw is not ready.
    assert troupe_mode.in_progress(ctx) == []


def test_the_cap_is_on_the_reference_pass_alone(ctx, svc):
    """A reference that was never approved stays unapproved forever, which is
    what the cap is for; a rig or a charsheet row is claimed by a serial queue
    and terminates. Capping those would hide work that is genuinely running."""
    for index in range(troupe_mode.MAX_IN_PROGRESS + 3):
        job_id = _plain_mesh(svc)
        svc.store.create("rig", f"ranger {index}", {"source_job": job_id})
    assert len(troupe_mode.in_progress(ctx)) == troupe_mode.MAX_IN_PROGRESS + 3


def test_the_preview_opens_paused(ctx):
    """Overturned on request 2026-08-23, and pinned so it cannot drift back.

    A clip already moving when you arrive is one you have to stop before you
    can look at any frame in it, and looking at a frame -- a hand, a
    silhouette, which way the feet point -- is the first thing anyone does with
    a new sheet.
    """
    state = troupe_mode.ensure(ctx)
    assert state.playing is False
    troupe_mode.advance(ctx, 10.0)
    assert state.frame == 0, "a paused preview advanced on its own"


def test_a_drawing_character_is_named_by_its_own_pose(ctx, svc):
    """The phase said "T-pose" flat, and the pose became a choice on
    2026-08-23. A row drawn against the A-pose guide reporting itself as a
    T-pose is the sidebar describing a different character to the one queued.
    """
    svc.store.create(
        "text",
        "a fire guardian",
        {"troupe": {"logical_size": 32}, "rig": True, "guide_pose": "apose"},
        stage="reference",
        status="queued",
    )
    (pending,) = troupe_mode.in_progress(ctx)
    assert "A-pose" in pending["phase"], pending["phase"]
    assert "T-pose" not in pending["phase"]


def test_a_row_with_no_recorded_pose_still_reads_as_a_t_pose(ctx, svc):
    """Rows queued before the pose existed were drawn against the T-pose --
    the same fallback ``_q_generate`` applies when it redraws one."""
    _troupe_reference(svc, status="queued")
    (pending,) = troupe_mode.in_progress(ctx)
    assert "T-pose" in pending["phase"], pending["phase"]


def test_no_pane_hard_codes_a_pose_it_cannot_know(ctx):
    """The empty state promised a T-pose before the user had picked one.

    Copy that names a pose has to get it from the row or the form. Where
    neither exists yet -- the cast pane with nothing in it -- the honest word
    is "reference", so this scans the two places that talk about the drawing
    and requires the label to come from ``POSE_LABELS``.
    """
    import ast
    from pathlib import Path

    from warlock.studio.panes import troupe_characters

    for module in (troupe_characters, troupe_mode):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Docstrings argue about the poses at length and should: they are not
        # on screen. What this scans is the strings that are.
        docstrings = {
            id(node.value.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node.value) in docstrings:
                continue
            if "T-pose" not in node.value:
                continue
            # The label table itself is where the words are allowed to live.
            assert node.value == "T-pose", (
                f"{Path(module.__file__).name}:{node.lineno} names a pose in prose: "
                f"{node.value!r}"
            )


# -- the cast is read on a clock, not per frame --------------------------------


def _count_store_list(ctx, monkeypatch):
    """Count ``store.list`` calls without changing what it answers."""
    calls: list[dict] = []
    real = ctx.svc.store.list

    def counted(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr(ctx.svc.store, "list", counted)
    return calls


def test_the_cast_pane_reads_the_store_once_across_many_frames(ctx, svc, monkeypatch):
    """A call-count assertion, deliberately, rather than a timing one.

    The defect was two full job-table walks *per draw* -- one ``kind``-filtered
    and one unfiltered, both up to ``SCAN_LIMIT`` rows, both behind the store's
    single lock while the worker wants it. What matters is the number of reads
    per frame, and that is what this pins; how long a read takes belongs to the
    perf lane.
    """
    _character(svc)
    calls = _count_store_list(ctx, monkeypatch)

    for _ in range(60):
        troupe_mode.cast_and_pending(ctx)

    assert len(calls) == 2, "one page per half, once -- not once per frame"


def test_the_cast_still_answers_the_same_thing_it_did_uncached(ctx, svc, monkeypatch):
    """The throttle must not change the answer, only how often it is computed."""
    job_id, _sheets = _character(svc)
    cast, pending = troupe_mode.cast_and_pending(ctx)
    assert [c["id"] for c in cast] == [job_id]
    assert [c["id"] for c in cast] == [c["id"] for c in troupe_mode.characters(ctx)]
    assert pending == troupe_mode.in_progress(ctx)


def test_a_landed_task_drops_the_cache_rather_than_waiting_the_interval(ctx, svc):
    """The interval exists to stop idle polling, not to delay news."""
    _character(svc)
    troupe_mode.cast_and_pending(ctx)
    assert troupe_mode.ensure(ctx).cast_cache is not None

    troupe_mode.on_task_failed(ctx, SimpleNamespace(key="troupe-send:x", error="no"))
    assert troupe_mode.ensure(ctx).cast_cache is None, "the next draw re-reads"


def test_an_idle_cast_is_reread_less_often_than_a_moving_one(ctx, svc):
    """Both cadences are ``jobs_cache``'s, over the same store."""
    assert troupe_mode.CAST_REFRESH_IDLE > troupe_mode.CAST_REFRESH_LIVE
    _character(svc)
    troupe_mode.cast_and_pending(ctx)
    state = troupe_mode.ensure(ctx)
    # Nothing on the chain, so the slower of the two.
    assert state.cast_cache is not None and state.cast_cache[1] == []


# --- the scores -------------------------------------------------------------
#
# ``troupe/qa.py`` is scored in the task runner and read back by the preview.
# What these pin: one submit per selection, adoption only for the sheet still
# on screen, no toast either way, and -- the one that matters -- the pane
# never scores in the frame loop.


class _SubmitCtx(_Ctx):
    def __init__(self, svc):
        super().__init__(svc)
        self.submitted: list[tuple] = []

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))

    def busy(self, key):
        return any(entry[0] == key for entry in self.submitted)


def _png_character(svc, size=16):
    """A character whose sheet PNG is a real atlas rather than four bytes."""
    import numpy as np
    from PIL import Image

    job_id, made = _v2_character(svc)
    atlas = np.zeros((size * 4, size * charsheet.COLUMNS, 4), dtype=np.uint8)
    atlas[..., :3] = 90
    atlas[..., 3] = 255
    path = rigging.sheet_png_path(svc.job_dir(job_id), made[0])
    Image.fromarray(atlas, "RGBA").save(path)
    record_path = rigging.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(record_path.read_text("utf-8"))
    record["frame_size"] = size
    record_path.write_text(json.dumps(record), "utf-8")
    return job_id, made


def test_scores_are_submitted_once_per_selection_and_not_in_the_frame_loop(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    assert troupe_mode.scores(ctx) is None
    assert len(ctx.submitted) == 1
    key, fn, args, _kwargs = ctx.submitted[0]
    assert key == troupe_mode.scores_key(job_id, made[0])
    # Asking again while it runs submits nothing more.
    assert troupe_mode.scores(ctx) is None
    assert len(ctx.submitted) == 1
    # The task itself is honest: run it here and it scores the real atlas.
    result = fn(*args)
    assert result.cells and result.worst is None


def test_a_landed_score_is_adopted_only_for_the_sheet_still_on_screen(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    troupe_mode.scores(ctx)
    key, fn, args, _kwargs = ctx.submitted[0]
    result = fn(*args)
    stale = SimpleNamespace(key=troupe_mode.scores_key(job_id, "other"), result=result)
    troupe_mode.on_task_done(ctx, stale)
    assert troupe_mode.scores(ctx) is None
    troupe_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))
    assert troupe_mode.scores(ctx) is result
    assert ctx.toasts == []


def test_a_failed_score_is_latched_and_never_toasted(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    troupe_mode.scores(ctx)
    key = ctx.submitted[0][0]
    ctx.submitted.clear()
    troupe_mode.on_task_failed(ctx, SimpleNamespace(key=key, error="boom"))
    assert troupe_mode.scores(ctx) is None
    assert troupe_mode.scores_failed(ctx)
    assert ctx.submitted == []
    assert ctx.toasts == []


def test_selecting_another_sheet_drops_the_scores(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _png_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    ctx.state.preview["troupe_scores"] = "stale"
    ctx.state.preview["troupe_scores:key"] = (job_id, made[0])
    troupe_mode.select(ctx, job_id, "")
    assert "troupe_scores" not in ctx.state.preview


def test_goto_points_the_preview_at_a_cell_and_stops(ctx, svc):
    job_id, made = _v2_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    state = troupe_mode.ensure(ctx)
    state.animation = "walk"
    state.playing = True
    troupe_mode.goto(ctx, "back", 99)
    assert (state.direction, state.frame, state.playing) == ("back", 5, False)


def test_a_sheet_that_does_not_say_its_cell_size_is_latched_rather_than_submitted(svc):
    ctx = _SubmitCtx(svc)
    job_id, made = _v2_character(svc)
    record_path = rigging.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(record_path.read_text("utf-8"))
    record["frame_size"] = 0
    record_path.write_text(json.dumps(record), "utf-8")
    troupe_mode.select(ctx, job_id, made[0])
    assert troupe_mode.scores(ctx) is None
    assert ctx.submitted == []
    assert troupe_mode.scores_failed(ctx)


def test_the_pane_never_scores_in_the_frame_loop() -> None:
    source = _preview_source()
    assert "score_sheet" not in source
    assert "troupe_mode.scores(" in source


def test_cell_geometry_reads_a_non_square_plan_in_the_right_order():
    assert troupe_mode.cell_geometry({"columns": 8, "frame_size": 32}) == (8, 32, 32)
    assert troupe_mode.cell_geometry(
        {"columns": 4, "frame_size": 0, "frame_w": 24, "frame_h": 40}
    ) == (4, 24, 40)
    assert troupe_mode.cell_geometry({"columns": 8}) is None
    assert troupe_mode.cell_geometry(None) is None


# --- increment 7: the view marks, the structural verdict, the package --------


class _TakenCtx(_SubmitCtx):
    """``_SubmitCtx`` whose ``submit`` answers like the real runner's: True for
    a key it took. The scoring tests above never read the answer; the export
    door does, because a refused press has to be distinguishable from a taken
    one."""

    def submit(self, key, fn, *args, **kwargs):
        super().submit(key, fn, *args, **kwargs)
        return True


def test_the_pivot_marker_reads_the_cell_and_not_a_constant():
    """**The claim: ``pivot_of`` looks the cell up.** The tempting shortcut is
    ``(w / 2, h)`` -- which is what ``sheet.sidecar`` itself writes when the
    renderer measured nothing -- and it would agree with the record on most
    cells and quietly disagree on the ones that matter: a character leaning
    into an attack does not stand on the middle of its own cell."""
    record = {
        "cells": [
            {"index": 0, "w": 32, "h": 32, "pivot_x": 16.0, "pivot_y": 32.0},
            {"index": 1, "w": 32, "h": 32, "pivot_x": 9.5, "pivot_y": 28.0},
        ]
    }
    assert troupe_mode.pivot_of(record, 1) == (9.5, 28.0), "the cell, not the first"
    assert troupe_mode.pivot_of(record, 0) == (16.0, 32.0)
    # A cell the sheet does not have is not the last cell's answer either.
    assert troupe_mode.pivot_of(record, 7) is None


def test_a_cell_with_no_pivot_answers_none_rather_than_the_centre():
    """The regression the preview's marker depends on: a marker at a guessed
    origin is a lie about where the engine will place the sprite, and it looks
    exactly like a measured one."""
    assert troupe_mode.pivot_of({"cells": [{"index": 0, "w": 32, "h": 32}]}, 0) is None
    assert troupe_mode.pivot_of({"cells": [{"index": 0, "pivot_x": 4.0}]}, 0) is None
    assert troupe_mode.pivot_of(None, 0) is None
    assert troupe_mode.pivot_of({"cells": []}, None) is None


def test_c_and_p_toggle_the_view_marks_on_the_press_only(ctx, svc):
    """``handle_key`` acted on ``event.key`` without reading ``event.type``
    once already, and every binding ran twice per press. A toggle that runs
    twice is a toggle that does nothing."""
    state = troupe_mode.ensure(ctx)
    assert (state.checker, state.show_pivot) == (False, True)

    assert troupe_mode.handle_key(ctx, _press(pygame.K_c)) is True
    assert state.checker is True
    assert troupe_mode.handle_key(
        ctx, SimpleNamespace(type=pygame.KEYUP, key=pygame.K_c, mod=0)
    ) is False
    assert state.checker is True, "a release must not undo the press"

    assert troupe_mode.handle_key(ctx, _press(pygame.K_p)) is True
    assert state.show_pivot is False


def test_typing_a_c_into_a_field_is_typing_and_not_a_view_toggle(ctx, monkeypatch):
    """The rule every plain-letter shortcut in the app lives by
    (``main._passes_text_field``): a focused text field takes the plain keys.
    Naming a character "packwright" must not toggle the checkerboard four
    times on the way past -- and the arrows above are exempt because the nav
    reservation already withholds them while a field has the keyboard."""
    monkeypatch.setattr(troupe_mode, "_typing", lambda: True)
    state = troupe_mode.ensure(ctx)
    before = (state.checker, state.show_pivot)

    assert troupe_mode.handle_key(ctx, _press(pygame.K_c)) is False
    assert troupe_mode.handle_key(ctx, _press(pygame.K_p)) is False
    assert (state.checker, state.show_pivot) == before


def test_a_sheet_that_needs_repair_says_what_is_wrong_and_still_plays(ctx, svc):
    """**Structural validation is not the QA heatmap, and neither one gates.**
    ``qa.py`` ranks the drawing; ``sheetcheck`` says a cell is clipped, empty
    or was never rendered. This pins both halves of the second: the
    diagnostics are the ones ``sheetcheck.describe`` writes, and the preview
    plays the sheet exactly as it would a clean one -- a verdict the user may
    disagree with must not take their sheet away."""
    job_id, made = _v2_character(svc)
    path = rigging.sheet_path(svc.job_dir(job_id), made[0])
    record = json.loads(path.read_text("utf-8"))
    record["validation"] = {
        "version": 1,
        "ok": False,
        "clipped": [3, 4],
        "blank": [],
        "missing": [],
        "metadata": [],
        "reframed": False,
    }
    path.write_text(json.dumps(record), "utf-8")
    troupe_mode.select(ctx, job_id, made[0])
    live = troupe_mode.active_sheet(ctx)

    assert troupe_mode.needs_repair(live) is True
    notes = troupe_mode.repair_notes(live)
    assert notes and "clipped at the frame edge" in notes[0]

    state = troupe_mode.ensure(ctx)
    state.playing = True
    troupe_mode.advance(ctx, 10.0)
    assert troupe_mode.cell_index(ctx) is not None, "a flagged sheet still plays"
    assert state.playing is True, "nothing stops the clock over a verdict"


def test_an_unchecked_sheet_is_not_accused_of_anything(ctx, svc):
    """A sheet rendered before ``validation`` existed carries no block, and
    "we did not look" must not read as "we looked and it is broken"."""
    job_id, made = _v2_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    record = troupe_mode.active_sheet(ctx)
    assert "validation" not in record
    assert troupe_mode.needs_repair(record) is False
    assert troupe_mode.repair_notes(record) == []
    assert troupe_mode.needs_repair({"validation": {"ok": True}}) is False


def test_exporting_a_package_writes_the_png_and_the_sidecar_together(svc, tmp_path):
    """**The pair is the deliverable.** A folder holding the atlas without the
    JSON holds an asset nothing can interpret -- the sidecar is what says which
    cell is ``walk`` facing south-east. Submitted under its own key, so a
    second press while one is in flight is refused rather than racing it."""
    ctx = _TakenCtx(svc)
    ctx.export_dir = str(tmp_path / "project")
    job_id, made = _v2_character(svc)
    troupe_mode.select(ctx, job_id, made[0])

    assert troupe_mode.export_package(ctx) is True
    assert len(ctx.submitted) == 1
    key, run, _args, _kwargs = ctx.submitted[0]
    assert key == f"troupe-export:{job_id}:{made[0]}"
    # In flight: the second press is refused rather than queued behind it.
    assert troupe_mode.export_package(ctx) is False
    assert len(ctx.submitted) == 1

    written = run()
    assert Path(written["png"]).exists() and Path(written["json"]).exists()
    assert Path(written["png"]).parent == Path(written["json"]).parent

    troupe_mode.on_task_done(ctx, SimpleNamespace(key=key, result=written))
    said = ctx.toasts[-1][0]
    assert Path(written["png"]).name in said and Path(written["json"]).name in said


def test_a_cancelled_export_picker_is_not_reported_as_an_export(svc):
    """``dialogs.select_folder`` answers None for a cancel and nothing else,
    and a toast naming two files nobody wrote would be the app claiming a
    write it did not make."""
    ctx = _SubmitCtx(svc)
    job_id, made = _v2_character(svc)
    troupe_mode.select(ctx, job_id, made[0])
    troupe_mode.export_package(ctx)
    key = ctx.submitted[0][0]

    troupe_mode.on_task_done(ctx, SimpleNamespace(key=key, result=None))
    assert ctx.toasts == []


def test_varying_a_character_loads_its_recipe_as_the_users_own_choices(ctx, svc, monkeypatch):
    """**Every field is marked as an override, and that is the whole claim.**

    Create's character form follows the prompt for anything the user has not
    touched (``settings_character.sync_from_prompt``), and the prompt in the
    box is whatever was last typed there -- so a recipe loaded without the
    override marks would have its species, theme and camera silently rewritten
    on the next keystroke, by a brief that is not about this character. The
    press of the button *is* the touch."""
    from warlock.studio import create_stages
    from warlock.studio.panes import settings_character
    from warlock.studio.state import default_form_2d

    went: list[str] = []
    monkeypatch.setattr(create_stages, "go", lambda c, stage, **kw: went.append(stage))
    ctx.state.form_2d = default_form_2d()
    ctx.state.form_2d["prompt"] = "a completely different brief"
    record = {
        "character": {
            "family": "wyvern",
            "family_version": 3,
            "recipe": {
                "family": "wyvern",
                "family_version": 3,
                "theme": "ember",
                "camera": "isometric",
                "animations": {"walk": 8, "idle": 4},
                "logical_size": 48,
                "colors": 16,
                "appearance": {"horn": 0.75},
                "seed": 4242,
                "name": "Ash",
            },
        }
    }

    assert troupe_mode.vary_in_create(ctx, record) is True
    form = ctx.state.form_2d
    assert went == ["reference"]
    assert form["asset_type"] == "character"
    assert form["character_family"] == "wyvern"
    assert form["character_camera"] == "isometric"
    assert form["character_pixel"] == "48" and form["character_colors"] == "16"
    assert '"horn"' in form["character_body"]
    assert form["seed"] == 4242

    marked = set(settings_character.overrides_of(form))
    assert set(settings_character.RECIPE_FIELDS) <= marked, "no field follows the prompt"

    # And the proof of what the marks are for: re-resolving the brief in the
    # box leaves every one of them exactly where this put it.
    before = {key: form[key] for key in settings_character.RECIPE_FIELDS}
    settings_character.sync_from_prompt(form)
    assert {key: form[key] for key in settings_character.RECIPE_FIELDS} == before


def test_a_sheet_with_no_recipe_offers_no_variation(ctx, svc):
    """A sheet built from a supplied mesh, or from Troupe's own form, was never
    described by a ``Recipe``. Refused rather than switching into a Create form
    that describes somebody else."""
    assert troupe_mode.vary_in_create(ctx, {}) is False
    assert troupe_mode.vary_in_create(ctx, {"character": {"family": "wolf"}}) is False
    assert troupe_mode.recipe_of(None) == {}

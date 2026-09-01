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
from types import SimpleNamespace

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

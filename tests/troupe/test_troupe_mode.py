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
    state.animation = "walk"
    walk = _table().animation("walk")
    troupe_mode.advance(ctx, walk.duration_ms / 1000.0 * 3)
    assert state.frame == 3


def test_a_cycle_loops_and_a_one_shot_holds_its_last_frame(ctx):
    """A preview that stops needs a control to start it again, and the point of
    the mode is that a bad frame is obvious without pressing anything."""
    state = troupe_mode.ensure(ctx)
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
    assert state.playing
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

"""Every Muse pane, drawn -- on a machine with no GPU and no sound card.

``test_sirens_panes_smoke.py``'s arrangement and its argument: the other pane
smoke tests build a *renderer* over a real GL context and skip where there is
none, which is most CI and every remote shell, and that skip is what lets a
wrong argument order ship. Nothing here presents anything, so no GL is needed --
declaring ``renderer_has_textures`` is what lets imgui finish a frame without a
backend claiming its font atlas -- and what the frame still does is run every
widget call, every draw-list call and every layout pass for real.

The panes are drawn into a window with a **stated size**, for that file's
reason: the brief's row gives way as the pane narrows, so a default-sized window
would exercise one branch of ``_row_widths`` and never the other.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_muse_mode import FakeCtx

from warlock.studio import muse_brief, muse_mode
from warlock.studio.panes import muse_player, muse_recipe, muse_results

PANES = (
    ("muse-brief", muse_brief),
    ("muse-recipe", muse_recipe),
    ("muse-results", muse_results),
    ("muse-player", muse_player),
)

#: Wide enough that the brief draws its count control and narrow enough to be a
#: plausible window. The tray wraps its cards against this too.
WINDOW = (900.0, 700.0)

#: The width at which ``muse_brief._row_widths`` gives the count away. Well
#: under ``TEXT_MIN_W`` plus the three fixed controls, so the branch is
#: certainly taken rather than nearly taken.
NARROW = (420.0, 700.0)


@pytest.fixture
def frames():
    """A bare imgui context, built and destroyed around this file.

    The save-and-restore is ``test_pane_guard``'s discipline for its reason: at
    most one imgui context may exist at a time, and a file that wants one builds
    and destroys it rather than relying on collection order.
    """
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = WINDOW) -> None:
        imgui.new_frame()
        imgui.set_next_window_size(size)
        imgui.begin("smoke")
        try:
            build()
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()

    yield draw
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    """No pane in this file may reach the mixer. CI has no card and a box that
    has one is not something a drawing test should depend on."""
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "")


class _Cache:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs


def _ctx(tmp_path, jobs: list[dict[str, Any]] | None = None) -> FakeCtx:
    """``test_muse_mode``'s context, with the *real* ``AppState`` on it.

    That substitution is the difference between this file and that one. The
    controller tests want a stub small enough to read; these panes go through
    ``focus.begin``/``item``, which keep their ring on the app state itself --
    so a stub would exercise a focus ring that does not exist, and pass.
    """
    from warlock.studio.state import AppState

    ctx = FakeCtx(tmp_path)
    ctx.state = AppState()
    ctx.cache = _Cache(jobs or [])
    return ctx


def _take(job_id: str, status: str = "done") -> dict[str, Any]:
    return {
        "id": job_id,
        "kind": "music",
        "stage": "music",
        "status": status,
        "prompt": "dark ambient, dungeon, low strings, slow",
        "params": {"duration": 60.0, "actual_duration": 60.0},
    }


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_with_takes_in_the_tray(name, pane, frames, tmp_path):
    ctx = _ctx(tmp_path, [_take("a"), _take("b"), _take("c", status="queued")])
    frames(lambda: pane.draw(ctx))


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_on_a_first_visit(name, pane, frames, tmp_path):
    """The state does not exist yet and there are no takes.

    This is the frame a user actually sees first, and it is the one a mode
    reached through the generic per-mode loops in ``test_studio_smoke`` gets --
    so an ``ensure`` missing anywhere shows up here rather than in the app.
    """
    frames(lambda: pane.draw(_ctx(tmp_path)))


def test_the_brief_gives_the_count_away_before_it_clips_generate(frames, tmp_path):
    """The row's give-way order, exercised rather than only written down.

    ``same_line`` past the pane edge draws a control *nowhere*, so an unstated
    order does not produce a cramped row -- it produces a missing Generate,
    which is the one control the bar exists for.
    """
    ctx = _ctx(tmp_path)
    frames(lambda: muse_brief.draw(ctx), NARROW)


def test_a_playing_take_draws_as_stop(frames, tmp_path, monkeypatch):
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "playing", lambda: True)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "a")
    ctx = _ctx(tmp_path, [_take("a")])
    assert muse_mode.is_playing(ctx, "a") is True
    frames(lambda: muse_results.draw(ctx))


def test_the_tray_shows_only_this_modes_rows(tmp_path):
    """A tray that listed meshes would be a second Library.

    Not a drawing assertion: ``plan_for`` is the filter, and asserting it
    directly is what makes the claim readable.
    """
    ctx = _ctx(
        tmp_path,
        [
            {"id": "m", "kind": "text", "status": "done"},
            _take("a"),
            {"id": "r", "kind": "rig", "status": "done"},
            _take("b"),
        ],
    )
    assert [job["id"] for job in muse_results.plan_for(ctx)] == ["b", "a"]
    assert muse_results.should_draw(ctx) is True


def test_the_tray_is_newest_first(tmp_path):
    # ``ctx.cache.jobs`` is oldest-first, and a results tray that showed the
    # first take you ever made at the top would bury every press.
    ctx = _ctx(tmp_path, [_take("old"), _take("new")])
    assert [job["id"] for job in muse_results.plan_for(ctx)] == ["new", "old"]


def _with_player(ctx, seconds: float = 6.0):
    """Put a decoded take under the strip, the way an audition would.

    The player draws a *buffer*, not a row, so a tray full of finished takes is
    not enough to exercise it -- which is also why ``should_draw`` gates on the
    buffer rather than on the rows.
    """
    import numpy as np

    from warlock.studio import muse_state
    from warlock.studio.muse import waveform

    rate = 44100
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    tone = (np.sin(2 * np.pi * 220.0 * t) * 12000).astype(np.int16)
    pcm = np.stack([tone, tone], axis=1)
    state = muse_mode.ensure(ctx)
    state.player = muse_state.Player(
        job="a", pcm=pcm, rate=rate, env=waveform.peaks(pcm), duration=seconds
    )
    return state.player


def test_the_player_draws_with_a_take_under_it(frames, tmp_path):
    ctx = _ctx(tmp_path, [_take("a")])
    _with_player(ctx)
    assert muse_player.should_draw(ctx) is True
    frames(lambda: muse_player.draw(ctx))


def test_the_player_draws_with_a_region_and_candidates(frames, tmp_path):
    """The branch with every control on it: markers, the fill, the candidate
    buttons, the crossfade slider and both exports."""
    from warlock.studio.muse.loops import Candidate

    ctx = _ctx(tmp_path, [_take("a")])
    one = _with_player(ctx)
    muse_mode.set_region(ctx, 1.0, 4.0)
    one.candidates = [Candidate(0, 44100, 0.1), Candidate(44100, 88200, 0.2)]
    frames(lambda: muse_player.draw(ctx))


def test_the_player_draws_while_the_finder_is_running(frames, tmp_path):
    ctx = _ctx(tmp_path, [_take("a")])
    _with_player(ctx).finding = True
    frames(lambda: muse_player.draw(ctx))


def test_the_strip_is_absent_until_a_take_has_been_auditioned(tmp_path):
    """A mode that reserved 148 dp for a picture it has no samples for is a
    mode with a hole in it -- so the two columns get the whole height until
    there is something to put under them."""
    ctx = _ctx(tmp_path, [_take("a")])
    assert muse_player.should_draw(ctx) is False


def test_no_control_appears_in_both_the_bar_and_the_column():
    """The one-owner rule, enforced by reading the two files.

    Create keeps the same split and states it in prose; this is the half that
    fails when somebody adds a duration slider to the recipe column because it
    felt like a setting.
    """
    from pathlib import Path

    bar = Path(muse_brief.__file__).read_text(encoding="utf-8")
    column = Path(muse_recipe.__file__).read_text(encoding="utf-8")
    for field in ("prompt", "lyrics", "duration", "count"):
        assert f'form["{field}"]' in bar, f"the bar should own {field}"
        assert f'form["{field}"]' not in column, f"{field} is in both panes"
    for field in ("infer_step", "guidance_scale", "scheduler_type", "cfg_type"):
        assert f'form["{field}"]' in column, f"the column should own {field}"
        assert f'form["{field}"]' not in bar, f"{field} is in both panes"

    # The third surface. ``panes/muse_results`` draws the derive popup, whose
    # controls are about *one finished take* rather than about the next press
    # -- so it must not touch the brief at all. Without this the popup is a
    # third place to look for a generation setting, which is the failure the
    # bar/column split exists to prevent, one surface further on.
    tray = Path(muse_results.__file__).read_text(encoding="utf-8")
    for field in ("prompt", "lyrics", "duration", "count", "infer_step",
                  "guidance_scale", "scheduler_type", "cfg_type", "omega_scale"):
        assert f'form["{field}"]' not in tray, (
            f"{field} is a brief control and the tray must not own one"
        )


def test_every_task_the_menu_offers_has_controls_and_a_door():
    """Three tables that have to agree, and nothing else makes them.

    ``DERIVE_ITEMS`` is what the menu offers, ``muse_mode.DERIVE_CONTROLS`` is
    what the popup draws for each, and ``_jobs_music.TASKS`` is what the door
    accepts. A task in the first and not the third is a menu item that always
    refuses; one in the third and not the first is capability with no way in.
    """
    from warlock.service._jobs_music import TASKS
    from warlock.studio import muse_mode

    offered = [one[0] for one in muse_results.DERIVE_ITEMS]
    assert sorted(offered) == sorted(TASKS)
    assert sorted(muse_mode.DERIVE_CONTROLS) == sorted(TASKS)
    assert len(set(offered)) == len(offered)


def test_every_derive_control_is_drawn_by_something():
    """A key in ``DEFAULT_DERIVE`` that no task lists is a value nobody can set.

    The reverse is the one that bites: a control named in ``DERIVE_CONTROLS``
    with no entry in ``DEFAULT_DERIVE`` is a ``KeyError`` the first time that
    task's popup opens, and no smoke test would reach it -- the popup only
    draws once a take exists and a menu item has been pressed.
    """
    from warlock.studio import muse_mode
    from warlock.studio.muse_state import DEFAULT_DERIVE

    named = {name for names in muse_mode.DERIVE_CONTROLS.values() for name in names}
    assert named <= set(DEFAULT_DERIVE)
    assert set(DEFAULT_DERIVE) - named == {"task", "count"}
    # The numeric ones each need a label and a bound; the two edit fields are
    # text and are drawn by their own branch.
    assert named - set(muse_results.DERIVE_FIELDS) == {"edit_prompt", "edit_lyrics"}

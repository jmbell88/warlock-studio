"""The motion primitives, and the one switch that turns all of them off.

Headless: :func:`motion._clock` is the module's only reach into imgui, so
substituting it is what lets every rule about easing, reduce-motion and the
idle-clamp wake condition be asserted without a GL context -- which is the same
argument ``splash.Startup`` is stdlib-only for.
"""

from __future__ import annotations

import pytest

from warlock.studio import layout, motion, theme, tokens


@pytest.fixture
def clock(monkeypatch):
    """A frame clock the test drives. ``tick()`` advances one 60 Hz frame."""

    state = {"dt": 1 / 60, "frame": 1}

    def fake() -> tuple[float, int]:
        return (state["dt"], state["frame"])

    monkeypatch.setattr(motion, "_clock", fake)

    def tick(dt: float = 1 / 60) -> None:
        state["dt"] = dt
        state["frame"] += 1

    state["tick"] = tick
    return state


@pytest.fixture(autouse=True)
def _clean():
    motion.reset()
    motion.set_reduced(False)
    yield
    motion.reset()
    motion.set_reduced(False)


# --- the curves --------------------------------------------------------------


@pytest.mark.parametrize("curve", [motion.ease_out_cubic, motion.ease_in_out_cubic])
def test_a_curve_starts_at_zero_ends_at_one_and_never_goes_back(curve):
    assert curve(0.0) == 0.0
    assert curve(1.0) == 1.0
    samples = [curve(i / 40) for i in range(41)]
    assert samples == sorted(samples)


@pytest.mark.parametrize("curve", [motion.ease_out_cubic, motion.ease_in_out_cubic])
def test_a_curve_clamps_rather_than_overshooting(curve):
    """Every caller divides an elapsed time by a duration, and a frame that
    lands a hair past the end would otherwise ask for an alpha above 1."""
    assert curve(-0.5) == 0.0
    assert curve(1.5) == 1.0


def test_ease_out_is_ahead_of_linear_and_ease_in_out_is_symmetric():
    assert motion.ease_out_cubic(0.5) > 0.5
    assert motion.ease_in_out_cubic(0.5) == pytest.approx(0.5)
    assert motion.ease_in_out_cubic(0.25) == pytest.approx(
        1.0 - motion.ease_in_out_cubic(0.75)
    )


# --- value / peek ------------------------------------------------------------


def test_a_first_sighting_snaps_and_a_changed_target_moves(clock):
    assert motion.value("k", 1.0) == 1.0  # settled, not assembling itself
    clock["tick"]()
    moved = motion.value("k", 0.0)
    assert 0.0 < moved < 1.0


def test_peek_reads_without_advancing(clock):
    motion.value("k", 0.0)
    clock["tick"]()
    motion.value("k", 1.0)
    seen = motion.peek("k")
    assert motion.peek("k") == seen
    assert motion.peek("nothing-here", 0.5) == 0.5


def test_peek_does_not_claim_a_key_is_moving(clock):
    """A peek is a read, not a target: ``animating`` answers for the widgets
    that set one, or a button drawn but never resolved would hold the frame
    loop at full rate on the strength of somebody looking at it."""
    motion.value("k", 0.0)
    clock["tick"]()
    motion.value("k", 1.0)
    clock["tick"]()
    motion.peek("k")
    assert not motion.animating()


# --- the idle-clamp wake condition -------------------------------------------


def test_animating_is_true_only_while_a_key_drawn_this_frame_is_moving(clock):
    motion.value("k", 0.0)
    clock["tick"]()
    motion.value("k", 1.0)
    assert motion.animating()
    for _ in range(200):
        clock["tick"]()
        motion.value("k", 1.0)
    assert not motion.animating()


def test_a_widget_that_leaves_the_screen_mid_move_stops_holding_the_app_awake(clock):
    """The half of ``animating`` that is easy to leave out. A card hovered and
    then navigated away from keeps a target it will never be asked for again,
    and without the frame stamp the app would never idle again either."""
    motion.value("card/lift", 0.0)
    clock["tick"]()
    motion.value("card/lift", 1.0)
    assert motion.animating()
    clock["tick"]()  # the pane is gone; nobody asks for that key
    assert not motion.animating()


def test_a_spring_crossing_its_target_at_speed_is_still_animating(clock):
    """``animating`` used to read only distance-to-target, and a spring's
    overshoot passes through zero distance at full speed -- the exact moment
    ``SPRING_DAMPING`` exists to produce (H12). Settled is distance *and*
    velocity under the floor, so still-moving is either over it."""
    motion._STATE["s"] = 1.0
    motion._TARGET["s"] = 1.0
    motion._VELOCITY["s"] = 0.5
    motion._FRAME["s"] = clock["frame"]
    assert motion.animating()
    # The frame-stamp discipline is untouched: a key nothing drew this frame
    # cannot hold the app awake, however fast it was going.
    clock["tick"]()
    assert not motion.animating()


def test_every_way_of_dropping_a_key_drops_its_velocity_too(clock):
    """H12's rule, which used to be tested through the ``SPRINGS``-off
    fallback: that path was the one that dropped a key without clearing its
    velocity, so a re-enabled spring resumed a move at a speed nothing asked
    for. The flag was folded away on 2026-08-12 and the fallback with it, but
    the rule it caught is a property of the state pair -- ``_STATE`` and
    ``_VELOCITY`` are cleared together or a stale speed outlives its value.
    ``seed`` is covered in ``test_ux_phase45``; these are the other two."""
    for drop in (lambda: motion.forget("k"), motion.reset):
        motion.seed("k", 0.0)
        for _ in range(10):
            clock["tick"]()
            motion.spring("k", 1.0)
        assert motion._VELOCITY["k"] != 0.0
        drop()
        assert motion._VELOCITY.get("k", 0.0) == 0.0
        assert "k" not in motion._STATE


# --- reduce motion -----------------------------------------------------------


def test_reduce_motion_snaps_every_value_and_reports_nothing_moving(clock):
    motion.value("k", 0.0)
    clock["tick"]()
    motion.set_reduced(True)
    assert motion.value("k", 1.0) == 1.0
    assert not motion.animating()


def test_reduce_motion_makes_an_arrival_arrived_rather_than_absent(clock):
    """The failure mode a naive "duration = 0" has: a one-shot ramp that reads
    0.0 forever is a popover that fades in to nothing and a splash veil that
    never clears."""
    motion.set_reduced(True)
    motion.restart("enter")
    assert motion.ease("enter", tokens.DUR_FAST) == 1.0


def test_ease_runs_from_zero_to_one_and_stays_there(clock):
    motion.restart("enter")
    first = motion.ease("enter", tokens.DUR_FAST)
    assert 0.0 < first < 1.0
    for _ in range(60):
        clock["tick"]()
        last = motion.ease("enter", tokens.DUR_FAST)
    assert last == 1.0


def test_an_unstarted_ease_key_reads_as_finished(clock):
    """A popup that has never been opened must not draw its first frame at
    alpha 0 with a 6 px offset it did not ask for."""
    assert motion.ease("never-started", tokens.DUR_FAST) == 1.0


def test_forget_makes_the_next_sighting_snap(clock):
    motion.value("k", 0.0)
    clock["tick"]()
    motion.forget("k")
    assert motion.value("k", 1.0) == 1.0


# --- the colour mix the animated widgets draw through ------------------------


def test_mix_lands_exactly_on_its_ends_and_clamps():
    low, high = 0x000000, 0xFFFFFF
    assert theme.mix(low, high, 0.0)[:3] == (0.0, 0.0, 0.0)
    assert theme.mix(low, high, 1.0)[:3] == (1.0, 1.0, 1.0)
    assert theme.mix(low, high, 2.0) == theme.mix(low, high, 1.0)
    assert theme.mix(low, high, -1.0) == theme.mix(low, high, 0.0)
    assert theme.mix(low, high, 0.5, 0.25)[3] == 0.25


def test_mix_resolves_roles_against_the_palette_in_force():
    try:
        tokens.set_theme("light")
        assert theme.mix(theme.PANEL, theme.ELEV_2, 0.0)[:3] == theme.rgba(
            tokens.PALETTES["light"]["PANEL"]
        )[:3]
    finally:
        tokens.set_theme("dark")


# --- the sidebar width, which is module state the frame loop advances --------


def test_the_construction_path_snaps_so_a_headless_caller_is_never_mid_move():
    try:
        layout.set_sidebar("wide")
        assert layout.SIDEBAR_WIDTHS["wide"] == layout.SIDEBAR_W
        assert layout.set_sidebar("nonsense") == "default"
        assert layout.SIDEBAR_WIDTHS["default"] == layout.SIDEBAR_W
    finally:
        layout.set_sidebar("default")


def test_changing_the_width_in_settings_eases_rather_than_jumping(clock):
    try:
        layout.set_sidebar("default")
        layout.set_sidebar("wide", animate=True)
        # Nothing has ticked yet, so the columns are still where they were.
        assert layout.SIDEBAR_WIDTHS["default"] == layout.SIDEBAR_W
        # The very first tick already moves: an animated change made before
        # anything has ticked must slide rather than jump, which is what
        # ``motion.seed`` is for.
        layout.tick()
        assert (
            layout.SIDEBAR_WIDTHS["default"]
            < layout.SIDEBAR_W
            < layout.SIDEBAR_WIDTHS["wide"]
        )
        for _ in range(120):
            clock["tick"]()
            layout.tick()
        assert layout.SIDEBAR_WIDTHS["wide"] == layout.SIDEBAR_W
    finally:
        layout.set_sidebar("default")


# --- the switch itself -------------------------------------------------------


def test_the_settings_toggle_moves_the_state_the_flag_and_the_file_together():
    """Three halves of one setting. The one that is easy to leave out is
    ``motion.REDUCED``: the checkbox would then tick, the file would remember
    it, and the app would go on animating until the next launch."""
    from types import SimpleNamespace

    from warlock.studio.panes import app_settings

    written: dict[str, object] = {}
    ctx = SimpleNamespace(
        state=SimpleNamespace(reduce_motion=False),
        settings=SimpleNamespace(set=lambda key, value: written.__setitem__(key, value)),
    )
    try:
        app_settings._apply_reduce_motion(ctx, True)
        assert ctx.state.reduce_motion is True
        assert motion.REDUCED is True
        assert written == {"reduce_motion": True}
    finally:
        motion.set_reduced(False)

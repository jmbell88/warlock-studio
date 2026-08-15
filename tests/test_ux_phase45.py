"""UX.md Phases 4 and 5: the parts a screenshot cannot check.

Same division of labour as ``tests/test_ux_phases.py`` states for Phases 2 and
3. The screenshot pass (``scripts/screenshot_modes.py``, now with ``--floating``
so the palette's translucency is in a still at all) is the instrument for the
half of these phases that is pixels. It cannot see that a resume row opens in
the pane that made it, that a nine-patch refuses a rectangle smaller than its
own corners, or that a spring settles rather than running out of ramp -- and
those are what this file pins.

The phase's first-run orientation card is gone: Home stopped being a chooser
with nothing on it but choices, and a card explaining the screen sat above a
screen that now explains itself.

The GL half of Phase 5 -- that a sprite really builds and a backdrop really
captures -- lives in ``tests/test_studio_smoke.py``, where the context, the
renderer and the framebuffer fixtures already are.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from warlock.studio import motion, ninepatch, shadows, surfaces, vibrancy
from warlock.studio.main import filter_shortcuts
from warlock.studio.panes import landing, library, overlay


class _Settings:
    """The two methods every settings reader in the app uses."""

    def __init__(self, **data) -> None:
        self.data = dict(data)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


def _ctx(**kwargs):
    base = {
        "state": SimpleNamespace(selected=None, home_index=0, mode="home"),
        "settings": _Settings(),
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


# --- Phase 4: resume --------------------------------------------------------


def _job(job_id, status="done", stage="model", **extra):
    return {"id": job_id, "status": status, "stage": stage, **extra}


def _cache(jobs):
    by_id = {job["id"]: job for job in jobs}
    return SimpleNamespace(jobs=list(jobs), get=lambda key: by_id.get(key))


def test_resuming_selects_what_it_opens_and_lands_in_the_right_pane():
    """The same reference/model split Continue used, plus the selection -- a
    resume that switched mode and left nothing selected would open the pane the
    asset belongs to and not the asset.

    The Continue *card* is gone: it was one row in front of a tile grid,
    answering "the newest finished thing" with one card, and the Resume list
    answers it with the whole list. What survives is the rule about how a row
    is opened.
    """
    picked = []
    state = SimpleNamespace(
        selected=None,
        home_index=0,
        mode="home",
        select=picked.append,
    )
    ctx = _ctx(cache=_cache([_job("ref", stage="reference")]), state=state)
    landing.activate(ctx, 0)
    assert picked == ["ref"]
    assert state.mode == "2d"


def test_a_queued_job_is_not_offered_and_a_bare_ctx_is_not_a_crash():
    """Opening a queued job lands on a pane with nothing in it; and Home is the
    screen the app opens on, so a headless or half-built ctx must draw an empty
    list rather than raise."""
    assert landing.rows(_ctx(cache=_cache([_job("q", status="queued")]))) == []
    assert landing.rows(_ctx(cache=_cache([]))) == []
    assert landing.rows(_ctx()) == []


# --- Phase 4: in-flow discovery ---------------------------------------------


def test_a_prefix_chip_types_itself_into_the_query():
    """Separated by a space, because ``rustytag:wood`` is one word that parses
    as free text and quietly matches nothing."""
    assert library.append_prefix("", "tag") == "tag:"
    assert library.append_prefix("rusty", "tag") == "rusty tag:"
    assert library.append_prefix("tag:wood ", "status") == "tag:wood status:"


def test_a_chip_clicked_twice_is_a_mis_click_rather_than_a_request():
    assert library.append_prefix("tag:", "tag") == "tag:"


def test_the_chips_are_the_parsers_own_prefixes():
    """A chip offering a prefix ``parse_query`` does not know is a button that
    turns the query into free text."""
    from warlock.studio.state import QUERY_FIELDS

    assert "QUERY_FIELDS" in inspect.getsource(library._prefix_chips)
    for field in QUERY_FIELDS:
        assert library.append_prefix("", field) == f"{field}:"


def test_the_shortcuts_filter_matches_a_group_by_name():
    """"clay" lists Clay's bindings, not the two whose text says the word."""
    sections = [
        ("Everywhere", [("Ctrl+K", "Command palette"), ("F1", "Manual")]),
        ("Clay", [("E", "Extrude"), ("F", "Frame the selection")]),
    ]
    assert filter_shortcuts(sections, "clay") == [sections[1]]


def test_the_shortcuts_filter_uses_the_palettes_matcher_and_drops_empty_groups():
    """A heading over nothing reads as a section that broke; and a second
    matcher is a second answer to "does ctz find Ctrl+Z"."""
    sections = [
        ("Everywhere", [("Ctrl+K", "Command palette"), ("F1", "Manual")]),
        ("Clay", [("E", "Extrude"), ("F", "Frame the selection")]),
    ]
    assert "palette" in inspect.getsource(filter_shortcuts)
    assert filter_shortcuts(sections, "ctk") == [
        ("Everywhere", [("Ctrl+K", "Command palette")])
    ]
    assert filter_shortcuts(sections, "zzzz") == []
    assert filter_shortcuts(sections, "") == sections


def test_the_filtered_rows_keep_their_hand_chosen_order():
    """A filter that also reshuffles is two changes to read at once."""
    rows = [("A", "alpha"), ("B", "alpha beta"), ("C", "alpha")]
    kept = filter_shortcuts([("G", rows)], "alpha")[0][1]
    assert kept == rows


# --- Phase 4: the splash ----------------------------------------------------


def test_the_splash_says_what_the_load_is_doing():
    """Three seconds of "Starting Warlock Studio..." spends the whole hold
    saying what the logo already said."""
    from warlock.studio import splash

    started = splash.Startup(lambda: None)
    assert started.message == splash.MESSAGE
    started.note("Checking this install")
    assert started.message == "Checking this install"


def test_a_quit_stops_the_narration_rather_than_arguing_with_it():
    from warlock.studio import splash

    started = splash.Startup(lambda: None)
    started.request_quit()
    started.note("Starting the worker")
    assert started.message == "Closing..."


def test_the_runtime_narrates_every_observable_pause():
    """A stage that has never taken measurable time gets no line; the ones that
    do are the doctor's probes and the worker's construction."""
    from warlock.studio import runtime

    source = inspect.getsource(runtime.Runtime._start)
    assert source.count("self._note(") >= 4
    # Defaulted rather than optional at every call site, so a start with
    # nothing listening cannot be a branch anybody forgets.
    assert "note: Callable[[str], None] | None = None" in inspect.getsource(
        runtime.Runtime.start
    )


# --- Phase 4: the progress card and the empty states ------------------------


def test_the_progress_card_can_be_faded_out_after_its_job_is_gone():
    """The job is gone by the time the card leaves, so the fade-out is drawn
    from the last snapshot -- with Cancel disabled, because a button acting on
    a finished job is worse than no button."""
    source = inspect.getsource(overlay.progress_card)
    assert "_LAST_PROGRESS" in source
    assert "motion.value(_PROGRESS_KEY" in source
    assert 'widgets.disabled_button("Cancel", live' in source


def test_the_progress_card_takes_the_same_depth_treatment_as_every_other_surface():
    source = inspect.getsource(overlay.progress_card)
    assert "push_surface_rounding" in source
    assert 'window_shadow("raised"' in source


def test_an_empty_state_has_three_registers_rather_than_one():
    """Three lines of muted text says "there is nothing here" in the typography
    as well as in the words, on one of the most-seen surfaces in the app."""
    from warlock.studio import widgets

    source = inspect.getsource(widgets.empty_state)
    assert "fonts.display" in source and "fonts.heading" in source
    assert "theme.TEXT" in source


# --- Phase 5: the effects, which no longer have switches ---------------------
#
# There were three tests here over ``effects.KEYS`` -- the table, the
# settings-file default, and the unknown-name guard. Phase 5 shipped the four
# items behind "a config flag per item while it stabilizes, folded away once
# trusted"; they were folded away on 2026-08-12 and the module is gone. What
# survives is the part the flags were standing in for: each effect answers
# "did it work" itself, which is the test below.


def test_every_effect_degrades_on_its_own_rather_than_raising(monkeypatch):
    """There is no flag in front of these any more, and there never was a
    promise the GL side is there. Each module answers "did it work" itself,
    which is what lets a host with no renderer draw the pre-Phase-5 app.

    The renderer is stood down explicitly rather than assumed absent: the GL
    smoke suite's own renderer is session-scoped, so "there is no renderer in
    this process" is true only when this file runs alone -- which is the shape
    of order-dependent test this repo has been bitten by before.
    """
    for module in (shadows, surfaces, vibrancy):
        assert module.available() in (True, False)
    monkeypatch.setattr(ninepatch, "_renderer", lambda: None)
    monkeypatch.setattr(vibrancy, "_PIPELINE", None)
    # Declines, and declines *without* latching the broken flag: a host that
    # simply has not built its renderer yet must not be written off for the
    # life of the process.
    assert shadows.sprite(11.0, 9.0) is None
    assert surfaces.sprite(11.0) is None
    assert vibrancy.backdrop() is None
    assert not shadows._BROKEN and not surfaces._BROKEN and not vibrancy._BROKEN


# --- Phase 5: the shadow sprite ---------------------------------------------


def test_the_shadow_sprite_is_a_blur_that_dies_before_its_own_edge():
    """A shadow that is still dark at the sprite's border is one the nine-slice
    will tile into a visible seam."""
    alpha = shadows.pixels(10.0, 9.0)
    size = alpha.shape[0]
    assert alpha.shape == (size, size)
    assert alpha[0, 0] <= 2, "the corner of the sprite is the far end of the blur"
    assert alpha[size // 2, size // 2] > 240, "and the middle is the surface"
    # Monotonic along the middle row from the edge inwards: that is what makes
    # it a falloff rather than a ring.
    middle = alpha[size // 2, : size // 2 + 1].astype(int)
    assert list(middle) == sorted(middle)


def test_the_shadow_sprites_middle_row_is_flat_where_it_gets_stretched():
    """The nine-slice stretches the middle strip, so the shape has to be
    constant along it -- otherwise a wide card's shadow is a smear of whatever
    the middle pixel happened to be."""
    alpha = shadows.pixels(10.0, 9.0)
    size = alpha.shape[0]
    column = alpha[:, size // 2]
    assert column[size // 2] == alpha[size // 2, size // 2]
    # Symmetric, which is what lets one sprite serve all four corners.
    assert np.array_equal(alpha, alpha[::-1, :])
    assert np.array_equal(alpha, alpha[:, ::-1])


def test_the_two_shadow_recipes_occupy_the_same_space():
    """A call site must not be able to tell them apart by layout: the sprite's
    reach is the outermost stroked band's outer bound."""
    from warlock.studio import tokens

    assert max(outer for _i, outer, _a in tokens.SHADOW_STEPS) == shadows.SPREAD


# --- Phase 5: the nine-slice ------------------------------------------------


class _Draw:
    def __init__(self) -> None:
        self.quads = []

    def add_image(self, ref, p_min, p_max, uv_min, uv_max, colour) -> None:
        self.quads.append((p_min, p_max, uv_min, uv_max))


def _sprite(corner=10):
    return ninepatch.Sprite(texture=None, ref=object(), corner=corner)


def test_a_shadow_leaves_the_interior_alone():
    """Eight quads, never nine: it is what lets one recipe be drawn under a
    card and over an already-painted window background."""
    draw = _Draw()
    assert ninepatch.paint(draw, _sprite(), (0, 0), (200, 100), 0, centre=False)
    assert len(draw.quads) == 8
    draw = _Draw()
    assert ninepatch.paint(draw, _sprite(), (0, 0), (200, 100), 0)
    assert len(draw.quads) == 9


def test_the_slice_covers_the_rectangle_exactly_once():
    """Gaps show as seams and overlaps double the alpha; both are visible in
    exactly the four places a nine-patch exists to get right."""
    draw = _Draw()
    ninepatch.paint(draw, _sprite(10), (0.0, 0.0), (200.0, 100.0), 0)
    xs = sorted({q[0][0] for q in draw.quads} | {q[1][0] for q in draw.quads})
    ys = sorted({q[0][1] for q in draw.quads} | {q[1][1] for q in draw.quads})
    assert xs == [0.0, 10.0, 190.0, 200.0]
    assert ys == [0.0, 10.0, 90.0, 100.0]
    assert len(draw.quads) == 9


def test_a_rectangle_smaller_than_its_own_corners_is_refused():
    """Shrinking the corner blocks distorts the shape the sprite is exact
    about; drawing them overlapped doubles the alpha at the corners."""
    draw = _Draw()
    assert not ninepatch.paint(draw, _sprite(20), (0, 0), (30, 30), 0)
    assert draw.quads == []


def test_the_uvs_split_where_the_pixels_do():
    """The middle strip is one texel wide, and it is the one the shape is
    constant along."""
    sprite = _sprite(10)
    lo, hi = sprite.splits()
    assert sprite.size == 21
    assert lo == 10 / 21 and hi == 11 / 21


# --- Phase 5: the squircle --------------------------------------------------


def test_a_squircle_is_fuller_than_the_circular_arc_it_replaces():
    """That is the whole of the visual difference: a superellipse holds the
    corner's area further out, which is what removes the pinch."""
    radius = 12.0
    mask = surfaces.pixels(radius)
    circle = shadows.pixels(radius, 1.0)  # the same corner, drawn as an arc
    assert mask[:, :].sum() > circle.sum()
    assert mask[0, 0] == 0, "the very corner is still outside the shape"
    size = mask.shape[0]
    assert mask[size // 2, size // 2] == 255


def test_the_squircle_is_symmetric_and_anti_aliased():
    mask = surfaces.pixels(12.0)
    assert np.array_equal(mask, mask[::-1, :])
    assert np.array_equal(mask, mask[:, ::-1])
    # Partial coverage somewhere on the arc, or the corner is a staircase.
    assert ((mask > 0) & (mask < 255)).any()


def test_a_small_radius_keeps_the_circular_corner():
    """UX.md's own rule: below about eight pixels the two shapes differ by less
    than one, and the sprite would be four texels a side."""
    assert not surfaces.enabled_for(surfaces.MIN_RADIUS - 1.0)
    assert surfaces.sprite(4.0) is None


# --- Phase 5: springs -------------------------------------------------------


class _Clock:
    """A stand-in for the frame clock ``motion`` reads off imgui."""

    def __init__(self, dt=1 / 60) -> None:
        self.dt = dt
        self.frame = 0

    def __call__(self):
        self.frame += 1
        return (self.dt, self.frame)


def _run(monkeypatch, key, target, frames, *, start=0.0, duration=0.2):
    monkeypatch.setattr(motion, "_clock", _Clock())
    motion.forget(key)
    motion.seed(key, start)
    return [motion.spring(key, target, duration=duration) for _ in range(frames)]


def test_a_spring_overshoots_and_settles(monkeypatch):
    """The characteristic that neither of the other two primitives has. An
    exponential approach cannot exceed its target by construction."""
    path = _run(monkeypatch, "test/spring", 1.0, 120)
    assert max(path) > 1.0, "it goes past"
    assert path[-1] == 1.0, "and it comes to rest, exactly"


def test_a_spring_stops_rather_than_approaching_forever(monkeypatch):
    """A spring is asymptotic, and ``animating()`` reads "still moving" off the
    distance to the target -- so without a floor the idle clamp never idles."""
    _run(monkeypatch, "test/settle", 1.0, 300)
    assert motion._STATE["test/settle"] == 1.0
    assert motion._VELOCITY["test/settle"] == 0.0


def test_reduce_motion_snaps_a_spring_like_everything_else(monkeypatch):
    monkeypatch.setattr(motion, "REDUCED", True)
    monkeypatch.setattr(motion, "_clock", _Clock())
    motion.forget("test/reduced")
    motion.seed("test/reduced", 0.0)
    assert motion.spring("test/reduced", 1.0) == 1.0


def test_a_spring_is_not_the_exponential_now_that_the_switch_is_gone(monkeypatch):
    """``effects.SPRINGS`` used to route :func:`motion.spring` back through
    :func:`motion.value`, and the old test here pinned that the fallback was
    *exact*. The flag was folded away on 2026-08-12, so what needs pinning is
    the opposite: the two primitives are genuinely different curves, and a
    careless fold-away that left ``spring`` delegating would pass unnoticed.
    Reduce-motion (tested above) is the only remaining way to stop the move."""
    monkeypatch.setattr(motion, "_clock", _Clock())
    motion.forget("test/off")
    motion.seed("test/off", 0.0)
    sprung = [motion.spring("test/off", 1.0) for _ in range(10)]
    monkeypatch.setattr(motion, "_clock", _Clock())
    motion.forget("test/off2")
    motion.seed("test/off2", 0.0)
    eased = [motion.value("test/off2", 1.0) for _ in range(10)]
    assert sprung != eased
    # And different in the way that matters: the spring carries velocity, so it
    # goes past its target and comes back. The exponential cannot.
    assert max(sprung) > 1.0
    assert max(eased) <= 1.0


def test_a_key_seeded_afresh_leaves_at_rest(monkeypatch):
    """Stating where a value *is* says nothing about where it was going, and a
    popover re-seeded to 0 while carrying the velocity of its last arrival
    would leave at a speed nothing asked for."""
    _run(monkeypatch, "test/seeded", 1.0, 30)
    motion.seed("test/seeded", 0.0)
    assert motion._VELOCITY.get("test/seeded", 0.0) == 0.0


def test_the_sliding_pill_and_the_popover_are_what_took_the_spring():
    """The two surfaces Phase 5 names, and the pill is the one whose target
    routinely changes while it is still moving."""
    from warlock.studio import widgets

    assert "motion.spring" in inspect.getsource(widgets.segmented_control)
    assert "motion.spring" in inspect.getsource(widgets.popover_enter)
    # The alpha is clamped and the rise is not: a spring goes a little past 1,
    # which for an opacity is a value that does not exist.
    assert "min(max(t, 0.0), 1.0)" in inspect.getsource(widgets.popover_enter)


# --- Phase 5: vibrancy ------------------------------------------------------


def test_a_floating_surface_never_blurs_itself():
    """The capture is skipped on any frame something sampled the backdrop,
    which is the whole of how the region under an open palette is kept from
    converging to smeared mush."""
    source = inspect.getsource(vibrancy.capture)
    assert "used, _USED = _USED, False" in source
    assert "or used:" in source


def test_the_backdrops_uvs_flip_the_y_axis():
    """The capture is a blit of the default framebuffer, whose origin is
    bottom-left; imgui's screen coordinates start at the top."""
    uv_min, uv_max = vibrancy.uv_for((0.0, 0.0), (400.0, 200.0), (800.0, 400.0))
    assert uv_min == (0.0, 1.0)
    assert uv_max == (0.5, 0.5)


def test_the_capture_is_downscaled_and_throttled():
    """One full-screen blit per frame on an app that is otherwise idle at
    twelve is the cost this effect could have been."""
    assert vibrancy.DOWNSCALE >= 2
    assert 0.0 < vibrancy.CAPTURE_INTERVAL <= 0.2
    assert vibrancy._target_size((1600.0, 900.0)) == (400, 225)
    # Never zero-sized, whatever the window does.
    assert min(vibrancy._target_size((0.0, 0.0))) >= 1


def test_every_frosted_surface_paints_a_fill_when_there_is_no_capture_yet():
    """The caller cleared imgui's own background on ``frosted``'s word, so a
    backdrop that then failed to arrive would be a floating surface with
    nothing in it."""
    from warlock.studio import widgets

    source = inspect.getsource(widgets.window_backdrop)
    assert "if ref is None:" in source
    assert "add_rect_filled" in source


def test_the_surfaces_that_float_are_the_ones_that_frost():
    """Toasts deliberately do not: the backdrop is the app as it was when the
    surface appeared, and a toast appears while what it reports on is moving."""
    from warlock.studio import dialogs, widgets
    from warlock.studio.manual import render as manual_render
    from warlock.studio.panes import palette as palette_pane

    assert "window_backdrop" in inspect.getsource(palette_pane.draw)
    assert "window_backdrop" in inspect.getsource(dialogs.ConfirmQueue.draw)
    assert "window_backdrop" in inspect.getsource(dialogs.PromptQueue.draw)
    # The Manual joined them when it stopped being a mode (REDESIGN.md wave 3):
    # it is a floating surface over the app now, so it frosts like one.
    assert "window_backdrop" in inspect.getsource(manual_render.draw_overlay)
    assert "window_shadow" in inspect.getsource(manual_render.draw_overlay)
    assert "window_backdrop" not in inspect.getsource(widgets.toasts)

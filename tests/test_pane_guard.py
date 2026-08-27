"""A pane that raises loses its pane, not the session.

``App.run`` wraps setup and the whole frame loop in one ``except``, so before
this every pane draw was a way to end the process -- and ``widgets.section``
reads a persisted dict every frame in most panes, so one wrong byte in
``settings.json`` was enough. ``studio/guard.py`` is the net; this file is what
says the net holds.

Most of it is about the *unwind*, because that is the part that cannot be
reasoned about from the outside. Dear ImGui's ``ErrorRecoveryTryToRecoverState``
promises to close every window, table, tab bar, tree, group, popup and stack
pushed past a stored mark, and the tests below assert that against all eleven
counters rather than trusting the promise. The clip-rect stack is asserted
separately and for the opposite reason: it is the one thing that recovery does
*not* restore, it lives on the draw list a whole host window shares, and a leak
there silently mis-clips every later pane in the frame rather than failing.

``STRICT`` is forced off per test here. ``tests/conftest.py`` turns it on autouse
for every other file in the suite, so a pane that raises anywhere else in these
twelve thousand tests still fails loudly instead of quietly drawing a
placeholder -- which is the same guarantee ``scripts/exercise_mode`` needs.
"""

from __future__ import annotations

import pytest

from warlock.studio import guard, layout

STACKS = (
    "size_of_window_stack",
    "size_of_id_stack",
    "size_of_tree_stack",
    "size_of_color_stack",
    "size_of_style_var_stack",
    "size_of_font_stack",
    "size_of_focus_scope_stack",
    "size_of_group_stack",
    "size_of_item_flags_stack",
    "size_of_begin_popup_stack",
    "size_of_disabled_stack",
)


@pytest.fixture
def imgui_ctx(gl):
    """This file's own imgui context, built and torn down around it.

    Copied from ``test_section_blocks`` rather than shared, for the reason its
    own docstring gives: two imgui contexts over the one GL context crash the
    process, so at most one may exist at a time and a file that wants one builds
    and destroys it rather than relying on collection order.
    """
    from imgui_bundle import imgui

    from warlock.studio import imgui_backend, theme

    prev_ctx = imgui.get_current_context()
    prev_screen = type(gl).__dict__.get("screen")
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    guard.configure()
    renderer = imgui_backend.ImguiRenderer(gl)
    yield imgui, renderer
    renderer.shutdown()
    imgui.destroy_context(ctx)
    if prev_screen is not None:
        type(gl).screen = prev_screen
    if prev_ctx is not None:
        imgui.set_current_context(prev_ctx)


@pytest.fixture(autouse=True)
def _catching(monkeypatch):
    """This file is the one that wants the guard to actually catch."""
    monkeypatch.setattr(guard, "STRICT", False)
    guard.reset()
    guard.HISTORY.clear()
    guard.FRAME_FAILURES.clear()
    yield
    guard.reset()
    guard.HISTORY.clear()
    guard.FRAME_FAILURES.clear()


@pytest.fixture
def frame(imgui_ctx):
    """Open a host window, run ``build`` in it, close the frame.

    ``render()`` is inside rather than after: a frame whose stacks were left
    unbalanced fails *there*, so calling it is the assertion that the unwind
    worked, and every test below gets it for free.
    """
    imgui, _renderer = imgui_ctx
    counter = [0]

    def run(build, *, size=(420.0, 900.0)):
        counter[0] += 1
        out = []
        imgui.new_frame()
        # Where the frame began, so an *escalating* failure can still be closed.
        # The tests below deliberately let ``KeyboardInterrupt``, ``SystemExit``
        # and ``MemoryError`` escape, which is what the app does too -- but the
        # app then tears the context down, and this fixture reuses it. A frame
        # left open makes the next ``new_frame`` assert "Forgot to call
        # Render()", in whichever file happens to run next in this xdist worker.
        # It did: the first run of this file took an unrelated
        # ``test_studio_controls`` case down with it.
        opened = imgui.internal.ErrorRecoveryState()
        imgui.internal.error_recovery_store_state(opened)
        imgui.set_next_window_size(size)
        imgui.begin(f"##guard-{counter[0]}")
        try:
            out.append(build())
        except BaseException:
            imgui.internal.error_recovery_try_to_recover_state(opened)
            imgui.end_frame()
            raise
        imgui.end()
        imgui.render()
        return out[0]

    return run


def _mess(imgui) -> None:
    """Leave every stack imgui can recover deeper than it found them."""
    imgui.push_style_var(imgui.StyleVar_.alpha.value, 0.5)
    imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(1, 0, 0, 1))
    imgui.push_id("leaked")
    imgui.begin_group()
    imgui.begin_disabled(True)
    imgui.begin_child("deeper", (50, 50))


def test_a_pane_that_raises_does_not_end_the_frame(frame, imgui_ctx):
    """The whole claim, at its smallest: the exception stops at the pane."""
    imgui, _ = imgui_ctx

    def build():
        with layout.pane("broken", (400.0, 200.0), title="Broken"):
            _mess(imgui)
            raise ValueError("the pane blew up")
        return "survived"

    assert frame(build) == "survived"


def test_the_stacks_come_back_to_where_the_pane_started(frame, imgui_ctx):
    """All eleven counters, asserted with the same API the guard unwinds with.

    Not a proxy for the balance -- it *is* the balance imgui checks in
    ``new_frame``, so a regression here is the next frame's crash.
    """
    imgui, _ = imgui_ctx
    seen = {}

    def build():
        before = imgui.internal.ErrorRecoveryState()
        imgui.internal.error_recovery_store_state(before)
        with layout.pane("broken", (400.0, 200.0), title="Broken"):
            _mess(imgui)
            raise ValueError("boom")
        after = imgui.internal.ErrorRecoveryState()
        imgui.internal.error_recovery_store_state(after)
        for name in STACKS:
            seen[name] = (getattr(before, name), getattr(after, name))
        return None

    frame(build)
    assert seen, "the build never reached the comparison"
    assert all(pair[0] == pair[1] for pair in seen.values()), seen


def test_a_leaked_clip_rect_is_popped_back_by_hand(frame, imgui_ctx):
    """The one thing imgui's own recovery does not restore.

    Asserted against ``guard`` directly rather than through ``layout.pane``,
    because that is where it is observable. ``ImDrawList::_ClipRectStack`` is not
    a field of ``ImGuiErrorRecoveryState``; a child shares its host window's
    draw list; and that list is rebuilt every frame. So the damage is confined
    to the rest of *this* frame -- every later pane in the same host clipped one
    level too tight, silently, with nothing raised and nothing left by the next
    frame to find. Measured here: two rects pushed, a child opened, recovered,
    and the stack comes back **2 deeper** than it started unless this pops it.

    ``plotter_canvas`` and ``packwright_preview`` both push a clip rect around
    the code in this app most likely to raise, with no ``finally``.
    """
    imgui, _ = imgui_ctx
    seen = {}

    def build():
        imgui.begin_child("pane", (400.0, 200.0))
        mark = guard.enter("clips")
        seen["marked"] = mark.clips
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect((0, 0), (10, 10))
        draw_list.push_clip_rect((0, 0), (5, 5))
        seen["leaked"] = len(draw_list._clip_rect_stack)
        # A child gets its *own* draw list, so the leak has to be read on the
        # pane's -- reading ``get_window_draw_list()`` from inside the child
        # measures a different object and reports no leak at all.
        imgui.begin_child("deeper", (50, 50))
        guard.recover(mark, "clips", "Clips", ValueError("boom"))
        seen["recovered"] = len(draw_list._clip_rect_stack)
        imgui.end_child()
        return None

    frame(build)
    # The leak is real: without the repair the recovery leaves these two behind.
    assert seen["leaked"] > seen["marked"]
    assert seen["recovered"] == seen["marked"], seen


def test_the_pane_after_the_broken_one_still_draws(frame, imgui_ctx):
    """The point of doing this per pane rather than per frame."""
    imgui, _ = imgui_ctx
    drew = []

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken"):
            _mess(imgui)
            raise ValueError("boom")
        with layout.pane("fine", (400.0, 100.0), title="Fine") as visible:
            drew.append(visible)
        return None

    frame(build)
    assert drew == [True]


@pytest.mark.parametrize("escape", [KeyboardInterrupt, SystemExit])
def test_the_two_that_must_never_be_caught_pass_straight_through(frame, escape):
    """``except Exception`` excludes both, and that is load-bearing rather than
    incidental: Ctrl-C and a deliberate exit are the user asking to leave."""

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken"):
            raise escape()

    with pytest.raises(escape):
        frame(build)


def test_a_fatal_failure_is_escalated_rather_than_dressed_up(frame):
    """Recovering from ``MemoryError`` and then drawing a tidy placeholder is
    not a claim this can make honestly."""

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken"):
            raise MemoryError("out of memory")

    with pytest.raises(MemoryError):
        frame(build)


def test_the_breaker_stops_re_attempting_after_three_failures(frame, imgui_ctx):
    """Attempts, not frames -- see ``guard.TRIP_AFTER``."""
    imgui, _ = imgui_ctx
    attempts = [0]

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken") as live:
            if live:
                attempts[0] += 1
                raise ValueError("boom")
        return None

    for _ in range(10):
        frame(build)
    assert attempts[0] == guard.TRIP_AFTER
    assert guard.tripped("broken")


def test_a_tripped_pane_still_opens_its_child_so_the_layout_editor_finds_it(
    frame, imgui_ctx
):
    """``layout.column`` records ``FRAME_PANES`` from the child's own rect, and
    the layout editor and the guided tour both draw from it. A pane that has
    stopped drawing has still not moved."""
    imgui, _ = imgui_ctx

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken") as live:
            if live:
                raise ValueError("boom")
            return imgui.internal.get_current_window().name

    for _ in range(guard.TRIP_AFTER):
        frame(build)
    assert guard.tripped("broken")
    assert frame(build).startswith("##guard-")


def test_try_again_closes_the_breaker(frame, imgui_ctx):
    """The placeholder's one action, and the only thing that reopens a pane."""
    imgui, _ = imgui_ctx
    attempts = [0]

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken") as live:
            if live:
                attempts[0] += 1
                raise ValueError("boom")
        return None

    for _ in range(6):
        frame(build)
    assert attempts[0] == guard.TRIP_AFTER
    guard.reset("broken")
    for _ in range(6):
        frame(build)
    assert attempts[0] == guard.TRIP_AFTER * 2


def test_one_announcement_however_many_frames_fail(frame, imgui_ctx):
    """A toast that expires in eight seconds would otherwise be raised again
    every eight seconds until the app closed; the breaker's own ``announced``
    flag is what makes "once" true."""
    imgui, _ = imgui_ctx
    said: list[tuple[str, str]] = []

    class _State:
        def note_error(self, text):
            said.append(("error", text))

    class _Ctx:
        state = _State()

        def toast(self, message, level="info", action=None):
            said.append(("toast", message))

    guard.begin_frame(_Ctx())

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken") as live:
            if live:
                raise ValueError("boom")
        return None

    for _ in range(30):
        frame(build)
    assert [kind for kind, _ in said] == ["toast", "error"]


def test_strict_mode_re_raises(frame, monkeypatch):
    """What every other test file in the suite runs under."""
    monkeypatch.setattr(guard, "STRICT", True)

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken"):
            raise ValueError("boom")

    with pytest.raises(ValueError):
        frame(build)


def test_a_failure_is_visible_to_a_harness(frame, imgui_ctx):
    """``exercise_mode`` and ``screenshot_modes`` read this, so a swallowed
    exception is still a finding rather than a green picture of a placeholder."""
    imgui, _ = imgui_ctx

    def build():
        with layout.pane("broken", (400.0, 100.0), title="Broken"):
            raise ValueError("a very specific boom")
        return None

    frame(build)
    (failure,) = guard.FRAME_FAILURES
    assert failure.key == "broken"
    assert failure.title == "Broken"
    assert failure.kind == "ValueError"
    assert "a very specific boom" in failure.traceback


def test_the_recovery_assert_is_off_before_anything_can_recover(imgui_ctx):
    """Left on, an ``IM_ASSERT`` inside the unwind surfaces as a ``RuntimeError``
    and the guard becomes the crash it exists to prevent. The debug log stays
    on: it names each thing imgui closed, which is the cheapest diagnosis there
    is."""
    imgui, _ = imgui_ctx
    io = imgui.get_io()
    assert io.config_error_recovery_enable_assert is False
    assert io.config_error_recovery_enable_tooltip is False
    assert io.config_error_recovery_enable_debug_log is True


# -- the rules that decay ---------------------------------------------------
#
# The guard is only a net while everything is inside it, and the overlays are
# where that will rot first: the next one added is a single ``x.draw(ctx)`` line
# in a method that is already forty lines of them.


def _overlays_tree():
    import ast
    import inspect
    import textwrap

    from warlock.studio import main as main_mod

    src = textwrap.dedent(inspect.getsource(main_mod.App._overlays))
    return ast.parse(src).body[0]


def test_every_overlay_is_drawn_through_the_guard():
    """One bare ``draw`` in ``_overlays`` is one surface back outside the net.

    ``_transition_overlay`` is the deliberate exception and is named here rather
    than pattern-matched: it is last, it paints a veil on the foreground list and
    submits no items, so there is nothing in it to fail and nothing a guard could
    put back.
    """
    import ast

    allowed_last = "_transition_overlay"
    bare = []
    for node in ast.walk(_overlays_tree()):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Name) and func.id in {"over", "run"}:
            continue
        if isinstance(func, ast.Attribute):
            if func.attr in {"run", "surface"}:
                continue
            if func.attr == allowed_last:
                continue
            bare.append(func.attr)
        elif isinstance(func, ast.Name):
            bare.append(func.id)
    assert not bare, f"drawn outside the guard in App._overlays: {bare}"


def test_the_guard_clears_its_census_beside_the_other_three():
    """``FRAME_FAILURES`` is one frame's record, like ``FRAME_PANES``,
    ``FRAME_ANCHORS`` and ``FRAME_CONTROLS``. A clear that goes missing turns it
    into a session-long list that a harness would read as this frame's."""
    import ast
    import inspect
    import textwrap

    from warlock.studio import main as main_mod

    src = textwrap.dedent(inspect.getsource(main_mod.App._build_ui))
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "begin_frame"
        for _ in [0]
    }
    assert calls == {"begin_frame"}
    owners = {
        node.func.value.id
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "begin_frame"
        and isinstance(node.func.value, ast.Name)
    }
    assert owners == {"guard", "layout_mod", "anchors", "probe"}, owners


def test_the_guard_never_catches_baseexception():
    """``KeyboardInterrupt`` and ``SystemExit`` are the user asking to leave, and
    a bare ``except`` would take both. Asserted structurally rather than by the
    two behaviour tests above, because those only cover the paths they name."""
    import ast
    import inspect

    from warlock.studio import layout as layout_mod

    for module in (guard, layout_mod):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            assert node.type is not None, f"bare except in {module.__name__}"
            names = (
                {node.type.id}
                if isinstance(node.type, ast.Name)
                else {e.id for e in getattr(node.type, "elts", []) if isinstance(e, ast.Name)}
            )
            assert "BaseException" not in names, f"BaseException caught in {module.__name__}"


def test_the_app_turns_the_recovery_assert_off_where_it_makes_its_context():
    """The flag defaults on and an ``IM_ASSERT`` is a ``RuntimeError`` here, so a
    context built without ``guard.configure()`` has a guard that cannot guard."""
    import ast
    import inspect
    import textwrap

    from warlock.studio import main as main_mod

    tree = ast.parse(textwrap.dedent(inspect.getsource(main_mod.App.setup_window)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    names = [f"{getattr(c.func.value, 'id', '')}.{c.func.attr}" for c in calls]
    assert "imgui.create_context" in names
    assert "guard.configure" in names
    assert names.index("guard.configure") > names.index("imgui.create_context")


def test_guarding_an_overlay_does_not_raise_imguis_implicit_debug_window(imgui_ctx):
    """The overlays are guarded at *host* scope -- ``main._overlays`` runs after
    ``imgui.end()`` -- and the mark the guard takes there used to be written
    against imgui's implicit ``Debug##Default`` window.

    ``NewFrame`` opens that fallback so a stray call cannot crash, and
    ``EndFrame`` closes it again *only if* ``WriteAccessed`` is still false.
    ``GetCurrentWindow`` sets that flag; ``GetCurrentWindowRead`` does not. With
    the write accessor a clean, failure-free frame rendered an empty 400x400
    "Debug" window over the app -- the state this asserts against, which is why
    the body below raises nothing.
    """
    imgui, _renderer = imgui_ctx
    imgui.new_frame()
    imgui.set_next_window_size((420.0, 900.0))
    imgui.begin("##host-overlay-scope")
    imgui.text("the workspace")
    imgui.end()
    # Host scope, exactly as ``_overlays`` runs: no window on the stack, and an
    # overlay that draws nothing this frame (a toast queue with no toasts).
    assert guard.run("overlay/quiet", lambda: None, draw_placeholder=False)
    imgui.render()

    fallback = imgui.internal.find_window_by_name("Debug##Default")
    assert fallback is None or not fallback.active, "the guard woke imgui's Debug window"
    drawn = {
        cmd_list.owner_name if hasattr(cmd_list, "owner_name") else ""
        for cmd_list in imgui.get_draw_data().cmd_lists
    }
    assert "Debug##Default" not in drawn


def test_no_studio_module_uses_the_write_marking_current_window_accessor():
    """``imgui.internal.get_current_window()`` is the one that sets
    ``WriteAccessed``. Anything asking at host scope -- the guard's mark, the
    probe's window column -- makes imgui render its implicit Debug window, so
    the read accessor is the only one this package may use.
    """
    import ast
    import pathlib

    import warlock.studio as studio_pkg

    root = pathlib.Path(studio_pkg.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_current_window"
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"use get_current_window_read instead: {offenders}"

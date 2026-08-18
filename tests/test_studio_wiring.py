"""Seams that existed but were not connected to anything.

Each of these pins one half of a pair that had drifted apart or never met: a
``persist`` nothing called at teardown, a mode switch written twice, a view
persisted as though it were a filter, an image-suffix list spelled five ways,
and a callback whose only handler was ``lambda _: None``.

Nothing here needs a GL context. The App is built with ``__new__`` and given
only the attributes the method under test reads, which is the shape
``test_studio_logging`` already uses -- the alternative is a real window, and a
teardown test that opens one is a teardown test that leaves one behind.
"""

from __future__ import annotations

import inspect
import re
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import (
    clay_mode,
    dialogs,
    filetypes,
    fonts,
    inker_mode,
    main,
    packwright_io,
    packwright_mode,
    palette,
    plotter_io,
    plotter_mode,
    plotter_tilesets,
    review_mode,
    theme,
    tokens,
    viewer_embed,
)
from warlock.studio import state as state_mod
from warlock.studio.state import AppState, Filters


class FakeSettings:
    """Just enough of ``Settings``: what was set, and how often it was written."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.flushes = 0

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def flush(self) -> None:
        self.flushes += 1


@pytest.fixture
def fake_pygame(monkeypatch):
    """``teardown`` imports pygame at call time and quits it.

    Stubbed, as ``test_splash`` does: a real ``pygame.quit()`` from a unit test
    tears down a display other tests in the session may be holding.
    """
    stub = SimpleNamespace(quit=lambda: None, display=SimpleNamespace(set_caption=lambda _t: None))
    monkeypatch.setitem(sys.modules, "pygame", stub)
    return stub


def _teardown_app(ctx: Any) -> main.App:
    """An App whose teardown reaches the persist steps and nothing after them."""
    app = main.App.__new__(main.App)
    app.runtime = SimpleNamespace(shutdown=lambda: None)
    app.window = app.imgui_renderer = app.viewer = app.eta = None
    app.clay_view = None
    app.app_ctx = ctx
    app._running = False
    app._last_frame = app._started_at = 0.0
    from warlock.studio.fps import FpsMeter

    app.fps = FpsMeter()
    return app


def _ctx(settings: FakeSettings, state: AppState | None = None) -> Any:
    return SimpleNamespace(
        settings=settings,
        state=state or AppState(),
        textures=None,
    )


# --- H48 / H49: the modes whose recents never reached the disk ---------------


def test_teardown_persists_every_mode_that_has_a_persist():
    """Plotter's and Packwright's ``persist`` were reachable only from inside
    their own save paths; the teardown list named Inker and Clay alone.

    Asserted against the set of modules that actually expose a ``persist``, not
    against four literals, because the next mode with a recent list is the one
    that would be forgotten again.
    """
    import importlib

    source = inspect.getsource(main.App)
    for name in ("inker_mode", "clay_mode", "plotter_mode", "packwright_mode"):
        module = importlib.import_module(f"warlock.studio.{name}")
        assert callable(module.persist)
        assert f"{name}.persist" in source, f"{name}.persist is never called"


def test_the_plotter_and_packwright_recents_survive_a_teardown(fake_pygame):
    settings = FakeSettings()
    state = AppState()
    ctx = _ctx(settings, state)

    plotter_mode.ensure(ctx)
    plotter_mode.remember_path(ctx, "D:/maps/one.wmap")
    packwright_mode.ensure(ctx)
    packwright_mode.remember_path(ctx, "D:/atlases/one.wpack")
    # The list is written by ``recents`` on the spot rather than flushed at
    # teardown -- which is the point: a mutation that forgot to persist used to
    # survive only because quitting wrote whatever the state happened to hold.
    _teardown_app(ctx).teardown()

    assert plotter_mode.recent_paths(ctx) == ["D:/maps/one.wmap"]
    assert packwright_mode.recent_paths(ctx) == ["D:/atlases/one.wpack"]


# --- H71: one write, and a label that names what it does ---------------------


def test_teardown_writes_the_settings_file_exactly_once(fake_pygame):
    """Every persist step used to flush, so quitting wrote the same JSON
    document once per mode. ``Settings`` holds the whole thing; the steps stay
    separate so one raising cannot cost the others, but the write does not."""
    settings = FakeSettings()
    ctx = _ctx(settings)
    plotter_mode.ensure(ctx)
    packwright_mode.ensure(ctx)

    _teardown_app(ctx).teardown()

    assert settings.flushes == 1


def test_teardown_releases_every_mode_texture_cache_before_the_viewer(fake_pygame, monkeypatch):
    """``inker_mode.release_all`` had no caller at all.

    Plotter's and Packwright's sweeps were both wired into teardown; Paint's --
    the largest of the three, holding a full-canvas composite, a floating
    buffer, a thumbnail per layer and up to 512 cel thumbnails per tab -- was
    written and never called.

    The order is asserted, not just the presence: every one of those textures
    was made from ``ctx.viewer.ctx``, so releasing the viewer first pulls the
    context out from under the sweep that is supposed to free them.
    """
    order: list[str] = []
    for name, module in (
        ("inker", inker_mode),
        ("plotter", plotter_mode),
        ("packwright", packwright_mode),
    ):
        monkeypatch.setattr(module, "release_all", lambda _ctx, _n=name: order.append(_n))

    app = _teardown_app(_ctx(FakeSettings()))
    app.viewer = SimpleNamespace(release=lambda: order.append("viewer"))
    app.teardown()

    assert order == ["inker", "plotter", "packwright", "viewer"]


def test_no_teardown_step_is_still_called_persist_build():
    """Clay was called Build once. A step labelled for a mode that no longer
    exists is what a log line says when the step fails."""
    source = inspect.getsource(main.App.teardown)
    assert "persist build" not in source
    assert "persist clay" in source


# --- H52: one mode switch, not two ------------------------------------------


def _palette_ctx(mode: str) -> Any:
    """The palette's own test shape: a duck-typed state, not an AppState."""
    return SimpleNamespace(
        state=SimpleNamespace(
            mode=mode,
            previous_mode=mode,
            mode_observed=mode,
        )
    )


def test_the_palette_goes_through_the_one_mode_switch():
    assert "set_mode" in inspect.getsource(palette._go)


def test_choosing_the_mode_you_are_already_in_keeps_the_history():
    """The drift: ``_set_mode`` early-returned on the current mode and the
    palette's copy did not, so re-picking the mode you were in recorded
    ``previous_mode == mode`` -- which ``_escape_mode`` reads as no history and
    answers with Home. Esc out of a pass-through mode is documented as going
    back to the mode it came from."""
    ctx = _palette_ctx("settings")
    ctx.state.previous_mode = "create"
    next(c for c in palette.commands(ctx) if c.key == "go:settings").run(ctx)

    assert ctx.state.mode == "settings"
    assert ctx.state.previous_mode == "create"


def test_the_palette_and_the_shortcut_agree_on_every_mode():
    """Parity, mode by mode, against ``App._set_mode`` driving the same state.

    A property rather than a case: the two were four lines each and differed in
    one of them, which no single example would have caught.
    """
    from warlock.studio import modes

    for start in modes.KEYS:
        for target in modes.KEYS:
            by_palette = _palette_ctx(start)
            by_palette.state.previous_mode = "create"
            palette._go(target)(by_palette)

            app = main.App.__new__(main.App)
            app.app_ctx = _palette_ctx(start)
            app.app_ctx.state.previous_mode = "create"
            app._set_mode(target)

            assert vars(by_palette.state) == vars(app.app_ctx.state), f"{start} -> {target}"


# --- H56: the trash is a view, and views are not persisted -------------------


def test_the_trash_view_is_not_written_to_the_settings_file(fake_pygame):
    settings = FakeSettings()
    state = AppState()
    state.filters = Filters(trash=True, status="error", text="chest")
    _teardown_app(_ctx(settings, state)).teardown()

    stored = settings.get("filters")
    assert "trash" not in stored
    # The rest of the bar is still remembered -- that is what it is for.
    assert stored["status"] == "error"
    assert stored["text"] == "chest"


def test_a_settings_file_that_still_carries_trash_opens_on_the_library():
    """Older builds really did write it, so the restore drops it too rather
    than trusting the writer."""
    restored = state_mod.filters_from_stored({"trash": True, "status": "done"})
    assert restored.trash is False
    assert restored.status == "done"


def test_an_unknown_stored_filter_key_is_dropped_rather_than_carried():
    assert not hasattr(state_mod.filters_from_stored({"nonesuch": 1}), "nonesuch")


def test_trash_is_the_stated_exception_and_the_list_is_reachable():
    """A name, not an inline literal at two call sites: excluding it on the way
    out and forgetting to on the way in is the failure this replaces."""
    assert state_mod.VOLATILE_FILTERS == ("trash",)
    for name in state_mod.VOLATILE_FILTERS:
        assert name in Filters.__annotations__


# --- H20: one spelling of where a reference image lives ----------------------


def test_review_mode_does_not_carry_its_own_copy_of_the_image_names():
    """The ``judge.STAGES``/``verdicts.STAGES`` hazard: the module held an
    identical tuple *and* read the service's, in two different functions."""
    assert not hasattr(review_mode, "REFERENCE_NAMES")


def test_the_reference_lookup_reads_the_service_tuple():
    from warlock.service import verdicts as verdicts_mod

    source = inspect.getsource(review_mode.reference_path)
    assert "verdicts_mod.IMAGE_NAMES" in source
    for name in verdicts_mod.IMAGE_NAMES:
        assert f'"{name}"' not in source


def test_the_reference_lookup_prefers_the_first_name_the_service_lists(tmp_path):
    """Order is the meaningful part: a text job writes both, and
    ``reference.png`` is what trellis actually saw."""
    from warlock.service import verdicts as verdicts_mod

    for name in verdicts_mod.IMAGE_NAMES:
        (tmp_path / name).write_bytes(b"x")
    found = review_mode.reference_path({"dir": str(tmp_path)})
    assert found is not None
    assert found.name == verdicts_mod.IMAGE_NAMES[0]


def test_the_reference_lookup_returns_none_when_neither_name_is_there(tmp_path):
    assert review_mode.reference_path({"dir": str(tmp_path)}) is None


# --- H75: the public advance ------------------------------------------------


def test_the_pane_skips_a_label_through_the_public_surface():
    assert hasattr(review_mode, "advance_labels")
    assert not hasattr(review_mode, "_advance_labels")
    assert "review_mode.advance_labels" in inspect.getsource(main.App)


# --- H51: the pose-dirty callback has a real consumer ------------------------


class FakeEditor:
    def __init__(self, dirty: bool = False) -> None:
        self.dirty = dirty
        self.cleared = False

    def has_unsaved_edits(self) -> bool:
        return self.dirty

    def clear(self) -> None:
        self.cleared = True
        self.dirty = False


def _viewer_shell(dirty: bool, pose_mode: bool = True) -> Any:
    """A ``Viewer`` with no GL: only what ``_notify_pose_dirty`` touches."""
    viewer = viewer_embed.Viewer.__new__(viewer_embed.Viewer)
    viewer.editor = FakeEditor(dirty)
    viewer.pose_mode = pose_mode
    viewer.on_pose_dirty = None
    return viewer


def test_the_viewer_reports_pose_dirt_to_whoever_is_listening():
    seen: list[bool] = []
    viewer = _viewer_shell(dirty=True)
    viewer.on_pose_dirty = seen.append
    viewer_embed.Viewer._notify_pose_dirty(viewer)
    assert seen == [True]


def test_leaving_the_editor_reports_clean():
    """Otherwise an indicator raised on the way in outlives every mesh loaded
    afterwards -- ``adopt_model`` and ``clear`` both exit pose mode."""
    seen: list[bool] = []
    viewer = _viewer_shell(dirty=True)
    viewer.on_pose_dirty = seen.append
    viewer._render_dirty = False
    viewer.pose_job_id = "a" * 12
    viewer.rotate_gizmo = SimpleNamespace(end_drag=lambda: None)
    viewer.translate_gizmo = SimpleNamespace(end_drag=lambda: None)

    viewer_embed.Viewer.exit_pose_mode(viewer)

    assert seen == [False]


def test_the_app_mirrors_pose_dirt_onto_the_state():
    """The handler was ``lambda _dirty: None`` -- a write with no reader."""
    app = main.App.__new__(main.App)
    app.app_ctx = _ctx(FakeSettings())
    titles: list[str] = []
    app._sync_title = lambda: titles.append(str(app.app_ctx.state.pose_dirty))

    app._on_pose_dirty(True)
    assert app.app_ctx.state.pose_dirty is True
    app._on_pose_dirty(True)
    # Unchanged: the caption is an OS call and the rotate gizmo reports on
    # every mouse-motion event while a joint is being dragged.
    assert titles == ["True"]
    app._on_pose_dirty(False)
    assert app.app_ctx.state.pose_dirty is False
    assert titles == ["True", "False"]


def test_the_handler_is_a_method_rather_than_a_no_op_lambda():
    source = inspect.getsource(main.App.setup_context)
    assert "self._on_pose_dirty" in source
    assert "lambda _dirty: None" not in source


# --- H50: the blocking strip is gone ----------------------------------------


def test_the_whole_strip_renderer_is_not_callable():
    """A draw plus a synchronous ``read_rgba`` per cell, sixteen cells to a
    frame. ``StripRender`` renders one per ``step()`` for exactly that reason,
    and a live method whose documented property is "never call this" is worse
    than an absent one."""
    assert not hasattr(viewer_embed.Viewer, "render_sheet_strip")
    assert hasattr(viewer_embed.Viewer, "begin_sheet_strip")
    assert hasattr(viewer_embed.Viewer, "step_sheet_strip")


# --- H55: one answer to "what counts as an image file" -----------------------


@pytest.mark.parametrize(
    "entry",
    [
        dialogs.IMAGE_FILTER,
        packwright_mode.IMAGE_FILTER,
        plotter_mode.TILESET_FILTER,
        inker_mode.OPEN_FILTER,
    ],
)
def test_every_image_picker_offers_every_droppable_suffix(entry):
    """The refusal, the accept and the picker have to agree. Two of these used
    to accept ``.jpeg`` and ``.bmp`` while advertising neither.

    ``entry[1]`` specifically -- the patterns of the *first* row, which is the
    one the picker opens on. This read ``" ".join(entry[1:])`` and so was
    satisfied by a suffix appearing anywhere in the tail, which is precisely
    what let three mis-paired filters through: joining the tail erases the
    pairing that decides which patterns are actually in force.
    """
    for suffix in filetypes.IMAGE_SUFFIXES:
        assert f"*{suffix}" in entry[1].split(), f"{suffix} missing from {entry[0]}"


# --- every picker filter is pairs, not a label and a pile of globs -----------


def _declared_filters():
    """Every ``*_FILTER`` list in the modules that own one, deduplicated.

    A scan rather than a hand-written parametrize list, because the bug this
    guards is one a *fourth* filter would reproduce: the two tests below were
    already parametrized over the four image pickers and passed while three of
    them were mis-paired.
    """
    modules = (
        clay_mode,
        dialogs,
        inker_mode,
        packwright_io,
        packwright_mode,
        plotter_io,
        plotter_mode,
        plotter_tilesets,
    )
    seen: dict[int, tuple[str, list[str]]] = {}
    for module in modules:
        for name, value in vars(module).items():
            if name.endswith("_FILTER") and isinstance(value, list):
                seen.setdefault(id(value), (f"{module.__name__.rpartition('.')[2]}.{name}", value))
    for suffix, value in dialogs.ARTIFACT_FILTERS.items():
        seen.setdefault(id(value), (f"ARTIFACT_FILTERS[{suffix!r}]", value))
    return sorted(seen.values())


_PICKER_FILTERS = _declared_filters()


@pytest.mark.parametrize(
    "name,entry", _PICKER_FILTERS, ids=[name for name, _ in _PICKER_FILTERS]
)
def test_a_picker_filter_is_label_pattern_pairs(name, entry):
    """portable-file-dialogs consumes ``filters`` **two at a time** -- a label,
    then that label's space-separated patterns -- and its loop is
    ``i + 1 < size``, so a trailing odd element is dropped without a word.

    A label followed by one entry per glob therefore does not mean "these
    patterns under this label"; it means the second glob becomes the first
    row's *only* pattern and the third glob becomes the second row's *label*.
    That is how "Add a tileset" came to open on a row advertising six formats
    and list only ``.tsx`` -- a folder of PNGs looked empty.

    ``pfd.all_files_filter()`` returning ``["All files", "*"]`` is the shape
    stated by the library itself.
    """
    assert len(entry) >= 2, f"{name} is not a label and its patterns"
    assert len(entry) % 2 == 0, f"{name} has {len(entry)} entries; pfd drops the last"
    for index in range(0, len(entry), 2):
        label, patterns = entry[index], entry[index + 1]
        assert not label.startswith("*"), f"{name}[{index}] is a pattern where a label goes"
        globs = patterns.split()
        assert globs, f"{name}[{index + 1}] has no patterns"
        for glob in globs:
            assert glob == "*" or glob.startswith("*."), f"{name}: {glob!r} is not a pattern"
        # A row that names formats must name *exactly* the ones beside it, both
        # directions. Advertising more is the dialog telling the user a format
        # is unsupported by the very picker that would have opened it;
        # advertising fewer is the same drift running the other way. A label
        # that names none ("Images") is deliberate and says nothing to break.
        advertised = set(re.findall(r"\*\.[A-Za-z0-9]+", label))
        if advertised:
            assert advertised == set(globs), f"{name}[{index}] says {advertised}, filters {globs}"


def test_no_module_writes_out_a_list_of_image_suffixes():
    """Five hand-written copies is how ``.webp`` comes to be droppable in three
    modes and refused in the fourth.

    "No line names two of them" rather than "no line names one": a single
    ``".png"`` is a legitimate answer to a different question (which filter a
    PNG export offers), while two on one line is a list being restated.
    """
    modules = (
        main,
        dialogs,
        inker_mode,
        packwright_mode,
        packwright_io,
        plotter_mode,
        plotter_io,
        plotter_tilesets,
    )
    for module in modules:
        for number, line in enumerate(inspect.getsource(module).splitlines(), 1):
            named = [s for s in filetypes.IMAGE_SUFFIXES if f'"{s}"' in line or f"*{s}" in line]
            assert len(named) < 2, f"{module.__name__}:{number} restates {named}"


def test_the_drop_router_names_the_shared_tuple():
    assert main.DROPPABLE_IMAGES is filetypes.IMAGE_SUFFIXES


def test_inker_opens_the_layered_format_on_top_of_the_shared_list():
    assert inker_mode.OPENABLE[0] == ".ora"
    assert tuple(inker_mode.OPENABLE[1:]) == filetypes.IMAGE_SUFFIXES


# --- H58 / H59 / H60 / H62: the inert names ----------------------------------


def test_the_palette_hook_resolves_only_palette_names():
    with pytest.raises(AttributeError):
        theme.__getattr__("RAISED")
    assert theme.__getattr__("ELEV_2") == theme.ELEV_2


def test_the_type_ramp_has_no_helper_for_the_default_face():
    """Regular is the first atlas font and is loaded at ``TEXT_BODY``, so it is
    already in force: pushing it would push what is pushed."""
    assert not hasattr(fonts, "body")
    for name in ("small", "label", "title"):
        assert hasattr(fonts, name)


def test_the_spacing_scale_carries_only_the_steps_in_use():
    """Every measurement token has a reader, and the rule is asserted rather
    than a list of the names that happened to fail it once.

    This used to name ``SP_6``, ``SP_10`` and ``RADIUS_L`` and assert their
    absence, which froze one afternoon's answer: ``SP_6`` came back the moment
    Home had a 24 px gap to own (UX.md Phase 0), and the test failed for the
    right reason with no way to say so. What ``tokens.py`` actually states is
    *a scale carries only the steps something spaces with* -- so the scan is
    the assertion, and a token added ahead of its call sites fails here by
    name.
    """
    import re
    from pathlib import Path

    root = Path(tokens.__file__).parent
    sources = [
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
        if path.name != "tokens.py"
    ]
    measurements = [
        name
        for name in vars(tokens)
        if re.fullmatch(r"(SP|RADIUS|TEXT|DUR)_[A-Z0-9]+", name)
    ]
    # A sanity floor on the scan itself: a regex that matched nothing would
    # make this pass by looking at an empty list.
    assert {"SP_1", "SP_4", "RADIUS_S", "TEXT_BODY", "DUR_BASE"} <= set(measurements)
    unread = [
        name
        for name in measurements
        if not any(re.search(rf"\b{name}\b", text) for text in sources)
    ]
    assert unread == []


def test_a_pending_task_carries_no_unread_timestamp():
    from warlock.studio.tasks import _Pending

    assert "started" not in _Pending.__dataclass_fields__


# --- H74: the popup stops stat-ing the log file per frame --------------------


def test_the_log_probe_stats_once_and_then_answers_from_the_cache(tmp_path, monkeypatch):
    """The popup draws every frame while it is open; the answer changes once a
    session, when the first line is written."""
    calls: list[str] = []
    real_exists = main.Path.exists

    def counting_exists(self):
        calls.append(str(self))
        return real_exists(self)

    monkeypatch.setattr(main.Path, "exists", counting_exists)
    monkeypatch.setattr(main, "_LOG_PROBE", ("", 0.0, False))

    assert main._log_exists(tmp_path) is False
    (tmp_path / "warlock.log").write_text("x")
    # Still the cached answer -- one stat, not two.
    assert main._log_exists(tmp_path) is False
    assert len(calls) == 1

    # And it is a *cache*, not a one-shot: the first line arrives after the
    # window does, so an answer frozen forever would grey the button for good.
    monkeypatch.setattr(main, "_LOG_PROBE", ("", 0.0, False))
    assert main._log_exists(tmp_path) is True


def test_the_diagnostics_popup_does_not_touch_the_filesystem_itself():
    source = inspect.getsource(main.App._diagnostics_popup)
    assert "_log_exists" in source
    assert ".exists()" not in source


# --- H61: the icon catalogue is deliberate -----------------------------------


def test_the_icon_module_says_it_is_a_catalogue():
    """Its numbers come from a named Lucide release and are never guessed, so
    an unreferenced constant here is correct rather than dead."""
    from warlock.studio import icons

    assert "catalogue" in (icons.__doc__ or "")

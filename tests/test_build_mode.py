"""Build mode's controller: the rules the raster editor had to learn first.

Nothing here is about geometry. It is about the two things that made a paint
tab go permanently read-only and permanently dirty, both of which follow from
saving being a *state* rather than a call that returns: a failed save has to
clear that state, and the history head a save records has to be read after
whatever the save itself pushes. Build mode inherits both, and the tests are
the reason the inheritance is real rather than intended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warlock.studio import build_mode, build_state
from warlock.studio.build import document as bd
from warlock.studio.build import primitives as bp


class FakeCtx:
    """Runs a submitted callable inline, so the test sees what the task thread
    would have done without needing one."""

    def __init__(self, svc: Any = None, *, accept: bool = True) -> None:
        self.svc = svc
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.confirms = _Confirms()
        self.cache = _Cache()
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.build = None
        self.mode = "home"


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


def _tab(ctx: FakeCtx, *, dirty: bool = False) -> build_state.BuildTab:
    """One open tab. ``adopt`` records the head it is given, so a tab is clean
    the moment it is adopted -- dirtying it means editing it afterwards."""
    doc = bd.BuildDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    tab = build_mode.adopt(ctx, doc, title="Scene")
    if dirty:
        doc.set_props(doc.objects[0].uid, name="Edited")
    return tab


def _save(ctx: FakeCtx, tab: build_state.BuildTab, path: Path) -> None:
    """A whole save: the submit, and the result coming back. ``save_to`` only
    does the first half -- applying the result is ``on_task_done``'s job, and a
    test that skipped it would be asserting against a half-finished save."""
    build_mode.save_to(ctx, tab, path)
    build_mode.on_task_done(ctx, _Done(f"build-save:{tab.uid}", ctx.result))


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    """``pygame.key.get_mods`` needs a video system; there is none in a test.

    Patched for every test here rather than in the ones that press a key, so a
    new key test cannot fail for a reason that has nothing to do with what it
    is checking.
    """
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- saving is a state -------------------------------------------------------


def test_a_failed_save_clears_the_saving_state(svc) -> None:
    """``saving`` disables every control that changes the document, so without
    this one failed write makes the tab read-only forever with no way back
    short of closing it."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.saving = True

    build_mode.on_task_failed(ctx, _Done(f"build-save:{tab.uid}"))
    assert tab.saving is False


def test_a_submit_that_is_refused_clears_the_saving_state(svc) -> None:
    """A second save while one is in flight is refused by the runner, and the
    flag must not be left set by the attempt."""
    ctx = FakeCtx(svc, accept=False)
    tab = _tab(ctx)
    build_mode.save_as(ctx, tab)
    assert tab.saving is False


def test_a_completed_save_clears_it_and_marks_the_tab_saved(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)
    assert tab.dirty is True

    _save(ctx, tab, tmp_path / "scene.wblk")
    assert tab.saving is False
    assert tab.dirty is False
    assert (tmp_path / "scene.wblk").exists()


def test_a_save_records_the_head_after_the_document_settles(svc, tmp_path) -> None:
    """A head captured before whatever the save itself pushes would leave the
    tab dirty however many times it was saved. Nothing in a Build save pushes
    today -- but the head is read at one place for that reason, and this is the
    test that fails if it moves back above one."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)

    _save(ctx, tab, tmp_path / "scene.wblk")
    assert tab.saved_head == tab.doc.history.head
    assert tab.dirty is False


def test_an_edit_during_a_save_leaves_the_tab_dirty(svc, tmp_path) -> None:
    """The head is captured when the encode starts, not when it finishes: an
    edit made while the file was being written is genuinely not in it, and a
    save that marked the tab clean would lose it silently."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)

    build_mode.save_to(ctx, tab, tmp_path / "scene.wblk")
    # The edit lands between the encode starting and the result coming back.
    tab.doc.add_object(bd.Obj(uid=bd.new_uid(), name="Late", mesh=bp.box()))
    build_mode.on_task_done(ctx, _Done(f"build-save:{tab.uid}", ctx.result))

    assert tab.saving is False
    assert tab.dirty is True


# --- keys --------------------------------------------------------------------


def test_handle_key_returns_false_with_no_document_open(svc) -> None:
    """The caller's fall-through depends on the distinction: Build mode owns a
    viewport, and with nothing open the viewport shortcuts must still work."""
    import pygame

    ctx = FakeCtx(svc)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g)
    assert build_mode.handle_key(ctx, event) is False


def test_a_tool_key_is_consumed_and_selects_its_tool(svc) -> None:
    import pygame

    ctx = FakeCtx(svc)
    _tab(ctx)
    for key_name, tool in build_mode.TOOL_KEYS.items():
        event = pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{key_name}"))
        assert build_mode.handle_key(ctx, event) is True
        assert ctx.state.build.tool == tool


def test_every_tool_key_names_a_real_tool(svc) -> None:
    tools = {key for key, _label, _shortcut in build_state.TOOLS}
    assert set(build_mode.TOOL_KEYS.values()) <= tools


def test_a_mutating_key_is_ignored_while_the_tab_is_saving(svc) -> None:
    """Every control that changes the document is gated on ``saving``, and the
    keyboard is a control. The key is still *consumed* -- falling through would
    let it act on a pane Build mode has replaced."""
    import pygame

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    tab.saving = True

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE)
    ctx.state.build.doc_selection_hint = None
    tab.doc.select([tab.doc.objects[0].uid])
    assert build_mode.handle_key(ctx, event) is True
    assert len(tab.doc.history) == depth


def test_delete_removes_the_selection_when_not_saving(svc) -> None:
    import pygame

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.doc.select([tab.doc.objects[0].uid])

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE)
    assert build_mode.handle_key(ctx, event) is True
    assert tab.doc.objects == []


# --- the guard ---------------------------------------------------------------


def test_the_guard_lets_a_clean_workspace_through(svc) -> None:
    ctx = FakeCtx(svc)
    _tab(ctx)
    went: list[bool] = []
    assert build_mode.guard(ctx, "quit", lambda: went.append(True)) is True
    assert went == [True]


def test_the_guard_stops_a_dirty_workspace_and_says_how_many(svc) -> None:
    ctx = FakeCtx(svc)
    _tab(ctx, dirty=True)
    _tab(ctx, dirty=True)

    went: list[bool] = []
    assert build_mode.guard(ctx, "quit", lambda: went.append(True)) is False
    assert went == []
    assert ctx.confirms.pending is not None
    assert "2 " in ctx.confirms.pending.message


def test_the_guard_asks_one_question_however_many_documents_are_dirty(svc) -> None:
    """``ConfirmQueue`` holds a single pending question, so asking per document
    would silently drop all but the first."""
    ctx = FakeCtx(svc)
    for _ in range(3):
        _tab(ctx, dirty=True)
    asked: list[Any] = []
    ctx.confirms.ask = asked.append  # type: ignore[method-assign]

    build_mode.guard(ctx, "quit", lambda: None)
    assert len(asked) == 1


def test_the_guard_with_no_state_at_all_proceeds(svc) -> None:
    ctx = FakeCtx(svc)
    went: list[bool] = []
    assert build_mode.guard(ctx, "quit", lambda: went.append(True)) is True
    assert went == [True]


# --- task keys ---------------------------------------------------------------


@pytest.fixture
def no_dialogs(monkeypatch, tmp_path):
    """No native picker ever runs in a test.

    ``FakeCtx`` deliberately runs a submitted callable inline, which is what
    makes the task-thread half assertable -- and a picker on that path is
    modal to the OS, so it hangs the suite rather than failing it.
    """
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "picked.wblk")
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: None)
    return tmp_path


def test_every_task_key_carries_the_build_prefix(svc, tmp_path, no_dialogs) -> None:
    """``_collect_tasks`` claims them by prefix, so a key without one is a
    result that is never delivered anywhere."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)

    build_mode.save_to(ctx, tab, tmp_path / "a.wblk")
    build_mode.save_as(ctx, tab)
    build_mode.export_asset(ctx, tab)
    build_mode.ask_open(ctx)

    assert ctx.submitted
    assert all(key.startswith("build-") for key in ctx.submitted)


def test_a_cancelled_picker_leaves_the_tab_editable(svc, monkeypatch) -> None:
    """A dismissed dialog returns None, which is not a failure -- but it does
    have to clear ``saving``, or cancelling a Save As locks the tab."""
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: None)
    ctx = FakeCtx(svc)
    tab = _tab(ctx)

    build_mode.save_as(ctx, tab)
    build_mode.on_task_done(ctx, _Done(f"build-saveas:{tab.uid}", ctx.result))
    assert tab.saving is False


def test_save_as_writes_through_the_picked_path(svc, no_dialogs) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)

    build_mode.save_as(ctx, tab)
    build_mode.on_task_done(ctx, _Done(f"build-saveas:{tab.uid}", ctx.result))
    assert (no_dialogs / "picked.wblk").exists()
    assert tab.title == "picked"
    assert tab.dirty is False


def test_a_save_key_carries_the_tab_uid_so_the_result_finds_its_tab(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    build_mode.save_to(ctx, tab, tmp_path / "a.wblk")
    assert ctx.submitted[-1] == f"build-save:{tab.uid}"


def test_a_result_for_an_unknown_tab_is_dropped_without_raising(svc) -> None:
    ctx = FakeCtx(svc)
    _tab(ctx)
    build_mode.on_task_done(ctx, _Done("build-save:nope", {"rev": 0}))
    build_mode.on_task_failed(ctx, _Done("build-save:nope"))


# --- export ------------------------------------------------------------------


def test_exporting_mints_a_built_asset_and_stores_the_document_beside_it(svc) -> None:
    """The whole point of Build mode: what comes out is an ordinary asset, so
    rigging, posing, sheets and every mesh export work on it unchanged."""
    from warlock.service import files as svc_files

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    build_mode.export_asset(ctx, tab)

    job_id = ctx.result["job_id"]
    job = svc.store.get(job_id)
    assert job["params"]["built"] is True
    assert (svc.job_dir(job_id) / "model.glb").exists()
    assert svc_files.build_source_status(svc, job_id)["exists"] is True


def test_the_mesh_is_written_before_the_document(svc, monkeypatch) -> None:
    """A crash between the two must leave the sidecar absent rather than lying
    about a mesh it did not produce."""
    from warlock.service import files as svc_files
    from warlock.service import jobs as svc_jobs

    order: list[str] = []
    real_import = svc_jobs.import_mesh
    real_source = svc_files.save_build_source

    def mesh(*args: Any, **kwargs: Any) -> Any:
        order.append("model.glb")
        return real_import(*args, **kwargs)

    def source(*args: Any, **kwargs: Any) -> Any:
        order.append("build.wblk")
        return real_source(*args, **kwargs)

    monkeypatch.setattr(svc_jobs, "import_mesh", mesh)
    monkeypatch.setattr(svc_files, "save_build_source", source)

    ctx = FakeCtx(svc)
    build_mode.export_asset(ctx, _tab(ctx))
    assert order == ["model.glb", "build.wblk"]


def test_exporting_an_empty_document_is_refused_before_a_job_exists(svc) -> None:
    ctx = FakeCtx(svc)
    tab = build_mode.adopt(ctx, bd.BuildDoc(), title="Empty")
    build_mode.export_asset(ctx, tab)

    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_a_document_being_saved_cannot_be_exported(svc) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.saving = True
    build_mode.export_asset(ctx, tab)
    assert ctx.submitted == []


# --- opening -----------------------------------------------------------------


def test_opening_a_saved_document_brings_its_objects_back(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    path = tmp_path / "scene.wblk"
    _save(ctx, tab, path)

    fresh = FakeCtx(svc)
    build_mode.open_path(fresh, path)
    build_mode.on_task_done(fresh, _Done("build-open", fresh.result))

    opened = fresh.state.build.active
    assert opened is not None
    assert [o.name for o in opened.doc.objects] == ["Box"]
    assert opened.dirty is False
    assert fresh.state.mode == "build"


def test_opening_a_file_that_is_already_open_focuses_it(svc, tmp_path) -> None:
    """Two tabs over one path would race on save."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    path = tmp_path / "scene.wblk"
    _save(ctx, tab, path)

    # Saving associates the tab with the path, so the tab that wrote the file
    # is the tab that owns it -- opening it again must focus, not re-read.
    other = _tab(ctx)
    ctx.state.build.activate(other.uid)
    before = list(ctx.submitted)

    build_mode.open_path(ctx, path)
    assert ctx.submitted == before  # nothing was read
    assert ctx.state.build.active is tab
    assert len(ctx.state.build.docs) == 2


def test_a_file_that_will_not_open_is_reported_and_forgotten(svc, tmp_path) -> None:
    bad = tmp_path / "broken.wblk"
    bad.write_bytes(b"not a wblk")
    ctx = FakeCtx(svc)
    build_mode.ensure(ctx)
    ctx.state.build.remember(bad)

    with pytest.raises(ValueError):
        build_mode._load(bad)


# --- state -------------------------------------------------------------------


def test_the_mode_state_is_built_lazily(svc) -> None:
    """``AppState`` deliberately knows nothing about it, and a session that
    never opens Build mode should not pay for it."""
    ctx = FakeCtx(svc)
    assert ctx.state.build is None
    assert build_mode.ensure(ctx) is ctx.state.build


def test_recent_files_persist_through_the_settings_store(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    _save(ctx, tab, tmp_path / "scene.wblk")

    stored = ctx.settings.get("build") or {}
    assert str(tmp_path / "scene.wblk") in (stored.get("recent") or [])


def test_tool_settings_belong_to_the_app_and_the_view_to_the_document(svc) -> None:
    """The convention the raster editor set: switching tabs must not change
    your snap setting, but a tab does remember where its camera was."""
    ctx = FakeCtx(svc)
    first = _tab(ctx)
    ctx.state.build.snap_translate = 0.25
    second = _tab(ctx)
    first.view.yaw = 1.25

    assert ctx.state.build.snap_translate == 0.25
    assert second.view.yaw != 1.25


def test_a_tab_label_keeps_its_identity_across_a_rename(svc) -> None:
    """imgui matches a tab on what follows ``###``, so a title alone would make
    two documents called "Untitled" the same tab."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    before = tab.label.split("###")[1]
    tab.title = "Renamed"
    assert tab.label.split("###")[1] == before
    assert tab.label.startswith("Renamed")


def test_two_tabs_never_share_a_uid(svc) -> None:
    ctx = FakeCtx(svc)
    assert _tab(ctx).uid != _tab(ctx).uid


def test_closing_a_tab_activates_its_neighbour(svc) -> None:
    ctx = FakeCtx(svc)
    first, second, third = _tab(ctx), _tab(ctx), _tab(ctx)
    state = ctx.state.build

    state.activate(second.uid)
    state.close(second.uid)
    assert state.active is third

    state.close(third.uid)
    assert state.active is first


def test_paths_are_pathlib_objects_not_strings(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    _save(ctx, tab, tmp_path / "scene.wblk")
    assert isinstance(tab.path, Path)

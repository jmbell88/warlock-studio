"""Clay's controller: the rules the raster editor had to learn first.

Nothing here is about geometry. It is about the two things that made a paint
tab go permanently read-only and permanently dirty, both of which follow from
saving being a *state* rather than a call that returns: a failed save has to
clear that state, and the history head a save records has to be read after
whatever the save itself pushes. Clay inherits both, and the tests are
the reason the inheritance is real rather than intended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warlock.studio import clay_mode, clay_state
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import primitives as bp


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
        self.clay = None
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


def _tab(ctx: FakeCtx, *, dirty: bool = False) -> clay_state.ClayTab:
    """One open tab. ``adopt`` records the head it is given, so a tab is clean
    the moment it is adopted -- dirtying it means editing it afterwards."""
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    if dirty:
        doc.set_props(doc.objects[0].uid, name="Edited")
    return tab


def _save(ctx: FakeCtx, tab: clay_state.ClayTab, path: Path) -> None:
    """A whole save: the submit, and the result coming back. ``save_to`` only
    does the first half -- applying the result is ``on_task_done``'s job, and a
    test that skipped it would be asserting against a half-finished save."""
    clay_mode.save_to(ctx, tab, path)
    clay_mode.on_task_done(ctx, _Done(f"clay-save:{tab.uid}", ctx.result))


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

    clay_mode.on_task_failed(ctx, _Done(f"clay-save:{tab.uid}"))
    assert tab.saving is False


def test_a_submit_that_is_refused_clears_the_saving_state(svc) -> None:
    """A second save while one is in flight is refused by the runner, and the
    flag must not be left set by the attempt."""
    ctx = FakeCtx(svc, accept=False)
    tab = _tab(ctx)
    clay_mode.save_as(ctx, tab)
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
    tab dirty however many times it was saved. Nothing in a Clay save pushes
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

    clay_mode.save_to(ctx, tab, tmp_path / "scene.wblk")
    # The edit lands between the encode starting and the result coming back.
    tab.doc.add_object(bd.Obj(uid=bd.new_uid(), name="Late", mesh=bp.box()))
    clay_mode.on_task_done(ctx, _Done(f"clay-save:{tab.uid}", ctx.result))

    assert tab.saving is False
    assert tab.dirty is True


# --- keys --------------------------------------------------------------------


def test_handle_key_returns_false_with_no_document_open(svc) -> None:
    """The caller's fall-through depends on the distinction: Clay owns a
    viewport, and with nothing open the viewport shortcuts must still work."""
    import pygame

    ctx = FakeCtx(svc)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g)
    assert clay_mode.handle_key(ctx, event) is False


def test_a_tool_key_is_consumed_and_selects_its_tool(svc) -> None:
    import pygame

    ctx = FakeCtx(svc)
    _tab(ctx)
    for key_name, tool in clay_mode.TOOL_KEYS.items():
        event = pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{key_name}"))
        assert clay_mode.handle_key(ctx, event) is True
        assert ctx.state.clay.tool == tool


def test_every_tool_key_names_a_real_tool(svc) -> None:
    tools = {key for key, _label, _shortcut in clay_state.TOOLS}
    assert set(clay_mode.TOOL_KEYS.values()) <= tools


def test_a_mutating_key_is_ignored_while_the_tab_is_saving(svc) -> None:
    """Every control that changes the document is gated on ``saving``, and the
    keyboard is a control. The key is still *consumed* -- falling through would
    let it act on a pane Clay has replaced."""
    import pygame

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    tab.saving = True

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE)
    ctx.state.clay.doc_selection_hint = None
    tab.doc.select([tab.doc.objects[0].uid])
    assert clay_mode.handle_key(ctx, event) is True
    assert len(tab.doc.history) == depth


def test_delete_removes_the_selection_when_not_saving(svc) -> None:
    import pygame

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.doc.select([tab.doc.objects[0].uid])

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE)
    assert clay_mode.handle_key(ctx, event) is True
    assert tab.doc.objects == []


# --- the guard ---------------------------------------------------------------


def test_the_guard_lets_a_clean_workspace_through(svc) -> None:
    ctx = FakeCtx(svc)
    _tab(ctx)
    went: list[bool] = []
    assert clay_mode.guard(ctx, "quit", lambda: went.append(True)) is True
    assert went == [True]


def test_the_guard_stops_a_dirty_workspace_and_says_how_many(svc) -> None:
    ctx = FakeCtx(svc)
    _tab(ctx, dirty=True)
    _tab(ctx, dirty=True)

    went: list[bool] = []
    assert clay_mode.guard(ctx, "quit", lambda: went.append(True)) is False
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

    clay_mode.guard(ctx, "quit", lambda: None)
    assert len(asked) == 1


def test_the_guard_with_no_state_at_all_proceeds(svc) -> None:
    ctx = FakeCtx(svc)
    went: list[bool] = []
    assert clay_mode.guard(ctx, "quit", lambda: went.append(True)) is True
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


def test_every_task_key_carries_the_clay_prefix(svc, tmp_path, no_dialogs) -> None:
    """``_collect_tasks`` claims them by prefix, so a key without one is a
    result that is never delivered anywhere."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)

    clay_mode.save_to(ctx, tab, tmp_path / "a.wblk")
    clay_mode.save_as(ctx, tab)
    clay_mode.export_asset(ctx, tab)
    clay_mode.ask_open(ctx)

    assert ctx.submitted
    assert all(key.startswith("clay-") for key in ctx.submitted)


def test_a_cancelled_picker_leaves_the_tab_editable(svc, monkeypatch) -> None:
    """A dismissed dialog returns None, which is not a failure -- but it does
    have to clear ``saving``, or cancelling a Save As locks the tab."""
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: None)
    ctx = FakeCtx(svc)
    tab = _tab(ctx)

    clay_mode.save_as(ctx, tab)
    clay_mode.on_task_done(ctx, _Done(f"clay-saveas:{tab.uid}", ctx.result))
    assert tab.saving is False


def test_save_as_writes_through_the_picked_path(svc, no_dialogs) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)

    clay_mode.save_as(ctx, tab)
    clay_mode.on_task_done(ctx, _Done(f"clay-saveas:{tab.uid}", ctx.result))
    assert (no_dialogs / "picked.wblk").exists()
    assert tab.title == "picked"
    assert tab.dirty is False


def test_save_as_leaves_a_readable_document_and_no_temporary_behind(
    svc, no_dialogs
) -> None:
    """Staged like ``save_to``: the ``.tmp`` is an implementation detail that
    must not survive the write, and what lands is a document that reopens."""
    from warlock.studio.clay import serialize

    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)

    clay_mode.save_as(ctx, tab)
    clay_mode.on_task_done(ctx, _Done(f"clay-saveas:{tab.uid}", ctx.result))

    written = no_dialogs / "picked.wblk"
    doc = serialize.read_wblk(written.read_bytes())
    assert [obj.name for obj in doc.objects] == [obj.name for obj in tab.doc.objects]
    assert list(no_dialogs.glob("*.tmp")) == []


def test_a_save_as_that_dies_partway_leaves_the_old_file_intact(
    svc, no_dialogs, monkeypatch
) -> None:
    """The reason for staging: a picker aimed at an existing document is the
    ordinary way to overwrite one, and an unstaged failure truncates it."""
    existing = no_dialogs / "picked.wblk"
    existing.write_bytes(b"the document that was already there")

    def boom(_self: Path, _data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", boom)
    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)

    # ``FakeCtx.submit`` runs the task inline, so the failure the real task
    # thread would report through ``on_task_failed`` surfaces here as a raise.
    with pytest.raises(OSError):
        clay_mode.save_as(ctx, tab)

    assert existing.read_bytes() == b"the document that was already there"


def test_a_save_key_carries_the_tab_uid_so_the_result_finds_its_tab(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    clay_mode.save_to(ctx, tab, tmp_path / "a.wblk")
    assert ctx.submitted[-1] == f"clay-save:{tab.uid}"


def test_a_result_for_an_unknown_tab_is_dropped_without_raising(svc) -> None:
    ctx = FakeCtx(svc)
    _tab(ctx)
    clay_mode.on_task_done(ctx, _Done("clay-save:nope", {"rev": 0}))
    clay_mode.on_task_failed(ctx, _Done("clay-save:nope"))


# --- export ------------------------------------------------------------------


def test_exporting_mints_a_built_asset_and_stores_the_document_beside_it(svc) -> None:
    """The whole point of Clay: what comes out is an ordinary asset, so
    rigging, posing, sheets and every mesh export work on it unchanged."""
    from warlock.service import files as svc_files

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    clay_mode.export_asset(ctx, tab)

    job_id = ctx.result["job_id"]
    job = svc.store.get(job_id)
    assert job["params"]["built"] is True
    assert (svc.job_dir(job_id) / "model.glb").exists()
    assert svc_files.clay_source_status(svc, job_id)["exists"] is True


def test_the_mesh_is_written_before_the_document(svc, monkeypatch) -> None:
    """A crash between the two must leave the sidecar absent rather than lying
    about a mesh it did not produce."""
    from warlock.service import files as svc_files
    from warlock.service import jobs as svc_jobs

    order: list[str] = []
    real_import = svc_jobs.import_mesh
    real_source = svc_files.save_clay_source

    def mesh(*args: Any, **kwargs: Any) -> Any:
        order.append("model.glb")
        return real_import(*args, **kwargs)

    def source(*args: Any, **kwargs: Any) -> Any:
        order.append("build.wblk")
        return real_source(*args, **kwargs)

    monkeypatch.setattr(svc_jobs, "import_mesh", mesh)
    monkeypatch.setattr(svc_files, "save_clay_source", source)

    ctx = FakeCtx(svc)
    clay_mode.export_asset(ctx, _tab(ctx))
    assert order == ["model.glb", "build.wblk"]


def test_exporting_an_empty_document_is_refused_before_a_job_exists(svc) -> None:
    ctx = FakeCtx(svc)
    tab = clay_mode.adopt(ctx, bd.ClayDoc(), title="Empty")
    clay_mode.export_asset(ctx, tab)

    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_a_document_being_saved_cannot_be_exported(svc) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.saving = True
    clay_mode.export_asset(ctx, tab)
    assert ctx.submitted == []


# --- opening -----------------------------------------------------------------


def test_opening_a_saved_document_brings_its_objects_back(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    path = tmp_path / "scene.wblk"
    _save(ctx, tab, path)

    fresh = FakeCtx(svc)
    clay_mode.open_path(fresh, path)
    clay_mode.on_task_done(fresh, _Done("clay-open", fresh.result))

    opened = fresh.state.clay.active
    assert opened is not None
    assert [o.name for o in opened.doc.objects] == ["Box"]
    assert opened.dirty is False
    assert fresh.state.mode == "clay"


def test_opening_a_file_that_is_already_open_focuses_it(svc, tmp_path) -> None:
    """Two tabs over one path would race on save."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    path = tmp_path / "scene.wblk"
    _save(ctx, tab, path)

    # Saving associates the tab with the path, so the tab that wrote the file
    # is the tab that owns it -- opening it again must focus, not re-read.
    other = _tab(ctx)
    ctx.state.clay.activate(other.uid)
    before = list(ctx.submitted)

    clay_mode.open_path(ctx, path)
    assert ctx.submitted == before  # nothing was read
    assert ctx.state.clay.active is tab
    assert len(ctx.state.clay.docs) == 2


def test_a_file_that_will_not_open_is_reported_and_forgotten(svc, tmp_path) -> None:
    bad = tmp_path / "broken.wblk"
    bad.write_bytes(b"not a wblk")
    ctx = FakeCtx(svc)
    clay_mode.ensure(ctx)
    clay_mode.remember_path(ctx, bad)

    with pytest.raises(ValueError):
        clay_mode._load(bad)


# --- state -------------------------------------------------------------------


def test_the_mode_state_is_built_lazily(svc) -> None:
    """``AppState`` deliberately knows nothing about it, and a session that
    never opens Clay should not pay for it."""
    ctx = FakeCtx(svc)
    assert ctx.state.clay is None
    assert clay_mode.ensure(ctx) is ctx.state.clay


def test_recent_files_persist_through_the_settings_store(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    _save(ctx, tab, tmp_path / "scene.wblk")

    # Through ``recents``, the one merged list, rather than a per-mode key:
    # four independent lists could not be merged into Home's Resume list at all.
    assert str(tmp_path / "scene.wblk") in clay_mode.recent_paths(ctx)


def test_tool_settings_belong_to_the_app_and_the_view_to_the_document(svc) -> None:
    """The convention the raster editor set: switching tabs must not change
    your snap setting, but a tab does remember where its camera was."""
    ctx = FakeCtx(svc)
    first = _tab(ctx)
    ctx.state.clay.snap_translate = 0.25
    second = _tab(ctx)
    first.view.yaw = 1.25

    assert ctx.state.clay.snap_translate == 0.25
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
    state = ctx.state.clay

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


# --- importing a mesh (T24) --------------------------------------------------


def _glb_bytes() -> bytes:
    from warlock.studio.viewer import glbwrite

    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return glbwrite.write_glb(bd.to_model(doc))


class _ImportSvc:
    """Just enough service for ``clay_source_path`` and ``config.job_dir``."""

    def __init__(self, root: Path) -> None:
        self.config = _ImportConfig(root)

    def job_dir(self, job_id: str) -> Path:
        return self.config.job_dir(job_id)


class _ImportConfig:
    def __init__(self, root: Path) -> None:
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


def test_importing_a_glb_parses_off_the_frame_thread(tmp_path: Path) -> None:
    ctx = FakeCtx()
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())

    clay_mode.import_glb_path(ctx, path)
    assert ctx.submitted[-1].startswith("clay-import:")
    assert ctx.result["title"] == "thing"
    assert len(ctx.result["doc"].objects) == 1
    assert ctx.result["triangles"] == 12


def test_a_parsed_import_is_adopted_and_switches_mode(tmp_path: Path) -> None:
    ctx = FakeCtx()
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())
    clay_mode.import_glb_path(ctx, path)
    clay_mode.on_task_done(ctx, _Done("clay-import:1", ctx.result))

    state = clay_mode.ensure(ctx)
    assert len(state.docs) == 1
    assert state.docs[0].path is None, "an import has no file of its own to save over"
    assert ctx.state.mode == "clay"


def test_a_big_import_asks_before_adopting(tmp_path: Path) -> None:
    ctx = FakeCtx()
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())
    clay_mode.import_glb_path(ctx, path)
    result = dict(ctx.result, triangles=clay_mode.SLOW_TRIANGLES + 1)

    clay_mode.on_task_done(ctx, _Done("clay-import:1", result))
    state = clay_mode.ensure(ctx)
    assert state.docs == [], "nothing is adopted until the user says so"
    assert ctx.confirms.pending is not None

    ctx.confirms.pending.on_confirm()
    assert len(state.docs) == 1
    assert ctx.state.mode == "clay"


def test_editing_an_asset_prefers_the_authored_document_over_the_mesh(
    tmp_path: Path,
) -> None:
    """The ``build.wblk`` sidecar was written and never read back until now.

    It is the document the user authored -- objects, names, generator
    parameters -- and importing the GLB instead would hand them back a single
    frozen triangle soup of their own work.
    """
    from warlock.studio.clay import serialize

    job_dir = tmp_path / "0123456789ab"
    job_dir.mkdir(parents=True)
    (job_dir / "model.glb").write_bytes(_glb_bytes())
    authored = bd.ClayDoc()
    authored.objects.append(
        bd.Obj(uid=bd.new_uid(), name="Authored", mesh=bp.cone(), generator="cone")
    )
    (job_dir / "build.wblk").write_bytes(serialize.wblk_bytes(authored))

    ctx = FakeCtx(_ImportSvc(tmp_path))
    clay_mode.edit_asset_in_clay(ctx, {"id": "0123456789ab", "name": "Job One"})
    assert ctx.result["doc"].objects[0].name == "Authored"
    assert ctx.result["doc"].objects[0].generator == "cone", "still parametric"


def test_editing_an_asset_without_a_sidecar_imports_the_served_mesh(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "0123456789cd"
    job_dir.mkdir(parents=True)
    (job_dir / "model.glb").write_bytes(_glb_bytes())
    (job_dir / "source.glb").write_bytes(b"never read")

    ctx = FakeCtx(_ImportSvc(tmp_path))
    clay_mode.edit_asset_in_clay(ctx, {"id": "0123456789cd", "name": "Job Two"})
    assert ctx.result["title"] == "Job Two"
    assert ctx.result["doc"].objects[0].generator is None, "an imported mesh is frozen"


def test_editing_an_asset_with_no_mesh_raises_rather_than_opening_nothing(
    tmp_path: Path,
) -> None:
    (tmp_path / "0123456789ef").mkdir(parents=True)
    ctx = FakeCtx(_ImportSvc(tmp_path))
    with pytest.raises(FileNotFoundError):
        clay_mode.edit_asset_in_clay(ctx, {"id": "0123456789ef", "name": "Empty"})


# --- adoption switches modes through set_mode (H14) --------------------------
#
# ``state.set_mode`` is the one implementation, and a bare ``ctx.state.mode =``
# skips the ``previous_mode``/``mode_observed`` pair it maintains -- so Esc out
# of the next pass-through mode would go back to wherever a *keypress* last
# was, not to Clay. The three adopt paths are the ones that used to assign.


def test_opening_a_document_records_where_clay_was_entered_from(svc, tmp_path) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    path = tmp_path / "scene.wblk"
    _save(ctx, tab, path)

    fresh = FakeCtx(svc)
    fresh.state.mode = "library"
    clay_mode.open_path(fresh, path)
    clay_mode.on_task_done(fresh, _Done("clay-open", fresh.result))

    assert fresh.state.mode == "clay"
    assert fresh.state.previous_mode == "library"
    assert fresh.state.mode_observed == "clay"


def test_an_adopted_import_records_where_clay_was_entered_from(tmp_path: Path) -> None:
    ctx = FakeCtx()
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())
    clay_mode.import_glb_path(ctx, path)
    clay_mode.on_task_done(ctx, _Done("clay-import:1", ctx.result))

    assert ctx.state.mode == "clay"
    assert ctx.state.previous_mode == "home"
    assert ctx.state.mode_observed == "clay"


def test_a_confirmed_big_import_records_where_clay_was_entered_from(tmp_path: Path) -> None:
    ctx = FakeCtx()
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())
    clay_mode.import_glb_path(ctx, path)
    result = dict(ctx.result, triangles=clay_mode.SLOW_TRIANGLES + 1)
    clay_mode.on_task_done(ctx, _Done("clay-import:1", result))

    ctx.confirms.pending.on_confirm()
    assert ctx.state.mode == "clay"
    assert ctx.state.previous_mode == "home"
    assert ctx.state.mode_observed == "clay"


def test_adopting_while_already_in_clay_keeps_the_escape_history(tmp_path: Path) -> None:
    """``set_mode`` early-returns on the mode it is already in -- the exact
    drift the palette's hand copy had: without it, ``previous_mode`` becomes
    ``clay`` itself and Esc's history is gone."""
    ctx = FakeCtx()
    ctx.state.mode = "clay"
    ctx.state.previous_mode = "create"
    ctx.state.mode_observed = "clay"
    path = tmp_path / "thing.glb"
    path.write_bytes(_glb_bytes())
    clay_mode.import_glb_path(ctx, path)
    clay_mode.on_task_done(ctx, _Done("clay-import:1", ctx.result))

    assert ctx.state.mode == "clay"
    assert ctx.state.previous_mode == "create"


# --- the way in -------------------------------------------------------------


def test_the_home_tile_mints_a_document_when_clay_is_empty() -> None:
    """The tile says "model something", so arriving with nothing open and no
    obvious way to begin is a dead end."""
    from warlock.studio.panes import landing

    ctx = FakeCtx()
    landing.start_clay(ctx)
    assert ctx.state.mode == "clay"
    state = clay_mode.ensure(ctx)
    assert len(state.docs) == 1
    assert state.active is not None
    assert state.active.title == "Untitled"


def test_the_home_tile_opens_nothing_over_work_already_there() -> None:
    """The other half of the contract Inker and Clay share: the documents *are*
    the work, so entering the mode must leave them exactly as they were."""
    from warlock.studio.panes import landing

    ctx = FakeCtx()
    tab = _tab(ctx)
    landing.start_clay(ctx)
    state = clay_mode.ensure(ctx)
    assert [doc.uid for doc in state.docs] == [tab.uid]


# --- what "hidden" means to a selection -------------------------------------
#
# ``visible=False`` is documented to mean an object does not render, does not
# export and *cannot be picked*. The object branch of select-all took every
# object anyway, while the element branch three lines below it had always
# skipped the invisible ones -- an asymmetry nobody saw until merge arrived and
# Ctrl+A then Ctrl+J pulled unseen geometry into a shape that is on screen.


def _two_objects(*, second_visible: bool = True) -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    first = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    second = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), visible=second_visible)
    )
    return doc, first.uid, second.uid


def test_select_all_in_object_mode_skips_hidden_objects():
    doc, first, _hidden = _two_objects(second_visible=False)
    clay_mode._select_all(doc)
    assert doc.selection == {first}


def test_invert_in_object_mode_never_selects_a_hidden_object():
    doc, first, _hidden = _two_objects(second_visible=False)
    doc.select([first])
    clay_mode._invert(doc)
    assert doc.selection == set()


def test_select_all_still_takes_every_visible_object():
    doc, first, second = _two_objects()
    clay_mode._select_all(doc)
    assert doc.selection == {first, second}


# --- the mesh check (Clay14) -------------------------------------------------
#
# ``check_manifold`` builds a whole adjacency and has never been drawable per
# frame, which is the constraint that shapes the whole feature: the panel runs
# it from a button and holds the answer against the ``Mesh`` it measured. Both
# halves are tested here rather than through imgui -- the staleness rule is a
# comparison and the click is three document calls.


def _stray_vertex_box() -> Any:
    """A box plus a vertex no face uses: one finding, in vertex mode."""
    import numpy as np

    from warlock.studio.clay import mesh as bm

    box = bp.box()
    return bm.Mesh(
        positions=np.vstack([box.positions, np.array([[9.0, 9.0, 9.0]], dtype="f4")]),
        loops=box.loops,
        starts=box.starts,
        material=box.material,
        smooth=box.smooth,
    )


def test_a_stored_check_is_stale_the_moment_the_mesh_is_replaced() -> None:
    from warlock.studio.clay import diagnose

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=_stray_vertex_box()))
    state = clay_state.ClayState()
    state.manifold[obj.uid] = (obj.mesh, diagnose.findings(obj.mesh))

    measured, rows = state.manifold[obj.uid]
    assert measured is doc.by_uid(obj.uid).mesh, "fresh while nothing has edited it"
    assert [row.kind for row in rows] == ["unused"]

    doc.set_mesh(obj.uid, bp.box())
    measured, _ = state.manifold[obj.uid]
    assert measured is not doc.by_uid(obj.uid).mesh, "an op replaces the mesh, so identity says so"


def test_clicking_a_finding_selects_exactly_its_elements_in_its_own_mode() -> None:
    from warlock.studio.clay import diagnose
    from warlock.studio.panes import clay_props

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=_stray_vertex_box()))
    other = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=_stray_vertex_box()))
    doc.set_element_mode("face")
    doc.set_element_sel(other.uid, el.select_all(other.mesh, "face"))

    row = diagnose.findings(obj.mesh)[0]
    clay_props._select_finding(doc, obj, row)

    assert doc.element_mode == "vertex"
    # The other object's stale face selection is gone, and the object selection
    # is derived rather than set: ``set_element_sel`` is what puts the uid in it.
    assert set(doc.element_sel) == {obj.uid}
    assert doc.selection == {obj.uid}
    assert doc.element_sel_of(obj.uid).verts.tolist() == row.sel.verts.tolist()


def test_a_finding_click_pushes_no_undo_step() -> None:
    """Selection is not undoable in Clay, and a diagnostic click is selection.

    A step here would move ``history.head`` and make a document ask to be saved
    because the user looked at a hole.
    """
    from warlock.studio.clay import diagnose
    from warlock.studio.panes import clay_props

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=_stray_vertex_box()))
    head = doc.history.head
    clay_props._select_finding(doc, obj, diagnose.findings(obj.mesh)[0])
    assert doc.history.head == head


# --- axis views and the orthographic toggle (Clay17) -------------------------
#
# Bound inside Clay's own handle_key, never in App._shortcut: a global binding
# is checked above the workspace modes and takes its key from them permanently,
# which is exactly why the mode switch went to Alt in Phase 3.


def test_every_axis_view_key_names_a_view_the_camera_knows() -> None:
    from warlock.studio.viewer.camera import Camera

    for name in clay_mode.AXIS_VIEW_KEYS.values():
        assert name in Camera.AXIS_VIEWS


def test_the_axis_views_point_along_the_axes_they_name() -> None:
    import numpy as np

    from warlock.studio.viewer.camera import Camera

    camera = Camera()
    camera.set_target(np.zeros(3))
    camera.distance = 4.0

    for name, axis in (("front", 2), ("right", 0), ("top", 1)):
        assert camera.look_along(name)
        offset = camera.position - camera.target
        assert offset[axis] > 3.9, name
        assert float(np.abs(np.delete(offset, axis)).max()) < 0.01, name


def test_an_axis_view_keeps_the_target_and_the_distance() -> None:
    """A change of angle, not of framing: reframing would lose the part of the
    model the user was about to line up."""
    import numpy as np

    from warlock.studio.viewer.camera import Camera

    camera = Camera()
    camera.set_target(np.array([1.0, 2.0, 3.0]))
    camera.distance = 7.5
    camera.look_along("front")
    assert camera.distance == 7.5
    assert list(camera.target) == [1.0, 2.0, 3.0]


def test_an_unknown_view_name_changes_nothing() -> None:
    from warlock.studio.viewer.camera import Camera

    camera = Camera()
    before = (camera.theta, camera.phi)
    assert camera.look_along("isometric") is False
    assert (camera.theta, camera.phi) == before


def test_the_orthographic_projection_matches_the_perspective_one_at_the_target():
    """What makes the toggle a change of projection rather than a jump cut."""
    import numpy as np

    from warlock.studio.viewer.camera import Camera

    camera = Camera(aspect=1.6)
    camera.distance = 5.0
    # Front view first, so "the plane through the target" is a plane the test
    # can name: from an arbitrary orbit angle it is not z = 0 and the point
    # below would be in front of it or behind it rather than on it.
    camera.look_along("front")
    point = np.array([0.4, 0.3, 0.0, 1.0])

    camera.orthographic = False
    near = camera.projection() @ camera.view() @ point
    camera.orthographic = True
    far = camera.projection() @ camera.view() @ point

    assert float(np.abs(near[:2] / near[3] - far[:2] / far[3]).max()) < 1e-6


def test_the_camera_is_perspective_unless_something_asks_otherwise() -> None:
    """The asset viewer shows what an engine will show, and an engine uses a
    perspective camera."""
    from warlock.studio.viewer.camera import Camera

    assert Camera().orthographic is False


# --- closing a document ------------------------------------------------------


def test_a_clean_document_closes_without_a_question(svc) -> None:
    """``ClayState.close`` had no caller at all: Clay could open documents and
    never shut one, and Ctrl+Tab cycled between them with nothing on screen
    saying there was more than one."""
    ctx = FakeCtx(svc)
    first = _tab(ctx)
    second = _tab(ctx)
    state = clay_mode.ensure(ctx)

    clay_mode.close_tab(ctx, second.uid)

    assert ctx.confirms.pending is None
    assert [tab.uid for tab in state.docs] == [first.uid]
    assert state.active_uid == first.uid


def test_a_dirty_document_asks_first(svc) -> None:
    ctx = FakeCtx(svc)
    tab = _tab(ctx, dirty=True)
    state = clay_mode.ensure(ctx)

    clay_mode.close_tab(ctx, tab.uid)

    assert ctx.confirms.pending is not None
    assert state.docs == [tab]

    ctx.confirms.pending.on_confirm()
    assert state.docs == []


def test_a_document_being_saved_cannot_be_closed(svc) -> None:
    """The serialise task reads the live document on a task thread. Closing it
    out from under that read loses the file being written, not merely the edits
    since -- which is a stronger reason than ``_MUTATING_CTRL``'s."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.saving = True
    state = clay_mode.ensure(ctx)

    clay_mode.close_tab(ctx, tab.uid)

    assert state.docs == [tab]
    assert ctx.confirms.pending is None


def test_ctrl_w_closes_the_active_document(svc) -> None:
    ctx = FakeCtx(svc)
    _tab(ctx)
    second = _tab(ctx)
    state = clay_mode.ensure(ctx)
    assert state.active_uid == second.uid

    clay_mode._ctrl_key(ctx, state, second, second.doc, "w", shift=False)

    assert second.uid not in [tab.uid for tab in state.docs]

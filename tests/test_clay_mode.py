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
    ctx.state.clay.remember(bad)

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

    stored = ctx.settings.get("clay") or {}
    assert str(tmp_path / "scene.wblk") in (stored.get("recent") or [])


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

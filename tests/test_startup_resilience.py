"""One wrong byte in a persisted file must not be a permanent boot loop.

``App.run()`` wraps setup *and* the whole frame loop in one ``except``, so
anything that raises during startup ends the session -- and nothing on the
failure path rewrites the file that caused it, which is what turns a single
failure into every failure after it. ``studio/guard.py`` put a net under the
*panes*; these are the triggers upstream of it, where there is not yet a window
to draw a placeholder in.

Two groups. The first is untrusted JSON reaching code that assumed its own
shapes: settings, the filter bar, a journal sidecar. The second is startup
failures that were real, unguarded and (under ``pythonw``, whose stderr is the
null device) completely silent -- the user double-clicks the icon and nothing
happens, twice.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from warlock.studio import journal
from warlock.studio import settings as settings_mod
from warlock.studio.settings import Settings

STUDIO = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"


def _write(path: Path, data: Any, version: int = settings_mod.VERSION) -> None:
    path.write_text(json.dumps({"version": version, "data": data}), encoding="utf-8")


# --- settings: the file is untrusted -----------------------------------------


def test_a_migration_that_raises_is_a_reset_rather_than_a_boot_loop(tmp_path):
    """``_migrate`` ran *outside* ``load``'s ``try``.

    It reads stored values as though they were the shapes it wrote: the
    legacy-asset-type branch asks ``value in ASSET_TYPES``, which on a stored
    ``asset_type`` of ``{}`` is ``TypeError: unhashable type``. That is an
    exception out of ``Settings.load``, which runs before there is a window, a
    GL context or an excepthook that can draw anything -- so the app did not
    start, and did not start the next time either, because nothing on that path
    rewrites the file.
    """
    _write(tmp_path / settings_mod.FILENAME, {"form_2d": {"asset_type": {}}})
    loaded = Settings.load(tmp_path)
    assert loaded.data == {}
    notice = loaded.take_notice()
    assert notice is not None and "reset to defaults" in notice
    assert list(tmp_path.glob("*.corrupt-*.json")), "the original is kept, not destroyed"


def test_a_version_mismatch_is_reported_and_the_file_is_kept(tmp_path):
    """It used to discard every preference in silence: no notice, no rename,
    and the first successful save overwrote the file. That is the same loss the
    unreadable case is careful about -- theme, UI scale, pane layout and every
    remembered form field -- and the likeliest way to reach it is an older
    build opening a newer file, where the data is not damaged at all."""
    _write(tmp_path / settings_mod.FILENAME, {"theme": "light"}, version=99)
    loaded = Settings.load(tmp_path)
    assert loaded.data == {}
    notice = loaded.take_notice()
    assert notice is not None and "different version" in notice
    kept = list(tmp_path.glob("*.corrupt-*.json"))
    assert kept, "the only copy of those preferences is not thrown away"
    assert json.loads(kept[0].read_text(encoding="utf-8"))["data"] == {"theme": "light"}


def test_a_top_level_key_of_the_wrong_type_reads_as_its_default(tmp_path):
    """``settings.get(k) or {}`` reads as a type guard and is not one: a stored
    string is truthy and then raises ``AttributeError`` on the ``.get`` after
    it. Seven sites read that way, one of them inside ``widgets.section``,
    which most panes draw every frame."""
    _write(
        tmp_path / settings_mod.FILENAME,
        {"panels_open": "yes", "layout": ["left"], "history": "a knight"},
    )
    loaded = Settings.load(tmp_path)
    assert settings_mod.as_dict(loaded.get("panels_open")) == {}
    assert settings_mod.as_dict(loaded.get("layout")) == {}
    assert settings_mod.as_list(loaded.get("history")) == []


def test_a_stored_string_does_not_become_a_list_of_letters():
    """``or []`` on a string is the quieter half of the same bug: the string
    *iterates*, so the result is a list of single characters that every
    downstream ``str()`` waves through."""
    shape = "abc"
    assert (shape or []) is shape, "the shape being refused"
    assert settings_mod.as_list(shape) == []


def test_the_frame_drawing_section_reader_survives_a_wrong_type(monkeypatch):
    """``widgets.section`` is the sharpest of the seven, because it is drawn
    every frame in most panes: one wrong byte was a crash on the first frame of
    whichever mode read it, every launch."""
    from warlock.studio import widgets

    class _Bad:
        def get(self, key, default=None):
            return "collapsed"

    monkeypatch.setattr(widgets, "_SETTINGS", _Bad())
    assert settings_mod.as_dict(widgets._SETTINGS.get("panels_open")) == {}


def test_the_layout_reader_survives_a_wrong_type():
    from warlock.studio.layout import Layout

    class _Bad:
        def get(self, key, default=None):
            return "wide"

        def set(self, key, value):  # pragma: no cover - never reached here
            raise AssertionError("reading a layout writes nothing")

    assert Layout(_Bad()).settings_share > 0.0


def test_a_layout_profile_survives_wrongly_typed_blobs():
    """``Arrangement.from_json`` asked the same non-question four times over
    values *inside* a stored blob, which is why the guard is a free function
    rather than a ``Settings`` method."""
    from warlock.studio.layouts import Arrangement

    got = Arrangement.from_json(
        {"columns": "left", "hidden": "abc", "widths": 3, "shares": "half"}
    )
    assert got.columns == {} and got.hidden == [] and got.widths == {} and got.shares == {}


def _fake_guards(tree: ast.AST, where: str) -> list[str]:
    """``<something settings-ish>.get(...) or {}`` / ``or []``, with lines."""
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        last = node.values[-1]
        empty_dict = isinstance(last, ast.Dict) and not last.keys
        empty_list = isinstance(last, ast.List) and not last.elts
        if not (empty_dict or empty_list):
            continue
        first = node.values[0]
        if not isinstance(first, ast.Call) or not isinstance(first.func, ast.Attribute):
            continue
        if first.func.attr != "get":
            continue
        receiver = first.func.value
        name = (
            receiver.attr
            if isinstance(receiver, ast.Attribute)
            else getattr(receiver, "id", "")
        )
        if "settings" in name.lower() or name == "_SETTINGS":
            offenders.append(f"{where}:{node.lineno}: {ast.unparse(node)}")
    return offenders


def test_no_new_settings_get_or_default_creeps_back():
    """The scan, in the idiom of the twenty-odd other structural scans here.

    ``settings.get(k) or {}`` is the shape, and it is easy to write again
    because it *looks* like a guard. ``as_dict``/``as_list`` are the answer and
    they read almost the same, which is the point.
    """
    offenders: list[str] = []
    for path in sorted(STUDIO.rglob("*.py")):
        if path.name == "settings.py":
            continue  # it documents the shape it refuses
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(_fake_guards(tree, path.name))
    assert not offenders, (
        "a settings read that looks like a type guard and is not: "
        + "; ".join(offenders)
        + " -- use settings.as_dict/as_list"
    )


def test_the_scan_catches_both_spellings_and_leaves_the_answer_alone():
    """A scan that never fires is a scan nobody notices has stopped working."""
    caught = _fake_guards(
        ast.parse(
            "a = settings.get('panes') or {}\n"
            "b = ctx.settings.get('history') or []\n"
            "c = _SETTINGS.get('panels_open') or {}\n"
        ),
        "x.py",
    )
    assert len(caught) == 3
    clean = _fake_guards(
        ast.parse(
            "a = settings.as_dict(settings.get('panes'))\n"
            "b = as_list(ctx.settings.get('history'))\n"
            "c = job.get('params') or {}\n"  # not a settings read
            "d = settings.get('name') or 'dark'\n"  # not a container default
        ),
        "x.py",
    )
    assert clean == []


# --- the filter bar ----------------------------------------------------------


def test_a_filter_of_the_wrong_type_does_not_reach_the_ui():
    """``filters_from_stored`` filtered *keys* and not types, so
    ``{"text": 5}`` built a ``Filters`` whose ``text`` is an int -- which then
    reached ``parse_query`` and imgui's ``input_text``, where it is the frame
    loop's problem rather than a bad byte in a file."""
    from warlock.studio.state import Filters, filters_from_stored

    got = filters_from_stored(
        {"text": 5, "status": None, "favorites_only": "yes", "kind": "model"}
    )
    blank = Filters()
    assert got.text == blank.text and got.status == blank.status
    assert got.favorites_only is blank.favorites_only
    assert got.kind == "model", "a well-typed value still comes back"


def test_a_filter_bar_that_is_not_a_dict_is_the_default():
    from warlock.studio.state import Filters, filters_from_stored

    assert filters_from_stored("newest") == Filters()


# --- the journal sidecar -----------------------------------------------------


def test_a_malformed_timestamp_does_not_kill_the_first_frame_after_a_crash(tmp_path):
    """``at=float(meta.get("at") or 0.0)`` was unvalidated, and its caller is
    ``snapshot`` -- which runs on the **first frame after a crash**, precisely
    when recovery matters. ``status_line`` wraps its whole call for this
    reason; the startup offer had no wrapper and needed one less."""
    from types import SimpleNamespace

    payload = tmp_path / "sketch-x.ora"
    payload.write_bytes(b"body")
    journal.meta_path(payload).write_text(
        json.dumps(
            {"version": journal.VERSION, "kind": "inker", "title": "s", "at": "half four"}
        ),
        encoding="utf-8",
    )
    ctx = SimpleNamespace(svc=SimpleNamespace(config=SimpleNamespace(autosave_dir=tmp_path)))
    found = journal.recoverable(ctx)
    assert [r.title for r in found] == ["s"], "the copy is still offered"
    assert found[0].at == 0.0, "an unreadable stamp sorts last and says nothing"


# --- the job database --------------------------------------------------------


def test_a_malformed_database_is_a_named_refusal(tmp_path):
    """``sqlite3.connect``/``executescript``/``_migrate`` were unguarded, so a
    malformed image raised a bare ``DatabaseError`` out of ``Runtime._start``
    and arrived as the generic "ran into a problem while starting" box -- on
    every launch, with no way in and nothing naming the file."""
    from warlock.db import JobStore, StoreUnreadable

    path = tmp_path / "jobs.sqlite"
    path.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4096)
    with pytest.raises(StoreUnreadable) as caught:
        JobStore(path)
    assert caught.value.path == path


def test_setting_a_broken_database_aside_keeps_every_part_of_it(tmp_path):
    """A rename, never a delete: a database sqlite will not open is still a
    file ``.recover`` or a newer sqlite might. All three WAL parts move
    together, because a fresh database beside a stale ``-wal`` is a database
    whose first read has to decide whether that journal is its own."""
    from warlock import db

    path = tmp_path / "jobs.sqlite"
    path.write_bytes(b"not a database")
    path.with_name("jobs.sqlite-wal").write_bytes(b"stale")
    path.with_name("jobs.sqlite-shm").write_bytes(b"stale")

    moved = db.set_aside(path)
    assert moved is not None and moved.exists()
    assert moved.read_bytes() == b"not a database"
    assert not path.exists()
    assert not path.with_name("jobs.sqlite-wal").exists()
    assert not path.with_name("jobs.sqlite-shm").exists()
    # And the store opens cleanly in the space that leaves.
    store = db.JobStore(path)
    try:
        assert store.list() == []
    finally:
        store.close()


def test_a_broken_database_shows_up_in_the_doctor(tmp_path):
    """The row that lets a database which has *started* to go be found while
    the app is still up, when a backup is still possible."""
    from warlock import doctor
    from warlock.config import Config

    config = Config(data_dir=tmp_path / "assets", db_path=tmp_path / "jobs.sqlite")
    config.db_path.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4096)
    row = doctor._store_check(config)
    assert row.name == "job database" and not row.ok

    config.db_path.unlink()
    assert doctor._store_check(config).ok, "a first run has no database and no fault"

    sqlite3.connect(config.db_path).close()
    store = __import__("warlock.db", fromlist=["JobStore"]).JobStore(config.db_path)
    store.close()
    assert doctor._store_check(config).ok


# --- the silent startup deaths -----------------------------------------------


def test_a_home_directory_that_cannot_be_prepared_is_said_out_loud(monkeypatch, tmp_path):
    """``get_config`` is the first thing in the process that touches the disk
    and was the only unguarded one: it runs the migration and then ``mkdir``s
    four directories, so a disconnected share or a read-only drive raised
    ``OSError`` before the window, before GL, before imgui and -- under
    ``pythonw`` -- with stderr pointed at the null device."""
    from warlock import instance
    from warlock.studio import main

    said: list[tuple[str, str]] = []
    monkeypatch.setattr(instance, "alert", lambda title, body: said.append((title, body)))
    monkeypatch.setattr(instance, "ask", lambda title, body: False)
    monkeypatch.setattr(main, "_setup_logging", lambda: None)
    monkeypatch.setattr(main, "_install_excepthooks", lambda: None)
    import warlock.config as config_mod

    def boom() -> Any:
        raise OSError(13, "Access is denied")

    monkeypatch.setattr(config_mod, "get_config", boom)
    assert main.run() == 1
    assert said and "home directory" in said[0][0]


def test_a_failure_before_the_window_is_a_dialog_rather_than_an_exit_code(monkeypatch):
    """``_run_locked``'s ``except`` logged and returned 1 with no dialog, which
    under ``pythonw`` is a process that starts, writes to a devnull stderr and
    vanishes."""
    from warlock import instance
    from warlock.studio import main

    said: list[tuple[str, str]] = []
    monkeypatch.setattr(instance, "alert", lambda title, body: said.append((title, body)))
    monkeypatch.setattr(main, "_note_previous_session", lambda: None)
    monkeypatch.setattr(main, "_write_session_marker", lambda: None)
    monkeypatch.setattr(main, "_clear_session_marker", lambda: None)
    monkeypatch.setattr(main, "App", lambda runtime: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setattr("warlock.studio.runtime.Runtime", lambda: object())
    assert main._run_locked() == 1
    assert said and "could not start" in said[0][0]


def test_a_refusal_with_words_of_its_own_says_them(monkeypatch):
    """The generic box names the exception type, which distinguishes "the port
    is in use" from "the database is malformed" without opening the log -- but
    for the failures that have a *remedy*, the remedy is what should be on
    screen."""
    from warlock import instance
    from warlock.studio import main

    said: list[tuple[str, str]] = []
    monkeypatch.setattr(instance, "alert", lambda title, body: said.append((title, body)))
    monkeypatch.setattr(main, "_note_previous_session", lambda: None)
    monkeypatch.setattr(main, "_write_session_marker", lambda: None)
    monkeypatch.setattr(main, "_clear_session_marker", lambda: None)
    refusal = main.StartupRefused("Warlock Studio needs OpenGL 3.3", "update your driver")
    monkeypatch.setattr(main, "App", lambda runtime: (_ for _ in ()).throw(refusal))
    monkeypatch.setattr("warlock.studio.runtime.Runtime", lambda: object())
    assert main._run_locked() == 1
    assert said == [("Warlock Studio needs OpenGL 3.3", "update your driver")]


def test_a_missing_font_is_named_rather_than_asserted(monkeypatch, tmp_path):
    """``fonts.load`` had no existence check, so a quarantined TTF surfaced as
    an ``IM_ASSERT`` in imgui's own wording. It is also reachable *mid-session*
    -- the UI-scale slider re-bakes the atlas -- which is why the file check
    runs before ``clear_fonts`` rather than after it."""
    from warlock.studio import fonts

    monkeypatch.setattr(fonts, "FONT_DIR", tmp_path)
    with pytest.raises(fonts.FontsUnavailable) as caught:
        fonts.load(object())
    assert set(caught.value.missing) == set(fonts.FACES)

    cleared: list[int] = []

    class _Io:
        fonts = type("F", (), {"clear_fonts": lambda self: cleared.append(1)})()

    with pytest.raises(fonts.FontsUnavailable):
        fonts.reload(type("I", (), {"get_io": staticmethod(lambda: _Io())})())
    assert cleared == [], "a rebuild that cannot happen leaves the atlas it had"

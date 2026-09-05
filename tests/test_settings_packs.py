"""Settings -> Packs: the rows that install torch, bpy and the music stack.

The shipped installer stages the base app and ``studio`` alone, so everything
heavy arrives from this pane -- which makes it the only place a user can turn
Create, Poser, Troupe or Muse on. The half worth asserting is the half a
screenshot cannot check: what each row *says*, and when the Cancel button is
offered at all.

Drawing is left to ``test_studio_smoke``'s category test; every helper here is
pure, for ``test_settings_health``'s reason.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock import packs
from warlock.service import packs as svc_packs
from warlock.studio import main as main_mod
from warlock.studio.panes import app_settings


class FakeService:
    """Only what ``service.packs.rows`` touches."""

    def __init__(self, home: Path) -> None:
        self.config = SimpleNamespace(home=home)


@pytest.fixture
def rows(tmp_path, monkeypatch):
    """The real registry's rows, with no manifest -- a source checkout."""
    monkeypatch.setattr(svc_packs, "manifest_path", lambda: tmp_path / "absent.json")
    return {row["key"]: row for row in svc_packs.rows(FakeService(tmp_path))}


# --- what a row says ---------------------------------------------------------


def test_the_cost_is_both_volumes_because_they_are_two_drives():
    """The wheels land in the cache under the user's Warlock home and the
    packages land in the application runtime, which on a per-user install is
    routinely another disk -- which is why ``packs.disk_refusal`` budgets both
    and why a row that quoted one figure would be answering half the question.
    """
    note = app_settings.pack_size_note(
        {"manifest": True, "wheels": 34, "download_gib": 2.94, "installed_gib": 6.13}
    )
    assert note == "34 packages, 2.9 GB to download, 6.1 GB installed"


def test_an_unmeasured_install_size_is_left_out_rather_than_shown_as_zero():
    """``0`` in the manifest means the generator could not read a RECORD, not
    that the pack is free -- and "0.0 GB installed" is the one reading of it
    that is actually false."""
    note = app_settings.pack_size_note(
        {"manifest": True, "wheels": 7, "download_gib": 0.32, "installed_gib": 0.0}
    )
    assert note == "7 packages, 0.3 GB to download"


def test_an_installed_pack_quotes_no_figures(rows):
    """Nothing is about to be downloaded, so there is no cost to state."""
    row = dict(rows["rig"], present=True, manifest=True, wheels=7, download_gib=0.32)
    assert app_settings.pack_size_note(row) == ""
    assert app_settings.pack_blocked(row) == ""


def test_a_row_names_the_modes_it_turns_on(rows):
    """``packs.Pack.modes`` is deliberately mode *keys*, so that
    ``warlock.packs`` imports no ``studio``. The pane is the one place holding
    both tables, and the user reads rail labels rather than keys."""
    assert app_settings.pack_unlocks(rows["rig"]) == "Unlocks Poser and Troupe"
    assert app_settings.pack_unlocks(rows["music"]) == "Unlocks Muse"


def test_a_source_checkout_offers_the_command_that_actually_works(rows):
    """A build with no manifest is a supported state, not a fault: the extras
    are installed with uv there. The row still says what the pack is -- and the
    remedy it prints is ``Pack.install_hint``, the one composed spelling."""
    # ``present`` is forced: the dev environment installs every extra, and the
    # case under test is the machine that has not.
    said = app_settings.pack_blocked(dict(rows["text2image"], present=False))
    assert "uv sync --extra text2image" in said
    assert packs.find("text2image").install_hint in said


def test_a_build_with_a_manifest_blocks_nothing(rows):
    assert app_settings.pack_blocked(dict(rows["rig"], present=False, manifest=True)) == ""


# --- when stopping is safe ---------------------------------------------------


def test_cancel_is_offered_while_downloading_and_withdrawn_once_pip_starts():
    """Killing the child mid-download costs a resumable ``.part`` and nothing
    else. Killing it mid-install leaves the application's own site-packages
    half written, which is the state the child exists to prevent -- so the
    button is withdrawn rather than left to mean two different things.

    Read off ``pack_worker``'s own phase word (H02), not a percent threshold:
    a threshold is a guess about when ``collect()`` ends and ``install()``
    begins, sampled once a frame, and the gap between an under-threshold
    sample and the child actually calling pip is where a quit used to slip
    through with no warning at all.
    """
    from warlock.pipelines import pack_worker

    assert app_settings.pack_cancellable("")
    assert app_settings.pack_cancellable(pack_worker.PHASE_DOWNLOAD)
    assert not app_settings.pack_cancellable(pack_worker.PHASE_COMMIT)


# --- what a finished install means -------------------------------------------


class _Toasts(list):
    def __call__(self, message, kind="info", action=None, **_kw):
        self.append((message, kind))


def _app():
    """An ``App`` with only what the ``pack:`` branch touches."""
    app = main_mod.App.__new__(main_mod.App)
    app._unclaimed = set()
    app.svc = object()
    submitted: list[str] = []
    app.app_ctx = SimpleNamespace(
        toast=_Toasts(),
        state=SimpleNamespace(preview={}),
        submit=lambda key, *a, **kw: submitted.append(key) or True,
        tasks=SimpleNamespace(set_progress=lambda *_a: None),
    )
    return app, submitted


def _done(key):
    return SimpleNamespace(key=key, result=None, ok=True, message="", action=None)


def test_a_finished_pack_install_re_probes_the_whole_installation(monkeypatch):
    """The child wrote into the site-packages this process runs out of, and
    every model, rigging and mode answer in the ctx was derived from a probe
    taken before it did -- the same wholesale re-probe a finished download
    gets, and for a stronger reason."""
    monkeypatch.setattr(svc_packs, "unresolved", lambda _keys: [])
    app, submitted = _app()
    app._on_task_done(_done("pack:rig"))
    assert submitted == [main_mod.VERIFY_KEY]
    assert app.app_ctx.toast == [("The rig pack is installed.", "success")]


def test_a_pack_that_still_will_not_import_asks_for_a_restart(monkeypatch):
    """``service.packs`` invalidates the import caches, so this is rare -- and
    when it happens the alternative is a mode that stays grey after an install
    that reported success, which is the failure the whole exercise exists to
    avoid."""
    monkeypatch.setattr(svc_packs, "unresolved", lambda _keys: ["bpy"])
    app, _submitted = _app()
    app._on_task_done(_done("pack:rig"))
    message, kind = app.app_ctx.toast[0]
    assert "restart" in message and kind == "warn"


def test_a_re_probe_that_raises_is_not_a_silent_install(monkeypatch):
    """A ``find_spec`` can raise on a package with broken metadata, and this
    runs on the frame thread: the toast still has to arrive."""

    def _boom(_keys):
        raise RuntimeError("dist-info is a directory of lies")

    monkeypatch.setattr(svc_packs, "unresolved", _boom)
    app, _submitted = _app()
    app._on_task_done(_done("pack:music"))
    assert app.app_ctx.toast == [("The music pack is installed.", "success")]


def test_restore_packs_reprobes_every_key_the_comma_joined_task_named(monkeypatch):
    """``app_settings._restore_packs`` (M02) installs several packs under one
    task key so the pane's one-install-at-a-time rule still applies; the
    ``pack:`` branch has to split that key back apart rather than handing the
    literal string "rig,music" to ``unresolved`` as if it were one pack."""
    seen: list[list[str]] = []
    monkeypatch.setattr(svc_packs, "unresolved", lambda keys: seen.append(list(keys)) or [])
    app, _submitted = _app()
    app._on_task_done(_done("pack:rig,music"))
    assert seen == [["rig", "music"]]
    assert app.app_ctx.toast == [("The rig, music pack is installed.", "success")]


# --- H02: a running install's commit phase cannot be quit through -----------


def test_a_quit_is_withheld_not_confirmed_during_a_packs_commit_phase():
    """Reproduces H02: before the fix, ``_quit_summary`` named no ``pack:``
    prefix at all, so a quit asked during a running install's commit phase
    produced an empty summary and went straight to ``_request_quit`` -- no
    dialog, no warning, nothing standing between the click and
    ``TaskRunner.shutdown`` eventually killing the child mid-write. The fixed
    guard must withhold the ask itself rather than show a confirm dialog that
    quitting would then honour by killing pip."""
    app, _ = _app()
    app.app_ctx.tasks = SimpleNamespace(
        commit_busy=lambda prefix: prefix == "pack:",
        set_progress=lambda *_a: None,
    )
    app.app_ctx.confirms = SimpleNamespace(
        ask=lambda *_a, **_k: pytest.fail("a confirm dialog must not be raised mid-commit")
    )
    app._running = True
    app._ask_quit()
    assert app._quit_deferred is True
    assert app._running is True  # not quit outright either
    message, kind = app.app_ctx.toast[-1]
    assert "cannot be interrupted" in message and kind == "warn"


def test_a_deferred_quit_resumes_once_the_commit_phase_clears(monkeypatch):
    """The other half: once nothing is left in the commit phase, the withheld
    ask has to actually happen rather than being forgotten (H02's "defer
    shutdown until the commit phase completes")."""
    monkeypatch.setattr(svc_packs, "unresolved", lambda _keys: [])
    app, _ = _app()
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx.cache = SimpleNamespace(active=None)
    app.app_ctx.tasks = SimpleNamespace(
        commit_busy=lambda _prefix: False,
        busy_keys=set(),
        set_progress=lambda *_a: None,
    )
    app.app_ctx.confirms = SimpleNamespace(ask=lambda *_a, **_k: None)
    app._quit_deferred = True
    requested: list[bool] = []
    app._request_quit = lambda: requested.append(True)
    app._on_task_done(_done("pack:rig"))
    assert app._quit_deferred is False


# --- muse-05: per-mode export prefixes never matched the export warning -----


def test_quit_summary_warns_while_a_muse_export_task_is_busy():
    """Reproduces muse-05: ``_quit_summary``'s guard tested
    ``k.startswith(("export", "save:", "bake:"))``, but every per-mode export
    queues under ``"<mode>-export:<name>"`` (``muse-export:``, ``sirens-export:``,
    ``clay-export:``, ``inker-export:``, ``packwright-export:``,
    ``plotter-export:``) -- none of which *start with* "export", so only the
    Library's bulk ``export-folder``/``export-zip`` keys ever matched. A user
    who quit mid "Export the loop" got no "An export is still being written"
    line and the app closed under the write with no notice."""
    app, _ = _app()
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx.cache = SimpleNamespace(active=None)
    for key in (
        "muse-export:loop",
        "sirens-export:abc123",
        "clay-export:abc123",
        "inker-export:abc123",
        "packwright-export:abc123",
        "plotter-export:abc123",
    ):
        app.app_ctx.tasks = SimpleNamespace(busy_keys={key})
        assert app._quit_summary() == "An export is still being written.", key

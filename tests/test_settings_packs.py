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

    The two numbers are ``pack_worker``'s: it caps the download bar at 89 and
    emits 92 as pip begins.
    """
    assert app_settings.pack_cancellable(0.0)
    assert app_settings.pack_cancellable(89.0)
    assert not app_settings.pack_cancellable(92.0)
    assert not app_settings.pack_cancellable(100.0)


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

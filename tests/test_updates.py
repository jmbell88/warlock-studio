"""The update check's parent half: the comparison, the spawn, and the staging.

Nothing here reaches the network. The child is a stub put in place of
``worker_argv``, which is what that function exists for -- ``test_service_packs``
makes the same trade for the same reason, and the half worth testing here is
the parent's: what it hands over, what it makes of what comes back, and what it
will and will not call "ready to run".
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

from warlock.service import updates as svc_updates
from warlock.service.errors import Invalid


class FakeService:
    """Only what this module touches. ``WarlockService`` needs a database."""

    def __init__(self, home: Path) -> None:
        self.config = type("C", (), {"home": home})()


@pytest.fixture
def svc(tmp_path):
    return FakeService(tmp_path / "home")


def _stub(monkeypatch, body: str) -> None:
    monkeypatch.setattr(
        svc_updates, "worker_argv", lambda: [sys.executable, "-c", textwrap.dedent(body)]
    )


# --- which version is newer ----------------------------------------------------


def test_a_higher_version_is_newer():
    assert svc_updates._is_newer("0.0.37", "0.0.36") is True
    assert svc_updates._is_newer("0.1.0", "0.0.99") is True


def test_a_tag_may_carry_its_v():
    """``tag_name`` is typed by a human at release time, and every tag this
    project has ever pushed carries one."""
    assert svc_updates._is_newer("v0.0.37", "0.0.36") is True


def test_the_same_version_is_not_newer():
    assert svc_updates._is_newer("0.0.36", "0.0.36") is False


def test_an_older_published_version_is_not_offered():
    """The case that matters on a developer's machine, and on any build made
    between releases: offering "0.0.36" to a copy running 0.0.37 would be
    offering a downgrade with an upgrade's wording."""
    assert svc_updates._is_newer("0.0.36", "0.0.37") is False


def test_nothing_published_is_not_newer():
    assert svc_updates._is_newer(None, "0.0.36") is False
    assert svc_updates._is_newer("", "0.0.36") is False


# --- the child, stubbed --------------------------------------------------------


def test_a_check_carries_the_installed_version_and_the_comparison(svc, monkeypatch):
    _stub(
        monkeypatch,
        """
        import json, sys
        spec = json.loads(sys.stdin.read())
        open(spec["result_path"], "w").write(
            json.dumps({"ok": True, "latest": "99.0.0", "installer_name": "x.exe"})
        )
        """,
    )
    out = svc_updates.check(svc)
    assert out["latest"] == "99.0.0"
    assert out["available"] is True
    assert out["current"]  # whatever this checkout is running


def test_a_release_with_no_manifest_reads_as_up_to_date(svc, monkeypatch):
    _stub(
        monkeypatch,
        """
        import json, sys
        spec = json.loads(sys.stdin.read())
        open(spec["result_path"], "w").write(json.dumps({"ok": True, "latest": None}))
        """,
    )
    assert svc_updates.check(svc)["available"] is False


def test_the_download_spec_carries_what_the_child_needs_and_progress_comes_back(
    svc, monkeypatch, tmp_path
):
    seen = tmp_path / "seen.json"
    _stub(
        monkeypatch,
        f"""
        import json, sys
        spec = json.loads(sys.stdin.read())
        open({str(seen)!r}, "w").write(json.dumps(spec))
        print(json.dumps({{"percent": 40.0, "label": "40 of 100 MB"}}), flush=True)
        open(spec["result_path"], "w").write(json.dumps({{"ok": True, "path": "C:/x.exe"}}))
        """,
    )
    told: list[tuple[float, str]] = []
    info = {
        "installer_url": "https://example.invalid/x.exe",
        "installer_name": "x.exe",
        "size_bytes": 100,
        "sha256": "a" * 64,
    }
    out = svc_updates.download(svc, info, on_progress=lambda p, label: told.append((p, label)))
    assert out["path"] == "C:/x.exe"
    assert told == [(40.0, "40 of 100 MB")]

    spec = json.loads(seen.read_text(encoding="utf-8"))
    assert spec["mode"] == "download"
    assert spec["installer_url"] == info["installer_url"]
    assert spec["sha256"] == info["sha256"]
    # Into the user's home, beside the wheel cache, so it survives the app.
    assert spec["dest_dir"] == str(svc_updates.staging_dir(svc))


def test_a_download_with_nothing_to_download_refuses_before_spawning(svc, monkeypatch):
    """The pane can only reach this by holding a stale check across a version
    it no longer has the digest for, and a refusal names that; a spawn would
    produce a traceback out of the child instead."""

    def never() -> list[str]:
        raise AssertionError("the child must not be spawned")

    monkeypatch.setattr(svc_updates, "worker_argv", never)
    with pytest.raises(Invalid):
        svc_updates.download(svc, {"installer_name": "x.exe"})


def test_a_failing_child_is_a_refusal_carrying_its_own_words(svc, monkeypatch):
    _stub(
        monkeypatch,
        """
        import json, sys
        spec = json.loads(sys.stdin.read())
        open(spec["result_path"], "w").write(
            json.dumps({"ok": False, "error": "the release feed did not answer"})
        )
        """,
    )
    with pytest.raises(Invalid) as caught:
        svc_updates.check(svc)
    assert "the release feed did not answer" in str(caught.value)


# --- what counts as ready ------------------------------------------------------


def _stage(svc, name: str, payload: bytes) -> Path:
    where = svc_updates.staging_dir(svc)
    where.mkdir(parents=True, exist_ok=True)
    path = where / name
    path.write_bytes(payload)
    return path


def test_a_staged_installer_whose_digest_matches_is_ready(svc):
    payload = b"MZ" + b"y" * 512
    path = _stage(svc, "x.exe", payload)
    info = {"installer_name": "x.exe", "sha256": hashlib.sha256(payload).hexdigest()}
    assert svc_updates.staged_installer(svc, info) == path


def test_a_staged_file_with_the_right_name_and_the_wrong_bytes_is_not_ready(svc):
    """The one that earns the digest: an installer left by a *different*
    release, or a download a full disk truncated, both sit under the name the
    pane would otherwise offer as "Run Installer"."""
    _stage(svc, "x.exe", b"MZ" + b"y" * 512)
    info = {"installer_name": "x.exe", "sha256": "c" * 64}
    assert svc_updates.staged_installer(svc, info) is None


def test_nothing_staged_is_not_ready(svc):
    info = {"installer_name": "x.exe", "sha256": "c" * 64}
    assert svc_updates.staged_installer(svc, info) is None


def test_an_answer_with_no_digest_is_never_ready(svc):
    _stage(svc, "x.exe", b"MZ")
    assert svc_updates.staged_installer(svc, {"installer_name": "x.exe"}) is None

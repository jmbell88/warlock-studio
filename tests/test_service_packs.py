"""Installing a pack, headlessly: the refusals, the spawn, and the protocol.

Nothing in this file downloads or installs anything. The child is a stub put in
place of ``worker_argv``, which is what that function exists for -- the half
worth testing here is the parent's: what it refuses before spawning at all,
what it hands over, and what it makes of what comes back.

The refusals are the point. A model download writes into a directory the app
only reads from; a pack install writes into the ``site-packages`` the app is
running out of, so "this must not start" is a much sharper claim and every one
of its reasons is pinned below.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from warlock import fetch, packs
from warlock.service import packs as svc_packs
from warlock.service.errors import Invalid, NotFound

GIB = 1024**3


def wheel(name: str, size: int, *, sha: str = "a" * 64, installed: int = 0, **kw) -> dict:
    out = {
        "filename": name,
        "url": f"https://example.invalid/{name}",
        "size_bytes": size,
        "sha256": sha,
        "installed_bytes": installed,
        "packs": ["rig"],
    }
    out.update(kw)
    return out


def manifest_file(tmp_path: Path, *wheels: dict) -> Path:
    path = tmp_path / packs.MANIFEST_NAME
    path.write_text(
        json.dumps({"version": packs.MANIFEST_VERSION, "wheels": list(wheels)}),
        encoding="utf-8",
    )
    return path


class FakeService:
    """Only what this module touches. ``WarlockService`` needs a database."""

    def __init__(self, home: Path) -> None:
        self.config = type("C", (), {"home": home})()


@pytest.fixture
def svc(tmp_path):
    return FakeService(tmp_path / "home")


@pytest.fixture
def manifest_at(monkeypatch, tmp_path):
    def use(*wheels: dict) -> Path:
        path = manifest_file(tmp_path, *wheels)
        monkeypatch.setattr(svc_packs, "manifest_path", lambda: path)
        return path

    return use


# --- a build that has no packs -------------------------------------------------


def test_a_build_with_no_manifest_refuses_with_the_remedy(monkeypatch, tmp_path):
    """A source checkout that has never run the generator is a *supported*
    state, so this is a refusal that says what to do instead rather than an
    error the pane has to translate."""
    monkeypatch.setattr(svc_packs, "manifest_path", lambda: tmp_path / "nope.json")
    with pytest.raises(NotFound) as caught:
        svc_packs.load()
    assert "uv" in str(caught.value)


def test_the_rows_still_describe_every_pack_without_a_manifest(monkeypatch, tmp_path, svc):
    """What is unavailable without one is *installing*. Settings still has to
    say which packs are present and what each unlocks."""
    monkeypatch.setattr(svc_packs, "manifest_path", lambda: tmp_path / "nope.json")
    rows = svc_packs.rows(svc)
    assert [row["key"] for row in rows] == list(packs.KEYS)
    assert all(row["manifest"] is False for row in rows)
    assert all(row["summary"] and row["modes"] for row in rows)


def test_a_malformed_manifest_is_a_refusal_naming_the_fault(monkeypatch, tmp_path):
    bad = tmp_path / packs.MANIFEST_NAME
    bad.write_text('{"version": 99, "wheels": []}', encoding="utf-8")
    monkeypatch.setattr(svc_packs, "manifest_path", lambda: bad)
    with pytest.raises(Invalid):
        svc_packs.load()


# --- what must never start ------------------------------------------------------


def test_a_pack_that_would_re_version_the_runtime_is_refused(manifest_at, svc, monkeypatch):
    """**A pack is a delta over the base runtime.** Pack and base come out of
    one lock, so every wheel is either absent or already at the pack's own
    version. Anything else means the two were built from different locks, and
    installing it would re-version a package the running application has
    already imported -- numpy under the app's feet, in the worst case.
    """
    manifest_at(wheel("numpy-2.5.1-cp313-cp313-win_amd64.whl", 10))
    monkeypatch.setattr(svc_packs, "installed_versions", lambda: {"numpy": "1.26.0"})
    said = svc_packs.refusal(svc, ["rig"])
    assert said is not None
    assert "numpy 1.26.0 would be replaced by 2.5.1" in said
    with pytest.raises(Invalid):
        svc_packs.install(svc, ["rig"])


def test_the_conflict_is_reported_before_the_disk(manifest_at, svc, monkeypatch):
    """A plan that would re-version the runtime is wrong however much room
    there is for it, and reporting the smaller problem would send the user to
    free up space for an install that must not run at all."""
    manifest_at(wheel("numpy-2.5.1-cp313-cp313-win_amd64.whl", 500 * GIB))
    monkeypatch.setattr(svc_packs, "installed_versions", lambda: {"numpy": "1.26.0"})
    monkeypatch.setattr(fetch, "free_gib", lambda _p: 0.0)
    assert "would be replaced" in (svc_packs.refusal(svc, ["rig"]) or "")


def test_a_plan_that_does_not_fit_is_refused(manifest_at, svc, monkeypatch):
    manifest_at(wheel("bpy-5.2.0-cp313-cp313-win_amd64.whl", 40 * GIB, installed=80 * GIB))
    monkeypatch.setattr(svc_packs, "installed_versions", dict)
    monkeypatch.setattr(fetch, "free_gib", lambda _p: 1.0)
    said = svc_packs.refusal(svc, ["rig"])
    assert said is not None and "Not enough disk space" in said


def test_an_already_installed_pack_is_not_a_spawn(manifest_at, svc, monkeypatch):
    """Everything the pack carries is already at the pack's own version. Not an
    error and not a child: running an install twice must be cheap."""
    manifest_at(wheel("bpy-5.2.0-cp313-cp313-win_amd64.whl", 10))
    monkeypatch.setattr(svc_packs, "installed_versions", lambda: {"bpy": "5.2.0"})

    def _never() -> list[str]:
        raise AssertionError("a pack with nothing to do must not spawn a child")

    monkeypatch.setattr(svc_packs, "worker_argv", _never)
    assert svc_packs.install(svc, ["rig"])["already"] is True


def test_only_what_is_missing_is_planned(manifest_at, svc, monkeypatch):
    """``music`` after ``text2image`` is the case: the wheels they share are
    already at the pack's version, so the second install is the difference."""
    manifest_at(
        wheel("bpy-5.2.0-cp313-cp313-win_amd64.whl", 10),
        wheel("attrs-26.1.0-py3-none-any.whl", 20),
    )
    have = {"bpy": "5.2.0"}
    plan = svc_packs.plan_for(["rig"])
    assert [w.filename for w in packs.to_install(plan, have)] == [
        "attrs-26.1.0-py3-none-any.whl"
    ]


# --- the child, stubbed ---------------------------------------------------------


def _stub(monkeypatch, body: str) -> None:
    """Put a one-file Python program in the child's place.

    It reads the spec on stdin exactly as the worker does, so what is being
    exercised is the real hand-over: the spec's shape, the progress protocol
    and the result file.
    """
    monkeypatch.setattr(
        svc_packs, "worker_argv", lambda: [sys.executable, "-c", textwrap.dedent(body)]
    )


def test_the_spec_carries_what_the_child_needs_and_the_result_comes_back(
    manifest_at, svc, monkeypatch, tmp_path
):
    seen = tmp_path / "seen.json"
    manifest_at(wheel("bpy-5.2.0-cp313-cp313-win_amd64.whl", 10, installed=30))
    monkeypatch.setattr(svc_packs, "installed_versions", dict)
    _stub(
        monkeypatch,
        f"""
        import json, sys
        spec = json.loads(sys.stdin.read())
        open({str(seen)!r}, "w").write(json.dumps(spec))
        print(json.dumps({{"percent": 50.0, "label": "half"}}), flush=True)
        open(spec["result_path"], "w").write(
            json.dumps({{"ok": True, "collected": ["bpy"], "installed": ["bpy"]}})
        )
        """,
    )
    seen_progress: list[tuple[float, str]] = []
    result = svc_packs.install(
        svc, ["rig"], on_progress=lambda p, label: seen_progress.append((p, label))
    )
    assert result["installed"] == ["bpy"]
    assert seen_progress == [(50.0, "half")]

    spec = json.loads(seen.read_text(encoding="utf-8"))
    assert [w["filename"] for w in spec["wheels"]] == [
        "bpy-5.2.0-cp313-cp313-win_amd64.whl"
    ]
    assert spec["wheels"][0]["url"].startswith("https://")
    assert spec["probe"] == ["bpy"]  # the pack's own imports, for the child to verify
    assert spec["pack_dir"].endswith("packs")


def test_a_child_that_fails_is_reported_in_its_own_words(manifest_at, svc, monkeypatch):
    manifest_at(wheel("bpy-5.2.0-cp313-cp313-win_amd64.whl", 10))
    monkeypatch.setattr(svc_packs, "installed_versions", dict)
    _stub(
        monkeypatch,
        """
        import json, sys
        spec = json.loads(sys.stdin.read())
        open(spec["result_path"], "w").write(
            json.dumps({"ok": False, "error": "ValueError: digest did not match"})
        )
        """,
    )
    with pytest.raises(Invalid) as caught:
        svc_packs.install(svc, ["rig"])
    assert "digest did not match" in str(caught.value)


def test_a_child_that_dies_before_writing_a_result_reports_its_stderr(
    manifest_at, svc, monkeypatch
):
    """No result file means the child died before it could write one, so its
    stderr is the only thing that knows why -- ``downloads``' SVC-05 lesson."""
    manifest_at(wheel("bpy-5.2.0-cp313-cp313-win_amd64.whl", 10))
    monkeypatch.setattr(svc_packs, "installed_versions", dict)
    _stub(
        monkeypatch,
        """
        import sys
        sys.stderr.write("ModuleNotFoundError: no pip in this runtime\\n")
        raise SystemExit(3)
        """,
    )
    with pytest.raises(Invalid) as caught:
        svc_packs.install(svc, ["rig"])
    assert "no pip in this runtime" in str(caught.value)


def test_an_install_that_overruns_its_deadline_is_killed(manifest_at, svc, monkeypatch):
    manifest_at(wheel("bpy-5.2.0-cp313-cp313-win_amd64.whl", 10))
    monkeypatch.setattr(svc_packs, "installed_versions", dict)
    _stub(
        monkeypatch,
        """
        import time
        time.sleep(60)
        """,
    )
    with pytest.raises(Invalid) as caught:
        svc_packs.install(svc, ["rig"], timeout=1.0)
    assert "timed out" in str(caught.value)

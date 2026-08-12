"""The two halves of a fetch that only fail when something goes wrong.

Both are about a promise that is invisible while everything works: a child that
stalls has to be killed rather than waited on forever, and a move that fails
partway has to leave the destination as it was rather than half-populated --
which is the state every presence probe in ``warlock.fetch`` reads as a finished
download forever.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time

import pytest

from warlock.service import downloads
from warlock.service.errors import Invalid

# ``fetch_worker`` is imported *inside* a fixture, never at module scope, and
# the flag is put back afterwards. Importing it sets ``HF_HUB_OFFLINE=0`` at
# module level -- which is the whole point of the module, and correct in the
# child process it is spawned as, but in the pytest process it is the one thing
# the offline invariant forbids: huggingface_hub reads that variable at import
# time, so a test run that imported it here would leave every later test (and
# any real hub call one of them made) online-capable. It leaked exactly that
# way, and ``test_fetch.py``'s own guard caught it two files later.


@pytest.fixture
def fetch_worker():
    import os as _os

    before = _os.environ.get("HF_HUB_OFFLINE")
    from warlock.pipelines import fetch_worker as mod

    if before is None:
        _os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        _os.environ["HF_HUB_OFFLINE"] = before
    return mod


class _Job:
    """The two attributes ``_run_worker`` touches on a ``fetch.Job``."""

    repo_id = "acme/stalled"

    def spec(self):
        return {"repo_id": self.repo_id, "dest": "unused"}


def _chatty_child(tmp_path):
    """A stand-in worker that talks forever and never exits.

    The real one does exactly this: ``_Sampler`` emits a progress line every
    half second whether or not bytes are arriving, so a child parked on a
    stalled socket keeps stdout open indefinitely. Draining to EOF before
    ``wait(timeout=)`` therefore never consulted the timeout at all.
    """
    script = tmp_path / "chatty.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys, time
            sys.stdin.read()
            while True:
                sys.stdout.write(json.dumps({"percent": 1.0, "label": "x"}) + "\\n")
                sys.stdout.flush()
                time.sleep(0.02)
            """
        ),
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_a_child_that_never_stops_talking_still_hits_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "worker_argv", lambda: _chatty_child(tmp_path))
    seen: list[float] = []

    started = time.monotonic()
    with pytest.raises(Invalid) as exc:
        downloads._run_worker(
            _Job(),
            on_progress=lambda percent, label: seen.append(percent),
            timeout=1.0,
        )
    elapsed = time.monotonic() - started

    assert "timed out" in str(exc.value)
    # The deadline, not "eventually": the point of the drain thread is that the
    # whole run is bounded, and the old code would have hung here forever.
    assert elapsed < 20.0
    # And progress was still reported while it ran -- the fix must not turn the
    # bar off to get the deadline.
    assert seen


def test_a_worker_that_exits_is_still_read_to_the_end(tmp_path, monkeypatch):
    """The ordinary path, which the deadline must leave alone."""
    script = tmp_path / "quick.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            spec = json.loads(sys.stdin.read())
            sys.stdout.write(json.dumps({"percent": 50.0, "label": "half"}) + "\\n")
            with open(spec["result_path"], "w") as fh:
                fh.write(json.dumps({"ok": True}))
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(downloads, "worker_argv", lambda: [sys.executable, str(script)])
    seen: list[tuple[float, str]] = []

    result = downloads._run_worker(
        _Job(), on_progress=lambda p, label: seen.append((p, label)), timeout=60.0
    )

    assert result == {"ok": True}
    assert (50.0, "half") in seen


# --- the move ---------------------------------------------------------------


def _staged(tmp_path, names):
    staging = tmp_path / ".dest.fetch.part"
    for name in names:
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    return staging


def test_a_move_that_fails_partway_leaves_the_destination_as_it_was(
    tmp_path, monkeypatch, fetch_worker
):
    """A half-populated shared directory reads as a finished download forever."""
    staging = _staged(tmp_path, ["a.safetensors", "b.safetensors", "c.safetensors"])
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "existing.txt").write_text("mine", encoding="utf-8")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, target):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        return real_replace(src, target)

    monkeypatch.setattr(fetch_worker.os, "replace", flaky)
    with pytest.raises(OSError):
        fetch_worker._move_into(staging, dest)

    # Nothing of the download landed, the pre-existing file is untouched, and
    # the whole staged tree is back where the caller's rmtree will find it.
    assert sorted(p.name for p in dest.iterdir()) == ["existing.txt"]
    assert (dest / "existing.txt").read_text(encoding="utf-8") == "mine"
    assert sorted(p.name for p in staging.iterdir()) == [
        "a.safetensors",
        "b.safetensors",
        "c.safetensors",
    ]


def test_a_failed_re_fetch_gives_back_what_it_overwrote(tmp_path, monkeypatch, fetch_worker):
    """``loras/`` is shared: a second fetch writes names an earlier one put there."""
    staging = _staged(tmp_path, ["shared.safetensors", "new.safetensors"])
    dest = tmp_path / "loras"
    dest.mkdir()
    (dest / "shared.safetensors").write_text("the old one", encoding="utf-8")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, target):
        calls["n"] += 1
        # Staged names are walked sorted, so: 1 moves new.safetensors in, 2
        # backs the old shared.safetensors out, 3 would overwrite it.
        if calls["n"] == 3:
            raise PermissionError("the file is open in another process")
        return real_replace(src, target)

    monkeypatch.setattr(fetch_worker.os, "replace", flaky)
    with pytest.raises(PermissionError):
        fetch_worker._move_into(staging, dest)

    assert (dest / "shared.safetensors").read_text(encoding="utf-8") == "the old one"
    assert not (dest / "new.safetensors").exists()
    assert not (tmp_path / ".loras.fetch.bak").exists()


def test_a_move_that_succeeds_leaves_no_backup_tree(tmp_path, fetch_worker):
    staging = _staged(tmp_path, ["a.bin", "sub/b.bin"])
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "a.bin").write_text("old", encoding="utf-8")

    moved = fetch_worker._move_into(staging, dest)

    assert sorted(moved) == ["a.bin", "sub/b.bin"]
    assert (dest / "a.bin").read_text(encoding="utf-8") == "a.bin"
    assert (dest / "sub" / "b.bin").exists()
    assert not (tmp_path / ".dest.fetch.bak").exists()


def test_a_failure_partway_names_what_already_landed(svc, monkeypatch):
    """MDL-10: repositories publish one at a time, so a failure leaves the
    earlier ones *installed*.

    Making that not so means staging the whole selection and publishing through
    a journal, which is a larger piece of work. What is wrong today is smaller
    and worse: the refusal said nothing about the gigabytes that had already
    arrived, so a user told "the download failed" had no way to know a 7 GiB
    checkpoint was on disk and only its adapter was missing.
    """
    from warlock.service import downloads
    from warlock.service.errors import Failed, Invalid

    seen: list[str] = []

    def _run(job, **_kw):
        seen.append(job.repo_id)
        if len(seen) > 1:
            raise Failed("The download stopped.")
        return {"ok": True}

    monkeypatch.setattr(downloads, "_run_worker", _run)
    with pytest.raises(Invalid) as caught:
        downloads.download(svc, ["base:sdxl"])

    message = caught.value.message
    assert "Already installed before this failed" in message
    assert seen[0] in message, "the repo that landed has to be named"
    assert "what downloaded is kept" in message


def test_staging_left_by_a_killed_fetch_is_swept_before_the_next_one(svc, monkeypatch):
    """Unwinding inside Python rolls a publish back; being *killed* mid-publish
    cannot. The names are deterministic, so the next download finds them --
    otherwise the disk quietly holds a second copy of a checkpoint that no
    presence probe will ever look at."""
    from warlock.service import downloads

    root = svc.config.t2i_model_root
    root.mkdir(parents=True, exist_ok=True)
    stranded = root / ".sdxl-base-1.0.fetch.part"
    stranded.mkdir()
    (stranded / "half.safetensors").write_bytes(b"x")
    orphan_backup = root / ".loras.fetch.bak"
    orphan_backup.mkdir()

    monkeypatch.setattr(downloads, "_run_worker", lambda job, **_kw: {"ok": True})
    downloads.download(svc, ["base:sdxl"])

    assert not stranded.exists()
    assert not orphan_backup.exists()

"""A reaped child must leave winjob's registry, on every exit path.

The registry's contract (winjob.py) is "entries are removed on reap". An entry
left behind after its child died is a recycled-pid hazard: a later
``terminate_tracked`` opens whatever process the OS handed that pid to next
with PROCESS_TERMINATE and kills it. rigging and matting both tracked their
children and never untracked them; these pin the untrack on the paths that
were missing it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from warlock import rigging, winjob
from warlock.pipelines import matting


@pytest.fixture
def tracked_pids(monkeypatch):
    """Every pid the code under test registers, while still registering it."""
    seen: list[int] = []
    real_track = winjob.track

    def spy(pid: int, what: str) -> None:
        seen.append(pid)
        real_track(pid, what)

    monkeypatch.setattr(winjob, "track", spy)
    return seen


def test_a_failed_rig_worker_leaves_no_registry_entry(tmp_path, tracked_pids):
    """The nonzero-exit path: the worker refuses the op and dies, and the
    registry must not go on saying it is alive."""
    with pytest.raises(rigging.BlenderError, match="code 2"):
        rigging.run_worker(
            {"op": "sculpt", "result_path": str(tmp_path / "r.json")}, timeout=60
        )
    assert tracked_pids, "the worker was never tracked at all"
    live = winjob.tracked()
    assert all(pid not in live for pid in tracked_pids)


def test_a_timed_out_rig_worker_leaves_no_registry_entry(tmp_path, monkeypatch, tracked_pids):
    """The kill path: a wedged worker is killed by the deadline and reaped,
    which is exactly the moment the contract says the entry comes out."""
    real_popen = subprocess.Popen

    def fake_popen(_cmd, **kw):
        return real_popen([sys.executable, "-c", "import time; time.sleep(120)"], **kw)

    monkeypatch.setattr(rigging.subprocess, "Popen", fake_popen)
    with pytest.raises(rigging.BlenderError, match="timed out"):
        rigging.run_worker(
            {"op": "rig", "result_path": str(tmp_path / "r.json")}, timeout=1.0
        )
    assert tracked_pids
    live = winjob.tracked()
    assert all(pid not in live for pid in tracked_pids)


def test_a_stopped_matting_child_leaves_no_registry_entry(monkeypatch, tracked_pids):
    """``_reset_child_locked`` kills and forgets; it must untrack too, or every
    checkpoint switch and unload leaves a dead pid behind for the life of the
    session."""
    monkeypatch.setattr(
        matting, "CHILD_ARGV", [sys.executable, "-c", "import time; time.sleep(120)"]
    )
    with matting._child_lock:
        matting._start_child("k")
    try:
        assert tracked_pids and tracked_pids[-1] in winjob.tracked()
        matting._stop_child()
        assert tracked_pids[-1] not in winjob.tracked()
    finally:
        matting._stop_child()

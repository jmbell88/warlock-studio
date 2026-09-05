"""The app-side handle on the music worker, over real pipes.

Driven against `fixtures/fake_music_worker.py`, which speaks the protocol and
imports no torch -- so this covers the machinery that actually costs sessions
(spawn, framing, progress, cancel, and above all the kill that is the point of
the whole exercise) without a weight load anywhere.

`test_t2i_client.py`'s shape, one for one. The two clients are deliberately
duplicated rather than sharing a base, so what keeps them from drifting is a
pair of test files making the same assertions in the same order.

The invariant behind every test here: `unload()` must genuinely end the process,
because an `unload()` that returns the VRAM and keeps the host commit is the
defect the child exists to fix
(`docs/measurements/2026-08-22-trampoline-child-pids.md`).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from warlock import models, vram, winjob
from warlock.pipelines import music_client
from warlock.pipelines.music_client import ChildFailed, MusicCancelled, MusicClient

FAKE = Path(__file__).parent / "fixtures" / "fake_music_worker.py"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A client whose child is the fake worker."""
    monkeypatch.setattr(music_client, "CHILD_ARGV", [sys.executable, str(FAKE)])
    made = MusicClient(models.MUSIC_MODELS["ace_step_v1"], tmp_path / "ace-step")
    try:
        yield made
    finally:
        made.close()


def _alive(proc) -> bool:
    return proc is not None and proc.poll() is None


# --- the surface the queue holds it by ---------------------------------------


def test_the_surface_is_the_one_the_queues_lifecycle_paths_expect(client):
    """Every member the idle-eviction and dispatch-credit paths touch.

    Named against ``Text2ImageClient``'s deliberately: the point of matching
    that surface is that ``_q_music`` needed no new lifecycle taught to the
    worker, only a parallel attribute.
    """
    for name in (
        "spec",
        "model_dir",
        "loaded",
        "last_used",
        "last_recipe",
        "load",
        "generate",
        "trim",
        "unload",
        "close",
    ):
        assert hasattr(client, name), f"{name} is missing from the client"


def test_the_silence_timeout_is_its_own_constant_not_the_image_workers():
    """They may be equal today and must not be *the same object*.

    The two are answers to different questions -- how long a diffusion step
    takes on an image latent versus on an audio one -- and sharing them would
    make the first measurement that moves one a change to both.
    """
    from warlock.pipelines import t2i_client

    assert "SILENCE_TIMEOUT" in vars(music_client)
    assert music_client.SILENCE_TIMEOUT is not t2i_client.__dict__.get("SILENCE_TIMEOUT")


def test_a_generate_returns_the_path_and_records_what_produced_it(client, tmp_path):
    out = tmp_path / "track.wav"
    got = client.generate("dark ambient, dungeon", out, audio_duration=30.0, seed=11)
    assert got == out
    assert client.last_recipe["model"] == "ace_step_v1"
    assert client.last_recipe["echo_duration"] == pytest.approx(30.0)
    # An explicit seed becomes a one-element ``manual_seeds`` list, which is the
    # shape ACE-Step takes -- the client is where that translation happens, so
    # no caller has to know it.
    assert client.last_recipe["echo_seeds"] == [11]
    assert client.last_used > 0
    assert client.loaded is True


def test_an_absent_seed_stays_absent_rather_than_becoming_a_number(client, tmp_path):
    client.generate("x", tmp_path / "track.wav")
    assert client.last_recipe["echo_seeds"] is None


def test_the_extra_kwargs_reach_the_child_verbatim(client, tmp_path):
    """Retake, repaint and edit travel as ``**extra``.

    Named here rather than in the signature because they are arguments to the
    same sampler call rather than modes of their own -- spelling each one out
    in the client would be a second copy of a table the worker already keeps,
    and two copies is how one of them loses a key.
    """
    client.generate("x", tmp_path / "track.wav", task="repaint", repaint_start=8)
    assert client.last_recipe["echo_task"] == "repaint"


def test_progress_is_forwarded_to_the_callbacks_the_queue_passes(client, tmp_path):
    states: list[str] = []
    steps: list[tuple[int, int]] = []
    client.generate(
        "x",
        tmp_path / "track.wav",
        on_state=states.append,
        on_step=lambda s, t: steps.append((s, t)),
    )
    assert states == ["load"]
    assert steps == [(1, 2), (2, 2)]


def test_library_chatter_is_not_mistaken_for_an_answer(client, tmp_path):
    # torch, transformers and loguru all print, and the sampler's bar lands on
    # the same stream as the answers. The marker is what tells the two apart; an
    # unmarked line must be logged and stepped over, not read as a response.
    resp = client._request({"op": "load", "chatter": True})
    assert resp["kind"] == "done"
    # And the stream was not left mid-message: the next request still lands.
    assert client.generate("x", tmp_path / "a.wav") == tmp_path / "a.wav"


def test_an_answer_glued_to_a_progress_bar_is_still_found(client, tmp_path):
    """The marker is matched *anywhere* in a line, not only at the start.

    A response written while tqdm's bar is mid-update shares that physical
    line. The parent that tested ``startswith`` dropped exactly those -- which
    cost every step message of every job while letting the final answer
    through, because the bar has finished by then.
    """
    steps: list[tuple[int, int]] = []
    got = client.generate(
        "x",
        tmp_path / "track.wav",
        on_step=lambda s, t: steps.append((s, t)),
        bar=True,
    )
    assert got == tmp_path / "track.wav"
    assert steps == [(1, 2), (2, 2)]


# --- failure, cancel, and the kill -------------------------------------------


def test_a_child_error_becomes_an_exception_carrying_the_childs_words(client):
    with pytest.raises(ChildFailed, match="weights are missing"):
        client._request({"op": "generate", "output": "x", "fail": True})


def test_a_cancel_event_is_relayed_to_a_child_that_is_mid_job(client, tmp_path):
    """The one message that has to arrive while another is being served.

    The fake worker blocks until a cancel line reaches it, so what is asserted
    is that the client *wrote* one -- not that it noticed a flag it set itself.
    Fired from a timer, because the `_request` call is what is blocking.
    """
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    resp = client._request(
        {
            "op": "generate",
            "prompt": "x",
            "output": str(tmp_path / "track.wav"),
            "await_cancel": True,
        },
        cancel_event=cancel,
    )
    assert resp["cancelled"] is True


def test_a_cancelled_answer_becomes_its_own_exception_not_a_failure(
    client, tmp_path, monkeypatch
):
    # ``MusicCancelled`` and not ``ChildFailed``: the queue tells a cancelled
    # job from a failed one by the exception type, and a cancel reported as a
    # failure is logged as one and shown to the user as one.
    monkeypatch.setattr(
        MusicClient, "_request", lambda self, *a, **k: {"cancelled": True}
    )
    with pytest.raises(MusicCancelled):
        client.generate("x", tmp_path / "track.wav")


def test_the_cancel_exception_is_not_the_image_pipelines(client):
    """Two exceptions because there are two pipelines.

    Nothing in the queue may catch one believing it is the other: the two
    stages have separate teardown, and a music cancel handled by the image
    stage's ``finally`` would unload the wrong pipe.
    """
    from warlock.pipelines.text2image import JobCancelled

    assert MusicCancelled is not JobCancelled
    assert not issubclass(MusicCancelled, JobCancelled)


def test_unload_ends_the_process_rather_than_only_dropping_a_reference(
    client, tmp_path
):
    """The defect the child exists to fix.

    In-process, `unload()` returns the VRAM and leaves the allocator's arenas
    holding host commit that nothing but exit gives back. Only a process that
    ends returns that, so the test is about the process.
    """
    client.generate("x", tmp_path / "track.wav")
    proc = client._proc
    assert _alive(proc)
    assert proc.pid in winjob.tracked()

    client.unload()

    assert not _alive(proc), "unload must end the child, not merely forget it"
    assert client.loaded is False
    assert proc.pid not in winjob.tracked()


def test_a_generate_after_unload_starts_a_fresh_child(client, tmp_path):
    client.generate("x", tmp_path / "a.wav")
    first = client._proc
    client.unload()
    client.generate("x", tmp_path / "b.wav")
    assert _alive(client._proc)
    assert client._proc.pid != first.pid


def test_a_child_that_dies_mid_request_is_reported_not_hung(client):
    with pytest.raises(ChildFailed, match="without answering"):
        client._request({"op": "generate", "output": "x", "die": True})
    assert client._proc is None


def test_a_dead_child_is_replaced_on_the_next_request(client, tmp_path):
    client.generate("x", tmp_path / "a.wav")
    client._proc.kill()
    client._proc.wait(timeout=10)
    # The next request must notice and respawn rather than writing into a
    # broken pipe: a second take after a crashed worker is an ordinary session.
    assert client.generate("x", tmp_path / "b.wav") == tmp_path / "b.wav"


def test_close_is_sticky_so_shutdown_cannot_be_outrun_by_a_load(client, tmp_path):
    client.generate("x", tmp_path / "a.wav")
    client.close()
    assert client._proc is None
    with pytest.raises(ChildFailed, match="shutting down"):
        client.generate("x", tmp_path / "b.wav")


def test_trim_on_a_client_with_no_child_does_not_spawn_one(client):
    """``_release_music`` trims unconditionally, so this must not be a load.

    A trim that spawned a child in order to tell it to release nothing would
    make the release path the thing that allocates 8.3 GiB.
    """
    client.trim()
    assert client._proc is None


# --- the reading the parent can no longer take -------------------------------


def test_the_childs_device_reading_reaches_vram(
    client, tmp_path, monkeypatch, real_device_memory
):
    """`vram.device_memory` has no torch reading to take in this process.

    Once the pipeline lives in a child, that is the app's steady state -- so the
    figure admission reads has to come from the child or it does not exist.

    ``live_memory`` is pinned to None so this stays about the *published* rung.
    It gained a rung above it on 2026-09-04 (NVML, which needs no torch), and
    without the pin this test asserts the child's figure on a machine with no
    NVIDIA driver and the real card's on a machine with one.
    """
    monkeypatch.setattr(vram, "_published", None)
    monkeypatch.setattr(vram, "live_memory", lambda: None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    client.generate("x", tmp_path / "track.wav")
    reading = vram.device_memory()
    assert reading is not None
    assert reading.free_gib == pytest.approx(20.5)
    assert reading.total_gib == pytest.approx(31.8)
    assert reading.name == "fake card"


def test_the_child_is_in_the_kill_on_close_job_while_it_lives(client, tmp_path):
    # It holds more memory than anything else the app spawns except the image
    # model; an orphan of this size is the crash the job object exists to
    # prevent.
    client.generate("x", tmp_path / "track.wav")
    if sys.platform == "win32" and winjob.armed():
        assert client._proc.pid in winjob.job_pids()

"""The app-side handle on the image worker, over real pipes.

Driven against `fixtures/fake_t2i_worker.py`, which speaks the protocol and
imports no torch -- so this covers the machinery that actually broke sessions
(spawn, framing, progress, cancel, and above all the kill that is the point of
the whole exercise) without a weight load anywhere.

The invariant behind every test here: `Text2ImageClient` must be substitutable
for `Text2Image` at every call site, and `unload()` must genuinely end the
process, because a `unload()` that returns the VRAM and keeps 21 GiB of host
commit is the defect this replaced
(`docs/measurements/2026-08-22-trampoline-child-pids.md`).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from warlock import models, vram, winjob
from warlock.pipelines import t2i_client
from warlock.pipelines.conditioning import Conditioning
from warlock.pipelines.t2i_client import ChildFailed, Text2ImageClient
from warlock.pipelines.text2image import JobCancelled

FAKE = Path(__file__).parent / "fixtures" / "fake_t2i_worker.py"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A client whose child is the fake worker."""
    monkeypatch.setattr(t2i_client, "CHILD_ARGV", [sys.executable, str(FAKE)])
    made = Text2ImageClient(models.BASE_MODELS["sdxl_cfg"], tmp_path)
    try:
        yield made
    finally:
        made.close()


def _alive(proc) -> bool:
    return proc is not None and proc.poll() is None


# --- the substitutable surface -----------------------------------------------


def test_the_surface_matches_the_class_it_stands_in_for(tmp_path):
    """Every member `queue.Worker` touches on a pipe exists on both.

    Compared on constructed instances rather than on the classes, because half
    of them (`last_used`, `last_recipe`) are instance attributes. A method added
    to `Text2Image` for the queue's benefit that went missing here would fail at
    runtime, on a job, inside a subprocess -- the worst place to learn it.
    Neither construction loads anything.
    """
    from warlock.pipelines.text2image import Text2Image

    spec = models.BASE_MODELS["sdxl_cfg"]
    real = Text2Image(spec, tmp_path)
    proxy = Text2ImageClient(spec, tmp_path)
    for name in (
        "spec",
        "model_dir",
        "loaded",
        "last_used",
        "last_prompt",
        "last_recipe",
        "load",
        "generate",
        "trim",
        "unload",
        "close",
    ):
        assert hasattr(real, name), f"{name} is not on Text2Image any more"
        assert hasattr(proxy, name), f"{name} is missing from the client"
    # And the one that has to agree in *value*: callers read it to resolve where
    # a checkpoint lives.
    assert proxy.model_dir == real.model_dir


def test_a_generate_returns_the_path_and_records_what_produced_it(client, tmp_path):
    out = tmp_path / "ref.png"
    got = client.generate("a copper lantern", out, seed=11, lora="pixelxl")
    assert got == out
    # last_prompt is the child's, not the caller's: the trigger-word prepend
    # happens inside generate and a job records what actually ran.
    assert client.last_prompt == "a copper lantern, trigger"
    assert client.last_recipe == {"seed": 11, "echo": "pixelxl"}
    assert client.last_used > 0
    assert client.loaded is True


def test_progress_is_forwarded_to_the_callbacks_the_queue_passes(client, tmp_path):
    states: list[str] = []
    steps: list[tuple[int, int]] = []
    client.generate(
        "a lantern",
        tmp_path / "ref.png",
        on_state=states.append,
        on_step=lambda s, t: steps.append((s, t)),
    )
    assert states == ["sample"]
    assert steps == [(1, 2), (2, 2)]


def test_library_chatter_is_not_mistaken_for_an_answer(client, tmp_path):
    # diffusers, transformers and PEFT all print, and their progress bars land
    # on the same stream as the answers. The marker is what tells the two apart;
    # an unmarked line must be logged and stepped over, not read as a response.
    resp = client._request({"op": "load", "chatter": True})
    assert resp["kind"] == "done"
    # And the stream was not left mid-message: the next request still lands.
    assert client.generate("x", tmp_path / "a.png") == tmp_path / "a.png"


def test_conditioning_crosses_by_every_field_rather_than_by_recipe(client, tmp_path):
    # as_dict() drops halves that are not in play, so it cannot round-trip; the
    # wire form has to carry each field by name.
    cond = Conditioning(
        ip_adapter="plus",
        ip_image=tmp_path / "ip.png",
        ip_scale=0.4,
        control="depth",
        control_image=tmp_path / "hint.png",
        control_scale=0.7,
        control_end=0.9,
        init_image=tmp_path / "init.png",
        strength=0.55,
    )
    payload = t2i_client._conditioning_payload(cond)
    assert payload == {
        "ip_adapter": "plus",
        "ip_image": str(tmp_path / "ip.png"),
        "ip_scale": 0.4,
        "control": "depth",
        "control_image": str(tmp_path / "hint.png"),
        "control_scale": 0.7,
        "control_end": 0.9,
        "init_image": str(tmp_path / "init.png"),
        "strength": 0.55,
    }
    assert t2i_client._conditioning_payload(None) is None


# --- failure, cancel, and the kill -------------------------------------------


def test_a_child_error_becomes_an_exception_carrying_the_childs_words(
    client, tmp_path
):
    with pytest.raises(ChildFailed, match="checkpoint is missing"):
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
            "output": str(tmp_path / "x.png"),
            "await_cancel": True,
        },
        cancel_event=cancel,
    )
    assert resp["cancelled"] is True


def test_a_cancelled_answer_becomes_the_exception_the_queue_already_handles(
    client, tmp_path, monkeypatch
):
    # `JobCancelled` and not `ChildFailed`: the queue tells a cancelled job from
    # a failed one by the exception type, and a cancel reported as a failure is
    # logged as one and shown to the user as one.
    monkeypatch.setattr(
        Text2ImageClient, "_request", lambda self, *a, **k: {"cancelled": True}
    )
    with pytest.raises(JobCancelled):
        client.generate("x", tmp_path / "x.png")


def test_unload_ends_the_process_rather_than_only_dropping_a_reference(
    client, tmp_path
):
    """The defect this whole change exists to fix.

    In-process, `unload()` returned the VRAM and left up to 21 GiB of host
    commit behind, because the allocator's arenas outlive every reference. Only
    a process that ends returns that, so the test is about the process.
    """
    client.generate("x", tmp_path / "x.png")
    proc = client._proc
    assert _alive(proc)
    assert proc.pid in winjob.tracked()

    client.unload()

    assert not _alive(proc), "unload must end the child, not merely forget it"
    assert client.loaded is False
    assert proc.pid not in winjob.tracked()


def test_a_generate_after_unload_starts_a_fresh_child(client, tmp_path):
    client.generate("x", tmp_path / "a.png")
    first = client._proc
    client.unload()
    client.generate("x", tmp_path / "b.png")
    assert _alive(client._proc)
    assert client._proc.pid != first.pid


def test_a_child_that_dies_mid_request_is_reported_not_hung(client, tmp_path):
    with pytest.raises(ChildFailed, match="without answering"):
        client._request({"op": "generate", "output": "x", "die": True})
    assert client._proc is None


def test_a_dead_child_is_replaced_on_the_next_request(client, tmp_path):
    client.generate("x", tmp_path / "a.png")
    client._proc.kill()
    client._proc.wait(timeout=10)
    # The next request must notice and respawn rather than writing into a
    # broken pipe: a reroll after a crashed worker is an ordinary session.
    assert client.generate("x", tmp_path / "b.png") == tmp_path / "b.png"


def test_close_is_sticky_so_shutdown_cannot_be_outrun_by_a_load(client, tmp_path):
    client.generate("x", tmp_path / "a.png")
    client.close()
    assert client._proc is None
    with pytest.raises(ChildFailed, match="shutting down"):
        client.generate("x", tmp_path / "b.png")


def test_trim_on_a_client_with_no_child_does_not_spawn_one(client):
    client.trim()
    assert client._proc is None


# --- the reading the parent can no longer take -------------------------------


def test_the_childs_device_reading_reaches_vram(client, tmp_path, monkeypatch):
    """`vram.device_memory` returns None without torch in this process.

    Once the pipe lives in a child, that is the app's steady state -- so the
    figure admission reads has to come from the child or it does not exist.
    """
    monkeypatch.setattr(vram, "_published", None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    client.generate("x", tmp_path / "x.png")
    reading = vram.device_memory()
    assert reading is not None
    assert reading.free_gib == pytest.approx(22.5)
    assert reading.total_gib == pytest.approx(31.8)
    assert reading.name == "fake card"


def test_a_live_torch_reading_still_wins_over_a_published_one(monkeypatch):
    # The in-process path must be unaffected: a GPU test, or any session that
    # has torch loaded for another reason, should read the card and not a
    # figure some child reported a minute ago.
    from warlock.vram import DeviceMemory

    vram.publish(1.0, 2.0, "stale")
    monkeypatch.setattr(
        vram, "_read", lambda _t: DeviceMemory(total_gib=31.8, free_gib=30.0, name="live")
    )
    monkeypatch.setitem(sys.modules, "torch", object())
    assert vram.device_memory().name == "live"


def test_the_child_is_in_the_kill_on_close_job_while_it_lives(client, tmp_path):
    # It holds more memory than anything else the app spawns; an orphan of this
    # size is the crash the job object exists to prevent.
    client.generate("x", tmp_path / "x.png")
    if sys.platform == "win32" and winjob.armed():
        assert client._proc.pid in winjob.job_pids()


def test_last_used_advances_so_the_idle_sweep_can_see_the_pipe(client, tmp_path):
    client.generate("x", tmp_path / "a.png")
    first = client.last_used
    time.sleep(0.01)
    client.generate("x", tmp_path / "b.png")
    assert client.last_used > first


def test_a_response_glued_to_a_progress_bar_is_still_read(client, tmp_path):
    """The defect that cost every step message of every job.

    diffusers writes its bar with carriage returns and no newline, onto the
    same merged stream the answers use. A response emitted mid-sample therefore
    lands on the end of "50%|#####     | 2/4", and a parent that tested
    `startswith(MARKER)` dropped it as chatter. The final answer always got
    through -- the bar has finished by then -- so the loss was invisible except
    as a progress bar that never moved.

    Both halves are asserted here: the client finds the marker anywhere in the
    line, and the steps arrive in order.
    """
    steps: list[tuple[int, int]] = []
    resp = client._request(
        {
            "op": "generate",
            "prompt": "x",
            "output": str(tmp_path / "x.png"),
            "bar": True,
        },
        on_step=lambda s, t: steps.append((s, t)),
    )
    assert resp["kind"] == "done"
    assert steps == [(1, 2), (2, 2)], "the glued step messages were dropped"

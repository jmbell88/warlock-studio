"""The t2i worker's protocol, driven in-process over two pipes.

The child exists so that a checkpoint's host commit is returned by a process
exit rather than by a ``gc.collect()`` that cannot return it
(``docs/measurements/2026-08-22-trampoline-child-pids.md``). What has to be
guaranteed here is the protocol around that: every request gets exactly one
terminal response, a failure is reported rather than thrown, progress arrives
before the answer it belongs to, and a cancel is not mistaken for a crash.

Driven through ``serve`` with a stub pipe, so none of it costs a weight load.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from warlock.pipelines import text2image_worker as worker


class _StubPipe:
    """Stands in for ``Text2Image``: the five members the worker touches."""

    def __init__(self, on_generate=None):
        self.loaded = True
        self.last_prompt = ""
        self.last_recipe: dict = {}
        self.calls: list[dict] = []
        self.trimmed = 0
        self._on_generate = on_generate

    def load(self, on_state=None):
        if on_state is not None:
            on_state("load")

    def trim(self):
        self.trimmed += 1

    def generate(self, prompt, output_path, **kw):
        self.calls.append({"prompt": prompt, "output": output_path, **kw})
        if self._on_generate is not None:
            self._on_generate(prompt, output_path, **kw)
        if kw.get("on_state") is not None:
            kw["on_state"]("sample")
        if kw.get("on_step") is not None:
            kw["on_step"](1, 2)
            kw["on_step"](2, 2)
        self.last_prompt = f"{prompt}, trigger words"
        self.last_recipe = {"base_model": "sdxl_cfg", "seed": kw.get("seed")}
        return output_path


def _run(requests: list[dict], pipe: _StubPipe) -> list[dict]:
    """Drive ``serve`` over pre-written stdin. -> the decoded response lines."""
    server = worker._Server("sdxl_cfg", "C:/models", None)
    server._t2i = pipe
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    assert worker.serve(server, stdin, stdout) == 0
    out = []
    for line in stdout.getvalue().splitlines():
        if not line:
            # Every response is written with a leading newline, to close off any
            # partial progress-bar line it might otherwise have landed on. With
            # nothing but responses in this stream that shows up as a blank line
            # before each one.
            continue
        assert line.startswith(worker.MARKER), f"unmarked line: {line!r}"
        out.append(json.loads(line[len(worker.MARKER) :]))
    return out


def _req(**kw):
    base = {
        "op": "generate",
        "prompt": "a copper lantern",
        "output": "out.png",
        "seed": 7,
        "lora_weight": 0.8,
    }
    base.update(kw)
    return base


def test_the_first_line_is_ready_so_the_parent_can_tell_startup_from_a_hang():
    assert _run([], _StubPipe())[0] == {"kind": "ready"}


def test_a_generate_answers_with_the_path_prompt_and_recipe():
    pipe = _StubPipe()
    msgs = _run([_req()], pipe)
    done = msgs[-1]
    assert done["kind"] == "done"
    assert done["path"] == "out.png"
    # last_prompt, not the prompt that was sent: the trigger-word prepend
    # happens inside generate, and a job records what actually produced it.
    assert done["prompt"] == "a copper lantern, trigger words"
    assert done["recipe"] == {"base_model": "sdxl_cfg", "seed": 7}


def test_progress_arrives_before_the_answer_it_belongs_to():
    msgs = _run([_req()], _StubPipe())
    kinds = [m["kind"] for m in msgs]
    assert kinds == ["ready", "state", "step", "step", "done"]
    assert [m for m in msgs if m["kind"] == "step"] == [
        {"kind": "step", "step": 1, "total": 2},
        {"kind": "step", "step": 2, "total": 2},
    ]


def test_every_generate_argument_survives_the_wire():
    pipe = _StubPipe()
    _run(
        [
            _req(
                lora="pixelxl",
                lora_weight=0.65,
                negative_prompt="blurry",
                tile=True,
                sheet=True,
                tilesheet=True,
                size=[1024, 512],
                conditioning={
                    "ip_adapter": "plus",
                    "ip_image": "ip.png",
                    "ip_scale": 0.4,
                    "control": "depth",
                    "control_image": "hint.png",
                    "control_scale": 0.7,
                    "control_end": 0.9,
                    "init_image": "init.png",
                    "strength": 0.55,
                },
            )
        ],
        pipe,
    )
    call = pipe.calls[0]
    assert call["seed"] == 7
    assert call["lora"] == "pixelxl"
    assert call["lora_weight"] == pytest.approx(0.65)
    assert call["negative_prompt"] == "blurry"
    assert call["tile"] is True and call["sheet"] is True
    assert call["tilesheet"] is True
    # A tuple, not the list JSON delivered -- generate unpacks it as (w, h) and
    # the isometric tile sheet is the caller that relies on a 2:1 frame.
    assert call["size"] == (1024, 512)
    cond = call["conditioning"]
    assert cond.ip_adapter == "plus" and str(cond.ip_image) == "ip.png"
    assert cond.control == "depth" and str(cond.control_image) == "hint.png"
    assert str(cond.init_image) == "init.png"
    assert cond.strength == pytest.approx(0.55)
    assert cond.control_end == pytest.approx(0.9)


def test_conditioning_absent_is_none_rather_than_an_empty_object():
    # None and a falsy Conditioning mean the same thing to the pipeline, but
    # only None takes the path that is bit-identical to what the app produced
    # before conditioning existed.
    pipe = _StubPipe()
    _run([_req()], pipe)
    assert pipe.calls[0]["conditioning"] is None


def test_a_cancel_reaches_the_event_a_running_generate_is_watching():
    """The cancel op is the only one that arrives while another is being served.

    Asserted by *waiting* rather than by sampling ``is_set()``: the reader
    thread and the main loop interleave freely, so a sample would pass or fail
    on scheduling. A generate that blocks until the flag arrives is both
    deterministic and what actually happens -- a 50-step sample is where a
    cancel lands.
    """
    seen = {}

    def _wait(prompt, output_path, **kw):
        seen["arrived"] = kw["cancel_event"].wait(timeout=10)

    _run([_req(), {"op": "cancel"}], _StubPipe(on_generate=_wait))
    assert seen["arrived"] is True


def test_a_cancel_for_a_finished_job_does_not_kill_the_next_one():
    """The race that clearing inside the handler would lose.

    A cancel written *before* a generate belongs to whatever came before it, and
    must not be carried into the job that follows. Deterministic: the reader
    clears on the generate line, and the main loop cannot dequeue that generate
    until after the reader has enqueued it.
    """
    seen = {}

    def _check(prompt, output_path, **kw):
        seen["set"] = kw["cancel_event"].is_set()

    _run([{"op": "cancel"}, _req()], _StubPipe(on_generate=_check))
    assert seen["set"] is False


def test_a_cancelled_generate_is_reported_as_cancelled_not_as_a_failure():
    from warlock.pipelines.text2image import JobCancelled

    def _raise(prompt, output_path, **kw):
        raise JobCancelled

    msgs = _run([_req()], _StubPipe(on_generate=_raise))
    assert msgs[-1] == {"kind": "error", "error": "cancelled", "cancelled": True}


def test_a_failing_generate_is_a_response_and_the_loop_survives_it():
    calls = {"n": 0}

    def _boom(prompt, output_path, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("checkpoint is missing")

    msgs = _run([_req(), _req()], _StubPipe(on_generate=_boom))
    first, second = msgs[1], msgs[-1]
    assert first["kind"] == "error" and first["cancelled"] is False
    assert "checkpoint is missing" in first["error"]
    # The child must not have died with the job: the whole reason it is
    # persistent is that the next request finds the pipe still loaded.
    assert second["kind"] == "done"


def test_an_unknown_op_is_refused_by_name_rather_than_ignored():
    msgs = _run([{"op": "enhance"}], _StubPipe())
    assert msgs[-1]["kind"] == "error"
    assert "enhance" in msgs[-1]["error"]


def test_an_unreadable_line_does_not_stall_the_ops_behind_it():
    server = worker._Server("sdxl_cfg", "C:/models", None)
    server._t2i = _StubPipe()
    stdin = io.StringIO("not json\n" + json.dumps(_req()) + "\n")
    stdout = io.StringIO()
    assert worker.serve(server, stdin, stdout) == 0
    kinds = [
        json.loads(line[len(worker.MARKER) :])["kind"]
        for line in stdout.getvalue().splitlines()
        if line
    ]
    assert kinds[-1] == "done"


def test_trim_is_served_without_touching_a_pipe_that_was_never_built():
    server = worker._Server("sdxl_cfg", "C:/models", None)
    stdin = io.StringIO(json.dumps({"op": "trim"}) + "\n")
    stdout = io.StringIO()
    assert worker.serve(server, stdin, stdout) == 0
    last = json.loads(stdout.getvalue().splitlines()[-1][len(worker.MARKER) :])
    assert last["kind"] == "done"
    assert last["loaded"] is False


def test_shutdown_ends_the_loop_without_answering_it():
    msgs = _run([{"op": "shutdown"}], _StubPipe())
    assert [m["kind"] for m in msgs] == ["ready"]


def test_main_refuses_an_argv_that_cannot_name_a_checkpoint():
    assert worker.main([]) == 2
    assert worker.main(["sdxl_cfg"]) == 2


# --- the deadlock the polling reader exists to avoid --------------------------


def _run_reader_probe(args: list[str], timeout: float) -> list[str]:
    """Spawn the reproduction harness, write a line, collect its stages."""
    import subprocess
    import threading as _threading
    from pathlib import Path

    script = Path(__file__).parent / "fixtures" / "stdin_reader_deadlock.py"
    proc = subprocess.Popen(
        [sys.executable, str(script), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stages: list[str] = []
    finished = _threading.Event()

    def _read() -> None:
        for line in proc.stdout:
            line = line.strip()
            if line:
                stages.append(line)
            if line.startswith("SEEN"):
                finished.set()

    _threading.Thread(target=_read, daemon=True).start()
    try:
        proc.stdin.write("during-the-import\n")
        proc.stdin.flush()
        finished.wait(timeout=timeout)
    finally:
        proc.kill()
        proc.wait(timeout=10)
    return stages


@pytest.mark.skipif(sys.platform != "win32", reason="the deadlock is a Win32 one")
def test_the_polling_reader_does_not_deadlock_a_native_import():
    """A daemon thread reading stdin must not stop the main thread importing.

    Measured 2026-08-22: with a *blocking* reader parked on the inherited pipe,
    `import numpy` never returns -- the faulthandler dump shows the main thread
    inside the loader, creating `_multiarray_umath`. Every blocking flavour
    fails alike (`for line in stdin`, `readline()`, `buffer.readline()`, a bare
    `os.read`), while an idle thread and a thread reading a regular file are
    both fine, so the trigger is a *pending* read on the pipe.

    The worker cannot simply read on its main loop the way `matting_worker`
    does -- a cancel has to be read while a generate is running -- so the reader
    peeks and reads only what has already arrived.
    """
    stages = _run_reader_probe([], timeout=60)
    assert "START" in stages
    assert "IMPORTED" in stages, f"the import never finished: {stages}"
    # And it is still a working reader, not merely a harmless one.
    assert "SEEN during-the-import" in stages, stages


@pytest.mark.skipif(sys.platform != "win32", reason="the deadlock is a Win32 one")
def test_the_naive_blocking_reader_still_reproduces_the_deadlock():
    """The guard above is only meaningful while the thing it guards is real.

    If a future Python or Windows fixes the underlying interaction, this fails
    and `_lines_from` can go back to being four lines. Until then it is the
    evidence that the polling is load-bearing rather than superstition.
    """
    # 12 s against an import that takes ~0.1 s when it is not deadlocked: a
    # hundredfold margin, and the whole cost this test adds to the suite.
    stages = _run_reader_probe(["--blocking"], timeout=12)
    assert "START" in stages
    assert "IMPORTED" not in stages, (
        "a blocking stdin reader no longer deadlocks the import; "
        "_lines_from's polling may no longer be necessary"
    )

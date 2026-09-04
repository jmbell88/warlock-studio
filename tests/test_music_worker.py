"""The music worker's protocol, driven in-process over two pipes.

``test_t2i_worker.py``'s shape, one for one, and that is the point: the two
workers are deliberately duplicated rather than sharing a ``_Server`` base, so
the thing that keeps them from drifting is a pair of test files that make the
same assertions in the same order. A reader diffing them should find only what
is genuinely different -- here, that the payload is flat scalars rather than
a value object with images in it.

Driven through ``serve`` with a stub pipeline, so none of it costs a weight
load, a card, or the `music` extra.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from warlock.pipelines import music_worker as worker


class _StubPipe:
    """Stands in for ``ACEStepPipeline``: the members the worker touches."""

    def __init__(self, on_call=None):
        self.loaded = True
        self.calls: list[dict] = []
        self.checkpoints: list[str] = []
        self._on_call = on_call

    def load_checkpoint(self, checkpoint_dir=None, **_kw):
        self.checkpoints.append(checkpoint_dir)

    def __call__(self, **kw):
        self.calls.append(dict(kw))
        if self._on_call is not None:
            self._on_call(**kw)
        if kw.get("on_step") is not None:
            kw["on_step"](1, 2)
            kw["on_step"](2, 2)
        return [kw.get("save_path")]


def _server(pipe: _StubPipe | None = None) -> worker._Server:
    server = worker._Server("ace_step_v1", "C:/models/ace-step-v1-3.5b")
    if pipe is not None:
        server._pipe = pipe
    return server


def _run(requests: list[dict], pipe: _StubPipe) -> list[dict]:
    """Drive ``serve`` over pre-written stdin. -> the decoded response lines."""
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    assert worker.serve(_server(pipe), stdin, stdout) == 0
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


def _req(tmp_path=None, **kw):
    base = {
        "op": "generate",
        "prompt": "dark ambient, dungeon, low strings",
        "lyrics": "",
        "output": str((tmp_path / "track.wav") if tmp_path else "track.wav"),
    }
    base.update(kw)
    return base


def test_the_marker_is_its_own_so_a_t2i_line_is_never_read_as_a_music_one():
    from warlock.pipelines import text2image_worker

    assert worker.MARKER != text2image_worker.MARKER


def test_the_first_line_is_ready_so_the_parent_can_tell_startup_from_a_hang(tmp_path):
    assert _run([], _StubPipe())[0] == {"kind": "ready"}


def test_a_generate_answers_with_the_path_and_the_recipe(tmp_path):
    msgs = _run([_req(tmp_path)], _StubPipe())
    done = msgs[-1]
    assert done["kind"] == "done"
    assert done["path"] == str(tmp_path / "track.wav")
    # The model key travels in the recipe, so a stored row names what actually
    # produced it rather than whatever the registry's default is today.
    assert done["recipe"]["model"] == "ace_step_v1"
    assert done["recipe"]["scheduler_type"] == "euler"


def test_progress_arrives_before_the_answer_it_belongs_to(tmp_path):
    msgs = _run([_req(tmp_path)], _StubPipe())
    assert [m["kind"] for m in msgs] == ["ready", "state", "step", "step", "done"]
    assert [m for m in msgs if m["kind"] == "step"] == [
        {"kind": "step", "step": 1, "total": 2},
        {"kind": "step", "step": 2, "total": 2},
    ]


def test_every_recipe_argument_survives_the_wire(tmp_path):
    pipe = _StubPipe()
    _run(
        [
            _req(
                tmp_path,
                lyrics="[verse]\na line",
                audio_duration=120.0,
                infer_step=27,
                guidance_scale=9.5,
                scheduler_type="heun",
                cfg_type="cfg",
                omega_scale=4.0,
                manual_seeds=[11],
                use_erg_tag=False,
            )
        ],
        pipe,
    )
    call = pipe.calls[0]
    assert call["prompt"] == "dark ambient, dungeon, low strings"
    assert call["lyrics"] == "[verse]\na line"
    assert call["audio_duration"] == pytest.approx(120.0)
    assert call["infer_step"] == 27
    assert call["guidance_scale"] == pytest.approx(9.5)
    assert call["scheduler_type"] == "heun"
    assert call["cfg_type"] == "cfg"
    assert call["omega_scale"] == pytest.approx(4.0)
    assert call["manual_seeds"] == [11]
    assert call["use_erg_tag"] is False
    # A WAV, always: the format is the worker's business and not the caller's,
    # because everything downstream -- files.MEDIA, the Sirens bridge, the
    # mixer -- is written against one.
    assert call["format"] == "wav"


def test_the_retake_and_edit_kwargs_are_the_same_call_rather_than_new_ops(tmp_path):
    """They change what the sampler is asked for, not what the worker is.

    A new op per task would have meant four copies of the load check, the cancel
    wiring and the vitals report -- which is exactly how two of them would come
    to disagree about one of those three.
    """
    pipe = _StubPipe()
    _run(
        [
            _req(
                tmp_path,
                task="repaint",
                retake_seeds=[3, 4],
                retake_variance=0.25,
                repaint_start=8,
                repaint_end=24,
                src_audio_path="prior.wav",
                edit_target_prompt="brighter",
                edit_n_min=0.1,
                edit_n_max=0.8,
            )
        ],
        pipe,
    )
    call = pipe.calls[0]
    assert call["task"] == "repaint"
    assert call["retake_seeds"] == [3, 4]
    assert call["retake_variance"] == pytest.approx(0.25)
    assert (call["repaint_start"], call["repaint_end"]) == (8, 24)
    assert call["src_audio_path"] == "prior.wav"
    assert call["edit_target_prompt"] == "brighter"
    assert call["edit_n_min"] == pytest.approx(0.1)
    assert call["edit_n_max"] == pytest.approx(0.8)


def test_an_absent_seed_is_none_rather_than_a_zero(tmp_path):
    # Zero is a *seed*, and a request that did not ask for one must not be
    # given the same one every time -- which is four identical takes.
    pipe = _StubPipe()
    _run([_req(tmp_path)], pipe)
    assert pipe.calls[0]["manual_seeds"] is None


def test_the_output_directory_is_created_before_the_model_writes_into_it(tmp_path):
    out = tmp_path / "jobs" / "abc123" / "track.wav"
    _run([_req(output=str(out))], _StubPipe())
    assert out.parent.is_dir()


def test_a_cancel_reaches_the_event_a_running_generate_is_watching(tmp_path):
    """The cancel op is the only one that arrives while another is being served.

    Asserted by *waiting* rather than by sampling ``is_set()``: the reader
    thread and the main loop interleave freely, so a sample would pass or fail
    on scheduling. A generate that blocks until the flag arrives is both
    deterministic and what actually happens -- a 60-step sample is where a
    cancel lands.
    """
    seen = {}

    def _wait(**kw):
        seen["arrived"] = kw["cancel_event"].wait(timeout=10)

    _run([_req(tmp_path), {"op": "cancel"}], _StubPipe(on_call=_wait))
    assert seen["arrived"] is True


def test_a_cancel_for_a_finished_job_does_not_kill_the_next_one(tmp_path):
    """The race that clearing inside the handler would lose.

    A cancel written *before* a generate belongs to whatever came before it, and
    must not be carried into the job that follows.
    """
    seen = {}

    def _check(**kw):
        seen["set"] = kw["cancel_event"].is_set()

    _run([{"op": "cancel"}, _req(tmp_path)], _StubPipe(on_call=_check))
    assert seen["set"] is False


def test_a_cancelled_generate_is_reported_as_cancelled_not_as_a_failure(tmp_path):
    # From ``_workerio``, not from the vendored pipeline: importing it from
    # there would drag torch into this test, which is the whole thing the
    # exception's placement avoids.
    from warlock.pipelines._workerio import WarlockCancelled

    def _raise(**kw):
        raise WarlockCancelled

    msgs = _run([_req(tmp_path)], _StubPipe(on_call=_raise))
    assert msgs[-1] == {"kind": "error", "error": "cancelled", "cancelled": True}


def test_a_failing_generate_is_a_response_and_the_loop_survives_it(tmp_path):
    calls = {"n": 0}

    def _boom(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("weights are missing")

    msgs = _run([_req(tmp_path), _req(tmp_path)], _StubPipe(on_call=_boom))
    # Found by kind rather than by index: a generate emits its "load" state
    # before it can fail, so the error is not the message after ready.
    first = next(m for m in msgs if m["kind"] == "error")
    second = msgs[-1]
    assert first["cancelled"] is False
    assert "weights are missing" in first["error"]
    # The child must not have died with the job: the whole reason it is
    # persistent is that the next request finds the pipeline still loaded.
    assert second["kind"] == "done"


def test_load_points_the_pipeline_at_the_directory_it_was_constructed_with(tmp_path):
    pipe = _StubPipe()
    msgs = _run([{"op": "load"}], pipe)
    # Through ``str(Path(...))``, because the server stores a Path and the
    # separator it renders back with is the platform's.
    assert pipe.checkpoints == [str(Path("C:/models/ace-step-v1-3.5b"))]
    assert msgs[-1]["kind"] == "done" and msgs[-1]["loaded"] is True


def test_an_unknown_op_is_refused_by_name_rather_than_ignored():
    msgs = _run([{"op": "remix"}], _StubPipe())
    assert msgs[-1]["kind"] == "error"
    assert "remix" in msgs[-1]["error"]


def test_an_unreadable_line_does_not_stall_the_ops_behind_it(tmp_path):
    stdin = io.StringIO("not json\n" + json.dumps(_req(tmp_path)) + "\n")
    stdout = io.StringIO()
    assert worker.serve(_server(_StubPipe()), stdin, stdout) == 0
    kinds = [
        json.loads(line[len(worker.MARKER) :])["kind"]
        for line in stdout.getvalue().splitlines()
        if line
    ]
    assert kinds[-1] == "done"


def test_trim_is_served_without_touching_a_pipe_that_was_never_built():
    """The op exists so the release path can call it unconditionally.

    ``_release_music`` trims and then unloads without asking which kind of pipe
    it holds, so a trim against a worker that has never constructed one has to
    be an answer rather than a construction -- otherwise the release path is
    what *loads* 8.3 GiB.
    """
    stdin = io.StringIO(json.dumps({"op": "trim"}) + "\n")
    stdout = io.StringIO()
    server = _server()
    assert worker.serve(server, stdin, stdout) == 0
    last = json.loads(stdout.getvalue().splitlines()[-1][len(worker.MARKER) :])
    assert last["kind"] == "done"
    assert last["loaded"] is False
    assert server._pipe is None


def test_shutdown_ends_the_loop_without_answering_it():
    msgs = _run([{"op": "shutdown"}], _StubPipe())
    assert [m["kind"] for m in msgs] == ["ready"]


def test_main_refuses_an_argv_that_cannot_name_a_model():
    assert worker.main([]) == 2
    assert worker.main(["ace_step_v1"]) == 2

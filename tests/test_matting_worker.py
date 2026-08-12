"""The matting child, driven in-process.

Its host half is covered in ``tests/test_matting.py`` -- this is the other end
of the pipe: ``handle`` for one request, and ``main`` for the loop and the
marker. ``loadprobe``'s test shape (``tests/test_doctor.py``), for
``loadprobe``'s reason: the module is deliberately importable and callable
without spawning anything.
"""

from __future__ import annotations

import io
import json
import sys

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import matting, matting_worker


@pytest.fixture(autouse=True)
def _no_model_carried_between_tests():
    matting.unload()
    yield
    matting.unload()


def _png(tmp_path, name="in.png"):
    path = tmp_path / name
    Image.new("RGB", (48, 32), (200, 200, 200)).save(path, "PNG")
    return path


def _request(tmp_path, **over):
    req = {
        "model_dir": str(tmp_path / "birefnet"),
        "device": "cpu",
        "input": str(_png(tmp_path)),
        "output": str(tmp_path / "mask.png"),
    }
    req.update(over)
    return req


def test_a_matte_is_written_as_a_png_at_the_inputs_own_size(tmp_path, monkeypatch):
    # Pixels go in a file, never down the pipe: blender_worker's rule, and here
    # for a second reason -- a 4096-square matte is 16 MB against a 64 KB pipe
    # buffer, which would deadlock a strict request/response exchange.
    monkeypatch.setattr(matting, "_load", lambda path, device: object())

    def fake_infer(image, model):
        out = np.zeros((image.size[1], image.size[0]), dtype=bool)
        out[4:8, 4:8] = True
        return out

    monkeypatch.setattr(matting, "_infer", fake_infer)

    req = _request(tmp_path)
    assert matting_worker.handle(req) == {"ok": True}

    with Image.open(req["output"]) as written:
        written.load()
        array = np.asarray(written.convert("L"))
    assert written.size == (48, 32)
    # Exactly 0 or 255, which is what the host's `> 127` reads back.
    assert set(np.unique(array).tolist()) == {0, 255}
    assert array[5, 5] == 255 and array[20, 20] == 0


def test_a_checkpoint_that_will_not_load_is_reported_as_a_load_failure(tmp_path, monkeypatch):
    # `stage` is the whole reason the response is structured: it is what the
    # parent's failure memo keys on. A load failure is permanent for the
    # session; a bad frame is not.
    def boom(*a, **k):
        raise RuntimeError("No module named 'einops'")

    # ``birefnet.load``, not ``transformers``: the modelling code is vendored
    # and the loader is stricter, so the failure this reports is now "the
    # checkpoint does not map onto the vendored architecture" as well as a
    # missing package. Both are permanent for the session, which is what the
    # memo below is about.
    from warlock.pipelines import birefnet

    monkeypatch.setattr(birefnet, "load", boom)
    resp = matting_worker.handle(_request(tmp_path))
    assert resp["ok"] is False
    assert resp["stage"] == "load"
    # The type as well as the message: an ImportError and a shape mismatch are
    # two entirely different repairs.
    assert "RuntimeError" in resp["error"] and "einops" in resp["error"]


def test_an_image_that_cannot_be_read_is_a_run_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(matting, "_load", lambda path, device: object())
    resp = matting_worker.handle(_request(tmp_path, input=str(tmp_path / "gone.png")))
    assert resp["ok"] is False
    assert resp["stage"] == "run"


def test_a_malformed_request_is_answered_rather_than_raised(tmp_path):
    # Anything that escaped handle() would kill the child mid-export and lose
    # the loaded model with it.
    resp = matting_worker.handle({"device": "cpu"})
    assert resp["ok"] is False
    assert resp["stage"] == "run"


def _drive(monkeypatch, lines: list[str]) -> list[str]:
    """Run main() over ``lines`` and return its marked output lines."""
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(lines)))
    monkeypatch.setattr(sys, "stdout", out)
    assert matting_worker.main([]) == 0
    return [
        line
        for line in out.getvalue().splitlines()
        if line.startswith(matting_worker.MARKER)
    ]


def test_main_answers_one_marked_line_per_request(tmp_path, monkeypatch):
    monkeypatch.setattr(matting, "_load", lambda path, device: object())
    monkeypatch.setattr(
        matting,
        "_infer",
        lambda image, model: np.ones((image.size[1], image.size[0]), dtype=bool),
    )
    reqs = [
        json.dumps(_request(tmp_path, output=str(tmp_path / f"m{i}.png"))) + "\n"
        for i in range(2)
    ]
    answers = _drive(monkeypatch, [reqs[0], "\n", reqs[1]])
    assert len(answers) == 2
    assert all(json.loads(a[len(matting_worker.MARKER) :])["ok"] for a in answers)


def test_an_unreadable_line_is_answered_and_the_loop_survives(tmp_path, monkeypatch):
    monkeypatch.setattr(matting, "_load", lambda path, device: object())
    monkeypatch.setattr(
        matting,
        "_infer",
        lambda image, model: np.ones((image.size[1], image.size[0]), dtype=bool),
    )
    answers = _drive(
        monkeypatch, ["not json\n", json.dumps(_request(tmp_path)) + "\n"]
    )
    assert len(answers) == 2
    first = json.loads(answers[0][len(matting_worker.MARKER) :])
    assert first["ok"] is False
    assert json.loads(answers[1][len(matting_worker.MARKER) :])["ok"] is True


def test_a_stray_print_cannot_land_inside_a_response(tmp_path, monkeypatch):
    # transformers prints, and so does BiRefNet's own modelling code. main()
    # points sys.stdout at stderr for the duration, so a print goes into the
    # log stream the parent drains; the marker is what makes that
    # belt-and-braces rather than a hope.
    monkeypatch.setattr(matting, "_load", lambda path, device: print("chatter") or object())
    monkeypatch.setattr(
        matting,
        "_infer",
        lambda image, model: np.ones((image.size[1], image.size[0]), dtype=bool),
    )
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_request(tmp_path)) + "\n"))
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    matting_worker.main([])

    assert "chatter" not in out.getvalue()
    assert "chatter" in err.getvalue()
    assert out.getvalue().strip().startswith(matting_worker.MARKER)
    # And the redirect is undone, or a second main() in one process stacks it.
    assert sys.stdout is out

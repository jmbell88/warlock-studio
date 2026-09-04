"""A worker that speaks the music protocol without importing torch.

Lets ``tests/test_music_client.py`` exercise the real pipe machinery -- spawn,
marker framing, progress forwarding, cancel, kill -- at subprocess speed and
with no weights anywhere. It deliberately mirrors ``music_worker``'s *shape* (a
stdin reader thread, one marked terminal line per request) rather than importing
it, so a change to the real worker that breaks the contract shows up here as a
failure instead of being inherited silently.

``fixtures/fake_t2i_worker.py``'s twin, key for key, because the two clients are
duplicated on purpose and a pair of fakes that diff cleanly is part of what
keeps them from drifting.

Behaviour is driven by keys on the request, so one script serves every case:

``fail``          answer with an error instead of a done
``await_cancel``  block until a cancel arrives, then answer cancelled
``die``           exit without answering at all
``chatter``       print an unmarked line first, as loguru and tqdm do
``bar``           write a carriage-returned partial line, then glue answers to it
"""

from __future__ import annotations

import json
import queue
import sys
import threading

MARKER = "@@warlock-music@@ "

_cancel = threading.Event()
_requests: queue.Queue = queue.Queue()


def _emit(msg: dict) -> None:
    # Leading newline like the real worker: it is what closes off a progress
    # bar's partial line so the marker starts one of its own.
    sys.stdout.write("\n" + MARKER + json.dumps(msg) + "\n")
    sys.stdout.flush()


def _emit_glued(msg: dict) -> None:
    """Emit *without* the leading newline, straight onto a bar's partial line.

    The failure this reproduces: a response written while the sampler's tqdm bar
    is mid-update shares that physical line, so a parent testing `startswith`
    never sees it.
    """
    sys.stdout.write(MARKER + json.dumps(msg) + "\n")
    sys.stdout.flush()


def _pump() -> None:
    try:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except ValueError:
                continue
            if req.get("op") == "cancel":
                _cancel.set()
                continue
            if req.get("op") == "generate":
                _cancel.clear()
            _requests.put(req)
    finally:
        _requests.put(None)


def _vitals(loaded: bool) -> dict:
    return {
        "loaded": loaded,
        "device_free_gib": 20.5,
        "device_total_gib": 31.8,
        "allocated_gib": 8.3,
        "reserved_gib": 8.9,
        "device_name": "fake card",
    }


def main() -> int:
    threading.Thread(target=_pump, daemon=True).start()
    _emit({"kind": "ready"})
    loaded = False
    while True:
        req = _requests.get()
        if req is None:
            return 0
        op = req.get("op")
        if req.get("chatter"):
            sys.stdout.write("Loading transformer weights: 100%\n")
            sys.stdout.flush()
        if req.get("die"):
            return 3
        if op == "load":
            loaded = True
            _emit({"kind": "done", **_vitals(loaded)})
            continue
        if op == "trim":
            _emit({"kind": "done", **_vitals(loaded)})
            continue
        if op == "generate":
            if req.get("fail"):
                _emit(
                    {
                        "kind": "error",
                        "error": "the weights are missing",
                        "cancelled": False,
                    }
                )
                continue
            if req.get("await_cancel"):
                if _cancel.wait(timeout=30):
                    _emit({"kind": "error", "error": "cancelled", "cancelled": True})
                else:
                    _emit(
                        {
                            "kind": "error",
                            "error": "no cancel arrived",
                            "cancelled": False,
                        }
                    )
                continue
            loaded = True
            _emit({"kind": "state", "text": "load"})
            if req.get("bar"):
                # tqdm mid-update: carriage return, no newline. The step
                # messages below are then written straight onto this line.
                sys.stdout.write("\r50%|#####     | 30/60")
                sys.stdout.flush()
                _emit_glued({"kind": "step", "step": 1, "total": 2})
                _emit_glued({"kind": "step", "step": 2, "total": 2})
                _emit(
                    {
                        "kind": "done",
                        "path": req["output"],
                        "recipe": {},
                        **_vitals(loaded),
                    }
                )
                continue
            _emit({"kind": "step", "step": 1, "total": 2})
            _emit({"kind": "step", "step": 2, "total": 2})
            _emit(
                {
                    "kind": "done",
                    "path": req["output"],
                    "recipe": {
                        "model": "ace_step_v1",
                        "echo_duration": req.get("audio_duration"),
                        "echo_seeds": req.get("manual_seeds"),
                        "echo_task": req.get("task"),
                    },
                    **_vitals(loaded),
                }
            )
            continue
        _emit({"kind": "error", "error": f"unknown op: {op!r}", "cancelled": False})


if __name__ == "__main__":
    raise SystemExit(main())

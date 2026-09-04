"""The music pipeline in a child process, one request per line.

**Why a child.** For ``text2image_worker``'s reason, measured the same way: a
3.5B transformer plus a DCAE and a vocoder charge host commit that the app never
gets back, and ``unload()`` in-process returns the VRAM while leaving the
allocator's arenas resident. The app process is the one that has to keep
running, so it is the one that must never pay it.

**Why persistent, and not one-shot.** A retake, a repaint and a second take on
the same tags are three generates against the same 8.3 GiB of weights, and a
one-shot child would pay the load each time. This one holds the pipeline across
requests and dies on ``music_client.MusicClient.unload()`` -- a process kill,
which genuinely returns everything -- or with the app, via the kill-on-close
job.

**What it is not.** It holds no model code of its own: it constructs the
vendored ``acestep.pipeline_ace_step.ACEStepPipeline`` and calls it, so the
model decisions stay in one auditable place. See that package's
``ATTRIBUTION.md`` for the three modifications, one of which -- the cancel hook
-- this module could not work without.

**The protocol.** One JSON object per stdin line in, one marked JSON line on
stdout back. The WAV travels as a *file*, never over the pipe, for the reason a
1024x1024 PNG does; ``ACEStepPipeline`` already takes a ``save_path``, so
nothing upstream had to change to satisfy it.

    {"op": "generate", "prompt": ..., "lyrics": ..., "output": ..., ...}
      -> marker {"kind": "state", "text": "load"}
      -> marker {"kind": "step", "step": 3, "total": 60}
      -> marker {"kind": "done", "path": ..., "recipe": {...}, ...}
      -> marker {"kind": "error", "error": ..., "cancelled": false}

``cancel`` is the one op that arrives *while* another is being served, so stdin
is drained by a reader thread rather than by the main loop -- and that thread
must never leave a read pending, which is what ``_workerio.lines_from`` is for.
"""

from __future__ import annotations

import json
import queue as _queue
import sys
import threading
from pathlib import Path
from typing import Any

from ._workerio import WarlockCancelled, lines_from

MARKER = "@@warlock-music@@ "
"""Prefixed onto every response line.

Its own marker rather than a shared one: the two model workers write to
separate pipes today, but a t2i log line read as a music answer would be a
silent, baffling failure, and one distinct string costs nothing.

torch, transformers and loguru all print, and ACE-Step's sampling loop writes a
tqdm bar; ``main`` also points ``sys.stdout`` at stderr for the duration, so
this is belt-and-braces rather than a hope.
"""


def _scalar_kwargs(req: dict[str, Any]) -> dict[str, Any]:
    """The recipe half of a generate request, defaulted.

    ACE-Step's payload is flat scalars and lists -- no images, no adapters, no
    staged temporary files -- so this is a ``.get()`` table rather than the
    value-object rebuild ``text2image_worker._conditioning`` needs. Every key is
    passed to ``__call__`` under the name upstream gives it.
    """
    seeds = req.get("manual_seeds")
    retake_seeds = req.get("retake_seeds")
    return {
        "audio_duration": float(req.get("audio_duration", 60.0)),
        "infer_step": int(req.get("infer_step", 60)),
        "guidance_scale": float(req.get("guidance_scale", 15.0)),
        "scheduler_type": str(req.get("scheduler_type", "euler")),
        "cfg_type": str(req.get("cfg_type", "apg")),
        "omega_scale": float(req.get("omega_scale", 10.0)),
        "guidance_interval": float(req.get("guidance_interval", 0.5)),
        "guidance_interval_decay": float(req.get("guidance_interval_decay", 0.0)),
        "min_guidance_scale": float(req.get("min_guidance_scale", 3.0)),
        "use_erg_tag": bool(req.get("use_erg_tag", True)),
        "use_erg_lyric": bool(req.get("use_erg_lyric", True)),
        "use_erg_diffusion": bool(req.get("use_erg_diffusion", True)),
        "manual_seeds": [int(s) for s in seeds] if seeds else None,
        # Retake / repaint / edit are kwargs of the same call rather than ops of
        # their own: they change what the sampler is asked for, not what the
        # worker is. A new op per task would have meant four copies of the
        # load-check, the cancel wiring and the vitals report.
        "task": str(req.get("task", "text2music")),
        "retake_seeds": [int(s) for s in retake_seeds] if retake_seeds else None,
        "retake_variance": float(req.get("retake_variance", 0.5)),
        "repaint_start": int(req.get("repaint_start", 0)),
        "repaint_end": int(req.get("repaint_end", 0)),
        "src_audio_path": req.get("src_audio_path"),
        "edit_target_prompt": req.get("edit_target_prompt"),
        "edit_target_lyrics": req.get("edit_target_lyrics"),
        "edit_n_min": float(req.get("edit_n_min", 0.0)),
        "edit_n_max": float(req.get("edit_n_max", 1.0)),
    }


class _Server:
    """One resident pipeline and the loop that serves it.

    A class rather than module globals so a test can drive a server over two
    pipes in-process, the way ``matting_worker.main`` can be called directly.
    """

    #: Deferred as *config* knobs deliberately -- see docs/INVARIANTS.md. Both
    #: are trades against VRAM that no measurement has been taken for yet, so
    #: they are class attributes a GPU-lane experiment can set rather than
    #: Config fields whose SETTINGS rows would have to be written first.
    cpu_offload = False
    overlapped_decode = False

    def __init__(self, model_key: str, model_dir: str) -> None:
        # Two arguments where the image worker takes three: ACE-Step ships one
        # weight set, with no base/variant split for a key to select between.
        # The key is still carried so the registry entry -- and therefore the
        # recipe the parent records -- is looked up rather than pickled.
        self.model_key = model_key
        self.model_dir = Path(model_dir)
        self._pipe: Any = None
        # Set *and cleared* by the reader thread, read by the sampling step
        # callback. Both on that one thread, because the parent's write order is
        # the only authority on which job a cancel belongs to: cleared as a
        # generate is enqueued, so a cancel written after it is seen and one
        # written before it is discarded with the job it was meant for. See
        # ``text2image_worker._Server.cancel`` for why clearing it inside
        # ``op_generate`` instead only looks equivalent.
        self.cancel = threading.Event()

    def pipe(self) -> Any:
        """The ``ACEStepPipeline``, constructed on first use.

        Deferred so startup is a Python import and not an 8 GiB read: the parent
        spawns this child when it decides to *have* a pipe, which is earlier
        than the moment it needs one loaded.

        ``torch_compile=False`` deliberately. The warm-up it charges is a bad
        trade against a queue that evicts this pipe on idle -- the compile would
        be paid again on the next take, and the app's whole VRAM story depends
        on being free to evict.
        """
        if self._pipe is None:
            from .acestep.pipeline_ace_step import ACEStepPipeline

            self._pipe = ACEStepPipeline(
                checkpoint_dir=str(self.model_dir),
                dtype="bfloat16",
                torch_compile=False,
                cpu_offload=self.cpu_offload,
                overlapped_decode=self.overlapped_decode,
            )
        return self._pipe

    # --- ops -----------------------------------------------------------------

    def op_load(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        emit({"kind": "state", "text": "load"})
        self.pipe().load_checkpoint(str(self.model_dir))
        return {"kind": "done", **self._vitals()}

    def op_generate(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        pipe = self.pipe()
        output = Path(req["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        kwargs = _scalar_kwargs(req)
        emit({"kind": "state", "text": "load"})
        try:
            pipe(
                format="wav",
                prompt=str(req.get("prompt") or ""),
                lyrics=str(req.get("lyrics") or ""),
                save_path=str(output),
                cancel_event=self.cancel,
                on_step=lambda step, total: emit(
                    {"kind": "step", "step": step, "total": total}
                ),
                **kwargs,
            )
        except WarlockCancelled:
            # Named on the wire rather than inferred from the message: the
            # parent re-raises the *same* exception type, and a cancel that
            # arrived as a generic error would be logged as a failure and shown
            # to the user as one.
            return {"kind": "error", "error": "cancelled", "cancelled": True}
        return {
            "kind": "done",
            "path": str(output),
            "recipe": {"model": self.model_key, **kwargs},
            **self._vitals(),
        }

    def op_trim(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        """Return cached blocks without dropping the pipeline.

        Thinner than the image worker's -- ACE-Step has no adapter cache to
        release -- but the op stays, so the parent's release path can call it
        unconditionally the way ``_release_t2i`` does rather than branching on
        which kind of pipe it holds.
        """
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 -- a hint must never fail a job
                pass
        return {"kind": "done", **self._vitals()}

    def _vitals(self) -> dict[str, Any]:
        """The device readings the parent can no longer take for itself.

        ``vram.device_memory`` reaches torch through ``sys.modules`` and returns
        None when it is absent -- which, once the pipe lives here, is the app
        process's steady state. Device-wide free/total (which sees the image
        worker and trellis-server too, because ``mem_get_info`` wraps
        ``cudaMemGetInfo``) plus this process's own allocated/reserved.
        """
        loaded = self._pipe is not None and bool(getattr(self._pipe, "loaded", False))
        out: dict[str, Any] = {"loaded": loaded}
        torch = sys.modules.get("torch")
        if torch is None:
            return out
        try:
            if not torch.cuda.is_available():
                return out
            gib = 1024**3
            free, total = torch.cuda.mem_get_info()
            out["device_free_gib"] = free / gib
            out["device_total_gib"] = total / gib
            out["allocated_gib"] = torch.cuda.memory_allocated() / gib
            out["reserved_gib"] = torch.cuda.memory_reserved() / gib
            name = getattr(torch.cuda, "get_device_name", None)
            if callable(name):
                out["device_name"] = str(name(0))
        except Exception:  # noqa: BLE001 -- a reading must never fail a job
            pass
        return out

    _OPS = {
        "load": op_load,
        "generate": op_generate,
        "trim": op_trim,
    }

    def handle(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        """Serve one request. Never raises.

        A failure that escaped here would kill the child mid-job and take the
        loaded checkpoint with it; every one is a response instead, and the
        parent turns it back into the exception the caller expected.
        """
        op = str(req.get("op") or "")
        handler = self._OPS.get(op)
        if handler is None:
            return {"kind": "error", "error": f"unknown op: {op!r}", "cancelled": False}
        try:
            return handler(self, req, emit)
        except Exception as exc:  # noqa: BLE001 -- the whole point is to report it
            return {
                "kind": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "cancelled": False,
            }


def serve(server: _Server, stdin: Any, stdout: Any) -> int:
    """Run the request loop until ``stdin`` closes. -> 0.

    Split from ``main`` so a test can drive it over two pipes without spawning
    anything -- the same reason ``matting_worker.main`` takes an ignored argv.
    """
    requests: Any = _queue.Queue()

    def _pump() -> None:
        try:
            for raw in lines_from(stdin):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    req = json.loads(raw)
                except ValueError:
                    # Unreadable lines are dropped here rather than queued: the
                    # main loop would have nothing to answer them with, and a
                    # cancel is the one message that must never wait behind a
                    # parse failure.
                    continue
                if req.get("op") == "cancel":
                    # Handled on this thread, which is the whole reason there
                    # is one: the main thread is inside the sampler when this
                    # arrives.
                    server.cancel.set()
                    continue
                if req.get("op") == "shutdown":
                    requests.put(None)
                    return
                if req.get("op") == "generate":
                    # Here rather than in the handler -- see ``_Server.cancel``.
                    server.cancel.clear()
                requests.put(req)
        finally:
            requests.put(None)

    reader = threading.Thread(target=_pump, name="music-worker-stdin", daemon=True)
    reader.start()

    def emit(msg: dict[str, Any]) -> None:
        # A *leading* newline as well as a trailing one. ACE-Step's sampling
        # loop writes carriage-returned partial tqdm lines with no newline of
        # their own, onto the same merged stream, so a response emitted
        # mid-sample would otherwise land on the end of a progress bar and stop
        # being a line that begins with the marker. The parent matches the
        # marker anywhere in a line; this is the other half of that pair, and
        # the cheaper half, because it keeps the log readable too.
        stdout.write("\n" + MARKER + json.dumps(msg) + "\n")
        stdout.flush()

    emit({"kind": "ready"})
    while True:
        req = requests.get()
        if req is None:
            return 0
        emit(server.handle(req, emit))


def main(argv: list[str] | None = None) -> int:
    """Serve until stdin closes. -> 0.

    ``argv`` is ``[model_key, model_dir]``, with the spec looked up by key
    rather than pickled, because the registry is the same module in both
    processes and a spec that crossed the wire could disagree with it.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        sys.stderr.write("usage: music_worker <model_key> <model_dir>\n")
        return 2
    out = sys.stdout
    # Everything torch, transformers and loguru print goes to the log stream
    # rather than into the middle of a response. Restored on the way out so a
    # test that calls main() twice does not stack the redirect.
    sys.stdout = sys.stderr
    try:
        server = _Server(args[0], args[1])
        return serve(server, sys.stdin, out)
    finally:
        sys.stdout = out


if __name__ == "__main__":
    raise SystemExit(main())

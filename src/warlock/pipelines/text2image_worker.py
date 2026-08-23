"""The image pipeline in a child process, one request per line.

**Why a child.** Loading a checkpoint charges host commit that the app never
gets back. Measured on 2026-08-22: ``sdxl_cfg`` cost +8.8 GiB of private commit
and returned 4.9 of it; ``flux_klein_distilled`` cost **+21.1 GiB and returned
0.1**, against the ``host_peak_gib=16.0`` the registry ships. ``unload()``
returns the VRAM exactly as it claims and the host figure does not move, because
what holds it is the allocator's arenas rather than a live object -- the same
finding ``docs/measurements/2026-08-08-load-probe-memory.md`` made for BiRefNet,
where dropping every reference plus ``gc.collect()`` left 71% resident.

That session ended with the app refusing its own sprite-sheet follow-up at 94%
commit, which is the correct behaviour and the wrong outcome. This is the fourth
instance of one rule -- ``bpy`` is process-global (``blender_worker``),
``HF_HUB_OFFLINE`` is read at import time (``fetch_worker``), RSS is
unreturnable (``loadprobe``) -- and the t2i loader was the one path paying it in
the process that has to keep running, where nothing but exit reclaims it.
``docs/measurements/2026-08-22-trampoline-child-pids.md`` carries the figures
and the two options not taken.

**Why persistent, and not one-shot.** A reroll, a sprite sheet and a tile sheet
are three generates against the same checkpoint, and a one-shot child would pay
the load each time. This one holds the pipe across requests and dies on
``t2i_client.Text2ImageClient.unload()`` -- which is a process kill, and so
genuinely returns the whole 21 GiB -- or with the app, via the kill-on-close job.

**What it is not.** It holds no model code of its own: it constructs the same
``text2image.Text2Image`` the app used to hold directly and calls the same
methods, so every recipe, adapter and scheduler decision stays in one place and
stays testable without a subprocess.

**The protocol.** One JSON object per stdin line in, one marked JSON line on
stdout back. Pixels travel as *files* -- ``blender_worker``'s rule, and here for
a second reason too: a 1024x1024 PNG down a 64 KB pipe would deadlock a strict
request/response exchange. ``generate`` already took an ``output_path``, so
nothing had to change to satisfy it.

    {"op": "generate", "prompt": ..., "output": ..., "seed": ..., ...}
      -> marker {"kind": "state", "text": "load"}
      -> marker {"kind": "step", "step": 3, "total": 30}
      -> marker {"kind": "done", "path": ..., "recipe": {...}, ...}
      -> marker {"kind": "error", "error": ..., "cancelled": false}

``cancel`` is the one op that arrives *while* another is being served, so stdin
is drained by a reader thread rather than by the main loop. It sets the
``threading.Event`` that ``Text2Image.generate`` already accepts as
``cancel_event``, so cancellation keeps the exact semantics it had in-process --
checked at the top of the call and once per diffusion step -- rather than
becoming "kill the child", which would throw away a warm pipe on an action the
user takes routinely.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue as _queue
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

MARKER = "@@warlock-t2i@@ "
"""Prefixed onto every response line.

diffusers, transformers and PEFT all print, and a bare progress bar on stdout
would otherwise be indistinguishable from an answer. ``main`` also points
``sys.stdout`` at stderr for the duration, so this is belt-and-braces rather
than a hope -- matting_worker's arrangement, for matting_worker's reason.
"""


def _conditioning(payload: dict[str, Any] | None) -> Any:
    """Rebuild a ``Conditioning`` from its wire form, or None.

    Not the inverse of ``Conditioning.as_dict()``: that method is the *recipe*
    rendering and deliberately drops halves that are not in play, so it cannot
    round-trip. This takes every field by name, which is what a value object
    crossing a process boundary needs.
    """
    if not payload:
        return None
    from .conditioning import Conditioning

    def _path(value: Any) -> Path | None:
        return Path(value) if value else None

    return Conditioning(
        ip_adapter=payload.get("ip_adapter"),
        ip_image=_path(payload.get("ip_image")),
        ip_scale=float(payload.get("ip_scale", 0.0)),
        control=payload.get("control"),
        control_image=_path(payload.get("control_image")),
        control_scale=float(payload.get("control_scale", 0.0)),
        control_end=float(payload.get("control_end", 0.0)),
        init_image=_path(payload.get("init_image")),
        strength=float(payload.get("strength", 0.0)),
    )


class _Server:
    """One resident pipe and the loop that serves it.

    A class rather than module globals so a test can drive a server over two
    pipes in-process, the way ``matting_worker.main`` can be called directly.
    """

    def __init__(self, base_key: str, model_root: str, model_dir: str | None) -> None:
        self.base_key = base_key
        self.model_root = Path(model_root)
        self.model_dir = Path(model_dir) if model_dir else None
        self._t2i: Any = None
        # Set *and cleared* by the reader thread, read by the diffusion step
        # callback. Both on that one thread, because the parent's write order is
        # the only authority on which job a cancel belongs to: cleared as a
        # generate is enqueued, so a cancel written after it is seen and one
        # written before it is discarded with the job it was meant for.
        #
        # Clearing inside ``op_generate`` instead looks equivalent and is not.
        # The main loop dequeues a request some time after the reader queued it,
        # and a cancel arriving in that window would be set by the reader and
        # then wiped by the handler -- losing exactly the cancel a user issues
        # immediately after starting a job, which is when they most often do.
        self.cancel = threading.Event()

    def pipe(self) -> Any:
        """The ``Text2Image``, constructed on first use.

        Deferred so startup is a Python import and not a checkpoint read: the
        parent spawns this child when it decides to *have* a pipe, which is
        earlier than the moment it needs one loaded.
        """
        if self._t2i is None:
            from .. import models
            from .text2image import Text2Image

            self._t2i = Text2Image(
                models.BASE_MODELS[self.base_key], self.model_root, self.model_dir
            )
        return self._t2i

    # --- ops -----------------------------------------------------------------

    def op_load(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        self.pipe().load(lambda text: emit({"kind": "state", "text": text}))
        return {"kind": "done", **self._vitals()}

    def op_generate(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        from .text2image import JobCancelled

        t2i = self.pipe()
        size = req.get("size")
        try:
            path = t2i.generate(
                str(req["prompt"]),
                Path(req["output"]),
                seed=int(req.get("seed", 42)),
                lora=req.get("lora"),
                lora_weight=float(req["lora_weight"]),
                negative_prompt=req.get("negative_prompt"),
                conditioning=_conditioning(req.get("conditioning")),
                on_state=lambda text: emit({"kind": "state", "text": text}),
                on_step=lambda step, total: emit(
                    {"kind": "step", "step": step, "total": total}
                ),
                cancel_event=self.cancel,
                tile=bool(req.get("tile")),
                sheet=bool(req.get("sheet")),
                scene=bool(req.get("scene")),
                tilesheet=bool(req.get("tilesheet")),
                size=(int(size[0]), int(size[1])) if size else None,
            )
        except JobCancelled:
            # Named on the wire rather than inferred from the message: the
            # parent re-raises the *same* exception type, and a cancel that
            # arrived as a generic error would be logged as a failure and shown
            # to the user as one.
            return {"kind": "error", "error": "cancelled", "cancelled": True}
        return {
            "kind": "done",
            "path": str(path),
            "prompt": t2i.last_prompt,
            "recipe": t2i.last_recipe,
            **self._vitals(),
        }

    def op_trim(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        if self._t2i is not None:
            self._t2i.trim()
        return {"kind": "done", **self._vitals()}

    def op_vram(self, req: dict[str, Any], emit: Any) -> dict[str, Any]:
        return {"kind": "done", **self._vitals()}

    def _vitals(self) -> dict[str, Any]:
        """The device readings the parent can no longer take for itself.

        ``vram.device_memory`` and ``queue.vram_gib`` both reach torch through
        ``sys.modules`` and return None when it is absent -- which, once the
        pipe lives here, is the app process's steady state. The figures are
        therefore reported rather than read: device-wide free/total (which sees
        trellis-server too, because ``mem_get_info`` wraps ``cudaMemGetInfo``)
        and this process's own allocated/reserved.
        """
        out: dict[str, Any] = {"loaded": self._t2i is not None and self._t2i.loaded}
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
        "vram": op_vram,
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


STDIN_POLL_SECONDS = 0.05
"""How often the reader thread asks whether a request has arrived.

Only reached while the pipe is empty. It bounds how late a *cancel* can be
noticed, and 50 ms is far below the duration of the diffusion step a cancel
interrupts -- while being long enough that an idle child costs nothing.
"""

_STD_INPUT_HANDLE = -10


def _peek_stdin(handle: Any) -> int:
    """Bytes already readable on the stdin pipe, or -1 if it is finished."""
    avail = wintypes.DWORD(0)
    ok = ctypes.windll.kernel32.PeekNamedPipe(
        handle, None, 0, None, ctypes.byref(avail), None
    )
    return int(avail.value) if ok else -1


def _lines_from(stdin: Any) -> Any:
    """Yield stdin lines *without ever leaving a read pending*.

    **This is not a style choice; a blocking read here deadlocks the process.**
    Measured 2026-08-22, minimally reproducible: a daemon thread parked in a
    read on the inherited stdin pipe stops the main thread's very next native
    extension import dead. `import numpy` takes 0.1 s with no such thread and
    never returns with one -- the faulthandler dump shows the main thread inside
    `_bootstrap_external.create_module`, loading numpy's `_multiarray_umath`
    DLL, indefinitely.

    Controls place the cause exactly: an *idle* second thread is fine, and a
    second thread blocked reading a *regular file* is fine. Every way of
    blocking on the pipe fails identically -- `for line in sys.stdin`,
    `readline()`, `sys.stdin.buffer.readline()` and a bare `os.read(0, ...)` --
    so it is below Python's IO layer, in the interaction between a pending
    synchronous pipe read and the Windows image loader.

    `matting_worker` never met this because it reads stdin *on its main loop*
    and imports before ever blocking. This worker cannot: a cancel has to be
    read while a generate is running, which is precisely a concurrent reader.
    So the reader peeks first and reads only what is already there, leaving no
    outstanding read for an import to trip over. Verified against the same
    reproduction: numpy and torch import at full speed, and a line written
    during the import still arrives.

    Falls back to plain iteration off Windows and for anything that is not the
    real stdin -- the in-process tests drive `serve` with a `StringIO`, which
    has no pipe to peek at and no loader to deadlock.
    """
    if sys.platform != "win32":
        yield from stdin
        return
    try:
        if stdin.fileno() != 0:
            yield from stdin
            return
    except (AttributeError, OSError, ValueError):
        yield from stdin
        return

    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    buf = b""
    while True:
        available = _peek_stdin(handle)
        if available < 0:
            # The parent closed its end: a broken pipe is how this loop is
            # meant to finish, not an error to report.
            return
        if available == 0:
            time.sleep(STDIN_POLL_SECONDS)
            continue
        try:
            chunk = os.read(0, available)
        except OSError:
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, _, buf = buf.partition(b"\n")
            yield line.decode("utf-8", "replace") + "\n"


def serve(server: _Server, stdin: Any, stdout: Any) -> int:
    """Run the request loop until ``stdin`` closes. -> 0.

    Split from ``main`` so a test can drive it over two pipes without spawning
    anything -- the same reason ``matting_worker.main`` takes an ignored argv.
    """
    requests: Any = _queue.Queue()

    def _pump() -> None:
        try:
            for raw in _lines_from(stdin):
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
                    # is one: the main thread is inside diffusers when this
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

    reader = threading.Thread(target=_pump, name="t2i-worker-stdin", daemon=True)
    reader.start()

    def emit(msg: dict[str, Any]) -> None:
        # A *leading* newline as well as a trailing one. diffusers' progress bar
        # writes carriage-returned partial lines with no newline of their own,
        # onto the same merged stream, so a response emitted mid-sample lands on
        # the end of "50%|####      | 2/4" and stops being a line that begins
        # with the marker. The parent dropped exactly those as chatter -- which
        # cost every step message of every job while letting the final answer
        # through, because the bar has finished by then. The parent matches the
        # marker anywhere in a line now; this is the other half of that pair,
        # and the cheaper half, because it keeps the log readable too.
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

    ``argv`` is ``[base_key, model_root]`` plus an optional ``model_dir`` --
    the three things ``Text2Image.__init__`` takes, with the spec looked up by
    key rather than pickled, because the registry is the same module in both
    processes and a spec that crossed the wire could disagree with it.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        sys.stderr.write("usage: text2image_worker <base_key> <model_root> [dir]\n")
        return 2
    out = sys.stdout
    # Everything diffusers and transformers print goes to the log stream rather
    # than into the middle of a response. Restored on the way out so a test that
    # calls main() twice does not stack the redirect.
    sys.stdout = sys.stderr
    try:
        server = _Server(args[0], args[1], args[2] if len(args) > 2 else None)
        return serve(server, sys.stdin, out)
    finally:
        sys.stdout = out


if __name__ == "__main__":
    raise SystemExit(main())

"""BiRefNet in a child process, one request per line.

**Why a child.** Loading BiRefNet on the CPU costs **1475 MB** of RSS on this
machine, and dropping every reference and calling ``gc.collect()`` leaves
**1053 MB** still resident, because what holds it is the allocator's arenas
rather than a live object (``docs/measurements/2026-08-08-load-probe-memory.md``).
``matting.unload()`` therefore *cannot* return it -- it cleared a dict and the
gigabyte stayed -- and only a process that ends can. This is the fourth
instance of one rule: ``bpy`` is process-global (``blender_worker``),
``HF_HUB_OFFLINE`` is read at import time (``fetch_worker``), RSS is
unreturnable (``loadprobe``), so all three are paid in a process that ends.
Warlock's worst crash to date is host-commit exhaustion; a gigabyte held for
the life of the app on behalf of a user who may never do a 2D export is exactly
the wrong trade.

**Why persistent, and not one-shot like ``loadprobe``.** ``service.derive``
mattes one artifact per call and an asset's icon, sprite set and pixel set are
three calls, so a one-shot child would pay the ~12 s load three times for one
asset. This one holds the loaded model across requests and dies on
``matting.unload()`` -- which is now a kill, and so genuinely returns all
1475 MB -- or with the app, via the kill-on-close job.

**The protocol.** One JSON object per stdin line in, one marked JSON line on
stdout back:

    {"model_dir": ..., "device": "cpu", "input": <png>, "output": <png>}
    -> @@warlock-matte@@ {"ok": true}
    -> @@warlock-matte@@ {"ok": false, "stage": "load"|"run", "error": "..."}

Pixels travel as *files*, never down the pipe -- ``blender_worker``'s rule, and
here for a second reason too: a 4096x4096 matte is 16 MB and the OS pipe buffer
is 64 KB, so a payload that size deadlocks a strict request/response exchange.
The mask is written as an 8-bit PNG, 0 or 255, at the input's own size.

``stage`` is what the parent's failure memo keys on: a checkpoint that will not
load will not load again and must be remembered (an export is a loop over
images), while one image that failed to matte says nothing about the next.

The marker exists because ``transformers`` and BiRefNet's own modelling code
both print. ``main`` also points ``sys.stdout`` at stderr for the duration, so a
stray print is merged into the log stream the parent drains rather than landing
in the middle of a response; the marker is what makes that belt-and-braces
rather than a hope.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Prefixed onto every response line. Nothing else this process writes carries
# it, so the parent can tell an answer from library chatter without parsing.
MARKER = "@@warlock-matte@@ "


def handle(req: dict[str, Any]) -> dict[str, Any]:
    """One matte. -> the response object. Never raises.

    A failure that escaped here would kill the child mid-export and lose the
    loaded model with it; every one is a sentence instead, and the parent's
    caller (``matting.mask``) turns any of them into the corner flood fill.
    """
    from . import matting

    try:
        model_dir = Path(req["model_dir"])
        device = str(req.get("device") or "cpu")
        source = Path(req["input"])
        dest = Path(req["output"])
    except (KeyError, TypeError) as exc:
        return {"ok": False, "stage": "run", "error": f"malformed request: {exc}"}

    try:
        # matting._load, not a second from_pretrained: it owns the fp16->fp32
        # CPU cast, the trust_remote_code flag and the per-(path, device) cache
        # that makes this child worth keeping alive between requests.
        model = matting._load(model_dir, device)
    except Exception as exc:  # noqa: BLE001 -- the whole point is to report it
        # The type as well as the message: "No module named 'einops'" and a
        # shape mismatch are two entirely different repairs.
        return {"ok": False, "stage": "load", "error": f"{type(exc).__name__}: {exc}"}

    try:
        import numpy as np
        from PIL import Image

        with Image.open(source) as im:
            im.load()
            found = matting._infer(im, model)
        Image.fromarray((np.asarray(found) * 255).astype("uint8"), "L").save(dest, "PNG")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "run", "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True}


def main(argv: list[str] | None = None) -> int:
    """Serve until stdin closes. Returns 0.

    ``argv`` is accepted and ignored, for ``loadprobe.main``'s shape -- it is
    what lets a test drive the loop in-process.
    """
    out = sys.stdout
    # Everything the model's own code prints goes to the log stream instead of
    # into the middle of a response. Restored on the way out so a test that
    # calls main() twice does not stack the redirect.
    sys.stdout = sys.stderr
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError as exc:
                resp: dict[str, Any] = {
                    "ok": False, "stage": "run", "error": f"unreadable request: {exc}"
                }
            else:
                resp = handle(req)
            out.write(MARKER + json.dumps(resp) + "\n")
            out.flush()
    finally:
        sys.stdout = out
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

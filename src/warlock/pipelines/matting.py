"""One boolean mask per image, from a model when there is one.

Three sources, in a fixed order of preference, and the order is the design:

* an alpha channel the image already has *and actually uses* -- ground truth,
  and a model asked to re-cut an existing cutout can only make it worse;
* BiRefNet, if its weights are on disk;
* the corner flood fill in ``pipelines/reference.py``, which needs nothing.

The fallback is what makes every 2D export work on a fresh checkout, and it is
never skipped on failure: a corrupt checkpoint, an out-of-memory, or a matte
that comes back empty all fall through to the flood fill rather than raising.
Missing weights cost the user edge quality, never the export.

The model is loaded on first use, kept on the CPU, and dropped by ``unload()``
-- never moved to the GPU by default. That is the same trade
``text2image._conditioned`` makes for the ControlNet, in the other direction:
this runs beside a resident trellis and a resident SDXL pipe, and a matting
model holding VRAM would take room from the models that are actually producing
the asset. A second or two of host compute per export is the cheaper half.

**And it is loaded in a child process** (``matting_worker``), which is what
finally makes ``unload()`` mean what its name says: loading BiRefNet costs
1475 MB of RSS and leaves 1053 MB resident after every reference is dropped and
``gc.collect()`` has run, because the allocator's arenas hold it. That module's
docstring carries the measurement and the argument; what matters here is the
shape of the boundary. It is inside ``_model_mask``, not at ``mask()``: the
three-source preference order, the never-fatal fallback and every caller
(``service.derive``, ``service.matte``) are untouched, and a spawn failure, a
timeout or a non-zero exit is an exception ``mask`` already catches and answers
with the corner flood fill.

Nothing here is imported at module level that a fresh checkout might not have:
torch, numpy and Pillow all load inside the functions that need them, so
``available()`` -- the question every caller asks first -- is answerable on a
machine with no image extra installed at all.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue as _queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import models, winjob
from .reference import has_alpha, subject_mask

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

log = logging.getLogger(__name__)

# What counts as subject in the model's probability map. 0.5 is the threshold
# BiRefNet's own demo uses; the map is confidently bimodal on a plain
# background, so the exact value only matters on the soft rim.
THRESHOLD = 0.5

# The model's expected input. It is fully convolutional, but it was trained at
# this size and a matte taken at the source resolution is measurably worse at
# the same cost.
INPUT_SIZE = 1024

# ImageNet statistics, which is what the published preprocessing normalises
# with. Written out here rather than pulled in through torchvision: this is
# three lines of arithmetic, and a dependency the project does not otherwise
# have would be a strange price for them.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# Below this, an alpha channel is carrying a cutout; at or above it every pixel
# is opaque and the channel says nothing. Not 255, because a saved cutout's rim
# is antialiased and a re-encode can leave the interior a step or two short of
# full. Having a channel is not the same as using one: half the tools in the
# world write RGBA unconditionally, and trusting an opaque one would hand every
# such upload back as a full-frame "cutout" labelled ground truth.
_OPAQUE = 250

# How long one request to the child may take. Generous on purpose: the *first*
# request pays the load as well as the inference, and the real checkpoint
# measured 11.5 s a frame at float32 on this machine on top of a ~12 s load. A
# timeout kills the child and falls through to the flood fill, so the cost of
# being wrong here is a worse-looking matte, not a failed export.
REQUEST_TIMEOUT = 180.0

# How long the child gets to die politely when ``unload`` asks.
STOP_TIMEOUT = 5.0


def model_dir(config: Any = None) -> Path:
    from ..config import get_config

    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = (config or get_config()).t2i_model_root
    return Path(root) / spec.dir_name


def available(config: Any = None) -> bool:
    """Whether the weights are on disk. Checked before any torch import, which
    is the ordering tests/test_offline.py requires everywhere."""
    return (model_dir(config) / "config.json").exists()


def mask(image: PILImage, config: Any = None, *, device: str = "cpu") -> tuple[Any, str]:
    """-> (boolean mask, which source produced it).

    The source is returned rather than logged because it goes in the manifest:
    an exported set whose alpha came from the flood fill has visibly rougher
    edges than one that did not, and that is a fact about the files.
    """
    if is_cutout(image):
        return (subject_mask(image), "alpha")
    # An alpha channel that says nothing must not be read by the fill either:
    # subject_mask keys on the *presence* of the channel, so it would take the
    # same branch and hand back the same all-true mask. Dropping it here is
    # what makes the fallback mean what its name says.
    flat = image.convert("RGB") if has_alpha(image) else image
    if available(config):
        try:
            found = _model_mask(flat, model_dir(config), device)
            if found is not None and found.any():
                return (found, models.DEFAULT_MATTING)
            log.warning("the matting model found no subject; falling back to the fill")
        except _AlreadyFailed as exc:
            # Decided once, reported once per image and without the traceback:
            # an export is a loop over images, and the same stack fifty times
            # buries every other line in the log.
            log.warning("%s; using the corner fill", exc)
        except Exception:
            # Never fatal. The flood fill is worse-looking and always right
            # enough to produce a file, which beats a failed export.
            log.exception("matting failed; falling back to the corner fill")
    return (subject_mask(flat), "flood")


def is_cutout(image: PILImage) -> bool:
    """Whether the image's alpha channel is a matte somebody already made.

    Exported rather than left private for the reason ``has_alpha`` was: a
    second caller now asks the same question about the same pixels.
    ``service.matte.approve`` records ``matte: approved`` on exactly this
    condition, and a copy of it that drifted would either claim an approval for
    an opaque upload or throw away a real one.

    Having the channel is not the same as using it. A PNG saved by almost any
    editor is RGBA whether or not anything in it is transparent, and treating a
    fully opaque one as ground truth returns the whole frame as subject --
    labelled ``"alpha"``, which is the manifest's way of saying "this boundary
    is exact". So the channel has to contain at least one non-opaque pixel
    before it is believed.
    """
    import numpy as np

    if not has_alpha(image):
        return False
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    return bool(alpha.min() < _OPAQUE)


def unload() -> None:
    """Drop the loaded model, and any memory of one that would not load.

    The counterpart to ``_load``: a model is held for as long as the process
    lives, which is right while exports keep coming and pure waste once the
    user has moved on, so the release is a call somebody can make rather than a
    decision baked into the load. It also forgets the failure sentinel, so a
    user who repairs a half-downloaded checkpoint gets another attempt without
    restarting the app.

    It now keeps that promise. It used to clear a dict, which returned about a
    third of what the load cost; killing the child returns all 1475 MB. Called
    from ``queue.Worker._maybe_evict_caches`` past the idle timeout -- through
    ``asyncio.to_thread``, because it is a process kill now and blocks.

    ``_cache`` is still cleared for the case where this module *is* the child
    (``matting_worker`` calls ``_load`` in its own process) and because the
    parent keeps its failure sentinels there.
    """
    global _last_error

    _stop_child()
    _cache.clear()
    _last_error = None


def last_error() -> str | None:
    """The words of the most recent failed load, or None.

    ``_FAILED`` is enough to stop a second attempt but it is an anonymous
    object: a reader could learn from it that matting is broken and never how.
    ``doctor`` runs in this process, so a module-level string is the whole
    mechanism needed -- no IPC, no file -- and it is what turns a green
    weights row above a silent fall-back to the flood fill into a red one that
    names the ModuleNotFoundError. Reset by ``unload`` with the sentinel it
    describes, or repairing an install leaves doctor reporting a failure that
    no longer happens.
    """
    return _last_error


class _AlreadyFailed(RuntimeError):
    """This checkpoint was tried once this session and could not be loaded."""


# The sentinel a failed load leaves behind, in the same dict as the model so
# that clearing one clears the other.
_FAILED = object()

_cache: dict[str, Any] = {}

_last_error: str | None = None


def _load(path: Path, device: str):
    """The BiRefNet at ``path``, built from *vendored* code (MDL-03).

    This used to be ``AutoModelForImageSegmentation.from_pretrained(...,
    trust_remote_code=True)``, which executes whatever ``birefnet.py`` the
    snapshot on disk happens to hold -- in this process, with the user's
    filesystem and network permissions. Pinning the revision narrowed that to
    "whatever that commit shipped"; the modelling code lives in
    ``pipelines/birefnet/`` now, so what runs is what is in this checkout and a
    reviewer can read it.

    The failure memo and its ``_last_error`` string are unchanged, and matter
    slightly more: the loader is stricter now (see ``birefnet.load``), so the
    new way to fail is a state dict that does not map onto the vendored
    architecture -- which is exactly the drift worth reporting loudly.
    """
    global _last_error

    from . import birefnet

    key = f"{path}|{device}"
    hit = _cache.get(key)
    if hit is _FAILED:
        # A checkpoint that cannot load cannot load again, and from_pretrained
        # is seconds of work per attempt. An export is a loop over images, so
        # without this the first broken install costs the whole batch.
        raise _AlreadyFailed(f"the matting model at {path} failed to load earlier this session")
    if hit is not None:
        return hit
    try:
        # ``strict=True`` inside, which is a parity assertion rather than a
        # setting: a checkpoint that does not map exactly onto the vendored
        # architecture would otherwise load with a layer left randomly
        # initialised, and a BiRefNet with a random decoder block does not
        # fail -- it returns a plausible, wrong mask.
        model = birefnet.load(path)
        model = model.to(device)
        if device == "cpu":
            # The published checkpoint stores fp16 weights, and half precision
            # on the CPU is emulated rather than native: the real BiRefNet
            # measured 73 s a frame at fp16 against 11.5 s at fp32 on this
            # machine. This module's whole bargain is "a second or two of host
            # compute per export instead of VRAM taken from the models
            # producing the asset", and only the cast keeps it. A CUDA device
            # is deliberately left alone -- there half is correct *and* faster.
            model = model.float()
    except Exception as exc:
        _cache[key] = _FAILED
        # The type as well as the message: "No module named 'einops'" and a
        # shape mismatch are two entirely different repairs, and a bare str()
        # of an ImportError names neither.
        _last_error = f"{type(exc).__name__}: {exc}"
        raise
    _cache[key] = model
    _last_error = None
    return model


# --- the child ---------------------------------------------------------------
#
# One process, one lock, and the request/response exchange serialised under it:
# service/matte.py and service/derive.py both run on TaskRunner's four threads
# and there is exactly one child.

CHILD_ARGV = [sys.executable, "-m", "warlock.pipelines.matting_worker"]
"""How the child is launched. A module constant rather than a literal inside
``_start_child`` so a test can substitute a scripted child and exercise the
*real* spawn, pipe, marker filtering and PNG round trip -- the parts a
patched-out ``_request`` would never reach."""

_proc: Any = None
_proc_key: str | None = None
_lines: Any = None
_child_lock = threading.Lock()


class _ChildFailed(RuntimeError):
    """The child could not be reached, or answered with a failure."""

    def __init__(self, message: str, *, stage: str = "run") -> None:
        super().__init__(message)
        self.stage = stage


def _stop_child() -> None:
    """Kill the child, if there is one. Idempotent and thread-safe."""
    with _child_lock:
        _reset_child_locked()


def _reset_child_locked() -> None:
    """Forget and kill the current child. Caller holds ``_child_lock``.

    Under the lock rather than beside it, because ``_request`` reaches this on
    four different failures while it is already inside the lock, and a second
    acquisition there would deadlock on a plain ``Lock``.
    """
    global _proc, _proc_key, _lines

    proc, _proc, _proc_key, _lines = _proc, None, None, None
    if proc is None:
        return
    if proc.poll() is None:
        log.info("stopping the matting worker (pid %s)", proc.pid)
        proc.kill()
        try:
            proc.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            # Nothing useful left to do: it is in the kill-on-close job, so it
            # cannot outlive the app, and this is an advisory release.
            log.warning("the matting worker (pid %s) did not exit", proc.pid)
    # Forgotten from winjob's registry as well as from the module globals:
    # entries are removed on reap by contract, or terminate_tracked later
    # opens a recycled pid with PROCESS_TERMINATE. Unconditional, like the
    # forget above -- a child that ignored the kill is being abandoned to the
    # kill-on-close job either way, and a second TerminateProcess adds nothing.
    winjob.untrack(proc.pid)
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()


def _start_child(key: str) -> None:
    """Spawn the worker. Caller holds ``_child_lock``."""
    global _proc, _proc_key, _lines

    # stderr is merged into stdout rather than DEVNULL'd: the reader thread
    # below drains it either way, so keeping it costs nothing and a failed
    # `import einops` is otherwise invisible. Merged rather than given a second
    # pipe, because an undrained one deadlocks the child once it fills.
    #
    # winjob.assign follows immediately, as it must: this child holds well over
    # a gigabyte, and the window between Popen and that call is the only one in
    # which a parent crash can still orphan it.
    proc = subprocess.Popen(
        list(CHILD_ARGV),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    winjob.assign(proc.pid)
    # Tracked as well as assigned. The job object kills it when *this* process
    # dies; the registry is what lets a shutdown that leaves the interpreter
    # alive stop it too (MDL-02).
    winjob.track(proc.pid, "birefnet matting")
    lines: Any = _queue.Queue()

    def _pump(stream: Any) -> None:
        try:
            for raw in stream:
                lines.put(raw)
        finally:
            lines.put(None)

    reader = threading.Thread(
        target=_pump, args=(proc.stdout,), name="matting-worker", daemon=True
    )
    reader.start()
    _proc, _proc_key, _lines = proc, key, lines


def _request(payload: dict[str, Any], key: str) -> None:
    """Send one request and wait for its answer. Raises ``_ChildFailed``.

    Serialised end to end under ``_child_lock``: the protocol is one response
    per request with nothing to correlate them by, so two callers interleaving
    would each read the other's answer.
    """
    from .matting_worker import MARKER

    with _child_lock:
        # No child, one that died under us, or one holding a different
        # checkpoint: in every case this is not the child the request wants.
        if _proc is None or _proc.poll() is not None or _proc_key != key:
            _reset_child_locked()
            try:
                _start_child(key)
            except OSError as exc:
                raise _ChildFailed(
                    f"could not start the matting worker: {exc}", stage="load"
                ) from exc
        proc, lines = _proc, _lines
        assert proc is not None and proc.stdin is not None
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            _reset_child_locked()
            raise _ChildFailed(f"the matting worker went away: {exc}") from exc

        deadline = time.monotonic() + REQUEST_TIMEOUT
        # Every *named* failure below resets the child itself, because each one
        # knows it has left the protocol mid-exchange. This catch is for the
        # ones with no name: a cancellation or a shutdown arriving while this
        # thread is parked in ``lines.get``. The request has been written and
        # its answer has not been read, so the child is not merely idle -- it
        # is one unread line ahead, and the *next* caller would take that line
        # as its own answer and matte an image against another image's mask.
        # Killing it is the only way back to a known state.
        #
        # Not a bare ``finally`` -- the analogous guard in
        # ``rigging.run_worker`` is one because its child is one-shot, and this
        # child is persistent by design: reaping it on the success path would
        # pay the model load again on every image.
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _reset_child_locked()
                    raise _ChildFailed(
                        f"the matting worker did not answer in {REQUEST_TIMEOUT:.0f}s"
                    )
                try:
                    raw = lines.get(timeout=remaining)
                except _queue.Empty:
                    continue
                if raw is None:
                    _reset_child_locked()
                    raise _ChildFailed("the matting worker exited without answering")
                if not raw.startswith(MARKER):
                    # Library chatter. BiRefNet's own modelling code prints, and
                    # so does transformers; the marker tells the two apart.
                    log.debug("matting worker: %s", raw.rstrip())
                    continue
                try:
                    resp = json.loads(raw[len(MARKER) :])
                except ValueError as exc:
                    _reset_child_locked()
                    raise _ChildFailed(
                        f"unreadable answer from the worker: {exc}"
                    ) from exc
                if resp.get("ok"):
                    return
                raise _ChildFailed(
                    str(resp.get("error") or "the matting worker failed"),
                    stage=str(resp.get("stage") or "run"),
                )
        except _ChildFailed:
            # Already handled: either the child was reset above, or this is the
            # worker's own reported failure, which leaves the protocol in step
            # and the child fit for the next request.
            raise
        except BaseException:
            _reset_child_locked()
            raise


def _model_mask(image: PILImage, path: Path, device: str):
    """The model half, run in the child. -> a boolean mask.

    Isolated so the fallback logic in ``mask`` can be tested without weights by
    patching exactly this -- which is unchanged, and is also why the process
    boundary is here rather than at ``mask``.

    The failure memo (``_FAILED`` / ``_last_error``) is kept in *this* process
    on purpose. It has to outlive the child's death, or a broken install costs
    one spawn and one ~12 s load attempt per image in an export loop, which is
    the exact cost ``_AlreadyFailed`` was introduced to stop. Only a
    ``stage == "load"`` failure memoizes: a checkpoint that will not load will
    not load again, while one image that failed to matte says nothing about the
    next.
    """
    global _last_error

    import numpy as np
    from PIL import Image

    key = f"{path}|{device}"
    if _cache.get(key) is _FAILED:
        raise _AlreadyFailed(
            f"the matting model at {path} failed to load earlier this session"
        )
    with tempfile.TemporaryDirectory(prefix="warlock-matte-") as tmp:
        scratch = Path(tmp)
        source = scratch / "in.png"
        dest = scratch / "mask.png"
        image.convert("RGB").save(source, "PNG")
        try:
            _request(
                {
                    "model_dir": str(path),
                    "device": device,
                    "input": str(source),
                    "output": str(dest),
                },
                key,
            )
        except _ChildFailed as exc:
            if exc.stage == "load":
                _cache[key] = _FAILED
                _last_error = str(exc)
            raise
        with Image.open(dest) as opened:
            opened.load()
            return np.asarray(opened.convert("L")) > 127


def _infer(image: PILImage, model: Any):
    """The arithmetic half: preprocess, run ``model``, threshold. -> bool mask.

    Split out of ``_model_mask`` when that became the IPC half, and it runs
    **in the child**. Keeping it here rather than in ``matting_worker`` keeps
    the preprocessing beside the constants it reads and beside the docstring
    that argues for them -- the worker imports it, exactly as
    ``blender_worker`` imports ``rigging.fit_template`` so the two halves can
    never disagree.
    """
    import numpy as np
    import torch
    from PIL import Image

    # Pillow and NumPy do what torchvision's Resize/ToTensor/Normalize do,
    # which is the whole of the published preprocessing: bilinear to the
    # trained size, scale to 0..1, subtract the ImageNet statistics.
    small = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    array = np.asarray(small, dtype=np.float32) / 255.0
    array = (array - np.asarray(_MEAN, dtype=np.float32)) / np.asarray(_STD, dtype=np.float32)
    # The dtype comes off the model rather than being assumed. The arithmetic
    # above is numpy's, i.e. float32, while the published checkpoint's weights
    # are fp16 -- and the two met at the first conv as "Input type (float) and
    # bias type (struct c10::Half)", raised inside mask()'s blanket except, so
    # every export on a host that *had* the weights fell back to the corner
    # fill and looked exactly like a host that did not.
    #
    # The *device* comes off the model for the same reason and one more: this
    # no longer takes a device argument, because it runs in the child where the
    # only thing that knows where the model was put is the model.
    parameter = next(model.parameters())
    tensor = (
        torch.from_numpy(array.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(device=parameter.device, dtype=parameter.dtype)
    )
    with torch.no_grad():
        # The published model returns a list of maps at descending scales; the
        # last is the full-resolution one, which is what its own demo reads.
        out = model(tensor)[-1].sigmoid().float().cpu()
    probability = out[0].squeeze()
    resized = Image.fromarray((probability.numpy() * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    return np.asarray(resized) > int(THRESHOLD * 255)

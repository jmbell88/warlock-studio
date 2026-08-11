"""Preflight checks: what's missing before you waste two minutes on a GPU job."""

from __future__ import annotations

import importlib.util
import shutil
import socket
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import fetch, guidance, models, native, rigging, vram, winjob
from .config import Config

# Cheap: matting itself imports nothing heavier than models and reference, both
# of which keep torch, cv2 and numpy inside the functions that need them. It is
# imported for last_error() alone -- doctor runs in this process, so the answer
# is a module attribute rather than anything that has to be carried.
from .pipelines import matting

MIN_FREE_DISK_GB = 5.0

# Importing bpy costs seconds and its answer cannot change while this process
# lives, so the probe runs once. The header's health poll re-runs these.
_blender: Check | None = None

# The row the four volatile checks are spliced in after: the last of the
# install rows, named rather than counted. The order used to be a positional
# slice of ``static_checks``' result, which meant adding an install row silently
# pushed it below the volatile block -- a misordering no test could see. Falling
# back to "volatile rows last" if the marker ever goes missing keeps a renamed
# check from dropping rows entirely.
VOLATILE_AFTER = "CUDA"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool


def run_checks(
    config: Config,
    *,
    trellis_running: bool = False,
    static: list[Check] | None = None,
    probe_slow: bool = True,
) -> list[Check]:
    """``trellis_running`` says the port is *ours*.

    Without it the port check reports a permanent false warning for the whole
    life of a warm process: the health poll runs these while trellis-server is
    resident and holding the port it is supposed to hold.

    ``static`` reuses a previous :func:`static_checks` result so a poller only
    pays for the volatile rows (C31); ``probe_slow=False`` skips the two
    genuinely slow probes -- the torch import and the bpy subprocess -- and
    reports them as still-checking rows instead (C29/C30). Startup uses both;
    the header health poll re-runs the slow probes once, off the frame thread,
    and their answers are cached from then on.
    """
    s = static_checks(config, probe_slow=probe_slow) if static is None else list(static)
    v = volatile_checks(config, trellis_running)
    # The historical display order: the install rows, then the four volatile
    # rows, then the per-model rows, then Blender (last in s). The seam is found
    # by name (VOLATILE_AFTER) rather than by a slice index, which was coupled
    # to how many install rows static_checks happened to return.
    cut = next((i + 1 for i, c in enumerate(s) if c.name == VOLATILE_AFTER), len(s))
    return [*s[:cut], *v, *s[cut:]]


def static_checks(config: Config, *, probe_slow: bool = True) -> list[Check]:
    """The rows that cannot change without the disk (or the venv) changing.

    Recomputed only on startup and on ``force`` -- a finished download is the
    one event that invalidates them, and its handler passes force.
    """
    return [
        _exe_check(config),
        _gguf_check(config),
        _birefnet_check(config),
        _gltfpack_check(config),
        _warlockc_check(),
        _cuda_check(probe=probe_slow),
        *_t2i_checks(config),
        *_matting_checks(config, probe_slow=probe_slow),
        *_pose_checks(config, probe_slow=probe_slow),
        blender_check(probe=probe_slow),
    ]


def volatile_checks(config: Config, trellis_running: bool = False) -> list[Check]:
    """The rows worth re-asking every poll: the card, the job object, the
    disk and the port are the four answers that change while the app runs."""
    return [
        _vram_check(config),
        _job_object_check(),
        _disk_check(config),
        _port_check(config, trellis_running),
    ]


BPY_PROBE_TIMEOUT = 120.0
BPY_INSTALL_HINT = "rigging unavailable; install with: uv sync --extra rig"


_blender_lock = threading.Lock()
# What a caller that declines to wait gets before the probe has run. ok=True
# because fatal-ness and the amber dot key on ok, and "still checking" is not a
# failure; the row flips to the real answer when the deferred probe lands.
_BLENDER_PENDING = Check(
    "Blender (rigging)", True,
    "still checking in the background -- rig controls appear when it finishes",
    fatal=False,
)


def blender_check(*, probe: bool = True) -> Check:
    """Can we rig? Probed in a subprocess, for the same reason rigging is.

    Non-fatal by design: bpy is an optional extra with cp313-only wheels, and
    a machine without it should generate meshes exactly as before with the
    rig/pose controls hidden -- the same way a missing image model degrades.

    ``probe=False`` never blocks: it returns the cached answer or a pending
    row (C30 -- the probe costs seconds and used to run inside startup). The
    lock keeps a deferred probe and an eager caller from racing two
    subprocesses; the answer cannot change while this process lives, so the
    first probe's result is everyone's.
    """
    global _blender
    if _blender is not None:
        return _blender
    if not probe:
        return _BLENDER_PENDING
    with _blender_lock:
        if _blender is None:
            _blender = _probe_blender()
        return _blender


def _probe_blender() -> Check:
    # Any template at all, not a hardcoded pair: templates are files, adding one
    # is the supported way to add a skeleton, and naming two of them here made
    # renaming or removing either a silent rigging outage.
    if not rigging.templates():
        return Check(
            "Blender (rigging)", False,
            f"no skeleton templates found in {rigging.TEMPLATE_DIR}",
            fatal=False,
        )
    try:
        # winjob.run, not subprocess.run: this fires during Runtime._start and
        # `import bpy` takes seconds, so killing Warlock while it starts used
        # to strand a python.exe mid-import.
        proc = winjob.run(
            [sys.executable, "-c", "import bpy; print(bpy.app.version_string)"],
            capture_output=True, text=True, timeout=BPY_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("Blender (rigging)", False, f"bpy probe failed: {exc}", fatal=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["import failed"]
        return Check("Blender (rigging)", False, f"{detail[0]} -- {BPY_INSTALL_HINT}", fatal=False)
    return Check("Blender (rigging)", True, f"bpy {proc.stdout.strip()}", fatal=False)


# The remedies for the only two fatal rows (F54). Every non-fatal model row has
# carried its exact ``hf download`` line since it was written; the two rows that
# actually stop the app said "not found at <path>" and stopped -- so the two
# failures a first run is most likely to hit were the two with no way forward.
#
# They are different *kinds* of remedy, which is why neither is a Fetch entry.
# The exe is a third-party release zip unpacked by hand (a fetcher would have to
# know how to unzip a GitHub release, and ``fetch_worker`` speaks one protocol,
# to one host); the GGUF weights are an ordinary ``hf download`` that is
# deliberately not in ``models.FETCHES`` because the app is unusable without
# them, so they belong in the install instructions rather than behind a button
# in a pane that cannot be reached until the app starts.
TRELLIS_EXE_HINT = (
    "download trellis-cuda-windows-x64.zip from "
    "https://github.com/pwilkin/trellis.cpp/releases and unpack it there "
    "(vendored build: v0.5.4), or point WARLOCK_TRELLIS_EXE at your own copy"
)
TRELLIS_GGUF_HINT = (
    'uvx hf download ilintar/trellis2-gguf --include "*.gguf" '
    '--exclude "q4/*" --exclude "q8/*" --local-dir models/trellis2-gguf'
)


def _exe_check(config: Config) -> Check:
    ok = config.trellis_server_exe.exists()
    detail = (
        str(config.trellis_server_exe)
        if ok
        else f"not found at {config.trellis_server_exe} -- {TRELLIS_EXE_HINT}"
    )
    return Check("trellis-server.exe", ok, detail, fatal=True)


def _gguf_check(config: Config) -> Check:
    ok = config.trellis_models_dir.exists() and any(config.trellis_models_dir.glob("*.gguf"))
    detail = (
        str(config.trellis_models_dir)
        if ok
        else f"no *.gguf found in {config.trellis_models_dir} -- download with:\n"
        f"  {TRELLIS_GGUF_HINT}"
    )
    return Check("TRELLIS GGUF weights", ok, detail, fatal=True)


def _birefnet_check(config: Config) -> Check:
    # The filename from guidance, not a second copy of it: guidance gates the
    # bg_removal default on this exact file, so a drifted spelling here would
    # report the weights missing while the app quietly kept asking for them.
    path = config.trellis_models_dir / guidance.BIREFNET_WEIGHTS
    ok = path.exists()
    detail = (
        str(path)
        if ok
        else f"missing at {path} -- background matting falls back to a threshold cutout"
    )
    # Named for the process that loads it: there is a second BiRefNet on the
    # host now (see _matting_checks) and the two are different downloads.
    return Check("trellis: birefnet.gguf (background removal)", ok, detail, fatal=False)


def _gltfpack_check(config: Config) -> Check:
    ok = config.gltfpack_exe.exists()
    detail = (
        str(config.gltfpack_exe)
        if ok
        else f"not found at {config.gltfpack_exe} -- meshes ship at full reconstruction density"
    )
    return Check("gltfpack (mesh optimizer)", ok, detail, fatal=False)


def _warlockc_check() -> Check:
    """The native kernels: built locally, optional, and *visibly* optional.

    Non-fatal for the reason gltfpack is: what it buys is speed, and every
    caller has a numpy path it falls back to. Worth a row anyway -- vendor/ is
    gitignored, so "the audit got slower after I moved machines" has exactly
    one cause and no other way to see it.
    """
    ok, detail = native.status()
    return Check("warlockc (native kernels)", ok, detail, fatal=False)


def _cuda_check(*, probe: bool = True) -> Check:
    """``probe=False`` refuses to *import* torch (seconds of module init, and
    this used to run inside startup -- C29). If torch is already loaded the
    answer is two attribute reads and is given regardless."""
    if not probe and sys.modules.get("torch") is None:
        return Check(
            "CUDA", True,
            "still checking in the background (importing torch takes a moment)",
            fatal=False,
        )
    try:
        import torch
    except ImportError:
        return Check(
            "CUDA", False, "torch not installed (uv sync --extra text2image)", fatal=False
        )
    ok = torch.cuda.is_available()
    detail = "available" if ok else "torch.cuda.is_available() is False"
    return Check("CUDA", ok, detail, fatal=False)


def _vram_check(config: Config) -> Check:
    """How large the card is, which mode that chose, and what a job may ask for.

    Fatal only when the budget cannot hold even a lone trellis run: every 3D
    job on such a host fails anyway, and it is better to say so at startup than
    two minutes into the first reconstruction.
    """
    device = vram.probe()
    resolved = vram.plan(
        exclusive=config.vram_exclusive,
        budget_gib=config.vram_budget_gib,
        total_gib=config.vram_total_gib,
        device=device,
        explicit=config.vram_exclusive_explicit,
    )
    if not resolved.enforced:
        return Check("VRAM budget", True, resolved.reason, fatal=False)
    # Only when the plan is actually about this device: with WARLOCK_VRAM_TOTAL
    # standing in for a card, the real card's free figure describes something
    # else entirely and reads as a contradiction.
    measured = device is not None and resolved.total_gib == device.total_gib
    free = f", {device.free_gib:.1f} GiB free now" if measured else ""  # type: ignore[union-attr]
    ok = resolved.budget_gib is not None and resolved.budget_gib >= vram.TRELLIS_GIB
    detail = resolved.reason + free
    if not ok:
        detail += f" -- below the {vram.TRELLIS_GIB:.0f} GiB a reconstruction needs"
    return Check("VRAM budget", ok, detail, fatal=not ok)


def _job_object_check() -> Check:
    """Is kill-on-close actually armed?

    Non-fatal -- a host without job objects is the status quo ante -- but it
    must be *visible*: the whole point of the guarantee is that GPU children
    die with the app, and silently not having it looks identical to having it
    right up until a crash strands trellis-server on port 17971.
    """
    ok = winjob.armed()
    detail = (
        "child processes die with Warlock (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)"
        if ok
        else "unavailable -- trellis-server/Blender can outlive a crash of this process"
    )
    return Check("subprocess cleanup", ok, detail, fatal=False)


def _disk_check(config: Config) -> Check:
    free_gb = shutil.disk_usage(config.data_dir).free / (1024**3)
    ok = free_gb >= MIN_FREE_DISK_GB
    return Check("free disk space", ok, f"{free_gb:.1f} GB free in {config.data_dir}", fatal=False)


def _port_check(config: Config, trellis_running: bool = False) -> Check:
    if trellis_running:
        return Check(
            "trellis port", True, f"port {config.trellis_port} held by trellis-server", fatal=False
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", config.trellis_port))
            ok, detail = True, f"port {config.trellis_port} is free"
        except OSError as exc:
            ok, detail = False, f"port {config.trellis_port} unavailable: {exc}"
    return Check("trellis port", ok, detail, fatal=False)


# The path rule and every presence probe live in warlock.fetch now, because the
# download button has to answer exactly the same questions and a second copy of
# "where does turbo live" is how the two would come to disagree. What stays here
# is the *wording*: a Check's detail is a sentence for a person, and that is
# doctor's business rather than the planner's.
_base_model_dir = fetch.base_model_dir


def _t2i_checks(config: Config) -> list[Check]:
    """One row per registry entry, all non-fatal.

    Every image model is an optional manual download and only the job that
    picks it cares, so a missing one is a note with its exact command rather
    than something that blocks startup. Listing them individually is the point:
    a single 'weights' row can't tell you *which* of five downloads you skipped.
    """
    checks: list[Check] = []
    for spec in models.BASE_MODELS.values():
        path = _base_model_dir(config, spec)
        ok, missing_lora = fetch.base_model_state(config, spec)
        if ok:
            detail = str(path)
        elif missing_lora is not None:
            detail = (
                f"weights present, but {spec.base_lora} is missing at "
                f"{missing_lora} -- this model cannot run without it; "
                f"download with:\n  {spec.download}"
            )
        else:
            detail = f"not found at {path} -- unavailable; download with:\n  {spec.download}"
        checks.append(Check(fetch.check_name("base", spec.label), ok, detail, fatal=False))
    for lora in models.STYLE_LORAS.values():
        path = config.t2i_model_root / "loras" / lora.filename
        ok = fetch.present(config, "lora", lora)
        detail = (
            str(path)
            if ok
            else f"not found at {path} -- style unavailable; download with:\n  {lora.download}"
        )
        checks.append(Check(fetch.check_name("lora", lora.label), ok, detail, fatal=False))
    for adapter in models.IP_ADAPTERS.values():
        root = config.t2i_model_root / adapter.dir_name
        weights = root / adapter.subfolder / adapter.weight_name
        # Both halves, deliberately: weights without the CLIP vision encoder
        # load fine and then fail at the first call, which is not a failure a
        # user can read back to a missing download.
        ok = fetch.present(config, "adapter", adapter)
        if ok:
            detail = str(root)
        else:
            missing = "weights" if not weights.exists() else "CLIP vision encoder"
            detail = (
                f"{missing} not found under {root} -- conditioning unavailable; "
                f"download with:\n  {adapter.download}"
            )
        checks.append(Check(fetch.check_name("adapter", adapter.label), ok, detail, fatal=False))
    for cn in models.CONTROLNETS.values():
        path = config.t2i_model_root / cn.dir_name
        ok = fetch.present(config, "control", cn)
        detail = (
            str(path)
            if ok
            else f"not found at {path} -- control unavailable; download with:\n  {cn.download}"
        )
        checks.append(Check(fetch.check_name("control", cn.label), ok, detail, fatal=False))
    checks.extend(_metric_checks(config))
    return checks


def _metric_checks(config: Config) -> list[Check]:
    """One row per measurement model, all non-fatal.

    These are only ever used by `python -m warlock.bench`; a missing one costs
    a metric, not a job, which is why they are reported here rather than
    failing anything.
    """
    checks: list[Check] = []
    for spec in models.METRIC_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = fetch.present(config, "metric", spec)
        # Weights present is not the same claim as "ranking is on", and the two
        # used to be indistinguishable here. Worker._rank_reference catches
        # every exception out of metrics.reference_cosine and scores on
        # composition alone, so a torchvision that will not import costs the
        # anchor similarity with nothing on screen to say so. Now that
        # torchvision is a declared dependency rather than something that
        # happened to be in the venv, this row names what it did not check
        # instead of pretending the distinction does not exist.
        #
        # Deliberately words and not a second import probe: unlike host
        # matting, nothing in the metric path records a real load failure, so a
        # probe here would be a guess with nothing to corroborate it -- and the
        # matting row already probes transformers on the same host.
        detail = (
            f"{path} -- not checked: whether the model loads "
            "(transformers and torchvision must import)"
            if ok
            else f"not found at {path} -- benchmark metric unavailable; download with:\n"
            f"  {spec.download}"
        )
        checks.append(Check(fetch.check_name("metric", spec.label), ok, detail, fatal=False))
    return checks


def _pose_checks(config: Config, *, probe_slow: bool = True) -> list[Check]:
    """The rig's joint-placement weights, non-fatal -- and only the weights.

    Missing, every humanoid rig still happens: ``rigging.fit_template`` scales
    the template onto the mesh bounding box, which is what every rig did before
    this model existed and is still right for a reference standing in a T-pose.
    What is lost is joint placement on the ones that are *not*, and that is a
    quality difference with no other visible cause -- the skeleton is simply in
    the wrong place inside the limbs, which looks like a bad rig rather than a
    missing download.

    A green row claims less than "informed fitting is on", the same honesty
    ``_matting_checks`` keeps: all this has looked at is a directory, and
    whether a detection then clears ``pose2d``'s sanity gates is a property of
    each reference image.
    """
    checks: list[Check] = []
    for spec in models.POSE_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = fetch.present(config, "pose", spec)
        if ok:
            loaded, note = _load_probe(config, "pose", probe=probe_slow)
            detail = (
                f"weights present at {path} -- {note}; rig joints are read off the "
                "reference image when the detection is confident, and fall back to "
                "the bbox fit when it is not"
            )
            ok = loaded
        else:
            detail = (
                f"not found at {path} -- joint placement falls back to the "
                f"bbox-proportional fit; download with:\n  {spec.download}"
            )
        checks.append(Check(fetch.check_name("pose", spec.label), ok, detail, fatal=False))
    return checks


# The load probes' answers, cached for the life of the process exactly as the
# bpy answer is (N112). Two facts make one attempt enough: ``_load`` in both
# modules caches a ``_FAILED`` sentinel and refuses to retry, and a checkpoint
# that cannot load cannot start loading. ``unload`` clears both sides, which is
# the supported way to make a repaired install re-probe.
#
# Keyed on the *resolved weights directory*, not on the kind. The bpy answer can
# be a bare module global because it is a fact about the interpreter; these are
# facts about a path, and ``WARLOCK_T2I_ROOT`` moves it -- so a kind-keyed cache
# answers the second config with the first config's result, which is a wrong
# green row and, in the suite, a wrong green row that depends on test order.
_probes: dict[tuple[str, str], tuple[bool, str]] = {}
_probe_lock = threading.Lock()

# What a caller that declines to wait is told. ``ok=True`` for the reason
# ``_BLENDER_PENDING`` is: still-checking is not a failure, and the amber dot
# keys on ok.
_PROBE_PENDING = (True, "still checking in the background whether the model loads")


LOAD_PROBE_TIMEOUT = 300.0


def _load_probe(config: Config, which: str, *, probe: bool = True) -> tuple[bool, str]:
    """Whether this model actually loads, once per process, **in a child**.

    ``probe=False`` never blocks -- it returns the cached answer or the pending
    one -- because this is the slowest probe in the file after bpy: a real
    ``from_pretrained`` is seconds and drags torch in, which is exactly what C29
    moved off the startup path. Startup passes ``probe_slow=False``; the first
    health poll, on a task thread, pays for it.

    Out of process because the cost cannot be given back in this one. Loading
    BiRefNet measures 1475 MB of RSS here and 1053 MB of that is still resident
    after every reference is dropped and ``gc.collect()`` has run -- the
    allocator keeps its arenas. See ``pipelines/loadprobe.py``.

    ``winjob.run``, not ``subprocess.run``: this fires from the health poll and
    holds a checkpoint open for seconds, so killing Warlock while it runs would
    otherwise strand a python.exe mid-load. It is also what keeps the
    every-spawn-is-in-the-kill-on-close-job scan satisfied.
    """
    from .pipelines import pose2d

    module = pose2d if which == "pose" else matting
    path = module.model_dir(config)
    key = (which, str(path))
    hit = _probes.get(key)
    if hit is not None:
        return hit
    if not probe:
        return _PROBE_PENDING
    with _probe_lock:
        if key not in _probes:
            _probes[key] = _run_load_probe(which, path)
        return _probes[key]


def _run_load_probe(which: str, path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "weights are not on disk"
    try:
        proc = winjob.run(
            [sys.executable, "-m", "warlock.pipelines.loadprobe", which, str(path)],
            capture_output=True, text=True, timeout=LOAD_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"load probe failed to run: {exc}"
    line = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    verdict, _, detail = line[0].partition(" ")
    if verdict == "ok":
        return True, detail or "loads"
    if verdict == "fail":
        return False, detail or "did not load"
    # Neither word: the child died before it printed. Its stderr is the only
    # thing that knows why, and the last line of it is the useful part.
    stderr = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
    return False, f"load probe did not report: {stderr[0]}"


# What BiRefNet's own modelling code reaches for, plus the package that runs
# it. Named here rather than derived from an import, because the whole point is
# to answer the question without paying for the import: importing timm imports
# torch, and this runs at startup before anything is on screen.
_MATTING_IMPORTS = ("einops", "kornia", "timm", "transformers")


def _missing_modules(names: Sequence[str]) -> list[str]:
    """Which of these do not resolve, without importing any of them.

    ``find_spec`` locates a top-level module without executing it, which is
    what keeps this cheap and keeps a startup check from dragging torch into
    the process. It raises rather than returns None for a dotted name whose
    parent is absent (and a package with broken metadata can raise anything at
    all), so every answer is wrapped: a probe that takes the app down at
    startup is strictly worse than a red row.
    """
    missing: list[str] = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except Exception:  # noqa: BLE001 -- a probe must never raise out of run_checks
            found = False
        if not found:
            missing.append(name)
    return missing


def _matting_checks(config: Config, *, probe_slow: bool = True) -> list[Check]:
    """The host-side matting stack, non-fatal: weights, imports, last failure.

    Missing, every 2D export still works -- the corner flood fill in
    pipelines/reference.py produces the alpha instead, with visibly rougher
    edges on anything that is not on a plain background. That is a quality
    difference the user should be able to see the cause of, which is what this
    row is for.

    The weights alone were not enough to see it by. BiRefNet is loaded with
    trust_remote_code, so what builds it is the checkpoint's own modelling code
    and its imports are invisible to any resolver -- which is exactly how this
    row came to be green on a host where ``_load`` raised
    ModuleNotFoundError on every export. So the three packages that code
    reaches for are probed by name, and ``matting.last_error`` -- the words of
    a load that already failed this session -- is reported beside them. A row
    that agrees with the filesystem and disagrees with the program is the worst
    of both answers.

    And since N112 it *does* claim that the checkpoint loads, because it tries:
    ``_load_probe`` runs a real CPU ``from_pretrained`` once per process, off
    the startup path and cached like the bpy answer. The import scan above is
    kept rather than replaced -- it is the cheap answer available before the
    slow one has run, and it names the missing package where a load failure
    only names the exception it raised.
    """
    missing = _missing_modules(_MATTING_IMPORTS)
    failure = matting.last_error()
    checks: list[Check] = []
    for spec in models.MATTING_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = fetch.present(config, "matting", spec)
        if ok:
            # No longer "not checked" (N112): the probe attempts a real CPU load
            # once per process, which is the only thing that settles the
            # question a green weights row above a silent fall-back could not.
            loaded, note = _load_probe(config, "matting", probe=probe_slow)
            detail = f"weights present at {path} -- {note}"
            ok = ok and loaded
            if spec.remote_code:
                detail += (
                    "; loading it executes third-party Python from this directory "
                    "in this process (transformers runs the repo's own modelling code)"
                )
            if missing:
                ok = False
                detail += (
                    "; cannot import " + ", ".join(missing) + " -- 2D exports fall back to "
                    "the corner fill; install with:\n  uv sync --extra text2image"
                )
            if failure:
                ok = False
                detail += f"; last load failed: {failure}"
        else:
            detail = (
                f"not found at {path} -- 2D exports fall back to the corner fill; "
                f"download with:\n  {spec.download}"
            )
        # "host matting", against _birefnet_check's "trellis": two rows, two
        # different BiRefNets -- one GGUF inside trellis-server, this one on
        # the host for 2D exports -- and a user with rough edges has to be able
        # to tell which download the row is asking for.
        checks.append(Check(fetch.check_name("matting", spec.label), ok, detail, fatal=False))
    return checks

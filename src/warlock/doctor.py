"""Preflight checks: what's missing before you waste two minutes on a GPU job."""

from __future__ import annotations

import ctypes
import os
import shutil
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import fetch, guidance, memlog, models, native, packs, rigging, vram, winjob
from .config import Config

# Cheap: matting itself imports nothing heavier than models and reference, both
# of which keep torch, cv2 and numpy inside the functions that need them. It is
# imported for last_error() alone -- doctor runs in this process, so the answer
# is a module attribute rather than anything that has to be carried.
from .pipelines import matting

MIN_FREE_DISK_GB = 5.0

#: Where the host-memory row stops being green. Below the queue's own
#: ``COMMIT_CEILING`` (0.90) on purpose: the doctor's job is to explain the
#: refusal *before* it happens, and a row that only turns amber at the exact
#: point jobs start failing tells the user nothing they had not already been
#: told by the failure.
COMMIT_WARN = 0.85

#: How far above installed RAM the commit limit must sit before the pagefile
#: stops being the thing to blame. See ``_commit_check``.
PAGEFILE_HEADROOM = 1.25

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
    """One diagnostic row.

    ``fatal`` and ``pending_install`` are different claims about a failing row
    and only one of them may be true. ``fatal`` means *this install is broken*:
    something the installer ships is missing, and nothing the user can do in
    the app will fix it. ``pending_install`` means *you have not downloaded
    this yet*, which is the ordinary state of a fresh machine and must not be
    reported as a fault -- a red banner on first launch taught every new user
    that a working app was broken, which is the incident this field exists for.

    Note the unrelated ``_BLENDER_PENDING``/``_MUSIC_DEPS_PENDING`` further
    down this module: those are ``ok=True`` placeholders meaning *the probe
    has not finished*, a third thing again. Hence the longer name here.
    """

    name: str
    ok: bool
    detail: str
    fatal: bool
    pending_install: bool = False

    def __post_init__(self) -> None:
        # A row cannot be both "your install is broken" and "you have not
        # installed this yet"; the banner and the exit code read the two
        # differently and a row claiming both would be reported twice.
        if self.fatal and self.pending_install:
            raise ValueError(f"{self.name}: fatal and pending_install are exclusive")


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
    genuinely slow probes -- the torch import, the bpy subprocess and the
    ACE-Step import -- and
    reports them as still-checking rows instead (C29/C30). Startup uses both;
    the header health poll re-runs the slow probes once, off the frame thread,
    and their answers are cached from then on.
    """
    s = static_checks(config, probe_slow=probe_slow) if static is None else list(static)
    v = volatile_checks(config, trellis_running, probe_slow=probe_slow)
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
        *_text_checks(config),
        *_pose_checks(config, probe_slow=probe_slow),
        *_music_checks(config),
        music_deps_check(probe=probe_slow),
        *_separation_checks(config),
        blender_check(probe=probe_slow),
    ]


def volatile_checks(
    config: Config, trellis_running: bool = False, *, probe_slow: bool = True
) -> list[Check]:
    """The rows worth re-asking every poll: the card, the job object, the
    disk and the port are the four answers that change while the app runs.

    ``probe_slow`` reaches here for one row. It is documented as skipping "the
    torch import and the bpy subprocess", and ``_cuda_check`` duly defers the
    import -- but ``_vram_check`` sat in this list calling ``vram.probe()``,
    which imports torch unconditionally, so startup paid the 1.57 s anyway and
    the deferral bought nothing on the recommended install. The flag now
    reaches both rows that can trigger it.
    """
    return [
        _vram_check(config, probe=probe_slow),
        _commit_check(),
        _job_object_check(),
        _instance_check(config),
        _environment_check(),
        _disk_check(config),
        _store_check(config),
        _port_check(config, trellis_running),
    ]


def _store_check(config: Config) -> Check:
    """Whether ``jobs.sqlite`` is still a database sqlite will open.

    The store is the app's single point of total failure and had the least
    diagnosis in it: ``JobStore.__init__`` connects, runs the schema and
    migrates with nothing guarding any of it, so a malformed image is the
    generic "ran into a problem while starting" box, on every launch, with no
    way in and nothing on screen naming the file. This row is what turns that
    into a sentence, and ``service.library.backup``'s copy is what it points
    at -- which is also why the check is worth having while the app is *up*: a
    database that has started to go is worth backing up before the launch that
    cannot open it.

    Volatile rather than static, because "is this file still readable" is
    exactly the kind of answer that changes while a process runs, and because
    a corruption that appears mid-session should not wait for a restart to be
    reported. Its own connection, opened read-only and closed immediately: this
    runs on a task thread and must not touch the live store's lock.

    ``PRAGMA quick_check`` rather than ``integrity_check``: it is the same
    walk minus the index cross-references, which on a database this size is
    milliseconds either way, and the failure this exists to catch (a torn page,
    a truncated file) shows up in both.
    """
    import sqlite3

    path = config.db_path
    if not path.exists():
        # A first run, or a home the user has just pointed somewhere new. The
        # store is created on demand, so absence is not a fault.
        return Check("job database", True, "not created yet", fatal=False)
    try:
        # ``as_uri()`` percent-encodes the path -- a raw f-string breaks on
        # anything a URI treats specially (``#`` truncates to a fragment,
        # ``%`` starts an escape), and a home directory is exactly the kind of
        # path a user picks without knowing that.
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return Check("job database", False, f"{path.name} cannot be opened: {exc}", fatal=False)
    answer = str(row[0]) if row else ""
    if answer.lower() == "ok":
        return Check("job database", True, f"{path.name} is intact", fatal=False)
    return Check(
        "job database",
        False,
        f"{path.name} reports {answer!r}; back the library up and see warlock.log",
        fatal=False,
    )


def _environment_check() -> Check:
    """Every ``WARLOCK_*`` value this process could not parse, in one row.

    One row and not one per variable: a machine with three typos in its
    environment has one problem, and three amber lines describing it would bury
    the rest of the diagnostics. Non-fatal because each of them fell back to its
    documented default, so the app is running -- just not with the settings the
    user believes they chose, which is precisely the thing worth saying out loud
    (RUN-03).
    """
    from .config import INVALID_ENV

    if not INVALID_ENV:
        return Check("environment", True, "every WARLOCK_* value parsed", fatal=False)
    parts = [f"{name}={raw!r} (expected {expected})" for name, raw, expected in INVALID_ENV]
    return Check(
        "environment",
        False,
        "ignored and left at the default: " + "; ".join(parts),
        fatal=False,
    )


def _instance_check(config: Config) -> Check:
    """Is this the only Warlock on this writable resource set?

    A row rather than only a startup refusal, because the refusal fires once and
    the condition it guards -- two processes over one job database and one engine
    port -- is exactly the sort of thing somebody goes looking for in
    diagnostics after the fact (RUN-01, A1-L9).

    Inside the app the answer is simply "we hold it" -- ``run()`` took the lock
    before anything else and keeps it for the process's life. Re-taking it here
    would answer nothing: re-locking one file from one process is allowed on
    some platforms (proving nothing) and refused on others (reading as though
    somebody else had it). So the live lock is asked first, and only a caller
    that holds none -- ``warlock doctor`` from a terminal -- probes.
    """
    from . import instance

    if instance.held_by_us():
        return Check(
            "single instance", True, f"this Warlock owns {config.home}", fatal=False
        )
    probe = instance.InstanceLocks(instance.lock_paths(config))
    if probe.acquire():
        probe.release()
        return Check(
            "single instance",
            True,
            "no other Warlock is using this home, database, or model root",
            fatal=False,
        )
    if probe.failure:
        return Check("single instance", False, probe.failure, fatal=False)
    return Check(
        "single instance",
        False,
        f"another Warlock holds {probe.path} -- close it, or give this copy "
        "separate WARLOCK_HOME, WARLOCK_DB, and WARLOCK_T2I_ROOT values",
        fatal=False,
    )


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
#
# The version is pinned in the URL rather than left to "the releases page",
# for ``TRELLIS_GGUF_REVISION``'s reason one paragraph down: this is a *fatal*
# row, so the remedy is the only path a fresh install has, and an unpinned one
# hands the user whatever shipped this week.
TRELLIS_EXE_VERSION = "v0.6.0"
TRELLIS_EXE_ASSET = "trellis-cuda-windows-x64.zip"
TRELLIS_EXE_URL = (
    f"https://github.com/pwilkin/trellis.cpp/releases/download/"
    f"{TRELLIS_EXE_VERSION}/{TRELLIS_EXE_ASSET}"
)
# The SHA-256 GitHub publishes for that exact asset. This is the one unsigned
# third-party binary in the whole setup, and a remedy that sent the user to a
# page with no digest gave them nothing to check a download against -- so the
# number travels with the URL, and **both move together or neither moves**:
# bumping ``TRELLIS_EXE_VERSION`` without re-reading the digest is worse than
# publishing none, because a mismatch then reads as tampering rather than as a
# stale constant.
TRELLIS_EXE_SHA256 = "4d08ab27e83094035fd8349aaf34d3460738df0466ef9c4991ddd958c0344bc2"
TRELLIS_EXE_HINT = (
    f"download {TRELLIS_EXE_ASSET} from {TRELLIS_EXE_URL} and unpack it there "
    f"(vendored build: {TRELLIS_EXE_VERSION}; sha256 {TRELLIS_EXE_SHA256}), "
    "or point WARLOCK_TRELLIS_EXE at your own copy"
)
def trellis_gguf_hint(config: Config) -> str:
    """The ``hf download`` line, landing in *this* install's models directory.

    A function rather than a constant because the literal ``models/...`` was a
    **relative** path, which is the mistake ``fetch``'s module docstring
    already names: the command is pasted into whatever directory the shell
    happens to be in, so it put 16 GB somewhere Warlock never inspects and
    left the fatal row standing. Quoted, because the resolved path contains
    spaces on any ordinary Windows profile.
    """
    return fetch.download_text(config, "engine", models.ENGINE_MODELS["trellis_gguf"])


def _registry_row(config: Config, kind: str, spec: Any, ok: bool, detail: str) -> Check:
    """One downloadable registry row.

    Never fatal, and ``pending_install`` whenever it is absent: every row this
    builds is a model the user can install from Settings -> Models, so "not
    downloaded" is a setup step rather than a fault. Written once because the
    nine call sites were nine copies of the same four arguments, and the
    ``fatal=False`` in each was the only thing saying so.

    **Downgraded uniformly when the files are damaged (M04).** ``fetch.present``
    only ever answered "is it here" -- a zero-byte weight file, the ordinary
    shape of a killed download, passes every ``Path.exists()`` in it and then
    fails minutes later with the checkpoint already resident in VRAM. That
    downgrade used to be hand-rolled once, in the base-model loop alone
    (``fetch.suspect_files`` called at exactly that one site): reproduced
    against a host with all ten engine probe files emptied, Doctor still
    returned OK, because the GGUF, LoRA, adapter, control, metric, music,
    separation, pose and matting rows never asked the question. Asked here
    instead, so every registry row gets the same answer for the price of one
    ``stat`` per candidate file rather than nine call sites remembering to
    repeat it (or not).
    """
    if ok:
        bad = fetch.suspect_files(config, kind, spec)
        if bad:
            ok = False
            detail = (
                f"{detail} -- but {len(bad)} file(s) are empty and will not "
                f"load; remove and reinstall this model. First: {bad[0]}"
            )
    return Check(
        fetch.check_name(kind, spec.label), ok, detail, fatal=False, pending_install=not ok
    )


def _exe_check(config: Config) -> Check:
    path = config.trellis_server_exe
    # L01: ``.exists()`` is true of a directory as well as a file, so a
    # broken unpack that left a *folder* named ``trellis-server.exe`` (an
    # archive extracted one level too shallow, or a leftover from a previous
    # attempt) passed this check and then failed to launch with no row
    # naming why. ``.is_file()`` plus a distinct message for the directory
    # case turns that into a damaged-install diagnostic instead of a second
    # "not found".
    ok = path.is_file()
    if ok:
        detail = str(path)
    elif path.exists():
        detail = f"{path} exists but is not a file -- {TRELLIS_EXE_HINT}"
    else:
        detail = f"not found at {path} -- {TRELLIS_EXE_HINT}"
    return Check("trellis-server.exe", ok, detail, fatal=True)


def _gguf_check(config: Config) -> Check:
    spec = models.ENGINE_MODELS["trellis_gguf"]
    ok = fetch.present(config, "engine", spec)
    missing = [name for name in spec.probe if not (config.trellis_models_dir / name).is_file()]
    # M04: this row called ``fetch.present`` alone, so a killed download that
    # left every probed GGUF at zero bytes still passed every ``is_file()`` in
    # it and reported OK -- reproduced with all ten engine probe files emptied
    # and Doctor still green. ``_registry_row`` gained the same check for the
    # other nine rows; this one is built by hand (its message names *missing*
    # files, which the registry helper does not), so the check is repeated
    # here rather than routed through it.
    bad = fetch.suspect_files(config, "engine", spec) if ok else []
    if bad:
        ok = False
    if bad:
        detail = (
            f"{len(bad)} required GGUF file(s) are empty and will not load; "
            f"remove and reinstall. First: {bad[0]}"
        )
    elif ok:
        detail = str(config.trellis_models_dir)
    else:
        detail = (
            f"{len(missing)} required GGUF file(s) missing from "
            f"{config.trellis_models_dir} ({', '.join(missing[:3])}) -- download with:\n"
            f"  {trellis_gguf_hint(config)}"
        )
    # **Not fatal, unlike the exe beside it, and the difference is the whole
    # point.** ``trellis-server.exe`` is staged by the installer, so its
    # absence is a broken install and nothing in the app can fix it. These
    # weights are a download the user has not made yet -- the ordinary state
    # of every fresh machine, and Settings -> Models is the button that fixes
    # it. Reporting it as fatal put a red banner on a healthy first launch and
    # made ``warlock doctor`` exit 1 on a machine with nothing wrong with it.
    return Check(
        "TRELLIS GGUF weights", ok, detail, fatal=False, pending_install=not ok
    )


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
    #
    # ``pending_install`` because this file arrives *inside* the trellis2-gguf
    # download (it is in that spec's own probe list), so an absent one is the
    # same not-downloaded-yet state the row above reports, not a fault of its
    # own. Left as a plain warning it was the one amber row a fresh install
    # still showed, which is exactly the false alarm this change removes.
    return Check(
        "trellis: birefnet.gguf (background removal)",
        ok,
        detail,
        fatal=False,
        pending_install=not ok,
    )


def _gltfpack_check(config: Config) -> Check:
    # L01, ``_exe_check``'s fix: a directory named like the binary passed
    # ``.exists()`` and read as "not found" either way.
    path = config.gltfpack_exe
    ok = path.is_file()
    if ok:
        detail = str(path)
    elif path.exists():
        detail = f"{path} exists but is not a file -- meshes ship at full reconstruction density"
    else:
        detail = f"not found at {path} -- meshes ship at full reconstruction density"
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


def _vram_check(config: Config, *, probe: bool = True) -> Check:
    """How large the card is, which mode that chose, and what a job may ask for.

    Fatal only when the budget cannot hold even a lone trellis run: every 3D
    job on such a host fails anyway, and it is better to say so at startup than
    two minutes into the first reconstruction. **Also fatal when the host has
    no CUDA device at all**: the plan then reports "admission control is off"
    and, read alone, an amber row saying a budget is not being enforced is
    indistinguishable from good news. There is no CPU fallback for
    reconstruction, so the honest verdict on such a host is that the 3D path
    will not run -- and the alternative is what it used to be, a
    ``RuntimeError("trellis-server exited during startup (code N)")`` two
    minutes into the first attempt, which is not a sentence anyone can act on.

    ``probe=False`` takes the reading without importing torch --
    ``_cuda_check``'s rule and its reason (C29), which this row was quietly
    exempt from. ``device_memory`` reads torch only if something else has
    already imported it and otherwise falls back to the last published reading,
    so on the startup pass the row is simply computed from whatever is known.
    """
    probed = probe or sys.modules.get("torch") is not None
    device = vram.probe() if probed else vram.device_memory()
    resolved = vram.plan(
        exclusive=config.vram_exclusive,
        budget_gib=config.vram_budget_gib,
        total_gib=config.vram_total_gib,
        device=device,
        explicit=config.vram_exclusive_explicit,
    )
    if not resolved.enforced:
        # Told a total by config (WARLOCK_VRAM_TOTAL) rather than by a card:
        # the operator has said what to assume, and contradicting them with a
        # fatal row would refuse the very override they reached for.
        if resolved.total_gib is None and config.vram_total_gib is None:
            if not probed:
                # "We have not looked yet" is not "there is no card", and
                # conflating them put a red row on every cold start of a
                # perfectly good machine: with the torch import deferred,
                # ``device_memory`` has nothing to read and no published
                # reading to fall back on until the first health poll. The
                # still-checking wording is ``_cuda_check``'s, for its reason.
                return Check(
                    "VRAM budget", True,
                    "still checking in the background (importing torch takes a moment)",
                    fatal=False,
                )
            return Check(
                "VRAM budget", False,
                resolved.reason + " -- no CUDA device means 3D reconstruction"
                " cannot run at all; there is no CPU fallback",
                fatal=True,
            )
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
    """Both volumes, not just the library's.

    ``db_path`` is deliberately allowed to sit outside the asset tree -- the
    advertised "SSD index over a spinning-disk library" -- so measuring
    ``data_dir`` alone is blind to exactly the split the config supports: a
    full DB volume showed a green row right up until a raw sqlite "disk full".
    Reported as the *worse* of the two, named, so the row says which one.
    """
    seen: dict[Any, tuple[float, Any]] = {}
    for where in (config.data_dir, config.db_path.parent):
        try:
            free_gb = shutil.disk_usage(where).free / (1024**3)
        except OSError:
            continue
        seen.setdefault(_volume_key(where), (free_gb, where))
    if not seen:
        return Check("free disk space", False, "no volume could be measured", fatal=False)
    free_gb, where = min(seen.values())
    ok = free_gb >= MIN_FREE_DISK_GB
    return Check("free disk space", ok, f"{free_gb:.1f} GB free in {where}", fatal=False)


def _physical_ram_gib() -> float | None:
    """Installed RAM in GiB, or None when it cannot be read.

    Its own function so the commit row can be tested without a machine of a
    particular shape -- ``memlog`` deliberately reports the commit *limit* and
    not the physical total, because the limit is what the ceiling divides by.
    """
    if sys.platform != "win32":
        return None
    try:
        info = memlog._PERFORMANCE_INFORMATION()  # type: ignore[attr-defined]
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return None
        return info.PhysicalTotal * info.PageSize / (1024**3)
    except (OSError, AttributeError):
        return None


def _commit_check() -> Check:
    """How close the host is to the commit wall, and whether the pagefile is why.

    The queue refuses a job at ``COMMIT_CEILING`` on a percentage of
    **system-wide** commit, and that refusal used to arrive with no context at
    all: "close other applications or restart Warlock". On 2026-08-21 that
    message was shown on a machine with **24 GiB of physical RAM free** --
    63.5 GiB installed, a 14.2 GiB pagefile, so a 77.7 GiB limit that ordinary
    GPU-driver and allocator commit had pushed to 96%. Closing applications was
    not the answer and restarting Warlock only postponed it.

    So the row distinguishes the two diagnoses, because the remedies are
    opposite. A limit barely above installed RAM means the *pagefile* is the
    constraint and growing it is the fix. A limit already well above RAM means
    something is genuinely consuming the memory and the pagefile advice would
    do nothing.

    Never fatal. It is a explanation for a refusal the queue owns, not a
    precondition for starting: a machine at 91% runs everything except the
    largest job, and refusing to start would be a worse answer than saying so.
    """
    sysmem = memlog.system_memory()
    if sysmem is None or sysmem.commit_limit <= 0:
        return Check("host memory", True, "commit not measured on this platform", fatal=False)
    pct = sysmem.commit_fraction * 100
    free = sysmem.commit_limit - sysmem.commit_total
    detail = (
        f"{sysmem.commit_total:.1f}/{sysmem.commit_limit:.1f} GiB committed "
        f"({pct:.0f}%), {free:.1f} GiB free"
    )
    if sysmem.commit_fraction < COMMIT_WARN:
        return Check("host memory", True, detail, fatal=False)
    ram = _physical_ram_gib()
    # 1.25x installed RAM: below that the pagefile is small enough that the
    # limit is essentially RAM itself, which is the shape that produces a
    # refusal beside plentiful free memory. A number rather than "any pagefile
    # at all" because Windows' default managed pagefile on a large-RAM machine
    # is genuinely small, and that default is the case being described.
    if ram is not None and sysmem.commit_limit < ram * PAGEFILE_HEADROOM:
        detail += (
            f" -- the commit limit is only {sysmem.commit_limit - ram:+.1f} GiB above "
            f"{ram:.1f} GiB of installed RAM, so the pagefile is the constraint "
            "rather than memory itself; growing it raises the limit"
        )
    else:
        detail += " -- jobs are refused past 90%; close other applications"
    return Check("host memory", False, detail, fatal=False)


def _volume_key(path: Path) -> Any:
    """What makes two paths the same volume, so one drive is not measured
    twice. ``os.stat`` where it is meaningful, the drive letter otherwise."""
    try:
        return os.stat(path).st_dev
    except OSError:
        return str(path.drive or path).lower()


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
            # The revision pin, when a fetch recorded one. Whether the files
            # are *usable* rather than merely present -- MDL-08's question --
            # is answered uniformly for every registry row inside
            # ``_registry_row`` now (M04), rather than here alone.
            recorded = fetch.read_manifest(path) or {}
            pins = {
                r.get("revision")
                for r in (recorded.get("repos") or {}).values()
                if r.get("revision")
            }
            if pins:
                detail += f" (revision {', '.join(sorted(pins))})"
        elif missing_lora is not None:
            detail = (
                f"weights present, but {spec.base_lora} is missing at "
                f"{missing_lora} -- this model cannot run without it; "
                f"download with:\n  {fetch.download_text(config, 'base', spec)}"
            )
        else:
            detail = (
                f"not found at {path} -- unavailable; download with:\n"
                f"  {fetch.download_text(config, 'base', spec)}"
            )
        checks.append(_registry_row(config, "base", spec, ok, detail))
    for lora in models.STYLE_LORAS.values():
        path = config.t2i_model_root / "loras" / lora.filename
        ok = fetch.present(config, "lora", lora)
        detail = (
            str(path)
            if ok
            else (
                f"not found at {path} -- style unavailable; download with:\n"
                f"  {fetch.download_text(config, 'lora', lora)}"
            )
        )
        checks.append(_registry_row(config, "lora", lora, ok, detail))
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
                f"download with:\n  {fetch.download_text(config, 'adapter', adapter)}"
            )
        checks.append(_registry_row(config, "adapter", adapter, ok, detail))
    for cn in models.CONTROLNETS.values():
        path = config.t2i_model_root / cn.dir_name
        ok = fetch.present(config, "control", cn)
        detail = (
            str(path)
            if ok
            else (
                f"not found at {path} -- control unavailable; download with:\n"
                f"  {fetch.download_text(config, 'control', cn)}"
            )
        )
        checks.append(_registry_row(config, "control", cn, ok, detail))
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
            f"  {fetch.download_text(config, 'metric', spec)}"
        )
        checks.append(_registry_row(config, "metric", spec, ok, detail))
    return checks


def _music_checks(config: Config) -> list[Check]:
    """One row per music model: are its weights on disk?

    Structurally the pose and matting loops, with one difference that is not
    cosmetic: there is no load probe. Those two are optional and degrade, so
    "the directory is there but the import is broken" is worth a distinct
    answer; ACE-Step is loaded only inside its own subprocess, and the app
    process deliberately never imports torch -- so a probe here would either
    lie or undo the thing the subprocess exists for.

    Non-fatal for the reason every model row is: a machine with no music
    weights runs the whole application except one mode, and doctor's fatal
    checks are the ones that mean nothing works.

    The row says weights and *only* weights. It used to read "Muse can
    generate", which was a claim about two things it never checked: the second
    is whether the ``music`` extra is installed at all, and that is
    :func:`music_deps_check`'s question, one row further down.
    """
    checks: list[Check] = []
    for spec in models.MUSIC_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = fetch.present(config, "music", spec)
        if ok:
            detail = f"weights present at {path}"
        else:
            detail = (
                f"not found at {path} -- Muse refuses at the door rather than "
                f"falling back; download with:\n"
                f"  {fetch.download_text(config, 'music', spec)}"
            )
        checks.append(
            _registry_row(config, "music", spec, ok, detail)
        )
    return checks


MUSIC_PROBE_TIMEOUT = 120.0
MUSIC_INSTALL_HINT = "Muse unavailable; install with: uv sync --extra music"

_music_deps: Check | None = None
_music_deps_lock = threading.Lock()
_MUSIC_DEPS_PENDING = Check(
    "Muse (dependencies)", True,
    "still checking in the background -- Muse appears when it finishes",
    fatal=False,
)


def music_deps_check(*, probe: bool = True) -> Check:
    """Is the ``music`` extra actually installed? Probed in a child.

    Structurally :func:`blender_check`, and here for a defect it would have
    caught: ``music`` was missing from the installer and both CI workflows for
    a release cycle, so a packaged build shipped Muse in the rail and failed
    its first take with ``ModuleNotFoundError``. The weights rows said nothing,
    because weights were never the thing that was missing.

    In a child for the same reason the pipeline is: ACE-Step drags torch in,
    and the app process deliberately never imports it. That constraint rules
    out an in-process probe -- it does not rule out asking a subprocess, which
    is the process the real job uses anyway.
    """
    global _music_deps
    if _music_deps is not None:
        return _music_deps
    if not probe:
        return _MUSIC_DEPS_PENDING
    with _music_deps_lock:
        if _music_deps is None:
            _music_deps = _probe_music_deps()
        return _music_deps


def _probe_music_deps() -> Check:
    try:
        # winjob.run for the reason the bpy probe uses it: this can fire from
        # startup, the import is seconds, and killing Warlock mid-probe used to
        # strand the child.
        proc = winjob.run(
            [sys.executable, "-c", "import warlock.pipelines.acestep.pipeline_ace_step"],
            capture_output=True, text=True, timeout=MUSIC_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("Muse (dependencies)", False, f"probe failed: {exc}", fatal=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["import failed"]
        return Check(
            "Muse (dependencies)", False, f"{detail[0]} -- {MUSIC_INSTALL_HINT}", fatal=False
        )
    return Check(
        "Muse (dependencies)", True, "the ACE-Step pipeline imports in a child", fatal=False
    )


def _separation_checks(config: Config) -> list[Check]:
    """One row per stem-separation model: are its weights on disk?

    ``_music_checks``' loop, with no load probe for its reason exactly -- the
    model runs only inside its own subprocess and the app process deliberately
    never imports torch.

    What differs is the *sentence*, because the consequence differs. A missing
    music model refuses a job at the door; a missing separation model refuses
    only the separation. Every take still generates, plays, exports and imports
    into Sirens -- what is lost is four extra files, and a row that said "Muse
    refuses" would be telling the user their mode was broken when it is not.
    """
    checks: list[Check] = []
    for spec in models.SEPARATION_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = fetch.present(config, "separation", spec)
        if ok:
            detail = f"weights present at {path} -- takes can be split into stems"
        else:
            detail = (
                f"not found at {path} -- Muse works without it; what is lost is "
                f"splitting a take into stems. Download with:\n"
                f"  {fetch.download_text(config, 'separation', spec)}"
            )
        checks.append(
            _registry_row(config, "separation", spec, ok, detail)
        )
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
                f"bbox-proportional fit; download with:\n"
                f"  {fetch.download_text(config, 'pose', spec)}"
            )
        checks.append(_registry_row(config, "pose", spec, ok, detail))
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


# One implementation, in ``packs``, because the pack registry asks the same
# question of the same interpreter: is this optional dependency actually here?
# Two copies of a probe whose whole job is to *never raise* is two places for
# one of them to start raising.
_missing_modules = packs.missing_modules


def _matting_checks(config: Config, *, probe_slow: bool = True) -> list[Check]:
    """The host-side matting stack, non-fatal: weights, imports, last failure.

    Missing, every 2D export still works -- the corner flood fill in
    pipelines/reference.py produces the alpha instead, with visibly rougher
    edges on anything that is not on a plain background. That is a quality
    difference the user should be able to see the cause of, which is what this
    row is for.

    The weights alone were not enough to see it by. BiRefNet's modelling code
    reaches for packages no resolver can see from the registry -- which is
    exactly how this row came to be green on a host where ``_load`` raised
    ModuleNotFoundError on every export. So the packages that code needs are
    probed by name, and ``matting.last_error`` -- the words of a load that
    already failed this session -- is reported beside them. A row that agrees
    with the filesystem and disagrees with the program is the worst of both
    answers.

    That code used to be the *checkpoint's own*, run under
    ``trust_remote_code``; it is vendored at ``pipelines/birefnet/`` now, so
    the import scan is checking this repo's dependencies rather than a
    downloaded file's. The scan is unchanged because the imports are.

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
                f"download with:\n  {fetch.download_text(config, 'matting', spec)}"
            )
        # "host matting", against _birefnet_check's "trellis": two rows, two
        # different BiRefNets -- one GGUF inside trellis-server, this one on
        # the host for 2D exports -- and a user with rough edges has to be able
        # to tell which download the row is asking for.
        checks.append(_registry_row(config, "matting", spec, ok, detail))
    return checks


_TEXT_IMPORTS = ("torch", "transformers")


def _text_checks(config: Config) -> list[Check]:
    """The local text model behind the Flourish prompt field, non-fatal.

    **Not a registry row.** Every entry in ``models`` carries a fetch pinned to
    a revision, and the pin comes from the measurement that picks the model --
    which has not been run (it is on the human's list). Until it has, the field probes one
    directory by name (``inker_flourish.TEXT_MODEL_DIR``) and this row says
    what it found there. Missing, the field still works:
    ``inker/flourish/keywords`` maps a fixed vocabulary of colours and
    adjectives, deterministically. What the model adds is the sentence the
    vocabulary does not cover, so the row says which of the two the field is
    using rather than "broken".
    """
    from .studio import inker_flourish

    path = inker_flourish.text_model_dir(config)
    missing = _missing_modules(_TEXT_IMPORTS)
    if inker_flourish.text_model_present(config):
        ok = not missing
        detail = f"weights present at {path} -- the Flourish prompt uses the model"
        if missing:
            detail += (
                "; cannot import " + ", ".join(missing) + " -- the prompt falls back to "
                "the keyword mapper; install with:\n  uv sync --extra text2image"
            )
    else:
        ok = True
        detail = (
            f"not found at {path} -- the Flourish prompt uses the keyword mapper. "
            "No download is offered yet: the model is chosen and its revision pinned "
            "by a measurement still owed; to try one, place an instruct "
            f"model's config.json and safetensors in {path}"
        )
    return [Check("text model: Flourish prompt", ok, detail, fatal=False, pending_install=not ok)]

"""The optional native kernel library, and the rule for using one.

``vendor/warlockc/warlockc.dll`` is built from ``native/*.c`` by
``native/build.ps1``. It is vendored the way ``trellis-server.exe`` and
``gltfpack.exe`` are -- a local build artifact under a gitignored directory,
never downloaded -- which means the honest default is that it is *absent*, and
every caller has to work without it.

So this module is pure in the way :mod:`~warlock.vram` and :mod:`~warlock.memlog`
are pure: stdlib only, no imports from ``service``, ``queue`` or ``studio``,
and a missing or unusable DLL is ``None`` rather than an exception. The call
sites read::

    if native.available():
        ...kernel...
    else:
        ...the numpy implementation...

and the numpy implementation is never deleted -- it is both the fallback and
the reference the parity tests measure the kernel against.

**The ABI check is the load-bearing part.** ``vendor/`` is gitignored, so a
working tree routinely holds a DLL built from older sources beside newer
Python. Without a version handshake that DLL would keep computing the previous
behaviour silently, which is the one failure mode a fallback path must not
have: an absent DLL is obvious, a stale one is not. ``ABI`` here must equal
``WARLOCKC_ABI`` in ``native/warlockc.h``; a mismatch is reported once and
treated as absent.

Two environment variables, both for situations that already exist elsewhere in
this project: ``WARLOCK_NATIVE=0`` forces the fallback (A/B timing, and CI on a
machine with no compiler), and ``WARLOCK_NATIVE_DLL`` relocates the file the
way ``WARLOCK_GLTFPACK`` relocates gltfpack -- a git worktree has no
``vendor/`` at all, which is why the other ``WARLOCK_*`` path overrides exist.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Must match WARLOCKC_ABI in native/warlockc.h.
ABI = 1

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DLL = _PROJECT_ROOT / "vendor" / "warlockc" / "warlockc.dll"

# The probe result, cached: ``...`` means "not tried yet", None means
# unavailable. Probing is a file check plus a load, and callers ask per job.
_lib: Any = ...


def dll_path() -> Path:
    """Where the library is expected, honouring the override."""
    override = os.environ.get("WARLOCK_NATIVE_DLL")
    return Path(override) if override else _DEFAULT_DLL


def _enabled() -> bool:
    return os.environ.get("WARLOCK_NATIVE", "1").strip().lower() not in {
        "0",
        "off",
        "false",
        "no",
    }


def _bind(lib: ctypes.CDLL) -> None:
    """Declare every prototype. ctypes defaults to int-returning and
    int-sized arguments, which silently truncates a 64-bit pointer."""
    d = ctypes.POINTER(ctypes.c_double)

    lib.warlockc_abi.restype = ctypes.c_int32
    lib.warlockc_abi.argtypes = []

    lib.warlockc_rasterise.restype = None
    lib.warlockc_rasterise.argtypes = [
        d, d, d, d, d, d, d,  # ax ay bx by cx cy area2
        ctypes.c_int64,  # n
        ctypes.c_int32,  # resolution
        ctypes.POINTER(ctypes.c_uint8),  # covered
    ]


def _load() -> Any:
    if not _enabled():
        return None
    path = dll_path()
    if not path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(path))
        _bind(lib)
        found = int(lib.warlockc_abi())
    except (OSError, AttributeError, ValueError) as exc:
        # A DLL built for the wrong architecture, a partial build missing an
        # export, or a file that is not a library at all. None of those should
        # stop the app: they mean "use numpy".
        log.warning("warlockc at %s is unusable (%s); using numpy fallbacks", path, exc)
        return None
    if found != ABI:
        log.warning(
            "warlockc at %s is ABI %d, this build expects %d -- rebuild with "
            "native/build.ps1; using numpy fallbacks",
            path,
            found,
            ABI,
        )
        return None
    return lib


def lib() -> Any:
    """The loaded library, or None. Probed once per process."""
    global _lib
    if _lib is ...:
        _lib = _load()
    return _lib


def available() -> bool:
    return lib() is not None


def reset() -> None:
    """Forget the cached probe.

    For tests that flip ``WARLOCK_NATIVE`` -- the whole point of the env var is
    running the same suite both ways in one process."""
    global _lib
    _lib = ...


def status() -> tuple[bool, str]:
    """(ok, detail) for the doctor row. Never raises, never loads twice."""
    path = dll_path()
    if not _enabled():
        return False, "disabled by WARLOCK_NATIVE=0 -- numpy fallbacks in use"
    if available():
        return True, f"{path} (ABI {ABI})"
    if not path.exists():
        return False, (
            f"not built at {path} -- run native\\build.ps1 "
            "(optional; numpy fallbacks in use)"
        )
    return False, f"{path} is unusable or ABI-mismatched -- rebuild with native\\build.ps1"


# --- typed views onto the kernels -------------------------------------------
#
# Callers pass numpy arrays; the conversion to pointers lives here so no call
# site has to know ctypes. Contiguity and dtype are the caller's to guarantee
# (they are cheap asserts there and would be a per-call copy here).


def _ptr(array: Any, ctype: Any) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def rasterise(
    ax: Any, ay: Any, bx: Any, by: Any, cx: Any, cy: Any, area2: Any, covered: Any
) -> None:
    """Fill ``covered`` (uint8, square, C-contiguous) from projected triangles.

    Every array must be float64, C-contiguous and the same length; ``covered``
    is written in place and never cleared, matching the numpy path.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_double = ctypes.c_double
    handle.warlockc_rasterise(
        _ptr(ax, c_double),
        _ptr(ay, c_double),
        _ptr(bx, c_double),
        _ptr(by, c_double),
        _ptr(cx, c_double),
        _ptr(cy, c_double),
        _ptr(area2, c_double),
        ctypes.c_int64(len(ax)),
        ctypes.c_int32(covered.shape[0]),
        _ptr(covered, ctypes.c_uint8),
    )


if sys.platform != "win32":  # pragma: no cover - the app is Windows-only
    # Not a hard failure: the loader simply will not find a .dll, and every
    # caller falls back. Stated here so the reason is in the module rather
    # than in a puzzled bug report.
    log.debug("warlockc is built as a Windows DLL; other platforms use numpy")

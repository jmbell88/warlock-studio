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
ABI = 4

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
    f = ctypes.POINTER(ctypes.c_float)
    i64 = ctypes.c_int64

    lib.warlockc_abi.restype = ctypes.c_int32
    lib.warlockc_abi.argtypes = []

    lib.warlockc_rasterise.restype = None
    lib.warlockc_rasterise.argtypes = [
        d, d, d, d, d, d, d,  # ax ay bx by cx cy area2
        ctypes.c_int64,  # n
        ctypes.c_int32,  # resolution
        ctypes.POINTER(ctypes.c_uint8),  # covered
    ]

    lib.warlockc_over_f32.restype = None
    lib.warlockc_over_f32.argtypes = [
        f, i64,  # backdrop, row stride in floats
        f, i64,  # source
        f, i64,  # out
        i64, i64,  # h, w
        ctypes.c_float,  # opacity
        ctypes.c_int32,  # mode
    ]

    lib.warlockc_paint_colour_f32.restype = None
    lib.warlockc_paint_colour_f32.argtypes = [
        f, i64,  # before
        f, i64,  # weight, one channel
        f, i64,  # out
        i64, i64,  # h, w
        ctypes.c_float * 4,  # colour, 0..255
    ]

    lib.warlockc_stack_f32.restype = None
    lib.warlockc_stack_f32.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # layer crops
        ctypes.POINTER(i64),  # their row strides, in bytes
        f,  # opacities
        ctypes.POINTER(ctypes.c_int32),  # modes
        i64,  # n
        f, i64,  # out
        i64, i64,  # h, w
        f, i64,  # base, or NULL
    ]

    lib.warlockc_to_uint8_f32.restype = None
    lib.warlockc_to_uint8_f32.argtypes = [
        f,  # pixels
        ctypes.POINTER(ctypes.c_uint8),  # out
        i64,  # count
    ]

    u8 = ctypes.POINTER(ctypes.c_uint8)
    i32 = ctypes.POINTER(ctypes.c_int32)
    lib.warlockc_contours.restype = i64
    lib.warlockc_contours.argtypes = [
        u8, i64,  # mask, row stride in bytes
        i64, i64,  # h, w
        ctypes.c_uint8,  # threshold
        u8,  # scratch, one zeroed byte per lattice edge
        i32, i64,  # points, capacity in vertices
        i32, i64,  # loop lengths, capacity in loops
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


def over_f32(
    backdrop: Any,
    backdrop_stride: int,
    source: Any,
    source_stride: int,
    out: Any,
    out_stride: int,
    height: int,
    width: int,
    opacity: float,
    mode: int,
) -> None:
    """Composite ``source`` onto ``backdrop`` into ``out``.

    Every array is float32 with four channels last and rows ``*_stride`` floats
    apart; ``mode`` indexes ``composite.BLEND_MODES``. The caller has already
    established all of that -- see ``composite._over_native``, which is the
    only one.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_float = ctypes.c_float
    handle.warlockc_over_f32(
        _ptr(backdrop, c_float),
        ctypes.c_int64(backdrop_stride),
        _ptr(source, c_float),
        ctypes.c_int64(source_stride),
        _ptr(out, c_float),
        ctypes.c_int64(out_stride),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        ctypes.c_float(opacity),
        ctypes.c_int32(mode),
    )


def paint_colour_f32(
    before: Any,
    before_stride: int,
    weight: Any,
    weight_stride: int,
    out: Any,
    out_stride: int,
    height: int,
    width: int,
    colour: tuple[int, int, int, int],
) -> None:
    """Write ``colour`` over ``before`` at ``weight`` into ``out``, 0..255."""
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_float = ctypes.c_float
    handle.warlockc_paint_colour_f32(
        _ptr(before, c_float),
        ctypes.c_int64(before_stride),
        _ptr(weight, c_float),
        ctypes.c_int64(weight_stride),
        _ptr(out, c_float),
        ctypes.c_int64(out_stride),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        (c_float * 4)(*(float(v) for v in colour)),
    )


def stack_f32(
    crops: list[Any],
    strides: list[int],
    opacities: list[float],
    modes: list[int],
    out: Any,
    out_stride: int,
    height: int,
    width: int,
    base: Any | None,
    base_stride: int,
) -> None:
    """Fold ``crops`` bottom-first onto ``base`` (or transparent black).

    ``crops`` are uint8 (h, w, 4) *views* into the layers' full canvases and the
    caller has to hold them alive across this call -- the pointer array below
    is built from their data addresses and ctypes keeps no reference to the
    arrays themselves.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_float = ctypes.c_float
    u8_ptr = ctypes.POINTER(ctypes.c_uint8)
    count = len(crops)
    handle.warlockc_stack_f32(
        (u8_ptr * count)(*(_ptr(crop, ctypes.c_uint8) for crop in crops)),
        (ctypes.c_int64 * count)(*strides),
        (c_float * count)(*opacities),
        (ctypes.c_int32 * count)(*modes),
        ctypes.c_int64(count),
        _ptr(out, c_float),
        ctypes.c_int64(out_stride),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        _ptr(base, c_float) if base is not None else None,
        ctypes.c_int64(base_stride),
    )


def to_uint8_f32(pixels: Any, out: Any, count: int) -> None:
    """Scale, round, clamp and narrow ``count`` floats into ``out``.

    Both arrays are C-contiguous and the shape is the caller's business -- this
    one is elementwise, so it sees a flat run.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    handle.warlockc_to_uint8_f32(
        _ptr(pixels, ctypes.c_float),
        _ptr(out, ctypes.c_uint8),
        ctypes.c_int64(count),
    )


def contours(
    mask: Any,
    threshold: int,
    scratch: Any,
    points: Any,
    loop_lens: Any,
) -> int:
    """Trace the closed boundary loops of ``mask >= threshold``.

    ``mask`` is a C-contiguous 2-D uint8 plane; ``scratch`` is a zeroed uint8
    buffer of ``w * (h + 1) + (w + 1) * h`` bytes; ``points`` is int32 with room
    for two values per vertex and ``loop_lens`` int32 with room for one per
    loop. Returns the number of loops, or -1 if either buffer was too small --
    the caller falls back to numpy rather than guessing a bigger one.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    height, width = mask.shape
    return int(
        handle.warlockc_contours(
            _ptr(mask, ctypes.c_uint8),
            ctypes.c_int64(mask.strides[0]),
            ctypes.c_int64(height),
            ctypes.c_int64(width),
            ctypes.c_uint8(threshold),
            _ptr(scratch, ctypes.c_uint8),
            _ptr(points, ctypes.c_int32),
            ctypes.c_int64(points.size // 2),
            _ptr(loop_lens, ctypes.c_int32),
            ctypes.c_int64(loop_lens.size),
        )
    )


if sys.platform != "win32":  # pragma: no cover - the app is Windows-only
    # Not a hard failure: the loader simply will not find a .dll, and every
    # caller falls back. Stated here so the reason is in the module rather
    # than in a puzzled bug report.
    log.debug("warlockc is built as a Windows DLL; other platforms use numpy")

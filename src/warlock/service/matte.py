"""The matte a mesh job will actually be reconstructed from, shown before it.

Two halves, and they answer different questions.

**The preview** is what the promote flow puts in front of the user: the host's
own BiRefNet cutout (or, with no weights, the corner fill) composited over a
checkerboard so transparency is visible as transparency, plus the composition
gate's own verdict on the same image. It exists because the matte is the single
decision that most often turns a good reference into a solid slab, and until
now it was made inside ``trellis-server.exe`` two minutes after the user
committed. Every byte of it is computed off the frame thread -- BiRefNet is
seconds of host compute -- and the result is cached by ``(job id, input.png
mtime)`` under git's racily-clean rule, exactly as ``files.attach_files``
caches its listings and for exactly the same reason: a Windows mtime comes off
a 15.6 ms clock, so a write landing inside the stamped tick would otherwise be
invisible to the stamp forever.

**Approval** is the other half. A reference that carries a real (non-opaque)
alpha channel is a matte somebody already made -- either by hand in Inker, or
by painting one -- and the whole point of having approved it is that it is the
matte trellis reconstructs from. ``trellis-server.exe --help`` states the
server's own rule::

    --bg-removal MODE   threshold | birefnet   (default: auto -- a pre-matted
                        image keeps its alpha; otherwise BiRefNet when its
                        model is present. ...)

So ``auto`` is the preserving mode and ``birefnet`` -- which is
``guidance.DEFAULT_BG_REMOVAL`` on a host that has the weights -- would re-cut
an approved cutout. ``approve`` therefore records ``matte: approved`` *and*
pins ``bg_removal`` to the preserving mode. An explicit override loses to it on
purpose: the user approved that cutout, and a mode that re-mattes would make
the approval a lie.

Pure in the way ``pipelines/reference.py`` is: numpy and Pillow are imported
inside the functions that need them, so importing this module costs nothing on
a host with no image extra.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .core import WarlockService
from .errors import Invalid
from .files import MTIME_RACE_NS
from .validation import check_job_id

# What ``params["matte"]`` says when the alpha on disk is a cutout somebody
# looked at. A string rather than a bool because it is a config-vector value
# (``vectors.VECTOR_PARAMS``) and a future ``matte: auto`` sits beside it.
APPROVED = "approved"

# The mode that keeps an existing alpha channel, per the exe's own help text
# quoted above. Named here rather than spelled "auto" at the call site, so the
# reason travels with the value.
PRESERVING_BG_REMOVAL = "auto"

# The checkerboard the cutout is drawn over, and its cell size in preview
# pixels. Two greys rather than the classic white/grey pair: the references
# this shows are overwhelmingly light subjects on a light background, and a
# white square behind a white rim is exactly the case the checkerboard exists
# to make visible.
CHECKER_LIGHT = 0xB4
CHECKER_DARK = 0x8C
CHECKER_CELL = 8

# The longest side the preview is scaled to. It is drawn in a modal at a few
# hundred design pixels, so a 1024-square source would be three quarters of a
# megabyte of texture spent on detail nothing can show.
PREVIEW_MAX = 384


@dataclass(frozen=True, slots=True)
class Preview:
    """One reference's matte, measured. Writes nothing, transforms nothing."""

    job_id: str
    # ``input.png``'s mtime when the pixels below were read, in ns. Read
    # *before* the read, so the clock the racily-clean rule compares it against
    # is unambiguously later than it.
    stamp: int | None
    width: int
    height: int
    # The cutout over the checkerboard, RGB8, ``width * height * 3`` bytes --
    # ready for a GL texture with no decode on the frame thread.
    rgb: bytes
    # Which of matting.py's three sources answered: alpha | birefnet | flood.
    source: str
    # Whether ``input.png`` already carries a matte somebody made, i.e. whether
    # promoting it now would record ``matte: approved``.
    approved: bool
    # The fraction of the frame the matte keeps. A cutout that kept 2% or 99%
    # of the frame is worth seeing as a number as well as a picture.
    coverage: float
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def stamp_for(path: Path) -> int | None:
    """``path``'s mtime in ns, or None when it is not there."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def cached(cache: dict, job_id: str, stamp: int | None) -> Preview | None:
    """The remembered preview for exactly this version of the file, or None."""
    hit = cache.get(job_id)
    if hit is not None and hit.stamp == stamp:
        return hit
    return None


def remember(cache: dict, preview: Preview) -> bool:
    """Store ``preview`` if its stamp is safely in the past. -> whether it was.

    git's racily-clean rule, and the whole of it. A file's mtime on Windows is
    written from the system clock, whose tick is 15.6 ms unless something has
    asked for better, so a write landing after the pixels were read but still
    inside the stamped mtime's own tick moves nothing the stamp can see -- and
    not for one tick but permanently, because every later comparison keeps
    matching. Here that is a re-save from Inker landing milliseconds after a
    preview was taken: the cutout on screen would then describe pixels that no
    longer exist, and no amount of re-checking would ever notice.

    **The clock is read here rather than in ``preview``, and that ordering is
    the proof**: the hazard needs a write later than the read yet inside the
    mtime's tick, which cannot exist if the read had already finished a tick
    after the mtime. This runs on the frame thread when the task's result is
    adopted, which is strictly after the read finished.

    A refusal costs one recomputation, which is the right way round.
    """
    if preview.stamp is None:
        return False
    if time.time_ns() - preview.stamp <= MTIME_RACE_NS:
        return False
    cache[preview.job_id] = preview
    return True


def replace_stamp(preview: Preview, stamp: int | None) -> Preview:
    """A copy wearing a different stamp -- the racily-clean rule's test hook,
    since a real one cannot be aged without sleeping through a tick."""
    return replace(preview, stamp=stamp)


def preview(svc: WarlockService, job_id: str) -> Preview:
    """Cut ``job_id``'s reference out and report on it. Blocking; off-thread.

    Seconds of host compute when BiRefNet's weights are present, which is why
    every caller goes through the TaskRunner and the frame thread only adopts
    the answer.
    """
    import numpy as np
    from PIL import Image

    from ..pipelines import reference

    src = _reference_path(svc, job_id)
    job = svc.require_job(job_id)

    # Before the read, so ``remember``'s clock is unambiguously later than it.
    stamp = stamp_for(src)
    rgba, source, approved = _cut(svc, src)
    coverage = float(rgba[:, :, 3].mean()) / 255.0

    small = Image.fromarray(rgba, "RGBA")
    small.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    composited = over_checkerboard(np.asarray(small, dtype=np.uint8))
    height, width = composited.shape[:2]

    report = (job.get("params") or {}).get("reference_report")
    if not isinstance(report, dict):
        report = reference.measure_file(src).as_dict()
    return Preview(
        job_id=job_id,
        stamp=stamp,
        width=int(width),
        height=int(height),
        rgb=composited.tobytes(),
        source=source,
        approved=approved,
        coverage=coverage,
        reasons=tuple(str(r) for r in (report.get("reasons") or ())),
        warnings=tuple(str(w) for w in (report.get("warnings") or ())),
    )


def alpha_plane(svc: WarlockService, job_id: str) -> tuple[Any, str]:
    """-> (the full-resolution matte as a uint8 plane, which source cut it).

    What the Inker hand-off applies: 255 keeps a pixel, 0 cuts it. Full
    resolution rather than the preview's, because this one becomes the pixels
    on disk and the preview's is a picture of them. Blocking; off-thread, for
    the reason ``preview`` is.
    """
    rgba, source, _approved = _cut(svc, _reference_path(svc, job_id))
    return rgba[:, :, 3], source


def _reference_path(svc: WarlockService, job_id: str) -> Path:
    check_job_id(job_id)
    src = svc.job_dir(job_id) / "input.png"
    if not src.exists():
        raise Invalid("this job has no reference image")
    return src


def _cut(svc: WarlockService, src: Path) -> tuple[Any, str, bool]:
    """-> (RGBA with the matte as its alpha, the matte's source, approved?).

    One place, because the preview and the hand-off must never disagree about
    where the edge is: the picture the user accepted is the alpha the editor
    opens with.
    """
    import numpy as np
    from PIL import Image

    from ..pipelines import matting

    with Image.open(src) as im:
        im.load()
        approved = matting.is_cutout(im)
        mask, source = matting.mask(im, svc.config)
        rgba = np.dstack(
            [
                np.asarray(im.convert("RGB"), dtype=np.uint8),
                np.asarray(mask, dtype=bool).astype(np.uint8) * np.uint8(255),
            ]
        )
    return rgba, source, approved


def over_checkerboard(rgba: Any) -> Any:
    """Composite an RGBA array onto the checkerboard. -> an RGB uint8 array.

    Done here rather than by drawing a checkerboard behind the image in imgui:
    it is one numpy expression on a 384-square, it keeps the pane to a single
    texture and a single draw, and -- the reason that actually decides it -- it
    makes "what does the user see" assertable in a headless test.
    """
    import numpy as np

    h, w = rgba.shape[:2]
    ys = (np.arange(h) // CHECKER_CELL)[:, None]
    xs = (np.arange(w) // CHECKER_CELL)[None, :]
    board = np.where((ys + xs) % 2 == 0, CHECKER_LIGHT, CHECKER_DARK).astype(np.float32)
    board = np.repeat(board[:, :, None], 3, axis=2)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgba[:, :, :3].astype(np.float32) * alpha + board * (1.0 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def approve(params: dict[str, Any], image: bytes | Path) -> bool:
    """Record an approved matte on ``params`` when ``image`` carries one.

    -> whether it did. Called from both doors onto a mesh job -- a promotion
    and an upload -- because both can arrive carrying a cutout: the promotion
    when the reference was fixed in Inker, the upload when Clay or Inker sent
    drawn pixels straight to 3D. ``files.to_png`` already preserves an alpha
    channel that was there, so nothing here has to make the alpha travel; what
    it has to do is stop the server throwing it away.
    """
    if not is_matted(image):
        return False
    params["matte"] = APPROVED
    # Last write wins over the normalized default (``birefnet`` on a host with
    # the weights), which would re-cut the cutout. See the module docstring for
    # the exe's own statement of what each mode does.
    params["bg_removal"] = PRESERVING_BG_REMOVAL
    return True


def is_matted(image: bytes | Path) -> bool:
    """Whether these pixels carry an alpha channel that is actually a matte.

    Defensive rather than strict: this decides an *addition* to params, so
    bytes that will not decode are simply not a matte -- the decode that
    matters already happened (``to_png``) or is about to (the worker's).
    """
    import io

    from PIL import Image

    from ..pipelines import matting

    try:
        source = io.BytesIO(image) if isinstance(image, bytes) else image
        with Image.open(source) as im:
            im.load()
            return matting.is_cutout(im)
    except Exception:
        return False

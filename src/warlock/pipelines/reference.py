"""What trellis actually sees: measuring and (optionally) normalising the
reference image before it is uploaded.

pipelines/trellis.py streams the file handle verbatim, so until now nothing
host-side knew whether the subject filled 4% or 80% of the frame, ran off the
edge, or was two objects. This module answers that -- and can recentre and
rescale the subject to a target occupancy.

Two rules keep it honest:

* The mask drives *geometry only* and is never written back as alpha.
  Background removal stays trellis's job. A bad mask therefore mis-crops --
  visible in reference.png, and caught by the rejection rules -- instead of
  punching holes in the subject, which is invisible until the mesh is wrong.
* ``prepare(enabled=False)`` is a byte-exact copy plus a measured report. The
  kill switch is a parameter, not a branch at the call site, so the report
  exists either way.

Pure and torch-free: cv2/numpy/Pillow imported inside functions.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

log = logging.getLogger(__name__)

# Fraction of the frame's area the subject should occupy after normalisation,
# and the minimum margin left around it.
DEFAULT_OCCUPANCY = 0.78
DEFAULT_PAD = 0.06

# Rejection thresholds. Deliberately loose -- these catch references that
# cannot reconstruct at all, not references that merely reconstruct badly.
MIN_OCCUPANCY = 0.04
MIN_SECOND_COMPONENT = 0.08
MAX_ASPECT = 8.0

# Per-channel tolerance for the corner flood fill, and the corner patch size
# sampled to find the background colour.
#
# 3 is small because the fill is deliberately *floating range* -- each
# candidate pixel is compared against its already-filled neighbour rather than
# against the seed. That is what lets it follow the soft vertical gradient
# SDXL puts behind every "plain light gray background", which a fixed-range
# fill cannot do at any tolerance. The cost is that the tolerance also decides
# how easily the fill crosses the subject's antialiased rim, so it has to be
# tight.
#
# Measured 2026-08-03 over four real reference images (subject fraction, the
# floating fill, seed gate 32):
#
#     tolerance    crate   lantern   first-aid box   barrel
#     1            0.811   0.459     0.459           0.669
#     2            0.516   0.404     0.404           0.639
#     3            0.514   0.370     0.370           0.635
#     4            0.512   0.205     0.205           0.632
#     8            0.485   0.023     0.023           0.410
#
# 2-3 is the plateau. Above it the white-on-light-gray first-aid box leaks and
# the mask collapses; below it the fill stops early and the "subject" grows to
# most of the frame. 3 sits at the stable end of the plateau. This is a
# heuristic with a real failure mode either side, which is why _leaked() below
# exists rather than the number being trusted on its own.
_FILL_TOLERANCE = 3
# How far a corner may sit from the sampled background before it is not a
# usable seed. Looser than the fill tolerance on purpose: the gradient means
# a corner legitimately differs from the four-corner median.
_SEED_TOLERANCE = 32
_CORNER_PATCH = 16

# Below this share the fill escaped through the subject's rim and ate it,
# which is indistinguishable from a genuine speck and vastly more common --
# so the honest answer is "not measured" rather than a rejection. Set below
# MIN_OCCUPANCY, not at it: between the two the "too small" rule still fires,
# which is where a real speck actually lands.
#
# There is deliberately no ceiling. A fill that takes nothing means the frame
# has no clean background in its corners, and "runs off the edge of the frame"
# is the right verdict for that, not a shrug -- TRELLIS needs a background
# either way.
_LEAK_FLOOR = 0.01


@dataclass(frozen=True, slots=True)
class Report:
    """A measurement of one reference image. Never a transform."""

    ok: bool = True
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    occupancy: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    touches: tuple[str, ...] = ()
    components: int = 0
    alpha_source: bool = False
    size: tuple[int, int] = (0, 0)
    normalised: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "occupancy": self.occupancy,
            "bbox": list(self.bbox) if self.bbox else None,
            "touches": list(self.touches),
            "components": self.components,
            "alpha_source": self.alpha_source,
            "size": list(self.size),
            "normalised": self.normalised,
            **self.extra,
        }


def has_alpha(image: PILImage) -> bool:
    """Whether ``subject_mask`` will read an alpha channel rather than flood fill.

    Exported rather than left inline because a second caller now has to make
    the same decision: ``pipelines/matting.py`` reports *which* source produced
    a matte and that string goes into an export manifest, so a copy of this
    condition that drifted would label a flood fill as ground truth.
    """
    return image.mode in ("RGBA", "LA") or "transparency" in image.info


def subject_mask(image: PILImage):
    """A boolean mask of the subject: the alpha channel when there is one,
    otherwise a corner flood fill of the background.

    The fill samples the four corner patches and takes their median as the
    background colour, which is exactly right for the images PROMPT_TEMPLATE
    produces ("plain light gray background") and degrades to a conservative
    all-subject mask on a busy photograph -- over-including is safe here,
    over-excluding would crop the subject.
    """
    import cv2
    import numpy as np

    if has_alpha(image):
        alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
        return alpha > 8

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    p = min(_CORNER_PATCH, max(1, h // 8), max(1, w // 8))
    corners = np.concatenate(
        [
            rgb[:p, :p].reshape(-1, 3),
            rgb[:p, -p:].reshape(-1, 3),
            rgb[-p:, :p].reshape(-1, 3),
            rgb[-p:, -p:].reshape(-1, 3),
        ]
    )
    bg = np.median(corners, axis=0)

    # A seed that is not itself background makes floodFill a no-op, which
    # would report the whole frame as subject. Seed only from corners that
    # match the sampled background.
    flood = np.zeros((h + 2, w + 2), np.uint8)
    # A writable copy: floodFill takes its image as an output argument, and
    # the array PIL hands back is read-only.
    work = np.array(rgb, dtype=np.uint8, order="C")
    tol = (_FILL_TOLERANCE,) * 3
    for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if np.abs(rgb[y, x].astype(float) - bg).max() <= _SEED_TOLERANCE:
            cv2.floodFill(work, flood, (x, y), (0, 0, 0), tol, tol, cv2.FLOODFILL_MASK_ONLY | 8)
    background = flood[1:-1, 1:-1] > 0
    return ~background


def _leaked(mask, alpha_source: bool) -> bool:
    """Whether the mask is a measurement or an artefact of the fill.

    Only ever true for a flood-filled mask: an alpha channel is ground truth
    and a genuinely tiny sprite is a real thing to report.
    """
    if alpha_source:
        return False
    return float(mask.mean()) < _LEAK_FLOOR


def _components(mask) -> list[int]:
    """Pixel counts of the mask's connected components, largest first."""
    import cv2
    import numpy as np

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return []
    sizes = np.bincount(labels.ravel(), minlength=count)[1:]
    return sorted((int(s) for s in sizes if s), reverse=True)


def unmeasured(reason: str) -> Report:
    """A report that makes no claim.

    ``ok`` is True deliberately: the rejection rules are about *composition*
    (nothing there, too small, cropped, two objects), and none of them can be
    evaluated on bytes that will not decode. Refusing here would turn "we
    could not measure this" into "this cannot reconstruct", which is a
    different statement -- and trellis is the authority on whether it can read
    a file at all. The warning still lands in params, so it is visible.
    """
    return Report(ok=True, warnings=(reason,), extra={"measured": False})


def measure(image: PILImage) -> Report:
    """Measure ``image``. Writes nothing, transforms nothing."""
    import numpy as np

    w, h = image.size
    mask = subject_mask(image)
    area = float(w * h) or 1.0
    total = int(mask.sum())
    alpha_source = image.mode in ("RGBA", "LA") or "transparency" in image.info

    if total == 0:
        return Report(
            ok=False,
            reasons=("No subject found against the background.",),
            size=(w, h),
            alpha_source=alpha_source,
        )
    if _leaked(mask, alpha_source):
        # The separation failed, so nothing below it means anything. Reported
        # as unmeasured rather than as a rejection: a light subject on a light
        # background is exactly what PROMPT_TEMPLATE produces, and refusing
        # those would refuse a large share of perfectly good references.
        return unmeasured(
            "The subject could not be separated from the background; not measured."
        )

    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = (x0, y0, x1 + 1, y1 + 1)
    occupancy = total / area

    touches = tuple(
        name
        for name, hit in (
            ("left", x0 == 0),
            ("right", x1 == w - 1),
            ("top", y0 == 0),
            ("bottom", y1 == h - 1),
        )
        if hit
    )
    sizes = _components(mask)
    components = len(sizes)

    reasons: list[str] = []
    warnings: list[str] = []

    if occupancy < MIN_OCCUPANCY:
        reasons.append(
            f"The subject fills only {occupancy * 100:.0f}% of the frame; "
            "TRELLIS needs it much larger."
        )
    opposite = ("left" in touches and "right" in touches) or (
        "top" in touches and "bottom" in touches
    )
    if len(touches) >= 3 or opposite:
        reasons.append("The subject runs off the edge of the frame.")
    if len(sizes) >= 2 and sizes[1] >= sizes[0] * MIN_SECOND_COMPONENT:
        reasons.append("There is more than one object in the reference.")

    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    aspect = max(bw, bh) / max(1, min(bw, bh))
    if aspect > MAX_ASPECT:
        warnings.append(
            f"The subject is {aspect:.0f}:1 -- very thin subjects reconstruct poorly."
        )
    if not reasons and len(touches) in (1, 2) and not opposite:
        warnings.append("The subject touches the edge of the frame.")

    return Report(
        ok=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        occupancy=occupancy,
        bbox=bbox,
        touches=touches,
        components=components,
        alpha_source=alpha_source,
        size=(w, h),
    )


def normalise(
    image: PILImage,
    *,
    occupancy: float = DEFAULT_OCCUPANCY,
    pad: float = DEFAULT_PAD,
    background: tuple[int, int, int] | None = None,
) -> tuple[PILImage, Report]:
    """Recentre and rescale the subject to ``occupancy`` of a square frame.

    Occupancy is measured against the subject's *bounding box*, not its mask
    area: a thin sword at 78% mask area would be scaled until it burst the
    frame. Returns the untouched image (and its report) when there is nothing
    to normalise.
    """
    from PIL import Image

    report = measure(image)
    if report.bbox is None:
        return image, report

    src = image.convert("RGBA") if report.alpha_source else image.convert("RGB")
    if background is None:
        background = _border_colour(src)

    x0, y0, x1, y1 = report.bbox
    subject = src.crop((x0, y0, x1, y1))
    side = max(image.size)
    target = max(1, round(side * (occupancy ** 0.5) * (1.0 - 2 * pad)))
    scale = target / max(subject.width, subject.height)
    nw = max(1, round(subject.width * scale))
    nh = max(1, round(subject.height * scale))
    subject = subject.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new(src.mode, (side, side), background + ((0,) if src.mode == "RGBA" else ()))
    box = ((side - nw) // 2, (side - nh) // 2)
    canvas.paste(subject, box, subject if src.mode == "RGBA" else None)

    out_report = measure(canvas)
    return canvas, Report(
        ok=out_report.ok,
        reasons=out_report.reasons,
        warnings=out_report.warnings,
        occupancy=out_report.occupancy,
        bbox=out_report.bbox,
        touches=out_report.touches,
        components=out_report.components,
        alpha_source=out_report.alpha_source,
        size=out_report.size,
        normalised=True,
        extra={"source_occupancy": report.occupancy, "source_size": list(report.size)},
    )


def _border_colour(image: PILImage) -> tuple[int, int, int]:
    """Median colour of the outermost ring -- the background trellis will see."""
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    ring = np.concatenate(
        [arr[0].reshape(-1, 3), arr[-1].reshape(-1, 3), arr[:, 0], arr[:, -1]]
    )
    return tuple(int(v) for v in np.median(ring, axis=0))  # type: ignore[return-value]


def measure_file(src: Path) -> Report:
    from PIL import Image

    try:
        with Image.open(src) as im:
            im.load()
            return measure(im)
    except Exception as exc:
        log.warning("could not measure reference %s: %s", src, exc)
        return unmeasured(f"The reference could not be read for measurement ({exc}).")


def prepare(
    src: Path,
    dest: Path,
    *,
    occupancy: float = DEFAULT_OCCUPANCY,
    enabled: bool = True,
) -> Report:
    """Produce the image trellis will actually be handed, and report on it.

    With ``enabled=False`` this is a byte-exact copy: the audit artifact and
    the rejection rules ship before the transform does, because whether
    host-side normalisation helps or fights the exe's own preprocessing is
    unmeasured (see DEFAULT_REFERENCE_PREP).
    """
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")

    report = None
    if enabled:
        try:
            with Image.open(src) as im:
                im.load()
                out, report = normalise(im, occupancy=occupancy)
                out.save(tmp, format="PNG")
        except Exception as exc:
            # Same rule as measure_file: an unreadable file is handed on
            # verbatim rather than failed here, and trellis decides.
            log.warning("could not normalise reference %s: %s", src, exc)
            report = None
    if report is None:
        shutil.copyfile(src, tmp)
        report = measure_file(src)

    os.replace(tmp, dest)
    return report

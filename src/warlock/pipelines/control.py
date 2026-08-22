"""Structure hints for ControlNet: pure, deterministic, torch-free.

Same split as pipelines/sheet.py -- everything decidable lives here and is
testable with a Pillow-drawn fixture and no GPU. cv2 is imported inside the
functions so this module stays importable without the text2image extra.

Canny goes through cv2 rather than Pillow's own edge filter on purpose: the
published controlnet-canny-sdxl-1.0 was trained on OpenCV Canny output, so
its model card's threshold advice only transfers if the hint comes from the
same operator.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

DEFAULT_LOW = 100
DEFAULT_HIGH = 200

# Kinds control.hint() itself can produce. A depth hint would need a torch
# model and so would have to be a module of its own: this one must stay
# torch-free, which test_offline.py asserts exactly. No such module exists yet.
PREPROCESSORS = ("canny",)


def _letterbox(image: PILImage, size: int) -> PILImage:
    """Fit ``image`` into a size x size RGB frame on black, preserving aspect.

    ControlNet wants the hint at the generation resolution exactly. Padding
    rather than cropping keeps the hint aligned with what the user supplied --
    a crop would silently promise structure that the reference never had.
    """
    from PIL import Image

    src = image.convert("RGB")
    scale = min(size / src.width, size / src.height)
    w = max(1, round(src.width * scale))
    h = max(1, round(src.height * scale))
    resized = src.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - w) // 2, (size - h) // 2))
    return canvas


def canny(
    image: PILImage,
    *,
    low: int = DEFAULT_LOW,
    high: int = DEFAULT_HIGH,
    size: int | None = None,
) -> PILImage:
    """An RGB Canny edge map, optionally letterboxed to ``size`` x ``size``.

    Edges are detected *before* padding, so the black bars never register as a
    frame-shaped rectangle of structure the model then tries to honour.
    """
    import cv2
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    edges = cv2.Canny(np.array(rgb), low, high)
    out = Image.fromarray(np.stack([edges] * 3, axis=-1), mode="RGB")
    return _letterbox(out, size) if size else out


def hint(
    kind: str,
    image: PILImage,
    *,
    size: int,
    low: int = DEFAULT_LOW,
    high: int = DEFAULT_HIGH,
) -> PILImage:
    """Dispatch on preprocessor name. Unknown (including "depth") raises."""
    if kind != "canny":
        raise ValueError(
            f"unsupported control preprocessor {kind!r}; "
            f"this module handles {', '.join(PREPROCESSORS)}"
        )
    return canny(image, low=low, high=high, size=size)


def edge_fraction(image: PILImage) -> float:
    """Share of pixels the detector marked as an edge.

    Recorded with every hint so "my silhouette lock did nothing" is an
    answerable question: a near-zero fraction means the thresholds found no
    structure, not that the ControlNet was ignored.
    """
    import numpy as np

    arr = np.asarray(image.convert("L"))
    if arr.size == 0:
        return 0.0
    return float((arr > 0).sum()) / float(arr.size)


def write_hint(
    src: Path,
    dest: Path,
    *,
    kind: str,
    size: int,
    low: int = DEFAULT_LOW,
    high: int = DEFAULT_HIGH,
) -> dict[str, Any]:
    """Preprocess ``src`` into ``dest`` and return the hint's provenance.

    Staged through a temp name and renamed, the same rule every other write
    onto a served path follows (optimize.staged_copy, rigging.finalize_rig):
    the file route may read this while a rerun is rewriting it.

    The ``finally`` is the other half of that rule, and it is the half this had
    missing: ``postprocess._staged``, ``trellis._atomic_write`` and
    ``optimize.run`` all unlink theirs. A raising ``save`` left the dotfile
    behind -- never a half-written served file, so this is consistency rather
    than correctness, but a stranded fragment is a fragment nothing will ever
    clean up.
    """
    from PIL import Image

    with Image.open(src) as im:
        im.load()
        out = hint(kind, im, size=size, low=low, high=high)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    try:
        out.save(tmp, format="PNG")
        os.replace(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return {
        "kind": kind,
        "low": low,
        "high": high,
        "size": size,
        "edge_fraction": edge_fraction(out),
    }


def hint_report(path: Path) -> dict[str, Any]:
    """The provenance block for a hint some other code already drew.

    Same shape :func:`write_hint` returns, so a job's ``control_hint`` means
    one thing whoever produced the image, and with ``edge_fraction`` measured
    over the file that will actually condition the generation -- which is the
    only reading that answers "did my guide draw anything".

    ``kind`` says ``"guide"`` rather than a preprocessor name because no
    preprocessor ran: the caller handed the ControlNet line art directly.
    """
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        return {
            "kind": "guide",
            "size": max(im.size),
            "edge_fraction": edge_fraction(im),
        }

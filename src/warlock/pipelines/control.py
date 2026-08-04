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

# Kinds control.hint() itself can produce. "depth" is a separate module
# (pipelines/depth.py) because it needs a torch model, and this one must stay
# torch-free -- test_offline.py asserts exactly that.
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
    """
    from PIL import Image

    with Image.open(src) as im:
        im.load()
        out = hint(kind, im, size=size, low=low, high=high)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    out.save(tmp, format="PNG")
    os.replace(tmp, dest)
    return {
        "kind": kind,
        "low": low,
        "high": high,
        "size": size,
        "edge_fraction": edge_fraction(out),
    }

"""Regenerating a selection with the image model: the arithmetic.

Pure numpy and Pillow, like the rest of this package. What leaves the editor
is a crop of the flattened canvas around the selection plus the selection's
own coverage as a mask, both sized for SDXL; what comes back is resized to
the crop and blended into the layer by the selection's weight through
``Document.apply_pixels`` -- the same ``masked_apply`` rule every other
selection-bounded write follows, so a feathered edge fades the regeneration
in exactly as it fades a filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: How far past the selection's bounds the crop reaches, so the model sees
#: the surroundings it has to match. In canvas pixels, clamped to the canvas.
MARGIN = 32

#: The long side the crop is resized to before it is sent. SDXL's native
#: frame; the short side follows at the same scale, rounded to the VAE's
#: stride.
SEND_LONG_SIDE = 1024
STRIDE = 64

#: Denoise strength for a regeneration. Higher than the reference form's
#: default: the mask already confines the change, so the model can be allowed
#: to invent more inside it.
DEFAULT_STRENGTH = 0.6


def crop_box(
    bounds: tuple[int, int, int, int], size: tuple[int, int], margin: int = MARGIN
) -> tuple[int, int, int, int]:
    """The selection's bounds grown by ``margin`` and clamped to the canvas."""
    width, height = size
    x0, y0, x1, y1 = bounds
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def send_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    """The size the crop is resized to: long side ``SEND_LONG_SIDE``, short
    side at the same scale rounded to ``STRIDE``, never below one stride."""
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    scale = SEND_LONG_SIDE / max(w, h)
    sw = max(STRIDE, int(round(w * scale / STRIDE)) * STRIDE)
    sh = max(STRIDE, int(round(h * scale / STRIDE)) * STRIDE)
    return sw, sh


def prepare(
    flat: np.ndarray, mask: np.ndarray, bounds: tuple[int, int, int, int]
) -> tuple[bytes, bytes, tuple[int, int, int, int]]:
    """-> (crop PNG bytes, mask PNG bytes, the box the crop covers).

    The crop is RGB over the flattened canvas -- transparent canvas reads as
    black, which is what the model sees. The mask is the selection's coverage
    (white = regenerate), sent at the crop's resized size.
    """
    from PIL import Image

    height, width = flat.shape[:2]
    box = crop_box(bounds, (width, height))
    x0, y0, x1, y1 = box
    crop = Image.fromarray(np.ascontiguousarray(flat[y0:y1, x0:x1, :3]), "RGB")
    weight = Image.fromarray(np.ascontiguousarray(mask[y0:y1, x0:x1]), "L")
    size = send_size(box)
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    weight = weight.resize(size, Image.Resampling.BILINEAR)
    return _png(crop), _png(weight), box


def fit_back(image: Any, box: tuple[int, int, int, int]) -> np.ndarray:
    """The model's picture resized to the crop's box, as RGBA uint8."""
    from PIL import Image

    x0, y0, x1, y1 = box
    out = image.convert("RGBA").resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    return np.asarray(out, dtype=np.uint8).copy()


def _png(image: Any) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

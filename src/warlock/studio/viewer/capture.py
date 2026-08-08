"""Turning a rendered viewport into a PNG.

One place, used by the thumbnail, the screenshot, the sheet preview and the
golden-image tests -- so a row-order or alpha bug is one bug rather than four.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from .glctx import Viewport


def image(viewport: Viewport) -> Any:
    """The viewport's current contents as an RGBA PIL image, top row first.

    The row flip is folded into Pillow's raw decoder (orientation -1 reads
    bottom-up) rather than done as a numpy slice-and-copy first (D41): one
    copy instead of two on a path the frame thread pays for thumbnails.
    """
    from PIL import Image

    return Image.frombuffer(
        "RGBA", viewport.size, viewport.read_raw(), "raw", "RGBA", 0, -1
    )


def png_bytes(viewport: Viewport, *, opaque: bool = True) -> bytes:
    """PNG bytes for the current frame.

    Opaque by default: a thumbnail is shown against the library's own
    background and a transparent one would let the card show through the model.
    The sheet preview is the exception and asks for alpha.
    """
    img = image(viewport)
    if opaque:
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    return buffer.getvalue()


def save_png(viewport: Viewport, path: Path, *, opaque: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(viewport, opaque=opaque))
    return path

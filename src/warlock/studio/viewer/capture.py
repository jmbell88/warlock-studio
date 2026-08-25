"""Turning a rendered viewport into a PNG.

One place, used by the thumbnail, the screenshot, the sheet preview and the
golden-image tests -- so a row-order or alpha bug is one bug rather than four.
"""

from __future__ import annotations

import io
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


# There was a ``save_png(viewport, path)`` here, and the only caller it had
# anywhere was the test that tested it. Every real capture path -- the
# thumbnail, the screenshot overlay, the sheet preview, the golden-image tests
# -- takes ``png_bytes`` and hands the bytes to whatever owns the destination,
# which is what lets those writes be staged. This one wrote in place, which is
# the opposite rule, so it was a trap sitting in a module four callers import.

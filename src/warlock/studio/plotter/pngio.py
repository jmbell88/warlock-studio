"""The one RGBA-to-PNG spelling in the repo.

Four byte-identical copies of this function existed -- ``plotter/tmx.py``,
``plotter/wmap.py``, ``packwright/compose.py`` and ``packwright/wpack.py`` --
and every one of them is on a *determinism* path: a ``.wmap`` and a ``.wpack``
have to be byte-identical across two saves of an unchanged document, and a
``.tmx`` export has to be byte-identical across two exports. Four copies is four
places for a compression level, an ICC profile or a Pillow version guard to be
added to one and not the others, at which point one of those guarantees quietly
becomes a claim about which writer happened to run.

Here rather than in ``packwright/`` because the dependency already runs this
way: ``packwright.tsxout`` reaches for ``plotter.tsx`` and ``tilegrid.tileset``,
and nothing in ``plotter/`` reaches back.

Pillow is imported inside the function, the rule the whole package follows:
importing it costs a tenth of a second and most of this package never touches a
pixel.
"""

from __future__ import annotations

import io

import numpy as np


def png_bytes(pixels: np.ndarray) -> bytes:
    """One RGBA array as the bytes of a PNG file."""
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8), "RGBA").save(out, "PNG")
    return out.getvalue()

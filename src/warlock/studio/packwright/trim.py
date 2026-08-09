"""Where a sprite's alpha actually stops.

One definition of "trim" in the repo, and this is not it -- ``pipelines.sheet.
measure_trim`` is, and has been since sprite sheets got a sidecar. What is here
is the same answer computed on a numpy array instead of a PIL image, because a
packer measures hundreds of sprites and decoding each into a PIL image to ask is
the wrong price for a question numpy answers directly.

The two are pinned against each other by ``tests/packwright/test_trim.py``. That
parity test is the whole justification for a second implementation existing at
all: without it this is a fork, and a packer whose idea of the subject differs
by a pixel from the sidecar's produces atlases that are correct and metadata
that is not.

**A fully transparent sprite is not dropped.** It packs as a 1x1 empty, and it
is marked ``empty`` so the caller can say so. A blank frame in the middle of a
clip is a real frame -- it is the pause -- and silently removing it renumbers
every frame after it.
"""

from __future__ import annotations

import numpy as np

# What an empty sprite occupies. One pixel rather than zero, because a zero-area
# rectangle is a special case for every consumer downstream: the packer, the
# compositor, and every engine reading the JSON.
EMPTY_SIZE = 1


def alpha_bbox(pixels: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(x, y, w, h)`` of the non-transparent region, or ``None`` if there is
    none. Row and column reductions rather than ``nonzero``, which would
    materialise one index pair per opaque pixel."""
    alpha = np.asarray(pixels)[..., 3]
    rows = np.any(alpha, axis=1)
    if not rows.any():
        return None
    columns = np.any(alpha, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(columns))
    x1 = int(len(columns) - np.argmax(columns[::-1]))
    return x0, y0, x1 - x0, y1 - y0


def trim_rect(pixels: np.ndarray, *, enabled: bool = True) -> tuple[int, int, int, int, bool]:
    """The rectangle to pack, plus whether the sprite was empty.

    With trimming off this is the whole image, which keeps the caller free of a
    branch: an untrimmed sprite is one whose trim rectangle happens to be its
    own bounds, and every consumer of ``spriteSourceSize`` then works unchanged.
    """
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    box = alpha_bbox(pixels)
    if box is None:
        return 0, 0, EMPTY_SIZE, EMPTY_SIZE, True
    if not enabled:
        return 0, 0, width, height, False
    return (*box, False)

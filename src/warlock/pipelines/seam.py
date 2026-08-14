"""Does this image tile, and by how much does it fail to?

The measurement is a ratio and that is the whole design. A hard number -- the
mean absolute difference between the first and last column -- says nothing on
its own, because a photograph of gravel legitimately differs by a lot between
*any* two adjacent columns while flat plaster differs by almost nothing. So the
wrap seam is divided by the mean interior difference: a value near 1 means the
seam is no more of a discontinuity than the texture already contains, and a
value of 8 means it is eight times worse than the picture's own grain.

Advisory, like every other measurement in this codebase. Nothing here fails a
job whose PNG is already on disk; the number goes on the row and the user
decides.

Pure: Pillow and NumPy inside the functions, no torch, no service imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Above this the seam is a visible edge rather than part of the texture.
#
# Measured, not guessed: docs/measurements/2026-08-08-seam-threshold.md. 72
# units on sdxl-turbo -- 48 generated through the circular-padding path and 24
# identical prompts and seeds with it off -- put the highest legitimately
# seamless tile at 2.50 and the lowest visible seam at 5.52, an empty band whose
# geometric centre is 3.72. 3.5 is the round value inside it, and it classifies
# every one of the 72 correctly where the previous 2.0 raised two false alarms.
#
# Both false alarms were large flat cells separated by thin hard lines (ceramic
# grout, metal ribs), which is the shape that breaks the ratio: the denominator
# is a mean over every adjacent pair and so is tiny on a mostly-flat picture,
# while the numerator is one column that may land on a grout line. Re-measure
# before moving it, and re-measure it per checkpoint -- the corpus is turbo at
# 4 steps, and a CFG base at 30 draws harder edges than any of it.
SEAM_MAX = 3.5

# Below this many pixels on a side there is no interior to compare against.
MIN_SIDE = 8


def _ratios(arr: Any) -> tuple[float, float]:
    import numpy as np

    def axis_ratio(a: Any) -> float:
        # a is (rows, columns, channels); the wrap seam is the first column
        # against the last, the interior is every adjacent pair.
        edge = float(np.abs(a[:, 0].astype(float) - a[:, -1].astype(float)).mean())
        interior = float(np.abs(np.diff(a.astype(float), axis=1)).mean())
        if interior <= 0.0:
            # A flat image has no grain to normalise against, and zero is the
            # honest answer: an image that is one colour tiles perfectly. The
            # second arm cannot fire while ``interior`` is a *mean* of absolute
            # differences -- a mean of zero means every adjacent pair is
            # identical, which makes the first column equal to the last and the
            # edge zero as well. It stays because that implication is a
            # property of the statistic and not of the idea: swap the mean for
            # a median (tempting, to stop one bright speck dominating a flat
            # texture) and a mostly-flat image with one hard join lands here
            # with a real seam, where returning 0.0 would call it seamless.
            return 0.0 if edge <= 0.0 else float("inf")
        return edge / interior

    horizontal = axis_ratio(arr)
    vertical = axis_ratio(arr.transpose(1, 0, 2))
    return (horizontal, vertical)


def report(path: Path) -> dict[str, Any]:
    """The seam verdict for one image.

    ``horizontal`` compares the left edge against the right, ``vertical`` the
    top against the bottom, and the worse of the two decides -- a tile that
    wraps one way and not the other is not a tile.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        if min(im.size) < MIN_SIDE:
            raise ValueError(f"{path.name} is too small to measure a seam in")
        arr = np.asarray(im.convert("RGB"))
    horizontal, vertical = _ratios(arr)
    worst = max(horizontal, vertical)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "worst": worst,
        "seamless": bool(worst <= SEAM_MAX),
        "threshold": SEAM_MAX,
    }


def wrap_preview(src: Path, dest: Path) -> Path:
    """The image rolled by half its size in both axes.

    What was the wrap seam is now a cross through the middle of the frame,
    which is the only way to *see* the failure the ratio above measures -- an
    edge is invisible at the edge of a picture and obvious in the centre of
    one.
    """
    import numpy as np
    from PIL import Image

    with Image.open(src) as im:
        im.load()
        arr = np.asarray(im.convert("RGBA" if "A" in im.getbands() else "RGB"))
    rolled = np.roll(arr, (arr.shape[0] // 2, arr.shape[1] // 2), axis=(0, 1))
    out = Image.fromarray(rolled)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest, "PNG")
    return dest

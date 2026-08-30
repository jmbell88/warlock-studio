"""Does this image tile, and by how much does it fail to?

The measurement is a ratio and that is the whole design. A hard number -- the
mean absolute difference between the first and last column -- says nothing on
its own, because a photograph of gravel legitimately differs by a lot between
*any* two adjacent columns while flat plaster differs by almost nothing. So the
wrap seam is divided by what the picture's own interior does.

**Which interior statistic divides it is the question three documents took to
settle.** Dividing by the interior *mean* asks "is the seam worse than this
picture's average join", and that is the shipped ratio (``worst``, kept below
because three published corpora are keyed on it). Dividing by the interior
*maximum* asks "is the seam the worst join in this picture", and that is
``dominance``, which decides the verdict as of 2026-08-30. The difference is one
word and it is the whole reason for the change: a texture of flat cells parted
by thin hard lines -- pixel art, ceramic grout, riveted panels -- has an average
that collapses toward zero while its maximum does not, so the mean-normalised
ratio inflates on exactly the population the seamless-tileset track generates.
On a held-out corpus it called 18 of 72 confirmed-seamless tiles seamed, 15 of
them under the pixel-art LoRA the track ships with; dominance called 0 of 72.

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
# while the numerator is one column that may land on a grout line.
#
# It has since been re-measured twice on the shipped default and it did not
# move either time, so "re-measure per checkpoint" is a settled question rather
# than an open one -- what those runs found is a limit of the *statistic*, and
# a third checkpoint corpus would find it again:
#
# - docs/measurements/2026-08-09-seam-threshold-cfg.md (run 2026-08-13),
#   sdxl_cfg at 30 steps, the same 72 units. The populations overlap and its
#   refusal rule fired: a wrap-preview-confirmed seamless tile scored 4.288
#   while the lowest visibly seamed unit scored 2.705. No value separates them.
# - docs/measurements/2026-08-29-seam-threshold-cfg.md, which reproduces that
#   corpus bit-identically (max delta 0.000000) and adds the population the
#   seamless-tileset track actually generates -- sdxl_cfg under the pixelxl
#   LoRA with SHEET_NEGATIVE_PROMPT, which neither earlier corpus contained.
#   There the ratio is *inverted*: 20 of 24 wrap-preview-confirmed seamless
#   tiles score above this constant, while the two lowest-scoring units in the
#   corpus are un-tiled pictures visibly chopped in four. The padding is fine;
#   the mean denominator collapses on flat-cell pixel art.
#
# That better denominator is now measured and shipped -- see
# ``SEAM_DOMINANCE_MAX`` below and
# docs/measurements/2026-08-30-seam-dominance.md -- so this constant no longer
# decides anything. It stays because ``worst`` is still reported beside the new
# verdict and because every row written before 2026-08-30 was judged against
# it; a stored report with no ``metric`` field is one of those, and
# ``inspector.seam_verdict`` still reads it as an edge/grain number.
SEAM_MAX = 3.5

# The verdict, as of 2026-08-30. Above this the wrap seam is the single largest
# discontinuity in the picture, which is the arithmetic spelling of "this does
# not tile".
#
# **Fixed by construction rather than fitted**, and that is the point of it:
# at exactly 1.0 the seam is precisely as large as the largest step the picture
# already contains, so the number is the statistic's own semantics and cannot
# drift with a corpus. docs/measurements/2026-08-30-seam-dominance.md
# pre-registered it before drawing a single held-out unit and then tried to
# falsify it: 144 tiled axes at seeds no published corpus contains -- plain,
# pixel-art-LoRA'd and hard-structured -- and **not one scored above 1.0**
# (highest 0.940). Every unit above 0.8 was read through ``wrap_preview`` and
# every one wraps.
#
# ``<=`` and not ``<``, and it is load-bearing: a triangle wave is the ideal
# seamless tile and its seam step equals its interior step exactly, so a strict
# comparison would false-alarm on the canonical good case.
#
# What it costs, measured on the same corpus and recorded rather than
# discovered later: dominance is a specificity instrument, and it misses a
# seamed picture whose interior already contains a step as hard as the seam --
# 4 of 44 visibly seamed control units, against the ratio's 5. Pooled it caught
# 40 where the ratio caught 39, which is the whole of its sensitivity claim and
# is deliberately not more than that.
SEAM_DOMINANCE_MAX = 1.0

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


def _dominance(arr: Any) -> tuple[float, float]:
    """The verdict statistic: the wrap seam against the *largest* interior step.

    Same numerator as ``_ratios`` and a different denominator, which is the
    whole of the difference. Each interior adjacent-column pair is reduced to
    one number the same way the seam is -- mean absolute difference over rows
    and colour channels -- and the largest of those is what the seam is
    measured against. So the question is "is the wrap the worst join in this
    picture", and a value of 1.0 means it ties with the worst.

    The flat-image arm is ``_ratios``' arm and is reachable for the same reason
    and no other: a maximum of zero means every adjacent pair is identical,
    which makes the first column equal to the last.
    """
    import numpy as np

    def axis_dominance(a: Any) -> float:
        a = a.astype(float)
        edge = float(np.abs(a[:, 0] - a[:, -1]).mean())
        pairs = np.abs(np.diff(a, axis=1)).mean(axis=(0, 2))
        interior = float(pairs.max()) if pairs.size else 0.0
        if interior <= 0.0:
            return 0.0 if edge <= 0.0 else float("inf")
        return edge / interior

    return (axis_dominance(arr), axis_dominance(arr.transpose(1, 0, 2)))


def report(path: Path) -> dict[str, Any]:
    """The seam verdict for one image.

    ``horizontal`` compares the left edge against the right, ``vertical`` the
    top against the bottom, and the worse of the two decides -- a tile that
    wraps one way and not the other is not a tile.

    Two statistics are reported and only one decides. ``dominance`` is the
    verdict (docs/measurements/2026-08-30-seam-dominance.md); ``worst`` is the
    edge-against-mean-grain ratio three published corpora are keyed on, kept so
    those documents stay readable against new output and so a row's number does
    not silently change meaning. ``metric`` names which one decided, and its
    absence marks a row written before 2026-08-30 -- which is what lets
    ``inspector.seam_verdict`` word an old row correctly instead of describing
    it with today's vocabulary.
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
    dom_h, dom_v = _dominance(arr)
    dominance = max(dom_h, dom_v)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "worst": worst,
        "dominance_horizontal": dom_h,
        "dominance_vertical": dom_v,
        "dominance": dominance,
        "metric": "dominance",
        "seamless": bool(dominance <= SEAM_DOMINANCE_MAX),
        "threshold": SEAM_DOMINANCE_MAX,
    }


#: The seam-erase pass: how wide a band around the wrapped seam cross is
#: regenerated (as a fraction of the side), and how hard. A band, not a line:
#: the model has to blend across the join, so it needs the join's neighbours.
ERASE_BAND = 0.125
ERASE_STRENGTH = 0.5


def roll_half(image: Any) -> Any:
    """The image rolled by half in both axes -- the wrap seam becomes a
    visible cross through the centre. Its own inverse."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(image)
    h, w = arr.shape[:2]
    return Image.fromarray(np.roll(np.roll(arr, h // 2, axis=0), w // 2, axis=1))


def cross_mask(size: tuple[int, int], band: float = ERASE_BAND) -> Any:
    """White where a rolled tile's seam cross is, black elsewhere -- the mask
    a seam-erase inpaint pass regenerates. Feathered by a linear ramp over
    half the band so the join blends rather than steps."""
    import numpy as np
    from PIL import Image

    w, h = size
    half_w = max(1, int(w * band / 2))
    half_h = max(1, int(h * band / 2))
    ys = np.abs(np.arange(h) - h // 2)
    xs = np.abs(np.arange(w) - w // 2)
    ramp_y = np.clip(1.0 - (ys - half_h / 2) / max(half_h / 2, 1), 0.0, 1.0)
    ramp_x = np.clip(1.0 - (xs - half_w / 2) / max(half_w / 2, 1), 0.0, 1.0)
    mask = np.maximum(ramp_y[:, None], ramp_x[None, :])
    return Image.fromarray((mask * 255).astype(np.uint8), "L")


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

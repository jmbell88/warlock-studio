"""Pixel-art processing and measurement: grids, palettes, dithering, cleanup.

The thing this module exists for is the *grid*. A pixel-art generator draws
logical pixels as square blocks -- typically 8x8 -- on a 1024 canvas, and that
block lattice has both a period and a **phase**: the first cell rarely starts at
x=0. Reducing such an image by cropping to the subject first and then scaling by
an arbitrary ratio destroys it twice over, once by moving the origin off the
lattice and once by sampling at non-cell-centre positions, and what comes out is
a small photograph of pixel art rather than pixel art. So the phase is measured
(``detect_grid``) and the reduction is taken at cell centres on the *full frame*
(``downscale_grid``); cropping happens afterwards, in logical space.

Processing and measurement live together on purpose: the bench's pixel metrics
(``bench/metrics.score_reference``) and the export path have to agree about what
"on the grid" means, and two implementations of that would drift.

Pure in the way ``seam.py`` is: stdlib at module level, NumPy and Pillow inside
the functions, no imports from ``service``/``queue``/``studio``. Nothing here
writes a file or decides where one goes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

RGB = tuple[int, int, int]

# Cell sizes a generator plausibly draws on a 1024 canvas. Integers only in v1:
# a fractional period would need sub-pixel resampling to reduce, which is the
# blending this whole module exists to avoid. 8 is what pixel-art-xl produces.
GRID_SCALES: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 16)

# Normalized within-cell variance below which the lattice is considered real.
# Provisional: Phase C's bench run over bench/suites/pixel-v1 calibrates it from
# the observed separation between an on-grid generation and an off-grid one, and
# docs/measurements/2026-08-06-pixel-art-xl.md is that document (procedure
# pre-registered, run not yet taken) and comes before it moves -- the pattern
# trellis_band and mesh_hole_max both set.
GRID_RESIDUAL_MAX = 0.05

# Alpha is binary in pixel art: a cell is drawn or it is not. The threshold is
# the midpoint rather than anything tuned, because the input is either already
# binary (a cutout) or a resample artefact.
ALPHA_THRESHOLD = 128

# Below this a "grid" is a handful of cells and the statistic is noise.
_MIN_CELLS = 4


@dataclass(frozen=True, slots=True)
class PixelOpts:
    """Everything about a pixel export that is not its size.

    Frozen and defaulted so that ``PixelOpts()`` is exactly the legacy export:
    no palette file, no dither, no cleanup, and ``colors`` carrying the old
    median-cut cap. Callers compare a stored manifest against these fields to
    decide staleness, which is why ``palette_hash`` is content-derived -- a
    palette file edited in place keeps its name and must still re-derive.
    """

    colors: int = 0
    palette_name: str | None = None
    palette: tuple[RGB, ...] | None = None
    palette_hash: str | None = None
    dither: bool = False
    cleanup: bool = False


# --------------------------------------------------------------------------
# Palette files
# --------------------------------------------------------------------------

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex(text: str) -> tuple[RGB, ...]:
    """A Lospec ``.hex`` palette: one ``rrggbb`` per line.

    Blank lines and ``;``/``#!``-style comments are skipped rather than
    rejected -- a palette downloaded from anywhere real carries a header line,
    and refusing the file over it would be a worse answer than ignoring it. A
    line that looks like it is *trying* to be a colour and is not still raises,
    because silently dropping one entry changes what every mapped pixel becomes.
    """
    colors: list[RGB] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "//")) or line.startswith("#!"):
            continue
        match = _HEX_RE.match(line)
        if match is None:
            raise ValueError(f"not a hex colour: {line!r}")
        value = match.group(1)
        colors.append(
            (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
        )
    if not colors:
        raise ValueError("the palette file contains no colours")
    return tuple(colors)


def parse_gpl(text: str) -> tuple[RGB, ...]:
    """A GIMP ``.gpl`` palette: a header, then ``r g b  name`` per line.

    The header is required to say GIMP Palette -- that is the format's own
    magic, and without it a stray text file parses as a palette of whatever
    numbers it happens to contain.
    """
    lines = text.splitlines()
    if not lines or not lines[0].strip().lower().startswith("gimp palette"):
        raise ValueError("not a GIMP palette: missing the 'GIMP Palette' header")
    colors: list[RGB] = []
    for raw in lines[1:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line.split()[0]:  # Name:/Columns: header fields
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = (int(parts[i]) for i in range(3))
        except ValueError:
            continue
        if not all(0 <= c <= 255 for c in (r, g, b)):
            raise ValueError(f"channel out of range: {line!r}")
        colors.append((r, g, b))
    if not colors:
        raise ValueError("the palette file contains no colours")
    return tuple(colors)


def parse_palette(text: str, suffix: str) -> tuple[RGB, ...]:
    """Dispatch on the file extension, which is the only thing that names the
    format -- both are plain text and one is a superset of nothing."""
    lowered = suffix.lower()
    if lowered == ".hex":
        return parse_hex(text)
    if lowered == ".gpl":
        return parse_gpl(text)
    raise ValueError(f"unsupported palette format {suffix!r} (expected .hex or .gpl)")


def palette_digest(colors: tuple[RGB, ...]) -> str:
    """A short fingerprint of a palette's *colours*, not of its file.

    Derived from the normalized triples so that reformatting a file, changing
    its comments or switching it from .hex to .gpl does not re-derive every
    artifact, while changing one channel of one colour does.
    """
    blob = ";".join(f"{r:02x}{g:02x}{b:02x}" for r, g, b in colors)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()[:12]


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


def _luma(arr: Any) -> Any:
    import numpy as np

    rgb = arr[:, :, :3].astype(np.float64)
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


def _axis_phase(luma: Any, scale: int, axis: int) -> int:
    """Where the cell boundaries sit along one axis.

    A block lattice puts all of its change *at* the boundaries, so the mean
    absolute gradient summed by position-modulo-period peaks at the boundary
    offset. O(W) after the gradient, and it is one argmax rather than a search
    over reconstructions.
    """
    import numpy as np

    grad = np.abs(np.diff(luma, axis=axis)).mean(axis=1 - axis)
    # grad[i] is the change between index i and i+1, i.e. a boundary *before*
    # index i+1; the phase is where a cell starts.
    totals = np.zeros(scale, dtype=np.float64)
    for offset in range(scale):
        totals[offset] = grad[(np.arange(len(grad)) % scale) == offset].sum()
    return int((int(np.argmax(totals)) + 1) % scale)


def grid_residual(arr: Any, scale: int, phase: tuple[int, int]) -> float:
    """How much of this image's change happens *between* cells rather than
    inside them.

    Mean absolute gradient at non-boundary positions over the mean absolute
    gradient everywhere -- the same ratio shape ``seam.py`` uses, and for the
    same reason: an absolute number of levels means nothing without the
    picture's own contrast to divide by. A block lattice scores 0.0 (nothing
    changes inside a cell); a smooth gradient scores ~1.0, because a gradient
    changes as much mid-cell as it does at a boundary. Comparing *within-cell
    variance to total variance* instead would look right and call every smooth
    image a grid, since a small patch of a gradient is always flatter than the
    whole frame.

    The worse of the two axes decides. A lattice holds in both directions; an
    image with horizontal banding and nothing vertical is not pixel art.
    """
    import numpy as np

    luma = _luma(arr)
    py, px = phase
    if min(luma.shape) < scale * _MIN_CELLS:
        return 1.0

    def axis_ratio(a: Any, offset: int) -> float:
        grad = np.abs(np.diff(a, axis=1))
        if not grad.size:
            return 1.0
        # grad[:, i] is the step between column i and i+1, i.e. a boundary iff
        # column i+1 starts a cell.
        positions = np.arange(grad.shape[1])
        boundary = ((positions + 1 - offset) % scale) == 0
        total = float(grad.mean())
        if total <= 0.0:
            # Nothing changes anywhere: a flat image is not evidence of a
            # lattice, however trivially constant each of its cells is.
            return 1.0
        interior = grad[:, ~boundary]
        return float(interior.mean()) / total if interior.size else 1.0

    return max(axis_ratio(luma, px), axis_ratio(luma.T, py))


def detect_grid(image: PILImage) -> dict[str, Any]:
    """The cell size and phase this image was drawn on, if any.

    Measured on the whole frame and on luminance: the lattice is a property of
    the generation, not of the subject, and a crop would move the phase before
    it was ever measured. Returns ``scale=None`` when nothing beats
    ``GRID_RESIDUAL_MAX`` -- an ordinary SDXL image, which is the common case
    and must keep the legacy export path.
    """
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    luma = _luma(arr)
    measured: list[tuple[int, tuple[int, int], float]] = []
    for scale in GRID_SCALES:
        if min(luma.shape) < scale * _MIN_CELLS:
            continue
        phase = (_axis_phase(luma, scale, 0), _axis_phase(luma, scale, 1))
        measured.append((scale, phase, grid_residual(arr, scale, phase)))
    if not measured:
        return {"scale": None, "phase": [0, 0], "residual": 1.0}
    # The *largest* passing scale, not the best-scoring one. Every divisor of a
    # true period passes just as cleanly -- an image of 8px blocks is trivially
    # also an image of 4px blocks -- and taking the smallest would double the
    # export's resolution and halve the size of every authored pixel.
    passing = [m for m in measured if m[2] < GRID_RESIDUAL_MAX]
    scale, phase, residual = (
        max(passing, key=lambda m: m[0])
        if passing
        else min(measured, key=lambda m: m[2])
    )
    return {
        "scale": scale if passing else None,
        "phase": [phase[0], phase[1]],
        "residual": float(residual),
    }


def downscale_grid(image: PILImage, scale: int, phase: tuple[int, int]) -> PILImage:
    """One output pixel per lattice cell, sampled at the cell's centre.

    The centre and not a corner: a corner sample sits on the boundary the
    generator drew, where a half-pixel of the neighbouring cell routinely
    bleeds. Whole-frame, because the phase is only meaningful against the frame
    the lattice was drawn on -- callers crop *after* this.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(image.convert("RGBA"))
    py, px = phase
    h, w = arr.shape[:2]
    ys = np.arange(py + scale // 2, h, scale)
    xs = np.arange(px + scale // 2, w, scale)
    if not len(ys) or not len(xs):
        raise ValueError("the grid leaves no cells in this image")
    return Image.fromarray(arr[np.ix_(ys, xs)], "RGBA")


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


def _to_oklab(rgb: Any) -> Any:
    """sRGB (0..255 float) -> Oklab. Closed form, no lookup, no dependency.

    Perceptual and not RGB because nearest-in-RGB picks by a distance nobody
    sees: it will map a mid-grey onto a saturated blue over a slightly darker
    grey whenever the arithmetic happens to be closer, which on a limited
    palette is a visible wrong colour rather than a subtle one.
    """
    import numpy as np

    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    lm = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    mm = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    sm = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(lm), np.cbrt(mm), np.cbrt(sm)
    return np.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        axis=-1,
    )


# A 4x4 Bayer matrix, in threshold units of 1/16. Ordered rather than
# error-diffused on purpose: diffusion is serial, spreads a decision across
# neighbouring cells and so is unstable under any later crop, while an ordered
# matrix is a pure function of position -- the same pixel dithers the same way
# whatever else is in the frame.
_BAYER4 = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)


def _nearest_native(flat: Any, plab: Any) -> Any:
    """Nearest Oklab entry per row from ``warlockc``, or ``None`` to fall back.

    The chunked numpy search below builds a ``(65536, p, 3)`` difference array
    per chunk -- about 100 MB at a 64-entry palette -- allocated, written and
    thrown away to read one index per row. Measured: 2488 ms for a 1024-square
    frame against 64 entries.

    **The kernel keeps the sqrt, and so does this docstring.** Comparing
    squared distances is tempting, since sqrt is monotonic -- but sqrt also
    *rounds*, so two distinct squared values can land on the same double.
    ``argmin`` over the rounded norms then takes the earlier index where an
    argmin over the exact squares takes the strictly smaller one, which may sit
    later: the same ordering, a different pick, and a visibly wrong colour on a
    ramp with near-equidistant entries.
    """
    import numpy as np

    from warlock import native

    if not native.available() or plab.shape[0] < 1:
        return None
    queries = np.ascontiguousarray(flat, dtype=np.float64)
    palette = np.ascontiguousarray(plab, dtype=np.float64)
    out = np.empty(queries.shape[0], dtype=np.int32)
    native.palette_nearest_f64(queries, palette, out)
    return out


def map_palette(
    image: PILImage, palette: tuple[RGB, ...], dither: bool = False
) -> PILImage:
    """Every opaque pixel becomes its nearest palette colour, in Oklab.

    Alpha is carried around the mapping and put back untouched, exactly as the
    legacy median-cut path does: a palette describes a subject's colours, and
    spending entries on the transparent background is the failure mode that
    convention exists to avoid.

    With ``dither``, an ordered 4x4 offset is added in Oklab before the nearest
    search. The offset is scaled by the palette's own mean nearest-neighbour
    spacing, so a 4-colour ramp dithers visibly and a 64-colour one barely at
    all -- a fixed offset would either do nothing on the second or destroy the
    first.
    """
    import numpy as np
    from PIL import Image

    if not palette:
        raise ValueError("an empty palette maps nothing")
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    lab = _to_oklab(rgba[:, :, :3])
    entries = np.asarray(palette, dtype=np.float64)
    plab = _to_oklab(entries)

    if dither and len(entries) > 1:
        # Mean distance to each entry's nearest neighbour: the size of the gap
        # dithering is being asked to bridge.
        gaps = np.linalg.norm(plab[:, None, :] - plab[None, :, :], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        spacing = float(gaps.min(axis=1).mean())
        bayer = np.asarray(_BAYER4, dtype=np.float64) / 16.0 - 0.5
        h, w = lab.shape[:2]
        tile = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        lab = lab + tile[:, :, None] * spacing

    flat = lab.reshape(-1, 3)
    picks = _nearest_native(flat, plab)
    if picks is not None:
        out = entries[picks].astype(np.uint8)
    else:
        # Chunked so a 1024x1024 frame against a 64-entry palette does not build
        # a single (1M, 64, 3) intermediate.
        out = np.empty((flat.shape[0], 3), dtype=np.uint8)
        step = 1 << 16
        for start in range(0, flat.shape[0], step):
            piece = flat[start : start + step]
            d = np.linalg.norm(piece[:, None, :] - plab[None, :, :], axis=-1)
            out[start : start + step] = entries[np.argmin(d, axis=1)].astype(np.uint8)
    mapped = np.dstack([out.reshape(lab.shape[0], lab.shape[1], 3), alpha])
    return Image.fromarray(mapped.astype(np.uint8), "RGBA")


def snap_alpha(image: PILImage, threshold: int = ALPHA_THRESHOLD) -> PILImage:
    """Alpha becomes 0 or 255 and nothing between.

    A reduction leaves partial alpha wherever a cell straddled the subject's
    edge, and a partial-alpha pixel in a 32px sprite reads as a smudge in every
    engine that does not blend the way the preview did.
    """
    import numpy as np
    from PIL import Image

    rgba = np.asarray(image.convert("RGBA")).copy()
    rgba[:, :, 3] = np.where(rgba[:, :, 3] >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def clean_orphans(image: PILImage) -> tuple[PILImage, int]:
    """Lone pixels take their neighbourhood's colour.

    A pixel none of whose eight neighbours shares its colour is, at 32px, an
    artefact of the reduction rather than a decision -- the generator does not
    place single-pixel detail at that size. It is replaced by the most common
    colour among its neighbours, so a deliberate two-pixel highlight survives
    (each of the pair has the other) while a stray does not.

    Transparent pixels are left alone in both roles: an isolated *hole* is a
    silhouette decision, and the alpha snap above already made it deliberate.
    """
    import numpy as np
    from PIL import Image

    rgba = np.asarray(image.convert("RGBA"))
    h, w = rgba.shape[:2]
    if h < 3 or w < 3:
        return (image.convert("RGBA"), 0)
    opaque = rgba[:, :, 3] > 0
    # One integer per distinct colour, so "same colour" is one comparison.
    rgb = rgba[:, :, :3].astype(np.int64)
    code = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
    code = np.where(opaque, code, -1)

    padded = np.pad(code, 1, constant_values=-1)
    neighbours = [
        padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    ]
    stack = np.stack(neighbours, axis=-1)
    lonely = opaque & ~(stack == code[:, :, None]).any(axis=-1)
    if not lonely.any():
        return (Image.fromarray(rgba, "RGBA"), 0)

    out = rgba.copy()
    ys, xs = np.nonzero(lonely)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        vals = [v for v in stack[y, x].tolist() if v >= 0]
        if not vals:
            continue
        winner = max(set(vals), key=vals.count)
        out[y, x, :3] = (
            (winner >> 16) & 0xFF,
            (winner >> 8) & 0xFF,
            winner & 0xFF,
        )
    return (Image.fromarray(out, "RGBA"), int(lonely.sum()))


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def report(
    image: PILImage,
    *,
    palette: tuple[RGB, ...] | None = None,
    grid: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Four advisory numbers about a finished pixel export.

    Advisory in the sense every measurement in this codebase is: nothing here
    fails anything, the numbers go on the manifest and the user decides.
    """
    import numpy as np

    rgba = np.asarray(image.convert("RGBA"))
    opaque = rgba[:, :, 3] > 0
    total = int(opaque.sum())
    pixels = rgba[:, :, :3][opaque]
    colors = int(len(np.unique(pixels.reshape(-1, 3), axis=0))) if total else 0

    out: dict[str, Any] = {"colors": colors, "opaque": total}
    if palette:
        entries = {tuple(int(c) for c in e) for e in palette}
        if total:
            member = sum(
                1 for row in np.unique(pixels.reshape(-1, 3), axis=0).tolist()
                if tuple(row) in entries
            )
            out["palette_fraction"] = float(member) / max(1, colors)
        else:
            out["palette_fraction"] = 1.0
    _, orphans = clean_orphans(image)
    out["orphan_fraction"] = (float(orphans) / total) if total else 0.0
    if grid is not None:
        out["grid_residual"] = grid.get("residual")
        out["grid_scale"] = grid.get("scale")
    return out


def verdict(rep: dict[str, Any]) -> str | None:
    """One sentence a person reads beside the export, or None when there is
    nothing to say.

    Basic-Latin plus Latin-1, like every other UI string: imgui's atlas is
    baked with that default glyph range, so the middle dot (U+00B7) is safe
    but anything past U+00FF would draw as the missing-glyph box.
    """
    if not rep:
        return None
    parts: list[str] = []
    colors = rep.get("colors")
    if colors is not None:
        parts.append(f"{colors} colours")
    fraction = rep.get("palette_fraction")
    if fraction is not None and fraction < 1.0:
        parts.append(f"{fraction * 100:.0f}% on palette")
    orphans = rep.get("orphan_fraction") or 0.0
    if orphans > 0.02:
        parts.append(f"{orphans * 100:.0f}% lone pixels")
    scale = rep.get("grid_scale")
    if scale:
        parts.append(f"{scale}px source grid")
    return " · ".join(parts) if parts else None

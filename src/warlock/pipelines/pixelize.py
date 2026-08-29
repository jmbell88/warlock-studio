"""The AI-free pixeliser: a rendered sheet becomes pixel art by arithmetic.

``create_pixel_sheet`` always queues an SDXL img2img job. That is the right
tool for restyling, and the wrong one for a Troupe character: at hard-edged,
alpha-255, low-colour pixel art a single drifted pixel is visible in every
frame and there is no averaging to hide it. Everything needed to do it
deterministically already shipped in ``pixel.py`` -- ``map_palette`` (nearest
in Oklab), ``snap_alpha``, ``clean_orphans`` -- and nothing wired them to a
sheet. This is that wiring, plus the two pieces that were genuinely missing:
an alpha-weighted supersample reduction and an outline pass.

The chain, in this order, and the order is the design:

1. **Reduce**, alpha-weighted -- whole-atlas via ``pixelize_atlas`` where the
   atlas exists at the render size, and **per frame** via ``reduce_frames``
   where it cannot, which for a 256-cell Troupe sheet at a 512px render is
   always (see that function: the packed atlas would be 4096x16384 against an
   8192 ceiling). Whole-atlas is preferred where it is available so a cell
   boundary is in the same place computed from either side
   (``spritesynth.reduce_atlas``'s argument); per frame gives up nothing there,
   because each frame reduces to exactly the cell size and no output pixel can
   straddle two cells. Alpha-weighted because a plain
   RGBA mean averages the transparent background's black into every silhouette
   edge, which is a dark fringe on every sprite -- the exact "muddy" failure
   this program exists to escape.
2. **Snap alpha.** A reduction leaves partial alpha wherever a cell straddled
   the edge, and partial alpha at 32px reads as a smudge.
3. **Map to a designed palette.** Not median-cut. ``quantize_shared`` picks the
   colours the render happens to contain, which is why sheets come out muddy;
   an authored ramp is the single highest-leverage art input in the program.
   The mapping is what puts back the contrast the box mean flattened.
4. **Clean orphans**, per cell.
5. **Outline**, per cell.

Steps 4 and 5 are per cell because a dense atlas has no gutter: a sprite
touching its cell's edge would otherwise see its neighbour's pixels as
neighbours, and outline across the seam.

Deterministic end to end -- no RNG, no dither unless asked, integer arithmetic
where it matters -- which is what makes the golden-image test in
``tests/troupe/test_pixelize.py`` the right bar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pixel import RGB, clean_orphans, map_palette, snap_alpha

__all__ = [
    "OUTLINE_MODES",
    "REDUCE_MODES",
    "outline",
    "pixelize",
    "pixelize_atlas",
    "reduce",
    "reduce_frames",
]

PILImage = Any

#: ``box`` averages every source pixel under an output pixel, weighted by
#: alpha; ``point`` takes the cell-centre sample ``pixel.downscale_grid``
#: takes. Box is the default: a 512px EEVEE render has real sub-pixel detail
#: and throwing 15 of every 16 samples away is noise, not crispness.
REDUCE_MODES = ("box", "point")

#: ``inner`` recolours the sprite's own edge pixels and cannot change its
#: silhouette, size, trim or pivot. ``outer`` grows the silhouette by a pixel,
#: which is the look most pixel art wants and the one that can clip at a cell
#: edge -- so it is opt-in and never the default.
OUTLINE_MODES = ("none", "inner", "outer")


def _weighted_box(arr: Any, sy: int, sx: int) -> Any:
    """``arr`` reduced by integer strides, RGB averaged weighted by alpha."""
    import numpy as np

    h, w = arr.shape[:2]
    blocks = arr[: h // sy * sy, : w // sx * sx].astype(np.float64)
    blocks = blocks.reshape(h // sy, sy, w // sx, sx, 4)
    alpha = blocks[..., 3]
    weight = alpha.sum(axis=(1, 3))
    rgb = (blocks[..., :3] * alpha[..., None]).sum(axis=(1, 3))
    # Where a whole block is transparent there is no colour to average; leaving
    # it at zero is right, because the alpha below is zero too and the alpha
    # snap will not resurrect it.
    safe = np.where(weight > 0.0, weight, 1.0)
    out = np.empty((*weight.shape, 4), dtype=np.float64)
    out[..., :3] = rgb / safe[..., None]
    out[..., 3] = alpha.mean(axis=(1, 3))
    return np.rint(out).clip(0, 255).astype(np.uint8)


def reduce(image: PILImage, size: tuple[int, int], *, mode: str = "box") -> PILImage:
    """``image`` at ``size``, by supersampling rather than by filtering.

    An integer stride takes the arithmetic path; anything else falls back to a
    single NEAREST resize -- which is exactly, and for every rung,
    ``spritesynth.reduce_atlas``. That function was the sprite path's whole
    reduction until ``SPRITE_DRAFT_VERSION`` 2 and has no caller in the program
    now; the fallback below is why the rungs that do not divide (24, 48, 96 out
    of 512) came through that change byte for byte, while the ones that do gained
    the 63 samples the single resize was throwing away.
    """
    import numpy as np
    from PIL import Image

    if mode not in REDUCE_MODES:
        raise ValueError(f"mode must be one of {list(REDUCE_MODES)}")
    w, h = size
    if w < 1 or h < 1:
        raise ValueError("a reduction target must be at least 1x1")
    rgba = image.convert("RGBA")
    if rgba.size == (w, h):
        return rgba
    if rgba.width % w or rgba.height % h:
        return rgba.resize((w, h), Image.NEAREST)
    sx, sy = rgba.width // w, rgba.height // h
    arr = np.asarray(rgba)
    if mode == "point":
        ys = np.arange(sy // 2, rgba.height, sy)
        xs = np.arange(sx // 2, rgba.width, sx)
        return Image.fromarray(arr[np.ix_(ys, xs)].copy(), "RGBA")
    return Image.fromarray(_weighted_box(arr, sy, sx), "RGBA")


def _edge_mask(alpha: Any, mode: str) -> Any:
    """Which pixels the outline claims, given a binary alpha plane."""
    import numpy as np

    opaque = alpha > 0
    padded = np.pad(opaque, 1, constant_values=False)
    h, w = opaque.shape
    # Four-neighbour, not eight: an eight-neighbour outline puts a pixel
    # diagonally off a corner, which reads as a stray rather than a line.
    neighbours = [
        padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
    ]
    any_open = ~np.logical_and.reduce(neighbours)
    any_solid = np.logical_or.reduce(neighbours)
    if mode == "inner":
        return opaque & any_open
    return (~opaque) & any_solid


def outline(
    image: PILImage,
    color: RGB,
    *,
    mode: str = "inner",
) -> PILImage:
    """A one-pixel hard outline. Returns ``image`` unchanged for ``"none"``.

    Hard outlines are most of what separates crisp pixel art from a shrunk
    render, and nothing else in the repo does this. Deliberately dumb: one
    pixel, one colour, four-neighbour. Anything cleverer is a decision the
    Inker cleanup pass should be making by hand.
    """
    import numpy as np
    from PIL import Image

    if mode not in OUTLINE_MODES:
        raise ValueError(f"mode must be one of {list(OUTLINE_MODES)}")
    rgba = image.convert("RGBA")
    if mode == "none":
        return rgba
    arr = np.asarray(rgba).copy()
    mask = _edge_mask(arr[:, :, 3], mode)
    if not mask.any():
        return Image.fromarray(arr, "RGBA")
    arr[mask, 0], arr[mask, 1], arr[mask, 2] = color
    arr[mask, 3] = 255
    return Image.fromarray(arr, "RGBA")


def darkest(palette: Sequence[RGB]) -> RGB:
    """The palette's darkest entry, by Rec.709 luma -- the default outline.

    Chosen from the ramp rather than made up, so an outline is always a colour
    the sheet already contains and the frame stays exactly N colours.
    """
    if not palette:
        raise ValueError("an empty palette has no darkest entry")
    return min(
        (tuple(int(v) for v in c) for c in palette),
        key=lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2],
    )


def pixelize(
    image: PILImage,
    *,
    size: tuple[int, int],
    palette: Sequence[RGB],
    dither: bool = False,
    reduce_mode: str = "box",
    outline_mode: str = "none",
    outline_color: RGB | None = None,
    alpha_threshold: int = 128,
    clean: bool = True,
) -> tuple[PILImage, dict[str, Any]]:
    """One frame, reduced and quantised. Returns ``(image, report)``.

    The report is what a caller shows a user and what a test asserts on:
    ``orphans`` cleaned, ``colors`` in the result, and whether the reduction
    took the exact-stride path or the NEAREST fallback.
    """
    import numpy as np

    entries = tuple(tuple(int(v) for v in c) for c in palette)
    if not entries:
        raise ValueError("an empty palette maps nothing")
    src = image.convert("RGBA")
    exact = not (src.width % size[0] or src.height % size[1])
    small = reduce(src, size, mode=reduce_mode)
    small = snap_alpha(small, alpha_threshold)
    small = map_palette(small, entries, dither=dither)
    orphans = 0
    if clean:
        small, orphans = clean_orphans(small)
    if outline_mode != "none":
        small = outline(
            small, outline_color or darkest(entries), mode=outline_mode
        )
    arr = np.asarray(small)
    opaque = arr[arr[:, :, 3] > 0][:, :3]
    return small, {
        "orphans": int(orphans),
        "colors": len({tuple(p) for p in opaque.tolist()}),
        "exact_stride": bool(exact),
        "reduce_mode": reduce_mode,
        "outline": outline_mode,
    }


def pixelize_atlas(
    atlas: PILImage,
    *,
    columns: int,
    rows: int,
    cell: int,
    palette: Sequence[RGB],
    dither: bool = False,
    reduce_mode: str = "box",
    outline_mode: str = "none",
    outline_color: RGB | None = None,
    alpha_threshold: int = 128,
    clean: bool = True,
) -> tuple[PILImage, dict[str, Any]]:
    """A whole rendered atlas at ``cell`` px per cell.

    Reduction and palette mapping run whole-atlas -- exact cell boundaries, one
    nearest search instead of ``columns * rows`` of them -- and the two
    neighbourhood passes run per cell, because a dense atlas has no gutter and
    a sprite touching its edge would otherwise be outlined against the sprite
    next to it.
    """
    import numpy as np
    from PIL import Image

    if columns < 1 or rows < 1 or cell < 1:
        raise ValueError("an atlas is at least one cell of at least one pixel")
    if atlas.width % columns or atlas.height % rows:
        raise ValueError(
            f"a {atlas.width}x{atlas.height} atlas is not {columns}x{rows} whole cells"
        )
    entries = tuple(tuple(int(v) for v in c) for c in palette)
    if not entries:
        raise ValueError("an empty palette maps nothing")

    target = (columns * cell, rows * cell)
    exact = not (atlas.width % target[0] or atlas.height % target[1])
    small = reduce(atlas.convert("RGBA"), target, mode=reduce_mode)
    small = snap_alpha(small, alpha_threshold)
    small = map_palette(small, entries, dither=dither)

    ink = outline_color or darkest(entries)
    arr = np.asarray(small).copy()
    orphans = 0
    for row in range(rows):
        for column in range(columns):
            y, x = row * cell, column * cell
            piece = Image.fromarray(arr[y : y + cell, x : x + cell].copy(), "RGBA")
            if clean:
                piece, count = clean_orphans(piece)
                orphans += count
            if outline_mode != "none":
                piece = outline(piece, ink, mode=outline_mode)
            arr[y : y + cell, x : x + cell] = np.asarray(piece)
    out = Image.fromarray(arr, "RGBA")
    opaque = arr[arr[:, :, 3] > 0][:, :3]
    return out, {
        "orphans": int(orphans),
        "colors": len({tuple(p) for p in opaque.tolist()}),
        "exact_stride": bool(exact),
        "reduce_mode": reduce_mode,
        "outline": outline_mode,
        "cell": int(cell),
        "columns": int(columns),
        "rows": int(rows),
    }


def reduce_frames(
    frames: Mapping[int, Any],
    size: int,
    out_dir: Any,
    *,
    mode: str = "box",
) -> dict[int, Any]:
    """Reduce rendered cell PNGs to ``size`` before they are packed.

    **The supersample step has to happen per frame, not on the packed atlas,
    and that is forced rather than preferred.** A Troupe sheet is 256 cells; at
    the 512px render the whole program is specified to supersample from, the
    packed atlas would be 4096x16384 and ``check_atlas_size`` refuses it at
    8192 -- correctly, because no engine would load it. So the atlas can only
    ever exist at the *logical* size, and the reduction is what stands between
    the two.

    Per frame costs nothing here that whole-atlas reduction was buying: the
    boundary argument (``spritesynth.reduce_atlas``) is about a target that
    does not divide the source, and every frame reduces to exactly ``size`` by
    construction, so no output pixel can straddle two cells.

    It also keeps ``sheet.pack`` off the pixel-art path. That function resizes
    a mismatched frame with **LANCZOS**, which is the right answer for a
    preview and the wrong one for a sprite: a filtered downscale is precisely
    the soft, fringed result the alpha snap then has to guess about.

    Returns a fresh ``{cell index: path}`` mapping for ``sheet.pack``.
    """
    from pathlib import Path

    from PIL import Image

    if int(size) < 1:
        raise ValueError("a reduction target must be at least 1 pixel")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reduced: dict[int, Any] = {}
    for index, path in frames.items():
        source = Path(path)
        if not source.exists():
            raise ValueError(f"no rendered frame for cell {index}")
        with Image.open(source) as frame:
            small = reduce(frame, (int(size), int(size)), mode=mode)
            dst = out / f"cell_{int(index):04d}.png"
            small.save(dst, "PNG")
        reduced[int(index)] = dst
    return reduced

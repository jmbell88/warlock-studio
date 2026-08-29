"""Turning a rendered sprite sheet into a pixel-art sprite sheet.

The differentiator, and the reason it works: the eight directions are not eight
generations that have to be talked into agreeing with each other. They are exact
orthographic renders of one mesh, so the geometry already agrees perfectly, and
the only thing being generated is *how it looks*. Everything here exists to keep
it that way -- the whole band is denoised in one latent under one seed (so there
is one identity, not eight), the silhouettes are re-imposed from the source
render afterwards (so they are exact by construction rather than by persuasion),
and the palette is quantized once across the whole atlas (so a colour means the
same thing in every direction).

Pure: Pillow and NumPy inside the functions, no torch, no service imports, no
decisions about where a file goes. The reduction and the palette work are
delegated to ``pipelines/pixel`` rather than reimplemented, for the reason that
module's docstring gives.

Geometry note. Eight yaws at 128 px is exactly 1024 wide, which is one SDXL
frame -- that is not a coincidence, it is why 128 is the frame size this is
qualified at. A sheet narrower than 1024 is letterboxed horizontally and cropped
back out exactly; a sheet *wider* than 1024 is refused rather than tiled,
because a cell split across two generations is the flicker this design exists to
avoid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .pixel import RGB
from .sheet import SHEET_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

PIXEL_SHEET_VERSION = 1

# One SDXL frame. The generation stays square and 1024 -- the same pin every
# other path here obeys -- and the band is what is fitted into it.
BAND_PX = 1024

# Neutral mid-grey behind a band, never a saturated key colour: SDXL bleeds the
# background into the subject's edges, and a magenta key would put magenta in
# the palette and a magenta rim on every sprite. The alpha is restored from the
# source render afterwards, so this colour never survives -- but its bleed
# would.
BAND_BACKGROUND = (128, 128, 128)


@dataclass(frozen=True, slots=True)
class Band:
    """One 1024x1024 generation's worth of the atlas.

    ``rows`` are sheet rows; ``x``/``y`` is where the band's content sits inside
    the square, which is what makes the crop back out exact.
    """

    index: int
    first_row: int
    rows: int
    frame: int
    x: int
    y: int
    width: int
    height: int


class NotRestylable(ValueError):
    """The sheet's geometry cannot be restyled as bands. Carries the reason."""


def check_restylable(meta: Mapping[str, Any], logical_size: int) -> None:
    """Refuse at the door, with the number and a remedy in the message.

    Two conditions, and both are about the grid rather than about taste. A sheet
    wider than one generation would have to be split mid-cell. And a logical
    size that does not divide the frame size exactly cannot be reduced without
    resampling across cell boundaries, which is the blending this whole feature
    exists to avoid.
    """
    frame = int(meta.get("frame_size") or 0)
    columns = int(meta.get("columns") or 0)
    if frame <= 0 or columns <= 0:
        raise NotRestylable("that sheet's sidecar does not describe a grid")
    width = frame * columns
    if width > BAND_PX:
        raise NotRestylable(
            f"that sheet is {width}px wide and a restyle generates {BAND_PX}px "
            f"at a time -- re-render it at a frame size of "
            f"{BAND_PX // columns}px or smaller"
        )
    if logical_size <= 0 or frame % logical_size:
        raise NotRestylable(
            f"a logical size of {logical_size}px does not divide the sheet's "
            f"{frame}px cells exactly"
        )


def bands(meta: Mapping[str, Any]) -> list[Band]:
    """The sheet split into whole-row groups, each one generation's worth.

    Whole rows, never part of one: a row is a subject's eight directions, and
    splitting it across two denoises is exactly the flicker the single-latent
    design removes. The last band is short rather than padded with rows that do
    not exist.
    """
    frame = int(meta["frame_size"])
    columns = int(meta["columns"])
    total_rows = int(meta["rows"])
    width = frame * columns
    per_band = max(1, BAND_PX // frame)
    out: list[Band] = []
    for index, first in enumerate(range(0, total_rows, per_band)):
        rows = min(per_band, total_rows - first)
        height = rows * frame
        out.append(
            Band(
                index=index,
                first_row=first,
                rows=rows,
                frame=frame,
                # Centred, so the model sees the content framed rather than
                # pushed into a corner; recorded, so the crop back is exact
                # rather than re-derived from the same arithmetic twice.
                x=(BAND_PX - width) // 2,
                y=(BAND_PX - height) // 2,
                width=width,
                height=height,
            )
        )
    return out


def band_init(atlas: PILImage, band: Band) -> PILImage:
    """The square RGB picture one band is denoised from.

    RGB and composited onto neutral grey, because an img2img init image has no
    alpha channel to carry -- the transparent background would otherwise be
    read as black and drawn as a black halo around every sprite.
    """
    from PIL import Image

    canvas = Image.new("RGB", (BAND_PX, BAND_PX), BAND_BACKGROUND)
    top = band.first_row * band.frame
    crop = atlas.convert("RGBA").crop((0, top, band.width, top + band.height))
    flat = Image.new("RGB", crop.size, BAND_BACKGROUND)
    flat.paste(crop, (0, 0), crop)
    canvas.paste(flat, (band.x, band.y))
    return canvas


def crop_back(generated: PILImage, band: Band) -> PILImage:
    """The band's content out of the square it was generated in."""
    return generated.convert("RGB").crop(
        (band.x, band.y, band.x + band.width, band.y + band.height)
    )


def remask(styled: PILImage, source: PILImage) -> PILImage:
    """The generated colours wearing the render's own alpha, exactly.

    This is what makes the silhouettes correct by *construction* rather than by
    persuasion: the source atlas is an orthographic render of real geometry, so
    its alpha is the truth about where the subject is, and anything SDXL
    invented outside it is background it was never asked for. Multiplied in
    verbatim, not blended -- a soft edge here would defeat the alpha snap that
    follows.
    """
    import numpy as np
    from PIL import Image

    rgb = np.asarray(styled.convert("RGB"))
    alpha = np.asarray(source.convert("RGBA"))[:, :, 3]
    if rgb.shape[:2] != alpha.shape:
        raise ValueError("the styled band and the source render are different sizes")
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def downscale(atlas: PILImage, factor: int) -> PILImage:
    """Integer-stride NEAREST reduction of the whole atlas at once.

    Whole-atlas and integer: every cell boundary is a multiple of the frame
    size, so an integer factor keeps each of them on an output pixel boundary.
    Any other ratio would put a cell's last column half into the next cell's
    first, which is a seam in every direction of every sprite.
    """
    from PIL import Image

    if factor < 1:
        raise ValueError("the reduction factor must be at least 1")
    if factor == 1:
        return atlas.convert("RGBA")
    if atlas.width % factor or atlas.height % factor:
        raise ValueError("the atlas does not divide by that factor exactly")
    return atlas.convert("RGBA").resize(
        (atlas.width // factor, atlas.height // factor), Image.NEAREST
    )


def quantize_shared(atlas: PILImage, colors: int) -> tuple[PILImage, list[str]]:
    """One palette across the whole atlas, and the palette it chose.

    Once, across every cell -- not per cell. Per-cell quantization is how the
    same shirt comes out two shades in two directions, which is the artefact
    that reads as flicker when a character turns.

    **The alpha is restored from the source, not quantized** -- ``opaque`` is
    computed before the median cut and stamped back after it, so no palette
    entry is spent representing a semi-transparent edge. That is as far as the
    handling goes, and the sentence here used to over-claim it as "opaque
    pixels only": the median cut is run over the *whole* RGB plane, transparent
    region included. Where that region is a uniform colour it costs one entry
    of the budget; a sprite sheet's transparent margin is black, so the cost is
    the black entry the sheet almost certainly wanted anyway.

    Worth stating because of the callers with a large transparent fraction --
    a sprite turnaround's cells are mostly margin -- where one entry out of
    eight is a different proposition from one out of sixty-four. (A tile sheet
    is the opposite case and costs nothing here: every cell is opaque edge to
    edge, so no entry of its budget is spent on a region nobody looks at.)
    Changing it means changing the bytes of every sheet already quantized, so
    it is written down rather than quietly fixed.
    """
    import numpy as np
    from PIL import Image

    from . import pixel as pixelmod

    rgba = np.asarray(atlas.convert("RGBA"))
    opaque = rgba[:, :, 3] >= pixelmod.ALPHA_THRESHOLD
    if not opaque.any():
        raise ValueError("that sheet's cells are empty")
    flat = Image.fromarray(rgba[:, :, :3], "RGB").quantize(
        colors=int(colors), method=Image.Quantize.MEDIANCUT
    )
    mapped = np.asarray(flat.convert("RGB"))
    out = np.dstack([mapped, np.where(opaque, 255, 0).astype(np.uint8)])
    used = {tuple(int(c) for c in p) for p in mapped[opaque]}
    palette = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in sorted(used)]
    return (Image.fromarray(out, "RGBA"), palette)


def resolve_palette(
    atlas: PILImage,
    *,
    colors: int,
    entries: tuple[RGB, ...] | None = None,
) -> tuple[tuple[RGB, ...], str]:
    """The colour set one sheet is mapped to, and where it came from.

    Two sources, one answer: an authored palette is returned verbatim as
    ``"designed"``, and with none given the median cut above picks one off the
    atlas and it comes back as ``"derived"``. ``None`` is how a caller says "I
    have no palette"; an empty tuple is not -- it is a palette of no colours,
    and ``pixel.map_palette`` refuses it, which is the right answer to a
    palette file somebody emptied.

    **The insight this whole shape rests on: a palette file does not replace
    the shared-across-cells property.** It is tempting to read "designed"
    against "derived" as a choice between an authored palette and
    ``quantize_shared``, and so to think naming a palette gives up whatever
    the median cut was doing for the sheet. It gives up nothing.
    ``quantize_shared`` stops being a **mapper** and becomes a **palette
    source**: the thing that made it shared was never the median cut, it is
    that one entry set is applied over the whole atlas in one pass. That
    property survives verbatim here, because ``pixel.map_palette`` maps
    whole-atlas and ``pixelize.pixelize_atlas`` already maps before its
    per-cell loop -- so the same shirt is the same shade in every direction
    whichever branch produced the entries.

    Which is also why the derived branch parses the hex list back to triples
    rather than handing on the mapped image ``quantize_shared`` returns: the
    caller wants *the colours*, and taking the pixels too would run the
    mapping twice and make a derived sheet a second code path with its own
    bugs.
    """
    if entries is not None:
        return (tuple(entries), "designed")
    _mapped, hexes = quantize_shared(atlas, colors)
    return (
        tuple(
            (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hexes
        ),
        "derived",
    )


def pixel_sidecar(
    meta: Mapping[str, Any],
    *,
    image: str,
    logical_size: int,
    palette: list[str],
    recipe: Mapping[str, Any],
    created: float,
) -> dict[str, Any]:
    """The pixel sheet's own sidecar, in the render sidecar's shape.

    Its own file rather than a new key in the render's, because the render's
    sidecar is that render's completion marker and describes *its* grid -- a
    pixel sheet is a different grid over the same subject, and merging the two
    would make one file's version number mean two things.

    Every pixel rectangle is the render's, divided by the reduction factor. The
    division is exact by construction: ``check_restylable`` refused any logical
    size that does not divide the frame size.
    """
    frame = int(meta["frame_size"])
    factor = frame // int(logical_size)

    def scaled(value: Any) -> Any:
        return None if value is None else (float(value) / factor)

    def reduced(rect: Any) -> dict[str, int] | None:
        """A pixel rectangle at the smaller size. Floor the origin and ceil the
        extent: a rectangle that rounded inward would clip a pixel off the thing
        it exists to describe."""
        if not rect:
            return None
        return {
            "x": int(rect["x"]) // factor,
            "y": int(rect["y"]) // factor,
            "w": -(-int(rect["w"]) // factor),
            "h": -(-int(rect["h"]) // factor),
        }

    cells = []
    for cell in meta.get("cells") or []:
        entry = dict(cell)
        entry["x"] = int(cell["x"]) // factor
        entry["y"] = int(cell["y"]) // factor
        entry["w"] = entry["h"] = int(logical_size)
        entry["pivot_x"] = scaled(cell.get("pivot_x"))
        entry["pivot_y"] = scaled(cell.get("pivot_y"))
        entry["trim"] = reduced(cell.get("trim"))
        # The slice block, at the smaller size and by the same two rules.
        # Present only when the render's own sidecar carried one, so a sheet
        # with no slices produces a pixel sidecar with none -- rather than an
        # empty list, which is a different statement.
        block = cell.get("slices")
        if block:
            entry["slices"] = [
                {
                    "name": one.get("name", ""),
                    "bounds": reduced(one.get("bounds")),
                    # A pivot is a *point*, so it divides rather than rounding:
                    # a nine-slice panel placed half a pixel out is visible, and
                    # the sidecar has always carried ``pivot_x`` as a float.
                    "pivot": (
                        None
                        if not one.get("pivot")
                        else {
                            "x": float(one["pivot"]["x"]) / factor,
                            "y": float(one["pivot"]["y"]) / factor,
                        }
                    ),
                    "center": reduced(one.get("center")),
                }
                for one in block
            ]
        cells.append(entry)
    return {
        "version": PIXEL_SHEET_VERSION,
        # The render's own version, carried rather than assumed: this file is a
        # derivative of that grid and says which one it was made from.
        "sheet_version": SHEET_VERSION,
        "id": meta.get("id"),
        "name": meta.get("name"),
        "source_job": meta.get("source_job"),
        "created": created,
        "image": image,
        "frame_size": int(logical_size),
        "columns": int(meta["columns"]),
        "rows": int(meta["rows"]),
        "width": int(meta["columns"]) * int(logical_size),
        "height": int(meta["rows"]) * int(logical_size),
        "elevation": meta.get("elevation"),
        "lighting": meta.get("lighting"),
        "yaws": list(meta.get("yaws") or []),
        "poses": list(meta.get("poses") or []),
        "palette": palette,
        # The restyle's own settings live here and not in the job's params: they
        # describe this file, and params are inputs a rerun copies.
        "restyle": dict(recipe),
        "cells": cells,
    }

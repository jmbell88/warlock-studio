"""Finished 2D assets from a reference image: icon, sprite, pixel art.

The same split ``pipelines/sheet.py`` makes, and for the same reason.
Everything about *what an export is* -- where the subject is trimmed to, how
much margin it keeps, where the pivot sits, how many colours survive -- is
decided here against a boolean mask, so the file, the manifest and any preview
can never disagree, and the whole thing is testable with a rectangle on a grey
field. Producing the mask is somebody else's job (``pipelines/matting.py``, or
``reference.subject_mask`` when the weights are absent), because that is the
part that needs a model.

Pure and torch-free: Pillow, NumPy and cv2 are imported inside the functions,
so this module stays importable without the text2image extra.

Every function takes ``(image, mask)`` and returns ``(image, metadata)``. The
metadata is what the manifest records, and it is returned rather than written
because a pure module has no business deciding where a file goes.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

# The icon canvas. Fixed rather than configurable because the artifact name is
# fixed: ``icon.png`` is one entry in the MEDIA allowlist, and a size knob
# would either need a name per size or would silently mean different things in
# two job directories. 512 downsamples cleanly to every icon size an engine or
# a launcher asks for.
ICON_SIZE = 512

# What a pixel-art export may be reduced to. Each is its own artifact name
# (pixel_32.png ...), for the same allowlist reason.
PIXEL_SIZES = (32, 64, 128)

# Margin left around an icon's subject, as a fraction of the canvas. Enough
# that a silhouette does not touch the frame -- which reads as cropped at
# thumbnail size -- and no more.
DEFAULT_PAD = 0.08

# Below this many pixels an alpha island is a speck of matting noise rather
# than a part of the subject, and counting it would make every export look
# like it had come apart.
MIN_ISLAND_PX = 16

# Alpha strictly between these is "partial" -- the soft rim a matte leaves.
_ALPHA_FLOOR = 0
_ALPHA_CEIL = 255


class NoSubject(ValueError):
    """The mask found nothing to export.

    A named exception rather than a blank image: every caller here is about to
    write a file, and a fully transparent icon.png is indistinguishable from a
    successful export until someone opens it.
    """


def trim_box(mask: Any) -> tuple[int, int, int, int] | None:
    """The subject's bounds as a PIL crop box (left, top, right, bottom).

    Right/bottom are exclusive, matching ``Image.crop``, so the box can be
    handed straight to it without an off-by-one at the call site.
    """
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def cutout(image: PILImage, mask: Any) -> PILImage:
    """The image with the mask as its alpha channel.

    This is the one place the mask becomes transparency, and it is deliberately
    *not* what ``pipelines/reference.py`` does: there the mask drives geometry
    only and is never written back as alpha, because a bad mask would punch
    holes in what trellis reconstructs. Here the export *is* a cutout, so a bad
    mask produces a visibly ragged PNG -- a failure the user can see, which is
    the whole difference.
    """
    import numpy as np
    from PIL import Image

    rgba = np.array(image.convert("RGBA"))
    rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _trimmed(image: PILImage, mask: Any) -> tuple[PILImage, tuple[int, int, int, int]]:
    box = trim_box(mask)
    if box is None:
        raise NoSubject("the matte found no subject in this image")
    return (cutout(image, mask).crop(box), box)


def icon(
    image: PILImage, mask: Any, *, size: int = ICON_SIZE, pad: float = DEFAULT_PAD
) -> tuple[PILImage, dict[str, Any]]:
    """A square, centred, transparent icon.

    Fitted inside the canvas rather than stretched to it: stretching would
    normalise the aspect ratio away, and a tall sword would come out the same
    shape as a round shield -- which is exactly the confusion an icon set
    exists to avoid.
    """
    from PIL import Image

    cropped, box = _trimmed(image, mask)
    inner = max(1, int(round(size * (1.0 - 2 * pad))))
    scale = min(inner / cropped.width, inner / cropped.height)
    target = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    resized = cropped.resize(target, Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # A straight copy, with no mask argument. Handing an RGBA image to paste as
    # its own mask does not composite it: PIL mixes every band including alpha
    # against the destination, so a rim pixel comes back as ``a*a/255`` with its
    # RGB dragged toward the transparent black canvas. The LANCZOS resample above
    # produces exactly that partial-alpha rim, so the masked form haloed every
    # icon and rounded rim alpha below ~16 away entirely. The canvas is empty,
    # so there is nothing to composite against in the first place.
    canvas.paste(resized, ((size - target[0]) // 2, (size - target[1]) // 2))
    return (
        canvas,
        {
            "kind": "icon",
            "canvas": [size, size],
            "trim": list(box),
            "source": [image.width, image.height],
            "pad": float(pad),
        },
    )


def sprite(
    image: PILImage, mask: Any, *, pad: float = 0.0
) -> tuple[PILImage, dict[str, Any]]:
    """The subject alone, at its native resolution, with a pivot.

    No resize, and by default no canvas either: a sprite is placed by its
    pivot, so anything that grows the image has to leave the pivot on the
    subject or it has moved the thing the pivot was meant to anchor. Padding
    exists for the engines that need a bleed margin around a sprite, and it is
    the *subject's* bottom-centre that is recorded either way -- measured from
    the padded canvas's own origin, so ``pivot_rule`` stays literally true at
    any pad rather than only at zero. Bottom-centre because that is where an
    engine puts a standing character's feet, and it is recorded rather than
    assumed -- an importer that guesses is an importer that is wrong for half
    a set.

    ``pad`` and ``margin`` are both recorded, as ``icon`` records its own pad:
    a consumer that wants the subject's own bounds back needs the margin in
    pixels, and rounding it out of a fraction at the far end would not always
    land on the same pixel we did.
    """
    from PIL import Image

    cropped, box = _trimmed(image, mask)
    subject = cropped.size
    margin = int(round(max(cropped.size) * pad)) if pad else 0
    if margin:
        canvas = Image.new(
            "RGBA",
            (cropped.width + 2 * margin, cropped.height + 2 * margin),
            (0, 0, 0, 0),
        )
        # No mask argument, for the reason spelled out in ``icon``: an RGBA
        # image used as its own paste mask has its alpha squared and its
        # colour pulled toward the empty canvas.
        canvas.paste(cropped, (margin, margin))
        cropped = canvas
    return (
        cropped,
        {
            "kind": "sprite",
            "canvas": [cropped.width, cropped.height],
            "trim": list(box),
            "source": [image.width, image.height],
            "pad": float(pad),
            "margin": margin,
            "pivot": [margin + subject[0] / 2.0, float(margin + subject[1])],
            "pivot_rule": "bottom-centre",
        },
    )


def _legacy_pixel(
    image: PILImage, mask: Any, size: int, colors: int
) -> tuple[PILImage, tuple[int, int, int, int], int | None]:
    """Crop to the subject, then scale by an arbitrary ratio.

    What every pixel export did before a generator that draws on a real grid
    existed, and still what happens to an image that has no grid to detect: a
    smooth SDXL render has no lattice to preserve, so there is nothing better to
    do than reduce it and take hard edges.
    """
    from PIL import Image

    cropped, box = _trimmed(image, mask)
    scale = size / max(cropped.width, cropped.height)
    target = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    small = cropped.resize(target, Image.NEAREST)
    palette = None
    if colors and colors > 0:
        palette = int(colors)
        alpha = small.getchannel("A")
        flat = small.convert("RGB").quantize(
            colors=palette, method=Image.Quantize.MEDIANCUT
        )
        # The alpha is carried around the quantize rather than through it, and
        # put back untouched: median cut only ever rewrites RGB, so the cutout
        # the resample produced survives the palette reduction exactly.
        small = flat.convert("RGBA")
        small.putalpha(alpha)
    return (small, box, palette)


def _grid_pixel(
    image: PILImage, mask: Any, size: int, grid: dict[str, Any]
) -> tuple[PILImage, tuple[int, int, int, int]]:
    """Reduce on the lattice the generator drew, *then* crop.

    This ordering is the whole point of detecting a grid at all. Cropping first
    moves the origin off the lattice by whatever the subject's bounding box
    happens to be, and the cell-centre sampling below then reads a different
    part of every cell -- which is a shear, not a reduction. So the full frame
    is reduced (each cell becoming one pixel, and its alpha decided by whether
    the cell was mostly subject), and the crop is taken afterwards in logical
    coordinates.

    ``meta["trim"]`` is still reported in *source* pixels, so every existing
    consumer of that field is unaffected by which branch ran.
    """
    import numpy as np
    from PIL import Image

    from . import pixel as pixelmod

    scale = int(grid["scale"])
    phase = (int(grid["phase"][0]), int(grid["phase"][1]))
    small = pixelmod.downscale_grid(cutout(image, mask), scale, phase)

    # A cell's alpha by majority coverage rather than by its centre sample: the
    # centre of an edge cell is as likely to fall just outside the subject as
    # just inside, and one wrong cell on a 32px sprite is a visible bite.
    solid = np.asarray(mask, dtype=bool)
    py, px = phase
    rows, cols = small.height, small.width
    # Zero-padded to a whole number of cells: the last row or column of cells
    # can legitimately run off the frame, and the part that is missing is not
    # subject.
    aligned = np.zeros((rows * scale, cols * scale), dtype=np.float64)
    usable = solid[py : py + rows * scale, px : px + cols * scale]
    aligned[: usable.shape[0], : usable.shape[1]] = usable
    coverage = aligned.reshape(rows, scale, cols, scale).mean(axis=(1, 3))
    rgba = np.asarray(small.convert("RGBA")).copy()
    rgba[:, :, 3] = np.where(coverage >= 0.5, 255, 0).astype(np.uint8)
    small = Image.fromarray(rgba, "RGBA")

    box = trim_box(rgba[:, :, 3] > 0)
    if box is None:
        raise NoSubject("the matte found no subject in this image")
    small = small.crop(box)
    if max(small.size) > size:
        # Only ever an integer stride: any other ratio resamples across cells
        # and undoes the alignment this branch exists to keep.
        stride = -(-max(small.size) // size)
        small = small.resize(
            (max(1, small.width // stride), max(1, small.height // stride)),
            Image.NEAREST,
        )
    # Back into source coordinates, so meta["trim"] means what it always meant.
    left, top, right, bottom = box
    source_box = (
        px + left * scale,
        py + top * scale,
        min(image.width, px + right * scale),
        min(image.height, py + bottom * scale),
    )
    return (small, source_box)


def pixel(
    image: PILImage,
    mask: Any,
    *,
    size: int,
    colors: int = 0,
    opts: Any = None,
) -> tuple[PILImage, dict[str, Any]]:
    """A downsampled, optionally palette-limited cutout.

    Nearest neighbour, never a filter: a resample that blends puts a ramp of
    in-between colours along every edge, and hard edges are the one property
    that makes the result read as pixel art rather than as a small photograph.

    Two branches, decided by the *image* rather than by a setting. An image a
    pixel-art model drew carries a block lattice with a period and a phase, and
    reducing it on that lattice recovers the pixels the model actually authored;
    an ordinary render carries none and takes the legacy path unchanged. With
    ``opts`` left at its default the result is byte-identical to what this
    function produced before either existed, which is pinned by a test --
    every asset already on disk was cut that way.

    ``colors`` is kept as a parameter rather than folded into ``opts`` because
    it is the one pixel setting that predates the dataclass; when both are
    given, ``opts.colors`` wins.
    """
    from . import pixel as pixelmod

    if opts is None:
        opts = pixelmod.PixelOpts(colors=int(colors or 0))
    grid = pixelmod.detect_grid(image)
    palette_cap: int | None = None
    if grid["scale"]:
        small, box = _grid_pixel(image, mask, size, grid)
    else:
        small, box, palette_cap = _legacy_pixel(
            image, mask, size, 0 if opts.palette else opts.colors
        )
    if opts.palette:
        small = pixelmod.map_palette(small, opts.palette, dither=opts.dither)
    elif grid["scale"] and opts.colors:
        # The grid branch does its own quantize, since _legacy_pixel's is
        # entangled with the resample it did not run.
        from PIL import Image

        palette_cap = int(opts.colors)
        alpha = small.getchannel("A")
        flat = small.convert("RGB").quantize(
            colors=palette_cap, method=Image.Quantize.MEDIANCUT
        )
        small = flat.convert("RGBA")
        small.putalpha(alpha)
    cleaned = 0
    if opts.cleanup:
        small, cleaned = pixelmod.clean_orphans(small)
    if grid["scale"] or opts.palette or opts.cleanup:
        # Only on the new branches: the legacy export's partial-alpha rim is
        # part of what "byte-identical" means above.
        small = pixelmod.snap_alpha(small)
    return (
        small,
        {
            "kind": "pixel",
            "size": int(size),
            "canvas": [small.width, small.height],
            "trim": list(box),
            "source": [image.width, image.height],
            "palette": palette_cap,
            "palette_file": opts.palette_name,
            "palette_hash": opts.palette_hash,
            "dither": bool(opts.dither),
            "grid": (
                {"scale": grid["scale"], "phase": grid["phase"],
                 "residual": grid["residual"]}
                if grid["scale"]
                else None
            ),
            "cleanup": int(cleaned),
            "qa": pixelmod.report(small, palette=opts.palette, grid=grid),
        },
    )


def alpha_report(image: PILImage) -> dict[str, Any]:
    """Two numbers about a finished cutout, both advisory.

    ``islands`` catches a matte that came apart -- a sword whose crossguard
    became a separate object is something the user should see before they ship
    a set. ``partial_fraction`` is the soft-rim measure: a flood-fill fallback
    matte has a hard edge and scores zero, while a model matte legitimately
    leaves a rim, so this is read as "which matte produced this", not as a
    fault.
    """
    import cv2
    import numpy as np

    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    solid = (alpha > _ALPHA_FLOOR).astype(np.uint8)
    count, labels = cv2.connectedComponents(solid, connectivity=8)
    islands = 0
    if count > 1:
        sizes = np.bincount(labels.ravel(), minlength=count)[1:]
        islands = int((sizes >= MIN_ISLAND_PX).sum())
    opaque = alpha > _ALPHA_FLOOR
    partial = np.logical_and(opaque, alpha < _ALPHA_CEIL)
    total = int(opaque.sum())
    return {
        "islands": islands,
        "partial_fraction": (float(partial.sum()) / total) if total else 0.0,
    }


def recipe_hash(recipe: dict[str, Any] | None) -> str | None:
    """A short, stable fingerprint of what produced the source image.

    Recorded in the manifest so an exported set can be traced back to the run
    that made it. Sorted keys, so two dicts that say the same thing hash the
    same however they were assembled -- and short, because this is a label in
    a JSON file a human reads, not a cryptographic claim.
    """
    if not recipe:
        return None
    blob = json.dumps(recipe, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]

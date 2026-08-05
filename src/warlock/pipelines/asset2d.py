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
    canvas.paste(resized, ((size - target[0]) // 2, (size - target[1]) // 2), resized)
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

    No canvas and no resize: a sprite is placed by its pivot, so padding it to
    a square would only move the pivot away from the thing it is meant to
    anchor. Bottom-centre is the default because that is where an engine puts
    a standing character's feet, and it is recorded rather than assumed --
    an importer that guesses is an importer that is wrong for half a set.
    """
    from PIL import Image

    cropped, box = _trimmed(image, mask)
    if pad:
        margin = int(round(max(cropped.size) * pad))
        canvas = Image.new(
            "RGBA",
            (cropped.width + 2 * margin, cropped.height + 2 * margin),
            (0, 0, 0, 0),
        )
        canvas.paste(cropped, (margin, margin), cropped)
        cropped = canvas
    return (
        cropped,
        {
            "kind": "sprite",
            "canvas": [cropped.width, cropped.height],
            "trim": list(box),
            "source": [image.width, image.height],
            "pivot": [cropped.width / 2.0, float(cropped.height)],
            "pivot_rule": "bottom-centre",
        },
    )


def pixel(
    image: PILImage, mask: Any, *, size: int, colors: int = 0
) -> tuple[PILImage, dict[str, Any]]:
    """A downsampled, optionally palette-limited cutout.

    Nearest neighbour, never a filter: a resample that blends puts a ramp of
    in-between colours along every edge, and hard edges are the one property
    that makes the result read as pixel art rather than as a small photograph.

    The quantization runs on RGB with alpha carried around it, because Pillow's
    median cut treats alpha as a fourth channel to spend palette entries on --
    which on a cutout means most of the palette describing the transparent
    background.
    """
    import numpy as np
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
        small = flat.convert("RGBA")
        small.putalpha(alpha)
        # Re-cut to the alpha we started with: quantize is nearest-in-palette
        # per pixel and knows nothing about the cutout, so a background pixel
        # can pick up a subject colour and reappear once alpha is restored.
        arr = np.array(small)
        arr[:, :, 3] = np.asarray(alpha)
        small = Image.fromarray(arr, "RGBA")
    return (
        small,
        {
            "kind": "pixel",
            "size": int(size),
            "canvas": [small.width, small.height],
            "trim": list(box),
            "source": [image.width, image.height],
            "palette": palette,
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

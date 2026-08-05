"""Making an EEVEE render and an SDXL reference comparable.

This is the heart of whether any of the automatic metrics mean anything. The
two images differ in three ways that have nothing to do with mesh quality --
background colour, framing, and scale -- and all three have to be cancelled
before a number is taken, or the metric measures the renderer instead of the
reconstruction.

What is *not* cancelled is the style gap: an EEVEE render will never look
like SDXL concept art. That difference is constant across recipes and cancels
in an A-against-B, which is exactly why the scores are only ever read as a
comparison.

Pure and torch-free; Pillow and numpy imported inside the functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# What both images are resized to. 224 is DINOv2's native input, so the
# reference pipeline resizes once rather than twice.
PAIR_SIZE = 224

# Relative margin left around the subject after cropping. Applied identically
# to both sides of the pair, which is the only property that matters -- an
# absolute pixel margin would mean different things at different crop sizes.
MARGIN = 0.06

# Alpha above which a rendered pixel counts as subject. The renders are
# transparent-background PNGs, so this is a hard edge with a little slack for
# the antialiased rim.
ALPHA_THRESHOLD = 8


def border_colour(image: Any) -> tuple[int, int, int]:
    """The reference's background, sampled from its outermost ring.

    PROMPT_TEMPLATE guarantees a plain light-gray background, so the ring
    median is the background colour rather than an average of it and the
    subject.
    """
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    ring = np.concatenate(
        [arr[0].reshape(-1, 3), arr[-1].reshape(-1, 3), arr[:, 0], arr[:, -1]]
    )
    return tuple(int(v) for v in np.median(ring, axis=0))  # type: ignore[return-value]


def render_mask(image: Any):
    """The rendered subject, from its alpha. None when nothing was rendered."""
    import numpy as np

    if "A" not in image.getbands():
        return None
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    mask = alpha > ALPHA_THRESHOLD
    return mask if mask.any() else None


def reference_mask(image: Any):
    """The reference's subject, from the same corner flood fill the app uses
    before every trellis upload -- so the benchmark and the app agree about
    what the subject is."""
    from ..pipelines import reference

    mask = reference.subject_mask(image)
    return mask if mask.any() else None


def _bbox(mask) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _square_crop(image: Any, box: tuple[int, int, int, int], background) -> Any:
    """Crop to ``box``, then pad to a square with a relative margin.

    Squared *after* cropping rather than before: the point is that both images
    end up with the subject at the same relative size in the frame, and a
    square taken from the original would carry each image's own framing with
    it.
    """
    from PIL import Image

    cropped = image.crop(box)
    side = int(round(max(cropped.width, cropped.height) * (1.0 + 2 * MARGIN)))
    canvas = Image.new("RGB", (side, side), background)
    canvas.paste(
        cropped.convert("RGB"),
        ((side - cropped.width) // 2, (side - cropped.height) // 2),
    )
    return canvas


def prepare_pair(
    reference_path: Path, render_path: Path, *, size: int = PAIR_SIZE
) -> tuple[Any, Any]:
    """-> (reference, render), both RGB, square, ``size`` px, on one background.

    Returns ``(None, None)`` when either side has no subject -- an
    all-transparent render is a failed reconstruction, which is a score of
    zero rather than an exception that kills a 160-item run.
    """
    from PIL import Image

    with Image.open(reference_path) as ref_im:
        ref_im.load()
        background = border_colour(ref_im)
        ref_mask = reference_mask(ref_im)
        ref_box = _bbox(ref_mask) if ref_mask is not None else None
        ref_rgb = ref_im.convert("RGB")

    with Image.open(render_path) as render_im:
        render_im.load()
        mask = render_mask(render_im)
        box = _bbox(mask) if mask is not None else None
        if box is None or ref_box is None:
            return (None, None)
        # Composited over the *reference's* background, so background hue
        # cancels between the two rather than becoming a difference the metric
        # can see.
        flat = Image.new("RGB", render_im.size, background)
        flat.paste(render_im.convert("RGBA"), (0, 0), render_im.convert("RGBA"))

    ref_out = _square_crop(ref_rgb, ref_box, background).resize((size, size), Image.LANCZOS)
    render_out = _square_crop(flat, box, background).resize((size, size), Image.LANCZOS)
    return (ref_out, render_out)


def prepare_references(
    a_path: Path, b_path: Path, *, size: int = PAIR_SIZE
) -> tuple[Any, Any]:
    """-> two references, cropped to their subjects and made comparable.

    The same crop-and-square treatment ``prepare_pair`` gives a render, but
    both sides measured with ``reference_mask``: two SDXL images are opaque,
    so the alpha route that identifies a rendered subject finds nothing at all
    and would score every pair as None.

    The background is taken from the *first* image, so a candidate is judged
    against the anchor's background rather than the two differing in a way the
    metric can see.
    """
    from PIL import Image

    prepared = []
    background = None
    for path in (a_path, b_path):
        with Image.open(path) as im:
            im.load()
            if background is None:
                background = border_colour(im)
            mask = reference_mask(im)
            box = _bbox(mask) if mask is not None else None
            if box is None:
                return (None, None)
            prepared.append(
                _square_crop(im.convert("RGB"), box, background).resize(
                    (size, size), Image.LANCZOS
                )
            )
    return (prepared[0], prepared[1])

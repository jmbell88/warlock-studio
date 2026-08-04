"""Blend-mode arithmetic and the layer compositor.

The formulas are the W3C separable blend modes -- the same ones OpenRaster's
``svg:*`` composite-ops are defined against, so a document saved here and
reopened in Krita composites identically. That is the whole reason not to
invent something prettier.

Everything in this module speaks *straight* (non-premultiplied) alpha, float32,
channels last, values in 0..1. Straight alpha is the format at every boundary
this app has -- Pillow, PNG, ORA, the moderngl upload -- so premultiplying
internally would mean converting twice per stroke to save one multiply.
"""

from __future__ import annotations

import numpy as np

# Names are ours; the ORA op is what goes on disk. ``add`` is svg:plus, which
# is a *compositing* op rather than a blend mode in the spec -- for opaque
# layers the two agree, and no other writer spells additive differently.
BLEND_MODES: tuple[str, ...] = ("normal", "multiply", "screen", "overlay", "add")

ORA_OPS: dict[str, str] = {
    "normal": "svg:src-over",
    "multiply": "svg:multiply",
    "screen": "svg:screen",
    "overlay": "svg:overlay",
    "add": "svg:plus",
}

# Read side only, and deliberately lossy: an op we cannot reproduce becomes
# normal rather than refusing the file.
OPS_ORA = {v: k for k, v in ORA_OPS.items()}


def blend(backdrop: np.ndarray, source: np.ndarray, mode: str) -> np.ndarray:
    """B(Cb, Cs) for a separable blend mode. Colour only; alpha is not here."""
    if mode == "normal":
        return source
    if mode == "multiply":
        return backdrop * source
    if mode == "screen":
        return backdrop + source - backdrop * source
    if mode == "overlay":
        # hard-light with the operands swapped, which is the spec's own wording
        return np.where(
            backdrop <= 0.5,
            2.0 * backdrop * source,
            1.0 - 2.0 * (1.0 - backdrop) * (1.0 - source),
        )
    if mode == "add":
        return np.minimum(backdrop + source, 1.0)
    raise ValueError(f"unknown blend mode {mode!r}")


def over(
    backdrop: np.ndarray, source: np.ndarray, *, opacity: float = 1.0, mode: str = "normal"
) -> np.ndarray:
    """Composite ``source`` onto ``backdrop``; both (H, W, 4) float32 straight.

    The general form from the compositing spec, in straight alpha:

        ao = as + ab(1 - as)
        Co = [ as(1-ab)Cs + as·ab·B(Cb,Cs) + (1-as)ab·Cb ] / ao

    which collapses to a plain lerp when the blend is normal, and to the
    backdrop wherever the source is empty -- the ``ao == 0`` guard is not an
    edge case, it is most of a brush stamp's bounding box.
    """
    if opacity >= 1.0 and mode == "normal" and float(source[..., 3].min()) >= 1.0:
        return source.copy()

    cb = backdrop[..., :3]
    cs = source[..., :3]
    ab = backdrop[..., 3:4]
    a_s = source[..., 3:4] * float(opacity)

    ao = a_s + ab * (1.0 - a_s)
    mixed = blend(cb, cs, mode) if mode != "normal" else cs
    num = a_s * (1.0 - ab) * cs + a_s * ab * mixed + (1.0 - a_s) * ab * cb

    out = np.empty_like(backdrop)
    np.divide(num, ao, out=out[..., :3], where=ao > 0.0)
    out[..., :3] = np.where(ao > 0.0, out[..., :3], 0.0)
    out[..., 3:4] = ao
    return out


def paint_colour(
    before: np.ndarray, colour: tuple[int, int, int, int], weight: np.ndarray
) -> np.ndarray:
    """Write ``colour`` over ``before`` at per-pixel ``weight``. 0..255 float32.

    The one formula every colour-writing tool shares -- brush, fill, gradient,
    shape -- so that a soft edge, a feathered selection and a gradient ramp all
    mean the same thing by "half". Straight alpha throughout: painting opaque
    red onto emptiness gives red, not red faded toward black, which is what a
    naive lerp against a transparent backdrop produces.
    """
    src_a = (colour[3] / 255.0) * weight
    dst_a = before[..., 3] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    share = np.divide(src_a, out_a, out=np.zeros_like(src_a), where=out_a > 0.0)
    rgb = np.array(colour[:3], dtype=np.float32)
    out = np.empty_like(before)
    out[..., :3] = before[..., :3] + (rgb - before[..., :3]) * share[..., None]
    out[..., 3] = out_a * 255.0
    return out


def stack_region(
    entries: list[tuple[np.ndarray, float, str]],
    rect: tuple[int, int, int, int],
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Composite a crop of several layers, bottom-first, onto ``base``.

    Deliberately takes tuples rather than ``Layer`` objects: the arithmetic has
    no opinion about names or ids, and keeping this module free of the layer
    model is what lets the below-cache be tested against a naive full-stack
    composite without the two sharing an implementation.
    """
    x0, y0, x1, y1 = rect
    if base is None:
        out = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
    else:
        out = base.astype(np.float32, copy=True)
    for pixels, opacity, mode in entries:
        out = over(out, to_float(pixels[y0:y1, x0:x1]), opacity=opacity, mode=mode)
    return out


# -- conversions -------------------------------------------------------------
#
# uint8 is the storage format (it is what a layer holds and what goes to GL);
# float32 is the arithmetic format. Keeping the two conversions in one place is
# what stops a stray /255 from turning up in a blend formula.


def to_float(pixels: np.ndarray) -> np.ndarray:
    return pixels.astype(np.float32) / 255.0


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    return np.clip(pixels * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def empty(width: int, height: int) -> np.ndarray:
    """A fully transparent layer-sized buffer."""
    return np.zeros((int(height), int(width), 4), dtype=np.uint8)


def flatten_onto(pixels: np.ndarray, matte: tuple[int, int, int, int] | None) -> np.ndarray:
    """Put ``matte`` behind the composite, or leave the alpha alone if None.

    This is where the old ``erase_color`` decision moved to. The eraser now
    always cuts alpha; whether the *flattened* export shows white or shows
    through is a property of the document, decided once at load, and applied
    once here -- so the choice is visible in the file rather than baked into
    every stroke.
    """
    if matte is None:
        return pixels
    back = np.empty_like(pixels, dtype=np.float32)
    back[..., 0] = matte[0] / 255.0
    back[..., 1] = matte[1] / 255.0
    back[..., 2] = matte[2] / 255.0
    back[..., 3] = matte[3] / 255.0
    return to_uint8(over(back, to_float(pixels)))

"""Blend-mode arithmetic and the layer compositor.

The formulas are the W3C separable blend modes -- the same ones OpenRaster's
``svg:*`` composite-ops are defined against, so a document saved here and
reopened in Krita composites identically. That is the whole reason not to
invent something prettier.

Everything in this module speaks *straight* (non-premultiplied) alpha, float32,
channels last, values in 0..1. Straight alpha is the format at every boundary
this app has -- Pillow, PNG, ORA, the moderngl upload -- so premultiplying
internally would mean converting twice per stroke to save one multiply.

``over`` and ``paint_colour`` each have a native kernel behind
``native.available()``, because the cost here was never the arithmetic: the
numpy body of ``over`` materialises about eight full-region temporaries and
``stack_region`` runs it once per layer, on the frame thread. The numpy bodies
are never deleted -- they are the fallback on a machine with no compiler and the
reference ``tests/inker/test_composite_native.py`` measures the kernel against,
bit for bit. Which path ran is not observable in the result.
"""

from __future__ import annotations

import numpy as np

from ... import native

# Names are ours; the ORA op is what goes on disk. ``add`` is svg:plus, which
# is a *compositing* op rather than a blend mode in the spec -- for opaque
# layers the two agree, and no other writer spells additive differently.
#
# The order is the *menu* order, grouped by what each family does to the
# backdrop -- darkening, lightening, contrast, comparison -- which is the order
# every raster editor a user has met puts them in. Nothing keys on the position:
# ``_MODE_IDS`` is written out, ORA stores the op name and ``animation.json``
# stores ours, so this list is free to be readable.
BLEND_MODES: tuple[str, ...] = (
    "normal",
    "darken",
    "multiply",
    "color-burn",
    "lighten",
    "screen",
    "color-dodge",
    "add",
    "overlay",
    "hard-light",
    "soft-light",
    "difference",
)

# The kernel takes a mode as a number, and the enum in native/warlockc.h spells
# the same ones out. Written here rather than derived from ``enumerate`` on
# purpose: a mode that is not in this map falls back to numpy, so adding one to
# BLEND_MODES and forgetting the C case costs a little speed, whereas an
# automatic index would hand the kernel a number it does not know and get
# ``normal`` back for it -- silently, which is the one thing a fallback path
# must never be. Written-out also means the numbers are free of the list's
# order, so the menu above could be regrouped without invalidating a DLL.
_MODE_IDS: dict[str, int] = {
    "normal": 0,
    "multiply": 1,
    "screen": 2,
    "overlay": 3,
    "add": 4,
    "darken": 5,
    "lighten": 6,
    "color-dodge": 7,
    "color-burn": 8,
    "hard-light": 9,
    "soft-light": 10,
    "difference": 11,
}

# Not a mode: what ``over``'s early-out does, spelled so the fused stack kernel
# can be told about it. The test behind it is a reduction over the whole region
# rather than a per-pixel one, so it stays on this side either way.
_MODE_REPLACE = -1

ORA_OPS: dict[str, str] = {
    "normal": "svg:src-over",
    "multiply": "svg:multiply",
    "screen": "svg:screen",
    "overlay": "svg:overlay",
    "add": "svg:plus",
    "darken": "svg:darken",
    "lighten": "svg:lighten",
    "color-dodge": "svg:color-dodge",
    "color-burn": "svg:color-burn",
    "hard-light": "svg:hard-light",
    "soft-light": "svg:soft-light",
    "difference": "svg:difference",
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
    if mode == "darken":
        return np.minimum(backdrop, source)
    if mode == "lighten":
        return np.maximum(backdrop, source)
    if mode == "difference":
        return np.abs(backdrop - source)
    if mode == "hard-light":
        # *Literally* overlay with the operands swapped, rather than the same
        # formula written out with the tests inverted. The two are equal on
        # paper and a hand-written copy would be equal to within a rounding of
        # the products, which is not the bar the kernel is held to -- and the
        # kernel says it the same way, so the relationship is true by
        # construction on both paths rather than checked on one.
        return blend(source, backdrop, "overlay")
    if mode == "color-dodge":
        # The spec's three cases, in its order: an empty backdrop stays empty
        # even under a full source. Dividing is guarded rather than clipped
        # afterwards because ``1 - Cs`` is exactly zero for a saturated channel,
        # which is most of a white brush stroke.
        denom = 1.0 - source
        ratio = np.minimum(backdrop / np.where(denom > 0.0, denom, 1.0), 1.0)
        return np.where(backdrop <= 0.0, 0.0, np.where(source >= 1.0, 1.0, ratio))
    if mode == "color-burn":
        ratio = np.minimum((1.0 - backdrop) / np.where(source > 0.0, source, 1.0), 1.0)
        return np.where(backdrop >= 1.0, 1.0, np.where(source <= 0.0, 0.0, 1.0 - ratio))
    if mode == "soft-light":
        # D(Cb): the spec's quartic below a quarter, a square root above it.
        # The clamp under the root is not defensive about the spec -- Cb is a
        # composited channel and ``over``'s divide is not clipped, so a value a
        # hair below zero would otherwise raise an invalid-value warning out of
        # a paint stroke.
        d = np.where(
            backdrop <= 0.25,
            ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop,
            np.sqrt(np.maximum(backdrop, 0.0)),
        )
        return np.where(
            source <= 0.5,
            backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop),
            backdrop + (2.0 * source - 1.0) * (d - backdrop),
        )
    raise ValueError(f"unknown blend mode {mode!r}")


# -- the native seam ---------------------------------------------------------
#
# Two helpers, both shaped the same way: return the kernel's answer, or None to
# mean "not this call's shape" so the caller runs the numpy body it was always
# going to run. Nothing below raises -- a refusal is a fallback, never an error.


def _row_stride(array: np.ndarray, channels: int) -> int | None:
    """Floats between row starts, or None if the kernels cannot take this array.

    They index ``row * stride + x * channels``, which is exactly the shape a
    rect slice of a larger canvas has -- so a crop passes without a copy, and
    anything else (a transpose, a strided channel view, float64) is numpy's.
    """
    if array.dtype != np.float32:
        return None
    item = array.itemsize
    if channels == 1:
        indexable = array.ndim == 2 and array.strides[1] == item
    else:
        indexable = (
            array.ndim == 3
            and array.shape[2] == channels
            and array.strides[2] == item
            and array.strides[1] == channels * item
        )
    if not indexable:
        return None
    row = array.strides[0]
    return row // item if row % item == 0 else None


def _over_native(
    backdrop: np.ndarray, source: np.ndarray, opacity: float, mode: str
) -> np.ndarray | None:
    code = _MODE_IDS.get(mode)
    if code is None or not native.available():
        return None
    if backdrop.ndim != 3 or backdrop.shape != source.shape:
        return None
    height, width, _ = backdrop.shape
    if height == 0 or width == 0:
        return None
    back_stride = _row_stride(backdrop, 4)
    src_stride = _row_stride(source, 4)
    if back_stride is None or src_stride is None:
        return None
    out = np.empty((height, width, 4), dtype=np.float32)
    native.over_f32(
        backdrop, back_stride, source, src_stride, out, width * 4, height, width, opacity, code
    )
    return out


def _paint_colour_native(
    before: np.ndarray, colour: tuple[int, int, int, int], weight: np.ndarray
) -> np.ndarray | None:
    if not native.available() or before.ndim != 3 or before.shape[2] != 4:
        return None
    height, width, _ = before.shape
    if height == 0 or width == 0 or weight.shape != (height, width):
        return None
    if not float(colour[3]).is_integer():
        # ``colour[3] / 255.0`` is a Python scalar, so numpy computes it in
        # double and rounds it to float32 once, at the multiply. The kernel is
        # handed the component already rounded to float32 and can only agree
        # when that rounding was lossless. Every integer 0..255 is; the type
        # says they all are; this is what happens when one is not.
        return None
    before_stride = _row_stride(before, 4)
    weight_stride = _row_stride(weight, 1)
    if before_stride is None or weight_stride is None:
        return None
    out = np.empty((height, width, 4), dtype=np.float32)
    native.paint_colour_f32(
        before, before_stride, weight, weight_stride, out, width * 4, height, width, colour
    )
    return out


def _stack_native(
    entries: list[tuple[np.ndarray, float, str]],
    rect: tuple[int, int, int, int],
    base: np.ndarray | None,
) -> np.ndarray | None:
    """The whole fold in one call, or None if any layer is not the kernel's shape.

    All-or-nothing on purpose: a mixed run would have to hand the kernel a
    partial result and resume, which is more seam than the saving is worth.
    """
    if not native.available():
        return None
    x0, y0, x1, y1 = rect
    height, width = y1 - y0, x1 - x0
    if height <= 0 or width <= 0 or x0 < 0 or y0 < 0:
        return None

    crops: list[np.ndarray] = []
    strides: list[int] = []
    opacities: list[float] = []
    modes: list[int] = []
    for pixels, opacity, mode in entries:
        code = _MODE_IDS.get(mode)
        if code is None or pixels.dtype != np.uint8 or pixels.ndim != 3:
            return None
        crop = pixels[y0:y1, x0:x1]
        if crop.shape != (height, width, 4):
            return None
        if crop.strides[2] != 1 or crop.strides[1] != 4:
            return None
        if opacity >= 1.0 and mode == "normal" and int(crop[..., 3].min()) == 255:
            # ``over`` would return ``source.copy()`` here, and to_float(255)
            # is exactly 1.0, so the uint8 test and the float32 one it stands
            # in for agree by construction.
            code = _MODE_REPLACE
        crops.append(crop)
        strides.append(crop.strides[0])
        opacities.append(float(opacity))
        modes.append(code)

    base_stride = 0
    if base is not None:
        if base.shape != (height, width, 4):
            return None
        stride = _row_stride(base, 4)
        if stride is None:
            return None
        base_stride = stride

    out = np.empty((height, width, 4), dtype=np.float32)
    native.stack_f32(
        crops, strides, opacities, modes, out, width * 4, height, width, base, base_stride
    )
    return out


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

    fast = _over_native(backdrop, source, float(opacity), mode)
    if fast is not None:
        return fast

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
    fast = _paint_colour_native(before, colour, weight)
    if fast is not None:
        return fast

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
    fast = _stack_native(entries, rect, base)
    if fast is not None:
        return fast

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


def _to_uint8_native(pixels: np.ndarray) -> np.ndarray | None:
    """Elementwise, so the kernel wants a flat contiguous run and nothing else.

    Every caller hands it a fresh array out of ``over`` or ``stack_region``, so
    the contiguity test is a formality that costs nothing and keeps the kernel
    free of a stride it would never use.
    """
    if not native.available() or pixels.dtype != np.float32:
        return None
    if pixels.size == 0 or not pixels.flags["C_CONTIGUOUS"]:
        return None
    out = np.empty(pixels.shape, dtype=np.uint8)
    native.to_uint8_f32(pixels, out, pixels.size)
    return out


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    fast = _to_uint8_native(pixels)
    if fast is not None:
        return fast
    return np.clip(pixels * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def to_uint8_255(pixels: np.ndarray) -> np.ndarray:
    """:func:`to_uint8` for values that are already levels, not fractions.

    The brush's ``_resolve`` and ``_filter`` and the document's ``write_colour``
    and ``gradient`` all do their arithmetic in the 0..255 domain -- that is
    what ``paint_colour`` speaks -- so none of them can use :func:`to_uint8`,
    which scales first. They hand-rolled the narrowing instead; this is that
    expression in one place, with the same kernel-or-numpy seam.

    The float32 gate is not a formality: ``np.clip(x + 0.5, 0, 255)`` on a
    float64 input rounds at a different width, so a caller who let a float64
    through must get the numpy path back rather than a silently different
    answer.
    """
    if (
        native.available()
        and pixels.dtype == np.float32
        and pixels.size
        and pixels.flags["C_CONTIGUOUS"]
    ):
        out = np.empty(pixels.shape, dtype=np.uint8)
        native.to_uint8_255_f32(pixels, out, pixels.size)
        return out
    return np.clip(pixels + 0.5, 0, 255).astype(np.uint8)


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

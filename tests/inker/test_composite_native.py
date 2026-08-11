"""The native compositor against the numpy one, bit for bit.

Not ``allclose``. A blend is a chain of six or seven float32 operations and the
kernel exists only because it is faster, so the property that makes it safe is
that no reader can tell which path ran -- and "no reader" includes the ORA
round-trip, the below-cache the document compares against a from-scratch
composite, and a screenshot diff. One ulp of drift in ``over`` is invisible in
any single image and shows up as a layer cache that no longer matches the
stack it was built from.

So: same inputs, both paths, ``np.array_equal``. If one of these fails, the C
is what changes.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from warlock import native
from warlock.studio.inker import composite as cp

needs_dll = pytest.mark.skipif(not native.available(), reason="warlockc.dll not built")

OPACITIES = (0.0, 0.37, 1.0)


def _both(call, monkeypatch):
    """Run ``call`` with the kernel and without it."""
    monkeypatch.setattr(native, "available", lambda: True)
    fast = call()
    monkeypatch.setattr(native, "available", lambda: False)
    slow = call()
    return fast, slow


def _canvas(rng, height, width):
    """Random pixels, with the alphas the formula actually branches on.

    Uniform noise never produces an exactly empty or exactly opaque pixel, and
    those are the two the ``ao > 0`` guard and the opacity early-out are made
    of -- most of a brush stamp's bounding box is the first one. The tiny band
    is there because ``num / ao`` at ao = 1e-30 is where a reassociated
    expression would first disagree.
    """
    pixels = rng.random((height, width, 4), dtype=np.float32)
    alpha = pixels[..., 3]
    alpha[0::4] = 0.0
    alpha[1::4] = 1.0
    alpha[2::4] = rng.random(alpha[2::4].shape, dtype=np.float32) * np.float32(1e-30)
    return pixels


@needs_dll
@pytest.mark.parametrize("mode", cp.BLEND_MODES)
@pytest.mark.parametrize("opacity", OPACITIES)
def test_over_is_bit_identical_on_both_paths(mode, opacity, monkeypatch):
    rng = np.random.default_rng(20260806)
    backdrop = _canvas(rng, 37, 53)
    source = _canvas(rng, 37, 53)

    fast, slow = _both(
        lambda: cp.over(backdrop, source, opacity=opacity, mode=mode), monkeypatch
    )
    assert fast.dtype == slow.dtype
    assert np.array_equal(fast, slow)


@needs_dll
@pytest.mark.parametrize("mode", cp.BLEND_MODES)
def test_over_agrees_on_a_rect_slice_of_a_larger_canvas(mode, monkeypatch):
    """The shape ``stack_region`` actually hands it.

    A crop is a strided view, not a copy, and the kernel takes a row stride
    precisely so it does not have to become one. Getting the stride wrong reads
    the wrong rows, which on a uniform image looks exactly like getting it
    right.
    """
    rng = np.random.default_rng(11)
    whole_back = _canvas(rng, 64, 96)
    whole_src = _canvas(rng, 64, 96)
    back = whole_back[9:41, 13:70]
    src = whole_src[9:41, 13:70]
    assert not back.flags["C_CONTIGUOUS"]

    fast, slow = _both(lambda: cp.over(back, src, opacity=0.62, mode=mode), monkeypatch)
    assert np.array_equal(fast, slow)


@needs_dll
def test_over_leaves_the_backdrop_alone_when_it_writes_a_fresh_array(monkeypatch):
    """The kernel may alias, the wrapper does not -- and callers rely on that:
    ``stack_region`` folds layer n+1 onto the result of layer n and the brush's
    ``before`` buffer is read again on the next dab."""
    rng = np.random.default_rng(3)
    backdrop = _canvas(rng, 8, 8)
    source = _canvas(rng, 8, 8)
    keep = backdrop.copy()

    monkeypatch.setattr(native, "available", lambda: True)
    out = cp.over(backdrop, source, opacity=0.5, mode="multiply")
    assert np.array_equal(backdrop, keep)
    assert out is not backdrop


@needs_dll
@pytest.mark.parametrize(
    "colour",
    [(255, 0, 0, 255), (12, 200, 90, 128), (0, 0, 0, 0), (255, 255, 255, 1)],
)
def test_paint_colour_is_bit_identical_on_both_paths(colour, monkeypatch):
    rng = np.random.default_rng(4242)
    before = rng.random((29, 41, 4), dtype=np.float32) * np.float32(255.0)
    weight = rng.random((29, 41), dtype=np.float32)
    # The three the tools actually produce: an untouched pixel, the half a
    # feathered selection and a soft brush edge both mean, and full coverage.
    weight[0::3] = 0.0
    weight[1::3] = np.float32(128.0 / 255.0)
    weight[2::3] = 1.0

    fast, slow = _both(lambda: cp.paint_colour(before, colour, weight), monkeypatch)
    assert fast.dtype == slow.dtype
    assert np.array_equal(fast, slow)


@needs_dll
def test_paint_colour_agrees_on_a_clipped_region(monkeypatch):
    rng = np.random.default_rng(5)
    whole = rng.random((48, 64, 4), dtype=np.float32) * np.float32(255.0)
    before = whole[5:33, 7:50]
    weight = rng.random((48, 64), dtype=np.float32)[5:33, 7:50]
    assert not before.flags["C_CONTIGUOUS"]

    fast, slow = _both(lambda: cp.paint_colour(before, (30, 60, 90, 200), weight), monkeypatch)
    assert np.array_equal(fast, slow)


# --- the fused fold -----------------------------------------------------------
#
# stack_region is where invalidate_all spends its time, and the fused kernel
# does not call the `over` one -- it inlines the same arithmetic with the
# u8 -> float conversion folded into the load. So it needs its own parity
# tests rather than inheriting the ones above.


def _layers(rng, height, width, count):
    return [rng.integers(0, 256, (height, width, 4), dtype=np.uint8) for _ in range(count)]


@needs_dll
def test_a_whole_stack_composites_identically(monkeypatch):
    """The integration shape: a deep stack folded bottom-first, every mode in
    play, which is what ``invalidate_all`` runs and the reason any of this
    exists.

    **One layer per mode, derived** -- a hand-written list of six was six
    because there were five modes, and adding one turned this into a zip-length
    error rather than into coverage of the new case.
    """
    rng = np.random.default_rng(77)
    opacities = itertools.cycle((1.0, 0.5, 0.9, 0.25, 0.66))
    entries = list(
        zip(
            _layers(rng, 40, 40, len(cp.BLEND_MODES)),
            opacities,
            cp.BLEND_MODES,
            strict=False,
        )
    )
    assert len(entries) == len(cp.BLEND_MODES)

    fast, slow = _both(lambda: cp.stack_region(entries, (4, 6, 36, 33)), monkeypatch)
    assert np.array_equal(fast, slow)


@needs_dll
def test_a_stack_onto_a_base_composites_identically(monkeypatch):
    """The below-cache path: ``composite_region`` hands the fold a float32 base
    that is itself a strided crop of the full-canvas cached composite."""
    rng = np.random.default_rng(101)
    entries = [(pixels, 0.7, "screen") for pixels in _layers(rng, 40, 40, 3)]
    below = rng.random((40, 40, 4), dtype=np.float32)
    base = below[6:33, 4:36]
    assert not base.flags["C_CONTIGUOUS"]

    fast, slow = _both(lambda: cp.stack_region(entries, (4, 6, 36, 33), base), monkeypatch)
    assert np.array_equal(fast, slow)


@needs_dll
def test_an_opaque_normal_layer_takes_the_early_out_on_both_paths(monkeypatch):
    """``over`` returns ``source.copy()`` for a fully opaque normal layer at
    full opacity, which is not what the per-pixel formula produces in the last
    ulp -- so the fold has to be told, and it is told by the same whole-region
    test. A Background layer is exactly this and is in every document."""
    rng = np.random.default_rng(202)
    under, over_layer = _layers(rng, 24, 24, 2)
    opaque = rng.integers(0, 256, (24, 24, 4), dtype=np.uint8)
    opaque[..., 3] = 255
    entries = [(under, 0.8, "multiply"), (opaque, 1.0, "normal"), (over_layer, 0.5, "overlay")]

    fast, slow = _both(lambda: cp.stack_region(entries, (0, 0, 24, 24)), monkeypatch)
    assert np.array_equal(fast, slow)

    # And one alpha short of opaque must *not* take it, or the two paths agree
    # only because they are both wrong.
    opaque[3, 3, 3] = 254
    fast, slow = _both(lambda: cp.stack_region(entries, (0, 0, 24, 24)), monkeypatch)
    assert np.array_equal(fast, slow)


@needs_dll
def test_an_empty_stack_is_transparent_black_on_both_paths(monkeypatch):
    """A document with every layer hidden. ``_entries`` filters them out, so
    the fold is handed nothing at all."""
    fast, slow = _both(lambda: cp.stack_region([], (0, 0, 8, 8)), monkeypatch)
    assert np.array_equal(fast, slow)
    assert not fast.any()


@needs_dll
def test_a_stack_of_float_layers_falls_back(monkeypatch):
    """A layer holds uint8; the fold refuses anything else rather than
    guessing, and the numpy body still answers."""
    monkeypatch.setattr(native, "available", lambda: True)
    entries = [(np.zeros((4, 4, 4), dtype=np.float32), 1.0, "normal")]
    assert cp._stack_native(entries, (0, 0, 4, 4), None) is None
    assert cp.stack_region(entries, (0, 0, 4, 4)).shape == (4, 4, 4)


@needs_dll
def test_a_whole_editing_session_produces_identical_pixels(monkeypatch):
    """End to end, because that is where a divergence would actually surface.

    The kernels are reached three different ways here -- a stroke resolves
    through ``paint_colour``, the dirty rect recomposites through the fused
    fold onto the below-cache, and the reorder rebuilds everything through
    ``invalidate_all``. Bit-parity on the arrays is one claim; that the
    document's *cached* composite is the same array either way is the one that
    would show as an editor displaying one image and saving another.
    """
    from warlock.studio import inker

    def session():
        rng = np.random.default_rng(909)
        layers = [
            inker.Layer(
                pixels=rng.integers(0, 256, (48, 48, 4), dtype=np.uint8),
                opacity=0.4 + 0.15 * i,
                blend=inker.BLEND_MODES[i % len(inker.BLEND_MODES)],
            )
            for i in range(4)
        ]
        doc = inker.Document(stack=inker.LayerStack(layers, 2))
        doc.begin_stroke((6, 6), (220, 40, 90, 255), size=9)
        doc.stroke_to((30, 22))
        doc.stroke_to((41, 40))
        doc.end_stroke()
        doc.stack.move(3, 0)
        doc.invalidate_all()
        doc.stack.layers[1].visible = False
        doc.invalidate_all()
        return doc.composite.copy(), doc.stack.flatten()

    fast, slow = _both(session, monkeypatch)
    assert np.array_equal(fast[0], slow[0])
    assert np.array_equal(fast[0], fast[1])


# --- the narrowing ------------------------------------------------------------


@needs_dll
def test_to_uint8_is_bit_identical_on_both_paths(monkeypatch):
    """Rounding, clamping and truncation in one pass rather than three arrays.

    The values that matter are the ones either side of a level boundary: the
    reference adds 0.5 and then *truncates*, so getting the two steps in the
    wrong order in C shifts every pixel by half a level -- which is invisible
    on a gradient and is a different file on disk.
    """
    rng = np.random.default_rng(31337)
    pixels = rng.random((23, 47, 4), dtype=np.float32)
    # Exact level boundaries, the two ends, and just outside both.
    pixels[0] = np.float32(0.0)
    pixels[1] = np.float32(1.0)
    pixels[2] = (np.arange(47 * 4, dtype=np.float32) / np.float32(255.0)).reshape(47, 4)
    pixels[3] = ((np.arange(47 * 4, dtype=np.float32) + 0.5) / np.float32(255.0)).reshape(47, 4)
    pixels[4] = np.float32(-0.004)
    pixels[5] = np.float32(1.004)

    fast, slow = _both(lambda: cp.to_uint8(pixels), monkeypatch)
    assert fast.dtype == slow.dtype == np.uint8
    assert np.array_equal(fast, slow)


@needs_dll
def test_to_uint8_round_trips_every_level_on_both_paths(monkeypatch):
    """The property ``test_uint8_survives_a_round_trip_through_float`` already
    asserts, run against the kernel: all 256 levels come back unchanged."""
    values = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(4, axis=2)
    fast, slow = _both(lambda: cp.to_uint8(cp.to_float(values)), monkeypatch)
    assert np.array_equal(fast, values)
    assert np.array_equal(fast, slow)


@needs_dll
def test_to_uint8_refuses_what_it_cannot_index_and_numpy_answers(monkeypatch):
    monkeypatch.setattr(native, "available", lambda: True)
    strided = np.zeros((8, 8, 4), dtype=np.float32)[:, ::2]
    assert cp._to_uint8_native(strided) is None
    assert cp._to_uint8_native(np.zeros((2, 2), dtype=np.float64)) is None
    assert cp.to_uint8(strided).shape == (8, 4, 4)


@needs_dll
def test_to_uint8_255_is_bit_identical_on_both_paths(monkeypatch):
    """The unscaled sibling, over the values its four call sites produce.

    Same trap as the scaled one and a sharper version of it: these values are
    already levels, so a half-level shift here is not a rounding question about
    a fraction, it is the wrong byte written straight into a layer.
    """
    rng = np.random.default_rng(4242)
    pixels = rng.random((23, 47, 4), dtype=np.float32) * np.float32(255.0)
    # Exact integers, exact halves, both ends, and just outside both -- the
    # last two because paint_colour's divide can overshoot by an ulp.
    pixels[0] = np.arange(47 * 4, dtype=np.float32).reshape(47, 4) % np.float32(256.0)
    pixels[1] = (np.arange(47 * 4, dtype=np.float32) + 0.5).reshape(47, 4) % np.float32(256.0)
    pixels[2] = np.float32(0.0)
    pixels[3] = np.float32(255.0)
    pixels[4] = np.float32(-0.4)
    pixels[5] = np.float32(255.6)

    fast, slow = _both(lambda: cp.to_uint8_255(pixels), monkeypatch)
    assert fast.dtype == slow.dtype == np.uint8
    assert np.array_equal(fast, slow)
    assert np.array_equal(slow, np.clip(pixels + 0.5, 0, 255).astype(np.uint8))


@needs_dll
def test_to_uint8_255_round_trips_every_level_on_both_paths(monkeypatch):
    """A layer's own pixels, widened and narrowed again, are themselves."""
    values = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(4, axis=2)
    call = lambda: cp.to_uint8_255(values.astype(np.float32))  # noqa: E731
    fast, slow = _both(call, monkeypatch)
    assert np.array_equal(fast, values)
    assert np.array_equal(fast, slow)


def test_to_uint8_255_refuses_what_it_cannot_index_and_numpy_answers(monkeypatch):
    """A non-contiguous or non-float32 input must reach numpy, not the kernel.

    float64 is the one that matters: ``x + 0.5`` rounds at a different width,
    so a kernel that took it would answer a different question rather than the
    same one faster.
    """
    monkeypatch.setattr(native, "available", lambda: True)
    strided = np.zeros((8, 8, 4), dtype=np.float32)[:, ::2]
    assert cp.to_uint8_255(strided).shape == (8, 4, 4)
    wide = np.full((2, 2), 127.5, dtype=np.float64)
    assert np.array_equal(cp.to_uint8_255(wide), np.full((2, 2), 128, dtype=np.uint8))
    assert cp.to_uint8_255(np.zeros((0, 4), dtype=np.float32)).shape == (0, 4)


# --- the refusals -------------------------------------------------------------
#
# ``_over_native`` returning None is not an error path, it is the whole design:
# a shape the kernel cannot index has to reach the numpy body and come back
# correct. These pin that the refusal happens *and* that the answer survives it.


@needs_dll
def test_float64_input_falls_back_and_still_composites(monkeypatch):
    monkeypatch.setattr(native, "available", lambda: True)
    backdrop = np.zeros((2, 2, 4), dtype=np.float64)
    source = np.full((2, 2, 4), 0.5, dtype=np.float64)

    assert cp._over_native(backdrop, source, 1.0, "normal") is None
    out = cp.over(backdrop, source, mode="multiply")
    assert out.dtype == np.float64
    assert np.allclose(out[..., :3], 0.5)


@needs_dll
def test_a_mode_the_kernel_does_not_know_never_reaches_it(monkeypatch):
    """``_MODE_IDS`` is written out rather than derived from ``BLEND_MODES``,
    so a mode added to one and not the other loses speed instead of silently
    compositing as normal -- and an outright unknown mode still raises."""
    monkeypatch.setattr(native, "available", lambda: True)
    px = np.zeros((1, 1, 4), dtype=np.float32)
    assert cp._over_native(px, px, 1.0, "dodge") is None
    with pytest.raises(ValueError):
        cp.over(px, px, mode="dodge")


@needs_dll
def test_an_empty_region_falls_back(monkeypatch):
    """A zero-width crop has meaningless strides, so the kernel never sees one."""
    monkeypatch.setattr(native, "available", lambda: True)
    empty = np.zeros((0, 5, 4), dtype=np.float32)
    assert cp._over_native(empty, empty, 1.0, "normal") is None
    assert cp.over(empty, empty, mode="screen").shape == (0, 5, 4)


@needs_dll
def test_a_transposed_view_falls_back(monkeypatch):
    """Channels have to be adjacent within a pixel; a transpose puts a whole
    row between them and there is no stride that says so."""
    monkeypatch.setattr(native, "available", lambda: True)
    view = np.zeros((4, 6, 8), dtype=np.float32).transpose(2, 1, 0)
    assert cp._row_stride(view, 4) is None


@needs_dll
@pytest.mark.parametrize("mode", cp.BLEND_MODES)
def test_over_is_bit_identical_on_a_nan_channel(mode, monkeypatch):
    """Parity holds for inputs that cannot occur, because that is the contract.

    ``composite.c``'s bar is ``np.array_equal`` against the numpy reference for
    *any* input, not agreement on the ones a uint8 layer can produce -- which
    is why the NaN substitutions in its min/max branches are written out at
    all. This pins the whole mode table against that bar in one place.

    It does **not** pin the divisor guards in colour-dodge and colour-burn,
    and the attempt to make it do so is worth recording: a NaN colour channel
    reaches the output through ``over``'s own ``k_src * cs`` term whatever
    ``B()`` returned, since ``0 * NaN`` is NaN and no alpha zeroes it out. The
    two kernels are therefore indistinguishable at this seam, which is the
    argument the C comment makes for why those guards are shape rather than
    fix. Anything that ever makes ``blend`` reachable on its own is what would
    turn this into a real test of them.

    ``array_equal`` alone would pass on two NaNs, so the assertion goes through
    ``equal_nan``: what is being pinned is that the two agree *including* on
    which cells are NaN at all.
    """
    rng = np.random.default_rng(20260811)
    backdrop = _canvas(rng, 8, 8)
    source = _canvas(rng, 8, 8)
    # A NaN in each operand's colour channels, and one in both at once.
    source[1, 1, :3] = np.nan
    backdrop[2, 2, :3] = np.nan
    source[3, 3, :3] = np.nan
    backdrop[3, 3, :3] = np.nan

    fast, slow = _both(lambda: cp.over(backdrop, source, mode=mode), monkeypatch)
    assert fast.dtype == slow.dtype
    assert np.array_equal(fast, slow, equal_nan=True), mode

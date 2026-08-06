"""The layer stack, and the one optimisation in it that can be wrong.

The below-cache is a cache, so every test that matters here compares it against
the naive full-stack composite it is standing in for. If those two ever
disagree the editor shows one thing and saves another.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import composite as cp
from warlock.studio.inker.layers import Layer, LayerStack


def _solid(w, h, rgba):
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[:, :] = rgba
    return Layer(pixels=pixels)


def _stack(*colours, active=0):
    return LayerStack([_solid(8, 8, c) for c in colours], active)


# --- the layer itself -------------------------------------------------------


def test_a_layer_must_be_rgba_uint8():
    with pytest.raises(ValueError):
        Layer(pixels=np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        Layer(pixels=np.zeros((4, 4, 4), dtype=np.float32))


def test_a_new_layer_is_transparent_not_black_and_white():
    layer = Layer.empty(4, 4)
    assert layer.pixels.max() == 0
    assert layer.size == (4, 4)


def test_a_duplicate_gets_its_own_uid_and_its_own_pixels():
    """Undo addresses layers by uid; a shared uid would let one layer's patch
    rewrite the other, and shared pixels would make the copy a mirror."""
    original = _solid(4, 4, (255, 0, 0, 255))
    copy = original.copy()
    assert copy.uid != original.uid
    copy.pixels[0, 0] = (0, 0, 255, 255)
    assert tuple(original.pixels[0, 0]) == (255, 0, 0, 255)


def test_a_layer_with_an_unknown_blend_mode_is_refused():
    with pytest.raises(ValueError):
        Layer(pixels=cp.empty(2, 2), blend="dodge")


# --- structure --------------------------------------------------------------


def test_a_stack_is_never_empty():
    with pytest.raises(ValueError):
        LayerStack([])
    stack = _stack((0, 0, 0, 255))
    with pytest.raises(ValueError):
        stack.remove(0)


def test_adding_puts_the_layer_above_the_active_one_and_selects_it():
    stack = _stack((0, 0, 0, 255), (255, 0, 0, 255), active=0)
    added = stack.add()
    assert stack.layers[1] is added
    assert stack.active is added


def test_a_layer_that_is_not_canvas_sized_is_refused():
    stack = _stack((0, 0, 0, 255))
    with pytest.raises(ValueError):
        stack.insert(0, Layer.empty(4, 4))


def test_moving_a_layer_keeps_it_active_and_reorders_the_rest():
    stack = _stack((1, 1, 1, 255), (2, 2, 2, 255), (3, 3, 3, 255))
    uids = [layer.uid for layer in stack]
    stack.move(0, 2)
    assert [layer.uid for layer in stack] == [uids[1], uids[2], uids[0]]
    assert stack.active.uid == uids[0]


def test_a_uid_survives_reordering_so_undo_can_find_its_layer():
    stack = _stack((1, 1, 1, 255), (2, 2, 2, 255))
    uid = stack[0].uid
    stack.move(0, 1)
    assert stack.index_of(uid) == 1
    assert stack.by_uid(uid) is stack[1]


def test_removing_keeps_the_active_index_inside_the_stack():
    stack = _stack((1, 1, 1, 255), (2, 2, 2, 255), active=2 - 1)
    stack.remove(1)
    assert stack.active_index == 0


def test_removing_a_layer_below_the_active_one_keeps_the_same_layer_active():
    """A clamp alone is not enough: removing below the active layer shifts
    every index above it down, so ``active_index`` kept pointing at whatever
    had been *above* it. The active layer silently became a different layer and
    the next stroke landed on it."""
    stack = _stack((1, 1, 1, 255), (2, 2, 2, 255), (3, 3, 3, 255), active=2)
    was = stack.active.uid

    stack.remove(0)

    assert stack.active.uid == was
    assert stack.active_index == 1


# --- compositing ------------------------------------------------------------


def test_an_opaque_top_layer_hides_everything_below_it():
    stack = _stack((255, 0, 0, 255), (0, 0, 255, 255))
    assert tuple(stack.flatten()[0, 0]) == (0, 0, 255, 255)


def test_an_invisible_layer_and_a_zero_opacity_layer_are_both_skipped():
    stack = _stack((255, 0, 0, 255), (0, 0, 255, 255))
    stack[1].visible = False
    assert tuple(stack.flatten()[0, 0]) == (255, 0, 0, 255)
    stack[1].visible = True
    stack[1].opacity = 0.0
    assert tuple(stack.flatten()[0, 0]) == (255, 0, 0, 255)


def test_the_below_cache_composites_exactly_what_the_naive_stack_does():
    rng = np.random.default_rng(7)
    layers = []
    for i in range(4):
        pixels = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
        blend = cp.BLEND_MODES[i % len(cp.BLEND_MODES)]
        layers.append(Layer(pixels=pixels, opacity=0.3 + 0.2 * i, blend=blend))
    stack = LayerStack(layers, active=2)
    rect = (2, 1, 7, 6)
    naive = stack.composite_region(rect)
    cached = stack.composite_region(rect, below=stack.composite_below())
    assert np.allclose(naive, cached, atol=1e-5)


def test_the_below_cache_still_applies_the_layers_above_the_active_one():
    """Blend modes are not associative, so there is no 'above' cache -- if the
    upper layers were dropped, painting would look right and export wrong."""
    stack = _stack((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), active=1)
    got = stack.composite_region((0, 0, 8, 8), below=stack.composite_below())
    assert np.allclose(got[0, 0], [0, 0, 1, 1], atol=1e-6)


def test_compositing_a_region_touches_only_that_region():
    stack = _stack((255, 0, 0, 255))
    got = stack.composite_region((2, 3, 5, 8))
    assert got.shape == (5, 3, 4)

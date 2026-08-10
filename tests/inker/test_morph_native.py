"""The morphology kernel against the numpy reference it replaced.

Unlike the contour tracer, this one's bar *is* output identity, and the licence
for demanding that is arithmetic: min and max over uint8 are exact, so there is
no summation order to disagree about and no reason two implementations of the
same neighbourhood should ever differ by a byte.

The neighbourhood is the thing worth testing hard. ``_morph`` alternates a
4-neighbour pass and an 8-neighbour pass, starting with the 4, so the element it
builds up is an *octagon* whose exact shape depends on the parity of the radius
-- neither a disc nor a square, and not expressible as one call to anybody's
dilate. Radii 1..16 are swept for both directions, because an off-by-one in
which pass is diagonal shows up only at odd or only at even radii.

Masks that touch the edges and corners are in the sweep on purpose: the kernel
handles the border by clamped row indices and two end-column lines rather than
by padding, which is the part most likely to disagree with ``mode="edge"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import native
from warlock.studio.inker import selection as sel

needs_dll = pytest.mark.skipif(not native.available(), reason="warlockc.dll not built")


def _numpy(mask: np.ndarray, radius: int, combine, monkeypatch) -> np.ndarray:
    """The reference body, with the kernel forced off."""
    monkeypatch.setattr(native, "available", lambda: False)
    return sel._morph(mask, radius, combine)


def _kernel(mask: np.ndarray, radius: int, combine, monkeypatch) -> np.ndarray:
    monkeypatch.setattr(native, "available", lambda: True)
    return sel._morph(mask, radius, combine)


def _masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    height, width = shape
    full = np.full(shape, 255, np.uint8)
    empty = np.zeros(shape, np.uint8)
    corners = np.zeros(shape, np.uint8)
    corners[0, 0] = corners[0, -1] = corners[-1, 0] = corners[-1, -1] = 255
    edges = np.zeros(shape, np.uint8)
    edges[0, :] = edges[-1, :] = 255
    edges[:, 0] = edges[:, -1] = 255
    blob = np.zeros(shape, np.uint8)
    blob[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
    return {
        # Soft values, not 0/255: the mask is antialiased and a max filter over
        # it has to move the edge while keeping it soft.
        "soft": rng.integers(0, 256, shape, dtype=np.uint8),
        "full": full,
        "empty": empty,
        "corners": corners,
        "edges": edges,
        "blob": blob,
    }


@needs_dll
@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (2, 2), (3, 3), (17, 9), (33, 32)])
@pytest.mark.parametrize("radius", list(range(1, 17)))
def test_the_kernel_is_byte_identical_to_the_numpy_body(shape, radius, monkeypatch):
    for name, mask in _masks(shape).items():
        for combine in (np.maximum, np.minimum):
            fast = _kernel(mask, radius, combine, monkeypatch)
            slow = _numpy(mask, radius, combine, monkeypatch)
            assert np.array_equal(fast, slow), f"{name} r={radius} {combine.__name__}"
            assert fast.dtype == slow.dtype


@needs_dll
def test_bordered_agrees_too_which_runs_both_directions_over_one_mask(monkeypatch):
    """``bordered`` is a grow and a shrink differenced, so it fails if either
    direction is wrong *or* if the two disagree about the same radius."""
    rng = np.random.default_rng(12)
    mask = rng.integers(0, 256, (40, 24), dtype=np.uint8)
    for width in (1, 2, 3, 7):
        monkeypatch.setattr(native, "available", lambda: True)
        fast = sel.SelectionMask(mask).bordered(width).mask
        monkeypatch.setattr(native, "available", lambda: False)
        slow = sel.SelectionMask(mask).bordered(width).mask
        assert np.array_equal(fast, slow), f"width {width}"


def test_a_radius_of_zero_never_reaches_the_kernel(monkeypatch):
    """The reference answers a plain copy, and the kernel's contract starts at
    1 -- so this is the one input the seam must keep to itself."""
    calls: list[int] = []
    monkeypatch.setattr(native, "available", lambda: True)
    monkeypatch.setattr(native, "morph_u8", lambda *a: calls.append(1))
    mask = np.full((4, 4), 200, np.uint8)
    assert np.array_equal(sel._morph(mask, 0, np.maximum), mask)
    assert calls == []


def test_an_unknown_combine_falls_back_rather_than_being_rendered_as_a_max(monkeypatch):
    """The kernel knows max and min. A third operation must reach the numpy
    body, not be silently answered as one of the two it does know."""
    monkeypatch.setattr(native, "available", lambda: True)
    mask = np.zeros((5, 5), np.uint8)
    mask[2, 2] = 255
    assert sel._morph_native(mask, 1, np.add) is None
    # and the public path still answers, through numpy
    assert sel._morph(mask, 1, np.add).shape == mask.shape


@needs_dll
def test_the_fallback_path_is_genuinely_taken_when_available_is_false(monkeypatch):
    """Without this the monkeypatch could stop toggling anything and both
    halves of every parity test above would be measuring the same code."""
    seen: list[str] = []
    real_spread = sel._spread

    def spy(*args, **kwargs):
        seen.append("numpy")
        return real_spread(*args, **kwargs)

    monkeypatch.setattr(sel, "_spread", spy)
    mask = np.zeros((6, 6), np.uint8)
    mask[3, 3] = 255

    monkeypatch.setattr(native, "available", lambda: True)
    sel._morph(mask, 2, np.maximum)
    assert seen == [], "the kernel path called the numpy spread"

    monkeypatch.setattr(native, "available", lambda: False)
    sel._morph(mask, 2, np.maximum)
    assert seen == ["numpy", "numpy"], "the numpy path did not run one pass per radius"

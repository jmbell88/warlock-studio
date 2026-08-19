"""Wall-clock budgets for ABI 9's two kernels and the flood's complexity.

Marked ``perf`` and therefore deselected by the default parallel run, for
``tests/clay/test_scale.py``'s reason: a wall-clock reading taken while eight
workers saturate the cores is a reading about the scheduler. Run them with
``uv run pytest -m perf -n 0``.

The budgets are ratios, not absolute times, everywhere they can be -- an
absolute number would fail on a slower machine and say nothing about whether the
kernel is still doing the work it was written to do.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from warlock import native

pytestmark = pytest.mark.perf


def _timed(fn, repeats: int = 3) -> float:
    """The best of *repeats*, which is the honest reading for a short kernel:
    the mean measures whatever else the machine did during the slow run."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _without_native(fn):
    import os

    os.environ["WARLOCK_NATIVE"] = "0"
    native.reset()
    try:
        return fn()
    finally:
        os.environ.pop("WARLOCK_NATIVE", None)
        native.reset()


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_the_palette_kernel_is_at_least_twice_the_numpy_path() -> None:
    """High-entropy 512 square: tens of thousands of distinct colours against a
    few dozen swatches, which is the case the N x P x 3 temporary dominates."""
    from warlock.studio.inker import indexed as ix

    rng = np.random.default_rng(1)
    pixels = np.zeros((512, 512, 4), dtype=np.uint8)
    pixels[..., :3] = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    pixels[..., 3] = 255
    palette = [tuple(int(v) for v in c) + (255,) for c in
               rng.integers(0, 256, (32, 3), dtype=np.uint8)]

    with_kernel = _timed(lambda: ix.snap(pixels, palette))
    without = _without_native(lambda: _timed(lambda: ix.snap(pixels, palette)))
    native.reset()

    assert without / with_kernel >= 2.0, (
        f"palette snap: {without:.3f}s numpy vs {with_kernel:.3f}s native "
        f"({without / with_kernel:.1f}x)"
    )


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_the_palette_curve_stays_flat_in_palette_size() -> None:
    """A flat curve in palette size is the dispatch-bound signature, and the
    native-batch-2 lesson was that spotting it is how you avoid writing a kernel
    for a problem that is not arithmetic. It must *stop* being flat once the
    work really is the arithmetic -- so a large palette costs meaningfully more
    than a small one, and a low-colour image does not regress."""
    from warlock.studio.inker import indexed as ix

    rng = np.random.default_rng(2)
    pixels = np.zeros((512, 512, 4), dtype=np.uint8)
    pixels[..., :3] = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    pixels[..., 3] = 255

    def palette(size: int):
        return [tuple(int(v) for v in c) + (255,) for c in
                rng.integers(0, 256, (size, 3), dtype=np.uint8)]

    small = _timed(lambda: ix.snap(pixels, palette(4)))
    large = _timed(lambda: ix.snap(pixels, palette(256)))
    assert large > small, f"256 entries ({large:.3f}s) must cost more than 4 ({small:.3f}s)"


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_a_low_colour_image_does_not_regress() -> None:
    """The distinct-colour reduction is what makes pixel art cheap, and the
    kernel must not have moved that: a 512 square of eight colours stays fast."""
    from warlock.studio.inker import indexed as ix

    rng = np.random.default_rng(3)
    swatches = rng.integers(0, 256, (8, 3), dtype=np.uint8)
    pixels = np.zeros((512, 512, 4), dtype=np.uint8)
    pixels[..., :3] = swatches[rng.integers(0, 8, (512, 512))]
    pixels[..., 3] = 255
    palette = [tuple(int(v) for v in c) + (255,) for c in swatches]

    elapsed = _timed(lambda: ix.snap(pixels, palette))
    assert elapsed < 0.25, f"a low-colour snap took {elapsed:.3f}s"


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_the_flood_is_linear_on_a_serpentine_corridor() -> None:
    """The pathological case for frontier dilation: the path distance to the
    furthest cell approaches the cell count, so the pass count does too and the
    work goes quadratic. The queue is linear in both cases, so the ratio between
    a corridor and an open room of the same area stays bounded."""
    from warlock.studio.plotter import tools

    size = 512
    corridor = np.zeros((size, size), dtype=bool)
    for row in range(0, size, 2):
        corridor[row, :] = True
        if (row // 2) % 2 == 0:
            corridor[min(row + 1, size - 1), size - 1] = True
        else:
            corridor[min(row + 1, size - 1), 0] = True
    room = np.ones((size, size), dtype=bool)

    corridor_time = _timed(lambda: tools.flood_mask(corridor, 0, 0), repeats=2)
    room_time = _timed(lambda: tools.flood_mask(room, 0, 0), repeats=2)
    # Both are O(cells) with the kernel; the corridor visits half as many, so a
    # generous multiple of the room's time is a real bound on quadratic work.
    assert corridor_time < max(room_time * 8.0, 1.0), (
        f"corridor {corridor_time:.3f}s vs room {room_time:.3f}s"
    )


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_the_flood_kernel_beats_the_dilation_on_the_corridor() -> None:
    from warlock.studio.plotter import tools

    size = 256
    corridor = np.zeros((size, size), dtype=bool)
    for row in range(0, size, 2):
        corridor[row, :] = True
        if (row // 2) % 2 == 0:
            corridor[min(row + 1, size - 1), size - 1] = True
        else:
            corridor[min(row + 1, size - 1), 0] = True

    with_kernel = _timed(lambda: tools.flood_mask(corridor, 0, 0), repeats=2)
    without = _without_native(
        lambda: _timed(lambda: tools.flood_mask(corridor, 0, 0), repeats=2)
    )
    native.reset()
    assert without / with_kernel >= 2.0, (
        f"corridor flood: {without:.3f}s dilation vs {with_kernel:.3f}s native"
    )


def test_the_picking_tree_beats_the_full_sweep() -> None:
    """A complexity change rather than a constant factor: the full sweep's cost
    is the mesh, and a 200k-triangle import is a quarter of a million
    Moller-Trumbore evaluations per mouse move."""
    from warlock.studio.clay import mesh as bm
    from warlock.studio.clay import primitives
    from warlock.studio.viewer import picking as pk

    mesh = primitives.uv_sphere(segments=200, rings=160)
    tris, _tf = bm.triangulate(mesh)
    positions = mesh.positions.astype("f8")
    bvh = pk.build_bvh(positions, tris)
    assert bvh is not None

    origin = np.array([0.0, 0.0, -3.0])
    direction = np.array([0.0, 0.0, 1.0])

    def sweep(tree):
        for _ in range(20):
            pk.ray_triangles(origin, direction, positions, tris, tree)

    full = _timed(lambda: sweep(None), repeats=2)
    narrowed = _timed(lambda: sweep(bvh), repeats=2)
    assert full / narrowed >= 2.0, (
        f"picking: {full:.3f}s full vs {narrowed:.3f}s with the tree"
    )

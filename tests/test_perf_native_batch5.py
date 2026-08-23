"""Wall-clock budgets for ABI 10's three kernels, and for the Tier 0 fixes.

Marked ``perf`` and therefore deselected by the default parallel run, for
``test_perf_native_batch3.py``'s reason: a wall-clock reading taken while eight
workers saturate the cores is a reading about the scheduler. Run them with
``uv run pytest -m perf -n 0``.

The budgets are ratios against the ``WARLOCK_NATIVE=0`` fallback wherever a
fallback exists, so they survive a slower machine and still say whether the
kernel is doing the work it was written to do. The two Tier 0 entries have no
seam to toggle -- they *replaced* their reference -- so their budget is against
the reference written out in the test, which is the only honest form.

Batch 5 also adds the coverage batch 3 lacked and the plan called for: the
Plotter render path and the Clay rebuild path, both of which are interactive.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from warlock import native

pytestmark = pytest.mark.perf


def _timed(fn, repeats: int = 3) -> float:
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


# --------------------------------------------------------------------------
# The kernels
# --------------------------------------------------------------------------


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_the_bvh_build_is_at_least_three_times_the_numpy_path() -> None:
    """Measured at 8.5x on a 200k-triangle mesh (1045 ms to 123 ms); three is
    the floor that fails only if the kernel has stopped being reached."""
    from warlock.studio.viewer import picking

    rng = np.random.default_rng(0xB00)
    positions = rng.random((60_000, 3)) * 10.0
    tris = rng.integers(0, 60_000, size=(120_000, 3)).astype("i8")

    with_kernel = _timed(lambda: picking.build_bvh(positions, tris))
    without = _without_native(lambda: _timed(lambda: picking.build_bvh(positions, tris)))
    native.reset()

    assert without / with_kernel >= 3.0, (
        f"bvh build: {without:.3f}s numpy vs {with_kernel:.3f}s native "
        f"({without / with_kernel:.1f}x)"
    )


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_the_plotter_render_is_at_least_three_times_the_numpy_path() -> None:
    """The interactive path batch 3's budgets never covered. Measured at 5.7x
    on the 200x200x3 map the plan benched (4348 ms to 759 ms)."""
    from warlock.studio.plotter import render
    from warlock.studio.plotter.tilemap import MapDoc
    from warlock.studio.tilegrid import gid
    from warlock.studio.tilegrid.tileset import Tileset

    rng = np.random.default_rng(0x71E)
    tiles, size, cells = 64, 32, 60
    sheet = rng.integers(0, 256, size=(size, size * tiles, 4), dtype=np.uint8)
    sheet[..., 3] = np.where(rng.random((size, size * tiles)) > 0.3, 255, 0)

    doc = MapDoc(cells, cells, size, size)
    doc.add_tileset(Tileset(name="t", pixels=sheet, tile_w=size, tile_h=size))
    for _ in range(3):
        layer = doc.add_tile_layer()
        data = rng.integers(1, tiles + 1, size=(cells, cells)).astype(gid.DTYPE)
        doc.write_region(layer.uid, 0, 0, data)

    with_kernel = _timed(lambda: render.render_map(doc), repeats=2)
    without = _without_native(lambda: _timed(lambda: render.render_map(doc), repeats=2))
    native.reset()

    assert without / with_kernel >= 3.0, (
        f"plotter render: {without:.3f}s numpy vs {with_kernel:.3f}s native "
        f"({without / with_kernel:.1f}x)"
    )


@pytest.mark.skipif(not native.available(), reason="warlockc is not built")
def test_the_oklab_palette_search_is_at_least_twice_the_numpy_path() -> None:
    """Only twice: unlike the other two, a large share of what is left is
    ``_to_oklab``, which the kernel deliberately does not swallow."""
    from PIL import Image

    from warlock.pipelines import pixel

    rng = np.random.default_rng(0xD1E)
    image = Image.fromarray(
        rng.integers(0, 256, (512, 512, 4), dtype=np.uint8), "RGBA"
    )
    palette = tuple(
        (int(r), int(g), int(b)) for r, g, b in rng.integers(0, 256, (64, 3))
    )

    with_kernel = _timed(lambda: pixel.map_palette(image, palette), repeats=2)
    without = _without_native(
        lambda: _timed(lambda: pixel.map_palette(image, palette), repeats=2)
    )
    native.reset()

    assert without / with_kernel >= 2.0, (
        f"oklab palette: {without:.3f}s numpy vs {with_kernel:.3f}s native "
        f"({without / with_kernel:.1f}x)"
    )


# --------------------------------------------------------------------------
# Tier 0 -- numpy against numpy, no DLL involved
# --------------------------------------------------------------------------


def test_the_clay_scatter_beats_the_unbuffered_ufunc_it_replaced() -> None:
    """``mesh.accumulate`` is on Clay's rebuild-on-every-edit path, and the
    reference is written out here because there is no seam to toggle -- the
    bincount *is* the implementation now."""
    from warlock.studio.clay.mesh import accumulate

    rng = np.random.default_rng(0x5EED)
    n_verts = 200_000
    loops = rng.integers(0, n_verts, size=n_verts * 6).astype("i8")
    values = rng.random((n_verts * 6, 3))

    def reference() -> None:
        out = np.zeros((n_verts, 3), dtype="f8")
        np.add.at(out, loops, values)

    fast = _timed(lambda: accumulate(loops, values, n_verts))
    slow = _timed(reference)
    assert slow / fast >= 1.4, f"accumulate: {slow:.3f}s add.at vs {fast:.3f}s bincount"


def test_the_edge_table_beats_the_axis_wise_unique_it_replaced() -> None:
    """``adjacency._build``'s one sort, which is the dominant cost of entering
    element mode. Measured at 6x; four is the floor."""
    rng = np.random.default_rng(0xC0FFEE)
    n_verts = 200_000
    a = rng.integers(0, n_verts, size=n_verts * 6)
    b = rng.integers(0, n_verts, size=n_verts * 6)
    pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype("i8")

    def reference() -> None:
        np.unique(pairs, axis=0, return_inverse=True)

    def packed() -> None:
        stride = int(pairs.max()) + 1
        keys, _ = np.unique(pairs[:, 0] * stride + pairs[:, 1], return_inverse=True)
        np.stack([keys // stride, keys % stride], axis=1)

    slow = _timed(reference, repeats=2)
    fast = _timed(packed, repeats=2)
    assert slow / fast >= 4.0, f"edge table: {slow:.3f}s lexsort vs {fast:.3f}s packed"


def test_the_palette_histogram_no_longer_scales_with_the_palette() -> None:
    """The shape of the fix, not just its size: the per-entry loop was linear in
    the palette and the packed pass is flat in it. A 16-entry and a 256-entry
    reading over the same canvas have to come out within a factor of two, which
    a reintroduced per-entry scan could not manage."""
    from warlock.studio.inker import indexed as ix

    rng = np.random.default_rng(0xA11CE)
    side = 512

    def canvas(entries: int):
        palette = [
            (int(r), int(g), int(b), 255)
            for r, g, b in rng.integers(0, 256, size=(entries, 3))
        ]
        table = np.asarray([p[:3] for p in palette], dtype=np.uint8)
        picks = rng.integers(0, entries, size=(side, side))
        pixels = np.dstack([table[picks], np.full((side, side), 255, dtype=np.uint8)])
        return pixels, palette

    small = canvas(16)
    large = canvas(256)
    thin = _timed(lambda: ix.histogram(*small))
    wide = _timed(lambda: ix.histogram(*large))
    assert wide / thin <= 2.0, (
        f"histogram: {thin:.3f}s at 16 entries vs {wide:.3f}s at 256 -- "
        "the cost is scaling with the palette again"
    )

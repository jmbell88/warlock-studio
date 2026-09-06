"""Measure the native-kernel candidates, one named case each.

The reusable harness the K-queue never had. Batches 2 and 3 benched from
ad-hoc scripts that were then thrown away, which is why their ranked follow-up
list rotted: by the time it was read again, three of its entries had been
fixed or invalidated by unrelated work and nobody could tell without
re-measuring. This lives in the repo so the next batch re-runs it instead of
re-inventing it, and ``docs/measurements/2026-08-22-native-batch-5.md`` is its
output.

Two triage rules are built in, both earned by earlier batches:

**Minimum of N runs in a fresh process.** Batch 2's stated method. The spread
on multi-megabyte float work is +-30%, which makes the mean a report on the
machine's other tenants rather than on the code; the minimum is the only
statistic that converges on what the code costs.

**Flat-vs-linear.** ``--sweep`` varies each case's work parameter. A cost curve
that is *flat* in it is dispatch-bound, and a kernel that replaces N numpy
calls with one wins enormously; a curve that *rises* is arithmetic-bound, and
the kernel buys only the constant factor. This predicted FS dither's 76x in
batch 3 before a line of C was written.

Usage::

    uv run python scripts/bench_native.py --list
    uv run python scripts/bench_native.py clay_normals
    uv run python scripts/bench_native.py --all --json out.json
    uv run python scripts/bench_native.py bvh_build --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Runs per measurement. Seven, taking the minimum: enough that one descheduled
#: run does not become the answer, few enough that a four-second case is still
#: a half-minute rather than a coffee break.
RUNS = 7


@dataclass
class Case:
    """One candidate, its gate, and how to build work for it."""

    name: str
    site: str
    #: What the plan requires before this becomes a kernel. Written down
    #: *before* the number is known -- a gate chosen after the fact is a
    #: rationalisation, not a decision.
    gate: str
    #: ``(size) -> callable`` -- the outer builds the inputs (not timed), the
    #: inner does the work (timed).
    build: Callable[[int], Callable[[], Any]]
    #: The work parameter, smallest first, for the flat-vs-linear sweep. The
    #: last is the size the gate is stated at.
    sizes: tuple[int, ...]
    variants: dict[str, Callable[[int], Callable[[], Any]]] = field(default_factory=dict)


def _timed(make: Callable[[], Any], runs: int = RUNS) -> float:
    """Milliseconds, minimum of *runs*, inputs built outside the clock."""
    best = float("inf")
    for _ in range(runs):
        work = make()
        start = time.perf_counter()
        work()
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


# --------------------------------------------------------------------------
# I1 -- np.add.at, eleven sites (clay/ops_topo, clay/ops_subdiv, viewer/scene)
# --------------------------------------------------------------------------


def _mesh_arrays(n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0x5EED)
    n_tris = n_verts * 2
    tris = rng.integers(0, n_verts, size=(n_tris, 3)).astype("i8")
    values = rng.random((n_tris * 3, 3))
    return tris.reshape(-1), values


def _case_add_at(n_verts: int) -> Callable[[], Any]:
    loops, values = _mesh_arrays(n_verts)

    def work() -> Any:
        out = np.zeros((n_verts, 3), dtype="f8")
        np.add.at(out, loops, values)
        return out

    return work


def _case_bincount(n_verts: int) -> Callable[[], Any]:
    loops, values = _mesh_arrays(n_verts)

    def work() -> Any:
        return np.stack(
            [np.bincount(loops, weights=values[:, k], minlength=n_verts) for k in range(3)],
            axis=1,
        )

    return work


def _case_add_at_hits(n_verts: int) -> Callable[[], Any]:
    loops, _ = _mesh_arrays(n_verts)

    def work() -> Any:
        hits = np.zeros(n_verts, dtype="f8")
        np.add.at(hits, loops, 1.0)
        return hits

    return work


def _case_bincount_hits(n_verts: int) -> Callable[[], Any]:
    loops, _ = _mesh_arrays(n_verts)

    def work() -> Any:
        return np.bincount(loops, minlength=n_verts).astype("f8")

    return work


# --------------------------------------------------------------------------
# I3 -- np.unique(pairs, axis=0) vs a packed scalar key
# --------------------------------------------------------------------------


def _edge_pairs(n_verts: int) -> np.ndarray:
    rng = np.random.default_rng(0xC0FFEE)
    n = n_verts * 6
    a = rng.integers(0, n_verts, size=n)
    b = rng.integers(0, n_verts, size=n)
    return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype("i8")


def _case_unique_axis(n_verts: int) -> Callable[[], Any]:
    pairs = _edge_pairs(n_verts)

    def work() -> Any:
        return np.unique(pairs, axis=0, return_inverse=True)

    return work


def _case_unique_packed(n_verts: int) -> Callable[[], Any]:
    pairs = _edge_pairs(n_verts)

    def work() -> Any:
        packed = pairs[:, 0].astype("u8") * np.uint64(n_verts) + pairs[:, 1].astype("u8")
        keys, inverse = np.unique(packed, return_inverse=True)
        verts = np.stack([keys // np.uint64(n_verts), keys % np.uint64(n_verts)], axis=1)
        return verts.astype("i8"), inverse

    return work


# --------------------------------------------------------------------------
# I6 -- indexed.histogram: one full-canvas pass per palette entry
# --------------------------------------------------------------------------


def _canvas(entries: int, side: int = 2048) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    rng = np.random.default_rng(0xA11CE)
    palette = [
        (int(r), int(g), int(b), 255)
        for r, g, b in rng.integers(0, 256, size=(entries, 3))
    ]
    table = np.asarray([p[:3] for p in palette], dtype=np.uint8)
    picks = rng.integers(0, entries, size=(side, side))
    pixels = np.dstack([table[picks], np.full((side, side), 255, dtype=np.uint8)])
    return pixels, palette


def _case_histogram_loop(entries: int) -> Callable[[], Any]:
    pixels, palette = _canvas(entries)

    def work() -> Any:
        counts = [0] * len(palette)
        visible = pixels[..., 3] > 0
        rgb = pixels[..., :3][visible]
        for index, colour in enumerate(palette):
            want = np.asarray(tuple(colour)[:3], dtype=np.uint8)
            counts[index] = int((rgb == want).all(axis=1).sum())
        return counts

    return work


def _case_histogram_packed(entries: int) -> Callable[[], Any]:
    pixels, palette = _canvas(entries)

    def work() -> Any:
        visible = pixels[..., 3] > 0
        rgb = pixels[..., :3][visible]
        packed = (
            rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
        )
        keys, hits = np.unique(packed, return_counts=True)
        want = np.asarray(
            [(c[0] << 16) | (c[1] << 8) | c[2] for c in palette], dtype=np.uint32
        )
        at = np.clip(np.searchsorted(keys, want), 0, max(len(keys) - 1, 0))
        found = keys[at] == want
        return np.where(found, hits[at], 0).tolist()

    return work


# --------------------------------------------------------------------------
# I2 -- viewer/picking.build_bvh
# --------------------------------------------------------------------------


def _case_bvh(n_tris: int) -> Callable[[], Any]:
    from warlock.studio.viewer import picking

    rng = np.random.default_rng(0xB00)
    n_verts = max(4, n_tris // 2)
    positions = rng.random((n_verts, 3)) * 10.0
    tris = rng.integers(0, n_verts, size=(n_tris, 3)).astype("i8")

    def work() -> Any:
        return picking.build_bvh(positions, tris)

    return work


# --------------------------------------------------------------------------
# I7 -- inker/filters._grow
# --------------------------------------------------------------------------


def _grow_mask() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random((1024, 1024)) > 0.98


def _case_grow(radius: int) -> Callable[[], Any]:
    from warlock.studio.inker import filters

    mask = _grow_mask()

    def work() -> Any:
        return filters._grow(mask, radius, 8, wrap=False)

    return work


def _grow_fused(mask: np.ndarray, steps: int) -> np.ndarray:
    """``filters._grow`` with the eight ``_shift`` calls fused, 8-connected.

    Same result, one temporary per step instead of nine: ``_shift`` allocates a
    whole-canvas ``zeros_like`` and slice-assigns into it, then ``|`` allocates
    the union -- so the shipped loop touches the canvas 8 x 3 times per step
    where this touches it 9 times. Batch 5 §8 asked for exactly this
    comparison before any C was considered.
    """
    out = mask
    for _ in range(steps):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        grown[1:, :-1] |= out[:-1, 1:]
        grown[:-1, 1:] |= out[1:, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        out = grown
    return out


def _grow_separable(mask: np.ndarray, steps: int) -> np.ndarray:
    """The 8-connected case as a 1x3 dilation followed by a 3x1 one.

    A 3x3 all-ones structuring element is separable, so the square that makes
    the 8-connected outline boxy can be had in 4 whole-canvas ORs per step
    rather than 8. **Only valid for 8-connected**: the 4-connected diamond is
    not separable, and that is the whole point of the flag.
    """
    out = mask
    for _ in range(steps):
        row = out.copy()
        row[:, 1:] |= out[:, :-1]
        row[:, :-1] |= out[:, 1:]
        col = row.copy()
        col[1:, :] |= row[:-1, :]
        col[:-1, :] |= row[1:, :]
        out = col
    return out


def _case_grow_fused(radius: int) -> Callable[[], Any]:
    mask = _grow_mask()
    return lambda: _grow_fused(mask, radius)


def _case_grow_separable(radius: int) -> Callable[[], Any]:
    mask = _grow_mask()
    return lambda: _grow_separable(mask, radius)


# --------------------------------------------------------------------------
# B4 -- inker/tiles.materialize's per-cell Python loop
# --------------------------------------------------------------------------


def _tiles_fixture(cells: int) -> tuple[np.ndarray, Any, tuple[int, int]]:
    """A ``cells`` x ``cells`` grid over a 64-tile, 32px sheet.

    Deliberately the same shape as ``_case_render_map`` so the two per-cell
    loops -- Plotter's, which got a kernel in batch 5, and Inker's, which did
    not -- are read against each other. A third of the refs carry a transform
    flag, because ``oriented()`` is called per cell with no memo and its cost
    only appears when the flags are actually set.
    """
    from warlock.studio.tilegrid import gid
    from warlock.studio.tilegrid.tileset import Tileset

    rng = np.random.default_rng(0x71E)
    tiles = 64
    sheet = rng.integers(0, 256, size=(32, 32 * tiles, 4), dtype=np.uint8)
    ts = Tileset(name="t", pixels=sheet, tile_w=32, tile_h=32)
    refs = rng.integers(0, tiles, size=(cells, cells)).astype(gid.DTYPE)
    flags = rng.choice(
        np.asarray(
            [0, gid.FLIP_H, gid.FLIP_V, gid.FLIP_D, gid.FLIP_H | gid.FLIP_D],
            dtype=gid.DTYPE,
        ),
        size=(cells, cells),
        p=[0.66, 0.11, 0.11, 0.06, 0.06],
    )
    refs = (refs | flags).astype(gid.DTYPE)
    return refs, ts, (cells * 32, cells * 32)


def _case_materialize(cells: int) -> Callable[[], Any]:
    from warlock.studio.inker import tiles

    refs, ts, size = _tiles_fixture(cells)
    return lambda: tiles.materialize(refs, ts, size)


def _case_materialize_memo(cells: int) -> Callable[[], Any]:
    """The same loop with ``oriented(tile_pixels(...))`` memoised per raw gid.

    There are at most ``tile_count x 8`` distinct answers, so a whole canvas
    asks for the same handful over and over. This isolates what the un-memoised
    ``oriented()`` per cell costs from what the per-cell *copy into the canvas*
    costs, which is the part no memo can remove.
    """
    from warlock.studio.inker import tiles
    from warlock.studio.tilegrid import gid

    refs, ts, size = _tiles_fixture(cells)

    def work() -> Any:
        width, height = size
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        grid_h, grid_w = refs.shape
        tile_w, tile_h = ts.tile_w, ts.tile_h
        memo: dict[int, np.ndarray] = {}
        for row in range(grid_h):
            y0 = row * tile_h
            if y0 >= height:
                break
            for col in range(grid_w):
                x0 = col * tile_w
                if x0 >= width:
                    break
                raw = int(refs[row, col])
                tile = memo.get(raw)
                if tile is None:
                    local = raw & gid.GID_MASK
                    if local >= ts.tile_count:
                        local = 0
                    tile = tiles.oriented(ts.tile_pixels(local), raw)
                    memo[raw] = tile
                h = min(tile.shape[0], height - y0)
                w = min(tile.shape[1], width - x0)
                canvas[y0 : y0 + h, x0 : x0 + w] = tile[:h, :w]
        return canvas

    return work


# --------------------------------------------------------------------------
# B6 -- pipelines/pixel._to_oklab, the residual around the shipped kernel
# --------------------------------------------------------------------------


def _oklab_fixture(entries: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0xD1E)
    side = 1024
    rgb = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    palette = rng.integers(0, 256, size=(entries, 3)).astype(np.float64)
    return rgb, palette


def _case_oklab_residual(entries: int) -> Callable[[], Any]:
    """Everything ``map_palette`` does in Oklab *except* the nearest search.

    Batch 5 kernelised the search (``warlockc_palette_nearest_f64``, 5.5x), so
    what is left of `map_palette` is these two conversions -- one over a
    1024-square frame, one over the palette. The sweep is over palette entries
    to sit beside ``pixel_map_palette``'s own 8/32/64, which means **the swept
    parameter is not this case's dominant work driver**; ``palette_only``
    below is the part that actually moves with it.
    """
    from warlock.pipelines import pixel

    rgb, palette = _oklab_fixture(entries)

    def work() -> Any:
        return pixel._to_oklab(rgb), pixel._to_oklab(palette)

    return work


def _case_oklab_palette(entries: int) -> Callable[[], Any]:
    from warlock.pipelines import pixel

    _, palette = _oklab_fixture(entries)
    return lambda: pixel._to_oklab(palette)


# --------------------------------------------------------------------------
# B8 -- packwright/maxrects.pack: _score's linear scan and _prune's O(F^2)
# --------------------------------------------------------------------------


def _pack_items(n: int) -> tuple[list[tuple[str, int, int]], int]:
    """``docs/measurements/2026-08-31-packwright-max-sprites.md``'s driver.

    Random 8..64 px rectangles under ``random.Random(7)``, run through
    ``order`` so the search is the real one and not an unsorted worst case, and
    an atlas side at the next power of two above twice the total area -- which
    is roughly ``_candidate_sizes``' first try.
    """
    import random

    from warlock.studio.packwright import layout, maxrects

    rng = random.Random(7)
    items = maxrects.order(
        [(f"s{i}", rng.randint(8, 64), rng.randint(8, 64)) for i in range(n)]
    )
    area = sum(w * h for _k, w, h in items)
    side = layout.next_pot(int((area * 2) ** 0.5) + 1)
    return items, side


def _case_pack(n: int) -> Callable[[], Any]:
    """One ``pack()``, the unit the 2026-08-31 document timed.

    Gate stated before the number: **>1.0 s at 1024 sprites**. 1024 is
    ``layout.MAX_SPRITES``, cut from 4096 on that document's 4.7 s figure, and a
    kernel is on the table only if a single pack at the ceiling is still over a
    second -- which is where a background "packing..." label starts reading as
    a hang. The sweep doubles n, so the cost ratio against 2x work tells the
    O(F^2)-per-placement story on its own.
    """
    from warlock.studio.packwright import maxrects

    items, side = _pack_items(n)
    return lambda: maxrects.pack(items, side, side)


def _pack_new_only(items: list[tuple[str, int, int]], width: int, height: int) -> Any:
    """``maxrects.pack`` with the prune restricted to pairs touching a new piece.

    After every prune no survivor contains another survivor -- that is what the
    prune *means* -- so on the next placement the only pairs that can fire are
    the ones involving a piece ``_split`` just made. The pair body, its order
    and the "of two identical rectangles the earlier is dropped" rule are the
    shipped ones verbatim; only old-vs-old pairs are skipped, and those cannot
    fire. Exact by induction, and the case builder asserts it on a small pack.
    """
    from bisect import bisect_right

    from warlock.studio.packwright.maxrects import Placement, Rect, _fits, _score, _split

    free = [Rect(0, 0, int(width), int(height))]
    placed: list[Placement] = []
    for key, w, h in items:
        w, h = int(w), int(h)
        best_score: Any = None
        best: Rect | None = None
        for candidate in free:
            if not _fits(candidate, w, h):
                continue
            score = _score(candidate, w, h)
            if best_score is None or score < best_score:
                best_score, best = score, candidate
        if best is None:
            return None
        used = Rect(best.x, best.y, w, h)
        placed.append(Placement(key=key, x=used.x, y=used.y, w=w, h=h))
        remaining: list[Rect] = []
        fresh: list[bool] = []
        for candidate in free:
            pieces = _split(candidate, used)
            if pieces is None:
                remaining.append(candidate)
                fresh.append(False)
            else:
                remaining.extend(pieces)
                fresh.extend([True] * len(pieces))
        count = len(remaining)
        bounds = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in remaining]
        dead = [False] * count
        new_idx = [k for k in range(count) if fresh[k]]
        for i in range(count):
            if dead[i]:
                continue
            ix, iy, ir, ib = bounds[i]
            js = range(i + 1, count) if fresh[i] else new_idx[bisect_right(new_idx, i):]
            for j in js:
                if dead[j]:
                    continue
                jx, jy, jr, jb = bounds[j]
                if jx <= ix and jy <= iy and jr >= ir and jb >= ib:
                    dead[i] = True
                    break
                if ix <= jx and iy <= jy and ir >= jr and ib >= jb:
                    dead[j] = True
        free = [rect for rect, gone in zip(remaining, dead, strict=True) if not gone]
    return placed


def _case_pack_new_only(n: int) -> Callable[[], Any]:
    from warlock.studio.packwright import maxrects

    small, small_side = _pack_items(256)
    assert _pack_new_only(small, small_side, small_side) == maxrects.pack(
        small, small_side, small_side
    ), "the restricted prune must reproduce the shipped pack exactly"
    items, side = _pack_items(n)
    return lambda: _pack_new_only(items, side, side)


# --------------------------------------------------------------------------
# B9 -- plotter/terrain.paint_wang_cells: the sequential per-cell re-fit
# --------------------------------------------------------------------------


def _wang_fixture(side: int) -> tuple[Any, Any, Any, list[tuple[int, int]]]:
    """A ``side`` x ``side`` layer of two blob terrains, and one diagonal drag.

    The layer is pre-filled with the two colours' interiors in random 8-cell
    blocks, so every cell has neighbours with an opinion -- an empty layer gives
    ``constraints_from`` nothing and the loop returns early, which is not the
    case anyone waits on. The gesture is a diagonal stroke corner to corner:
    the box around it is the whole layer and almost none of it is *asserted*,
    so every cell in the box is re-fit. A solid fill is the cheap shape here
    (its interior is asserted and skipped); the drag is the expensive one.
    """
    from warlock.studio.tilegrid import blob, gid
    from warlock.studio.tilegrid.tileset import Tileset, TilesetRef
    from warlock.studio.tilegrid.wang import blob_wangset

    wangset = blob_wangset(["grass", "sand"], ["#00ff00", "#ffff00"])
    interior = {
        1: next(i for i, w in wangset.tiles.items() if all(c == 1 for c in w)),
        2: next(i for i, w in wangset.tiles.items() if all(c == 2 for c in w)),
    }
    rng = np.random.default_rng(0x3A6)
    blocks = rng.integers(1, 3, size=(side // 8 + 1, side // 8 + 1))
    colours = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)[:side, :side]
    data = np.where(colours == 1, 1 + interior[1], 1 + interior[2]).astype(gid.DTYPE)
    pixels = np.zeros((16, 16 * (2 * blob.TILE_COUNT), 4), dtype=np.uint8)
    ref = TilesetRef(
        firstgid=1,
        tileset=Tileset(name="w", pixels=pixels, tile_w=16, tile_h=16, wangsets=(wangset,)),
    )
    cells = [(i, i) for i in range(side)]
    return data, ref, wangset, cells


def _case_wang_drag(side: int) -> Callable[[], Any]:
    """Gate stated before the number: **>100 ms for a 128-cell diagonal drag**.

    ``paint_wang_cells`` runs on the frame thread from ``plotter_canvas``, so
    the budget is a gesture the user is mid-way through, not a task. 128 is a
    mid-size map's diagonal; the box it opens is 128^2 = 16k cells, each a
    ``constraints_from`` (eight neighbour reads through a closure) and a memo
    lookup. The sweep doubles the side, so the work quadruples per step.
    """
    from warlock.studio.plotter import terrain

    data, ref, wangset, cells = _wang_fixture(side)
    return lambda: terrain.paint_wang_cells(data, cells, 1, ref, wangset)


def _wang_field_hoisted(data: np.ndarray, ref: Any, wangset: Any) -> Any:
    """``terrain.wang_field`` with ``ref.holds`` hoisted out of the closure.

    ``holds`` is ``firstgid <= id <= last_gid`` and ``last_gid`` walks
    ``max_local_id -> tile_count -> columns/rows -> image_w/h`` on every call;
    at eight neighbour reads per cell that chain is where the profile puts the
    time. The bounds are facts about the ref that do not change during one
    gesture, so they are read once. Same answers, by inspection of ``holds``.
    """
    from warlock.studio.tilegrid import gid as gidlib

    height, width = data.shape
    first, last, tiles, mask = ref.firstgid, ref.last_gid, wangset.tiles, gidlib.GID_MASK

    def field_of(x: int, y: int) -> Any:
        if not (0 <= x < width and 0 <= y < height):
            return None
        value = int(data[y, x]) & mask
        if not value or not (first <= value <= last):
            return None
        return tiles.get(value - first)

    return field_of


def _case_wang_drag_hoisted(side: int) -> Callable[[], Any]:
    from warlock.studio.plotter import terrain

    data, ref, wangset, cells = _wang_fixture(side)
    shipped = terrain.wang_field
    try:
        terrain.wang_field = _wang_field_hoisted
        fast = terrain.paint_wang_cells(data, cells, 1, ref, wangset)
    finally:
        terrain.wang_field = shipped
    slow = terrain.paint_wang_cells(data, cells, 1, ref, wangset)
    assert slow is not None and fast is not None
    assert slow[:2] == fast[:2] and np.array_equal(slow[2], fast[2]), "hoisting changed the re-fit"

    def work() -> Any:
        terrain.wang_field = _wang_field_hoisted
        try:
            return terrain.paint_wang_cells(data, cells, 1, ref, wangset)
        finally:
            terrain.wang_field = shipped

    return work


# --------------------------------------------------------------------------
# B7 -- clay/ops_bevel.bevel_edges
# --------------------------------------------------------------------------


def _quad_grid(side: int) -> Any:
    """``side`` x ``side`` quads -- ``tests/clay/test_scale.py``'s own fixture."""
    from warlock.studio.clay import mesh as bm
    from warlock.studio.clay import topo

    xs = np.arange(side + 1, dtype="f4")
    positions = np.stack(
        [
            np.repeat(xs, side + 1),
            np.zeros((side + 1) ** 2, dtype="f4"),
            np.tile(xs, side + 1),
        ],
        axis=1,
    )

    def v(i: int, j: int) -> int:
        return i * (side + 1) + j

    faces = [
        [v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)]
        for i in range(side)
        for j in range(side)
    ]
    return bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * len(faces)),
        material=np.zeros(len(faces), dtype="i4"),
        smooth=np.zeros(len(faces), dtype=bool),
    )


def _bevel_edges_for(side: int, count: int) -> np.ndarray:
    """*count* interior, pairwise vertex-disjoint edges of a ``side`` grid.

    Disjoint on purpose: ``bevel_edges`` refuses a boundary vertex carrying two
    beveled edges, and a disjoint set keeps every corner in the one-beveled-edge
    row of its own table, so the sweep varies count and nothing else.
    """
    pairs = [
        (i * (side + 1) + j, (i + 1) * (side + 1) + j)
        for i in range(1, side - 1, 2)
        for j in range(1, side)
    ]
    if count > len(pairs):
        raise ValueError(f"a {side}x{side} grid has only {len(pairs)} disjoint edges")
    return np.asarray(pairs[:count], dtype="i4")


def _case_bevel(side: int) -> Callable[[int], Callable[[], Any]]:
    def build(count: int) -> Callable[[], Any]:
        from warlock.studio.clay import adjacency as adj
        from warlock.studio.clay import elements as el
        from warlock.studio.clay import ops_bevel as ob

        mesh = _quad_grid(side)
        sel = el.ElementSel(edges=_bevel_edges_for(side, count))
        # Built outside the clock, and cached against the mesh: by the time a
        # user clicks Bevel the adjacency is already there, so timing its build
        # here would be measuring the wrong click.
        adj.adjacency(mesh)
        return lambda: ob.bevel_edges(mesh, sel, width=0.05)

    return build


# --------------------------------------------------------------------------
# B2 -- pipelines/pixel.map_palette
# --------------------------------------------------------------------------


def _case_map_palette(entries: int) -> Callable[[], Any]:
    from PIL import Image

    from warlock.pipelines import pixel

    rng = np.random.default_rng(0xD1E)
    side = 1024
    arr = rng.integers(0, 256, size=(side, side, 4), dtype=np.uint8)
    image = Image.fromarray(arr, "RGBA")
    palette = tuple(
        (int(r), int(g), int(b))
        for r, g, b in rng.integers(0, 256, size=(entries, 3))
    )

    def work() -> Any:
        return pixel.map_palette(image, palette)

    return work


# --------------------------------------------------------------------------
# B1 -- plotter/render.render_layer's per-cell Python loop
# --------------------------------------------------------------------------


def _case_render_map(cells: int) -> Callable[[], Any]:
    """A square map of *cells* cells per side, three layers, 32px tiles.

    Binary alpha and a 64-tile sheet: the corpus batch 2 said it did not have.
    Its 33 us/cell figure was measured on a map like this one, and the point of
    re-measuring is that Part J's work has moved several neighbouring numbers.
    """
    from warlock.studio.plotter import render
    from warlock.studio.plotter.tilemap import MapDoc
    from warlock.studio.tilegrid import gid
    from warlock.studio.tilegrid.tileset import Tileset

    rng = np.random.default_rng(0x71E)
    tiles = 64
    pixels = rng.integers(0, 256, size=(tiles, 32, 32 * 1, 4), dtype=np.uint8)
    sheet = rng.integers(0, 256, size=(32, 32 * tiles, 4), dtype=np.uint8)
    # Binary alpha: a tile is either drawn or it is not. Partial alpha is the
    # case batch 2 left unmeasured and it is still unmeasured here.
    sheet[..., 3] = np.where(rng.random((32, 32 * tiles)) > 0.3, 255, 0)
    del pixels

    doc = MapDoc(cells, cells, 32, 32)
    doc.add_tileset(Tileset(name="t", pixels=sheet, tile_w=32, tile_h=32))
    for _ in range(3):
        layer = doc.add_tile_layer()
        data = rng.integers(1, tiles + 1, size=(cells, cells)).astype(gid.DTYPE)
        doc.write_region(layer.uid, 0, 0, data)

    def work() -> Any:
        return render.render_map(doc)

    return work


# --------------------------------------------------------------------------
# B5 -- meshreport._welded's unique
# --------------------------------------------------------------------------


def _case_welded(n_verts: int) -> Callable[[], Any]:
    rng = np.random.default_rng(0x1234)
    vertices = rng.random((n_verts, 3)).astype("f8") * 2.0 - 1.0

    def work() -> Any:
        tolerance = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) * 1e-6
        return np.unique(
            np.round(vertices / tolerance), axis=0, return_index=True, return_inverse=True
        )

    return work


CASES: dict[str, Case] = {
    "clay_normals": Case(
        name="clay_normals",
        site="clay/ops_topo.py:338, ops_subdiv.py:237-262, viewer/scene.py:243",
        gate="any win at 20k verts (Tier 0: numpy only, no ABI bump)",
        build=_case_add_at,
        sizes=(2_000, 20_000, 200_000),
        variants={"add_at": _case_add_at, "bincount": _case_bincount},
    ),
    "clay_hits": Case(
        name="clay_hits",
        site="clay/ops_topo.py:339,487,544 -- the 1-D counters",
        gate="any win at 20k verts",
        build=_case_add_at_hits,
        sizes=(2_000, 20_000, 200_000),
        variants={"add_at": _case_add_at_hits, "bincount": _case_bincount_hits},
    ),
    "clay_edges": Case(
        name="clay_edges",
        site="clay/adjacency.py:154 and :395",
        gate=">30% of _build (Tier 0)",
        build=_case_unique_axis,
        sizes=(2_000, 20_000, 200_000),
        variants={"unique_axis": _case_unique_axis, "packed": _case_unique_packed},
    ),
    "inker_histogram": Case(
        name="inker_histogram",
        site="inker/indexed.py:343",
        gate=">50 ms at 2048 square, 256 entries",
        build=_case_histogram_loop,
        sizes=(16, 64, 256),
        variants={"per_entry": _case_histogram_loop, "packed": _case_histogram_packed},
    ),
    "bvh_build": Case(
        name="bvh_build",
        site="viewer/picking.py:180-264",
        gate=">50 ms at 200k tris, per mesh edit",
        build=_case_bvh,
        sizes=(2_000, 20_000, 200_000),
    ),
    "inker_grow": Case(
        name="inker_grow",
        site="inker/filters.py:402-418",
        gate=">100 ms at r=32",
        build=_case_grow,
        sizes=(4, 16, 32),
        variants={
            "shipped": _case_grow,
            "fused": _case_grow_fused,
            "separable": _case_grow_separable,
        },
    ),
    "pixel_map_palette": Case(
        name="pixel_map_palette",
        site="pipelines/pixel.py:382-391",
        gate=">500 ms at 1024 square x 64 entries",
        build=_case_map_palette,
        sizes=(8, 32, 64),
    ),
    "plotter_render": Case(
        name="plotter_render",
        site="plotter/render.py:261 -- the per-cell loop",
        gate="any win (the old K4; 3992 ms at 200x200x3 when last measured)",
        build=_case_render_map,
        sizes=(50, 100, 200),
    ),
    "tiles_materialize": Case(
        name="tiles_materialize",
        site="inker/tiles.py:253-290 -- the per-cell loop",
        gate=">200 ms per conversion (batch 5, item B4)",
        build=_case_materialize,
        sizes=(50, 100, 200),
        variants={"as_shipped": _case_materialize, "memo_oriented": _case_materialize_memo},
    ),
    "oklab_fold": Case(
        name="oklab_fold",
        site="pipelines/pixel.py:332-356 -- _to_oklab, around the shipped kernel",
        gate=">200 ms of residual at 1024 square x 64 entries "
        "(the size batch 5 item B2 kernelised the search at)",
        build=_case_oklab_residual,
        sizes=(8, 32, 64),
        variants={"residual": _case_oklab_residual, "palette_only": _case_oklab_palette},
    ),
    "packwright_pack": Case(
        name="packwright_pack",
        site="packwright/maxrects.py:175 -- pack, and _prune at :130",
        gate=">1.0 s at 1024 sprites (layout.MAX_SPRITES)",
        build=_case_pack,
        sizes=(128, 256, 512, 1024),
        variants={"shipped": _case_pack, "prune_new_only": _case_pack_new_only},
    ),
    "wang_drag": Case(
        name="wang_drag",
        site="plotter/terrain.py:424 -- paint_wang_cells' per-cell loop",
        gate=">100 ms for a 128-cell diagonal drag",
        build=_case_wang_drag,
        sizes=(32, 64, 128),
        variants={"shipped": _case_wang_drag, "holds_hoisted": _case_wang_drag_hoisted},
    ),
    "clay_bevel": Case(
        name="clay_bevel",
        site="clay/ops_bevel.py:258 -- bevel_edges",
        gate=">1.0 s on the 200x200-quad mesh, a third of "
        "tests/clay/test_scale.py's BUDGET = 3.0 s",
        build=_case_bevel(200),
        sizes=(10, 100, 1_000),
        variants={"grid_200": _case_bevel(200), "grid_50": _case_bevel(50)},
    ),
    "mesh_welded": Case(
        name="mesh_welded",
        site="meshreport.py:213",
        gate=">200 ms",
        build=_case_welded,
        sizes=(20_000, 200_000, 500_000),
    ),
}


def run_case(case: Case, sweep: bool) -> dict[str, Any]:
    sizes = case.sizes if sweep else case.sizes[-1:]
    rows: list[dict[str, Any]] = []
    for size in sizes:
        row: dict[str, Any] = {"size": size}
        if case.variants:
            for label, build in case.variants.items():
                row[label] = _timed(lambda b=build, s=size: b(s))
        else:
            row["ms"] = _timed(lambda s=size: case.build(s))
        rows.append(row)
    return {"name": case.name, "site": case.site, "gate": case.gate, "rows": rows}


def _report(result: dict[str, Any]) -> None:
    print(f"\n{result['name']}  --  {result['site']}")
    print(f"  gate: {result['gate']}")
    for row in result["rows"]:
        parts = [f"{k}={v:.2f}ms" for k, v in row.items() if k != "size"]
        print(f"  n={row['size']:>8}  " + "  ".join(parts))
    # Flat-vs-linear: how the cost grew against how the work grew. A ratio far
    # below the work ratio means most of the time is dispatch, not arithmetic.
    if len(result["rows"]) > 1:
        span = result["rows"][-1]["size"] / result["rows"][0]["size"]
        for key in [k for k in result["rows"][0] if k != "size"]:
            first, last = result["rows"][0][key], result["rows"][-1][key]
            shape = "dispatch-bound" if last / first < span / 4 else "arithmetic-bound"
            print(f"  {key}: {last / first:.1f}x cost over {span:.0f}x work -- {shape}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case names; default is every case")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="vary the work parameter")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--child", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        for case in CASES.values():
            print(f"{case.name:<20} {case.site}")
            print(f"{'':<20} gate: {case.gate}")
        return 0

    if args.child:
        # One case per process: a case that allocates 500 MB leaves the
        # allocator in a state the next case would otherwise be measured
        # through, which is the fresh-process half of batch 2's method.
        print("@@JSON@@" + json.dumps(run_case(CASES[args.child], args.sweep)))
        return 0

    names = list(CASES) if (args.all or not args.cases) else args.cases
    results = []
    for name in names:
        if name not in CASES:
            print(f"unknown case: {name}", file=sys.stderr)
            return 2
        cmd = [sys.executable, __file__, "--child", name]
        if args.sweep:
            cmd.append("--sweep")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith("@@JSON@@")), None
        )
        if line is None:
            print(f"{name}: FAILED\n{proc.stdout}\n{proc.stderr}", file=sys.stderr)
            continue
        result = json.loads(line[len("@@JSON@@") :])
        results.append(result)
        _report(result)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

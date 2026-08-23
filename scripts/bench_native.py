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


def _case_grow(radius: int) -> Callable[[], Any]:
    from warlock.studio.inker import filters

    rng = np.random.default_rng(7)
    mask = rng.random((1024, 1024)) > 0.98

    def work() -> Any:
        return filters._grow(mask, radius, 8, wrap=False)

    return work


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
        site="inker/filters.py:402-419",
        gate=">100 ms at r=32",
        build=_case_grow,
        sizes=(4, 16, 32),
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

"""Silhouette hole measurement for generated meshes.

trellis-server.exe's narrow-band DC remesh emits a thin crust made of many
disconnected-but-individually-closed plates, so the usual integrity checks all
pass on a mesh that is visibly perforated: every component is watertight, every
component is wound outward, and no edge is left dangling. What is actually wrong
is that the plates do not meet, and the gaps between them are see-through.

The only way to catch that is to look at the mesh the way the user does. This
rasterises the silhouette from several directions and flood-fills the uncovered
pixels from the image border; whatever the fill cannot reach is enclosed by
geometry on all sides, i.e. a hole you can see through.

Backface culling is deliberately not simulated. On a closed, consistently wound
mesh it cannot open a hole (measured: 4-35 px out of ~45,000), so including it
would only add a second explanation for a number that has one.

Used two ways: the worker measures every finished mesh at REQUEST_PATH_RESOLUTION
and stores the result on the job, and the band sweep (sweep.py) measures at full
resolution offline. Cost is superlinear in resolution -- the flood fill and blob
count both iterate ~O(resolution) times over a resolution^2 array -- which is
why the request path uses the lower one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# Front, two three-quarter views, and one from below. The underside is
# consistently the worst case: it is the least constrained by the reference
# image, so it is where the shape flow hallucinates most.
DEFAULT_VIEWS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.4, 1.0),
    (-1.0, 0.6, 0.8),
    (0.2, -1.0, 0.3),
)

# Above this bbox size a triangle gets its own rasterisation pass rather than
# joining a vectorised batch, to keep the (n, k, k) working set bounded.
_BATCH_MAX_SPAN = 16

# Cap on the (n, k, k) working set of one vectorised pass, in cells. k is
# bounded by _BATCH_MAX_SPAN but n is not: a 500k-triangle trellis mesh can put
# hundreds of thousands of triangles in a single k-group, and each pass holds
# about a dozen float64 arrays of that shape at once. Unbounded, that is a
# multi-gigabyte commit spike inside a mesh audit -- which is precisely the
# kind of transient the 2026-08-03 exhaustion crash was made of. 4M cells is
# ~32 MB per temporary, so a few hundred MB for the pass, and the chunk loop
# costs nothing when n is small (the common case is one chunk).
_BATCH_MAX_CELLS = 4_000_000

# What the worker measures every finished mesh at. Half the diagnostic default:
# the cost of both fixpoint loops scales with resolution^3, so 512 is ~8x
# cheaper than 1024 -- seconds rather than tens of seconds on a job that
# already took minutes. Hole fractions stay comparable across resolutions
# (test_meshaudit asserts a higher resolution does not manufacture holes);
# only sub-pixel gaps are missed.
REQUEST_PATH_RESOLUTION = 512


def hole_fraction(
    glb_path: Path,
    views: tuple[tuple[float, float, float], ...] = DEFAULT_VIEWS,
    resolution: int = 1024,
) -> dict[str, Any]:
    """Measure the fraction of each silhouette that is see-through.

    Returns per-view results plus ``worst`` / ``mean`` hole fractions. A solid
    object measures ~0.0; the perforated meshes this was written for measure
    0.07-0.15 depending on the direction.
    """
    positions, faces = load_mesh(glb_path)
    per_view = []
    for direction in views:
        covered = _coverage(positions, faces, direction, resolution)
        holes = _enclosed_gaps(covered)
        hole_px = int(holes.sum())
        silhouette_px = int(covered.sum()) + hole_px
        per_view.append(
            {
                "direction": tuple(float(c) for c in direction),
                "silhouette_px": silhouette_px,
                "hole_px": hole_px,
                "hole_fraction": hole_px / silhouette_px if silhouette_px else 0.0,
                "blobs": _count_blobs(holes),
            }
        )
    fractions = [v["hole_fraction"] for v in per_view]
    return {
        "resolution": resolution,
        "faces": int(len(faces)),
        "views": per_view,
        "worst": max(fractions) if fractions else 0.0,
        "mean": float(np.mean(fractions)) if fractions else 0.0,
    }


def load_mesh(glb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (vertices, faces) with every node transform already applied."""
    loaded = trimesh.load(glb_path, process=False)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _screen_basis(direction: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    view = np.asarray(direction, dtype=np.float64)
    view /= np.linalg.norm(view)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(view @ up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(up, view)
    right /= np.linalg.norm(right)
    return right, np.cross(view, right)


def _coverage(
    positions: np.ndarray,
    faces: np.ndarray,
    direction: tuple[float, float, float],
    resolution: int,
) -> np.ndarray:
    """Orthographic coverage mask. No depth buffer -- only "is anything here".

    Depth is irrelevant to whether a pixel is see-through, and leaving it out
    means the result cannot depend on a near/far choice.
    """
    covered = np.zeros((resolution, resolution), dtype=bool)
    if len(faces) == 0:
        return covered

    right, up = _screen_basis(direction)
    sx, sy = positions @ right, positions @ up
    lo = np.array([sx.min(), sy.min()])
    hi = np.array([sx.max(), sy.max()])
    span = float((hi - lo).max())
    if span <= 0:
        return covered
    pad = span * 0.06
    scale = (resolution - 1) / (span + 2 * pad)
    px = (sx - (lo[0] - pad)) * scale
    py = (sy - (lo[1] - pad)) * scale

    a = np.stack([px[faces[:, 0]], py[faces[:, 0]]], axis=1)
    b = np.stack([px[faces[:, 1]], py[faces[:, 1]]], axis=1)
    c = np.stack([px[faces[:, 2]], py[faces[:, 2]]], axis=1)
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep = area2 != 0
    a, b, c, area2 = a[keep], b[keep], c[keep], area2[keep]
    if len(a) == 0:
        return covered

    last = resolution - 1
    x0 = np.clip(np.floor(np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0])), 0, last).astype(int)
    x1 = np.clip(np.ceil(np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0])), 0, last).astype(int)
    y0 = np.clip(np.floor(np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1])), 0, last).astype(int)
    y1 = np.clip(np.ceil(np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1])), 0, last).astype(int)

    # A triangle smaller than a pixel cannot be sampled by a pixel-centre test,
    # so mark the pixel it falls in. At the resolutions used here this is <1% of
    # triangles; the icosphere control in the tests is what proves it does not
    # invent holes.
    span_x, span_y = x1 - x0, y1 - y0
    subpixel = (span_x <= 1) & (span_y <= 1)
    if subpixel.any():
        cx = np.clip(((a[:, 0] + b[:, 0] + c[:, 0]) / 3).astype(int), 0, last)
        cy = np.clip(((a[:, 1] + b[:, 1] + c[:, 1]) / 3).astype(int), 0, last)
        covered[cy[subpixel], cx[subpixel]] = True

    rest = np.flatnonzero(~subpixel)
    if len(rest) == 0:
        return covered

    spans = np.maximum(span_x[rest], span_y[rest]) + 1
    batched = rest[spans <= _BATCH_MAX_SPAN]
    for k in np.unique(spans[spans <= _BATCH_MAX_SPAN]):
        sel = batched[spans[spans <= _BATCH_MAX_SPAN] == k]
        _rasterise_batch(covered, a[sel], b[sel], c[sel], area2[sel], x0[sel], y0[sel], int(k))
    for i in rest[spans > _BATCH_MAX_SPAN]:
        _rasterise_batch(
            covered,
            a[i : i + 1],
            b[i : i + 1],
            c[i : i + 1],
            area2[i : i + 1],
            x0[i : i + 1],
            y0[i : i + 1],
            int(max(span_x[i], span_y[i]) + 1),
        )
    return covered


def _rasterise_batch(
    covered: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    area2: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    k: int,
) -> None:
    """Pixel-centre barycentric fill for triangles sharing a k x k window size.

    Chunked over the triangle axis so the working set stays bounded regardless
    of how many triangles share a window size; `covered` is written in place,
    so chunking is exactly equivalent to one pass.
    """
    chunk = max(1, _BATCH_MAX_CELLS // (k * k))
    if len(a) > chunk:
        for s in range(0, len(a), chunk):
            e = s + chunk
            _rasterise_batch(covered, a[s:e], b[s:e], c[s:e], area2[s:e], x0[s:e], y0[s:e], k)
        return
    resolution = covered.shape[0]
    offs = np.arange(k)
    gx = (x0[:, None, None] + offs[None, None, :]).astype(np.float64) + 0.5
    gy = (y0[:, None, None] + offs[None, :, None]).astype(np.float64) + 0.5
    e0 = (b[:, 0, None, None] - a[:, 0, None, None]) * (gy - a[:, 1, None, None]) - (
        b[:, 1, None, None] - a[:, 1, None, None]
    ) * (gx - a[:, 0, None, None])
    e1 = (c[:, 0, None, None] - b[:, 0, None, None]) * (gy - b[:, 1, None, None]) - (
        c[:, 1, None, None] - b[:, 1, None, None]
    ) * (gx - b[:, 0, None, None])
    e2 = (a[:, 0, None, None] - c[:, 0, None, None]) * (gy - c[:, 1, None, None]) - (
        a[:, 1, None, None] - c[:, 1, None, None]
    ) * (gx - c[:, 0, None, None])
    sign = np.sign(area2)[:, None, None]
    inside = (e0 * sign >= 0) & (e1 * sign >= 0) & (e2 * sign >= 0)
    ix = np.clip(x0[:, None, None] + offs[None, None, :], 0, resolution - 1)
    iy = np.clip(y0[:, None, None] + offs[None, :, None], 0, resolution - 1)
    flat = np.broadcast_to(iy, inside.shape) * resolution + np.broadcast_to(ix, inside.shape)
    covered.reshape(-1)[flat[inside]] = True


def _enclosed_gaps(covered: np.ndarray) -> np.ndarray:
    """Uncovered pixels that the background cannot reach -- i.e. real holes."""
    free = ~covered
    reached = np.zeros_like(free)
    reached[0] |= free[0]
    reached[-1] |= free[-1]
    reached[:, 0] |= free[:, 0]
    reached[:, -1] |= free[:, -1]
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if grown.sum() == reached.sum():
            break
        reached = grown
    return free & ~reached


def _count_blobs(mask: np.ndarray) -> int:
    """Number of 4-connected regions, by iterative label propagation."""
    if not mask.any():
        return 0
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels[mask] = np.arange(1, int(mask.sum()) + 1)
    while True:
        grown = labels.copy()
        grown[1:] = np.maximum(grown[1:], labels[:-1])
        grown[:-1] = np.maximum(grown[:-1], labels[1:])
        grown[:, 1:] = np.maximum(grown[:, 1:], labels[:, :-1])
        grown[:, :-1] = np.maximum(grown[:, :-1], labels[:, 1:])
        grown[~mask] = 0
        if np.array_equal(grown, labels):
            break
        labels = grown
    return len(np.unique(labels)) - 1



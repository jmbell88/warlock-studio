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
resolution offline.

The reachability half is one ``cv2.connectedComponents`` pass. It used to be two
iterative fixpoints -- grow a boolean plane by four shifted ORs until it stops
changing, then propagate integer labels the same way to count blobs -- each of
which is O(image diameter) full-array passes, i.e. a cost scaling with
resolution^3 and the whole reason the request path measures at half resolution.
Connected components answer both questions exactly and in one linear pass, so
the result is identical rather than merely close: a hole is a component of the
uncovered pixels that does not touch the border, which is precisely what the
flood fill could not reach.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from . import native

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

# What the worker measures every finished mesh at.
#
# This was 512 for one reason: the two fixpoint loops cost ~resolution^3, so
# full resolution meant tens of seconds per mesh. Connected components and the
# native rasteriser removed that term -- a four-view audit at 1024 is ~0.13 s on
# a test sphere and ~0.8 s on a 290k-face reconstruction -- so the halving buys
# nothing now and costs the sub-pixel gaps 512 could not see.
#
# The reason it is safe to change a number the corpus is keyed on is measured,
# not assumed: docs/measurements/2026-08-06-audit-resolution.md walks meshes
# from 0.0 to 0.84 hole fraction at both resolutions and finds the largest
# disagreement is 0.00045 -- two orders of magnitude below the empty band from
# 0.0308 to 0.1010 that Config.mesh_hole_max sits in, and below the ~0.3% noise
# floor the band sweep already established. Nothing reclassifies.
REQUEST_PATH_RESOLUTION = 1024


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
        holes, blobs = _enclosed_gaps(covered)
        hole_px = int(holes.sum())
        silhouette_px = int(covered.sum()) + hole_px
        per_view.append(
            {
                "direction": tuple(float(c) for c in direction),
                "silhouette_px": silhouette_px,
                "hole_px": hole_px,
                "hole_fraction": hole_px / silhouette_px if silhouette_px else 0.0,
                "blobs": blobs,
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
    projected = _project(positions, faces, direction, resolution)
    if projected is None:
        return covered
    a, b, c, area2 = projected

    if native.available():
        # The kernel takes one column per coordinate, so the (n, 2) stacks are
        # split here rather than in C. ``ascontiguousarray`` is not decoration:
        # ``a[:, 0]`` is a strided view over interleaved pairs, and a pointer
        # to it would read every other value.
        native.rasterise(
            *(
                np.ascontiguousarray(col)
                for col in (a[:, 0], a[:, 1], b[:, 0], b[:, 1], c[:, 0], c[:, 1])
            ),
            np.ascontiguousarray(area2),
            # bool and uint8 are the same byte; the view shares the buffer, so
            # the kernel writes into ``covered`` itself.
            covered.view(np.uint8),
        )
        return covered

    _fill(covered, a, b, c, area2, resolution)
    return covered


def _project(
    positions: np.ndarray,
    faces: np.ndarray,
    direction: tuple[float, float, float],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Triangles in pixel space, degenerates dropped. None if there are none.

    Cheap and vectorised, so it stays in numpy whichever fill runs: this is a
    few passes over the vertex array, against a rasteriser that touches every
    pixel of every triangle's bounding box.
    """
    if len(faces) == 0:
        return None

    right, up = _screen_basis(direction)
    sx, sy = positions @ right, positions @ up
    lo = np.array([sx.min(), sy.min()])
    hi = np.array([sx.max(), sy.max()])
    span = float((hi - lo).max())
    if span <= 0:
        return None
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
    return None if len(a) == 0 else (a, b, c, area2)


def _fill(
    covered: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    area2: np.ndarray,
    resolution: int,
) -> None:
    """The numpy rasteriser: the reference the kernel is measured against.

    Kept whole rather than deleted once ``warlockc`` existed. It is the
    fallback on a machine with no compiler, and it is what
    ``test_the_native_rasteriser_matches_numpy_bit_for_bit`` compares to --
    a native path with no reference is a native path nobody can check.
    """
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
        return

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


def _enclosed_gaps(covered: np.ndarray) -> tuple[np.ndarray, int]:
    """-> (holes, blob count). Uncovered pixels the background cannot reach.

    One labelling answers both: a component of the uncovered pixels is either
    the background -- which is exactly the set of components touching the image
    border, since the fill used to start from all four edges -- or a hole. The
    count of the rest is the blob count, with no second pass.

    ``connectivity=4`` is not a default worth taking silently: both fixpoints
    this replaces grew by four shifted ORs, so eight-connectivity would merge
    holes that touch only at a corner and change the count.
    """
    import cv2

    free = (~covered).astype(np.uint8)
    count, labels = cv2.connectedComponents(free, connectivity=4)
    if count <= 1:  # nothing uncovered at all
        return np.zeros(covered.shape, dtype=bool), 0

    outside = np.zeros(count, dtype=bool)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    outside[np.unique(border)] = True
    outside[0] = True  # label 0 is the covered pixels, not a gap
    holes = ~outside[labels]
    return holes, int(count - int(outside.sum()))



"""Picking elements on screen: projection, nearest-hit and marquee masks.

Pure numpy, and that is the point. The obvious way to pick a vertex in a 3D
viewport is to render an id buffer and read the pixel back, and ``viewer/
picking.py``'s own docstring already rejects that for the gizmo: a readback
cannot be asserted without a GL context, so every behaviour it encodes becomes
untestable at exactly the moment it is most fiddly. Everything here is a matmul
and a comparison, so the whole of "what does clicking there select" is pinned by
tests that never open a window.

**Occlusion is a depth test against the ray-picked surface, plus a bias.** The
frame already ray-picks the surface under the cursor to decide which object was
clicked, which gives a distance from the eye to whatever is actually visible
there; an element is pickable when it is no further away than that, within a
relative bias. Three behaviours fall out of that one rule rather than being
special-cased, and each is what Wings3D does:

* On a **closed** mesh the far side is rejected -- the surface hit is the near
  wall, and a vertex on the back is metres behind it.
* On an **open sheet** both sides pick, because there is only one surface and
  every element on it sits at the hit depth.
* On a **silhouette click** there is no surface hit at all -- the ray misses --
  and the rim elements still pick, which is how a user grabs the outline of a
  shape rather than having to click just inside it.

**A marquee is through-selection, with no occlusion at all.** A vertex counts if
it is in the rectangle, an edge if *both* its endpoints are, a face if *all* its
corners are. That is deliberately not what a click does: a marquee over a
blockout is asking for "everything in this region", the far side included, and
an occluded marquee makes a user rotate the model twice to select a box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "EDGE_RADIUS",
    "OCCLUSION_BIAS",
    "VERT_RADIUS",
    "Screen",
    "marquee_edges",
    "marquee_faces",
    "marquee_verts",
    "nearest_edge",
    "nearest_vertex",
    "project",
    "visible",
]

# Click radii in pixels. A vertex is a point and needs the larger target; an
# edge is a line the cursor can follow, so it needs less. Both are generous
# enough for a trackpad and small enough that two adjacent vertices of a dense
# imported mesh do not overlap into one blob.
VERT_RADIUS = 8.0
EDGE_RADIUS = 6.0

# Relative slack on the occlusion depth test. An element *on* the picked
# surface is at the same distance as the hit in exact arithmetic and a few
# ulps either side of it in practice, and a vertex that failed its own
# surface's depth test would be unpickable from the front -- the worst possible
# failure, because it looks like the click was simply ignored.
OCCLUSION_BIAS = 1e-3


@dataclass(frozen=True)
class Screen:
    """A mesh's vertices, projected once for one frame and one object."""

    xy: np.ndarray  # (V, 2) f8 -- pixels within the viewport rect
    depth: np.ndarray  # (V,)  f8 -- euclidean distance from the eye
    front: np.ndarray  # (V,)  bool -- in front of the camera at all


def project(
    positions: np.ndarray,
    matrix: np.ndarray,
    view_proj: np.ndarray,
    eye: np.ndarray,
    width: int,
    height: int,
) -> Screen:
    """Project object-space positions to viewport pixels, in one matmul.

    ``matrix`` is the object's world transform and ``view_proj`` the camera's;
    they are kept separate rather than pre-multiplied because ``depth`` is
    measured in *world* space against the eye, which the composed matrix has
    thrown away.

    A vertex behind the eye comes back with ``front=False`` and its ``xy``
    forced far outside the viewport rather than mirrored through the origin,
    which is what dividing by a negative ``w`` would do -- a vertex behind the
    camera would otherwise be pickable at a plausible-looking position.
    """
    positions = np.asarray(positions, dtype="f8").reshape(-1, 3)
    if len(positions) == 0:
        return Screen(
            np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool)
        )
    homo = np.hstack([positions, np.ones((len(positions), 1))])
    world = (np.asarray(matrix, dtype="f8") @ homo.T).T
    clip = (np.asarray(view_proj, dtype="f8") @ world.T).T

    w = clip[:, 3]
    front = w > 1e-9
    safe = np.where(front, w, 1.0)
    ndc = clip[:, :3] / safe[:, None]
    xy = np.stack(
        [(ndc[:, 0] + 1.0) * 0.5 * width, (1.0 - ndc[:, 1]) * 0.5 * height], axis=1
    )
    xy[~front] = -1e9
    depth = np.linalg.norm(world[:, :3] - np.asarray(eye, dtype="f8"), axis=1)
    return Screen(xy=xy, depth=depth, front=front)


def visible(screen: Screen, surface_depth: float | None) -> np.ndarray:
    """A ``(V,)`` bool: which vertices the occlusion rule admits.

    ``surface_depth`` is ``None`` when the ray missed everything -- a
    silhouette click -- and then nothing is occluded, which is the whole reason
    the rim of a shape is clickable.
    """
    if surface_depth is None:
        return screen.front
    return screen.front & (screen.depth <= surface_depth * (1.0 + OCCLUSION_BIAS))


def _pick_nearest(
    distance: np.ndarray, ok: np.ndarray, depth: np.ndarray, radius: float
) -> int | None:
    """The nearest candidate within *radius*, breaking ties by depth.

    Ties are real: on a cube seen down an axis the near and far vertices of an
    edge project to the same pixel, and picking the nearer one is what the user
    means by clicking it.
    """
    within = ok & (distance <= radius)
    if not within.any():
        return None
    candidates = np.flatnonzero(within)
    order = np.lexsort((depth[candidates], np.round(distance[candidates], 3)))
    return int(candidates[order[0]])


def nearest_vertex(
    screen: Screen,
    point: tuple[float, float],
    *,
    surface_depth: float | None = None,
    radius: float = VERT_RADIUS,
) -> int | None:
    """The vertex under the cursor, or ``None``."""
    if len(screen.xy) == 0:
        return None
    delta = screen.xy - np.asarray(point, dtype="f8")
    return _pick_nearest(
        np.linalg.norm(delta, axis=1), visible(screen, surface_depth), screen.depth, radius
    )


def _segment_distance(
    a: np.ndarray, b: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(distance, t)`` from *point* to each segment, with the clamped parameter."""
    ab = b - a
    length2 = np.einsum("ij,ij->i", ab, ab)
    t = np.divide(
        np.einsum("ij,ij->i", point - a, ab),
        length2,
        out=np.zeros(len(a)),
        where=length2 > 0.0,
    )
    t = np.clip(t, 0.0, 1.0)
    closest = a + ab * t[:, None]
    return np.linalg.norm(closest - point, axis=1), t


def nearest_edge(
    screen: Screen,
    edges: np.ndarray,
    point: tuple[float, float],
    *,
    surface_depth: float | None = None,
    radius: float = EDGE_RADIUS,
) -> int | None:
    """The edge under the cursor as an index into *edges*, or ``None``.

    An edge's depth is interpolated at the closest point along it rather than
    taken from either endpoint: grabbing the middle of a long edge that runs
    away from the camera should compare against the depth *there*, or the edge
    becomes unpickable over half its length on a closed mesh. The interpolation
    is linear in *screen* space, which is not exactly the depth of the 3D point
    under the cursor -- perspective is not affine -- but it is monotonic along
    the edge and within a pixel of right at any angle a user can click, and the
    exact version costs a per-edge reciprocal for an answer that is then
    compared against a biased threshold anyway.
    """
    edges = np.asarray(edges, dtype="i8").reshape(-1, 2)
    if len(edges) == 0 or len(screen.xy) == 0:
        return None
    a, b = screen.xy[edges[:, 0]], screen.xy[edges[:, 1]]
    distance, t = _segment_distance(a, b, np.asarray(point, dtype="f8"))
    depth = screen.depth[edges[:, 0]] * (1.0 - t) + screen.depth[edges[:, 1]] * t
    ok = screen.front[edges].all(axis=1)
    if surface_depth is not None:
        ok &= depth <= surface_depth * (1.0 + OCCLUSION_BIAS)
    return _pick_nearest(distance, ok, depth, radius)


# --- marquee ----------------------------------------------------------------


def _inside(screen: Screen, rect: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = (
        min(rect[0], rect[2]),
        min(rect[1], rect[3]),
        max(rect[0], rect[2]),
        max(rect[1], rect[3]),
    )
    x, y = screen.xy[:, 0], screen.xy[:, 1]
    return screen.front & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def marquee_verts(
    screen: Screen, rect: tuple[float, float, float, float]
) -> np.ndarray:
    """Vertex indices inside the rectangle. Through-selection; see the module docs."""
    return np.flatnonzero(_inside(screen, rect)).astype("i4")


def marquee_edges(
    screen: Screen, edges: np.ndarray, rect: tuple[float, float, float, float]
) -> np.ndarray:
    """Edges with **both** endpoints inside, as ``(n, 2)`` vertex pairs.

    Both rather than either: a marquee that took every edge with one endpoint
    inside would drag in a fringe of edges leading out of the region, and the
    user would have to Ctrl-drag them back off one at a time.
    """
    edges = np.asarray(edges, dtype="i4").reshape(-1, 2)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype="i4")
    inside = _inside(screen, rect)
    return edges[inside[edges].all(axis=1)]


def marquee_faces(
    screen: Screen,
    loops: np.ndarray,
    starts: np.ndarray,
    rect: tuple[float, float, float, float],
) -> np.ndarray:
    """Faces with **all** corners inside, as face indices.

    ``reduceat`` over the CSR spans rather than a loop per face -- the marquee
    runs while the mouse is moving, on whatever the user dragged over.
    """
    starts = np.asarray(starts, dtype="i8")
    if len(starts) < 2 or len(loops) == 0:
        return np.zeros(0, dtype="i4")
    inside = _inside(screen, rect)[np.asarray(loops, dtype="i8")]
    whole = np.minimum.reduceat(inside.astype("i1"), starts[:-1]) > 0
    return np.flatnonzero(whole).astype("i4")

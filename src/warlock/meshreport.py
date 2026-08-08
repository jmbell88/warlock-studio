"""Is this mesh usable in a game engine, and if not, why not.

Replaces the single "watertight" badge, which was a silhouette measurement
wearing a topology word. The two questions are genuinely different: meshaudit
answers "can you see through it", which is what a player notices, and this
module answers "will an importer accept it and will it sit on the floor", which
is what an engine notices. Both are reported; only topology may use the word
watertight.

Deliberately advisory. Nothing here rejects a mesh -- a `review` model is still
downloadable, because a warning the user can act on beats a job that refuses to
hand over the thing it already spent two minutes making.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# A pivot this far off the floor, as a fraction of the model's height, is a
# grounding failure rather than float noise.
GROUND_TOLERANCE = 0.001

# Achieved longest axis may miss the requested size by this fraction before it
# is worth saying so.
SIZE_TOLERANCE = 0.01

# Above this the mesh is a source reconstruction, not a game asset. The default
# trellis output is ~290k triangles, so this fires until the optimizer runs.
TRIANGLE_BUDGET = 150_000

# Silhouette hole fraction at which the mesh stops being cosmetically fine.
HOLE_WARN = 0.02

# Two vertices closer than this fraction of the bounding-box diagonal are the
# same point, for the purpose of the welded analysis copy below.
#
# The mesh is loaded with `process=False` on purpose -- the UV and material
# checks need the unwelded vertices -- but that made every xatlas UV-seam split
# count as a boundary edge, so the watertight figure was mostly measuring
# seams. A seam split carries *identical* positions, so any positive tolerance
# welds it; the fraction is there only so a rewriter that round-trips a
# position through float32 does not leave two copies a few ulps apart. It is
# relative to the model because an absolute epsilon means something different
# on a 0.02 m gear than on a 30 m building.
#
# No `docs/measurements/` document backs this number: nothing in the stored
# corpus is keyed on it. It decides one boolean, and the boolean it replaces
# was measuring the wrong thing entirely.
WELD_TOLERANCE = 1e-5


def build(
    glb_path: Path,
    *,
    target_size_m: float | None = None,
    silhouette: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a finished GLB and classify it ready / review / invalid."""
    import numpy as np
    import trimesh

    try:
        loaded = trimesh.load(glb_path, process=False)
        mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0 or not np.isfinite(vertices).all():
            raise ValueError("mesh has no finite geometry")
    except Exception as exc:
        log.warning("mesh report could not parse %s: %s", glb_path, exc)
        return {
            "status": "invalid",
            "reasons": [f"could not read the mesh: {exc}"],
            "bytes": _size(glb_path),
            "silhouette": silhouette,
        }

    reasons: list[str] = []

    # Topology. trimesh's own predicates, not a reimplementation: they are what
    # every other consumer of this format uses to decide the same questions.
    watertight = bool(mesh.is_watertight)
    components, boundary_edges, nonmanifold_edges = _topology(trimesh, np, mesh)
    degenerate = int((~mesh.nondegenerate_faces()).sum())

    # The same three questions again, on a copy welded by position -- which is
    # the answer anyone actually means by "is it watertight". Falls back to the
    # raw numbers rather than raising: this module is advisory throughout.
    welded = _welded(trimesh, np, vertices, faces)
    if welded is None:
        welded_watertight = watertight
        welded_components, welded_boundary_edges = components, boundary_edges
    else:
        welded_watertight = bool(welded.is_watertight)
        welded_components, welded_boundary_edges, _ = _topology(trimesh, np, welded)

    extents = [float(v) for v in mesh.extents]
    achieved = float(max(extents)) if extents else 0.0
    bounds_min = [float(v) for v in mesh.bounds[0]]
    height = extents[1] if len(extents) > 2 else 0.0
    # glTF is Y-up, so the floor is minimum Y.
    grounded = abs(bounds_min[1]) <= max(height * GROUND_TOLERANCE, 1e-6)

    has_normals = bool(
        getattr(mesh, "vertex_normals", None) is not None
        and len(mesh.vertex_normals) == len(vertices)
    )
    has_uvs, textures = _materials(mesh)

    triangles = int(len(faces))
    if not welded_watertight:
        reasons.append(
            f"not watertight: {welded_boundary_edges} boundary edge(s) in "
            f"{welded_components} component(s), after welding vertices by position"
        )
    if nonmanifold_edges:
        reasons.append(f"{nonmanifold_edges} non-manifold edge(s)")
    if degenerate:
        reasons.append(f"{degenerate} degenerate triangle(s)")
    if triangles > TRIANGLE_BUDGET:
        reasons.append(f"{triangles:,} triangles is above the {TRIANGLE_BUDGET:,} budget")
    if not has_uvs:
        reasons.append("no UV coordinates")
    if not textures["base_color"]:
        reasons.append("no base-color texture")
    if not textures["metallic_roughness"]:
        reasons.append("no metallic/roughness texture")
    if not grounded:
        reasons.append(f"pivot is {bounds_min[1]:.4f} m off the floor")
    if (
        target_size_m
        and achieved > 0
        and abs(achieved - target_size_m) / target_size_m > SIZE_TOLERANCE
    ):
        reasons.append(f"longest axis is {achieved:.3f} m, asked for {target_size_m:.3f} m")
    worst = (silhouette or {}).get("worst")
    if isinstance(worst, (int, float)) and worst > HOLE_WARN:
        reasons.append(f"{worst * 100:.1f}% of the worst silhouette is see-through")

    return {
        "status": "review" if reasons else "ready",
        "reasons": reasons,
        "triangles": triangles,
        "vertices": int(len(vertices)),
        "components": components,
        "degenerate": degenerate,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "watertight": watertight,
        # Additive, and the raw three above keep their meaning: every reader is
        # `.get`-based, and the unwelded numbers still say how badly the file is
        # split (a rig or an exporter cares).
        "welded_watertight": welded_watertight,
        "welded_boundary_edges": welded_boundary_edges,
        "welded_components": welded_components,
        "has_uvs": has_uvs,
        "has_normals": has_normals,
        "textures": textures,
        "extents_m": extents,
        "achieved_size_m": achieved,
        "grounded": grounded,
        "bytes": _size(glb_path),
        "silhouette": silhouette,
    }


def _topology(trimesh: Any, np: Any, mesh: Any) -> tuple[int, int, int]:
    """-> (components, boundary edges, non-manifold edges)."""
    # Only the *count* is wanted, so the face-adjacency graph is walked
    # directly rather than through mesh.split(), which builds a full Trimesh --
    # vertices, faces, visual and all -- for every shell it finds. On a
    # 500k-triangle trellis reconstruction with a few hundred stray shells that
    # is a large transient allocation to compute one integer.
    components = int(
        len(
            # `nodes` is not optional here: face_adjacency omits any face with
            # no neighbour, and a lone floating triangle is a component.
            trimesh.graph.connected_components(
                mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
            )
        )
    )
    # An edge with exactly one adjacent face is a boundary edge; more than two
    # is non-manifold. Both fall out of counting how many times each unique
    # edge is referenced, which is one bincount rather than two traversals.
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return components, int((counts == 1).sum()), int((counts > 2).sum())


def _welded(trimesh: Any, np: Any, vertices: Any, faces: Any) -> Any | None:
    """A copy of the mesh with coincident vertices merged, or None.

    Positions are quantised onto a `WELD_TOLERANCE * diagonal` lattice and
    `np.unique` does the merging, which is one sort rather than a spatial
    query. Quantising can in principle leave two points a hair under the
    tolerance on opposite sides of a cell boundary; that is fine here, because
    the case this exists for -- a UV seam split -- duplicates the position
    *exactly*, so both copies land in the same cell whatever the offset.

    Faces that collapse to fewer than three distinct vertices are dropped: they
    have no area, and trimesh's edge bookkeeping would count their repeated
    edge as a boundary and undo the whole point.
    """
    try:
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        tolerance = diagonal * WELD_TOLERANCE
        if not (tolerance > 0.0):
            return None
        _, index, inverse = np.unique(
            np.round(vertices / tolerance), axis=0, return_index=True, return_inverse=True
        )
        merged = faces[:, :3] if faces.ndim == 2 else faces
        merged = inverse.reshape(-1)[merged]
        keep = (
            (merged[:, 0] != merged[:, 1])
            & (merged[:, 1] != merged[:, 2])
            & (merged[:, 0] != merged[:, 2])
        )
        merged = merged[keep]
        if len(merged) == 0:
            return None
        # The representative vertex is an original position, not a rounded one:
        # the lattice is a lookup key, never geometry.
        return trimesh.Trimesh(vertices=vertices[index], faces=merged, process=False)
    except Exception as exc:  # pragma: no cover - defensive, like the rest here
        log.warning("mesh report could not weld the mesh: %s", exc)
        return None


def _materials(mesh: Any) -> tuple[bool, dict[str, bool]]:
    """-> (has UVs, which PBR maps are present).

    Reads the material off the merged mesh rather than the glTF JSON: trimesh
    has already resolved which image feeds which slot, and re-deriving that from
    the raw JSON would be a second, disagreeing answer to the same question.
    """
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    has_uvs = uv is not None and len(uv) > 0
    material = getattr(visual, "material", None)
    return has_uvs, {
        "base_color": getattr(material, "baseColorTexture", None) is not None,
        "metallic_roughness": getattr(material, "metallicRoughnessTexture", None) is not None,
        "normal": getattr(material, "normalTexture", None) is not None,
    }


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0

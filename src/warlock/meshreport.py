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
    components = int(len(mesh.split(only_watertight=False)))
    # An edge with exactly one adjacent face is a boundary edge; more than two
    # is non-manifold. Both fall out of counting how many times each unique
    # edge is referenced, which is one bincount rather than two traversals.
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary_edges = int((counts == 1).sum())
    nonmanifold_edges = int((counts > 2).sum())
    degenerate = int((~mesh.nondegenerate_faces()).sum())

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
    if not watertight:
        reasons.append(
            f"not watertight: {boundary_edges} boundary edge(s) in {components} component(s)"
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
        "has_uvs": has_uvs,
        "has_normals": has_normals,
        "textures": textures,
        "extents_m": extents,
        "achieved_size_m": achieved,
        "grounded": grounded,
        "bytes": _size(glb_path),
        "silhouette": silhouette,
    }


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

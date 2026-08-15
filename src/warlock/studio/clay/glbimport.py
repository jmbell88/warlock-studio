"""A GLB into a Clay document, so an imported asset is an editable one.

The loop the rest of Clay was built to close: ``jobs.import_mesh`` turns a Clay
document into an ordinary library asset, and this turns an ordinary asset back
into a Clay document. It goes through :func:`~..viewer.gltf.load`, which already
composes every node's transform (including the grounding ``normalize_glb``
inserts under each root), already decodes textures, and already refuses sparse
accessors and non-triangle modes -- so nothing here re-implements a loader, and
an asset that will not open in the 3D viewport will not half-open here either.

**One ``Obj`` per primitive, not per node.** A glTF primitive carries exactly
one material, and Clay stores material per *face*, so the two are not the same
split -- but a node with three primitives is three materials, and merging them
into one object would need a palette lookup per face at import time to say
nothing the user asked for. Three objects the outliner shows separately, that
can be selected and moved separately, is what a modeller expects from a file
that was authored that way.

**Vertices are merged bitwise, not by tolerance.** An exporter splits a vertex
wherever a normal or a uv disagrees, and the split copies are *bit-identical* in
position -- so ``np.unique`` over the raw twelve bytes puts them back together
exactly, with no epsilon to choose and no chance of welding two features that
happen to be a hair apart. Tolerance welding is a repair tool, and Clay has one
(``ops_topo.weld``); making it the import default would silently reshape assets.

**The uv survives the merge because it is per corner.** That is precisely what
the per-corner ``Mesh.uv`` is for: the exporter's split vertices carried
different uvs at one position, and after the merge those become two corners at
one vertex, which is exactly how a seam is expressed here.

**Smoothing is a heuristic, and it is stated as one.** glTF has no smooth flag;
it has normals. A face whose corner normals all agree with its own geometric
normal was flat-shaded, and one where they do not was smooth -- measured against
a cosine of 0.999, which is about 2.5 degrees. A mesh with no normals at all is
taken as smooth, because that is what a reader that computes its own normals
will do with it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..viewer import gltf
from ..viewer import math3d as m3
from . import topo
from .document import ClayDoc, Obj, new_uid
from .elements import OpError

__all__ = ["MAX_TRIANGLES", "glb_to_claydoc"]

# Above this an import is refused rather than attempted. A Clay document holds
# every mesh in memory twice per undo step, and the rebuild-per-edit cost is
# linear in the triangle count -- two million is already "usable but slow" and
# past it the app stops responding rather than becoming slower.
MAX_TRIANGLES = 2_000_000

# How closely a corner normal must agree with its face's own normal to count as
# flat. cos(2.5 degrees); tighter than this and float noise in an exporter's
# normals reads every flat face as smooth.
FLAT_COSINE = 0.999


def glb_to_claydoc(data: bytes, name: str = "Imported") -> ClayDoc:
    """Parse GLB bytes into a fresh :class:`~.document.ClayDoc`.

    The document comes back with a clean history and nothing selected: it has
    just been opened, so it is not unsaved, and the objects are placed by
    construction rather than through ``add_object``, which would push a step
    apiece.
    """
    try:
        model = gltf.load(data)
    except Exception as exc:  # noqa: BLE001 - the loader raises several types
        raise OpError(f"This file could not be read as a GLB. {exc}") from exc

    if model.skins:
        raise OpError(
            "This GLB is rigged, and Clay has no skinning -- editing it here "
            "would drop the rig. Open it in Create instead."
        )
    total = sum(len(p.indices) // 3 for prims in model.meshes for p in prims)
    if total > MAX_TRIANGLES:
        raise OpError(
            f"This mesh has {total:,} triangles, past the {MAX_TRIANGLES:,} Clay "
            "can edit. Retarget its triangle budget in the library first."
        )

    materials: list[gltf.Material] = []
    palette: dict[int, int] = {}
    objects: list[Obj] = []
    taken: set[str] = set()
    for node in model.nodes:
        if node.mesh is None:
            continue
        for prim in model.meshes[node.mesh]:
            obj = _object_for(prim, node, name, materials, palette, taken)
            if obj is not None:
                objects.append(obj)

    if not objects:
        raise OpError("This GLB has no meshes in it.")
    return ClayDoc(objects=objects, materials=materials or None)


def _object_for(
    prim: gltf.Primitive,
    node: gltf.Node,
    base: str,
    materials: list[gltf.Material],
    palette: dict[int, int],
    taken: set[str],
) -> Obj | None:
    if len(prim.indices) < 3 or len(prim.positions) == 0:
        return None
    mesh = _mesh_for(prim, _material_index(prim.material, materials, palette))
    translation, rotation, scale = m3.decompose(node.world)
    return Obj(
        uid=new_uid(),
        name=_unique(node.name or base, taken),
        mesh=mesh,
        translation=translation,
        rotation=rotation,
        scale=scale,
        # An imported mesh is not describable by a generator's parameters, so
        # the properties panel shows counts rather than a size field that would
        # discard the file the moment it was touched.
        generator=None,
    )


def _material_index(
    material: gltf.Material | None,
    materials: list[gltf.Material],
    palette: dict[int, int],
) -> int:
    """The palette slot for a loader material, adding it the first time.

    Deduplicated by the loader's own object identity -- two primitives that
    reference one glTF material share the object, and two that reference
    different ones must stay separate even if every factor matches, because the
    user is about to edit them.
    """
    if material is None:
        material = gltf.Material(name="imported")
    key = id(material)
    if key not in palette:
        palette[key] = len(materials)
        materials.append(material)
    return palette[key]


def _mesh_for(prim: gltf.Primitive, material: int) -> Any:
    """One primitive's triangles as a CSR mesh, vertices merged bitwise."""
    positions = np.ascontiguousarray(prim.positions, dtype="f4").reshape(-1, 3)
    indices = np.asarray(prim.indices, dtype="i8").reshape(-1)
    tris = indices[: (len(indices) // 3) * 3].reshape(-1, 3)

    # ``np.unique`` over the raw rows: an exporter's split copies are
    # bit-identical in position, so this is exact rather than tolerant.
    merged, inverse = np.unique(positions, axis=0, return_inverse=True)
    loops = inverse.reshape(-1)[tris].reshape(-1)

    uv = None
    if prim.uvs is not None:
        uv = np.ascontiguousarray(prim.uvs, dtype="f4").reshape(-1, 2)[tris].reshape(-1, 2)

    n_faces = len(tris)
    return topo.rebuild(
        merged,
        loops,
        topo.starts_from_counts(np.full(n_faces, 3, dtype="i8")),
        np.full(n_faces, int(material), dtype="i4"),
        _smooth_flags(prim, merged, loops, tris),
        uv=uv,
    )


def _smooth_flags(
    prim: gltf.Primitive, positions: np.ndarray, loops: np.ndarray, tris: np.ndarray
) -> np.ndarray:
    """Per-face smooth flags, from how far the corner normals bend. See the module docs."""
    n_faces = len(tris)
    if prim.normals is None or n_faces == 0:
        return np.ones(n_faces, dtype=bool)

    corners = positions[loops.reshape(-1, 3)].astype("f8")
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    face = _unit(face)
    supplied = _unit(
        np.asarray(prim.normals, dtype="f8").reshape(-1, 3)[tris].reshape(-1, 3)
    ).reshape(-1, 3, 3)
    agreement = np.einsum("ijk,ik->ij", supplied, face).min(axis=1)
    return agreement <= FLAT_COSINE


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0.0)


def _unique(name: str, taken: set[str]) -> str:
    """``Box`` -> ``Box.001``, the same counting rule the outliner elsewhere uses."""
    if name not in taken:
        taken.add(name)
        return name
    for n in range(1, len(taken) + 2):
        candidate = f"{name}.{n:03d}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    return name  # pragma: no cover - unreachable, the loop is bounded above

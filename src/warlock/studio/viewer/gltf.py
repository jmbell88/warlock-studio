"""A GLB loader, in numpy and Pillow.

Hand-rolled rather than delegated. trimesh discards a scene root's transform --
which is exactly where ``normalize_glb`` puts the grounding transform, so a
trimesh-loaded model would sit in the wrong place and at the wrong size -- and
it has no notion of a skin at all. pygltflib parses the JSON but still leaves
accessor decoding here. What is left after those two is small enough to own.

The node graph stays live after loading: posing a rig means setting a joint
node's local rotation and recomputing world matrices, so nothing is baked.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...glbio import read_glb
from . import math3d as m3

log = logging.getLogger(__name__)

# glTF componentType -> numpy dtype.
_COMPONENT = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class Material:
    """A pbrMetallicRoughness material, with its textures already decoded.

    Images are kept as raw RGBA bytes plus a size rather than as PIL objects:
    the only consumer uploads them to the GPU, and a decoded PIL image held for
    the life of the model is 30 MB of nothing.
    """

    name: str = ""
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    # (width, height, rgba_bytes) per slot, or None.
    base_color: tuple[int, int, bytes] | None = None
    metallic_roughness: tuple[int, int, bytes] | None = None
    normal: tuple[int, int, bytes] | None = None
    emissive: tuple[int, int, bytes] | None = None
    occlusion: tuple[int, int, bytes] | None = None


@dataclass
class Primitive:
    positions: np.ndarray  # (n, 3) f4
    indices: np.ndarray  # (m,) u4
    normals: np.ndarray | None = None  # (n, 3) f4
    uvs: np.ndarray | None = None  # (n, 2) f4
    # Promoted from uint8 to int32: GLSL 3.30 has no unsigned-byte vertex
    # attribute that survives as an integer, and ivec4 is what the skinning
    # shader indexes its palette with.
    joints: np.ndarray | None = None  # (n, 4) i4
    weights: np.ndarray | None = None  # (n, 4) f4
    material: Material = field(default_factory=Material)


@dataclass
class Skin:
    joints: list[int]  # node indices, in palette order
    inverse_bind: np.ndarray  # (n, 4, 4)


@dataclass
class Node:
    name: str = ""
    translation: np.ndarray = field(default_factory=lambda: m3.vec3())
    rotation: np.ndarray = field(default_factory=m3.quat_identity)
    scale: np.ndarray = field(default_factory=lambda: m3.vec3(1, 1, 1))
    children: list[int] = field(default_factory=list)
    mesh: int | None = None
    skin: int | None = None
    # Filled by Model.update_world(); never trusted before that runs.
    world: np.ndarray = field(default_factory=m3.identity)

    def local(self) -> np.ndarray:
        return m3.compose(self.translation, self.rotation, self.scale)


class Model:
    """A loaded GLB: a node graph, its meshes and its skins."""

    def __init__(
        self,
        nodes: list[Node],
        roots: list[int],
        meshes: list[list[Primitive]],
        skins: list[Skin],
    ) -> None:
        self.nodes = nodes
        self.roots = roots
        self.meshes = meshes
        self.skins = skins
        # A joint node is addressed by name everywhere above this layer: a
        # pose is a bone->rotation map, and the browser never saw an index.
        self.by_name: dict[str, int] = {}
        for i, node in enumerate(nodes):
            # First wins: a duplicate name is a broken rig either way, and
            # picking the later one would silently move a different joint.
            if node.name and node.name not in self.by_name:
                self.by_name[node.name] = i
        self.rest_rotations = [n.rotation.copy() for n in nodes]
        self.update_world()

    # -- transforms --------------------------------------------------------

    def update_world(self) -> None:
        """Recompute every node's world matrix from the roots down.

        Called once at load and again after every pose change. Iterative
        rather than recursive: a skeleton is shallow but a mesh hierarchy from
        an exporter need not be, and a blown stack in the frame loop is not a
        failure mode worth having.
        """
        stack = [(r, m3.identity()) for r in reversed(self.roots)]
        while stack:
            index, parent = stack.pop()
            node = self.nodes[index]
            node.world = parent @ node.local()
            for child in reversed(node.children):
                stack.append((child, node.world))

    def mesh_instances(self) -> list[tuple[Node, list[Primitive]]]:
        return [(n, self.meshes[n.mesh]) for n in self.nodes if n.mesh is not None]

    def joint_palette(self, node: Node) -> list[np.ndarray] | None:
        """The skinning matrices for one mesh node, in palette order.

        Per the glTF spec the mesh node's own world transform is divided back
        out: the vertices are already in the skin's space, so leaving it in
        would apply the grounding transform twice.
        """
        if node.skin is None:
            return None
        skin = self.skins[node.skin]
        inv_mesh = np.linalg.inv(node.world)
        return [
            inv_mesh @ self.nodes[j].world @ skin.inverse_bind[i]
            for i, j in enumerate(skin.joints)
        ]

    # -- posing ------------------------------------------------------------

    def set_rotation(self, bone: str, quat_xyzw: Any) -> bool:
        """Set one joint node's *local* rotation. -> whether the bone exists.

        Local, not world: that is the entire pose contract. A glTF joint's
        local transform is what the Blender worker reconstructs its basis from,
        so anything else here would silently disagree with the bake.
        """
        index = self.by_name.get(bone)
        if index is None:
            return False
        self.nodes[index].rotation = m3.quat_normalize(np.asarray(quat_xyzw, dtype="f8"))
        return True

    def get_rotation(self, bone: str) -> np.ndarray | None:
        index = self.by_name.get(bone)
        return None if index is None else self.nodes[index].rotation.copy()

    def reset_rotation(self, bone: str) -> bool:
        index = self.by_name.get(bone)
        if index is None:
            return False
        self.nodes[index].rotation = self.rest_rotations[index].copy()
        return True

    def reset_all(self) -> None:
        for i, node in enumerate(self.nodes):
            node.rotation = self.rest_rotations[i].copy()
        self.update_world()

    def pose(self) -> dict[str, list[float]]:
        """Every joint whose rotation differs from rest, as XYZW lists."""
        out: dict[str, list[float]] = {}
        for name, index in self.by_name.items():
            node = self.nodes[index]
            if not np.allclose(node.rotation, self.rest_rotations[index], atol=1e-7):
                out[name] = [float(v) for v in node.rotation]
        return out

    # -- measurement -------------------------------------------------------

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World-space AABB over every mesh, at the current pose.

        Computed from the eight corners of each primitive's own box rather
        than from every vertex: a 443k-vertex transform per frame would be the
        most expensive thing in the viewer, and framing does not need to be
        tighter than the box.
        """
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for node, prims in self.mesh_instances():
            for prim in prims:
                if len(prim.positions) == 0:
                    continue
                pmin = prim.positions.min(axis=0)
                pmax = prim.positions.max(axis=0)
                corners = np.array(
                    [
                        [x, y, z]
                        for x in (pmin[0], pmax[0])
                        for y in (pmin[1], pmax[1])
                        for z in (pmin[2], pmax[2])
                    ],
                    dtype="f8",
                )
                world = (node.world @ np.hstack([corners, np.ones((8, 1))]).T).T[:, :3]
                lo = np.minimum(lo, world.min(axis=0))
                hi = np.maximum(hi, world.max(axis=0))
        if not np.isfinite(lo).all():
            return m3.vec3(), m3.vec3()
        return lo, hi

    @property
    def triangle_count(self) -> int:
        return sum(len(p.indices) // 3 for prims in self.meshes for p in prims)


# --- loading ----------------------------------------------------------------


def load(path: Path | bytes) -> Model:
    gltf, buffer = read_glb(path)
    reader = _Reader(gltf, buffer)
    materials = [reader.material(m) for m in gltf.get("materials", [])]
    meshes = [
        [reader.primitive(p, materials) for p in mesh.get("primitives", [])]
        for mesh in gltf.get("meshes", [])
    ]
    skins = [reader.skin(s) for s in gltf.get("skins", [])]
    nodes = [reader.node(n) for n in gltf.get("nodes", [])]
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    roots = list(scene.get("nodes", range(len(nodes))))
    return Model(nodes, roots, meshes, skins)


class _Reader:
    def __init__(self, gltf: dict, buffer: bytes) -> None:
        self.gltf = gltf
        self.buffer = buffer

    # -- accessors ---------------------------------------------------------

    def accessor(self, index: int) -> np.ndarray:
        acc = self.gltf["accessors"][index]
        if "sparse" in acc:
            # Nothing in this pipeline emits one, and silently dropping the
            # overrides would render a subtly wrong mesh rather than fail.
            raise ValueError("sparse accessors are not supported")
        dtype = _COMPONENT[acc["componentType"]]
        ncomp = _NCOMP[acc["type"]]
        count = acc["count"]
        if "bufferView" not in acc:
            # Legal glTF: an accessor with no view reads as zeros.
            return np.zeros((count, ncomp), dtype=dtype)
        view = self.gltf["bufferViews"][acc["bufferView"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        item = np.dtype(dtype).itemsize * ncomp
        stride = view.get("byteStride") or item
        if stride == item:
            flat = np.frombuffer(self.buffer, dtype=dtype, count=count * ncomp, offset=start)
            return flat.reshape(count, ncomp)
        # Interleaved: take the raw bytes and slice the rows out. Nothing this
        # pipeline writes is interleaved, but a hand-supplied GLB may be.
        raw = np.frombuffer(self.buffer, dtype=np.uint8, count=stride * count, offset=start)
        rows = raw.reshape(count, stride)[:, :item]
        return np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)

    # -- pieces ------------------------------------------------------------

    def primitive(self, prim: dict, materials: list[Material]) -> Primitive:
        attrs = prim.get("attributes", {})
        if "POSITION" not in attrs:
            raise ValueError("a primitive with no POSITION is not renderable")
        if prim.get("mode", 4) != 4:
            raise ValueError(f"unsupported primitive mode {prim.get('mode')}")
        positions = self.accessor(attrs["POSITION"]).astype("f4")
        if "indices" in prim:
            indices = self.accessor(prim["indices"]).reshape(-1).astype("u4")
        else:
            indices = np.arange(len(positions), dtype="u4")
        out = Primitive(positions=positions, indices=indices)
        if "NORMAL" in attrs:
            out.normals = self.accessor(attrs["NORMAL"]).astype("f4")
        if "TEXCOORD_0" in attrs:
            out.uvs = self.accessor(attrs["TEXCOORD_0"]).astype("f4")
        if "JOINTS_0" in attrs:
            out.joints = self.accessor(attrs["JOINTS_0"]).astype("i4")
        if "WEIGHTS_0" in attrs:
            raw = self.accessor(attrs["WEIGHTS_0"])
            # glTF allows weights as normalized ubyte/ushort as well as float.
            # Reading the integer forms as-is would give every vertex a weight
            # of 65535, which renders as an explosion rather than as a mesh.
            if raw.dtype == np.uint8:
                out.weights = raw.astype("f4") / 255.0
            elif raw.dtype == np.uint16:
                out.weights = raw.astype("f4") / 65535.0
            else:
                out.weights = raw.astype("f4")
        if "material" in prim and prim["material"] < len(materials):
            out.material = materials[prim["material"]]
        return out

    def material(self, mat: dict) -> Material:
        pbr = mat.get("pbrMetallicRoughness", {})
        out = Material(
            name=mat.get("name", ""),
            base_color_factor=tuple(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))),
            metallic_factor=float(pbr.get("metallicFactor", 1.0)),
            roughness_factor=float(pbr.get("roughnessFactor", 1.0)),
            emissive_factor=tuple(mat.get("emissiveFactor", (0.0, 0.0, 0.0))),
            double_sided=bool(mat.get("doubleSided", False)),
            alpha_mode=mat.get("alphaMode", "OPAQUE"),
            alpha_cutoff=float(mat.get("alphaCutoff", 0.5)),
        )
        out.base_color = self.texture(pbr.get("baseColorTexture"))
        out.metallic_roughness = self.texture(pbr.get("metallicRoughnessTexture"))
        out.normal = self.texture(mat.get("normalTexture"))
        out.emissive = self.texture(mat.get("emissiveTexture"))
        out.occlusion = self.texture(mat.get("occlusionTexture"))
        return out

    def texture(self, ref: dict | None) -> tuple[int, int, bytes] | None:
        if not ref:
            return None
        from PIL import Image

        tex = self.gltf.get("textures", [])[ref["index"]]
        image = self.gltf.get("images", [])[tex["source"]]
        if "bufferView" not in image:
            # An external URI would be a runtime file read of a path from
            # inside the asset; every GLB this app produces is self-contained.
            log.warning("skipping a texture stored outside the GLB")
            return None
        view = self.gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        data = self.buffer[start : start + view["byteLength"]]
        with Image.open(io.BytesIO(data)) as im:
            rgba = im.convert("RGBA")
            return rgba.width, rgba.height, rgba.tobytes()

    def skin(self, skin: dict) -> Skin:
        joints = list(skin["joints"])
        if "inverseBindMatrices" in skin:
            # glTF stores each matrix column-major; ours are M @ v with the
            # translation in the last column, so every one is transposed here
            # and nowhere else.
            raw = self.accessor(skin["inverseBindMatrices"]).astype("f8")
            ibm = raw.reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            ibm = np.tile(np.eye(4), (len(joints), 1, 1))
        return Skin(joints=joints, inverse_bind=ibm)

    def node(self, node: dict) -> Node:
        out = Node(
            name=node.get("name", ""),
            children=list(node.get("children", [])),
            mesh=node.get("mesh"),
            skin=node.get("skin"),
        )
        if "matrix" in node:
            # A node gives either a matrix or TRS, never both.
            mat = np.array(node["matrix"], dtype="f8").reshape(4, 4).T
            out.translation, out.rotation, out.scale = m3.decompose(mat)
        else:
            out.translation = np.array(node.get("translation", (0.0, 0.0, 0.0)), dtype="f8")
            out.rotation = np.array(node.get("rotation", (0.0, 0.0, 0.0, 1.0)), dtype="f8")
            out.scale = np.array(node.get("scale", (1.0, 1.0, 1.0)), dtype="f8")
        return out

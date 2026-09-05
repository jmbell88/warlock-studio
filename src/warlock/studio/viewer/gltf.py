"""A GLB loader, in numpy and Pillow.

Hand-rolled rather than delegated. trimesh discards a scene root's transform --
the constraint ``normalize_glb`` writes around, and one this loader is under no
obligation to inherit -- and it has no notion of a skin at all. pygltflib parses
the JSON but still leaves accessor decoding here. What is left after those two
is small enough to own.

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

#: Ceilings on what a hand-supplied file may declare. ``check_glb`` at the
#: import door is deliberately structural-only, so a ≤100 MB GLB reaches this
#: loader with whatever numbers its JSON chunk claims -- and this loader runs on
#: the frame thread. Both are module-level so a test can lower them rather than
#: craft a hostile asset.
#:
#: A JSON chunk declaring millions of nodes is a hang, not a scene.
MAX_NODES = 100_000
#: Mirrors ``service.validation.MAX_IMAGE_PIXELS`` without importing service
#: into the viewer (the viewer imports no business-logic layer).
MAX_TEXTURE_PIXELS = 16_000_000
#: What one accessor may decode to. Every *other* accessor path is bounded by
#: the BIN chunk it reads out of -- ``_check_span`` refuses a span past the end
#: of it, so a 400-byte GLB cannot ask for more than 400 bytes of geometry. The
#: no-``bufferView`` path is the one that allocates without touching the buffer
#: at all, so nothing downstream of it ever gets a turn: ``count:
#: 1_000_000_000, type: "MAT4"`` in a file that fits in a packet asks numpy for
#: 64 GB. 256 MiB is two orders past the largest accessor this pipeline
#: produces (a 2 M-triangle mesh's index stream is 24 MB) and four orders short
#: of what the field can express.
MAX_ACCESSOR_BYTES = 1 << 28

#: What one *document* may decode to, across every accessor and texture
#: combined. MAX_ACCESSOR_BYTES bounds a single array; nothing bounded the
#: *sum* of them, so a file with many primitives -- or one primitive's
#: accessor replayed by many nodes -- allocated without limit as long as each
#: individual array stayed under the per-accessor ceiling (H01). Measured: a
#: 4,036-byte bufferless GLB with 128 primitives produced 6,144,000 bytes of
#: geometry across 128 separate arrays, and nothing stopped scaling the
#: primitive count further. 768 MiB is comfortably above the largest asset
#: this pipeline produces (MAX_ACCESSOR_BYTES's own docstring: a 2 M-triangle
#: index stream is 24 MB; five fully populated 16-megapixel texture slots are
#: another ~320 MB) while still refusing a file that keeps asking for more.
MAX_TOTAL_BYTES = 768 * (1 << 20)


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
    #: The primitive's own axis-aligned box in *its* space, computed once.
    #: ``Model.bounds`` runs per frame for the inspector and for framing, and
    #: its docstring's claim that it does not touch every vertex was only true
    #: of the transform: the ``min``/``max`` that make the box were a full pass
    #: over the positions, every frame, on 443k vertices. Positions never move
    #: -- a pose moves the *node* -- so the box is a property of the primitive.
    _box: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )

    def box(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(min, max)`` over this primitive's positions, or None if empty."""

        if self._box is None:
            if len(self.positions) == 0:
                return None
            self._box = (self.positions.min(axis=0), self.positions.max(axis=0))
        return self._box


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
        skipped_textures: int = 0,
    ) -> None:
        self.nodes = nodes
        self.roots = roots
        self.meshes = meshes
        self.skins = skins
        # How many of this file's images the loader could not use (D42). The
        # stated policy is that a texture is a cosmetic loss and never a reason
        # to refuse a file -- which is right, and left the loss reported only in
        # the log, so an untextured-looking mesh was indistinguishable from a
        # mesh that was never textured. This is the count the UI says it with.
        self.skipped_textures = skipped_textures
        # A joint node is addressed by name everywhere above this layer: a
        # pose is a bone->rotation map, and the browser never saw an index.
        self.by_name: dict[str, int] = {}
        for i, node in enumerate(nodes):
            # First wins: a duplicate name is a broken rig either way, and
            # picking the later one would silently move a different joint.
            if node.name and node.name not in self.by_name:
                self.by_name[node.name] = i
        self.rest_rotations = [n.rotation.copy() for n in nodes]
        # The mirror of rest_rotations, for the one translation posing may
        # move: the root joint, when the Poser previews a root offset as
        # ``rest + delta``. Remembered for every node because which one is the
        # root is the editor's knowledge, not the file's.
        self.rest_translations = [n.translation.copy() for n in nodes]
        self.update_world()

    # -- transforms --------------------------------------------------------

    def update_world(self) -> None:
        """Recompute every node's world matrix from the roots down.

        Called once at load and again after every pose change. Iterative
        rather than recursive: a skeleton is shallow but a mesh hierarchy from
        an exporter need not be, and a blown stack in the frame loop is not a
        failure mode worth having.
        """
        # ``seen`` and the range check are not defensiveness about our own
        # exporter: a hand-supplied GLB whose children form a cycle, or name a
        # node index that does not exist, is reachable through import (which is
        # deliberately structural-only) and would spin or raise here -- on the
        # frame thread. A malformed graph costs the malformed part of itself.
        seen: set[int] = set()
        stack = [(r, m3.identity()) for r in reversed(self.roots)]
        while stack:
            index, parent = stack.pop()
            if index in seen or not 0 <= index < len(self.nodes):
                continue
            seen.add(index)
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
                local = prim.box()
                if local is None:
                    continue
                pmin, pmax = local
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

    @property
    def vertex_count(self) -> int:
        """Total vertices, computed once (B18): the primitives are immutable
        after load and the inspector asks for this every frame."""
        count = getattr(self, "_vertex_count", None)
        if count is None:
            count = sum(len(p.positions) for prims in self.meshes for p in prims)
            self._vertex_count = count
        return count


# --- loading ----------------------------------------------------------------


#: Extensions this loader implements. Anything a file lists as *required*
#: beyond these changes what its bytes mean, so it is refused rather than
#: decoded as though the extension were absent.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset()


def load(path: Path | bytes) -> Model:
    gltf, buffer = read_glb(path)
    # Refused rather than misread. KHR_mesh_quantization is the one that will
    # arrive first -- gltfpack -c writes it -- and a quantized position stream
    # decoded as though it were plain floats is geometry that looks like
    # nothing, with nothing in the data to say why.
    required = set(gltf.get("extensionsRequired") or []) - SUPPORTED_EXTENSIONS
    if required:
        raise ValueError(
            "this GLB requires glTF extensions this viewer does not implement: "
            + ", ".join(sorted(required))
        )
    declared = len(gltf.get("nodes", []))
    if declared > MAX_NODES:
        raise ValueError(f"this GLB declares {declared} nodes, more than this viewer will load")
    reader = _Reader(gltf, buffer)
    materials = [reader.material(m) for m in gltf.get("materials", [])]
    meshes = [
        [reader.primitive(p, materials) for p in mesh.get("primitives", [])]
        for mesh in gltf.get("meshes", [])
    ]
    skins = [reader.skin(s) for s in gltf.get("skins", [])]
    nodes = [reader.node(n) for n in gltf.get("nodes", [])]
    return Model(
        nodes, _roots(gltf, nodes), meshes, skins, skipped_textures=reader.skipped
    )


def _roots(gltf: dict, nodes: list[Node]) -> list[int]:
    """The scene's root nodes, or the ones nobody parents.

    The fallback matters more than it looks. Taking *every* node as a root
    makes ``update_world`` visit each child twice: once correctly under its
    parent, and again later as a root with an identity parent, which overwrites
    the world matrix it just computed. Every node ends up with world == local,
    so anything parented renders in the wrong place and ``joint_palette``
    inverts a matrix that was never right.
    """
    scenes = gltf.get("scenes") or []
    index = gltf.get("scene", 0)
    if scenes and 0 <= index < len(scenes) and "nodes" in scenes[index]:
        return list(scenes[index]["nodes"])
    parented = {child for node in nodes for child in node.children}
    return [i for i in range(len(nodes)) if i not in parented]


class _Reader:
    def __init__(self, gltf: dict, buffer: bytes) -> None:
        self.gltf = gltf
        self.buffer = buffer
        # Decoded pixels per glTF image *source* index (D39). Several
        # materials routinely reference one atlas, and the PNG decode is the
        # dominant cost of parse_model -- so each image is decoded once and
        # the same (w, h, bytes) tuple is shared, which is also what lets the
        # GPU side de-duplicate its uploads by buffer identity.
        self._images: dict[int, tuple[int, int, bytes] | None] = {}
        # Images this file carries that could not be used (D42). Counted per
        # *source*, not per material reference, because ``_images`` memoizes:
        # one unreadable atlas shared by six primitives is one loss, and
        # counting references would report six.
        self.skipped = 0
        # Decoded accessor arrays, by accessor index (H01). Two or more
        # primitives naming the same accessor -- an instanced mesh's shared
        # POSITION stream is the ordinary case -- used to decode it once per
        # reference; caching makes a repeated reference free instead of a
        # repeated allocation, the same way ``_images`` already does for
        # textures.
        self._accessors: dict[int, np.ndarray] = {}
        # The dtype-converted form of an accessor -- ``positions.astype("f4")``
        # and friends -- shared the same way, and for the same reason (H01):
        # see ``_typed``.
        self._typed_cache: dict[Any, np.ndarray] = {}
        # Running total this document has allocated, against MAX_TOTAL_BYTES.
        self._spent = 0

    # -- accessors ---------------------------------------------------------

    def _charge(self, nbytes: int) -> None:
        """Add ``nbytes`` to this document's running total, refusing once the
        shared ceiling (H01) is crossed.

        Called *before* the allocation it describes is handed back, not after
        it lands in a caller's hands -- the whole point of a document-wide
        budget is that the refusal happens instead of the bytes, not
        alongside them.
        """
        self._spent += nbytes
        if self._spent > MAX_TOTAL_BYTES:
            raise ValueError(
                f"this GLB's decoded geometry and textures pass the "
                f"{MAX_TOTAL_BYTES:,} byte budget this viewer holds open at once"
            )

    def accessor(self, index: int) -> np.ndarray:
        cached = self._accessors.get(index)
        if cached is not None:
            # Already charged and decoded once; a second primitive naming the
            # same accessor gets the same array rather than a second
            # allocation (H01). Safe to share: nothing downstream mutates an
            # accessor's array in place, only reads or copies it.
            return cached
        out = self._decode_accessor(index)
        self._accessors[index] = out
        return out

    def _decode_accessor(self, index: int) -> np.ndarray:
        acc = self.gltf["accessors"][index]
        if "sparse" in acc:
            # Nothing in this pipeline emits one, and silently dropping the
            # overrides would render a subtly wrong mesh rather than fail.
            raise ValueError("sparse accessors are not supported")
        dtype = _COMPONENT[acc["componentType"]]
        ncomp = _NCOMP[acc["type"]]
        count = int(acc["count"])
        # Before the branch and not inside it, so the bound is a property of
        # *reading an accessor* rather than a rule the zeros path below had to
        # remember. The interleaved path pays for it twice over -- ``_check_span``
        # will refuse it again against the real buffer -- and that is the point:
        # the branch that has no buffer to be checked against is the one that
        # was allocating from a number nothing had looked at.
        if count < 0 or count * ncomp * np.dtype(dtype).itemsize > MAX_ACCESSOR_BYTES:
            raise ValueError(
                f"an accessor in this GLB declares {count} {acc['type']} elements,"
                f" which is more than the {MAX_ACCESSOR_BYTES} bytes this viewer"
                " will allocate for one"
            )
        if "bufferView" not in acc:
            # Legal glTF: an accessor with no view reads as zeros -- and the
            # branch H01 was written for: nothing here reads the buffer, so
            # nothing about *this* accessor's cost is bounded by how small the
            # file is. Charged like every other allocation below.
            self._charge(count * ncomp * np.dtype(dtype).itemsize)
            return np.zeros((count, ncomp), dtype=dtype)
        view = self.gltf["bufferViews"][acc["bufferView"]]
        self._check_buffer(view)
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        item = np.dtype(dtype).itemsize * ncomp
        stride = view.get("byteStride") or item
        if stride == item:
            self._check_span(start, count * item)
            self._charge(count * item)
            flat = np.frombuffer(self.buffer, dtype=dtype, count=count * ncomp, offset=start)
            return flat.reshape(count, ncomp)
        # Interleaved: take the raw bytes and gather the rows out. Nothing this
        # pipeline writes is interleaved, but a hand-supplied GLB may be.
        #
        # The span is stride*(count-1) + item, not stride*count: the spec only
        # requires the *elements* to be present, so the padding after the last
        # one need not exist. Demanding it made an accessor sitting at the tail
        # of the BIN chunk raise "buffer is smaller than requested size" -- the
        # model simply failed to load.
        span = stride * (count - 1) + item if count else 0
        self._check_span(start, span)
        # Two real copies land here -- the fancy-indexed ``rows`` and the
        # ``ascontiguousarray`` beneath it -- on top of the raw span this reads
        # out of the buffer, which is exactly the doubling H01's budget exists
        # to account for rather than charge for the final array alone.
        self._charge(span + 2 * count * item)
        raw = np.frombuffer(self.buffer, dtype=np.uint8, count=span, offset=start)
        rows = raw[np.arange(count)[:, None] * stride + np.arange(item)[None, :]]
        return np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)

    def decoded(self, index: int) -> np.ndarray:
        """An accessor with its ``normalized`` flag honoured.

        The flag says the integers encode a float in 0..1 (or -1..1 signed), so
        reading them raw gives a UV of 65535 or a colour of 255 -- geometry and
        materials that are silently, enormously wrong rather than broken. The
        skin-weight path below already did this by hand for the one case that
        turned up in practice; this is the same rule for every attribute.
        """
        acc = self.gltf["accessors"][index]
        raw = self.accessor(index)
        if not acc.get("normalized") or raw.dtype.kind not in "iu":
            return raw
        info = np.iinfo(raw.dtype)
        if raw.dtype.kind == "u":
            return raw.astype("f4") / float(info.max)
        # Signed: the spec's own formula, which clamps -128 and -32768 to -1.
        return np.maximum(raw.astype("f4") / float(info.max), -1.0)

    def _check_buffer(self, view: dict) -> None:
        """Refuse a view that points at a buffer we do not have.

        ``read_glb`` returns the GLB's single BIN chunk, which is buffer 0. A
        view naming buffer 1 (a data URI, or an external .bin) was read at the
        same offsets *into buffer 0* -- silently wrong geometry rather than an
        error, which is the worst way for this to fail.
        """
        if view.get("buffer", 0) != 0:
            raise ValueError(
                "this GLB stores geometry outside its binary chunk, which is not supported"
            )

    def _check_span(self, start: int, span: int) -> None:
        if start < 0 or start + span > len(self.buffer):
            raise ValueError("an accessor reads past the end of the binary chunk")

    def _astype(self, arr: np.ndarray, dtype: str) -> np.ndarray:
        """``arr.astype(dtype)``, charged against the document budget.

        ``astype`` copies even when the requested dtype already matches, so
        every conversion below -- positions to ``f4``, indices to ``u4``, the
        weight/joint promotions -- is a *second* array on top of whatever
        ``accessor``/``decoded`` already charged for the first one (H01).
        Charging only the final accessor size, as the per-accessor ceiling
        does, undercounted exactly this doubling.
        """
        out = arr.astype(dtype)
        self._charge(out.nbytes)
        return out

    def _typed(self, key: Any, raw: np.ndarray, dtype: str) -> np.ndarray:
        """A converted array, shared by every primitive asking for the same
        ``key`` (H01).

        Caching ``accessor()``'s *raw* output already stops a repeated
        reference from decoding twice; this closes the gap one layer up.
        Every attribute here still runs its own ``.astype`` per primitive
        even once the raw accessor is cached, because each call built a fresh
        copy of the *converted* array too -- an instanced mesh with a
        thousand nodes sharing one glTF mesh definition paid for a thousand
        ``positions.astype("f4")`` copies of an identical result. Safe to
        share: nothing downstream mutates a primitive's positions, indices,
        normals, uvs or joints in place (unlike weights, renormalised right
        below -- deliberately left out of this cache).
        """
        cached = self._typed_cache.get(key)
        if cached is not None:
            return cached
        out = self._astype(raw, dtype)
        self._typed_cache[key] = out
        return out

    # -- pieces ------------------------------------------------------------

    def primitive(self, prim: dict, materials: list[Material]) -> Primitive:
        attrs = prim.get("attributes", {})
        if "POSITION" not in attrs:
            raise ValueError("a primitive with no POSITION is not renderable")
        if prim.get("mode", 4) != 4:
            raise ValueError(f"unsupported primitive mode {prim.get('mode')}")
        positions = self._typed(
            ("POSITION", attrs["POSITION"]), self.decoded(attrs["POSITION"]), "f4"
        )
        if "indices" in prim:
            # Indices are never normalized -- they are indices -- so they take
            # the raw path deliberately.
            indices = self._typed(
                ("indices", prim["indices"]),
                self.accessor(prim["indices"]).reshape(-1),
                "u4",
            )
        else:
            # Synthesised rather than read off the buffer, but no less real an
            # allocation -- and the one H01 names explicitly: a primitive with
            # no ``indices`` used to get this array for free regardless of how
            # large ``positions`` was. Cached on the length rather than an
            # accessor index -- there is no accessor to key on -- which still
            # dedupes the ordinary case of several instances of one unindexed
            # mesh.
            key = ("arange", len(positions))
            indices = self._typed_cache.get(key)
            if indices is None:
                indices = np.arange(len(positions), dtype="u4")
                self._charge(indices.nbytes)
                self._typed_cache[key] = indices
        out = Primitive(positions=positions, indices=indices)
        if "NORMAL" in attrs:
            out.normals = self._typed(
                ("NORMAL", attrs["NORMAL"]), self.decoded(attrs["NORMAL"]), "f4"
            )
        if "TEXCOORD_0" in attrs:
            out.uvs = self._typed(
                ("TEXCOORD_0", attrs["TEXCOORD_0"]), self.decoded(attrs["TEXCOORD_0"]), "f4"
            )
        if "JOINTS_0" in attrs:
            out.joints = self._typed(
                ("JOINTS_0", attrs["JOINTS_0"]), self.accessor(attrs["JOINTS_0"]), "i4"
            )
        if "WEIGHTS_0" in attrs:
            raw = self.accessor(attrs["WEIGHTS_0"])
            # glTF allows weights as normalized ubyte/ushort as well as float.
            # Reading the integer forms as-is would give every vertex a weight
            # of 65535, which renders as an explosion rather than as a mesh.
            if raw.dtype == np.uint8:
                weights = self._astype(raw, "f4") / 255.0
            elif raw.dtype == np.uint16:
                weights = self._astype(raw, "f4") / 65535.0
            else:
                weights = self._astype(raw, "f4")
            # **Renormalised, and a zero-sum vertex pinned to its first joint.**
            # The shader sums ``u_joints[j] * w`` over the four influences with
            # no division, so the spec's "the weights of a vertex sum to 1" is
            # a requirement this renderer *relies* on rather than one it
            # checks. A file whose weights sum to 0 -- a stray vertex an
            # exporter left unweighted, which is common enough in rigs that
            # come back from a round trip -- collapsed every such vertex onto
            # the origin, and the mesh grew a spike to the world centre.
            # Renormalising is exact for a well-formed file (a sum of 1 divides
            # by 1) and is the only reading available for a malformed one.
            total = weights.sum(axis=1, keepdims=True)
            dead = total[:, 0] <= 0.0
            if dead.any():
                weights[dead] = 0.0
                weights[dead, 0] = 1.0
                total = weights.sum(axis=1, keepdims=True)
            out.weights = weights / total
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

    def _image_bytes(self, image: dict) -> bytes | None:
        """The encoded pixels for one glTF image, or None if unreachable.

        Three cases, and the old code collapsed them into one warning that was
        false for two of them. A bufferView is the normal path. A ``data:`` URI
        is equally self-contained -- it is *inside* the file, it just is not in
        the binary chunk -- and dropping it lost a texture the asset carried
        while reporting it as "stored outside the GLB". Only a genuine external
        URI is a runtime file read of a path from inside the asset, which is
        the thing actually worth refusing.

        Every unreachable case *returns* here rather than raising, unlike the
        geometry path. A mesh read from the wrong buffer is a plausible-looking
        wrong answer and worth failing the load over; a texture that cannot be
        found is a cosmetic loss, and raising turned any third-party GLB whose
        images sit in a second buffer into "this file will not open".
        """
        if "bufferView" in image:
            view = self.gltf["bufferViews"][image["bufferView"]]
            try:
                self._check_buffer(view)
            except Exception as exc:
                log.warning("skipping a texture in an unreachable buffer: %s", exc)
                self.skipped += 1
                return None
            start = view.get("byteOffset", 0)
            return self.buffer[start : start + view["byteLength"]]
        uri = str(image.get("uri") or "")
        if uri.startswith("data:"):
            import base64

            _, _, payload = uri.partition(",")
            try:
                return base64.b64decode(payload)
            except Exception:
                log.warning("a texture's embedded data URI could not be decoded")
                self.skipped += 1
                return None
        log.warning("skipping a texture stored in a separate file (%s)", uri or "no uri")
        self.skipped += 1
        return None

    def texture(self, ref: dict | None) -> tuple[int, int, bytes] | None:
        if not ref:
            return None
        from PIL import Image

        tex = self.gltf.get("textures", [])[ref["index"]]
        source = tex["source"]
        if source in self._images:
            return self._images[source]
        image = self.gltf.get("images", [])[source]
        data = self._image_bytes(image)
        if data is None:
            self._images[source] = None
            return None
        try:
            with Image.open(io.BytesIO(data)) as im:
                # ``open`` is lazy, so the size is known before any pixel is
                # decoded: an absurd one costs a log line rather than the RAM.
                # Same policy as a corrupt map -- the texture, not the model.
                if im.width * im.height > MAX_TEXTURE_PIXELS:
                    log.warning(
                        "skipping a %dx%d texture: over this viewer's %d-pixel ceiling",
                        im.width,
                        im.height,
                        MAX_TEXTURE_PIXELS,
                    )
                    self.skipped += 1
                    self._images[source] = None
                    return None
                # Charged into the *document-wide* budget (H01) before the
                # decode it describes, and deliberately outside the
                # cosmetic-loss ``except`` below: an over-budget document is a
                # refusal, unlike one corrupt map, because this is bounding
                # the sum of every texture the file carries rather than
                # judging any one of them.
                self._charge(im.width * im.height * 4)
                rgba = im.convert("RGBA")
                decoded = rgba.width, rgba.height, rgba.tobytes()
        except ValueError:
            raise
        except Exception:
            # The stated policy for images, applied to the decode as well as to
            # the lookup: a texture that cannot be read is a cosmetic loss, and
            # raising here turned one corrupt map into "this file will not
            # open" -- for a mesh that is otherwise entirely intact.
            log.warning("skipping a texture whose image data could not be decoded")
            self.skipped += 1
            decoded = None
        self._images[source] = decoded
        return decoded

    def skin(self, skin: dict) -> Skin:
        joints = list(skin["joints"])
        if "inverseBindMatrices" in skin:
            # glTF stores each matrix column-major; ours are M @ v with the
            # translation in the last column, so every one is transposed here
            # and nowhere else.
            raw = self._astype(self.accessor(skin["inverseBindMatrices"]), "f8")
            ibm = raw.reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            ibm = np.tile(np.eye(4), (len(joints), 1, 1))
        return Skin(joints=joints, inverse_bind=ibm)

    def node(self, node: dict) -> Node:
        # Bounds-checked against the file's own declared counts, not decoded
        # length -- both are read from ``self.gltf`` before any node is built,
        # so this holds regardless of load()'s decode order. The 2026-09-05
        # audit, finding create-01: an out-of-range mesh/skin index used to
        # reach ``GpuModel.__init__`` (scene.py) as a bare ``IndexError`` --
        # *after* real ``ctx.buffer``/``ctx.texture`` objects had already been
        # allocated for earlier, valid nodes in the same document, which
        # leaked them forever (this app sets no moderngl ``gc_mode``: a
        # dropped reference frees nothing). Refusing here, on the task thread
        # inside ``load()``, means no GPU resource is ever created for a file
        # that will not finish loading -- the same ceiling ``prim["material"]``
        # already gets a few lines below.
        mesh = node.get("mesh")
        if mesh is not None:
            n_meshes = len(self.gltf.get("meshes", []))
            if not 0 <= mesh < n_meshes:
                raise ValueError(
                    f"node {node.get('name', '') or '<unnamed>'!r} references mesh "
                    f"{mesh}, but this GLB declares {n_meshes} mesh(es)"
                )
        skin = node.get("skin")
        if skin is not None:
            n_skins = len(self.gltf.get("skins", []))
            if not 0 <= skin < n_skins:
                raise ValueError(
                    f"node {node.get('name', '') or '<unnamed>'!r} references skin "
                    f"{skin}, but this GLB declares {n_skins} skin(s)"
                )
        out = Node(
            name=node.get("name", ""),
            children=list(node.get("children", [])),
            mesh=mesh,
            skin=skin,
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

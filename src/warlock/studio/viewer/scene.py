"""GPU resources for a loaded model: buffers, textures, vertex arrays.

A :class:`GpuModel` owns everything that has to be released when a model is
swapped out, and holds a reference to the CPU-side :class:`~.gltf.Model` it was
built from -- posing mutates that graph and only the joint palette is recomputed
on the GPU side, so the two are deliberately not independent copies.
"""

from __future__ import annotations

import logging
from typing import Any

import moderngl
import numpy as np

from . import math3d as m3
from .gltf import Material, Model, Node, Primitive
from .programs import MAX_JOINTS

log = logging.getLogger(__name__)

# Slot -> (define, uniform, is_srgb). Mirrors postprocess._TEXTURE_SLOTS: the
# meshes this app makes only ever carry the first two, and the rest are here so
# a hand-supplied GLB is not silently flattened.
TEXTURE_SLOTS = (
    ("base_color", "HAS_BASE_COLOR_MAP", "u_base_color_map", True),
    ("metallic_roughness", "HAS_MR_MAP", "u_mr_map", False),
    ("normal", "HAS_NORMAL_MAP", "u_normal_map", False),
    ("emissive", "HAS_EMISSIVE_MAP", "u_emissive_map", True),
    ("occlusion", "HAS_AO_MAP", "u_ao_map", False),
)


class GpuMaterial:
    """Uploaded textures plus the factors that go with them.

    sRGB-ness is *not* handled by an sRGB texture format: the fragment shader
    decodes base colour and emissive itself, so a texture is a plain RGBA8
    either way and the shader's ``srgbToLinear`` is the single place the
    convention lives.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        material: Material,
        texture_cache: dict[int, moderngl.Texture] | None = None,
    ) -> None:
        """``texture_cache`` de-duplicates uploads by decoded buffer (D40):
        keyed on ``id(pixels)``, which is sound because the loader decodes
        each glTF image source once and shares the bytes object, and the
        Model keeps those bytes alive for as long as this material exists. A
        texture found in the cache is *borrowed* -- only the creator releases
        it -- so two materials over one atlas cost one upload and one free.
        """
        self.material = material
        self.textures: dict[str, moderngl.Texture] = {}
        self._owned: list[moderngl.Texture] = []
        self.defines: list[str] = []
        for slot, define, _uniform, _srgb in TEXTURE_SLOTS:
            data = getattr(material, slot)
            if data is None:
                continue
            width, height, pixels = data
            texture = None if texture_cache is None else texture_cache.get(id(pixels))
            if texture is None:
                texture = ctx.texture((width, height), 4, pixels)
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                texture.build_mipmaps()
                texture.anisotropy = min(8.0, ctx.max_anisotropy)
                self._owned.append(texture)
                if texture_cache is not None:
                    texture_cache[id(pixels)] = texture
            self.textures[slot] = texture
            self.defines.append(define)

    def bind(self, program: Any) -> None:
        unit = 0
        for slot, _define, uniform, _srgb in TEXTURE_SLOTS:
            texture = self.textures.get(slot)
            if texture is None or uniform not in program:
                continue
            texture.use(unit)
            program[uniform].value = unit
            unit += 1
        mat = self.material
        if "u_base_color_factor" in program:
            program["u_base_color_factor"].value = tuple(mat.base_color_factor)
        if "u_metallic" in program:
            program["u_metallic"].value = mat.metallic_factor
            program["u_roughness"].value = mat.roughness_factor
            program["u_emissive_factor"].value = tuple(mat.emissive_factor)
            program["u_alpha_cutoff"].value = mat.alpha_cutoff
            # 0 opaque, 1 masked, 2 blended -- the shader only distinguishes
            # "cut out at the threshold" from "ignore alpha entirely", and
            # BLEND is the one mode where the sampled alpha is kept as-is.
            program["u_alpha_mask"].value = {"MASK": 1, "BLEND": 2}.get(mat.alpha_mode, 0)

    def release(self) -> None:
        # Only what this material created: a texture borrowed from the shared
        # cache belongs to the material that uploaded it, and a double release
        # frees a GL name the driver may already have reissued.
        for texture in self._owned:
            texture.release()
        self._owned.clear()
        self.textures.clear()


class GpuPrimitive:
    """One drawable primitive: an interleaved buffer and a vertex array per program.

    Interleaved rather than one buffer per attribute: it is a single upload and
    a single binding, and nothing here ever updates one attribute alone (a pose
    changes the joint palette, not the vertices).
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        primitive: Primitive,
        material: GpuMaterial,
        skinned: bool,
    ) -> None:
        self.ctx = ctx
        self.primitive = primitive
        self.material = material
        self.skinned = skinned and primitive.joints is not None
        count = len(primitive.positions)

        normals = primitive.normals
        if normals is None:
            normals = _face_normals(primitive)
        uvs = primitive.uvs if primitive.uvs is not None else np.zeros((count, 2), "f4")

        columns = [primitive.positions, normals, uvs]
        # (format, attribute, byte width). The width is what a *missing*
        # attribute costs: GLSL strips anything a program does not actually
        # read -- an untextured mesh's shader never mentions a_uv -- so the
        # layout has to become padding rather than name a location that no
        # longer exists.
        self._parts = [("3f", "a_position", 12), ("3f", "a_normal", 12), ("2f", "a_uv", 8)]
        if self.skinned:
            # int32 joints ride in the same interleaved buffer; moderngl's "4i"
            # binds them with glVertexAttribIPointer, which is what an ivec4
            # needs (a float pointer would silently truncate index 16+).
            columns += [primitive.joints.view("f4"), primitive.weights]
            self._parts += [("4i", "a_joints", 16), ("4f", "a_weights", 16)]

        self.vbo = ctx.buffer(self._interleave(columns, count))
        self.ibo = ctx.buffer(np.ascontiguousarray(primitive.indices, "u4").tobytes())
        # Sorted here, once, so ProgramCache.get can key on the tuple as-is
        # instead of canonicalising it per draw call (B16).
        self.defines = tuple(
            sorted(material.defines + (["SKINNED"] if self.skinned else []))
        )
        self._vaos: dict[int, moderngl.VertexArray] = {}

    @staticmethod
    def _interleave(columns: list[Any], count: int) -> bytes:
        return np.ascontiguousarray(
            np.concatenate(
                [np.ascontiguousarray(c, dtype="f4").reshape(count, -1) for c in columns],
                axis=1,
            )
        ).tobytes()

    def update_vertices(self, primitive: Primitive) -> None:
        """Rewrite the vertex data in place, keeping the buffer and every VAO.

        For a *live* edit that moves vertices without changing topology -- a
        gizmo drag over a mesh selection. Rebuilding the ``GpuPrimitive`` would
        release and recreate the buffer and drop the cached vertex arrays, once
        per frame of the drag, for data that is the same size and the same
        layout as what is already there.

        The layout is not duplicated at the call site: this rebuilds the same
        column list ``__init__`` does, so a future attribute is added in one
        place. It refuses a primitive whose vertex count differs, because that
        is a topology change wearing a movement's clothes and writing it would
        silently truncate.
        """
        count = len(primitive.positions)
        if count != len(self.primitive.positions):
            raise ValueError(
                f"update_vertices needs the same vertex count: had "
                f"{len(self.primitive.positions)}, got {count}"
            )
        normals = primitive.normals
        if normals is None:
            normals = _face_normals(primitive)
        uvs = primitive.uvs if primitive.uvs is not None else np.zeros((count, 2), "f4")
        columns = [primitive.positions, normals, uvs]
        if self.skinned:
            columns += [primitive.joints.view("f4"), primitive.weights]
        self.vbo.write(self._interleave(columns, count))
        self.primitive = primitive

    def vao(self, program: Any) -> moderngl.VertexArray:
        """One vertex array per program, cached: the same primitive is drawn by
        the lit program and by the unlit one, and a VAO is bound to neither."""
        key = id(program)
        vao = self._vaos.get(key)
        if vao is None:
            fields, names = [], []
            for fmt, name, width in self._parts:
                if name in program:
                    fields.append(fmt)
                    names.append(name)
                else:
                    fields.append(f"{width}x")
            vao = self.ctx.vertex_array(
                program,
                [(self.vbo, " ".join(fields), *names)],
                index_buffer=self.ibo,
                index_element_size=4,
            )
            self._vaos[key] = vao
        return vao

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        self._vaos.clear()
        self.vbo.release()
        self.ibo.release()


def _face_normals(primitive: Primitive) -> np.ndarray:
    """Flat normals for a primitive that shipped none.

    Legal glTF and rare here (trellis always writes them), but a mesh with no
    normals shades pure black, which looks like a load failure rather than like
    a missing attribute.
    """
    positions = primitive.positions
    tri = primitive.indices.reshape(-1, 3)
    edge1 = positions[tri[:, 1]] - positions[tri[:, 0]]
    edge2 = positions[tri[:, 2]] - positions[tri[:, 0]]
    face = np.cross(edge1, edge2)
    # Deliberately *not* the ``np.bincount`` scatter ``clay.mesh.accumulate``
    # uses. That one is bit-identical because its accumulator is already f8;
    # here ``positions`` is f4, so the loop sums in single precision and
    # bincount would sum in double and round differently. A rare path (trellis
    # always writes normals) is not worth a changed last bit on the one
    # attribute the shading reads.
    out = np.zeros_like(positions)
    for i in range(3):
        np.add.at(out, tri[:, i], face)
    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, np.where(lengths == 0, 1.0, lengths)).astype("f4")


class GpuModel:
    """Everything on the GPU for one loaded GLB."""

    def __init__(self, ctx: moderngl.Context, model: Model) -> None:
        self.ctx = ctx
        self.model = model
        self.materials: list[GpuMaterial] = []
        self._by_material: dict[int, GpuMaterial] = {}
        # Shared across this model's materials so one decoded buffer is one
        # GPU texture (D40). Lives here so it dies with the model.
        self._texture_cache: dict[int, moderngl.Texture] = {}
        # Per-node normal matrices, keyed on the world matrix's bytes (B15):
        # recomputed only when a pose actually moves the node.
        self._normal_cache: dict[int, tuple[bytes, bytes]] = {}
        self.draws: list[tuple[Node, GpuPrimitive]] = []
        over_budget = False
        for node, primitives in model.mesh_instances():
            skinned = node.skin is not None
            if skinned and len(model.skins[node.skin].joints) > MAX_JOINTS:
                # Drawn unskinned rather than not at all: a mesh in its bind
                # pose is still recognisably the mesh, and every rig this app
                # produces has 20 joints.
                over_budget = True
                skinned = False
            for primitive in primitives:
                material = self._gpu_material(primitive.material)
                self.draws.append((node, GpuPrimitive(ctx, primitive, material, skinned)))
        if over_budget:
            log.warning("a skin exceeds %d joints; drawing it at rest", MAX_JOINTS)
        # Recomputed on every pose change, not per draw call.
        self._palettes: dict[int, bytes] = {}
        self.refresh_palettes()

    def _gpu_material(self, material: Material) -> GpuMaterial:
        key = id(material)
        gpu = self._by_material.get(key)
        if gpu is None:
            gpu = GpuMaterial(self.ctx, material, self._texture_cache)
            self._by_material[key] = gpu
            self.materials.append(gpu)
        return gpu

    def normal_matrix_bytes(self, node: Node, world: np.ndarray) -> bytes:
        """The GL bytes of ``inv(world[:3,:3]).T``, cached per node (B15).

        The identity fallback keeps a zero-scaled node from raising out of
        the draw, exactly as the inline version did.
        """
        world_key = world.tobytes()
        hit = self._normal_cache.get(id(node))
        if hit is not None and hit[0] == world_key:
            return hit[1]
        try:
            normal_matrix = np.linalg.inv(world[:3, :3]).T
        except np.linalg.LinAlgError:
            normal_matrix = np.eye(3)
        out = np.ascontiguousarray(normal_matrix.T, dtype="f4").tobytes()
        self._normal_cache[id(node)] = (world_key, out)
        return out

    def refresh_palettes(self) -> None:
        """Recompute the joint matrices for every skinned node."""
        self._palettes.clear()
        for node, _prim in self.draws:
            if node.skin is None or id(node) in self._palettes:
                continue
            palette = self.model.joint_palette(node)
            if palette is not None:
                # Padded out to the declared array length: moderngl writes an
                # array uniform whole, and a short write is rejected rather
                # than treated as a prefix.
                padded = palette + [m3.identity()] * (MAX_JOINTS - len(palette))
                self._palettes[id(node)] = m3.gl_bytes_many(padded)

    def palette(self, node: Node) -> bytes | None:
        return self._palettes.get(id(node))

    def release(self) -> None:
        for _node, primitive in self.draws:
            primitive.release()
        for material in self.materials:
            material.release()
        self.draws.clear()
        self.materials.clear()
        self._by_material.clear()
        self._texture_cache.clear()
        self._normal_cache.clear()


def placement(model: Model) -> np.ndarray:
    """The transform that centres a model on the origin and rests it on the grid.

    The frontend does exactly this to every model it shows, on top of whatever
    ``normalize_glb`` already did -- so a compare view lines two meshes up, and
    a mesh from before grounding existed still sits on the floor. Reproducing
    it is what keeps the two viewers framing the same asset identically.
    """
    lo, hi = model.bounds()
    centre = (lo + hi) * 0.5
    return m3.translation(m3.vec3(-centre[0], -lo[1], -centre[2]))

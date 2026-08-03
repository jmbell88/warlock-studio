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

    def __init__(self, ctx: moderngl.Context, material: Material) -> None:
        self.material = material
        self.textures: dict[str, moderngl.Texture] = {}
        self.defines: list[str] = []
        for slot, define, _uniform, _srgb in TEXTURE_SLOTS:
            data = getattr(material, slot)
            if data is None:
                continue
            width, height, pixels = data
            texture = ctx.texture((width, height), 4, pixels)
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.build_mipmaps()
            texture.anisotropy = min(8.0, ctx.max_anisotropy)
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
        for texture in self.textures.values():
            texture.release()
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

        interleaved = np.concatenate(
            [np.ascontiguousarray(c, dtype="f4").reshape(count, -1) for c in columns],
            axis=1,
        )
        self.vbo = ctx.buffer(np.ascontiguousarray(interleaved).tobytes())
        self.ibo = ctx.buffer(np.ascontiguousarray(primitive.indices, "u4").tobytes())
        self.defines = tuple(material.defines + (["SKINNED"] if self.skinned else []))
        self._vaos: dict[int, moderngl.VertexArray] = {}

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
            gpu = GpuMaterial(self.ctx, material)
            self._by_material[key] = gpu
            self.materials.append(gpu)
        return gpu

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

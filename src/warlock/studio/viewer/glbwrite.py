"""The inverse of :mod:`~warlock.studio.viewer.gltf`: a :class:`Model` to GLB bytes.

The loader beside this file is hand-rolled because trimesh drops a scene root's
transform and has no notion of a skin; the writer is hand-rolled for the mirror
reason. There is no encoder in the dependency set that takes *these* types, and
the round trip against the loader that already ships is a stronger test than
either half could be tested with alone -- an authored ``Model``, written, read
back, and compared array for array.

It lives here rather than at package root because of the layering rule the rest
of the project keeps: ``service/`` never imports ``studio/``. Clay
produces bytes and hands them to the service layer, which writes them wherever
an artifact goes. Nothing in the direction ``service -> studio`` is created by
putting the writer next to the loader it inverts.

Four details are easy to get wrong and each is load-bearing rather than
pedantic:

* **``min`` and ``max`` are required on a POSITION accessor.** Most viewers
  compute the scene bound from them rather than from the vertices, so an asset
  without them loads and then frames wrongly -- tiny, or off-screen -- with
  nothing in the file to say why.
* **Buffer views are four-byte aligned.** The loader reads an accessor with
  ``np.frombuffer`` straight at the view's offset; an f4 read from an odd
  offset is a misaligned read where that still matters and a silently wrong
  number where it does not.
* **The index component type has to agree with the bytes.** Unsigned short up
  to 65535 vertices and unsigned int past it -- and the accessor must declare
  whichever was actually written, which is the one place a writer disagrees
  with itself and produces a file that is subtly, not obviously, corrupt.
* **The chunks pad differently.** The BIN chunk pads with zeros and the JSON
  chunk with spaces; the spec says so and a strict reader that hands its JSON
  chunk to a parser unstripped chokes on a trailing NUL.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from ...glbio import CHUNK_BIN, rebuild_glb
from . import gltf

_GLB_MAGIC = 0x46546C67
_U16_MAX = 65535

_FLOAT = 5126
_USHORT = 5123
_UINT = 5125


def _png(image: tuple[int, int, bytes]) -> bytes:
    """A decoded ``(width, height, rgba)`` slot back to PNG bytes."""
    import io

    from PIL import Image

    width, height, data = image
    out = io.BytesIO()
    Image.frombytes("RGBA", (int(width), int(height)), bytes(data)).save(out, "PNG")
    return out.getvalue()


class _Buffer:
    """The BIN chunk under construction, and the views into it.

    Every blob is appended at a four-byte boundary and padded up to one, so a
    view's offset is aligned by construction rather than by a check afterwards.
    """

    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, Any]] = []

    def view(self, array: np.ndarray) -> int:
        return self.view_bytes(np.ascontiguousarray(array).tobytes())

    def view_bytes(self, blob: bytes) -> int:
        """A view over bytes that are already encoded -- a PNG, for instance.

        An image is not an array, and going through one to reach the same
        ``tobytes`` would mean deciding a dtype for something whose bytes are a
        file format. Same alignment rule: every blob starts on a four-byte
        boundary and pads up to one.
        """
        offset = len(self.data)
        self.views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        self.data += blob
        self.data += b"\x00" * (-len(blob) % 4)
        return len(self.views) - 1


class _Writer:
    def __init__(self) -> None:
        self.buffer = _Buffer()
        self.accessors: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        # Identity, not value: the palette is a list of slots the user edits,
        # and two entries that happen to match today must not become one that
        # changes both when either is edited. It is also the same key
        # ``GpuMaterial`` de-duplicates on, so the file and the GPU agree.
        self._material_index: dict[int, int] = {}
        self._material_keep: list[gltf.Material] = []
        # Images and textures, deduplicated by the identity of the
        # ``(width, height, rgba)`` tuple -- the same rule the palette follows,
        # and the reason a document whose eight objects share one baked
        # base-colour map writes one PNG rather than eight.
        self.images: list[dict[str, Any]] = []
        self.textures: list[dict[str, Any]] = []
        self._texture_index: dict[int, int] = {}
        self._texture_keep: list[Any] = []

    # -- accessors ---------------------------------------------------------

    def accessor(
        self,
        array: np.ndarray,
        kind: str,
        component: int,
        *,
        bounds: bool = False,
    ) -> int:
        acc: dict[str, Any] = {
            "bufferView": self.buffer.view(array),
            "componentType": component,
            "count": int(len(array)),
            "type": kind,
        }
        if bounds and len(array):
            acc["min"] = [float(v) for v in array.min(axis=0)]
            acc["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def indices(self, indices: np.ndarray) -> int:
        """The index accessor, narrowed to unsigned short where it fits.

        Halving the index buffer is worth having on a mesh of any size, but the
        reason to do it here rather than leave everything at unsigned int is
        that some readers -- and some engines' importers -- still assume short
        indices for small meshes. The declared component type is derived from
        the same expression the cast is, so the two cannot drift.
        """
        wide = len(indices) and int(indices.max()) > _U16_MAX
        dtype, component = ("<u4", _UINT) if wide else ("<u2", _USHORT)
        return self.accessor(
            np.asarray(indices, dtype=dtype).reshape(-1), "SCALAR", component
        )

    # -- textures ----------------------------------------------------------

    def texture(self, image: tuple[int, int, bytes]) -> int:
        """One decoded image slot as a texture index, encoding it once.

        The image goes into the BIN chunk as a PNG buffer view, which is what
        makes the file self-contained -- a URI would point at something the
        recipient does not have. PNG rather than JPEG because a normal map or a
        metallic-roughness map is not photographic data and JPEG artefacts in
        one are visible as shading noise, not as softness.
        """
        key = id(image)
        if key in self._texture_index:
            return self._texture_index[key]
        index = len(self.textures)
        self.images.append(
            {"bufferView": self.buffer.view_bytes(_png(image)), "mimeType": "image/png"}
        )
        self.textures.append({"source": len(self.images) - 1})
        self._texture_index[key] = index
        # Alive for the length of the call, for the reason ``material`` states:
        # an ``id`` key is only sound while nothing it names can be collected.
        self._texture_keep.append(image)
        return index

    # -- materials ---------------------------------------------------------

    def material(self, material: gltf.Material) -> int:
        key = id(material)
        if key in self._material_index:
            return self._material_index[key]
        doc: dict[str, Any] = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(v) for v in material.base_color_factor],
                "metallicFactor": float(material.metallic_factor),
                "roughnessFactor": float(material.roughness_factor),
            }
        }
        if material.name:
            doc["name"] = material.name
        self._material_textures(material, doc)
        if any(material.emissive_factor):
            doc["emissiveFactor"] = [float(v) for v in material.emissive_factor]
        if material.double_sided:
            doc["doubleSided"] = True
        if material.alpha_mode != "OPAQUE":
            doc["alphaMode"] = material.alpha_mode
            if material.alpha_mode == "MASK":
                doc["alphaCutoff"] = float(material.alpha_cutoff)
        self.materials.append(doc)
        index = len(self.materials) - 1
        self._material_index[key] = index
        # Keyed on ``id``, which is only sound while every keyed object is
        # alive: a collected material could have its id reissued to a different
        # one and de-duplicate the two into a single entry. The model holds
        # every material through its primitives for the whole call, so that
        # cannot happen here -- this list makes the requirement explicit rather
        # than leaving it resting on a caller's object graph.
        self._material_keep.append(material)
        return index

    def _material_textures(self, material: gltf.Material, doc: dict[str, Any]) -> None:
        """Attach whichever of the five slots this material carries.

        Base colour and metallic-roughness live under ``pbrMetallicRoughness``
        and the other three at material level -- which is the spec's layout and
        not a choice, and is the one part of this that a reader will silently
        ignore if it is put in the wrong place.
        """
        pbr = doc["pbrMetallicRoughness"]
        for slot, where, name in (
            ("base_color", pbr, "baseColorTexture"),
            ("metallic_roughness", pbr, "metallicRoughnessTexture"),
            ("normal", doc, "normalTexture"),
            ("emissive", doc, "emissiveTexture"),
            ("occlusion", doc, "occlusionTexture"),
        ):
            image = getattr(material, slot, None)
            if image is not None:
                where[name] = {"index": self.texture(image)}

    # -- primitives --------------------------------------------------------

    def primitive(self, prim: gltf.Primitive) -> dict[str, Any] | None:
        """One glTF primitive, or None for a primitive with no geometry.

        None rather than an empty one because the spec requires an accessor's
        ``count`` to be at least one: a zero-count accessor is not a degenerate
        file, it is an invalid one, and an object that the outliner is holding
        with nothing in it yet is a state Clay reaches routinely.
        """
        if len(prim.positions) == 0:
            return None
        attributes = {
            "POSITION": self.accessor(
                np.asarray(prim.positions, dtype="<f4"), "VEC3", _FLOAT, bounds=True
            )
        }
        # Written only when present. A zero-filled NORMAL or TEXCOORD_0 is four
        # or twelve bytes a vertex of a lie, and both are optional in the spec:
        # a reader that wants normals computes them, which is better than being
        # handed ones that all point along +X.
        if prim.normals is not None:
            attributes["NORMAL"] = self.accessor(
                np.asarray(prim.normals, dtype="<f4"), "VEC3", _FLOAT
            )
        if prim.uvs is not None:
            attributes["TEXCOORD_0"] = self.accessor(
                np.asarray(prim.uvs, dtype="<f4"), "VEC2", _FLOAT
            )
        doc: dict[str, Any] = {"attributes": attributes}
        if len(prim.indices):
            doc["indices"] = self.indices(prim.indices)
        doc["material"] = self.material(prim.material)
        return doc

    # -- nodes -------------------------------------------------------------

    def node(self, node: gltf.Node) -> dict[str, Any]:
        """TRS rather than a matrix, so the values survive readably.

        A node gives either a matrix or TRS and never both. TRS is what the
        editor holds and what the loader reads back; going through a matrix
        would put every rotation through a decomposition that is exact only up
        to sign, for no gain.
        """
        doc: dict[str, Any] = {}
        if node.name:
            doc["name"] = node.name
        if not np.allclose(node.translation, 0.0):
            doc["translation"] = [float(v) for v in node.translation]
        if not np.allclose(node.rotation, (0.0, 0.0, 0.0, 1.0)):
            doc["rotation"] = [float(v) for v in node.rotation]  # XYZW
        if not np.allclose(node.scale, 1.0):
            doc["scale"] = [float(v) for v in node.scale]
        if node.children:
            doc["children"] = [int(c) for c in node.children]
        if node.mesh is not None:
            doc["mesh"] = int(node.mesh)
        return doc


def write_glb(model: gltf.Model) -> bytes:
    """``model`` as the bytes of a self-contained binary glTF.

    Self-contained: one buffer, no URIs, nothing outside the file -- **textures
    included**, PNG-encoded into the BIN chunk as buffer views. Clay still
    paints none, but it imports them now, and an asset that lost its baked maps
    on the way back out would be a round trip that quietly destroys work.
    Images are deduplicated by identity, so a palette sharing one map writes it
    once.

    Skins are refused rather than dropped, for the same reason stated the other
    way round: dropping one produces a rig-shaped file with no rig in it, and
    the failure surfaces a long way from here.
    """
    if model.skins:
        raise ValueError("write_glb does not write a skin; Clay has none")

    writer = _Writer()
    meshes = []
    # ``primitives`` has a minimum length of one in the spec, so a mesh that
    # wrote nothing -- an object whose geometry is empty, or whose every
    # primitive was refused -- is dropped rather than emitted as
    # ``{"primitives": []}``, which strict importers reject outright. Dropping
    # renumbers the survivors, so the nodes that referenced them are remapped
    # and a node whose mesh has gone simply carries no mesh.
    remap: dict[int, int] = {}
    for index, prims in enumerate(model.meshes):
        written = [w for w in (writer.primitive(p) for p in prims) if w is not None]
        if not written:
            continue
        remap[index] = len(meshes)
        meshes.append({"primitives": written})

    nodes = [writer.node(n) for n in model.nodes]
    for node in nodes:
        if "mesh" in node:
            moved = remap.get(node["mesh"])
            if moved is None:
                del node["mesh"]
            else:
                node["mesh"] = moved

    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "Warlock Studio"},
        "scene": 0,
        "scenes": [{"nodes": [int(r) for r in model.roots]}],
        "nodes": nodes,
    }
    if meshes:
        doc["meshes"] = meshes
    if writer.materials:
        doc["materials"] = writer.materials
    if writer.textures:
        doc["images"] = writer.images
        doc["textures"] = writer.textures
    if writer.accessors:
        doc["accessors"] = writer.accessors
        doc["bufferViews"] = writer.buffer.views

    binary = bytes(writer.buffer.data)
    # No empty buffer and no empty BIN chunk. ``byteLength`` has a minimum of
    # one in the spec, so a document with no geometry in it -- an empty scene,
    # or one object the user has not given a shape yet -- would otherwise write
    # a file that is invalid for a reason that has nothing to do with its
    # contents. A glTF with no buffer at all is perfectly legal.
    chunk = b""
    if binary:
        doc["buffers"] = [{"byteLength": len(binary)}]
        chunk = struct.pack("<II", len(binary), CHUNK_BIN) + binary
    # ``rebuild_glb`` takes the header for its magic and version, writes the
    # total length itself, and pads the JSON chunk with spaces -- which is the
    # whole reason the writer goes through it rather than packing its own
    # header: one place decides how a GLB of ours is framed.
    header = struct.pack("<III", _GLB_MAGIC, 2, 0)
    return rebuild_glb(header, doc, chunk)

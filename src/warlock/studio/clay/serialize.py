"""``.wblk`` -- the Clay document on disk, as a zip.

A zip of ``scene.json`` plus one ``meshes/<uid>.npz`` per object, mirroring the
raster editor's ``.ora`` in shape and for the same reason: the small,
human-meaningful half of the document is text that a person can read and a diff
can show, and the large numeric half is stored in a form that does not swell by
a factor of ten on the way through JSON. A Phase 3 subdivided mesh written as
nested float lists would be a megabyte of ``0.7071067811865476``; the same
arrays in an npz are their own bytes.

**Every timestamp in the archive is fixed.** A zip stamps each member with the
wall clock, and an npz is itself a zip, so a document that has not changed would
otherwise produce different bytes every time it was written -- which makes the
file undiffable, makes a content hash useless and makes "has this actually
changed" unanswerable outside the app. Both levels are written at the epoch the
zip format starts at, so two saves of an unchanged document are byte-identical.

**A half-read document is worse than a refused one.** Two failures are caught
rather than papered over: a file written by a newer version, and a scene naming
a mesh the archive does not carry. Substituting an empty mesh for a missing one
would open the file, show the user an object with nothing in it, and let them
save that over their work.

The reader restores a document by *construction* rather than through
``add_object``, which would push one undo step per object and open every file
already dirty; and it raises the process-wide uid floor as it goes, so a fresh
object cannot be minted onto a uid a restored one already wears.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from ..viewer import gltf
from . import mesh as bm
from .document import ClayDoc, Obj, reserve_uid

VERSION = 1
SCENE = "scene.json"
MESH_DIR = "meshes"

# 1980-01-01, the earliest a zip can express. Any fixed value would do; this one
# is the conventional choice for a reproducible archive and is obviously not a
# real modification time, which is the point -- nobody should read it as one.
_EPOCH = (1980, 1, 1, 0, 0, 0)

_MESH_FIELDS = ("positions", "loops", "starts", "material", "smooth")


# --- writing ----------------------------------------------------------------


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """A ``.npz`` built member by member, so its timestamps are ours.

    ``np.savez`` would be one line, but it stamps each member with the wall
    clock and there is no argument that turns that off. An npz *is* a zip of
    ``.npy`` members, so writing it here costs a few lines and buys the
    byte-identity the whole format is claimed to have. ``np.load`` reads the
    result with no idea it was not written by numpy.
    """
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(array))
            zf.writestr(zipfile.ZipInfo(f"{name}.npy", _EPOCH), member.getvalue())
    return out.getvalue()


def _material_json(material: gltf.Material) -> dict[str, Any]:
    """The factors, and only the factors.

    Every texture slot is deliberately dropped. Clay paints none, they
    are decoded pixel buffers rather than files, and a format that carried them
    would be making a promise about textures that nothing upstream can keep.
    """
    return {
        "name": material.name,
        "base_color_factor": [float(v) for v in material.base_color_factor],
        "metallic_factor": float(material.metallic_factor),
        "roughness_factor": float(material.roughness_factor),
        "emissive_factor": [float(v) for v in material.emissive_factor],
        "double_sided": bool(material.double_sided),
        "alpha_mode": str(material.alpha_mode),
        "alpha_cutoff": float(material.alpha_cutoff),
    }


def _object_json(obj: Obj) -> dict[str, Any]:
    return {
        "uid": int(obj.uid),
        "name": obj.name,
        "translation": [float(v) for v in obj.translation],
        "rotation": [float(v) for v in obj.rotation],  # XYZW, as everywhere
        "scale": [float(v) for v in obj.scale],
        "generator": obj.generator,
        "params": obj.params,
        "visible": bool(obj.visible),
        "material": int(obj.material),
    }


def scene_json(doc: ClayDoc) -> str:
    """``scene.json``'s text: sorted keys, indented, one object per entry.

    Sorted and indented rather than compact because this half of the file
    exists to be *read* -- by a person looking at why a document opens wrong,
    and by a diff. The mesh arrays are the reason the format is a zip; there is
    no size argument left for minifying the small half.
    """
    scene = {
        "version": VERSION,
        "materials": [_material_json(m) for m in doc.materials],
        "objects": [_object_json(o) for o in doc.objects],
    }
    return json.dumps(scene, sort_keys=True, indent=2)


def wblk_bytes(doc: ClayDoc) -> bytes:
    """The document as the bytes of a ``.wblk`` archive."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(SCENE, _EPOCH), scene_json(doc))
        for obj in doc.objects:
            arrays = {name: getattr(obj.mesh, name) for name in _MESH_FIELDS}
            zf.writestr(
                zipfile.ZipInfo(f"{MESH_DIR}/{int(obj.uid)}.npz", _EPOCH),
                _npz_bytes(arrays),
            )
    return out.getvalue()


# --- reading ----------------------------------------------------------------


def _read_mesh(zf: zipfile.ZipFile, uid: int) -> bm.Mesh:
    name = f"{MESH_DIR}/{uid}.npz"
    try:
        raw = zf.read(name)
    except KeyError as exc:
        raise ValueError(
            f"this clay document names an object whose mesh is missing ({name})"
        ) from exc
    # ``allow_pickle=False`` is the default and is left explicit: an npz is a
    # zip a user can be handed, and object arrays in one are arbitrary code.
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        try:
            arrays = {name: npz[name] for name in _MESH_FIELDS}
        except KeyError as exc:
            raise ValueError(f"a mesh in this clay document is incomplete: {exc}") from exc
    mesh = bm.Mesh(**arrays)
    # Validate rather than trust: the CSR offsets are the one part of the file
    # that a truncated write or a hand edit makes *quietly* wrong -- ``edges``
    # and ``_face_normals`` do not raise on a bad ``starts``, they produce
    # nonsense. Better to refuse the file than to render it.
    bm.validate(mesh)
    return mesh


def _material_from(entry: dict[str, Any]) -> gltf.Material:
    return gltf.Material(
        name=str(entry.get("name", "")),
        base_color_factor=tuple(entry.get("base_color_factor", (1.0, 1.0, 1.0, 1.0))),
        metallic_factor=float(entry.get("metallic_factor", 1.0)),
        roughness_factor=float(entry.get("roughness_factor", 1.0)),
        emissive_factor=tuple(entry.get("emissive_factor", (0.0, 0.0, 0.0))),
        double_sided=bool(entry.get("double_sided", False)),
        alpha_mode=str(entry.get("alpha_mode", "OPAQUE")),
        alpha_cutoff=float(entry.get("alpha_cutoff", 0.5)),
    )


def read_wblk(data: bytes) -> ClayDoc:
    """A ``.wblk``'s bytes back into a :class:`ClayDoc`.

    The returned document has an empty history and reads clean: a file that has
    just been opened is not unsaved, and the objects are placed directly rather
    than through ``add_object``, which would push a step apiece.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        scene = json.loads(zf.read(SCENE))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("this is not a Warlock Clay document") from exc

    version = int(scene.get("version", 0))
    if version > VERSION:
        raise ValueError(
            f"this clay document was written by a newer version of Warlock "
            f"(format {version}, this build reads {VERSION})"
        )

    with zf:
        objects = []
        for entry in scene.get("objects", []):
            uid = int(entry["uid"])
            reserve_uid(uid)
            objects.append(
                Obj(
                    uid=uid,
                    name=str(entry.get("name", "")),
                    mesh=_read_mesh(zf, uid),
                    translation=np.array(entry.get("translation", (0.0, 0.0, 0.0))),
                    rotation=np.array(entry.get("rotation", (0.0, 0.0, 0.0, 1.0))),
                    scale=np.array(entry.get("scale", (1.0, 1.0, 1.0))),
                    generator=entry.get("generator"),
                    params=dict(entry.get("params") or {}),
                    visible=bool(entry.get("visible", True)),
                    material=int(entry.get("material", 0)),
                )
            )

    return ClayDoc(
        objects=objects,
        materials=[_material_from(m) for m in scene.get("materials", [])],
    )

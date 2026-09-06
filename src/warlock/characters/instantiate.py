"""Recipe in, ``source.glb`` / ``model.glb`` / ``character.json`` out.

**Archetype-agnostic on purpose.** Nothing below knows what a humanoid is. It
reads the species' baked mesh and the mask file beside it, adds the appearance
channels the recipe asks for, paints the regions the theme names, scales the
result to the species' height and writes the three files a job directory holds.
The other three body plans plug in by shipping their own bake; this module does
not grow a branch when they do.

**The three files are the artifact rule, applied.** ``source.glb`` is the
reconstruction -- here, the generated body at the requested appearance;
``model.glb`` is derived from it, and today it is the same bytes because there is
no optimisation pass a generated mesh needs; ``character.json`` is the sidecar
naming the joints, the sockets, the materials and the recipe that produced them.
Every one is staged to a temp name and ``os.replace``d, never written in place,
because a half-written ``model.glb`` under a served name is the failure that rule
exists to prevent.

**Deterministic.** Two runs of the same recipe produce byte-identical files.
That is what makes a character cacheable and a rerun meaningful, and it is why
nothing here consults a clock, a random number generator or a set iteration
order.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CharacterError
from .family import Family, get_family
from .recipe import VERSION as RECIPE_VERSION
from .recipe import Recipe

__all__ = ["Instance", "instantiate", "transformed_joints"]

#: The three names a job directory serves. Named here rather than spelled at
#: each write so the sidecar's own manifest cannot disagree with what was
#: written.
SOURCE_NAME = "source.glb"
MODEL_NAME = "model.glb"
SIDECAR_NAME = "character.json"


@dataclass(frozen=True)
class Instance:
    """What was built, in the terms the rest of the program asks in."""

    #: ``rigging.validate_joints``' shape -- ``{"name", "parent", "head",
    #: "tail"}`` per bone, in template order, Blender axes, **world metres**.
    joints: list[dict[str, Any]]
    #: ``socket name -> {"bone", "position", "reach"}``, world metres.
    sockets: dict[str, dict[str, Any]]
    family: str
    version: int
    #: ``region -> #rrggbb``, exactly what was written into the GLB.
    materials: dict[str, str]

    @property
    def bone_names(self) -> list[str]:
        return [b["name"] for b in self.joints]


def _load_base(fam: Family) -> tuple[list[Any], np.ndarray, dict[str, np.ndarray]]:
    """``(primitives, concatenated positions, mask arrays)`` for a species.

    The digest is checked here and nowhere else. A ``.masks.npz`` whose channels
    were baked against different positions would displace the wrong vertices --
    silently, and worst on the vertices that moved the most -- so the two files
    are treated as one artifact that happens to be stored twice.
    """
    from ..studio.viewer import gltf

    if not fam.base_glb.is_file():
        raise CharacterError(
            f"{fam.label} has no baked mesh at {fam.base_glb.name}; run "
            "scripts/author_humanoid.py --write",
            field="family",
        )
    model = gltf.load(fam.base_glb)
    prims = [p for mesh in model.meshes for p in mesh]
    stacked = np.concatenate([p.positions for p in prims]).astype("f4")

    with np.load(fam.masks_npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    digest = hashlib.blake2b(stacked.tobytes(), digest_size=16).digest()
    if bytes(arrays["positions_digest"].tobytes()) != digest:
        raise CharacterError(
            f"{fam.label}'s mask file was baked against a different {fam.base_glb.name}",
            field="family",
        )
    return prims, stacked.astype("f8"), arrays


def _displaced(
    positions: np.ndarray, arrays: dict[str, np.ndarray], appearance: dict[str, float]
) -> np.ndarray:
    out = positions.copy()
    # Sorted, not insertion order: floating-point addition is not associative,
    # and "deterministic" has to survive the recipe having been built by a
    # different code path with the same values in a different order.
    for key in sorted(appearance):
        field = arrays.get(f"disp/{key}")
        if field is None:
            raise CharacterError(f"the baked mesh has no {key!r} channel", field="appearance")
        out = out + float(appearance[key]) * field.astype("f8")
    return out


def _displaced_joints(
    joints: np.ndarray, arrays: dict[str, np.ndarray], appearance: dict[str, float]
) -> np.ndarray:
    out = joints.copy()
    for key in sorted(appearance):
        out = out + float(appearance[key]) * arrays[f"jdisp/{key}"].astype("f8")
    return out


def _ground_and_scale(
    positions: np.ndarray, height_m: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """``(positions, scale, offset)`` -- grounded, centred, the right height.

    Grounding always runs, the rule ``model.glb`` already lives by: a character
    whose feet are not on y = 0 renders half a cell high in every sprite of
    every sheet, and the offset is not recoverable from the image afterwards.
    """
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = float(hi[1] - lo[1])
    scale = float(height_m) / (span if span > 1e-9 else 1.0)
    offset = np.array([(lo[0] + hi[0]) / 2.0, float(lo[1]), (lo[2] + hi[2]) / 2.0])
    return (positions - offset) * scale, scale, offset


def _to_blender(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def _to_gltf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _hex_to_rgba(value: str) -> tuple[float, float, float, float]:
    text = value.lstrip("#")
    if len(text) != 6:
        raise CharacterError(f"{value!r} is not a #rrggbb colour", field="theme")
    r, g, b = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, 1.0)


def _smooth_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tri = np.asarray(indices, dtype="i8").reshape(-1, 3)
    a, b, c = positions[tri[:, 0]], positions[tri[:, 1]], positions[tri[:, 2]]
    face = np.cross(b - a, c - a)
    out = np.zeros_like(positions, dtype="f8")
    for k in range(3):
        np.add.at(out, tri[:, k], face)
    norm = np.linalg.norm(out, axis=1)
    norm[norm == 0] = 1.0
    return (out / norm[:, None]).astype("f4")


def transformed_joints(
    joints: list[dict[str, Any]], transform: np.ndarray
) -> list[dict[str, Any]]:
    """A joint list through a 4x4, column-vector convention (``M @ v``).

    The same convention ``clay.mesh.transformed`` uses, so a caller holding one
    matrix can move the mesh and the skeleton with it and not have to remember
    which of the two wanted the transpose.
    """
    matrix = np.asarray(transform, dtype="f8")
    if matrix.shape != (4, 4):
        raise CharacterError("a joint transform is a 4x4 matrix")

    def move(point: Any) -> list[float]:
        v = np.append(np.asarray(point, dtype="f8"), 1.0)
        out = matrix @ v
        return [float(x) for x in out[:3]]

    return [
        {"name": b["name"], "parent": b["parent"], "head": move(b["head"]), "tail": move(b["tail"])}
        for b in joints
    ]


def _sockets(fam: Family, joints: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name = {b["name"]: b for b in joints}
    out: dict[str, dict[str, Any]] = {}
    for socket in fam.sockets:
        bone = by_name.get(socket.bone)
        if bone is None:  # pragma: no cover - a socket naming no bone is a bad row
            raise CharacterError(f"socket {socket.name!r} names no bone of {fam.template}")
        head = np.array(bone["head"], dtype="f8")
        tail = np.array(bone["tail"], dtype="f8")
        span = tail - head
        length = float(np.linalg.norm(span)) or 1.0
        along = span / length
        # A frame with no arbitrary twist: "lateral" is whatever is perpendicular
        # to the bone in the world's horizontal plane, "up" completes it. Enough
        # for a prop to hang off; a full bone roll is the poser's business.
        lateral = np.cross(along, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(lateral) < 1e-9:
            lateral = np.array([1.0, 0.0, 0.0])
        lateral = lateral / np.linalg.norm(lateral)
        up = np.cross(along, lateral)
        a, b, c = socket.offset
        position = head + length * (a * along + b * lateral + c * up)
        out[socket.name] = {
            "bone": socket.bone,
            "position": [float(v) for v in position],
            "reach": float(socket.reach * length),
        }
    return out


def _write(path: Path, data: bytes) -> None:
    """Stage onto a served name. Never in place -- the repo's rule for these."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def instantiate(recipe: Recipe, out_dir: Any) -> Instance:
    """Build one character into *out_dir*, and say what was built."""
    from ..studio.viewer import glbwrite, gltf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fam = get_family(recipe.family)
    theme = fam.theme(recipe.theme)

    prims, positions, arrays = _load_base(fam)
    appearance = dict(recipe.appearance)
    positions = _displaced(positions, arrays, appearance)

    joint_points = _to_gltf(_displaced_joints(arrays["joints"].astype("f8"), arrays, appearance))
    # One transform for both, derived from the mesh: a skeleton grounded against
    # its own bounding box rather than the body's would sit a few millimetres off
    # in every pose, and the error would look like bad weights.
    positions, scale, offset = _ground_and_scale(positions, fam.height_m)
    joint_points = (joint_points - offset) * scale

    names = [str(n) for n in arrays["joint_names"]]
    parents = [str(p) or None for p in arrays["joint_parents"]]
    blender = _to_blender(joint_points)
    joints = [
        {
            "name": name,
            "parent": parent,
            "head": [float(v) for v in blender[i, 0]],
            "tail": [float(v) for v in blender[i, 1]],
        }
        for i, (name, parent) in enumerate(zip(names, parents, strict=True))
    ]
    _check_against_template(fam, joints)

    offsets = arrays["prim_offsets"].astype("i8")
    regions = arrays["prim_regions"].astype("i8")
    region_names = fam.regions
    materials = {name: theme.materials[name] for name in region_names if name in theme.materials}

    built: list[gltf.Primitive] = []
    for index, prim in enumerate(prims):
        block = positions[offsets[index] : offsets[index + 1]].astype("f4")
        region = region_names[int(regions[index])]
        colour = theme.materials.get(region)
        if colour is None:
            raise CharacterError(
                f"the {theme.key} look does not paint {region!r}", field="theme"
            )
        # ``_make_flat`` in the Blender worker drives emission from the base
        # colour, so an accent region is lit by being *coloured*, with no texture
        # to compose and no image for the worker to fail to decode.
        lit = region == "accent" and bool(theme.effects)
        emissive = _hex_to_rgba(colour)[:3] if lit else (0.0, 0.0, 0.0)
        built.append(
            gltf.Primitive(
                positions=block,
                indices=np.asarray(prim.indices, dtype="u4"),
                normals=_smooth_normals(block.astype("f8"), prim.indices),
                material=gltf.Material(
                    name=region,
                    base_color_factor=_hex_to_rgba(colour),
                    metallic_factor=0.0,
                    roughness_factor=0.85,
                    emissive_factor=tuple(float(v) for v in emissive),
                ),
            )
        )

    node = gltf.Node(name=recipe.name, mesh=0)
    data = glbwrite.write_glb(gltf.Model(nodes=[node], roots=[0], meshes=[built], skins=[]))
    _write(out_dir / SOURCE_NAME, data)
    # Derived from source.glb, and today derived by the identity: a generated
    # body needs no decimation and no seam repair. Written as its own file
    # anyway, because every export downstream is a pure function of *model.glb*
    # and a missing one would send them all to the reconstruction instead.
    _write(out_dir / MODEL_NAME, data)

    sockets = _sockets(fam, joints)
    sidecar = {
        "version": RECIPE_VERSION,
        "family": fam.key,
        "family_version": fam.version,
        "archetype": fam.archetype,
        "label": fam.label,
        "template": fam.template,
        "clip_library": fam.clip_library,
        "height_m": fam.height_m,
        "theme": theme.key,
        "materials": materials,
        "joints": joints,
        "sockets": sockets,
        "recipe": recipe.as_dict(),
    }
    _write(
        out_dir / SIDECAR_NAME,
        json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return Instance(
        joints=joints,
        sockets=sockets,
        family=fam.key,
        version=fam.version,
        materials=materials,
    )


def _check_against_template(fam: Family, joints: list[dict[str, Any]]) -> None:
    """The joints have to be the ones the rig template names, exactly.

    Through ``rigging.validate_joints`` rather than by comparing name sets: that
    function is the door every corrected skeleton already comes in by, and a
    generated one that would not survive it is a rig job that fails inside
    Blender instead of here.
    """
    from .. import rigging

    template = rigging.get_template(fam.template)
    payload = {"bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in joints]}
    try:
        rigging.validate_joints(payload, template)
    except ValueError as exc:
        raise CharacterError(
            f"{fam.label}'s baked skeleton does not fit the {fam.template} template: {exc}",
            field="family",
        ) from None

"""Mesh conversion and scaling. GLB is the primary output; STL/OBJ are derived on demand.

Decimation and UV atlasing happen inside trellis-server (quadric simplify to
300K faces @1024), so post-processing here is format conversion plus the one
transform trellis-server cannot do itself: scaling the result to a real-world
size (see guidance.py).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import trimesh

from .. import glbio

log = logging.getLogger(__name__)

# The container format lives in warlock.glbio, shared with the viewer's loader.
# Kept as module-level aliases because this module's own callers (and its
# tests) have always spelled them with the leading underscore.
_split_glb = glbio.split_glb
_rebuild_glb = glbio.rebuild_glb


@contextlib.contextmanager
def _staged(dest: Path) -> Iterator[Path]:
    """Yield a temp path beside ``dest``, renamed onto it on clean exit.

    Both exports are produced on demand by the file-serving route, so two
    requests for the same artifact (a double-clicked download link is enough)
    can run concurrently. Writing to ``dest`` directly lets a reader observe a
    half-written file; a per-call unique temp plus os.replace -- an atomic
    rename on POSIX and Windows alike when both share a filesystem -- means a
    reader sees either the old file or the complete new one. Same reasoning as
    scale_glb below, which had this bug first.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(raw)
    try:
        yield tmp
        _replace_or_accept(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _replace_or_accept(tmp: Path, dest: Path) -> None:
    """os.replace(tmp, dest), tolerating a concurrent writer.

    Windows fails the rename with ERROR_ACCESS_DENIED when two threads replace
    the same target at once, where POSIX would just let the last one win. These
    artifacts are pure functions of the GLB, so whoever landed first produced a
    file identical to ours: losing the race is a success, not an error. Only a
    failure that leaves no file at all is worth raising.
    """
    for attempt in range(3):
        try:
            os.replace(tmp, dest)
            return
        except OSError:
            if attempt < 2:
                time.sleep(0.05)
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("%s was written by a concurrent export; keeping theirs", dest)
        return
    os.replace(tmp, dest)  # re-raise the real error with the real traceback


def glb_to_stl(glb_path: Path, stl_path: Path) -> Path:
    """Geometry-only STL (for reference / printing previews)."""
    mesh = _load_merged(glb_path)
    # Exported to bytes rather than to the temp path: trimesh infers the format
    # from the suffix, and the temp name deliberately doesn't have one.
    data = mesh.export(file_type="stl")
    with _staged(stl_path) as tmp:
        tmp.write_bytes(data)
    return stl_path


def glb_to_obj_zip(glb_path: Path, zip_path: Path) -> Path:
    """OBJ + MTL + textures bundled in a zip (OBJ is inherently multi-file)."""
    scene = trimesh.load(glb_path)
    # A private temp dir, not a fixed "obj_export" next to the zip: the fixed
    # name was both left behind forever (duplicating the model and every
    # texture beside the zip) and shared by concurrent exports, which
    # interleaved their writes into it.
    workdir = Path(tempfile.mkdtemp(dir=zip_path.parent, prefix=".obj_export."))
    try:
        scene.export(workdir / "model.obj")
        with _staged(zip_path) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            # rglob, not iterdir: trimesh may emit textures into a subdirectory,
            # which zf.write would refuse as a directory entry.
            for f in sorted(workdir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(workdir).as_posix())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return zip_path


# glTF PBR slot -> the filename it gets in the zip. Named for what the map *is*
# rather than what the GLB called it, because a texture pulled out for editing
# ends up in an engine's material slot, not back in a glTF.
_TEXTURE_SLOTS = {
    "baseColorTexture": "base_color.png",
    "metallicRoughnessTexture": "metallic_roughness.png",
    "normalTexture": "normal.png",
    "emissiveTexture": "emissive.png",
    "occlusionTexture": "occlusion.png",
}


def glb_to_textures_zip(glb_path: Path, zip_path: Path) -> Path:
    """Every PBR map in the GLB, as loose PNGs in a zip.

    Raises ValueError when there is nothing to extract: an empty zip looks like
    a successful export of a model with no textures, which is a different and
    much more alarming thing than "this GLB has no maps".
    """
    mesh = _load_merged(glb_path)
    material = getattr(getattr(mesh, "visual", None), "material", None)
    found: list[tuple[str, Any]] = []
    for slot, filename in _TEXTURE_SLOTS.items():
        image = getattr(material, slot, None)
        if image is not None:
            found.append((filename, image))
    if not found:
        raise ValueError("this model has no textures to extract")

    import io

    with _staged(zip_path) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, image in found:
            buf = io.BytesIO()
            image.save(buf, "PNG")
            zf.writestr(filename, buf.getvalue())
    return zip_path


def glb_to_collision(glb_path: Path, out_path: Path, *, max_hull_faces: int = 256) -> Path:
    """A convex collision shape for the mesh, as its own GLB.

    A hull rather than a decomposition: every engine accepts a convex shape
    directly, hull generation is deterministic and takes milliseconds, and a
    proper decomposition needs a VHACD binary this project would have to vendor.
    For the props Warlock produces the hull is the shape a hand-made collider
    would have been anyway.

    Simplified to a face budget because a hull over a 290k-triangle
    reconstruction can still carry thousands of faces, and a physics engine
    pays for every one of them each frame.
    """
    mesh = _load_merged(glb_path)
    hull = mesh.convex_hull
    if len(hull.faces) > max_hull_faces:
        # Simplify then re-hull: simplification can push vertices inward and
        # make the result non-convex, and a collider that is not convex is
        # silently wrong rather than loudly broken.
        #
        # Best-effort, because trimesh delegates decimation to an optional
        # backend (fast_simplification) that this project does not depend on.
        # An unsimplified hull is a correct collider that costs the physics
        # engine more per frame; refusing to produce one, or pulling in a
        # dependency to shave faces off it, are both worse trades.
        try:
            hull = hull.simplify_quadric_decimation(face_count=max_hull_faces).convex_hull
        except Exception as exc:
            log.info(
                "collision hull for %s kept at %d faces (no decimation backend: %s)",
                glb_path.name, len(hull.faces), exc,
            )
    data = trimesh.Scene(hull).export(file_type="glb")
    with _staged(out_path) as tmp:
        tmp.write_bytes(data)
    return out_path


def normalize_glb(glb_path: Path, target_max_m: float | None) -> dict[str, Any]:
    """Scale to a real-world size, centre in X/Z and sit the model on Y=0.

    Returns the transform that was applied. Scaling alone was never enough: a
    trellis GLB's origin is wherever the reconstruction volume's centre happened
    to be, so every import into Godot or Unity needed a manual origin fixup
    before the asset would stand on the ground.

    The measurement goes through trimesh but the write does not -- re-exporting
    would re-encode every texture. Only the JSON chunk is rewritten, inserting
    the scale and translation as *one* node above the whole scene; buffers and
    images are copied through byte-for-byte.

    **The composition is ``T . S . M_root``, never ``M_root . T . S``.** The
    bounds are measured in *world* space, after every node's TRS has composed,
    so the transform has to apply outside each root's own. That is what
    ``_insert_transform_below`` does by **emptying** the root and folding its
    TRS down into the inserted child (``_composed``). Sandwiching the grounding
    under an untouched root was wrong twice over: a rotated root rotated the
    grounding offset, and the same world-space translation applied once per
    root -- with each root's offset left unscaled above it -- meant a Clay
    export, which emits one root per object, grounded every object as though it
    alone were the scene. Three boxes asked to be 2 m across came out 8.2 m and
    reported 2. trellis output hid both, because trimesh writes one identity
    root.

    The transform still may not sit *on* a root: trimesh treats one as the
    graph's base frame and silently discards its transform, which would leave
    the GLB grounded and the derived STL/OBJ exports not. A bare wrapper node
    above the roots does not work either -- trimesh's base frame is named
    ``world`` and so is the root its own exporter writes, so the collision
    resolves into a cycle that drops the transform again.
    """
    scene = trimesh.load(glb_path)
    extents = scene.extents
    bounds = scene.bounds
    extent = float(max(extents))
    factor = 1.0
    if target_max_m and extent > 0 and target_max_m > 0:
        factor = target_max_m / extent

    # In the *scaled* frame: centre X and Z on the origin, put minimum Y at zero.
    # glTF is Y-up, which is why Y is the odd one out here and Z is not.
    lo, hi = bounds[0], bounds[1]
    translation = [
        -float(lo[0] + hi[0]) / 2.0 * factor,
        -float(lo[1]) * factor,
        -float(lo[2] + hi[2]) / 2.0 * factor,
    ]

    header, gltf, rest = _split_glb(glb_path.read_bytes())
    gltf_scene = gltf["scenes"][gltf.get("scene", 0)]
    nodes = gltf.setdefault("nodes", [])
    for root in gltf_scene.get("nodes", []):
        _insert_transform_below(nodes, root, factor, translation)
    # Write to a temp file and rename into place instead of write_bytes(),
    # which opens "wb" and truncates glb_path to 0 bytes before writing --
    # a concurrent reader (the file-serving route, or another export) could
    # observe a half-written file. os.replace is an atomic rename on both
    # POSIX and Windows as long as tmp and glb_path share a filesystem.
    tmp = glb_path.with_suffix(".glb.tmp")
    tmp.write_bytes(_rebuild_glb(header, gltf, rest))
    os.replace(tmp, glb_path)
    return {
        "scale": factor,
        "translation": translation,
        "achieved_size_m": extent * factor,
    }


def scale_glb(glb_path: Path, target_max_m: float) -> float:
    """Back-compatible wrapper: normalize and return only the scale factor."""
    return normalize_glb(glb_path, target_max_m)["scale"]


def _insert_transform_below(
    nodes: list[dict], root: int, factor: float, translation: list[float]
) -> None:
    """Move everything hanging off ``root`` -- *including the root's own
    transform* -- into a new child that carries the grounding as well.

    The transform still cannot go on the root: trimesh treats a scene root as
    the graph's base frame and silently discards its transform, which would
    leave the GLB grounded and every derived STL and OBJ not. Nor can a new
    root be added above it, which is the shape this looks like it wants: the
    node trimesh calls the base frame is named ``world``, and so is the root
    its own exporter writes, so a wrapper produces two nodes with one name and
    trimesh resolves the collision into a cycle that drops the transform again.

    So the composition is fixed here instead. The bounds were measured in world
    space, after every node's TRS composed, and the result must therefore apply
    *outside* the root's own transform -- ``T . S . M_root``, never
    ``M_root . T . S``. Sandwiching it under an untouched root was the latter,
    so any rotation on a root rotated the grounding offset and a mesh authored
    in Blender or exported from Clay came back sunk and shifted. It also meant
    the same world-space translation was applied once per root while each
    root's own offset stayed unscaled above it, so a Clay export -- one root
    per object -- grounded every object as though it alone were the scene:
    three boxes asked to be 2 m across came out 8.2 m and reported 2.

    Emptying the root is what makes it correct for every root at once: with the
    root left as a bare parent, the child's transform *is* the world transform,
    and ``T . S . M_root`` is the same grounding applied to each.
    """
    node = nodes[root]
    child: dict = {}
    for key in ("mesh", "children", "skin", "weights"):
        if key in node:
            child[key] = node.pop(key)
    child.update(_composed(node, factor, translation))
    nodes.append(child)
    node["children"] = [len(nodes) - 1]


def _composed(node: dict, factor: float, translation: list[float]) -> dict:
    """``T . S . M_node`` as a node transform, consuming the node's own.

    Stays in TRS whenever the node was in TRS, which is every exporter we meet:
    the scale is uniform, so the product decomposes exactly -- rotation is
    untouched, scale multiplies through, and the translation is the node's own
    scaled and then offset. A node carrying an explicit ``matrix`` (glTF allows
    either, never both) is composed as a matrix, because a general one need not
    decompose into TRS at all.
    """
    if "matrix" in node:
        # glTF matrices are column-major, which is the transpose of what numpy
        # multiplies -- hence the round trip through .T at both ends.
        import numpy as np

        own = np.array(node.pop("matrix"), dtype=float).reshape(4, 4).T
        outer = np.diag([factor, factor, factor, 1.0])
        outer[:3, 3] = translation
        return {"matrix": [float(v) for v in (outer @ own).T.reshape(16)]}

    own_t = [float(v) for v in node.pop("translation", (0.0, 0.0, 0.0))]
    own_s = [float(v) for v in node.pop("scale", (1.0, 1.0, 1.0))]
    rotation = node.pop("rotation", None)
    out = {
        "translation": [translation[i] + factor * own_t[i] for i in range(3)],
        "scale": [factor * own_s[i] for i in range(3)],
    }
    if rotation is not None:
        out["rotation"] = rotation
    return out


def _load_merged(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        return loaded.to_mesh()
    return loaded

"""Game-ready remesh: the host-side half, and every number the panel and the
door agree on.

A TRELLIS reconstruction is a triangle soup at ~300k faces with xatlas islands
-- fine to look at, wrong to ship: no edge loops, a UV layout no artist can
paint, and holes wherever the narrow band gave up. Every commercial pipeline
(Meshy's Remesh, Tripo's quad remesher, Rodin's quad tiers) closes exactly that
gap with one step: **remesh to a face budget, unwrap the new surface, bake the
old surface's colour, roughness and normals onto it.** This module is the
arithmetic and the vocabulary of that step; the Blender half is
``blender_worker.op_remesh``, and the two never import each other.

Unlike a gltfpack retarget (``pipelines.optimize``), which *simplifies* the
reconstruction and keeps its UVs, a remesh produces a **new** surface -- so it
bakes rather than preserves, and ``tiercheck.compare`` against the mesh it
replaced is what says whether anything a tier must keep was lost.

Pure: stdlib only, no bpy, no service, no queue.
"""

from __future__ import annotations

from typing import Any

#: Quad budgets, keyed by the word the panel shows. Faces, not triangles: a
#: quadriflow target is a quad count, and a "2k quad" prop is ~4k triangles --
#: the ladder a mobile/indie engine actually budgets in. ``custom`` takes a
#: number in ``FACES_MIN..FACES_MAX``.
FACE_PROFILES: dict[str, int] = {
    "low": 2_000,
    "medium": 8_000,
    "high": 30_000,
}
DEFAULT_PROFILE = "medium"
FACES_MIN = 500
FACES_MAX = 200_000

#: Bake resolutions. ``None`` at the door means "match the mesh's own atlas",
#: resolved by the worker against the file for ``retexture``'s reason.
TEXTURE_SIZES = (512, 1024, 2048)
DEFAULT_TEXTURE_PX = 1024

#: Voxel size, as a fraction of the mesh's bounding diagonal, for the
#: hole-closing pre-pass. One number rather than a slider: at 1/200 of the
#: diagonal a 1 m prop remeshes at 5 mm voxels, which closes the plate-crust
#: gaps ``meshaudit`` flags without rounding off a sword's edge; finer than
#: that is minutes in Blender for no visible change.
VOXEL_FRACTION = 0.005

#: The seed quadriflow takes. Fixed so two remeshes of one mesh at one budget
#: are the same mesh -- a reroll is a different job, not a different seed.
QUADRIFLOW_SEED = 0

#: Every export a remesh invalidates: the whole of ``service.files.DERIVED``,
#: restated here because the queue may not import ``service``. Geometry *and*
#: skin change, so this is the superset of ``retexture.SURFACE_DERIVED``;
#: ``tests/test_remesh.py`` asserts the two spellings agree.
GEOMETRY_DERIVED = ("model.stl", "model_obj.zip", "collision.glb", "textures.zip", "model.fbx")

#: The margin (texels) the bake grows past every island edge, so bilinear
#: filtering and the first two mips never read the background.
BAKE_MARGIN_PX = 8


def resolve(profile: str, custom: int | None = None) -> int:
    """The quad budget a profile names, or a refusal that names the range.

    ``ValueError`` rather than a service error: this is the one implementation
    the door *and* the panel's pre-flight call, exactly as ``optimize.resolve``
    is for the triangle tiers, so both refuse in the same words.
    """
    if profile == "custom":
        if custom is None:
            raise ValueError("a custom remesh needs a face count")
        try:
            value = int(custom)
        except (TypeError, ValueError) as exc:
            raise ValueError("face count must be a whole number") from exc
        if not FACES_MIN <= value <= FACES_MAX:
            raise ValueError(f"face count must be between {FACES_MIN:,} and {FACES_MAX:,}")
        return value
    if profile not in FACE_PROFILES:
        raise ValueError(
            f"unknown remesh profile {profile!r}; one of "
            f"{sorted(FACE_PROFILES)} or 'custom'"
        )
    return FACE_PROFILES[profile]


def profile_label(key: str, faces: int | None = None) -> str:
    """"Medium (8k quads)" -- derived from the table so a label cannot
    disagree with the number the worker is asked for."""
    if key == "custom":
        return "Custom..."
    faces = FACE_PROFILES[key] if faces is None else faces
    if faces % 1000 == 0:
        return f"{key.capitalize()} ({faces // 1000}k quads)"
    return f"{key.capitalize()} ({faces:,} quads)"


def report_line(report: Any) -> str | None:
    """One sentence about what the last remesh produced, or None.

    Says which path made the surface -- quadriflow, or the decimate fallback a
    non-manifold input forces -- because a "remeshed" mesh that is really a
    decimated one has triangles where the user was promised quads, and the
    panel must not let the two read the same.
    """
    if not isinstance(report, dict):
        return None
    faces = report.get("faces")
    if faces is None:
        return None
    quads = report.get("quads")
    method = report.get("method")
    size = report.get("texture_size")
    if method == "decimate":
        head = f"{faces:,} triangles (decimated: the surface was not manifold enough to quad)"
    elif quads is not None:
        head = f"{faces:,} faces, {quads * 100:.0f}% quads"
    else:
        head = f"{faces:,} faces"
    tail = f", baked at {size} px" if size else ""
    verdict = report.get("tiercheck") or {}
    if verdict.get("ok") is False:
        tail += "; lost: " + ", ".join(verdict.get("failures") or ()) or "; lost something"
    return head + tail

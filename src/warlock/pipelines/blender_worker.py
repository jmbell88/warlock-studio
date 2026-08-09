"""The ``bpy`` side of rigging. Runs as a subprocess, never inside the app.

Invoked as ``python -m warlock.pipelines.blender_worker`` with a JSON spec
on stdin (see ``rigging.run_worker``). Writes its result to
``spec["result_path"]`` and progress to stdout as ``[blender] <frac> <label>``.

Why a subprocess at all is argued in ``rigging.py``'s docstring; the short
version is that ``bpy`` is process-global, not thread-safe, and can take the
interpreter down rather than raise on the kind of non-manifold geometry
trellis-server routinely produces.

Everything that does not need Blender lives in ``rigging.py`` and is imported
from there, so the host and this process can never disagree about where a
joint goes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import meshreport, rigging
from . import sheet


def progress(frac: float, label: str) -> None:
    print(f"{rigging.PROGRESS_PREFIX} {frac:.3f} {label}", flush=True)


# --- scene helpers ----------------------------------------------------------


def _reset_scene(bpy: Any) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_glb(bpy: Any, path: Path) -> Any:
    """Import and return one joined mesh object.

    A trellis GLB is usually a single mesh, but nothing guarantees it, and
    ``parent_set`` skins the selection rather than the scene -- so joining
    first is what makes "rig the model" mean the whole model.
    """
    bpy.ops.import_scene.gltf(filepath=str(path))
    _purge_import_helpers(bpy)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{path.name} contains no mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    # The importer parents meshes under a rotation empty rather than baking the
    # Y-up -> Z-up conversion into the data. Applying it means the bbox we
    # measure and the bbox the exporter sees are the same one.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return mesh


# Blender's glTF importer parks helper objects it invents -- bone-shape widgets
# for an imported armature -- in a collection with this name, so that a
# re-export skips them. It is excluded from the *view layer*, so it never
# renders and is easy to miss; but it is still in scene.objects, and a unit
# icosphere sitting in there silently inflated the measured bounds of every
# rigged sprite sheet and framed the subject at a third of its proper size.
IMPORT_HELPER_COLLECTION = "glTF_not_exported"


def _purge_import_helpers(bpy: Any) -> None:
    collection = bpy.data.collections.get(IMPORT_HELPER_COLLECTION)
    if collection is None:
        return
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def _world_bounds(mesh: Any) -> tuple[list[float], list[float]]:
    from mathutils import Vector

    corners = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return lo, hi


def _build_armature(bpy: Any, bones: list[dict[str, Any]], name: str = "rig") -> Any:
    armature = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, armature)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    created: dict[str, Any] = {}
    for spec in bones:
        eb = armature.edit_bones.new(spec["name"])
        eb.head = spec["head"]
        eb.tail = spec["tail"]
        created[spec["name"]] = eb
    for spec in bones:
        parent = spec["parent"]
        if parent is None:
            continue
        eb = created[spec["name"]]
        eb.parent = created[parent]
        # Connect only where the joint actually coincides, so a shoulder
        # offset from the chest tail isn't yanked onto it.
        eb.use_connect = _close(eb.head, created[parent].tail)
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def _close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    return all(abs(a[i] - b[i]) <= tol for i in range(3))


def weld_distance(lo: Sequence[float], hi: Sequence[float]) -> float:
    """The merge-by-distance epsilon for a mesh with this bounding box.

    Relative to the model, and the same fraction ``meshreport`` welds its
    analysis copy by -- imported rather than restated, because the two are the
    same judgement about the same meshes: a UV-seam split carries an
    *identical* position, so any positive tolerance welds it, and the fraction
    is only insurance against a rewriter that round-tripped a position through
    float32. An absolute epsilon means something different on a 0.02 m gear
    than on a 30 m building.

    Zero (a degenerate bbox) means "do not weld", which the caller reads as an
    ordinary unwelded run rather than as a failure.
    """
    diagonal = sum((float(b) - float(a)) ** 2 for a, b in zip(lo, hi, strict=True)) ** 0.5
    return diagonal * meshreport.WELD_TOLERANCE


def _skin_steps(merged: int) -> tuple[tuple[str, bool], ...]:
    """The heat attempts to make after a weld merged ``merged`` vertices.

    Each step is ``(the method name to record, restore the pre-weld mesh
    first)``. Pure, and its own function for the reason ``_rig_bones`` is:
    everything around it needs bpy, so this is the part of the fallback chain a
    test can reach.

    A weld that merged *nothing* leaves exactly the mesh the heat solve would
    have seen anyway, so it is reported as plain ``automatic`` and there is no
    second, identical attempt to fall back to -- retrying it would be two
    minutes of Laplacian solve for a guaranteed repeat of the same answer.
    """
    if merged <= 0:
        return (("automatic", False),)
    return (("automatic-welded", False), ("automatic", True))


def _weld(bpy: Any, mesh: Any, distance: float) -> tuple[Any, int]:
    """Merge coincident vertices in place. -> (the pre-weld mesh data, merged).

    The hypothesis this exists for: trellis meshes are UV-atlased, and a
    seam-split vertex makes the surface non-manifold for bone-heat's Laplacian
    solve even though nothing is actually open -- the same root cause that made
    ``meshreport`` measure seams and call them holes. Welding by distance is
    what turns that back into a closed surface for the solve.

    It must not change what the user sees, and the whole argument that it does
    not is that Blender's merge-by-distance keeps face-corner data: UVs are
    per-loop, so the two halves of a seam keep their own texture coordinates
    while sharing one position. ``tests/test_rigging.py`` proves that on a real
    Blender rather than asserting it here.

    The pre-weld mesh datablock is returned rather than discarded, because the
    fallback chain may have to put it back: a welded solve that fails is not
    evidence the unwelded one would.
    """
    original = mesh.data.copy()
    before = len(mesh.data.vertices)
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=distance)
        # Heat weighting reads the surface, and a weld can leave two merged
        # shells disagreeing about which way is out.
        bpy.ops.mesh.normals_make_consistent(inside=False)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return original, before - len(mesh.data.vertices)


def _restore_mesh(bpy: Any, mesh: Any, original: Any) -> None:
    """Put the pre-weld geometry back on the object and drop the welded copy."""
    welded = mesh.data
    mesh.data = original
    bpy.data.meshes.remove(welded)


def _skin(bpy: Any, mesh: Any, arm_obj: Any, *, weld: float = 0.0) -> tuple[str, str | None]:
    """Bind mesh to armature, preferring heat weights. -> (method, why not).

    The chain is weld -> verify -> unwelded heat -> envelope, and every rung
    below the first is a degraded outcome that names itself.

    Bone-heat weighting solves a Laplacian over the surface and fails outright
    on non-manifold input, which describes a good share of trellis meshes -- in
    large part because they are UV-atlased, and an xatlas seam split is a
    non-manifold edge that is not a hole. ``weld`` is the distance to merge
    coincident vertices by first (see ``weld_distance``); zero skips the weld
    entirely and restores exactly the old behaviour.

    The failure is reported two different ways depending on Blender version (an
    operator RuntimeError, or a 'FINISHED' that quietly leaves every vertex
    group empty), so both are checked at every rung. Envelope weights are worse
    but they always exist, and a mediocre rig beats a failed job.

    The *reason* is returned rather than only printed. Envelope is a degraded
    outcome, not a second success, and while the only trace of it was a line on
    this subprocess's stdout a rig that quietly fell back was indistinguishable
    from one that did not -- which is exactly the state a user needs to be told
    about, because it is the one where the deformation will look wrong.
    ``None`` on either automatic path: there is nothing to explain about a
    solve that took.
    """
    def bind(kind: str) -> None:
        bpy.ops.object.select_all(action="DESELECT")
        mesh.select_set(True)
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.parent_set(type=kind)

    causes: list[str] = []
    original, merged = None, 0
    if weld > 0.0:
        try:
            original, merged = _weld(bpy, mesh, weld)
        except RuntimeError as exc:
            # The weld is an optimisation on the way to a rig, so its failure
            # costs the welded attempt and nothing else.
            causes.append(f"the weld pass failed: {str(exc).strip() or 'raised'}")
            print(f"weld before weighting failed: {exc}", flush=True)
        else:
            print(f"welded {merged} coincident vertice(s) before weighting", flush=True)

    for method, restore in _skin_steps(merged if original is not None else 0):
        if restore and original is not None:
            _restore_mesh(bpy, mesh, original)
            original = None
        try:
            bind("ARMATURE_AUTO")
            if _has_weights(mesh):
                if original is not None:
                    bpy.data.meshes.remove(original)
                return method, None
            causes.append(f"{method}: produced no vertex weights")
        except RuntimeError as exc:
            causes.append(f"{method}: {str(exc).strip() or 'raised'}")
        # parent_set already made the mesh a child; clear it so the next bind
        # doesn't stack a second armature modifier on top of the empty one.
        _unbind(bpy, mesh)

    if original is not None:
        # Envelope weights do not care whether the mesh is welded, but what is
        # exported should be the geometry the user's model.glb describes when
        # nothing was gained by changing it.
        _restore_mesh(bpy, mesh, original)
    reason = "bone-heat weighting failed: " + "; ".join(causes)
    print(f"{reason}; falling back to envelope", flush=True)
    bind("ARMATURE_ENVELOPE")
    return "envelope", reason


def _unbind(bpy: Any, mesh: Any) -> None:
    for mod in list(mesh.modifiers):
        if mod.type == "ARMATURE":
            mesh.modifiers.remove(mod)
    mesh.vertex_groups.clear()
    mesh.parent = None


def _has_weights(mesh: Any) -> bool:
    if not mesh.vertex_groups:
        return False
    groups = {g.index for g in mesh.vertex_groups}
    return any(g.group in groups and g.weight > 0.0 for v in mesh.data.vertices for g in v.groups)


def _export(bpy: Any, out_glb: Path) -> None:
    """Write the whole scene as a skinned GLB.

    ``export_rest_position_armature=False`` is what makes a posed export mean
    anything: left at its default the exporter writes every joint node at its
    rest transform and relegates the current pose to an animation track, so a
    baked pose would come back looking like a T-pose.
    """
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        export_skins=True,
        export_yup=True,
        export_animations=False,
        export_rest_position_armature=False,
    )


# --- posing -----------------------------------------------------------------


def _import_rig(bpy: Any, path: Path) -> Any:
    """Import a rig GLB and return its armature object.

    ``bone_heuristic="BLENDER"`` keeps the imported bone matrices identical to
    the ones the file was exported from. The default heuristic re-aims bones at
    their children for editing comfort, which changes ``matrix_local`` -- and
    ``matrix_local`` is exactly what :func:`_rest_local_rotation` reconstructs
    the glTF node frame from, so a re-aimed bone would silently rotate the pose.
    """
    bpy.ops.import_scene.gltf(filepath=str(path), bone_heuristic="BLENDER")
    _purge_import_helpers(bpy)
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError(f"{path.name} contains no armature")
    return armatures[0]


def _rest_local_rotation(bone: Any) -> Any:
    """The joint's rest rotation in the frame the browser sees it in.

    Blender composes a posed bone as::

        pose(b) = pose(parent) @ (rest(parent)^-1 @ rest(b)) @ basis(b)

    and a glTF joint node's local transform is ``pose(parent)^-1 @ pose(b)``,
    so ``node_local == rest_node_local @ basis``. That identity is the whole
    bridge between the two ends: three.js hands us ``node_local``, and
    ``basis = rest_node_local^-1 @ node_local`` is what Blender wants. No
    per-bone axis correction is involved, and the Z-up/Y-up conversion cancels
    because the exporter applies it once at the root, not per joint.
    """
    local = bone.matrix_local
    if bone.parent is not None:
        local = bone.parent.matrix_local.inverted() @ local
    return local.to_quaternion()


def _reset_pose(arm_obj: Any) -> None:
    """Back to rest. A pose is a partial map, so without this a later row of a
    sheet would inherit whatever the row above it left on the bones it omits."""
    for pbone in arm_obj.pose.bones:
        pbone.rotation_mode = "QUATERNION"
        pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pbone.location = (0.0, 0.0, 0.0)
        pbone.scale = (1.0, 1.0, 1.0)


def _apply_pose(arm_obj: Any, bones: dict[str, Any]) -> tuple[int, list[str]]:
    from mathutils import Quaternion

    applied = 0
    unknown: list[str] = []
    for name, quat in bones.items():
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            # Skipped, not fatal: a pose saved against one skeleton should
            # still mostly apply after a re-rig with a different template.
            unknown.append(name)
            continue
        x, y, z, w = (float(v) for v in quat)   # stored XYZW, three.js order
        pbone.rotation_mode = "QUATERNION"
        pbone.rotation_quaternion = _rest_local_rotation(pbone.bone).inverted() @ Quaternion(
            (w, x, y, z)
        )
        applied += 1
    return applied, unknown


# --- sprite-sheet rendering -------------------------------------------------
#
# Layout is decided in pipelines/sheet.py and packing happens back on the host;
# this end only renders one square, transparent frame per cell. The split keeps
# the grid arithmetic testable without Blender and keeps this function to the
# one thing only Blender can do.


def _scene_bounds(bpy: Any) -> tuple[list[float], list[float]]:
    from mathutils import Vector

    corners = [
        obj.matrix_world @ Vector(c)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
        for c in obj.bound_box
    ]
    if not corners:
        raise RuntimeError("nothing to render: the scene has no mesh")
    return (
        [min(c[i] for c in corners) for i in range(3)],
        [max(c[i] for c in corners) for i in range(3)],
    )


def _setup_render(bpy: Any, size: int) -> None:
    scene = bpy.context.scene
    # EEVEE was renamed in 4.2 and the old id is gone in 5.x. Assigning an
    # unknown enum raises, so try the current name first and fall back.
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    # The filepath already carries .png and is exact; letting Blender append
    # its own would give us model.png.png.
    scene.render.use_file_extension = False
    # Standard, not the default filmic-style transform: a sprite should come
    # out the colour the texture says it is. A tone curve here would quietly
    # desaturate every frame relative to the 3D preview beside it.
    with contextlib.suppress(TypeError):
        scene.view_settings.view_transform = "Standard"


def _world(bpy: Any, strength: float) -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("sheet_world")
    # Blender 5.0 deprecates World.use_nodes (worlds always have a node
    # tree there); only reach for it on older versions where a fresh world
    # has no tree until the flag is set.
    if scene.world.node_tree is None:
        scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs["Strength"].default_value = strength


def _make_flat(bpy: Any) -> None:
    """Rewire every material to emit its own base colour.

    Flat means unlit, not untextured: the albedo map still shows, it just
    receives no shading, which is the look most 2D pipelines expect from a
    sprite. Driving an Emission node from whatever fed Base Color keeps the
    texture and drops the lighting in one step.
    """
    # A mesh with no material at all falls back to Blender's default diffuse
    # surface, which under the black world flat mode uses renders as a
    # silhouette. Giving it a material of its own is the difference between a
    # sheet of the model and a sheet of its shadow.
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and not obj.data.materials:
            fallback = bpy.data.materials.new("flat_fallback")
            # Material.use_nodes is deprecated in Blender 5.0, where a new
            # material always carries a node tree already.
            if fallback.node_tree is None:
                fallback.use_nodes = True
            obj.data.materials.append(fallback)

    for material in bpy.data.materials:
        if material.node_tree is None:
            continue
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        principled = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if output is None or principled is None:
            continue
        emission = nodes.new("ShaderNodeEmission")
        base = principled.inputs["Base Color"]
        if base.is_linked:
            links.new(base.links[0].from_socket, emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = base.default_value
        links.new(emission.outputs["Emission"], output.inputs["Surface"])


def _make_lit(bpy: Any, centre: list[float], radius: float) -> None:
    """A conventional three-point setup, scaled to the subject."""
    distance = radius * 4.0
    for name, offset, energy in (
        ("key", (0.7, -0.9, 0.9), 5.0),
        ("fill", (-1.0, -0.5, 0.2), 1.6),
        ("rim", (0.1, 1.0, 0.7), 3.0),
    ):
        light = bpy.data.lights.new(name, type="SUN")
        light.energy = energy
        obj = bpy.data.objects.new(name, light)
        obj.location = [centre[i] + offset[i] * distance for i in range(3)]
        _aim_at(obj, centre)
        bpy.context.scene.collection.objects.link(obj)
    _world(bpy, 0.25)


def _aim_at(obj: Any, target: list[float]) -> None:
    from mathutils import Vector

    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _setup_camera(bpy: Any, extent: float, distance: float) -> Any:
    data = bpy.data.cameras.new("sheet_cam")
    data.type = "ORTHO"
    # One ortho_scale for every cell, so the subject stays the same size as it
    # turns instead of breathing between columns.
    data.ortho_scale = extent
    data.clip_start = max(distance * 0.001, 1e-4)
    data.clip_end = distance * 4.0
    cam = bpy.data.objects.new("sheet_cam", data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def _aim_camera(
    cam: Any, centre: list[float], yaw_deg: float, elevation_deg: float, distance: float
) -> None:
    """Place the camera on a turntable around the subject.

    Yaw 0 looks along +Y, i.e. from -Y, which is the front: the skeleton
    templates put the subject's forward direction at -Y, so column 0 of every
    sheet is the front view. Yaw increases clockwise seen from above.
    """
    import math

    from mathutils import Euler, Vector

    elevation = math.radians(elevation_deg)
    spin = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
    # A camera looks down its local -Z; rotating X by 90-elevation aims that at
    # +Y and tilts it down by the elevation, and the Z rotation then spins it.
    cam.rotation_euler = Euler((math.radians(90.0 - elevation_deg), 0.0, spin.z), "XYZ")
    forward = Vector((0.0, math.cos(elevation), -math.sin(elevation)))
    forward.rotate(spin)
    cam.location = Vector(centre) - forward * distance


def _project(bpy: Any, cam: Any, point: Sequence[float], size: int) -> tuple[float, float]:
    """World point -> pixel coordinates within one square frame.

    bpy_extras' own camera projection rather than reimplementing the ortho
    matrix: it already accounts for ortho_scale, the sensor fit and the aspect,
    and a hand-rolled version that disagreed would put every sprite's feet in
    the wrong place with nothing to indicate why.
    """
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    bpy.context.view_layer.update()
    ndc = world_to_camera_view(bpy.context.scene, cam, Vector(point))
    # world_to_camera_view returns 0..1 with y up; image pixels are y down.
    return (float(ndc.x) * size, (1.0 - float(ndc.y)) * size)


# --- operations -------------------------------------------------------------


def _rig_bones(
    spec: dict[str, Any], lo: Sequence[float], hi: Sequence[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The joints to build the armature from, and the record of where they
    came from. See ``rigging.rig_spec`` for the order of preference.

    Its own function because it is the only decision in ``op_rig``, and
    everything around it needs Blender -- so this is what a test can reach on a
    machine with no bpy.

    A ``template_bones`` list that does not name exactly this template's bones
    is ignored rather than trusted. The spec arrives over a pipe, and a bone
    list whose names do not match builds an armature whose parents do not
    resolve; falling back to the fit that is always available costs the
    informed placement and never the rig.
    """
    template = rigging.get_template(spec["template"])
    if spec.get("bones"):
        # Caller-supplied joints win over any fit. They are already validated
        # against the template host-side (rigging.validate_joints), so this is
        # a straight substitution rather than a second, disagreeing check.
        return spec["bones"], spec.get("fit") or {"method": "manual"}

    landmarks = spec.get("template_bones")
    informed = bool(landmarks) and {b["name"] for b in landmarks} == {
        b["name"] for b in template.bones
    }
    if landmarks and not informed:
        print(
            "template_bones does not name this template's bones; using the bbox fit",
            flush=True,
        )
    if informed:
        template = dataclasses.replace(template, bones=tuple(landmarks))
    fit = spec.get("fit") or {"method": "pose2d" if informed else "bbox"}
    return rigging.fit_template(template, lo, hi), fit


def _rig_meta(
    template: Any,
    *,
    bones: list[dict[str, Any]],
    lo: Any,
    hi: Any,
    weighting: str,
    weighting_reason: str | None,
    adjusted: bool,
    fit: dict[str, Any],
) -> dict[str, Any]:
    """Everything rig.json says about a rig, as a plain dict.

    Its own function for the reason ``_rig_bones`` is: everything around it in
    ``op_rig`` needs bpy, so pulling the *content* of the file out is what
    makes it assertable on a machine with no Blender -- which is every machine
    the app ships on.
    """
    return {
        "version": 1,
        "template": template.key,
        "label": template.label,
        "root": template.root,
        "weighting": weighting,
        # Additive beside ``weighting``, and no version bump with it for the
        # same reason ``fit`` needed none: every reader is .get-based, so a
        # rig.json written before this field stays readable and one written
        # after it stays readable by anything that has not heard of it. None on
        # the automatic path -- there is nothing to explain about a success.
        "weighting_reason": weighting_reason,
        "bounds": {"min": lo, "max": hi},
        "bones": bones,
        "mirror_pairs": [list(pair) for pair in template.mirror_pairs],
        "adjusted": adjusted,
        "fit": fit,
    }


def op_rig(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    template = rigging.get_template(spec["template"])
    source = Path(spec["source_glb"])
    if not source.exists():
        raise RuntimeError(f"no mesh to rig at {source}")

    progress(0.05, "Loading mesh")
    _reset_scene(bpy)
    mesh = _import_glb(bpy, source)

    progress(0.25, "Fitting skeleton")
    lo, hi = _world_bounds(mesh)
    bones, fit = _rig_bones(spec, lo, hi)
    arm_obj = _build_armature(bpy, bones)

    progress(0.40, "Computing weights")
    weighting, weighting_reason = _skin(bpy, mesh, arm_obj, weld=weld_distance(lo, hi))

    progress(0.85, "Exporting rig")
    _export(bpy, Path(spec["out_glb"]))

    rig_meta = _rig_meta(
        template,
        bones=bones,
        lo=lo,
        hi=hi,
        weighting=weighting,
        weighting_reason=weighting_reason,
        adjusted=bool(spec.get("bones")),
        fit=fit,
    )
    Path(spec["out_json"]).write_text(json.dumps(rig_meta, indent=2), encoding="utf-8")
    progress(1.0, "Rig complete")
    return {
        "ok": True,
        "weighting": weighting,
        "weighting_reason": weighting_reason,
        "bones": len(bones),
    }


def op_pose(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Bake one saved pose into its own GLB, next to the rig it came from."""
    rig_glb = Path(spec["rig_glb"])
    if not rig_glb.exists():
        raise RuntimeError(f"no rig to pose at {rig_glb}")

    progress(0.10, "Loading rig")
    _reset_scene(bpy)
    arm_obj = _import_rig(bpy, rig_glb)

    progress(0.50, "Applying pose")
    applied, unknown = _apply_pose(arm_obj, spec["bones"])
    if unknown:
        print(f"pose names {len(unknown)} bone(s) this rig does not have: {unknown}", flush=True)

    progress(0.70, "Exporting pose")
    _export(bpy, Path(spec["out_glb"]))
    progress(1.0, "Pose complete")
    return {"ok": True, "bones": applied, "unknown": unknown}


def op_sheet(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Render one PNG per sheet cell into ``frames_dir``.

    The host decided the grid and will do the packing; this walks the cell list
    in order, posing and spinning the camera as it goes. The framing is
    computed once from the *rest* bounding box and never touched again --
    reframing per pose would make the subject jump between rows of the finished
    sheet.
    """
    source = Path(spec["source_glb"])
    if not source.exists():
        raise RuntimeError(f"nothing to render at {source}")
    frames_dir = Path(spec["frames_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    size = int(spec["frame_size"])
    elevation = float(spec["elevation"])
    cells = spec["cells"]

    progress(0.05, "Loading model")
    _reset_scene(bpy)
    bpy.ops.import_scene.gltf(filepath=str(source), bone_heuristic="BLENDER")
    _purge_import_helpers(bpy)
    armature = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)

    lo, hi = _scene_bounds(bpy)
    centre = [(a + b) / 2.0 for a, b in zip(lo, hi, strict=True)]
    span = [b - a for a, b in zip(lo, hi, strict=True)]
    # The widest the subject can look from any yaw is its horizontal diagonal;
    # sizing to that rather than to a single axis is what stops the corner
    # views clipping.
    # sheet.FRAME_MARGIN, not a literal and not spec.get("margin", ...): nothing
    # has ever written `margin` into a sheet spec, so that fallback was the only
    # value it ever took -- and it was a second copy of the figure the in-app
    # preview frames with, which has to match this one.
    extent = max((span[0] ** 2 + span[1] ** 2) ** 0.5, span[2], 1e-6) * sheet.FRAME_MARGIN
    distance = extent * 2.0

    progress(0.10, "Setting up")
    _setup_render(bpy, size)
    if spec["lighting"] == "lit":
        _make_lit(bpy, centre, extent / 2.0)
    else:
        _make_flat(bpy)
        _world(bpy, 0.0)
    cam = _setup_camera(bpy, extent, distance)

    # The subject's ground origin: horizontally centred, sitting on the bbox
    # floor. Projected once, from yaw 0, because the ortho camera is framed once
    # and only spins -- so this pixel is the same in every direction, which is
    # exactly what makes it usable as a sprite pivot. Aiming here also keeps the
    # projection out of the render loop.
    _aim_camera(cam, centre, 0.0, elevation, distance)
    pivot = _project(bpy, cam, (centre[0], centre[1], lo[2]), size)

    posed: Any = "__rest__"
    rendered = []
    for i, cell in enumerate(cells):
        # Cells arrive grouped by row, so this re-poses once per row rather than
        # once per frame -- eight times less work on an eight-yaw sheet. A
        # clip's rows differ only by frame, which is why the cache key carries
        # it: without that every frame of a clip would render the first one.
        key = (cell.get("pose"), cell.get("frame", 0))
        if armature is not None and key != posed:
            _reset_pose(armature)
            _apply_pose(armature, cell.get("bones") or {})
            posed = key
        _aim_camera(cam, centre, float(cell["yaw"]), elevation, distance)
        out = frames_dir / f"{cell['index']:04d}.png"
        bpy.context.scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        rendered.append(cell["index"])
        progress(0.10 + 0.85 * (i + 1) / max(len(cells), 1), f"Rendering {i + 1}/{len(cells)}")

    progress(1.0, "Frames rendered")
    return {
        "ok": True,
        "frames": rendered,
        "bounds": {"min": lo, "max": hi},
        "pivot": list(pivot),
    }


def op_fbx(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Import a GLB and write it back out as FBX, skins and all."""
    source = Path(spec["source_glb"])
    if not source.exists():
        raise RuntimeError(f"nothing to convert at {source}")

    progress(0.10, "Loading model")
    _reset_scene(bpy)
    bpy.ops.import_scene.gltf(filepath=str(source), bone_heuristic="BLENDER")
    _purge_import_helpers(bpy)

    progress(0.60, "Writing FBX")
    out = Path(spec["out_fbx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(out),
        use_selection=True,
        path_mode="COPY",
        embed_textures=True,
        # Unity and Unreal both read Y-up FBX; matching the GLB's axes means the
        # FBX and the GLB describe the same orientation rather than two.
        axis_forward="-Z",
        axis_up="Y",
        bake_anim=False,
    )
    progress(1.0, "FBX written")
    return {"ok": True, "objects": len(bpy.context.scene.objects)}


# The scratch UV layer a projection lands in. The mesh's own atlas stays the
# active layer and is what every bake writes *into*; this only ever carries one
# view at a time and is rebuilt per view.
PROJECT_UV = "wl_proj"


def _view_direction(yaw: float, pitch: float) -> tuple[float, float, float]:
    """The unit direction the camera sits in, in Blender axes.

    ``pipelines.retexture.view_matrix`` is the same arithmetic and is the one a
    test can reach without bpy; this is the worker's copy, which imports
    nothing from the host half by design -- this module runs inside a bpy
    interpreter and `rigging.py`'s split is what keeps that one-way.
    ``tests/test_retexture.py`` pins the two against each other, which is the
    same treatment ``rigging.fit_template`` gets for the same reason.
    """
    import math

    y, p = math.radians(yaw), math.radians(pitch)
    return (math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p))


def _retexture_frame(bpy: Any, source: Path, size: int):
    """Import, measure, and frame the one camera both re-texture ops use.

    Shared rather than written twice because the two ops have to agree about
    that camera *exactly*: ``op_views`` renders through it and ``op_project``
    projects through it, and a framing that differed by a pixel between them
    would shift the whole atlas by that pixel with nothing on screen to say
    why. -> (mesh, centre, extent, distance)
    """
    _reset_scene(bpy)
    mesh = _import_glb(bpy, source)
    lo, hi = _world_bounds(mesh)
    centre = [(a + b) / 2.0 for a, b in zip(lo, hi, strict=True)]
    span = [b - a for a, b in zip(lo, hi, strict=True)]
    # The horizontal diagonal, as op_sheet sizes to: one axis clips the corner
    # views, and here a clipped view is a strip of atlas nothing covers.
    extent = max((span[0] ** 2 + span[1] ** 2) ** 0.5, span[2], 1e-6) * 1.05
    distance = extent * 2.0
    _setup_render(bpy, size)
    return mesh, centre, extent, distance


def op_views(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Render the mesh once per view direction, flat.

    **Flat, not lit, and that is the load-bearing choice.** These renders are
    restyled and then baked back into the *albedo*, so any shading in them
    becomes shading painted permanently into the texture -- a highlight that
    stays put as the object turns, which is the one artefact a base-colour map
    must not have. ``_make_flat`` keeps the existing texture and drops the
    lighting, which is exactly the signal an img2img pass should be restyling.

    The alpha matters as much as the colour: ``film_transparent`` leaves the
    background clear, and ``op_project`` uses that alpha as the mask saying
    which texels this view is entitled to speak about at all.
    """
    source = Path(spec["source_glb"])
    if not source.exists():
        raise RuntimeError(f"nothing to render at {source}")
    # Resolved, and that is not tidiness. ``render.filepath`` is one of the
    # paths Blender interprets *itself*, relative to the .blend file rather
    # than to the process's directory -- and there is no .blend file here, so a
    # relative path renders successfully and saves the PNG somewhere the caller
    # will never look. Nothing raises: ``bpy.ops.render.render`` reports
    # completion and ``op_views`` returns ok. ``op_sheet`` has always been safe
    # only because its caller hands it a TemporaryDirectory, which is absolute
    # by construction.
    views_dir = Path(spec["views_dir"]).resolve()
    views_dir.mkdir(parents=True, exist_ok=True)
    views = spec["views"]

    progress(0.05, "Loading model")
    _mesh, centre, extent, distance = _retexture_frame(bpy, source, int(spec["size"]))
    _make_flat(bpy)
    _world(bpy, 0.0)
    cam = _setup_camera(bpy, extent, distance)

    for i, (yaw, pitch) in enumerate(views):
        _aim_camera(cam, centre, float(yaw), float(pitch), distance)
        bpy.context.scene.render.filepath = str(views_dir / f"view_{i:02d}.png")
        bpy.ops.render.render(write_still=True)
        progress(0.05 + 0.9 * (i + 1) / max(len(views), 1), f"View {i + 1}/{len(views)}")

    progress(1.0, "Views rendered")
    return {"ok": True, "views": len(views), "extent": extent}


def _project_material(bpy: Any, colour_png: Path, mask_png: Path, direction):
    """One material carrying both bakes. -> (material, colour_emit, weight_emit)

    Both emissions share one projection and one pair of textures, so switching
    which node feeds the output is the whole difference between the colour bake
    and the weight bake -- they cannot come to disagree about where the view
    landed.

    The mask is the **original** render's alpha rather than the restyled one's,
    because img2img returns RGB and drops it. Two textures over one UV layer,
    which also makes "outside the camera frustum" free: ``CLIP`` extension
    returns alpha 0 out there, so a texel the camera never saw gets weight 0
    without a frustum test of its own.
    """
    material = bpy.data.materials.new("wl_project")
    if material.node_tree is None:
        material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")

    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = PROJECT_UV

    colour_tex = tree.nodes.new("ShaderNodeTexImage")
    colour_tex.image = bpy.data.images.load(str(colour_png))
    colour_tex.extension = "CLIP"
    tree.links.new(uv.outputs["UV"], colour_tex.inputs["Vector"])

    mask_tex = tree.nodes.new("ShaderNodeTexImage")
    mask_tex.image = bpy.data.images.load(str(mask_png))
    mask_tex.extension = "CLIP"
    tree.links.new(uv.outputs["UV"], mask_tex.inputs["Vector"])

    colour_emit = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(colour_tex.outputs["Color"], colour_emit.inputs["Color"])

    # facing = max(0, dot(N, the direction the camera is in)), masked by the
    # render's own alpha. Clamped at zero rather than made absolute: a face
    # pointing away from this camera is not "seen from behind", it is not seen.
    geo = tree.nodes.new("ShaderNodeNewGeometry")
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = direction
    tree.links.new(geo.outputs["Normal"], dot.inputs[0])
    clamp = tree.nodes.new("ShaderNodeMath")
    clamp.operation = "MAXIMUM"
    clamp.inputs[1].default_value = 0.0
    tree.links.new(dot.outputs["Value"], clamp.inputs[0])
    masked = tree.nodes.new("ShaderNodeMath")
    masked.operation = "MULTIPLY"
    tree.links.new(clamp.outputs["Value"], masked.inputs[0])
    tree.links.new(mask_tex.outputs["Alpha"], masked.inputs[1])
    weight_emit = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(masked.outputs["Value"], weight_emit.inputs["Color"])

    tree.links.new(colour_emit.outputs["Emission"], out.inputs["Surface"])
    return material, colour_emit, weight_emit


def op_project(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Bake each restyled view into the atlas, with a weight image beside it.

    One pair of images per view, combined on the host by
    ``pipelines.retexture.assemble`` -- deliberately not accumulated here,
    because a weighted mean is arithmetic and belongs where ``sheet.py``'s grid
    does.

    The projection is Blender's own UVProject modifier onto a scratch UV layer,
    applied per view and rebuilt for the next. The mesh's real atlas stays the
    *active* layer throughout, because that is the one every bake writes into
    -- leaving the scratch layer active would bake the projection into itself.
    """
    source = Path(spec["source_glb"])
    if not source.exists():
        raise RuntimeError(f"nothing to project onto at {source}")
    # Both resolved, for op_views' reason: ``Image.filepath_raw`` is the same
    # kind of path as ``render.filepath`` and Blender resolves it the same way,
    # so a relative out_dir saves every bake somewhere the host will not find
    # and reports success doing it.
    views_dir = Path(spec["views_dir"]).resolve()
    out_dir = Path(spec["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    views = spec["views"]
    texture_size = int(spec["texture_size"])

    progress(0.05, "Loading model")
    mesh, centre, extent, distance = _retexture_frame(bpy, source, int(spec["size"]))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    # An emission bake carries no noise, so one sample is the whole budget.
    scene.cycles.samples = 1
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    # No margin: the host dilates, and it has to, because a margin Blender
    # grew per view would be grown from that view's colours before the views
    # were ever combined.
    scene.render.bake.margin = 0

    if not mesh.data.uv_layers:
        raise RuntimeError("this mesh has no UVs to bake into")
    atlas_uv = mesh.data.uv_layers.active.name
    cam = _setup_camera(bpy, extent, distance)

    original = list(mesh.data.materials)
    done = []
    for i, (yaw, pitch) in enumerate(views):
        colour_png = views_dir / f"restyled_{i:02d}.png"
        mask_png = views_dir / f"view_{i:02d}.png"
        if not colour_png.exists() or not mask_png.exists():
            # A view whose restyle never arrived contributes nothing rather
            # than failing the bake: five good projections beat none.
            continue
        _aim_camera(cam, centre, float(yaw), float(pitch), distance)

        if PROJECT_UV in mesh.data.uv_layers:
            mesh.data.uv_layers.remove(mesh.data.uv_layers[PROJECT_UV])
        mesh.data.uv_layers.new(name=PROJECT_UV)
        modifier = mesh.modifiers.new("wl_project", "UV_PROJECT")
        modifier.uv_layer = PROJECT_UV
        modifier.projector_count = 1
        modifier.projectors[0].object = cam
        modifier.aspect_x = modifier.aspect_y = 1.0
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.modifier_apply(modifier="wl_project")
        mesh.data.uv_layers.active = mesh.data.uv_layers[atlas_uv]

        material, colour_emit, weight_emit = _project_material(
            bpy, colour_png, mask_png, _view_direction(float(yaw), float(pitch))
        )
        mesh.data.materials.clear()
        mesh.data.materials.append(material)
        tree = material.node_tree
        out_node = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")

        for suffix, emit in (("bake", colour_emit), ("weight", weight_emit)):
            for link in list(out_node.inputs["Surface"].links):
                tree.links.remove(link)
            tree.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
            image = bpy.data.images.new(
                f"wl_{suffix}_{i}", texture_size, texture_size, alpha=False
            )
            node = tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            tree.nodes.active = node
            bpy.ops.object.bake(type="EMIT")
            image.filepath_raw = str(out_dir / f"{suffix}_{i:02d}.png")
            image.file_format = "PNG"
            image.save()
            tree.nodes.remove(node)

        mesh.data.materials.clear()
        done.append(i)
        progress(0.05 + 0.9 * (i + 1) / max(len(views), 1), f"Baking {i + 1}/{len(views)}")

    for material in original:
        mesh.data.materials.append(material)
    progress(1.0, "Projections baked")
    return {"ok": True, "baked": done, "uv_layer": atlas_uv}


OPS = {
    "rig": op_rig,
    "pose": op_pose,
    "sheet": op_sheet,
    "fbx": op_fbx,
    "views": op_views,
    "project": op_project,
}


def main() -> int:
    spec = json.loads(sys.stdin.read())
    op = OPS.get(spec.get("op"))
    if op is None:
        print(f"unknown op {spec.get('op')!r}", file=sys.stderr)
        return 2
    try:
        import bpy
    except ImportError as exc:
        print(f"Blender (bpy) is not installed: {exc}", file=sys.stderr)
        return 3
    result = op(bpy, spec)
    Path(spec["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

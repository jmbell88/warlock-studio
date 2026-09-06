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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import meshreport, poselib, rigging
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
    try:
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
    except BaseException:
        # ``remove_doubles`` raising is a real failure mode -- ``_skin`` catches
        # the ``RuntimeError`` and carries on down the fallback chain -- and the
        # copy taken two lines up is the caller's only by *return*. A raise
        # never returns it, so nothing downstream had a reference to free and
        # the datablock stayed resident for the life of the subprocess.
        with contextlib.suppress(Exception):
            bpy.data.meshes.remove(original)
        raise
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
        _unbind(mesh)

    if original is not None:
        # Envelope weights do not care whether the mesh is welded, but what is
        # exported should be the geometry the user's model.glb describes when
        # nothing was gained by changing it.
        _restore_mesh(bpy, mesh, original)
    reason = "bone-heat weighting failed: " + "; ".join(causes)
    print(f"{reason}; falling back to envelope", flush=True)
    bind("ARMATURE_ENVELOPE")
    return "envelope", reason


def _unbind(mesh: Any) -> None:
    """Every armature modifier, its weights and its parenting, off one mesh.

    No ``bpy``: it took one and never used it, which in this module is a
    misleading signature rather than a harmless one -- ``bpy`` in a parameter
    list is how every function here says it touches the global Blender state,
    and this one only walks the mesh it was handed."""
    for mod in list(mesh.modifiers):
        if mod.type == "ARMATURE":
            mesh.modifiers.remove(mod)
    mesh.vertex_groups.clear()
    mesh.parent = None


def _strip_incoming_rig(bpy: Any, mesh: Any) -> int:
    """Drop any skin and skeleton the source GLB brought with it. -> bones removed.

    **Every mesh this path had ever seen was a TRELLIS reconstruction**, which
    carries no armature and no vertex groups, so nothing here was needed and
    nothing noticed it was missing. A *user-supplied* base mesh is the case the
    Troupe intake exists for, and a supplied humanoid usually arrives rigged --
    at which point two things go wrong at once, neither of them loudly.

    **``_skin``'s guard stops working.** Its contract is that bone-heat
    weighting reports failure two ways, one of them a ``FINISHED`` that quietly
    leaves every vertex group empty, and ``_has_weights`` is what catches the
    quiet one. That check asks whether *any* group holds a weight -- so an
    incoming skin answers yes before the new armature has been bound at all,
    and a bind that produced nothing is reported as a clean ``automatic`` rig.
    The user is then told the rig succeeded, and the character does not deform.

    **The old skeleton is exported beside the new one.** ``_import_glb``
    returns the joined mesh and leaves the rest of the scene alone, while
    ``_export`` writes *the whole scene*; the result is a GLB carrying two
    armatures, one of which nothing is weighted to.

    **And the measurements are wrong, which is the worst of the three.**
    ``_import_glb`` bakes the Y-up -> Z-up rotation into the vertex data, but a
    skinned import parents the mesh to its armature and *that* still carries
    the rotation -- so ``matrix_world`` applies it a second time and
    ``_world_bounds`` returns a box rotated once too far. Measured on
    CesiumMan: ``(0.505, 0.896, 1.458)`` against a true
    ``(1.138, 0.312, 1.507)``, an arm span reported at under half its real
    width. ``_rig_bones`` fits the template to that box, so every joint lands
    in the wrong place while the height stays plausible enough to look fine.
    Unparenting is what makes the two agree, which is why this must run
    *before* ``_world_bounds`` and not merely before ``_skin``.

    None of the three can happen to a TRELLIS reconstruction: no skin, no
    armature, no parent. All three happen to a supplied humanoid.

    Discarding rather than adopting is deliberate. Bone names would have to map
    onto the shipped template, and a supplied rig generally does not: CesiumMan
    is 19 bones like the template and still does not fit it -- 3 per arm and 4
    per leg against the template's 4 and 3. Warlock fits its own skeleton, and
    the one the file arrived with is not evidence about where those joints go.
    """
    _unbind(mesh)
    removed = 0
    for obj in [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]:
        removed += len(obj.data.bones)
        bpy.data.objects.remove(obj, do_unlink=True)
    if removed:
        print(f"discarded an incoming rig of {removed} bone(s)", flush=True)
    return removed


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


def _apply_root_translation(arm_obj: Any, bone_name: Any, offset_world: Sequence[float]) -> bool:
    """Displace one pose bone by a *world-space* offset. -> whether it applied.

    ``pbone.location`` lives in the bone's own rest frame, so the world offset
    is carried through the inverse of the bone's world rest orientation. For a
    parentless root -- the only bone anything writes an offset against today --
    the exporter then emits ``rest + d`` as the node translation exactly.

    An unknown bone is reported like ``_apply_pose``'s unknowns, never fatal: a
    library pose applied after a re-rig with a different template should cost
    the offset, not the bake.
    """
    from mathutils import Vector

    pbone = arm_obj.pose.bones.get(str(bone_name or ""))
    if pbone is None:
        print(f"root offset names a bone this rig does not have: {bone_name!r}", flush=True)
        return False
    if pbone.parent is not None:
        # The inverse below carries the offset through the bone's *rest* frame
        # only; a parented bone composes through its parent's pose, which this
        # arithmetic never sees. Every shipped template's root is parentless
        # (enforced at registry load in rigging._parse_template), so this is a
        # foreign or hand-edited rig.json -- it costs the offset, not the bake,
        # the same rule as an unknown bone above.
        print(f"root offset bone {bone_name!r} has a parent; skipping the offset", flush=True)
        return False
    pbone.location = (arm_obj.matrix_world @ pbone.bone.matrix_local).to_3x3().inverted() @ Vector(
        [float(v) for v in offset_world]
    )
    return True


#: The two frames a stored rotation can be in, and they are not interchangeable.
#:
#: ``node`` is what the pose editor saves and every shipped *pose* uses: the
#: joint's orientation in the glTF node frame, i.e. **absolute** relative to its
#: parent joint. Identity there does not mean "at rest", it means "aligned with
#: the parent", which is why :func:`_rest_local_rotation` has to be divided out.
#:
#: ``delta`` is a rotation **from the bone's own rest orientation**, which is
#: exactly what Blender's pose basis already is -- so it applies with no
#: correction at all. Clips are authored in it because an author thinks in it
#: ("swing the thigh forward 24 degrees") and, more importantly, because it is
#: the frame that survives a *re-fit*: a node-local value bakes in the rest
#: orientation of the skeleton it was authored against, so the same numbers on a
#: rig whose joints were measured off the mesh rather than fitted to its bbox
#: produce a different -- and usually broken -- pose.
POSE_SPACES = ("node", "delta")


def _apply_pose(
    arm_obj: Any, bones: dict[str, Any], space: str = "node"
) -> tuple[int, list[str]]:
    from mathutils import Quaternion

    applied = 0
    unknown: list[str] = []
    delta = str(space) == "delta"
    for name, quat in bones.items():
        pbone = arm_obj.pose.bones.get(name)
        if pbone is None:
            # Skipped, not fatal: a pose saved against one skeleton should
            # still mostly apply after a re-rig with a different template.
            unknown.append(name)
            continue
        node = [float(v) for v in quat]   # stored XYZW, three.js order
        pbone.rotation_mode = "QUATERNION"
        if delta:
            x, y, z, w = node
        else:
            # ``rigging.delta_from_node``, not a local ``inverted() @``: a pose
            # bone's ``rotation_quaternion`` *is* a rotation from rest, and the
            # host's pose editor makes the same conversion against the viewer's
            # rest quaternions. The order and which side is conjugated are the
            # parts that drift, and a drifted one contorts a skeleton silently
            # -- so there is one definition and both ends call it.
            rest = _rest_local_rotation(pbone.bone)
            x, y, z, w = rigging.delta_from_node(
                [rest.x, rest.y, rest.z, rest.w], node
            )
        pbone.rotation_quaternion = Quaternion((w, x, y, z))
        applied += 1
    return applied, unknown


# --- sprite-sheet rendering -------------------------------------------------
#
# Layout is decided in pipelines/sheet.py and packing happens back on the host;
# this end only renders one square, transparent frame per cell. The split keeps
# the grid arithmetic testable without Blender and keeps this function to the
# one thing only Blender can do.


def _render_meshes(bpy: Any) -> list[Any]:
    """The objects a render actually shows.

    Its own function because two callers now share the predicate -- the rest
    bounds and the posed union below -- and a union taken over a *different*
    set of objects than the rest box would frame from two disagreeing subjects.
    ``hide_render`` is what keeps Blender's glTF importer's bone-shape widgets
    out; see ``test_a_rigged_subject_is_framed_by_its_own_size``.
    """
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]


def _scene_bounds(bpy: Any) -> tuple[list[float], list[float]]:
    """The rest bounding box of everything that renders.

    Through ``_transform`` rather than ``mathutils`` since 2026-09-05, when the
    posed union arrived: ``_union_framing`` takes a ``max`` over corners from
    *both* sources, and two arithmetics for one corner is a difference that
    would show up as a hair of framing nobody could account for. (mathutils
    vectors are single precision; this is double, so the rest box moved by
    about a part in 10^7 -- far below a pixel at any frame size, and the
    2026-09-05 union-framing measurement records it.)
    """
    corners = [
        _transform(obj.matrix_world, c)
        for obj in _render_meshes(bpy)
        for c in obj.bound_box
    ]
    if not corners:
        raise RuntimeError("nothing to render: the scene has no mesh")
    return (
        [min(c[i] for c in corners) for i in range(3)],
        [max(c[i] for c in corners) for i in range(3)],
    )


def _transform(matrix: Any, point: Sequence[float]) -> tuple[float, float, float]:
    """A 4x4 row-major matrix applied to a point, in plain arithmetic.

    ``mathutils`` is not imported here on purpose: everything below this line
    that decides the sheet's *framing* has to be reachable from the ordinary
    (bpy-less) test lane, and a matrix-point product is three dot products. The
    rows of a ``mathutils.Matrix`` index exactly like the nested sequences a
    test hands it.
    """
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    return tuple(  # type: ignore[return-value]
        float(matrix[r][0]) * x
        + float(matrix[r][1]) * y
        + float(matrix[r][2]) * z
        + float(matrix[r][3])
        for r in range(3)
    )


def _evaluated_corners(bpy: Any, meshes: Sequence[Any]) -> list[tuple[float, float, float]]:
    """World-space bounding-box corners of ``meshes`` **as currently posed**.

    ``_scene_bounds`` reads ``obj.bound_box`` off the original object, and for
    a skinned mesh that is its *rest* box: armature deformation is a modifier,
    so the posed extent exists only on the depsgraph-evaluated copy. Framing a
    whole sheet from the rest box is what clipped the top off every overhead
    attack wind and every jump apex -- on every cell of the run, because the
    camera is framed once.

    ``view_layer.update()`` is the caller's job: it has just applied a pose and
    knows whether anything moved.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners: list[tuple[float, float, float]] = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        matrix = evaluated.matrix_world
        corners.extend(_transform(matrix, corner) for corner in evaluated.bound_box)
    return corners


def _socket_world_point(
    arm_obj: Any, socket: Mapping[str, Any]
) -> tuple[float, float, float] | None:
    """Where a named socket sits in the world, in the pose that is applied now.

    A socket is ``{"bone", "offset": [along, lateral, up], "reach"}`` with the
    offset in *bone-length* units, so it survives a re-fit onto a character of
    a different size -- the same reasoning ``delta`` pose space is authored
    under. Blender puts a bone's own +Y along the bone, +X lateral and +Z up,
    which is the order the offset is written in.

    An unknown bone costs the socket and never the sheet, the rule
    ``_apply_pose`` and ``_apply_root_translation`` already follow: a socket
    list authored against one template applied after a re-rig should lose the
    attachment point, not 256 frames.
    """
    pbone = arm_obj.pose.bones.get(str(socket.get("bone") or ""))
    if pbone is None:
        print(f"socket names a bone this rig does not have: {socket.get('bone')!r}", flush=True)
        return None
    along, lateral, up = (list(socket.get("offset") or (0.0, 0.0, 0.0)) + [0.0, 0.0, 0.0])[:3]
    length = float(pbone.bone.length)
    local = (float(lateral) * length, float(along) * length, float(up) * length)
    # pose-bone matrix is armature-object space; the object's own transform
    # carries it the rest of the way.
    return _transform(arm_obj.matrix_world, _transform(pbone.matrix, local))


def _sphere_corners(
    point: Sequence[float], radius: float
) -> list[tuple[float, float, float]]:
    """The eight corners of the box around a sphere of ``radius`` at ``point``.

    A flame or a muzzle flash drawn at a socket has extent of its own, and
    framing to the body alone would clip it. Cube corners rather than the six
    axis extremes: over-covering a sphere by its diagonal frames a little wide,
    which is the safe direction, where under-covering it clips.
    """
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    r = float(radius)
    return [
        (x + sx * r, y + sy * r, z + sz * r)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def _union_framing(
    rest_lo: Sequence[float],
    rest_hi: Sequence[float],
    corners: Sequence[Sequence[float]],
    *,
    margin: float,
) -> tuple[list[float], float]:
    """The ortho window that holds every posed corner. -> ``(centre, extent)``.

    **The orbit axis stays the rest ground origin.** Only the window widens:
    ``centre``'s x and y are the *rest* box's, so the projected pivot -- which
    sits on that axis -- lands on the same pixel at every yaw, which is the one
    property an engine placing a sprite by it depends on. Letting the axis
    follow the union would make the pivot drift as the character turns, which
    is a worse defect than the clipping this fixes.

    ``extent`` is the larger of the union's full height and twice the furthest
    any corner reaches *from that axis* horizontally -- the widest the subject
    can look from any yaw, which is the argument the single-bbox version made
    with its horizontal diagonal. On a rest-only sheet the union is the rest
    box and this returns exactly what that arithmetic did, to the bit.
    """
    import math

    cx = (float(rest_lo[0]) + float(rest_hi[0])) / 2.0
    cy = (float(rest_lo[1]) + float(rest_hi[1])) / 2.0
    points = [tuple(float(v) for v in c) for c in corners]
    points.extend(
        (x, y, z)
        for x in (float(rest_lo[0]), float(rest_hi[0]))
        for y in (float(rest_lo[1]), float(rest_hi[1]))
        for z in (float(rest_lo[2]), float(rest_hi[2]))
    )
    lo_z = min(p[2] for p in points)
    hi_z = max(p[2] for p in points)
    radius = max(math.hypot(p[0] - cx, p[1] - cy) for p in points)
    extent = max(2.0 * radius, hi_z - lo_z, 1e-6) * float(margin)
    return [cx, cy, (lo_z + hi_z) / 2.0], extent


def _view_forward(yaw_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """The unit vector the camera looks *along*, for one turntable position.

    The same construction ``_aim_camera`` places the camera with, restated in
    plain arithmetic for the same reason ``_transform`` is: depth ordering is
    part of the framing decision and has to be testable without Blender. Yaw 0
    looks along +Y.
    """
    import math

    elevation = math.radians(float(elevation_deg))
    yaw = math.radians(float(yaw_deg))
    fx, fy, fz = 0.0, math.cos(elevation), -math.sin(elevation)
    return (
        fx * math.cos(yaw) - fy * math.sin(yaw),
        fx * math.sin(yaw) + fy * math.cos(yaw),
        fz,
    )


def _view_depth(
    centre: Sequence[float],
    point: Sequence[float],
    *,
    yaw_deg: float,
    elevation_deg: float,
    distance: float,
) -> float:
    """How far ``point`` is from the camera, measured along the view direction.

    Not the straight-line distance: an orthographic camera has no eye point to
    measure from, and what an overlay compositor needs is the ordering along
    the view axis. ``_aim_camera`` puts the camera at ``centre - forward *
    distance``, so this is ``dot(point - centre, forward) + distance`` -- a
    positive number that grows away from the viewer, which is what makes
    ``behind`` a plain ``>``.
    """
    forward = _view_forward(yaw_deg, elevation_deg)
    return float(distance) + sum(
        (float(point[i]) - float(centre[i])) * forward[i] for i in range(3)
    )


def _pose_union(
    bpy: Any,
    armature: Any,
    cells: Sequence[Mapping[str, Any]],
    sockets: Sequence[Mapping[str, Any]],
    *,
    rest_height: float,
) -> tuple[
    list[tuple[float, float, float]],
    dict[Any, dict[str, tuple[float, float, float]]],
    dict[Any, tuple[float, float, float]],
]:
    """Walk every distinct pose the sheet contains and measure what it reaches.

    Returns the union's corners, each pose's socket world points, and each
    pose's own body centre (what ``behind`` is measured against below).

    Keyed on ``(pose, frame)`` -- the render loop's own cache key, reused
    deliberately: measuring a key the loop would not pose would frame for a
    pose the sheet does not contain, and measuring fewer would clip. Every
    frame of a clip shares a pose id, which is why the frame is in the key.

    The pose is left at rest afterwards, so the render loop's ``posed`` cache
    starts from the state it claims to.
    """
    meshes = _render_meshes(bpy)
    corners: list[tuple[float, float, float]] = []
    socket_points: dict[Any, dict[str, tuple[float, float, float]]] = {}
    body_centres: dict[Any, tuple[float, float, float]] = {}
    seen: set[Any] = set()
    for cell in cells:
        key = (cell.get("pose"), cell.get("frame", 0))
        if key in seen:
            continue
        seen.add(key)
        if armature is not None:
            # Exactly what the render loop does, in the same order: a pose
            # measured differently from the way it is rendered is a window
            # sized for a picture nobody gets.
            _reset_pose(armature)
            _apply_pose(
                armature, cell.get("bones") or {}, str(cell.get("pose_space") or "node")
            )
            if cell.get("root_offset"):
                _apply_root_translation(armature, cell.get("root_bone"), cell["root_offset"])
        bpy.context.view_layer.update()
        mine = _evaluated_corners(bpy, meshes)
        if not mine:
            raise RuntimeError("nothing to render: the scene has no mesh")
        corners.extend(mine)
        body_centres[key] = tuple(  # type: ignore[assignment]
            (min(c[i] for c in mine) + max(c[i] for c in mine)) / 2.0 for i in range(3)
        )
        if sockets and armature is not None:
            points: dict[str, tuple[float, float, float]] = {}
            for socket in sockets:
                point = _socket_world_point(armature, socket)
                if point is None:
                    continue
                points[str(socket.get("name") or socket.get("bone"))] = point
                reach = float(socket.get("reach") or 0.0) * float(rest_height)
                if reach > 0.0:
                    corners.extend(_sphere_corners(point, reach))
            socket_points[key] = points
    if armature is not None:
        _reset_pose(armature)
        bpy.context.view_layer.update()
    return corners, socket_points, body_centres


def _setup_render(bpy: Any, size: int, *, taa_samples: int | None = None) -> None:
    """Render settings shared by the sheet and the view-bake paths.

    ``taa_samples`` is the crispness knob. A native low-res render comes back
    antialiased and soft otherwise, which is a shrunk render rather than pixel
    art -- and the softness survives the reduction, because a partial-alpha
    fringe is exactly what the alpha snap then has to guess about. One sample
    is right wherever the surface emits rather than shades: an emission render
    has no noise for TAA to average away, which is the argument ``op_views``
    already makes at its own call. Left alone by default, because a *lit*
    sheet does have noise and one sample would show it.
    """
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
    if taa_samples is not None:
        with contextlib.suppress(AttributeError):
            scene.eevee.taa_render_samples = int(taa_samples)


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
    _strip_incoming_rig(bpy, mesh)

    progress(0.25, "Fitting skeleton")
    lo, hi = _world_bounds(mesh)
    if spec.get("joints") == "measured" and not spec.get("bones"):
        # Measured off the geometry rather than scaled to its box. Done here
        # and not on the host because this is the only process that can read a
        # GLB's vertices; it lands in ``spec["bones"]``, so from ``_rig_bones``
        # onward it is indistinguishable from a user's own joint correction --
        # which is exactly what it is, taken automatically.
        from . import jointfit

        verts = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
        try:
            measured = jointfit.payload([tuple(v) for v in verts])
        except ValueError as exc:
            # Costs the measurement, never the rig: the bbox fit is still a
            # rig, and a mesh this cannot read is exactly the mesh whose
            # measurements would be worth least.
            print(f"joint measurement failed, using the template fit: {exc}", flush=True)
        else:
            spec = {**spec, "bones": rigging.validate_joints(measured, template)}
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
    applied, unknown = _apply_pose(
        arm_obj, spec["bones"], str(spec.get("pose_space") or "node")
    )
    if unknown:
        print(f"pose names {len(unknown)} bone(s) this rig does not have: {unknown}", flush=True)
    if spec.get("root_offset"):
        # Only present when a library pose carried a nonzero root translation
        # (rigging.pose_spec adds the keys conditionally), so a spec without it
        # bakes exactly what it always did.
        _apply_root_translation(arm_obj, spec.get("root_bone"), spec["root_offset"])

    progress(0.70, "Exporting pose")
    _export(bpy, Path(spec["out_glb"]))
    progress(1.0, "Pose complete")
    return {"ok": True, "bones": applied, "unknown": unknown}


def op_armature(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Export one template's armature over the canonical unit box, meshless.

    The Poser preview: no source mesh, no skinning, just the skeleton fitted to
    ``poselib.UNIT_LO``/``UNIT_HI`` -- where ``fit_template``'s ``place()`` is
    the identity on the normalized landmarks, so the exported armature is
    exactly one character-height tall and a root translation authored against
    it is in character-height units literally.

    ``_build_armature`` and ``_export`` are the same calls ``op_rig`` makes, on
    purpose: the preview's bone frames and a real bake's must be the same
    frames, and sharing the code path is what makes that divergence-proof.
    """
    template = rigging.get_template(spec["template"])

    progress(0.10, "Building armature")
    _reset_scene(bpy)
    bones = rigging.fit_template(template, poselib.UNIT_LO, poselib.UNIT_HI)
    _build_armature(bpy, bones)

    progress(0.60, "Exporting armature")
    _export(bpy, Path(spec["out_glb"]))
    progress(1.0, "Armature complete")
    return {"ok": True, "template": template.key, "bones": len(bones)}


def op_sheet(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Render one PNG per sheet cell into ``frames_dir``.

    The host decided the grid and will do the packing; this walks the cell list
    in order, posing and spinning the camera as it goes. The framing is
    computed **once, from the union of every pose the sheet contains**, and
    never touched again -- reframing per pose would make the subject jump
    between rows of the finished sheet, and framing from the rest box alone
    clipped every pose whose apex leaves it (an overhead attack wind, a jump)
    on every cell of that run.

    The pre-pass is universal rather than gated on which animations were asked
    for: a clipped apex is a defect on any sheet, and on a rest-only sheet the
    union *is* the rest box, so those sheets come out byte-identical.
    """
    source = Path(spec["source_glb"])
    if not source.exists():
        raise RuntimeError(f"nothing to render at {source}")
    frames_dir = Path(spec["frames_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    size = int(spec["frame_size"])
    elevation = float(spec["elevation"])
    cells = spec["cells"]
    sockets = [dict(s) for s in (spec.get("sockets") or [])]

    progress(0.05, "Loading model")
    _reset_scene(bpy)
    bpy.ops.import_scene.gltf(filepath=str(source), bone_heuristic="BLENDER")
    _purge_import_helpers(bpy)
    armature = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)

    lo, hi = _scene_bounds(bpy)
    # ``spec.get("margin")`` has two writers and only one of them invents a
    # number. ``_q_troupe``'s validation retry chooses one, re-rendering a sheet
    # whose first attempt clipped at a wider margin; a *subset* re-render passes
    # back the ``frame_margin`` its base sidecar recorded, because cells landing
    # beside existing ones have to be framed exactly as those were or the
    # character changes size inside rectangles that must not change.
    # Every other caller omits the key and gets ``sheet.FRAME_MARGIN`` -- the
    # same figure ``studio.viewer.sheet`` frames the in-app preview with, which
    # is why it is that constant and not a literal here.
    margin = float(spec.get("margin") or sheet.FRAME_MARGIN)
    progress(0.07, "Measuring poses")
    union, socket_points, body_centres = _pose_union(
        bpy, armature, cells, sockets, rest_height=max(hi[2] - lo[2], 1e-6)
    )
    if not union:
        # An empty cell list -- nothing to pose, so the union is the rest box.
        union = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    centre, extent = _union_framing(lo, hi, union, margin=margin)
    distance = extent * 2.0

    progress(0.10, "Setting up")
    # Flat is unlit emission, so one sample is the whole answer and antialiasing
    # is pure loss -- see _setup_render. A lit sheet keeps Blender's default.
    _setup_render(bpy, size, taa_samples=1 if spec["lighting"] != "lit" else None)
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
    projected: dict[int, dict[str, dict[str, Any]]] = {}
    for i, cell in enumerate(cells):
        # Cells arrive grouped by row, so this re-poses once per row rather than
        # once per frame -- eight times less work on an eight-yaw sheet. A
        # clip's rows differ only by frame, which is why the cache key carries
        # it: without that every frame of a clip would render the first one.
        key = (cell.get("pose"), cell.get("frame", 0))
        if armature is not None and key != posed:
            _reset_pose(armature)
            _apply_pose(
                armature, cell.get("bones") or {}, str(cell.get("pose_space") or "node")
            )
            if cell.get("root_offset"):
                # A sheet built from snapshotted library poses must not
                # silently disagree with the bake -- one meaning per pose.
                # _reset_pose zeroes pbone.location between rows, so an offset
                # never leaks into the next pose's cells.
                _apply_root_translation(armature, cell.get("root_bone"), cell["root_offset"])
            posed = key
        _aim_camera(cam, centre, float(cell["yaw"]), elevation, distance)
        if sockets:
            # Per cell, because a socket is attached to a bone and both the
            # pose and the yaw move it. Projected through the same ``_project``
            # the pivot goes through, so a socket and the feet are in one
            # coordinate system -- pixels within the rendered frame, which the
            # host converts to cell pixels with ``charsheet.point_in_cell``.
            yaw = float(cell["yaw"])
            body = body_centres.get(key)
            here: dict[str, dict[str, Any]] = {}
            for name, point in (socket_points.get(key) or {}).items():
                px, py = _project(bpy, cam, point, size)
                depth = _view_depth(
                    centre, point, yaw_deg=yaw, elevation_deg=elevation, distance=distance
                )
                behind = body is not None and depth > _view_depth(
                    centre, body, yaw_deg=yaw, elevation_deg=elevation, distance=distance
                )
                here[name] = {
                    "x": float(px),
                    "y": float(py),
                    "depth": float(depth),
                    "behind": bool(behind),
                }
            projected[int(cell["index"])] = here
        out = frames_dir / f"{cell['index']:04d}.png"
        bpy.context.scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        rendered.append(cell["index"])
        progress(0.10 + 0.85 * (i + 1) / max(len(cells), 1), f"Rendering {i + 1}/{len(cells)}")

    progress(1.0, "Frames rendered")
    result: dict[str, Any] = {
        "ok": True,
        "frames": rendered,
        "bounds": {"min": lo, "max": hi},
        "pivot": list(pivot),
        # What the window was actually sized to, so the host can record it and
        # a retry at a wider margin can say what changed. ``bounds`` above stays
        # the *rest* box: it is what the pivot is derived from and readers of it
        # predate the union.
        "framing": {
            "extent": float(extent),
            "margin": float(margin),
            "union_bounds": {
                "min": [min(c[i] for c in union) for i in range(3)],
                "max": [max(c[i] for c in union) for i in range(3)],
            },
        },
    }
    if sockets:
        # Only when they were asked for, so every sheet rendered before sockets
        # existed comes back with the dict it always came back with. The keys
        # are cell indices and arrive at the host as JSON object keys, i.e.
        # strings -- the result travels through ``result.json``.
        result["sockets"] = projected
    return result


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


def _depth_terms(extent: float, distance: float) -> tuple[float, float]:
    """The (offset, scale) of the camera-depth encoding: enc = (offset - d) * scale.

    Inverted -- near 1, far 0 -- so a pixel where nothing rendered decodes to
    the far plane and "no occluder here" needs no special case on the host.
    Pure arithmetic, importable without bpy, and pinned against
    ``pipelines.retexture.depth_encode`` by ``tests/test_retexture.py`` -- the
    ``_view_direction`` treatment, because the host decodes what these two
    numbers encoded and a drift reads every visibility compare against the
    wrong plane, which looks like random dropout rather than like a bug.
    """
    span = max(2.0 * extent, 1e-9)
    return (distance + extent, 1.0 / span)


def _depth_chain(tree: Any, centre, extent: float, distance: float):
    """The node chain computing this surface point's encoded camera depth.

    -> (dot_node, value_socket). The caller points ``dot_node.inputs[1]`` at
    each view's direction; the socket then carries
    ``depth_encode(distance - dot(P - centre, dir))`` for that view. Shared by
    the depth *render* material and the depth-pair *bake* material so the two
    cannot disagree about what a texel's own depth is.
    """
    offset, scale = _depth_terms(extent, distance)
    geo = tree.nodes.new("ShaderNodeNewGeometry")
    rel = tree.nodes.new("ShaderNodeVectorMath")
    rel.operation = "SUBTRACT"
    rel.inputs[1].default_value = tuple(centre)
    tree.links.new(geo.outputs["Position"], rel.inputs[0])
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    tree.links.new(rel.outputs["Vector"], dot.inputs[0])
    # d = distance - dot(P - centre, dir); enc = (offset - d) * scale, clamped.
    depth = tree.nodes.new("ShaderNodeMath")
    depth.operation = "SUBTRACT"
    depth.inputs[0].default_value = distance
    tree.links.new(dot.outputs["Value"], depth.inputs[1])
    inverted = tree.nodes.new("ShaderNodeMath")
    inverted.operation = "SUBTRACT"
    inverted.inputs[0].default_value = offset
    tree.links.new(depth.outputs["Value"], inverted.inputs[1])
    scaled = tree.nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    scaled.use_clamp = True
    scaled.inputs[1].default_value = scale
    tree.links.new(inverted.outputs["Value"], scaled.inputs[0])
    return dot, scaled.outputs["Value"]


def _depth_material(bpy: Any, centre, extent: float, distance: float):
    """One emission material rendering the camera-depth encoding. -> (material, dot)

    The dot node is returned so ``op_views`` can retarget the view direction
    per render instead of rebuilding the material ten times.
    """
    material = bpy.data.materials.new("wl_depth")
    if material.node_tree is None:
        material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    dot, value = _depth_chain(tree, centre, extent, distance)
    emit = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(value, emit.inputs["Color"])
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return material, dot


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

    if spec.get("depth"):
        # A second pass rather than interleaved: it costs the same either way
        # and leaves the colour loop exactly what it was. Every mesh wears the
        # one depth material -- the colour pass is over, so nothing needs its
        # materials back in this process.
        scene = bpy.context.scene
        depth_mat, dot = _depth_material(bpy, centre, extent, distance)
        for obj in scene.objects:
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(depth_mat)
        # Raw, not Standard: the encoding is a linear ramp and Standard's sRGB
        # curve would bend it before the host's decode. 16-bit because the
        # depth-pair bake samples this file inside Blender, where the extra
        # precision is kept even though Pillow reads it back at 8.
        with contextlib.suppress(TypeError):
            scene.view_settings.view_transform = "Raw"
        scene.render.image_settings.color_depth = "16"
        with contextlib.suppress(AttributeError):
            # One sample: the occlusion source needs hard edges, and an
            # emission render has no noise for TAA to average away.
            scene.eevee.taa_render_samples = 1
        for i, (yaw, pitch) in enumerate(views):
            dot.inputs[1].default_value = _view_direction(float(yaw), float(pitch))
            _aim_camera(cam, centre, float(yaw), float(pitch), distance)
            scene.render.filepath = str(views_dir / f"depth_{i:02d}.png")
            bpy.ops.render.render(write_still=True)

    progress(1.0, "Views rendered")
    return {"ok": True, "views": len(views), "extent": extent}


def _project_material(
    bpy: Any,
    colour_png: Path,
    mask_png: Path,
    direction,
    depth_png: Path | None = None,
    frame: tuple[Any, float, float] | None = None,
):
    """One material carrying every bake. -> (material, colour_emit, weight_emit, depth_emit)

    All emissions share one projection and one set of textures, so switching
    which node feeds the output is the whole difference between the colour bake
    and the weight bake -- they cannot come to disagree about where the view
    landed.

    The mask is the **original** render's alpha rather than the restyled one's,
    because img2img returns RGB and drops it. Two textures over one UV layer,
    which also makes "outside the camera frustum" free: ``CLIP`` extension
    returns alpha 0 out there, so a texel the camera never saw gets weight 0
    without a frustum test of its own.

    With ``depth_png`` and ``frame`` (= centre, extent, distance) a third
    emission carries the depth pair: R is the depth render sampled through the
    same projected UVs -- what the camera actually saw at this texel's pixel
    -- and G is the texel's own depth from ``_depth_chain``. R is a
    pass-through sample, so the only encode formula lives in the node graphs
    fed by ``_depth_terms`` within one process run; ``depth_emit`` is ``None``
    when not asked for.
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

    depth_emit = None
    if depth_png is not None and frame is not None:
        centre, extent, distance = frame
        depth_tex = tree.nodes.new("ShaderNodeTexImage")
        depth_tex.image = bpy.data.images.load(str(depth_png))
        # Non-Color or the compare silently rots: the render is a linear
        # encoding, and the sRGB decode every loaded PNG gets by default
        # would bend zread against the zsurf computed in nodes.
        depth_tex.image.colorspace_settings.name = "Non-Color"
        depth_tex.extension = "CLIP"
        tree.links.new(uv.outputs["UV"], depth_tex.inputs["Vector"])
        dot, own_depth = _depth_chain(tree, centre, extent, distance)
        dot.inputs[1].default_value = direction
        pair = tree.nodes.new("ShaderNodeCombineColor")
        # R = what the camera saw at this texel's pixel, G = this texel's own
        # depth. The host subtracts them; nothing here decides visibility.
        tree.links.new(depth_tex.outputs["Color"], pair.inputs["Red"])
        tree.links.new(own_depth, pair.inputs["Green"])
        depth_emit = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(pair.outputs["Color"], depth_emit.inputs["Color"])

    tree.links.new(colour_emit.outputs["Emission"], out.inputs["Surface"])
    return material, colour_emit, weight_emit, depth_emit


def _free_material(bpy: Any, material: Any) -> None:
    """Free a projection material and every image it loaded.

    Blender's datablocks are reference-counted only for *saving*: an image or a
    material with no user still sits in ``bpy.data`` for the life of the
    process. ``mesh.data.materials.clear()`` empties an object's slots and
    ``tree.nodes.remove`` detaches a node; neither frees anything, which is why
    both of those already being called is not enough.

    Written defensively -- a datablock already gone is not an error worth
    failing a bake over -- and it collects the images *before* removing the
    material, because removing the material invalidates its node tree.
    """
    images = []
    tree = getattr(material, "node_tree", None)
    if tree is not None:
        images = [
            node.image
            for node in tree.nodes
            if getattr(node, "type", "") == "TEX_IMAGE" and node.image is not None
        ]
    with contextlib.suppress(Exception):
        bpy.data.materials.remove(material)
    for image in images:
        with contextlib.suppress(Exception):
            bpy.data.images.remove(image)


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
    depth_wanted = bool(spec.get("depth"))
    done = []
    for i, (yaw, pitch) in enumerate(views):
        colour_png = views_dir / f"restyled_{i:02d}.png"
        mask_png = views_dir / f"view_{i:02d}.png"
        depth_png = views_dir / f"depth_{i:02d}.png"
        if not colour_png.exists() or not mask_png.exists():
            # A view whose restyle never arrived contributes nothing rather
            # than failing the bake: five good projections beat none.
            continue
        if depth_wanted and not depth_png.exists():
            # Same rule: without its depth render this view cannot be
            # occlusion-tested, and the host's all-or-nothing assemble would
            # refuse a bake that arrived without its pair.
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

        material, colour_emit, weight_emit, depth_emit = _project_material(
            bpy,
            colour_png,
            mask_png,
            _view_direction(float(yaw), float(pitch)),
            depth_png if depth_wanted else None,
            (centre, extent, distance) if depth_wanted else None,
        )
        mesh.data.materials.clear()
        mesh.data.materials.append(material)
        tree = material.node_tree
        out_node = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL")

        bakes = [("bake", colour_emit), ("weight", weight_emit)]
        if depth_emit is not None:
            bakes.append(("depthpair", depth_emit))
        for suffix, emit in bakes:
            for link in list(out_node.inputs["Surface"].links):
                tree.links.remove(link)
            tree.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
            image = bpy.data.images.new(
                f"wl_{suffix}_{i}", texture_size, texture_size, alpha=False
            )
            if suffix in ("depthpair", "weight"):
                # The bake target's colorspace decides how save() encodes the
                # PNG. Non-Color writes linear values raw; the default would
                # sRGB-encode them and the host would read a bent curve as a
                # straight one.
                #
                # Both of these are *data*, not colour. ``depthpair`` is the
                # near/far pair the host subtracts. ``weight`` is
                # ``max(0, N.V) * mask_alpha`` -- a facing ratio -- and it was
                # left on the sRGB default while ``retexture.assemble`` read it
                # back as linear and thresholded it against ``MIN_FACING``.
                # Since srgb_encode(0.0196) is about 0.15, a floor meant to
                # drop views past ~81 degrees off-normal was really dropping
                # only those past ~89, and the curve's compression handed
                # grazing views roughly twice their intended share of every
                # texel. ``bake`` -- the colour target -- stays sRGB, which is
                # correct for it and is why this is a tuple rather than a flip.
                #
                # See docs/measurements/2026-08-20-retexture-weight-colorspace.md.
                image.colorspace_settings.name = "Non-Color"
            node = tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            tree.nodes.active = node
            bpy.ops.object.bake(type="EMIT")
            image.filepath_raw = str(out_dir / f"{suffix}_{i:02d}.png")
            image.file_format = "PNG"
            image.save()
            tree.nodes.remove(node)
            # The PNG is on disk and nothing reads the datablock again.
            # ``nodes.remove`` only detaches it: an image datablock outlives
            # every node that pointed at it and is freed only by an explicit
            # ``images.remove``. At ``texture_size`` up to 2048 and three
            # targets a view, ten views left the better part of a gibibyte
            # resident in the one subprocess whose host-commit budget the rest
            # of the codebase guards carefully.
            bpy.data.images.remove(image)

        mesh.data.materials.clear()
        # Same argument, for what ``_project_material`` allocated: the material
        # and the two or three PNGs it loaded. ``materials.clear()`` empties the
        # object's *slots*, which is not the same as freeing the datablock.
        _free_material(bpy, material)
        done.append(i)
        progress(0.05 + 0.9 * (i + 1) / max(len(views), 1), f"Baking {i + 1}/{len(views)}")

    for material in original:
        mesh.data.materials.append(material)
    progress(1.0, "Projections baked")
    return {"ok": True, "baked": done, "uv_layer": atlas_uv}


# --- remesh --------------------------------------------------------------------

#: The voxel size of the hole-closing pre-pass, as a fraction of the mesh's
#: bounding diagonal, and the bake margin in texels. Restated from
#: ``pipelines.remesh`` because this side may not import the host package;
#: ``tests/test_remesh.py`` pins the pair.
VOXEL_FRACTION = 0.005
BAKE_MARGIN_PX = 8


def _face_stats(mesh: Any) -> tuple[int, float]:
    """(face count, fraction of faces that are quads)."""
    polys = mesh.data.polygons
    if len(polys) == 0:
        return 0, 0.0
    quads = sum(1 for p in polys if len(p.vertices) == 4)
    return len(polys), quads / len(polys)


def _bake_image(bpy: Any, name: str, size: int, *, data: bool) -> Any:
    image = bpy.data.images.new(name, size, size, alpha=False)
    if data:
        # Roughness and normals are data, not colour: left on the sRGB
        # default the exporter would bend a straight ramp
        # (docs/measurements/2026-08-20-retexture-weight-colorspace.md).
        image.colorspace_settings.name = "Non-Color"
    return image


def _source_metallic(source: Any) -> float:
    """The source's metallic factor, when it is a constant.

    Cycles has no metallic bake type, and rewiring every source material's
    metallic input into an emission is a second bake pipeline for a channel a
    reconstruction almost never varies. The constant is honest: it is what the
    importer wrote, averaged over the slots, and the report says "constant".
    """
    values = []
    for material in source.data.materials:
        if material is None or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                socket = node.inputs.get("Metallic")
                if socket is not None and not socket.is_linked:
                    values.append(float(socket.default_value))
    return sum(values) / len(values) if values else 0.0


def op_remesh(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Remesh to a quad budget, unwrap, and bake the old surface onto the new.

    The one step every commercial pipeline sells as "game-ready" and a
    reconstruction lacks. Four stages, all in this process:

    1. **Remesh.** Optionally a voxel pass first (it closes the plate-crust
       holes ``meshaudit`` counts, at the cost of rounding sharp edges), then
       quadriflow to ``target_faces``. Quadriflow refuses non-manifold input,
       which a reconstruction routinely is; the fallback is a decimate to the
       same budget in triangles, and the result says which path ran -- a
       decimated mesh must not be reported as a quad one.
    2. **Unwrap.** Smart UV project on the new surface. Not an artist's
       layout, but every island is a real region of the mesh, which is what
       the reconstruction's xatlas soup was not.
    3. **Bake.** Selected-to-active from the *original* object: base colour
       (the diffuse colour pass alone -- no lighting, ``op_views``' rule),
       roughness, and tangent-space normals, which carry the high-resolution
       geometry the budget threw away. Metallic is a constant, see
       ``_source_metallic``.
    4. **Export** the new object alone, textures packed into the GLB.
    """
    import math

    source_path = Path(spec["source_glb"])
    out_glb = Path(spec["out_glb"]).resolve()
    if not source_path.exists():
        raise RuntimeError(f"nothing to remesh at {source_path}")
    target = int(spec["target_faces"])
    texture_size = int(spec["texture_size"])
    seed = int(spec.get("seed", 0))

    progress(0.02, "Loading model")
    _reset_scene(bpy)
    source = _import_glb(bpy, source_path)
    faces_before, _ = _face_stats(source)
    lo, hi = _world_bounds(source)
    diagonal = max(math.dist(lo, hi), 1e-6)

    # A working copy: the original keeps its materials and UVs as the bake
    # source, and is deleted before export.
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    bpy.ops.object.duplicate()
    work = bpy.context.view_layer.objects.active
    work.name = "wl_remesh"
    work.data.materials.clear()

    # **Weld before anything else touches the topology.** glTF cannot share a
    # position between two texture coordinates, so every GLB splits its
    # vertices at each UV seam -- which makes an imported mesh non-manifold
    # before anything is actually wrong with it. ``_weld``'s docstring carries
    # that argument already; what is new here is that *every* input on this
    # path is a GLB, so the quadriflow branch below could never succeed and
    # every remesh silently produced the triangle fallback instead.
    #
    # Measured 2026-08-30 on a UV sphere: 1,106 vertices before export, 4,512
    # after the round trip, and quadriflow answering "Remeshing failed".
    # Welded back to 1,106 it returns 479 faces, all of them quads.
    #
    # ``work`` alone. ``source`` keeps its own vertices, UVs and materials
    # because it is the bake's selected-to-active source, and welding it would
    # change the surface the colour and normal passes are read from.
    weld = weld_distance(lo, hi)
    if weld > 0.0:
        pre_weld, _merged = _weld(bpy, work, weld)
        # Never restored: unlike the skin chain, there is no fallback here that
        # wants the split mesh back, so the copy ``_weld`` takes is freed at
        # once rather than living until the subprocess exits.
        with contextlib.suppress(Exception):
            bpy.data.meshes.remove(pre_weld)

    method = "quadriflow"
    if spec.get("close_holes"):
        progress(0.08, "Closing holes")
        work.data.remesh_voxel_size = diagonal * VOXEL_FRACTION
        work.data.remesh_voxel_adaptivity = 0.0
        work.data.use_remesh_fix_poles = False
        bpy.ops.object.voxel_remesh()

    progress(0.15, f"Remeshing to {target:,} quads")
    try:
        bpy.ops.object.quadriflow_remesh(
            target_faces=target,
            use_mesh_symmetry=False,
            use_preserve_sharp=False,
            use_preserve_boundary=False,
            seed=seed,
            mode="FACES",
        )
        if len(work.data.polygons) == 0:
            raise RuntimeError("quadriflow produced no faces")
    except Exception:
        # Non-manifold input, or a mesh quadriflow gave up on. The budget is
        # still honoured, in triangles, and the result says so.
        method = "decimate"
        modifier = work.modifiers.new("wl_decimate", "DECIMATE")
        tris = max(len(work.data.polygons), 1)
        modifier.ratio = max(min((target * 2) / tris, 1.0), 0.001)
        bpy.context.view_layer.objects.active = work
        bpy.ops.object.modifier_apply(modifier="wl_decimate")

    progress(0.45, "Unwrapping")
    bpy.ops.object.select_all(action="DESELECT")
    work.select_set(True)
    bpy.context.view_layer.objects.active = work
    while work.data.uv_layers:
        work.data.uv_layers.remove(work.data.uv_layers[0])
    work.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")

    # The new material, with the three bake targets already in it: a bake
    # writes into the active image node of the active object's material.
    material = bpy.data.materials.new("wl_remeshed")
    material.use_nodes = True
    tree = material.node_tree
    principled = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    base_img = _bake_image(bpy, "wl_base_color", texture_size, data=False)
    rough_img = _bake_image(bpy, "wl_roughness", texture_size, data=True)
    normal_img = _bake_image(bpy, "wl_normal", texture_size, data=True)
    nodes = {}
    for key, image in (("base", base_img), ("rough", rough_img), ("normal", normal_img)):
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        nodes[key] = node
    work.data.materials.append(material)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.cage_extrusion = diagonal * 0.02
    bake.max_ray_distance = diagonal * 0.05
    bake.margin = BAKE_MARGIN_PX
    bake.use_pass_direct = False
    bake.use_pass_indirect = False
    bake.use_pass_color = True

    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    work.select_set(True)
    bpy.context.view_layer.objects.active = work

    for frac, label, key, kind, extra in (
        (0.55, "Baking colour", "base", "DIFFUSE", {}),
        (0.70, "Baking roughness", "rough", "ROUGHNESS", {}),
        (0.82, "Baking normals", "normal", "NORMAL", {"normal_space": "TANGENT"}),
    ):
        progress(frac, label)
        tree.nodes.active = nodes[key]
        bpy.ops.object.bake(type=kind, **extra)
        nodes[key].image.pack()

    # Wire the bakes into the material the exporter reads. glTF packs
    # roughness in G and metallic in B of one image; the exporter builds that
    # image itself when roughness is a texture and metallic a constant.
    metallic = _source_metallic(source)
    tree.links.new(nodes["base"].outputs["Color"], principled.inputs["Base Color"])
    tree.links.new(nodes["rough"].outputs["Color"], principled.inputs["Roughness"])
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    tree.links.new(nodes["normal"].outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
    principled.inputs["Metallic"].default_value = metallic

    progress(0.92, "Exporting")
    bpy.data.objects.remove(source, do_unlink=True)
    faces, quads = _face_stats(work)
    _export(bpy, out_glb)
    progress(1.0, "Remeshed")
    return {
        "ok": True,
        "method": method,
        "faces_before": faces_before,
        "faces": faces,
        "quads": quads,
        "texture_size": texture_size,
        "metallic": metallic,
    }


OPS = {
    "rig": op_rig,
    "pose": op_pose,
    "armature": op_armature,
    "sheet": op_sheet,
    "fbx": op_fbx,
    "views": op_views,
    "project": op_project,
    "remesh": op_remesh,
}


def main() -> int:
    # ``fetch_worker``'s rule: a malformed spec is reported in a sentence and
    # an exit code, never as a traceback. The host reads the tail of stdout to
    # build its error message, and a stack trace from a JSON decoder tells the
    # user nothing about the job they submitted. The result path is resolved
    # here rather than after the op, so a spec that could never hand anything
    # back is refused before Blender spends minutes on it.
    try:
        spec = json.loads(sys.stdin.read())
        result_path = Path(spec["result_path"])
    except (ValueError, TypeError, KeyError) as exc:
        print(f"the worker spec on stdin is not usable: {exc}", file=sys.stderr)
        return 2
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
    # Staged and renamed, like every other write onto a name something else
    # reads: the host polls for this file's existence, so a partial write is a
    # result it would parse as a failure of the op rather than of the write.
    tmp = result_path.with_name(result_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(result_path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

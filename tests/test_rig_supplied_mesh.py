"""Rigging a mesh that arrives already rigged — the Troupe intake's real input.

Every mesh the rig path had ever been given was a TRELLIS reconstruction: no
armature, no vertex groups, no skin. That is not what the *supplied base mesh*
path receives. A humanoid a user brings to Troupe has usually been rigged by
whoever made it, and two defects lived in exactly that blind spot until
``tests/fixtures/humanoid/cesium_man.glb`` was added and this file asked.

**1. ``_skin``'s failure guard stopped working.** Bone-heat weighting reports
failure two ways -- an operator ``RuntimeError``, or a ``FINISHED`` that quietly
leaves every vertex group empty -- and ``_has_weights`` is what catches the
quiet one. It asks whether *any* vertex group carries a weight. An incoming
skin answers yes before the new armature is bound at all, so a bind that
produced nothing was reported as a clean ``automatic`` rig: the app says the
rig succeeded and the character does not deform.

**2. The old skeleton was exported beside the new one.** ``_import_glb``
returns the joined mesh and leaves the scene alone; ``_export`` writes the
*whole scene*. Two armatures in the output GLB, one with nothing weighted to it.

Both are fixed by ``_strip_incoming_rig``, and both are pinned below. Neither
could be reproduced with a synthetic fixture without first building a rigged
GLB, which is the fixture.

The fixture is CesiumMan (CC-BY 4.0, Cesium) -- see its ``ATTRIBUTION.md``. It
is a low-poly specification sample, so what it can settle is whether the
*mechanism* handles a skinned input. Whether a rig deforms well enough to ship
is an art verdict and needs the mesh the plan file asks for.

Run with: uv run pytest tests/test_rig_supplied_mesh.py -n 0
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: A real bone-heat solve over 3k vertices, plus an import and an export.
pytestmark = pytest.mark.timeout(600)

FIXTURE = Path(__file__).parent / "fixtures" / "humanoid" / "cesium_man.glb"

#: What the file is known to carry, asserted so that swapping the fixture for a
#: different mesh fails here rather than silently weakening every test below.
SOURCE_BONES = 19
SOURCE_GROUPS = 19


@pytest.fixture
def imported():
    """The fixture through the real importer, on a clean scene. -> the mesh."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker as bw

    assert FIXTURE.is_file(), f"missing fixture: {FIXTURE}"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy, bw, bw._import_glb(bpy, FIXTURE)


def test_the_fixture_is_the_rigged_textured_mesh_these_tests_assume(imported):
    """Guards every assertion below against a fixture swap.

    Also the record of what the file is: if this fails, the mesh changed and
    the defects the rest of this file pins may no longer be reachable through
    it -- which would make a green run mean nothing.
    """
    bpy, _bw, mesh = imported
    assert len(mesh.vertex_groups) == SOURCE_GROUPS
    assert len(mesh.data.materials) == 1, "the fixture must be textured"
    assert len(mesh.data.uv_layers) == 1
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    assert len(armatures) == 1
    assert len(armatures[0].data.bones) == SOURCE_BONES


def test_the_incoming_skin_defeats_the_weight_guard(imported):
    """The defect, stated as a property of the mesh rather than of the fix.

    This is what made defect 1 invisible: ``_has_weights`` is True on arrival,
    before anything has been bound. Asserted directly, because it is the reason
    ``_strip_incoming_rig`` has to run *before* ``_skin`` rather than inside
    its fallback chain -- and a future refactor that moves the call would pass
    every other test in this file while restoring the bug.
    """
    _bpy, bw, mesh = imported
    assert bw._has_weights(mesh) is True, (
        "the fixture is supposed to arrive skinned; without that this file "
        "cannot reach the defect it exists to pin"
    )


def test_stripping_leaves_no_skin_and_no_skeleton(imported):
    """After the strip: no groups, no armature, and the geometry untouched.

    The geometry half matters as much as the rest. ``_unbind`` also clears
    parenting and modifiers, and a strip that quietly dropped vertices would
    be a far worse bug than the one it fixes.
    """
    bpy, bw, mesh = imported
    verts_before = len(mesh.data.vertices)
    polys_before = len(mesh.data.polygons)

    removed = bw._strip_incoming_rig(bpy, mesh)

    assert removed == SOURCE_BONES
    assert len(mesh.vertex_groups) == 0
    assert not [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    assert not [m for m in mesh.modifiers if m.type == "ARMATURE"]
    assert mesh.parent is None
    assert len(mesh.data.vertices) == verts_before, "the strip moved geometry"
    assert len(mesh.data.polygons) == polys_before
    assert len(mesh.data.materials) == 1, "the strip dropped the material"


def test_the_guard_can_detect_a_failed_bind_once_stripped(imported):
    """The fix, stated as the restored invariant rather than as a call count.

    ``_has_weights`` is only a guard while False means "nothing is weighted".
    After the strip it does, which is the whole point: a subsequent bind that
    produces no weights is now detectable, and ``_skin`` can fall through to
    envelope and *say so* instead of reporting a rig that is not there.
    """
    bpy, bw, mesh = imported
    bw._strip_incoming_rig(bpy, mesh)
    assert bw._has_weights(mesh) is False


def _live_size(mesh) -> tuple[float, float, float]:
    """The mesh's real world-space extent, read off the vertices themselves."""
    points = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return tuple(hi[i] - lo[i] for i in range(3))


def test_a_skinned_import_measures_wrong_until_it_is_stripped(imported):
    """The third consequence, and the one that would have ruined the rig quietly.

    ``_import_glb`` bakes the Y-up -> Z-up rotation into the vertex data, but a
    skinned import parents the mesh to its armature and that parent still
    carries the rotation -- so ``matrix_world`` applies it twice and
    ``_world_bounds`` returns a box rotated once too far.

    This is pinned as a *disagreement* rather than against fixed numbers: what
    makes it a bug is that the cached box and the actual vertices describe
    different objects, and any future change that keeps them consistent is a
    fix however the numbers land.
    """
    _bpy, bw, mesh = imported
    lo, hi = bw._world_bounds(mesh)
    reported = tuple(hi[i] - lo[i] for i in range(3))
    actual = _live_size(mesh)
    assert reported != pytest.approx(actual, abs=1e-3), (
        "the double rotation is gone from _import_glb -- if so, delete this "
        "test and keep the one below"
    )


def test_the_stripped_mesh_stands_upright_and_measures_true(imported):
    """After the strip: the bounds agree with the geometry, and it is standing.

    ``_rig_bones`` fits the template to exactly this box, so a wrong one puts
    every joint in the wrong place while the stature stays plausible enough to
    pass a glance. The stored-pose spike already produced one sheet rendered
    lying down; this is the measurement that would let it happen again.
    """
    bpy, bw, mesh = imported
    bw._strip_incoming_rig(bpy, mesh)

    lo, hi = bw._world_bounds(mesh)
    reported = tuple(hi[i] - lo[i] for i in range(3))
    assert reported == pytest.approx(_live_size(mesh), abs=1e-3)

    width, depth, height = reported
    assert height > width > depth, "this is not a standing figure in world space"
    assert 1.0 < height < 2.5, f"unexpected stature: {height:.2f} m"
    assert abs(lo[2]) < 0.1, f"feet are not near the floor: z={lo[2]:.3f}"

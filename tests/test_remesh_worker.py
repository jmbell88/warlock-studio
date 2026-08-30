"""The remesh worker, run for real. The one thing ``test_remesh.py`` cannot ask.

``test_remesh.py`` fakes Blender at ``rigging.run_worker`` -- deliberately, and
its subject is the queue's rework contract: publish over ``model.glb`` by
rename, invalidate the derived exports, never touch ``source.glb``. Every
assertion there is about the *host* half, and the child is a stub that writes
a byte string.

So on 2026-08-30, when the remesh shipped, **nothing had ever executed
``op_remesh``**. Four stages -- voxel pre-pass, quadriflow, smart UV project,
selected-to-active bake -- existed with no evidence that any of them ran, and
the failure that would show up first on a user's card (a Blender build without
the quadriflow operator, a bake that silently produces a blank atlas) is
invisible to a fake by construction.

This file is that evidence, and it is deliberately **not** in the ``gpu`` lane.
That marker means "requires local GPU + model weights"; a remesh requires
neither -- Cycles bakes here at 4 samples on whatever device Blender defaults
to, and the input is a sphere this file builds. Putting it behind ``-m gpu``
would mean it ran only when somebody opted into a lane about model weights,
which is the opposite of what a regression test is for. It skips without
``bpy`` exactly as ``test_rig_weld.py`` and ``test_rigging.py`` do.

The subject is a UV sphere and not a reconstruction. That is the honest scope:
this asks *whether the four stages run and produce a mesh with the maps they
promise*, not whether a 300k-face trellis soup survives them. The second
question needs a real reconstruction and a person looking at it, and it is
`TODO.md` P3's session rather than a test.

Run with: uv run pytest tests/test_remesh_worker.py -n 0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock import rigging, tiercheck
from warlock.pipelines import remesh

#: A real quadriflow plus three bakes is far past the suite's 120 s hang net,
#: which is sized for the default lane's ~5 s worst case. Ten minutes is still
#: a hang net rather than a budget: what it catches is a wedged child, not a
#: slow one.
pytestmark = pytest.mark.timeout(600)

#: Small on purpose. The budget only has to be one quadriflow can actually hit
#: on a sphere; a game budget here would buy minutes of runtime and no extra
#: claim.
TARGET_FACES = 500
TEXTURE_PX = 512


@pytest.fixture(scope="module")
def source_glb(tmp_path_factory) -> Path:
    """A textured sphere, exported as GLB.

    Module-scoped: it is the input to every test below and building it twice
    would only re-roll the same deterministic mesh.

    It carries a Principled material with a non-default base colour, roughness
    and metallic, because the bake is *selected-to-active from the original*.
    An untextured input would bake three blank images and the maps below would
    pass while proving nothing about whether the bake read anything.
    """
    pytest.importorskip("bpy")
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    obj = bpy.context.view_layer.objects.active

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    material = bpy.data.materials.new("probe")
    material.use_nodes = True
    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.8, 0.2, 0.1, 1.0)
    principled.inputs["Roughness"].default_value = 0.35
    principled.inputs["Metallic"].default_value = 0.0
    obj.data.materials.append(material)

    out = tmp_path_factory.mktemp("remesh-src") / "source.glb"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB")
    return out


@pytest.fixture(scope="module")
def remeshed(source_glb, tmp_path_factory) -> tuple[Path, dict]:
    """One real remesh. -> (the written GLB, the child's result payload)."""
    work = tmp_path_factory.mktemp("remesh-out")
    out_glb = work / "remeshed.glb"
    spec = rigging.remesh_spec(
        source_glb,
        out_glb,
        work,
        target_faces=TARGET_FACES,
        texture_size=TEXTURE_PX,
    )
    result = rigging.run_worker(spec)
    return out_glb, result


def test_the_child_runs_and_writes_the_glb_it_was_asked_for(remeshed):
    """The whole contract in one assertion: spec in, result JSON and a file out.

    Nothing had ever asserted this. A missing operator, an import error inside
    bpy or a bad spec key all land here, and all of them reach the user as a
    job that goes to ``error`` after Blender has already started.
    """
    out_glb, result = remeshed
    assert result["ok"] is True
    assert out_glb.is_file(), "the worker reported ok and wrote no mesh"
    assert out_glb.stat().st_size > 0


def test_the_budget_is_respected(remeshed):
    """``faces`` lands near the budget, and the mesh is genuinely reduced."""
    _out_glb, result = remeshed
    assert result["faces"] > 0
    assert result["faces_before"] > result["faces"], "nothing was reduced"
    # Generous: quadriflow hits a budget approximately, and the exact ratio is
    # a property of the operator rather than of this code.
    assert result["faces"] <= TARGET_FACES * 3


def test_the_surface_comes_back_as_quads(remeshed):
    """Quadriflow ran, and this is a regression gate rather than a formality.

    **The first run of this file failed here**, and the defect it found is the
    reason the assertion is this strict. glTF cannot share a vertex position
    between two texture coordinates, so a GLB splits its vertices at every UV
    seam -- and quadriflow refuses non-manifold input. Every input on this path
    is a GLB, so before the weld was added to ``op_remesh`` the quadriflow
    branch *could never succeed*: every remesh silently took the decimate
    fallback, and a feature whose profiles are spelled in quads shipped
    triangles.

    Measured on this sphere: 1,106 vertices before export, 4,512 after the
    round trip, quadriflow answering "Remeshing failed"; welded back to 1,106
    it returns ~479 faces, all quads.

    ``decimate`` remains a legitimate outcome for genuinely bad geometry, which
    is why ``report_line`` still distinguishes them. It is not a legitimate
    outcome for a UV sphere, and accepting it here would let the fallback
    become the only path again without a test noticing -- which is exactly what
    happened.
    """
    _out_glb, result = remeshed
    assert result["method"] == "quadriflow", (
        "the remesh fell back to decimate on a closed, manifold sphere; if the "
        "weld in op_remesh is gone or ineffective, every remesh ships triangles"
    )
    assert result["quads"] > 0.9, f"only {result['quads']:.0%} of the faces are quads"


def test_the_report_line_never_calls_a_decimated_mesh_quads(remeshed):
    """``report_line`` against a payload that came from the real worker.

    ``test_remesh.py`` already pins this rule against hand-built dicts. The
    thing only a real run can check is that the worker's *actual* keys are the
    ones ``report_line`` reads -- a renamed key would leave the pure test green
    and the panel showing nothing.
    """
    _out_glb, result = remeshed
    line = remesh.report_line(result)
    assert line, "the worker's payload produced no report line"
    if result["method"] == "decimate":
        assert "decimated" in line and "quads" not in line
    else:
        assert "quads" in line


def test_the_bake_produced_the_maps_the_remesh_promises(remeshed):
    """Base colour **and a normal map**, which is the point of the whole step.

    A TRELLIS reconstruction ships base colour and metallic/roughness and *no*
    normal map; the remesh's argument for existing is that it bakes the
    high-resolution geometry it just threw away into a tangent-space normal
    map. Read off the written GLB through ``tiercheck.survey`` -- the same
    reader the tier qualification uses -- rather than from the child's own
    report, because a worker reporting on itself is not evidence that the
    bytes on disk carry the texture.
    """
    out_glb, _result = remeshed
    survey = tiercheck.survey(out_glb)
    assert survey is not None, "the written GLB did not parse"
    assert survey.base_color, "no base colour survived the bake"
    assert survey.normal_map, "the remesh promises a baked normal map and wrote none"


def test_the_new_surface_is_uv_unwrapped(remeshed):
    """Every bake target needs UVs, so a mesh with none is a blank atlas.

    Asserted off the file for ``test_the_bake_...``'s reason: the unwrap is a
    stage whose failure is silent -- smart UV project on a mesh it cannot lay
    out leaves the bake reading background, and the maps above would still be
    *present*.
    """
    out_glb, _result = remeshed
    survey = tiercheck.survey(out_glb)
    assert survey is not None
    assert survey.primitives > 0
    assert survey.uv_primitives == survey.primitives, (
        "a primitive came back without UVs, so its bake read background"
    )

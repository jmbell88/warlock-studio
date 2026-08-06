"""The two end-to-end paths the plan asks to be checked by hand, headlessly.

Neither replaces sitting in front of the app, but both walk exactly the same
sequence of calls a user's clicks and keypresses produce, so a regression in
the *joins* -- op to document, document to serialization, serialization to
import -- fails here rather than in a window nobody has open.
"""

from __future__ import annotations

import numpy as np

from warlock.studio import clay_ops
from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import glbimport, serialize
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as bp
from warlock.studio.viewer import glbwrite, gltf


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    """``toast(text, level)`` and nothing else -- the real ``Ctx``'s API."""

    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def test_place_a_box_select_a_face_extrude_drag_and_undo() -> None:
    """Clay -> box -> 3 -> click a face -> Extrude -> drag with W -> Ctrl+Z.

    The two things the interactive check is really for: that the extrude leaves
    the caps selected so the drag has something to grab, and that the undo takes
    the geometry back *and* drops the selection over it.
    """
    ctx = _Ctx()
    doc = bd.ClayDoc()
    box = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box")
    )

    doc.set_element_mode("face")
    doc.set_element_sel(box.uid, el.ElementSel(faces=[1]))  # the +Y face
    assert doc.selection == {box.uid}

    assert clay_ops.run(ctx, doc, clay_ops.get("extrude"))
    assert bm.face_count(doc.by_uid(box.uid).mesh) == 10
    assert doc.element_sel_of(box.uid).faces.tolist() == [1], "the cap is what you drag"
    assert doc.by_uid(box.uid).generator is None, "and the object is frozen"

    # The drag: the viewport moves the affected vertices and commits one step.
    before = doc.by_uid(box.uid).mesh
    verts = el.affected_verts(before, doc.element_sel_of(box.uid))
    moved = np.array(before.positions)
    moved[verts] += [0.0, 0.5, 0.0]
    from dataclasses import replace

    doc.set_mesh(
        box.uid, replace(before, positions=moved), select=doc.element_sel_of(box.uid)
    )
    assert bm.bounds(doc.by_uid(box.uid).mesh)[1][1] == 1.0

    doc.undo()
    assert np.allclose(doc.by_uid(box.uid).mesh.positions, before.positions)
    assert box.uid not in doc.element_sel, "an undo that moves geometry drops it"
    assert not ctx.toasts.errors


def test_bevel_an_edge_with_a_width_and_the_result_is_still_closed() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    box = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.set_element_mode("edge")
    doc.set_element_sel(
        box.uid, el.ElementSel(edges=[adj.adjacency(box.mesh).edge_verts[0]])
    )

    assert clay_ops.run(ctx, doc, clay_ops.get("bevel"), width=0.08)
    out = doc.by_uid(box.uid).mesh
    bm.validate(out)
    assert bm.face_count(out) == 7
    assert adj.check_manifold(out).clean
    assert not ctx.toasts.errors


def test_import_a_model_glb_repair_it_and_export_it_again(tmp_path) -> None:
    """Drop a library ``model.glb`` -> weld -> fill hole -> export -> reopen.

    The whole point of Phase 5 in one path: an asset that came out of the
    pipeline goes back in with its textures and its uvs intact, having been
    edited in between.
    """
    ctx = _Ctx()

    # A "library asset": an open tube with a texture, exported as a GLB.
    tube = bp.cylinder(segments=8)
    open_tube = bm.Mesh(
        positions=tube.positions,
        loops=tube.loops[: 8 * 4],
        starts=tube.starts[:9],
        material=tube.material[:8],
        smooth=tube.smooth[:8],
        uv=np.tile(np.array([[0.25, 0.75]], dtype="f4"), (8 * 4, 1)),
    )
    image = (2, 2, bytes(range(16)))
    source = bd.ClayDoc(materials=[gltf.Material(name="baked", base_color=image)])
    source.objects.append(bd.Obj(uid=bd.new_uid(), name="Tube", mesh=open_tube))
    asset = glbwrite.write_glb(bd.to_model(source))

    # Import it, and check what arrived.
    doc = glbimport.glb_to_claydoc(asset, name="Tube")
    uid = doc.objects[0].uid
    assert doc.materials[0].base_color == image
    assert doc.by_uid(uid).mesh.uv is not None

    # Weld the whole thing (a no-op on a clean mesh, but the repair a user
    # reaches for first), then cap both ends.
    doc.set_element_mode("vertex")
    doc.set_element_sel(uid, el.select_all(doc.by_uid(uid).mesh, "vertex"))
    assert clay_ops.run(ctx, doc, clay_ops.get("weld"), eps=1e-5)

    doc.set_element_mode("edge")
    boundary = adj.check_manifold(doc.by_uid(uid).mesh).boundary_edges
    assert len(boundary) == 16, "two open ends of eight edges each"
    doc.set_element_sel(uid, el.ElementSel(edges=boundary))
    assert clay_ops.run(ctx, doc, clay_ops.get("fill-hole"))
    assert adj.check_manifold(doc.by_uid(uid).mesh).clean, "closed now"
    assert not ctx.toasts.errors

    # Export it as an asset, and save the document beside it -- what
    # ``clay_mode.export_asset`` does, minus the service call.
    exported = glbwrite.write_glb(bd.to_model(doc))
    sidecar = tmp_path / "build.wblk"
    sidecar.write_bytes(serialize.wblk_bytes(doc))

    # Open the exported asset in 3D mode: the textures are still there.
    loaded = gltf.load(exported)
    assert loaded.meshes[0][0].material.base_color == image
    assert loaded.meshes[0][0].uvs is not None

    # And reopening the sidecar gives back the document, not a frozen soup.
    reopened = serialize.read_wblk(sidecar.read_bytes())
    assert len(reopened.objects) == 1
    assert reopened.materials[0].base_color == image
    assert bm.face_count(reopened.objects[0].mesh) == bm.face_count(
        doc.by_uid(uid).mesh
    )
    assert reopened.dirty is False

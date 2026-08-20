"""New faces inherit their material slot and shading from a source face.

The rule every face-growing op follows -- subdivide repeats ``material[faces]``,
extrude copies its owner, dissolve keeps the group's -- stated here for the two
that used to mint slot 0 / flat instead: a bevel on a painted, smooth mesh
sprouted strips of the palette's first material, and a filled hole came back in
it too.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from warlock.studio.clay import ops_bevel, ops_topo
from warlock.studio.clay import primitives as bp
from warlock.studio.clay.adjacency import adjacency
from warlock.studio.clay.elements import ElementSel


def _painted_box():
    mesh = bp.box()
    return replace(
        mesh,
        material=np.full_like(mesh.material, 2),
        smooth=np.ones_like(mesh.smooth),
    )


def test_a_bevels_new_faces_inherit_material_and_shading() -> None:
    mesh = _painted_box()
    edge = adjacency(mesh).edge_verts[0]
    out, sel = ops_bevel.bevel_edges(mesh, ElementSel(edges=np.array([edge])), width=0.1)
    new = np.asarray(sel.faces)
    assert len(new), "the bevel grew faces"
    assert (out.material[new] == 2).all()
    assert out.smooth[new].all()


def test_a_filled_holes_cap_inherits_material_and_shading() -> None:
    mesh = _painted_box()
    held, _ = ops_topo.delete_faces(mesh, ElementSel(faces=[0]))
    a = adjacency(held)
    boundary = np.flatnonzero(a.edge_uses == 1)
    out, sel = ops_topo.fill_hole(held, ElementSel(edges=a.edge_verts[boundary[:1]]))
    new = np.asarray(sel.faces)
    assert len(new) == 1, "one cap for one hole"
    assert (out.material[new] == 2).all()
    assert out.smooth[new].all()

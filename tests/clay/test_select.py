"""Loops, rings, linked, grow, shrink, boundary and mirror pairs.

Every one of these is a thing a modeller does dozens of times an hour and none
of them existed: Clay could select an element and add another with Shift, and
that was the whole vocabulary. So selecting the ring of edges round a cylinder
meant clicking each of them, and selecting one of two shapes welded into one
mesh was not possible at all.

The counts are checked against solids whose answers are arithmetic rather than
remembered. A 4x4 grid has 25 vertices, 40 edges and 16 faces; a row of it is
four edges and the ring across that row is five, because a ring of n quads has
n+1 edges. Those are the numbers a wrong walk gets wrong.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import primitives as bp
from warlock.studio.clay import select


def _grid():
    return bp.grid()


def _interior_edge(mesh):
    """An edge of the grid whose endpoints both have four edges, which is where
    a loop is defined at all."""
    a = adj.adjacency(mesh)
    valence = np.zeros(len(mesh.positions), dtype=int)
    for index in range(a.n_edges):
        valence[a.edge_verts[index]] += 1
    for index in range(a.n_edges):
        pair = a.edge_verts[index]
        if valence[pair[0]] == 4 and valence[pair[1]] == 4:
            return tuple(int(v) for v in pair)
    raise AssertionError("the grid has no interior edge")


# --- loops and rings ----------------------------------------------------------


def test_a_loop_runs_the_width_of_the_grid():
    mesh = _grid()
    assert len(select.edge_loop(mesh, _interior_edge(mesh))) == 4


def test_a_ring_crosses_the_quads_and_is_one_longer():
    """A ring of n quads has n+1 edges, which is the arithmetic that catches a
    walk that stops one short or wraps one too far."""
    mesh = _grid()
    assert len(select.edge_ring(mesh, _interior_edge(mesh))) == 5


def test_a_loop_stops_at_a_pole():
    """A cube's vertices have three edges, not four, so there is no edge
    opposite the one arrived on -- and a loop that ran through one would wander
    off round the mesh. The seed alone is the honest answer."""
    mesh = bp.box()
    a = adj.adjacency(mesh)
    seed = tuple(int(v) for v in a.edge_verts[0])

    assert len(select.edge_loop(mesh, seed)) == 1


def test_a_ring_still_works_where_a_loop_does_not():
    """The two are different traversals, and the cube is the case that shows
    it: no loop, and a four-edge band round the cube."""
    mesh = bp.box()
    a = adj.adjacency(mesh)
    seed = tuple(int(v) for v in a.edge_verts[0])

    assert len(select.edge_ring(mesh, seed)) == 4


def test_a_loop_and_a_ring_both_contain_their_seed():
    mesh = _grid()
    seed = _interior_edge(mesh)
    for pairs in (select.edge_loop(mesh, seed), select.edge_ring(mesh, seed)):
        rows = {tuple(sorted(int(v) for v in row)) for row in pairs}
        assert tuple(sorted(seed)) in rows


def test_an_edge_that_is_not_on_the_mesh_selects_nothing():
    """Reached by a keystroke during a selection, so a refusal is empty rather
    than an exception -- a key that raises is a key that takes the window down."""
    mesh = _grid()
    assert len(select.edge_loop(mesh, (999, 998))) == 0
    assert len(select.edge_ring(mesh, (999, 998))) == 0


def test_a_face_loop_takes_both_strips_through_the_face():
    """A quad sits on two loops at right angles, and picking one arbitrarily
    would make the result depend on corner order rather than on anything the
    user can see."""
    mesh = _grid()
    # A corner face: 4 along one strip and 4 along the other, sharing itself.
    assert len(select.face_loop(mesh, 0)) == 7


def test_a_face_loop_refuses_a_face_that_is_not_a_quad():
    mesh = bp.cylinder()
    arity = np.diff(np.asarray(mesh.starts, dtype="i8"))
    ngon = int(np.flatnonzero(arity != 4)[0])
    assert len(select.face_loop(mesh, ngon)) == 0


# --- linked -------------------------------------------------------------------


def test_linked_takes_the_whole_shell():
    mesh = _grid()
    assert len(select.linked(mesh, [0])) == len(mesh.positions)


def test_linked_stops_at_a_shell_boundary():
    """The verb that makes two shapes welded into one mesh separable again."""
    from warlock.studio.clay import document as bd
    from warlock.studio.clay import ops

    far = bd.Obj(
        uid=bd.new_uid(),
        name="b",
        mesh=bp.box(),
        translation=np.array([9.0, 0.0, 0.0]),
    )
    near = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    joined = ops.join([near, far])

    first = select.linked(joined, [0])

    assert len(first) == 8, "one cube's worth, not both"
    assert len(first) < len(joined.positions)


def test_linked_from_nothing_selects_nothing():
    assert len(select.linked(_grid(), [])) == 0


# --- more and less ------------------------------------------------------------


def test_grow_takes_one_ring_outward():
    mesh = _grid()
    corner = select.grow(mesh, [0])
    assert len(corner) == 3, "a corner vertex and its two neighbours"


def test_shrink_peels_the_border_off():
    """The definition that matters: shrinking leaves "the middle of what I
    have", which is what a user reaching for it wants."""
    mesh = _grid()
    everything = np.arange(len(mesh.positions))

    inner = select.shrink(mesh, everything)

    assert len(inner) == 9, "the 3x3 interior of a 5x5 lattice"
    assert len(inner) < len(everything)


def test_shrinking_a_selection_with_no_interior_leaves_nothing():
    mesh = _grid()
    assert len(select.shrink(mesh, [0])) == 0


def test_grow_then_shrink_returns_the_interior_rather_than_the_original():
    """They are inverses only on an infinite lattice, and saying so is the
    point: a selection touching the border loses that border to the shrink."""
    mesh = _grid()
    seed = select.linked(mesh, [0])
    assert len(select.shrink(mesh, select.grow(mesh, seed))) <= len(seed)


# --- boundary and material ----------------------------------------------------


def test_the_boundary_of_a_grid_is_its_border():
    assert len(select.boundary(_grid())) == 16


def test_a_closed_solid_has_no_boundary():
    assert len(select.boundary(bp.box())) == 0


def test_by_material_finds_the_faces_using_a_slot():
    mesh = bp.box()
    assert len(select.by_material(mesh, 0)) == 6
    assert len(select.by_material(mesh, 7)) == 0


# --- mirror pairs -------------------------------------------------------------


def test_a_symmetric_mesh_pairs_every_vertex():
    pairs = select.mirror_pairs(bp.box(), 0)
    assert len(pairs) == 8
    # And the pairing is an involution: the mirror of a vertex's mirror is
    # itself, which is what a mirrored drag depends on.
    for index, twin in pairs.items():
        assert pairs[twin] == index


def test_a_vertex_on_the_plane_maps_to_itself():
    """The case that has to be handled rather than excluded: those are the ones
    a mirrored drag must slide *along* the plane instead of moving off it."""
    mesh = bp.grid()
    pairs = select.mirror_pairs(mesh, 0)
    on_plane = [
        index
        for index in range(len(mesh.positions))
        if abs(float(mesh.positions[index][0])) < 1e-6
    ]
    assert on_plane
    for index in on_plane:
        assert pairs.get(index) == index


def test_an_asymmetric_mesh_reports_what_it_can_rather_than_pretending():
    """X-mirror can only be as good as the mesh: one that is not symmetric has
    no pairs to find, and reporting the ones it has beats inventing the rest."""
    import dataclasses

    mesh = bp.box()
    moved = np.asarray(mesh.positions, dtype="f8").copy()
    moved[0][0] += 0.5
    shifted = dataclasses.replace(mesh, positions=moved)

    pairs = select.mirror_pairs(shifted, 0)

    assert len(pairs) < len(moved)


def test_the_mirror_axis_is_a_parameter():
    box = bp.box()
    assert len(select.mirror_pairs(box, 0)) == 8
    assert len(select.mirror_pairs(box, 1)) == 8
    assert len(select.mirror_pairs(box, 2)) == 8

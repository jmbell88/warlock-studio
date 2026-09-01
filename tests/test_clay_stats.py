"""What Clay's statistics overlay reports, checked against known solids.

Every number here was unavailable anywhere in Clay before the overlay existed:
the outliner counted objects and nothing counted vertices, edges, faces or
triangles -- so "is this mesh 500 triangles or 50,000" was a question the app
could not answer about the thing on screen, which is the question that decides
whether a game asset is finished.

Which makes the numbers being *right* the whole of the feature, and Euler is how
that is checked rather than asserted: V - E + F = 2 for any closed surface of
genus 0, so a cube reporting 24 edges (the corner count, which is what a cheap
implementation reports) fails here rather than being read off a screenshot by
somebody who happens to remember a cube has twelve.
"""

from __future__ import annotations

import pytest

from warlock.studio import clay_hints
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import primitives as bp


def _doc(*meshes):
    doc = bd.ClayDoc()
    uids = [
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"o{index}", mesh=mesh)).uid
        for index, mesh in enumerate(meshes)
    ]
    return doc, uids


def _numbers(line: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in line.split("  "):
        parts = chunk.split()
        if len(parts) == 2 and parts[0].replace(",", "").isdigit():
            out[parts[1].rstrip("s")] = int(parts[0].replace(",", ""))
    return out


# --- the counts --------------------------------------------------------------


def test_a_cube_reports_the_numbers_a_cube_has():
    doc, _uids = _doc(bp.box())
    got = _numbers(clay_hints.stats(doc))
    assert got["vert"] == 8
    assert got["edge"] == 12, "24 is the corner count -- every edge is shared"
    assert got["face"] == 6
    assert got["tri"] == 12


@pytest.mark.parametrize(
    "name", ["box", "cylinder", "cone", "uv_sphere", "icosphere", "capsule", "torus"]
)
def test_every_closed_primitive_satisfies_eulers_formula(name):
    """V - E + F = 2 for a closed surface of genus 0, and 0 for a torus. The
    check that makes the edge count right rather than plausible."""
    make = getattr(bp, name)
    doc, _uids = _doc(make())
    got = _numbers(clay_hints.stats(doc))

    characteristic = got["vert"] - got["edge"] + got["face"]
    assert characteristic == (0 if name == "torus" else 2), (name, got)


def test_an_open_surface_is_not_expected_to_close():
    """A plane is one face with a boundary: 4 - 4 + 1 = 1, which is the disc's
    characteristic and not a failure."""
    doc, _uids = _doc(bp.plane())
    got = _numbers(clay_hints.stats(doc))
    assert got["vert"] - got["edge"] + got["face"] == 1


def test_an_empty_document_reports_zeros_rather_than_nothing():
    assert "0 objects" in clay_hints.stats(bd.ClayDoc())


def test_counts_are_summed_over_the_objects():
    doc, _uids = _doc(bp.box(), bp.box())
    got = _numbers(clay_hints.stats(doc))
    assert got["vert"] == 16 and got["face"] == 12


def test_a_hidden_object_is_not_counted():
    """The overlay describes what is on screen. An object you have hidden is
    not on screen, and counting it would make the numbers disagree with the
    picture they sit on."""
    doc, uids = _doc(bp.box(), bp.box())
    doc.set_visibility({uids[0]: False})

    got = _numbers(clay_hints.stats(doc))

    assert got["vert"] == 8 and got["object"] == 1


# --- what is selected --------------------------------------------------------


def test_nothing_selected_says_nothing():
    doc, _uids = _doc(bp.box())
    assert "selected" not in clay_hints.stats(doc)


def test_object_mode_counts_objects():
    doc, uids = _doc(bp.box(), bp.box())
    doc.select(uids)
    assert "2 selected" in clay_hints.stats(doc)


@pytest.mark.parametrize(
    ("mode", "sel", "expected"),
    [
        ("vertex", {"verts": [0, 1, 2]}, "3 verts selected"),
        ("face", {"faces": [0]}, "1 face selected"),
        ("edge", {"edges": [[0, 1]]}, "1 edge selected"),
    ],
)
def test_an_element_mode_counts_the_elements_it_is_about(mode, sel, expected):
    """A face count while vertices are being picked is a number about a
    selection the user does not have."""
    doc, uids = _doc(bp.box())
    doc.element_mode = mode
    doc.set_element_sel(uids[0], el.ElementSel(**sel))

    assert expected in clay_hints.stats(doc)


def test_element_counts_are_summed_across_objects():
    doc, uids = _doc(bp.box(), bp.box())
    doc.element_mode = "face"
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0, 1]))

    assert "4 faces selected" in clay_hints.stats(doc)


# --- the memo ----------------------------------------------------------------


def test_the_edge_count_is_memoised_on_the_mesh_itself():
    """Keyed on the ``Mesh`` rather than an id, which is ``ClayState.manifold``'s
    rule: a mesh is immutable and every op replaces it, so ``mesh is measured``
    is exactly "this count is still about what is on screen" -- where an id
    would be recycled onto a different mesh and report the last edit's edges."""
    mesh = bp.box()
    clay_hints._EDGE_CACHE.clear()

    first = clay_hints._unique_edges(mesh)
    assert mesh in clay_hints._EDGE_CACHE
    assert clay_hints._unique_edges(mesh) == first


def test_the_memo_is_bounded_so_a_readout_is_not_a_second_undo_stack():
    """The key pins the mesh, so an unbounded cache would keep every mesh a
    session ever measured alive."""
    clay_hints._EDGE_CACHE.clear()
    for _ in range(clay_hints._EDGE_CACHE_MAX + 5):
        clay_hints._unique_edges(bp.box())

    assert len(clay_hints._EDGE_CACHE) <= clay_hints._EDGE_CACHE_MAX


def test_a_mesh_with_no_faces_counts_no_edges_rather_than_raising():
    class Bare:
        loops = None
        starts = None
        positions = ()

    assert clay_hints._unique_edges(Bare()) == 0

"""The one selection type: canonical form, conversion and the click modifiers."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import elements as el
from warlock.studio.clay import primitives as prim


def test_arrays_are_canonicalised_copied_and_frozen() -> None:
    src = np.array([5, 1, 5, 3], dtype="i4")
    sel = el.ElementSel(verts=src, edges=[[7, 2], [2, 7], [1, 0]], faces=[2, 2])
    assert sel.verts.tolist() == [1, 3, 5]
    assert sel.edges.tolist() == [[0, 1], [2, 7]], "low-first, lexsorted, deduped"
    assert sel.faces.tolist() == [2]

    src[0] = 99
    assert sel.verts.tolist() == [1, 3, 5], "the selection copied its input"
    with pytest.raises(ValueError):
        sel.verts[0] = 0


def test_an_empty_selection_is_empty_whichever_way_it_is_asked() -> None:
    assert el.is_empty(el.empty())
    assert el.is_empty(None)
    assert len(el.empty()) == 0
    assert not el.is_empty(el.ElementSel(verts=[0]))


def test_two_selections_of_the_same_elements_compare_the_same() -> None:
    a = el.ElementSel(edges=[[3, 1], [0, 2]])
    b = el.ElementSel(edges=[[2, 0], [1, 3]])
    assert a.same_as(b)
    assert a is not b, "identity stays distinct -- id(sel) keys the overlay cache"


def test_affected_verts_unions_every_kind() -> None:
    m = prim.box()
    sel = el.ElementSel(verts=[7], edges=[[0, 1]], faces=[0])
    got = el.affected_verts(m, sel).tolist()
    assert set(got) >= {0, 1, 7}
    assert got == sorted(set(got))
    assert el.affected_verts(m, el.empty()).tolist() == []


def test_going_down_is_a_union_and_going_up_is_a_conjunction() -> None:
    m = prim.box()
    face0 = el.ElementSel(faces=[0])
    corners = sorted(m.loops[m.starts[0] : m.starts[1]].tolist())

    verts = el.convert(m, face0, "vertex")
    assert verts.verts.tolist() == corners

    edges = el.convert(m, face0, "edge")
    assert len(edges.edges) == 4, "a quad's four edges, and no others"

    # Back up: those four corners belong to exactly one face all of whose
    # corners are selected. A neighbouring face shares two of them and is not
    # selected, which is the conjunction rule.
    back = el.convert(m, verts, "face")
    assert back.faces.tolist() == [0]

    assert el.convert(m, face0, "object").verts.size == 0


def test_converting_a_partial_vertex_set_selects_no_face() -> None:
    m = prim.box()
    two = el.ElementSel(verts=m.loops[:2])
    assert el.convert(m, two, "face").faces.tolist() == []
    assert el.convert(m, two, "edge").edges.tolist() == [
        sorted(m.loops[:2].tolist())
    ], "two adjacent verts imply the edge between them"


def test_combine_replaces_adds_and_subtracts() -> None:
    a = el.ElementSel(verts=[1, 2, 3], faces=[0])
    b = el.ElementSel(verts=[3, 4], faces=[1])
    assert el.combine(a, b, "replace") is b
    assert el.combine(a, b, "add").verts.tolist() == [1, 2, 3, 4]
    assert el.combine(a, b, "add").faces.tolist() == [0, 1]
    assert el.combine(a, b, "subtract").verts.tolist() == [1, 2]
    assert el.combine(a, b, "subtract").faces.tolist() == [0]
    with pytest.raises(ValueError):
        el.combine(a, b, "nonsense")  # type: ignore[arg-type]


def test_subtracting_edges_matches_whole_pairs_not_endpoints() -> None:
    a = el.ElementSel(edges=[[0, 1], [1, 2], [2, 3]])
    b = el.ElementSel(edges=[[2, 1]])
    assert el.combine(a, b, "subtract").edges.tolist() == [[0, 1], [2, 3]]
    # Subtracting something that was never selected is a no-op, not an error:
    # a Ctrl-drag marquee sweeps over mostly-unselected geometry.
    assert el.combine(a, el.ElementSel(edges=[[9, 8]]), "subtract").same_as(a)


def test_select_all_and_invert_are_per_mode() -> None:
    m = prim.box()
    assert len(el.select_all(m, "vertex").verts) == len(m.positions)
    assert len(el.select_all(m, "face").faces) == 6
    assert len(el.select_all(m, "edge").edges) == 12
    assert el.select_all(m, "object").verts.size == 0

    some = el.ElementSel(faces=[0, 1])
    assert el.invert(m, some, "face").faces.tolist() == [2, 3, 4, 5]


def test_restrict_drops_what_a_shrunken_mesh_no_longer_has() -> None:
    m = prim.plane()  # 4 verts, 1 face
    wide = el.ElementSel(verts=[0, 9], edges=[[0, 1], [0, 9]], faces=[0, 4])
    got = el.restrict(m, wide)
    assert got.verts.tolist() == [0]
    assert got.edges.tolist() == [[0, 1]]
    assert got.faces.tolist() == [0]


def test_op_error_is_a_value_error_so_a_forgotten_catch_still_fails_loudly() -> None:
    assert issubclass(el.OpError, ValueError)

"""Where a dragged layer lands, and which rows the panel draws.

The layers panel draws top-first while the document is bottom-first, so every
drop is a translation between two orders. An off-by-one there is a layer that
lands one place from where it was released -- in a gesture whose *entire*
feedback is where it landed, which makes it the kind of defect a user reports
as "drag and drop is broken" rather than as an off-by-one.

So these assert the resulting order by actually performing the move, not by
comparing the returned index against a number written out here. A test that
restates the arithmetic cannot catch the arithmetic being wrong.
"""

from __future__ import annotations

from warlock.studio.plotter import layer_rows
from warlock.studio.plotter.tilemap import MapDoc


def _doc() -> MapDoc:
    doc = MapDoc(width=4, height=4, tile_w=16, tile_h=16)
    for layer in list(doc.layers):
        doc.remove_layer(layer.uid)
    return doc


def _named(doc: MapDoc) -> list[str]:
    """The root layers bottom-first, which is the document's own order."""
    return [layer.name for layer in doc.layers]


def _three() -> tuple[MapDoc, list[int]]:
    doc = _doc()
    uids = []
    for name in ("a", "b", "c"):
        doc.add_tile_layer()
        layer = doc.layers[-1]
        doc.set_layer_props(layer.uid, name=name)
        uids.append(layer.uid)
    assert _named(doc) == ["a", "b", "c"]
    return doc, uids


def _apply(doc: MapDoc, source: int, target: int) -> None:
    landing = layer_rows.drop_target(doc, source, target)
    assert landing is not None
    parent, index = landing
    doc.move_layer(source, index, parent_uid=parent)


def test_dragging_the_bottom_row_onto_the_top_row_takes_the_top():
    """Screen ``c b a``; drop ``a`` on ``c``'s row and ``a`` is the top row."""
    doc, (a, _b, c) = _three()
    _apply(doc, a, c)
    # Document is bottom-first, so the top of the screen is the end of the list.
    assert _named(doc) == ["b", "c", "a"]


def test_dragging_the_top_row_onto_the_bottom_row_takes_the_bottom():
    """The other direction, which is where an off-by-one would show."""
    doc, (a, _b, c) = _three()
    _apply(doc, c, a)
    assert _named(doc) == ["c", "a", "b"]


def test_dragging_one_row_swaps_with_its_neighbour():
    doc, (_a, b, c) = _three()
    _apply(doc, b, c)
    assert _named(doc) == ["a", "c", "b"]


def test_a_drop_onto_the_row_it_came_from_is_declined():
    """``None`` rather than a move that pushes an undo step doing nothing."""
    doc, (a, _b, _c) = _three()
    assert layer_rows.drop_target(doc, a, a) is None
    head = doc.history.head
    assert layer_rows.drop_target(doc, a, a) is None
    assert doc.history.head == head


def test_a_group_is_never_dropped_into_its_own_subtree():
    """``move_layer`` raises on this, and nothing wraps a pane draw.

    Declined *by returning None* rather than by letting the ValueError out:
    the row menu already filters the same case out of "Move into", and a drag
    that offers a landing it will then refuse is worse than one that does not
    light up.
    """
    doc = _doc()
    doc.add_group_layer()
    group = doc.layers[-1]
    doc.add_tile_layer()
    child = doc.layers[-1]
    doc.move_layer(child.uid, 0, parent_uid=group.uid)
    assert group.children and group.children[0].uid == child.uid

    assert layer_rows.drop_target(doc, group.uid, child.uid) is None
    assert layer_rows.drop_into_group(doc, group.uid, group.uid) is None


def test_dropping_into_a_group_appends_to_it():
    """The same landing ``Move into`` gives from the row menu."""
    doc = _doc()
    doc.add_group_layer()
    group = doc.layers[-1]
    doc.add_tile_layer()
    loose = doc.layers[-1]

    landing = layer_rows.drop_into_group(doc, loose.uid, group.uid)
    assert landing == (group.uid, 0)
    doc.move_layer(loose.uid, landing[1], parent_uid=landing[0])
    assert [child.uid for child in group.children] == [loose.uid]
    assert loose.uid not in [layer.uid for layer in doc.layers]


def test_visible_rows_is_top_first_and_carries_depth():
    doc = _doc()
    doc.add_group_layer()
    group = doc.layers[-1]
    doc.set_layer_props(group.uid, name="group")
    doc.add_tile_layer()
    inner = doc.layers[-1]
    doc.set_layer_props(inner.uid, name="inner")
    doc.move_layer(inner.uid, 0, parent_uid=group.uid)
    doc.add_tile_layer()
    doc.set_layer_props(doc.layers[-1].uid, name="top")

    rows = layer_rows.visible_rows(doc.layers, set())
    assert [(layer.name, depth) for layer, depth in rows] == [
        ("top", 0),
        ("group", 0),
        ("inner", 1),
    ]

    # A folded group keeps its own row and drops its children's.
    folded = layer_rows.visible_rows(doc.layers, {group.uid})
    assert [(layer.name, depth) for layer, depth in folded] == [
        ("top", 0),
        ("group", 0),
    ]


def test_only_rows_with_something_under_them_offer_a_fold():
    doc = _doc()
    doc.add_tile_layer()
    plain = doc.layers[-1]
    doc.add_group_layer()
    group = doc.layers[-1]

    assert layer_rows.can_fold(plain) is False
    assert layer_rows.can_fold(group) is False, "an empty group folds nothing"
    doc.add_tile_layer()
    doc.move_layer(doc.layers[-1].uid, 0, parent_uid=group.uid)
    assert layer_rows.can_fold(group) is True


def test_subtree_uids_agrees_with_the_documents_own_refusal():
    """Both walk ``children`` and nothing else, which is why they agree."""
    doc = _doc()
    doc.add_group_layer()
    outer = doc.layers[-1]
    doc.add_group_layer()
    inner = doc.layers[-1]
    doc.move_layer(inner.uid, 0, parent_uid=outer.uid)
    doc.add_tile_layer()
    leaf = doc.layers[-1]
    doc.move_layer(leaf.uid, 0, parent_uid=inner.uid)

    assert layer_rows.subtree_uids(outer) == {outer.uid, inner.uid, leaf.uid}

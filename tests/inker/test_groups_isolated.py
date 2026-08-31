"""Isolated group compositing: the second pass, and what it changes.

A group was a membership fact. An *isolated* one is a compositing stage: its
members blend onto transparency and the result blends once, which is what makes
a group-level blend mode mean anything and what confines a member's own mode to
its siblings.

The oracle here is :func:`_naive` -- a composite that walks the group tree the
slow, obvious way and shares no code with ``LayerStack._entries``. That
separation is the point. The real path flattens the tree into a list of entries
and hands the flat list to ``composite.stack_region``; a test that re-derived
the same flattening would agree with the implementation about a shared mistake.
The naive walk recurses over the tree itself and never builds an entry list at
all, so the two agree only if the flattening is right.

Two things also asserted here rather than left to read like an accident: a
document with nothing isolated composites down the path it always did, and the
``_below`` cache is refused for exactly one arrangement rather than for every
document that owns a group.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import composite as cp
from warlock.studio.inker import groups as gp
from warlock.studio.inker.document import Document
from warlock.studio.inker.layers import _shown_pixels

RED = (255, 0, 0, 255)
WHITE = (255, 255, 255, 255)
GREY = (128, 128, 128, 255)
BLUE = (0, 0, 255, 255)


def _doc(layers: int = 3) -> Document:
    doc = Document.blank(8, 8)
    doc.stack[0].name = "L0"
    doc.stack[0].pixels[:, :] = RED
    for i in range(1, layers):
        doc.add_layer(f"L{i}")
    doc.invalidate_all()
    return doc


def _group(doc: Document, rows: list[int], **props) -> gp.GroupNode:
    node = doc.group_layers(rows)
    assert node is not None
    if props:
        doc.set_group_props(node.uid, **props)
    doc.invalidate_all()
    return node


# --- the naive oracle --------------------------------------------------------


def _children(doc: Document, parent: int | None) -> list[int]:
    """The direct members of ``parent`` (None = root), bottom-first.

    A group takes the position of its lowest leaf, which is well defined
    because leaves are contiguous -- the invariant the whole tree rests on.
    """
    out: list[int] = []
    seen: set[int] = set()
    for uid in doc.member_uids():
        chain = gp.ancestry(doc.group_of, uid)
        if parent is None:
            top = chain[-1] if chain else None
        else:
            if parent not in chain:
                continue
            at = chain.index(parent)
            top = chain[at - 1] if at > 0 else None
        key = uid if top is None else top
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _layer_of(doc: Document, uid: int):
    order = doc.member_uids()
    return doc.stack[order.index(uid)]


def _naive(doc: Document, rect=None) -> np.ndarray:
    """Composite by walking the tree, with no reference to ``_entries``."""
    width, height = doc.size
    rect = rect or (0, 0, width, height)
    x0, y0, x1, y1 = rect
    out = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)

    def walk(parent, out, carry_opacity, carry_visible):
        for uid in _children(doc, parent):
            node = doc.groups.get(uid)
            if node is None:
                layer = _layer_of(doc, uid)
                opacity = layer.opacity * carry_opacity
                if not (layer.visible and carry_visible) or opacity <= 0.0:
                    continue
                source = cp.to_float(_shown_pixels(layer)[y0:y1, x0:x1])
                out = cp.over(out, source, opacity=opacity, mode=layer.blend)
                continue
            visible = carry_visible and node.visible
            if not gp.isolated(node):
                out = walk(uid, out, carry_opacity * node.opacity, visible)
                continue
            if not visible:
                continue
            inner = walk(uid, np.zeros_like(out), 1.0, True)
            out = cp.over(
                out,
                inner,
                opacity=node.opacity * carry_opacity,
                mode=node.blend,
            )
        return out

    return walk(None, out, 1.0, True)


def _shown(doc: Document, rect=None) -> np.ndarray:
    width, height = doc.size
    return doc.stack.composite_region(rect or (0, 0, width, height))


# --- the pure model ----------------------------------------------------------


def test_a_blend_mode_on_a_group_implies_isolation():
    assert gp.isolated(gp.GroupNode()) is False
    assert gp.isolated(gp.GroupNode(isolate=True)) is True
    assert gp.isolated(gp.GroupNode(blend="multiply")) is True


def test_isolation_is_meaningful_without_a_blend_mode():
    """The case that stops ``isolate`` being derived from ``blend``."""
    node = gp.GroupNode(isolate=True, blend="normal", opacity=1.0)
    assert gp.isolated(node) is True


def test_resolve_stops_opacity_at_the_isolated_ancestor():
    outer = gp.GroupNode(name="outer", opacity=0.5, isolate=True)
    inner = gp.GroupNode(name="inner", opacity=0.5)
    groups = {outer.uid: outer, inner.uid: inner}
    group_of = {inner.uid: outer.uid, 7: inner.uid}

    visible, opacity, locked = gp.resolve(groups, group_of, 7)
    # The inner pass-through half is folded in; the isolated group's own half
    # is carried by its Span instead, or it would be applied twice.
    assert opacity == pytest.approx(0.5)
    assert visible is True
    assert locked is False


def test_resolve_still_folds_visibility_and_the_lock_through_isolation():
    outer = gp.GroupNode(opacity=0.5, isolate=True, visible=False, locked=True)
    groups = {outer.uid: outer}
    visible, _opacity, locked = gp.resolve(groups, {7: outer.uid}, 7)
    assert visible is False
    assert locked is True


def test_spans_are_none_until_something_is_isolated():
    doc = _doc(3)
    _group(doc, [0, 1])
    assert doc.group_spans() is None
    assert doc.stack.group_spans is None


def test_a_span_is_the_groups_slice_of_the_stack():
    doc = _doc(4)
    node = _group(doc, [1, 2], blend="multiply")
    spans = doc.stack.group_spans
    assert spans is not None and len(spans) == 1
    assert (spans[0].guid, spans[0].lo, spans[0].hi) == (node.uid, 1, 3)
    assert spans[0].blend == "multiply"


def test_a_span_carries_its_pass_through_ancestrys_opacity():
    doc = _doc(4)
    # Nesting is by *narrowing*: the wider run first, so the narrower one lands
    # inside it -- ``group_layers`` takes its parent from the members it is given.
    outer = _group(doc, [1, 2, 3], opacity=0.5)
    inner = _group(doc, [1, 2], isolate=True, opacity=0.5)

    spans = doc.stack.group_spans
    assert [s.guid for s in spans] == [inner.uid]
    assert outer.uid not in {s.guid for s in spans}
    # A pass-through holder dims each member individually, and an isolated
    # group is one of its members: 0.5 of its own, 0.5 inherited.
    assert spans[0].opacity == pytest.approx(0.25)


def test_spans_come_back_outermost_first():
    doc = _doc(4)
    outer = _group(doc, [0, 1, 2], isolate=True)
    inner = _group(doc, [1, 2], isolate=True)
    assert [s.guid for s in doc.stack.group_spans] == [outer.uid, inner.uid]


def test_top_level_leaves_out_a_span_the_range_cuts():
    spans = [gp.Span(1, 0, 4, 1.0, "normal"), gp.Span(2, 1, 3, 1.0, "normal")]
    assert [s.guid for s in gp.top_level(spans, 0, 4)] == [1]
    assert [s.guid for s in gp.top_level(spans, 1, 4)] == [2]
    # 0..2 cuts both, so neither is returned rather than either being halved.
    assert gp.top_level(spans, 0, 2) == []


# --- what isolation changes --------------------------------------------------


def test_an_isolated_group_confines_a_members_blend_to_its_siblings():
    doc = _doc(3)
    doc.stack[0].pixels[:, :] = RED  # the document beneath
    doc.stack[1].pixels[0:4, :] = WHITE  # the group's own base, top half only
    doc.stack[2].pixels[:, :] = GREY
    doc.stack[2].blend = "multiply"
    node = _group(doc, [1, 2])

    through = _shown(doc).copy()
    doc.set_group_props(node.uid, isolate=True)
    doc.invalidate_all()
    isolated = _shown(doc)

    assert not np.allclose(through, isolated)
    # Where the group has a base of its own the two agree -- grey multiplies
    # white either way, and the white hides the red from both.
    assert np.allclose(through[0, 0], isolated[0, 0], atol=2 / 255)
    # Where it does not, the difference is the whole feature. Pass-through,
    # the multiply reaches the red beneath and darkens it.
    assert through[5, 0, 0] == pytest.approx(128 / 255, abs=2 / 255)
    assert through[5, 0, 1] == pytest.approx(0.0, abs=2 / 255)
    # Isolated, it multiplies the transparency that is all the group holds
    # there, and what lands on the red is an ordinary grey.
    assert isolated[5, 0, 0] == pytest.approx(128 / 255, abs=2 / 255)
    assert isolated[5, 0, 1] == pytest.approx(128 / 255, abs=2 / 255)
    assert np.allclose(isolated, _naive(doc), atol=1e-6)


def test_group_opacity_applies_once_under_isolation():
    doc = _doc(3)
    doc.stack[0].pixels[:, :] = RED
    doc.stack[1].pixels[:, :] = WHITE
    doc.stack[2].pixels[0:4, :] = BLUE  # covers half of its sibling
    node = _group(doc, [1, 2], opacity=0.5)

    through = _shown(doc).copy()
    doc.set_group_props(node.uid, isolate=True)
    doc.invalidate_all()
    isolated = _shown(doc)

    # Where the two members overlap, pass-through shows the lower one through
    # the upper one and isolation does not. That is the whole difference.
    assert not np.allclose(through[0, 0], isolated[0, 0])
    # Pass-through: the white goes on at 50%, and the blue goes on at 50% over
    # *that* -- so a washed-out white is still in the answer.
    assert through[0, 0] == pytest.approx([0.5, 0.25, 0.75, 1.0], abs=2 / 255)
    # Isolated: white then blue merge to an opaque blue, and the pair goes on
    # once at 50%. The white is covered before the opacity is ever applied.
    assert isolated[0, 0] == pytest.approx([0.5, 0.0, 0.5, 1.0], abs=2 / 255)
    assert np.allclose(isolated, _naive(doc), atol=1e-6)


def test_a_group_blend_mode_acts_on_the_groups_result():
    doc = _doc(3)
    doc.stack[0].pixels[:, :] = RED
    doc.stack[1].pixels[:, :] = GREY
    doc.stack[2].pixels[:, :] = (0, 0, 0, 0)
    _group(doc, [1, 2], blend="multiply")
    assert np.allclose(_shown(doc), _naive(doc), atol=1e-6)
    # Grey multiplied onto red: red stays, green and blue stay at nothing.
    shown = _shown(doc)
    assert shown[0, 0, 0] == pytest.approx(128 / 255, abs=2 / 255)
    assert shown[0, 0, 1] == pytest.approx(0.0, abs=2 / 255)


def test_a_hidden_isolated_group_draws_nothing():
    doc = _doc(3)
    doc.stack[1].pixels[:, :] = BLUE
    node = _group(doc, [1, 2], isolate=True)
    before = _shown(doc).copy()
    doc.set_group_props(node.uid, visible=False)
    doc.invalidate_all()
    after = _shown(doc)
    assert not np.allclose(before, after)
    assert np.allclose(after, _naive(doc), atol=1e-6)


def test_a_pass_through_group_composites_exactly_as_it_always_did():
    doc = _doc(4)
    for index in range(1, 4):
        doc.stack[index].pixels[index : index + 3, :] = BLUE
        doc.stack[index].opacity = 0.7
    _group(doc, [1, 2], opacity=0.5)
    assert doc.stack.group_spans is None
    assert np.allclose(_shown(doc), _naive(doc), atol=1e-6)


# --- the oracle, over an awkward tree ----------------------------------------


def _tangled() -> Document:
    """Nested isolation, mixed modes, partial coverage, a hidden member."""
    doc = _doc(6)
    colours = [RED, WHITE, BLUE, GREY, WHITE, BLUE]
    for index, colour in enumerate(colours):
        doc.stack[index].pixels[index : index + 3, index : index + 4] = colour
    doc.stack[2].blend = "multiply"
    doc.stack[3].blend = "screen"
    doc.stack[4].opacity = 0.6
    doc.stack[5].visible = False

    outer = doc.group_layers([1, 2, 3, 4])
    assert outer is not None
    inner = doc.group_layers([2, 3])
    assert inner is not None
    doc.set_group_props(outer.uid, isolate=True, opacity=0.75)
    doc.set_group_props(inner.uid, blend="overlay", opacity=0.5)
    doc.invalidate_all()
    return doc


def test_a_nested_tree_matches_the_naive_walk():
    doc = _tangled()
    assert len(doc.stack.group_spans) == 2
    assert np.allclose(_shown(doc), _naive(doc), atol=1e-6)


@pytest.mark.parametrize("rect", [(0, 0, 8, 8), (2, 1, 6, 7), (3, 3, 4, 4)])
def test_a_partial_rect_matches_the_naive_walk_over_the_same_rect(rect):
    doc = _tangled()
    assert np.allclose(_shown(doc, rect), _naive(doc, rect), atol=1e-6)


def test_a_group_that_is_the_whole_stack_matches_the_naive_walk():
    doc = _doc(3)
    for index in range(3):
        doc.stack[index].pixels[index:, :] = BLUE
    _group(doc, [0, 1, 2], blend="multiply", opacity=0.4)
    assert np.allclose(_shown(doc), _naive(doc), atol=1e-6)


# --- how the result reaches the compositor -----------------------------------


def test_a_group_arrives_as_one_prepared_entry_sized_to_the_rect():
    doc = _tangled()
    rect = (2, 1, 6, 7)
    entries = doc.stack._entries(0, len(doc.stack), rect)
    prepared = [e for e in entries if isinstance(e[0], cp.PreparedRegion)]
    assert len(prepared) == 1  # the outer group; the inner one is inside it
    region = prepared[0][0].region
    # Rect-sized, not canvas-sized: that is what keeps a dab-sized
    # invalidation a dab-sized allocation.
    assert region.shape == (6, 4, 4)
    assert region.dtype == np.float32
    assert prepared[0][1] == pytest.approx(0.75)


def test_the_native_stack_kernel_declines_a_prepared_region():
    region = cp.PreparedRegion(np.zeros((4, 4, 4), dtype=np.float32))
    entries = [(region, 1.0, "normal")]
    assert cp._stack_native(entries, (0, 0, 4, 4), None) is None


def test_stack_region_takes_a_prepared_region_uncropped():
    plane = np.zeros((8, 8, 4), dtype=np.uint8)
    plane[:, :] = RED
    region = np.zeros((2, 2, 4), dtype=np.float32)
    region[:, :] = (0.0, 0.0, 1.0, 1.0)
    out = cp.stack_region(
        [(plane, 1.0, "normal"), (cp.PreparedRegion(region), 1.0, "normal")],
        (1, 1, 3, 3),
    )
    assert out.shape == (2, 2, 4)
    assert np.allclose(out[..., 2], 1.0)


# --- the below cache ---------------------------------------------------------


def test_an_isolated_group_below_the_active_layer_keeps_the_cache():
    doc = _doc(4)
    for index in range(4):
        doc.stack[index].pixels[index : index + 2, :] = BLUE
    _group(doc, [0, 1], isolate=True, opacity=0.5)
    doc.set_active_layer(3)
    assert doc.stack.cuts_isolated(doc.stack.active_index) is False

    doc.invalidate((0, 0, 8, 8))
    assert doc._below is not None
    assert np.allclose(_shown(doc), _naive(doc), atol=1e-6)


def test_the_cache_is_refused_when_the_active_layer_is_inside_the_group():
    doc = _doc(4)
    for index in range(4):
        doc.stack[index].pixels[index : index + 2, :] = BLUE
    _group(doc, [1, 2], isolate=True, opacity=0.5)
    # The span is [1, 3). Active row 2 puts the split strictly inside it, so
    # the cached base would be the lower half of a group rendered alone.
    doc.set_active_layer(2)
    assert doc.stack.cuts_isolated(doc.stack.active_index) is True

    doc.invalidate((0, 0, 8, 8))
    assert doc._below is None
    assert np.allclose(_shown(doc), _naive(doc), atol=1e-6)


def test_a_boundary_is_not_a_cut():
    doc = _doc(4)
    _group(doc, [1, 2], isolate=True)
    # The span is [1, 3): a split at 1 puts it wholly above, at 3 wholly below.
    assert doc.stack.cuts_isolated(1) is False
    assert doc.stack.cuts_isolated(3) is False
    assert doc.stack.cuts_isolated(2) is True


def test_repeated_invalidation_inside_a_group_stays_correct():
    """The failure mode the cache rule exists to prevent: right once, then not."""
    doc = _doc(4)
    for index in range(4):
        doc.stack[index].pixels[index : index + 2, :] = BLUE
    _group(doc, [1, 2], isolate=True, blend="multiply")
    doc.set_active_layer(2)

    for _ in range(3):
        doc.stack[2].pixels[0:2, 0:2] = GREY
        doc.invalidate((0, 0, 4, 4))
        assert np.allclose(_shown(doc), _naive(doc), atol=1e-6)


# --- crossed with per-cel z --------------------------------------------------


def _animated_group() -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].name = "L0"
    doc.stack[0].pixels[:, :] = RED
    for index in range(1, 4):
        doc.add_layer(f"L{index}")
        doc.stack[index].pixels[:, :] = BLUE if index == 1 else GREY
    doc.invalidate_all()
    doc.ensure_animation()
    doc.set_current_frame(0)
    return doc


def test_a_lifted_row_cannot_leave_its_isolated_group():
    doc = _animated_group()
    _group(doc, [1, 2], isolate=True)
    # +8 is far more than the stack is tall; inside the span it can only reach
    # the top of its own group.
    assert doc.set_cel_z(8, track_index=1) is True
    doc.invalidate_all()

    entries = doc.stack._entries(0, len(doc.stack), (0, 0, 4, 4))
    prepared = [index for index, e in enumerate(entries) if isinstance(e[0], cp.PreparedRegion)]
    # Still one group, still holding both its rows: the lift stopped at the
    # boundary rather than hoisting the row out of the group.
    assert len(prepared) == 1
    assert len(entries) == 3  # L0, the group, L3


def test_a_lifted_row_outside_a_group_can_still_cross_the_whole_group():
    doc = _animated_group()
    _group(doc, [1, 2], isolate=True)
    assert doc.set_cel_z(8, track_index=0) is True
    doc.invalidate_all()

    entries = doc.stack._entries(0, len(doc.stack), (0, 0, 4, 4))
    kinds = ["group" if isinstance(e[0], cp.PreparedRegion) else "row" for e in entries]
    # L0 lifted over the group and L3: a row may cross a group, never enter it.
    assert kinds == ["group", "row", "row"]

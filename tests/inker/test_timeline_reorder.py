"""The layer drag-reorder source, and the assert it used to fire.

``_reorder`` drags a layer to a new position in the stack, and the item it
drags *from* is the layer's name -- which is ``imgui.text``, an item imgui adds
with id **0**. ``BeginDragDropSource`` asserts outright on a null id unless
``SourceAllowNullID`` tells it to derive one from the item's rectangle.

Reaching that assert needs the mouse held *and* either the row hovered or
something else in the same window holding the active id -- which is what made
this a live crash rather than a dormant one: holding the mouse on a timeline
cell or frame header makes it the active id inside ``inker-timeline-grid``, and
the next row's ``_reorder`` asserted on the same frame. What surfaced was
neither the row nor the reason: the exception unwound through
``layout.pane``'s ``finally``, whose ``end_child`` asserted in turn on the
now-unbalanced frame, so the error read ``Missing PopID()``. It also means
drag-reorder had never once worked -- the gesture it exists for takes exactly
the path that asserted.

**What this file does and does not cover.** The behavioural reproduction is
``scripts/exercise_mode.py --mode inker``, which crashed deterministically on
the timeline's frame buttons before the fix and completes after it. It is not
reproduced here, and the attempt is worth recording rather than quietly
dropping: standing the conditions up in a bare context did not reach the
assert, because a synthetic press never took the active id and a hovered
null-id item is not on its own enough. A test that passes with the fix removed
is worse than no test, so what is left is the flag pinned directly -- the
assert is native and aborts the frame rather than raising something a refactor
would notice -- plus the busy guard, which is ordinary Python and genuinely
testable.
"""

from __future__ import annotations

import inspect

from warlock.studio.panes import inker_timeline


def test_the_drag_source_allows_a_null_id():
    """The fix itself. Removing the flag restores a crash on a real click."""

    source = inspect.getsource(inker_timeline._reorder)
    assert "source_allow_null_id" in source
    assert "begin_drag_drop_source(flags)" in source


def test_the_name_the_drag_starts_from_is_still_a_null_id_item():
    """Why the flag is needed, pinned against the row that draws the name.

    All three branches draw *text*. If one ever became a selectable or a
    button the item would carry an id of its own and the flag would stop being
    load-bearing -- so this is the assumption the fix rests on, written down
    where it would break.
    """
    row = inspect.getsource(inker_timeline._track_row)
    assert "_reorder(ctx, tab, doc, track_index)" in row
    before = row.split("_reorder(ctx, tab, doc, track_index)")[0]
    tail = before[before.rindex("if active_track:") :]
    assert "widgets.text_colored(theme.ACCENT, label)" in tail
    assert "widgets.muted(label)" in tail
    assert "imgui.text(label)" in tail
    for id_bearing in ("controls.button", "controls.selectable", "imgui.selectable"):
        assert id_bearing not in tail, id_bearing


def test_a_busy_tab_registers_no_drag_source_at_all():
    """Unchanged, and load-bearing: ``begin_disabled`` does not stop a source.

    No imgui context: the guard is the first line of the function and returns
    before touching any of it, which is exactly the property being pinned.
    """

    class _Tab:
        busy = True

    moved: list[tuple[int, int]] = []

    class _Doc:
        stack = [object(), object()]

        def move_layer(self, source, target):
            moved.append((source, target))

    inker_timeline._reorder(None, _Tab(), _Doc(), 1)
    assert moved == []

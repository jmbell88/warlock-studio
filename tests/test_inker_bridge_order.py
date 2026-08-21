"""The bridge panel's section order is a decision, not an accident.

The right-hand column reads Layers (its own pane), then this panel: Canvas,
Animation, Pipeline, Document, File, with the recent list trailing the File
section. Pinned at the source level -- the sections are emitted straight-line
from ``draw``, so their order in the source *is* their order on screen.
"""

from __future__ import annotations

import inspect


def test_bridge_sections_draw_most_touched_first() -> None:
    from warlock.studio.panes import inker_bridge

    source = inspect.getsource(inker_bridge.draw)
    marks = (
        'section("Canvas")',
        "_animation(",
        "_pipeline(",
        'section("Document")',
        'section("File")',
    )
    positions = [source.index(mark) for mark in marks]
    assert positions == sorted(positions), (
        "inker_bridge.draw emits its sections out of the pinned order "
        "Canvas, Animation, Pipeline, Document, File"
    )


def test_recent_files_trail_the_file_buttons() -> None:
    from warlock.studio.panes import inker_bridge

    source = inspect.getsource(inker_bridge._file)
    assert source.index("widgets.recent_files") > source.rindex("_button"), (
        "the recent list draws after every File button, so its own Recent "
        "heading lands last in the column"
    )


def test_the_sheet_import_survives_having_no_tab() -> None:
    """Importing a sheet *makes* a document, so it has to be reachable when
    there is none -- which is exactly the moment a user wants it. It must not
    be parked behind an early return that only a tab reaches.

    New and Open are deliberately *not* on this list any more. They live on the
    empty canvas one column to the left, which is a full ``nothing_open`` with
    the presets, Open and the recent list on it; drawing them here as well was
    one pair of buttons in two places under a fourth copy of "Nothing open."
    """
    from warlock.studio.panes import inker_bridge

    source = inspect.getsource(inker_bridge.draw)
    head = source[: source.index("_sheet_import(")]
    assert "return" not in head, (
        "a return above the sheet import would make an empty session lose the "
        "one action that can create a document from this panel"
    )


def test_the_file_section_is_behind_the_no_tab_return() -> None:
    """The other half of the rule above: New/Open belong to the empty canvas,
    so the File block is for a document that exists."""
    from warlock.studio.panes import inker_bridge

    source = inspect.getsource(inker_bridge.draw)
    head = source[: source.index('section("File")')]
    assert "return" in head, (
        "the File block draws a second New/Open pair against the empty "
        "canvas's own; it belongs behind the no-tab return"
    )

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


def test_the_document_section_survives_having_no_tab() -> None:
    """Importing a sheet makes a document, so Document and File must both be
    emitted on the no-tab path -- the reorder must not park them behind an
    early return that only a tab reaches."""
    from warlock.studio.panes import inker_bridge

    source = inspect.getsource(inker_bridge.draw)
    document = source.index('section("Document")')
    file = source.index('section("File")')
    for position in (document, file):
        head = source[:position]
        assert "return" not in head, (
            "a return above the Document/File sections would make an empty "
            "session lose its New/Open/Import buttons"
        )

"""One document header in every workspace.

The 2026-09-05 consistency review: Packwright and Clay drew their own
New/Open/Save/Save As grids, Sirens and Plotter had none, Inker had no file
block in any pane, and the saved-state line was worded three ways. One widget
now, and ``controls.DOCUMENT_ACTION_ORDER`` -- declared and unread since the
redesign -- is what it renders from.
"""

from __future__ import annotations

import inspect

from warlock.studio import controls, widgets
from warlock.studio.panes import (
    clay_bridge,
    inker_generate,
    packwright_bridge,
    plotter_bridge,
    sirens_bridge,
)

BRIDGES = (clay_bridge, inker_generate, packwright_bridge, plotter_bridge, sirens_bridge)


def test_every_document_mode_draws_the_shared_header():
    for module in BRIDGES:
        assert "widgets.document_header(" in inspect.getsource(module), module.__name__


def test_no_bridge_hand_rolls_the_four_verbs_any_more():
    for module in BRIDGES:
        source = inspect.getsource(module)
        assert "Save As..." not in source, module.__name__
        assert '"unsaved changes"' not in source, module.__name__
        assert '"Unsaved changes."' not in source, module.__name__


def test_the_header_renders_from_the_declared_order():
    for word in controls.DOCUMENT_ACTION_ORDER[:4]:
        assert word in widgets.DOCUMENT_ACTION_LABELS
    assert widgets.DOCUMENT_ACTION_LABELS["Save"].endswith("Save (Ctrl+S)")
    assert "DOCUMENT_ACTION_ORDER" in inspect.getsource(widgets.document_header)


def test_the_status_ladder_is_one_sentence_and_exclusive():
    text = widgets.document_status_text
    assert text(None, False) == "Nothing to save yet."
    assert text(None, True) == "Not saved to a file yet."
    assert text("song.wsng", False) == "Saved."
    assert text("song.wsng", True) == "Unsaved changes."
    assert text("song.wsng", True, saving=True) == "Saving..."

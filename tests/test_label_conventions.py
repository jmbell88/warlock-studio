"""Two label conventions, asserted rather than trusted to review.

Both are the same argument: a label is read *before* the click, and its job is
to say what the click will cost. A convention that holds in four places and
lapses in a fifth is worse than none, because the reader has learned to believe
it by then.
"""

from __future__ import annotations

from warlock.studio import clay_ops, dialogs, inker_ops

#: An action that opens a picker, a popup or a confirm before it does anything
#: ends in an ellipsis. The op registries already keep it; the toolbars and the
#: context menus beside them did not, so the same three exports were "Export
#: sheet..." in the File menu and "Export sheet" on the timeline's own bar.
_OPENS_A_DIALOG = (
    "export_png",
    "export_sheet",
    "export_gif",
    "export_slices",
)


def test_an_inker_export_says_it_will_ask_first():
    for name in _OPENS_A_DIALOG:
        assert inker_ops.get(name).label.endswith("..."), name


def test_the_timeline_bar_agrees_with_the_file_menu_about_the_ellipsis():
    """The same three exports, reached from the other door. Read out of the
    source rather than by building a bar: the items are constructed inside a
    draw and every one of them opens a save dialog."""
    import re
    from pathlib import Path

    from warlock.studio.panes import inker_timeline

    source = Path(inker_timeline.__file__).read_text(encoding="utf-8")
    # Every ``"Export ..."`` string literal in the file. The one multi-line
    # docstring that opens with the word is excluded by the newline: a label is
    # one line by construction.
    labels = set(re.findall('"(Export[^"\n]*)"', source))
    assert len(labels) >= 10, labels
    for label in sorted(labels):
        assert label.endswith("..."), f"{label!r} opens a dialog and must say so"


def test_a_clay_op_that_opens_a_dialog_says_so():
    """The registry's own rule: ``hint`` is only shown by a parameterised op,
    which is exactly the set that opens a dialog."""
    for op in clay_ops.OPS:
        if op.params:
            assert op.label.endswith("..."), op.name


def test_the_cancel_labels_are_three_and_are_written_down():
    """A reader answers on the buttons, so the one thing worth keeping
    identical across five modes and six purges is the pair. The library used to
    disagree with itself -- "Keep" on two purges and "Cancel" on the three
    larger ones beside them."""
    assert dialogs.CANCEL_LABELS == ("Keep editing", "Keep", "Cancel")
    assert dialogs.Confirm(title="t", message="m").cancel_label == "Keep editing"


def test_every_close_without_saving_goes_through_the_one_helper():
    """Five modes ask it and it was spelled two ways -- four over [Discard],
    Inker over [Close]."""
    from pathlib import Path

    from warlock.studio import (
        clay_mode,
        inker_mode,
        packwright_mode,
        plotter_mode,
        sirens_mode,
    )

    for module in (clay_mode, inker_mode, packwright_mode, plotter_mode, sirens_mode):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "ask_close_unsaved" in source, module.__name__
        assert "Close without saving?" not in source, module.__name__

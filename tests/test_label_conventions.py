"""Two label conventions, asserted rather than trusted to review.

Both are the same argument: a label is read *before* the click, and its job is
to say what the click will cost. A convention that holds in four places and
lapses in a fifth is worse than none, because the reader has learned to believe
it by then.
"""

from __future__ import annotations

from warlock.studio import clay_ops, dialogs, inker_ops

#: Every one-line ``"Export ..."`` string literal in a module. A label is one
#: line by construction, so the newline excludes the docstrings that open with
#: the same word.
_EXPORT_LABEL = '"(Export[^"\n]*)"'

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


def test_every_export_door_agrees_with_the_file_menu_about_the_ellipsis():
    """The same exports, reached from the other door.

    This used to scan ``inker_timeline``'s source for ``"Export ..."``
    literals, because the five doors were toolbar items built inside a draw and
    there was no record to read. They left that row on 2026-09-05 and
    ``inker_export.DOORS`` is now the record, so the convention is asserted
    against the labels themselves -- stronger than a regex, and immune to the
    tooltips and failure messages in that module that legitimately open with
    the same word and are not labels.
    """
    from warlock.studio import inker_export

    assert len(inker_export.DOORS) == 5, inker_export.DOORS
    for door in inker_export.DOORS:
        assert door.label.endswith("..."), f"{door.label!r} opens a dialog and must say so"


def test_the_timeline_still_says_so_on_the_labels_it_kept():
    """The frame context menu's range and tag exports stayed behind, and the
    convention did not leave with the doors."""
    import re
    from pathlib import Path

    from warlock.studio.panes import inker_timeline

    source = Path(inker_timeline.__file__).read_text(encoding="utf-8")
    labels = set(re.findall(_EXPORT_LABEL, source))
    assert len(labels) >= 5, labels
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
        docmodes,
        inker_mode,
        packwright_mode,
        plotter_mode,
        sirens_mode,
    )

    # The question moved into ``docmodes.close_tab`` on 2026-09-05, so the
    # five modes ask it by calling that and none spells it out any more.
    assert "ask_close_unsaved" in Path(docmodes.__file__).read_text(encoding="utf-8")
    for module in (clay_mode, inker_mode, packwright_mode, plotter_mode, sirens_mode):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "docmodes.close_tab(" in source, module.__name__
        assert "Close without saving?" not in source, module.__name__

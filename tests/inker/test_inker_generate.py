"""The Generation panel: the four bridge verbs, surfaced as buttons.

Nothing here is new behaviour -- every button dispatches the op the File menu
row already dispatched. What is worth pinning is that the pane and the registry
cannot drift apart, and that a greyed button says why, which is the whole
reason the panel exists rather than staying a menu.
"""

from __future__ import annotations

from warlock.studio import inker_ops
from warlock.studio.panes import inker_generate


def test_every_button_names_an_op_that_actually_exists():
    """A renamed op would otherwise leave a button that raises on click."""

    names = [name for name, _note in inker_generate.OPS]
    assert names == ["send_to_3d", "save_as_reference", "add_to_packwright", "revert"]
    assert [op.name for op in inker_generate.ops()] == names


def test_every_button_is_a_file_menu_row_and_carries_a_refusal_sentence():
    """The panel is a second door onto the menu, not a second implementation --
    and a greyed control with no reason is worse than a missing one."""

    for op in inker_generate.ops():
        assert op.menu == "File", op.name
        assert op.reason, op.name


def test_each_button_has_a_note_saying_what_it_is_for():
    for _name, note in inker_generate.OPS:
        assert note and note[0].isupper() and note.endswith(".")


class _Tab:
    def __init__(self, job_id="", link_kind="", has_original=False):
        self.job_id = job_id
        self.link_kind = link_kind
        self.has_original = has_original

    @property
    def linked(self):
        return bool(self.job_id)


def test_the_link_line_explains_a_greyed_revert():
    """Revert is refused on a document with no original kept, and this line is
    where a user reads which half of that they are looking at."""

    assert "No drawing is open" in inker_generate.link_line(None)
    unlinked = inker_generate.link_line(_Tab())
    assert "Not linked" in unlinked and "Save as reference" in unlinked
    fresh = inker_generate.link_line(_Tab("job7", "reference-edit"))
    assert "job7" in fresh and "reference-edit" in fresh and "no original kept" in fresh
    edited = inker_generate.link_line(_Tab("job7", "reference-edit", has_original=True))
    assert "with the original kept" in edited


def test_revert_is_refused_with_a_sentence_on_an_unlinked_document():
    """The example the manual gives, asserted rather than described."""

    op = inker_ops.get("revert")

    class _State:
        pass

    reason = inker_ops.reason_for(op, _State(), _Tab())
    assert "no original" in reason.lower()



# --- 6.10: what the Undo History panel reads --------------------------------


def test_the_history_reads_as_one_timeline_oldest_first():
    """The undone half lives newest-first on its own stack; the panel shows one
    timeline in the order things happened."""
    from warlock.studio import inker

    doc = inker.Document.blank(8, 8)
    doc.add_layer()
    doc.add_layer()
    doc.undo()
    assert doc.history.history() == [("layer add", True), ("layer add", False)]


def test_stepping_to_an_index_goes_through_undo_and_redo():
    """Jumping is not a third operation on the stack -- an implementation that
    spliced the two lists would be a second definition of what a step is."""
    from warlock.studio import inker

    doc = inker.Document.blank(8, 8)
    doc.add_layer()
    doc.add_layer()
    assert len(doc.stack) == 3

    assert doc.history.step_to(doc, 0) is True
    assert len(doc.stack) == 1
    assert doc.history.step_to(doc, 2) is True
    assert len(doc.stack) == 3
    # And a step to where it already is moves nothing rather than churning.
    assert doc.history.step_to(doc, 2) is False


def test_stepping_past_either_end_is_clamped():
    from warlock.studio import inker

    doc = inker.Document.blank(8, 8)
    doc.add_layer()
    doc.history.step_to(doc, 99)
    assert len(doc.stack) == 2
    doc.history.step_to(doc, -5)
    assert len(doc.stack) == 1


def test_a_step_is_labelled_from_its_own_class():
    """An Edit has no name of its own, and giving each of the twenty a string
    is twenty places for a rename to be missed."""
    from warlock.studio import undo

    class LayerAddEdit:
        pass

    assert undo._label(LayerAddEdit()) == "layer add"

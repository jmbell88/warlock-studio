"""Which situation the keyboard is in, as a table rather than as an if-ladder.

The defect this closes is one Enter has caused in every editor that grew
modes: it means "apply the transform", "close the polygon" and "play", and if
the only thing keeping those apart is the order of three branches, the fourth
meaning is added at the wrong end of the function. First-match-wins over one
tuple makes them mutually exclusive by construction, and makes "which context
is this" a plain assertion.
"""

from __future__ import annotations

from warlock.studio import inker, inker_ops, inker_state


def _tab():
    doc = inker.Document.blank(16, 16)
    return inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")


def _session(tool: str = "brush"):
    state = inker_state.InkerState(tool=tool)
    tab = _tab()
    state.add(tab)
    return state, tab


def test_there_is_always_a_context():
    """Including with no document at all, which is the state the menu strip
    and the shortcut sheet are drawn in."""

    state, tab = _session(tool="eyedropper")
    assert inker_state.key_context(state, tab) == "Normal"
    assert inker_state.key_context(inker_state.InkerState(tool="move"), None) == "MoveTool"


def test_the_last_row_is_the_one_that_always_applies():
    name, applies = inker_state.KEY_CONTEXTS[-1]
    assert name == "Normal"
    assert applies(inker_state.InkerState(), None) is True


def test_a_transform_outranks_everything_under_it():
    state, tab = _session()
    tab.doc.select_all()
    state.gesture_pts = [(0, 0), (1, 1)]
    state.transforming = True
    assert inker_state.key_context(state, tab) == "Transformation"


def test_a_gesture_outranks_the_selection_it_will_replace():
    state, tab = _session(tool="lasso_poly")
    tab.doc.select_all()
    state.gesture_pts = [(0, 0), (1, 1)]
    assert inker_state.key_context(state, tab) == "Gesture"


def test_a_selection_outranks_the_tool_in_hand():
    state, tab = _session(tool="move")
    tab.doc.select_all()
    assert inker_state.key_context(state, tab) == "Selection"


def test_the_tool_contexts_are_reached_with_nothing_else_going_on():
    for tool, expected in (
        ("move", "MoveTool"),
        ("rect", "ShapeTool"),
        ("brush", "FreehandTool"),
        ("eyedropper", "Normal"),
    ):
        state, tab = _session(tool=tool)
        assert inker_state.key_context(state, tab) == expected, tool


def test_a_frame_range_is_its_own_context():
    state, tab = _session()
    tab.range_sel = (0, 0, 0, 0)
    assert inker_state.key_context(state, tab) == "FramesSelection"


def test_the_registry_can_only_name_a_context_the_table_can_return():
    """A binding in a context nothing produces is a key that never fires, and
    the failure would be silence rather than an error."""

    produced = {name for name, _applies in inker_state.KEY_CONTEXTS}
    assert set(inker_ops.CONTEXTS) == produced | {""}
    for op in inker_ops.OPS:
        assert op.context in inker_ops.CONTEXTS, op.name


def test_the_modal_arms_read_the_context_rather_than_the_state():
    import inspect

    from warlock.studio import inker_mode

    source = inspect.getsource(inker_mode.handle_key)
    assert "inker_state.key_context(state, tab)" in source
    # The three-branch ladder is gone from the key handler itself.
    assert "if state.transforming:" not in source
    assert "if state.gesture_pts:" not in source

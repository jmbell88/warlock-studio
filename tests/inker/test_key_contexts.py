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


# --- the registry is the binding table (W2.7) --------------------------------


def test_a_chord_is_spelled_the_way_the_menu_prints_it():
    """One spelling for the binding and the label, which is why ``Op.key`` is
    one field: a menu row cannot advertise a chord the keyboard will not
    answer if the row and the branch read the same string."""

    import pygame

    from warlock.studio import inker_mode

    def chord(key, *, ctrl=False, shift=False):
        event = pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)
        return inker_mode.chord_of(event, ctrl=ctrl, shift=shift)

    assert chord(pygame.K_z, ctrl=True) == "Ctrl+Z"
    assert chord(pygame.K_s, ctrl=True, shift=True) == "Ctrl+Shift+S"
    assert chord(pygame.K_RETURN) == "Enter"
    assert chord(pygame.K_UP, ctrl=True, shift=True) == "Ctrl+Shift+Up"
    assert chord(pygame.K_0, ctrl=True) == "Ctrl+0"


def test_every_bound_op_is_reachable_by_its_own_chord():
    """The property the whole table exists for: what the menu prints beside a
    row is what fires it."""


    from warlock.studio import inker_ops

    for op in inker_ops.OPS:
        if not op.key:
            continue
        assert inker_ops.by_key(op.key, op.context) is not None, op.name


def test_the_key_handler_asks_the_registry_before_its_own_branches():
    import inspect

    from warlock.studio import inker_mode

    source = inspect.getsource(inker_mode.handle_key)
    registry = source.index("inker_ops.by_key")
    fallback = source.index("_ctrl_key(")
    assert registry < fallback


def test_the_digits_set_brush_opacity_with_zero_meaning_full():
    """A key that made the brush invisible would be one nobody could tell from
    a broken tool."""

    from types import MethodType, SimpleNamespace

    import pygame

    from warlock.studio import inker, inker_mode
    from warlock.studio import state as state_mod

    doc = inker.Document.blank(8, 8)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="t")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    ctx = SimpleNamespace(state=app, toast=app.toast)

    for key, expected in ((pygame.K_3, 0.3), (pygame.K_9, 0.9), (pygame.K_0, 1.0)):
        inker_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        assert state.opacity == expected


def test_every_bound_key_resolves_in_every_context_it_can_be_pressed_in():
    """The guard the spelling check could not give. ``_contexts`` already
    refuses a context name ``key_context`` can never return, but a *correctly
    spelled* one that is almost never live is the same failure -- silence.

    ``play`` declared ``context="Normal"``, the last row of ``KEY_CONTEXTS`` and
    so the one that only answers when no other matched. ``by_key`` tries the
    requested context and then ops with *no* context, and "Normal" is truthy, so
    with any paint tool in hand the binding resolved to nothing and the op's
    refusal never spoke."""
    from warlock.studio import inker_ops, inker_state

    live = [name for name, _applies in inker_state.KEY_CONTEXTS]
    for op in inker_ops.OPS:
        if not op.key or op.context:
            continue
        for context in live:
            found = inker_ops.by_key(op.key, context)
            assert found is not None, f"{op.name} ({op.key}) is unreachable in {context}"


def test_a_context_bound_op_is_reachable_in_its_own_context():
    """And that a context-bound op is not shadowed out of existence: if it
    declares one, ``by_key`` has to find it there."""
    from warlock.studio import inker_ops

    for op in inker_ops.OPS:
        if not (op.key and op.context):
            continue
        assert inker_ops.by_key(op.key, op.context) is not None, op.name


def test_enter_resolves_to_play_whatever_tool_is_held():
    from warlock.studio import inker_ops, inker_state

    for name, _applies in inker_state.KEY_CONTEXTS:
        if name in ("Transformation", "Gesture"):
            # Consumed by ``_modal`` before ``by_key`` is asked.
            continue
        assert inker_ops.by_key("Enter", name).name == "play", name


def test_enter_on_a_still_document_says_why_with_a_brush_in_hand():
    """The user-visible half. With the binding unreachable, ``handle_key`` fell
    through to its raw ``toggle_play`` branch, which has no refusal to give --
    so Enter said nothing with a brush in hand and toasted correctly with an
    eyedropper, for the same document."""
    from types import MethodType, SimpleNamespace

    import pygame

    from warlock.studio import inker, inker_mode, inker_state
    from warlock.studio import state as state_mod

    doc = inker.Document.blank(8, 8)
    assert doc.anim is None, "a still document"
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="t")
    state = inker_state.InkerState()
    state.add(tab)
    state.set_tool("brush")
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    ctx = SimpleNamespace(state=app, toast=app.toast)

    assert inker_state.key_context(state, tab) == "FreehandTool"
    inker_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))

    assert state.tip is not None, "the refusal reached the user"
    assert "no frames yet" in str(state.tip.text)


def test_no_ctrl_chord_has_both_an_op_and_a_branch():
    """``_ctrl_key`` runs only after the registry has been asked, so a branch
    for a chord some op already carries is dead code -- and a reader has to
    check the registry to find that out. Ctrl+Shift+E, Ctrl+Shift+D and
    Ctrl+Shift+J were all in that state; Ctrl+4 and Ctrl+5 were half in it,
    with unadvertised shifted halves that no ``Op.key`` printed."""
    import inspect
    import re

    from warlock.studio import inker_mode, inker_ops

    source = inspect.getsource(inker_mode._ctrl_key)
    for chord in {op.key for op in inker_ops.OPS if op.key.startswith("Ctrl+")}:
        letter = chord.rsplit("+", 1)[-1].lower()
        if len(letter) != 1:
            continue
        if "Shift" in chord:
            pattern = rf'name == "{re.escape(letter)}" and shift'
        else:
            # Not followed by ``and shift``: Ctrl+Shift+Z is deliberately a
            # branch (redo's second spelling, which no op carries because an op
            # has one key), and it must not read as a collision with Ctrl+Z.
            pattern = rf'name == "{re.escape(letter)}"(?! and shift)'
        assert not re.search(pattern, source), f"{chord} has both an op and a branch"


def test_every_view_rotation_chord_is_advertised():
    from warlock.studio import inker_ops

    assert inker_ops.by_key("Ctrl+4", "").name == "rotate_view"
    assert inker_ops.by_key("Ctrl+Shift+4", "").name == "rotate_view_back"
    assert inker_ops.by_key("Ctrl+5", "").name == "flip_view"

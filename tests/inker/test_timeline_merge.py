"""The timeline absorbed the layers panel: one grid, always on screen.

The riskiest single change of the Inker wave, because it deletes a pane every
session uses. What is testable without a frame is the part that decides whether
a user can find their layers at all -- which is now simply *always* -- and the
fact that a still document has a row per layer and exactly one frame column.

The auto-show rule and the ``Tab`` toggle were pinned here until 2026-08-23.
Both are gone with the hiding they served: this strip holds the layer list, so
every state in which it is off screen is a document whose layers cannot be
seen or reached, and each of this pane's two shipped defects was an instance of
exactly that. What replaces them is one composition assertion -- the strip is
drawn on an unconditional path -- and the height drag, which is what hiding it
was really being used for.
"""

from __future__ import annotations

from warlock.studio import inker, inker_state
from warlock.studio.panes import inker_timeline


def _tab(uid: str = "t1"):
    doc = inker.Document.blank(16, 16)
    return inker_state.InkerDoc(doc=doc, uid=uid, title="Untitled")


def _session():
    state = inker_state.InkerState()
    tab = _tab()
    state.add(tab)
    return state, tab


def test_a_still_document_is_a_one_frame_sprite():
    """One column, and no model change: ``doc.anim`` is still None."""

    tab = _tab()
    assert tab.doc.anim is None
    assert inker_timeline.frame_uids(tab.doc) == [None]


def test_an_animated_document_has_a_column_per_frame():
    tab = _tab()
    tab.doc.ensure_animation()
    tab.doc.add_frame()
    assert len(inker_timeline.frame_uids(tab.doc)) == len(tab.doc.anim.frames) == 2


def test_the_rows_run_bottom_up():
    """Aseprite's order, Photoshop's order, and the order the grid's own frame
    columns already implied: the panel counted down and the grid counts up.

    Asserted through ``row_plan`` -- which is the walk now that group headers
    are folded into it -- rather than off the source, plus the source check
    that the old descending loop has not come back."""

    import inspect

    tab = _tab()
    for name in ("one", "two"):
        tab.doc.add_layer(name)
    plan = inker_timeline.row_plan(tab.doc)
    assert [entry.index for entry in plan] == [0, 1, 2]
    source = inspect.getsource(inker_timeline._grid)
    assert "for index in range(len(doc.stack))" in source
    assert "range(len(anim.tracks) - 1, -1, -1)" not in source


def test_the_range_readers_came_with_the_rows():
    assert callable(inker_timeline.track_range)
    assert callable(inker_timeline.extend_range)


def test_the_layers_pane_is_gone():
    from pathlib import Path

    panes = Path(inker_timeline.__file__).parent
    assert not (panes / "inker_layers.py").exists()


def test_an_eye_drag_paints_one_state_rather_than_flipping_each_row():
    """The value is the one the first row took; a flip-per-row drag would
    leave a striped stack behind whichever way the hand moved."""

    import inspect

    source = inspect.getsource(inker_timeline._drag_toggle)
    assert "state.eye_drag = not tab.doc.stack[index].visible" in source
    assert "!= state.eye_drag" in source


def test_toggle_all_shows_everything_when_anything_is_hidden():
    """With three of ten hidden, the button a user reaches for means "show
    everything" -- a strict-all rule would hide the other seven."""

    import inspect

    source = inspect.getsource(inker_timeline._toggle_all)
    assert "any(not layer.visible for layer in doc.stack)" in source


def test_the_eye_drag_starts_with_nothing_in_flight():
    assert inker_state.InkerState().eye_drag is None


def test_the_playback_tick_runs_where_the_strip_is_drawn():
    """``_tick`` is the only caller of ``tick_playback``, which is the only
    thing that advances ``play_index`` or ends a clip naturally.

    It used to have to sit above an early return, and then outside the pane
    altogether, because a hidden strip left ``tab.playing`` true for ever and
    ``tab.busy`` with it -- and the canvas then refused every gesture silently
    while the Stop button explaining why was off screen. The strip cannot be
    hidden any more, so the tick simply lives at the top of ``draw``; what is
    pinned is that it is still *unconditional* within it.
    """

    from pathlib import Path

    source = Path(inker_timeline.__file__).read_text(encoding="utf-8")
    body = source[source.index("def draw(ctx: Any) -> None:") :]
    body = body[: body.index("def _tick")]
    assert "_tick(tab)" in body
    assert body.index("_tick(tab)") < body.index("if tab.doc.anim is not None:"), (
        "the tick must not sit behind the animated branch"
    )


def test_the_strip_is_neither_hideable_nor_auto_shown_any_more():
    """The two doors that could take the layer list off screen, kept shut."""

    assert not hasattr(inker_timeline, "toggle")
    assert not hasattr(inker_timeline, "autoshow")
    assert not hasattr(inker_timeline, "tick_and_autoshow")
    assert not hasattr(inker_state.InkerState(), "timeline_open")
    assert not hasattr(inker_state.InkerState(), "timeline_shown")


def test_the_timeline_strip_is_drawn_unconditionally():
    """The gate that hid the layer list, pinned shut by AST.

    ``2a56df6`` merged the layers panel into this strip and said "the strip is
    always available, Tab toggles it" -- but the ``doc.anim is not None`` gate
    in ``_inker_workspace`` predated that commit and was never lifted, so a
    still document had no layer list at all and every headless test passed,
    because they all call this module's functions directly rather than walking
    the composition. A composition walk is the missing coverage class, and
    ``tests/test_layout.py`` already pins ``main.py`` by AST the same way.

    Now that the strip cannot be hidden either, the assertion is stronger: no
    condition anywhere in the workspace may mention the animation or an open
    flag, and the only thing the strip's own branch may ask about is whether
    there is a document at all.
    """
    import ast
    import inspect
    from pathlib import Path

    from warlock.studio import main as main_mod

    source = Path(inspect.getfile(main_mod)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    workspace = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_inker_workspace"
    )
    conditions = [
        ast.unparse(node.test)
        for node in ast.walk(workspace)
        if isinstance(node, ast.If)
    ]
    assert not any("anim" in text for text in conditions), conditions
    assert not any("timeline_open" in text for text in conditions), conditions
    # And the strip is composed: ``inker_timeline.draw`` is reached from a
    # branch that asks only whether a document is open.
    calls = [
        ast.unparse(node)
        for node in ast.walk(workspace)
        if isinstance(node, ast.Call)
    ]
    assert "inker_timeline.draw(ctx)" in calls


def test_the_range_drag_can_actually_ask_which_cell_it_is_over():
    """The v0.0.28 crash, and the shape that let it ship.

    ``_range_gesture`` splatted the grid's whole per-draw scratch into a
    keyword-only signature -- ``cell_index(point, **geom)`` -- so every
    *addition* to that scratch was a ``TypeError`` at the call rather than at
    import. `17b8210` cached ``columns`` and ``order`` there to stop the row
    loop rebuilding them, and the next press-and-drag in the timeline took the
    frame loop down with ``unexpected keyword argument 'columns'``.

    Nothing caught it because nothing headless drags, and this pane's own tests
    call ``cell_index`` directly with exactly the five arguments it wants --
    which is the one way to use it that could not fail. So what is pinned here
    is the *mapping*: the named key tuple against the real signature, in both
    directions, plus one call through the door the gesture actually uses.
    """
    import inspect

    signature = inspect.signature(inker_timeline.cell_index)
    keyword_only = tuple(
        name
        for name, param in signature.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert keyword_only == inker_timeline.CELL_GEOM_KEYS

    # And the scratch dict a real ``_grid`` builds carries more than those, so
    # the splat this replaced would still be a TypeError today.
    geom = {
        "x0": 0.0,
        "tops": {0: 0.0},
        "cell": 20.0,
        "gutter": 2.0,
        "frames": 3,
        "columns": [1, 2, 3],
        "order": [7],
    }
    assert set(geom) - set(inker_timeline.CELL_GEOM_KEYS) == {"columns", "order"}
    assert inker_timeline.hit_cell(geom, (5.0, 5.0)) == (0, 0)
    assert inker_timeline.hit_cell(geom, (45.0, 5.0)) == (0, 2)
    # Off the end is None rather than the nearest cell, exactly as
    # ``cell_index`` promises -- the wrapper adds no behaviour of its own.
    assert inker_timeline.hit_cell(geom, (500.0, 5.0)) is None


def test_the_strip_never_gets_less_than_its_floor():
    """"At minimum 150px" as arithmetic rather than as a screenshot: the share
    names the strip's portion of the centre column, and a tall window makes it
    bigger while a short one cannot make it smaller than ``STRIP_H``."""

    def strip(available: float, share: float, scale: float = 1.0) -> float:
        return max(inker_timeline.STRIP_H * scale, available * share)

    assert inker_timeline.STRIP_H == 150.0
    assert strip(400.0, 0.28) == 150.0
    assert strip(1000.0, 0.28) == 280.0
    # And the floor is design px, so it grows with the UI scale rather than
    # becoming a smaller and smaller fraction of the window.
    assert strip(400.0, 0.28, 1.5) == 225.0


# -- one gesture, one undo step ----------------------------------------------


def test_hiding_every_layer_is_one_undo_step():
    """``set_layer_props`` pushes its own edit per call that changes something,
    so the header's loop cost a ten-layer document ten Ctrl+Z to reverse one
    click -- against the one-gesture-one-step rule the filters, the palette
    conversion and ``apply_matte`` all follow."""

    doc = inker.Document.blank(8, 8)
    for _ in range(4):
        doc.add_layer()
    head = doc.history.head
    assert doc.set_all_layer_props(visible=False) is True
    assert all(not layer.visible for layer in doc.stack)
    assert doc.undo() is True
    assert all(layer.visible for layer in doc.stack)
    assert doc.history.head == head


def test_locking_every_layer_of_an_animated_document_is_one_step_too():
    doc = inker.Document.blank(8, 8)
    doc.add_layer()
    doc.ensure_animation()
    assert doc.set_all_layer_props(locked=True) is True
    assert all(track.locked for track in doc.anim.tracks)
    assert doc.undo() is True
    assert not any(track.locked for track in doc.anim.tracks)


def test_a_stack_that_already_agrees_records_nothing():
    """A no-op must not make a saved document ask to be saved again."""
    doc = inker.Document.blank(8, 8)
    doc.add_layer()
    assert doc.set_all_layer_props(visible=True) is False


def test_only_the_rows_that_change_contribute_a_step():
    doc = inker.Document.blank(8, 8)
    for _ in range(3):
        doc.add_layer()
    doc.set_layer_props(1, visible=False)
    head = doc.history.head
    assert doc.set_all_layer_props(visible=False) is True
    assert doc.undo() is True
    assert doc.history.head == head
    # Row 1 was already hidden and must stay hidden, not be shown again.
    assert [layer.visible for layer in doc.stack] == [True, False, True, True]


def test_an_eye_drag_is_one_step_for_every_row_it_crossed():
    """The gesture writes the rows live so the column follows the cursor, and
    asks for its undo entry once, on release. Eight rows crossed used to cost
    eight Ctrl+Z to put back."""

    doc = inker.Document.blank(8, 8)
    for _ in range(3):
        doc.add_layer()
    head = doc.history.head
    # What ``_drag_toggle`` does across three rows, then what release does.
    was = {}
    for index in (0, 1, 2):
        was[index] = {"visible": doc.stack[index].visible}
        doc.stack[index].visible = False
    assert doc.set_layers_props([0, 1, 2], was=was, visible=False) is True
    assert doc.undo() is True
    assert [layer.visible for layer in doc.stack] == [True, True, True, True]
    assert doc.history.head == head

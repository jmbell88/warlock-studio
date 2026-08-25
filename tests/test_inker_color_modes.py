"""The studio half of the colour modes: the mode door, and which slot the brush holds.

Two things the engine cannot check for itself. The mode buttons are a
whole-document conversion behind a click, so what matters is that a refusal
reaches the user rather than leaving a button that does nothing. And
``fg_slot`` is the answer to "which of two identical swatches did I click",
which a colour alone cannot give -- so what matters is that every *other* way of
choosing a colour clears it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from warlock.studio import inker, inker_mode
from warlock.studio.inker_state import InkerDoc, InkerState

BLACK = (0, 0, 0, 255)
RED = (200, 20, 20, 255)
GREEN = (20, 200, 20, 255)


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=InkerState())
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", *_: Any) -> None:
        self.toasts.append((text, level))


def _tab(ctx: _Ctx, width: int = 4, height: int = 4) -> InkerDoc:
    doc = inker.Document.blank(width, height)
    doc.stack.active.pixels[:, :] = RED
    # Two colours, not one: a palette built from a uniform drawing has a single
    # entry, and every "move the transparent slot" assertion below needs two.
    doc.stack.active.pixels[0, 0] = GREEN
    tab = InkerDoc(doc=doc, title="art")
    ctx.state.inker.add(tab)
    return tab


# --- the mode door ----------------------------------------------------------


def test_entering_indexed_with_no_palette_builds_one_from_the_drawing():
    """Two published operations and no third one, exactly as
    ``palette_from_document`` does for the constrained mode."""
    ctx = _Ctx()
    tab = _tab(ctx)

    assert inker_mode.set_color_mode(ctx, tab, "indexed") is True
    assert tab.doc.is_indexed and tab.doc.palette
    tab.doc.check_materialized()
    assert ctx.toasts and "transparent" in ctx.toasts[0][0]


def test_entering_indexed_keeps_a_table_the_user_already_authored():
    ctx = _Ctx()
    tab = _tab(ctx)
    tab.doc.set_palette([BLACK, RED, GREEN])

    inker_mode.set_color_mode(ctx, tab, "indexed")
    assert tab.doc.palette == [BLACK, RED, GREEN]
    tab.doc.check_materialized()


def test_the_mode_it_is_already_in_does_nothing_and_pushes_nothing():
    ctx = _Ctx()
    tab = _tab(ctx)
    head = tab.doc.history.head
    assert inker_mode.set_color_mode(ctx, tab, "rgb") is False
    assert tab.doc.history.head == head and not ctx.toasts


def test_a_refusal_reaches_the_user_with_the_attempt_in_front_of_it():
    """A silent False would leave a button that does nothing. The frame is the
    house rule: library text with no subject makes the reader work out what was
    being tried."""
    ctx = _Ctx()
    tab = _tab(ctx)
    tab.doc.set_palette([(i, i, i, 255) for i in range(257)])

    assert inker_mode.set_color_mode(ctx, tab, "indexed") is False
    assert tab.doc.color_mode == "rgb"
    text, level = ctx.toasts[0]
    assert text.startswith("Cannot switch to Indexed:") and "256" in text
    assert level == "warn"


def test_a_busy_tab_refuses_every_mode_change():
    """``index_to``'s gate: these rebind whole layer planes, and one landing
    mid-save writes an archive whose parts disagree."""
    ctx = _Ctx()
    tab = _tab(ctx)
    tab.saving = True
    assert inker_mode.set_color_mode(ctx, tab, "indexed") is False
    assert inker_mode.set_transparent_slot(ctx, tab, 1) is False


def test_grayscale_and_back_is_two_steps_and_two_toasts():
    ctx = _Ctx()
    tab = _tab(ctx)

    assert inker_mode.set_color_mode(ctx, tab, "grayscale") is True
    assert inker_mode.set_color_mode(ctx, tab, "rgb") is True
    tab.doc.undo()
    assert tab.doc.color_mode == "grayscale"
    tab.doc.undo()
    assert tab.doc.color_mode == "rgb"


def test_moving_the_transparent_slot_goes_through_the_mode_layer():
    ctx = _Ctx()
    tab = _tab(ctx)
    inker_mode.set_color_mode(ctx, tab, "indexed")
    ctx.toasts.clear()

    assert inker_mode.set_transparent_slot(ctx, tab, 1) is True
    assert tab.doc.transparent_index == 1
    tab.doc.check_materialized()
    assert "transparent" in ctx.toasts[0][0]
    # The same slot twice is not a change, so it is not a toast either.
    ctx.toasts.clear()
    assert inker_mode.set_transparent_slot(ctx, tab, 1) is False
    assert not ctx.toasts


def test_a_mode_change_clears_the_slot_selection_and_the_brush_s_slot():
    """All three are answers about a table that has just been replaced."""
    ctx = _Ctx()
    tab = _tab(ctx)
    state = ctx.state.inker
    state.palette_slot, state.palette_slots, state.fg_slot = 3, [2, 3], 3

    inker_mode.set_color_mode(ctx, tab, "indexed")
    assert (state.palette_slot, state.palette_slots, state.fg_slot) == (0, [], None)


# --- which slot the brush holds ---------------------------------------------


def test_only_a_palette_click_records_a_slot():
    """``set_fg`` is one door because what has to be right is the *clearing*: a
    clear that lives at three of the four sites is a brush that goes on claiming
    a slot the user moved away from."""
    state = InkerState()

    state.set_fg(RED, 3)
    assert (state.fg, state.fg_slot) == (RED, 3)

    state.set_fg(GREEN)
    assert (state.fg, state.fg_slot) == (GREEN, None)


def test_every_foreground_assignment_in_the_studio_goes_through_set_fg():
    """A scan rather than four separate tests, so a fifth way of choosing a
    colour cannot be added without answering the question."""
    import ast
    from pathlib import Path

    import warlock.studio as studio

    root = Path(studio.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "fg"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "state"
                ):
                    offenders.append(f"{path.name}:{target.lineno}")
    assert offenders == [], "assign the foreground through InkerState.set_fg"


def test_the_slot_reaches_the_document_before_any_branch_of_a_press():
    """One handover for every tool at once, and it has to be **above** the tool
    branches: three of them return early (the slice tool, an inert right button,
    a locked layer), so a handover placed beside the brush would be missing for
    the fill, the shape and the gradient.

    Asserted on the source rather than by driving a press, because ``_press``
    reaches imgui on its second line and standing up a second imgui context in
    the test process crashes it -- the hazard ``test_studio_smoke`` documents.
    """
    import ast
    import inspect
    import textwrap

    from warlock.studio.panes import inker_canvas

    body = ast.parse(textwrap.dedent(inspect.getsource(inker_canvas._press))).body[0].body
    statements = [node for node in body if not isinstance(node, ast.Expr)]
    assigns = [
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "paint_slot"
    ]
    assert assigns, "_press must hand the palette slot to the document"
    returns = [
        i for i, node in enumerate(statements) for _ in ast.walk(node) if isinstance(_, ast.Return)
    ]
    assert assigns[0] < min(returns, default=len(statements)), (
        "the handover sits after a branch that returns"
    )


def test_the_document_holds_the_slot_the_state_was_given():
    """The other half, without imgui: what ``_press`` copies over is exactly
    what ``set_fg`` recorded, including the None."""
    ctx = _Ctx()
    tab = _tab(ctx)
    state = ctx.state.inker
    inker_mode.set_color_mode(ctx, tab, "indexed")

    state.set_fg(tab.doc.palette[1], 1)
    tab.doc.paint_slot = state.fg_slot
    tab.doc.begin_stroke((1, 1), tab.doc.palette[1], size=1, nib="pixel")
    tab.doc.end_stroke()
    tab.doc.check_materialized()
    assert tab.doc.stack.active.indices[1, 1] == 1

    state.set_fg(GREEN)
    assert state.fg_slot is None


# --- the dither the mode change uses ----------------------------------------


def _ramp_tab(ctx: _Ctx) -> InkerDoc:
    """A horizontal grey ramp: something a dither has an answer for and a
    nearest-match snap does not. Four colours because slot 0 is the transparent
    index, so a two-entry table leaves one usable colour and no choice to make.
    """
    import numpy as np

    doc = inker.Document.blank(16, 4)
    ramp = np.linspace(0, 255, 16).astype("uint8")
    doc.stack.active.pixels[:, :, :3] = ramp[None, :, None]
    doc.stack.active.pixels[:, :, 3] = 255
    tab = InkerDoc(doc=doc, title="ramp")
    ctx.state.inker.add(tab)
    return tab


def test_the_mode_door_takes_the_dither_method():
    """``set_color_mode`` hard-coded ``"nearest"``, so the Convert popup's
    choice of matrix could not reach the one operation that changes mode."""
    import numpy as np

    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    assert (
        inker_mode.set_color_mode(ctx, tab, "indexed", method="floyd-steinberg", max_colours=4)
        is True
    )
    plane = tab.doc.composite[..., 0]
    # A nearest snap makes every row identical -- the answer depends only on
    # the column. A diffused one does not.
    assert not np.array_equal(plane[0], plane[1])


def test_the_default_is_still_a_nearest_snap():
    import numpy as np

    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    assert inker_mode.set_color_mode(ctx, tab, "indexed", max_colours=4) is True
    plane = tab.doc.composite[..., 0]
    assert np.array_equal(plane[0], plane[1])

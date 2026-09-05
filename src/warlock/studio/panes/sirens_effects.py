"""Sirens' third right-column pane: the song's sound effects.

**A list over the document, and nothing more.** A one-shot has been a
first-class part of a ``.wsng`` since Phase 1 -- ``SongDoc.oneshots`` holds
``OneShot(uid, name, pattern, tempo, speed)``, ``add_oneshot`` mints an effect
*and a pattern of its own* as one collapsed undo step, and ``synth.render_oneshot``
renders it at its own tempo -- so what was missing was never a model, it was a
way in. This is that: add, remove, rename, the two timing fields, and Audition.

**The grid is the effect editor.** Selecting an effect points the caret at that
effect's pattern (``sirens_mode.set_caret`` takes a pattern uid), so writing a
coin pickup uses the same five columns, the same piano row and the same undo
stack as writing a bassline. A second, smaller grid in this sidebar would be a
second set of keyboard bindings for the same job, and the one it would be
smaller than is the one people already know.

**Why an effect keeps its own tempo and speed.** ``document.OneShot`` states it:
a coin pickup is forty milliseconds whatever the music is doing, and tying it to
the song's tempo would mean every effect in the document changed length the
moment somebody slowed the track down. So the two fields are here, per row, and
the transport's pair does not reach them.

**Auditioning does not touch the song's buffer.** It goes through
``sirens_mode.audition``, which renders under its own task key and hands the
samples straight to the mixer -- see ``AUDITION_PREFIX`` for why sharing
``sirens-render:`` would leave a coin pickup where the song used to be.
"""

from __future__ import annotations

from typing import Any

from .. import anchors, controls, icons, sirens_audio, sirens_mode, tokens, widgets
from ..manual import render as manual_render
from ..sirens import document as D
from ..sirens import instruments as inst
from ..tokens import sp

#: How many rows a new effect's pattern gets. ``document.add_oneshot``'s own
#: default, named here because this pane draws the sentence that explains it: at
#: the default speed eight rows is about a third of a second, which is longer
#: than nearly every effect and short enough that the whole thing is on screen
#: without scrolling.
NEW_ROWS = 8


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    anchors.mark_window("sirens/effects")
    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Sound effects")
    manual_render.help_button(ctx, "sirens-effects")

    if tab is None:
        return

    doc = tab.doc
    editable = not tab.busy
    busy_why = "This song is being written; the buttons come back when it lands."

    width = widgets.grid_width(2)
    if widgets.disabled_button(f"{icons.PLUS} Add", editable, (width, 0), reason=busy_why):
        _add(ctx, state, tab)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.TRASH} Delete",
        editable and state.oneshot is not None,
        (width, 0),
        reason=busy_why if editable else "No sound effect is selected.",
    ):
        _remove(ctx, state, tab)

    if not doc.oneshots:
        widgets.muted_wrapped(
            "No sound effects yet. Each one is a little pattern of its own,"
            " played at its own tempo and exported to sfx/ beside the song."
        )
        return

    imgui.dummy((0, sp(tokens.SP_1)))
    _rows(ctx, state, tab, editable)

    selected = None if state.oneshot is None else doc.oneshot(state.oneshot)
    if selected is None:
        return
    imgui.dummy((0, sp(tokens.SP_2)))
    _fields(state, tab, selected, editable)


def _add(ctx: Any, state: Any, tab: Any) -> None:
    """One new effect, selected, with the caret already in its pattern.

    Selecting it here rather than leaving the user to click the row they just
    made: ``add_oneshot`` creates a pattern nobody is looking at, and an Add
    button whose only visible result is one more row in a list is a button that
    appears not to have worked.
    """
    try:
        one = tab.doc.add_oneshot(rows=NEW_ROWS)
    except ValueError as exc:
        # ``MAX_ONESHOTS``, which is the only way this refuses. Framed rather
        # than swallowed: the ceiling is reachable by working, and a button that
        # silently stops adding is worse than one that says why.
        ctx.toast(f"That sound effect was not added: {exc}", "error")
        return
    _select(ctx, state, one)


def _remove(ctx: Any, state: Any, tab: Any) -> None:
    """Drop the selected effect. Its pattern is **left in the document.**

    ``remove_oneshot``'s own behaviour, and the one worth saying out loud here
    because this pane is where somebody would notice: the pattern the effect
    named survives in the pattern list, exactly as an instrument survives a
    sample being removed. It is undoable either way, and a removal that also
    deleted a pattern the user had put in the song's order would be a removal
    that changed the music.
    """
    uid = state.oneshot
    try:
        removed = tab.doc.remove_oneshot(uid)
    except ValueError as exc:
        ctx.toast(f"That sound effect was not removed: {exc}", "error")
        return
    if removed:
        state.oneshot = None
        sirens_mode.clamp_caret(ctx, tab)


def _select(ctx: Any, state: Any, one: Any) -> None:
    """Point the pane's selection and the grid's caret at the same effect."""
    state.oneshot = one.uid
    sirens_mode.set_caret(ctx, pattern=one.pattern)


def _label(doc: Any, one: Any, index: int) -> str:
    rows = doc.pattern(one.pattern)
    length = f"{rows.rows} rows" if rows is not None else "no pattern"
    return f"{one.name or f'Effect {index + 1}'}  ({length})"


def _rows(ctx: Any, state: Any, tab: Any, editable: bool) -> None:
    """One row per effect: the name, and the button that plays it.

    Audition is per row rather than one button under the selection, because the
    thing a user does with a folder of sound effects is listen down the list --
    and a design where that costs two clicks per effect is one where nobody
    checks the last five.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    device = sirens_audio.available()
    # One sentence per way the button can be dead, and the device's is
    # ``sirens_audio``'s own so this pane and the transport cannot say two
    # different things about the same missing card.
    why = (
        "This song is being written; the buttons come back when it lands."
        if not editable
        else sirens_audio.unavailable_reason()
    )
    for index, one in enumerate(list(doc.oneshots)):
        if controls.selectable(
            f"{_label(doc, one, index)}###sirens-oneshot-{one.uid}",
            state.oneshot == one.uid,
        )[0]:
            _select(ctx, state, one)
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.PLAY}###sirens-oneshot-play-{one.uid}",
            editable and device,
            (0, 0),
            reason=why,
            tooltip="Render this effect and play it once.",
        ):
            sirens_mode.audition(ctx, tab, one.uid)


def _fields(state: Any, tab: Any, selected: Any, editable: bool) -> None:
    """The selected effect's name, tempo and speed, and where it is edited.

    Every change goes through ``update_oneshot``, which is what makes it one
    reversible step and what refuses a no-op -- a slider held still is a stream
    of frames, and every one of them would otherwise be an undo step.

    None of the three re-arms the renderer, and that is not an omission: an
    effect is not in the song's order list, so nothing about it can change what
    ``song.wav`` sounds like. What it changes is the next audition and the next
    export, both of which read the document when they run.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    imgui.set_next_item_width(-1)
    name = widgets.input_text(
        "Name", selected.name, max_length=inst.MAX_NAME_LEN, commit=True
    )
    if name != selected.name:
        doc.update_oneshot(selected.uid, name=name)

    imgui.set_next_item_width(-1)
    changed, value = controls.slider_int(
        "Tempo", selected.tempo, D.MIN_TEMPO, D.MAX_TEMPO, enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed:
        doc.update_oneshot(selected.uid, tempo=int(value))
    imgui.set_next_item_width(-1)
    changed, value = controls.slider_int(
        "Speed", selected.speed, D.MIN_SPEED, D.MAX_SPEED, enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed:
        doc.update_oneshot(selected.uid, speed=int(value))

    pattern = doc.pattern(selected.pattern)
    if pattern is None:
        # Unreachable through the app -- ``add_oneshot`` mints the pattern and
        # the pair is one undo step -- but a hand-edited ``.wsng`` can carry it,
        # and a row that renders as an exception is worse than one that says
        # what is wrong.
        widgets.muted_wrapped("This effect names a pattern this song no longer holds.")
        return
    # Only while the caret is actually in it: clicking a song pattern in the
    # Order panel leaves this row selected -- the selection is what the two
    # sliders edit -- and a panel that then insisted the grid was showing the
    # effect would be telling the user something they can see is untrue.
    if state.pattern == selected.pattern:
        widgets.muted_wrapped(
            "The grid is editing this effect. Click a pattern in the Order panel"
            " to go back to the song."
        )
    else:
        widgets.muted_wrapped("Select this effect again to edit it in the grid.")

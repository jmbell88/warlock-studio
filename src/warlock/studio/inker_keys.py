"""Inker's keyboard: one dispatcher, the chord speller and the two gates.

handle_key is the mode's whole keyboard and the arms are in a deliberate order
-- the modal gates first (a transform and a multi-click gesture consume
everything), then the *registry*, which answers for every op that carries a
key, then the bindings that are not ops: the tool letters, the sizes, the
nudges and the modeless view keys.

Lifted out of ``studio/inker_mode`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed, so the
move is code motion over tested behaviour rather than a rewrite.

``inker_mode`` is imported as a *module* and never ``from``-imported: every
attribute is resolved at call time, so this file and its parent may be
imported in either order. The parent serves these names back through a PEP
562 ``__getattr__``, which is what keeps ``inker_mode.export_png`` and the
rest working for every caller and every test.
"""

from __future__ import annotations

from typing import Any

from . import inker_mode, inker_ops, inker_state
from .inker_state import InkerDoc, InkerState


def handle_key(ctx: Any, event: Any) -> bool:
    """Inker's shortcuts. -> whether the key was consumed.

    Consumed unconditionally while a document is open, exactly as the old
    inline editor did: F, W and S would otherwise frame and wireframe a
    viewport that is not on screen.
    """
    import pygame

    state = ctx.state.inker
    if state is None:
        return False

    if event.key == pygame.K_SPACE:
        # Seen on both edges: space-to-pan is a hold, not a toggle.
        #
        # **Above the "is there a document" returns below**, which is the whole
        # of the fix: they used to sit in front of this, so holding Space and
        # closing the last tab dropped the release and left the flag on for the
        # rest of the session -- every left-drag panned instead of painting, and
        # every tool press was suppressed. A release is honoured whether or not
        # there is anything to pan, because the flag outlives the document.
        # ``plotter_mode.handle_key`` learned the same lesson at its own door.
        state.space_held = event.type == pygame.KEYDOWN
        return bool(state.docs)
    if not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False

    quick_key = ""
    if event.key in (pygame.K_LALT, pygame.K_RALT):
        quick_key = "Alt"
    elif event.key in (pygame.K_LCTRL, pygame.K_RCTRL):
        quick_key = "Ctrl"
    if quick_key:
        if event.type == pygame.KEYUP and state.quick_key == quick_key:
            previous = state.quick_tool
            state.quick_key = state.quick_tool = ""
            if previous:
                state.set_tool(previous)
            return True
        if event.type == pygame.KEYDOWN and not state.quick_key:
            quick = inker_ops.resolve_binding(
                quick_key,
                inker_state.key_context(state, tab),
                state.shortcut_overrides,
                trigger="hold",
            )
            if quick is not None and quick.kind == "tool":
                state.quick_key, state.quick_tool = quick_key, state.tool
                state.set_tool(quick.target)
                return True
    doc = tab.doc
    if event.type != pygame.KEYDOWN:
        return True

    # ``event.mod``, never ``pygame.key.get_mods()`` -- ``main._shortcut``'s
    # rule (main.py:2340), and Inker was the last mode still breaking it.
    # ``mod`` is the modifier state at the moment this key was *pressed*;
    # ``get_mods()`` is the state now, after the event batch drained, so a
    # Ctrl released between the press and this call read as never held.
    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    alt = bool(mods & pygame.KMOD_ALT)
    name = pygame.key.name(event.key)
    # Built here rather than at module scope: pygame is imported lazily in this
    # function, so a module-level table would drag it into every import of the
    # mode. Four entries costs nothing per keypress.
    arrows = {
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
    }

    # **The modal arms are a table now** (W2.8). Which situation the keyboard is
    # in is ``inker_state.key_context``, first-match-wins over one tuple, so the
    # contexts are mutually exclusive by construction instead of by three
    # branches each remembering the other two -- which is how Enter came to mean
    # "apply the transform", "close the polygon" and "play" in one function with
    # the order of the ifs as the only thing keeping them apart.
    context = inker_state.key_context(state, tab)
    if _modal(ctx, state, tab, context, name, event, ctrl=ctrl):
        return True

    # **The registry answers first.** Every op that carries a key is bound
    # here, once, from the same field the menu row prints -- so a chord cannot
    # be advertised in one place and implemented in another, which is what
    # eleven of these branches used to be. What is left below is the bindings
    # that are not ops: the tool letters, the sizes, the nudges, the modeless
    # view keys.
    chord = chord_of(event, ctrl=ctrl, shift=shift, alt=alt)
    op = inker_ops.by_key(chord, context, state.shortcut_overrides)
    if op is not None:
        # No ``_MUTATING_CTRL`` check here any more. ``inker_ops.run`` enforces
        # the op's own ``enabled`` gate and *says why* when it refuses, and
        # every op bound to one of those chords is ``when_ready``-gated -- so
        # the second check refused the same presses silently, which is the one
        # thing ``run``'s refusal exists to stop. The set still guards the raw
        # arms in ``_ctrl_key``, which are not ops and have no gate of their
        # own; ``tests/inker/test_findings_engine.py`` pins the pairing.
        inker_ops.run(ctx, op)
        return True
    binding = inker_ops.resolve_binding(chord, context, state.shortcut_overrides)
    if binding is not None and binding.kind == "tool":
        state.set_tool(inker_state.cycle_in_group(state.tool, binding.target))
        return True

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, event, shift=shift)

    if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        # Aseprite's zoom in, and ``=`` unshifted answers it too because that
        # is the same physical key on every layout this ships to.
        tab.view.pending_zoom_rung = 1
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        tab.view.pending_zoom_rung = -1
    elif name == "x":
        state.swap_colours()
    elif alt and name.isdigit() and name != "0":
        # Aseprite's Alt+1..9. Shift stores, plain recalls -- Plotter's stamp
        # slots' rule and its reason: recall happens hundreds of times a
        # session and storing nine times, so the cheap gesture goes to the
        # frequent one.
        slot = int(name)
        if shift:
            if not state.store_stamp(slot):
                state.say("There is no captured brush to store -- Ctrl+B captures one.")
        elif not state.recall_stamp(slot):
            state.say(f"Brush {slot} is empty -- Alt+Shift+{slot} stores one.")
    elif not shift and name.isdigit():
        # **The number row was entirely unbound in Inker**, and Aseprite's
        # answer to the same spare keys is the same one: opacity in tenths,
        # with 0 meaning full rather than nothing -- a key that made the brush
        # invisible would be one nobody could tell from a broken tool.
        digit = int(name)
        state.opacity = 1.0 if digit == 0 else digit / 10.0
    elif event.key == pygame.K_LEFTBRACKET:
        if shift:
            state.hardness = max(0.0, state.hardness - 0.05)
        else:
            state.brush_size = inker_state.step_size(state.brush_size, -1)
    elif event.key == pygame.K_RIGHTBRACKET:
        if shift:
            state.hardness = min(1.0, state.hardness + 0.05)
        else:
            state.brush_size = inker_state.step_size(state.brush_size, +1)
    # ``,`` / ``.`` / Enter had a raw arm here as well as their ops
    # (``prev_frame``, ``next_frame``, ``play``). That was a workaround for a
    # context bug in the ``play`` op which is fixed -- see its registration --
    # and it outlived it: two bindings for one key, of which only the registry's
    # is remappable and only the registry's carries the "no frames yet"
    # refusal. The registry answers first, a hundred lines up, so these arms
    # were unreachable for a still document and redundant for an animated one.
    elif event.key in arrows:
        step = inker_mode.NUDGE_STEP if shift else 1
        dx, dy = arrows[event.key]
        # **The return value is read now.** A inker_mode.nudge onto a locked layer came
        # back False and the answer was thrown away, so the arrows did nothing
        # and said nothing -- while the *same* refusal reached by a mouse press
        # raised a toast. The two doors gave two different answers to one
        # question, and the quiet one is the one a user meets by accident.
        # Only when the lock is what actually refused it, though: ``inker_mode.nudge``
        # also declines for a busy tab and for a tool the arrows do not serve
        # (no floating buffer, not the move tool), and blaming the lock for
        # either would be a toast naming the wrong problem.
        if (
            not inker_mode.nudge(state, tab, dx * step, dy * step)
            and not tab.busy
            and (state.tool == "move" or doc.floating is not None)
            and doc.write_locked()
        ):
            state.say(inker_mode.LOCKED_LAYER, remedy="layer_properties", remedy_label="Unlock")
    elif event.key == pygame.K_DELETE:
        if not tab.busy and not doc.delete_selection() and doc.write_locked():
            state.say(inker_mode.LOCKED_LAYER, remedy="layer_properties", remedy_label="Unlock")
    elif event.key == pygame.K_ESCAPE:
        # Never leaves the mode: Esc means "drop what I am doing", and losing a
        # workspace full of tabs to a stray keypress is not that.
        # The move session goes back **unconditionally**, beside ``clear_drag``
        # below and for a stronger version of its reason. It is the one open
        # gesture that has already *written* previewed pixels into the layer
        # with no undo step behind them, so dropping the drag state without it
        # leaves the layer moved, clean and unrecoverable -- and mid-save those
        # pixels are exactly what the encoder is reading off the live document,
        # so they reach the file. Cancelling puts back only what this session
        # itself wrote, which makes it as safe mid-save as abandoning the drag.
        moved = doc.cancel_layer_move()
        if tab.playing:
            inker_mode.stop_play(tab)
        elif not tab.saving and not moved:
            # Only when the move did not already answer the keypress: Esc means
            # "drop the one thing I am doing", not "unwind everything at once".
            if doc.floating is not None:
                doc.cancel_floating()
            elif doc.mask is not None:
                doc.deselect()
        # Always: abandoning a half-finished drag is safe mid-save, because it
        # touches the pane's own state and never the document.
        state.clear_drag()
    return True


# Ctrl-shortcuts that change the document. A save encodes the *live* document
# on a task thread; that is safe only for a stroke landing mid-write, because
# pixels are written in place. Everything here restructures the layer stack or
# moves the history head the save captured, so it waits for the save the same
# way a brush stroke on the canvas already does.
# ``e`` joins them because plain Ctrl+E now writes the document into the
# library: it flattens the layer stack, which is the same read a save makes and
# is just as wrong to take while one is in flight.
# ``j`` joins them for the ordinary reason: layer-from-selection adds a layer
# (a track, on an animated document) and may cut pixels out of another.
_MUTATING_CTRL = frozenset({"z", "y", "a", "d", "x", "v", "i", "t", "e", "j"})


#: The keys whose chord label is not simply the character on them.
#:
#: Written out rather than derived from ``pygame.key.name``: the names it
#: returns are lowercase and platform-shaped ("return", "left", "escape"), and
#: the label is what a menu row and the shortcut sheet *print*, so the two have
#: to be the same string. ``Op.key`` is both the binding and the label for
#: exactly this reason.
_CHORD_NAMES: dict[str, str] = {
    "return": "Enter",
    "enter": "Enter",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "tab": "Tab",
    "escape": "Esc",
    "delete": "Delete",
    "backspace": "Backspace",
    "space": "Space",
    "[": "[",
    "]": "]",
    ",": ",",
    ".": ".",
    "home": "Home",
    "end": "End",
    **{f"f{index}": f"F{index}" for index in range(1, 13)},
}


def chord_of(event: Any, *, ctrl: bool, shift: bool, alt: bool = False) -> str:
    """The chord this key press *is*, in ``Op.key``'s spelling, or ``""``.

    One spelling, used by the binding and by the printed label, so a menu row
    can never advertise a chord the keyboard does not answer -- which is the
    whole reason ``Op.key`` is one field rather than two.
    """
    import pygame

    name = pygame.key.name(event.key).lower()
    label = _CHORD_NAMES.get(name)
    if label is None and len(name) == 1:
        label = name.upper()
    if label is None:
        return ""
    parts = []
    if ctrl:
        parts.append("Ctrl")
    if alt:
        parts.append("Alt")
    if shift:
        parts.append("Shift")
    parts.append(label)
    return "+".join(parts)


def _modal(
    ctx: Any, state: InkerState, tab: InkerDoc, context: str, name: str, event, *, ctrl: bool
) -> bool:
    """Enter and Escape, answered by the context they are pressed in.

    -> whether the key was consumed *and* the rest of ``handle_key`` skipped.

    Only the three modal contexts appear here; every other context falls
    through to the ordinary bindings, which is what "modal" means. A
    transformation swallows every key, not just these two: nothing may change
    the tool out from under a half-finished one.
    """
    import pygame

    enter = event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
    escape = event.key == pygame.K_ESCAPE
    if context == "Transformation":
        if enter:
            inker_mode.end_transform(ctx, commit=True)
        elif escape or (ctrl and name == "z"):
            # Ctrl+Z during a transform means "undo the transform", which is
            # cancelling it -- not stepping back through the history behind it.
            inker_mode.end_transform(ctx, commit=False)
        return True
    if context == "Gesture":
        # An open multi-click gesture answers Enter and Escape before anything
        # else does, and consumes them: Enter would otherwise start playback on
        # an animated document, and Escape would drop the *previous* selection
        # while leaving the half-drawn polygon on screen. Ahead of the tool
        # letters as well, so neither can be reached with a gesture open.
        if enter:
            inker_mode.commit_gesture(state, tab)
            return True
        if escape:
            state.clear_gesture()
            state.clear_drag()
            return True
    return False


def _ctrl_key(
    ctx: Any, state: InkerState, tab: InkerDoc, doc: Any, name: str, event, *, shift: bool
):
    import pygame

    # ``busy``, not ``saving``: playback is the second reason the document may
    # not be restructured, and it is the same list of keys for the same reason.
    if tab.busy and name in _MUTATING_CTRL:
        return True

    if name == "z" and shift:
        # Ctrl+Shift+Z is redo's second spelling, which the registry does not
        # carry: an op has one key, and Ctrl+Y is the one the menu prints.
        doc.redo()
    elif event.key == pygame.K_TAB:
        state.cycle(-1 if shift else 1)
    # Ctrl+Shift+E, Ctrl+Shift+D and Ctrl+Shift+J used to have branches here.
    # All three are ops (``export_png``, ``reselect``, ``move_to_layer``), and
    # the registry is consulted before this function is reached -- so the
    # branches had been unreachable since those ops gained their keys, and a
    # reader would have had to check the registry to know it.
    #
    # Ctrl+4 and Ctrl+5 went the same way. Their *shifted* halves were the
    # awkward part: Ctrl+Shift+4 rotated the other way and Ctrl+Shift+5 was a
    # silent alias for Ctrl+5, neither advertised by any ``Op.key`` -- which is
    # the one thing that field exists to prevent. The reverse rotation is now
    # ``rotate_view_back``, printed on its menu row like every other binding,
    # and the alias is gone.
    return True


def release_all(ctx: Any) -> None:
    from .panes import inker_textures

    inker_textures.release_all(ctx)

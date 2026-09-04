"""Sirens' keyboard: one dispatch, five alphabets.

Split out of :mod:`.sirens_mode` on 2026-09-04 (T7 of the 2026-09-02 review).

**Which column the caret is in decides what a key means**, and that decision is
made once, here, rather than inside each writer -- the columns overlap on the
keyboard and nowhere else, so ``c`` is a note in the first column, the hex digit
twelve in the third and fifth, and nothing at all in the fourth. Four guards
that had to agree with each other is what one dispatch on ``state.column``
replaces.

Every verb it reaches lives in :mod:`.sirens_edit` or :mod:`.sirens_play` and is
called through ``sirens_mode``, which forwards; see the ``_MOVED`` table at the
foot of that file.
"""

from __future__ import annotations

from typing import Any

from . import sirens_audio, sirens_mode, sirens_state
from .sirens_state import SirensState, SongTab, ensure  # noqa: F401


def release_all(ctx: Any) -> None:
    """Nothing to release but the device. No textures: the grid is imgui
    primitives, which is why this mode registers none with the backend."""
    sirens_audio.stop()


#: The piano rows, as pygame key names -> semitone. Two rows, the tracker
#: layout every FamiTracker and DefleMask user already has in their hands:
#: ``zsxdcvgbhnjm`` is the lower octave and ``q2w3er5t6y7u`` the one above. Not
#: derived from a keyboard layout, deliberately -- a physical-position mapping
#: would be right on AZERTY and wrong for the user who learned the letters.
PIANO_KEYS: dict[str, int] = {
    "z": 0, "s": 1, "x": 2, "d": 3, "c": 4, "v": 5, "g": 6, "b": 7, "h": 8,
    "n": 9, "j": 10, "m": 11, ",": 12,
    "q": 12, "2": 13, "w": 14, "3": 15, "e": 16, "r": 17, "5": 18, "t": 19,
    "6": 20, "y": 21, "7": 22, "u": 23,
}

# The Ctrl chords a busy tab refuses. Copy is not among them: it reads the
# document and pushes nothing, so a tab mid-save can still be copied from.
_MUTATING_CTRL = frozenset({"z", "y", "x", "v", "g"})


def handle_key(ctx: Any, event: Any) -> bool:
    """Sirens' keyboard. Returns whether the key was consumed; the app returns
    afterwards either way, as it does for every workspace mode."""
    import pygame

    from .sirens import document as D

    if event.type != pygame.KEYDOWN:
        return False
    state = ensure(ctx)
    tab = state.active
    # Off ``event.mod``, never ``pygame.key.get_mods()`` -- ``main._shortcut``'s
    # rule (UX-12): ``mod`` is the state when this key was pressed, and
    # ``get_mods()`` is the state now, after the event batch drained.
    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key).lower()

    if ctrl:
        if tab is not None and tab.busy and name in _MUTATING_CTRL:
            return True
        return _ctrl_key(ctx, state, tab, name, shift=shift)

    if tab is None:
        return False

    if event.key == pygame.K_SPACE:
        sirens_mode.toggle_play(ctx, tab)
        return True
    if event.key in (pygame.K_UP, pygame.K_DOWN):
        sirens_mode.move_caret(ctx, drow=1 if event.key == pygame.K_DOWN else -1, select=shift)
        return True
    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        step = 1 if event.key == pygame.K_RIGHT else -1
        # Shift moves by *channel* rather than by column, because a selection
        # is a rectangle over channels and rows -- the column a block starts in
        # is not something a tracker selection has ever carried.
        if shift:
            sirens_mode.move_caret(ctx, dchannel=step, select=True)
        else:
            sirens_mode.move_caret(ctx, dcolumn=step)
        return True
    if event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
        # A beat is four rows (``document.ROWS_PER_BEAT``); a page is four
        # beats, which is one bar in every time signature this idiom uses.
        sirens_mode.move_caret(
            ctx, drow=16 if event.key == pygame.K_PAGEDOWN else -16, select=shift
        )
        return True
    if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
        # Shifted, these pull the rows below up over this one -- the tracker's
        # "take this row out", and Insert's inverse. Unshifted they blank in
        # place, which is a different verb and the more common one.
        if shift:
            sirens_mode.shift_rows(ctx, -1)
        else:
            sirens_mode.clear_cell(ctx)
        return True
    if event.key == pygame.K_INSERT:
        sirens_mode.shift_rows(ctx, 1)
        return True
    if event.key in (pygame.K_HOME, pygame.K_END):
        pattern = sirens_mode.caret_pattern(ctx)
        rows = 1 if pattern is None else pattern.rows
        sirens_mode.jump_row(ctx, 0 if event.key == pygame.K_HOME else rows - 1, select=shift)
        return True
    if event.key == pygame.K_ESCAPE:
        # Staged, and consumed only when it had something to drop -- Inker's and
        # Plotter's rule. This cleared the anchor and returned True either way,
        # so Esc with nothing selected was swallowed here rather than reaching
        # whatever the app would otherwise do with it (closing an overlay, say),
        # and a user pressing it twice got no answer to the second press.
        if state.anchor is None:
            return False
        state.anchor = None
        return True
    if name in ("=", "+", "-"):
        state.octave = max(
            sirens_state.MIN_OCTAVE,
            min(sirens_state.MAX_OCTAVE, state.octave + (-1 if name == "-" else 1)),
        )
        return True
    if event.key == pygame.K_1 and shift:
        # Shift+1/2 transpose the block, the FamiTracker chord. Plain digits
        # are the upper piano row, so the shifted pair is what is left.
        sirens_mode.transpose(ctx, -1)
        return True
    if event.key == pygame.K_2 and shift:
        sirens_mode.transpose(ctx, 1)
        return True
    # The five columns take five different alphabets, and which one is live is
    # decided *here* rather than inside each writer, because the columns overlap
    # on the keyboard and nowhere else: ``c`` is a note in the first column, the
    # hex digit twelve in the third and fifth, and the halt effect in the
    # fourth. One dispatch on ``state.column`` is what keeps that from being
    # four guards that have to agree with each other.
    if state.column == D.NOTE:
        # ``e`` in the effect column is the letter of an effect, not an
        # E-natural, and a piano row that fired everywhere would make four of
        # the five columns untypable. That guard is unchanged; what changed is
        # that the other four columns now have alphabets of their own below,
        # rather than nothing at all.
        if name in PIANO_KEYS:
            sirens_mode.write_note(ctx, PIANO_KEYS[name])
            return True
        if event.key == pygame.K_BACKQUOTE:
            from .sirens import notes

            # Backtick cuts (``===``), Shift+backtick releases (``~~~``). They
            # are one physical key because they are one gesture with two
            # endings, and the shifted half is the character the cell itself
            # draws -- the tilde is on that key on the layout this piano row is
            # already spelled for. FamiTracker puts release beside cut for the
            # same reason; it does not agree with this repo about *which* key
            # cut is on, and moving cut now would retrain the one binding
            # somebody may already have in their hands.
            sirens_mode.write_cell(
                ctx, notes.NOTE_RELEASE if shift else notes.NOTE_OFF, column=D.NOTE
            )
            return True
        return False
    if state.column == D.EFFECT:
        return sirens_mode.write_effect(ctx, name)
    if len(name) == 1 and name in "0123456789abcdef":
        return sirens_mode.write_hex(ctx, int(name, 16))
    return False


def _ctrl_key(
    ctx: Any, state: SirensState, tab: SongTab | None, name: str, *, shift: bool
) -> bool:
    if name == "n":
        sirens_mode.new_document(ctx)
        return True
    if name == "o":
        sirens_mode.ask_open(ctx)
        return True
    if tab is None:
        return False
    if name == "w":
        sirens_mode.close_tab(ctx, tab.uid)
        return True
    if name == "s":
        sirens_mode.save_as(ctx, tab) if shift else sirens_mode.save(ctx, tab)
        return True
    if name == "z":
        # Ctrl+Shift+Z redoes as well, which is what Inker, Clay, Plotter and
        # Packwright accept and what a user arriving from any of them already
        # has in their hand. Ctrl+Y keeps working: this adds a spelling rather
        # than replacing one.
        sirens_mode.redo(ctx, tab) if shift else sirens_mode.undo(ctx, tab)
        return True
    if name == "y":
        sirens_mode.redo(ctx, tab)
        return True
    if name == "c":
        sirens_mode.copy_selection(ctx)
        return True
    if name == "x":
        sirens_mode.cut_selection(ctx)
        return True
    if name == "v":
        sirens_mode.paste(ctx)
        return True
    if name == "tab":
        state.cycle(-1 if shift else 1)
        return True
    if name == "g":
        # FamiTracker's chord for it, and the letter is free here.
        sirens_mode.interpolate_selection(ctx)
        return True
    if name in ("up", "down"):
        # The instrument the next note is stamped with, from the keyboard.
        # Ctrl and the vertical pair because the list *is* vertical and the
        # unmodified arrows are the caret's -- see :func:`cycle_instrument`.
        sirens_mode.cycle_instrument(ctx, -1 if name == "up" else 1)
        return True
    return False

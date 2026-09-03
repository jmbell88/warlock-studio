"""Sirens' controller: documents, the caret, rendering, playback and keys.

The layer that knows about task threads and a sound device; the engine under
``sirens/`` knows about neither, and that separation is the whole reason a
machine with no card can still use this mode. The panes draw, this decides.

**Rendering is a task, and re-arming it is a flag.** A three-minute song is a
tick loop over every channel plus a 4x-oversampled decimation per voice, which
is seconds of numpy -- not frame-thread work -- so a render goes through
``ctx.submit`` under ``sirens-render:<uid>`` and the result is adopted in
:func:`on_task_done`. What re-arms it is ``SongTab.render_dirty``, pumped from
the grid pane's draw and cleared *only when a submit is accepted*: this is
``PackTab.pack_dirty`` verbatim, including its rule that the flag is cleared at
the submit and never at the adoption. ``TaskRunner.submit`` refuses a key
already in flight and nothing re-arms it, so a burst of keystrokes -- which is
what typing a bar *is* -- would otherwise render the song as it stood at the
first one and drop every note after it.

**Rendering reads a snapshot, not the document.** ``wsng.wsng_bytes`` is the
snapshot: the task re-reads a document from those bytes and renders *that*, so
the frame thread may keep editing the live one while the task runs. It costs a
zip round-trip per render and it buys the only version of this that cannot tear
-- a numpy view handed to a thread is a view of an array the caret is writing
into.

**Playback never blocks and never fails loudly.** ``sirens_audio`` answers
rather than raises, so :func:`play` on a machine with no device toasts once and
leaves the document exactly as it was.

Every task key carries the ``sirens-`` prefix, because the app claims results
by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import dialogs, docmodes, journal, recents, sirens_audio, sirens_io, sirens_state

# ``ensure`` and ``active`` live in :mod:`.sirens_state` -- they touch nothing
# but ``ctx.state.sirens`` -- and the file layer lives in :mod:`.sirens_io`.
# Both are re-exported here, as **plain imports rather than wrappers**, because
# every pane, every key binding and every test says ``sirens_mode.save(ctx)``:
# a wrapper would be a second object where the callers reach for one.
from .sirens_io import (  # noqa: F401
    EXPORT_PREFIX,
    SAMPLE_FILTER,
    SAMPLE_PREFIX,
    SFX_DIR,
    SONG_FILTER,
    SONG_NAME,
    STEM_DIR,
    WAV_FILTER,
    _load,
    _start,
    ask_open,
    ask_sample,
    export_files,
    export_plan,
    export_to,
    free_sample_key,
    import_sample,
    open_path,
    save,
    save_as,
    save_to,
)
from .sirens_state import (  # noqa: F401
    COLUMN_DIGITS,
    SirensState,
    SongTab,
    active,
    ensure,
)
from .state import set_mode

log = logging.getLogger(__name__)


def remember_path(ctx: Any, path: Any) -> None:
    """Put ``path`` at the front of the merged recent list -- :mod:`.recents`,
    which is the one list Home's Resume rows are built from."""
    recents.remember(ctx.settings, "sirens", path)


def forget_path(ctx: Any, path: Any) -> None:
    """Drop a path that turned out not to open -- :mod:`.recents`' own rule,
    named here so a caller does not have to know this mode's kind string."""
    recents.forget(ctx.settings, "sirens", path)


def recent_paths(ctx: Any) -> list[str]:
    """This mode's recent files, newest first. What its own panel draws."""
    return recents.paths(ctx.settings, "sirens")


def persist(ctx: Any) -> None:
    """Nothing to write: the recent list is :mod:`.recents`, which persists
    itself on every write. Kept as a no-op for ``packwright_mode.persist``'s
    reason -- it is called after every open and save, and turning each of those
    into "call this only if the mode still has settings" is how one of them
    comes to skip a write that mattered later."""


# --- documents ----------------------------------------------------------------


def adopt(ctx: Any, doc: Any, *, path: Path | None = None, title: str | None = None) -> SongTab:
    state = ensure(ctx)
    tab = SongTab(
        doc=doc,
        title=title or sirens_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> SongTab:
    """A song that plays the moment somebody types into it.

    ``document.new_song`` rather than a bare ``SongDoc()``: an empty order list
    is a document where Space does nothing and there is no way to find out why.
    """
    from .sirens import document

    return adopt(ctx, document.new_song(), title="Untitled")


# --- the caret ----------------------------------------------------------------


def caret_pattern(ctx: Any, tab: SongTab | None = None) -> Any:
    """The pattern the caret is in, or ``None``.

    Every caller wants the object rather than the uid, and reaching for it
    through ``doc.pattern(state.pattern)`` at six sites is six places to forget
    that the uid can be stale -- a pattern the user has just deleted is the
    ordinary way that happens.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or state.pattern is None:
        return None
    return tab.doc.pattern(state.pattern)


def caret_pattern_label(ctx: Any, tab: SongTab | None = None) -> str:
    """What the grid is editing, in words. Empty when it is editing nothing.

    **Adding a sound effect silently repoints the grid.** ``add_oneshot`` mints
    the effect *and* a pattern of its own and moves the caret onto it, which is
    the right behaviour -- the grid is the effect editor and there is no second
    one -- but until this readout existed the only thing on screen that said so
    was a line of muted text in a panel in the right-hand column, and a reader
    who had scrolled it away had no way to tell a song pattern from an effect.
    Which one is loaded decides what every keystroke is editing, so it belongs
    above the grid rather than beside it.

    A reverse lookup, because a one-shot *is* a pattern (``sirens/document``'s
    "One-shots are patterns"): the effect owns a pattern uid, so the question
    "is this pattern an effect" is answered by asking the effects, not the
    pattern. Pure state to string, so it is assertable without a frame.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or state.pattern is None:
        return ""
    pattern = tab.doc.pattern(state.pattern)
    if pattern is None:
        return ""
    for one in tab.doc.oneshots:
        if one.pattern == state.pattern:
            # The effect's name, not the pattern's: the effect is the thing the
            # user named and the thing the exported WAV is called after.
            return f"{one.name or 'effect'} - sound effect"
    return f"pattern {pattern.name}" if pattern.name else "song pattern"


def clamp_caret(ctx: Any, tab: SongTab | None = None) -> None:
    """Put the caret back inside the pattern it names.

    Called after every step that can shrink the grid under it -- a resize, a
    channel removal, an undo of either. Without it the next keystroke writes at
    a row the pattern no longer has, which ``set_cells`` clips to nothing: the
    key does nothing and there is no way to see why.
    """
    from .sirens import document as D

    state = ensure(ctx)
    tab = tab or state.active
    if tab is None:
        return
    doc = tab.doc
    if state.pattern is None or doc.pattern(state.pattern) is None:
        state.pattern = doc.patterns[0].uid if doc.patterns else None
        state.anchor = None
    pattern = None if state.pattern is None else doc.pattern(state.pattern)
    rows = pattern.rows if pattern is not None else 1
    chans = pattern.channels if pattern is not None else 1
    state.row = max(0, min(state.row, rows - 1))
    state.channel = max(0, min(state.channel, chans - 1))
    state.column = max(0, min(state.column, D.COLUMNS - 1))
    # Whatever half-typed byte was in flight belonged to a cell that may not be
    # under the caret any more. Cleared here as well as in :func:`move_caret`
    # because this is the arrival every *other* route takes -- a click through
    # :func:`set_caret`, a pattern resized under the caret, an undo of either.
    state.digit = 0
    if state.instrument is not None and doc.instrument(state.instrument) is None:
        state.instrument = doc.instruments[0].uid if doc.instruments else None
    if state.oneshot is not None and doc.oneshot(state.oneshot) is None:
        # Cleared rather than moved to a neighbour, unlike the instrument above:
        # an instrument is what the *next note* is stamped with and needs some
        # answer, while a selected effect is what the grid is editing -- and
        # silently switching the grid to a different effect after an undo is
        # the caret bug this function exists to prevent, one level up.
        state.oneshot = None


def move_caret(ctx: Any, drow: int = 0, dchannel: int = 0, dcolumn: int = 0,
               *, select: bool = False) -> None:
    """Step the caret, wrapping rows and clamping the rest.

    Rows wrap because a pattern is a loop and every tracker wraps them; columns
    and channels clamp because their ends are the edge of the grid rather than
    the edge of a cycle -- a right arrow at the last channel that jumped to the
    first would move the eye across the whole screen for one keystroke.

    ``select`` is what Shift+arrow passes: it anchors the block at wherever the
    caret was *before* this step, once, so a run of shifted arrows grows one
    rectangle rather than re-anchoring each time.
    """
    from .sirens import document as D

    state = ensure(ctx)
    pattern = caret_pattern(ctx)
    if pattern is None:
        return
    # A step of any size ends the byte being typed. Without this the second
    # nibble of ``4_`` lands in whatever cell the arrow key moved to, which is
    # a value nobody typed in a cell nobody was looking at.
    state.digit = 0
    if select:
        if state.anchor is None:
            state.anchor = (state.row, state.channel)
    else:
        state.anchor = None
    if drow:
        if select:
            # A selection has an edge where a caret has a loop: Shift+Up at
            # row 0 wrapped to the last row and selected the whole pattern.
            state.row = max(0, min(state.row + drow, pattern.rows - 1))
        else:
            state.row = (state.row + drow) % pattern.rows
    if dchannel:
        state.channel = max(0, min(state.channel + dchannel, pattern.channels - 1))
    if dcolumn:
        state.column = max(0, min(state.column + dcolumn, D.COLUMNS - 1))


def set_caret(ctx: Any, *, pattern: int | None = None, row: int | None = None,
              channel: int | None = None, column: int | None = None) -> None:
    """Put the caret somewhere absolute -- what a click and the order list do.

    Through here rather than by assignment so the clamp is not something four
    call sites remember; the selection is dropped for the same reason a tab
    switch drops it, because a rectangle with one corner in a pattern the user
    has left is not a rectangle.
    """
    state = ensure(ctx)
    if pattern is not None and pattern != state.pattern:
        state.pattern = pattern
        state.anchor = None
    if row is not None:
        state.row = int(row)
    if channel is not None:
        state.channel = int(channel)
    if column is not None:
        state.column = int(column)
    clamp_caret(ctx)


# --- editing ------------------------------------------------------------------


def _touch(tab: SongTab, changed: bool) -> bool:
    """Arm the renderer if something moved. -> what it was told.

    One line, and it exists so no mutator can forget it: an edit that does not
    set ``render_dirty`` is an edit you cannot hear, and that is indisting-
    uishable from an edit that did not happen.
    """
    if changed:
        tab.render_dirty = True
    return changed


def write_cell(
    ctx: Any, value: int, column: int | None = None, *, advance: bool = True
) -> bool:
    """Put one value in the cell under the caret, and step by the edit step.

    **The one door every keystroke that changes a cell goes through**, whichever
    column it lands in: a note, a hex nibble, an effect letter and a blanking
    Delete all arrive here, because every one of them owes the same three
    things -- the document's refusal framed as a toast rather than a traceback,
    the renderer re-armed through :func:`_touch`, and the caret stepped. A
    second path grown for the numeric columns would be a second place to forget
    one of the three, and an edit that forgets ``_touch`` is an edit you cannot
    hear, which is indistinguishable from an edit that did not happen.

    ``advance`` is what a half-finished entry passes. The high nibble of a
    two-digit column writes a real value into the cell -- it is not a keystroke
    held in a buffer somewhere -- and must nevertheless *not* drop the caret by
    the edit step, or the low nibble would land a row further down.
    """
    return _write_at_caret(ctx, column, [int(value)], advance=advance)


def _write_at_caret(
    ctx: Any, column: int | None, values: list[int], *, advance: bool
) -> bool:
    """:func:`write_cell`'s body over a run of adjacent columns.

    A note and the instrument stamped beside it are one keystroke and have to
    be one undo step -- as two ``set_cell`` calls, Ctrl+Z after a note removed
    the instrument and left the note. One ``set_cells`` over a ``1×1×n`` block
    is one step; ``write_cell`` is the ``n == 1`` case.
    """
    import numpy as np

    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    column = state.column if column is None else int(column)
    block = np.array(values, dtype=np.int16).reshape(1, 1, len(values))
    try:
        changed = tab.doc.set_cells(state.pattern, state.row, state.channel, column, block)
    except ValueError as exc:
        ctx.toast(f"That note was not written: {exc}", "error")
        return False
    _touch(tab, changed)
    if advance:
        # The entry is finished, so the next hex key starts a fresh byte.
        # Cleared here as well as in :func:`move_caret` because a step of zero
        # leaves the caret exactly where it is and never reaches that one.
        state.digit = 0
        if state.step:
            move_caret(ctx, drow=state.step)
    # Deliberately True even for a no-op write: retyping the note that is
    # already there is a real user action and the caret still steps, so the key
    # was consumed. ``SongDoc``'s False means "no history step", not "ignored".
    return True


def write_note(ctx: Any, semitone: int) -> bool:
    """A note from a piano-row key, in the caret's octave, with its instrument.

    The instrument column is stamped alongside, which is the behaviour every
    tracker has and the absence of which is the single most common "why is it
    silent" report: a note with no instrument plays nothing, and the user who
    typed it has no reason to suspect a second column they never touched.
    """
    from .sirens import notes

    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    value = state.octave * 12 + int(semitone)
    if not notes.is_note(value):
        return False
    from .sirens import document as D

    values = [value]
    if state.instrument is not None:
        # The note and its instrument as one block, so one Ctrl+Z takes back
        # the whole keystroke -- see ``_write_at_caret``.
        values.append(int(state.instrument))
    return _write_at_caret(ctx, D.NOTE, values, advance=True)


def _column_ceiling(column: int) -> int:
    """The largest value ``column`` may hold, read off the engine not copied.

    The instrument column is the one that matters. Ids are minted out of a
    per-document space bounded by ``document.MAX_INSTRUMENTS``, so ``80`` and
    upwards name a slot no song can contain: a cell that renders as silence and
    reads, to the person who typed it, as the synthesiser being broken. The
    volume column is the engine's own ``0..15``, and the parameter column is a
    byte because that is exactly what every effect's ``xx`` is.
    """
    from .sirens import document as D
    from .sirens import instruments as inst

    if column == D.INSTRUMENT:
        return D.MAX_INSTRUMENTS - 1
    if column == D.VOLUME:
        return inst.MAX_VOLUME
    return 0xFF


def write_hex(ctx: Any, value: int) -> bool:
    """Type one hex digit into the caret's column. -> whether it was taken.

    Tracker entry, which is nibble-at-a-time and **in place**: the digit
    replaces one nibble of whatever the cell already holds rather than starting
    a fresh byte, so correcting the low half of ``4F`` is one keystroke on the
    second character instead of retyping both. An empty cell counts as ``00``
    for that purpose, which is what makes the first digit of a fresh entry land
    in the high nibble and read back the way it was typed.

    A digit that would take the cell past :func:`_column_ceiling` writes
    **nothing at all** -- ``8`` in the high nibble of the instrument column
    names a slot no song has -- for the same reason an unknown effect letter
    writes nothing: a value the engine cannot use is worse in the cell than
    absent, because the grid then shows the user something the song does not
    play.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    column = state.column
    width = COLUMN_DIGITS[column]
    if width == 0:
        return False
    pattern = caret_pattern(ctx, tab)
    if pattern is None:
        return False
    # An empty cell is ``00`` for the purpose of a nibble write; anything else
    # would make the first digit of a fresh entry depend on ``EMPTY``'s value.
    current = int(pattern.cells[state.row, state.channel, column])
    current = 0 if current < 0 else current & 0xFF
    if width == 1:
        wanted = int(value)
    elif state.digit == 0:
        wanted = (int(value) << 4) | (current & 0x0F)
    else:
        wanted = (current & 0xF0) | int(value)
    if wanted > _column_ceiling(column):
        return False
    last = state.digit >= width - 1
    history = tab.doc.history
    if last and width > 1 and state.digit_head == history.head and history.can_undo:
        # The high nibble already went in as its own step (it is drawn in the
        # grid while the low one is awaited). Two steps for one byte meant
        # Ctrl+Z took back half a number; the first step is withdrawn here,
        # without a redo, and the whole byte lands as one.
        history.undo(tab.doc, redoable=False)
        tab.render_dirty = True
    before = history.head
    written = write_cell(ctx, wanted, column=column, advance=last)
    if written and not last:
        state.digit = 1
        # Remembered only when the nibble pushed a step: retyping the digit
        # already there pushes nothing, and withdrawing "the top step" would
        # then withdraw somebody else's.
        state.digit_head = history.head if history.head != before else -1
    return written


def write_effect(ctx: Any, letter: str) -> bool:
    """Type an effect letter into the effect column. -> whether it was taken.

    ``synth.EFFECT_NAMES`` is the authority and is **read rather than copied**,
    so a tenth effect added to the engine becomes typable by existing and a
    letter that is not in the table writes nothing. The refusal is the point: an
    effect id the tick loop has no handler for draws as ``?`` and plays as
    silence, which the person who typed it cannot tell from a bug in the
    synthesiser.
    """
    from .sirens import document as D
    from .sirens import synth

    state = ensure(ctx)
    if state.column != D.EFFECT:
        return False
    for effect, (name, _description) in synth.EFFECT_NAMES.items():
        if name.lower() == letter.lower():
            return write_cell(ctx, effect, column=D.EFFECT)
    return False


def clear_cell(ctx: Any) -> bool:
    """Blank the block, or -- with no block -- the one column under the caret.

    Delete narrows to a column when nothing is selected, which is the tracker
    convention and is the half that makes the other four columns *editable*
    rather than merely writable: a wrong instrument number is taken back by
    clearing two characters, not by clearing the note beside them as well. A
    block is still blanked across every column, because a selection is a
    rectangle over rows and channels and has never had a column axis to narrow
    along.
    """
    from .sirens import notes

    state = ensure(ctx)
    if state.anchor is not None:
        return clear_selection(ctx)
    return write_cell(ctx, notes.EMPTY)


def clear_selection(ctx: Any) -> bool:
    """Blank the block, or the cell under the caret when there is no block."""
    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    state.digit = 0  # the edit moved on; a half-typed nibble must not land in it
    block = state.selection() or (state.row, state.channel, 1, 1)
    return _touch(tab, tab.doc.clear_cells(state.pattern, *block))


def transpose(ctx: Any, by: int) -> bool:
    """Shift the notes in the block (or the caret's cell) by semitones."""
    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    state.digit = 0  # the edit moved on; a half-typed nibble must not land in it
    block = state.selection() or (state.row, state.channel, 1, 1)
    return _touch(tab, tab.doc.transpose(state.pattern, *block, by))


# --- the block clipboard ------------------------------------------------------
#
# The last thing the Experimental chip named, closed 2026-09-02. Three verbs
# and no new document API: a copy is a read of ``pattern.cells``, a paste is
# ``SongDoc.set_cells`` -- the one door every pattern write already goes through,
# which clips at the pattern edge and pushes one ``CellsEdit`` (``edits.py``
# says the class exists for exactly this) -- and a cut is a copy followed by
# ``clear_cells``, which is one history step because ``clear_cells`` is one
# ``set_cells``. The clipboard is read-only from the document's point of view:
# ``copy_selection`` takes a private copy, so a later edit to the source block
# cannot change what a paste puts down.


def copy_selection(ctx: Any) -> bool:
    """Copy the block (or the caret's cell) to the app-level clipboard.

    Not an edit -- the document is untouched and no history is pushed -- so
    it is allowed on a busy tab, which is why it is not in ``_MUTATING_CTRL``.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None or state.pattern is None:
        return False
    pattern = tab.doc.pattern(state.pattern)
    if pattern is None:
        return False
    row, chan, rows, chans = state.selection() or (state.row, state.channel, 1, 1)
    block = pattern.cells[row : row + rows, chan : chan + chans, :]
    if block.size == 0:
        return False
    state.clip = np.ascontiguousarray(block, dtype=np.int16).copy()
    return True


def cut_selection(ctx: Any) -> bool:
    """Copy, then blank -- one history step, because the blanking is one
    ``set_cells``. -> whether anything was blanked; the copy happens either
    way, so cutting an already-empty block still fills the clipboard."""
    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    state.digit = 0  # the edit moved on; a half-typed nibble must not land in it
    if not copy_selection(ctx):
        return False
    row, chan, rows, chans = state.selection() or (state.row, state.channel, 1, 1)
    return _touch(tab, tab.doc.clear_cells(state.pattern, row, chan, rows, chans))


def paste(ctx: Any) -> bool:
    """Put the clipboard down with its top-left corner at the caret.

    A block that runs off the pattern's edge is clipped, not refused
    (``SongDoc.set_cells``). Nothing on the clipboard is a no-op that pushes
    no history, the way retyping the note already there is.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None or state.clip is None:
        return False
    state.digit = 0  # the edit moved on; a half-typed nibble must not land in it
    return _touch(tab, tab.doc.set_cells(state.pattern, state.row, state.channel, 0, state.clip))


# --- instruments --------------------------------------------------------------

#: The four sequences an instrument is made of, in the order the editor stacks
#: them. Named here rather than in the pane because the pane is not the only
#: caller: a test asserting "a drag is one step" walks them, and a fifth
#: sequence added to the engine has to reach both through one list.
ENVELOPE_FIELDS: tuple[str, ...] = ("volume", "arpeggio", "pitch", "duty")


def set_sequence(ctx: Any, tab: SongTab | None, uid: int, field: str, sequence: Any) -> bool:
    """Put one sequence on one instrument. -> whether the document moved.

    Through ``update_instrument`` rather than by assignment, which is what makes
    it one reversible ``InstrumentEdit`` and what refuses a no-op -- a pointer
    held still over a bar it has already painted is a stream of frames, and
    every one of them would otherwise be an undo step.
    """
    if tab is None or tab.busy or field not in ENVELOPE_FIELDS:
        return False
    try:
        changed = tab.doc.update_instrument(uid, **{field: sequence})
    except ValueError as exc:
        ctx.toast(f"That envelope was not changed: {exc}", "error")
        return False
    return _touch(tab, changed)


def begin_envelope_drag(ctx: Any, tab: SongTab | None, field: str, grip: str) -> None:
    """Open a gesture, recording where the history stood when it started.

    The depth is the whole mechanism: painting a decay curve is one edit per
    column the pointer crosses, and forty of them in the stack means forty
    Ctrl+Z presses to undo one drag. ``document.add_oneshot`` records a depth
    for the same reason and folds at the same place.
    """
    state = ensure(ctx)
    if tab is None or tab.busy:
        return
    state.env_field, state.env_grip = field, grip
    state.env_depth = tab.doc.history.mark()
    state.env_step = -1


def end_envelope_drag(ctx: Any, tab: SongTab | None) -> bool:
    """Close the gesture and fold its run into one step. -> whether it folded.

    False for a drag that changed nothing (``update_instrument`` refused every
    frame of it) and for one that changed exactly one thing -- see
    ``UndoStack.collapse_since``, which leaves a lone step reading as what it
    did rather than as a compound of one.
    """
    state = ensure(ctx)
    depth = state.env_depth
    state.env_field, state.env_grip, state.env_depth, state.env_step = "", "", -1, -1
    if tab is None or depth < 0:
        return False
    return tab.doc.history.collapse_since(depth)


def adopt_sample(ctx: Any, tab: SongTab, result: dict[str, Any]) -> str:
    """Land a decoded ``.wav`` in the document. -> the key it went under.

    On the frame thread, and that is the point: which key is free depends on the
    sample table, ``doc.set_sample`` pushes an undo step, and both of those are
    the document's -- so the task decodes bytes and this decides what to do with
    them.

    **The sample and the instrument that asked for it are one step.** A picker
    opened from an instrument's ``sample`` field is one action by the user, and
    an undo that took the assignment back while leaving an orphan sample in the
    table would be a second press to finish a single mistake --
    ``document.add_oneshot``'s rule, at the same depth-and-collapse.
    """
    doc = tab.doc
    key = free_sample_key(doc, str(result.get("name", "")))
    depth = doc.history.mark()
    try:
        doc.set_sample(key, result["pcm"])
    except ValueError as exc:
        ctx.toast(f"That sample was not added: {exc}", "error")
        doc.history.collapse_since(depth)
        return ""
    instrument = result.get("instrument")
    if instrument is not None and doc.instrument(instrument) is not None:
        doc.update_instrument(instrument, sample=key)
    doc.history.collapse_since(depth)
    tab.render_dirty = True
    ctx.toast(f"Added the sample {key}.")
    return key


def remove_sample(ctx: Any, tab: SongTab, key: str) -> bool:
    """Drop one entry of the sample table. -> whether anything went.

    The instruments that named it are **left alone**, which is
    ``remove_instrument``'s rule one level down and for its reason: a sample
    instrument pointing at a key nothing answers to is silent and can be put
    back by an undo, while rewriting every instrument that used it cannot.
    """
    if tab is None or tab.busy:
        return False
    try:
        return _touch(tab, tab.doc.set_sample(key, None))
    except ValueError as exc:
        ctx.toast(f"That sample was not removed: {exc}", "error")
        return False


def undo(ctx: Any, tab: Any) -> None:
    """One step back, whichever surface asked for it.

    The clamp and the re-render belong to *undoing* rather than to the
    keyboard, which is why this exists at all: the bridge pane draws the same
    Undo button and must carry the same side effects.
    """
    tab.doc.undo()
    tab.render_dirty = True
    clamp_caret(ctx, tab)


def redo(ctx: Any, tab: Any) -> None:
    """One step forward. :func:`undo`'s twin, and its reasoning."""
    tab.doc.redo()
    tab.render_dirty = True
    clamp_caret(ctx, tab)


# --- rendering ----------------------------------------------------------------


def request_rerender(ctx: Any, tab: SongTab | None = None) -> None:
    """Mark the audio stale so the next pump renders it.

    Does not render here: rendering is a worker job the grid pane's pump owns,
    and a pane that started one would be doing a full synthesis pass on the
    frame thread.
    """
    tab = tab or active(ctx)
    if tab is None:
        return
    tab.render_dirty = True


def request_render(ctx: Any, tab: SongTab | None = None) -> None:
    """Ask for a render. Safe to call every frame -- that is the point."""
    tab = tab or active(ctx)
    if tab is None or not tab.render_dirty:
        return
    if not tab.doc.order:
        # Nothing to play, and nothing to render. Cleared rather than left
        # armed, so an empty song does not resubmit a no-op every frame.
        tab.pcm, tab.loop, tab.render_dirty, tab.render_error = None, None, False, ""
        return

    uid = tab.uid
    # Asked *before* the snapshot, because the snapshot is the expensive half:
    # ``wsng_bytes`` DEFLATEs every pattern and encodes every sample, on the
    # frame thread. ``submit`` refuses a key already in flight and
    # ``render_dirty`` deliberately stays armed when it does, so a song being
    # rendered was re-serialised in full on every frame until the render landed
    # -- the work thrown away, once per frame, for as long as the render took.
    if ctx.busy(f"sirens-render:{uid}"):
        return

    from .sirens import wsng

    # The snapshot; see the module docstring. Taken on the frame thread, which
    # is where the document is safe to read.
    data = wsng.wsng_bytes(tab.doc)

    def run() -> dict[str, Any]:
        from ..service.errors import invalid_from
        from .sirens import synth, wavout

        try:
            doc = wsng.read_wsng(data)
            samples, loop = synth.render(doc)
        except ValueError as exc:
            # Framed, because only a ``ServiceError``'s text survives the task
            # classifier -- and the engine's own sentence (a song past the
            # render ceiling says so, with the ceiling in it) is the half that
            # tells the user what to do about it.
            raise invalid_from(exc, "That song did not render") from exc
        return {"pcm": wavout.to_int16(samples), "loop": loop, "uid": uid}

    tab.rendering = True
    if ctx.submit(f"sirens-render:{uid}", run):
        # Cleared *only* on an accepted submit. The runner refuses a key
        # already in flight, and clearing regardless would drop the note that
        # arrived while the previous render was running.
        tab.render_dirty = False
    else:
        tab.rendering = False


def pump(ctx: Any) -> None:
    """Called from the grid pane's draw, which is the only thing that runs
    every frame in this mode -- the ``motion.py`` idiom."""
    request_render(ctx)


#: The key prefix an audition carries. **Not ``sirens-render:``**, and that is
#: the decision in this pair of functions: ``TaskRunner.submit`` refuses a key
#: already in flight, so sharing the song's key would make a press of Audition
#: during a re-render do nothing -- and the arm that adopts a ``sirens-render``
#: result puts the samples on ``SongTab.pcm``, which is the buffer Space plays.
#: An effect landing there would replace the song with a coin pickup until the
#: next edit re-armed the renderer, which is the sort of thing a user reports as
#: "the song vanished". So: its own key, its own arm, and the tab's buffer is
#: never touched.
AUDITION_PREFIX = "sirens-audition:"


def audition(ctx: Any, tab: SongTab | None, uid: int) -> bool:
    """Render one sound effect and play it. -> whether the render started.

    The song's render shape, one document object smaller: the snapshot is taken
    on the frame thread, ``synth.render_oneshot`` runs on the task thread, and
    :func:`on_task_done` hands the result straight to the mixer rather than
    storing it. Nothing about an audition outlives the sound, so there is
    nothing on the tab for it to live in.

    The device is checked *before* the submit, because a machine with no card
    would otherwise spend seconds of numpy on a buffer with nowhere to go and
    say nothing about why.
    """
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return False
    if tab.doc.oneshot(uid) is None:
        return False
    if not sirens_audio.available():
        ctx.toast(sirens_audio.unavailable_reason(), "warn")
        return False

    from .sirens import wsng

    data = wsng.wsng_bytes(tab.doc)
    effect = int(uid)

    def run() -> dict[str, Any]:
        from ..service.errors import invalid_from
        from .sirens import synth, wavout

        try:
            doc = wsng.read_wsng(data)
            samples = synth.render_oneshot(doc, effect)
        except ValueError as exc:
            raise invalid_from(exc, "That sound effect did not render") from exc
        return {"pcm": wavout.to_int16(samples), "oneshot": effect}

    return bool(ctx.submit(f"{AUDITION_PREFIX}{tab.uid}", run))


# --- playback -----------------------------------------------------------------


def play(ctx: Any, tab: SongTab | None = None) -> bool:
    """Hand the last render to the device. -> whether it started.

    Refuses rather than renders when the buffer is stale: the pump is a frame
    away, and playing the *previous* version of a bar the user has just edited
    is the one outcome that makes them doubt what they heard.
    """
    tab = tab or active(ctx)
    if tab is None:
        return False
    if not sirens_audio.available():
        ctx.toast(sirens_audio.unavailable_reason(), "warn")
        return False
    if tab.rendering or tab.render_dirty:
        # Stale, whether or not an older buffer exists: the transport reads
        # "Rendering..." and the old bar must not play under it.
        ctx.toast("Still rendering your latest edits -- try again in a moment.", "info")
        return False
    if tab.pcm is None:
        ctx.toast("There is nothing in the order list to play yet.", "error")
        return False
    if not sirens_audio.play(tab.pcm):
        ctx.toast("That song could not be played; see the log for details.", "error")
        return False
    return True


def stop(ctx: Any) -> None:
    """Silence, from any surface. A no-op with no device."""
    sirens_audio.stop()


def toggle_play(ctx: Any, tab: SongTab | None = None) -> bool:
    """What Space does: stop if sounding, start if not."""
    if sirens_audio.playing():
        stop(ctx)
        return True
    return play(ctx, tab)


def playhead_row(ctx: Any, tab: SongTab | None = None) -> int | None:
    """Which pattern row is sounding, or ``None``.

    Derived from :func:`sirens_audio.position` and the document's tick rate
    rather than tracked, because the mixer is the only thing that knows how far
    it has got and it does not say -- see ``sirens_audio``'s playhead note. It
    is an estimate to within a buffer and it moves a highlight; nothing depends
    on it.
    """
    tab = tab or active(ctx)
    if tab is None or not sirens_audio.playing():
        return None
    seconds_per_row = tab.doc.speed / tab.doc.tick_rate
    if seconds_per_row <= 0:
        return None
    return int(sirens_audio.position() / seconds_per_row)


# --- task results -------------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "sirens-open":
        if isinstance(result, dict):
            adopt(
                ctx,
                result["doc"],
                path=Path(result["path"]) if result.get("path") else None,
                title=result.get("title"),
            )
            set_mode(ctx.state, "sirens")
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        return

    if name == "sirens-render":
        if isinstance(result, dict):
            tab.adopt_render(result["pcm"], result.get("loop"))
        else:
            tab.rendering = False
        return

    if name == "sirens-audition":
        # Straight to the mixer. Deliberately not through :func:`play`, which
        # is about ``tab.pcm`` -- see :data:`AUDITION_PREFIX` for why an effect
        # never lands there.
        if isinstance(result, dict) and not sirens_audio.play(result["pcm"]):
            ctx.toast("That sound effect could not be played; see the log.", "error")
        return

    if name == "sirens-sample":
        if isinstance(result, dict):
            adopt_sample(ctx, tab, result)
        return

    if name == "sirens-export":
        # Its own arm rather than the fall-through below, because an export is
        # not a save: it writes files *derived* from the document and leaves the
        # document exactly as dirty as it was. Falling through would call
        # ``mark_saved`` and drop the crash copy of work still only in memory.
        tab.saving = False
        if isinstance(result, dict):
            ctx.toast(f"Exported {result['files']} file(s) to {result['directory']}")
        return

    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    tab.mark_saved(result.get("head"))
    # Saved is the moment the crash copy stops describing anything at risk
    # (UX-05); see ``inker_mode``.
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = sirens_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked, and a failed *render*
    must clear ``rendering`` and record why -- a dead Play button with no
    sentence beside it is the worst outcome of a song that would not render."""
    if done.key.startswith(sirens_io.OPEN_PREFIX):
        # Before the tab lookup, because an open that failed has no tab: what
        # it has is a path that does not open, and a Resume list that keeps
        # offering one is worse than a short one.
        forget_path(ctx, done.key.split(":", 1)[1])
        return
    if done.key.startswith(AUDITION_PREFIX) or done.key.startswith(sirens_io.SAMPLE_PREFIX):
        # A refused sample or audition has nothing to unlock: the tab was never
        # locked for either, and clearing ``saving`` here would unlock a *save*
        # that happened to be running alongside. The sentence the user is owed
        # is already the toast the classifier drew before this was called. An
        # *export* is not in this clause, deliberately -- it does lock the tab
        # (``docmodes.start_save``), so it falls through to the unlock below.
        return
    state = ctx.state.sirens
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is None:
        return
    tab.saving = False
    if done.key.startswith("sirens-render"):
        tab.rendering = False
        tab.render_error = done.message or "That song did not render."


# --- guard and keys -----------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them, the ``packwright_mode.guard`` shape. Only
    quitting and closing a tab are destructive: switching modes is not, because
    Sirens is a mode rather than a takeover and its tabs are still there on the
    way back.
    """
    return docmodes.guard(ctx, "sirens", "song", "songs", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    state = ensure(ctx)
    tab = state.get(uid)
    if tab is None:
        return

    def drop() -> None:
        # The document is on disk under a name the user chose, or is gone from
        # the session: either way the crash copy describes work that is no
        # longer at risk, and one left behind is exactly the file that gets
        # offered back after a clean session and confuses somebody (UX-05).
        journal.drop(ctx, tab)
        if state.active_uid == uid:
            # The buffer on this tab is what the mixer is playing; a tab closed
            # mid-bar would otherwise keep sounding with nothing on screen to
            # stop it.
            stop(ctx)
        state.close(uid)

    if not tab.dirty:
        drop()
        return
    dialogs.ask_close_unsaved(ctx, tab.title, drop)


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
_MUTATING_CTRL = frozenset({"z", "y", "x", "v"})


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
        toggle_play(ctx, tab)
        return True
    if event.key in (pygame.K_UP, pygame.K_DOWN):
        move_caret(ctx, drow=1 if event.key == pygame.K_DOWN else -1, select=shift)
        return True
    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        step = 1 if event.key == pygame.K_RIGHT else -1
        # Shift moves by *channel* rather than by column, because a selection
        # is a rectangle over channels and rows -- the column a block starts in
        # is not something a tracker selection has ever carried.
        if shift:
            move_caret(ctx, dchannel=step, select=True)
        else:
            move_caret(ctx, dcolumn=step)
        return True
    if event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
        # A beat is four rows (``document.ROWS_PER_BEAT``); a page is four
        # beats, which is one bar in every time signature this idiom uses.
        move_caret(ctx, drow=16 if event.key == pygame.K_PAGEDOWN else -16, select=shift)
        return True
    if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
        clear_cell(ctx)
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
        transpose(ctx, -1)
        return True
    if event.key == pygame.K_2 and shift:
        transpose(ctx, 1)
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
            write_note(ctx, PIANO_KEYS[name])
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
            write_cell(ctx, notes.NOTE_RELEASE if shift else notes.NOTE_OFF, column=D.NOTE)
            return True
        return False
    if state.column == D.EFFECT:
        return write_effect(ctx, name)
    if len(name) == 1 and name in "0123456789abcdef":
        return write_hex(ctx, int(name, 16))
    return False


def _ctrl_key(
    ctx: Any, state: SirensState, tab: SongTab | None, name: str, *, shift: bool
) -> bool:
    if name == "n":
        new_document(ctx)
        return True
    if name == "o":
        ask_open(ctx)
        return True
    if tab is None:
        return False
    if name == "w":
        close_tab(ctx, tab.uid)
        return True
    if name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
        return True
    if name == "z":
        # Ctrl+Shift+Z redoes as well, which is what Inker, Clay, Plotter and
        # Packwright accept and what a user arriving from any of them already
        # has in their hand. Ctrl+Y keeps working: this adds a spelling rather
        # than replacing one.
        redo(ctx, tab) if shift else undo(ctx, tab)
        return True
    if name == "y":
        redo(ctx, tab)
        return True
    if name == "c":
        copy_selection(ctx)
        return True
    if name == "x":
        cut_selection(ctx)
        return True
    if name == "v":
        paste(ctx)
        return True
    if name == "tab":
        state.cycle(-1 if shift else 1)
        return True
    return False


# --- crash recovery (UX-05) ---------------------------------------------------
#
# ``packwright_mode``'s four answers, for songs. See :mod:`studio.journal`.
#
# The rendered PCM is deliberately not journalled, for the reason it is not
# saved: it is derived, and a re-render of an unchanged document is identical
# with nothing to invalidate. A recovered song re-renders itself, which is
# seconds and is the only answer that cannot be stale.


def _journal_slots(ctx: Any) -> list[Any]:
    state = getattr(ctx.state, "sirens", None)
    if state is None:
        return []
    return [tab for tab in state.docs if tab.dirty and not tab.busy]


def _journal_encode(tab: Any) -> bytes:
    from .sirens import wsng

    return wsng.wsng_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    from .sirens import wsng

    ensure(ctx)
    try:
        doc = wsng.read_wsng(sirens_io._within_ceiling(Path(path)).read_bytes())
    except Exception:
        log.exception("could not reopen the recovered song at %s", path)
        ctx.toast("A recovered song could not be reopened.", "warn", action="log")
        return False
    title = f"{meta.get('title') or Path(path).stem} (recovered)"
    tab = adopt(ctx, doc, path=None, title=title)
    # A recovered document must read dirty until the user saves it somewhere:
    # ``read_wsng`` hands back a document already marked saved, and a clean
    # recovered tab closes without a confirm -- taking the journal copy, the
    # only surviving copy of the work, with it. ``SongTab.dirty`` delegates to
    # the document, so the never-matching head goes on the *document*; the
    # tab's mirror follows so ``mark_saved`` keeps them in step.
    doc.saved_head = -1
    tab.saved_head = -1
    tab.journal_name = Path(path).name
    return True


JOURNAL = journal.register(
    journal.Provider(
        kind="sirens",
        ext=".wsng",
        label="song",
        slots=_journal_slots,
        uid_of=lambda tab: tab.uid,
        title_of=lambda tab: tab.title,
        head_of=lambda tab: tab.doc.history.head,
        encode=_journal_encode,
        adopt=_journal_adopt,
    )
)

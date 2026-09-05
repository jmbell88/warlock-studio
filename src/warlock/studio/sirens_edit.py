"""Sirens' caret and every verb that changes a document.

Split out of :mod:`.sirens_mode` on 2026-09-04 (T7 of the 2026-09-02 review),
which is the one thing section 8 asked of this file's *shape*. Pure code motion
over behaviour the tests beside it had just pinned: the caret, the five columns'
writers, the block clipboard, the instrument sequences and the sample table.

What is **not** here is the reason the split works: rendering and playback went
to :mod:`.sirens_play` and the keyboard to :mod:`.sirens_keys`, and the three
reach each other through ``sirens_mode`` -- imported as a *module*, so its
attribute lookups happen at call time and the import graph has no cycle to
order. Every name below is still reachable at ``sirens_mode.<name>``; see the
``_MOVED`` table at the foot of that file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import sirens_mode
from .sirens_state import COLUMN_DIGITS, SongTab, ensure  # noqa: F401


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
    pattern = sirens_mode.caret_pattern(ctx)
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


def jump_row(ctx: Any, row: int, *, select: bool = False) -> None:
    """Put the caret on an absolute row, keeping Shift's anchor rule.

    Home and End, which is the one movement :func:`move_caret` cannot express:
    a delta large enough to reach the top of a 256-row pattern from row 3
    wraps, because rows are a loop there. Not :func:`set_caret`, which drops
    the selection -- Shift+End selecting to the end of the pattern is the whole
    reason a tracker has the key.
    """
    state = ensure(ctx)
    pattern = sirens_mode.caret_pattern(ctx)
    if pattern is None:
        return
    state.digit = 0
    if select:
        if state.anchor is None:
            state.anchor = (state.row, state.channel)
    else:
        state.anchor = None
    state.row = max(0, min(int(row), pattern.rows - 1))


def set_caret(ctx: Any, *, pattern: int | None = None, row: int | None = None,
              channel: int | None = None, column: int | None = None,
              order_index: int | None = None) -> None:
    """Put the caret somewhere absolute -- what a click and the order list do.

    Through here rather than by assignment so the clamp is not something four
    call sites remember; the selection is dropped for the same reason a tab
    switch drops it, because a rectangle with one corner in a pattern the user
    has left is not a rectangle.

    ``order_index`` says *which entry of the order list* the caret is in, and is
    the order list's to pass (S3, 2026-09-05): a pattern used twice is one uid
    at two places in the song, so a caret that knows only the uid always played
    and highlighted the first of them. A caret set from the grid passes none,
    and the entry is cleared -- there is no honest answer there, and keeping the
    previous entry would be worse than having none.
    """
    state = ensure(ctx)
    if pattern is not None and pattern != state.pattern:
        state.pattern = pattern
        state.anchor = None
    if pattern is not None or order_index is not None:
        state.order_index = None if order_index is None else int(order_index)
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
    written = _write_at_caret(ctx, D.NOTE, values, advance=True)
    if written:
        # After the write, not before: a preview of a note that the document
        # refused would be the editor saying yes and the song saying no.
        sirens_mode.preview_note(ctx, value)
    return written


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
    pattern = sirens_mode.caret_pattern(ctx, tab)
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


def shift_rows(ctx: Any, by: int) -> bool:
    """Insert a blank row at the caret, or take the caret's row out.

    The tracker's Insert and Shift+Delete. Scoped to the block when there is
    one and to the caret's channel when there is not, which is the convention
    and the useful default: rows are where one *part* is written, and pushing
    every channel down to fix the bass line's timing puts a hole in the drums.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    state.digit = 0  # the edit moved on; a half-typed nibble must not land in it
    row, chan, _rows, chans = state.selection() or (state.row, state.channel, 1, 1)
    # From the caret's row, not from the block's top: Insert is about where the
    # caret is, and a block is here only to say *which channels* it reaches.
    return _touch(tab, tab.doc.shift_rows(state.pattern, state.row, chan, chans, by))


def interpolate_selection(ctx: Any) -> bool:
    """Fill the rows between the block's ends with a straight ramp.

    Refuses out loud rather than silently: a block of two rows has nothing
    between its ends and no selection at all has no ends, and both are presses
    a user makes while learning what the verb is for.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None or tab.busy or state.pattern is None:
        return False
    block = state.selection()
    if block is None or block[2] < 3:
        ctx.toast(
            "Select at least three rows to interpolate: the ends are the values"
            " it ramps between.",
            "warn",
        )
        return False
    state.digit = 0  # the edit moved on; a half-typed nibble must not land in it
    return _touch(tab, tab.doc.interpolate(state.pattern, *block))


def update_channel(ctx: Any, uid: int, **values: Any) -> bool:
    """Rename, repan or re-kind one channel, with the refusal framed as a toast.

    ``SongDoc.update_channel`` had no caller at all until 2026-09-04 -- the
    model could do all three and nothing on screen could ask for any of them
    (the 2026-09-02 review, section 8). The header strip over the grid is where
    they went, because that is the one surface that already names a channel.
    """
    tab = ensure(ctx).active
    if tab is None or tab.busy:
        return False
    try:
        changed = tab.doc.update_channel(int(uid), **values)
    except ValueError as exc:
        ctx.toast(f"That channel was not changed: {exc}", "error")
        return False
    return _touch(tab, changed)


def cycle_instrument(ctx: Any, by: int) -> bool:
    """Step the selected instrument, which is what the next note is stamped with.

    From the keyboard because that is where the notes come from: reaching for
    the instrument list between two phrases costs the hand that is typing them.
    Clamped rather than wrapped -- the list has ends, and a step past the last
    one landing on the first is a stamp nobody meant.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None or not tab.doc.instruments:
        return False
    uids = [one.uid for one in tab.doc.instruments]
    index = uids.index(state.instrument) if state.instrument in uids else 0
    state.instrument = uids[max(0, min(index + int(by), len(uids) - 1))]
    return True


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
    key = sirens_mode.free_sample_key(doc, str(result.get("name", "")))
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


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    """Jump to a position in the undo stack -- the history popover's door,
    carrying :func:`undo`'s side effects for :func:`undo`'s reason."""
    moved = tab.doc.history.step_to(tab.doc, index)
    tab.render_dirty = True
    clamp_caret(ctx, tab)
    return moved

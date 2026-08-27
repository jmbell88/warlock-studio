"""Note numbers, the names a tracker column shows, and what they sound like.

**One numbering, and it is not MIDI's.** ``0`` is ``C-0`` and the range runs to
``B-9`` at 119, which is what the octave digit in a pattern cell means: a user
who types ``C-4`` gets middle C, and the number stored in the cell is 48. MIDI
would put middle C at 60 and the column would have to subtract twelve every time
it drew or parsed a cell -- a conversion in two places, which is one place for
the two to disagree about what octave 4 is.

**The sentinels live above the notes rather than beside them.** A pattern cell
is an ``int16`` and its note column holds either a note, one of two commands, or
:data:`EMPTY`. Putting :data:`NOTE_OFF` at 120 rather than at some negative
number means every real note is ``0 <= n <= MAX_NOTE`` and the ordering of the
scale is the ordering of the integers, so a transpose is arithmetic and a clamp
is a clamp. :data:`EMPTY` is ``-1`` for the same reason it is in the other
columns: an empty cell must not be a valid value of anything.

**Equal temperament, A-4 = 440 Hz, and deliberately not a period table.** The
2A03 tunes by writing an 11-bit divider, so its notes are quantised and its top
octave is audibly sharp; reproducing that would be the register-level emulation
this engine is explicitly not (see ``docs/INVARIANTS.md``). The timbre is the
part that makes the era, and the tuning is the part that makes it unusable
beside anything else -- a song exported from here has to sit under a modern
soundtrack without beating against it.
"""

from __future__ import annotations

import math

#: Semitone names, sharps only. A tracker column is three characters wide and
#: ``C#4`` fits where ``Db4`` would need a second spelling of the same pitch.
SEMITONES: tuple[str, ...] = (
    "C-",
    "C#",
    "D-",
    "D#",
    "E-",
    "F-",
    "F#",
    "G-",
    "G#",
    "A-",
    "A#",
    "B-",
)

OCTAVES = 10
MAX_NOTE = OCTAVES * 12 - 1

#: No value in this cell. Shared by every column of a pattern, which is why it
#: is here rather than in :mod:`.document`: a cell is empty in one sense.
EMPTY = -1

#: Stop the note now. The instrument's sequences end and the voice goes silent.
NOTE_OFF = 120

#: Enter the instrument's release phase -- what a volume sequence's release
#: point is for. A note with no release point in its instrument is cut, which
#: makes this identical to :data:`NOTE_OFF` for a simple instrument and is the
#: correct degradation rather than a special case.
NOTE_RELEASE = 121

#: ``A-4``. The one number the rest of the tuning is derived from.
A4_NOTE = 57
A4_HZ = 440.0

#: The reference an instrument's ``sample`` is assumed to have been recorded at.
#: A sample plays back unresampled at this note, so a one-shot drum hit written
#: at ``C-4`` comes out at its recorded speed.
SAMPLE_BASE_NOTE = 48


def is_note(value: int) -> bool:
    """Whether ``value`` is a pitch rather than a command or an empty cell."""
    return 0 <= int(value) <= MAX_NOTE


def name(value: int) -> str:
    """A cell's note column as the three characters the grid draws.

    Every case, including the ones that are not notes, because the caller is a
    draw loop and a ``None`` here becomes a branch at every cell.
    """
    value = int(value)
    if value == EMPTY:
        return "..."
    if value == NOTE_OFF:
        return "==="
    if value == NOTE_RELEASE:
        return "~~~"
    if not is_note(value):
        return "???"
    return f"{SEMITONES[value % 12]}{value // 12}"


def parse(text: str) -> int:
    """``"C-4"`` -> 48. The inverse of :func:`name`, sentinels included.

    Raises ``ValueError`` on anything else rather than returning :data:`EMPTY`:
    this reads hand-edited files and a typo that silently becomes an empty cell
    is a note that vanishes with nothing anywhere to say so.
    """
    text = text.strip()
    if text in ("...", ""):
        return EMPTY
    if text == "===":
        return NOTE_OFF
    if text == "~~~":
        return NOTE_RELEASE
    if len(text) != 3:
        raise ValueError(f"{text!r} is not a note")
    try:
        index = SEMITONES.index(text[:2].upper())
    except ValueError:
        raise ValueError(f"{text!r} is not a note") from None
    if not text[2].isdigit():
        raise ValueError(f"{text!r} is not a note")
    value = int(text[2]) * 12 + index
    if not is_note(value):
        raise ValueError(f"{text!r} is outside the range this build plays")
    return value


def frequency(value: float) -> float:
    """A note number as hertz. Fractional, so a slide can pass between notes.

    Taking a float rather than an int is the whole reason pitch effects need no
    second unit: portamento, vibrato and the pitch sequences all move the note
    number and ask again.
    """
    return A4_HZ * math.pow(2.0, (float(value) - A4_NOTE) / 12.0)


def cents(value: float, offset: float) -> float:
    """``value`` shifted by ``offset`` cents, still in note numbers.

    The effect column's pitch units are cents because a semitone is far too
    coarse for a slide and a raw divider delta means nothing to a reader. One
    hundredth of a semitone is one hundredth of a semitone at every octave,
    which a period delta is not.
    """
    return float(value) + float(offset) / 100.0

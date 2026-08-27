"""What an instrument is: four sequences ticked once per engine tick.

This is the FamiTracker instrument model, and it is the model rather than a
simplification of it because it is the one the music was actually written in.
An NES instrument is not an ADSR envelope with a filter -- it is a short list of
numbers per parameter, stepped once a frame, with a loop point and a release
point. ``volume`` ``[15,14,12,8,4,2,1,0]`` is a pluck; ``arpeggio``
``[0,4,7]`` on loop is a chord played by one voice; ``duty`` ``[0,1,2]`` is the
timbre sweep every NES lead has. A four-parameter ADSR can express none of them.

**A sequence is frozen and a document holds them by value.** They are tiny (a
few dozen ints at most), they are shared between instruments constantly, and an
undo step that stored a reference to one the user then edited in place would
restore the edit it was supposed to reverse. ``tuple`` rather than ``list`` is
what makes that impossible rather than merely unlikely.

**:meth:`Sequence.index_at` is the whole engine, and it is a pure function.**
Everything about how a note evolves over time -- the loop, the release, what
happens when a sequence runs off its end -- is decided in eight lines that take
two integers and return an integer or ``None``. The tick loop in :mod:`.synth`
holds a tick counter per channel and asks; it keeps no sequence cursors of its
own, because a cursor is a second copy of the position and the two drift the
first time a note is retriggered mid-sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The volume scale, and it *is* four bits. Not a hardware limit that got left
#: in: sixteen steps is what makes a hand-typed volume column legible as a
#: single hex digit, and every envelope shape in the genre was written against
#: it. The mixer downstream is float, so nothing is quantised twice.
MAX_VOLUME = 15

#: How far a duty sequence's values reach. Four pulse widths, as on the chip;
#: for a noise channel the same column carries the LFSR mode (0 long, 1 short).
MAX_DUTY = 3

#: What a channel can be. A document's channel says which of these it is and an
#: instrument says which it is *for*, so a pulse instrument played on the noise
#: channel is a refusal rather than a surprise.
KINDS: tuple[str, ...] = ("pulse", "triangle", "noise", "sample")

#: The ceiling on one sequence. Long enough for any envelope anybody writes by
#: hand (four seconds at the default tick rate) and short enough that a
#: hand-edited document cannot ask the tick loop to walk a million entries.
MAX_SEQUENCE_LEN = 256

MAX_NAME_LEN = 32


@dataclass(frozen=True)
class Sequence:
    """One parameter's values over time, with a loop point and a release point.

    **``release`` splits the sequence in two.** Everything before it is what a
    held note plays; everything from it onwards plays only after a note-off.
    That is the FamiTracker rule and it is what makes a sustain expressible at
    all: without it, a decay tail written into the sequence would play while the
    key was still down. ``-1`` means there is no release phase and a note-off
    cuts the voice, which is right for a percussive instrument and is not a
    missing feature.

    ``loop`` is the index the held half jumps back to when it runs off *its*
    end -- which is ``release`` when there is one and the end of the values when
    there is not. ``-1`` holds the last held value forever.

    The released half never loops. A release that loops is a note that never
    ends.
    """

    values: tuple[int, ...] = ()
    loop: int = -1
    release: int = -1

    def __post_init__(self) -> None:
        if len(self.values) > MAX_SEQUENCE_LEN:
            raise ValueError(
                f"a sequence of {len(self.values)} steps is past the"
                f" {MAX_SEQUENCE_LEN} this build ticks"
            )

    def __bool__(self) -> bool:
        return bool(self.values)

    def index_at(self, tick: int, release_tick: int | None = None) -> int | None:
        """Which entry is playing at ``tick``. ``None`` once the sequence is over.

        ``release_tick`` is when the note was released, in the same tick
        numbering, or ``None`` while it is still held.

        The two halves are deliberately not symmetrical. Held, a sequence that
        runs off its end either loops or holds its last value, and both go on
        forever; released, it plays to the end and finishes. That asymmetry is
        what "release" means.
        """
        count = len(self.values)
        if count == 0:
            return None
        if release_tick is None or tick < release_tick:
            # Where the held half stops: the release point when there is one,
            # otherwise the end of the values.
            end = self.release if 0 <= self.release < count else count
            if end <= 0:
                # Every value is release material. Degenerate rather than
                # illegal -- a hand-edited file can say it -- and silence while
                # held is the only reading of it that is not a guess.
                return None
            if tick < end:
                return tick
            if 0 <= self.loop < end:
                # The tail from ``loop`` onwards, repeated. A modulo rather
                # than a while loop because a note held for a minute is three
                # thousand ticks.
                return self.loop + (tick - end) % (end - self.loop)
            return end - 1
        if not 0 <= self.release < count:
            return None
        index = self.release + (tick - release_tick)
        return index if index < count else None

    def value_at(self, tick: int, release_tick: int | None = None, default: int = 0) -> int:
        """:meth:`index_at`'s entry, or ``default`` once the sequence is over."""
        index = self.index_at(tick, release_tick)
        return default if index is None else int(self.values[index])

    def finished(self, tick: int, release_tick: int | None = None) -> bool:
        """Whether a *volume* sequence has ended, which ends the note.

        Only the volume sequence gets a vote on that. An arpeggio that runs out
        does not silence a voice, and asking this of one would.
        """
        return bool(self.values) and self.index_at(tick, release_tick) is None


@dataclass
class Instrument:
    """A named set of sequences, plus which kind of voice they are written for.

    ``uid`` is what a pattern cell stores, never the list position: inserting an
    instrument at the top of the list must not retune the song, which is exactly
    what an index would do. The rest of this package addresses everything the
    same way, for the reason ``studio/undo.py`` states once for all of them.

    It is the one uid in this engine that is **per document and bounded** --
    ``0 <= uid < document.MAX_INSTRUMENTS``, minted by
    ``document._free_instrument_id`` -- because it is the one uid a pattern cell
    has to hold, and a cell is an ``int16``. Everything else keeps
    ``document.new_uid``. That file's docstring carries the argument.
    """

    uid: int
    name: str = ""
    kind: str = "pulse"
    volume: Sequence = field(default_factory=Sequence)
    arpeggio: Sequence = field(default_factory=Sequence)
    pitch: Sequence = field(default_factory=Sequence)
    duty: Sequence = field(default_factory=Sequence)
    #: For ``kind == "sample"``: which entry of the document's sample table.
    sample: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.kind!r} is not one of {', '.join(KINDS)}")
        self.name = self.name[:MAX_NAME_LEN]


def default(uid: int, kind: str = "pulse", name: str = "") -> Instrument:
    """A new instrument that makes a sound the first time it is played.

    An instrument whose sequences are all empty is silent, and a user who adds
    one, types a note and hears nothing has been told the app is broken. So a
    new one is a plain sustained tone at full volume with a short decay tail on
    release -- audible, unremarkable, and obviously a starting point.
    """
    return Instrument(
        uid=uid,
        name=name or f"{kind.capitalize()} {uid}",
        kind=kind,
        # Held: full volume, forever (index 0, looped). Released: a five-tick
        # decay to silence. Audible immediately, obviously a starting point.
        volume=Sequence(values=(15, 12, 9, 6, 3, 0), loop=0, release=1),
        duty=Sequence(values=(2,)) if kind == "pulse" else Sequence(),
    )

"""The tick loop: a document in, samples out.

## Parameters change once per tick, and never between ticks

That is not a shortcut, it is how the hardware works: the CPU writes the sound
registers once a frame and the chip holds those values until the next write.
Reproducing it makes the whole renderer fall out. Within one tick every voice
has a constant frequency, a constant volume and a constant timbre, so its
samples are one vectorised expression over a linear phase ramp -- no per-sample
Python anywhere, and no interpolation decisions to get wrong.

It also makes the block size for :class:`~.voices.Decimator` free: a tick is the
block. There is no boundary between ticks that the filter has to be told about,
because the filter carries its state across them (see that class for what
happens when it does not).

## What is stateful and where it lives

:class:`Voice` is one channel's playing state -- the note, how many ticks it has
been held, where its oscillator phase is, and what each effect has accumulated.
:class:`Player` is the position in the song: which order entry, which row, which
tick of that row. Sequence positions are deliberately *not* state: a voice holds
a tick counter and asks :meth:`~.instruments.Sequence.index_at`, because a
cursor per sequence is a second copy of the position and the copies drift the
first time a note is retriggered mid-envelope.

## The output is a pure function of the document

Same document, same bytes, on every machine and every numpy version this build
supports. That is what makes a re-export reproducible and it is what
``tests/sirens/test_synth.py`` asserts. It is also why the decimation filter in
:mod:`.voices` is written out by hand rather than taken from ``scipy.signal``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from . import document as D
from . import instruments as inst
from . import notes, voices

SAMPLE_RATE = 44100

#: Per-voice amplitude is 0..1, so a mix of several full-volume voices would
#: clip. Five at once -- the default channel set, everything playing -- reaches
#: 0.9 at this gain, which leaves the headroom a sixth channel needs and keeps a
#: single loud voice well clear of the limiter.
MASTER_GAIN = 0.18

#: The ceiling on one render. Ten minutes of 44.1 kHz stereo is a little over
#: 100 MB of ``int16``, which is already past what anything here should be
#: producing; a document that asks for more has a hand-edited order list.
MAX_RENDER_SECONDS = 600

#: How long to keep rendering after the last row, so a release tail is not cut
#: off mid-decay. Only for a song that does **not** loop: a looping song's end
#: is its loop point, and samples past it would make the seam audible.
TAIL_SECONDS = 2.0

#: What the noise channel's note column means. The 2A03 has sixteen fixed noise
#: rates and no way to play a scale; with those limits gone the useful thing is
#: for the keyboard to sweep the timbre, so a note picks the register clock as a
#: multiple of that note's pitch. 32 puts a usable range of hiss under the
#: octaves people actually type -- C-2 rumbles, C-4 is a snare, C-6 is a hat.
NOISE_RATIO = 32.0

# --- the effect column --------------------------------------------------------
#
# Nine effects, and the choice of *which* nine is the design. These are the ones
# without which the idiom cannot be written: an arpeggio (one voice playing a
# chord is how three channels sound like a band), the three pitch effects, a
# vibrato, a volume slide, and the three that control the player itself. Adding
# a tenth is a handler and a row in EFFECT_NAMES; the tick loop below does not
# grow.
#
# **``Fxx`` is a tempo, and that is a deliberate departure from FamiTracker**,
# where it sets ticks per row and changes the tempo as a side effect of doing
# so. Here a row is a sixteenth note by definition (``document.ROWS_PER_BEAT``)
# and ``speed`` is how finely a row is subdivided, so setting it mid-song would
# change how quickly envelopes run and leave the music at exactly the tempo it
# was -- a control that appears to do nothing. What the user reaching for the
# speed effect wants is the audible half of it, so that is what the letter does:
# ``F78`` is 120 BPM. The letter is kept because it is the one every tracker
# uses for "the song's timing changes here".

FX_ARPEGGIO = 0x0
FX_SLIDE_UP = 0x1
FX_SLIDE_DOWN = 0x2
FX_PORTAMENTO = 0x3
FX_VIBRATO = 0x4
FX_VOLUME_SLIDE = 0xA
FX_JUMP = 0xB
FX_HALT = 0xC
FX_BREAK = 0xD
FX_TEMPO = 0xF

#: ``id -> (the letter the grid draws, what it does)``. The letters are the
#: tracker convention rather than an invention: a user arriving from FamiTracker
#: or DefleMask already knows every one of them, and a fresh set of letters
#: would be a dialect for no gain.
EFFECT_NAMES: dict[int, tuple[str, str]] = {
    FX_ARPEGGIO: ("0", "Arpeggio: cycle the note, +x, +y semitones each tick"),
    FX_SLIDE_UP: ("1", "Slide the pitch up, xx cents per tick"),
    FX_SLIDE_DOWN: ("2", "Slide the pitch down, xx cents per tick"),
    FX_PORTAMENTO: ("3", "Slide to the note, xx cents per tick"),
    FX_VIBRATO: ("4", "Vibrato: x is the speed, y the depth"),
    FX_VOLUME_SLIDE: ("A", "Slide the volume, x up or y down per tick"),
    FX_JUMP: ("B", "Jump to order position xx"),
    FX_HALT: ("C", "Stop the song here"),
    FX_BREAK: ("D", "End this pattern, resume at row xx of the next"),
    FX_TEMPO: ("F", "Set the tempo to xx beats per minute"),
}


@dataclass
class Voice:
    """One channel's playing state. See the module docstring for what is not here."""

    kind: str = "pulse"
    pan: float = 0.0

    active: bool = False
    #: The note as it sounds now: fractional, because every pitch effect moves it.
    note: float = 0.0
    #: The note as it was typed. Arpeggio cycles around this, not around ``note``.
    base: float = 0.0
    #: Where a portamento is heading.
    target: float = 0.0

    instrument: inst.Instrument | None = None
    #: Ticks since the note started, which is the sequences' clock.
    tick: int = 0
    release_tick: int | None = None

    #: 0..15, from the volume column, slid by ``Axy``. Persists across rows --
    #: an empty volume column means "unchanged", not "full".
    column_volume: float = 15.0

    slide: float = 0.0
    portamento: float = 0.0
    arpeggio: tuple[int, int] = (0, 0)
    vibrato_speed: float = 0.0
    vibrato_depth: float = 0.0
    vibrato_phase: float = 0.0
    volume_slide: float = 0.0
    #: Cents accumulated by the instrument's pitch sequence, which is relative.
    pitch_bend: float = 0.0

    phase: float = 0.0

    def cut(self) -> None:
        self.active = False
        self.release_tick = None

    def trigger(self, note: int) -> None:
        """Start a note. Resets the phase, which is what makes an attack crisp."""
        self.active = True
        self.note = float(note)
        self.base = float(note)
        self.target = float(note)
        self.tick = 0
        self.release_tick = None
        self.pitch_bend = 0.0
        self.vibrato_phase = 0.0
        self.phase = 0.0


@dataclass
class Player:
    """Where in the song we are, and what the row processor may change about it."""

    order_index: int = 0
    row: int = 0
    tick: int = 0
    #: Ticks per row: how finely a row is subdivided, not how long it lasts.
    speed: int = D.DEFAULT_SPEED
    #: Beats per minute, and therefore how long a row lasts. ``Fxx`` moves it.
    tempo: int = D.DEFAULT_TEMPO
    #: Set by ``Bxx``/``Dxx`` during a row and consumed when the row ends.
    jump_to: int | None = None
    break_to: int | None = None
    halted: bool = False
    voices: list[Voice] = field(default_factory=list)


def _instrument_for(doc: D.SongDoc, uid: int) -> inst.Instrument | None:
    for one in doc.instruments:
        if one.uid == uid:
            return one
    return None


def _apply_row(doc: D.SongDoc, player: Player, cells: np.ndarray) -> None:
    """Read one row of a pattern into the voices. Called on tick 0 of the row.

    ``cells`` is ``(channels, COLUMNS)``.
    """
    for index, voice in enumerate(player.voices):
        if index >= cells.shape[0]:
            break
        note, instrument, volume, effect, param = (int(v) for v in cells[index])

        # The effect first, because a portamento on the same row as a note means
        # "slide to it" rather than "play it", and the note handler has to know.
        voice.arpeggio = (0, 0)
        voice.slide = 0.0
        voice.vibrato_depth = 0.0 if effect != FX_VIBRATO else voice.vibrato_depth
        voice.volume_slide = 0.0
        gliding = effect == FX_PORTAMENTO
        if effect != notes.EMPTY:
            _apply_effect(player, voice, effect, max(0, param))

        if instrument != notes.EMPTY:
            found = _instrument_for(doc, instrument)
            if found is not None:
                voice.instrument = found
        if volume != notes.EMPTY:
            voice.column_volume = float(max(0, min(inst.MAX_VOLUME, volume)))

        if note == notes.NOTE_OFF:
            voice.cut()
        elif note == notes.NOTE_RELEASE:
            if voice.active and voice.release_tick is None:
                voice.release_tick = voice.tick
        elif notes.is_note(note):
            if gliding and voice.active:
                # Slide to it. The envelope keeps running: retriggering here is
                # the difference between a legato line and a stutter.
                voice.target = float(note)
                voice.base = float(note)
            else:
                voice.trigger(note)


def _apply_effect(player: Player, voice: Voice, effect: int, param: int) -> None:
    x, y = (param >> 4) & 0xF, param & 0xF
    if effect == FX_ARPEGGIO:
        voice.arpeggio = (x, y)
    elif effect == FX_SLIDE_UP:
        voice.slide = float(param)
    elif effect == FX_SLIDE_DOWN:
        voice.slide = -float(param)
    elif effect == FX_PORTAMENTO:
        voice.portamento = float(param)
    elif effect == FX_VIBRATO:
        voice.vibrato_speed = x / 16.0
        voice.vibrato_depth = float(y) * 8.0
    elif effect == FX_VOLUME_SLIDE:
        voice.volume_slide = (x - y) / 4.0
    elif effect == FX_JUMP:
        player.jump_to = param
    elif effect == FX_BREAK:
        player.break_to = param
    elif effect == FX_TEMPO:
        if param > 0:
            player.tempo = max(D.MIN_TEMPO, min(D.MAX_TEMPO, param))
    elif effect == FX_HALT:
        player.halted = True


def _advance(voice: Voice) -> None:
    """One tick of everything that moves between rows."""
    if voice.slide:
        voice.note = notes.cents(voice.note, voice.slide)
    if voice.portamento and voice.target != voice.note:
        step = voice.portamento / 100.0
        if voice.target > voice.note:
            voice.note = min(voice.target, voice.note + step)
        else:
            voice.note = max(voice.target, voice.note - step)
    if voice.volume_slide:
        voice.column_volume = max(
            0.0, min(float(inst.MAX_VOLUME), voice.column_volume + voice.volume_slide)
        )
    if voice.vibrato_depth:
        voice.vibrato_phase = (voice.vibrato_phase + voice.vibrato_speed) % 1.0
    if voice.instrument is not None and voice.instrument.pitch:
        voice.pitch_bend += voice.instrument.pitch.value_at(voice.tick, voice.release_tick, 0)


def _sound(voice: Voice, count: int, rate: float, samples: dict[str, np.ndarray]) -> np.ndarray:
    """This voice's contribution for one tick, or an empty array if it is silent.

    Returns the *oversampled* block; the caller sums these and decimates once.
    """
    instrument = voice.instrument
    if not voice.active or instrument is None:
        return np.zeros(0, dtype=np.float32)

    if instrument.volume.finished(voice.tick, voice.release_tick):
        voice.cut()
        return np.zeros(0, dtype=np.float32)

    level = instrument.volume.value_at(voice.tick, voice.release_tick, inst.MAX_VOLUME)
    amp = (level / inst.MAX_VOLUME) * (voice.column_volume / inst.MAX_VOLUME)
    if amp <= 0.0:
        return np.zeros(0, dtype=np.float32)

    # The three things that move a note within a tick's constant parameters.
    arp = 0
    if voice.arpeggio != (0, 0):
        arp = (0, voice.arpeggio[0], voice.arpeggio[1])[voice.tick % 3]
    arp += instrument.arpeggio.value_at(voice.tick, voice.release_tick, 0)
    bend = voice.pitch_bend
    if voice.vibrato_depth:
        bend += math.sin(2.0 * math.pi * voice.vibrato_phase) * voice.vibrato_depth
    pitch = notes.cents(voice.note + arp, bend)

    if voice.kind == "noise":
        mode = instrument.duty.value_at(voice.tick, voice.release_tick, 0)
        clock = notes.frequency(pitch) * NOISE_RATIO
        ramp, voice.phase = voices.phase_ramp(voice.phase, clock / rate, count)
        # Wrapped to the table's period rather than left to grow: a float64
        # phase counting register steps for three minutes loses its low bits.
        voice.phase = math.fmod(voice.phase, float(voices._NOISE[1 if mode else 0].size))
        wave = voices.noise(ramp, mode)
    elif voice.kind == "sample":
        pcm = samples.get(instrument.sample)
        if pcm is None or pcm.size == 0:
            return np.zeros(0, dtype=np.float32)
        ratio = notes.frequency(pitch) / notes.frequency(notes.SAMPLE_BASE_NOTE)
        ramp, voice.phase = voices.phase_ramp(
            voice.phase, ratio / voices.OVERSAMPLE, count
        )
        if voice.phase >= pcm.size:
            # A one-shot that has run out. Cut rather than left "active" with a
            # phase climbing forever, so the voice stops costing anything.
            voice.cut()
        wave = voices.sampled(pcm, ramp)
    else:
        default_duty = 2 if voice.kind == "pulse" else 0
        duty = instrument.duty.value_at(voice.tick, voice.release_tick, default_duty)
        ramp, voice.phase = voices.phase_ramp(voice.phase, notes.frequency(pitch) / rate, count)
        voice.phase = math.fmod(voice.phase, 1.0)
        wave = voices.pulse(ramp, duty) if voice.kind == "pulse" else voices.triangle(ramp)

    return (wave * amp).astype(np.float32)


def _render(
    doc: D.SongDoc,
    order: list[int],
    *,
    tempo: int,
    speed: int,
    loop_order: int = -1,
    rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, tuple[int, int] | None, tuple[tuple[int, int, int, int], ...]]:
    """The shared engine. -> ``(samples (n, 2) float32, loop or None, marks)``."""
    channels = doc.channels
    player = Player(
        speed=max(D.MIN_SPEED, min(D.MAX_SPEED, int(speed))),
        tempo=max(D.MIN_TEMPO, min(D.MAX_TEMPO, int(tempo))),
        voices=[Voice(kind=one.kind, pan=one.pan) for one in channels],
    )
    # **One per side.** A decimator carries the tail of its last block as
    # filter state, so feeding both channels through a single instance makes
    # each one's state the other's history -- which mixes the two together at
    # the filter and puts signal on a channel that is panned fully away from it.
    decimate_l, decimate_r = voices.Decimator(), voices.Decimator()
    over = float(rate * voices.OVERSAMPLE)

    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    produced = 0
    ticks = 0
    # Where the current tempo started, in ticks and in samples. The fractional
    # accumulator below counts from here rather than from zero, so an ``Fxx``
    # halfway through a song does not retroactively re-time what came before it.
    anchor_tick = 0
    anchor_samples = 0
    tempo_now = player.tempo
    ceiling = int(MAX_RENDER_SECONDS * rate)
    looping = 0 <= loop_order < len(order)
    loop_start: int | None = None
    body_end: int | None = None
    tail_left = 0 if looping else int(TAIL_SECONDS * rate)
    # Sample offset at which each order entry was first entered at row 0 --
    # what a backward ``Bxx`` loops back to. See the jump check below.
    order_starts: dict[int, int] = {}
    # One entry per row *as it was actually played*: where it starts in the
    # output, which order entry it belonged to, which pattern that was, and
    # which row. Ascending in the first field by construction, so a playhead is
    # a bisect rather than an estimate -- see :func:`render_marked`.
    marks: list[tuple[int, int, int, int]] = []

    while produced < ceiling:
        playing = 0 <= player.order_index < len(order) and not player.halted
        if not playing:
            if body_end is None:
                body_end = produced
                # Let go of everything still held, so the tail is the
                # instruments' own decay rather than two seconds of sustain
                # followed by a cut. A voice whose envelope loops forever never
                # ends on its own, and the last chord of a song is usually one.
                for one in player.voices:
                    if one.active and one.release_tick is None:
                        one.release_tick = one.tick
            if tail_left <= 0 or not any(one.active for one in player.voices):
                break

        if playing:
            pattern = doc.pattern(order[player.order_index])
            if pattern is None:
                player.order_index += 1
                continue
            if player.row >= pattern.rows:
                player.row = 0
                player.order_index += 1
                continue
            if player.row == 0 and player.tick == 0:
                order_starts.setdefault(player.order_index, produced)
            at_loop = player.order_index == loop_order and player.row == 0
            if loop_start is None and looping and at_loop:
                loop_start = produced
            if player.tick == 0:
                marks.append(
                    (produced, player.order_index, pattern.uid, player.row)
                )
                _apply_row(doc, player, pattern.cells[player.row])

        # How many output samples this tick is worth. Accumulated against the
        # running total rather than rounded per tick, so a tempo whose ticks are
        # not a whole number of samples does not drift over three minutes.
        #
        # Read fresh each tick because ``Fxx`` moves the tempo mid-song, and
        # ``elapsed`` is what makes the accumulator survive that: after a tempo
        # change the samples already produced are the new baseline, so the
        # remaining ticks are counted against it rather than against a total
        # that assumes the old rate ran all along.
        if player.tempo != tempo_now:
            anchor_tick, anchor_samples, tempo_now = ticks, produced, player.tempo
        tick_rate = player.speed * tempo_now * D.ROWS_PER_BEAT / 60.0
        spt = rate / tick_rate
        want = int(round((ticks - anchor_tick + 1) * spt)) + anchor_samples - produced
        want = max(1, want)
        if not playing:
            want = min(want, tail_left)
            tail_left -= want
        count = want * voices.OVERSAMPLE

        mix_l = np.zeros(count, dtype=np.float32)
        mix_r = np.zeros(count, dtype=np.float32)
        for voice in player.voices:
            block = _sound(voice, count, over, doc.samples)
            if block.size:
                mix_l += block * (1.0 - max(0.0, voice.pan))
                mix_r += block * (1.0 + min(0.0, voice.pan))
            if voice.active:
                _advance(voice)
                voice.tick += 1
        left.append(decimate_l.process(mix_l))
        right.append(decimate_r.process(mix_r))
        produced += want
        ticks += 1

        if playing:
            player.tick += 1
            if player.tick >= player.speed:
                player.tick = 0
                was = player.order_index
                jumped = _end_of_row(player, order, pattern.rows)
                if jumped and player.order_index <= was:
                    # A ``Bxx`` to an entry at or before this one is the
                    # tracker idiom for "loop the song here". Without this it
                    # was an infinite song that rendered to
                    # ``MAX_RENDER_SECONDS`` -- ten minutes, on every
                    # keystroke. The body ends at the jump and the loop
                    # points at where the target entry first started.
                    looping = True
                    loop_start = order_starts.get(player.order_index, 0)
                    tail_left = 0
                    player.halted = True

    if not left:
        return np.zeros((0, 2), dtype=np.float32), None, ()
    out = np.stack([np.concatenate(left), np.concatenate(right)], axis=1)
    out = np.clip(out * MASTER_GAIN, -1.0, 1.0).astype(np.float32)
    loop = None
    if looping and loop_start is not None:
        loop = (loop_start, body_end if body_end is not None else out.shape[0])
    return out, loop, tuple(marks)


def _end_of_row(player: Player, order: list[int], rows: int) -> bool:
    """Advance the row, honouring whichever of ``Bxx``/``Dxx`` the row set.
    -> whether a ``Bxx`` was taken.

    A jump and a break on the same row is a real thing people write: the jump
    picks the order entry and the break picks the row within it, so both are
    consumed together rather than one winning.
    """
    jump, brk = player.jump_to, player.break_to
    player.jump_to = player.break_to = None
    if jump is not None or brk is not None:
        player.order_index = jump if jump is not None else player.order_index + 1
        player.row = brk or 0
        return jump is not None
    player.row += 1
    if player.row >= rows:
        player.row = 0
        player.order_index += 1
    return False


def render_marked(
    doc: D.SongDoc, *, rate: int = SAMPLE_RATE
) -> tuple[np.ndarray, tuple[int, int] | None, tuple[tuple[int, int, int, int], ...]]:
    """The whole song, plus the row map. -> ``(samples, loop, marks)``.

    Each mark is ``(sample offset, order index, pattern uid, row)``, one per row
    *as it was actually played* and ascending in the offset -- so "which row is
    sounding" is a bisect of what the renderer did, rather than an estimate.

    The estimate is what this replaces, and it was wrong the moment a song had
    two patterns in it: seconds divided by the document's own seconds-per-row
    describes a single imaginary pattern of unbounded length, ignoring the
    order list, the patterns' own lengths, every ``Fxx`` tempo change and every
    ``Bxx``/``Dxx`` jump. With two patterns the highlight was off the bottom of
    the grid within five seconds and follow mode pinned the view there (the
    2026-09-02 review, section 8).
    """
    return _render(
        doc,
        list(doc.order),
        tempo=doc.tempo,
        speed=doc.speed,
        loop_order=doc.loop_order,
        rate=rate,
    )


def render(doc: D.SongDoc, *, rate: int = SAMPLE_RATE) -> tuple[np.ndarray, tuple[int, int] | None]:
    """The whole song. -> ``(samples (n, 2) float32 in [-1, 1], loop or None)``.

    The two-value form, for every caller that wants audio and nothing else --
    the exports, the tests, the WAV writer. :func:`render_marked` is the same
    render with the row map beside it.

    The loop points are sample offsets into the returned array and are what
    :func:`~.wavout.wav_bytes` writes into the file's ``smpl`` chunk -- which is
    how a game engine is told where to loop without a sidecar nobody reads.
    """
    out, loop, _marks = render_marked(doc, rate=rate)
    return out, loop


def render_only(
    doc: D.SongDoc, channels: Iterable[int], *, rate: int = SAMPLE_RATE
) -> tuple[np.ndarray, tuple[int, int] | None, tuple[tuple[int, int, int, int], ...]]:
    """The song with every channel outside ``channels`` silenced.

    **Not a second rendering path.** There is one tick loop in this build and a
    per-channel variant of it would be a second thing to keep in step with the
    first; what a mute *is*, is the same render of a document with the silenced
    channels' notes taken out. So the note, instrument and volume columns are
    blanked on a copy of each pattern's cells and the ordinary render runs
    unchanged -- which is what a stem export has always done
    (``sirens_io._stem_render``, whose body this now is).

    **The effect column survives, and that is the whole subtlety.** ``Bxx``,
    ``Cxx``, ``Dxx`` and ``Fxx`` are the player's rather than the voice's, and
    any channel may carry them -- so a mix rendered with the muted channels
    wiped clean would jump differently, halt somewhere else and run at a
    different tempo than the mix without the mute. Left in, a muted render is
    sample-aligned with the full one and its row map is the same map. The voice
    effects that survive alongside them (a slide, a vibrato) act on a voice that
    was never triggered, which is silence.

    ``channels`` holds *indices*, not uids: it is the grid's own axis, and the
    cells are indexed by it.
    """
    keep = {int(one) for one in channels}
    saved = {pattern.uid: pattern.cells for pattern in doc.patterns}
    try:
        for pattern in doc.patterns:
            cells = pattern.cells.copy()
            for channel in range(pattern.channels):
                if channel not in keep:
                    cells[:, channel, :D.EFFECT] = notes.EMPTY
            pattern.cells = cells
        return render_marked(doc, rate=rate)
    finally:
        for pattern in doc.patterns:
            if pattern.uid in saved:
                pattern.cells = saved[pattern.uid]


def render_pattern(
    doc: D.SongDoc, pattern: int, *, tempo: int | None = None, speed: int | None = None,
    rate: int = SAMPLE_RATE
) -> np.ndarray:
    """One pattern, once. What auditioning a pattern in the editor plays."""
    if doc.pattern(pattern) is None:
        raise ValueError(D.MISSING_PATTERN)
    out, _loop, _marks = _render(
        doc,
        [pattern],
        tempo=doc.tempo if tempo is None else tempo,
        speed=doc.speed if speed is None else speed,
        rate=rate,
    )
    return out


def render_oneshot(doc: D.SongDoc, uid: int, *, rate: int = SAMPLE_RATE) -> np.ndarray:
    """One sound effect, at its own tempo. What gets exported to ``sfx/``."""
    oneshot = doc.oneshot(uid)
    if oneshot is None:
        raise ValueError(D.MISSING_ONESHOT)
    return render_pattern(
        doc, oneshot.pattern, tempo=oneshot.tempo, speed=oneshot.speed, rate=rate
    )

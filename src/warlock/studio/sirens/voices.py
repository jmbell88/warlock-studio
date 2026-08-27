"""The oscillators, as pure functions over sample arrays.

Nothing here knows what a note or a pattern is. A voice takes a phase array and
returns samples in ``[-1, 1]``; :mod:`.synth` owns every decision about what the
phase should be. That split is what makes the timbre testable on its own -- a
waveform is a few hundred numbers you can assert about, and a song is not.

## Everything runs at :data:`OVERSAMPLE` times the output rate

A square wave is an infinite stack of odd harmonics, and generating one directly
at 44.1 kHz folds every harmonic above Nyquist back down as an inharmonic tone.
It is worst exactly where chiptune leads live: a C-6 pulse at 44.1 kHz has its
3rd, 5th and 7th partials all aliased, and what a listener hears is not a bright
square but a square that is slightly *out of tune*. That is the single most
common complaint about naive chip synthesis and it is not a matter of taste.

So voices render at four times the output rate and the summed mix is decimated
once, through :class:`Decimator`. Four is enough: it moves the first fold to
176 kHz, and what remains below Nyquist after the filter is 60 dB down. Because
mixing and decimation are both linear, **one decimator runs over the finished
mix rather than one per voice** -- which is why this module exposes a filter and
:mod:`.synth` owns the only instance of it.

**The quantisation stays.** :func:`triangle` steps through the chip's 32-entry
staircase and :func:`pulse` has four widths, because that stepping is the
timbre. Anti-aliasing removes an artefact of the sample rate; it does not
smooth the instrument.

:data:`OVERSAMPLE` and the filter width are keyed on by every rendered corpus.
Changing either changes the output of every existing document, so both get a
``docs/measurements/`` entry first -- the rule ``docs/INVARIANTS.md`` states.
"""

from __future__ import annotations

import numpy as np

#: How many samples are generated per output sample. See the module docstring.
OVERSAMPLE = 4

#: The decimation filter's length, in oversampled samples. Odd, so it is
#: symmetric and its delay is an exact integer.
FILTER_TAPS = 63

#: The four pulse widths, as a fraction of the cycle. 25% and 75% are the same
#: waveform inverted and therefore sound identical in isolation -- both are here
#: because they differ when summed with another voice, and because a user
#: reading a duty column expects the four the chip had.
DUTIES: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75)

#: The triangle's staircase: sixteen levels up, sixteen down. Not a parameter.
TRIANGLE_STEPS = 32


def _lfsr_bits(tap: int) -> np.ndarray:
    """One full period of the 2A03 noise register, as ``+1``/``-1``.

    The register is fifteen bits with feedback from bit 0 against bit ``tap``,
    and the output is the *inverse* of bit 0. Two taps are wired on the chip: 1
    gives a 32767-step sequence that sounds like hiss, and 6 gives a 93-step one
    that sounds like a metallic ring -- which is the whole of why the mode bit
    exists and why both are here.

    **Generated once and indexed thereafter.** An LFSR is sequential and cannot
    be vectorised, but it is also periodic, so a table of one period turns the
    whole voice into an array lookup. Without this, a two-minute song at the top
    noise rate would step the register twenty million times in Python.
    """
    register = 1
    out: list[float] = []
    while True:
        out.append(-1.0 if register & 1 else 1.0)
        feedback = (register ^ (register >> tap)) & 1
        register = (register >> 1) | (feedback << 14)
        if register == 1:
            break
    return np.array(out, dtype=np.float32)


_NOISE = (_lfsr_bits(1), _lfsr_bits(6))


def phase_ramp(start: float, increment: float, count: int) -> tuple[np.ndarray, float]:
    """``count`` phases from ``start``, stepping by ``increment``, and the next one.

    Returned as a pair because phase continuity across blocks is the whole
    thing: a voice that restarts its phase at every tick clicks sixty times a
    second. The caller stores the second value and hands it back.

    The running phase is wrapped to ``[0, 1)`` -- for a sample voice, to
    ``[0, length)`` -- by the caller rather than here, because a float32 phase
    that is allowed to grow for three minutes loses the precision that makes the
    top octave in tune.
    """
    ramp = start + increment * np.arange(count, dtype=np.float64)
    return ramp, start + increment * count


def pulse(phase: np.ndarray, duty: int) -> np.ndarray:
    """A square wave of one of the four :data:`DUTIES`."""
    width = DUTIES[int(duty) % len(DUTIES)]
    return np.where(np.mod(phase, 1.0) < width, 1.0, -1.0).astype(np.float32)


def triangle(phase: np.ndarray) -> np.ndarray:
    """The chip's 32-step staircase, not a linear ramp.

    Sixteen levels up and sixteen down, which is what makes an NES triangle
    sound like an NES triangle rather than like a soft sine: the steps put a
    small amount of high harmonic content on what is otherwise a very dull wave,
    and it is audible on a bassline.
    """
    step = np.floor(np.mod(phase, 1.0) * TRIANGLE_STEPS).astype(np.int32)
    level = np.where(step < 16, step, TRIANGLE_STEPS - 1 - step)
    return (level.astype(np.float32) / 7.5) - 1.0


def noise(phase: np.ndarray, mode: int = 0) -> np.ndarray:
    """The noise channel. ``phase`` counts register steps, not cycles.

    ``mode`` picks the tap: 0 is the long, hiss-like sequence and 1 the short,
    pitched one.
    """
    table = _NOISE[1 if int(mode) else 0]
    index = np.mod(phase.astype(np.int64), table.size)
    return table[index]


def sampled(pcm: np.ndarray, phase: np.ndarray, *, loop: bool = False) -> np.ndarray:
    """A PCM sample, linearly interpolated. ``phase`` counts source samples.

    Linear rather than a windowed resampler on purpose: what this plays is a
    drum hit or a bass note at a handful of pitches, the source is usually
    already lo-fi, and the interpolation error sits under the decimation filter
    that runs over the mix anyway.

    Past the end a one-shot is silent -- **not** held at its last sample, which
    would leave a DC step on the mix for as long as the note is held.
    """
    if pcm.size == 0:
        return np.zeros(phase.size, dtype=np.float32)
    position = np.mod(phase, float(pcm.size)) if loop else phase
    left = np.floor(position).astype(np.int64)
    frac = (position - left).astype(np.float32)
    inside = (left >= 0) & (left < pcm.size)
    # Clipped indices so the gather is in bounds; ``inside`` is what actually
    # decides the output, so the clipped lanes contribute nothing.
    low = np.clip(left, 0, pcm.size - 1)
    high = np.clip(left + 1, 0, pcm.size - 1)
    out = pcm[low] * (1.0 - frac) + pcm[high] * frac
    return np.where(inside, out, 0.0).astype(np.float32)


def _lowpass(taps: int, cutoff: float) -> np.ndarray:
    """A Hamming-windowed sinc, normalised to unity gain at DC.

    Written out rather than taken from ``scipy.signal``: it is six lines, it is
    what makes the rendered output byte-identical across scipy versions, and
    "the same document renders the same bytes" is the bar this engine is held
    to. A filter designed by a dependency is a filter that can change under a
    ``uv sync``.
    """
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    h *= 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(taps) / (taps - 1))
    return h / h.sum()


class Decimator:
    """Streaming decimation by :data:`OVERSAMPLE`, with the filter state carried.

    One instance per render, fed one tick's worth of mix at a time. The state is
    the last ``taps - 1`` input samples: without it, filtering each tick
    independently zero-pads both ends of every block and puts a click at sixty
    per second into the output -- which is the failure mode that makes people
    conclude block-based rendering does not work.
    """

    def __init__(self, factor: int = OVERSAMPLE, taps: int = FILTER_TAPS) -> None:
        if taps % 2 == 0:
            raise ValueError("the decimation filter must have an odd number of taps")
        self.factor = int(factor)
        self.taps = int(taps)
        # 0.9 of the target Nyquist: the transition band has to *end* by the
        # time it reaches the output's Nyquist, not begin there.
        self._h = _lowpass(self.taps, 0.9 * 0.5 / self.factor)[::-1].copy()
        self._tail = np.zeros(self.taps - 1, dtype=np.float64)

    def process(self, block: np.ndarray) -> np.ndarray:
        """One block of oversampled input as output samples.

        ``len(block)`` must be a multiple of :attr:`factor`; the caller works in
        whole output samples and multiplies up, so it always is.
        """
        if block.size % self.factor:
            raise ValueError(
                f"a block of {block.size} does not divide into {self.factor} samples"
            )
        padded = np.concatenate([self._tail, block.astype(np.float64, copy=False)])
        windows = np.lib.stride_tricks.sliding_window_view(padded, self.taps)
        out = windows[:: self.factor] @ self._h
        self._tail = padded[-(self.taps - 1) :].copy()
        return out.astype(np.float32)

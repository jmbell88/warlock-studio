"""Where a take can be cut so that it repeats without a seam.

**The model cannot do this and is not asked to.** ACE-Step has no cyclic
objective; the ``loop`` task makes the *joint* something the model wrote rather
than a cut, by repainting across a rolled seam, and that is a different and
smaller claim. It still does not make the first and last samples equal. This
module is the other half: finding the two positions where the music is most
nearly the same, joining them exactly, and crossfading what is left.

Five stages, all numpy, all pure.

1. **Downmix and decimate to 11 025 Hz.** Nyquist at 5.5 kHz is above every
   partial a seam is judged on, and the sizzle above it is what the crossfade
   smooths anyway. 10.6 M stereo frames become 2.6 M mono, which is the
   difference between an analysis you run and one you wait for.
2. **A 24-band log-spaced feature per frame**, L2-normalised. Per *frame*, so
   the score is about content rather than loudness -- level comes back as its
   own term, where it can be weighted.
3. **A coarse pair search**, scored on the frame, its ~370 ms of lead-in, the
   level difference and a length penalty. The lead-in term is what makes this
   work at all: matching one frame finds moments that happen to share a
   spectrum, while matching what *arrives at* each point asks whether the music
   coming into the loop end sounds like the music coming into the loop start.
4. **Sample-accurate refinement** by normalised cross-correlation of the raw
   waveform, then a rising-edge zero-crossing snap.
5. **An equal-power crossfade**, cos/sin rather than linear.

Returns the top few candidates, never one. This is a heuristic over material
nobody composed to loop, and a single answer with no alternatives claims a
confidence the method does not have.

**On the measured-constant rule, honestly.** None of these figures is keyed on
the stored corpus -- music contributes nothing to ``VECTOR_PARAMS``, because no
aggregator reads it -- so ``docs/measurements/`` is not compelled. Every
constant with a derivation carries it below. The three score weights are the
only ones chosen by ear, and they say so. An honest unmeasured constant beats a
measured-sounding one.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: The analysis rate. See stage 1 in the module docstring.
ANALYSIS_RATE = 11025

#: The STFT. 2048 at 11 025 Hz is a 186 ms window -- long enough to resolve a
#: bass note's fundamental (a 40 Hz period is 25 ms) and short enough that a
#: drum hit does not smear across the whole of it. The hop is a quarter of it,
#: which is the standard overlap and puts a frame every 46 ms.
FFT = 2048
HOP = 512

#: How many log-spaced bands the spectrum is reduced to. Roughly a third of an
#: octave from 40 Hz to 5.5 kHz, which is the resolution at which two moments
#: "sound like" each other rather than "are the same recording".
BANDS = 24

#: How much lead-in each candidate is judged on, in frames. Eight hops is
#: ~370 ms -- long enough to carry a bar's worth of onset at most tempos, short
#: enough that a loop point is not forced to agree with the phrase before it.
CONTEXT_FRAMES = 8

#: The coarse grid, in frames. Four hops is 186 ms; stage 4 refines to the
#: sample, so this only has to be fine enough not to miss a candidate's basin.
COARSE_STRIDE = 4

#: How many candidates :func:`find` returns. See the module docstring.
TOP_N = 5

#: How much of the take a loop must cover, as a fraction. A "loop" that is two
#: seconds of a four-minute piece is a sample, not a loop, and offering one as
#: the best answer is the heuristic finding the quietest pair of moments rather
#: than the most similar.
MIN_SPAN = 0.35

#: How much of the raw waveform stage 4 correlates, in milliseconds. Two hops'
#: worth: long enough to lock the phase of a bass note, short enough that the
#: correlation is about the seam rather than about the bar around it.
REFINE_MS = 100.0

#: How far the zero-crossing snap will look, in milliseconds. One period of
#: 50 Hz -- below the fundamental of anything a game soundtrack carries, so a
#: crossing inside this window is always the *same* cycle the correlation
#: chose, never the next one.
ZERO_CROSS_MS = 20.0

# --- the three weights ------------------------------------------------------
#
# **Chosen by ear, and unmeasured.** They are not corpus-keyed, so no
# measurement document is compelled -- but nothing has established that these
# are better than a neighbouring set either, and saying so here is cheaper than
# a comment implying otherwise. What each one is *for* is written down, which is
# what makes them adjustable rather than magic.

#: How much the lead-in matters beside the seam frame itself. Above 1.0 because
#: the lead-in is the term that distinguishes "these two moments sound alike"
#: from "the music arriving here is the music arriving there", and the second is
#: the question.
W_CONTEXT = 1.5

#: How much a level difference costs. A seam between a loud moment and a quiet
#: one is audible as a step even when the spectra match, and the per-frame
#: normalisation in stage 2 deliberately threw level away so it could be
#: weighted here rather than buried in the spectral distance.
W_LEVEL = 0.6

#: How much a short loop costs, applied to how far below :data:`MIN_SPAN` the
#: span falls. A penalty rather than a hard floor so that a genuinely excellent
#: short loop can still be offered -- ranked below a good long one.
W_LENGTH = 2.0


class Candidate(NamedTuple):
    """One loop, in samples of the source take."""

    #: Where the loop begins and ends, as sample offsets into the source.
    start: int
    end: int
    #: The score it was ranked on. Lower is better; it is a distance, not a
    #: confidence, and it is comparable only between candidates of one take.
    score: float

    @property
    def frames(self) -> int:
        return self.end - self.start


def _mono(pcm: np.ndarray) -> np.ndarray:
    data = np.asarray(pcm)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(max(np.iinfo(data.dtype).max, 1))
    return np.ascontiguousarray(data, dtype=np.float32)


def _decimate(mono: np.ndarray, rate: int) -> np.ndarray:
    """Stage 1. Box-filtered decimation to :data:`ANALYSIS_RATE`.

    A box rather than a windowed sinc, and deliberately: this feeds a
    third-octave band analysis, so the alias energy a box lets through lands
    inside bands whose contents are being summed anyway. ``sirens/voices``
    writes a real sinc because *its* output is heard; this output is compared.
    """
    factor = max(1, int(round(rate / ANALYSIS_RATE)))
    if factor == 1:
        return mono
    usable = (mono.size // factor) * factor
    if usable == 0:
        return mono
    return mono[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)


def _bands(count: int) -> np.ndarray:
    """The bin index each of :data:`BANDS` bands ends at. Log-spaced."""
    low, high = 40.0, ANALYSIS_RATE / 2.0
    edges = np.geomspace(low, high, BANDS + 1)
    bins = np.round(edges / (ANALYSIS_RATE / 2.0) * (count - 1)).astype(int)
    # Monotonic and at least one bin wide, so ``reduceat`` never sees a
    # backwards or empty span at the low end where the spacing is tightest.
    return np.maximum.accumulate(np.clip(bins, 1, count - 1))


def features(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2. -> ``(bands, levels)``, one row and one figure per frame.

    ``bands`` is ``(frames, BANDS)`` L2-normalised per frame; ``levels`` is the
    frame's RMS in dB, which is the level the normalisation removed, kept so
    stage 3 can weight it on purpose.

    "Removed" is *almost* exact and not quite: ``log1p`` is non-linear, so
    scaling a frame by 20 dB compresses its band vector rather than scaling it,
    and normalising afterwards leaves a residue of about half a percent. That
    errs the useful way -- two passages differing only in level still read as
    the same content, and the difference is carried at full weight by the level
    term instead.
    """
    if mono.size < FFT:
        mono = np.pad(mono, (0, FFT - mono.size))
    count = 1 + (mono.size - FFT) // HOP
    index = np.arange(FFT)[None, :] + HOP * np.arange(count)[:, None]
    frames = mono[index] * np.hanning(FFT).astype(np.float32)

    spectrum = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    edges = _bands(spectrum.shape[1])
    summed = np.add.reduceat(spectrum, np.r_[0, edges[:-1]], axis=1)[:, :BANDS]
    bands = np.log1p(summed)
    norm = np.linalg.norm(bands, axis=1, keepdims=True)
    bands = bands / np.maximum(norm, 1e-6)

    rms = np.sqrt(np.maximum((frames**2).mean(axis=1), 1e-12))
    return bands.astype(np.float32), (20.0 * np.log10(rms)).astype(np.float32)


def _coarse(bands: np.ndarray, levels: np.ndarray) -> list[tuple[int, int, float]]:
    """Stage 3. -> up to :data:`TOP_N` ``(start_frame, end_frame, score)``.

    One matmul over pre-stacked context vectors rather than a loop over pairs:
    the context term is a dot product of two ``BANDS * CONTEXT_FRAMES`` vectors,
    so stacking them turns the whole search into a single Gram matrix.
    """
    count = bands.shape[0]
    if count <= CONTEXT_FRAMES * 2:
        return []
    grid = np.arange(CONTEXT_FRAMES, count, COARSE_STRIDE)
    if grid.size < 2:
        return []

    # Each grid point as itself plus its lead-in, one flat vector, re-normalised
    # so the dot product below is a cosine over the pair.
    lead = np.stack([bands[g - CONTEXT_FRAMES : g].ravel() for g in grid])
    lead /= np.maximum(np.linalg.norm(lead, axis=1, keepdims=True), 1e-6)
    here = bands[grid]

    # Distance is 1 - cosine on both terms, so both are in [0, 2] and the
    # weights below are comparing like with like.
    seam = 1.0 - here @ here.T
    context = 1.0 - lead @ lead.T
    level = np.abs(levels[grid][:, None] - levels[grid][None, :]) / 40.0

    span = (grid[None, :] - grid[:, None]).astype(np.float32) / max(count, 1)
    shortfall = np.maximum(MIN_SPAN - span, 0.0)
    score = seam + W_CONTEXT * context + W_LEVEL * level + W_LENGTH * shortfall
    # Only start-before-end pairs. ``inf`` rather than a mask so that argsort
    # below needs no second index.
    score = np.where(span > 0.0, score, np.inf)

    order = np.argsort(score, axis=None)[: TOP_N * 8]
    out: list[tuple[int, int, float]] = []
    taken: list[tuple[int, int]] = []
    for flat in order:
        i, j = int(flat // len(grid)), int(flat % len(grid))
        if not np.isfinite(score[i, j]):
            break
        start, end = int(grid[i]), int(grid[j])
        # Candidates within a lead-in of one already taken are the same basin
        # seen twice. Offering five readings of one answer would be the single
        # answer this function refuses to give, wearing a list's clothes.
        if any(
            abs(start - s) < CONTEXT_FRAMES and abs(end - e) < CONTEXT_FRAMES
            for s, e in taken
        ):
            continue
        taken.append((start, end))
        out.append((start, end, float(score[i, j])))
        if len(out) == TOP_N:
            break
    return out


def _refine(mono: np.ndarray, start: int, end: int, rate: int) -> tuple[int, int]:
    """Stage 4. Correlate, then snap both markers to a *rising* crossing.

    Rising for both, never nearest-of-either: two crossings of matching sign
    have matching slope, whereas joining a rising edge to a falling one clicks
    audibly even though both samples are zero. That is the whole reason this
    function does not simply take ``argmin(abs(x))``.

    A marker with no crossing in its window is left where the correlation put
    it -- ``envelope.marker_bounds``' rule, and the honest floor: moving it
    further to find one would move it out of the basin that was chosen.
    """
    half = int(REFINE_MS * rate / 1000.0) // 2
    if half < 2 or start + half >= mono.size or end + half >= mono.size:
        return start, end

    target = mono[max(start - half, 0) : start + half]
    lo = max(end - half * 2, 0)
    hi = min(end + half * 2, mono.size)
    region = mono[lo:hi]
    if region.size <= target.size or target.size == 0:
        return start, end

    # Normalised, so the match is about shape rather than about which candidate
    # happens to be louder -- the same reason stage 2 normalises per frame.
    corr = np.correlate(region, target, mode="valid")
    energy = np.sqrt(
        np.maximum(
            np.convolve(region**2, np.ones(target.size, dtype=np.float32), mode="valid"),
            1e-9,
        )
    )
    end = lo + int(np.argmax(corr / energy)) + half

    return _snap(mono, start, rate), _snap(mono, end, rate)


def _snap(mono: np.ndarray, at: int, rate: int) -> int:
    """The nearest rising zero crossing within :data:`ZERO_CROSS_MS`, or ``at``."""
    reach = int(ZERO_CROSS_MS * rate / 1000.0)
    lo, hi = max(at - reach, 0), min(at + reach, mono.size - 1)
    if hi - lo < 2:
        return at
    window = mono[lo:hi]
    rising = np.flatnonzero((window[:-1] <= 0.0) & (window[1:] > 0.0))
    if rising.size == 0:
        return at
    return int(lo + rising[np.argmin(np.abs(lo + rising - at))])


def find(pcm: np.ndarray, rate: int) -> list[Candidate]:
    """The loop points of ``pcm``, best first. -> up to :data:`TOP_N`.

    Runs on a task thread, which is why being slow costs a spinner rather than
    a frozen window -- and why no attempt is made to bound the analysis by
    anything but the take's own length.
    """
    mono = _mono(pcm)
    if mono.size < rate:
        return []
    small = _decimate(mono, rate)
    ratio = mono.size / max(small.size, 1)

    bands, levels = features(small)
    out: list[Candidate] = []
    for start, end, score in _coarse(bands, levels):
        # Analysis frames back to source samples: hop, then the decimation.
        a = int(start * HOP * ratio)
        b = int(end * HOP * ratio)
        a, b = _refine(mono, min(a, mono.size - 1), min(b, mono.size - 1), rate)
        if b > a:
            out.append(Candidate(a, b, score))
    return out


def _wrap_step(a: np.ndarray, b: np.ndarray) -> float:
    """How far the signal jumps going from sample ``a`` straight to sample ``b``.

    Summed across channels rather than taken per channel, so the three
    candidate joins below are ranked by one number: a stereo loop has one
    seam, and picking a join that fixes the left channel by wrecking the right
    is not an improvement anybody can hear as one.
    """
    lhs = np.asarray(a, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(rhs - lhs)))


def crossfade(pcm: np.ndarray, start: int, end: int, fade: int) -> np.ndarray:
    """The loop body, joined at whichever seam is measurably smallest. -> same
    dtype, same length as ``end - start``.

    **The thing being minimised is the wrap, and a fade that does not reduce it
    is not applied (2026-09-05, finding M08).** What a repeated loop plays
    across is the join from the body's *last* sample straight back to its
    *first*. That is the only discontinuity a repeat has, so it is the only
    number worth spending a fade on -- and a fade is only worth applying when
    it makes that number smaller. Two earlier versions each blended material in
    somewhere and called the job done without ever measuring the wrap they were
    supposedly fixing: the first blended pre-``start`` material into the head
    (which changes the loop's first instant and leaves the tail exactly where
    it was), and the second replaced the tail with ``tail * cos + head * sin``,
    which ends the body on ``head[fade - 1]`` -- so playback wraps to
    ``head[0]`` and time jumps *backwards* by ``fade`` samples every repeat.
    Measured on a 440 Hz tone, region ``[1000, 60000)``, 2048-sample fade: that
    version's seam stepped 14502 against an interior maximum of 1320, where
    doing nothing at all stepped 11695. The fade was making the click worse.

    So the three joins are costed first, all in O(1) off the source:

    ==========================  ===============================  =============
    choice                      resulting wrap                   needs
    ==========================  ===============================  =============
    no fade                     ``|data[end-1] - data[start]|``  --
    fade tail toward pre-start  ``|data[start-1] - data[start]|`` ``start > 0``
    fade post-end into head     ``|data[end-1] - data[end]|``    ``end < len``
    ==========================  ===============================  =============

    The two fades are the two ways to make the wrap land on a join that is
    *already continuous in the source*: end the body on ``data[start - 1]``, or
    begin it on ``data[end]``. Either way the repeat crosses a step the take
    itself contains, which is generally tiny. The smallest of the three wins,
    and when neither fade beats leaving it alone the body is returned untouched
    -- which is what closes M08's own repro, a constant plateau whose untreated
    wrap is already exactly zero and cannot be improved on.

    **Equal power, cos/sin, not linear.** Two decorrelated signals crossfaded
    linearly sum to about 0.71 of their level at the midpoint -- a ~3 dB dip,
    which is an audible hole once per repeat. ``cos``/``sin`` sum to 1 in
    *power*, which is what two unrelated signals actually add as.

    **Duration policy.** ``fade`` is clamped to at most half the body's own
    length, so the faded region can never cover the body twice, and to however
    much material the chosen side actually has outside the region.
    """
    data = np.asarray(pcm)
    dtype = data.dtype
    body = data[start:end].astype(np.float32).copy()
    n = body.shape[0]
    fade = int(max(0, min(fade, n // 2)))
    if fade <= 0 or n == 0:
        return body.astype(dtype)

    # Cost every join before touching a sample. ``plain`` is the incumbent: a
    # fade has to beat it outright, not merely tie, or it is churn.
    plain = _wrap_step(data[end - 1], data[start])
    candidates: list[tuple[float, str, int]] = []
    if start > 0:
        candidates.append((_wrap_step(data[start - 1], data[start]), "tail", min(fade, start)))
    if end < data.shape[0]:
        room = data.shape[0] - end
        candidates.append((_wrap_step(data[end - 1], data[end]), "head", min(fade, room)))
    usable = [c for c in candidates if c[2] > 0 and c[0] < plain]
    if not usable:
        return body.astype(dtype)
    _, side, fade = min(usable, key=lambda c: c[0])

    angle = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
    rising, falling = np.sin(angle), np.cos(angle)
    if body.ndim == 2:
        rising, falling = rising[:, None], falling[:, None]

    if side == "tail":
        # The body fades out into the material that immediately precedes the
        # region, so it ends on ``data[start - 1]`` -- the sample the source
        # itself puts before ``data[start]``.
        lead_out = data[start - fade : start].astype(np.float32)
        body[n - fade : n] = body[n - fade : n] * falling + lead_out * rising
    else:
        # The body begins on ``data[end]`` and fades into its own head, so the
        # wrap crosses the source's own ``end - 1 -> end`` step.
        lead_in = data[end : end + fade].astype(np.float32)
        body[:fade] = lead_in * falling + body[:fade] * rising

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        body = np.clip(body, info.min, info.max)
    return body.astype(dtype)


__all__ = [
    "ANALYSIS_RATE",
    "BANDS",
    "CONTEXT_FRAMES",
    "MIN_SPAN",
    "TOP_N",
    "ZERO_CROSS_MS",
    "Candidate",
    "crossfade",
    "features",
    "find",
]

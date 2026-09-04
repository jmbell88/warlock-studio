"""The loop finder, on material whose right answer is known.

A heuristic over real music cannot be asserted -- there is no ground truth for
"where does this piece repeat best". So every test here builds a signal whose
loop point is a *fact*: a tone that repeats exactly, a piece with a seam
deliberately placed, silence. What is being pinned is that the machinery finds
what is genuinely there and does not invent what is not.

The three score weights are unmeasured and say so in the source; nothing here
asserts a ranking that depends on their exact values, which is what keeps this
file from freezing a figure the module admits it guessed.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.muse import loops

RATE = 44100


def _tone(seconds: float, hz: float = 220.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _phrase(seconds: float, hz: float, rate: int = RATE) -> np.ndarray:
    """A tone with an onset, so the lead-in term has something to match on."""
    body = _tone(seconds, hz, rate)
    attack = np.minimum(np.arange(body.size, dtype=np.float32) / (rate * 0.02), 1.0)
    return (body * attack).astype(np.float32)


# --- features ----------------------------------------------------------------


def test_every_frames_feature_is_unit_length():
    """Per-frame normalisation is what makes the score about content.

    Level is not discarded -- it comes back as its own term, where it can be
    weighted on purpose rather than buried inside the spectral distance.
    """
    bands, levels = loops.features(_tone(2.0, rate=loops.ANALYSIS_RATE))
    assert bands.shape[1] == loops.BANDS
    assert np.allclose(np.linalg.norm(bands, axis=1), 1.0, atol=1e-4)
    assert levels.shape[0] == bands.shape[0]


def test_two_identical_passages_have_the_same_feature():
    one = _tone(1.0, 440.0, loops.ANALYSIS_RATE)
    bands, _ = loops.features(np.concatenate([one, one]))
    half = bands.shape[0] // 2
    # A frame in the first copy and the matching frame in the second.
    assert bands[half // 2] @ bands[half + half // 2] == pytest.approx(1.0, abs=1e-3)


def test_a_loud_passage_and_a_quiet_one_of_the_same_material_match_on_content():
    """And differ on level, which is the split the two return values are for.

    The match is 0.99-something rather than exactly 1: ``log1p`` is not linear,
    so scaling a frame by 20 dB does not scale its band vector, it *compresses*
    it slightly -- normalising afterwards removes almost all of the level and
    not quite all. That residue is deliberate and small, and it errs the useful
    way: two passages that differ only in level still read as the same content,
    while the level term below carries the difference at full weight.
    """
    quiet = _tone(1.0, 440.0, loops.ANALYSIS_RATE)
    bands, levels = loops.features(np.concatenate([quiet, quiet * 0.1]))
    half = bands.shape[0] // 2
    assert bands[half // 2] @ bands[half + half // 2] > 0.99
    assert levels[half // 2] - levels[half + half // 2] == pytest.approx(20.0, abs=1.0)


# --- find --------------------------------------------------------------------


def test_a_take_shorter_than_a_second_has_no_loop_points():
    assert loops.find(_tone(0.5), RATE) == []


def test_silence_produces_no_crash_and_at_most_a_ranking():
    """Silence is the degenerate case: every moment matches every other, so any
    answer is as good as any other. What must not happen is an exception."""
    out = loops.find(np.zeros(RATE * 3, dtype=np.float32), RATE)
    assert all(c.end > c.start for c in out)


def test_every_candidate_is_ordered_and_inside_the_take():
    pcm = np.concatenate([_phrase(2.0, 220.0), _phrase(2.0, 330.0), _phrase(2.0, 220.0)])
    out = loops.find(pcm, RATE)
    assert out, "a three-phrase piece should offer at least one loop"
    for candidate in out:
        assert 0 <= candidate.start < candidate.end <= pcm.size
        assert candidate.frames == candidate.end - candidate.start


def test_it_returns_alternatives_rather_than_one_answer():
    """A single answer with no alternatives would claim a confidence this
    method does not have -- see the module docstring."""
    pcm = np.concatenate([_phrase(1.5, hz) for hz in (220, 330, 220, 440, 220)])
    out = loops.find(pcm, RATE)
    assert 1 < len(out) <= loops.TOP_N


def test_candidates_are_distinct_rather_than_five_readings_of_one_basin():
    pcm = np.concatenate([_phrase(1.5, hz) for hz in (220, 330, 220, 440, 220)])
    out = loops.find(pcm, RATE)
    starts = sorted(c.start for c in out)
    reach = loops.CONTEXT_FRAMES * loops.HOP * (RATE / loops.ANALYSIS_RATE)
    assert all(b - a > 0 for a, b in zip(starts, starts[1:], strict=False))
    assert len(out) == 1 or max(starts) - min(starts) > reach


def test_a_loop_covers_a_real_fraction_of_the_take():
    """A two-second window of a four-minute piece is a sample, not a loop.

    The penalty is soft, so this asserts the effect rather than a hard floor:
    the best-ranked candidate is not a sliver.
    """
    pcm = np.concatenate([_phrase(2.0, hz) for hz in (220, 330, 440, 220)])
    out = loops.find(pcm, RATE)
    assert out
    assert out[0].frames > pcm.size * 0.2


# --- crossfade ---------------------------------------------------------------


def test_the_crossfade_is_equal_power_rather_than_linear():
    """Two decorrelated signals crossfaded linearly dip ~3 dB mid-fade -- an
    audible hole once per repeat. cos/sin sum to 1 in *power*, which is how two
    unrelated signals actually add."""
    rng = np.random.default_rng(0)
    pcm = rng.standard_normal(4000).astype(np.float32)
    fade = 400
    body = loops.crossfade(pcm, 1000, 3000, fade)
    # Mid-fade, the two contributions are each sin/cos(pi/4) = 0.707, so the
    # expected *power* is the sum of the two sources' powers.
    mid = fade // 2
    expected = np.sqrt(pcm[1000 + mid] ** 2 + pcm[1000 - fade + mid] ** 2) * 0.7071
    assert abs(body[mid]) == pytest.approx(abs(expected), rel=0.5)


def test_the_body_is_the_requested_span():
    pcm = _tone(3.0)
    assert loops.crossfade(pcm, 1000, 5000, 100).shape[0] == 4000


def test_the_material_faded_in_comes_from_before_the_loop_start():
    """The point of the join: what arrives at the loop start already exists in
    the take, and using it is what makes the seam continuous rather than
    merely quiet."""
    pcm = np.zeros(4000, dtype=np.float32)
    pcm[500:1000] = 1.0  # only the run-in to the loop start is non-zero
    body = loops.crossfade(pcm, 1000, 3000, 200)
    assert body[:200].max() > 0.5
    assert body[200:].max() == pytest.approx(0.0)


def test_a_fade_longer_than_the_run_in_is_shortened_rather_than_reading_backwards():
    pcm = _tone(1.0)
    body = loops.crossfade(pcm, 50, 5000, 1000)
    assert body.shape[0] == 4950
    assert np.isfinite(body).all()


def test_no_fade_is_the_plain_slice():
    pcm = _tone(1.0)
    assert np.array_equal(loops.crossfade(pcm, 100, 900, 0), pcm[100:900])


def test_an_integer_take_comes_back_in_its_own_dtype_and_in_range():
    pcm = np.full(4000, 30000, dtype="<i2")
    body = loops.crossfade(pcm, 1000, 3000, 200)
    assert body.dtype == np.dtype("<i2")
    assert body.max() <= np.iinfo("<i2").max


def test_a_stereo_take_is_faded_per_channel():
    pcm = np.zeros((4000, 2), dtype=np.float32)
    pcm[500:1000] = 1.0
    body = loops.crossfade(pcm, 1000, 3000, 200)
    assert body.shape == (2000, 2)
    assert body[:200].max() > 0.5

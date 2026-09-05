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
#
# M08: the previous version blended material from *before* ``start`` into the
# loop's head, which does nothing about the seam a repeated loop actually
# plays across -- the join between the *tail* of one repetition and the *head*
# of the next. Every test below is about that seam, on the *repeated* output,
# not about the buffer's own first samples.


def test_the_crossfade_is_equal_power_rather_than_linear():
    """The blend at the tail is exactly ``tail * cos + head * sin`` -- weights
    whose *squares* sum to 1, which is what keeps two decorrelated signals from
    dipping ~3 dB at mid-fade the way a linear (weights summing to 1) blend
    would. Asserted as the exact per-sample formula rather than a statistical
    approximation over random material, which is fragile by construction: any
    single sample's instantaneous value depends on both signals' signs, not
    only on their power.
    """
    pcm = np.arange(4000, dtype=np.float32)
    fade = 400
    body = loops.crossfade(pcm, 0, 4000, fade)
    tail, head = pcm[4000 - fade : 4000], pcm[:fade]
    angle = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
    expected_tail = tail * np.cos(angle) + head * np.sin(angle)
    np.testing.assert_allclose(body[4000 - fade :], expected_tail, rtol=1e-4)


def test_the_body_is_the_requested_span():
    pcm = _tone(3.0)
    assert loops.crossfade(pcm, 1000, 5000, 100).shape[0] == 4000


def test_crossfade_does_not_introduce_a_click_into_an_already_seamless_loop():
    """M08's own repro. A constant loop has a zero-jump seam already: blending
    whatever comes *before* ``start`` -- a different constant here -- into the
    head introduced a jump the untreated loop never had. Fails against the
    unfixed code, which reads exactly that 40000-unit jump at the wrap.
    """
    pcm = np.concatenate(
        [np.full(500, -20000, dtype=np.int16), np.full(2000, 20000, dtype=np.int16)]
    )
    start, end, fade = 500, 2500, 200
    original_seam = abs(int(pcm[end - 1]) - int(pcm[start]))
    assert original_seam == 0

    body = loops.crossfade(pcm, start, end, fade)
    repeated = np.concatenate([body, body])
    seam_jump = abs(int(repeated[len(body) - 1]) - int(repeated[len(body)]))
    assert seam_jump == 0


def test_the_material_faded_into_the_tail_comes_from_the_bodys_own_head():
    """The join is tail-against-head, both from *inside* the body -- never
    material from before ``start``, which the old version reached for and
    which need not resemble the body's own end at all.
    """
    pcm = np.zeros(4000, dtype=np.float32)
    pcm[1000:1200] = 1.0  # only the head of the body (from 1000) is non-zero
    body = loops.crossfade(pcm, 1000, 3000, 200)
    # The tail (the last 200 samples of the body) now leans toward the head's
    # content; the untouched middle stays at zero.
    assert body[-200:].max() > 0.5
    assert body[200:-200].max() == pytest.approx(0.0)


def test_crossfade_still_fades_when_the_region_starts_at_the_top_of_the_take():
    """The old algorithm needed lead-in material from before ``start`` and so
    clamped ``fade`` by ``start`` -- at ``start == 0`` there is none, and it
    silently fell back to no fade at all. Fails against the unfixed code,
    which returns the plain, un-blended slice whenever ``start == 0``.
    """
    pcm = np.concatenate(
        [np.full(1000, 1000.0, dtype=np.float32), np.full(1000, -1000.0, dtype=np.float32)]
    )
    body = loops.crossfade(pcm, 0, 2000, 200)
    assert body[-1] != pytest.approx(-1000.0)


def test_a_fade_longer_than_half_the_body_is_capped_so_head_and_tail_never_overlap():
    """Duration policy: the fade cannot exceed half the body, or the fading
    head and tail regions would read the same samples twice. A short body with
    a distinct head and tail still gets a seamless repeat -- fails against the
    unfixed code, whose ``fade = min(fade, start, len(body))`` has no notion
    of "half the body" and, at ``start == 0``, disables the fade outright
    (leaving the 2000-unit jump between B and A at the repeat).
    """
    pcm = np.concatenate([np.full(25, 1000, dtype=np.int16), np.full(25, -1000, dtype=np.int16)])
    body = loops.crossfade(pcm, 0, 50, 1000)  # fade requested far past the body
    assert body.shape[0] == 50
    assert np.isfinite(body).all()
    repeated = np.concatenate([body, body])
    seam_jump = abs(int(repeated[49]) - int(repeated[50]))
    assert seam_jump < 2000  # smaller than the untreated jump between B and A


def test_no_fade_is_the_plain_slice():
    pcm = _tone(1.0)
    assert np.array_equal(loops.crossfade(pcm, 100, 900, 0), pcm[100:900])


def test_an_integer_take_comes_back_in_its_own_dtype_and_in_range():
    pcm = np.full(4000, 30000, dtype="<i2")
    body = loops.crossfade(pcm, 1000, 3000, 200)
    assert body.dtype == np.dtype("<i2")
    assert body.max() <= np.iinfo("<i2").max


def test_a_stereo_takes_seam_is_faded_independently_per_channel():
    pcm = np.zeros((4000, 2), dtype=np.float32)
    pcm[1000:1200, 0] = 1.0  # only channel 0's head is non-zero
    pcm[1000:1200, 1] = -1.0  # channel 1's head is the opposite sign
    body = loops.crossfade(pcm, 1000, 3000, 200)
    assert body.shape == (2000, 2)
    assert body[-200:, 0].max() > 0.5
    assert body[-200:, 1].min() < -0.5

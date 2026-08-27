"""The oscillators and the decimation filter.

The waveform tests are exact: a 50% pulse at eight samples per cycle is four
highs and four lows, and asserting that is worth more than any spectral
measurement because it fails legibly.

The filter tests are the ones that matter for how it *sounds*, and they exist
because both of the failures they cover are silent. A filter with the wrong
gain makes everything quiet; a filter without carried state clicks sixty times
a second, which people diagnose as "block rendering doesn't work".
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.sirens import voices


def test_the_two_noise_taps_have_the_periods_the_chip_has():
    """32767 is hiss and 93 is a metallic ring, and the mode bit exists to pick
    between them. A wrong tap gives a sequence of the wrong length and the
    short mode stops sounding pitched."""
    assert voices._NOISE[0].size == 32767
    assert voices._NOISE[1].size == 93


def test_the_noise_table_is_only_ever_plus_or_minus_one():
    for table in voices._NOISE:
        assert set(np.unique(table)) == {-1.0, 1.0}


def test_a_fifty_percent_pulse_is_half_high():
    phase = np.arange(8) / 8.0
    assert list(voices.pulse(phase, 2)) == [1, 1, 1, 1, -1, -1, -1, -1]


def test_the_four_duties_are_the_four_widths():
    phase = np.arange(16) / 16.0
    highs = [int((voices.pulse(phase, d) > 0).sum()) for d in range(4)]
    assert highs == [2, 4, 8, 12]


def test_the_triangle_is_a_staircase_and_not_a_ramp():
    """The stepping is the timbre. A linear triangle sounds like a soft sine and
    is the single most common way an NES bassline comes out wrong."""
    wave = voices.triangle(np.arange(64) / 64.0)
    assert len(np.unique(wave)) == 16
    assert wave.min() == pytest.approx(-1.0)
    assert wave.max() == pytest.approx(1.0)


def test_the_triangle_is_symmetric_about_its_peak():
    """Sixteen steps up then the same sixteen back down -- the chip's sequence,
    not a resampled ramp that happens to look like one."""
    wave = voices.triangle(np.arange(32) / 32.0)
    assert list(wave[16:32]) == list(wave[15::-1])


def test_a_one_shot_sample_is_silent_past_its_end_rather_than_held():
    """Held would leave a DC step on the mix for as long as the note is, which
    is inaudible on its own and shifts everything else off centre."""
    pcm = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    out = voices.sampled(pcm, np.arange(6, dtype=np.float64))
    assert list(out) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_a_looping_sample_wraps():
    pcm = np.array([1.0, 2.0], dtype=np.float32)
    out = voices.sampled(pcm, np.arange(4, dtype=np.float64), loop=True)
    assert list(out) == [1.0, 2.0, 1.0, 2.0]


def test_an_empty_sample_is_silence_rather_than_an_error():
    out = voices.sampled(np.zeros(0, dtype=np.float32), np.arange(4, dtype=np.float64))
    assert out.size == 4 and not out.any()


def test_the_phase_ramp_hands_back_where_it_got_to():
    ramp, nxt = voices.phase_ramp(0.25, 0.5, 3)
    assert list(ramp) == [0.25, 0.75, 1.25]
    assert nxt == pytest.approx(1.75)


def test_the_decimation_filter_has_unity_gain_at_dc():
    """Off by a few percent and the whole app is quieter than it should be, with
    nothing on screen to say so."""
    assert voices.Decimator()._h.sum() == pytest.approx(1.0)


def test_decimating_in_blocks_matches_decimating_in_one_go():
    """The carried-state guarantee, stated as an equality rather than as an
    absence of clicks. Filtering each block independently zero-pads both of its
    ends and puts a discontinuity at every seam."""
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(4 * 512).astype(np.float64)
    whole = voices.Decimator().process(signal)
    blocked = voices.Decimator()
    pieces = [blocked.process(signal[i * 512 : (i + 1) * 512]) for i in range(4)]
    assert np.allclose(whole, np.concatenate(pieces))


def test_a_block_that_does_not_divide_is_refused():
    with pytest.raises(ValueError, match="does not divide"):
        voices.Decimator().process(np.zeros(7))


def test_the_filter_must_be_symmetric():
    with pytest.raises(ValueError, match="odd number of taps"):
        voices.Decimator(taps=64)


def test_oversampling_removes_the_foldover_that_makes_high_notes_sound_flat():
    """The measurement behind :data:`voices.OVERSAMPLE`.

    A 5 kHz square at 44.1 kHz has its 5th harmonic at 25 kHz, which folds back
    to 19 kHz -- an inharmonic tone a listener hears as the note being out of
    tune rather than as brightness. Rendering four times up and filtering down
    should leave far less energy in the bins that are not multiples of the
    fundamental.
    """
    rate, freq, count = 44100, 5000.0, 4096

    naive = voices.pulse(np.arange(count) * (freq / rate), 2)

    over = voices.pulse(np.arange(count * 4) * (freq / (rate * 4)), 2)
    clean = voices.Decimator().process(over)

    def inharmonic(signal: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
        bins = np.arange(spectrum.size) * rate / signal.size
        harmonics = np.zeros(spectrum.size, dtype=bool)
        for n in range(1, 30):
            harmonics |= np.abs(bins - n * freq) < 120.0
        return float(spectrum[~harmonics].sum() / spectrum.sum())

    assert inharmonic(clean) < inharmonic(naive) / 2.0

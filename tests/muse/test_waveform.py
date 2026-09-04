"""The envelope, and the mapping the player's four controls share."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.muse import waveform


def test_a_silent_take_draws_flat():
    env = waveform.peaks(np.zeros(44100, dtype=np.float32), columns=64)
    assert env.shape == (2, 64)
    assert np.allclose(env, 0.0)


def test_the_two_rows_are_the_minimum_and_the_maximum():
    pcm = np.array([-1.0, 1.0, -0.5, 0.5], dtype=np.float32)
    env = waveform.peaks(pcm, columns=2)
    assert env[0].tolist() == [-1.0, -0.5]
    assert env[1].tolist() == [1.0, 0.5]


def test_an_integer_take_is_scaled_by_its_dtypes_range_not_by_its_own_peak():
    """Normalising per take would make a quiet piece draw as loud as a loud one.

    A tray of candidates exists to be compared, so two takes' pictures have to
    mean the same thing.
    """
    quiet = waveform.peaks(np.full(1000, 3276, dtype="<i2"), columns=4)
    loud = waveform.peaks(np.full(1000, 32767, dtype="<i2"), columns=4)
    assert quiet[1].max() == pytest.approx(0.1, abs=0.01)
    assert loud[1].max() == pytest.approx(1.0, abs=0.001)


def test_stereo_is_downmixed_because_the_envelope_is_of_the_piece():
    pcm = np.stack([np.full(100, 1.0), np.full(100, -1.0)], axis=1).astype(np.float32)
    env = waveform.peaks(pcm, columns=4)
    assert np.allclose(env, 0.0)


def test_the_tail_is_padded_rather_than_dropped():
    """A truncated tail would draw the take shorter than it is, and the playhead
    -- placed from the true duration -- would run past its own picture."""
    env = waveform.peaks(np.ones(101, dtype=np.float32), columns=10)
    assert env.shape == (2, 10)
    # The last block is part signal, part pad, so its minimum is the pad.
    assert env[1, -1] == pytest.approx(1.0)


def test_an_empty_take_is_a_flat_picture_rather_than_an_error():
    env = waveform.peaks(np.zeros(0, dtype=np.float32), columns=16)
    assert env.shape == (2, 16)


# --- window ------------------------------------------------------------------


def test_bucketing_down_is_exact():
    """The direction ``COLUMNS`` is sized to guarantee.

    A display column is a whole number of stored columns, so its extremes are
    the extremes of those -- the same answer computing from the samples would
    have given.
    """
    env = np.stack(
        [np.arange(-8, 0, dtype=np.float32), np.arange(1, 9, dtype=np.float32)]
    )
    out = waveform.window(env, 4)
    assert out[0].tolist() == [-8.0, -6.0, -4.0, -2.0]
    assert out[1].tolist() == [2.0, 4.0, 6.0, 8.0]


def test_a_wider_request_repeats_rather_than_inventing_detail():
    env = np.stack([np.array([-1.0, -2.0]), np.array([1.0, 2.0])]).astype(np.float32)
    out = waveform.window(env, 4)
    assert out.shape == (2, 4)
    assert set(out[1].tolist()) <= {1.0, 2.0}


def test_a_window_of_one_column_is_the_whole_takes_extremes():
    env = np.stack([np.array([-3.0, -1.0]), np.array([2.0, 5.0])]).astype(np.float32)
    out = waveform.window(env, 1)
    assert out[0, 0] == pytest.approx(-3.0)
    assert out[1, 0] == pytest.approx(5.0)


# --- the mapping -------------------------------------------------------------


def test_time_and_pixels_round_trip():
    """One mapping, four readings. Four copies of ``x / duration * width``
    disagree by half a column the first time one of them rounds differently."""
    for seconds in (0.0, 12.5, 60.0):
        x = waveform.at(seconds, 60.0, 800.0)
        assert waveform.seconds_at(x, 60.0, 800.0) == pytest.approx(seconds)


def test_the_mapping_clamps_rather_than_running_off_the_pane():
    assert waveform.at(-5.0, 60.0, 800.0) == 0.0
    assert waveform.at(90.0, 60.0, 800.0) == pytest.approx(800.0)
    assert waveform.seconds_at(-10.0, 60.0, 800.0) == 0.0
    assert waveform.seconds_at(9999.0, 60.0, 800.0) == pytest.approx(60.0)


def test_a_zero_length_take_maps_to_the_left_edge_rather_than_dividing_by_zero():
    assert waveform.at(3.0, 0.0, 800.0) == 0.0
    assert waveform.seconds_at(400.0, 60.0, 0.0) == 0.0

"""The WAV writer and reader.

The writer is hand-rolled so it can carry loop points, which means the RIFF
layout is this repo's problem rather than the standard library's -- so it is
read back with ``wave`` and with ``struct`` here, rather than with the reader
that shares its assumptions.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from warlock.studio.sirens import wavout


def _chunks(raw: bytes) -> dict[bytes, bytes]:
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    out: dict[bytes, bytes] = {}
    at = 12
    while at + 8 <= len(raw):
        tag = raw[at : at + 4]
        size = struct.unpack("<I", raw[at + 4 : at + 8])[0]
        out[tag] = raw[at + 8 : at + 8 + size]
        at += 8 + size + (size % 2)
    return out


def test_a_written_file_reads_back_through_the_standard_library():
    pcm = np.zeros((100, 2), dtype=np.float32)
    with wave.open(io.BytesIO(wavout.wav_bytes(pcm, 44100))) as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 44100
        assert handle.getnframes() == 100


def test_the_declared_size_matches_what_is_there():
    raw = wavout.wav_bytes(np.zeros((37, 1), dtype=np.float32), 22050)
    assert struct.unpack("<I", raw[4:8])[0] == len(raw) - 8


def test_two_writes_of_the_same_samples_are_byte_identical():
    """No timestamp, no writer string, nothing that depends on when. It is what
    makes a re-export of an unchanged document diffable."""
    pcm = np.linspace(-1, 1, 500, dtype=np.float32)
    assert wavout.wav_bytes(pcm, 44100) == wavout.wav_bytes(pcm, 44100)


def test_a_loop_writes_an_smpl_chunk_and_no_loop_writes_none():
    pcm = np.zeros((500, 2), dtype=np.float32)
    assert b"smpl" not in _chunks(wavout.wav_bytes(pcm, 44100))
    assert b"smpl" in _chunks(wavout.wav_bytes(pcm, 44100, loop=(100, 400)))


def test_the_loop_end_is_the_last_sample_inside_the_loop():
    """RIFF is inclusive here. An exclusive value leaves a one-sample gap at the
    seam, which is a click on every repeat."""
    raw = wavout.wav_bytes(np.zeros((500, 2), dtype=np.float32), 44100, loop=(100, 400))
    smpl = _chunks(raw)[b"smpl"]
    count = struct.unpack("<I", smpl[28:32])[0]
    assert count == 1
    _cue, kind, start, end, _frac, _plays = struct.unpack("<6I", smpl[36:60])
    assert kind == wavout.LOOP_FORWARD
    assert (start, end) == (100, 399)


def test_the_sample_period_describes_the_rate():
    raw = wavout.wav_bytes(np.zeros((10, 1), dtype=np.float32), 48000, loop=(0, 10))
    assert struct.unpack("<I", _chunks(raw)[b"smpl"][8:12])[0] == round(1e9 / 48000)


def test_full_scale_does_not_wrap():
    """Scaling by 32768 turns +1.0 into a value ``int16`` does not have, and the
    positive peak comes out as the negative one -- audible as a loud crack on
    exactly the loudest sample."""
    out = wavout.to_int16(np.array([-1.0, 0.0, 1.0]))
    assert list(out) == [-32767, 0, 32767]


def test_samples_outside_the_rails_are_clipped_rather_than_wrapped():
    out = wavout.to_int16(np.array([-4.0, 4.0]))
    assert list(out) == [-32767, 32767]


def test_mono_input_is_accepted_as_one_channel():
    raw = wavout.wav_bytes(np.zeros(10, dtype=np.float32), 8000)
    with wave.open(io.BytesIO(raw)) as handle:
        assert handle.getnchannels() == 1


def test_the_reader_and_the_writer_are_exact_inverses():
    """Not an aesthetic point: opening a song and saving it has to produce the
    file that was opened, and a sample that drifts by one bit per round trip
    makes every ``.wsng`` in a repository churn."""
    source = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    once = wavout.read_wav(wavout.wav_bytes(source, 44100), 44100)
    twice = wavout.read_wav(wavout.wav_bytes(once, 44100), 44100)
    assert np.array_equal(once, twice)


def test_a_stereo_sample_is_read_as_mono():
    """A chip voice has one waveform; two played through it is two samples
    fighting over the same oscillator."""
    stereo = np.stack([np.full(64, 0.5), np.full(64, -0.5)], axis=1).astype(np.float32)
    mono = wavout.read_wav(wavout.wav_bytes(stereo, 44100), 44100)
    assert mono.shape == (64,)
    assert np.allclose(mono, 0.0, atol=1e-4)


def test_a_sample_recorded_at_another_rate_is_resampled():
    """The synth advances a sample's phase in output samples, so a 22 kHz source
    left alone would play an octave out."""
    source = np.zeros(1000, dtype=np.float32)
    out = wavout.read_wav(wavout.wav_bytes(source, 22050), 44100)
    assert out.size == 2000


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_every_width_this_build_claims_to_read_is_readable(width):
    """Built by hand rather than through the writer, which only emits 16-bit."""
    frames = 32
    raw = (b"\x80" if width == 1 else b"\x00") * (frames * width)
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 8000 * width, width, width * 8)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(raw)) + raw
    wav = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body
    out = wavout.read_wav(wav, 8000)
    assert out.size == frames
    assert np.allclose(out, 0.0, atol=1e-2)


def test_an_unreadable_width_is_refused_by_name():
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 8000 * 8, 8, 64)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", 8) + b"\x00" * 8
    wav = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body
    with pytest.raises(ValueError, match="64-bit"):
        wavout.read_wav(wav, 8000)


def test_something_that_is_not_a_wav_is_refused():
    with pytest.raises(ValueError, match="not a WAV"):
        wavout.read_wav(b"this is a PNG, honestly", 44100)


def test_a_sample_past_the_ceiling_is_refused_before_it_is_decoded(monkeypatch):
    monkeypatch.setattr(wavout, "MAX_SAMPLE_FRAMES", 10)
    raw = wavout.wav_bytes(np.zeros(100, dtype=np.float32), 44100)
    with pytest.raises(ValueError, match="past the"):
        wavout.read_wav(raw, 44100)


def test_writing_to_a_path_produces_the_same_bytes(tmp_path):
    pcm = np.zeros((16, 2), dtype=np.float32)
    target = tmp_path / "a.wav"
    wavout.write(target, pcm, 44100, loop=(0, 16))
    assert target.read_bytes() == wavout.wav_bytes(pcm, 44100, loop=(0, 16))

"""The format a take is written in, and the reader that has to open it.

Muse wrote 48 kHz IEEE-float WAVs for its whole life, and *nothing caught it*:
``tests/test_muse_mode.py`` stubs ``import_sample`` out, so the bridge to Sirens
-- the headline pairing of the two audio modes -- was asserted against a mock
while the real one raised ``unknown format: 3`` on every take. This file is what
stops that coming back, and it is deliberately three cheap tests rather than one
expensive one: none of them needs weights, a card, or the ``music`` extra for
the scan.

The vendored change they pin is ``WARLOCK 5/5``; see
``pipelines/acestep/ATTRIBUTION.md`` for the argument, including why 16-bit
rather than a float branch in the tracker's reader.
"""

from __future__ import annotations

import io
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from warlock.studio.sirens import wavout

_ACESTEP = Path(__file__).resolve().parents[1] / "src" / "warlock" / "pipelines" / "acestep"
_PIPELINE = _ACESTEP / "pipeline_ace_step.py"
_ATTRIBUTION = _ACESTEP / "ATTRIBUTION.md"


def test_the_two_call_sites_carry_the_format_kwargs():
    """A source scan, the ``winjob.assign`` precedent.

    The defect is two absent keyword arguments, so the regression is two absent
    keyword arguments -- and a re-vendoring drops them silently, because it is a
    file copy over the top rather than a merge. Scanning the source is the only
    check that survives someone re-pinning the model without reading
    ATTRIBUTION.md's "Updating" section.
    """
    source = _PIPELINE.read_text(encoding="utf-8")

    save = re.search(r"torchaudio\.save\((.*?)\n        \)", source, re.S)
    assert save is not None, "torchaudio.save call not found -- was the vendor copy reshaped?"
    assert 'encoding="PCM_S"' in save.group(1)
    assert "bits_per_sample=16" in save.group(1)

    call = re.search(r"self\.latents2audio\((.*?)\n        \)", source, re.S)
    assert call is not None, "the latents2audio call site was not found"
    assert "sample_rate=44100" in call.group(1)


def test_the_marker_count_matches_the_attribution_document():
    """``WARLOCK n/N`` and the document's numbered list must agree.

    They did not: the document said "Four ... `WARLOCK n/4`" while the source
    comments said ``n/3``, and the "Updating" section said "the three
    modifications". Drift in a file whose entire job is telling a future
    re-vendorer what to re-apply is the drift that costs a feature, so it is
    pinned rather than proofread.
    """
    sources = [_PIPELINE.read_text(encoding="utf-8")]
    sources.append((_ACESTEP / "__init__.py").read_text(encoding="utf-8"))
    markers = set()
    total = set()
    for text in sources:
        for n, of in re.findall(r"WARLOCK (\d+)/(\d+):", text):
            markers.add(int(n))
            total.add(int(of))

    assert len(total) == 1, f"the markers disagree on how many there are: {sorted(total)}"
    count = total.pop()
    assert markers == set(range(1, count + 1)), (
        f"markers {sorted(markers)} do not number 1..{count} -- one was added or"
        " removed without renumbering the rest"
    )

    doc = _ATTRIBUTION.read_text(encoding="utf-8")
    assert f"`WARLOCK n/{count}:`" in doc
    entries = re.findall(r"^(\d+)\. \*\*", doc, re.M)
    assert [int(e) for e in entries] == list(range(1, count + 1)), (
        f"ATTRIBUTION.md lists {entries} modifications but the source carries {count}"
    )


def _saved_bytes(tmp_path: Path) -> bytes:
    """Write 0.1 s of tone through the vendored writer. -> the file's bytes.

    ``save_wav_file`` reads no model state -- it is a path calculation and a
    ``torchaudio.save`` -- so it is called unbound on ``None`` rather than
    through a pipeline nobody wants to construct for a format assertion.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    module = pytest.importorskip("warlock.pipelines.acestep.pipeline_ace_step")

    t = torch.linspace(0.0, 0.1, 4410)
    wave_ = torch.stack([torch.sin(t * 440.0), torch.sin(t * 660.0)]) * 0.5
    out = tmp_path / "track.wav"
    module.ACEStepPipeline.save_wav_file(
        None, wave_, 0, save_path=str(out), sample_rate=44100
    )
    return out.read_bytes()


def test_a_written_take_is_44100_hz_16_bit_stereo(tmp_path):
    with wave.open(io.BytesIO(_saved_bytes(tmp_path))) as handle:
        assert handle.getframerate() == 44100
        assert handle.getsampwidth() == 2
        assert handle.getnchannels() == 2


def test_the_tracker_can_read_a_written_take(tmp_path):
    """The assertion "Open in Sirens" needs and nothing else made.

    ``wavout.read_wav`` is the reader behind ``sirens_io.import_sample``, which
    is what the button calls. Feeding it the writer's own bytes is the whole
    bridge, minus the file dialog.
    """
    mono = wavout.read_wav(_saved_bytes(tmp_path), 44100)
    assert mono.dtype == np.float32
    # The render rate asked for is the take's own, so no resampling happens and
    # the frame count survives -- which is the second half of the fix: at the
    # old 48 kHz this would have been a resample even had the width been right.
    assert len(mono) == 4410
    assert np.abs(mono).max() <= 1.0

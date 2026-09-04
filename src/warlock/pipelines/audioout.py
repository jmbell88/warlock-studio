"""Re-encoding a finished take into the formats an engine actually imports.

**No new dependency, which is the opposite of the obvious assumption.** The
``soundfile`` already in the core dependencies is libsndfile 1.2.2, which
encodes WAV, FLAC, MP3 (LAME) and OGG (Vorbis) on its own. No ffmpeg, no
``lameenc``, no second binary to ship or to sign. It is worth saying plainly,
because every plan for this reaches for a converter first.

**Deliberately not ``pipeline_ace_step.save_wav_file``**, which dispatches on a
``format`` string and would look like the reuse. It selects ``backend="sox"``
for ogg, and ``torchaudio.list_audio_backends()`` here returns ``['soundfile']``
alone -- sox is not built on Windows. So the worker keeps ``format="wav"`` and
every other format is a *derived artifact* of ``track.wav``, which is also the
shape ``files.MEDIA`` requires: that allowlist is literal filenames, so a
per-job format would be a per-job artifact name, which is exactly what an
allowlist exists to prevent.

**No staleness rule, and that is stated rather than omitted.** ``input.png`` has
three writers, which is ``files.fresh_2d``'s whole reason for existing;
``track.wav`` has one and nothing rewrites it after the run. So existence is the
freshness test here, as it is for the mesh exports.

No torch: this module is imported in the app process, on a path that has no
reason to pay for it.
"""

from __future__ import annotations

from pathlib import Path

#: Which libsndfile format and subtype each derived name is written as.
#:
#: The subtypes are chosen, not defaulted. FLAC at ``PCM_16`` matches what
#: ``WARLOCK 5/5`` writes, so a FLAC of a take is *lossless with respect to the
#: file it came from* rather than lossless with respect to a re-quantisation.
#: MP3 and Vorbis take libsndfile's own VBR default, and there is deliberately
#: no bitrate knob: one would cost a Config field, a SETTINGS row and the
#: bidirectional test that pairs them, and no measurement says the default is
#: insufficient for what these are for.
FORMATS: dict[str, tuple[str, str]] = {
    "track.flac": ("FLAC", "PCM_16"),
    "track.mp3": ("MP3", "MPEG_LAYER_III"),
    "track.ogg": ("OGG", "VORBIS"),
}


def convert(source: Path, out: Path, name: str) -> None:
    """Re-encode ``source`` into ``name``'s format, writing to ``out``. Blocking.

    ``name`` is passed rather than read off ``out``, and that is not redundancy:
    every derivation here is staged through ``.{name}.tmp``, so the path being
    written to is called ``.track.flac.tmp`` and has no format in its suffix at
    all. Dispatching on the artifact name keeps the choice on the allowlisted
    string rather than on a filename this function was handed.

    Read and written at the file's own rate and channel count: this is a format
    change and nothing else, so resampling or downmixing here would be a second,
    undeclared transformation riding along inside an export.
    """
    import soundfile as sf

    if name not in FORMATS:
        raise ValueError(f"{name} is not a format this build writes")
    fmt, subtype = FORMATS[name]
    data, rate = sf.read(str(source), dtype="float32", always_2d=True)
    sf.write(str(out), data, int(rate), format=fmt, subtype=subtype)


__all__ = ["FORMATS", "convert"]

"""``.wsng`` -- the song document on disk, as a zip.

``song.json`` plus one ``patterns/<n>.npy`` per pattern and one
``samples/<n>.wav`` per sample, epoch-stamped throughout, so two saves of an
unchanged document are byte-identical. That is the ``.wblk``/``.wmap``/``.wpack``
rule applied a fourth time and for the fourth reason: a file that changes every
time it is written is undiffable and its content hash is worthless.

**Members are numbered, not named after their contents.** A pattern is stored at
``patterns/3.npy`` and the manifest says which uid that is; a sample is at
``samples/1.wav`` and the manifest carries its key. Naming a member after a
user-supplied string is how ``../`` and a Windows reserved name get into an
archive, and the indirection costs one line in each direction.

**The rendered audio is not in here.** A ``.wsng`` is the composition; every WAV
is a pure function of it (``docs/INVARIANTS.md``). Storing a render would make
the file able to disagree with the notes beside it, and exporting is a separate
act with its own destination.

Read through :mod:`..zipguard` and :mod:`..npyguard`. The first bounds what the
archive may unpack to; the second bounds what a ``.npy`` header may *ask for*,
which the archive's own directory cannot see -- a 128-byte member declaring a
terabyte-shaped array is honest about every byte of itself.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from .. import npyguard, zipguard
from . import document as D
from . import instruments as inst
from . import notes, synth, wavout

VERSION = 1
MANIFEST = "song.json"
PATTERN_DIR = "patterns"
SAMPLE_DIR = "samples"
SUFFIX = ".wsng"

_EPOCH = (1980, 1, 1, 0, 0, 0)

#: ``packwright/wpack.py``'s constant verbatim, and read from module globals at
#: call time for its reason too: a test lowers it rather than building a
#: gigabyte. A song of two hundred patterns and a full sample table is a few
#: megabytes, so this is only ever reached by a file we did not write.
MAX_DECOMPRESSED_BYTES = 1 << 30

_NOT_A_SONG = "this is not a Warlock song"
_MALFORMED = "this song's manifest is malformed"


def _sequence_json(sequence: inst.Sequence) -> dict[str, Any]:
    """Written only when it has something to say.

    An instrument's four sequences are usually two empty ones, and an empty
    sequence that writes ``{"values": [], "loop": -1, "release": -1}`` is three
    keys of noise in every instrument of every file. The reader is ``.get``
    based, so absent and empty are the same thing to it.
    """
    out: dict[str, Any] = {"values": [int(v) for v in sequence.values]}
    if sequence.loop >= 0:
        out["loop"] = int(sequence.loop)
    if sequence.release >= 0:
        out["release"] = int(sequence.release)
    return out


def _sequence_from(raw: Any) -> inst.Sequence:
    if not isinstance(raw, dict):
        return inst.Sequence()
    values = raw.get("values", [])
    if not isinstance(values, list):
        raise ValueError(_MALFORMED)
    if len(values) > inst.MAX_SEQUENCE_LEN:
        raise ValueError(
            f"a sequence of {len(values)} steps is past the"
            f" {inst.MAX_SEQUENCE_LEN} this build ticks"
        )
    try:
        return inst.Sequence(
            values=tuple(int(v) for v in values),
            loop=int(raw.get("loop", -1)),
            release=int(raw.get("release", -1)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(_MALFORMED) from exc


def manifest_json(doc: D.SongDoc) -> str:
    payload = {
        "version": VERSION,
        "title": doc.title,
        "author": doc.author,
        "tempo": int(doc.tempo),
        "speed": int(doc.speed),
        "loop_order": int(doc.loop_order),
        "channels": [
            {"uid": one.uid, "name": one.name, "kind": one.kind, "pan": round(float(one.pan), 4)}
            for one in doc.channels
        ],
        "instruments": [
            {
                "uid": one.uid,
                "name": one.name,
                "kind": one.kind,
                **({"sample": one.sample} if one.sample else {}),
                "volume": _sequence_json(one.volume),
                "arpeggio": _sequence_json(one.arpeggio),
                "pitch": _sequence_json(one.pitch),
                "duty": _sequence_json(one.duty),
            }
            for one in doc.instruments
        ],
        "patterns": [
            {"uid": one.uid, "name": one.name, "rows": one.rows, "member": index}
            for index, one in enumerate(doc.patterns)
        ],
        "order": [int(one) for one in doc.order],
        "oneshots": [
            {
                "uid": one.uid,
                "name": one.name,
                "pattern": int(one.pattern),
                "tempo": int(one.tempo),
                "speed": int(one.speed),
            }
            for one in doc.oneshots
        ],
        "samples": [
            {"key": key, "member": index}
            for index, key in enumerate(sorted(doc.samples))
        ],
    }
    # ``sort_keys`` is off and the dict order above is the file's order, so the
    # manifest reads top-down the way the document does. Determinism comes from
    # the sample table being sorted, which is the one collection here whose
    # Python order is not the user's.
    return json.dumps(payload, indent=1)


def wsng_bytes(doc: D.SongDoc) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(MANIFEST, _EPOCH), manifest_json(doc))
        for index, pattern in enumerate(doc.patterns):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(pattern.cells, dtype=np.int16), version=(1, 0)
            )
            zf.writestr(zipfile.ZipInfo(f"{PATTERN_DIR}/{index}.npy", _EPOCH), buffer.getvalue())
        for index, key in enumerate(sorted(doc.samples)):
            zf.writestr(
                zipfile.ZipInfo(f"{SAMPLE_DIR}/{index}.wav", _EPOCH),
                wavout.wav_bytes(doc.samples[key], synth.SAMPLE_RATE),
            )
    return out.getvalue()


def read_wsng(data: bytes) -> D.SongDoc:
    """A ``.wsng``'s bytes back into a :class:`~.document.SongDoc`.

    Restored by construction, so the document reads clean: a file that has just
    been opened is not unsaved.
    """
    try:
        zf = zipguard.BoundedZip(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError(_NOT_A_SONG) from exc

    # Everything that can raise is inside the ``with``, the version check and
    # the manifest parse included -- the ``read_wpack`` shape, and for its
    # reason: a refusal in the gap between the open and the ``with`` leaves the
    # archive open with nothing but the collector to close it.
    with zf:
        claimed = sum(int(info.file_size) for info in zf.infolist())
        if claimed > MAX_DECOMPRESSED_BYTES:
            raise ValueError(
                f"this song claims {claimed} bytes unpacked, past the"
                f" {MAX_DECOMPRESSED_BYTES} this build will read"
            )
        try:
            manifest = json.loads(zf.read(MANIFEST))
        except (
            zipfile.BadZipFile,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(_NOT_A_SONG) from exc
        if not isinstance(manifest, dict):
            raise ValueError(_MALFORMED)
        try:
            version = int(manifest.get("version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(_MALFORMED) from exc
        if version > VERSION:
            raise ValueError(
                f"this song was written by a newer version of Warlock"
                f" (format {version}, this build reads {VERSION})"
            )

        channels = _channels_from(manifest)
        instruments = _instruments_from(manifest)
        patterns = _patterns_from(zf, manifest, len(channels))
        samples = _samples_from(zf, manifest)
        known = {one.uid for one in patterns}
        order = [int(one) for one in _list(manifest, "order")][: D.MAX_ORDER]
        # An order entry naming a pattern the file does not contain is dropped
        # rather than refused: the rest of the song is intact and readable, and
        # refusing the whole document over one stale number would lose it.
        order = [one for one in order if one in known]
        oneshots = [one for one in _oneshots_from(manifest) if one.pattern in known]

        doc = D.SongDoc(
            channels=channels,
            instruments=instruments,
            patterns=patterns,
            order=order,
            oneshots=oneshots,
            samples=samples,
            title=str(manifest.get("title", "")),
            author=str(manifest.get("author", "")),
            tempo=_int(manifest, "tempo", D.DEFAULT_TEMPO),
            speed=_int(manifest, "speed", D.DEFAULT_SPEED),
            loop_order=_int(manifest, "loop_order", -1),
        )
    if doc.loop_order >= len(doc.order):
        doc.loop_order = -1
    # See ``document.reserve_uid``: without this, the first pattern added after
    # an open collides with one the file already used.
    highest = max(
        [0]
        + [one.uid for one in doc.channels]
        + [one.uid for one in doc.instruments]
        + [one.uid for one in doc.patterns]
        + [one.uid for one in doc.oneshots]
    )
    D.reserve_uid(highest)
    return doc


def _list(manifest: dict, key: str) -> list:
    value = manifest.get(key, [])
    if not isinstance(value, list):
        raise ValueError(_MALFORMED)
    return value


def _int(manifest: dict, key: str, default: int) -> int:
    try:
        return int(manifest.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(_MALFORMED) from exc


def _channels_from(manifest: dict) -> list[D.Channel]:
    out: list[D.Channel] = []
    for entry in _list(manifest, "channels")[: D.MAX_CHANNELS]:
        if not isinstance(entry, dict):
            raise ValueError(_MALFORMED)
        kind = str(entry.get("kind", "pulse"))
        if kind not in inst.KINDS:
            raise ValueError(f"this song has a {kind!r} channel, which this build cannot play")
        out.append(
            D.Channel(
                uid=int(entry.get("uid", D.new_uid())),
                name=str(entry.get("name", "")),
                kind=kind,
                pan=float(entry.get("pan", 0.0)),
            )
        )
    # A song with no channels has no voices and no pattern shape, so there is
    # nothing to open. The five defaults are a better answer than a refusal for
    # a file whose channel list was truncated.
    return out or D.default_channels()


def _instruments_from(manifest: dict) -> list[inst.Instrument]:
    out: list[inst.Instrument] = []
    for entry in _list(manifest, "instruments")[: D.MAX_INSTRUMENTS]:
        if not isinstance(entry, dict):
            raise ValueError(_MALFORMED)
        kind = str(entry.get("kind", "pulse"))
        if kind not in inst.KINDS:
            raise ValueError(f"this song has a {kind!r} instrument, which this build cannot play")
        out.append(
            inst.Instrument(
                uid=int(entry.get("uid", D.new_uid())),
                name=str(entry.get("name", "")),
                kind=kind,
                sample=str(entry.get("sample", "")),
                volume=_sequence_from(entry.get("volume")),
                arpeggio=_sequence_from(entry.get("arpeggio")),
                pitch=_sequence_from(entry.get("pitch")),
                duty=_sequence_from(entry.get("duty")),
            )
        )
    return out


def _patterns_from(zf: Any, manifest: dict, channels: int) -> list[D.Pattern]:
    out: list[D.Pattern] = []
    for entry in _list(manifest, "patterns")[: D.MAX_PATTERNS]:
        if not isinstance(entry, dict):
            raise ValueError(_MALFORMED)
        member = f"{PATTERN_DIR}/{int(entry.get('member', len(out)))}.npy"
        try:
            raw = zf.read(member)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"this song is missing {member}") from exc
        cells = npyguard.read_array(raw, member)
        rows = _int(entry, "rows", 0) or int(cells.shape[0] if cells.ndim else 0)
        if cells.ndim != 3 or cells.shape != (rows, channels, D.COLUMNS):
            raise ValueError(
                f"{member} is {cells.shape}, and this song's patterns are"
                f" ({rows}, {channels}, {D.COLUMNS})"
            )
        if not D.MIN_ROWS <= rows <= D.MAX_ROWS:
            raise ValueError(f"{member} has {rows} rows, which this build does not play")
        # Clipped rather than trusted: every column has a range, and a
        # hand-edited array can hold anything an ``int16`` can. A note of 9000
        # would index past the end of the frequency table at render time.
        grid = np.ascontiguousarray(cells, dtype=np.int16).copy()
        np.clip(grid, notes.EMPTY, 255, out=grid)
        out.append(
            D.Pattern(
                uid=int(entry.get("uid", D.new_uid())),
                name=str(entry.get("name", "")),
                cells=grid,
            )
        )
    return out


def _oneshots_from(manifest: dict) -> list[D.OneShot]:
    out: list[D.OneShot] = []
    for entry in _list(manifest, "oneshots")[: D.MAX_ONESHOTS]:
        if not isinstance(entry, dict):
            raise ValueError(_MALFORMED)
        out.append(
            D.OneShot(
                uid=int(entry.get("uid", D.new_uid())),
                name=str(entry.get("name", "")),
                pattern=_int(entry, "pattern", 0),
                tempo=_int(entry, "tempo", D.DEFAULT_TEMPO),
                speed=_int(entry, "speed", D.DEFAULT_SPEED),
            )
        )
    return out


def _samples_from(zf: Any, manifest: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for index, entry in enumerate(_list(manifest, "samples")[: D.MAX_SAMPLES]):
        if not isinstance(entry, dict):
            raise ValueError(_MALFORMED)
        member = f"{SAMPLE_DIR}/{int(entry.get('member', index))}.wav"
        try:
            raw = zf.read(member)
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"this song is missing {member}") from exc
        out[str(entry.get("key", f"sample{index}"))] = wavout.read_wav(raw, synth.SAMPLE_RATE)
    return out

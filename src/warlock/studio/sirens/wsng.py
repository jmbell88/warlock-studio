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


#: ``id(array) -> (array, wav)``. The array is held so its id cannot be
#: recycled under the entry. Bounded at twice the sample ceiling: a document's
#: worth plus the one it replaced.
_WAV_CACHE: dict[int, tuple[np.ndarray, bytes]] = {}
_WAV_CACHE_MAX = 2 * D.MAX_SAMPLES


def _wav_of(pcm: np.ndarray) -> bytes:
    """A sample's WAV bytes, encoded once per array.

    ``wsng_bytes`` is the render snapshot *and* the journal encode, both on the
    frame thread, and both fire on every accepted edit -- so every sample was
    re-encoded to int16 WAV on every keystroke that reached the synthesiser,
    which at eight seconds of 44.1 kHz per sample is the longest thing the
    mode did per frame. Samples are replaced, never written in place
    (``SongDoc.set_sample`` stores a fresh contiguous copy), so identity is
    the cache key. Keyed on the object, not its bytes: hashing the array
    would cost what the encode does.
    """
    hit = _WAV_CACHE.get(id(pcm))
    if hit is not None and hit[0] is pcm:
        return hit[1]
    data = wavout.wav_bytes(pcm, synth.SAMPLE_RATE)
    if len(_WAV_CACHE) >= _WAV_CACHE_MAX:
        _WAV_CACHE.pop(next(iter(_WAV_CACHE)))
    _WAV_CACHE[id(pcm)] = (pcm, data)
    return data


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
                _wav_of(doc.samples[key]),
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
        instruments, remap = _instruments_from(manifest)
        patterns = _patterns_from(zf, manifest, len(channels), remap)
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
    #
    # **The instruments are deliberately not in this maximum.** Their ids are a
    # per-document space bounded by ``MAX_INSTRUMENTS`` and have nothing to do
    # with the global counter (``document``'s docstring); reserving above one
    # would walk the counter toward its own ceiling on behalf of a number that
    # never came out of it.
    highest = max(
        [0]
        + [one.uid for one in doc.channels]
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


def _instruments_from(manifest: dict) -> tuple[list[inst.Instrument], dict[int, int]]:
    """The instrument list, and the renumbering the file needed -- usually none.

    An instrument id is bounded by ``document.MAX_INSTRUMENTS`` and unique
    within the document (see ``document._free_instrument_id``), because it is
    what an ``int16`` cell holds. A file can say otherwise: one written by an
    earlier build minted instrument ids from the process-global counter, and a
    hand-edited manifest can say anything at all.

    Such a file is **renumbered rather than refused**, and the pattern cells are
    carried through the same map (:func:`_patterns_from`) so the song still
    plays the instruments it named. The list is already capped at
    ``MAX_INSTRUMENTS``, so numbering by position always fits.

    The map is empty when every id was already legal and distinct, which is the
    only case a file this build wrote can be in -- so a save/open/save round
    trip stays byte-identical and a ``.wsng`` in a repository stays diffable.
    """
    entries = _list(manifest, "instruments")[: D.MAX_INSTRUMENTS]
    stored: list[int] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(_MALFORMED)
        try:
            stored.append(int(entry.get("uid", index)))
        except (TypeError, ValueError) as exc:
            raise ValueError(_MALFORMED) from exc
    legal = all(0 <= uid < D.MAX_INSTRUMENTS for uid in stored) and len(set(stored)) == len(stored)
    remap: dict[int, int] = {} if legal else {uid: i for i, uid in enumerate(stored)}

    out: list[inst.Instrument] = []
    for index, entry in enumerate(entries):
        kind = str(entry.get("kind", "pulse"))
        if kind not in inst.KINDS:
            raise ValueError(f"this song has a {kind!r} instrument, which this build cannot play")
        out.append(
            inst.Instrument(
                uid=stored[index] if legal else index,
                name=str(entry.get("name", "")),
                kind=kind,
                sample=str(entry.get("sample", "")),
                volume=_sequence_from(entry.get("volume")),
                arpeggio=_sequence_from(entry.get("arpeggio")),
                pitch=_sequence_from(entry.get("pitch")),
                duty=_sequence_from(entry.get("duty")),
            )
        )
    return out, remap


def _patterns_from(
    zf: Any, manifest: dict, channels: int, remap: dict[int, int] | None = None
) -> list[D.Pattern]:
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
        #
        # One basic-index slice per column, deliberately: ``grid[:, :, [a, b]]``
        # is advanced indexing and hands back a *copy*, so a clip written
        # through it lands nowhere and every bound here would be decoration.
        for column in (D.NOTE, D.VOLUME, D.EFFECT, D.PARAM):
            plane = grid[:, :, column]
            np.clip(plane, notes.EMPTY, 255, out=plane)
        # **The instrument column's range is the instrument id space**, which
        # is ``0..MAX_INSTRUMENTS-1`` and per document (``document``'s
        # docstring). The 255 this used to be was nearly that number and not
        # quite; the int16 ceiling that replaced it was right only while ids
        # came from the process-global counter, which is the ceiling this
        # phase removed. A file whose ids needed renumbering is carried
        # through the same map its instrument list was, so the cells still
        # name the instruments they named -- and a value that matched *no*
        # instrument in that file is blanked rather than left to land on
        # whichever slot the renumbering has since put in its place.
        plane = grid[:, :, D.INSTRUMENT]
        if remap:
            stored = plane.copy()
            plane[:] = notes.EMPTY
            for old, new in remap.items():
                plane[stored == old] = new
        np.clip(plane, notes.EMPTY, D.MAX_INSTRUMENTS - 1, out=plane)
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
        key = str(entry.get("key", f"sample{index}"))
        if key in out:
            # Refused by name rather than collapsed. An instrument names its
            # sample by this string, so two manifest entries claiming one key
            # is a file that contradicts itself -- and the silent answer (the
            # last one wins) is a song that opens and plays the wrong sound on
            # whichever instrument lost, with nothing anywhere saying so.
            raise ValueError(f"this song lists the sample {key!r} twice")
        out[key] = wavout.read_wav(raw, synth.SAMPLE_RATE)
    return out

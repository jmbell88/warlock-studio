"""The ``.aseprite`` round-trip corpus gate.

``tests/inker/fixtures/aseprite/`` is a *committed* corpus in the shape
``tests/plotter/fixtures/tiled/`` set the precedent for: real files on disk,
a manifest naming the stems the gate requires, and a ``FIXTURES.md`` carrying
the provenance of each one. What differs from ``test_aseout.py`` -- which
already exercises this same writer/reader pair with documents built fresh in
Python every run -- is durability: a committed file is inspectable, survives
a refactor of the builders that produced it, and is where an Aseprite-authored
fixture will eventually replace a synthesized one without the gate changing
shape (see ``FIXTURES.md``'s "What is owed").

**The gate, precisely.** ``aseprite_bytes`` normalises some documents on their
first trip through it -- a palette-constrained RGB document is the one this
corpus carries (``ASEPRITE_PARITY.md`` divergence 19) -- so a fixture's
committed bytes are not asserted to equal some builder's raw output; they are
asserted to be a *fixed point*: read them, write what was read, and get the
same bytes back. That is the write-read-write property ``test_aseout.py``'s
``test_a_re_imported_document_writes_the_same_bytes_twice`` checks for one
document; this file checks it for every committed one, plus a second
independent read to confirm the model the fixed point encodes is stable too.

No committed golden of the *model* exists beside the bytes -- deep-comparing
two reads of the same bytes is a weaker claim than comparing against an
originally-built document, but the stronger claim (bytes equal a specific
builder's output) is exactly what a checked-in generator would have to pin
and immediately duplicates the fixed-point property above. See
``FIXTURES.md`` for why the files are committed rather than regenerated.
"""

from __future__ import annotations

import numpy as np
import pytest
from _asecorpus import EXPECTED_WARNINGS, FIXTURE_DIR, MANIFEST, available, read

from warlock.studio.inker import asein, aseout
from warlock.studio.inker.document import Document
from warlock.studio.inker.tiles import TilemapCel


def test_the_fixture_directory_exists():
    assert FIXTURE_DIR.is_dir()


def test_every_manifest_stem_has_a_file():
    missing = [stem for stem in MANIFEST if not (FIXTURE_DIR / f"{stem}.aseprite").is_file()]
    assert not missing, f"MANIFEST names fixtures with no file on disk: {missing}"


def test_the_directory_holds_nothing_the_manifest_does_not_list():
    """A file dropped in without a manifest entry is a file nothing tests --
    the ``_corpus.py`` rule's other direction."""
    extra = set(available()) - set(MANIFEST)
    assert not extra, f"fixtures on disk with no MANIFEST entry: {sorted(extra)}"


# --- a structural snapshot, uid-independent ----------------------------------
#
# ``uid`` is minted from a process-wide counter (``layers.py``'s ``new_uid``),
# so two independent parses of the same bytes -- which is what the fixed-point
# check below performs -- never agree on the numbers even though they agree on
# everything the numbers *name*. Every comparison here is by name or by stack
# position instead, which is what the document's own round-trip tests do
# (``back.tracks[ti]``, never a raw uid) and is the only kind of equality that
# means anything across two mints.


def _row_props(row) -> dict:
    props = {
        "name": row.name,
        "opacity": round(float(row.opacity), 4),
        "visible": row.visible,
        "locked": row.locked,
        "blend": row.blend,
        "alpha_lock": row.alpha_lock,
    }
    if hasattr(row, "continuous"):
        props["continuous"] = row.continuous
    return props


def _tileset_name(doc: Document, uid: int | None) -> str | None:
    if not uid:
        return None
    return doc.tileset_slot(uid).tileset.name


def _cel_payload(doc: Document, layer) -> dict:
    payload: dict[str, object] = {"pixels": layer.pixels.tobytes()}
    if layer.indices is not None:
        payload["indices"] = layer.indices.tobytes()
    if isinstance(layer, TilemapCel):
        payload["refs"] = layer.refs.tobytes() if layer.refs is not None else None
        payload["tileset"] = _tileset_name(doc, layer.tileset_uid)
    return payload


def _link_ids(doc: Document) -> list[list[int | None]] | None:
    """Which cels are the *same object*, as small canonical integers.

    Track-major, frame-minor first-appearance order -- deterministic given a
    fixed track/frame order, which both reads of one fixture share."""
    if doc.anim is None:
        return None
    seen: dict[int, int] = {}
    out: list[list[int | None]] = []
    for track in doc.anim.tracks:
        row: list[int | None] = []
        for frame in doc.anim.frames:
            cel = doc.anim.cels.get((track.uid, frame.uid))
            if cel is None:
                row.append(None)
            else:
                row.append(seen.setdefault(id(cel), len(seen)))
        out.append(row)
    return out


def _group_snapshot(doc: Document) -> tuple[dict, dict]:
    name_of: dict[int, str] = {}
    for row in doc.stack if doc.anim is None else doc.anim.tracks:
        name_of[row.uid] = row.name
    for uid, node in doc.groups.items():
        name_of[uid] = node.name

    nodes = {
        node.name: {
            "visible": node.visible,
            "opacity": round(float(node.opacity), 4),
            "locked": node.locked,
            "parent": name_of.get(doc.group_of.get(uid)),
        }
        for uid, node in doc.groups.items()
    }
    membership = {
        name_of[uid]: name_of.get(doc.group_of.get(uid)) for uid in doc.member_uids()
    }
    return nodes, membership


def _slice_snapshot(doc: Document) -> list[dict]:
    frame_uids: list[int | None] = [None] if doc.anim is None else [
        frame.uid for frame in doc.anim.frames
    ]
    out = []
    for entry in doc.slices:
        keys = [entry.at(uid) for uid in frame_uids]
        out.append(
            {
                "name": entry.name,
                "keys": [(k.bounds, k.pivot, k.center) for k in keys],
            }
        )
    return out


def _tileset_snapshot(doc: Document) -> list[dict]:
    return [
        {
            "name": slot.tileset.name,
            "tile_w": slot.tileset.tile_w,
            "tile_h": slot.tileset.tile_h,
            "tile_count": slot.tileset.tile_count,
            "pixels": slot.tileset.pixels.tobytes(),
        }
        for slot in doc.tilesets
    ]


def _snapshot(doc: Document) -> dict:
    out: dict[str, object] = {
        "color_mode": doc.color_mode,
        "size": doc.size,
        "palette": None if doc.palette is None else [tuple(c) for c in doc.palette],
        "transparent_index": doc.transparent_index,
        "groups": _group_snapshot(doc),
        "slices": _slice_snapshot(doc),
        "tilesets": _tileset_snapshot(doc),
    }
    if doc.anim is None:
        out["rows"] = [_row_props(layer) for layer in doc.stack]
        out["cels"] = [_cel_payload(doc, layer) for layer in doc.stack]
        out["links"] = None
    else:
        out["rows"] = [_row_props(track) for track in doc.anim.tracks]
        out["tileset_binding"] = [
            _tileset_name(doc, track.tileset_uid) for track in doc.anim.tracks
        ]
        out["durations"] = [frame.duration_ms for frame in doc.anim.frames]
        out["tags"] = [
            (tag.name, tag.start, tag.end, tag.loop, tag.direction, tag.repeat)
            for tag in doc.anim.tags
        ]
        out["links"] = _link_ids(doc)
        cels: list[dict | None] = []
        for track in doc.anim.tracks:
            for frame in doc.anim.frames:
                cel = doc.anim.cels.get((track.uid, frame.uid))
                cels.append(None if cel is None else _cel_payload(doc, cel))
        out["cels"] = cels
    return out


# --- the gate -----------------------------------------------------------------


@pytest.mark.parametrize("stem", MANIFEST)
def test_fixture_is_a_write_read_write_fixed_point(stem: str):
    data = read(stem)
    doc_a, warnings_a = asein.document_from_aseprite(data)
    rewritten = aseout.aseprite_bytes(doc_a)
    assert rewritten == data, (
        f"{stem}.aseprite is not a fixed point of the writer -- "
        "regenerate it from a doc that has already made one round trip"
    )

    doc_b, warnings_b = asein.document_from_aseprite(rewritten)
    assert _snapshot(doc_a) == _snapshot(doc_b), (
        f"{stem}: two independent reads of the same bytes disagree"
    )

    expected = list(EXPECTED_WARNINGS.get(stem, ()))
    assert warnings_a == expected, f"{stem}: unexpected reader warnings {warnings_a}"
    assert warnings_b == expected


@pytest.mark.parametrize("stem", MANIFEST)
def test_fixture_pixels_are_never_all_zero(stem: str):
    """A sanity floor on the matrix itself: a fixture whose canvas is entirely
    empty would round-trip trivially and prove nothing about the feature it
    names."""
    doc, _ = asein.document_from_aseprite(read(stem))
    rows = doc.stack if doc.anim is None else doc.anim.tracks
    assert rows, f"{stem}: no layers at all"
    any_ink = False
    layers = doc.stack if doc.anim is None else doc.anim.unique_cel_layers()
    for layer in layers:
        if np.any(layer.pixels[..., 3] > 0):
            any_ink = True
            break
    assert any_ink, f"{stem}: every cel is fully transparent"

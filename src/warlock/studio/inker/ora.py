"""OpenRaster: a zip of layer PNGs and a stack.xml describing them.

The native format is ORA rather than something of our own for one reason --
a layered document that only this app can open is a document the user cannot
get out. ORA is a handful of stdlib calls (``zipfile`` plus Pillow), and Krita
and GIMP both read and write it.

The writer follows the spec's fiddly parts because readers depend on them: the
``mimetype`` entry is first and stored uncompressed (it is a magic number, read
at a fixed offset), the stack is listed *top layer first*, and ``mergedimage.png``
is required -- a viewer that does not composite shows that and nothing else.

The reader is deliberately tolerant. A composite-op we cannot reproduce becomes
normal, a layer with an offset is pasted at it rather than refused, and a nested
stack is flattened into the list. An unreadable file is a bug report; a file
that opens slightly wrong is a file the user still has.

An **animated** document adds a second member, ``animation.json``, and that one
is authoritative: it carries the grid -- durations, track properties, which
slots hold which cel, and which slots share one -- while ``stack.xml`` becomes
an interop *projection* of it, one nested ``<stack>`` group per frame. The two
cannot be derived from each other (XML has nowhere to put a duration and no way
to say "these two entries are the same cel, not two equal ones"), so rather than
split the truth between them, one is the record and the other is the picture
foreign editors get. Frames after the first are written hidden, on the group and
on every layer inside it, so an editor that flattens groups -- Krita, GIMP, and
this reader's own fallback -- shows frame 1 rather than the whole clip stacked
on top of itself.

The JSON stores **indices, not uids**: uids are minted per process and mean
nothing in a file. And the reader treats any way of ``animation.json`` not
making sense -- a wrong version, an index outside the grid it declares, a cel
naming a PNG the archive does not hold -- as a reason to fall back to the flat
read rather than to raise, which is the same bargain the rest of this module
makes: a file that opens as a still image is a file the user still has. What it
deliberately does *not* do is cross-check the JSON against ``stack.xml``. The
XML is the projection and the JSON is the record, so a disagreement between them
is the XML being wrong about a document the JSON describes correctly, and
throwing the grid away over it would lose the animation to fix the picture of
it.

A third member, ``warlock.json``, carries what this app knows about a document
that ORA has nowhere for and that is not part of the grid -- today, its slices.
It is written **only when there is something to write**, so an archive from a
document with no slices is byte-for-byte what this writer produced before the
member existed. It is separate from ``animation.json`` because slices live on
still documents too, and because that member fails whole-grid on purpose while a
malformed slice must never cost a document its timeline.

The accepted cost, stated because it is not recoverable: an older build of this
app opens an animated file (seeing frame 1) and saving it writes it back flat.
That is inherent to forward compatibility with a format that has no version
gate; the hidden groups limit the damage to what is displayed, not to what a
foreign save discards.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from . import composite as cp
from . import gpl
from .animation import (
    DEFAULT_DURATION_MS,
    Animation,
    DirectionalLayout,
    Frame,
    Tag,
    Track,
)
from .layers import Layer, LayerStack

THUMBNAIL_MAX = 256

#: Bumped when a reader written against version N could get an N+1 file wrong in
#: a way it cannot detect. Adding a key does not qualify -- every read below is
#: ``.get``-based -- so this has stayed at 1 through the whole of v1.
ANIMATION_VERSION = 1
ANIMATION_MEMBER = "animation.json"

#: An indexed document's colour table, written as a plain ``.gpl``. Additive
#: and versionless: a reader that does not know about it opens the file as an
#: ordinary RGBA document, which is exactly what the pixels already are -- see
#: :mod:`.indexed`. ``.gpl`` rather than another JSON key because ``gpl.py``
#: already reads and writes it, because it is the one interchange format for a
#: row of swatches, and because a member of that name is a palette anyone can
#: pull out of the zip. Alpha does not survive it and does not need to: a
#: palette constrains colour and never opacity.
PALETTE_MEMBER = "palette.gpl"

#: Everything this app knows about a document that ORA has no place for and
#: that is not part of the *grid*. Today that is slices; the version gate is
#: here for the same reason ``animation.json`` has one and has stayed at 1 for
#: the same reason -- every read below is ``.get``-based, so adding a key does
#: not qualify.
#:
#: A member of its own rather than a key in ``animation.json``, and the two
#: reasons are both about failure. Slices exist on a **still** document (a
#: nine-slice button is one PNG), which has no ``animation.json`` at all; and
#: ``animation.json`` fails whole-grid by design -- half a timeline is harder to
#: notice than none of one -- where a slice going wrong must never cost the
#: document its frames.
WARLOCK_MEMBER = "warlock.json"
WARLOCK_VERSION = 1

# 1980-01-01, the earliest a zip can express, and the same constant the three
# younger formats in this repo (``.wblk``, ``.wmap``, ``.wpack``) fix their
# members at. Without it every member is stamped with the wall clock, so two
# saves of a document nobody touched produced two different files -- which makes
# a save look like a change to anything that hashes or diffs one.
#
# It is safe against a foreign reader, and the file itself is the evidence: the
# ``mimetype`` member has carried this exact stamp since this writer was written
# (it is what a bare ``ZipInfo`` defaults to) and it is the first thing every ORA
# reader touches. The OpenRaster spec says nothing whatever about modification
# times -- it specifies member *names*, the ordering of ``mimetype`` and its
# being stored uncompressed -- and Krita's and GIMP's readers are ordinary zip
# readers that never look. This is also the floor rather than an arbitrary
# choice: MS-DOS date fields cannot express anything earlier, and ``zipfile``
# raises on a date below it.
_EPOCH = (1980, 1, 1, 0, 0, 0)

log = logging.getLogger(__name__)


def _member(name: str) -> zipfile.ZipInfo:
    """A deflated archive member at the fixed epoch.

    ``writestr`` builds one of these itself when it is handed a plain name --
    with the wall clock, which is the whole problem -- and takes the compression
    and the mode off the ``ZipInfo`` when it is handed one instead. So both have
    to be restated here: a bare ``ZipInfo`` says ``ZIP_STORED``, and a writer
    that forgot this line would silently stop compressing and treble the size of
    every file it wrote.
    """
    info = zipfile.ZipInfo(name, _EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info


def _png(pixels: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def _stack_xml(doc) -> bytes:
    width, height = doc.size
    root = ElementTree.Element(
        "image", {"version": "0.0.3", "w": str(width), "h": str(height)}
    )
    stack = ElementTree.SubElement(root, "stack")
    # Top first: ORA's document order is the painter's, reversed.
    for index, layer in enumerate(reversed(list(doc.stack))):
        ElementTree.SubElement(
            stack,
            "layer",
            {
                "name": layer.name,
                "src": f"data/layer{index}.png",
                "x": "0",
                "y": "0",
                "opacity": f"{float(layer.opacity):.6f}",
                "visibility": "visible" if layer.visible else "hidden",
                "composite-op": cp.ORA_OPS.get(layer.blend, "svg:src-over"),
                **_lock_attr(layer.alpha_lock),
            },
        )
    return ElementTree.tostring(root, encoding="UTF-8", xml_declaration=True)


# Ours, and prefixed with the app name to say so. ORA readers ignore attributes
# they do not recognise -- that tolerance is what the format is built on and
# what ``_foreign`` exercises from the other side -- so a locked layer opens in
# Krita as an ordinary layer rather than as an error. It is written only when
# set, so a document nobody has locked anything in produces byte-identical XML
# to the one this build wrote before the attribute existed.
#
# A hyphen, **not** ``warlock:alpha-lock``: a colon in an attribute name is an
# XML namespace prefix, and an undeclared one makes the whole ``stack.xml``
# unparseable -- so the private attribute would have cost every reader the
# entire file, this one included.
LOCK_ATTR = "warlock-alpha-lock"


def _lock_attr(locked: bool) -> dict[str, str]:
    return {LOCK_ATTR: "1"} if locked else {}


def _cel_names(anim) -> dict[int, str]:
    """One PNG per *distinct* cel, so a linked cel is stored once.

    Keyed by ``id(layer)`` and ordered by ``unique_cel_layers``, which is
    deterministic -- so saving an unchanged document twice produces the same
    names, and a slot sharing another's ``src`` is exactly how a link survives
    into the file.
    """
    return {id(layer): f"data/cel{i}.png" for i, layer in enumerate(anim.unique_cel_layers())}


def _stack_xml_animated(doc, names: dict[int, str]) -> bytes:
    anim = doc.anim
    width, height = doc.size
    root = ElementTree.Element(
        "image", {"version": "0.0.3", "w": str(width), "h": str(height)}
    )
    outer = ElementTree.SubElement(root, "stack")
    for index, frame in enumerate(anim.frames):
        hidden = index > 0
        group = ElementTree.SubElement(
            outer,
            "stack",
            {
                "name": f"frame:{index + 1:04d}",
                "visibility": "hidden" if hidden else "visible",
            },
        )
        # Top first inside each group, as in the still writer. The properties
        # come off the *track*, which is authoritative; a cel's own copy is a
        # materialisation detail and may be stale.
        for track in reversed(anim.tracks):
            layer = anim.cels.get((track.uid, frame.uid))
            if layer is None:
                continue
            ElementTree.SubElement(
                group,
                "layer",
                {
                    "name": track.name,
                    "src": names[id(layer)],
                    "x": "0",
                    "y": "0",
                    "opacity": f"{float(track.opacity):.6f}",
                    # Hidden on every layer as well as on the group: a reader
                    # that flattens groups keeps the layers and would otherwise
                    # show every frame at once.
                    "visibility": "visible" if track.visible and not hidden else "hidden",
                    "composite-op": cp.ORA_OPS.get(track.blend, "svg:src-over"),
                    **_lock_attr(track.alpha_lock),
                },
            )
    return ElementTree.tostring(root, encoding="UTF-8", xml_declaration=True)


def _animation_json(doc, names: dict[int, str]) -> bytes:
    anim = doc.anim
    tracks = {track.uid: i for i, track in enumerate(anim.tracks)}
    frames = {frame.uid: i for i, frame in enumerate(anim.frames)}
    cels = [
        {
            "track": tracks[track_uid],
            "frame": frames[frame_uid],
            "data": names[id(layer)],
        }
        for (track_uid, frame_uid), layer in anim.cels.items()
        if track_uid in tracks and frame_uid in frames
    ]
    cels.sort(key=lambda cel: (cel["frame"], cel["track"]))
    payload = {
        "version": ANIMATION_VERSION,
        "frames": [{"duration_ms": int(frame.duration_ms)} for frame in anim.frames],
        "tracks": [track.props() for track in anim.tracks],
        "cels": cels,
        # ``direction`` is additive and the version is unchanged deliberately:
        # every reader of this section is ``.get``-based, so an older build
        # opens the file and plays every tag forwards, which is exactly what it
        # did before the field existed.
        "tags": [
            {
                "name": tag.name,
                "start": int(tag.start),
                "end": int(tag.end),
                "loop": bool(tag.loop),
                "direction": str(tag.direction),
            }
            for tag in anim.tags
        ],
    }
    if anim.layout is not None:
        # Additive and the version stays 1, for the reason ``direction`` above
        # gives: an older build reads this section with ``.get`` and simply has
        # no layout, which is what every document had before sprite sheets
        # existed. Written only when set, so a layout-less document's
        # ``animation.json`` is byte-identical to what it was.
        #
        # Only the kind, because that is all a ``DirectionalLayout`` is -- the
        # grid is derived, so there is no second number here to disagree with
        # the reader's.
        payload["layout"] = {"kind": anim.layout.kind}
    return json.dumps(payload, indent=2).encode("utf-8")


def _rect_json(rect) -> dict[str, int]:
    """A rectangle as ``{x, y, w, h}``.

    The sidecar's spelling, not this package's ``x0 y0 x1 y1``, and
    deliberately: the ``.ora`` is a file other programs open, ``{x, y, w, h}``
    is what every one of them already means by a rectangle, and the two
    conversions live in this module's two functions rather than in a reader
    somebody else writes.
    """
    x0, y0, x1, y1 = (int(v) for v in rect)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _rect_of(entry, key: str):
    """``{x, y, w, h}`` back into exclusive bounds, or None when it is absent.

    Raises for a rectangle that is present and malformed, which is what makes
    the reader's "an entry failing on its own terms drops the member" rule
    reachable: a missing key is a file written before the field, and a key
    holding a string is a file that is wrong about itself.
    """
    raw = entry.get(key)
    if raw is None:
        return None
    x, y = int(raw["x"]), int(raw["y"])
    return (x, y, x + int(raw["w"]), y + int(raw["h"]))


def _slice_json(entry, frames: dict[int, int]) -> dict:
    """One slice, with only the fields it actually carries.

    ``pivot``, ``center`` and ``keys`` are written **only when set**, which is
    what keeps this member small and, more usefully, keeps a document whose
    slices are plain rectangles producing the same bytes it did before pivots
    existed.

    Keys are stored by frame **index**, the ``cels`` precedent: a uid is minted
    per process and means nothing in a file. A key whose frame has left the grid
    is skipped rather than failing the save -- the same accepted leak
    ``_placeholder_uids`` takes, and the alternative is refusing to write a
    document over metadata for a frame that no longer exists.
    """
    out: dict = {"name": entry.name, "bounds": _rect_json(entry.bounds)}
    if entry.pivot is not None:
        out["pivot"] = {"x": float(entry.pivot[0]), "y": float(entry.pivot[1])}
    if entry.center is not None:
        out["center"] = _rect_json(entry.center)
    keys = []
    for frame_uid, key in entry.keys.items():
        index = frames.get(frame_uid)
        if index is None:
            continue
        record: dict = {"frame": index, "bounds": _rect_json(key.bounds)}
        if key.pivot is not None:
            record["pivot"] = {"x": float(key.pivot[0]), "y": float(key.pivot[1])}
        if key.center is not None:
            record["center"] = _rect_json(key.center)
        keys.append(record)
    if keys:
        # Sorted, so two saves of an unchanged document are byte-identical
        # however the dictionary happened to be built -- the same property
        # ``_animation_json`` sorts its cels for.
        keys.sort(key=lambda record: record["frame"])
        out["keys"] = keys
    return out


def _warlock_json(doc) -> bytes:
    anim = getattr(doc, "anim", None)
    frames = (
        {} if anim is None else {frame.uid: i for i, frame in enumerate(anim.frames)}
    )
    payload = {
        "version": WARLOCK_VERSION,
        "slices": [_slice_json(entry, frames) for entry in doc.slices],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def _frame_flatten(doc, frame) -> np.ndarray:
    """Frame 1's pixels, whatever the playhead is on.

    ``mergedimage.png`` has to be a function of the document and not of where
    the user happened to be looking, or saving the same file twice produces two
    different files.
    """
    layers = doc.anim.layers_for(frame, doc.size)
    return cp.flatten_onto(LayerStack(list(layers), 0).flatten(), doc.matte)


def write_ora(doc, path: Path) -> None:
    """Blocking; callers encode on a task thread."""
    from PIL import Image

    path = Path(path)
    anim = getattr(doc, "anim", None)
    names = _cel_names(anim) if anim is not None else {}
    merged = doc.flatten() if anim is None else _frame_flatten(doc, anim.frames[0])
    thumb = Image.fromarray(merged, "RGBA")
    thumb.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX))
    thumb_buf = io.BytesIO()
    thumb.save(thumb_buf, "PNG")

    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        # Stored, and first: the spec makes this a magic number at a fixed
        # offset, and a deflated one is not readable as such.
        zf.writestr(
            zipfile.ZipInfo("mimetype", _EPOCH), b"image/openraster", zipfile.ZIP_STORED
        )
        if anim is None:
            zf.writestr(_member("stack.xml"), _stack_xml(doc))
            for index, layer in enumerate(reversed(list(doc.stack))):
                zf.writestr(_member(f"data/layer{index}.png"), _png(layer.pixels))
        else:
            zf.writestr(_member("stack.xml"), _stack_xml_animated(doc, names))
            # One PNG per name, with no de-duplication needed: ``_cel_names``
            # is built from the same ``unique_cel_layers`` walk and gives each
            # distinct cel its own name, so the two can only ever agree.
            for layer in anim.unique_cel_layers():
                zf.writestr(_member(names[id(layer)]), _png(layer.pixels))
            zf.writestr(_member(ANIMATION_MEMBER), _animation_json(doc, names))
        if getattr(doc, "palette", None):
            zf.writestr(_member(PALETTE_MEMBER), gpl.dumps(doc.palette).encode("utf-8"))
        # Only when there are slices. A document with none produces an archive
        # byte-identical to the one this build wrote before the member existed,
        # which is what the determinism suite pins and what makes this addition
        # invisible to every reader that has never heard of it.
        if getattr(doc, "slices", None):
            zf.writestr(_member(WARLOCK_MEMBER), _warlock_json(doc))
        zf.writestr(_member("mergedimage.png"), _png(merged))
        zf.writestr(_member("Thumbnails/thumbnail.png"), thumb_buf.getvalue())
    tmp.replace(path)


def ora_bytes(doc) -> bytes:
    """The same file, in memory -- for a save that goes through a service."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.ora"
        write_ora(doc, path)
        return path.read_bytes()


# --- reading ----------------------------------------------------------------


def _layer_elements(node) -> list:
    """Depth-first, flattening nested stacks -- we have no group layers, and
    dropping a group's contents would silently lose most of a Krita file."""
    found = []
    for child in node:
        if child.tag == "layer":
            found.append(child)
        elif child.tag == "stack":
            found.extend(_layer_elements(child))
    return found


def _place(pixels: np.ndarray, size: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    """Paste a layer onto a canvas-sized plane at its ORA offset.

    Offsets exist on disk and not in memory: every op in this app is a plain
    slice, and the price of that is doing the placement once, here.
    """
    from .transform import resize_canvas

    return resize_canvas(pixels, size, offset)


def _decode(data: bytes, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        pixels = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    width, height = size
    if (pixels.shape[1], pixels.shape[0]) != (width, height):
        pixels = _place(pixels, size, (0, 0))
    return pixels


def _known_blend(name: object, zf: zipfile.ZipFile) -> str:
    """A track's blend mode, or ``normal`` and a log line for one we lack.

    The same tolerance the ``stack.xml`` reader has always had, which
    ``animation.json`` did not: ``Layer.__post_init__`` refuses an unknown mode,
    that refusal happens while the grid is being built, and the ``except`` around
    it drops the **whole timeline** back to a flat read. So a file written by a
    build that carries one more mode than this one cost a user every frame of
    their animation over a string. A mode is how a layer composites, not what it
    contains -- ``Tag.direction``'s rule, for the same reason.
    """
    if isinstance(name, str) and name in cp.BLEND_MODES:
        return name
    log.warning(
        "unknown blend mode %r in %s; using normal", name, getattr(zf, "filename", "?")
    )
    return "normal"


def _read_animation(zf: zipfile.ZipFile, size: tuple[int, int]) -> Animation | None:
    """Rebuild the grid from ``animation.json``, or return None to fall back.

    Every way of being wrong ends the same way -- a log line and the flat read
    -- because the alternative is refusing to open a file whose pixels are all
    present and intact. The one thing worth being strict about is *silent*
    wrongness, so a cel naming a missing PNG or an out-of-range index fails the
    whole grid rather than being skipped: half a timeline is harder to notice
    than none of one, and the flat fallback at least looks like what it is.
    """
    try:
        raw = zf.read(ANIMATION_MEMBER)
    except KeyError:
        return None
    try:
        payload = json.loads(raw)
        if int(payload.get("version", 0)) != ANIMATION_VERSION:
            raise ValueError(f"animation.json version {payload.get('version')!r}")
        # An absent duration is a file written by something that does not carry
        # one, so it gets the default a new frame gets. Falling through to
        # ``clamp_duration``'s floor instead gave it 1 ms -- a hundred times too
        # fast, and silently, since a clip that plays is not obviously wrong.
        frames = [
            Frame(duration_ms=entry.get("duration_ms", DEFAULT_DURATION_MS))
            for entry in payload["frames"]
        ]
        tracks = [
            Track(
                name=entry.get("name") or f"Layer {i + 1}",
                opacity=float(entry.get("opacity", 1.0)),
                visible=bool(entry.get("visible", True)),
                blend=_known_blend(entry.get("blend", "normal"), zf),
                alpha_lock=bool(entry.get("alpha_lock", False)),
            )
            for i, entry in enumerate(payload["tracks"])
        ]
        if not frames or not tracks:
            raise ValueError("an animation has at least one frame and one track")

        # Decoded once per distinct ``data`` path and *shared* across the slots
        # that name it -- which is the whole of how a link survives a save and a
        # reload. Decoding per slot would give equal pixels in separate objects,
        # and the break would only show on the next stroke.
        planes: dict[str, Layer] = {}
        cels: dict[tuple[int, int], Layer] = {}
        for entry in payload["cels"]:
            ti, fi, src = int(entry["track"]), int(entry["frame"]), entry["data"]
            if not (0 <= ti < len(tracks) and 0 <= fi < len(frames)):
                raise ValueError(f"cel at ({ti}, {fi}) is outside the grid")
            layer = planes.get(src)
            if layer is None:
                track = tracks[ti]
                layer = Layer(
                    pixels=_decode(zf.read(src), size),
                    name=track.name,
                    opacity=track.opacity,
                    visible=track.visible,
                    blend=track.blend,
                    alpha_lock=track.alpha_lock,
                )
                planes[src] = layer
            cels[(tracks[ti].uid, frames[fi].uid)] = layer

        tags = [
            Tag(
                name=entry.get("name") or "tag",
                start=int(entry.get("start", 0)),
                end=int(entry.get("end", 0)),
                loop=bool(entry.get("loop", True)),
                # A file written before the field, or by something that spells
                # it differently, gets ``Tag``'s own coercion to forward rather
                # than failing the whole grid: a direction is how a tag plays,
                # not what it contains.
                direction=str(entry.get("direction", "forward")),
            )
            for entry in payload.get("tags", [])
        ]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        log.warning("ignoring animation.json in %s: %s", getattr(zf, "filename", "?"), exc)
        return None

    # Its own guard, outside the block above, and deliberately: a layout is
    # metadata *about* a grid whose pixels are all present and correct, so a
    # malformed or unknown one must cost the document its export shortcut, not
    # its timeline. ``DirectionalLayout.of`` already answers None for a kind
    # this build does not carry; this catches a "layout" that is not a mapping
    # at all.
    layout = None
    try:
        raw_layout = payload.get("layout")
        if raw_layout is not None:
            layout = DirectionalLayout.of(raw_layout["kind"])
    except (KeyError, TypeError) as exc:
        log.warning("ignoring animation.json layout in %s: %s",
                    getattr(zf, "filename", "?"), exc)

    return Animation(
        tracks=tracks, frames=frames, cels=cels, tags=tags, current=0, layout=layout
    )


def _read_palette(zf) -> list | None:
    """The document's colour table, or None when the file carries none.

    Tolerant in the way the rest of this reader is: a palette member that will
    not parse costs the *indexed constraint*, never the file. The pixels are
    already snapped -- they were written that way -- so a document that opens
    without its table is the same picture with the constraint lifted, which is
    a far better outcome than refusing to open it.
    """
    try:
        raw = zf.read(PALETTE_MEMBER)
    except KeyError:
        return None
    try:
        return gpl.parse(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        log.warning("ignoring %s in %s: %s", PALETTE_MEMBER, getattr(zf, "filename", "?"), exc)
        return None


def _read_slices(zf: zipfile.ZipFile, anim: Animation | None) -> list:
    """Rebuild ``doc.slices`` from ``warlock.json``, or answer with none.

    All-or-nothing *within the member*, and never beyond it. A wrong version, a
    payload that is not the shape it claims, or any single entry failing on its
    own terms drops the whole member and the file opens as a drawing with no
    slices on it -- which is what every ORA in the world already is. Half a
    slice list is the outcome worth avoiding: a nine-slice panel missing its
    centre still exports, silently, and stretches wrong in the game.

    Keys are read by frame index and resolved against the grid. When there is no
    grid -- either a still document or, more importantly, an
    ``animation.json`` the reader has already rejected -- the keys are dropped
    with a line in the log rather than guessed at: an index into a timeline that
    was not restored names nothing.
    """
    from .slices import Slice, SliceKey

    try:
        raw = zf.read(WARLOCK_MEMBER)
    except KeyError:
        return []
    frames = [] if anim is None else anim.frames
    dropped = 0
    try:
        payload = json.loads(raw)
        if int(payload.get("version", 0)) != WARLOCK_VERSION:
            raise ValueError(f"{WARLOCK_MEMBER} version {payload.get('version')!r}")
        out = []
        for entry in payload.get("slices", []):
            bounds = _rect_of(entry, "bounds")
            if bounds is None:
                raise ValueError("a slice with no bounds")
            pivot = entry.get("pivot")
            keys: dict[int, SliceKey] = {}
            for record in entry.get("keys", []):
                index = int(record["frame"])
                if not 0 <= index < len(frames):
                    dropped += 1
                    continue
                key_bounds = _rect_of(record, "bounds")
                if key_bounds is None:
                    raise ValueError("a slice key with no bounds")
                key_pivot = record.get("pivot")
                keys[frames[index].uid] = SliceKey(
                    bounds=key_bounds,
                    pivot=(
                        None
                        if key_pivot is None
                        else (float(key_pivot["x"]), float(key_pivot["y"]))
                    ),
                    center=_rect_of(record, "center"),
                )
            out.append(
                Slice(
                    name=str(entry.get("name") or f"Slice {len(out) + 1}"),
                    bounds=bounds,
                    pivot=(
                        None if pivot is None else (float(pivot["x"]), float(pivot["y"]))
                    ),
                    center=_rect_of(entry, "center"),
                    keys=keys,
                )
            )
    except (AttributeError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        # ``AttributeError`` as well as the four ``_read_animation`` catches:
        # a ``"slices"`` that is a *string* iterates into characters, and the
        # first ``.get`` on one is the shape check this member would otherwise
        # be missing.
        log.warning("ignoring %s in %s: %s", WARLOCK_MEMBER, getattr(zf, "filename", "?"), exc)
        return []
    if dropped:
        log.warning(
            "dropped %d slice key(s) in %s: no such frame",
            dropped,
            getattr(zf, "filename", "?"),
        )
    return out


def read_ora(path: Path, *, budget: int | None = None):
    from PIL import Image

    from .document import Document, matte_for
    from .undo import UNDO_BYTES, UndoStack

    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
        width = int(root.get("w") or 0)
        height = int(root.get("h") or 0)

        # JSON first, and only when the XML told us how big the canvas is: the
        # grid's cels are decoded against that size, and guessing it from the
        # first PNG would be guessing for every later one too.
        palette = _read_palette(zf)
        anim = _read_animation(zf, (width, height)) if width and height else None
        # After the grid, and outside its guard: the keys are stored by frame
        # index, so they can only be resolved against the timeline that was
        # actually restored -- and a grid that fell back to the flat read has no
        # timeline to resolve them against.
        found_slices = _read_slices(zf, anim)
        if anim is not None:
            stack = LayerStack(
                anim.layers_for(anim.frames[0], (width, height)),
                len(anim.tracks) - 1,
            )
            doc = Document(
                stack=stack,
                history=UndoStack(UNDO_BYTES if budget is None else budget),
                anim=anim,
                slices=found_slices,
            )
            doc.matte = matte_for(doc.composite)
            doc.file_format = "ora"
            doc.path = Path(path)
            # ``snap=False``: the pixels in the file were written snapped, so
            # re-snapping them would cost a whole-document rewrite on every
            # open and push an undo step for opening a file.
            doc.set_palette(palette, snap=False)
            return doc

        layers: list[Layer] = []
        for element in _layer_elements(root):
            src = element.get("src")
            if not src:
                continue
            try:
                data = zf.read(src)
            except KeyError:
                continue
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                pixels = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
            if not width or not height:
                width, height = pixels.shape[1], pixels.shape[0]
            offset = (int(element.get("x") or 0), int(element.get("y") or 0))
            if offset != (0, 0) or (pixels.shape[1], pixels.shape[0]) != (width, height):
                pixels = _place(pixels, (width, height), offset)
            layers.append(
                Layer(
                    pixels=pixels,
                    name=element.get("name") or f"Layer {len(layers) + 1}",
                    opacity=float(element.get("opacity") or 1.0),
                    visible=element.get("visibility", "visible") != "hidden",
                    blend=cp.OPS_ORA.get(element.get("composite-op", ""), "normal"),
                    alpha_lock=element.get(LOCK_ATTR) == "1",
                )
            )

    if not layers:
        layers = [Layer.empty(max(1, width), max(1, height), "Background")]
    layers.reverse()  # file order is top-first; ours is bottom-first
    doc = Document(
        stack=LayerStack(layers, len(layers) - 1),
        history=UndoStack(UNDO_BYTES if budget is None else budget),
        slices=found_slices,
    )
    doc.matte = matte_for(doc.composite)
    doc.file_format = "ora"
    doc.path = Path(path)
    doc.set_palette(palette, snap=False)
    return doc

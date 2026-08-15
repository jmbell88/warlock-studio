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


def _group_nester(doc, container):
    """A ``member uid -> element`` function that opens a ``<stack>`` per group.

    ORA has no membership list: a group *is* a nested element, so writing the
    tree means opening and closing elements as the walk moves through the
    stack. That is sound for exactly the reason the tree has an invariant --
    a group's leaves are contiguous, so each group is opened once and closed
    once, and the walk never has to come back to it.

    On a document with no groups every uid answers ``container`` and no element
    is ever created, so the XML is byte-identical to what this writer produced
    before groups existed. That is the negative control, and it is a property
    of this function rather than of a branch at the call site.
    """
    from . import groups as gp

    open_uids: list[int] = []
    open_els = [container]

    def element_for(member_uid: int):
        chain = list(reversed(gp.ancestry(doc.group_of, member_uid)))
        shared = 0
        while (
            shared < len(open_uids)
            and shared < len(chain)
            and open_uids[shared] == chain[shared]
        ):
            shared += 1
        del open_uids[shared:]
        del open_els[shared + 1 :]
        for guid in chain[shared:]:
            node = doc.groups.get(guid)
            if node is None:  # pragma: no cover - a dangling parent
                continue
            element = ElementTree.SubElement(
                open_els[-1],
                "stack",
                {
                    "name": node.name,
                    "opacity": f"{float(node.opacity):.6f}",
                    "visibility": "visible" if node.visible else "hidden",
                    # ``auto`` is ORA's spelling of pass-through, which is what
                    # this build composites -- see ``inker/groups.py``. Saying
                    # so explicitly is what stops Krita opening the file as an
                    # *isolated* group and rendering it differently.
                    "isolation": "auto",
                    **_lock_attr(False, node.locked),
                },
            )
            open_uids.append(guid)
            open_els.append(element)
        return open_els[-1]

    return element_for


def _stack_xml(doc) -> bytes:
    width, height = doc.size
    root = ElementTree.Element(
        "image", {"version": "0.0.3", "w": str(width), "h": str(height)}
    )
    stack = ElementTree.SubElement(root, "stack")
    parent_for = _group_nester(doc, stack)
    # Top first: ORA's document order is the painter's, reversed. Reversing
    # keeps each group's members adjacent, because contiguity is a property of
    # the order and not of its direction.
    for index, layer in enumerate(reversed(list(doc.stack))):
        ElementTree.SubElement(
            parent_for(layer.uid),
            "layer",
            {
                "name": layer.name,
                "src": f"data/layer{index}.png",
                "x": "0",
                "y": "0",
                "opacity": f"{float(layer.opacity):.6f}",
                "visibility": "visible" if layer.visible else "hidden",
                "composite-op": cp.ORA_OPS.get(layer.blend, "svg:src-over"),
                **_lock_attr(layer.alpha_lock, layer.locked),
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

#: The content lock, on the same terms and for the same reasons. A separate
#: attribute rather than a value on the first, because the two locks are
#: independent -- a layer can preserve transparency, refuse writes, both or
#: neither -- and because a reader that only knows the older one must go on
#: reading it correctly.
CONTENT_LOCK_ATTR = "warlock-content-lock"


def _lock_attr(alpha_lock: bool, locked: bool = False) -> dict[str, str]:
    """The lock attributes, written only where they are set.

    Absent rather than ``"0"``, so a document nobody has locked anything in
    produces byte-identical XML to the one this build wrote before either
    attribute existed -- which is what the determinism pin is measuring.
    """
    out: dict[str, str] = {}
    if alpha_lock:
        out[LOCK_ATTR] = "1"
    if locked:
        out[CONTENT_LOCK_ATTR] = "1"
    return out


#: Attributes this writer puts on a group's ``<stack>``, and therefore the ones
#: its reader knows how to put back. Anything else a foreign file carries on one
#: -- Krita's ``composite-op``, a selection or an alpha-inheritance flag -- is
#: dropped with a log line rather than guessed at, which is the same bargain the
#: layer reader makes with a composite-op it cannot reproduce.
GROUP_ATTRS = frozenset(
    {"name", "opacity", "visibility", "isolation", LOCK_ATTR, CONTENT_LOCK_ATTR}
)


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
        #
        # A fresh nester per frame: the layer groups are nested *inside* each
        # frame's group, because the XML is the picture a foreign editor gets
        # and that picture is one frame with its folders in it. The record --
        # where the grouping survives a round trip -- is ``animation.json``.
        parent_for = _group_nester(doc, group)
        for track in reversed(anim.tracks):
            layer = anim.cels.get((track.uid, frame.uid))
            if layer is None:
                continue
            ElementTree.SubElement(
                parent_for(track.uid),
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
                    **_lock_attr(track.alpha_lock, track.locked),
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
    grouping = _groups_payload(doc, tracks)
    if grouping is not None:
        # Additive and the version stays 1, for ``layout``'s reason: the reader
        # is guarded on its own and a build that does not know the key opens the
        # file as a flat grid, which is exactly the document it was before
        # groups existed. Written only when there is a group, so an ungrouped
        # document's ``animation.json`` is byte-identical to what it was.
        payload["groups"] = grouping
    return json.dumps(payload, indent=2).encode("utf-8")


def _groups_payload(doc, tracks: dict[int, int]) -> dict | None:
    """The layer-group tree as indices, or None when there is no tree.

    Indices, not uids, for ``_animation_json``'s reason -- uids are minted per
    process and mean nothing in a file. Groups are numbered by the order they
    are first opened walking the tracks bottom-first, which is deterministic
    and is also the order the nested ``<stack>`` elements come out in.
    """
    from . import groups as gp

    if not getattr(doc, "groups", None):
        return None
    order: list[int] = []
    for track_uid in tracks:
        for guid in reversed(gp.ancestry(doc.group_of, track_uid)):
            if guid in doc.groups and guid not in order:
                order.append(guid)
    if not order:
        return None
    index_of = {guid: i for i, guid in enumerate(order)}
    return {
        "nodes": [
            {
                "name": doc.groups[guid].name,
                "visible": bool(doc.groups[guid].visible),
                "opacity": float(doc.groups[guid].opacity),
                "locked": bool(doc.groups[guid].locked),
            }
            for guid in order
        ],
        "tracks": [
            {"track": tracks[uid], "group": index_of[doc.group_of[uid]]}
            for uid in tracks
            if doc.group_of.get(uid) in index_of
        ],
        "nesting": [
            {"group": index_of[guid], "parent": index_of[doc.group_of[guid]]}
            for guid in order
            if doc.group_of.get(guid) in index_of
        ],
    }


def _frame_flatten(doc, frame) -> np.ndarray:
    """Frame 1's pixels, whatever the playhead is on.

    ``mergedimage.png`` has to be a function of the document and not of where
    the user happened to be looking, or saving the same file twice produces two
    different files.
    """
    # ``frame_stack``, not a bare ``LayerStack``: it carries the layer-group
    # fold, without which a hidden group would be hidden on screen and visible
    # in ``mergedimage.png``.
    return cp.flatten_onto(doc.frame_stack(frame).flatten(), doc.matte)


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


def _layer_elements(node, tree=None, parent=None) -> list:
    """Depth-first, flattening nested stacks *and* recording them as groups.

    The flattening is unchanged and still load-bearing: the flat stack is
    authoritative for paint order, so a nested file has always come out as one
    list and always will. What is new is that the nesting is no longer thrown
    away -- ``tree`` collects ``(groups, group_of)`` on the way through, so a
    Krita file's folders survive a round trip instead of being silently lost.

    Each layer comes back as ``(element, parent group uid or None)``. Depth-
    first order is what makes the result satisfy the contiguity invariant for
    free: a group's layers are exactly the run this walk emits while it is
    inside that group's element.

    An **empty** ``<stack>`` records no group. Empty groups are disallowed in
    the model, and a foreign file is entitled to contain one -- as is one of
    ours whose layer PNGs went missing.
    """
    from .groups import GroupNode

    found: list = []
    for child in node:
        if child.tag == "layer":
            found.append((child, parent))
        elif child.tag == "stack":
            if tree is None:
                found.extend(_layer_elements(child))
                continue
            groups, group_of = tree
            node_group = GroupNode(
                name=child.get("name") or f"Group {len(groups) + 1}",
                visible=child.get("visibility", "visible") != "hidden",
                opacity=float(child.get("opacity") or 1.0),
                locked=child.get(CONTENT_LOCK_ATTR) == "1",
            )
            inner = _layer_elements(child, tree, node_group.uid)
            if not inner:
                continue
            unmodelled = set(child.attrib) - GROUP_ATTRS
            if unmodelled:
                log.debug(
                    "dropping unmodelled group attributes on %r: %s",
                    node_group.name,
                    ", ".join(sorted(unmodelled)),
                )
            groups[node_group.uid] = node_group
            if parent is not None:
                group_of[node_group.uid] = parent
            found.extend(inner)
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


def _read_animation(zf: zipfile.ZipFile, size: tuple[int, int]):
    """The grid and the payload it came from, or None to fall back to flat.

    The payload rides back out because the *grouping* is read from it too, and
    it has to be read after the ``Document`` exists -- membership is keyed on
    track uids, which the tracks only have once they are built. Re-reading the
    member to get at it would parse the same JSON twice and give two chances
    for the two readers to disagree about what it said.

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
                blend=entry.get("blend", "normal"),
                alpha_lock=bool(entry.get("alpha_lock", False)),
                # ``.get``-based like every other key here, so a file written
                # before the content lock existed reads as unlocked rather than
                # failing the whole grid. That is why the version stays 1.
                locked=bool(entry.get("locked", False)),
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
                    locked=track.locked,
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
    ), payload


def _read_groups(doc, payload: dict) -> None:
    """Rebuild the layer-group tree from ``animation.json``, or leave it empty.

    Guarded exactly like ``layout`` and for the same reason, one level up: a
    grouping is metadata *about* a grid whose pixels are all present and
    correct, so a malformed one must cost the document its folders and never
    its timeline. Anything wrong -- a track index outside the grid, a parent
    that is not a group, a "groups" that is not a mapping -- logs and leaves
    the document flat.
    """
    from .groups import GroupNode

    raw = payload.get("groups")
    if raw is None:
        return
    try:
        nodes = [
            GroupNode(
                name=entry.get("name") or f"Group {i + 1}",
                visible=bool(entry.get("visible", True)),
                opacity=float(entry.get("opacity", 1.0)),
                locked=bool(entry.get("locked", False)),
            )
            for i, entry in enumerate(raw["nodes"])
        ]
        tracks = doc.anim.tracks
        group_of: dict[int, int] = {}
        for entry in raw.get("tracks", []):
            ti, gi = int(entry["track"]), int(entry["group"])
            if not (0 <= ti < len(tracks) and 0 <= gi < len(nodes)):
                raise ValueError(f"track {ti} names group {gi}")
            group_of[tracks[ti].uid] = nodes[gi].uid
        for entry in raw.get("nesting", []):
            gi, pi = int(entry["group"]), int(entry["parent"])
            if not (0 <= gi < len(nodes) and 0 <= pi < len(nodes)) or gi == pi:
                raise ValueError(f"group {gi} names parent {pi}")
            group_of[nodes[gi].uid] = nodes[pi].uid
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        log.warning("ignoring animation.json groups: %s", exc)
        return
    _install_groups(doc, ({node.uid: node for node in nodes}, {}), group_of)


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


def _install_groups(doc, tree: tuple[dict, dict], parents: dict[int, int]) -> None:
    """Put a read tree onto the document, dropping whatever is now empty.

    A group whose layers were all skipped -- a missing ``src``, a PNG the
    archive does not hold -- must not survive as an empty folder, and neither
    must its ancestors. Pruning here rather than refusing the file is the whole
    of this reader's bargain: a file that opens slightly wrong is a file the
    user still has.
    """
    from . import groups as gp

    nodes, nesting = tree
    if not nodes:
        return
    doc.groups = dict(nodes)
    doc.group_of = {**nesting, **parents}
    order = doc.member_uids()
    for guid in list(doc.groups):
        if not gp.leaves_of(doc.group_of, order, guid):
            doc._drop_group(guid)
    doc.invalidate_all()


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
        got = _read_animation(zf, (width, height)) if width and height else None
        anim, grid_payload = got if got is not None else (None, None)
        if anim is not None:
            stack = LayerStack(
                anim.layers_for(anim.frames[0], (width, height)),
                len(anim.tracks) - 1,
            )
            doc = Document(
                stack=stack,
                history=UndoStack(UNDO_BYTES if budget is None else budget),
                anim=anim,
            )
            _read_groups(doc, grid_payload)
            doc.matte = matte_for(doc.composite)
            doc.file_format = "ora"
            doc.path = Path(path)
            # ``snap=False``: the pixels in the file were written snapped, so
            # re-snapping them would cost a whole-document rewrite on every
            # open and push an undo step for opening a file.
            doc.set_palette(palette, snap=False)
            return doc

        layers: list[Layer] = []
        tree: tuple[dict, dict] = ({}, {})
        parents: dict[int, int] = {}
        # From the document's own root ``<stack>``, not from ``<image>``: the
        # outer stack is the document, and reading it as a group would wrap
        # every file this reader opens in one folder called "Group 1".
        outer = root.find("stack")
        for element, parent in _layer_elements(root if outer is None else outer, tree):
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
                    # A foreign file carries neither attribute, so every layer
                    # in it opens unlocked -- which is the answer a Krita or
                    # GIMP document should give, since neither writes ours.
                    locked=element.get(CONTENT_LOCK_ATTR) == "1",
                )
            )
            if parent is not None:
                # Recorded against the layer's uid only once the layer exists,
                # which is what keeps a group whose PNGs are all missing from
                # coming back as an empty one.
                parents[layers[-1].uid] = parent

    if not layers:
        layers = [Layer.empty(max(1, width), max(1, height), "Background")]
    layers.reverse()  # file order is top-first; ours is bottom-first
    doc = Document(
        stack=LayerStack(layers, len(layers) - 1),
        history=UndoStack(UNDO_BYTES if budget is None else budget),
    )
    _install_groups(doc, tree, parents)
    doc.matte = matte_for(doc.composite)
    doc.file_format = "ora"
    doc.path = Path(path)
    doc.set_palette(palette, snap=False)
    return doc

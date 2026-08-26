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

import contextlib
import io
import json
import logging
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .. import pixelguard, xmlguard, zipguard
from ..tilegrid import gid
from ..tilegrid.tileset import Tileset
from . import composite as cp
from . import gpl
from . import index_plane as ixp
from .animation import (
    DEFAULT_DURATION_MS,
    Animation,
    DirectionalLayout,
    Frame,
    Tag,
    Track,
)
from .layers import Layer, LayerStack
from .tiles import TilemapCel, TilesetSlot, materialize

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

#: A document's tilesets, track bindings and cel refs -- Wave 3 chunk 3.4.
#: Its own member and its own version, ``WARLOCK_MEMBER``'s reasons restated
#: one layer further in: tile *structure* is metadata about a picture that is
#: already fully and honestly on the canvas -- every ``TilemapCel``'s own PNG
#: is written and read exactly like any other layer's -- so a broken or
#: unknown member here must cost the structure and never a pixel or a frame.
#: Written only when ``doc.tilesets`` is non-empty, so a document that has
#: never touched a tileset produces the exact archive this writer wrote before
#: tilesets existed.
TILES_MEMBER = "tiles.json"
TILES_VERSION = 1

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

# A zip's directory declares what each member unpacks to and nothing makes that
# number honest -- a few kilobytes of archive can claim terabytes, and the read
# that discovers this is the one that has already exhausted memory. The
# ``clay/serialize.py`` constant verbatim, as ``packwright/wpack.py`` and
# ``plotter/wmap.py`` also carry it; ORA was the one container door in the tree
# without it, which mattered more here than anywhere else because a ``.wblk``
# is ours and an ORA is explicitly *anyone's*. Read from module globals at call
# time for their reason too: a test lowers it rather than building a gigabyte.
MAX_DECOMPRESSED_BYTES = 1 << 30

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


def _png_indexed(indices: np.ndarray, palette, transparent: int) -> bytes:
    """An index plane as a **PNG mode-P image**: ``PLTE``, ``tRNS``, raw slots.

    The record/projection doctrine a third time (``animation.json`` and
    ``mergedimage.png`` are the other two). These bytes are the record for both
    the indices and the *full-alpha* table; ``palette.gpl`` stays the interop
    projection and ``stack.xml``'s colours stay the picture a foreign editor
    gets. Writing our own container instead would have bought a format only this
    app can read, in exchange for nothing.

    **Old and foreign readers degrade perfectly**, which is the whole reason
    this shape was chosen: ``_decode`` already ends in ``im.convert("RGBA")``,
    which honours ``PLTE`` + ``tRNS``, so a build that has never heard of index
    planes opens the file with the right pixels and lands in
    palette-constrained RGB via ``palette.gpl``. Krita and GIMP decode a P-PNG
    natively. The stated forward-compat cost mirrors ``animation.json``'s: an
    old build that *re-saves* writes RGBA planes back, and index identity is
    gone.

    The transparent slot keeps its **stored colour** in ``PLTE`` and gets alpha
    zero in ``tRNS``. Zeroing the colour too would be an edit rather than a
    projection, and a ``.aseprite`` writer has to put that colour back.
    """
    from PIL import Image

    height, width = indices.shape
    im = Image.frombytes("P", (width, height), indices.tobytes())
    entries = [tuple(colour) for colour in palette]
    rgb: list[int] = []
    for colour in entries:
        rgb.extend(int(c) for c in colour[:3])
    im.putpalette(rgb)
    alpha = bytes(
        0 if i == transparent else int(colour[3]) if len(colour) > 3 else 255
        for i, colour in enumerate(entries)
    )
    buf = io.BytesIO()
    # ``optimize`` is deliberately off (it is off by default and stated here
    # because turning it on would be a plausible-looking change): Pillow's
    # optimiser drops unused palette entries and renumbers the plane, which is
    # exactly the identity this member exists to preserve. A slot nothing is
    # painted in is still a slot the user authored.
    im.save(buf, "PNG", transparency=alpha)
    return buf.getvalue()


def _read_indexed_png(data: bytes, size: tuple[int, int]):
    """``(indices, palette)`` out of a P-PNG, or None when it is not one.

    None rather than a raise, this reader's rule throughout: a document whose
    ``color`` block says indexed but whose planes are ordinary RGBA -- an old
    build re-saved it, or the archive was assembled by hand -- opens as the
    picture it is, with the indices re-inferred from the pixels. Losing
    duplicate-slot identity is a real cost and a much smaller one than refusing
    the file.
    """
    with pixelguard.opened(io.BytesIO(data), "a layer in this drawing") as im:
        if im.mode != "P":
            return None
        im.load()
        indices = np.asarray(im, dtype=np.uint8).copy()
        raw = im.getpalette() or []
        transparency = im.info.get("transparency", b"")
    count = len(raw) // 3
    if not count:
        return None
    # ``transparency`` comes back in **two shapes**, and reading only one of
    # them silently loses the table's alpha. Pillow writes a ``tRNS`` chunk only
    # as long as it needs to (trailing opaque entries are dropped), and when
    # exactly one leading entry is transparent it hands the whole chunk back as
    # a bare ``int`` -- the "single transparent index" form. A palette carrying
    # a translucent swatch gets the ``bytes`` form. Both mean the same thing:
    # every entry the chunk does not reach is opaque.
    if isinstance(transparency, int):
        alphas = bytes([255] * transparency) + b"\x00"
    elif isinstance(transparency, (bytes, bytearray)):
        alphas = bytes(transparency)
    else:
        alphas = b""
    palette = [
        (
            raw[i * 3],
            raw[i * 3 + 1],
            raw[i * 3 + 2],
            alphas[i] if i < len(alphas) else 255,
        )
        for i in range(count)
    ]
    width, height = size
    if (indices.shape[1], indices.shape[0]) != (width, height):
        return None
    return indices, palette


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


def _read_resolution(root) -> tuple[int, int] | None:
    """``(xres, yres)`` off the ``<image>`` element, or None.

    ORA's physical-size fields. Anything non-numeric or non-positive reads as
    absent rather than raising: a resolution is metadata, and a drawing whose
    pixels are perfectly readable must not fail to open over it.
    """
    try:
        xres, yres = int(root.get("xres") or 0), int(root.get("yres") or 0)
    except (TypeError, ValueError):
        return None
    return (xres, yres) if xres > 0 and yres > 0 else None


def _image_attrs(doc) -> dict[str, str]:
    """The ``<image>`` element's attributes, resolution included when known.

    ``xres``/``yres`` are ORA's physical-size fields. Nothing in this editor
    renders at a physical size, so they are read into ``Document.dpi`` and
    written straight back out -- carried, not used. Omitted entirely when the
    document has none, because writing a guessed 72 would be asserting a fact
    about a canvas nobody stated one for.
    """
    width, height = doc.size
    attrs = {"version": "0.0.3", "w": str(width), "h": str(height)}
    dpi = getattr(doc, "dpi", None)
    if dpi:
        attrs["xres"], attrs["yres"] = str(int(dpi[0])), str(int(dpi[1]))
    # Unconditional, unlike its siblings -- see ``MATTE_ATTR``. Both writers go
    # through here, so the still and animated stacks agree for free.
    attrs[MATTE_ATTR] = _matte_attr(getattr(doc, "matte", None))
    return attrs


def _stack_xml(doc) -> bytes:
    root = ElementTree.Element("image", _image_attrs(doc))
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
                **_lock_attr(
                    layer.alpha_lock,
                    layer.locked,
                    getattr(layer, "background", False),
                    getattr(layer, "reference", False),
                ),
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

#: The two layer types of 6.5, written **only when set** -- the content lock's
#: own rule, and its reason: an ordinary document's ``stack.xml`` stays
#: byte-identical to what this writer produced before they existed.
BACKGROUND_ATTR = "warlock-background"
REFERENCE_ATTR = "warlock-reference"

#: The document's flatten matte, on the ``<image>`` root -- it is document
#: state, not a layer's.
#:
#: Written **unconditionally**, which is the one place this writer breaks the
#: "only when set" rule its four siblings above follow, and deliberately: the
#: setting is tri-state -- on, off, and *no stored answer* -- and absence has
#: to go on meaning "infer with ``matte_for``" for Krita files and for files
#: written before this attribute existed. A write-only-when-set rule cannot
#: express the third state, so "the user turned the matte off" would be
#: indistinguishable from "nobody said", and an opaque drawing would silently
#: get its matte back on every reopen.
MATTE_ATTR = "warlock-matte"


def _matte_attr(matte) -> str:
    """``none`` or ``#rrggbbaa``. See ``MATTE_ATTR`` for why never absent."""
    if matte is None:
        return "none"
    red, green, blue, alpha = (list(matte) + [255, 255, 255, 255])[:4]
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}{int(alpha):02x}"


def _read_matte(doc, root) -> None:
    """Put the stored matte back, or infer one. Never raises.

    Absent -> ``matte_for``, the call this reader always made. ``none`` -> off.
    ``#rrggbbaa`` -> that colour. Anything else warns and falls back to
    ``matte_for``: the same degradation contract every optional member in this
    reader follows -- a drawing whose pixels are perfectly readable must not
    fail to open over one bad metadata attribute.
    """
    from .document import matte_for

    raw = root.get(MATTE_ATTR)
    if raw is None:
        doc.matte = matte_for(doc.composite)
        return
    text = raw.strip().lower()
    if text == "none":
        doc.matte = None
        return
    try:
        if not text.startswith("#") or len(text) != 9:
            raise ValueError(raw)
        doc.matte = tuple(int(text[at : at + 2], 16) for at in (1, 3, 5, 7))
    except ValueError:
        log.warning("ignoring unreadable %s=%r; inferring instead", MATTE_ATTR, raw)
        doc.matte = matte_for(doc.composite)


def _lock_attr(
    alpha_lock: bool,
    locked: bool = False,
    background: bool = False,
    reference: bool = False,
) -> dict[str, str]:
    """The lock and layer-type attributes, written only where they are set.

    Absent rather than ``"0"``, so a document nobody has locked anything in --
    and nobody has made a background of -- produces byte-identical XML to the
    one this build wrote before any of these attributes existed, which is what
    the determinism pin is measuring.
    """
    out: dict[str, str] = {}
    if alpha_lock:
        out[LOCK_ATTR] = "1"
    if locked:
        out[CONTENT_LOCK_ATTR] = "1"
    if background:
        out[BACKGROUND_ATTR] = "1"
    if reference:
        out[REFERENCE_ATTR] = "1"
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
    root = ElementTree.Element("image", _image_attrs(doc))
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
                    **_lock_attr(
                        track.alpha_lock,
                        track.locked,
                        getattr(track, "background", False),
                        getattr(track, "reference", False),
                    ),
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
        # ``continuous`` is written **only when it is set**, which is
        # ``repeat``'s rule below rather than ``direction``'s: false is what
        # every track ever written already means, so omitting it keeps a
        # document with no continuous rows byte-identical to what it was, and
        # the determinism pin depends on exactly that. Built here rather than
        # inside ``Track.props`` because that dict is *also* the six-property
        # copy-down list, and this is not one of the six.
        "tracks": [
            {**track.props(), **({"continuous": True} if track.continuous else {})}
            for track in anim.tracks
        ],
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
                # ``repeat`` goes one step further than ``direction`` and is
                # written **only when it is set**, which is ``layout``'s rule
                # rather than this list's: 0 is the value every tag ever
                # written already has, so omitting it keeps a repeat-less
                # document's ``animation.json`` byte-identical to what it was
                # -- the determinism pin depends on exactly that.
                **({"repeat": int(tag.repeat)} if tag.repeat else {}),
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


def _tile_refs_names(doc, anim) -> dict[int, str]:
    """One name per *distinct* tilemap cel: ``data/tilerefs{n}.u32``.

    ``_cel_names``'s pattern, extended: keyed by ``id(layer)`` so a linked cel
    answers to one name rather than one per slot it occupies, and ``n`` counts
    only the tilemap cels found walking the very order the cel PNGs themselves
    use -- ``unique_cel_layers`` on an animated document, stack order on a
    still one (``_stack_xml``'s own walk, not the reversed one ``write_ora``
    writes PNGs in -- what matters here is a deterministic, repeatable order,
    and ``enumerate(doc.stack)`` is the one :func:`_tiles_json`'s ``"layer"``
    index already commits to).
    """
    layers = anim.unique_cel_layers() if anim is not None else list(doc.stack)
    out: dict[int, str] = {}
    n = 0
    for layer in layers:
        if isinstance(layer, TilemapCel):
            out[id(layer)] = f"data/tilerefs{n}.u32"
            n += 1
    return out


def _tiles_json(
    doc, anim, names: dict[int, str], refs_names: dict[int, str]
) -> bytes:
    """The tileset/track/cel record. Indices, not uids -- ``_animation_json``'s
    convention, restated: a uid is minted per process and means nothing in a
    file.

    Every tileset in ``doc.tilesets`` is listed, referenced or not -- a user's
    spare tileset is not garbage, and this is what lets it survive a save it
    was never drawn with. ``"tracks"`` appears only on an animated document (a
    still one has no tracks at all); ``"cels"`` keys each entry by the cel's
    own PNG member name on an animated document -- the stable cross-member key
    :func:`_animation_json` already uses -- and by stack index on a still one.
    """
    tileset_index = {slot.uid: i for i, slot in enumerate(doc.tilesets)}
    payload: dict = {
        "version": TILES_VERSION,
        "tilesets": [
            {
                "name": slot.tileset.name,
                "tile_w": int(slot.tileset.tile_w),
                "tile_h": int(slot.tileset.tile_h),
                "data": f"data/tileset{i}.png",
            }
            for i, slot in enumerate(doc.tilesets)
        ],
    }
    cels: list[dict] = []
    if anim is not None:
        payload["tracks"] = [
            {"track": ti, "tileset": tileset_index[track.tileset_uid]}
            for ti, track in enumerate(anim.tracks)
            if track.tileset_uid is not None and track.tileset_uid in tileset_index
        ]
        for layer in anim.unique_cel_layers():
            if isinstance(layer, TilemapCel) and layer.tileset_uid in tileset_index:
                grid_h, grid_w = layer.refs.shape
                cels.append(
                    {
                        "cel": names[id(layer)],
                        "tileset": tileset_index[layer.tileset_uid],
                        "grid_w": int(grid_w),
                        "grid_h": int(grid_h),
                        "refs": refs_names[id(layer)],
                    }
                )
    else:
        for index, layer in enumerate(doc.stack):
            if isinstance(layer, TilemapCel) and layer.tileset_uid in tileset_index:
                grid_h, grid_w = layer.refs.shape
                cels.append(
                    {
                        "layer": index,
                        "tileset": tileset_index[layer.tileset_uid],
                        "grid_w": int(grid_w),
                        "grid_h": int(grid_h),
                        "refs": refs_names[id(layer)],
                    }
                )
    payload["cels"] = cels
    return json.dumps(payload, indent=2).encode("utf-8")


def _write_tiles(zf: zipfile.ZipFile, doc, anim, names: dict[int, str]) -> None:
    """The three tile members, in their fixed order: tileset strips, then the
    refs blobs that name them, then ``tiles.json`` last -- it is what a reader
    needs the first two members' own names for, so it goes out once they are
    already in the archive.
    """
    for i, slot in enumerate(doc.tilesets):
        zf.writestr(_member(f"data/tileset{i}.png"), _png(slot.tileset.pixels))
    refs_names = _tile_refs_names(doc, anim)
    layers = anim.unique_cel_layers() if anim is not None else list(doc.stack)
    for layer in layers:
        if isinstance(layer, TilemapCel):
            # Raw, little-endian, row-major -- kilobytes at most and trivially
            # deterministic, so a member of its own buys nothing a JSON string
            # would not, except that JSON cannot hold binary at all.
            zf.writestr(_member(refs_names[id(layer)]), layer.refs.astype("<u4").tobytes())
    zf.writestr(_member(TILES_MEMBER), _tiles_json(doc, anim, names, refs_names))


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
    colour = _colour_block(doc)
    if colour is not None:
        payload["color"] = colour
    return json.dumps(payload, indent=2).encode("utf-8")


def _colour_block(doc) -> dict | None:
    """The colour mode, or None on an ordinary RGB document.

    **Additive, and written only when it is not the default**, which is what
    keeps ``WARLOCK_VERSION`` at 1 and every RGB document's archive byte-for-byte
    what this writer produced before the block existed. The bump rule -- "a
    reader could get it wrong in a way it cannot detect" -- is not met: an older
    reader ignores the key and opens the file as the RGBA picture the planes
    already are.

    A malformed block costs the *mode metadata* and never the file. The P-PNGs
    are self-describing, so the indices survive; the transparent index recovers
    from ``tRNS`` or falls back to zero, with a line in the log.
    """
    mode = getattr(doc, "color_mode", "rgb")
    if mode == "rgb":
        return None
    block: dict = {"mode": mode}
    if mode == "indexed":
        block["transparent"] = int(getattr(doc, "transparent_index", 0))
    return block


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
    # ``try/finally`` around the encode for ``plotter_io``/``packwright_io``/
    # ``journal``'s reason: ``replace`` only runs on success, so a failed encode
    # left the staging file sitting beside the user's document forever. Not data
    # loss -- the destination is never touched -- just a stray dotfile per
    # failure, and the unlink is a no-op once the replace has renamed it away.
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            # Stored, and first: the spec makes this a magic number at a fixed
            # offset, and a deflated one is not readable as such.
            zf.writestr(
                zipfile.ZipInfo("mimetype", _EPOCH), b"image/openraster", zipfile.ZIP_STORED
            )
            # One encoder for both paths, chosen by the document's mode rather than
            # per layer: a document is indexed or it is not, and a mixed archive is
            # a state no reader (ours least of all) has a sensible answer for.
            def encode(layer) -> bytes:
                if getattr(doc, "color_mode", "rgb") == "indexed" and layer.indices is not None:
                    return _png_indexed(layer.indices, doc.palette, doc.transparent_index)
                return _png(layer.pixels)

            if anim is None:
                zf.writestr(_member("stack.xml"), _stack_xml(doc))
                for index, layer in enumerate(reversed(list(doc.stack))):
                    zf.writestr(_member(f"data/layer{index}.png"), encode(layer))
            else:
                zf.writestr(_member("stack.xml"), _stack_xml_animated(doc, names))
                # One PNG per name, with no de-duplication needed: ``_cel_names``
                # is built from the same ``unique_cel_layers`` walk and gives each
                # distinct cel its own name, so the two can only ever agree.
                for layer in anim.unique_cel_layers():
                    zf.writestr(_member(names[id(layer)]), encode(layer))
                zf.writestr(_member(ANIMATION_MEMBER), _animation_json(doc, names))
            # Only when there is a tileset to record -- a document that has never
            # touched one produces the exact archive this writer wrote before
            # tilesets existed, which is what the determinism suite pins.
            if getattr(doc, "tilesets", None):
                _write_tiles(zf, doc, anim, names)
            if getattr(doc, "palette", None):
                zf.writestr(_member(PALETTE_MEMBER), gpl.dumps(doc.palette).encode("utf-8"))
            # Only when there are slices *or* a colour mode to record. A plain RGB
            # document with neither produces an archive byte-identical to the one
            # this build wrote before either existed, which is what the determinism
            # suite pins and what makes both additions invisible to every reader
            # that has never heard of them.
            if getattr(doc, "slices", None) or _colour_block(doc) is not None:
                zf.writestr(_member(WARLOCK_MEMBER), _warlock_json(doc))
            zf.writestr(_member("mergedimage.png"), _png(merged))
            zf.writestr(_member("Thumbnails/thumbnail.png"), thumb_buf.getvalue())
        tmp.replace(path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


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

    ``tree`` of ``None`` flattens without recording anything, which is exactly
    what this function did before groups existed. :func:`read_ora` passes None
    for the frame-projection case; see :func:`has_frame_groups`.
    """
    from .groups import GroupNode

    found: list = []
    for child in node:
        if child.tag == "layer":
            found.append((child, parent))
        elif child.tag == "stack":
            if tree is None:
                found.extend(_layer_elements(child, None, parent))
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


#: What ``_stack_xml_animated`` names each frame's ``<stack>``: ``frame:0001``
#: and up. It is a *projection* marker rather than a folder, so the reader has
#: to recognise it.
FRAME_GROUP_PREFIX = "frame:"


def has_frame_groups(node) -> bool:
    """Whether this root stack is one of our animated frame projections.

    It matters only on the **flat fallback** -- when ``animation.json`` is
    missing or would not parse and the grid could not be rebuilt. The XML still
    says what it always said: one nested ``<stack>`` per frame. Those are frames,
    never folders, so reading them as layer groups turned a degraded path into a
    noisy one, handing back N folders called ``frame:0001`` where the same file
    previously opened flat.

    Group construction is suppressed for the whole read rather than per element,
    and that is the point: each frame group carries its *own* copy of the
    document's real folders, so recognising the frames alone would still give a
    forty-frame clip forty folders called "Ink". The pixels are all present
    either way, which is the bargain this whole reader makes.

    A foreign file with a top-level group genuinely called ``frame:0001`` loses
    its folders here. That is the accepted cost, and it is bounded: the layers,
    their order and their properties are unaffected.
    """
    return any(
        child.tag == "stack" and (child.get("name") or "").startswith(FRAME_GROUP_PREFIX)
        for child in node
    )


def _place(pixels: np.ndarray, size: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    """Paste a layer onto a canvas-sized plane at its ORA offset.

    Offsets exist on disk and not in memory: every op in this app is a plain
    slice, and the price of that is doing the placement once, here.
    """
    from .transform import resize_canvas

    return resize_canvas(pixels, size, offset)


def _decode(data: bytes, size: tuple[int, int]) -> np.ndarray:
    with pixelguard.opened(io.BytesIO(data), "a layer in this drawing") as im:
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


def _read_animation(zf: zipfile.ZipFile, size: tuple[int, int], reader=None):
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
                blend=_known_blend(entry.get("blend", "normal"), zf),
                alpha_lock=bool(entry.get("alpha_lock", False)),
                # ``.get``-based like every other key here, so a file written
                # before the content lock existed reads as unlocked rather than
                # failing the whole grid. That is why the version stays 1.
                locked=bool(entry.get("locked", False)),
                # The seventh and eighth track properties. ``Track.props``
                # has written both since 6.5; this reader never read them
                # back, so every animated document lost its background and
                # reference flags on load -- the ``Layer.copy`` bug again,
                # in the one copy site that hand-lists its fields.
                background=bool(entry.get("background", False)),
                reference=bool(entry.get("reference", False)),
                # Same ``.get``, same reason -- and no ``stack.xml`` attribute
                # to go with it: a foreign editor has no concept for "new cels
                # start as a copy of the last one", so there is nowhere honest
                # to put it outside our own section.
                continuous=bool(entry.get("continuous", False)),
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
                # Read once and used twice: the RGBA decode and the index plane
                # beside it are two readings of the same bytes, and reading the
                # member twice would be two chances for them to disagree.
                data = zf.read(src)
                layer = Layer(
                    pixels=_decode(data, size),
                    name=track.name,
                    opacity=track.opacity,
                    visible=track.visible,
                    blend=track.blend,
                    alpha_lock=track.alpha_lock,
                    locked=track.locked,
                    background=track.background,
                    reference=track.reference,
                )
                if reader is not None:
                    reader.attach(layer, data)
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
                # Absent in every file written before repeats existed, and 0
                # is exactly "the loop flag decides" -- so an old document
                # reads back playing precisely as it did.
                repeat=int(entry.get("repeat", 0) or 0),
            )
            for entry in payload.get("tags", [])
        ]
    except (
        AttributeError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        # ``_read_colour``'s verbatim five, plus ``OSError``: this member is
        # the one whose entries name *PNGs* to decode, and Pillow's
        # ``UnidentifiedImageError`` on a member holding non-image bytes is an
        # ``OSError`` -- without it a corrupt cel crashed the open instead of
        # falling back flat. ``AttributeError`` for ``_read_slices``' reason:
        # a payload key of the wrong shape fails on its first ``.get``.
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


def _read_colour(zf) -> tuple[str, int]:
    """``(mode, transparent index)``, defaulting to plain RGB.

    Read from ``warlock.json`` before anything decodes a plane, because it is
    what decides *how* the planes are read. It fails the way every optional
    member here fails -- to the default, with a log line -- and the failure is
    cheap by design: the P-PNGs are self-describing, so a lost block costs the
    mode label and the transparent index, not a pixel.
    """
    try:
        raw = zf.read(WARLOCK_MEMBER)
    except KeyError:
        return ("rgb", 0)
    try:
        payload = json.loads(raw)
        if int(payload.get("version", 0)) != WARLOCK_VERSION:
            raise ValueError(f"{WARLOCK_MEMBER} version {payload.get('version')!r}")
        block = payload.get("color")
        if not isinstance(block, dict):
            return ("rgb", 0)
        mode = str(block.get("mode", "rgb"))
        if mode not in ("rgb", "indexed", "grayscale"):
            raise ValueError(f"unknown colour mode {mode!r}")
        return (mode, int(block.get("transparent", 0)))
    except (AttributeError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        log.warning("ignoring colour mode in %s: %s", getattr(zf, "filename", "?"), exc)
        return ("rgb", 0)


class _IndexReader:
    """Collects index planes off the P-PNGs as the layers decode.

    A small object rather than a return value threaded through two decode paths,
    because the *table* is a whole-document fact discovered inside a per-layer
    loop: the first plane that carries one wins, and every later plane in a
    well-formed archive carries the same one. It is optional throughout -- every
    RGB document reads with ``None`` here and pays nothing.
    """

    def __init__(self, size):
        self.size = size
        self.palette = None

    def attach(self, layer, data: bytes) -> None:
        got = _read_indexed_png(data, self.size)
        if got is None:
            return
        indices, palette = got
        layer.indices = indices
        if self.palette is None:
            self.palette = palette


def _finish_colour(doc, mode: str, transparent: int, reader, palette) -> None:
    """Install the colour mode once the layers exist, and make it consistent.

    Last rather than first because it needs the document: the planes have to be
    on layers before the table can be applied to them, and a half-built indexed
    document is exactly the state ``check_materialized`` exists to catch.

    Two degradations are deliberate. A file whose block says indexed but which
    carries **no table at all** stays RGB with a log line -- an indexed document
    with nothing to index onto is not a document. A file whose block says
    indexed but whose *planes* are RGBA (an old build re-saved it) has its
    indices re-inferred, which is exactly the forward-compat cost this format
    states: the picture is intact and the duplicate-slot identity is gone.

    The table comes from the P-PNGs where they carry one, because ``tRNS`` holds
    per-entry alpha and ``palette.gpl`` cannot. The ``.gpl`` is the fallback and
    the interop projection, never the record.
    """
    if mode == "grayscale":
        doc.color_mode = "grayscale"
        # The writer records ``palette.gpl`` for a grayscale document too --
        # the funnel's own rule is two ``if``s, not an ``elif``, because "a
        # grayscale document with a palette gets both". Dropping the table
        # here lost it permanently on the first save-open-save. ``snap=False``
        # for the RGB branch's reason: the pixels were written snapped.
        if palette:
            doc.set_palette(palette, snap=False)
        return
    if mode != "indexed":
        return
    table = (reader.palette if reader is not None else None) or palette
    if not table:
        log.warning("a document declared indexed carries no colour table; opening as RGB")
        return
    doc.palette = [tuple(colour) for colour in table]
    doc.color_mode = "indexed"
    doc.transparent_index = transparent if 0 <= transparent < len(doc.palette) else 0
    # The transparent slot's ``tRNS`` byte is *forced* to zero on the way out --
    # it has to be, or the hole is not a hole to Krita, GIMP or a browser -- so
    # reading it back literally would be reading back the writer's own
    # requirement as though it were the user's data. Canonically opaque here, in
    # the one place that knows which slot it is: the RGB (which Aseprite
    # displays and a writer must put back) survives untouched, the alpha does
    # not exist to survive, and save-open-save is a fixpoint.
    hole = doc.transparent_index
    entry = doc.palette[hole]
    doc.palette[hole] = (entry[0], entry[1], entry[2], 255)
    lut = doc._index_lut()
    layers = doc.stack if doc.anim is None else doc.anim.unique_cel_layers()
    for layer in layers:
        if layer.indices is None:
            layer.indices = ixp.resolve(layer.pixels, lut, doc.transparent_index)
        doc._rematerialize(layer, lut, notify=False)
    doc.invalidate_all()


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


def _read_tiles(zf: zipfile.ZipFile, doc, anim: Animation | None) -> None:
    """Rebuild ``doc.tilesets`` and swap the cels ``tiles.json`` names into
    :class:`~.tiles.TilemapCel`, or leave the document exactly as the grid or
    flat read already built it.

    Read after the document and its stack exist -- a still document's layers
    are already at their final positions and an animated one's cels are
    already the objects ``anim.cels`` and ``doc.stack`` share, so a name in
    ``tiles.json`` resolves straight onto them. Its own guard, outside every
    other member's, for ``_read_slices``'s reason one level further: tile
    *structure* is metadata about a picture that is already fully and
    honestly on the canvas -- every tilemap cel's own PNG decoded RGBA,
    complete -- so a wrong version or any single entry failing on its own
    terms drops the whole member and costs the structure alone. Nothing is
    written onto ``doc`` until every entry in the member has validated
    cleanly, so a failure partway through leaves the document exactly as it
    already was, never half-swapped.

    On success every named cel is replaced **in place**: the same uid, and
    the same *object* wherever it is linked, which is what lets a linked cel
    stay linked. Trusting the file's own pixels would let stale ones survive
    a hand-edited archive, so every replaced cel is re-materialized from its
    own refs afterwards -- refs are authoritative once this runs, the
    funnel's own invariant restated for the one path that does not go through
    it.
    """
    try:
        raw = zf.read(TILES_MEMBER)
    except KeyError:
        return
    try:
        payload = json.loads(raw)
        if int(payload.get("version", 0)) != TILES_VERSION:
            raise ValueError(f"{TILES_MEMBER} version {payload.get('version')!r}")

        slots: list[TilesetSlot] = []
        for entry in payload["tilesets"]:
            with pixelguard.opened(
                io.BytesIO(zf.read(entry["data"])), "a tileset in this drawing"
            ) as im:
                im.load()
                pixels = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
            tileset = Tileset(
                name=str(entry.get("name") or "tiles"),
                pixels=pixels,
                tile_w=int(entry["tile_w"]),
                tile_h=int(entry["tile_h"]),
            )
            slots.append(TilesetSlot(tileset=tileset))

        track_binds: list[tuple[int, int]] = []
        for entry in payload.get("tracks", []):
            ti, si = int(entry["track"]), int(entry["tileset"])
            if anim is None or not (0 <= ti < len(anim.tracks) and 0 <= si < len(slots)):
                raise ValueError(f"{TILES_MEMBER} track {ti} names tileset {si}")
            track_binds.append((ti, si))

        def _new_cel(layer: Layer, entry: dict) -> TilemapCel:
            si = int(entry["tileset"])
            if not 0 <= si < len(slots):
                raise ValueError(f"{TILES_MEMBER} cel names tileset {si}")
            grid_h, grid_w = int(entry["grid_h"]), int(entry["grid_w"])
            blob = zf.read(entry["refs"])
            expected = grid_h * grid_w * 4
            if len(blob) != expected:
                raise ValueError(
                    f"{entry['refs']} is {len(blob)} bytes, expected {expected}"
                )
            refs = np.frombuffer(blob, dtype="<u4").reshape(grid_h, grid_w).astype(gid.DTYPE)
            ts = slots[si].tileset
            if ts.tile_w != ts.tile_h and (refs & gid.DTYPE(gid.FLIP_D)).any():
                # The refs door's own mask (``_doc_tiles._strip_diagonal``),
                # applied to what a file carries: a diagonal flip of a
                # non-square tile has the wrong footprint, and a commit over
                # one reads neighbour pixels back into the atlas. A file from
                # before the door was sealed degrades to the unturned
                # placement, the member's own log-and-keep contract.
                log.warning(
                    "dropping diagonal flips on a %dx%d tileset in %s",
                    ts.tile_w,
                    ts.tile_h,
                    getattr(zf, "filename", "?"),
                )
                refs = refs & gid.DTYPE(0xFFFFFFFF ^ gid.FLIP_D)
            return TilemapCel(
                pixels=layer.pixels,
                name=layer.name,
                opacity=layer.opacity,
                visible=layer.visible,
                blend=layer.blend,
                alpha_lock=layer.alpha_lock,
                locked=layer.locked,
                indices=layer.indices,
                uid=layer.uid,
                refs=refs,
                tileset_uid=slots[si].uid,
            )

        replacements: dict[int, TilemapCel] = {}
        if anim is not None:
            # ``_cel_names`` recomputed, not passed in: it is a pure function
            # of ``anim.unique_cel_layers()``, which walks in frame-then-track
            # order -- the same deterministic walk that named this exact cel
            # when the file was written, so the names agree without anything
            # having to be threaded through the read path to prove it.
            names = _cel_names(anim)
            by_name: dict[str, Layer] = {}
            for cel_layer in anim.unique_cel_layers():
                by_name[names[id(cel_layer)]] = cel_layer
            for entry in payload.get("cels", []):
                layer = by_name.get(entry.get("cel"))
                if layer is None:
                    raise ValueError(
                        f"{TILES_MEMBER} cel names an unknown layer: {entry.get('cel')!r}"
                    )
                replacements[id(layer)] = _new_cel(layer, entry)
        else:
            stack_layers = list(doc.stack)
            for entry in payload.get("cels", []):
                li = int(entry["layer"])
                if not 0 <= li < len(stack_layers):
                    raise ValueError(f"{TILES_MEMBER} cel names layer {li}")
                layer = stack_layers[li]
                replacements[id(layer)] = _new_cel(layer, entry)
    except (
        AttributeError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        # ``OSError`` beyond ``_read_colour``'s verbatim five, for
        # ``_read_animation``'s reason: this member's entries name embedded
        # images, and Pillow's ``UnidentifiedImageError`` on a strip member
        # holding non-image bytes is an ``OSError``.
        log.warning("ignoring %s in %s: %s", TILES_MEMBER, getattr(zf, "filename", "?"), exc)
        return

    # Everything above is read-only against ``doc`` -- what follows is the
    # atomic apply, safe now that every entry in the member has validated.
    doc.tilesets.extend(slots)
    if anim is not None:
        for ti, si in track_binds:
            anim.tracks[ti].tileset_uid = slots[si].uid
        # Every slot a linked cel occupies is re-registered, not just one --
        # ``id(cel)`` is the same key for all of them, so this is what keeps a
        # link a link rather than unlinking it into one real cel and one stale
        # copy.
        for key, cel in list(anim.cels.items()):
            new = replacements.get(id(cel))
            if new is not None:
                anim.cels[key] = new
    for i, layer in enumerate(doc.stack.layers):
        new = replacements.get(id(layer))
        if new is not None:
            doc.stack.layers[i] = new

    by_uid = {slot.uid: slot for slot in slots}
    for new_cel in replacements.values():
        slot = by_uid[new_cel.tileset_uid]
        new_cel.pixels = materialize(new_cel.refs, slot.tileset, new_cel.size)
    doc.invalidate_all()


def _parse_stack(data: bytes):
    """``stack.xml``, through the shared XML door.

    This used to be a second copy of ``plotter/tsx.py``'s four-line substring
    probe, written twice deliberately because this package imports nothing from
    ``plotter`` and the alternative was a new shared leaf for four lines. The
    leaf exists now (:mod:`..xmlguard`) because the probe was wrong in both
    copies: a UTF-16 document encodes the declaration as ``<\\x00!\\x00D...``,
    and five thousand bytes of legal prolog comment put it past byte 4096.
    Both were reproduced by direct execution. The refusal is on the parser's
    own DOCTYPE event now, which sees the declaration whatever it is spelled
    in, and the same door caps nesting depth -- a value this reader could not
    have checked afterwards, because the walkers that would trip over it run on
    the frame thread once a document is open.
    """
    return xmlguard.fromstring(data, "this drawing's stack.xml")


def read_ora(path: Path, *, budget: int | None = None):
    from .document import Document
    from .undo import UNDO_BYTES, UndoStack

    with zipguard.BoundedZip(path) as zf:
        # Before the first read, which is the only place the refusal is cheap:
        # the directory says what every member unpacks to, and a read that
        # discovers the archive lied has already spent the memory. The
        # ``wpack``/``wmap``/``wblk`` door, fourth instance.
        claimed = sum(int(info.file_size) for info in zf.infolist())
        ceiling = MAX_DECOMPRESSED_BYTES
        if claimed > ceiling:
            raise ValueError(
                f"this drawing claims {claimed} bytes unpacked, "
                f"which is past the {ceiling}-byte ceiling"
            )
        root = _parse_stack(zf.read("stack.xml"))
        # Checked here, where the numbers arrive, because nothing downstream
        # ever asks: ``_read_animation``, ``_place``/``resize_canvas`` and
        # ``Layer.empty`` all take this size as given and allocate from it, so
        # ``w="200000" h="200000"`` in a 2 KB archive is a 160 GB request that
        # the directory ceiling four lines up cannot see -- the archive is
        # honest about every byte it holds, and one 1x1 PNG is all it needs.
        width = int(root.get("w") or 0)
        height = int(root.get("h") or 0)
        pixelguard.check(width, height, "this drawing's canvas")
        # Carried, not used -- see ``Document.dpi``. Both axes or neither: a
        # file that states one and not the other is stating nothing usable, and
        # inventing the missing half would be inventing a canvas shape.
        found_dpi = _read_resolution(root)

        # JSON first, and only when the XML told us how big the canvas is: the
        # grid's cels are decoded against that size, and guessing it from the
        # first PNG would be guessing for every later one too.
        palette = _read_palette(zf)
        # Before any plane decodes, because it decides how they are read.
        mode, transparent = _read_colour(zf)
        reader = _IndexReader((width, height)) if mode == "indexed" else None
        got = _read_animation(zf, (width, height), reader) if width and height else None
        anim, grid_payload = got if got is not None else (None, None)
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
            _read_groups(doc, grid_payload)
            _read_matte(doc, root)
            doc.file_format = "ora"
            doc.dpi = found_dpi
            doc.path = Path(path)
            # ``snap=False``: the pixels in the file were written snapped, so
            # re-snapping them would cost a whole-document rewrite on every
            # open and push an undo step for opening a file.
            if mode == "rgb":
                doc.set_palette(palette, snap=False)
            else:
                _finish_colour(doc, mode, transparent, reader, palette)
            # Last: tile structure is metadata about a picture that is
            # already fully and correctly built above it.
            _read_tiles(zf, doc, anim)
            return doc

        layers: list[Layer] = []
        planes_data: list[bytes] = []
        tree: tuple[dict, dict] = ({}, {})
        parents: dict[int, int] = {}
        # From the document's own root ``<stack>``, not from ``<image>``: the
        # outer stack is the document, and reading it as a group would wrap
        # every file this reader opens in one folder called "Group 1".
        outer = root.find("stack")
        top = root if outer is None else outer
        # Reaching here with frame groups present means the animated read fell
        # back: no grid, and the nested stacks are frames rather than folders.
        collect = None if has_frame_groups(top) else tree
        for element, parent in _layer_elements(top, collect):
            src = element.get("src")
            if not src:
                continue
            try:
                data = zf.read(src)
            except KeyError:
                continue
            try:
                with pixelguard.opened(io.BytesIO(data), "a layer in this drawing") as im:
                    im.load()
                    pixels = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
            except OSError as exc:
                # A member that is not an image (Pillow's
                # ``UnidentifiedImageError`` is an ``OSError``) costs that one
                # layer, exactly as a missing member does one branch up --
                # the degradation contract every optional member here follows.
                log.warning("skipping undecodable %s in %s: %s", src, path, exc)
                continue
            planes_data.append(data)
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
                    background=element.get(BACKGROUND_ATTR) == "1",
                    reference=element.get(REFERENCE_ATTR) == "1",
                )
            )
            if parent is not None:
                # Recorded against the layer's uid only once the layer exists,
                # which is what keeps a group whose PNGs are all missing from
                # coming back as an empty one.
                parents[layers[-1].uid] = parent

        # Inside the same ``with`` as the animated branch above, rather than
        # closing it first, ``tiles.json`` needs it -- widened deliberately so
        # a still document reading it stays on the same footing as an
        # animated one.
        if not layers:
            layers = [Layer.empty(max(1, width), max(1, height), "Background")]
        layers.reverse()  # file order is top-first; ours is bottom-first
        doc = Document(
            stack=LayerStack(layers, len(layers) - 1),
            history=UndoStack(UNDO_BYTES if budget is None else budget),
            slices=found_slices,
        )
        _install_groups(doc, tree, parents)
        _read_matte(doc, root)
        doc.file_format = "ora"
        doc.dpi = found_dpi
        doc.path = Path(path)
        if mode == "rgb":
            doc.set_palette(palette, snap=False)
        else:
            if reader is not None:
                # Paired after the reverse, not inside the loop: the flat path
                # reverses its list before building the stack, so attaching in
                # file order would put every plane on the wrong layer.
                for layer, data in zip(reversed(layers), planes_data, strict=False):
                    reader.attach(layer, data)
            _finish_colour(doc, mode, transparent, reader, palette)
        _read_tiles(zf, doc, None)
        return doc

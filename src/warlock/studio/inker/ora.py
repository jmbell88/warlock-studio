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
from .animation import DEFAULT_DURATION_MS, Animation, Frame, Tag, Track
from .layers import Layer, LayerStack

THUMBNAIL_MAX = 256

#: Bumped when a reader written against version N could get an N+1 file wrong in
#: a way it cannot detect. Adding a key does not qualify -- every read below is
#: ``.get``-based -- so this has stayed at 1 through the whole of v1.
ANIMATION_VERSION = 1
ANIMATION_MEMBER = "animation.json"

log = logging.getLogger(__name__)


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
            },
        )
    return ElementTree.tostring(root, encoding="UTF-8", xml_declaration=True)


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
        "tags": [
            {"name": tag.name, "start": int(tag.start), "end": int(tag.end), "loop": bool(tag.loop)}
            for tag in anim.tags
        ],
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
            zipfile.ZipInfo("mimetype"), b"image/openraster", zipfile.ZIP_STORED
        )
        if anim is None:
            zf.writestr("stack.xml", _stack_xml(doc))
            for index, layer in enumerate(reversed(list(doc.stack))):
                zf.writestr(f"data/layer{index}.png", _png(layer.pixels))
        else:
            zf.writestr("stack.xml", _stack_xml_animated(doc, names))
            # One PNG per name, with no de-duplication needed: ``_cel_names``
            # is built from the same ``unique_cel_layers`` walk and gives each
            # distinct cel its own name, so the two can only ever agree.
            for layer in anim.unique_cel_layers():
                zf.writestr(names[id(layer)], _png(layer.pixels))
            zf.writestr(ANIMATION_MEMBER, _animation_json(doc, names))
        zf.writestr("mergedimage.png", _png(merged))
        zf.writestr("Thumbnails/thumbnail.png", thumb_buf.getvalue())
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
                blend=entry.get("blend", "normal"),
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
                )
                planes[src] = layer
            cels[(tracks[ti].uid, frames[fi].uid)] = layer

        tags = [
            Tag(
                name=entry.get("name") or "tag",
                start=int(entry.get("start", 0)),
                end=int(entry.get("end", 0)),
                loop=bool(entry.get("loop", True)),
            )
            for entry in payload.get("tags", [])
        ]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        log.warning("ignoring animation.json in %s: %s", getattr(zf, "filename", "?"), exc)
        return None
    return Animation(tracks=tracks, frames=frames, cels=cels, tags=tags, current=0)


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
        anim = _read_animation(zf, (width, height)) if width and height else None
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
            doc.matte = matte_for(doc.composite)
            doc.file_format = "ora"
            doc.path = Path(path)
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
                )
            )

    if not layers:
        layers = [Layer.empty(max(1, width), max(1, height), "Background")]
    layers.reverse()  # file order is top-first; ours is bottom-first
    doc = Document(
        stack=LayerStack(layers, len(layers) - 1),
        history=UndoStack(UNDO_BYTES if budget is None else budget),
    )
    doc.matte = matte_for(doc.composite)
    doc.file_format = "ora"
    doc.path = Path(path)
    return doc

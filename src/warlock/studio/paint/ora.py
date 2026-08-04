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
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from . import composite as cp
from .layers import Layer, LayerStack

THUMBNAIL_MAX = 256


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


def write_ora(doc, path: Path) -> None:
    """Blocking; callers encode on a task thread."""
    from PIL import Image

    path = Path(path)
    merged = doc.flatten()
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
        zf.writestr("stack.xml", _stack_xml(doc))
        for index, layer in enumerate(reversed(list(doc.stack))):
            zf.writestr(f"data/layer{index}.png", _png(layer.pixels))
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


def read_ora(path: Path, *, budget: int | None = None):
    from PIL import Image

    from .document import Document, matte_for
    from .undo import UNDO_BYTES, UndoStack

    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
        width = int(root.get("w") or 0)
        height = int(root.get("h") or 0)

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

"""OpenRaster round-trips, and the spec details other readers depend on.

The interop assertions are not pedantry: ``mimetype`` first and stored is how a
reader recognises the file at all, and a missing ``mergedimage.png`` makes an
ORA that Krita opens as blank.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from warlock.studio import inker

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(size=(8, 8)):
    doc = inker.Document.blank(*size)
    doc.stack[0].pixels[:, :] = RED
    doc.stack[0].name = "Background"
    layer = doc.add_layer("Top")
    layer.pixels[0:4, 0:4] = BLUE
    layer.opacity = 0.5
    layer.blend = "multiply"
    doc.invalidate_all()
    return doc


def _saved(tmp_path: Path) -> Path:
    path = tmp_path / "doc.ora"
    inker.write_ora(_doc(), path)
    return path


# --- the round trip ---------------------------------------------------------


def test_every_layer_comes_back_byte_for_byte(tmp_path: Path):
    original = _doc()
    path = tmp_path / "doc.ora"
    inker.write_ora(original, path)
    reopened = inker.Document.load(path)
    assert len(reopened.stack) == len(original.stack)
    for before, after in zip(original.stack, reopened.stack, strict=True):
        assert np.array_equal(before.pixels, after.pixels)


def test_layer_names_opacity_visibility_and_blend_all_survive(tmp_path: Path):
    reopened = inker.Document.load(_saved(tmp_path))
    names = [layer.name for layer in reopened.stack]
    assert names == ["Background", "Top"]
    assert reopened.stack[1].opacity == 0.5
    assert reopened.stack[1].blend == "multiply"
    assert all(layer.visible for layer in reopened.stack)


def test_a_hidden_layer_stays_hidden(tmp_path: Path):
    doc = _doc()
    doc.stack[1].visible = False
    path = tmp_path / "doc.ora"
    inker.write_ora(doc, path)
    assert inker.Document.load(path).stack[1].visible is False


def test_the_reopened_composite_is_the_one_that_was_saved(tmp_path: Path):
    original = _doc()
    path = tmp_path / "doc.ora"
    inker.write_ora(original, path)
    assert np.array_equal(inker.Document.load(path).composite, original.composite)


def test_a_reopened_document_knows_it_is_an_ora(tmp_path: Path):
    doc = inker.Document.load(_saved(tmp_path))
    assert doc.file_format == "ora"
    assert doc.path is not None and doc.path.suffix == ".ora"


def test_ora_bytes_is_the_same_file_without_a_path(tmp_path: Path):
    data = inker.ora_bytes(_doc())
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert "stack.xml" in zf.namelist()


# --- what other readers depend on -------------------------------------------


def test_the_mimetype_entry_is_first_and_uncompressed(tmp_path: Path):
    """It is a magic number read at a fixed offset; deflated, it is not one."""
    with zipfile.ZipFile(_saved(tmp_path)) as zf:
        first = zf.infolist()[0]
        assert first.filename == "mimetype"
        assert first.compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == b"image/openraster"


def test_a_merged_image_and_a_thumbnail_are_both_written(tmp_path: Path):
    with zipfile.ZipFile(_saved(tmp_path)) as zf:
        names = zf.namelist()
        assert "mergedimage.png" in names
        assert "Thumbnails/thumbnail.png" in names
        with Image.open(io.BytesIO(zf.read("mergedimage.png"))) as merged:
            assert merged.size == (8, 8)


def test_the_stack_lists_the_top_layer_first(tmp_path: Path):
    """ORA's document order is the painter's, reversed. Getting this backwards
    round-trips perfectly here and shows every other editor an upside-down
    stack."""
    with zipfile.ZipFile(_saved(tmp_path)) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    layers = root.find("stack").findall("layer")
    assert [element.get("name") for element in layers] == ["Top", "Background"]


def test_blend_modes_are_written_as_svg_composite_ops(tmp_path: Path):
    with zipfile.ZipFile(_saved(tmp_path)) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    ops = [element.get("composite-op") for element in root.iter("layer")]
    assert ops == ["svg:multiply", "svg:src-over"]


def test_the_image_element_carries_the_canvas_size(tmp_path: Path):
    with zipfile.ZipFile(_saved(tmp_path)) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    assert (root.get("w"), root.get("h")) == ("8", "8")


# --- tolerance when reading -------------------------------------------------


def _foreign(path: Path, layers: str, *, w=8, h=8, extra=None) -> None:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[:, :] = RED
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buf, "PNG")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr(
            "stack.xml",
            f'<image version="0.0.3" w="{w}" h="{h}"><stack>{layers}</stack></image>',
        )
        zf.writestr("data/a.png", buf.getvalue())
        for name, data in (extra or {}).items():
            zf.writestr(name, data)


def test_a_composite_op_we_cannot_reproduce_becomes_normal(tmp_path: Path):
    """A file that opens slightly wrong is a file the user still has."""
    path = tmp_path / "foreign.ora"
    _foreign(path, '<layer name="L" src="data/a.png" composite-op="svg:color-dodge"/>')
    assert inker.Document.load(path).stack[0].blend == "normal"


def test_a_layer_offset_is_pasted_rather_than_refused(tmp_path: Path):
    path = tmp_path / "offset.ora"
    _foreign(path, '<layer name="L" src="data/a.png" x="2" y="2"/>', w=16, h=16)
    doc = inker.Document.load(path)
    assert doc.size == (16, 16)
    assert doc.stack[0].size == (16, 16)
    assert tuple(doc.stack[0].pixels[3, 3]) == RED
    assert int(doc.stack[0].pixels[0, 0][3]) == 0


def test_a_nested_stack_is_flattened_into_the_layer_list(tmp_path: Path):
    """We have no group layers; dropping a group would lose most of a Krita
    file rather than a little of it."""
    path = tmp_path / "grouped.ora"
    _foreign(
        path,
        '<stack name="G"><layer name="inner" src="data/a.png"/></stack>'
        '<layer name="outer" src="data/a.png"/>',
    )
    doc = inker.Document.load(path)
    assert sorted(layer.name for layer in doc.stack) == ["inner", "outer"]


def test_a_layer_whose_file_is_missing_is_skipped_not_fatal(tmp_path: Path):
    path = tmp_path / "broken.ora"
    _foreign(
        path,
        '<layer name="gone" src="data/missing.png"/><layer name="here" src="data/a.png"/>',
    )
    doc = inker.Document.load(path)
    assert [layer.name for layer in doc.stack] == ["here"]


def test_an_ora_with_no_layers_at_all_still_opens(tmp_path: Path):
    path = tmp_path / "empty.ora"
    _foreign(path, "")
    doc = inker.Document.load(path)
    assert len(doc.stack) == 1
    assert doc.size == (8, 8)


def test_the_topmost_layer_is_the_active_one_when_a_document_opens(tmp_path: Path):
    doc = inker.Document.load(_saved(tmp_path))
    assert doc.stack.active_index == len(doc.stack) - 1
    assert doc.stack.active.name == "Top"

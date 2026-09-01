"""What a hostile file gets to allocate before anything asks how big it is.

Every document format this app opens is a file somebody was *handed*. The
2026-08-24 sweep closed the archive-level half of that -- ``zipguard`` at the
four zip doors, ``asein._inflate``, ``tmx._decompress`` -- and each of those
bounds a member against what its own container declared. What this file is
about is the layer *below* that: a number inside the payload that decides an
allocation, checked afterwards or not at all.

The shape is the same at every door and it is why these cases live together
rather than beside the format each one names. A file of a few hundred bytes
states a size, something allocates from it, and the check that would have
refused it runs on the line after -- so the refusal arrives correctly, once the
memory is already spent. The fixtures here are all tiny for exactly that
reason: if any of these tests could only be written with a large file, the bug
it pins would not be a bug.

Nothing here builds a real bomb. Every ceiling is module-level so a test can
lower it, which is the rule ``zipguard.BoundedZip.ceiling`` already states, and
the cases that cannot lower a ceiling declare an absurd *number* rather than
producing an absurd *file*.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from warlock.glbio import CHUNK_BIN, CHUNK_JSON, GLB_MAGIC
from warlock.studio.inker import asein, ora, sheetout
from warlock.studio.plotter import tmx, tsx, wmap
from warlock.studio.tilegrid.tileset import Tileset
from warlock.studio.viewer import gltf

# --- fixtures -----------------------------------------------------------------


def _pixels(side: int = 32) -> np.ndarray:
    array = np.zeros((side, side, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def _tsx_loader(_source: str) -> Tileset:
    return Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16)


def _image_loader(_source: str) -> np.ndarray:
    return _pixels()


LOADERS = {"image_loader": _image_loader, "tsx_loader": _tsx_loader}


def _glb(document: dict, binary: bytes = b"") -> bytes:
    """A GLB with whatever JSON is handed over, valid or not.

    ``glbwrite.write_glb`` cannot build these: it writes a *correct* file from
    a model, and every case here is about a file whose JSON says something no
    writer would say.
    """
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    body = struct.pack("<II", len(payload), CHUNK_JSON) + payload
    if binary:
        padded = binary + b"\x00" * (-len(binary) % 4)
        body += struct.pack("<II", len(padded), CHUNK_BIN) + padded
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(body)) + body


def _npy(shape: tuple[int, ...], dtype: str) -> bytes:
    """A ``.npy`` **header** for an array nobody could hold, and no data.

    The whole point: the header is 128 bytes whatever it claims, and it is the
    header that decides the allocation. ``write_array`` would have to build the
    array first, which is the thing being refused.
    """
    out = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        out, {"descr": dtype, "fortran_order": False, "shape": shape}
    )
    return out.getvalue()


def _ora(stack: str, members: dict[str, bytes] | None = None) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("mimetype", "image/openraster")
        zf.writestr("stack.xml", stack)
        for name, data in (members or {}).items():
            zf.writestr(name, data)
    return out.getvalue()


def _aseprite(width: int, height: int) -> bytes:
    """A 128-byte Aseprite header and one empty frame."""
    head = bytearray(128)
    struct.pack_into("<IHHHHHI", head, 0, 128 + 16, 0xA5E0, 1, width, height, 32, 0)
    frame = struct.pack("<IHHHHI", 16, 0xF1FA, 0, 100, 0, 0)
    return bytes(head) + frame


def _map(body: str, attrs: str = "") -> bytes:
    return (
        f'<map version="1.10" orientation="orthogonal" width="2" height="2"'
        f' tilewidth="16" tileheight="16" {attrs}>'
        f'<tileset firstgid="1" source="t.tsx"/>{body}</map>'
    ).encode()


# --- 6a: allocating from a declared number ------------------------------------


def test_a_glb_accessor_with_no_buffer_view_cannot_declare_a_terabyte():
    """The one accessor path that allocates without touching the buffer.

    An accessor with no ``bufferView`` legally reads as zeros, so the loader
    returns ``np.zeros((count, ncomp))`` -- and ``_check_span``, which bounds
    every other path against the length of the BIN chunk, never runs. This file
    is under a kilobyte and used to ask numpy for 64 GB.
    """
    document = {
        "accessors": [{"componentType": 5126, "type": "MAT4", "count": 1_000_000_000}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}],
    }
    assert len(_glb(document)) < 1024
    with pytest.raises(ValueError, match="allocate"):
        gltf.load(_glb(document))


def test_a_wmap_layer_is_sized_from_its_npy_header_before_it_is_read():
    """``read_array`` sizes the buffer from the header; the shape check that
    would refuse it is the line after."""
    doc = wmap.read_wmap(_wmap_with_layer(None))
    assert doc.layers  # the honest file still opens
    hostile = _wmap_with_layer(_npy((1 << 20, 1 << 20), "<u4"))
    assert len(hostile) < 4096
    with pytest.raises(ValueError, match="allocate"):
        wmap.read_wmap(hostile)


def _wmap_with_layer(payload: bytes | None) -> bytes:
    """A real ``.wmap``, with its layer member replaced by *payload* or left be."""
    from warlock.studio.plotter.tilemap import MapDoc

    doc = MapDoc(2, 2, 16, 16)
    doc.add_tile_layer("L")
    data = wmap.wmap_bytes(doc)
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            body = src.read(info.filename)
            if payload is not None and info.filename.endswith(".npy"):
                body = payload
            dst.writestr(info.filename, body)
    return out.getvalue()


def test_an_npz_inside_a_wblk_is_read_through_the_bounded_zip():
    """``np.load`` on an ``.npz`` opens a nested zip with numpy's own plain
    ``zipfile``, so the outer ``BoundedZip`` is not in the path at all."""
    from warlock.studio import npyguard

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("verts.npy", _npy((1 << 25, 4), "<f8"))
    with pytest.raises(ValueError, match="allocate"):
        npyguard.read_npz(inner.getvalue(), "a mesh")


def test_an_npy_declaring_object_dtype_is_refused_by_name():
    """``allow_pickle=False``'s refusal, made from the header instead."""
    from warlock.studio import npyguard

    with pytest.raises(ValueError, match="unpickle"):
        npyguard.read_array(_npy((2,), "|O"), "a mesh")


def test_an_ora_canvas_size_has_a_ceiling(tmp_path):
    """``w`` and ``h`` from ``stack.xml`` feed ``Layer.empty`` and
    ``resize_canvas``, and nothing downstream asks a second time."""
    path = tmp_path / "big.ora"
    path.write_bytes(_ora('<image w="200000" h="200000"><stack/></image>'))
    assert path.stat().st_size < 4096
    with pytest.raises(ValueError, match="pixels"):
        ora.read_ora(path)


def test_an_ora_layer_count_has_a_ceiling(tmp_path, monkeypatch):
    """``pixelguard`` bounds one canvas and ``zipguard`` bounds the claimed
    bytes; neither can see the *product*. Every ``<layer>`` is placed onto the
    canvas, so a fifteen-kilobyte archive naming five hundred layers of one
    tiny PNG asks for five hundred full canvases."""
    from PIL import Image

    from warlock.studio.inker import transform

    tiny = io.BytesIO()
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(tiny, "PNG")
    count = 2000
    layers = "".join(f'<layer name="L{i}" src="t.png" x="0" y="0"/>' for i in range(count))
    path = tmp_path / "many.ora"
    path.write_bytes(
        _ora(
            f'<image w="2048" h="2048"><stack>{layers}</stack></image>',
            {"t.png": tiny.getvalue()},
        )
    )
    assert path.stat().st_size < 1 << 20

    # The refusal has to land *before* the allocations, not after: counting
    # ``resize_canvas`` calls is what says so, since a bound that only checked
    # the total at the end would already have spent the memory.
    placed = 0
    real = transform.resize_canvas

    def counted(*args, **kwargs):
        nonlocal placed
        placed += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(transform, "resize_canvas", counted)
    with pytest.raises(ValueError, match="layers"):
        ora.read_ora(path)
    assert placed < count


def test_an_aseprite_canvas_size_has_a_ceiling():
    """Both fields are u16 and were checked only for ``< 1``; 65535 squared is
    17 GB on the first drawable row."""
    data = _aseprite(65535, 65535)
    assert len(data) < 256
    with pytest.raises(ValueError, match="pixels"):
        asein.parse(data)
    # The floor still refuses what it always refused.
    with pytest.raises(ValueError, match="not one to draw on"):
        asein.parse(_aseprite(0, 8))


def test_a_gif_is_bounded_by_composed_pixels_and_not_by_its_file_size(
    tmp_path, monkeypatch
):
    """GIF stores deltas, so the file says nothing about what composing it
    costs. The budget is spent on what is built."""
    from PIL import Image

    from warlock.studio import pixelguard
    from warlock.studio.inker import gifin

    frames = [Image.new("RGBA", (8, 8), (i, 0, 0, 255)) for i in range(6)]
    path = tmp_path / "clip.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50)
    assert len(gifin.frames_of_gif(path)[0]) == 6
    # One canvas' worth of pixels, times as many frames as fit in it.
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 8 * 8 * 3)
    with pytest.raises(ValueError, match="frames"):
        gifin.frames_of_gif(path)


def test_layer_data_stops_one_cell_past_the_declaration():
    """The refusal has to arrive *before* the cells are built, not after.

    A generator that records what it was asked for is the only way to see the
    difference: the old spelling built one Python string and one Python int per
    cell for the whole of ``<data>`` and then asked ``_gid_array`` whether the
    count matched. Four cells declared, five taken -- the fifth is what proves
    the layer carries too many, which is ``BoundedZip``'s declared-plus-one
    trick in another format.
    """
    asked: list[int] = []

    def cells():
        for i in range(100_000):
            asked.append(i)
            yield 0

    with pytest.raises(ValueError, match="carries"):
        tmx._gid_array(cells(), 2, 2)
    assert len(asked) == 5


def test_a_csv_layer_carrying_too_many_cells_is_still_refused():
    """The end-to-end half. It passes both ways on purpose -- ``_gid_array``
    always refused a wrong count, and this guards the property the lazy
    rewrite could have broken."""
    cells = ",".join("0" for _ in range(4096))
    data = _map(f'<layer id="1" name="L" width="2" height="2">'
                f'<data encoding="csv">{cells}</data></layer>')
    with pytest.raises(ValueError, match="carries"):
        tmx.read_tmx(data, **LOADERS)


def test_the_honest_csv_layer_still_reads():
    """The lazy spelling has to be the same reader for a file that is fine."""
    data = _map('<layer id="1" name="L" width="2" height="2">'
                "<data encoding=\"csv\">0,0,\n0,0</data></layer>")
    doc = tmx.read_tmx(data, **LOADERS)
    assert doc.layers[0].data.shape == (2, 2)


# --- 6b: per-item bounds with no item count -----------------------------------


def test_a_map_cannot_declare_more_layers_than_this_build_reads(monkeypatch):
    """``_decompress`` is correct per layer and nothing capped how many."""
    monkeypatch.setattr(tmx, "MAX_LAYERS", 3)
    body = "".join(
        f'<layer id="{i}" name="L{i}" width="2" height="2">'
        '<data encoding="csv">0,0,0,0</data></layer>'
        for i in range(6)
    )
    with pytest.raises(ValueError, match="layers"):
        tmx.read_tmx(_map(body), **LOADERS)


def test_the_layer_budget_counts_across_the_group_recursion(monkeypatch):
    """A budget checked per call would be a budget per group."""
    monkeypatch.setattr(tmx, "MAX_LAYERS", 4)
    leaf = ('<layer id="9" name="L" width="2" height="2">'
            '<data encoding="csv">0,0,0,0</data></layer>')
    body = f"<group>{leaf}</group><group>{leaf}</group><group>{leaf}</group>"
    with pytest.raises(ValueError, match="layers"):
        tmx.read_tmx(_map(body), **LOADERS)


def test_a_map_cannot_declare_more_chunks_than_this_build_reads(monkeypatch):
    monkeypatch.setattr(tmx, "MAX_CHUNKS", 2)
    chunks = "".join(
        f'<chunk x="{i * 16}" y="0" width="2" height="2">0,0,0,0</chunk>'
        for i in range(4)
    )
    body = (f'<layer id="1" name="L" width="2" height="2">'
            f'<data encoding="csv">{chunks}</data></layer>')
    with pytest.raises(ValueError, match="chunks"):
        tmx.read_tmx(_map(body, attrs='infinite="1"'), **LOADERS)


def test_a_tmj_chunk_side_goes_through_the_same_cap_as_a_tmx_one():
    """The JSON half had the cap written beside it and never called it."""
    payload = {
        "type": "map",
        "version": "1.10",
        "orientation": "orthogonal",
        "renderorder": "right-down",
        "infinite": True,
        "width": 2,
        "height": 2,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [
            {
                "type": "tilelayer",
                "name": "L",
                "id": 1,
                "chunks": [
                    {"x": 0, "y": 0, "width": 4_000_000_000, "height": 1, "data": []}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="cells a side"):
        tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)


# --- 6c/6d: the XML door ------------------------------------------------------


_LAUGHS = (
    '<!DOCTYPE map [<!ENTITY a "AAAAAAAAAA">'
    '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
)
_TSX_BODY = '<tileset name="t" tilewidth="16" tileheight="16"><image source="a.png"/></tileset>'


def test_a_dtd_in_utf16_is_refused_although_the_byte_probe_never_saw_it():
    """The substring probe this replaced looked for ``b"<!DOCTYPE"``. A UTF-16
    document spells it ``<\\x00!\\x00D\\x00...`` and expat parses it happily."""
    data = ('<?xml version="1.0" encoding="UTF-16"?>' + _LAUGHS + _TSX_BODY).encode(
        "utf-16"
    )
    assert b"<!DOCTYPE" not in data[:4096].upper()
    with pytest.raises(ValueError, match="DTD"):
        tsx.tsx_source(data)


def test_a_dtd_behind_a_padded_prolog_is_refused():
    """A comment is legal prolog content, so 5,000 bytes of one put the
    declaration past byte 4096 -- which is the sentence the old docstring rested
    on."""
    data = (b"<!-- " + b"x" * 5000 + b" -->" + (_LAUGHS + _TSX_BODY).encode())
    assert b"<!DOCTYPE" not in data[:4096].upper()
    with pytest.raises(ValueError, match="DTD"):
        tsx.tsx_source(data)


def test_an_ora_stack_hides_its_dtd_the_same_two_ways(tmp_path):
    """One leaf for both doors, which is what makes fixing it once enough."""
    for stack in (
        ('<?xml version="1.0" encoding="UTF-16"?>'
         '<!DOCTYPE image [<!ENTITY a "b">]><image w="8" h="8"><stack/></image>'
         ).encode("utf-16"),
        (b"<!-- " + b"x" * 5000 + b" -->"
         + b'<!DOCTYPE image [<!ENTITY a "b">]><image w="8" h="8"><stack/></image>'),
    ):
        path = tmp_path / "dtd.ora"
        path.write_bytes(_ora(stack))
        with pytest.raises(ValueError, match="DTD"):
            ora.read_ora(path)


def test_xml_nesting_is_capped_at_the_door(monkeypatch):
    """``ET.fromstring`` will build a 20,001-deep tree out of 300 KB, and the
    recursion limit is 1000."""
    from warlock.studio import xmlguard

    monkeypatch.setattr(xmlguard, "MAX_DEPTH", 8)
    deep = b"<a>" + b"<b>" * 20 + b"</b>" * 20 + b"</a>"
    with pytest.raises(ValueError, match="deep"):
        xmlguard.fromstring(deep, "a document")
    assert xmlguard.fromstring(b"<a><b><c/></b></a>", "a document").tag == "a"


def test_a_deeply_nested_tmx_is_refused_rather_than_opened():
    """The nastier half is the *shallow* one: a document nested a few hundred
    deep loads, and then the frame-thread walkers blow up once a frame."""
    from warlock.studio import xmlguard

    depth = xmlguard.MAX_DEPTH + 5
    body = "<group>" * depth + "</group>" * depth
    data = _map(body)
    assert len(data) < 4096
    with pytest.raises(ValueError, match="deep"):
        tmx.read_tmx(data, **LOADERS)


def test_the_plotter_open_path_frames_a_recursion_error(tmp_path, monkeypatch):
    """``_load`` caught only ``ValueError``, so a ``RecursionError`` from any
    walker left the task thread raw."""
    from warlock.service.errors import ServiceError
    from warlock.studio import plotter_io

    path = tmp_path / "m.wmap"
    path.write_bytes(b"not a map")

    def boom(*_args, **_kwargs):
        raise RecursionError("too deep")

    monkeypatch.setattr("warlock.studio.plotter.wmap.read_wmap", boom)
    with pytest.raises(ServiceError, match="nested deeper"):
        plotter_io._load(path)


# --- 6e/6f: the ceilings that existed and were not wired ----------------------


def test_clay_refuses_a_document_past_its_own_ceiling(tmp_path, monkeypatch):
    """Clay had no size ceiling anywhere, though the number has existed since
    the format did -- applied at the upload and at neither door a user reaches."""
    from warlock.service.errors import TooLarge
    from warlock.studio import clay_mode

    path = tmp_path / "big.wblk"
    path.write_bytes(b"x" * 4096)
    monkeypatch.setattr("warlock.service.files.MAX_CLAY_SOURCE_BYTES", 1024)
    with pytest.raises(TooLarge):
        clay_mode._load(path)


def test_inker_refuses_an_aseprite_past_its_ceiling(tmp_path, monkeypatch):
    from warlock.service.errors import TooLarge
    from warlock.studio import inker_mode

    path = tmp_path / "big.aseprite"
    path.write_bytes(_aseprite(8, 8))
    monkeypatch.setattr("warlock.service.files.MAX_INKER_BYTES", 8)
    with pytest.raises(TooLarge):
        inker_mode._load_aseprite(path)


def test_the_pixel_ceiling_is_asked_before_convert(monkeypatch):
    """``packwright/wpack.py``'s rule, verbatim, at the door every other mode
    reaches an image through. Pillow's own default only *warns* between one and
    two times itself, and nothing in this repo ever set it."""
    from PIL import Image

    from warlock.studio import pixelguard

    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "PNG")
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 1024)
    with pytest.raises(ValueError, match="pixels"):
        pixelguard.decode_rgba(io.BytesIO(buf.getvalue()), "a picture")


def test_the_pixel_ceiling_reaches_the_shared_document_decoder(tmp_path, monkeypatch):
    from PIL import Image

    from warlock.studio import docmodes, pixelguard

    path = tmp_path / "p.png"
    Image.new("RGB", (64, 64)).save(path)
    assert docmodes.decode_rgba(path).shape == (64, 64, 4)
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 1024)
    with pytest.raises(ValueError, match="pixels"):
        docmodes.decode_rgba(path)


# --- 6g: the two contained ones -----------------------------------------------


def test_an_alternate_data_stream_source_is_refused(tmp_path):
    """``PureWindowsPath("sheet.png:secret").drive`` is ``''``, so the
    absolute/UNC filter passed an NTFS stream straight through."""
    from warlock.studio import plotter_io

    with pytest.raises(ValueError, match="colon"):
        plotter_io._resolve_source(tmp_path, "sheet.png:$DATA")
    # The documented ``..`` trade-off is unchanged.
    assert plotter_io._resolve_source(tmp_path, "../t/g.tsx") == tmp_path / "../t/g.tsx"


def test_an_overlong_composed_path_is_a_framed_refusal(tmp_path):
    """It used to raise a bare ``OSError`` from the open, which leaves this
    module's refusal contract by the back door."""
    from warlock.studio import plotter_io

    with pytest.raises(ValueError, match="too long"):
        plotter_io._resolve_source(tmp_path, "a" * 300 + ".png")


def test_two_export_names_that_differ_only_in_case_are_refused():
    """NTFS and APFS are both case-insensitive by default, so a split export
    over ``Walk`` and ``walk`` wrote one file over the other -- exactly what
    this function's docstring says it exists to prevent."""
    with pytest.raises(ValueError, match="one filename"):
        sheetout.require_distinct_names(["Walk.png", "walk.png"])
    with pytest.raises(ValueError, match="both be called"):
        sheetout.require_distinct_names(["walk.png", "walk.png"])
    sheetout.require_distinct_names(["walk.png", "run.png"])


def test_two_export_names_that_differ_only_in_normalisation_are_refused():
    """``sanitize_stem`` preserves non-ASCII, so a tag typed on macOS arrives
    decomposed and one typed on Windows composed."""
    composed = "é.png"
    decomposed = "é.png"
    assert composed != decomposed
    with pytest.raises(ValueError, match="one filename"):
        sheetout.require_distinct_names([composed, decomposed])


def test_the_guard_leaves_are_where_the_engines_can_reach_them():
    """The vacuous-pass guard for the file: every case above imports through a
    format module, so a renamed leaf would fail here rather than everywhere."""
    from warlock.studio import npyguard, pixelguard, xmlguard

    for module in (npyguard, pixelguard, xmlguard):
        assert Path(module.__file__).parent.name == "studio"

"""OpenRaster round-trips, and the spec details other readers depend on.

The interop assertions are not pedantry: ``mimetype`` first and stored is how a
reader recognises the file at all, and a missing ``mergedimage.png`` makes an
ORA that Krita opens as blank.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from warlock.studio import inker
from warlock.studio.inker import animation
from warlock.studio.inker import ora as inker_ora
from warlock.studio.inker.animation import Tag

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


def test_every_blend_mode_survives_a_real_file_round_trip(tmp_path: Path):
    """One layer per mode, written and reopened.

    ``ORA_OPS``/``OPS_ORA`` agreeing with each other is one test; this is the
    other half -- that the op string actually reaches ``stack.xml`` and comes
    back as the same mode through the encoder and the parser, which is what a
    user opening their file in Krita is relying on.
    """
    from warlock.studio.inker import composite as cp

    doc = inker.Document.blank(8, 8)
    for index, mode in enumerate(cp.BLEND_MODES):
        if index >= len(doc.stack):
            doc.add_layer(f"L{index}")
        doc.stack[index].blend = mode
    path = tmp_path / "modes.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    written = {element.get("composite-op") for element in root.iter("layer")}
    assert written == {cp.ORA_OPS[mode] for mode in cp.BLEND_MODES}

    reopened = inker.Document.load(path)
    assert [layer.blend for layer in reopened.stack] == [
        layer.blend for layer in doc.stack
    ]


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
    """A file that opens slightly wrong is a file the user still has.

    The example has to be a mode we genuinely cannot reproduce, or this asserts
    the fallback of something that no longer falls back -- which is what
    happened to the previous one: ``svg:hue`` stood here until the four
    non-separable modes were implemented. ``krita:dissolve`` is not a blend of
    two colours at all (it is a per-pixel coin toss on alpha), so it is a
    genuinely absent mode rather than one nobody has got to yet.
    """
    path = tmp_path / "foreign.ora"
    _foreign(path, '<layer name="L" src="data/a.png" composite-op="krita:dissolve"/>')
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


# --- animated documents ------------------------------------------------------


def _animated(size=(8, 8)):
    """Two frames over two tracks: a linked background and a per-frame top."""
    doc = inker.Document.blank(*size)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    doc.add_layer("Top")
    doc.invalidate_all()
    doc.ensure_animation()
    anim = doc.anim
    anim.frames[0].duration_ms = 40
    doc.set_active_layer(1)
    doc.write_colour((0, 0, 4, 4), BLUE, np.ones((4, 4), dtype=np.float32))
    # Frame two: the background stays *linked* (one object, two slots) while the
    # top gets a cel of its own, painted differently.
    doc.add_frame(link=True)
    doc.set_frame_duration(1, 120)
    doc.unlink_cel(track_index=1, frame_index=1)
    doc.set_current_frame(1)
    doc.set_active_layer(1)
    doc.write_colour((4, 4, 8, 8), BLUE, np.ones((4, 4), dtype=np.float32))
    doc.anim.tags.append(Tag(name="walk", start=0, end=1, loop=True))
    doc.set_current_frame(0)
    return doc


def test_an_animated_document_round_trips_its_whole_grid(tmp_path: Path):
    original = _animated()
    path = tmp_path / "anim.ora"
    inker.write_ora(original, path)

    doc = inker.Document.load(path)

    assert doc.animated
    assert [f.duration_ms for f in doc.anim.frames] == [40, 120]
    assert [t.name for t in doc.anim.tracks] == ["Background", "Top"]
    assert [(t.name, t.start, t.end, t.loop) for t in doc.anim.tags] == [
        ("walk", 0, 1, True)
    ]
    assert len(doc.anim.cels) == len(original.anim.cels)


def test_a_link_is_still_a_link_after_a_save_and_a_reload(tmp_path: Path):
    """Decoding per slot would give equal pixels in separate objects. The break
    only shows on the next stroke -- one frame changes and the other does not --
    which arrives as a bug report about drawing, three sessions later."""
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)

    anim = inker.Document.load(path).anim
    back, top = anim.tracks[0], anim.tracks[1]

    assert anim.cel(back.uid, anim.frames[0].uid) is anim.cel(
        back.uid, anim.frames[1].uid
    )
    # ...and the top track, deliberately unlinked, is still not shared.
    assert anim.cel(top.uid, anim.frames[0].uid) is not anim.cel(
        top.uid, anim.frames[1].uid
    )


def test_a_linked_cel_is_stored_once(tmp_path: Path):
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)

    with zipfile.ZipFile(path) as zf:
        cels = [name for name in zf.namelist() if name.startswith("data/cel")]
        payload = json.loads(zf.read("animation.json"))

    # Three distinct cels (one shared background, two tops) across four slots.
    assert len(cels) == 3
    assert len(payload["cels"]) == 4
    assert len({c["data"] for c in payload["cels"] if c["track"] == 0}) == 1


def test_frames_after_the_first_are_written_hidden(tmp_path: Path):
    """So Krita, GIMP and this reader's own fallback show frame 1 rather than
    the whole clip stacked on top of itself."""
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)

    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    groups = root.find("stack").findall("stack")

    assert [g.get("name") for g in groups] == ["frame:0001", "frame:0002"]
    assert groups[0].get("visibility") == "visible"
    assert groups[1].get("visibility") == "hidden"
    assert all(layer.get("visibility") == "hidden" for layer in groups[1])
    assert any(layer.get("visibility") == "visible" for layer in groups[0])


def test_the_merged_image_is_frame_one_whatever_the_playhead_is_on(tmp_path: Path):
    """A save has to be a function of the document, not of where the user
    happened to be looking, or saving twice gives two different files."""
    doc = _animated()
    first, second = tmp_path / "a.ora", tmp_path / "b.ora"
    inker.write_ora(doc, first)
    doc.set_current_frame(1)
    inker.write_ora(doc, second)

    with zipfile.ZipFile(first) as zf:
        a = zf.read("mergedimage.png")
    with zipfile.ZipFile(second) as zf:
        b = zf.read("mergedimage.png")
    assert a == b


def _rewritten(path: Path, dest: Path, edit) -> Path:
    """A copy of an ORA with ``animation.json`` passed through ``edit``."""
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(dest, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "animation.json":
                data = edit(data)
                if data is None:
                    continue
            dst.writestr(item, data)
    return dest


def test_an_absent_animation_json_reads_exactly_as_it_always_did(tmp_path: Path):
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)
    stripped = _rewritten(path, tmp_path / "flat.ora", lambda _data: None)

    doc = inker.Document.load(stripped)

    assert doc.anim is None
    # Both frames' layers, flattened out of their groups by the tolerant reader.
    assert len(doc.stack) == 4


def test_a_corrupt_animation_json_falls_back_to_the_flat_read(tmp_path: Path):
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)
    broken = _rewritten(path, tmp_path / "broken.ora", lambda _data: b"{not json")

    doc = inker.Document.load(broken)

    assert doc.anim is None
    assert doc.size == (8, 8)


def _retagged(data: bytes, **changes) -> bytes:
    payload = json.loads(data)
    payload.update(changes)
    return json.dumps(payload).encode()


def test_a_cel_naming_a_missing_png_fails_the_whole_grid(tmp_path: Path):
    """Not just that cel. Half a timeline is harder to notice than none of one,
    and the flat fallback at least looks like what it is."""
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)

    def break_one(data: bytes) -> bytes:
        payload = json.loads(data)
        payload["cels"][0]["data"] = "data/nope.png"
        return json.dumps(payload).encode()

    broken = _rewritten(path, tmp_path / "broken.ora", break_one)
    assert inker.Document.load(broken).anim is None


def test_a_cel_pointing_outside_the_grid_fails_the_whole_grid(tmp_path: Path):
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)

    def out_of_range(data: bytes) -> bytes:
        payload = json.loads(data)
        payload["cels"][0]["frame"] = 99
        return json.dumps(payload).encode()

    broken = _rewritten(path, tmp_path / "broken.ora", out_of_range)
    assert inker.Document.load(broken).anim is None


def test_a_future_version_is_not_guessed_at(tmp_path: Path):
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)
    future = _rewritten(path, tmp_path / "future.ora", lambda d: _retagged(d, version=99))
    assert inker.Document.load(future).anim is None


def test_a_still_document_writes_exactly_the_members_it_always_did(tmp_path: Path):
    """The regression pin for "plain documents are byte-compatible". Every
    branch the animation added is gated on ``doc.anim``, and this is what says
    so from outside the module."""
    with zipfile.ZipFile(_saved(tmp_path)) as zf:
        names = zf.namelist()

    assert names == [
        "mimetype",
        "stack.xml",
        "data/layer0.png",
        "data/layer1.png",
        "mergedimage.png",
        "Thumbnails/thumbnail.png",
    ]


def test_the_grid_survives_a_second_save(tmp_path: Path):
    """Uids are minted per process and the file stores indices, so a reload
    renumbers everything. That is only safe if writing the reloaded document
    produces the same grid again."""
    first, second = tmp_path / "a.ora", tmp_path / "b.ora"
    inker.write_ora(_animated(), first)
    inker.write_ora(inker.Document.load(first), second)

    with zipfile.ZipFile(first) as zf:
        a = json.loads(zf.read("animation.json"))
    with zipfile.ZipFile(second) as zf:
        b = json.loads(zf.read("animation.json"))
    assert a == b


def test_a_frame_with_no_duration_gets_the_default_and_not_the_floor(tmp_path: Path):
    """A hundred times too fast, and silently: a clip that plays is not
    obviously wrong. Falling through to ``clamp_duration``'s floor gave an
    absent ``duration_ms`` one millisecond, where a frame that never said what
    it lasts should last what a new frame lasts."""
    path = tmp_path / "anim.ora"
    inker.write_ora(_animated(), path)

    def drop_durations(data: bytes) -> bytes:
        payload = json.loads(data)
        for entry in payload["frames"]:
            entry.pop("duration_ms", None)
        return json.dumps(payload).encode()

    stripped = _rewritten(path, tmp_path / "nodur.ora", drop_durations)
    doc = inker.Document.load(stripped)

    assert doc.anim is not None
    assert [f.duration_ms for f in doc.anim.frames] == [
        animation.DEFAULT_DURATION_MS
    ] * len(doc.anim.frames)


def test_tags_survive_a_round_trip(tmp_path: Path):
    """The timeline can create them now, so the file has to carry them: a tag
    is the only part of the grid a playback engine reads by name."""
    doc = _animated()  # already carries a hand-built "walk"
    doc.add_tag("hit", 1, loop=False)
    path = tmp_path / "tagged.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    assert [(t.name, t.start, t.end, t.loop) for t in back.anim.tags] == [
        ("walk", 0, 1, True),
        ("hit", 1, 1, False),
    ]


def test_a_tags_direction_survives_a_round_trip(tmp_path: Path):
    doc = _animated()
    doc.add_tag("swing", 0, 1)
    doc.set_tag(len(doc.anim.tags) - 1, direction="pingpong")
    path = tmp_path / "swing.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    assert [t.direction for t in back.anim.tags] == ["forward", "pingpong"]


def test_a_file_written_before_directions_existed_reads_as_forward(tmp_path: Path):
    """The field is additive and the version is unchanged, so this is the file
    an older build writes -- and it has to keep opening rather than failing the
    whole grid over how a tag *plays*."""
    doc = _animated()
    path = tmp_path / "old.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    payload = json.loads(members[inker_ora.ANIMATION_MEMBER])
    for tag in payload["tags"]:
        del tag["direction"]
    members[inker_ora.ANIMATION_MEMBER] = json.dumps(payload).encode("utf-8")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    back = inker.Document.load(path)
    assert back.anim is not None
    assert [t.direction for t in back.anim.tags] == ["forward"]


# --- the directional layout -------------------------------------------------


def _layout_doc(kind="walk"):
    doc = _animated()
    doc.anim.layout = animation.DirectionalLayout(kind)
    return doc


def test_a_layout_round_trips(tmp_path):
    path = tmp_path / "sheet.ora"
    inker_ora.write_ora(_layout_doc(), path)
    assert inker_ora.read_ora(path).anim.layout == animation.DirectionalLayout("walk")


def test_a_layout_less_document_writes_the_animation_json_it_always_did(tmp_path):
    """Additive means additive: the key is written only when it is set, so an
    ordinary animation's ``animation.json`` is byte-identical to before."""
    plain = tmp_path / "plain.ora"
    inker_ora.write_ora(_animated(), plain)
    with zipfile.ZipFile(plain) as zf:
        payload = json.loads(zf.read(inker_ora.ANIMATION_MEMBER))
    assert "layout" not in payload
    assert payload["version"] == inker_ora.ANIMATION_VERSION


def test_the_layout_does_not_bump_the_version(tmp_path):
    path = tmp_path / "sheet.ora"
    inker_ora.write_ora(_layout_doc(), path)
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(inker_ora.ANIMATION_MEMBER))
    assert payload["version"] == inker_ora.ANIMATION_VERSION
    assert payload["layout"] == {"kind": "walk"}


def test_saving_a_layout_document_twice_is_byte_identical(tmp_path):
    first, second = tmp_path / "a.ora", tmp_path / "b.ora"
    doc = _layout_doc()
    inker_ora.write_ora(doc, first)
    inker_ora.write_ora(doc, second)
    assert first.read_bytes() == second.read_bytes()


def _rewrite_animation(src, dest, mutate):
    with zipfile.ZipFile(src) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    payload = json.loads(members[inker_ora.ANIMATION_MEMBER])
    mutate(payload)
    members[inker_ora.ANIMATION_MEMBER] = json.dumps(payload).encode("utf-8")
    with zipfile.ZipFile(dest, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_a_malformed_layout_costs_the_layout_and_not_the_grid(tmp_path):
    """Its own guard, outside the block that rebuilds the timeline: the pixels
    are all present and correct, so a bad layout must not fall the file back to
    a flat read."""
    src, dest = tmp_path / "a.ora", tmp_path / "b.ora"
    inker_ora.write_ora(_layout_doc(), src)
    _rewrite_animation(src, dest, lambda p: p.update(layout={"nope": 1}))
    doc = inker_ora.read_ora(dest)
    assert doc.anim is not None and len(doc.anim.frames) == 2
    assert doc.anim.layout is None


def test_a_layout_this_build_does_not_know_degrades_to_none(tmp_path):
    src, dest = tmp_path / "a.ora", tmp_path / "b.ora"
    inker_ora.write_ora(_layout_doc(), src)
    _rewrite_animation(src, dest, lambda p: p.update(layout={"kind": "isometric"}))
    doc = inker_ora.read_ora(dest)
    assert doc.anim is not None and doc.anim.layout is None


def test_a_file_written_before_layouts_existed_reads_as_none(tmp_path):
    path = tmp_path / "old.ora"
    inker_ora.write_ora(_animated(), path)
    assert inker_ora.read_ora(path).anim.layout is None

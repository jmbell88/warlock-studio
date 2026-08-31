"""Layer groups through an ``.ora``, in both directions.

Two things are being pinned and the second is the one that could quietly go
wrong. The tree has to survive a round trip -- for our own files and, in the
still case, for a Krita-authored one, since a nested ``<stack>`` is *the* way
ORA says "group" and our reader used to throw the nesting away. And a document
with **no** groups has to produce exactly the bytes it produced before groups
existed, because a save is supposed to look like a save rather than like a
change to anything that hashes or diffs one.
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
from warlock.studio.inker import groups as gp
from warlock.studio.inker import ora as inker_ora
from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(layers: int = 3) -> Document:
    doc = Document.blank(8, 8)
    doc.stack[0].name = "L0"
    doc.stack[0].pixels[:, :] = RED
    for i in range(1, layers):
        doc.add_layer(f"L{i}")
        doc.stack[i].pixels[0:2, 0:2] = BLUE
    doc.invalidate_all()
    return doc


def _animated() -> Document:
    doc = _doc(3)
    doc.add_frame(copy=True)
    doc.set_current_frame(0)
    return doc


def _xml(path: Path):
    with zipfile.ZipFile(path) as zf:
        return ElementTree.fromstring(zf.read("stack.xml"))


def _ok(doc: Document) -> None:
    gp.check(doc.groups, doc.group_of, doc.member_uids())


# --- the negative control ----------------------------------------------------


def test_an_ungrouped_document_is_byte_identical(tmp_path: Path):
    """The control the whole feature is measured against: no groups means no
    nested elements, no ``groups`` key, and therefore the same file."""
    doc = _doc()
    plain = inker.ora_bytes(doc)

    grouped = _doc()
    node = grouped.group_layers([1, 2], name="Ink")
    assert node is not None
    assert grouped.ungroup(node.uid)

    assert inker.ora_bytes(grouped) == plain


def test_an_ungrouped_animated_document_is_byte_identical(tmp_path: Path):
    """The animated writer takes a different branch and has a second place to
    leak -- ``animation.json`` -- so it needs its own control."""
    plain = inker.ora_bytes(_animated())

    grouped = _animated()
    node = grouped.group_layers([1, 2], name="Ink")
    assert grouped.ungroup(node.uid)

    assert inker.ora_bytes(grouped) == plain


def test_no_group_writes_no_nested_stack_and_no_groups_key(tmp_path: Path):
    path = tmp_path / "flat.ora"
    inker.write_ora(_animated(), path)
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(inker_ora.ANIMATION_MEMBER))
    assert "groups" not in payload


# --- the still round trip ----------------------------------------------------


def test_a_group_becomes_a_nested_stack(tmp_path: Path):
    doc = _doc()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "grouped.ora"
    inker.write_ora(doc, path)

    outer = _xml(path).find("stack")
    nested = outer.findall("stack")
    assert [g.get("name") for g in nested] == ["Ink"]
    # Top first, as everywhere in this writer, and the group holds both layers.
    assert [layer.get("name") for layer in nested[0]] == ["L2", "L1"]
    # The one layer outside it is a sibling of the group rather than inside it.
    assert [layer.get("name") for layer in outer.findall("layer")] == ["L0"]
    assert nested[0].get("isolation") == "auto"


def test_a_group_round_trips(tmp_path: Path):
    doc = _doc()
    node = doc.group_layers([1, 2], name="Ink")
    doc.set_group_props(node.uid, opacity=0.5, visible=False, locked=True)
    path = tmp_path / "grouped.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    _ok(back)
    assert len(back.groups) == 1
    got = next(iter(back.groups.values()))
    assert (got.name, got.opacity, got.visible, got.locked) == ("Ink", 0.5, False, True)
    assert gp.leaves_of(back.group_of, back.member_uids(), got.uid) == [
        back.stack[1].uid,
        back.stack[2].uid,
    ]


def test_nesting_round_trips(tmp_path: Path):
    doc = _doc(4)
    outer = doc.group_layers([1, 2, 3], name="outer")
    doc.group_layers([2, 3], name="inner")
    assert outer is not None
    path = tmp_path / "nested.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    _ok(back)
    assert len(back.groups) == 2
    depths = [len(gp.ancestry(back.group_of, layer.uid)) for layer in back.stack]
    assert depths == [0, 1, 2, 2]


def test_a_hidden_group_hides_its_layers_in_the_merged_image(tmp_path: Path):
    """``mergedimage.png`` is what a viewer that does not composite shows, so a
    group hidden on screen and visible in the file would be the same picture
    disagreeing with itself."""
    doc = _doc(2)
    doc.stack[1].pixels[:, :] = BLUE
    node = doc.group_layers([1], name="Ink")
    doc.set_group_props(node.uid, visible=False)
    path = tmp_path / "hidden.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf, Image.open(io.BytesIO(zf.read("mergedimage.png"))) as im:
        merged = np.asarray(im.convert("RGBA"))
    assert tuple(merged[4, 4]) == RED


# --- a foreign file ----------------------------------------------------------


def _foreign(path: Path, body: str, *, w: int = 8, h: int = 8) -> None:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[:, :] = RED
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buf, "PNG")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr(
            "stack.xml",
            f'<image version="0.0.3" w="{w}" h="{h}"><stack>{body}</stack></image>',
        )
        zf.writestr("data/a.png", buf.getvalue())


def test_a_krita_style_group_is_read_as_a_group(tmp_path: Path):
    """A nested ``<stack>`` *is* how ORA spells a group. The flat list is still
    what the compositor sees -- that has not changed and must not -- but the
    nesting is now recorded rather than dropped."""
    path = tmp_path / "krita.ora"
    _foreign(
        path,
        '<stack name="G" opacity="0.500000" visibility="hidden" '
        'composite-op="svg:src-over" isolate-mode="0">'
        '<layer name="inner" src="data/a.png"/>'
        "</stack>"
        '<layer name="outer" src="data/a.png"/>',
    )
    doc = inker.Document.load(path)
    _ok(doc)
    assert sorted(layer.name for layer in doc.stack) == ["inner", "outer"]
    assert len(doc.groups) == 1
    node = next(iter(doc.groups.values()))
    assert (node.name, node.opacity, node.visible) == ("G", 0.5, False)
    inner = next(layer for layer in doc.stack if layer.name == "inner")
    assert doc.group_of[inner.uid] == node.uid


def test_an_empty_foreign_group_records_nothing(tmp_path: Path):
    """Empty groups are disallowed in the model, and a foreign file is entitled
    to contain one."""
    path = tmp_path / "empty.ora"
    _foreign(path, '<stack name="G"></stack><layer name="L" src="data/a.png"/>')
    doc = inker.Document.load(path)
    assert not doc.groups
    _ok(doc)


def test_a_group_whose_layers_are_all_missing_is_pruned(tmp_path: Path):
    path = tmp_path / "gone.ora"
    _foreign(
        path,
        '<stack name="G"><layer name="gone" src="data/missing.png"/></stack>'
        '<layer name="here" src="data/a.png"/>',
    )
    doc = inker.Document.load(path)
    assert [layer.name for layer in doc.stack] == ["here"]
    assert not doc.groups
    _ok(doc)


def test_unmodelled_group_attributes_are_dropped_not_refused(tmp_path: Path, caplog):
    path = tmp_path / "extra.ora"
    _foreign(
        path,
        '<stack name="G" selected="true" alpha-preserve="1">'
        '<layer name="L" src="data/a.png"/></stack>',
    )
    with caplog.at_level("DEBUG", logger="warlock.studio.inker.ora"):
        doc = inker.Document.load(path)
    assert len(doc.groups) == 1
    assert "selected" in caplog.text


# --- the animated record -----------------------------------------------------


def test_an_animated_grouping_round_trips(tmp_path: Path):
    doc = _animated()
    assert doc.group_layers([1, 2], name="outer") is not None
    inner = doc.group_layers([2], name="inner")
    doc.set_group_props(inner.uid, opacity=0.25, locked=True)
    path = tmp_path / "anim.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    assert back.anim is not None
    _ok(back)
    assert len(back.groups) == 2
    by_name = {node.name: node for node in back.groups.values()}
    assert by_name["inner"].opacity == 0.25
    assert by_name["inner"].locked is True
    assert back.group_of[by_name["inner"].uid] == by_name["outer"].uid
    # Keyed on the *track*, which is what the placeholder trap requires.
    assert back.group_of[back.anim.tracks[2].uid] == by_name["inner"].uid
    assert back.group_of[back.anim.tracks[1].uid] == by_name["outer"].uid


def test_the_animated_version_stays_one(tmp_path: Path):
    doc = _animated()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "v.ora"
    inker.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(inker_ora.ANIMATION_MEMBER))
    assert payload["version"] == 1
    assert [entry["name"] for entry in payload["groups"]["nodes"]] == ["Ink"]


def test_the_frame_groups_carry_the_layer_groups_inside_them(tmp_path: Path):
    """The XML is the picture a foreign editor gets, and that picture is one
    frame with its folders in it."""
    doc = _animated()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "anim.ora"
    inker.write_ora(doc, path)

    frames = _xml(path).find("stack").findall("stack")
    assert [f.get("name") for f in frames] == ["frame:0001", "frame:0002"]
    for frame in frames:
        assert [g.get("name") for g in frame.findall("stack")] == ["Ink"]


def test_a_malformed_grouping_costs_the_folders_and_not_the_grid(tmp_path: Path):
    """Guarded exactly like ``layout``: a grouping is metadata about a grid
    whose pixels are all present and correct."""
    doc = _animated()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "bad.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    payload = json.loads(members[inker_ora.ANIMATION_MEMBER])
    payload["groups"]["tracks"] = [{"track": 99, "group": 0}]
    members[inker_ora.ANIMATION_MEMBER] = json.dumps(payload).encode("utf-8")
    broken = tmp_path / "broken.ora"
    with zipfile.ZipFile(broken, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    back = inker.Document.load(broken)
    assert back.anim is not None
    assert len(back.anim.frames) == 2
    assert len(back.anim.tracks) == 3
    assert not back.groups


def test_a_groups_key_that_is_not_a_mapping_is_ignored(tmp_path: Path):
    doc = _animated()
    path = tmp_path / "anim.ora"
    inker.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    payload = json.loads(members[inker_ora.ANIMATION_MEMBER])
    payload["groups"] = "nonsense"
    members[inker_ora.ANIMATION_MEMBER] = json.dumps(payload).encode("utf-8")
    broken = tmp_path / "broken.ora"
    with zipfile.ZipFile(broken, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    back = inker.Document.load(broken)
    assert back.anim is not None
    assert not back.groups


def _without(path: Path, out: Path, member: str) -> None:
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist() if n != member}
    with zipfile.ZipFile(out, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_an_animated_file_that_falls_back_flat_records_no_folders(tmp_path: Path):
    """The degraded path must not get *noisier*. Without ``animation.json`` the
    grid cannot be rebuilt and the reader falls back to the flat XML -- whose
    nested stacks are **frames**, not folders. Reading them as groups handed
    back one folder per frame where the same file previously opened flat."""
    doc = _animated()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "anim.ora"
    inker.write_ora(doc, path)

    flat = tmp_path / "flat.ora"
    _without(path, flat, inker_ora.ANIMATION_MEMBER)
    back = inker.Document.load(flat)

    assert back.anim is None
    assert not back.groups
    # And the flattening is unchanged: every cel of every frame is still a
    # layer, which is the whole reason the fallback exists.
    assert len(back.stack) == 6
    _ok(back)


def test_an_ungrouped_animated_file_falls_back_flat_with_no_folders(tmp_path: Path):
    doc = _animated()
    path = tmp_path / "anim.ora"
    inker.write_ora(doc, path)
    flat = tmp_path / "flat.ora"
    _without(path, flat, inker_ora.ANIMATION_MEMBER)

    back = inker.Document.load(flat)
    assert back.anim is None
    assert not back.groups


def test_frame_groups_are_recognised_by_name():
    """The projection marker `_stack_xml_animated` writes. Group construction
    is suppressed for the *whole* read when one is present, because each frame
    group carries its own copy of the document's real folders -- recognising
    the frames alone would still give a forty-frame clip forty folders."""
    animated = ElementTree.fromstring(
        '<stack><stack name="frame:0001"><layer name="L"/></stack></stack>'
    )
    plain = ElementTree.fromstring(
        '<stack><stack name="Ink"><layer name="L"/></stack></stack>'
    )
    assert inker_ora.has_frame_groups(animated) is True
    assert inker_ora.has_frame_groups(plain) is False


def test_a_still_file_is_unaffected_by_the_frame_group_rule(tmp_path: Path):
    """The suppression is scoped to files that carry a frame projection; an
    ordinary grouped document still reads its folders back."""
    doc = _doc()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "still.ora"
    inker.write_ora(doc, path)
    assert len(inker.Document.load(path).groups) == 1


# --- isolation and the group blend mode --------------------------------------
#
# Divergence retired: the two attributes used to be dropped on read (logged, as
# unmodelled) and hard-coded to the pass-through answer on write, because the
# document model had nowhere to put them. It has, so they are carried.


def test_a_pass_through_group_still_says_auto_and_names_no_mode(tmp_path: Path):
    """The negative control, and the reason the mode is written conditionally.

    ``svg:src-over`` is what a reader assumes from an absent ``composite-op``,
    so spelling it out would change the bytes of every file that has ever had a
    folder in it and say nothing new.
    """
    doc = _doc()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "plain.ora"
    inker.write_ora(doc, path)

    nested = _xml(path).find("stack").findall("stack")[0]
    assert nested.get("isolation") == "auto"
    assert nested.get("composite-op") is None


def test_an_isolated_group_says_so_and_names_its_mode(tmp_path: Path):
    doc = _doc()
    node = doc.group_layers([1, 2], name="Ink")
    doc.set_group_props(node.uid, isolate=True, blend="multiply")
    path = tmp_path / "isolated.ora"
    inker.write_ora(doc, path)

    nested = _xml(path).find("stack").findall("stack")[0]
    assert nested.get("isolation") == "isolate"
    assert nested.get("composite-op") == "svg:multiply"


def test_a_mode_alone_is_written_as_isolated(tmp_path: Path):
    """A blend mode implies isolation, so the file must not claim otherwise."""
    doc = _doc()
    node = doc.group_layers([1, 2], name="Ink")
    doc.set_group_props(node.uid, blend="screen")
    path = tmp_path / "mode-only.ora"
    inker.write_ora(doc, path)

    nested = _xml(path).find("stack").findall("stack")[0]
    assert nested.get("isolation") == "isolate"
    assert nested.get("composite-op") == "svg:screen"


def test_isolation_and_the_mode_round_trip(tmp_path: Path):
    doc = _doc()
    node = doc.group_layers([1, 2], name="Ink")
    doc.set_group_props(node.uid, isolate=True, blend="multiply", opacity=0.5)
    path = tmp_path / "grouped.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    _ok(back)
    got = next(iter(back.groups.values()))
    assert (got.isolate, got.blend, got.opacity) == (True, "multiply", 0.5)
    assert back.stack.group_spans is not None


def test_a_krita_group_set_to_multiply_opens_as_multiply(tmp_path: Path):
    """The case the divergence named: it used to render pass-through on load."""
    doc = _doc()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "krita.ora"
    inker.write_ora(doc, path)
    _rewrite_group_attrs(path, {"composite-op": "svg:multiply", "isolation": "isolate"})

    back = inker.Document.load(path)
    got = next(iter(back.groups.values()))
    assert got.blend == "multiply"
    assert got.isolate is True


def test_a_mode_this_build_cannot_reproduce_reads_as_normal(tmp_path: Path):
    """The reader's own tolerance rule, one level up from a layer."""
    doc = _doc()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "odd.ora"
    inker.write_ora(doc, path)
    _rewrite_group_attrs(path, {"composite-op": "krita:greater"})

    back = inker.Document.load(path)
    got = next(iter(back.groups.values()))
    assert got.blend == "normal"


def test_isolation_round_trips_through_an_animated_document(tmp_path: Path):
    """The second serialisation: ``animation.json``, not ``stack.xml``."""
    doc = _animated()
    node = doc.group_layers([1, 2], name="Ink")
    doc.set_group_props(node.uid, isolate=True, blend="overlay")
    path = tmp_path / "anim.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read("animation.json"))
    assert payload["groups"]["nodes"][0]["blend"] == "overlay"
    assert payload["groups"]["nodes"][0]["isolate"] is True

    back = inker.Document.load(path)
    got = next(iter(back.groups.values()))
    assert (got.isolate, got.blend) == (True, "overlay")


def test_an_animated_pass_through_group_writes_neither_key(tmp_path: Path):
    doc = _animated()
    doc.group_layers([1, 2], name="Ink")
    path = tmp_path / "anim-plain.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read("animation.json"))
    node = payload["groups"]["nodes"][0]
    assert "blend" not in node and "isolate" not in node


def _rewrite_group_attrs(path: Path, attrs: dict) -> None:
    """Re-save ``path`` with ``attrs`` set on its first nested ``<stack>``."""
    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    root = ElementTree.fromstring(members["stack.xml"].decode("utf-8"))
    nested = root.find("stack").findall("stack")[0]
    for key, value in attrs.items():
        nested.set(key, value)
    members["stack.xml"] = ElementTree.tostring(root, encoding="utf-8")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

"""The one property model, and the three pairs of codecs over it.

Properties used to be modelled three times -- once in ``tsx`` for the XML
spelling, once in ``tmx`` for the JSON one, once in ``wmap`` for our own
archive -- and each copy decided for itself what a property may be. This file
is the gate on the single model that replaced them: nine types, one refusal
sentence, and a codec pair per syntax that agrees with the other two about
what a document holds.

The interesting cases are the three types that carry something other than a
scalar: ``file`` is a string the host resolves, ``object`` is a persistent id
where zero means *none*, and ``class`` is a recursive block of properties
whose ``propertytype`` names a type Plotter does not define and therefore
copies verbatim.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from warlock.studio.plotter import props as P
from warlock.studio.plotter.props import Prop, TiledUnsupported


def _xml_round_trip(values: dict[str, Prop]) -> dict[str, Prop]:
    """Write a ``<properties>`` block and read it straight back."""
    root = ET.Element("map")
    P.write_properties(root, values)
    return P.read_properties(root)


def _xml_text(values: dict[str, Prop]) -> str:
    root = ET.Element("map")
    P.write_properties(root, values)
    return ET.tostring(root, encoding="unicode")


def _json_round_trip(values: dict[str, Prop]) -> dict[str, Prop]:
    """Through ``json.dumps`` deliberately: the writer's output has to be
    JSON, not merely a dict that looks like it."""
    payload = json.loads(json.dumps(P.write_json_properties(values)))
    return P.read_json_properties(payload)


def _wmap_round_trip(values: dict[str, Prop]) -> dict[str, Prop]:
    payload = json.loads(json.dumps(P.write_wmap_properties(values), sort_keys=True))
    return P.read_wmap_properties(payload)


# --- the model ----------------------------------------------------------------


def test_the_type_list_is_tileds_nine():
    assert P.PROPERTY_TYPES == (
        "string",
        "int",
        "float",
        "bool",
        "color",
        "file",
        "object",
        "class",
        "list",
    )


def test_a_type_outside_the_nine_is_refused_by_name():
    with pytest.raises(TiledUnsupported) as exc:
        Prop("vector2", "1,2")
    assert "vector2" in str(exc.value)
    assert exc.value.feature == "a custom property of type 'vector2'"


def test_a_property_carries_a_propertytype_and_it_defaults_to_empty():
    """``propertytype`` names a *user-defined* type -- a class or an enum
    declared in a Tiled project this editor never reads. It is carried rather
    than validated, because the only alternative is dropping the name of a
    type we cannot check and writing the file back without it."""
    assert Prop("string", "x").propertytype == ""
    assert Prop("string", "GREEN", propertytype="Colour").propertytype == "Colour"


def test_a_class_property_holds_props_and_nothing_else():
    inner = {"hp": Prop("int", 3)}
    assert Prop("class", inner, propertytype="NPC").value == inner
    with pytest.raises(ValueError):
        Prop("class", {"hp": 3})
    with pytest.raises(ValueError):
        Prop("class", "not a block")


def test_a_list_property_holds_props_and_nothing_else():
    assert Prop("list", [Prop("int", 1)]).value == [Prop("int", 1)]
    with pytest.raises(ValueError):
        Prop("list", [1, 2])


def test_an_object_property_is_an_id_and_zero_means_none():
    assert Prop("object", 0).value == 0
    assert Prop("object", 12).value == 12
    with pytest.raises(ValueError):
        Prop("object", "12")


def test_a_file_property_is_the_path_verbatim():
    """Resolution is the host's problem: the engine never opens a file, so a
    path it cannot resolve is a path it must not rewrite."""
    prop = Prop("file", "../art/tiles.png")
    assert prop.value == "../art/tiles.png"
    assert _xml_round_trip({"art": prop})["art"] == prop


# --- the XML codec ------------------------------------------------------------


def test_the_three_new_scalarish_types_round_trip_through_xml():
    values = {
        "art": Prop("file", "art/hero.png"),
        "target": Prop("object", 7),
        "none": Prop("object", 0),
    }
    assert _xml_round_trip(values) == values


def test_a_class_property_round_trips_through_xml_with_its_members_typed():
    values = {
        "npc": Prop(
            "class",
            {
                "name": Prop("string", "Bob"),
                "hp": Prop("int", 100),
                "tint": Prop("color", "#ff00ff00"),
                "friendly": Prop("bool", True),
            },
            propertytype="NPC",
        )
    }
    assert _xml_round_trip(values) == values


def test_a_class_property_is_written_as_tiled_writes_one():
    """Tiled 1.12.2 spells a class property as a ``<property>`` with no
    ``value`` attribute whose members are a nested ``<properties>`` block, and
    names the user's type in ``propertytype``."""
    text = _xml_text({"npc": Prop("class", {"hp": Prop("int", 1)}, propertytype="NPC")})
    node = ET.fromstring(text).find("properties/property")
    assert node.get("type") == "class"
    assert node.get("propertytype") == "NPC"
    assert node.get("value") is None
    member = node.find("properties/property")
    assert (member.get("name"), member.get("type"), member.get("value")) == ("hp", "int", "1")


def test_a_class_property_nests_and_keeps_an_unknown_propertytype():
    values = {
        "outer": Prop(
            "class",
            {"inner": Prop("class", {"n": Prop("int", 2)}, propertytype="Unknown")},
            propertytype="Outer",
        )
    }
    assert _xml_round_trip(values) == values


def test_an_empty_class_property_survives_the_round_trip():
    """A class with no members is still a class -- writing nothing at all
    would read back as an empty *string*, which is a different document."""
    values = {"npc": Prop("class", {}, propertytype="NPC")}
    assert _xml_round_trip(values) == values


def test_class_members_are_written_in_sorted_order():
    """Determinism reaches all the way down: two saves of one document are
    byte-identical, and a nested dict's order is not a fact about the map."""
    text = _xml_text(
        {"npc": Prop("class", {"z": Prop("int", 1), "a": Prop("int", 2)}, propertytype="N")}
    )
    assert text.index('name="a"') < text.index('name="z"')


def test_a_propertytype_on_a_plain_property_is_preserved():
    """Tiled writes ``propertytype`` on an *enum*-valued string or int too.
    Carrying it is free and dropping it retypes the user's enum on save."""
    values = {"mood": Prop("string", "GREEN", propertytype="Mood")}
    assert _xml_round_trip(values) == values
    assert 'propertytype="Mood"' in _xml_text(values)


# --- the JSON codec -----------------------------------------------------------


def test_the_three_new_scalarish_types_round_trip_through_json():
    values = {
        "art": Prop("file", "art/hero.png"),
        "target": Prop("object", 7),
        "none": Prop("object", 0),
    }
    assert _json_round_trip(values) == values


def test_a_class_property_is_written_as_tiled_writes_one_in_json():
    """The JSON spelling is a ``propertytype`` beside an *object* value whose
    keys are the member names -- not the list-of-descriptors form the top
    level uses."""
    entries = P.write_json_properties(
        {"npc": Prop("class", {"hp": Prop("int", 1)}, propertytype="NPC")}
    )
    assert entries == [
        {"name": "npc", "type": "class", "propertytype": "NPC", "value": {"hp": 1}}
    ]


def test_a_class_property_round_trips_through_json_for_the_types_json_can_tell_apart():
    values = {
        "npc": Prop(
            "class",
            {"name": Prop("string", "Bob"), "hp": Prop("int", 2), "ok": Prop("bool", True)},
            propertytype="NPC",
        )
    }
    assert _json_round_trip(values) == values


def test_a_colour_member_of_a_class_returns_as_a_string_through_json():
    """The one documented asymmetry between the two syntaxes, pinned rather
    than hidden. Tiled's JSON stores class *members* as bare values and takes
    their types from the project's ``propertytypes.json``, which Plotter does
    not read -- so ``color`` and ``string``, both JSON strings, are
    indistinguishable on the way back in. The XML spelling types every member
    and is lossless; guessing from the text would be exactly the silent
    retyping this model exists to prevent.
    """
    values = {"npc": Prop("class", {"tint": Prop("color", "#ff00ff00")}, propertytype="N")}
    back = _json_round_trip(values)
    assert back["npc"].value["tint"] == Prop("string", "#ff00ff00")


def test_a_nested_class_in_json_keeps_its_members_and_loses_only_the_type_name():
    values = {"outer": Prop("class", {"inner": Prop("class", {"n": Prop("int", 2)})})}
    assert _json_round_trip(values) == values


# --- the wmap codec -----------------------------------------------------------


def test_every_type_round_trips_through_the_wmap_record():
    values = {
        "s": Prop("string", "x"),
        "i": Prop("int", 3),
        "f": Prop("float", 0.5),
        "b": Prop("bool", True),
        "c": Prop("color", "#ff00ff00"),
        "art": Prop("file", "a/b.png"),
        "target": Prop("object", 9),
        "npc": Prop("class", {"tint": Prop("color", "#ff112233")}, propertytype="NPC"),
        "bag": Prop("list", [Prop("int", 1), Prop("string", "two")]),
    }
    assert _wmap_round_trip(values) == values


def test_a_plain_propertys_wmap_record_is_what_version_2_wrote():
    """Extended in place, with no version bump: a record that gained no new
    field has to be the bytes it already was, or every ``.wmap`` in the world
    changes on the next save for nothing."""
    assert P.write_wmap_properties({"n": Prop("int", 1)}) == {"n": {"type": "int", "value": 1}}


def test_a_version_2_record_without_a_propertytype_reads_as_an_empty_one():
    """The ``locked`` precedent: an older file is missing a key, not wrong."""
    back = P.read_wmap_properties({"n": {"type": "string", "value": "x"}})
    assert back == {"n": Prop("string", "x")}


def test_a_malformed_wmap_record_is_refused():
    with pytest.raises(ValueError):
        P.read_wmap_properties({"n": "not a record"})


# --- list, which Tiled has no syntax for --------------------------------------


def test_a_list_property_is_refused_by_name_on_both_tiled_writers():
    """Modelled in the document and in our own archive, refused at the Tiled
    door: Tiled 1.12.2 has no list-valued property, so there is no syntax to
    write that it would read back. Refusing beats inventing one."""
    values = {"bag": Prop("list", [Prop("int", 1)])}
    with pytest.raises(TiledUnsupported) as exc:
        _xml_text(values)
    assert exc.value.feature == "a list-valued custom property"
    with pytest.raises(TiledUnsupported) as exc:
        P.write_json_properties(values)
    assert exc.value.feature == "a list-valued custom property"


def test_a_list_property_is_refused_by_name_on_both_tiled_readers():
    root = ET.fromstring(
        '<map><properties><property name="bag" type="list" value="1"/></properties></map>'
    )
    with pytest.raises(TiledUnsupported) as exc:
        P.read_properties(root)
    assert exc.value.feature == "a list-valued custom property"
    with pytest.raises(TiledUnsupported) as exc:
        P.read_json_properties([{"name": "bag", "type": "list", "value": [1]}])
    assert exc.value.feature == "a list-valued custom property"


def test_a_list_inside_a_class_is_refused_by_the_same_name():
    values = {"npc": Prop("class", {"bag": Prop("list", [])}, propertytype="N")}
    with pytest.raises(TiledUnsupported) as exc:
        _xml_text(values)
    assert exc.value.feature == "a list-valued custom property"

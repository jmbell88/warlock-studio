"""One custom-property model, and the three codec pairs over it.

A Tiled property is a *typed* value with a name, and this module is the only
place in the package that decides what that sentence means. It used to be
decided three times -- :mod:`.tsx` for the XML spelling, :mod:`.tmx` for the
JSON one, :mod:`.wmap` for our own archive -- and three copies of a rule is
three chances for a file to be read as one document and written back as
another. The pairs live here, beside the type they encode, so adding a type is
one edit rather than three that have to agree.

**The type is stored, never inferred.** ``color`` and ``string`` are both
``str``, ``object`` and ``int`` are both ``int``: a round trip that guessed
from the Python value would silently retype every colour a user set in Tiled,
and the user would find out when the map came back with a broken tint. The
same rule is why a ``propertytype`` -- the name of a class or an enum declared
in a Tiled *project*, which this editor never reads -- is carried verbatim
even when it names a type we know nothing about. The alternative to copying an
unknown name is dropping it, and dropping it rewrites the user's file.

**The nine types, and what each holds:**

``string`` ``color``
    ``str``. ``color`` is Tiled's ``#aarrggbb``, kept as text.
``int`` ``float`` ``bool``
    the obvious Python scalar.
``file``
    ``str``, verbatim. The path is relative to the document that names it and
    resolving it means touching a filesystem, which is the *host's* job -- the
    engine never opens a file, so a path it cannot resolve is a path it must
    not rewrite.
``object``
    ``int``, a persistent Tiled object id. **Zero means none**, which is Tiled's
    own spelling of an unset object reference and the reason this is not
    ``int | None``.
``class``
    ``dict[str, Prop]``, recursively. ``propertytype`` names the user's class.
``list``
    ``list[Prop]``. Modelled and stored in ``.wmap``, **refused at the Tiled
    door**: Tiled 1.12.2 has no list-valued property, so there is no syntax to
    write that it would read back, and inventing one would produce a file that
    only this editor can open while claiming to be a Tiled file.

**The asymmetry between the syntaxes, stated rather than hidden.** XML writes
a class's members as a nested ``<properties>`` block -- real ``<property>``
elements, each carrying its own ``type`` and ``propertytype`` -- so the XML
codec is lossless. Tiled's JSON writes them as *bare values* inside the
parent's value object, and that costs two things on the way back in:

- **member types**, which Tiled recovers from the project's
  ``propertytypes.json`` and this editor does not read: a ``color`` member
  comes back from a ``.tmj`` as a ``string``, and so does a ``file`` member;
- **a nested class's ``propertytype``**, because that name is an attribute of
  a property *record* and a member is not one -- Tiled's schema has nowhere to
  put it. The outermost class keeps its name; every class inside it comes back
  with an empty one, members intact.

Both are real losses, and both beat the alternatives: guessing a member's type
from the shape of its text is precisely the silent retyping this model exists
to prevent, and inventing somewhere to hang the nested name would write a
``.tmj`` Tiled does not read back. ``docs/PLOTTER_COMPAT.md`` enumerates both
and ``tests/plotter/test_props.py`` pins them.

Nothing here imports anything: it is the package's leaf, below :mod:`.tsx`,
which re-exports :class:`Prop` and :class:`TiledUnsupported` so every existing
caller keeps working.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

#: Every type a custom property may have -- Tiled 1.12.2's own set, in Tiled's
#: own order. One tuple, in one module: the two copies this replaced were
#: byte-identical twins that nothing kept in step.
PROPERTY_TYPES = (
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

#: The two types whose value holds other properties. Named because a caller
#: that draws an editor row needs to know which types it cannot draw on one
#: line, and asking for the list beats each caller writing the pair out.
CONTAINER_TYPES = ("class", "list")


class TiledUnsupported(ValueError):
    """A Tiled file using a feature this editor does not model.

    ``feature`` is the name to put in front of the user; the message already
    contains it, and the attribute exists so a test can assert on the feature
    rather than on the sentence around it.

    Defined here rather than in :mod:`.tsx` because the property model is the
    package's leaf and the refusal travels with it; ``tsx`` re-exports the name
    so every ``from .tsx import TiledUnsupported`` keeps working.
    """

    def __init__(self, feature: str, detail: str = "") -> None:
        self.feature = feature
        tail = f" ({detail})" if detail else ""
        super().__init__(
            f"this file uses {feature}, which Plotter does not support{tail}. "
            "Open it in Tiled and remove or flatten that feature first."
        )


def _refuse_list(where: str = "") -> None:
    """The one refusal this module raises for a type it *models*.

    A list property is a real document field -- ``.wmap`` stores it and the
    editor shows it -- and it is refused only where it would have to be
    written in Tiled's syntax, because Tiled 1.12.2 has no such syntax.
    """
    raise TiledUnsupported(
        "a list-valued custom property",
        where or "Tiled has no list-valued property to write it as",
    )


@dataclass(frozen=True)
class Prop:
    """One typed custom property, as Tiled stores it.

    Frozen because a property is a value: the editors replace a whole
    properties dict rather than writing into one, which is what makes an undo
    step's snapshot honest.
    """

    type: str
    value: Any
    #: The user-defined type this value belongs to -- a class name, or an enum
    #: name on an otherwise plain ``string``/``int``. Empty when Tiled wrote
    #: none. Never validated: it names a type declared in a project file this
    #: editor does not read.
    propertytype: str = ""

    def __post_init__(self) -> None:
        if self.type not in PROPERTY_TYPES:
            raise TiledUnsupported(
                f"a custom property of type {self.type!r}",
                f"supported types are {', '.join(PROPERTY_TYPES)}",
            )
        if self.type == "class":
            if not isinstance(self.value, dict) or not all(
                isinstance(key, str) and isinstance(item, Prop)
                for key, item in self.value.items()
            ):
                raise ValueError("a class property's value is a mapping of name to Prop")
            object.__setattr__(self, "value", dict(self.value))
        elif self.type == "list":
            if not isinstance(self.value, list) or not all(
                isinstance(item, Prop) for item in self.value
            ):
                raise ValueError("a list property's value is a list of Prop")
            object.__setattr__(self, "value", list(self.value))
        elif self.type == "object":
            # ``bool`` is an ``int`` in Python and is not an object id.
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValueError("an object property's value is a Tiled object id, 0 for none")


# --- the XML codec ------------------------------------------------------------


def _parse_value(kind: str, text: str) -> Any:
    if kind == "bool":
        return text.strip().lower() == "true"
    if kind == "int":
        return int(float(text))
    if kind == "object":
        return int(float(text or 0))
    if kind == "float":
        return float(text)
    return str(text)


def read_properties(parent: ET.Element | None) -> dict[str, Prop]:
    """The ``<properties>`` child of an element, as a mapping.

    An element with no properties gives an empty dict, and so does one with an
    empty ``<properties>`` block -- the two are the same document. Recursive,
    because a ``class`` property's members are exactly this block one level
    down, which is how Tiled spells them.
    """
    if parent is None:
        return {}
    node = parent.find("properties")
    if node is None:
        return {}
    out: dict[str, Prop] = {}
    for entry in node.findall("property"):
        name = entry.get("name")
        if not name:
            continue
        kind = entry.get("type", "string")
        if kind not in PROPERTY_TYPES:
            raise TiledUnsupported(f"a custom property of type {kind!r}", f"property {name!r}")
        if kind == "list":
            _refuse_list(f"property {name!r}")
        propertytype = entry.get("propertytype", "")
        if kind == "class":
            out[name] = Prop("class", read_properties(entry), propertytype=propertytype)
            continue
        # Tiled puts a multi-line string in the element's text instead of in
        # the attribute, so the attribute is preferred and the text is the
        # fallback rather than the other way round.
        raw = entry.get("value")
        if raw is None:
            raw = entry.text or ""
        out[name] = Prop(
            type=kind, value=_parse_value(kind, raw), propertytype=propertytype
        )
    return out


def _value_text(prop: Prop) -> str:
    if prop.type == "bool":
        return "true" if prop.value else "false"
    if prop.type in ("int", "object"):
        return str(int(prop.value))
    if prop.type == "float":
        return repr(float(prop.value))
    return str(prop.value)


def _write_block(parent: ET.Element, props: dict[str, Prop]) -> None:
    """The ``<properties>`` element itself, written unconditionally.

    Separate from :func:`write_properties` because an *empty* one is only
    meaningless at the top level: a class property with no members that wrote
    no block at all would read back as an empty string, which is a different
    document.
    """
    node = ET.SubElement(parent, "properties")
    for name in sorted(props):
        prop = props[name]
        entry = ET.SubElement(node, "property", {"name": name})
        # Tiled omits type="string", and matching that keeps a file written
        # here diff-clean against the same file written there.
        if prop.type != "string":
            entry.set("type", prop.type)
        if prop.propertytype:
            entry.set("propertytype", prop.propertytype)
        if prop.type == "list":
            _refuse_list(f"property {name!r}")
        elif prop.type == "class":
            _write_block(entry, prop.value)
        else:
            entry.set("value", _value_text(prop))


def write_properties(parent: ET.Element, props: dict[str, Prop]) -> None:
    """Append a ``<properties>`` block, or nothing at all when there are none.

    Written in sorted name order rather than in whatever order they were read,
    at every level of nesting. The output is canonical on purpose -- two saves
    of an unchanged document have to be byte-identical, and a dict's order is
    not a property of the document.
    """
    if not props:
        return
    _write_block(parent, props)


# --- the JSON codec (Tiled's ``.tmj``/``.tsj`` spelling) -----------------------


def _json_member(name: str, raw: Any) -> Prop:
    """One member of a class property, read back from Tiled's JSON.

    The types are *inferred here and nowhere else*, because this is the one
    place Tiled does not write them: a class member's type lives in the
    project's ``propertytypes.json``. See the module docstring -- ``color``
    and ``file`` members arrive as ``string``, and that loss is the reason the
    XML spelling is the lossless one.
    """
    if isinstance(raw, bool):
        return Prop("bool", raw)
    if isinstance(raw, int):
        return Prop("int", raw)
    if isinstance(raw, float):
        return Prop("float", raw)
    if isinstance(raw, str):
        return Prop("string", raw)
    if isinstance(raw, dict):
        return Prop("class", {str(k): _json_member(str(k), v) for k, v in raw.items()})
    if isinstance(raw, list):
        _refuse_list(f"class member {name!r}")
    raise ValueError(f"class member {name!r} has a value this reader cannot type")


def read_json_properties(entries: Any) -> dict[str, Prop]:
    """Tiled's ``properties`` array as a mapping.

    A list of ``{name, type, value}`` records rather than XML's elements, and
    the same model on the other side of it -- one document, two syntaxes.
    """
    out: dict[str, Prop] = {}
    for entry in entries or []:
        name = str(entry.get("name", ""))
        if not name:
            continue
        kind = str(entry.get("type", "string"))
        if kind == "list":
            _refuse_list(f"property {name!r}")
        propertytype = str(entry.get("propertytype", "") or "")
        raw = entry.get("value")
        if kind == "class":
            if not isinstance(raw, dict):
                raise ValueError(f"class property {name!r} has no member block")
            out[name] = Prop(
                "class",
                {str(k): _json_member(str(k), v) for k, v in raw.items()},
                propertytype=propertytype,
            )
            continue
        if kind == "object":
            raw = int(raw or 0)
        elif kind == "file":
            raw = str(raw or "")
        # ``Prop`` refuses an unknown type by name, so the JSON side needs no
        # list of its own -- one place decides what a property may be.
        out[name] = Prop(type=kind, value=raw, propertytype=propertytype)
    return out


def _json_value(name: str, prop: Prop) -> Any:
    if prop.type == "list":
        _refuse_list(f"property {name!r}")
    if prop.type == "class":
        return {
            member: _json_value(member, prop.value[member]) for member in sorted(prop.value)
        }
    if prop.type == "object":
        return int(prop.value)
    if prop.type == "file":
        return str(prop.value)
    return prop.value


def write_json_properties(props: dict[str, Prop]) -> list[dict[str, Any]]:
    """The ``properties`` array, in sorted name order.

    A class property is written as Tiled writes one: ``propertytype`` beside a
    *value object* whose keys are the member names, sorted -- not the
    list-of-descriptors form the top level uses.
    """
    out: list[dict[str, Any]] = []
    for name in sorted(props):
        prop = props[name]
        record: dict[str, Any] = {"name": name, "type": prop.type}
        if prop.propertytype:
            record["propertytype"] = prop.propertytype
        record["value"] = _json_value(name, prop)
        out.append(record)
    return out


# --- the ``.wmap`` codec ------------------------------------------------------


def _wmap_record(prop: Prop) -> dict[str, Any]:
    record: dict[str, Any] = {"type": prop.type}
    if prop.type == "class":
        record["value"] = {
            member: _wmap_record(prop.value[member]) for member in sorted(prop.value)
        }
    elif prop.type == "list":
        record["value"] = [_wmap_record(item) for item in prop.value]
    else:
        record["value"] = prop.value
    # Only when set, which is what keeps a document that uses none byte-for-byte
    # the file version 2 already wrote. The reader defaults it, the ``locked``
    # precedent, so this needs no format version of its own.
    if prop.propertytype:
        record["propertytype"] = prop.propertytype
    return record


def write_wmap_properties(props: dict[str, Prop]) -> dict[str, Any]:
    """Our own archive's spelling: a mapping of name to a typed record.

    Unlike the two Tiled codecs this one is *complete* -- every type,
    including ``list``, and every member typed at every depth. It has one
    reader, so there is no foreign schema to fit inside.
    """
    return {name: _wmap_record(props[name]) for name in sorted(props)}


def _wmap_prop(name: str, record: Any) -> Prop:
    if not isinstance(record, dict):
        raise ValueError(f"property {name!r} is malformed")
    kind = str(record.get("type", "string"))
    propertytype = str(record.get("propertytype", "") or "")
    raw = record.get("value")
    if kind == "class":
        if not isinstance(raw, dict):
            raise ValueError(f"property {name!r} is malformed")
        return Prop(
            "class",
            {str(key): _wmap_prop(f"{name}.{key}", item) for key, item in raw.items()},
            propertytype=propertytype,
        )
    if kind == "list":
        if not isinstance(raw, list):
            raise ValueError(f"property {name!r} is malformed")
        return Prop(
            "list",
            [_wmap_prop(f"{name}[{index}]", item) for index, item in enumerate(raw)],
            propertytype=propertytype,
        )
    if kind == "object":
        return Prop("object", int(raw or 0), propertytype=propertytype)
    if kind == "file":
        return Prop("file", str(raw or ""), propertytype=propertytype)
    # ``Prop`` refuses an unknown type by name, so there is one list of legal
    # types and this is not a second copy of it.
    return Prop(type=kind, value=raw, propertytype=propertytype)


def read_wmap_properties(entry: Any) -> dict[str, Prop]:
    """The mapping :func:`write_wmap_properties` wrote, back as properties.

    Tolerant where a version 2 file is merely *older*: a record with no
    ``propertytype`` has an empty one rather than being refused.
    """
    return {str(name): _wmap_prop(str(name), record) for name, record in (entry or {}).items()}

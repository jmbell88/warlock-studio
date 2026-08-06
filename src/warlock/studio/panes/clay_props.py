"""The selected object: its transform, its generator's parameters, its material.

**The parameter widgets are generated from the registry, not written by hand.**
``primitives.GENERATORS`` maps a name to ``(defaults, builder)`` and every
default dictionary is a complete call, so the panel enumerates it and binds one
widget per key. A seventh primitive therefore needs no edit here at all -- which
is the entire reason that registry is data rather than a chain of ``if``s, and
is asserted by a test that registers a fake generator and looks for its
parameters.

A change to a parameter regenerates the mesh as **one** ``MeshEdit`` plus the
props edit that recorded the new parameters, so a Ctrl+Z takes the object back
to the shape it had. That only works while ``generator`` is not None; the first
topology edit clears it and this panel switches to a vertex and face count with
a "frozen" note -- which is the state an *imported* object arrives in too.

Every control here is disabled while a save is in flight, for the reason the
tool panel states.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, icons, widgets
from ..clay import primitives as bp
from ..manual import render as manual_render

# How a parameter's type decides its widget. Read off the *default value*,
# because a registry entry carries no schema and does not need one: a float
# default means a float field, and a tuple means one field per component.
_STEP = {"segments": 1, "rings": 1, "sides": 1}


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("properties")
    manual_render.help_button(ctx, "clay-props")
    if tab is None:
        widgets.muted("Nothing open.")
        return
    doc = tab.doc
    _element_summary(doc)
    obj = _selected(doc)
    if obj is None:
        widgets.empty_state(icons.BOX, "Nothing selected", "Click an object in the viewport.")
        return

    imgui.begin_disabled(tab.saving)
    _identity(doc, obj)
    imgui.dummy((0, 6))
    _transform(doc, obj)
    imgui.dummy((0, 6))
    _generator(doc, obj)
    imgui.dummy((0, 6))
    _material(doc, obj)
    imgui.end_disabled()


def _element_summary(doc: Any) -> None:
    """One line saying what is selected inside the objects, in element modes.

    The object panel below stays exactly as it was -- an element selection is
    still an object selection, by the document's own invariant -- so this adds
    a line rather than replacing the pane. It is also where the *frozen* branch
    of the generator section finally becomes reachable: an op that edits
    topology clears ``generator``, and this is usually the first thing the user
    sees afterwards.
    """
    if doc.element_mode == "object":
        return
    total = sum(sel.count(doc.element_mode) for sel in doc.element_sel.values())
    noun = {"vertex": "vertices", "edge": "edges", "face": "faces"}[doc.element_mode]
    objects = len(doc.element_sel)
    if total == 0:
        widgets.muted(f"{doc.element_mode} mode -- nothing selected")
    else:
        across = "1 object" if objects == 1 else f"{objects} objects"
        widgets.muted(f"{doc.element_mode} mode -- {total} {noun} across {across}")
    imgui.dummy((0, 4))


def _selected(doc: Any) -> Any:
    """The one selected object, or None.

    One rather than the first of many: a properties panel that silently edited
    whichever object happened to sort first under a multi-selection is worse
    than one that says it cannot.
    """
    if len(doc.selection) != 1:
        return None
    try:
        return doc.by_uid(next(iter(doc.selection)))
    except KeyError:
        return None


def _identity(doc: Any, obj: Any) -> None:
    name = widgets.input_text("name##buildname", obj.name, max_length=120)
    if name != obj.name:
        doc.set_props(obj.uid, name=name)
    changed, value = widgets.toggle(f"{icons.EYE} Visible", obj.visible, tag=str(obj.uid))
    if changed:
        doc.set_props(obj.uid, visible=value)


def _transform(doc: Any, obj: Any) -> None:
    widgets.field_label("transform")
    was = tuple(v.copy() for v in obj.trs())
    changed = False
    edited, translation = imgui.input_float3("position##bt", list(obj.translation))
    changed |= edited
    edited, scale = imgui.input_float3("scale##bs", list(obj.scale))
    changed |= edited
    edited, rotation = imgui.input_float4("rotation##br", list(obj.rotation))
    changed |= edited
    if changed:
        # ``was`` is the values the fields started from. imgui writes the new
        # ones into the widget's own state as they are typed, so reading
        # "before" off the object here would compare a value against itself and
        # record an empty step -- which is the trap ``set_transform``'s ``was``
        # argument exists for.
        doc.set_transform(
            obj.uid,
            translation=translation,
            rotation=rotation,
            scale=scale,
            was=was,
        )


def _generator(doc: Any, obj: Any) -> None:
    if obj.generator is None:
        # A frozen object: edited topology, or imported. The panel says what
        # the object is rather than pretending it still has parameters that
        # would silently discard the edit if changed.
        widgets.field_label("mesh")
        widgets.muted(
            f"frozen -- {len(obj.mesh.positions)} vertices, "
            f"{len(obj.mesh.starts) - 1} faces"
        )
        return
    entry = bp.GENERATORS.get(obj.generator)
    if entry is None:
        widgets.muted(f"unknown generator '{obj.generator}'")
        return

    defaults, build = entry
    widgets.field_label(obj.generator.replace("_", " "))
    params = dict(defaults)
    params.update({k: v for k, v in obj.params.items() if k in defaults})
    edited = dict(params)
    changed = False
    for key, default in defaults.items():
        was, changed_here = _widget(key, params.get(key, default), default)
        if changed_here:
            edited[key] = was
            changed = True
    if not changed:
        return
    try:
        mesh = build(**edited)
    except Exception:
        # A generator raises on a value it cannot build -- a zero segment
        # count, a tube thicker than its radius. The old mesh stays; the field
        # keeps the number the user typed, so they can correct it.
        return
    doc.set_props(obj.uid, params=edited, was={"params": params})
    doc.set_mesh(obj.uid, mesh)


def _widget(key: str, value: Any, default: Any) -> tuple[Any, bool]:
    """One widget for one parameter, chosen from the default's type."""
    label = f"{key.replace('_', ' ')}##gen{key}"
    if isinstance(default, bool):
        return imgui.checkbox(label, bool(value))[::-1]
    if isinstance(default, int):
        changed, out = imgui.input_int(label, int(value), _STEP.get(key, 1))
        return out, changed
    if isinstance(default, float):
        changed, out = imgui.input_float(label, float(value), 0.05)
        return out, changed
    if isinstance(default, (tuple, list)):
        values = list(value) + [0.0] * (len(default) - len(value))
        if len(default) == 2:
            changed, out = imgui.input_float2(label, values[:2])
        else:
            changed, out = imgui.input_float3(label, values[:3])
        return tuple(float(v) for v in out), changed
    # A parameter type nobody has added yet: shown, not editable, rather than
    # silently dropped from the panel.
    imgui.text_disabled(f"{key}: {value!r}")
    return value, False


def _material(doc: Any, obj: Any) -> None:
    widgets.field_label("material")
    if not doc.materials:
        widgets.muted("the palette is empty")
        return
    options = [(str(i), m.name or f"slot {i}") for i, m in enumerate(doc.materials)]
    picked = widgets.labeled_combo("slot", str(obj.material), options)
    if picked != str(obj.material):
        doc.set_props(obj.uid, material=int(picked))

    index = min(max(int(obj.material), 0), len(doc.materials) - 1)
    material = doc.materials[index]
    _texture_chip(material)
    changed, colour = imgui.color_edit4("base colour##bm", list(material.base_color_factor))
    metal_changed, metallic = imgui.slider_float(
        "metallic##bm", float(material.metallic_factor), 0.0, 1.0
    )
    rough_changed, roughness = imgui.slider_float(
        "roughness##bm", float(material.roughness_factor), 0.0, 1.0
    )
    if changed or metal_changed or rough_changed:
        # A *replacement*, never an in-place edit. Identity is what the GPU
        # cache, ``to_model`` and the writer all de-duplicate on, so editing
        # the object in place would leave every one of them showing the old
        # values with nothing in the data to say why.
        from dataclasses import replace

        # ``replace`` rather than a fresh ``Material``: the five texture slots
        # are fields on it, and building a new one from the three the panel
        # shows would silently delete a baked map an import carried in.
        fresh = replace(
            material,
            base_color_factor=tuple(float(c) for c in colour),
            metallic_factor=float(metallic),
            roughness_factor=float(roughness),
        )
        doc.set_material(index, fresh)


TEXTURE_SLOTS = ("base_color", "metallic_roughness", "normal", "emissive", "occlusion")


def _texture_chip(material: Any) -> None:
    """A read-only line naming the baked maps this material carries.

    Read-only on purpose: Clay paints no textures, and a control that offered
    to replace one would be promising an import/export path that does not
    exist. What it *does* promise is that the maps are still there -- which is
    the thing a user editing an imported asset most needs to know before they
    export it again.
    """
    present = [slot for slot in TEXTURE_SLOTS if getattr(material, slot, None) is not None]
    if not present:
        return
    widgets.muted("baked: " + ", ".join(s.replace("_", " ") for s in present))

"""Words into a recipe change, deterministically -- and the validator every
model-written change goes through.

Two doors, one funnel. ``apply(recipe, text)`` reads plain words ("colder,
more sparks, bigger, no smoke, green flames") with a fixed vocabulary and
turns them into parameter edits; it is what the prompt field does with no
language model on the machine, and it is what the tests can pin. ``apply_diff
(recipe, diff)`` takes a *structured* change -- the shape a language model is
asked to produce -- and lands only what the recipe already knows how to hold:
every value goes through ``Layer.with_param``'s clamp, an unknown layer or
parameter is dropped and *named* in the notes, and the model can therefore
only ever narrow what is already legal. That is the whitelist-as-safety idea
the deleted prompt expander used, applied to numbers.

Both return ``(recipe, notes)``: the notes are the toast, one line per change
made or refused, because a prompt that silently did nothing is worse than one
that says which word it did not know.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from . import prims
from .curves import Curve
from .recipe import Layer, Phase, Recipe, clamp

# -- vocabulary ------------------------------------------------------------------------

#: Colour words -> a warm/hot colour and a cooler/outer one.
COLOURS: dict[str, tuple[str, str]] = {
    "red": ("#FFB090", "#E02010"),
    "orange": ("#FFE0A0", "#FF6A1A"),
    "yellow": ("#FFFFD0", "#FFC020"),
    "gold": ("#FFF4C0", "#E0A020"),
    "green": ("#D8FFC0", "#30C040"),
    "poison": ("#C0FF80", "#40A030"),
    "blue": ("#D0E8FF", "#3070FF"),
    "cyan": ("#E0FFFF", "#20C0E0"),
    "ice": ("#FFFFFF", "#80C0FF"),
    "purple": ("#F0D0FF", "#A040FF"),
    "violet": ("#F0D0FF", "#8020E0"),
    "pink": ("#FFE0F0", "#FF60B0"),
    "white": ("#FFFFFF", "#E0E0FF"),
    "silver": ("#FFFFFF", "#C0C8D8"),
    "black": ("#404040", "#101010"),
    "dark": ("#503060", "#180828"),
    "necrotic": ("#B0FF80", "#204020"),
    "holy": ("#FFFFF0", "#FFE080"),
    "blood": ("#FF4050", "#800810"),
    "fire": ("#FFE0A0", "#FF5010"),
}

#: The parameters a colour word repaints, per primitive: (hot, cool) roles.
_COLOUR_SLOTS: dict[str, tuple[tuple[str, int], ...]] = {
    "core": (("color_inner", 0), ("color_outer", 1)),
    "flame": (("color_base", 0), ("color_tip", 1)),
    "particles": (("color_start", 0), ("color_end", 1)),
    "trail": (("color_head", 0), ("color_tail", 1)),
    "ring": (("color", 0),),
    "flash": (("color", 0),),
    "glow": (("tint", 1),),
    "sprite": (("tint", 0),),
}

#: Size-ish parameters a "bigger"/"smaller" word scales.
_SIZE_PARAMS = {"radius", "width", "height", "size", "spawn_radius", "thickness", "length"}
#: Speed-ish parameters.
_SPEED_PARAMS = {"speed", "rise", "drift", "noise_speed", "pulse_hz", "spin", "flicker_hz"}
#: Count-ish parameters.
_COUNT_PARAMS = {"count"}
#: Brightness-ish parameters.
_BRIGHT_PARAMS = {"intensity", "strength", "alpha", "opacity"}

_SCALE_WORDS: dict[str, tuple[str, float]] = {
    "bigger": ("size", 1.3),
    "larger": ("size", 1.3),
    "huge": ("size", 1.6),
    "wider": ("size", 1.3),
    "smaller": ("size", 0.75),
    "tiny": ("size", 0.5),
    "faster": ("speed", 1.4),
    "quicker": ("speed", 1.4),
    "slower": ("speed", 0.7),
    "brighter": ("bright", 1.3),
    "stronger": ("bright", 1.3),
    "dimmer": ("bright", 0.7),
    "softer": ("bright", 0.7),
    "longer": ("length", 1.3),
    "shorter": ("length", 0.75),
    "wilder": ("turbulence", 1.4),
    "calmer": ("turbulence", 0.7),
}

_MORE = ("more", "extra", "lots of", "many")
_FEWER = ("fewer", "less", "no", "without")

#: Layer-kind words: "sparks" names particles layers, "smoke" smoke layers...
_KIND_WORDS: dict[str, tuple[str, ...]] = {
    "particles": ("sparks", "spark", "embers", "ember", "particles", "debris", "droplets"),
    "smoke": ("smoke", "mist", "dust", "cloud"),
    "flame": ("flames", "flame", "fire"),
    "core": ("core", "ball", "orb"),
    "ring": ("ring", "shockwave", "wave", "halo"),
    "trail": ("trail", "tail"),
    "glow": ("glow", "bloom"),
    "flash": ("flash",),
    "distortion": ("heat", "shimmer", "distortion", "warp"),
}


# -- the deterministic mapper --------------------------------------------------------------


def apply(recipe: Recipe, text: str) -> tuple[Recipe, list[str]]:
    """Plain words -> a changed recipe and what changed. Unknown words are
    ignored silently; a prompt that changed nothing says so in one note."""
    # Punctuation is kept as a token so "blue, no smoke" reads as two clauses:
    # a kind word only binds to the colour or quantifier *before the comma*.
    words = re.findall(r"[a-z]+|[,;.]", text.lower())
    notes: list[str] = []
    if not any(w.isalpha() for w in words):
        return recipe, ["Nothing to apply."]
    layers = list(recipe.layers)

    # Colour: "green flames" repaints the flame layers; a bare colour repaints
    # every layer that has colours.
    for i, word in enumerate(words):
        if word not in COLOURS:
            continue
        hot, cool = COLOURS[word]
        target = _kind_after(words, i)
        touched = 0
        for j, layer in enumerate(layers):
            if target is not None and layer.kind != target:
                continue
            slots = _COLOUR_SLOTS.get(layer.kind)
            if not slots:
                continue
            for name, role in slots:
                layers[j] = layers[j].with_param(name, hot if role == 0 else cool)
            touched += 1
        if touched:
            notes.append(f"{word}: recoloured {touched} layer(s)")

    # "more sparks" / "no smoke": counts and visibility per kind.
    for i, word in enumerate(words):
        if word in _MORE or word in _FEWER:
            target = _kind_after(words, i)
            if target is None:
                continue
            more = word in _MORE
            for j, layer in enumerate(layers):
                if layer.kind != target:
                    continue
                specs = prims.params_of(layer.kind)
                if word in ("no", "without"):
                    layers[j] = replace(layer, visible=False)
                    notes.append(f"hid {layer.name}")
                    continue
                for name in _COUNT_PARAMS & set(specs):
                    layers[j] = _scaled(layers[j], name, 1.5 if more else 0.6)
                if not (_COUNT_PARAMS & set(specs)):
                    # A kind with no count: "more glow" is a stronger one.
                    for name in _BRIGHT_PARAMS & set(specs):
                        layers[j] = _scaled(layers[j], name, 1.3 if more else 0.7)
                notes.append(f"{'more' if more else 'fewer'} {target}")

    # Scale words: "bigger", "faster", "brighter", "longer", "wilder".
    for word in words:
        if word not in _SCALE_WORDS:
            continue
        what, factor = _SCALE_WORDS[word]
        if what == "length":
            phases = tuple(
                replace(p, frames=max(1, int(round(p.frames * factor)))) for p in recipe.phases
            )
            recipe = replace(recipe, phases=phases)
            notes.append(f"{word}: phases x{factor}")
            continue
        names = {
            "size": _SIZE_PARAMS,
            "speed": _SPEED_PARAMS,
            "bright": _BRIGHT_PARAMS,
            "turbulence": {"turbulence", "noise", "raggedness", "unevenness"},
        }[what]
        for j, layer in enumerate(layers):
            for name in names & set(prims.params_of(layer.kind)):
                layers[j] = _scaled(layers[j], name, factor)
        notes.append(f"{word}: x{factor}")

    # Temperature: "hotter" / "colder" nudge every colour toward fire or ice.
    for word in words:
        if word in ("hotter", "warmer", "colder", "cooler"):
            hot, cool = COLOURS["fire"] if word in ("hotter", "warmer") else COLOURS["ice"]
            for j, layer in enumerate(layers):
                for name, role in _COLOUR_SLOTS.get(layer.kind, ()):
                    layers[j] = layers[j].with_param(name, hot if role == 0 else cool)
            notes.append(word)

    if not notes:
        notes.append("No words I know: try colours, bigger/smaller, faster/slower, more sparks.")
    return clamp(replace(recipe, layers=tuple(layers))), notes


def _kind_after(words: list[str], i: int) -> str | None:
    """The primitive kind named by the next word or two of the same clause."""
    for word in words[i + 1 : i + 3]:
        if not word.isalpha():
            return None
        for kind, names in _KIND_WORDS.items():
            if word in names:
                return kind
    return None


def _scaled(layer: Layer, name: str, factor: float) -> Layer:
    specs = prims.params_of(layer.kind)
    spec = specs.get(name)
    if spec is None:
        return layer
    value = layer.params.get(name, spec.default)
    if spec.kind in ("float", "int"):
        return layer.with_param(name, float(value) * factor)
    if spec.kind == "curve":
        curve = Curve.from_json(value)
        scaled = Curve(tuple((t, v * factor) for t, v in curve.keys), curve.easing)
        return layer.with_param(name, scaled.to_json())
    return layer


# -- the structured diff, and its validation ---------------------------------------------------

#: What a language model is told it may return. Kept beside the validator so
#: the prompt and the funnel cannot drift.
DIFF_SCHEMA = (
    '{"seed": int?, "fps": int?, "layers": {"<layer name>": {"<parameter>": value, ...}}?, '
    '"phases": {"<phase name>": {"frames": int?, "loop": bool?}}?, '
    '"hide": ["<layer name>", ...]?, "show": ["<layer name>", ...]?}'
)


def apply_diff(recipe: Recipe, diff: Any) -> tuple[Recipe, list[str]]:
    """Land a structured change. Every value is clamped by the parameter it
    names; anything the recipe cannot hold is dropped and named."""
    notes: list[str] = []
    if not isinstance(diff, dict):
        return recipe, ["The change was not a JSON object."]
    layers = list(recipe.layers)
    by_name = {layer.name.lower(): i for i, layer in enumerate(layers)}
    by_uid = {str(layer.uid): i for i, layer in enumerate(layers)}

    def index_of(key: Any) -> int | None:
        text = str(key).strip().lower()
        return by_name.get(text, by_uid.get(text))

    for key, params in (diff.get("layers") or {}).items():
        i = index_of(key)
        if i is None:
            notes.append(f"no layer called {key!r}")
            continue
        if not isinstance(params, dict):
            notes.append(f"{key}: not a mapping")
            continue
        specs = prims.params_of(layers[i].kind)
        for name, value in params.items():
            if name == "visible":
                layers[i] = replace(layers[i], visible=bool(value))
                notes.append(f"{layers[i].name}: visible={bool(value)}")
                continue
            if name == "opacity":
                try:
                    layers[i] = replace(layers[i], opacity=min(1.0, max(0.0, float(value))))
                    notes.append(f"{layers[i].name}: opacity")
                except (TypeError, ValueError):
                    notes.append(f"{layers[i].name}: opacity is not a number")
                continue
            if name not in specs:
                notes.append(f"{layers[i].name} has no {name!r}")
                continue
            layers[i] = layers[i].with_param(name, value)
            notes.append(f"{layers[i].name}: {name}")
    for key in diff.get("hide") or []:
        i = index_of(key)
        if i is None:
            notes.append(f"no layer called {key!r}")
        else:
            layers[i] = replace(layers[i], visible=False)
            notes.append(f"hid {layers[i].name}")
    for key in diff.get("show") or []:
        i = index_of(key)
        if i is not None:
            layers[i] = replace(layers[i], visible=True)
            notes.append(f"showed {layers[i].name}")
    phases = list(recipe.phases)
    for key, change in (diff.get("phases") or {}).items():
        j = next((k for k, p in enumerate(phases) if p.name.lower() == str(key).lower()), None)
        if j is None or not isinstance(change, dict):
            notes.append(f"no phase called {key!r}")
            continue
        frames = change.get("frames", phases[j].frames)
        loop = change.get("loop", phases[j].loop)
        try:
            phases[j] = Phase(phases[j].name, int(frames), bool(loop))
            notes.append(f"{phases[j].name}: {int(frames)} frames")
        except (TypeError, ValueError):
            notes.append(f"{key}: frames is not a number")
    out = replace(recipe, layers=tuple(layers), phases=tuple(phases))
    for field in ("seed", "fps"):
        if field in diff:
            try:
                out = replace(out, **{field: int(diff[field])})
                notes.append(f"{field}={int(diff[field])}")
            except (TypeError, ValueError):
                notes.append(f"{field} is not a number")
    if not notes:
        notes.append("The change named nothing this effect has.")
    return clamp(out), notes


def describe_for_model(recipe: Recipe) -> dict[str, Any]:
    """The compact view a language model is shown: layer names, kinds, the
    parameters each has with their current values and ranges. No uids, no
    curves' inner keys -- the two things it would only get wrong."""
    layers = []
    for layer in recipe.layers:
        specs = prims.params_of(layer.kind)
        params = {}
        for name, spec in specs.items():
            value = layer.params.get(name, spec.default)
            if spec.kind in ("curve", "life"):
                curve = Curve.from_json(value)
                first, last = curve.keys[0][1], curve.keys[-1][1]
                value = first if curve.is_const else [first, last]
            entry: dict[str, Any] = {"value": value}
            if spec.kind in ("float", "int", "curve", "life"):
                entry["range"] = [spec.lo, spec.hi]
            elif spec.kind == "choice":
                entry["choices"] = list(spec.choices)
            params[name] = entry
        layers.append(
            {"name": layer.name, "kind": layer.kind, "visible": layer.visible, "params": params}
        )
    return {
        "name": recipe.name,
        "seed": recipe.seed,
        "fps": recipe.fps,
        "phases": [{"name": p.name, "frames": p.frames, "loop": p.loop} for p in recipe.phases],
        "layers": layers,
    }

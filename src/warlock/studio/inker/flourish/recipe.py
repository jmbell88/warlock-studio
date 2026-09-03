"""The recipe: what an effect *is*, and its JSON codec.

A recipe is a seed, a canvas, a list of phases and a list of layers. A phase
is a named run of frames with a loop flag -- ``cast``, ``projectile``,
``impact`` -- and a layer is one primitive with its parameters, active in some
or all phases. There is no graph: the layer list *is* the composite order,
bottom first, exactly as Inker's stack is, and that is what lets a recipe
become an Inker layer group with nothing lost in translation.

Two rules of the codec. **Loading clamps.** Every parameter is passed through
its primitive's ``Param.clamp`` on the way in, so a hand-edited preset with a
count of ten thousand or a hex colour with a typo becomes a recipe that
renders, and the text model in a later phase can only ever *narrow* what is
already legal. **Unknown keys are dropped, unknown kinds refused.** A layer of
a kind this build does not know cannot render and is not silently kept as a
blank; a parameter this build does not know is somebody else's problem and is
not carried around to be misread.

Identity is by uid, never by index -- ``Layer.uid`` is what an undo step and a
render cache are addressed by, the rule the rest of Inker follows.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, replace
from typing import Any

from . import prims

SCHEMA_VERSION = 1

BLENDS = ("normal", "add")
MODES = ("painterly", "pixel")

MAX_SIZE = 1024
MAX_SUPERSAMPLE = 8
MAX_FRAMES_PER_PHASE = 240
MAX_PHASES = 12
MAX_LAYERS = 32
MAX_DIRECTIONS = 16
MAX_COLORS = 256

_uids = itertools.count(1)


def new_uid() -> int:
    return next(_uids)


@dataclass(frozen=True)
class Phase:
    name: str
    frames: int = 12
    loop: bool = False


@dataclass(frozen=True)
class Layer:
    uid: int
    kind: str
    name: str = ""
    #: Parameter name -> stored JSON value (number, string, or a curve dict).
    params: dict[str, Any] = field(default_factory=dict)
    blend: str = "normal"
    opacity: float = 1.0
    visible: bool = True
    #: Phase names this layer renders in; empty means every phase.
    phases: tuple[str, ...] = ()

    def active_in(self, phase: str) -> bool:
        return self.visible and (not self.phases or phase in self.phases)

    def with_param(self, name: str, value: Any) -> Layer:
        spec = prims.params_of(self.kind).get(name)
        if spec is None:
            raise KeyError(f"{self.kind} has no parameter {name!r}")
        params = dict(self.params)
        params[name] = spec.clamp(value)
        return replace(self, params=params)


@dataclass(frozen=True)
class Recipe:
    name: str = "Effect"
    seed: int = 1
    width: int = 128
    height: int = 128
    supersample: int = 4
    fps: int = 18
    mode: str = "painterly"
    #: Hex colours for pixel mode; ``None`` means derive one from the render.
    palette: tuple[str, ...] | None = None
    #: How many colours a derived palette gets.
    colors: int = 16
    directions: int = 1
    phases: tuple[Phase, ...] = (Phase("main", 12, True),)
    layers: tuple[Layer, ...] = ()
    version: int = SCHEMA_VERSION

    # -- timing ------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return sum(p.frames for p in self.phases)

    def phase_at(self, frame: int) -> tuple[Phase, int, int]:
        """``(phase, phase_index, frame_within_phase)`` for a global frame."""
        if frame < 0:
            raise IndexError(frame)
        start = 0
        for i, phase in enumerate(self.phases):
            if frame < start + phase.frames:
                return phase, i, frame - start
            start += phase.frames
        raise IndexError(f"frame {frame} is past the recipe's {self.frame_count} frames")

    def phase_start(self, index: int) -> int:
        return sum(p.frames for p in self.phases[:index])

    def phase_named(self, name: str) -> Phase | None:
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    def layer(self, uid: int) -> Layer:
        for layer in self.layers:
            if layer.uid == uid:
                return layer
        raise KeyError(uid)

    def replace_layer(self, layer: Layer) -> Recipe:
        layers = tuple(layer if each.uid == layer.uid else each for each in self.layers)
        return replace(self, layers=layers)


# -- clamping -------------------------------------------------------------------


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return int(min(hi, max(lo, int(value))))
    except (TypeError, ValueError):
        return default


def clamp_layer(layer: Layer) -> Layer:
    specs = prims.params_of(layer.kind)
    params = {
        name: spec.clamp(layer.params.get(name, spec.default)) for name, spec in specs.items()
    }
    return replace(
        layer,
        params=params,
        blend=layer.blend if layer.blend in BLENDS else "normal",
        opacity=min(1.0, max(0.0, float(layer.opacity))),
        name=str(layer.name or layer.kind),
    )


def clamp(recipe: Recipe) -> Recipe:
    """Every field inside its range; every layer's parameters filled and clamped."""
    phases = tuple(
        Phase(
            str(p.name or f"phase{i}"),
            _clamp_int(p.frames, 1, MAX_FRAMES_PER_PHASE, 12),
            bool(p.loop),
        )
        for i, p in enumerate(recipe.phases[:MAX_PHASES])
    ) or (Phase("main", 12, True),)
    names = {p.name for p in phases}
    layers = tuple(
        replace(clamp_layer(each), phases=tuple(n for n in each.phases if n in names))
        for each in recipe.layers[:MAX_LAYERS]
    )
    palette = None
    if recipe.palette:
        kept = []
        for entry in recipe.palette:
            try:
                prims.parse_color(str(entry))
            except ValueError:
                continue
            kept.append(str(entry))
        palette = tuple(kept) or None
    return replace(
        recipe,
        name=str(recipe.name or "Effect"),
        seed=int(recipe.seed) & 0x7FFFFFFF,
        width=_clamp_int(recipe.width, 8, MAX_SIZE, 128),
        height=_clamp_int(recipe.height, 8, MAX_SIZE, 128),
        supersample=_clamp_int(recipe.supersample, 1, MAX_SUPERSAMPLE, 4),
        fps=_clamp_int(recipe.fps, 1, 120, 18),
        mode=recipe.mode if recipe.mode in MODES else "painterly",
        palette=palette,
        colors=_clamp_int(recipe.colors, 2, MAX_COLORS, 16),
        directions=_clamp_int(recipe.directions, 1, MAX_DIRECTIONS, 1),
        phases=phases,
        layers=layers,
        version=SCHEMA_VERSION,
    )


# -- codec ----------------------------------------------------------------------


def to_dict(recipe: Recipe) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "name": recipe.name,
        "seed": recipe.seed,
        "size": [recipe.width, recipe.height],
        "supersample": recipe.supersample,
        "fps": recipe.fps,
        "mode": recipe.mode,
        "palette": list(recipe.palette) if recipe.palette else None,
        "colors": recipe.colors,
        "directions": recipe.directions,
        "phases": [{"name": p.name, "frames": p.frames, "loop": p.loop} for p in recipe.phases],
        "layers": [
            {
                "uid": each.uid,
                "kind": each.kind,
                "name": each.name,
                "params": dict(each.params),
                "blend": each.blend,
                "opacity": each.opacity,
                "visible": each.visible,
                "phases": list(each.phases),
            }
            for each in recipe.layers
        ],
    }


def from_dict(raw: dict[str, Any]) -> Recipe:
    """A clamped recipe from JSON data. Raises ``ValueError`` on a layer kind
    this build does not know or a payload that is not a recipe at all."""
    if not isinstance(raw, dict):
        raise ValueError("a recipe is a JSON object")
    version = int(raw.get("version") or SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(f"recipe version {version} is newer than this build's {SCHEMA_VERSION}")
    size = raw.get("size") or [128, 128]
    phases = tuple(
        Phase(str(p.get("name", "")), int(p.get("frames", 12)), bool(p.get("loop", False)))
        for p in (raw.get("phases") or [])
        if isinstance(p, dict)
    )
    layers = []
    seen: set[int] = set()
    for entry in raw.get("layers") or []:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "")
        if kind not in prims.KINDS:
            raise ValueError(f"{kind!r} is not a primitive this build knows")
        uid = entry.get("uid")
        uid = int(uid) if isinstance(uid, int) and uid > 0 and uid not in seen else new_uid()
        seen.add(uid)
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        layers.append(
            Layer(
                uid=uid,
                kind=kind,
                name=str(entry.get("name") or kind),
                params=dict(params),
                blend=str(entry.get("blend") or "normal"),
                opacity=_as_float(entry.get("opacity"), 1.0),
                visible=bool(entry.get("visible", True)),
                phases=tuple(str(n) for n in (entry.get("phases") or [])),
            )
        )
    palette = raw.get("palette")
    recipe = Recipe(
        name=str(raw.get("name") or "Effect"),
        seed=int(raw.get("seed") or 1),
        width=int(size[0]),
        height=int(size[1]),
        supersample=int(raw.get("supersample", 4)),
        fps=int(raw.get("fps", 18)),
        mode=str(raw.get("mode") or "painterly"),
        palette=tuple(str(c) for c in palette) if isinstance(palette, list) else None,
        colors=int(raw.get("colors", 16)),
        directions=int(raw.get("directions", 1)),
        phases=phases,
        layers=tuple(layers),
    )
    return clamp(recipe)


def dumps(recipe: Recipe) -> str:
    return json.dumps(to_dict(recipe), indent=2, sort_keys=True)


def loads(text: str) -> Recipe:
    return from_dict(json.loads(text))


def bump_uids(recipe: Recipe) -> Recipe:
    """The same recipe with fresh layer uids -- for inserting a preset twice."""
    return replace(recipe, layers=tuple(replace(each, uid=new_uid()) for each in recipe.layers))


def reserve_uids(recipe: Recipe) -> None:
    """Advance the uid counter past every uid in ``recipe`` so a layer added
    later cannot collide with one that was loaded."""
    global _uids
    top = max((each.uid for each in recipe.layers), default=0)
    current = next(_uids)
    _uids = itertools.count(max(current, top + 1))

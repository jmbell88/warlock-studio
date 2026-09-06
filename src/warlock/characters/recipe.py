"""One flat record of everything a character sheet needs, validated at the door.

Flat on purpose. A recipe is written by a command bar, edited by a settings
column, stored in a job's ``params``, read back by a rerun and diffed against a
previous one -- and every one of those is easier over sixteen scalar keys than
over a tree. Nesting appears exactly once, in ``appearance``, because the
channel set belongs to the species and cannot be a fixed list of columns.

**It refuses; it never clamps.** Every rejection is a
:class:`~warlock.characters.errors.CharacterError` naming the ``field`` it came
from, which ``service.errors.invalid_from`` passes straight through to the
control. The rule is the one ``service/jobs`` already states about tile size: a
request for 96px tiles answered with 32px tiles and nobody told is the failure
mode, and a slider quietly snapping back is the same failure wearing a nicer
coat.

The vocabulary this module checks against -- the size ladder, the colour ladder,
the outline and reduce modes -- is **restated here rather than imported**,
because ``characters`` may not import ``service`` and the pixel ladders live
behind it. ``tests/characters/test_recipe.py`` owns the agreement between the
two copies, in the ``pipelines.charsheet`` / ``studio.troupe.spec`` arrangement:
a change to one is a change to both plus that test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .errors import CharacterError
from .family import Family, get_family

__all__ = [
    "COLOR_CHOICES",
    "DEFAULT_RECIPE",
    "DIRECTION_CHOICES",
    "LOGICAL_SIZES",
    "OUTLINE_MODES",
    "REDUCE_MODES",
    "THEMES",
    "VERSION",
    "Recipe",
]

#: The recipe format's own version, bumped when a key changes meaning. Not the
#: species' version -- that lives on the row in ``family`` and moves when the
#: baked asset does.
VERSION = 1

#: ``pipelines.charsheet.SIZES``. See the module docstring for why it is copied.
LOGICAL_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128)
#: ``service.troupe.TROUPE_COLOR_CHOICES``.
COLOR_CHOICES: tuple[int, ...] = (8, 16, 32, 64)
#: ``pipelines.pixelize.OUTLINE_MODES`` / ``REDUCE_MODES``.
OUTLINE_MODES: tuple[str, ...] = ("none", "inner", "outer")
REDUCE_MODES: tuple[str, ...] = ("box", "point")
#: ``pipelines.charsheet.DIRECTION_PRESETS``' keys.
DIRECTION_CHOICES: tuple[int, ...] = (1, 4, 8, 16)

#: The most a name may be, matching ``rigging.validate_pose``'s cap so a
#: character and a pose cannot disagree about what a long name is.
MAX_NAME = 64


def _number(raw: Any, field_name: str, what: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise CharacterError(f"{what} must be a number", field=field_name) from None
    if value != value or value in (float("inf"), float("-inf")):
        raise CharacterError(f"{what} must be a real number", field=field_name)
    return value


def _integer(raw: Any, field_name: str, what: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise CharacterError(f"{what} must be a whole number", field=field_name) from None


def _on_ladder(value: int, ladder: tuple[int, ...], field_name: str, what: str) -> int:
    if value not in ladder:
        raise CharacterError(f"{what} must be one of {list(ladder)}", field=field_name)
    return value


@dataclass(frozen=True)
class Recipe:
    """A request for one character sheet, whole."""

    family: str
    family_version: int
    appearance: Mapping[str, float]
    theme: str
    camera: str
    elevation: float
    #: ``movement -> frames``. A ``charsheet`` movement name, so the layout this
    #: expands into is the one Troupe already renders; a movement this program
    #: has no clip for is refused here rather than at the renderer.
    animations: Mapping[str, int]
    directions: int
    logical_size: int
    colors: int
    outline: str
    reduce_mode: str
    dither: bool
    palette: str
    seed: int
    name: str

    @property
    def spec(self) -> Family:
        return get_family(self.family)

    @property
    def archetype(self) -> str:
        """The body plan this species belongs to. Derived, never stored: a
        recipe carrying its own copy would be one edit away from a request whose
        archetype and species name different skeletons."""
        return self.spec.archetype

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None = None) -> Recipe:
        """Validate a request. Absent keys take the species' defaults."""
        raw = dict(raw or {})
        fam = get_family(raw.get("family", DEFAULT_FAMILY))

        version = _integer(
            raw.get("family_version", fam.version), "family_version", "family version"
        )
        if version > fam.version:
            # By name, not by falling back: a recipe written by a newer build
            # describes a mesh this one does not have, and quietly rebuilding it
            # from the older asset is how a saved character comes back looking
            # like somebody else.
            raise CharacterError(
                f"this {fam.label} was made with {fam.label} v{version}, and this "
                f"build ships v{fam.version}",
                field="family_version",
            )

        appearance = dict(fam.appearance_defaults())
        raw_appearance = raw.get("appearance") or {}
        if not isinstance(raw_appearance, Mapping):
            raise CharacterError("appearance must be a set of named sliders", field="appearance")
        by_key = {c.key: c for c in fam.channels}
        for key, value in raw_appearance.items():
            channel = by_key.get(str(key))
            if channel is None:
                raise CharacterError(
                    f"{fam.label} has no {key!r} slider; try "
                    + ", ".join(sorted(by_key)),
                    field="appearance",
                )
            number = _number(value, "appearance", channel.label)
            if not channel.lo <= number <= channel.hi:
                raise CharacterError(
                    f"{channel.label} must be between {channel.lo:g} and {channel.hi:g}",
                    field="appearance",
                )
            appearance[channel.key] = number

        theme = str(raw.get("theme") or fam.themes[0].key)
        fam.theme(theme)  # raises, carrying field="theme"

        camera = str(raw.get("camera") or DEFAULT_CAMERA)
        _check_camera(camera)

        elevation = _number(raw.get("elevation", DEFAULT_ELEVATION), "elevation", "elevation")
        if not -89.0 <= elevation <= 89.0:
            raise CharacterError("elevation must be between -89 and 89 degrees", field="elevation")

        animations = _check_animations(raw.get("animations"))

        directions = _on_ladder(
            _integer(raw.get("directions", 8), "directions", "directions"),
            DIRECTION_CHOICES, "directions", "directions",
        )
        logical_size = _on_ladder(
            _integer(raw.get("logical_size", 64), "logical_size", "logical size"),
            LOGICAL_SIZES, "logical_size", "logical size",
        )
        colors = _on_ladder(
            _integer(raw.get("colors", 32), "colors", "colours"),
            COLOR_CHOICES, "colors", "colours",
        )

        outline = str(raw.get("outline") or "outer")
        if outline not in OUTLINE_MODES:
            raise CharacterError(
                f"outline must be one of {list(OUTLINE_MODES)}", field="outline"
            )
        reduce_mode = str(raw.get("reduce_mode") or "box")
        if reduce_mode not in REDUCE_MODES:
            raise CharacterError(
                f"reduce_mode must be one of {list(REDUCE_MODES)}", field="reduce_mode"
            )

        seed = _integer(raw.get("seed", 0), "seed", "seed")
        if seed < 0:
            raise CharacterError("seed must not be negative", field="seed")

        name = str(raw.get("name") or fam.label).strip()
        if not name:
            raise CharacterError("a character needs a name", field="name")
        if len(name) > MAX_NAME:
            raise CharacterError(f"a name is at most {MAX_NAME} characters", field="name")

        return cls(
            family=fam.key,
            family_version=version,
            appearance=appearance,
            theme=theme,
            camera=camera,
            elevation=elevation,
            animations=animations,
            directions=directions,
            logical_size=logical_size,
            colors=colors,
            outline=outline,
            reduce_mode=reduce_mode,
            dither=bool(raw.get("dither", False)),
            palette=str(raw.get("palette") or "").strip(),
            seed=seed,
            name=name,
        )

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["appearance"] = dict(self.appearance)
        out["animations"] = dict(self.animations)
        out["version"] = VERSION
        return out

    def replace(self, **changes: Any) -> Recipe:
        return Recipe.from_dict({**self.as_dict(), **changes})

    # -- what it expands into -----------------------------------------------

    def layout_payload(self) -> dict[str, Any]:
        """The v2 Troupe layout request this recipe means."""
        return {
            "version": 2,
            "movements": [
                {"key": name, "frames": frames, "directions": self.directions}
                for name, frames in self.animations.items()
            ],
        }

    def layout_dict(self) -> dict[str, Any]:
        """The resolved frame table, as ``charsheet`` states it.

        Through ``charsheet.resolve_layout`` rather than by counting here: the
        cell order and the run table are what the renderer and the sidecar
        agree on, and a second arithmetic in this module would be a second
        opinion about what cell 137 depicts.
        """
        from ..pipelines import charsheet

        return charsheet.resolve_layout(self.layout_payload()).as_dict()

    @property
    def cell_count(self) -> int:
        return sum(frames for frames in self.animations.values()) * self.directions


def _check_camera(camera: str) -> None:
    from ..pipelines import charsheet

    keys = [key for key, _label, _elev in charsheet.CAMERA_PRESETS]
    if camera not in keys:
        raise CharacterError(f"camera must be one of {keys}", field="camera")


def _check_animations(raw: Any) -> dict[str, int]:
    from ..pipelines import charsheet

    if raw is None:
        raw = dict(DEFAULT_ANIMATIONS)
    if not isinstance(raw, Mapping) or not raw:
        raise CharacterError("a character sheet needs at least one animation", field="animations")
    known = {name for name, *_rest in charsheet.ANIMATIONS}
    out: dict[str, int] = {}
    for name, frames in raw.items():
        key = str(name)
        if key not in known:
            raise CharacterError(
                f"{key!r} is not an animation; try " + ", ".join(sorted(known)),
                field="animations",
            )
        count = _integer(frames, "animations", f"{key} frames")
        low = charsheet.MOVEMENT_MIN_FRAMES[key]
        if not low <= count <= charsheet.MAX_FRAMES:
            raise CharacterError(
                f"{key} must have {low}-{charsheet.MAX_FRAMES} frames", field="animations"
            )
        out[key] = count
    return out


DEFAULT_FAMILY = "ogre"


def _default_camera() -> tuple[str, float]:
    """The default preset and the angle it means, read from the one table.

    Not two literals here. ``charsheet.CAMERA_PRESETS`` is the table the pane
    offers from, the door validates against and the Blender worker frames the
    ortho window with, and
    ``tests/troupe/test_camera_presets.py::test_the_form_and_the_door_read_one_preset_table``
    scans the whole package for a second module spelling the default key --
    which is exactly how this file's first draft was caught. A preset whose
    angle is edited in one place and copied in another is a form offering a
    framing nothing renders.

    Function-scoped like every other ``charsheet`` reach in this module, so the
    package's import pin keeps measuring module-level imports.
    """
    from ..pipelines import charsheet

    key = charsheet.DEFAULT_CAMERA_PRESET
    elevation = next(
        angle for preset, _label, angle in charsheet.CAMERA_PRESETS if preset == key
    )
    return key, float(elevation)


DEFAULT_CAMERA, DEFAULT_ELEVATION = _default_camera()
#: idle, walk and attack -- **not** run and jump. Not a subset chosen for
#: brevity: those three are what a character needs to read as alive in a top-down
#: game, and the legacy five-animation ``charsheet.ANIMATIONS`` table is
#: untouched so every sheet Troupe has ever rendered still means what it did.
DEFAULT_ANIMATIONS: Mapping[str, int] = {"idle": 4, "walk": 8, "attack": 6}

#: The brief, exactly: a fire ogre seen 3/4 top-down, eight directions, 64px
#: logical cells, 32 colours, an outer outline -- 18 frames across 8 directions,
#: 144 cells.
DEFAULT_RECIPE: Recipe = Recipe.from_dict(
    {
        "family": DEFAULT_FAMILY,
        "theme": "fire",
        "camera": DEFAULT_CAMERA,
        "elevation": DEFAULT_ELEVATION,
        "animations": dict(DEFAULT_ANIMATIONS),
        "directions": 8,
        "logical_size": 64,
        "colors": 32,
        "outline": "outer",
    }
)


def _themes() -> dict[str, tuple[str, ...]]:
    from .family import families

    return {key: tuple(t.key for t in fam.themes) for key, fam in families().items()}


#: ``species -> the look keys it offers``, for a picker that has no reason to
#: hold a whole registry.
THEMES: dict[str, tuple[str, ...]] = _themes()

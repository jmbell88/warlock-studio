"""What a character *is*, as data: archetypes, species, channels, themes.

Two registries, deliberately, because two different things were being called a
"family" in the first draft of this layer and they have different lifetimes:

* An :class:`Archetype` is a **body plan**. It owns the rig skeleton key, the
  clip library key, the attachment sockets and the material region names --
  everything a pose, a clip and a sprite sheet are expressed in terms of. There
  are four of them and adding a fifth is a rig template plus a clip library plus
  a generator, which is a programme and not a parameter.
* A :class:`Family` is a **species**: one row of generator parameters, one set of
  channel defaults, one palette of themes, one height. Adding a species is a row
  in :data:`_FAMILIES`. That asymmetry is the whole point of the split -- a
  knight and a zombie are a human with a different palette and two nudged
  channels, and a registry that hid that behind two hand-modelled meshes would
  be lying about the cost of the next one.

**A species does not own a mesh; a silhouette group does.** Every species names
a ``silhouette``, and the baked asset (``<silhouette>.glb`` plus
``<silhouette>.masks.npz``) is shared by every species that names it. What
separates two species inside a group is entirely channel defaults, height and
materials -- all of which are applied at instantiation from the same base mesh.
Two species land in *different* groups only when they differ topologically
(tusks, ear shape, a snout), because no smooth displacement field can grow a
tusk. That is the bake decision, and ``scripts/author_humanoid.py`` reports the
numbers it rests on.

This module imports nothing but the standard library and :mod:`.errors`. It is
read by the door, by the worker and by a resolver that has to turn the words in
a prompt into a species key, and none of those three may drag numpy, a GL
context or ``service`` in behind it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .errors import CharacterError

__all__ = [
    "ARCHETYPE_KEYS",
    "Archetype",
    "Channel",
    "Family",
    "Socket",
    "Theme",
    "archetypes",
    "families",
    "families_of",
    "get_archetype",
    "get_family",
    "silhouettes",
]

#: The four body plans. Named as a tuple as well as a registry so a caller can
#: check membership without building the registry -- a resolver's vocabulary
#: table is built at import time and has no business constructing dataclasses.
ARCHETYPE_KEYS: tuple[str, ...] = ("humanoid", "quadruped", "winged", "amorphous")


@dataclass(frozen=True, slots=True)
class Channel:
    """One continuous appearance control, in channel units.

    ``default`` is the **species'** value, not a neutral: the baked mesh is the
    silhouette group's channel-zero shape and a species is the offsets that turn
    it into itself. So ``get_family("ogre")`` already reads +0.9 bulk, and a
    recipe that says nothing about appearance still produces an ogre rather than
    a large human.

    ``lo``/``hi`` are the range a *user* may ask for. They are not clamped
    anywhere: :meth:`Recipe.from_dict` refuses an out-of-range value by name,
    because a silently clamped slider is how a character comes back not looking
    like the one that was asked for with nothing on screen saying why.
    """

    key: str
    label: str
    default: float
    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class Socket:
    """Where a prop hangs off a skeleton.

    ``offset`` is ``(along, lateral, up)`` in **bone-length units** off the
    bone's head, so it survives a species being three times as tall; ``reach``
    is the radius, in the same units, a prop may occupy before it clips the
    body. Data only -- nothing in this increment attaches anything.
    """

    name: str
    bone: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reach: float = 1.0


@dataclass(frozen=True, slots=True)
class Theme:
    """A palette over an archetype's regions, plus what it emits.

    ``materials`` maps a region name to ``#rrggbb``. Every region the archetype
    declares must appear: a theme with a hole would fall back to whatever the
    baked material happened to be, which is exactly the silent-wrong-colour
    failure the region list exists to prevent.

    ``effects`` and ``effect_params`` are carried but unused here -- the
    Flourish edge is Increment 4's, and a theme that names an effect nothing
    reads is still the honest place to record that "fire" means embers.
    """

    key: str
    label: str
    materials: Mapping[str, str]
    effects: tuple[str, ...] = ()
    effect_params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Archetype:
    """A body plan: the rig, the clips, the sockets and the region names."""

    key: str
    label: str
    #: ``rigging.get_template`` key. Not a file path -- the registry owns it.
    template: str
    #: ``rigging.clip_library`` key. The same string today for every archetype
    #: whose skeleton it shares; kept separate because a body plan may
    #: eventually want its own walk without wanting its own skeleton.
    clip_library: str
    sockets: tuple[Socket, ...]
    #: Material slots, **in index order**. A baked mesh stores one region id per
    #: face and that id is an index into this tuple, so reordering it silently
    #: repaints every checked-in asset. Append only.
    regions: tuple[str, ...]
    #: The channel *set*, with neutral defaults. A species overrides the values.
    channels: tuple[Channel, ...]
    #: The subpackage under ``warlock.characters`` holding the generator and the
    #: baked assets.
    package: str

    def channel(self, key: str) -> Channel:
        for channel in self.channels:
            if channel.key == key:
                return channel
        raise CharacterError(f"{key!r} is not an appearance channel", field="appearance")


@dataclass(frozen=True, slots=True)
class Family:
    """One species: parameters, defaults, palette, height."""

    key: str
    version: int
    label: str
    #: An :data:`ARCHETYPE_KEYS` member, not an :class:`Archetype` -- the
    #: registry resolves it, so a species row stays readable as one line.
    archetype: str
    #: The baked asset stem this species is grown from. Several species share
    #: one; see the module docstring.
    silhouette: str
    #: Words a prompt may use for this species, longest first is *not* required
    #: -- ordering is the resolver's problem, spelling is this table's.
    aliases: tuple[str, ...]
    height_m: float
    #: ``channel key -> this species' default``. Every key must be one of the
    #: archetype's channels; a key that is not is a broken build.
    channel_defaults: Mapping[str, float]
    themes: tuple[Theme, ...]
    #: Species a resolver may offer when this one is not what was asked for,
    #: nearest first. Every entry names a real species, which
    #: ``test_every_nearest_hint_names_a_real_species`` pins.
    nearest: tuple[str, ...] = ()

    # -- what the archetype owns, readable off the species ------------------
    #
    # Delegated rather than duplicated: a caller holding a Family wants to know
    # which template to rig with and never wants to have gone to a second
    # registry to find out, but a copy on the species row would be one edit away
    # from a species whose skeleton disagrees with its own clips.

    @property
    def arch(self) -> Archetype:
        return get_archetype(self.archetype)

    @property
    def template(self) -> str:
        return self.arch.template

    @property
    def clip_library(self) -> str:
        return self.arch.clip_library

    @property
    def sockets(self) -> tuple[Socket, ...]:
        return self.arch.sockets

    @property
    def regions(self) -> tuple[str, ...]:
        return self.arch.regions

    @property
    def channels(self) -> tuple[Channel, ...]:
        """The archetype's channels, carrying *this species'* defaults."""
        return tuple(
            Channel(c.key, c.label, float(self.channel_defaults.get(c.key, c.default)), c.lo, c.hi)
            for c in self.arch.channels
        )

    @property
    def asset_dir(self) -> Path:
        return Path(__file__).parent / self.arch.package

    @property
    def base_glb(self) -> Path:
        return self.asset_dir / f"{self.silhouette}.glb"

    @property
    def masks_npz(self) -> Path:
        return self.asset_dir / f"{self.silhouette}.masks.npz"

    def theme(self, key: str) -> Theme:
        for theme in self.themes:
            if theme.key == key:
                return theme
        raise CharacterError(
            f"{self.label} has no {key!r} look; try "
            + ", ".join(t.key for t in self.themes),
            field="theme",
        )

    def appearance_defaults(self) -> dict[str, float]:
        return {c.key: c.default for c in self.channels}


# --- the humanoid archetype --------------------------------------------------

_HUMANOID_CHANNELS: tuple[Channel, ...] = (
    Channel("bulk", "Bulk", 0.0, -1.0, 1.0),
    Channel("stature", "Leg length", 0.0, -1.0, 1.0),
    Channel("head_size", "Head size", 0.0, -1.0, 1.0),
    Channel("limb_length", "Limb length", 0.0, -1.0, 1.0),
    Channel("shoulder_width", "Shoulder width", 0.0, -1.0, 1.0),
    Channel("hunch", "Hunch", 0.0, -1.0, 1.0),
)

#: Material slots, in the order the baked region ids index. Append only.
_HUMANOID_REGIONS: tuple[str, ...] = (
    "skin",
    "belly",
    "tooth",
    "eye",
    "garment",
    "accent",
)

_HUMANOID_SOCKETS: tuple[Socket, ...] = (
    Socket("weapon_main", "hand.R", (1.0, 0.0, 0.0), 3.0),
    Socket("weapon_off", "hand.L", (1.0, 0.0, 0.0), 3.0),
    Socket("back", "chest", (0.4, 0.0, -0.6), 2.0),
    Socket("crown", "head", (1.0, 0.0, 0.0), 1.2),
    Socket("belt", "hips", (0.6, 0.0, 0.0), 1.5),
)

# --- the quadruped archetype -------------------------------------------------

_QUADRUPED_CHANNELS: tuple[Channel, ...] = (
    Channel("bulk", "Bulk", 0.0, -1.0, 1.0),
    Channel("leg_length", "Leg length", 0.0, -1.0, 1.0),
    Channel("neck_length", "Neck length", 0.0, -1.0, 1.0),
    Channel("head_size", "Head size", 0.0, -1.0, 1.0),
    Channel("body_length", "Body length", 0.0, -1.0, 1.0),
    Channel("tail_length", "Tail length", 0.0, -1.0, 1.0),
)

#: Material slots, in the order the baked region ids index. Append only.
#:
#: ``horn`` is every keratin surface at once -- hooves, claws, antlers, tusks
#: and the scaled group's dorsal ridge -- because one material is what they all
#: are, and a separate ``hoof`` slot would have been a colour every theme had to
#: name and no theme ever wanted to differ on.
_QUADRUPED_REGIONS: tuple[str, ...] = (
    "hide",
    "underbelly",
    "horn",
    "eye",
    "mane",
    "accent",
)

_QUADRUPED_SOCKETS: tuple[Socket, ...] = (
    Socket("saddle", "spine", (0.5, 0.0, 0.9), 1.6),
    Socket("collar", "neck", (0.6, 0.0, 0.5), 1.2),
    Socket("crown", "head", (0.5, 0.0, 0.7), 1.2),
    Socket("pack", "hips", (0.4, 0.0, 0.9), 1.6),
)


# --- the winged archetype ----------------------------------------------------

_WINGED_CHANNELS: tuple[Channel, ...] = (
    Channel("bulk", "Bulk", 0.0, -1.0, 1.0),
    Channel("wingspan", "Wing span", 0.0, -1.0, 1.0),
    Channel("neck_length", "Neck length", 0.0, -1.0, 1.0),
    Channel("tail_length", "Tail length", 0.0, -1.0, 1.0),
    Channel("head_size", "Head size", 0.0, -1.0, 1.0),
    Channel("leg_length", "Leg length", 0.0, -1.0, 1.0),
)

_WINGED_REGIONS: tuple[str, ...] = (
    "hide",
    "underbelly",
    "wing",
    "beak",
    "eye",
    "horn",
    "accent",
)

_WINGED_SOCKETS: tuple[Socket, ...] = (
    Socket("saddle", "chest", (0.6, 0.0, 0.9), 1.6),
    Socket("crown", "head", (0.6, 0.0, 0.7), 1.2),
    Socket("talon", "foot.R", (1.0, 0.0, 0.0), 1.5),
    Socket("back", "spine", (0.4, 0.0, -0.8), 1.6),
)


# --- the amorphous archetype -------------------------------------------------

_AMORPHOUS_CHANNELS: tuple[Channel, ...] = (
    Channel("bulk", "Bulk", 0.0, -1.0, 1.0),
    Channel("viscosity", "Viscosity", 0.0, -1.0, 1.0),
    Channel("lobe", "Lobes", 0.0, -1.0, 1.0),
    Channel("ripple", "Surface", 0.0, -1.0, 1.0),
    Channel("crown", "Crown", 0.0, -1.0, 1.0),
)

#: ``core`` is the nucleus, and it is a *surface* region rather than an inner
#: shell for a reason worth recording: the sprite pipeline's ``_make_flat``
#: rewires every material to emit its own base colour and drops alpha on the
#: way, so a translucent body with something suspended in it renders as an
#: opaque body with nothing in it. The nucleus therefore breaks the surface --
#: shards for an elemental, a bright pool for a slime -- and translucency is a
#: palette, not a material property. See ``characters/amorphous/generate.py``.
_AMORPHOUS_REGIONS: tuple[str, ...] = (
    "body",
    "core",
    "eye",
    "accent",
)

_AMORPHOUS_SOCKETS: tuple[Socket, ...] = (
    Socket("crown", "top", (1.0, 0.0, 0.0), 1.2),
    Socket("core", "core", (0.5, 0.0, 0.0), 1.0),
)


_ARCHETYPES: dict[str, Archetype] = {
    "humanoid": Archetype(
        key="humanoid",
        label="Humanoid",
        # The shipped 19-bone skeleton and the shipped clip library, unchanged.
        # A species that wants its arms lower says so with a rest offset in its
        # own row, never with a second copy of the walk cycle -- two libraries
        # that start identical are two libraries that drift.
        template="humanoid",
        clip_library="humanoid",
        sockets=_HUMANOID_SOCKETS,
        regions=_HUMANOID_REGIONS,
        channels=_HUMANOID_CHANNELS,
        package="humanoid",
    ),
    "quadruped": Archetype(
        key="quadruped",
        label="Quadruped",
        template="quadruped",
        clip_library="quadruped",
        sockets=_QUADRUPED_SOCKETS,
        regions=_QUADRUPED_REGIONS,
        channels=_QUADRUPED_CHANNELS,
        package="quadruped",
    ),
    "winged": Archetype(
        key="winged",
        label="Winged",
        # The ``bird`` template, and a dragon is one of its species rather than
        # a substitution for one: the wing chain is
        # ``wing_{base,mid,tip}.{L,R}`` either way, and what separates a
        # membrane from a feather is a mesh the generator grows over that
        # chain. A second skeleton for the same three bones would be two
        # skeletons that start identical and drift.
        template="bird",
        clip_library="bird",
        sockets=_WINGED_SOCKETS,
        regions=_WINGED_REGIONS,
        channels=_WINGED_CHANNELS,
        package="winged",
    ),
    "amorphous": Archetype(
        key="amorphous",
        label="Amorphous",
        template="blob",
        clip_library="blob",
        sockets=_AMORPHOUS_SOCKETS,
        regions=_AMORPHOUS_REGIONS,
        channels=_AMORPHOUS_CHANNELS,
        package="amorphous",
    ),
}


# --- themes ------------------------------------------------------------------


def _theme(key: str, label: str, **hexes: str) -> Theme:
    """A palette over :data:`_HUMANOID_REGIONS`, keyword by region.

    Written as a helper rather than twelve dict literals so a missing region is
    a ``TypeError`` at import rather than a black face at render time --
    ``test_every_theme_paints_every_region`` makes the same claim over the
    registry, but failing at import is the earlier and cheaper of the two.
    """
    effects = ()
    params: dict[str, float] = {}
    if key == "fire":
        effects = ("embers",)
        params = {"rate": 24.0, "rise": 0.35}
    return Theme(key, label, dict(hexes), effects, params)


def _skin_theme(key: str, label: str, skin: str, garment: str, accent: str) -> Theme:
    """The common shape: skin and belly a shade apart, bone-white teeth."""
    return _theme(
        key,
        label,
        skin=skin,
        belly=_lighten(skin, 0.12),
        tooth="#efe6cf",
        eye="#1b1410",
        garment=garment,
        accent=accent,
    )


def _lighten(hexcolor: str, amount: float) -> str:
    r = int(hexcolor[1:3], 16)
    g = int(hexcolor[3:5], 16)
    b = int(hexcolor[5:7], 16)
    mix = lambda v: min(255, int(round(v + (255 - v) * amount)))  # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


_COMMON_THEMES = {
    "human": (
        _skin_theme("natural", "Natural", "#c68a63", "#5a4632", "#8a3a2a"),
        _skin_theme("ashen", "Ashen", "#8e8577", "#3b3a38", "#5c6b7a"),
    ),
    "elf": (
        _skin_theme("natural", "Natural", "#e0c3a4", "#2f5a44", "#c8a33a"),
        _skin_theme("moonlit", "Moonlit", "#d9dced", "#2a3358", "#9fd8e8"),
    ),
    "dwarf": (
        _skin_theme("natural", "Natural", "#c08055", "#4a3324", "#b5651d"),
        _skin_theme("forge", "Forge", "#a9704b", "#3a2a20", "#e2571e"),
    ),
    "goblin": (
        _skin_theme("natural", "Natural", "#7ea24e", "#4b3a26", "#c2452d"),
        _skin_theme("swamp", "Swamp", "#5d7f43", "#3c4028", "#8fbf3a"),
    ),
    "orc": (
        _skin_theme("natural", "Natural", "#6f8f5a", "#3d2f22", "#9b3226"),
        _skin_theme("blood", "Blood", "#7d6a58", "#41221c", "#c22a1c"),
    ),
    "ogre": (
        _skin_theme("natural", "Natural", "#9a8a63", "#5b4630", "#7a4a2a"),
        # The brief's default. ``accent`` is the crack region the generator
        # picks out by noise threshold, and it is the one region a theme lights.
        _theme(
            "fire",
            "Fire",
            skin="#6b4a35",
            belly="#8a6244",
            tooth="#efe6cf",
            eye="#ffb43a",
            garment="#3a2a1e",
            accent="#ff5a1e",
        ),
        _skin_theme("stone", "Stone", "#8d8d86", "#4a4a44", "#5f7a8a"),
    ),
    "troll": (
        _skin_theme("natural", "Natural", "#5f7a6a", "#3a3428", "#8a5a3a"),
        _skin_theme("frost", "Frost", "#8fb6c8", "#2f3d4a", "#d8f0ff"),
    ),
    "skeleton": (
        _theme(
            "natural",
            "Bone",
            skin="#ddd6c0",
            belly="#cfc7ae",
            tooth="#f4eddb",
            eye="#0d0b09",
            garment="#3a3730",
            accent="#7a2f2f",
        ),
        _theme(
            "cursed",
            "Cursed",
            skin="#b9c4b0",
            belly="#a8b39f",
            tooth="#e6efdd",
            eye="#6fe36f",
            garment="#26302a",
            accent="#5cff8a",
        ),
    ),
    "knight": (
        _theme(
            "natural",
            "Steel",
            skin="#b9c0c8",
            belly="#9aa3ac",
            tooth="#efe6cf",
            eye="#1b1410",
            garment="#2f3a56",
            accent="#c8a33a",
        ),
        _theme(
            "blackened",
            "Blackened",
            skin="#4a4e55",
            belly="#3c4046",
            tooth="#efe6cf",
            eye="#c81f1f",
            garment="#1d1f24",
            accent="#8a1f1f",
        ),
    ),
    "wizard": (
        _skin_theme("natural", "Natural", "#d6a37c", "#3b2f6b", "#c8a33a"),
        _skin_theme("verdant", "Verdant", "#cfa27e", "#274a2e", "#8fd45a"),
    ),
    "zombie": (
        _skin_theme("natural", "Natural", "#8fa27a", "#4a4438", "#6b2b2b"),
        _skin_theme("drowned", "Drowned", "#7b93a0", "#33403f", "#3f6b5a"),
    ),
    "lizardfolk": (
        _skin_theme("natural", "Natural", "#4f8f6a", "#3f3626", "#c8922a"),
        _skin_theme("sand", "Sand", "#b09a5e", "#4a4028", "#d4622a"),
    ),
}


def _beast_theme(key: str, label: str, hide: str, horn: str, accent: str) -> Theme:
    """A palette over :data:`_QUADRUPED_REGIONS`, keyword by region.

    The underbelly is the hide lightened rather than a colour of its own,
    because that is what it is on every animal anybody asked for and a second
    knob would have been a second thing to get subtly wrong per species.
    """
    return _theme(
        key,
        label,
        hide=hide,
        underbelly=_lighten(hide, 0.22),
        horn=horn,
        eye="#14100c",
        mane=_darken(hide, 0.25),
        accent=accent,
    )


def _wing_theme(
    key: str, label: str, hide: str, wing: str, beak: str, accent: str
) -> Theme:
    """A palette over :data:`_WINGED_REGIONS`. The wing is its own colour: a
    membrane is not the shade of the hide it hangs off, and a flight feather is
    not the shade of the down beside it."""
    return _theme(
        key,
        label,
        hide=hide,
        underbelly=_lighten(hide, 0.25),
        wing=wing,
        beak=beak,
        eye="#120e0a",
        horn=_darken(beak, 0.2),
        accent=accent,
    )


def _blob_theme(key: str, label: str, body: str, core: str, accent: str) -> Theme:
    """A palette over :data:`_AMORPHOUS_REGIONS`.

    A translucent body is *painted*, not made translucent: the sprite
    pipeline's flat mode drives emission from the base colour and drops alpha,
    so a slime reads as glass by being pale and letting its nucleus glow, and
    an alpha value would have been a number that changed nothing on screen.
    """
    return _theme(key, label, body=body, core=core, eye="#0f0d0b", accent=accent)


def _darken(hexcolor: str, amount: float) -> str:
    r = int(hexcolor[1:3], 16)
    g = int(hexcolor[3:5], 16)
    b = int(hexcolor[5:7], 16)
    mix = lambda v: max(0, int(round(v * (1.0 - amount))))  # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


_BEAST_THEMES = {
    "wolf": (
        _beast_theme("natural", "Grey", "#6d6a63", "#2b2823", "#3e3a34"),
        _beast_theme("black", "Black", "#3a3835", "#1a1817", "#6a6255"),
    ),
    "dog": (
        _beast_theme("natural", "Tan", "#b98a4e", "#3a2c1c", "#7a5228"),
        _beast_theme("ashen", "Ash", "#8e8577", "#2f2b26", "#4a453d"),
    ),
    "big_cat": (
        _beast_theme("natural", "Tawny", "#c08f45", "#3a2c18", "#4a331a"),
        _beast_theme("panther", "Panther", "#2e2c2b", "#151313", "#5a5450"),
    ),
    "bear": (
        _beast_theme("natural", "Brown", "#7a5433", "#2c1e12", "#4a3220"),
        _beast_theme("frost", "Frost", "#d8dde4", "#3a4048", "#9fb6c8"),
    ),
    "horse": (
        _beast_theme("natural", "Bay", "#8a5a34", "#2f2721", "#2a1c12"),
        _beast_theme("dapple", "Dapple", "#9b9791", "#33302c", "#5a5650"),
    ),
    "deer": (
        _beast_theme("natural", "Fallow", "#b08050", "#7a6a4a", "#e8dcc4"),
        _beast_theme("winter", "Winter", "#8d8272", "#6a6252", "#d8d2c4"),
    ),
    "boar": (
        _beast_theme("natural", "Bristle", "#5a4a3e", "#d8cdb4", "#2f2620"),
        _beast_theme("ashen", "Ash", "#4a4642", "#cfc8b6", "#26221e"),
    ),
    "lizard": (
        _beast_theme("natural", "Green", "#5a8f52", "#c8b45a", "#2f5a34"),
        _beast_theme("sand", "Sand", "#b09a5e", "#d8c98a", "#7a6230"),
    ),
}

_WINGED_THEMES = {
    "dragon": (
        _wing_theme("natural", "Crimson", "#8a2320", "#5a1614", "#d8b45a", "#e8a83a"),
        _theme(
            "fire",
            "Fire",
            hide="#5a1a12",
            underbelly="#8a3a1e",
            wing="#3a120c",
            beak="#c8a23a",
            eye="#ffc23a",
            horn="#2a1810",
            accent="#ff5a1e",
        ),
        _wing_theme("verdant", "Verdant", "#2f5a34", "#1e3a24", "#c8b45a", "#8fd45a"),
    ),
    "wyvern": (
        _wing_theme("natural", "Slate", "#4a5058", "#2f343a", "#b8b2a0", "#7a8a9a"),
        _wing_theme("venom", "Venom", "#3a4a2e", "#26301e", "#c8c05a", "#9fe03a"),
    ),
    "bat": (
        _wing_theme("natural", "Umber", "#4a3a30", "#2e2420", "#3a2c24", "#8a6a4a"),
        _wing_theme("pale", "Pale", "#9a8e82", "#6a6058", "#5a5048", "#d8c8a8"),
    ),
    "bird": (
        _wing_theme("natural", "Sparrow", "#a8845a", "#8a6a44", "#d8b45a", "#e8dcc4"),
        _wing_theme("azure", "Azure", "#3a5a8a", "#2a4570", "#d8b45a", "#8fc8e8"),
    ),
    "raven": (
        _wing_theme("natural", "Raven", "#26262a", "#1a1a1e", "#141414", "#5a5a6a"),
        _wing_theme("albino", "Albino", "#dcd8d0", "#c8c4bc", "#d8b45a", "#e8b0b0"),
    ),
    "griffin": (
        _wing_theme("natural", "Eagle", "#b09a72", "#8a7450", "#d8b45a", "#c8a33a"),
        _wing_theme("storm", "Storm", "#6a7078", "#4a5058", "#c8c05a", "#9fd8e8"),
    ),
}

_AMORPHOUS_THEMES = {
    "slime": (
        _blob_theme("natural", "Green", "#5abf6a", "#d8ff9a", "#2f8a3a"),
        _blob_theme("azure", "Azure", "#5a9fd8", "#c8ecff", "#2f5a9a"),
    ),
    "ooze": (
        _blob_theme("natural", "Bile", "#7a7a3a", "#d8d86a", "#4a4a1e"),
        _blob_theme("tar", "Tar", "#2a2826", "#6a6258", "#141312"),
    ),
    "elemental": (
        _theme(
            "fire",
            "Fire",
            body="#8a2a10",
            core="#ffd45a",
            eye="#fff0c8",
            accent="#ff6a1e",
        ),
        _blob_theme("stone", "Stone", "#7a736a", "#c8b48a", "#4a453e"),
        _blob_theme("frost", "Frost", "#8fb6c8", "#e8f8ff", "#4a6a7a"),
    ),
    "cloud": (
        _blob_theme("natural", "Cumulus", "#d8dce4", "#ffffff", "#9fa8b8"),
        _blob_theme("storm", "Storm", "#5a606a", "#c8d8ff", "#2f3440"),
    ),
    "ghost": (
        _blob_theme("natural", "Pale", "#b8c8d8", "#e8f4ff", "#7a8a9a"),
        _blob_theme("cursed", "Cursed", "#8fb89a", "#c8ffd8", "#3a6a4a"),
    ),
}


# --- the species table -------------------------------------------------------
#
# ``silhouette`` is the bake group. Four of them for twelve species, and the
# split is topological, not stylistic: ``plain`` is round-eared and tuskless,
# ``pointed`` adds long ears, ``tusked`` adds tusks and a heavy brow, ``saurian``
# adds a snout and drops the ears. Everything else -- and there is a lot of it --
# is a channel default and a palette.

_FAMILIES: dict[str, Family] = {
    "human": Family(
        key="human", version=1, label="Human", archetype="humanoid",
        silhouette="plain", aliases=("human", "humans", "villager", "peasant", "adventurer"),
        height_m=1.80, channel_defaults={},
        themes=_COMMON_THEMES["human"], nearest=("knight", "wizard", "elf"),
    ),
    "elf": Family(
        key="elf", version=1, label="Elf", archetype="humanoid",
        silhouette="pointed", aliases=("elf", "elves", "elven", "high elf"),
        height_m=1.85,
        channel_defaults={
            "bulk": -0.35, "stature": 0.20, "head_size": -0.10,
            "limb_length": 0.15, "shoulder_width": -0.15,
        },
        themes=_COMMON_THEMES["elf"], nearest=("human", "wizard"),
    ),
    "dwarf": Family(
        key="dwarf", version=1, label="Dwarf", archetype="humanoid",
        silhouette="plain", aliases=("dwarf", "dwarves", "dwarven"),
        height_m=1.35,
        channel_defaults={
            "bulk": 0.50, "stature": -0.60, "head_size": 0.25,
            "limb_length": -0.45, "shoulder_width": 0.35, "hunch": 0.10,
        },
        themes=_COMMON_THEMES["dwarf"], nearest=("human", "goblin"),
    ),
    "goblin": Family(
        key="goblin", version=1, label="Goblin", archetype="humanoid",
        silhouette="pointed", aliases=("goblin", "goblins", "gobbo", "hobgoblin"),
        height_m=1.20,
        channel_defaults={
            "bulk": -0.10, "stature": -0.50, "head_size": 0.40,
            "limb_length": -0.20, "shoulder_width": -0.20, "hunch": 0.50,
        },
        themes=_COMMON_THEMES["goblin"], nearest=("orc", "dwarf"),
    ),
    "orc": Family(
        key="orc", version=1, label="Orc", archetype="humanoid",
        silhouette="tusked", aliases=("orc", "orcs", "orcish", "orc warrior"),
        height_m=2.00,
        channel_defaults={
            "bulk": 0.50, "stature": 0.15, "head_size": 0.10,
            "shoulder_width": 0.50, "hunch": 0.35,
        },
        themes=_COMMON_THEMES["orc"], nearest=("ogre", "troll", "goblin"),
    ),
    "ogre": Family(
        key="ogre", version=1, label="Ogre", archetype="humanoid",
        silhouette="tusked", aliases=("ogre", "ogres", "ogre warrior", "ogre brute"),
        height_m=2.60,
        channel_defaults={
            "bulk": 0.90, "stature": 0.10, "head_size": 0.20,
            "limb_length": 0.30, "shoulder_width": 0.75, "hunch": 0.60,
        },
        themes=_COMMON_THEMES["ogre"], nearest=("troll", "orc"),
    ),
    "troll": Family(
        key="troll", version=1, label="Troll", archetype="humanoid",
        silhouette="tusked", aliases=("troll", "trolls", "cave troll"),
        height_m=3.00,
        channel_defaults={
            "bulk": 0.60, "stature": 0.40, "head_size": -0.05,
            "limb_length": 0.80, "shoulder_width": 0.50, "hunch": 0.75,
        },
        themes=_COMMON_THEMES["troll"], nearest=("ogre", "orc"),
    ),
    "skeleton": Family(
        key="skeleton", version=1, label="Skeleton", archetype="humanoid",
        silhouette="plain", aliases=("skeleton", "skeletons", "skeletal", "bones"),
        height_m=1.75,
        channel_defaults={"bulk": -0.90, "shoulder_width": -0.30},
        themes=_COMMON_THEMES["skeleton"], nearest=("zombie", "human"),
    ),
    "knight": Family(
        key="knight", version=1, label="Knight", archetype="humanoid",
        silhouette="plain", aliases=("knight", "knights", "paladin", "soldier", "guard"),
        height_m=1.85,
        channel_defaults={"bulk": 0.25, "shoulder_width": 0.30},
        themes=_COMMON_THEMES["knight"], nearest=("human", "wizard"),
    ),
    "wizard": Family(
        key="wizard", version=1, label="Wizard", archetype="humanoid",
        silhouette="plain", aliases=("wizard", "wizards", "mage", "sorcerer", "witch"),
        height_m=1.75,
        channel_defaults={"bulk": -0.15, "stature": -0.05, "hunch": 0.25},
        themes=_COMMON_THEMES["wizard"], nearest=("human", "elf"),
    ),
    "zombie": Family(
        key="zombie", version=1, label="Zombie", archetype="humanoid",
        silhouette="plain", aliases=("zombie", "zombies", "ghoul", "undead"),
        height_m=1.75,
        channel_defaults={"bulk": -0.40, "shoulder_width": -0.10, "hunch": 0.40},
        themes=_COMMON_THEMES["zombie"], nearest=("skeleton", "human"),
    ),
    "lizardfolk": Family(
        key="lizardfolk", version=1, label="Lizardfolk", archetype="humanoid",
        silhouette="saurian",
        aliases=("lizardfolk", "lizardman", "lizard folk", "saurian", "kobold"),
        height_m=1.90,
        channel_defaults={"bulk": 0.15, "limb_length": 0.10, "hunch": 0.30},
        themes=_COMMON_THEMES["lizardfolk"], nearest=("orc", "goblin"),
    ),

    # -- the quadruped archetype. Five silhouettes for eight species, and the
    # split is topological exactly as the humanoid's is: ``paw`` has soft feet
    # and nothing growing off the skull, ``hoof`` adds hooves and a mane,
    # ``antlered`` adds antlers, ``tusker`` adds tusks off the jaw, ``scaled``
    # drops the ears and grows a dorsal ridge. A wolf, a bear and a big cat are
    # one mesh and three rows.
    "wolf": Family(
        key="wolf", version=1, label="Wolf", archetype="quadruped",
        silhouette="paw", aliases=("wolf", "wolves", "dire wolf", "warg"),
        height_m=0.85,
        channel_defaults={
            "bulk": -0.10, "leg_length": 0.15, "body_length": 0.05, "tail_length": 0.25,
        },
        themes=_BEAST_THEMES["wolf"], nearest=("dog", "big_cat", "bear"),
    ),
    "dog": Family(
        key="dog", version=1, label="Dog", archetype="quadruped",
        silhouette="paw", aliases=("dog", "dogs", "hound", "mastiff", "puppy"),
        height_m=0.60,
        channel_defaults={
            "bulk": -0.10, "leg_length": -0.20, "head_size": 0.15,
            "body_length": -0.10, "tail_length": 0.05,
        },
        themes=_BEAST_THEMES["dog"], nearest=("wolf", "big_cat"),
    ),
    "big_cat": Family(
        key="big_cat", version=1, label="Big cat", archetype="quadruped",
        silhouette="paw",
        aliases=("big cat", "cat", "lion", "tiger", "panther", "leopard", "cougar"),
        height_m=0.95,
        channel_defaults={
            "bulk": -0.05, "neck_length": -0.10, "head_size": -0.05,
            "body_length": 0.15, "tail_length": 0.60,
        },
        themes=_BEAST_THEMES["big_cat"], nearest=("wolf", "bear"),
    ),
    "bear": Family(
        key="bear", version=1, label="Bear", archetype="quadruped",
        silhouette="paw", aliases=("bear", "bears", "grizzly", "ursine"),
        height_m=1.30,
        channel_defaults={
            "bulk": 0.85, "leg_length": -0.25, "neck_length": -0.35,
            "head_size": 0.15, "body_length": -0.05, "tail_length": -0.60,
        },
        themes=_BEAST_THEMES["bear"], nearest=("boar", "wolf"),
    ),
    "horse": Family(
        key="horse", version=1, label="Horse", archetype="quadruped",
        silhouette="hoof", aliases=("horse", "horses", "steed", "pony", "mare", "stallion"),
        height_m=1.70,
        channel_defaults={
            "bulk": 0.10, "leg_length": 0.55, "neck_length": 0.35,
            "head_size": -0.05, "tail_length": 0.20,
        },
        themes=_BEAST_THEMES["horse"], nearest=("deer", "boar"),
    ),
    "deer": Family(
        key="deer", version=1, label="Deer", archetype="quadruped",
        silhouette="antlered", aliases=("deer", "stag", "elk", "hart", "buck"),
        height_m=1.50,
        channel_defaults={
            "bulk": -0.40, "leg_length": 0.55, "neck_length": 0.30,
            "head_size": -0.20, "body_length": -0.10, "tail_length": -0.50,
        },
        themes=_BEAST_THEMES["deer"], nearest=("horse", "big_cat"),
    ),
    "boar": Family(
        key="boar", version=1, label="Boar", archetype="quadruped",
        silhouette="tusker", aliases=("boar", "boars", "hog", "pig", "warthog"),
        height_m=1.00,
        channel_defaults={
            "bulk": 0.55, "leg_length": -0.45, "neck_length": -0.40,
            "head_size": 0.20, "tail_length": -0.30,
        },
        themes=_BEAST_THEMES["boar"], nearest=("bear", "horse"),
    ),
    "lizard": Family(
        key="lizard", version=1, label="Lizard", archetype="quadruped",
        silhouette="scaled",
        aliases=("lizard", "lizards", "monitor", "giant lizard", "basilisk", "drake lizard"),
        height_m=0.45,
        channel_defaults={
            "bulk": -0.20, "leg_length": -0.50, "neck_length": -0.10,
            "body_length": 0.35, "tail_length": 0.80,
        },
        themes=_BEAST_THEMES["lizard"], nearest=("big_cat", "wolf"),
    ),

    # -- the winged archetype. ``dragon`` is a species and not a stand-in for
    # one: the brief's rule is that anything nameable is makeable, and this is
    # where the nameable flying things live. What it is *not* is four-legged --
    # the ``bird`` template has one pair of legs, so a dragon here is a winged
    # biped, which is the shape a wyvern already is and the shape a griffin is
    # approximated to. Recorded rather than hidden: a four-legged dragon needs a
    # winged-quadruped template, and that is a body plan, not a parameter.
    "dragon": Family(
        key="dragon", version=1, label="Dragon", archetype="winged",
        silhouette="drake", aliases=("dragon", "dragons", "draconic", "wyrm", "great wyrm"),
        height_m=4.00,
        channel_defaults={
            "bulk": 0.60, "wingspan": 0.55, "neck_length": 0.45,
            "tail_length": 0.55, "head_size": 0.15, "leg_length": 0.20,
        },
        themes=_WINGED_THEMES["dragon"], nearest=("wyvern", "griffin"),
    ),
    "wyvern": Family(
        key="wyvern", version=1, label="Wyvern", archetype="winged",
        silhouette="drake", aliases=("wyvern", "wyverns", "drake", "lesser dragon"),
        height_m=3.00,
        channel_defaults={
            "bulk": 0.10, "wingspan": 0.35, "neck_length": -0.10,
            "tail_length": 0.75, "head_size": -0.05, "leg_length": 0.35,
        },
        themes=_WINGED_THEMES["wyvern"], nearest=("dragon", "bat"),
    ),
    "bat": Family(
        key="bat", version=1, label="Bat", archetype="winged",
        silhouette="membrane", aliases=("bat", "bats", "giant bat", "vampire bat"),
        height_m=0.35,
        channel_defaults={
            "bulk": -0.35, "wingspan": 0.70, "neck_length": -0.55,
            "tail_length": -0.55, "head_size": 0.40, "leg_length": -0.40,
        },
        themes=_WINGED_THEMES["bat"], nearest=("raven", "wyvern"),
    ),
    "bird": Family(
        key="bird", version=1, label="Bird", archetype="winged",
        silhouette="feathered",
        aliases=("bird", "birds", "sparrow", "songbird", "eagle", "hawk", "falcon"),
        height_m=0.40,
        channel_defaults={
            "bulk": -0.20, "wingspan": 0.20, "neck_length": -0.25, "head_size": 0.15,
        },
        themes=_WINGED_THEMES["bird"], nearest=("raven", "griffin"),
    ),
    "raven": Family(
        key="raven", version=1, label="Raven", archetype="winged",
        silhouette="feathered", aliases=("raven", "ravens", "crow", "rook", "blackbird"),
        height_m=0.50,
        channel_defaults={
            "bulk": -0.10, "wingspan": 0.10, "neck_length": -0.10,
            "tail_length": 0.25, "head_size": 0.05,
        },
        themes=_WINGED_THEMES["raven"], nearest=("bird", "bat"),
    ),
    "griffin": Family(
        key="griffin", version=1, label="Griffin", archetype="winged",
        silhouette="feathered", aliases=("griffin", "gryphon", "griffon", "hippogriff"),
        height_m=1.70,
        channel_defaults={
            "bulk": 0.55, "wingspan": 0.60, "neck_length": 0.15,
            "tail_length": 0.30, "head_size": 0.10, "leg_length": 0.30,
        },
        themes=_WINGED_THEMES["griffin"], nearest=("dragon", "bird"),
    ),

    # -- the amorphous archetype. Three silhouettes for five species: ``smooth``
    # is a settled body, ``crystal`` breaks its own surface with a nucleus,
    # ``puff`` is a cluster of lumps. Viscosity, lobe prominence and surface
    # ripple are channels because they are displacement fields; a shard is not.
    "slime": Family(
        key="slime", version=1, label="Slime", archetype="amorphous",
        silhouette="smooth", aliases=("slime", "slimes", "gelatinous cube", "jelly", "blob"),
        height_m=0.80,
        channel_defaults={"viscosity": -0.35, "lobe": 0.30, "crown": -0.20},
        themes=_AMORPHOUS_THEMES["slime"], nearest=("ooze", "elemental"),
    ),
    "ooze": Family(
        key="ooze", version=1, label="Ooze", archetype="amorphous",
        silhouette="smooth", aliases=("ooze", "oozes", "pudding", "sludge", "gray ooze"),
        height_m=0.70,
        channel_defaults={
            "bulk": 0.25, "viscosity": -0.75, "lobe": 0.60, "ripple": 0.35, "crown": -0.55,
        },
        themes=_AMORPHOUS_THEMES["ooze"], nearest=("slime", "cloud"),
    ),
    "elemental": Family(
        key="elemental", version=1, label="Elemental", archetype="amorphous",
        silhouette="crystal",
        aliases=("elemental", "elementals", "fire elemental", "earth elemental", "golem core"),
        height_m=1.80,
        channel_defaults={
            "bulk": 0.20, "viscosity": 0.65, "lobe": -0.20, "ripple": 0.45, "crown": 0.55,
        },
        themes=_AMORPHOUS_THEMES["elemental"], nearest=("slime", "ghost"),
    ),
    "cloud": Family(
        key="cloud", version=1, label="Cloud", archetype="amorphous",
        silhouette="puff", aliases=("cloud", "clouds", "vapour", "vapor", "mist", "fog"),
        height_m=1.20,
        channel_defaults={"bulk": 0.30, "viscosity": -0.40, "lobe": 0.45, "ripple": 0.60},
        themes=_AMORPHOUS_THEMES["cloud"], nearest=("ghost", "ooze"),
    ),
    "ghost": Family(
        key="ghost", version=1, label="Ghost", archetype="amorphous",
        silhouette="puff", aliases=("ghost", "ghosts", "spectre", "specter", "wraith", "shade"),
        height_m=1.40,
        channel_defaults={
            "bulk": -0.30, "viscosity": 0.20, "lobe": -0.35, "ripple": 0.25, "crown": 0.60,
        },
        themes=_AMORPHOUS_THEMES["ghost"], nearest=("cloud", "elemental"),
    ),
}


def archetypes() -> dict[str, Archetype]:
    """``key -> Archetype``. A copy: the registry is not a scratchpad."""
    return dict(_ARCHETYPES)


def families() -> dict[str, Family]:
    """``key -> Family``, every shipped species of every archetype."""
    return dict(_FAMILIES)


def families_of(archetype: str) -> dict[str, Family]:
    """Every species of one body plan, for a picker grouped by archetype."""
    return {k: f for k, f in _FAMILIES.items() if f.archetype == archetype}


def silhouettes(archetype: str) -> dict[str, tuple[str, ...]]:
    """``silhouette -> the species baked from it``, for one archetype.

    The authoring script's unit of work and the import pin's unit of counting:
    there is exactly one ``<silhouette>.glb`` per key here.
    """
    out: dict[str, list[str]] = {}
    for key, fam in _FAMILIES.items():
        if fam.archetype == archetype:
            out.setdefault(fam.silhouette, []).append(key)
    return {k: tuple(v) for k, v in sorted(out.items())}


def get_archetype(key: str) -> Archetype:
    try:
        return _ARCHETYPES[str(key)]
    except KeyError:
        raise CharacterError(
            f"{key!r} is not a body plan; try " + ", ".join(sorted(_ARCHETYPES)),
            field="archetype",
        ) from None


def get_family(key: str) -> Family:
    """One species by key. Never by alias -- resolving words is the resolver's."""
    try:
        return _FAMILIES[str(key)]
    except KeyError:
        raise CharacterError(
            f"{key!r} is not a character we can make; try "
            + ", ".join(sorted(_FAMILIES)),
            field="family",
        ) from None


def _check_registry(_families: Mapping[str, Family] = _FAMILIES) -> None:
    """Import-time consistency, so a bad row costs the build and not a render."""
    for key, fam in _families.items():
        if fam.key != key:
            raise CharacterError(f"species {key!r} calls itself {fam.key!r}")
        arch = get_archetype(fam.archetype)
        known = {c.key for c in arch.channels}
        unknown = set(fam.channel_defaults) - known
        if unknown:
            raise CharacterError(f"{key}: {sorted(unknown)} are not {arch.key} channels")
        if not fam.themes:
            raise CharacterError(f"{key} has no themes")


_check_registry()

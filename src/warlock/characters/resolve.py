"""Words in, a :class:`Resolution` out: what the prompt asked for, and what of it we can make.

The Create surface takes a plain-language brief -- "fire ogre, 3/4 top down
sprite sheet" -- and has to show the user *controls*, not a guess: a species
picker already on Ogre, a theme already on Fire, a camera already on 3/4
top-down, and every word it could not account for listed where the user can see
it. That is all this module does. It never builds anything and it never refuses:
:mod:`.recipe` is the layer that refuses by name, and it does so against a
species the user has by then seen and can change.

Three decisions are load-bearing.

**The scan is longest-alias-first, then a fixed category order.** ``bird's eye``
is a camera and ``bird`` is (or will be) a species, and the only reason the
first does not silently become the second is that the two-token alias is tried
at that position before the one-token one. Where two aliases of the *same*
length collide the order is family > camera > theme > action > creature >
noise: "top down view" is in the noise table (it is the sort of thing people
type around a request) *and* in the camera table, and the tie order is what
makes it a camera rather than three words thrown away.

**Nothing is ever substituted.** If the prompt names a creature we do not make,
:attr:`Resolution.family` stays ``None`` and :attr:`Resolution.offer` carries
the species to *propose*. A resolver that quietly returned the nearest species
would produce a wyvern from the word "dragon" with nothing on screen saying so,
which is the same silent-wrong-output failure the channel ranges refuse to
clamp for. :func:`offer_sentence` is the one home for the wording, because the
Create surface, a tooltip and a toast all say it and three copies would drift.

**The vocabulary is derived from the registry, not written next to it.** Family
aliases come from :func:`family.families`, and a theme word is only in the table
if some species actually declares that theme key -- so a sibling adding a
species adds its aliases here for free, and a theme nobody paints can never be
resolved to.

Standard library only, plus :mod:`.family`. The camera keys are literals rather
than an import of ``warlock.pipelines.charsheet`` because this module is on the
door's import path and has no arithmetic to do with a preset; the *tests* import
charsheet and pin every key here against it, which is where a renamed preset is
caught.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .family import ARCHETYPE_KEYS, Family, families

__all__ = [
    "ACTION_WORDS",
    "CAMERA_WORDS",
    "KNOWN_CREATURES",
    "NOISE_WORDS",
    "STOPWORDS",
    "THEME_WORDS",
    "Creature",
    "Resolution",
    "Span",
    "offer_sentence",
    "resolve",
    "vocabulary",
]


# --- the vocabulary ----------------------------------------------------------

#: ``camera preset key -> the spellings a prompt may use``. Every key here must
#: be a ``charsheet.CAMERA_PRESETS`` key; ``test_every_camera_key_is_a_real_preset``
#: is the pin, because a preset renamed in the pipeline and not here would
#: resolve to a camera the sheet planner has never heard of.
CAMERA_WORDS: dict[str, tuple[str, ...]] = {
    "three_quarter_top_down": (
        "3/4 top down",
        "3/4 top-down",
        "3/4 topdown",
        "three quarter top down",
        "three-quarter top down",
        "three quarter",
        "three-quarter",
        "threequarter",
        "3/4 view",
        "3/4",
        "34 view",
    ),
    "isometric": ("isometric", "iso", "isometric view", "2:1 dimetric", "dimetric"),
    "side": ("side view", "side-on", "side on", "profile", "sideview", "side elevation"),
    "top_down": (
        "top down",
        "top-down",
        "topdown",
        # Also in NOISE_WORDS. Same length, so the family > camera > theme >
        # action > creature > noise tie order decides, and it decides camera:
        # a user who typed "top down view" asked for a camera.
        "top down view",
        "top-down view",
        "overhead",
        "birds eye",
        "birds eye view",
        "bird eye view",
        "aerial",
    ),
}

#: ``theme key -> spellings``. Filtered at build time against the theme keys the
#: registry's species actually declare, so this table may name a look no species
#: has yet without :func:`vocabulary` ever offering it.
THEME_WORDS: dict[str, tuple[str, ...]] = {
    "fire": (
        "fire", "flame", "flames", "flaming", "fiery", "burning", "lava",
        "magma", "ember", "embers", "inferno", "blaze", "blazing", "molten",
    ),
    "natural": ("natural", "plain", "default", "normal", "untinted"),
    "ashen": ("ashen", "ash", "ashy", "pale", "grey", "gray"),
    "moonlit": ("moonlit", "moon", "lunar", "silvered"),
    "forge": ("forge", "forged", "foundry", "smithy"),
    "swamp": ("swamp", "swampy", "bog", "marsh", "mire", "boggy"),
    "blood": ("blood", "bloody", "bloodied", "crimson", "gore", "gory"),
    "stone": ("stone", "stony", "rock", "rocky", "granite", "marble"),
    "frost": ("frost", "frosty", "frozen", "ice", "icy", "glacial", "snow", "snowy", "arctic"),
    "cursed": ("cursed", "curse", "hexed", "haunted", "spectral", "wraithlike"),
    # "dark" is deliberately absent: a "dark elf" is a species with a mood, not
    # a request for blackened steel, and the word is too common to spend.
    "blackened": ("blackened", "black", "obsidian", "sooty", "charred"),
    "verdant": ("verdant", "green", "mossy", "moss", "leafy", "overgrown"),
    "drowned": ("drowned", "sunken", "waterlogged", "sodden", "sea-soaked"),
    "sand": ("sand", "sandy", "desert", "dune", "dusty", "sunbleached"),
}

#: ``movement key -> spellings``. The keys are ``charsheet.ANIMATIONS`` names,
#: pinned by ``test_every_action_key_is_a_real_animation``.
ACTION_WORDS: dict[str, tuple[str, ...]] = {
    "idle": ("idle", "standing", "stand", "idling", "breathing", "rest"),
    "walk": ("walk", "walking", "walk cycle", "walkcycle", "stroll"),
    "run": ("run", "running", "sprint", "sprinting", "dash", "charge"),
    "attack": (
        "attack", "attacking", "strike", "striking", "swing", "swinging",
        "slash", "slashing", "melee", "punch",
    ),
    "jump": ("jump", "jumping", "leap", "leaping", "hop"),
}

#: Words about the *deliverable* rather than the character. Consumed with a span
#: so the UI can grey them out, and never reported as unrecognised: a user who
#: typed "sprite sheet" at a sprite sheet generator has not made a mistake.
NOISE_WORDS: tuple[str, ...] = (
    "sprite sheet", "sprite sheets", "spritesheet", "spritesheets", "sprite",
    "sprites", "sheet", "sheets", "character", "characters", "char",
    "pixel art", "pixelart", "pixel", "art", "artwork", "asset", "assets",
    "game asset", "game assets", "game", "game ready", "2d", "3d",
    "8 direction", "8 directions", "8-direction", "8-directional",
    "eight direction", "eight directions", "4 direction", "4 directions",
    "four direction", "four directions", "directional", "directions",
    "animation", "animations", "animated", "frame", "frames", "fps",
    "top down view", "turnaround", "full body", "fullbody", "reference",
    "render", "rendering", "png", "transparent", "no background",
    "high quality", "detailed", "clean", "retro", "16 bit", "8 bit", "hd",
)

#: Dropped without a span. Grammar, not vocabulary.
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "for", "with", "and", "or", "in", "on", "to",
    "into", "from", "as", "at", "by", "my", "me", "i", "we", "you",
    "want", "wants", "need", "needs", "give", "gives", "make", "makes",
    "making", "create", "creates", "creating", "generate", "generates",
    "draw", "drawing", "please", "just", "some", "that", "this", "it", "its",
    "is", "are", "be", "can", "could", "would", "should", "style", "styled",
    "look", "looking", "like", "new", "set", "up",
})


@dataclass(frozen=True, slots=True)
class Creature:
    """A creature noun the vocabulary knows, whether or not we make one.

    ``archetype`` is the **body plan hint**: which of :data:`family.ARCHETYPE_KEYS`
    this thing is shaped like. It is what makes the offer for "chimera"
    four-legged and the offer for "phoenix" winged instead of whichever species
    happened to sort first. ``None`` means genuinely unknown ("monster"), and
    only then does the offer range over every body plan.

    ``kin`` names species this creature is *specifically* nearest, ahead of the
    rest of its body plan. Entries that the registry does not ship are dropped,
    so naming a species a sibling has not added yet is free rather than fatal.
    """

    archetype: str | None
    kin: tuple[str, ...] = ()


def _c(archetype: str | None, *kin: str) -> Creature:
    return Creature(archetype, tuple(kin))


#: Creature nouns the resolver recognises *as creatures* even when no species
#: matches, so a refusal can name what was asked for. Deliberately much wider
#: than the registry: the whole point is to answer "we do not make a kraken"
#: with a sentence rather than with "kraken" in a list of unknown words.
KNOWN_CREATURES: dict[str, Creature] = {
    # -- humanoid -----------------------------------------------------------
    "giant": _c("humanoid", "ogre", "troll"),
    "cyclops": _c("humanoid", "ogre", "troll"),
    "titan": _c("humanoid", "ogre", "troll"),
    "minotaur": _c("humanoid", "ogre", "orc"),
    "gnome": _c("humanoid", "dwarf", "goblin"),
    "halfling": _c("humanoid", "dwarf", "goblin"),
    "hobbit": _c("humanoid", "dwarf", "goblin"),
    "dark elf": _c("humanoid", "elf"),
    "drow": _c("humanoid", "elf"),
    "vampire": _c("humanoid", "human", "zombie"),
    "werewolf": _c("humanoid", "orc", "troll"),
    "mummy": _c("humanoid", "zombie", "skeleton"),
    "lich": _c("humanoid", "skeleton", "wizard"),
    "wraith": _c("humanoid", "skeleton", "zombie"),
    "necromancer": _c("humanoid", "wizard"),
    "druid": _c("humanoid", "wizard"),
    "cleric": _c("humanoid", "wizard", "knight"),
    "monk": _c("humanoid", "human"),
    "bandit": _c("humanoid", "human"),
    "pirate": _c("humanoid", "human"),
    "ninja": _c("humanoid", "human"),
    "samurai": _c("humanoid", "knight"),
    "archer": _c("humanoid", "human", "elf"),
    "ranger": _c("humanoid", "human", "elf"),
    "rogue": _c("humanoid", "human"),
    "thief": _c("humanoid", "human"),
    "barbarian": _c("humanoid", "human", "orc"),
    "berserker": _c("humanoid", "orc", "human"),
    "demon": _c("humanoid", "orc", "troll"),
    "devil": _c("humanoid", "orc", "troll"),
    "satyr": _c("humanoid", "goblin"),
    "yeti": _c("humanoid", "troll", "ogre"),
    "sasquatch": _c("humanoid", "troll", "ogre"),
    "golem": _c("humanoid", "ogre", "knight"),
    "automaton": _c("humanoid", "knight"),
    "robot": _c("humanoid", "knight"),
    "android": _c("humanoid", "human"),
    "treant": _c("humanoid", "troll"),
    "ent": _c("humanoid", "troll"),
    "gnoll": _c("humanoid", "orc", "goblin"),
    "kenku": _c("humanoid", "goblin"),
    "merfolk": _c("humanoid", "human"),
    "mermaid": _c("humanoid", "human"),
    "naga": _c("humanoid", "lizardfolk"),
    "troglodyte": _c("humanoid", "lizardfolk", "goblin"),
    "sahuagin": _c("humanoid", "lizardfolk"),
    "imp": _c("winged", "goblin"),
    "gargoyle": _c("winged", "ogre"),
    "angel": _c("winged", "human"),
    "harpy": _c("winged", "goblin"),
    # -- quadruped ----------------------------------------------------------
    "wolf": _c("quadruped"),
    "dire wolf": _c("quadruped"),
    "direwolf": _c("quadruped"),
    "dog": _c("quadruped"),
    "hound": _c("quadruped"),
    "hellhound": _c("quadruped"),
    "cerberus": _c("quadruped"),
    "cat": _c("quadruped"),
    "lion": _c("quadruped"),
    "tiger": _c("quadruped"),
    "panther": _c("quadruped"),
    "leopard": _c("quadruped"),
    "sabertooth": _c("quadruped"),
    "horse": _c("quadruped"),
    "pony": _c("quadruped"),
    "unicorn": _c("quadruped"),
    "donkey": _c("quadruped"),
    "mule": _c("quadruped"),
    "camel": _c("quadruped"),
    "llama": _c("quadruped"),
    "bear": _c("quadruped"),
    "boar": _c("quadruped"),
    "pig": _c("quadruped"),
    "cow": _c("quadruped"),
    "bull": _c("quadruped"),
    "ox": _c("quadruped"),
    "goat": _c("quadruped"),
    "sheep": _c("quadruped"),
    "deer": _c("quadruped"),
    "stag": _c("quadruped"),
    "elk": _c("quadruped"),
    "moose": _c("quadruped"),
    "elephant": _c("quadruped"),
    "mammoth": _c("quadruped"),
    "rhino": _c("quadruped"),
    "hippo": _c("quadruped"),
    "fox": _c("quadruped"),
    "hyena": _c("quadruped"),
    "jackal": _c("quadruped"),
    "badger": _c("quadruped"),
    "weasel": _c("quadruped"),
    "otter": _c("quadruped"),
    "beaver": _c("quadruped"),
    "raccoon": _c("quadruped"),
    "squirrel": _c("quadruped"),
    "rabbit": _c("quadruped"),
    "hare": _c("quadruped"),
    "rat": _c("quadruped"),
    "mouse": _c("quadruped"),
    "ferret": _c("quadruped"),
    "monkey": _c("quadruped"),
    "ape": _c("humanoid", "orc"),
    "gorilla": _c("humanoid", "ogre"),
    "crocodile": _c("quadruped"),
    "alligator": _c("quadruped"),
    "lizard": _c("quadruped"),
    "turtle": _c("quadruped"),
    "tortoise": _c("quadruped"),
    "frog": _c("quadruped"),
    "toad": _c("quadruped"),
    "spider": _c("quadruped"),
    "scorpion": _c("quadruped"),
    "crab": _c("quadruped"),
    "beetle": _c("quadruped"),
    "ant": _c("quadruped"),
    "mantis": _c("quadruped"),
    "chimera": _c("quadruped"),
    "manticore": _c("quadruped"),
    "basilisk": _c("quadruped"),
    "cockatrice": _c("winged"),
    # A hydra has a lizard's body and legs in most of the art people mean by
    # the word, so it is offered a four-legged thing, not a serpent.
    "hydra": _c("quadruped"),
    "centaur": _c("quadruped"),
    "behemoth": _c("quadruped"),
    "kirin": _c("quadruped"),
    # -- winged -------------------------------------------------------------
    "dragon": _c("winged", "wyvern", "drake"),
    "wyrm": _c("winged", "wyvern", "drake"),
    "drake": _c("winged", "wyvern"),
    "wyvern": _c("winged"),
    "griffin": _c("winged"),
    "gryphon": _c("winged"),
    "hippogriff": _c("winged"),
    "pegasus": _c("winged"),
    "phoenix": _c("winged"),
    "roc": _c("winged"),
    "bird": _c("winged"),
    "crow": _c("winged"),
    "raven": _c("winged"),
    "eagle": _c("winged"),
    "hawk": _c("winged"),
    "falcon": _c("winged"),
    "owl": _c("winged"),
    "vulture": _c("winged"),
    "bat": _c("winged"),
    "moth": _c("winged"),
    "butterfly": _c("winged"),
    "dragonfly": _c("winged"),
    "wasp": _c("winged"),
    "bee": _c("winged"),
    "fairy": _c("winged"),
    "pixie": _c("winged"),
    "sphinx": _c("winged"),
    # -- amorphous ----------------------------------------------------------
    "slime": _c("amorphous"),
    "ooze": _c("amorphous"),
    "blob": _c("amorphous"),
    "jelly": _c("amorphous"),
    "gelatinous cube": _c("amorphous"),
    "pudding": _c("amorphous"),
    "amoeba": _c("amorphous"),
    "ghost": _c("amorphous"),
    "spirit": _c("amorphous"),
    "phantom": _c("amorphous"),
    "banshee": _c("amorphous"),
    "wisp": _c("amorphous"),
    "elemental": _c("amorphous"),
    "mimic": _c("amorphous"),
    "kraken": _c("amorphous"),
    "octopus": _c("amorphous"),
    "squid": _c("amorphous"),
    "jellyfish": _c("amorphous"),
    "snake": _c("amorphous"),
    "serpent": _c("amorphous"),
    "worm": _c("amorphous"),
    "slug": _c("amorphous"),
    "snail": _c("amorphous"),
    "eel": _c("amorphous"),
    "fish": _c("amorphous"),
    "shark": _c("amorphous"),
    "whale": _c("amorphous"),
    # -- shape unknown: only these fall back across every body plan ----------
    "monster": _c(None),
    "creature": _c(None),
    "beast": _c(None),
    "critter": _c(None),
    "mob": _c(None),
    "enemy": _c(None),
    "boss": _c(None),
    "familiar": _c(None),
    "npc": _c(None),
}

#: When a creature's body plan ships no species, which plan to look at next.
#: A winged thing would rather be a four-legged thing than a puddle, and a
#: puddle is the last resort for everything -- which is what keeps
#: ``test_a_winged_creature_is_not_offered_a_slime`` true once the amorphous
#: species land.
_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "humanoid": ("humanoid", "winged", "quadruped", "amorphous"),
    "quadruped": ("quadruped", "winged", "humanoid", "amorphous"),
    "winged": ("winged", "quadruped", "humanoid", "amorphous"),
    "amorphous": ("amorphous", "quadruped", "humanoid", "winged"),
}

#: How many species an offer names. Three is what the Create surface has room
#: for under the prompt bar; :func:`offer_sentence` only ever says the first.
MAX_OFFER = 3

#: The scan's tie order. Longest alias first *then* this, so the two-token
#: "birds eye" beats the one-token "bird" on length, and a same-length
#: collision between a camera and a noise phrase goes to the camera.
_PRIORITY: tuple[str, ...] = ("family", "camera", "theme", "action", "creature", "noise")

#: ``charsheet.ANIMATIONS`` order, so two prompts that name the same movements
#: in different orders produce the same sheet.
#: ``test_the_action_order_is_the_frame_tables_order`` pins it.
_ACTION_ORDER: tuple[str, ...] = ("idle", "walk", "run", "attack", "jump")

_IRREGULAR: dict[str, str] = {
    "wolves": "wolf", "elves": "elf", "dwarves": "dwarf", "thieves": "thief",
    "mice": "mouse", "geese": "goose", "men": "man", "women": "woman",
    "children": "child", "oxen": "ox", "feet": "foot", "teeth": "tooth",
    "knives": "knife", "leaves": "leaf", "halves": "half", "wolfs": "wolf",
}


# --- the record --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Span:
    """What covered which tokens, so the UI can underline the prompt.

    ``applied`` is false for a span that was understood but not used -- the
    second species in "orc and goblin", say. The distinction matters: the token
    is not unknown, it is outvoted, and an underline that said "unknown" there
    would be a lie.
    """

    start: int
    end: int
    kind: str
    key: str
    text: str
    applied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Span:
        return cls(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            kind=str(data.get("kind", "")),
            key=str(data.get("key", "")),
            text=str(data.get("text", "")),
            applied=bool(data.get("applied", True)),
        )


@dataclass(frozen=True, slots=True)
class Resolution:
    """Everything a prompt yielded, including what it did not.

    ``family`` is ``None`` whenever no species matched -- including, especially,
    when a creature *was* recognised and we do not make it. The nearest species
    live in ``offer`` and are shown, never applied.
    """

    family: str | None = None
    archetype: str | None = None
    theme: str | None = None
    camera_preset: str | None = None
    actions: tuple[str, ...] = ()
    unrecognised: tuple[str, ...] = ()
    creature_words: tuple[str, ...] = ()
    offer: tuple[str, ...] = ()
    spans: tuple[Span, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "archetype": self.archetype,
            "theme": self.theme,
            "camera_preset": self.camera_preset,
            "actions": list(self.actions),
            "unrecognised": list(self.unrecognised),
            "creature_words": list(self.creature_words),
            "offer": list(self.offer),
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Resolution:
        def _strs(name: str) -> tuple[str, ...]:
            return tuple(str(v) for v in data.get(name, ()) or ())

        return cls(
            family=data.get("family") or None,
            archetype=data.get("archetype") or None,
            theme=data.get("theme") or None,
            camera_preset=data.get("camera_preset") or None,
            actions=_strs("actions"),
            unrecognised=_strs("unrecognised"),
            creature_words=_strs("creature_words"),
            offer=_strs("offer"),
            spans=tuple(Span.from_dict(s) for s in data.get("spans", ()) or ()),
        )


# --- tokenising --------------------------------------------------------------

#: Kept inside a token. ``/`` so "3/4" survives as one token and ``-`` so
#: "top-down" does; everything else is a separator.
_KEEP = set("abcdefghijklmnopqrstuvwxyz0123456789/-")
_SEPARATORS = ",;:!?.()[]{}\"“”‘’<>|*_+=@#$%^&~`\\\n\t\r"


def _normal(raw: str) -> str:
    """One token's matching form: lowercase, apostrophes gone, edges trimmed.

    Apostrophes are removed rather than kept so "bird's eye" and "birds eye"
    are the same two tokens; the *original* spelling is carried separately and
    is what an unrecognised word is reported as.
    """
    text = raw.lower().replace("'", "").replace("’", "")
    text = "".join(ch if ch in _KEEP else " " for ch in text)
    return " ".join(text.split()).replace(" ", "")


def _tokenise(text: str) -> tuple[list[str], list[str]]:
    """``(original spellings, matching forms)``, one entry each, same length."""
    scrubbed = "".join(" " if ch in _SEPARATORS else ch for ch in text)
    raws: list[str] = []
    normals: list[str] = []
    for raw in scrubbed.split():
        normal = _normal(raw)
        if not normal:
            continue
        raws.append(raw.strip(_SEPARATORS) or raw)
        normals.append(normal)
    return raws, normals


def _variants(token: str) -> list[str]:
    """A token's plausible dictionary forms, most likely first.

    Plurals and ``-ing`` only, because those are the two inflections a prompt
    actually uses ("goblins", "walking", "flaming") and every extra rule is a
    chance to turn a word we do not know into a word we do.
    """
    out = [token]
    if token in _IRREGULAR:
        out.append(_IRREGULAR[token])
    if len(token) > 3:
        if token.endswith("ies"):
            out.append(token[:-3] + "y")
        if token.endswith("es"):
            out.append(token[:-2])
        if token.endswith("s") and not token.endswith("ss"):
            out.append(token[:-1])
        if token.endswith("ing"):
            stem = token[:-3]
            out.extend((stem, stem + "e"))
            if len(stem) > 2 and stem[-1] == stem[-2]:
                out.append(stem[:-1])
        if token.endswith("ed"):
            out.extend((token[:-2], token[:-1]))
    seen: set[str] = set()
    return [v for v in out if v and not (v in seen or seen.add(v))]


# --- the tables --------------------------------------------------------------


def _families_map(registry: Mapping[str, Family] | None) -> Mapping[str, Family]:
    return families() if registry is None else registry


def _declared_themes(registry: Mapping[str, Family]) -> set[str]:
    return {theme.key for fam in registry.values() for theme in fam.themes}


def vocabulary(registry: Mapping[str, Family] | None = None) -> dict[str, dict[str, Any]]:
    """Every alias the scan can match, by category. Read by tests and the manual.

    ``families`` is built from the registry rather than written down, and
    ``themes`` is filtered to the keys some species declares -- the two halves
    of "a sibling adding a species must not have to edit this module".
    """
    registry = _families_map(registry)
    fams: dict[str, str] = {}
    for key, fam in registry.items():
        for alias in fam.aliases:
            fams.setdefault(_alias_key(alias), key)
    declared = _declared_themes(registry)
    themes = {
        _alias_key(word): key
        for key, words in THEME_WORDS.items()
        if key in declared
        for word in words
    }
    cameras = {_alias_key(w): k for k, words in CAMERA_WORDS.items() for w in words}
    actions = {_alias_key(w): k for k, words in ACTION_WORDS.items() for w in words}
    noise = {_alias_key(w): "noise" for w in NOISE_WORDS}
    creatures = {_alias_key(w): c for w, c in KNOWN_CREATURES.items()}
    return {
        "families": fams,
        "cameras": cameras,
        "themes": themes,
        "actions": actions,
        "creatures": creatures,
        "noise": noise,
        "stopwords": dict.fromkeys(sorted(STOPWORDS), "stopword"),
    }


def _alias_key(alias: str) -> str:
    """An alias in matching form: space-separated normalised tokens."""
    return " ".join(_normal(part) for part in alias.split() if _normal(part))


def _table(
    registry: Mapping[str, Family],
) -> tuple[dict[tuple[str, ...], list[tuple[str, str]]], int]:
    """``token tuple -> [(category, key)]`` plus the longest alias's length."""
    vocab = vocabulary(registry)
    table: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for category, entries in (
        ("family", vocab["families"]),
        ("camera", vocab["cameras"]),
        ("theme", vocab["themes"]),
        ("action", vocab["actions"]),
        ("creature", {k: k for k in vocab["creatures"]}),
        ("noise", vocab["noise"]),
    ):
        for alias, key in entries.items():
            table.setdefault(tuple(alias.split(" ")), []).append((category, key))
    longest = max((len(k) for k in table), default=1)
    return table, longest


def _lookup(
    table: Mapping[tuple[str, ...], list[tuple[str, str]]], span: list[str]
) -> list[tuple[str, str]]:
    """Hits for a token span, trying the literal spelling before any inflection.

    Only the *last* token is lemmatised: "walk cycles" and "fire ogres" are the
    shapes a prompt inflects, and lemmatising every position would multiply the
    ways an unknown phrase could collide with a known one.
    """
    key = tuple(span)
    if key in table:
        return table[key]
    for variant in _variants(span[-1])[1:]:
        candidate = (*span[:-1], variant)
        if candidate in table:
            return table[candidate]
    return []


# --- the scan ----------------------------------------------------------------


def resolve(text: str, *, families: Mapping[str, Family] | None = None) -> Resolution:
    """Turn a prompt into what we understood of it.

    Never raises and never defaults: a prompt with no species in it resolves to
    ``family=None``, because a Create surface that quietly filled in "human"
    would be answering a question the user has not been asked yet.
    """
    registry = _families_map(families)
    table, longest = _table(registry)
    raws, normals = _tokenise(text or "")

    family_key: str | None = None
    theme: str | None = None
    camera: str | None = None
    actions: list[str] = []
    creature_words: list[str] = []
    unrecognised: list[str] = []
    spans: list[Span] = []
    hints: list[str] = []  # creature words, in matching form, in order

    index = 0
    total = len(normals)
    while index < total:
        hit: tuple[str, str] | None = None
        length = 0
        for width in range(min(longest, total - index), 0, -1):
            found = _lookup(table, normals[index : index + width])
            if found:
                hit = min(found, key=lambda h: _PRIORITY.index(h[0]))
                length = width
                break
        if hit is None:
            if normals[index] not in STOPWORDS:
                unrecognised.append(raws[index])
            index += 1
            continue

        category, key = hit
        text_span = " ".join(raws[index : index + length])
        applied = True
        if category == "family":
            creature_words.append(text_span)
            if family_key is None:
                family_key = key
            else:
                # Two species named. The first wins, and the second is reported
                # as unaccounted for rather than silently dropped -- a prompt
                # saying "orc and goblin" got one of the two, and the user has
                # to be able to see which word did nothing.
                applied = False
                unrecognised.extend(raws[index : index + length])
        elif category == "creature":
            creature_words.append(text_span)
            hints.append(key)
        elif category == "camera":
            if camera is None:
                camera = key
            else:
                applied = False
        elif category == "theme":
            if theme is None:
                theme = key
            else:
                applied = False
        elif category == "action":
            if key in actions:
                applied = False
            else:
                actions.append(key)
        spans.append(Span(index, index + length, category, key, text_span, applied))
        index += length

    archetype = registry[family_key].archetype if family_key else None
    offer: tuple[str, ...] = ()
    if family_key is None and hints:
        creature = KNOWN_CREATURES[hints[0]]
        archetype = creature.archetype
        offer = _offer_for(creature, registry)

    return Resolution(
        family=family_key,
        archetype=archetype,
        theme=theme,
        camera_preset=camera,
        actions=tuple(sorted(actions, key=_action_rank)),
        unrecognised=tuple(unrecognised),
        creature_words=tuple(creature_words),
        offer=offer,
        spans=tuple(spans),
    )


def _action_rank(key: str) -> tuple[int, str]:
    return (_ACTION_ORDER.index(key) if key in _ACTION_ORDER else len(_ACTION_ORDER), key)


# --- the offer ---------------------------------------------------------------


def _offer_for(creature: Creature, registry: Mapping[str, Family]) -> tuple[str, ...]:
    """The species to propose for a creature we do not make, nearest first.

    Body plan first, always: the offer comes entirely from one archetype -- the
    creature's own if the registry ships any of it, otherwise the nearest plan
    that has species. Within the plan, the creature's own ``kin`` lead, then the
    *registry's* ``nearest`` for whichever species that put first (the species
    row is where kinship is really written down), then the rest of the plan.
    """
    plan = _plan_for(creature.archetype, registry)
    pool = (
        {k: f for k, f in registry.items() if f.archetype == plan}
        if plan
        else dict(registry)
    )
    ordered: list[str] = []

    def add(keys: Iterable[str]) -> None:
        for key in keys:
            if key in pool and key not in ordered:
                ordered.append(key)

    add(creature.kin)
    if ordered:
        add(registry[ordered[0]].nearest)
    add(pool)
    return tuple(ordered[:MAX_OFFER])


def _plan_for(archetype: str | None, registry: Mapping[str, Family]) -> str | None:
    """The body plan an offer may draw from: the hint, or its nearest neighbour
    that ships species. ``None`` only for a creature whose shape we never knew."""
    if archetype is None:
        return None
    for candidate in _NEIGHBOURS.get(archetype, ARCHETYPE_KEYS):
        if any(f.archetype == candidate for f in registry.values()):
            return candidate
    return None


def offer_sentence(
    resolution: Resolution, *, families: Mapping[str, Family] | None = None
) -> str | None:
    """The exact wording for "we do not make that; here is what we do make".

    One home for it because the Create surface, the species picker's tooltip and
    the toast after a failed brief all say it, and three copies of a sentence
    that names a substitution are three chances to word one of them as though
    the substitution had already happened.

    ``None`` when there is nothing to offer -- including when the creature *is*
    supported, which is the common case.
    """
    if resolution.family is not None or not resolution.offer:
        return None
    registry = _families_map(families)
    asked = _asked_for(resolution)
    if not asked:
        return None
    key = resolution.offer[0]
    if key not in registry:
        return None
    label = registry[key].label.lower()
    return f"Warlock has no {asked} yet. The closest it makes is {_article(label)} {label}."


def _asked_for(resolution: Resolution) -> str | None:
    """The dictionary form of the first creature word, so the sentence reads
    "no chimera yet" for a prompt that said "chimeras"."""
    for word in resolution.creature_words:
        tokens = [_normal(part) for part in word.split() if _normal(part)]
        if not tokens:
            continue
        for variant in _variants(tokens[-1]):
            candidate = " ".join([*tokens[:-1], variant])
            if candidate in KNOWN_CREATURES:
                return candidate
    return None


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"

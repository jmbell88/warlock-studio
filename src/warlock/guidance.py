"""Design guidance: the structured creative direction attached to a job.

trellis-server.exe takes only an image, a seed and a geometry resolution, so
guidance can act on exactly three things:

* the SDXL prompt -- text jobs only, since image jobs never touch SDXL;
* the per-request geometry resolution, which the platform preset supplies;
* the physical scale of the finished GLB (pipelines/postprocess.scale_glb).

Deliberately *not* texture resolution: --tex-res is a server launch flag, not a
per-request one, and config.py pins it to 512 because the vendored exe bakes
per-texel noise into the atlas above that.

This module owns the taxonomy for both the API layer and the worker. The prompt
fragments here describe the *subject* only; the single-object/plain-background
scaffolding that keeps images TRELLIS-friendly stays in
pipelines/prompt.PROMPT_TEMPLATE, which wraps whatever we produce.

Every Option.prompt fragment here is kept to 2-4 words. Chunked encoding in
pipelines/prompt.py and pipelines/text2image.py removes CLIP's hard 77-token
ceiling, but not the soft one -- a longer conditioning sequence still dilutes
cross-attention, so brevity stays a rule even though truncation no longer is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import models


class GuidanceError(ValueError):
    """A refused guidance value, naming the field it came from.

    A ``ValueError`` subclass, deliberately: every caller in the repo already
    catches ``ValueError`` and this module has raised one since it was written,
    so the addition is purely ``field`` and nothing that reads it today has to
    change. What ``field`` buys is an *address* -- ``service.errors.Invalid``
    has carried a ``field`` since it was written and the UI highlights the
    control it names, but the guidance passthroughs could never supply one, so
    the twelve most common refusals in the app arrived as a toast pointing at
    nothing in particular (S137).

    Recovering the field by parsing the message was the alternative and is the
    translation-table hazard ``judge.STAGES``/``verdicts.STAGES`` is written
    down about: two spellings of one fact, drifting silently the first time a
    message is reworded.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


SIZE_MIN_M = 0.01
SIZE_MAX_M = 100.0

# Fallback when no category is chosen and so no default size applies.
DEFAULT_SIZE_M = 1.0
DEFAULT_PLATFORM = "3d"

# How trellis-server mattes the input image. Not an Option table: these are
# server capabilities, not prompt fragments, so they never reach compose_prompt.
BG_REMOVAL = ("auto", "birefnet", "threshold")

# birefnet, and it is the one thing the 2026-08-07 review of the rogue sweep
# found any signal at all for. 3 accepts in 83; all three were
# bg_removal=birefnet and `auto` went 0 for 80, over matched pairs whose
# input.png hashed identically -- bg_removal is passed to trellis-server at
# reconstruction time, so nothing about the picture differed. The failure mode
# moved as well as the rate: 58 of 80 `auto` rejects were tagged `broken`, 0 of
# 4 birefnet were. The mechanism is in doctor.py's own words -- without the
# weights matting "falls back to a threshold cutout", and `auto` lets the
# server decide -- and a threshold cutout on a dark brief leaves background
# attached, which TRELLIS reconstructs into a solid slab.
#
# Gated, though, and default_bg_removal below is the gate: on a host that never
# downloaded birefnet.gguf there is nothing to load, and `auto` is still right
# there. n=4 is thin and the blind confirmation the roadmap asks for has not
# been run; what tips it is that the alternative is a default already measured
# at 0 for 80.
DEFAULT_BG_REMOVAL = "birefnet"
FALLBACK_BG_REMOVAL = "auto"

# The weights DEFAULT_BG_REMOVAL needs, named once. doctor._birefnet_check
# reports this same file, and a second spelling of it would let the app pick a
# matte the doctor is telling the user is unavailable.
BIREFNET_WEIGHTS = "birefnet.gguf"


def default_bg_removal(trellis_models_dir: Path) -> str:
    """The matte to use when the caller named none: the learned one when its
    weights are on disk, ``auto`` when they are not."""
    return (
        DEFAULT_BG_REMOVAL
        if (trellis_models_dir / BIREFNET_WEIGHTS).exists()
        else FALLBACK_BG_REMOVAL
    )

# What a TRELLIS reference image must not be. A second subject or a cropped one
# is the single most common cause of a mesh that reconstructs into nonsense, so
# this is a default rather than an empty field the user has to discover.
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, multiple objects, cropped, cut off, "
    "text, watermark, signature, busy background, human hands"
)
MAX_NEGATIVE_PROMPT = 1000


@dataclass(frozen=True, slots=True)
class Option:
    key: str
    label: str
    prompt: str
    # Categories only: the real-world size a member of this category usually is.
    default_size_m: float | None = None
    # Platforms only: the geometry resolution passed to trellis-server.
    resolution: int | None = None


def _table(*options: Option) -> dict[str, Option]:
    return {opt.key: opt for opt in options}


GENRES = _table(
    Option("fantasy", "Fantasy", "high fantasy, medieval, ornate craftsmanship"),
    Option("scifi", "Sci-fi", "science fiction, sleek futuristic technology, panel seams"),
    Option("modern", "Modern", "contemporary real-world design, everyday manufactured object"),
    Option(
        "postapoc", "Post-apocalyptic",
        "post-apocalyptic, scavenged and improvised, worn and rusted",
    ),
    Option("horror", "Horror", "dark horror aesthetic, grim and unsettling, decayed"),
    Option("cartoon", "Cartoon", "playful cartoon design, exaggerated friendly proportions"),
)

# Console eras rather than abstract style names: "PS1 era" tells a user what
# the result will look like in a way "Low-poly" does not, and the ladder from
# NES to PS5 is itself the fidelity axis.
#
# Deliberately no literal "pixel art" token in any fragment -- this binds the
# nes and snes entries in particular. At 512/1024 SDXL renders fake chunky
# pixels that alias under the real NEAREST downscale in asset2d.pixel, and the
# same reference feeds trellis on promotion. What survives downscaling is flat
# shading and a bold silhouette.
ART_STYLES = _table(
    Option(
        "nes", "NES era",
        "flat colour shading, bold readable silhouette, clean dark outlines",
    ),
    Option("snes", "SNES era", "vivid saturated colours, bold simple shapes"),
    Option("ps1", "PS1 era", "low-poly, flat-shaded faceted surfaces, minimal detail"),
    Option(
        "ps2", "PS2 era",
        "hand-painted texture style, painterly brushwork, baked lighting",
    ),
    Option("ps3", "PS3/360 era", "stylized realistic game art, strong surface detail"),
    Option(
        "ps5", "PS5 era",
        "photorealistic PBR materials, physically accurate surfaces",
    ),
)

CATEGORIES = _table(
    Option("prop", "Prop", "a game prop", default_size_m=0.4),
    Option("weapon", "Weapon", "a game weapon", default_size_m=1.0),
    Option(
        "character", "Character",
        "a game character, full body, standing, T-pose neutral stance",
        default_size_m=1.8,
    ),
    Option("vehicle", "Vehicle", "a game vehicle", default_size_m=4.5),
    Option("environment", "Environment piece", "a modular environment piece", default_size_m=8.0),
    Option("consumable", "Consumable", "a small consumable pickup item", default_size_m=0.15),
)

# What the asset is *for*, which is the only thing the user actually knows at
# the point of asking: a 2D asset is going to be seen flat and small, a 3D one
# in a scene. The geometry resolution follows from that rather than from a
# hardware tier the user has to translate.
PLATFORMS = _table(
    Option(
        "2d", "2D",
        "clean readable silhouette, simple flat forms",
        resolution=512,
    ),
    Option(
        "3d", "3D",
        "moderate surface detail, clear material separation",
        resolution=1024,
    ),
)

# The camera the reference is drawn under, and the one taxonomy table that does
# *not* compose into the subject clause: its fragment is injected into
# pipelines/prompt.PROMPT_TEMPLATE's ``{view}`` slot, where the literal "3/4
# perspective view" used to sit. ``three_quarter`` carries that exact string,
# which is the whole reason PROMPT_VERSION stays at 4 -- the default
# composition is byte-identical to the pre-framing one, so no recipe, no
# prompt_hash and no stored vector is re-keyed.
#
# ``front_ortho`` is a measurement axis rather than a new default: whether a
# straight-on plate reconstructs better than a 3/4 view is a question for a
# sweep, and TODO's answer is "make it expressible, then measure". Two things
# about its wording are deliberate. It names a *camera* and never a pose, so it
# cannot contradict the T-pose fragment CATEGORIES["character"] already
# carries; and it says "one view only", because front-on plus T-pose is
# exactly the character-sheet/turnaround layout that caused all 17 refusals of
# the 2026-08-07 rogue sweep, and that guard is cheaper than the refusal.
FRAMINGS = _table(
    Option("three_quarter", "3/4 view", "3/4 perspective view"),
    Option(
        "front_ortho", "Front orthographic",
        "front orthographic view, facing the camera, one view only",
    ),
)

# Like DEFAULT_PLATFORM: always written into params, never left to absence.
DEFAULT_FRAMING = "three_quarter"

MATERIALS = _table(
    Option("wood", "Wood", "wooden construction"),
    Option("iron", "Iron", "iron construction"),
    Option("steel", "Steel", "steel construction"),
    Option("bronze", "Bronze", "bronze construction"),
    Option("stone", "Stone", "carved stone"),
    Option("leather", "Leather", "leather construction"),
    Option("bone", "Bone", "bone construction"),
    Option("crystal", "Crystal", "faceted crystal"),
    Option("glass", "Glass", "glass construction"),
    Option("ceramic", "Ceramic", "glazed ceramic"),
    Option("gold", "Gold", "gilded gold accents"),
    Option("fabric", "Fabric", "woven fabric"),
)

CONDITIONS = _table(
    Option("pristine", "Pristine", "pristine condition"),
    Option("worn", "Worn", "worn weathered surfaces"),
    Option("damaged", "Damaged", "battle-damaged surfaces"),
    Option("ancient", "Ancient", "ancient timeworn surfaces"),
    Option("rusted", "Rusted", "rusted corroded surfaces"),
    Option("overgrown", "Overgrown", "moss-covered and overgrown"),
    Option("burned", "Burned", "charred burned surfaces"),
)

SETTINGS = _table(
    Option("medieval", "Medieval", "medieval European setting"),
    Option("norse", "Norse", "Norse Viking setting"),
    Option("japanese", "Japanese", "feudal Japanese setting"),
    Option("egyptian", "Egyptian", "ancient Egyptian setting"),
    Option("greco", "Greco-Roman", "Greco-Roman setting"),
    Option("steampunk", "Steampunk", "steampunk brass setting"),
    Option("cyberpunk", "Cyberpunk", "cyberpunk neon setting"),
    Option("tribal", "Tribal", "tribal primitive setting"),
    Option("deco", "Art Deco", "art deco setting"),
    Option("military", "Military", "modern military setting"),
)

PALETTES = _table(
    Option("earth", "Earth tones", "earthy natural palette"),
    Option("steel", "Cool steel", "cool steel palette"),
    Option("muted", "Muted", "muted desaturated palette"),
    Option("vibrant", "Vibrant", "vibrant saturated palette"),
    Option("mono", "Monochrome", "monochrome palette"),
    Option("crimson", "Crimson", "crimson red palette"),
    Option("verdigris", "Verdigris", "verdigris green patina"),
    Option("ivory", "Ivory", "ivory cream palette"),
)

EMISSIVES = _table(
    Option("runes", "Glowing runes", "glowing magic runes"),
    Option("neon", "Neon", "glowing neon accents"),
    Option("molten", "Molten cracks", "glowing molten cracks"),
    Option("holo", "Holographic", "holographic light accents"),
    Option("arcane", "Arcane glow", "arcane energy glow"),
    Option("toxic", "Toxic glow", "toxic green glow"),
)

RARITIES = _table(
    Option("common", "Common", "common plain quality"),
    Option("uncommon", "Uncommon", "uncommon refined quality"),
    Option("rare", "Rare", "rare exceptional quality"),
    Option("epic", "Epic", "epic masterwork quality"),
    Option("legendary", "Legendary", "legendary mythical quality"),
)

SILHOUETTES = _table(
    Option("bulky", "Bulky", "bulky heavy silhouette"),
    Option("slender", "Slender", "slender narrow silhouette"),
    Option("compact", "Compact", "compact dense silhouette"),
    Option("angular", "Angular", "angular sharp silhouette"),
    Option("rounded", "Rounded", "rounded soft silhouette"),
    Option("elongated", "Elongated", "elongated tall silhouette"),
)

MOODS = _table(
    Option("heroic", "Heroic", "heroic noble mood"),
    Option("grim", "Grim", "grim dark mood"),
    Option("whimsical", "Whimsical", "whimsical playful mood"),
    Option("sacred", "Sacred", "sacred reverent mood"),
    Option("sinister", "Sinister", "sinister menacing mood"),
    Option("regal", "Regal", "regal majestic mood"),
)

_OPTION_TABLES: dict[str, dict[str, Option]] = {
    "genre": GENRES,
    "art_style": ART_STYLES,
    "category": CATEGORIES,
    "platform": PLATFORMS,
    "framing": FRAMINGS,
    "material": MATERIALS,
    "condition": CONDITIONS,
    "setting": SETTINGS,
    "palette": PALETTES,
    "emissive": EMISSIVES,
    "rarity": RARITIES,
    "silhouette": SILHOUETTES,
    "mood": MOODS,
}

# Keys that used to exist, mapped onto the ones that replaced them. Every job
# row already on disk carries the old spellings, and rerun/promotion re-run
# normalize() over stored params -- without this a stored ``platform: "hero"``
# would 400 rather than reroll. normalize() writes the canonical key back, so
# params migrate the first time a job is touched.
#
# The one accepted cost: findings and verdict buckets split between the old and
# the new key, because a vector recorded before the rename is a different
# string. Evidence gathered under the old names simply stops accumulating.
_LEGACY_ALIASES: dict[str, dict[str, str]] = {
    "platform": {"mobile": "2d", "desktop": "3d", "hero": "3d"},
    "art_style": {
        "realistic": "ps5",
        "stylized": "ps3",
        "lowpoly": "ps1",
        "handpainted": "ps2",
        "toon": "snes",
        "pixelart": "nes",
    },
}


def _canonical(field: str, value: str) -> str:
    """The current key for a value, which for anything current is itself."""
    return _LEGACY_ALIASES.get(field, {}).get(value, value)

# The model-selection fields are validated and stored here so the API gets its
# 400-on-unknown from the same place as everything else, but they are owned by
# models.py and are deliberately absent from _PROMPT_FIELDS: a checkpoint is
# not a prompt fragment, and a LoRA's trigger words are model scaffolding that
# belongs next to PROMPT_TEMPLATE, not creative direction.
_TABLES: dict[str, dict[str, Any]] = {
    **_OPTION_TABLES,
    "base_model": models.BASE_MODELS,
    "style_lora": models.STYLE_LORAS,
    # Conditioning selections. Same reasoning as base_model/style_lora: they
    # are validated here so the 400 comes from one place, and are absent from
    # _PROMPT_FIELDS because they are not prompt fragments -- compose_prompt's
    # output is byte-identical with and without any of them.
    "ip_adapter": models.IP_ADAPTERS,
    "control": models.CONTROLNETS,
}

# Order matters: this is the order fragments appear in the composed prompt, so
# it should read like a sentence. This preserves the relative order of the
# four original fields (category -> genre -> art_style -> platform), which
# tests/test_guidance.py pins.
_PROMPT_FIELDS = (
    "category", "silhouette", "material", "condition", "rarity", "emissive",
    "setting", "genre", "mood", "art_style", "palette", "platform",
)


def form_fields() -> tuple[str, ...]:
    """Every field name the API accepts as a taxonomy value."""
    return tuple(_TABLES)


def _lookup(field: str, value: Any) -> Any | None:
    """None for 'not specified'; ValueError for a value we do not know."""
    if value is None or value == "":
        return None
    table = _TABLES[field]
    option = table.get(_canonical(field, str(value)))
    if option is None:
        raise GuidanceError(
            f"unknown {field} {value!r}; expected one of {sorted(table)}", field=field
        )
    return option


def _number(raw: dict[str, Any], field: str, *, default: float, low: float, high: float) -> float:
    """A bounded float field, with the same missing-or-empty-means-default rule
    lora_weight has had since it was the only one."""
    value = raw.get(field)
    if value in (None, ""):
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise GuidanceError(f"{field} must be a number, got {value!r}", field=field) from exc
    if not low <= value <= high:
        raise GuidanceError(f"{field} must be between {low} and {high}", field=field)
    return value


def normalize(raw: dict[str, Any], *, bg_default: str | None = None) -> dict[str, Any]:
    """Validate submitted guidance and fill in what it implies.

    Returns the dict stored in jobs.params, so it also carries the derived
    ``resolution`` the worker actually sends to trellis-server. Raises
    ValueError (which the API turns into a 400) on any unrecognised value.

    ``bg_default`` is the matte to use when ``raw`` names none -- what
    ``default_bg_removal`` resolved against the host's weights directory. It is
    a passed value rather than a lookup because this module is pure and holds
    no Config; ``None`` means "no gate was applied", which is the stated
    preference and what the validation-only callers (bench recipes, the sweep
    parser) want, since they never reach a server.
    """
    chosen = {field: _lookup(field, raw.get(field)) for field in _TABLES}

    platform = chosen["platform"] or PLATFORMS[DEFAULT_PLATFORM]
    # Written explicitly for the same reason platform is: the prompt compiler
    # needs a view clause for every text job and must never have to guess one,
    # and a bucket keyed on "unset" would describe the default population as a
    # different configuration from the one that names it.
    framing = chosen["framing"] or FRAMINGS[DEFAULT_FRAMING]
    resolution = raw.get("resolution")
    # An explicit resolution stays an override; otherwise the platform supplies it.
    resolution = int(resolution) if resolution not in (None, "") else platform.resolution

    size_m = raw.get("size_m")
    if size_m in (None, ""):
        category = chosen["category"]
        size_m = category.default_size_m if category else DEFAULT_SIZE_M
    else:
        try:
            size_m = float(size_m)
        except (TypeError, ValueError) as exc:
            raise GuidanceError(
                f"size_m must be a number, got {size_m!r}", field="size_m"
            ) from exc
        if not SIZE_MIN_M <= size_m <= SIZE_MAX_M:
            raise GuidanceError(
                f"size_m must be between {SIZE_MIN_M} and {SIZE_MAX_M} metres", field="size_m"
            )

    base_model = chosen["base_model"] or models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    style_lora = chosen["style_lora"]

    lora_weight = raw.get("lora_weight")
    if lora_weight in (None, ""):
        # The LoRA's own tuned default, so picking a style is a one-click action.
        lora_weight = style_lora.default_weight if style_lora else models.DEFAULT_LORA_WEIGHT
    else:
        try:
            lora_weight = float(lora_weight)
        except (TypeError, ValueError) as exc:
            raise GuidanceError(
                f"lora_weight must be a number, got {lora_weight!r}", field="lora_weight"
            ) from exc
        if not models.LORA_WEIGHT_MIN <= lora_weight <= models.LORA_WEIGHT_MAX:
            raise GuidanceError(
                f"lora_weight must be between {models.LORA_WEIGHT_MIN} "
                f"and {models.LORA_WEIGHT_MAX}",
                field="lora_weight",
            )

    if style_lora is not None and base_model.family != models.FAMILY_SDXL:
        # Symmetric with the ControlNet refusal below, and refused for a
        # stronger reason: every style LoRA in the registry names SDXL UNet
        # modules, so loading one onto another architecture raises at load time
        # with the checkpoint already in VRAM rather than merely doing nothing.
        # ``base_model``, not ``style_lora``: two controls are in conflict and
        # only one of them can be highlighted, so it is the one the message
        # already tells the user to change. A highlight that disagreed with the
        # sentence beside it would be worse than none.
        raise GuidanceError(
            f"base_model {base_model.key!r} cannot take a style LoRA; "
            f"pick one of {models.lora_bases()}",
            field="base_model",
        )

    ip_adapter = chosen["ip_adapter"]
    control = chosen["control"]
    if control is not None and not base_model.controlnet:
        raise GuidanceError(
            f"base_model {base_model.key!r} cannot run a ControlNet; "
            f"pick one of {models.controlnet_bases()}",
            field="base_model",
        )

    # The selection's own tuned default, so picking an adapter is one click --
    # the same rule lora_weight follows.
    ip_scale = _number(
        raw, "ip_scale",
        default=ip_adapter.default_scale if ip_adapter else models.DEFAULT_IP_SCALE,
        low=models.IP_SCALE_MIN, high=models.IP_SCALE_MAX,
    )
    control_scale = _number(
        raw, "control_scale",
        default=control.default_scale if control else models.DEFAULT_CONTROL_SCALE,
        low=models.CONTROL_SCALE_MIN, high=models.CONTROL_SCALE_MAX,
    )
    control_end = _number(
        raw, "control_end",
        default=control.default_end if control else models.DEFAULT_CONTROL_END,
        low=models.CONTROL_END_MIN, high=models.CONTROL_END_MAX,
    )

    bg_removal = raw.get("bg_removal")
    if bg_removal in (None, ""):
        bg_removal = bg_default or DEFAULT_BG_REMOVAL
    # Validated after the fallback rather than only on the explicit branch: the
    # default is a value a caller hands in now, and an unknown one would
    # otherwise reach trellis-server unchecked -- the one path where "the
    # server decides" is not a safe answer.
    if str(bg_removal) not in BG_REMOVAL:
        raise GuidanceError(f"bg_removal must be one of {list(BG_REMOVAL)}", field="bg_removal")

    # Unlike bg_removal, a missing key and an explicit "" are NOT the same
    # thing here: missing means "use the game-asset default", an explicit
    # empty string means the user deliberately wants no negative prompt at all.
    negative = raw.get("negative_prompt")
    if negative is None:
        negative = DEFAULT_NEGATIVE_PROMPT
    negative = str(negative).strip()
    if len(negative) > MAX_NEGATIVE_PROMPT:
        raise GuidanceError(
            f"negative_prompt must be at most {MAX_NEGATIVE_PROMPT} characters",
            field="negative_prompt",
        )

    out: dict[str, Any] = {
        "resolution": resolution,
        "size_m": size_m,
        "platform": platform.key,
        "framing": framing.key,
        "bg_removal": str(bg_removal),
        "negative_prompt": negative,
        # Always present, unlike the optional fields below: the worker needs a
        # base model for every text job and must never have to guess one.
        "base_model": base_model.key,
    }
    # Only carried when a style was actually chosen -- a stored lora_weight with
    # no lora would read as "a LoRA at 0.9" on rerun.
    if style_lora is not None:
        out["style_lora"] = style_lora.key
        out["lora_weight"] = lora_weight
    # Same rule again for both conditioning halves: a scale with nothing to
    # scale reads as a live setting on rerun.
    if ip_adapter is not None:
        out["ip_adapter"] = ip_adapter.key
        out["ip_scale"] = ip_scale
    if control is not None:
        out["control"] = control.key
        out["control_scale"] = control_scale
        out["control_end"] = control_end
    # Every optional taxonomy field except platform and framing, which are
    # always written explicitly above. Derived from _OPTION_TABLES rather than a
    # hand-picked tuple so a new table can never be silently dropped from
    # params again.
    for field in _OPTION_TABLES:
        if field in ("platform", "framing"):
            continue
        option = chosen[field]
        if option is not None:
            out[field] = option.key
    return out


def framing_clause(value: Any) -> str:
    """The view clause ``PROMPT_TEMPLATE``'s ``{view}`` slot takes.

    Takes the value rather than the params dict, because the two callers hold
    different things: prompt.build() has the whole params, text2image.generate()
    has one string handed down beside the composed subject.

    Skips an unknown value rather than raising, for compose_prompt's reason and
    not _lookup's: this runs at sample time on a job whose row is already
    written, and a spelling this build does not carry should compose the
    default view rather than fail a job. Absence means the same thing --
    ``three_quarter`` is the clause every row on disk was composed under when
    it was a template literal.
    """
    option = FRAMINGS.get(_canonical("framing", str(value or "")))
    return (option or FRAMINGS[DEFAULT_FRAMING]).prompt


def compose_prompt(
    user_prompt: str, params: dict[str, Any], fields: tuple[str, ...] | None = None
) -> str:
    """Fold the guidance fragments into the subject clause of the SDXL prompt.

    ``fields`` restricts which tables contribute, defaulting to all of them in
    their canonical order. The one caller that narrows it is the tile path: a
    tile has no subject, so category, silhouette and rarity describe an object
    that is not in the picture, and a prompt that names one gets an object.

    Unknown or absent values are skipped rather than raising: params may come
    from a job row created before a taxonomy entry was renamed or removed, and
    a slightly less specific prompt beats failing an otherwise valid job.
    """
    parts = [user_prompt.strip()]
    for field in _PROMPT_FIELDS:
        if fields is not None and field not in fields:
            continue
        option = _TABLES[field].get(_canonical(field, str(params.get(field, ""))))
        if option is not None:
            parts.append(option.prompt)
    return ", ".join(p for p in parts if p)


# Whole recipes: a prompt skeleton plus every guidance field that makes the
# style land. Defined here rather than in the browser because the fields name
# taxonomy and model keys, and normalize() is what decides whether those are
# still valid -- a preset that names a removed LoRA should fail this module's
# tests, not a user's submit.
PRESETS: tuple[dict[str, Any], ...] = (
    {
        "key": "handpainted_prop",
        "label": "Hand-painted fantasy prop",
        "prompt": "a weathered wooden crate bound with iron",
        "fields": {
            "category": "prop",
            "genre": "fantasy",
            "art_style": "ps2",
            "platform": "3d",
            "base_model": "sdxl",
            "style_lora": "render3d",
            "material": "wood",
            "condition": "worn",
            "setting": "medieval",
            "palette": "earth",
        },
    },
    {
        "key": "ps1_character",
        "label": "PS1 low-poly character",
        "prompt": "a hooded adventurer standing in a neutral pose",
        "fields": {
            "category": "character",
            "genre": "fantasy",
            "art_style": "ps1",
            "platform": "2d",
            "base_model": "sdxl",
            "style_lora": "ps1",
            "silhouette": "slender",
            "condition": "worn",
            "setting": "medieval",
            "mood": "heroic",
        },
    },
    {
        "key": "scifi_hero_weapon",
        "label": "Sci-fi hero weapon",
        "prompt": "a compact energy rifle with panel seams and glowing vents",
        "fields": {
            "category": "weapon",
            "genre": "scifi",
            "art_style": "ps5",
            "platform": "3d",
            "base_model": "playground",
            "material": "steel",
            "condition": "pristine",
            "emissive": "neon",
            "rarity": "epic",
        },
    },
    {
        "key": "modern_pickup",
        "label": "Modern consumable pickup",
        "prompt": "a small first-aid kit",
        "fields": {
            "category": "consumable",
            "genre": "modern",
            "art_style": "ps3",
            "platform": "2d",
            "base_model": "turbo",
            "material": "fabric",
            "condition": "pristine",
            "palette": "vibrant",
            "rarity": "common",
        },
    },
    {
        # art_style "nes" and not a new taxonomy option: its fragment is
        # already what survives a real downscale, and the word "pixel" belongs
        # to the LoRA's trigger (models.STYLE_LORAS) rather than to any prompt
        # fragment. The base is the LCM arm because that is the pixel-art-xl
        # author's documented recipe -- the null hypothesis a bench run over
        # bench/suites/pixel-v1 is there to overturn or keep.
        "key": "pixel_sprite",
        "label": "Pixel-art sprite",
        "prompt": "a hooded adventurer standing in a neutral pose",
        "fields": {
            "category": "character",
            "genre": "fantasy",
            "art_style": "nes",
            "platform": "2d",
            "base_model": "pixel",
            # No lora_weight: normalize() fills the LoRA's own default (1.2),
            # so the author's recipe lives in one place.
            "style_lora": "pixelxl",
            "silhouette": "slender",
            "palette": "vibrant",
        },
    },
)


def catalog(*, bg_default: str | None = None) -> dict[str, Any]:
    """The taxonomy as JSON, so the UI builds its selects from one source.

    ``bg_default`` is ``normalize``'s, for the same reason and with the same
    meaning: the form's initial matte must be the one a submit would pick, or
    the pane shows a setting the job will not run at.
    """
    return {
        "fields": {
            field: [
                {
                    "key": opt.key,
                    "label": opt.label,
                    **({"default_size_m": opt.default_size_m} if opt.default_size_m else {}),
                    **({"resolution": opt.resolution} if opt.resolution else {}),
                }
                for opt in table.values()
            ]
            for field, table in _OPTION_TABLES.items()
        }
        | models.catalog(),
        "bg_removal": list(BG_REMOVAL),
        # Not a select, so not under "fields": what the UI checks the chosen
        # base against before it draws the Structure group at all.
        "controlnet_bases": models.controlnet_bases(),
        # Same shape and the same purpose as controlnet_bases: what the UI
        # checks the chosen base against before it presents the negative
        # prompt as a live control rather than an inert one.
        "cfg_bases": models.cfg_bases(),
        # And again for the style LoRA: what the UI checks before it offers the
        # picker as a live control rather than a disabled one with a reason.
        "lora_bases": models.lora_bases(),
        # Copied, not handed out: the UI reads these and a shared dict would let
        # a caller mutate the shipped table.
        "presets": [dict(p) for p in PRESETS],
        "defaults": {
            "platform": DEFAULT_PLATFORM,
            "framing": DEFAULT_FRAMING,
            "size_m": DEFAULT_SIZE_M,
            "base_model": models.DEFAULT_BASE_MODEL,
            "lora_weight": models.DEFAULT_LORA_WEIGHT,
            "bg_removal": bg_default or DEFAULT_BG_REMOVAL,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        },
        "size_range_m": [SIZE_MIN_M, SIZE_MAX_M],
        "lora_weight_range": [models.LORA_WEIGHT_MIN, models.LORA_WEIGHT_MAX],
        "ip_scale_range": [models.IP_SCALE_MIN, models.IP_SCALE_MAX],
        "control_scale_range": [models.CONTROL_SCALE_MIN, models.CONTROL_SCALE_MAX],
        "control_end_range": [models.CONTROL_END_MIN, models.CONTROL_END_MAX],
    }

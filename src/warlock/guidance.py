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
from typing import Any

from . import models

SIZE_MIN_M = 0.01
SIZE_MAX_M = 100.0

# Fallback when no category is chosen and so no default size applies.
DEFAULT_SIZE_M = 1.0
DEFAULT_PLATFORM = "desktop"

# How trellis-server mattes the input image. Not an Option table: these are
# server capabilities, not prompt fragments, so they never reach compose_prompt.
BG_REMOVAL = ("auto", "birefnet", "threshold")
DEFAULT_BG_REMOVAL = "auto"

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

ART_STYLES = _table(
    Option(
        "realistic", "Realistic PBR",
        "photorealistic PBR materials, physically accurate surfaces",
    ),
    Option("stylized", "Stylized", "stylized game art, bold shapes, slightly exaggerated forms"),
    Option("lowpoly", "Low-poly", "low-poly, flat-shaded faceted surfaces, minimal detail"),
    Option(
        "handpainted", "Hand-painted",
        "hand-painted texture style, painterly brushwork, baked lighting",
    ),
    Option("toon", "Toon", "cel-shaded toon look, clean flat colours, crisp outlines"),
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

PLATFORMS = _table(
    Option(
        "mobile", "Mobile / VR",
        "clean readable silhouette, simple forms, minimal fine detail",
        resolution=512,
    ),
    Option(
        "desktop", "Indie desktop",
        "moderate surface detail, clear material separation",
        resolution=1024,
    ),
    Option(
        "hero", "Hero asset",
        "intricate fine surface detail, rich material variation, high fidelity",
        resolution=1536,
    ),
)

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
    "material": MATERIALS,
    "condition": CONDITIONS,
    "setting": SETTINGS,
    "palette": PALETTES,
    "emissive": EMISSIVES,
    "rarity": RARITIES,
    "silhouette": SILHOUETTES,
    "mood": MOODS,
}

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
    option = table.get(str(value))
    if option is None:
        raise ValueError(f"unknown {field} {value!r}; expected one of {sorted(table)}")
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
        raise ValueError(f"{field} must be a number, got {value!r}") from exc
    if not low <= value <= high:
        raise ValueError(f"{field} must be between {low} and {high}")
    return value


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate submitted guidance and fill in what it implies.

    Returns the dict stored in jobs.params, so it also carries the derived
    ``resolution`` the worker actually sends to trellis-server. Raises
    ValueError (which the API turns into a 400) on any unrecognised value.
    """
    chosen = {field: _lookup(field, raw.get(field)) for field in _TABLES}

    platform = chosen["platform"] or PLATFORMS[DEFAULT_PLATFORM]
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
            raise ValueError(f"size_m must be a number, got {size_m!r}") from exc
        if not SIZE_MIN_M <= size_m <= SIZE_MAX_M:
            raise ValueError(f"size_m must be between {SIZE_MIN_M} and {SIZE_MAX_M} metres")

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
            raise ValueError(f"lora_weight must be a number, got {lora_weight!r}") from exc
        if not models.LORA_WEIGHT_MIN <= lora_weight <= models.LORA_WEIGHT_MAX:
            raise ValueError(
                f"lora_weight must be between {models.LORA_WEIGHT_MIN} "
                f"and {models.LORA_WEIGHT_MAX}"
            )

    ip_adapter = chosen["ip_adapter"]
    control = chosen["control"]
    if control is not None and not base_model.controlnet:
        raise ValueError(
            f"base_model {base_model.key!r} cannot run a ControlNet; "
            f"pick one of {models.controlnet_bases()}"
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
        bg_removal = DEFAULT_BG_REMOVAL
    elif str(bg_removal) not in BG_REMOVAL:
        raise ValueError(f"bg_removal must be one of {list(BG_REMOVAL)}")

    # Unlike bg_removal, a missing key and an explicit "" are NOT the same
    # thing here: missing means "use the game-asset default", an explicit
    # empty string means the user deliberately wants no negative prompt at all.
    negative = raw.get("negative_prompt")
    if negative is None:
        negative = DEFAULT_NEGATIVE_PROMPT
    negative = str(negative).strip()
    if len(negative) > MAX_NEGATIVE_PROMPT:
        raise ValueError(f"negative_prompt must be at most {MAX_NEGATIVE_PROMPT} characters")

    out: dict[str, Any] = {
        "resolution": resolution,
        "size_m": size_m,
        "platform": platform.key,
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
    # Every optional taxonomy field except platform, which is always written
    # explicitly above. Derived from _OPTION_TABLES rather than a hand-picked
    # tuple so a new table can never be silently dropped from params again.
    for field in _OPTION_TABLES:
        if field == "platform":
            continue
        option = chosen[field]
        if option is not None:
            out[field] = option.key
    return out


def compose_prompt(user_prompt: str, params: dict[str, Any]) -> str:
    """Fold the guidance fragments into the subject clause of the SDXL prompt.

    Unknown or absent values are skipped rather than raising: params may come
    from a job row created before a taxonomy entry was renamed or removed, and
    a slightly less specific prompt beats failing an otherwise valid job.
    """
    parts = [user_prompt.strip()]
    for field in _PROMPT_FIELDS:
        option = _TABLES[field].get(str(params.get(field, "")))
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
            "art_style": "handpainted",
            "platform": "desktop",
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
            "art_style": "lowpoly",
            "platform": "mobile",
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
            "art_style": "realistic",
            "platform": "hero",
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
            "art_style": "stylized",
            "platform": "mobile",
            "base_model": "turbo",
            "material": "fabric",
            "condition": "pristine",
            "palette": "vibrant",
            "rarity": "common",
        },
    },
)


def catalog() -> dict[str, Any]:
    """The taxonomy as JSON, so the UI builds its selects from one source."""
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
        # Copied, not handed out: the UI reads these and a shared dict would let
        # a caller mutate the shipped table.
        "presets": [dict(p) for p in PRESETS],
        "defaults": {
            "platform": DEFAULT_PLATFORM,
            "size_m": DEFAULT_SIZE_M,
            "base_model": models.DEFAULT_BASE_MODEL,
            "lora_weight": models.DEFAULT_LORA_WEIGHT,
            "bg_removal": DEFAULT_BG_REMOVAL,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        },
        "size_range_m": [SIZE_MIN_M, SIZE_MAX_M],
        "lora_weight_range": [models.LORA_WEIGHT_MIN, models.LORA_WEIGHT_MAX],
        "ip_scale_range": [models.IP_SCALE_MIN, models.IP_SCALE_MAX],
        "control_scale_range": [models.CONTROL_SCALE_MIN, models.CONTROL_SCALE_MAX],
        "control_end_range": [models.CONTROL_END_MIN, models.CONTROL_END_MAX],
    }

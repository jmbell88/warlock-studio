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
pipelines/text2image.PROMPT_TEMPLATE, which wraps whatever we produce.
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

_OPTION_TABLES: dict[str, dict[str, Option]] = {
    "genre": GENRES,
    "art_style": ART_STYLES,
    "category": CATEGORIES,
    "platform": PLATFORMS,
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
}

# Order matters: this is the order fragments appear in the composed prompt.
_PROMPT_FIELDS = ("category", "genre", "art_style", "platform")


def _lookup(field: str, value: Any) -> Any | None:
    """None for 'not specified'; ValueError for a value we do not know."""
    if value is None or value == "":
        return None
    table = _TABLES[field]
    option = table.get(str(value))
    if option is None:
        raise ValueError(f"unknown {field} {value!r}; expected one of {sorted(table)}")
    return option


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
    for field in ("genre", "art_style", "category"):
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
    }

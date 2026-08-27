"""The normalized contract behind Studio's Create workflow.

The UI and the historical service doors used to describe the same request in
different vocabularies.  This module is deliberately torch-free and owns the
small, serializable vocabulary that connects them.  It is safe to use from
settings migration, API validation, and worker planning code.

The request is an input document.  A :class:`ResolvedRecipe` is the immutable
answer to "what will run".  Keeping those two separate means automatic routing
can evolve without changing the meaning of an already queued job.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import models

GENERATION_TYPES = ("image", "3d_model", "seamless_material", "tileset", "sprite_sheet")
GENERATION_TYPE_OPTIONS = (
    ("image", "Image"),
    ("3d_model", "3D Model"),
    ("seamless_material", "Seamless Material"),
    ("tileset", "Tileset"),
    ("sprite_sheet", "Sprite Sheet"),
)
QUALITY_TIERS = ("fast", "quality")
MODEL_MODES = ("auto", "advanced")
REFERENCE_MODES = ("none", "single", "multi")
TARGET_CELL_PRESETS = (16, 24, 32, 48, 64, 96, 128, 256)
TARGET_CELL_MIN = 8
TARGET_CELL_MAX = 256
SPRITE_ACTIONS = ("idle", "walk", "run", "attack", "jump")
SPRITE_FRAME_COUNTS = {"idle": 4, "walk": 8, "run": 8, "attack": 6, "jump": 6}
VIEW_NAMES = ("front", "left", "right", "back")


@dataclass(frozen=True, slots=True)
class TileSettings:
    mode: str = "collection"
    view: str = "top_down"
    content_kind: str = "terrain"
    prompt_items: tuple[str, ...] = ()
    inner_terrain: str = ""
    outer_terrain: str = ""
    boundary: str = ""
    ground: str = ""
    path: str = ""
    edge: str = ""
    variants: int = 1
    target_cell_px: int | None = None


@dataclass(frozen=True, slots=True)
class SpriteSettings:
    mode: str = "action"
    action: str = "idle"
    directions: int = 4
    frame_count: int | None = None
    candidate_count: int = 2
    target_cell_px: int | None = None

    def resolved_frame_count(self) -> int:
        return self.frame_count or SPRITE_FRAME_COUNTS[self.action]


@dataclass(frozen=True, slots=True)
class ModelSettings:
    backend: str = "trellis_single_view"
    views: Mapping[str, str] = field(default_factory=dict)
    texture_mode: str = "pbr"
    output_profile: str = "raw"
    custom_triangles: int | None = None
    license_acknowledged: bool = False


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A stable, JSON-friendly Create request.

    ``model_mode='auto'`` leaves model choice to :func:`resolve_recipe`.
    ``model_override`` is retained even when it is incompatible so the UI can
    explain the refusal instead of silently replacing the user's choice.
    """

    generation_type: str = "3d_model"
    prompt: str = ""
    negative_prompt: str = ""
    quality: str = "quality"
    model_mode: str = "auto"
    model_override: str | None = None
    style_lora: str | None = None
    lora_weight: float | None = None
    references: tuple[str, ...] = ()
    reference_mode: str = "none"
    seed: int = 0
    count: int = 1
    tile: TileSettings = field(default_factory=TileSettings)
    sprite: SpriteSettings = field(default_factory=SpriteSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GenerationRequest":
        tile = raw.get("tile") if isinstance(raw.get("tile"), Mapping) else {}
        sprite = raw.get("sprite") if isinstance(raw.get("sprite"), Mapping) else {}
        model = raw.get("model") if isinstance(raw.get("model"), Mapping) else {}
        generation_type = raw.get("generation_type")
        if not generation_type:
            generation_type = legacy_asset_type(raw)
        return cls(
            generation_type=str(generation_type or "3d_model"),
            prompt=str(raw.get("prompt") or ""),
            negative_prompt=str(raw.get("negative_prompt") or ""),
            quality=str(raw.get("quality") or "quality"),
            model_mode=str(raw.get("model_mode") or "auto"),
            model_override=(str(raw["model_override"]) if raw.get("model_override") else None),
            style_lora=(str(raw["style_lora"]) if raw.get("style_lora") else None),
            lora_weight=raw.get("lora_weight"),
            references=tuple(str(x) for x in raw.get("references") or ()),
            reference_mode=str(raw.get("reference_mode") or "none"),
            seed=int(raw.get("seed") or 0),
            count=int(raw.get("count") or 1),
            tile=TileSettings(**{k: (tuple(v) if k == "prompt_items" else v) for k, v in tile.items() if k in TileSettings.__dataclass_fields__}),
            sprite=SpriteSettings(**{k: v for k, v in sprite.items() if k in SpriteSettings.__dataclass_fields__}),
            model=ModelSettings(**{k: v for k, v in model.items() if k in ModelSettings.__dataclass_fields__}),
            schema_version=int(raw.get("schema_version") or 1),
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    key: str
    label: str
    generation_types: tuple[str, ...]
    quality: str
    base_model: str
    default_lora: str | None = None
    working_resolution: tuple[int, int] = (1024, 1024)
    reference_modes: tuple[str, ...] = ("none", "single")
    negative_prompt: bool = False
    vram_gib: float = 0.0
    ram_gib: float = 0.0
    license: str = ""
    commercial: bool = True
    required_downloads: tuple[str, ...] = ()
    rank: int = 0

    @property
    def supports_negative_prompt(self) -> bool:
        return self.negative_prompt


@dataclass(frozen=True, slots=True)
class ResolvedRecipe:
    recipe: Recipe
    base_model: str
    style_lora: str | None
    model_checksum: str | None = None
    lora_checksum: str | None = None
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": asdict(self.recipe),
            "base_model": self.base_model,
            "style_lora": self.style_lora,
            "model_checksum": self.model_checksum,
            "lora_checksum": self.lora_checksum,
            "warning": self.warning,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    field: str
    message: str
    severity: str = "error"


def _recipe_table() -> tuple[Recipe, ...]:
    """Build recipes from the authoritative model registry.

    No checkpoint is duplicated here.  The registry describes how a model is
    run; this table describes which asset outcome that run is qualified for.
    """
    return (
        Recipe("image_fast", "Fast image", ("image",), "fast", "sdxl_cfg", working_resolution=(1024, 1024), negative_prompt=True, vram_gib=7, ram_gib=8, required_downloads=("base:sdxl_cfg",), rank=20),
        Recipe("image_quality", "Quality image", ("image", "3d_model"), "quality", "sdxl_cfg", working_resolution=(1024, 1024), negative_prompt=True, vram_gib=7, ram_gib=8, required_downloads=("base:sdxl_cfg",), rank=30),
        Recipe("material_sdxl", "Seamless material", ("seamless_material",), "quality", "sdxl_cfg", working_resolution=(1024, 1024), reference_modes=("none", "single"), negative_prompt=True, vram_gib=7, ram_gib=8, required_downloads=("base:sdxl_cfg",), rank=40),
        Recipe("tileset_sdxl", "Tileset", ("tileset",), "quality", "sdxl_cfg", working_resolution=(1024, 1024), reference_modes=("none", "single"), negative_prompt=True, vram_gib=7, ram_gib=8, required_downloads=("base:sdxl_cfg",), rank=40),
        Recipe("sprite_sdxl", "Sprite sheet", ("sprite_sheet",), "quality", "sdxl_cfg", working_resolution=(1024, 1024), reference_modes=("none", "single", "multi"), negative_prompt=True, vram_gib=7, ram_gib=8, required_downloads=("base:sdxl_cfg", "adapter:plus", "control:canny", "lora:pixelxl"), rank=40),
        # SDXL remains the default qualified recipe. FLUX.2 is selected when
        # it is the only installed candidate, or explicitly in Advanced; its
        # native reference path is still available without changing the
        # existing SDXL default.
        Recipe("image_flux2", "FLUX.2 Klein image", ("image", "3d_model"), "quality", "flux_klein", working_resolution=(1024, 1024), reference_modes=("none", "single", "multi"), negative_prompt=True, vram_gib=10, ram_gib=16, required_downloads=("base:flux_klein",), rank=25),
    )


RECIPES = _recipe_table()
RECIPE_REGISTRY = RECIPES
RECIPE_REGISTRY_VERSION = 1


def recipe_registry() -> tuple[Recipe, ...]:
    return RECIPES


def _present(key: str, config: Any | None) -> bool:
    if config is None:
        return True
    try:
        from . import fetch

        entry = fetch.find(key)
        return entry is not None and fetch.present(config, entry.kind, entry.spec)
    except Exception:
        return False


def _checksum(key: str, config: Any | None) -> str | None:
    if config is None:
        return None
    try:
        from . import fetch, provenance

        entry = fetch.find(key)
        if entry is None:
            return None
        paths = [fetch.destination(config, entry, one) for one in entry.fetch]
        if paths:
            return provenance.model_fingerprints({key: paths[0]})[key]
    except Exception:
        pass
    return None


def resolve_recipe(
    request: GenerationRequest,
    config: Any | None = None,
    *,
    installed: Iterable[str] | None = None,
) -> ResolvedRecipe | None:
    """Resolve a request without falling back from an explicit choice.

    Automatic selection filters non-commercial recipes and missing downloads.
    Explicit Advanced selection is allowed to select a non-commercial model;
    the warning travels with the resolved recipe and must be displayed by the
    caller.
    """
    register_imported_loras(config)
    candidates = [r for r in RECIPES if request.generation_type in r.generation_types and r.quality == request.quality]
    available = set(installed) if installed is not None else None
    if request.model_mode == "advanced" or request.model_override:
        key = request.model_override or request.model.backend
        base = models.BASE_MODELS.get(key)
        if base is None:
            return None
        candidate = next((r for r in candidates if r.base_model == key), None)
        if candidate is None:
            candidate = Recipe(f"advanced_{key}", base.label, (request.generation_type,), request.quality, key, working_resolution=(base.image_size, base.image_size), reference_modes=("none", "single"), negative_prompt=base.guidance_scale > 1, vram_gib=base.vram_gib, ram_gib=base.host_peak_gib or base.vram_gib, license=base.license or "", commercial=base.commercial, rank=0)
        warning = "" if base.commercial else f"{base.license or 'This model'} does not permit commercial use."
        lora_checksum = _checksum(f"lora:{request.style_lora}", config) if request.style_lora else None
        manifest = imported_lora(config, request.style_lora or "")
        if manifest is not None:
            lora_checksum = manifest.checksum
        return ResolvedRecipe(candidate, key, request.style_lora or candidate.default_lora, _checksum(f"base:{key}", config), lora_checksum, warning)
    for candidate in sorted(candidates, key=lambda r: r.rank, reverse=True):
        if not candidate.commercial:
            continue
        if available is not None:
            if not all(k in available for k in candidate.required_downloads):
                continue
        elif not all(_present(k, config) for k in candidate.required_downloads):
            continue
        lora_checksum = None
        manifest = imported_lora(config, request.style_lora or "")
        if manifest is not None:
            lora_checksum = manifest.checksum
        return ResolvedRecipe(candidate, candidate.base_model, request.style_lora or candidate.default_lora, _checksum(f"base:{candidate.base_model}", config), lora_checksum)
    return None


def capability_controls(request: GenerationRequest, resolved: ResolvedRecipe | None) -> dict[str, bool]:
    """Return visibility/availability for adaptive controls."""
    recipe = resolved.recipe if resolved else None
    return {
        "negative_prompt": bool(recipe and recipe.supports_negative_prompt),
        "references": bool(recipe and recipe.reference_modes != ("none",)),
        "multi_reference": bool(recipe and "multi" in recipe.reference_modes),
        "controlnet": bool(recipe and models.BASE_MODELS.get(recipe.base_model, None) and models.BASE_MODELS[recipe.base_model].controlnet),
        "style_lora": bool(recipe and recipe.base_model in models.lora_bases()),
        "tile": request.generation_type == "seamless_material",
        "tiles": request.generation_type == "tileset",
        "sprites": request.generation_type == "sprite_sheet",
        "model_backend": request.generation_type == "3d_model",
    }


def validate_target_cell(target_cell_px: int | None, *, isometric: bool = False) -> list[CompatibilityIssue]:
    if target_cell_px is None:
        return []
    try:
        value = int(target_cell_px)
    except (TypeError, ValueError):
        return [CompatibilityIssue("target_cell_px", "Cell target must be a whole number.")]
    if not TARGET_CELL_MIN <= value <= TARGET_CELL_MAX:
        return [CompatibilityIssue("target_cell_px", f"Cell target must be between {TARGET_CELL_MIN} and {TARGET_CELL_MAX} pixels.")]
    if isometric and value % 2:
        return [CompatibilityIssue("target_cell_px", "Isometric cell widths must be even so height can be exactly half the width.")]
    return []


def cell_dimensions(working: tuple[int, int], target_cell_px: int | None, *, isometric: bool = False) -> tuple[int, int]:
    """Return output dimensions; a blank target never reduces or upscales."""
    if target_cell_px is None:
        return int(working[0]), int(working[1])
    issues = validate_target_cell(target_cell_px, isometric=isometric)
    if issues:
        raise ValueError(issues[0].message)
    width = int(target_cell_px)
    height = width // 2 if isometric else width
    if width > working[0] or height > working[1]:
        raise ValueError("target cell size is larger than the working cell; generation will not upscale")
    return width, height


def validate_request(request: GenerationRequest, resolved: ResolvedRecipe | None = None) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    if request.generation_type not in GENERATION_TYPES:
        issues.append(CompatibilityIssue("generation_type", "Choose one of the supported generation types."))
    if not request.prompt.strip():
        issues.append(CompatibilityIssue("prompt", "A prompt is required."))
    if request.quality not in QUALITY_TIERS:
        issues.append(CompatibilityIssue("quality", "Quality must be Fast or Quality."))
    if request.model_mode not in MODEL_MODES:
        issues.append(CompatibilityIssue("model_mode", "Model mode must be automatic or Advanced."))
    if request.reference_mode not in REFERENCE_MODES:
        issues.append(CompatibilityIssue("reference_mode", "Unknown reference mode."))
    if request.reference_mode == "multi" and len(request.references) < 2:
        issues.append(CompatibilityIssue("references", "Multi-reference mode needs at least two images."))
    if request.count < 1:
        issues.append(CompatibilityIssue("count", "Count must be at least one."))
    if request.generation_type == "tileset":
        t = request.tile
        if t.mode not in ("collection", "terrain_transition", "path"):
            issues.append(CompatibilityIssue("tile.mode", "Tilesets support Collection, Terrain transition, and Path modes."))
        if t.mode == "collection":
            if not 1 <= len(t.prompt_items) <= 16:
                issues.append(CompatibilityIssue("tile.prompt_items", "Collection tilesets accept 1–16 prompt lines."))
            if not 1 <= t.variants <= 4 or len(t.prompt_items) * t.variants > 64:
                issues.append(CompatibilityIssue("tile.variants", "Collection variants must be 1–4 and total no more than 64 cells."))
        elif t.mode == "terrain_transition" and (not t.inner_terrain.strip() or not t.outer_terrain.strip()):
            issues.append(CompatibilityIssue("tile.terrain", "Terrain transitions need inner and outer terrain descriptions."))
        elif t.mode == "path" and (not t.ground.strip() or not t.path.strip()):
            issues.append(CompatibilityIssue("tile.path", "Path sets need ground and path descriptions."))
        issues.extend(validate_target_cell(t.target_cell_px, isometric=t.view == "isometric"))
    if request.generation_type == "sprite_sheet":
        s = request.sprite
        if s.mode not in ("turnaround", "action"):
            issues.append(CompatibilityIssue("sprite.mode", "Sprites support Turnaround and action sheets."))
        if s.mode == "action" and s.action not in SPRITE_ACTIONS:
            issues.append(CompatibilityIssue("sprite.action", "Unknown sprite action."))
        if s.directions not in (4, 8):
            issues.append(CompatibilityIssue("sprite.directions", "Sprites support 4 or 8 directions."))
        if not 1 <= s.candidate_count <= 2:
            issues.append(CompatibilityIssue("sprite.candidate_count", "Sprite candidate count must be 1 or 2."))
        issues.extend(validate_target_cell(s.target_cell_px))
    if request.generation_type == "3d_model":
        if request.model.backend not in ("trellis_single_view", "hunyuan3d_multiview"):
            issues.append(CompatibilityIssue("model.backend", "Choose TRELLIS or the experimental Hunyuan3D backend."))
        if request.model.backend == "hunyuan3d_multiview":
            view_errors = [name for name in VIEW_NAMES if not request.model.views.get(name)]
            if view_errors:
                issues.append(CompatibilityIssue("model.views", "Approve Front, Left, Right, and Back views before using Hunyuan3D."))
            if not request.model.license_acknowledged:
                issues.append(CompatibilityIssue("model.license_acknowledged", "Acknowledge the Hunyuan3D regional license exclusions before use."))
    if resolved is None:
        issues.append(CompatibilityIssue("recipe", "No compatible installed recipe is available. Install a qualified recipe or choose a compatible Advanced model."))
    else:
        if request.reference_mode not in resolved.recipe.reference_modes:
            issues.append(CompatibilityIssue("reference_mode", f"{resolved.recipe.label} does not support {request.reference_mode} references."))
        if request.negative_prompt.strip() and not resolved.recipe.supports_negative_prompt:
            issues.append(CompatibilityIssue("negative_prompt", f"{resolved.recipe.label} does not support negative prompts; remove Avoid text or choose another model."))
        if request.style_lora:
            lora = models.STYLE_LORAS.get(request.style_lora)
            base = models.BASE_MODELS.get(resolved.base_model)
            if lora is None:
                issues.append(CompatibilityIssue("style_lora", "The selected style LoRA is not in the catalog."))
            elif base is not None and not models.lora_fits(base, lora):
                issues.append(CompatibilityIssue("style_lora", "The selected style LoRA is not fitted to the resolved model."))
    return issues


def legacy_asset_type(form: Mapping[str, Any]) -> str:
    """Map old Create fields to the five new generation types."""
    if form.get("generation_type") in GENERATION_TYPES:
        return str(form["generation_type"])
    output = form.get("output")
    if output == "tile":
        return "seamless_material"
    if output == "sheet":
        if form.get("sheet_type") == "sprite":
            return "sprite_sheet"
        return "tileset"
    return "3d_model" if form.get("asset_type") != "image_2d" else "image"


def request_from_legacy(form: Mapping[str, Any]) -> GenerationRequest:
    generation_type = legacy_asset_type(form)
    projection = str(form.get("projection") or "top_down")
    if projection == "orthogonal":
        projection = "top_down"
    tile = TileSettings(view=projection, target_cell_px=_optional_int(form.get("target_cell_px")))
    sprite = SpriteSettings(
        mode="turnaround" if form.get("sheet_layout", "turnaround") == "turnaround" else "action",
        action="walk" if form.get("sheet_layout") == "walk" else "idle",
        target_cell_px=_optional_int(form.get("target_cell_px") or form.get("cell_size")),
    )
    return GenerationRequest(
        generation_type=generation_type,
        prompt=str(form.get("prompt") or ""),
        negative_prompt=str(form.get("negative_prompt") or ""),
        quality=str(form.get("quality") or "quality"),
        model_mode=str(form.get("model_mode") or "auto"),
        model_override=str(form["model_override"] or form["base_model"]) if form.get("model_mode") == "advanced" and (form.get("model_override") or form.get("base_model")) else None,
        style_lora=str(form["style_lora"]) if form.get("style_lora") else None,
        lora_weight=form.get("lora_weight"),
        references=(str(form["ref_path"]),) if form.get("ref_path") else (),
        reference_mode="single" if form.get("ref_path") else "none",
        seed=int(form.get("seed") or 0),
        count=int(form.get("count") or 1),
        tile=tile,
        sprite=sprite,
    )


def request_to_legacy(request: GenerationRequest) -> dict[str, Any]:
    """Compatibility adapter for existing ``create_job`` doors."""
    output = {"image": "reference", "3d_model": "reference", "seamless_material": "tile", "tileset": "sheet", "sprite_sheet": "sheet"}[request.generation_type]
    out: dict[str, Any] = {
        "asset_type": request.generation_type,
        "asset_intent": {"image": "refine_2d", "3d_model": "reconstruct_3d", "seamless_material": "refine_2d", "tileset": "tileset", "sprite_sheet": "sprite"}[request.generation_type],
        "output": output,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or None,
        "seed": request.seed,
        "count": request.count,
    }
    if request.style_lora:
        out["style_lora"] = request.style_lora
        out["lora_weight"] = request.lora_weight
    if request.references:
        out["references"] = list(request.references)
    if request.generation_type == "tileset":
        out["projection"] = request.tile.view
        out["tile_settings"] = asdict(request.tile)
    if request.generation_type == "sprite_sheet":
        out["sprite_settings"] = asdict(request.sprite)
    return out


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class LoraManifest:
    key: str
    label: str
    family: str
    trigger_text: str
    tuned_weight: float
    license: str
    commercial: bool
    source: str
    checksum: str
    filename: str
    schema_version: int = 1


def lora_manifest_path(config: Any) -> Path:
    return Path(config.t2i_model_root) / "loras" / "manifests.json"


def load_lora_manifests(config: Any) -> list[LoraManifest]:
    # Total by construction: managed adapters are an optional extra, and
    # ``resolve_recipe`` calls this on every submit. A config that cannot say
    # where the model root is -- a partial one, or a stub -- means "no imported
    # adapters", never a failed generate. The *write* path above still requires
    # the root, because there is nowhere to put the file without it.
    try:
        raw = json.loads(lora_manifest_path(config).read_text(encoding="utf-8"))
    except (AttributeError, TypeError, FileNotFoundError, OSError, ValueError):
        return []
    rows = raw.get("manifests", raw) if isinstance(raw, Mapping) else []
    out: list[LoraManifest] = []
    for row in rows:
        try:
            out.append(LoraManifest(**row))
        except (TypeError, ValueError):
            continue
    return out


def imported_lora(config: Any | None, key: str) -> LoraManifest | None:
    if config is None:
        return None
    return next((row for row in load_lora_manifests(config) if row.key == key), None)


def register_imported_loras(config: Any | None) -> None:
    """Expose managed local adapters through the existing model loader."""
    if config is None:
        return
    for manifest in load_lora_manifests(config):
        existing = models.STYLE_LORAS.get(manifest.key)
        if existing is not None and existing.filename == manifest.filename:
            continue
        models.STYLE_LORAS[manifest.key] = models.StyleLora(
            key=manifest.key,
            label=manifest.label,
            filename=manifest.filename,
            trigger=manifest.trigger_text,
            default_weight=manifest.tuned_weight,
            family=manifest.family,
        )


def lora_catalog(config: Any | None = None) -> list[dict[str, Any]]:
    register_imported_loras(config)
    out = [{"key": x.key, "label": x.label, "family": x.family, "trigger_text": x.trigger, "tuned_weight": x.default_weight, "license": "", "commercial": True, "source": "built-in", "checksum": ""} for x in models.STYLE_LORAS.values()]
    if config is not None:
        out.extend(asdict(x) for x in load_lora_manifests(config))
    return out


def import_lora(config: Any, source: Path | str, *, label: str, family: str = models.FAMILY_SDXL, trigger_text: str = "", tuned_weight: float = models.DEFAULT_LORA_WEIGHT, license: str = "", commercial: bool = False, source_url: str = "local file") -> LoraManifest:
    """Copy one local safetensors adapter into managed storage and register it."""
    source_path = Path(source)
    if source_path.suffix.lower() != ".safetensors":
        raise ValueError("a LoRA must be a .safetensors file")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    key = f"imported_{digest[:16]}"
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("._-") or key
    filename = f"{key}_{safe_label}.safetensors"
    root = Path(config.t2i_model_root) / "loras"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if not destination.exists() or destination.stat().st_size != source_path.stat().st_size:
        shutil.copy2(source_path, destination)
    manifest = LoraManifest(key, label, family, trigger_text, float(tuned_weight), license, bool(commercial), source_url, digest, filename)
    path = lora_manifest_path(config)
    manifests = [x for x in load_lora_manifests(config) if x.key != key]
    manifests.append(manifest)
    payload = json.dumps({"version": 1, "manifests": [asdict(x) for x in manifests]}, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False, prefix=".manifests-", suffix=".tmp") as fh:
        fh.write(payload)
        temp = Path(fh.name)
    temp.replace(path)
    register_imported_loras(config)
    return manifest

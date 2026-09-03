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
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
#: The seven actions a sprite sheet can depict, and how many frames each is.
#:
#: ``pipelines.spritesynth.ACTIONS``' table, restated rather than imported for
#: this module's stated reason -- it is the request *document*, and naming its
#: own vocabulary should not cost it an import of a pipeline. It had five, and
#: the two it was missing were not idle curiosities: ``cast`` and ``hurt`` are
#: the pair that exist **only** on this path (they have no Blender clip behind
#: them), so the one table that could not name them was the one a request is
#: written in. ``tests/test_sprite_geometry_agreement.py`` owns the agreement.
SPRITE_ACTIONS = ("idle", "walk", "run", "attack", "cast", "hurt", "jump")
SPRITE_FRAME_COUNTS = {
    "idle": 4,
    "walk": 8,
    "run": 8,
    "attack": 6,
    "cast": 6,
    "hurt": 4,
    "jump": 6,
}

#: How many directions a sprite sheet may carry. ``spritesynth.DIRECTION_COUNTS``.
SPRITE_DIRECTION_COUNTS = (4, 8)

#: The shapes a sprite request comes in. Two of them name a *sheet* and one
#: names a kind of sheet: ``turnaround`` and ``walk`` are the two fixed atlases
#: this path has always drawn -- the second being a four-frame cycle over four
#: directions, which is emphatically not the eight-frame ``walk4`` the action
#: table plans -- and ``action`` is "the action and direction count named
#: below". A mode that names itself is how a legacy kind stays requestable
#: without pretending to be an action whose frame count it does not have.
SPRITE_MODES = ("turnaround", "walk", "action")
SPRITE_LEGACY_MODES = ("turnaround", "walk")

#: ``kind -> (action, directions)`` for every sheet the action table plans. A
#: table and not a parse of the trailing digits, for ``spritesynth.plan_kind``'s
#: reason: an action is free to be named ``dash2`` one day, and a parser would
#: quietly read that as a two-direction ``dash``.
SPRITE_SHEET_KINDS: dict[str, tuple[str, int]] = {
    f"{action}{count}": (action, count)
    for action in SPRITE_ACTIONS
    for count in SPRITE_DIRECTION_COUNTS
}


#: The mode words a tile request may carry, and the two shapes they name.
#:
#: ``asset_workflows.TILE_MODE_ALIASES``' own table, restated here rather than
#: imported. It was a cycle until the grid planner was deleted (2026-08-29) and
#: ``asset_workflows`` stopped importing this module; the copy stays because
#: this file is the request *document* and naming its own vocabulary should not
#: cost it an import of a planning helper.
#: ``tests/test_tileset_service.py`` pins the two copies together.
TILE_MODES = ("collection", "materials", "terrain_transition", "terrain", "path")


@dataclass(frozen=True, slots=True)
class TileSettings:
    #: ``"collection"`` and not ``"materials"``: this default is what
    #: ``from_dict`` fills in for every stored row that predates the field, and
    #: moving it would silently re-read old rows as a different mode. The two
    #: words mean the same shape -- see :data:`TILE_MODES`.
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
    #: Which layout a terrain set is laid out on. One value ships --
    #: ``pipelines.tilemask``'s forty-seven blob cases -- and it is a field
    #: rather than a constant because it is *recorded*: a set's layout is what
    #: says which column means which coverage case, and a reader of a stored row
    #: must not have to infer it from the year it was made.
    terrain_layout: str = "blob47"
    #: Whether the cells of one request are held to a single sampler seed so N
    #: separately generated materials read as one sheet, rather than each taking
    #: its own seed from ``tileatlas.material_seeds``' derived family. Recorded
    #: here and acted on by the worker; the door's only job is to carry it, so a
    #: stored row can say which of the two a set was made under. Off by default,
    #: because the derived family is what makes "reroll material 3 only"
    #: reproducible and that is the commoner ask.
    style_lock: bool = False
    #: A second, masked img2img pass over each material's wrapped seam cross.
    seam_erase: bool = False
    target_cell_px: int | None = None
    #: The stem of an authored palette file, or ``""`` for "derive one". The
    #: structured request had no way to name one, so a tileset submitted here
    #: could not use a capability the pane path could -- and the door has taken
    #: both of these since the day it grew them. Validated at that door
    #: (``service.tilesheets.create_tile_sheet`` -> ``check_pixel_options``,
    #: which loads the file and throws it away), never here: this is a document,
    #: and a request naming a palette that has since been deleted has to be
    #: refused against the filesystem it is submitted on rather than the one it
    #: was written on.
    palette: str = ""
    #: The ordered 4x4 offset in ``pixel.map_palette``. Independent of
    #: :attr:`palette` on this path, and deliberately: ``tilesheet.quantize_tiles``
    #: branches on ``not entries and not dither``, so a dithered sheet that names
    #: no palette still derives its own table by median cut and then dithers
    #: against it.
    dither: bool = False


@dataclass(frozen=True, slots=True)
class SpriteSettings:
    mode: str = "action"
    action: str = "idle"
    directions: int = 4
    frame_count: int | None = None
    #: ``None`` is "nobody said", exactly as :attr:`frame_count` and
    #: :attr:`target_cell_px` mean it, and it is the default because the right
    #: answer depends on how big the sheet is: a pair of a four-cell turnaround
    #: is the feature, and a pair of an eight-direction walk is sixteen
    #: generations. ``spritesynth.default_candidates`` decides at the door; a
    #: request that names 1 or 2 is honoured.
    candidate_count: int | None = None
    target_cell_px: int | None = None
    #: :attr:`TileSettings.palette` and :attr:`TileSettings.dither`, for the
    #: sprite half of the same gap: the follow-up block this request compiles
    #: reaches ``sprites._check_options`` through ``_check_sprite_sheet``, which
    #: has taken both since it started sharing that checker. The outline is
    #: **not** here -- the sprite path's outline default is forced by its
    #: geometry (``spritesynth.DEFAULT_SPRITE_OUTLINE``) and nothing on this
    #: route has ever named one, so a field carrying it would be the dead field
    #: ``check_pixel_options`` exists to refuse.
    palette: str = ""
    dither: bool = False

    def resolved_frame_count(self) -> int:
        return self.frame_count or SPRITE_FRAME_COUNTS[self.action]


@dataclass(frozen=True, slots=True)
class ModelSettings:
    output_profile: str = "raw"
    custom_triangles: int | None = None


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
    #: The ControlNet key this request asks for, or ``""``. A *declaration*,
    #: not a route: :func:`request_to_legacy` deliberately does not emit it,
    #: because the structured door has never carried structure control and
    #: growing it one here would be a new capability rather than the honesty
    #: fix this field exists for. What it is for is :func:`validate_request` --
    #: a recipe whose base has ``controlnet=False`` (Fast) must refuse the
    #: pairing in a sentence naming a control the user can see, rather than
    #: letting the legacy params carry the selection to ``guidance.normalize``
    #: and come back naming ``base_model``, which under automatic routing is
    #: not drawn at all.
    structure_control: str = ""
    #: img2img: start the denoise from the first reference, at this strength.
    init_image: bool = False
    init_strength: float | None = None
    seed: int = 0
    count: int = 1
    tile: TileSettings = field(default_factory=TileSettings)
    sprite: SpriteSettings = field(default_factory=SpriteSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GenerationRequest:
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
            structure_control=str(raw.get("structure_control") or ""),
            seed=int(raw.get("seed") or 0),
            count=int(raw.get("count") or 1),
            tile=TileSettings(
                **{
                    k: (tuple(v) if k == "prompt_items" else v)
                    for k, v in tile.items()
                    if k in TileSettings.__dataclass_fields__
                }
            ),
            sprite=SpriteSettings(
                **{k: v for k, v in sprite.items() if k in SpriteSettings.__dataclass_fields__}
            ),
            model=ModelSettings(
                **{k: v for k, v in model.items() if k in ModelSettings.__dataclass_fields__}
            ),
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
    #: One sentence saying what picking this recipe costs, shown under the
    #: resolved-recipe line in Create. A tier that is honestly worse and says
    #: so is a choice; one that is quietly worse is a defect, and Fast was the
    #: second of those for as long as it named the same checkpoint as Quality.
    #: Empty on the recipes a user cannot choose between -- the sheet arms pin
    #: their own base and never reach the Fast/Quality control.
    note: str = ""
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
        # Fast is ``sdxl`` -- SDXL 1.0 with Hyper-SD on top, four steps at
        # guidance 0 -- and Quality is ``sdxl_cfg``, the same base weights run
        # at 30 steps with real CFG. Both tiers named ``sdxl_cfg`` until
        # 2026-08-29, which made the Fast/Quality control a label over one
        # picture: two recipe *names*, identical checkpoint, identical
        # sampler, identical resolution. A control that changes nothing is the
        # defect; the fix is to let it change something and say what.
        #
        # The trade is measured, not guessed.
        # ``docs/measurements/2026-08-11-default-base-model.md`` scores this
        # exact arm: ``sdxl`` took 2 of 4 accepted and ``sdxl_cfg`` 3 of 3 --
        # tiny n, no significance claimed, and a documented quality trade is
        # precisely what "Fast" is supposed to mean. ``sdxl`` and not
        # ``turbo``/``lightning``: those are 512 px, non-commercial or their
        # own 7 GB checkpoint, while this shares Quality's base weights and is
        # the backend where style LoRAs behave as trained.
        #
        # ``base_model`` is in ``vectors.VECTOR_PARAMS``, so this changes what
        # a job records. Nothing stored is invalidated: until today *both*
        # tiers wrote ``base_model=sdxl_cfg`` and no finding in the corpus can
        # tell a Fast job from a Quality one, so the corpus gets more truthful
        # rather than less.
        Recipe(
            "image_fast",
            "Fast image",
            ("image",),
            "fast",
            "sdxl",
            working_resolution=(1024, 1024),
            # Inert here and therefore declared inert. ``sdxl`` runs at
            # guidance 0 and ``text2image`` encodes the negative branch only
            # above 1.0 (``models.cfg_bases``), so a True here would have moved
            # the lie out of the tier and into the field: an Avoid box that
            # takes text, stores it in params and changes no pixel.
            negative_prompt=False,
            vram_gib=7,
            ram_gib=8,
            # One row covers it: ``sdxl``'s ``fetch`` tuple already carries the
            # shared SDXL 1.0 base *and* the 0.8 GB Hyper-SD LoRA, so there is
            # no second key to name here.
            required_downloads=("base:sdxl",),
            note=(
                "Four steps instead of thirty, on the same SDXL weights. No "
                "structure control and no negative prompt: it runs at guidance "
                "0, so there is no unconditioned branch for either to steer. "
                "Quality is measurably better at holding a shape."
            ),
            rank=20,
        ),
        Recipe(
            "image_quality",
            "Quality image",
            ("image", "3d_model"),
            "quality",
            "sdxl_cfg",
            working_resolution=(1024, 1024),
            negative_prompt=True,
            vram_gib=7,
            ram_gib=8,
            required_downloads=("base:sdxl_cfg",),
            note=(
                "Thirty steps with full classifier-free guidance: the only "
                "tier that takes a ControlNet, and the only one where the "
                "negative prompt carries weight."
            ),
            rank=30,
        ),
        Recipe(
            "material_sdxl",
            "Seamless material",
            ("seamless_material",),
            "quality",
            "sdxl_cfg",
            working_resolution=(1024, 1024),
            reference_modes=("none", "single"),
            negative_prompt=True,
            vram_gib=7,
            ram_gib=8,
            required_downloads=("base:sdxl_cfg",),
            rank=40,
        ),
        Recipe(
            "tileset_sdxl",
            "Tileset",
            ("tileset",),
            "quality",
            "sdxl_cfg",
            working_resolution=(1024, 1024),
            reference_modes=("none", "single"),
            negative_prompt=True,
            vram_gib=7,
            ram_gib=8,
            required_downloads=("base:sdxl_cfg",),
            rank=40,
        ),
        Recipe(
            "sprite_sdxl",
            "Sprite sheet",
            ("sprite_sheet",),
            "quality",
            "sdxl_cfg",
            working_resolution=(1024, 1024),
            reference_modes=("none", "single", "multi"),
            negative_prompt=True,
            vram_gib=7,
            ram_gib=8,
            required_downloads=("base:sdxl_cfg", "adapter:plus", "control:canny", "lora:pixelxl"),
            rank=40,
        ),
        # SDXL remains the default qualified recipe. FLUX.2 is selected when
        # it is the only installed candidate, or explicitly in Advanced; its
        # native reference path is still available without changing the
        # existing SDXL default.
        Recipe(
            "image_flux2",
            "FLUX.2 Klein image",
            ("image", "3d_model"),
            "quality",
            "flux_klein",
            working_resolution=(1024, 1024),
            reference_modes=("none", "single", "multi"),
            negative_prompt=True,
            vram_gib=10,
            ram_gib=16,
            required_downloads=("base:flux_klein",),
            rank=25,
        ),
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
    candidates = [
        r
        for r in RECIPES
        if request.generation_type in r.generation_types and r.quality == request.quality
    ]
    available = set(installed) if installed is not None else None
    if request.model_mode == "advanced" or request.model_override:
        key = request.model_override or ""
        base = models.BASE_MODELS.get(key)
        if base is None:
            return None
        candidate = next((r for r in candidates if r.base_model == key), None)
        if candidate is None:
            candidate = Recipe(
                f"advanced_{key}",
                base.label,
                (request.generation_type,),
                request.quality,
                key,
                working_resolution=(base.image_size, base.image_size),
                reference_modes=("none", "single"),
                negative_prompt=base.guidance_scale > 1,
                vram_gib=base.vram_gib,
                ram_gib=base.host_peak_gib or base.vram_gib,
                license=base.license or "",
                commercial=base.commercial,
                rank=0,
            )
        warning = (
            ""
            if base.commercial
            else f"{base.license or 'This model'} does not permit commercial use."
        )
        lora_checksum = (
            _checksum(f"lora:{request.style_lora}", config) if request.style_lora else None
        )
        manifest = imported_lora(config, request.style_lora or "")
        if manifest is not None:
            lora_checksum = manifest.checksum
        return ResolvedRecipe(
            candidate,
            key,
            request.style_lora or candidate.default_lora,
            _checksum(f"base:{key}", config),
            lora_checksum,
            warning,
        )
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
        return ResolvedRecipe(
            candidate,
            candidate.base_model,
            request.style_lora or candidate.default_lora,
            _checksum(f"base:{candidate.base_model}", config),
            lora_checksum,
        )
    return None


def _takes_controlnet(resolved: ResolvedRecipe | None) -> bool:
    """Whether the checkpoint this recipe resolves to can run a ControlNet.

    The registry's ``controlnet`` flag and nothing else, so the pane's picker,
    :func:`capability_controls` and :func:`validate_request`'s refusal cannot
    come to disagree -- ``models.lora_fits``' rule for the other pairing.
    """
    if resolved is None:
        return False
    spec = models.BASE_MODELS.get(resolved.base_model)
    return bool(spec is not None and spec.controlnet)


def capability_controls(
    request: GenerationRequest, resolved: ResolvedRecipe | None
) -> dict[str, bool]:
    """Return visibility/availability for adaptive controls."""
    recipe = resolved.recipe if resolved else None
    return {
        "negative_prompt": bool(recipe and recipe.supports_negative_prompt),
        "references": bool(recipe and recipe.reference_modes != ("none",)),
        "multi_reference": bool(recipe and "multi" in recipe.reference_modes),
        "controlnet": _takes_controlnet(resolved),
        "style_lora": bool(recipe and recipe.base_model in models.lora_bases()),
        "tile": request.generation_type == "seamless_material",
        "tiles": request.generation_type == "tileset",
        "sprites": request.generation_type == "sprite_sheet",
    }


def validate_target_cell(
    target_cell_px: int | None, *, isometric: bool = False
) -> list[CompatibilityIssue]:
    if target_cell_px is None:
        return []
    try:
        value = int(target_cell_px)
    except (TypeError, ValueError):
        return [CompatibilityIssue("target_cell_px", "Cell target must be a whole number.")]
    if not TARGET_CELL_MIN <= value <= TARGET_CELL_MAX:
        return [
            CompatibilityIssue(
                "target_cell_px",
                f"Cell target must be between {TARGET_CELL_MIN} and {TARGET_CELL_MAX} pixels.",
            )
        ]
    if isometric and value % 2:
        return [
            CompatibilityIssue(
                "target_cell_px",
                "Isometric cell widths must be even so height can be exactly half the width.",
            )
        ]
    return []


def cell_dimensions(
    working: tuple[int, int], target_cell_px: int | None, *, isometric: bool = False
) -> tuple[int, int]:
    """Return output dimensions; a blank target never reduces or upscales."""
    if target_cell_px is None:
        return int(working[0]), int(working[1])
    issues = validate_target_cell(target_cell_px, isometric=isometric)
    if issues:
        raise ValueError(issues[0].message)
    width = int(target_cell_px)
    height = width // 2 if isometric else width
    if width > working[0] or height > working[1]:
        raise ValueError(
            "target cell size is larger than the working cell; generation will not upscale"
        )
    return width, height


def validate_request(
    request: GenerationRequest, resolved: ResolvedRecipe | None = None
) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    if request.generation_type not in GENERATION_TYPES:
        issues.append(
            CompatibilityIssue("generation_type", "Choose one of the supported generation types.")
        )
    if not request.prompt.strip():
        issues.append(CompatibilityIssue("prompt", "A prompt is required."))
    if request.quality not in QUALITY_TIERS:
        issues.append(CompatibilityIssue("quality", "Quality must be Fast or Quality."))
    if request.model_mode not in MODEL_MODES:
        issues.append(CompatibilityIssue("model_mode", "Model mode must be automatic or Advanced."))
    if request.reference_mode not in REFERENCE_MODES:
        issues.append(CompatibilityIssue("reference_mode", "Unknown reference mode."))
    if request.reference_mode == "multi" and len(request.references) < 2:
        issues.append(
            CompatibilityIssue("references", "Multi-reference mode needs at least two images.")
        )
    if request.count < 1:
        issues.append(CompatibilityIssue("count", "Count must be at least one."))
    if request.generation_type == "tileset":
        t = request.tile
        if t.mode not in TILE_MODES:
            issues.append(
                CompatibilityIssue(
                    "tile.mode",
                    "Tilesets support Materials and Terrain modes.",
                )
            )
        if t.mode in ("collection", "materials"):
            if not 1 <= len(t.prompt_items) <= 16:
                issues.append(
                    CompatibilityIssue(
                        "tile.prompt_items", "Collection tilesets accept 1–16 prompt lines."
                    )
                )
            if not 1 <= t.variants <= 4 or len(t.prompt_items) * t.variants > 64:
                issues.append(
                    CompatibilityIssue(
                        "tile.variants",
                        "Collection variants must be 1–4 and total no more than 64 cells.",
                    )
                )
        elif t.mode in ("terrain", "terrain_transition") and (
            not t.inner_terrain.strip() or not t.outer_terrain.strip()
        ):
            # Both halves, because both are generated: a terrain set is two
            # seamless materials composited through a coverage field, so a
            # request that describes one of them has nothing to put on the other
            # side of every boundary.
            issues.append(
                CompatibilityIssue(
                    "tile.terrain", "Terrain sets need inner and outer terrain descriptions."
                )
            )
        elif t.mode == "path" and (not t.ground.strip() or not t.path.strip()):
            issues.append(
                CompatibilityIssue("tile.path", "Path sets need ground and path descriptions.")
            )
        if t.mode in ("terrain", "terrain_transition", "path") and t.terrain_layout != "blob47":
            issues.append(
                CompatibilityIssue(
                    "tile.terrain_layout",
                    f"{t.terrain_layout!r} is not a terrain layout; the blob-47 "
                    "autotile set is the one that ships.",
                )
            )
        issues.extend(validate_target_cell(t.target_cell_px, isometric=t.view == "isometric"))
    if request.generation_type == "sprite_sheet":
        s = request.sprite
        if s.mode not in SPRITE_MODES:
            issues.append(
                CompatibilityIssue("sprite.mode", "Sprites support Turnaround and action sheets.")
            )
        if s.mode == "action" and s.action not in SPRITE_ACTIONS:
            issues.append(CompatibilityIssue("sprite.action", "Unknown sprite action."))
        if s.directions not in SPRITE_DIRECTION_COUNTS:
            issues.append(
                CompatibilityIssue("sprite.directions", "Sprites support 4 or 8 directions.")
            )
        if s.candidate_count is not None and not 1 <= s.candidate_count <= 2:
            issues.append(
                CompatibilityIssue(
                    "sprite.candidate_count", "Sprite candidate count must be 1 or 2."
                )
            )
        issues.extend(validate_target_cell(s.target_cell_px))
    if resolved is None:
        issues.append(
            CompatibilityIssue(
                "recipe",
                "No compatible installed recipe is available. Install a qualified "
                "recipe or choose a compatible Advanced model.",
            )
        )
    else:
        if request.reference_mode not in resolved.recipe.reference_modes:
            issues.append(
                CompatibilityIssue(
                    "reference_mode",
                    f"{resolved.recipe.label} does not support "
                    f"{request.reference_mode} references.",
                )
            )
        if request.structure_control and not _takes_controlnet(resolved):
            # Refuse rather than drop. The conditioning is the whole point of
            # attaching it, and a tier that quietly ran without it would be the
            # same defect as a Fast tier that quietly drew the same picture as
            # Quality. The field named is the one the user can act on: under
            # automatic routing that is the Fast/Quality control, because the
            # model combo is only drawn under Advanced.
            field_name = "base_model" if request.model_mode == "advanced" else "quality"
            issues.append(
                CompatibilityIssue(
                    field_name,
                    f"{resolved.recipe.label} runs at guidance 0 and cannot run a "
                    "ControlNet. "
                    + (
                        "Choose a full-CFG model."
                        if field_name == "base_model"
                        else "Switch the recipe to Quality, or clear the structure control."
                    ),
                )
            )
        # A saved brief can carry Avoid text from a full-CFG model into a
        # distilled one. That is not an invalid request: the negative branch
        # is simply absent for the selected recipe. Refusing here stranded the
        # text behind a hidden control and stopped FLUX.2 klein distilled
        # generations from starting. The worker omits the inert value through
        # ``effective_negative_prompt`` instead.
        if request.style_lora:
            lora = models.STYLE_LORAS.get(request.style_lora)
            base = models.BASE_MODELS.get(resolved.base_model)
            if lora is None:
                issues.append(
                    CompatibilityIssue(
                        "style_lora", "The selected style LoRA is not in the catalog."
                    )
                )
            elif base is not None and not models.lora_fits(base, lora):
                issues.append(
                    CompatibilityIssue(
                        "style_lora", "The selected style LoRA is not fitted to the resolved model."
                    )
                )
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


def sprite_from_layout(layout: Any) -> tuple[str, str, int]:
    """``(mode, action, directions)`` for a form's stored ``sheet_layout``.

    That field holds a *sheet kind* -- one of the two legacy atlases, or one of
    the planned ``f"{action}{directions}"`` names -- and this is the one place
    it is taken apart. Unknown falls back to the turnaround rather than raising,
    which is this whole adapter's contract: it reads settings written by an
    older or newer build, and a form is not the place a bad enum should stop
    the app. The doors refuse what they cannot draw.
    """
    key = str(layout or "turnaround")
    if key in SPRITE_LEGACY_MODES:
        # The legacy kinds name themselves. ``action``/``directions`` are the
        # fields of the *other* mode and are left at their defaults rather than
        # filled with a plausible lie -- a legacy ``walk`` is four frames, and
        # writing ``action="walk"`` beside it would make a request that reads as
        # the eight-frame ``walk4`` to anything consulting SPRITE_FRAME_COUNTS.
        return (key, "idle", 4)
    spec = SPRITE_SHEET_KINDS.get(key)
    if spec is None:
        return ("turnaround", "idle", 4)
    return ("action", spec[0], spec[1])


def sprite_layout_of(sprite: SpriteSettings) -> str:
    """:func:`sprite_from_layout` the other way: the kind a request names."""
    if sprite.mode in SPRITE_LEGACY_MODES:
        return sprite.mode
    return f"{sprite.action}{sprite.directions}"


def request_from_legacy(form: Mapping[str, Any]) -> GenerationRequest:
    generation_type = legacy_asset_type(form)
    projection = str(form.get("projection") or "top_down")
    if projection == "orthogonal":
        projection = "top_down"
    # Every field of the tile document, not just the two that used to be here.
    # ``TileSettings(view=..., target_cell_px=...)`` was the whole of it, which
    # is why ``mode`` has been unreachable from the UI since it was written: the
    # form could carry a mode, a material list and two terrain descriptions and
    # this adapter dropped all four on the floor, so every request arrived as
    # the default collection of nothing.
    tile = TileSettings(
        mode=str(form.get("tile_mode") or "collection"),
        view=projection,
        prompt_items=_prompt_lines(form.get("prompt_items")),
        inner_terrain=str(form.get("inner_terrain") or ""),
        outer_terrain=str(form.get("outer_terrain") or ""),
        boundary=str(form.get("boundary") or ""),
        ground=str(form.get("ground") or ""),
        path=str(form.get("path") or ""),
        edge=str(form.get("edge") or ""),
        variants=int(form.get("variants") or 1),
        terrain_layout=str(form.get("terrain_layout") or "blob47"),
        style_lock=bool(form.get("style_lock")),
        seam_erase=bool(form.get("seam_erase")),
        target_cell_px=_optional_int(form.get("target_cell_px")),
        palette=str(form.get("palette") or ""),
        dither=bool(form.get("dither")),
    )
    mode, action, sprite_directions = sprite_from_layout(form.get("sheet_layout"))
    sprite = SpriteSettings(
        mode=mode,
        action=action,
        directions=sprite_directions,
        candidate_count=_optional_int(form.get("sprite_candidates")),
        target_cell_px=_optional_int(form.get("target_cell_px") or form.get("cell_size")),
        palette=str(form.get("palette") or ""),
        dither=bool(form.get("dither")),
    )
    return GenerationRequest(
        generation_type=generation_type,
        prompt=str(form.get("prompt") or ""),
        negative_prompt=str(form.get("negative_prompt") or ""),
        quality=str(form.get("quality") or "quality"),
        model_mode=str(form.get("model_mode") or "auto"),
        model_override=str(form["model_override"] or form["base_model"])
        if form.get("model_mode") == "advanced"
        and (form.get("model_override") or form.get("base_model"))
        else None,
        style_lora=str(form["style_lora"]) if form.get("style_lora") else None,
        lora_weight=form.get("lora_weight"),
        references=(str(form["ref_path"]),) if form.get("ref_path") else (),
        reference_mode="single" if form.get("ref_path") else "none",
        # Only when there is an image to derive a hint from: the pane clears
        # ``control`` with the reference and refuses the pair separately, so a
        # key left over from a session whose VOLATILE ``ref_path`` did not
        # survive must not read here as a structure request.
        structure_control=str(form.get("control") or "") if form.get("ref_path") else "",
        init_image=bool(form.get("init_image")) and bool(form.get("ref_path")),
        init_strength=(
            float(form["init_strength"])
            if form.get("init_image") and form.get("init_strength") not in (None, "")
            else None
        ),
        seed=int(form.get("seed") or 0),
        count=int(form.get("count") or 1),
        tile=tile,
        sprite=sprite,
    )


def effective_negative_prompt(
    request: GenerationRequest, resolved: ResolvedRecipe | None
) -> str:
    """Return the Avoid text a resolved generation can actually consume."""
    if resolved is None or resolved.recipe.supports_negative_prompt:
        return request.negative_prompt
    return ""


def request_to_legacy(
    request: GenerationRequest, resolved: ResolvedRecipe | None = None
) -> dict[str, Any]:
    """Compatibility adapter for existing ``create_job`` doors.

    ``sprite_sheet`` is ``reference`` and not ``sheet``, which looks like a
    mismatch and is the correction: a sprite sheet **is** an ordinary reference
    job carrying a follow-up request -- the rig checkbox's shape, so the
    character is a row in its own right and a sheet the user hates still leaves
    them the drawing. ``create_job`` takes ``reference``, ``model`` or ``tile``
    and nothing else, so this said ``sheet`` and every structured sprite request
    was refused at ``field="output"`` before it reached the sprite block at all.

    ``tileset`` keeps ``sheet`` because that arm never reaches ``create_job``:
    ``create_generation_request`` sends it to ``create_tile_sheet``, which is
    its own job kind with its own door.
    """
    output = {
        "image": "reference",
        "3d_model": "reference",
        "seamless_material": "tile",
        "tileset": "sheet",
        "sprite_sheet": "reference",
    }[request.generation_type]
    out: dict[str, Any] = {
        "asset_type": request.generation_type,
        "asset_intent": {
            "image": "refine_2d",
            "3d_model": "reconstruct_3d",
            "seamless_material": "refine_2d",
            "tileset": "tileset",
            "sprite_sheet": "sprite",
        }[request.generation_type],
        "output": output,
        "prompt": request.prompt,
        "negative_prompt": effective_negative_prompt(request, resolved) or None,
        "seed": request.seed,
        "count": request.count,
    }
    if request.style_lora:
        out["style_lora"] = request.style_lora
        out["lora_weight"] = request.lora_weight
    if request.references:
        out["references"] = list(request.references)
        if request.init_image:
            out["init_image"] = True
            if request.init_strength is not None:
                out["init_strength"] = float(request.init_strength)
    if request.generation_type == "tileset":
        out["projection"] = request.tile.view
        out["tile_settings"] = asdict(request.tile)
    if request.generation_type == "sprite_sheet":
        out["sprite_settings"] = asdict(request.sprite)
    return out


def _prompt_lines(value: Any) -> tuple[str, ...]:
    """A material list from whatever the form is holding.

    A multi-line text control gives one string with newlines in it and a list
    control gives a list, and both spellings reach here from persisted forms
    of different ages. Blank lines are dropped rather than kept as empty
    materials -- a trailing newline is what a text box has, not a material
    somebody asked for and forgot to describe.
    """
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parts: Iterable[Any] = value.splitlines()
    elif isinstance(value, Iterable):
        parts = value
    else:
        return ()
    return tuple(text for text in (str(part).strip() for part in parts) if text)


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


#: ``{path: (stamp_or_None, rows)}`` -- ``None`` for a cached miss (absent or
#: unreadable), which can never collide with a real stamp. The stamp is
#: ``(st_mtime_ns, st_size)`` rather than the mtime alone: an import followed
#: by a remove rewrites the file twice inside one filesystem timestamp tick
#: (NTFS reports 100 ns, and two atomic renames land inside that on a warm
#: cache), and the second read then served the first write's rows --
#: ``tests/test_loras.py`` failed on exactly that, two runs in three. Every
#: write below also drops the entry outright, so the stamp is the second
#: line of defence, not the first. ``bench/findings.py``
#: verbatim, and for the same reason: ``resolve_recipe`` calls this and the
#: Create pane resolves a recipe *three times a frame*, so an idle Create tab
#: re-read and re-parsed ``manifests.json`` 180 times a second.
_MANIFEST_CACHE: dict[Any, tuple[tuple[int, int] | None, list[LoraManifest]]] = {}


def _forget_manifests(path: Path) -> None:
    """Drop the cached rows for ``path``: called after every rewrite so a
    reader in the same tick cannot be served the previous contents."""
    _MANIFEST_CACHE.pop(path, None)


def load_lora_manifests(config: Any) -> list[LoraManifest]:
    # Total by construction: managed adapters are an optional extra, and
    # ``resolve_recipe`` calls this on every submit. A config that cannot say
    # where the model root is -- a partial one, or a stub -- means "no imported
    # adapters", never a failed generate. The *write* path above still requires
    # the root, because there is nowhere to put the file without it.
    try:
        path = lora_manifest_path(config)
    except (AttributeError, TypeError, ValueError):
        return []
    try:
        st = path.stat()
        mtime: tuple[int, int] | None = (st.st_mtime_ns, st.st_size)
    except OSError:
        _MANIFEST_CACHE[path] = (None, [])
        return []
    cached = _MANIFEST_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        # The same list object, deliberately: every reader here treats it as
        # read-only, and copying it per call would give back the allocation the
        # cache exists to save.
        return cached[1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (AttributeError, TypeError, FileNotFoundError, OSError, ValueError):
        _MANIFEST_CACHE[path] = (mtime, [])
        return []
    rows = raw.get("manifests", raw) if isinstance(raw, Mapping) else []
    out: list[LoraManifest] = []
    for row in rows:
        try:
            out.append(LoraManifest(**row))
        except (TypeError, ValueError):
            continue
    _MANIFEST_CACHE[path] = (mtime, out)
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


def remove_imported_lora(config: Any, key: str) -> bool:
    """Forget an imported adapter: its manifest row, its file, its registry entry.

    Only an *imported* key -- a built-in ``STYLE_LORAS`` entry has no manifest
    and is left alone. The manifest is rewritten first, so a crash between the
    two leaves an orphan file rather than a registered entry with no file.
    """
    manifests = load_lora_manifests(config)
    gone = next((m for m in manifests if m.key == key), None)
    if gone is None:
        return False
    root = Path(config.t2i_model_root) / "loras"
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": 1, "manifests": [asdict(x) for x in manifests if x.key != key]},
        indent=2,
        sort_keys=True,
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=root, delete=False, prefix=".manifests-", suffix=".tmp"
    ) as fh:
        fh.write(payload)
        temp = Path(fh.name)
    temp.replace(lora_manifest_path(config))
    _forget_manifests(lora_manifest_path(config))
    # Resolved and re-checked against the directory before it is deleted, the
    # rule ``service.palettes._path`` and ``fetch.removal_plan`` both follow.
    # ``import_lora`` cannot write a filename with a separator in it, so today
    # this refuses nothing -- but ``manifests.json`` is a file on disk that a
    # user can edit and a restore can replace, and this is a call to ``unlink``.
    # A file outside the directory is left where it is: the manifest entry is
    # already gone, so the style has disappeared from the app either way, and
    # an orphan is a much better outcome than an ``unlink`` somewhere else.
    target = (root / gone.filename).resolve()
    if target.parent == root.resolve():
        target.unlink(missing_ok=True)
    models.STYLE_LORAS.pop(key, None)
    return True


def lora_catalog(config: Any | None = None) -> list[dict[str, Any]]:
    register_imported_loras(config)
    out = [
        {
            "key": x.key,
            "label": x.label,
            "family": x.family,
            "trigger_text": x.trigger,
            "tuned_weight": x.default_weight,
            "license": "",
            "commercial": True,
            "source": "built-in",
            "checksum": "",
        }
        for x in models.STYLE_LORAS.values()
    ]
    if config is not None:
        out.extend(asdict(x) for x in load_lora_manifests(config))
    return out


def import_lora(
    config: Any,
    source: Path | str,
    *,
    label: str,
    family: str = models.FAMILY_SDXL,
    trigger_text: str = "",
    tuned_weight: float = models.DEFAULT_LORA_WEIGHT,
    license: str = "",
    commercial: bool = False,
    source_url: str = "local file",
) -> LoraManifest:
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
    manifest = LoraManifest(
        key,
        label,
        family,
        trigger_text,
        float(tuned_weight),
        license,
        bool(commercial),
        source_url,
        digest,
        filename,
    )
    path = lora_manifest_path(config)
    manifests = [x for x in load_lora_manifests(config) if x.key != key]
    manifests.append(manifest)
    payload = json.dumps(
        {"version": 1, "manifests": [asdict(x) for x in manifests]}, indent=2, sort_keys=True
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=root, delete=False, prefix=".manifests-", suffix=".tmp"
    ) as fh:
        fh.write(payload)
        temp = Path(fh.name)
    temp.replace(path)
    _forget_manifests(path)
    register_imported_loras(config)
    return manifest

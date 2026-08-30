"""Creating, resubmitting, editing and removing jobs -- a facade over five siblings.

This module was 1,446 lines covering five unrelated subjects, and it is now the
front of them: ``_jobs_create``, ``_jobs_list``, ``_jobs_lifecycle``,
``_jobs_resubmit`` and ``_jobs_rework``. Sibling *files* rather than a
``service/jobs/`` package, which is this codebase's naming everywhere else.

**Why a facade and not a rename.** Every caller in the repo -- panes, API,
sweeps and tests alike -- imports this module and calls by attribute
(``svc_jobs.create_job(...)``), and several tests monkeypatch names *on it*:
``create_job``, ``import_mesh``, ``list_jobs``, ``storage_sizes`` and
``MAX_LIST_LIMIT``. Re-exporting keeps every one of those patches landing where
it always did. The two that are read as module globals rather than called --
``MAX_LIST_LIMIT`` in ``list_jobs`` and ``prune_jobs`` -- resolve it back
through this module at call time for exactly that reason, with a comment at
each site.

**The accepted caveat**: intra-package calls bind at import, not through here.
``_jobs_lifecycle.update_job`` calls ``_jobs_list.get_job`` directly, and
``_jobs_resubmit.promote_candidates`` calls ``promote_to_model`` in its own
module -- so a facade patch of *those* would not redirect them. That was true
of the single-module version too (a patched ``jobs.get_job`` never redirected
``update_job``'s internal call either), no test relies on it, and the
alternative is a lazy self-import on every internal call for a redirection
nobody has ever asked for.

The names below that come from ``errors``, ``files`` and ``validation`` are not
part of the split: they have always been importable from here, and a caller
that learned that spelling should not have to learn another.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import verdicts as verdicts_mod  # noqa: F401  -- historically importable
from ._jobs_create import (  # noqa: F401  -- the facade's re-export
    _normalize_guidance,
    _resolve_profile,
    create_job,
    import_mesh,
    import_reference,
    resolve_profile,
)
from ._jobs_lifecycle import (  # noqa: F401  -- the facade's re-export
    _refuse_if_busy,
    cancel_job,
    clean_jobs,
    delete_job,
    dependent_jobs,
    empty_trash,
    prune_jobs,
    restore_job,
    retained_job_ids,
    trash_job,
    trash_size,
    update_job,
    worker_is_inside,
)
from ._jobs_list import (  # noqa: F401  -- the facade's re-export
    get_job,
    list_jobs,
    storage,
    storage_sizes,
)
from ._jobs_resubmit import (  # noqa: F401  -- the facade's re-export
    keep_candidate,
    promote_candidates,
    promote_to_model,
    rerun_job,
)
from ._jobs_rework import (  # noqa: F401  -- the facade's re-export
    optimize_job,
    remesh_job,
    retexture_job,
    stale_rig_artifacts,
    stale_surface_artifacts,
)
from .core import WarlockService  # noqa: F401  -- historically importable
from .errors import (  # noqa: F401  -- historically importable from here
    Conflict,
    Failed,
    Invalid,
    NotFound,
    TooLarge,
    invalid_from,
)
from .files import (  # noqa: F401  -- historically importable from here
    ImageTooLarge,
    attach_files,
    dir_size,
    measure_storage,
    to_png,
)
from .validation import (  # noqa: F401  -- historically importable from here
    ALLOWED_RESOLUTIONS,
    CONDITIONING_PARAMS,
    DERIVED_PARAMS,
    MAX_JOB_NAME,
    MAX_LIST_LIMIT,
    MAX_MESH_BYTES,
    MAX_MESH_CANDIDATES,
    MAX_PROMPT,
    MAX_REFERENCE_COUNT,
    MAX_UPLOAD_BYTES,
    check_glb,
    check_job_id,
    check_seed,
    check_trellis_band,
    check_trellis_tex_res,
    check_vram,
    check_weights,
    normalize_tags,
    not_done_message,
    random_seed,
    valid_template,
)

# ``jobs.log`` has always been importable, and the sibling modules each have
# their own (``warlock.service._jobs_*``). Nothing asserts a logger name.
log = logging.getLogger(__name__)


def create_generation_request(svc: WarlockService, request: Any, **uploads: Any) -> dict[str, Any]:
    """Queue a normalized :class:`warlock.generation.GenerationRequest`.

    The historical ``create_job`` function remains the persistence boundary;
    this adapter translates the new document once and keeps old rows and
    sidecars readable.
    """
    from .. import generation

    if not isinstance(request, generation.GenerationRequest):
        request = generation.GenerationRequest.from_dict(request)
    resolved = generation.resolve_recipe(request, svc.config)
    issues = generation.validate_request(request, resolved)
    if issues:
        raise Invalid(issues[0].message, field=issues[0].field)
    legacy = generation.request_to_legacy(request)
    native_payloads: list[bytes] = []
    if request.references:
        from .files import to_png

        for name in request.references:
            try:
                native_payloads.append(to_png(Path(name).read_bytes()))
            except OSError as exc:
                raise Invalid(
                    f"could not read reference {name!r}: {exc}", field="references"
                ) from exc
            except Exception as exc:
                raise Invalid(f"could not decode reference {name!r}", field="references") from exc
    model_view_payloads: dict[str, bytes] = {}
    if request.generation_type == "3d_model" and request.model.backend == "hunyuan3d_multiview":
        for view_name, view_path in request.model.views.items():
            try:
                model_view_payloads[view_name] = to_png(Path(view_path).read_bytes())
            except OSError as exc:
                raise Invalid(
                    f"could not read {view_name} view {view_path!r}: {exc}",
                    field="model.views",
                ) from exc
            except Exception as exc:
                raise Invalid(f"could not decode {view_name} view", field="model.views") from exc
    reference = uploads.get("reference")
    if reference is None and native_payloads:
        try:
            reference = native_payloads[0]
        except IndexError:
            reference = None
    guidance_fields = {"base_model": resolved.base_model}
    if resolved.style_lora:
        guidance_fields["style_lora"] = resolved.style_lora
    # Keep the two historical follow-up doors intact while making the new
    # request document authoritative.  A sprite starts as an approved
    # reference and then queues its sheet; a tileset uses the dedicated sheet
    # worker, which owns atomic atlas publication and palette quantization.
    if request.generation_type == "tileset":
        from . import tilesheets

        target = request.tile.target_cell_px
        # Refused rather than coerced. This used to read
        # ``target if target in TILE_SIZES else DEFAULT_TILE_SIZE``, which
        # answered "make me 96px tiles" with a sheet of 32px tiles and told
        # nobody -- the request document went on saying 96, so the row, the
        # sidecar and the pane all agreed about a number the pixels did not
        # have. ``tilesheet.geometry`` and ``sheet_subject`` both refuse an
        # unbuildable ask for this reason; so does this.
        if target is None:
            tile_size = tilesheets.DEFAULT_TILE_SIZE
        elif int(target) in tilesheets.TILE_SIZES:
            tile_size = int(target)
        else:
            raise Invalid(
                f"a tile sheet cannot publish {int(target)}px tiles; "
                f"choose one of {list(tilesheets.TILE_SIZES)}",
                field="tile.target_cell_px",
            )
        # The structured request goes through whole. It used to be flattened
        # into one prompt string -- "inner terrain: X; outer terrain: Y; Z" --
        # and handed to a single generation, which is what made the planner
        # decorative: the mode, the material list and the two terrain
        # descriptions were compiled into a sentence and the door then had no
        # way to tell a materials sheet from a terrain set. The sheet-level
        # prompt is the style sentence; the materials are their own lines.
        from ..asset_workflows import TILE_MODE_ALIASES

        tile = request.tile
        # ``validate_request`` has already refused a mode that is not in the
        # table, so the fallback is unreachable rather than a tolerance; it is
        # here because a ``KeyError`` out of a door is not a sentence.
        mode = TILE_MODE_ALIASES.get(tile.mode, tilesheets.DEFAULT_MODE)
        if tile.mode == "path":
            # A path *is* a terrain transition with the two surfaces named
            # differently: the path is what appears as the blob shapes and the
            # ground is what surrounds it, which is exactly ``inner``/``outer``.
            inner, outer, boundary = tile.path, tile.ground, tile.edge
        else:
            inner, outer, boundary = tile.inner_terrain, tile.outer_terrain, tile.boundary
        result = tilesheets.create_tile_sheet(
            svc,
            prompt=request.prompt,
            tile_size=tile_size,
            view=tile.view,
            seed=request.seed,
            negative_prompt=request.negative_prompt or None,
            reference=reference,
            asset_type="tileset",
            asset_intent="tileset",
            mode=mode,
            prompt_items=tile.prompt_items,
            variants=tile.variants,
            inner_terrain=inner,
            outer_terrain=outer,
            boundary=boundary,
            terrain_layout=tile.terrain_layout,
            style_lock=tile.style_lock,
            seam_erase=tile.seam_erase,
            # The two the request document could not name until now. The door
            # has taken both since it grew them, so a tileset submitted through
            # here was refused nothing -- it simply could not ask, while the
            # pane path could, which is the same capability reachable from one
            # door and not the other.
            palette=tile.palette,
            dither=tile.dither,
        )
    else:
        sprite_block = None
        if request.generation_type == "sprite_sheet":
            from . import sprites as svc_sprites

            sprite = request.sprite
            legacy_size = (
                sprite.target_cell_px
                if sprite.target_cell_px in svc_sprites.SPRITE_LOGICAL_SIZES
                else svc_sprites.DEFAULT_SPRITE_LOGICAL_SIZE
            )
            sprite_block = {
                # The request's own action and direction count, not a collapse
                # of them. This read ``"turnaround" if mode == "turnaround" else
                # "walk"``, which answered every one of the seven actions and
                # both direction counts with the legacy four-by-four walk: a
                # request for an eight-direction idle was admitted, queued and
                # published as a sheet nobody asked for, with the request
                # document still saying "idle" beside it. The door composes the
                # kind from the pair (``sprites.resolve_sheet_kind``) and
                # refuses an action with no pose guide behind it.
                **(
                    {"sheet_type": sprite.mode}
                    if sprite.mode in generation.SPRITE_LEGACY_MODES
                    else {"action": sprite.action, "directions": sprite.directions}
                ),
                "candidates": sprite.candidate_count,
                "logical_size": legacy_size,
                "colors": svc_sprites.DEFAULT_SPRITE_COLORS,
                # ``_check_sprite_sheet`` validates these at *this* door through
                # ``sprites._check_options``, so a palette deleted since the
                # request was written costs the request rather than the
                # reference generation plus an hour.
                "palette": sprite.palette,
                "dither": sprite.dither,
            }
        result = create_job(
            svc,
            kind="text",
            prompt=request.prompt,
            output=legacy["output"],
            count=request.count,
            seed=request.seed,
            negative_prompt=request.negative_prompt or None,
            reference=reference,
            guidance_fields=guidance_fields,
            asset_type=request.generation_type,
            asset_intent=legacy["asset_intent"],
            lora_weight=(
                request.lora_weight
                if request.lora_weight is not None
                else next(
                    (
                        row["tuned_weight"]
                        for row in generation.lora_catalog(svc.config)
                        if row["key"] == resolved.style_lora
                    ),
                    None,
                )
            ),
            sprite_sheet=sprite_block,
        )
    # ``guidance.normalize`` intentionally rejects unknown fields, so recipe
    # provenance is merged after the legacy door has normalized its settings.
    recipe_payload = {"version": generation.RECIPE_REGISTRY_VERSION, **resolved.to_dict()}
    for job_id in result.get("ids", [result["id"]]):
        extra = {
            "generation_request": request.to_dict(),
            "resolved_recipe": recipe_payload,
            # These copies make the new contract inspectable by old result
            # views and by rerun tools without requiring them to deserialize
            # the entire request first.
            "quality": request.quality,
            "model_mode": request.model_mode,
            "target_cell_px": (
                request.tile.target_cell_px
                if request.generation_type == "tileset"
                else request.sprite.target_cell_px
                if request.generation_type == "sprite_sheet"
                else None
            ),
        }
        if request.references:
            native_files = []
            for index, data in enumerate(native_payloads):
                native_name = f"native_reference_{index}.png"
                Path(svc.config.job_dir(job_id), native_name).write_bytes(data)
                native_files.append(native_name)
            extra["native_reference_files"] = native_files
        if request.generation_type == "3d_model" and request.model.backend == "hunyuan3d_multiview":
            view_files: dict[str, str] = {}
            for view_name, view_bytes in model_view_payloads.items():
                filename = f"view_{view_name}.png"
                Path(svc.config.job_dir(job_id), filename).write_bytes(view_bytes)
                view_files[view_name] = filename
            extra.update({
                "backend": "hunyuan3d_multiview",
                "texture_mode": request.model.texture_mode,
                "view_assets": view_files,
                "license_acknowledged": True,
            })
        svc.store.merge_params(job_id, extra)
    return result

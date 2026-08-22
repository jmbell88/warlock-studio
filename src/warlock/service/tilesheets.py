"""AI tile sheets: what one may ask for, and queuing the paint.

The door onto ``kind="tile_sheet"``. Modelled on ``sprites.create_sprite_synthesis``
and for its reason: every refusal that can be made from the form belongs here,
before a row exists, because the alternative is a place in the queue and a
minute of GPU spent on a request that was never going to produce a usable
sheet.

**Three fields, and that is deliberate.** A prompt, a tile size and a
projection. The grid is not a control (``tilesheet.COLS``/``ROWS`` are fixed at
the pixel-art LoRA's own art resolution), the palette is not a control (one
shared palette is what makes sixty-four tiles read as one sheet), and the
reference image is optional -- it conditions the style through the IP-Adapter
and is never required, so the common path is a prompt and nothing else.

**The geometry rules are restated, not imported.** ``service/`` may not import
``studio/`` and a pipeline is the wrong place to look up a form's ceiling, so
the tile sizes and the projection list are literals here -- and
``tests/test_tilesheet_service.py`` pins each of them to
``pipelines.tilesheet``, so the copy cannot drift without a red test saying
which one moved.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .. import models
from .core import WarlockService
from .errors import Invalid, TooLarge
from .files import ImageTooLarge, to_png
from .validation import (
    MAX_JOB_NAME,
    MAX_PROMPT,
    MAX_UPLOAD_BYTES,
    check_base_model_weights,
    check_seed,
    check_vram,
    install_remedy,
    random_seed,
)

#: The base a tile sheet is pinned to. Not ``models.DEFAULT_BASE_MODEL``: this
#: path wants full CFG so the negative prompt actually steers, and it has to
#: keep wanting it if the default ever moves. ``sprites.py``'s constant, for
#: ``sprites.py``'s reason.
TILE_SHEET_BASE_MODEL = "sdxl_cfg"

#: The two non-base weights every sheet loads: (registry kind, registry key,
#: the form field a refusal about it names). The grid guide *is* the ControlNet
#: and the art style *is* the LoRA, so a missing one is not a plainer picture,
#: it is the feature not happening.
_TILE_SHEET_REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("control", "canny", "control"),
    ("lora", models.PIXEL_SHEET_LORA, "style_lora"),
)

#: What an *attached reference* additionally needs. Separate from the tuple
#: above because a reference is optional: making the adapter unconditional
#: would refuse the common prompt-only request on a host that has everything
#: the common request actually uses.
_REFERENCE_REQUIRED: tuple[str, str, str] = ("adapter", "plus", "ip_adapter")

#: Every registry row a tile sheet needs on this host, in ``fetch.Entry``'s
#: ``row_key`` spelling -- so a pane can offer "install what this needs"
#: without knowing what a tile sheet is made of.
TILE_SHEET_ROWS: tuple[str, ...] = (
    f"base:{TILE_SHEET_BASE_MODEL}",
    *(f"{kind}:{key}" for kind, key, _field in _TILE_SHEET_REQUIRED),
)

#: With a reference attached. A superset, in check order.
TILE_SHEET_REFERENCE_ROWS: tuple[str, ...] = (
    *TILE_SHEET_ROWS,
    f"{_REFERENCE_REQUIRED[0]}:{_REFERENCE_REQUIRED[1]}",
)

#: ``tilesheet.TILE_SIZES`` and ``tilesheet.VIEWS``, restated.
TILE_SIZES: tuple[int, ...] = (16, 32, 48, 64)
VIEWS: tuple[str, ...] = ("top_down", "three_quarter", "isometric")

#: The one stored spelling that is not in :data:`VIEWS`. Restated here for the
#: same reason the list is: a door may not import the pipeline's constants, and
#: a request carrying the old word has to be accepted rather than refused --
#: rerolling a sheet made last week is not an error.
LEGACY_VIEWS: dict[str, str] = {"orthogonal": "top_down"}

#: One palette across the whole sheet, at the size the ground run measured
#: (``docs/measurements/2026-08-17-ground-reduction.md``: occupancy at 32 was
#: saturated, 64 stayed above 95%). Not a control, for the reason the module
#: docstring gives.
SHEET_COLORS = 64

DEFAULT_TILE_SIZE = 32
DEFAULT_VIEW = "top_down"


def tile_sheet_options() -> dict[str, Any]:
    """What a tile-sheet request may ask for. One source for the form, so the
    pane never hardcodes a ceiling this module enforces.

    The grid summary comes from ``tilesheet.geometry`` rather than being written
    out again here -- ``sprite_options``' rule: the pane says "64 tiles, 8x8"
    under the control, and a second copy of that arithmetic is a label that
    goes stale the first time the grid moves.
    """
    from ..pipelines import tilesheet

    sizes = []
    for size in TILE_SIZES:
        entries = {}
        for view in VIEWS:
            geom = tilesheet.geometry(size, view)
            entries[view] = {
                "tile_w": geom.tile_w,
                "tile_h": geom.tile_h,
                "sheet_w": geom.sheet_size[0],
                "sheet_h": geom.sheet_size[1],
            }
        sizes.append({"key": size, "views": entries})
    reference = tilesheet.geometry(DEFAULT_TILE_SIZE, DEFAULT_VIEW)
    return {
        "tile_sizes": list(TILE_SIZES),
        "views": list(VIEWS),
        # The label the form draws, beside the key it submits. Here rather than
        # in the pane for ``tile_sizes``' reason: the pane hardcoded its two
        # labels and so could not have grown a third without an edit in a file
        # that knows nothing about what a view is.
        "view_labels": {
            "top_down": "Top-down",
            "three_quarter": "3/4",
            "isometric": "Isometric",
        },
        "sizes": sizes,
        "columns": reference.columns,
        "rows": reference.rows,
        "tiles": reference.tiles,
        "colors": SHEET_COLORS,
        "base_model": TILE_SHEET_BASE_MODEL,
        "rows_needed": list(TILE_SHEET_ROWS),
        "reference_rows_needed": list(TILE_SHEET_REFERENCE_ROWS),
        "defaults": {
            "tile_size": DEFAULT_TILE_SIZE,
            "view": DEFAULT_VIEW,
        },
    }


def _check_weights(svc: WarlockService, *, with_reference: bool) -> None:
    """Everything a sheet loads, refused by name with its download line.

    ``validation.check_weights`` stays text-only on purpose -- it is keyed on
    ``params`` a text job wrote -- so this kind brings its own, exactly as
    ``sprites._check_weights`` does.
    """
    from .. import fetch
    from .downloads import needed_keys

    wanted = TILE_SHEET_REFERENCE_ROWS if with_reference else TILE_SHEET_ROWS
    check_base_model_weights(
        svc,
        models.BASE_MODELS[TILE_SHEET_BASE_MODEL],
        rows=needed_keys(svc, wanted),
    )
    required = list(_TILE_SHEET_REQUIRED)
    if with_reference:
        required.append(_REFERENCE_REQUIRED)
    for kindname, key, field in required:
        entry = fetch.find(f"{kindname}:{key}")
        assert entry is not None, f"{kindname}:{key} is not a registry row"
        spec = entry.spec
        if fetch.present(svc.config, kindname, spec):
            continue
        raise Invalid(
            f"A tile sheet needs {spec.label!r}, which is not downloaded. "
            f"{install_remedy(spec.label, fetch.download_text(svc.config, kindname, spec))}",
            field=field,
            # Every row the feature is short of, not merely the one that
            # tripped: a user offered "install what this needs" wants one
            # download, not three refusals in a row.
            rows=needed_keys(svc, wanted),
        )


def create_tile_sheet(
    svc: WarlockService,
    *,
    prompt: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    view: str = DEFAULT_VIEW,
    seed: int | None = None,
    negative_prompt: str | None = None,
    reference: bytes | None = None,
    asset_type: str | None = None,
    asset_intent: str | None = None,
) -> dict[str, Any]:
    """Queue an 8x8 sheet of generated tiles.

    Nothing is written until every check has passed, which is the shape
    ``create_pixel_sheet`` uses: a form that asks for an impossible tile, or
    that asks on a host with no pixel-art LoRA, costs the request rather than a
    minute of GPU and a sheet that merely came back plain.
    """
    from ..pipelines import tilesheet

    text = str(prompt or "").strip()
    if not text:
        raise Invalid("a tile sheet needs a prompt", field="prompt")
    if len(text) > MAX_PROMPT:
        raise Invalid(f"prompt must be at most {MAX_PROMPT} characters", field="prompt")

    # ``None`` means "the form said nothing", and what the absent value means is
    # the pipeline's own default -- an empty *string* is a user explicitly
    # asking for no negative prompt, and is honoured. ``grounds.py``'s rule.
    negative = (
        tilesheet.SHEET_NEGATIVE_PROMPT
        if negative_prompt is None
        else str(negative_prompt)
    )
    if len(negative) > MAX_PROMPT:
        raise Invalid(
            f"negative_prompt must be at most {MAX_PROMPT} characters",
            field="negative_prompt",
        )

    # Coerced inside a guard: these are form values, and a bare ``int()`` over
    # one turns a typo into an unhandled TypeError/ValueError rather than the
    # refusal-with-a-field this module exists to produce.
    try:
        size = int(tile_size)
    except (TypeError, ValueError):
        raise Invalid("tile_size must be a whole number", field="tile_size") from None
    if size not in TILE_SIZES:
        raise Invalid(f"tile_size must be one of {list(TILE_SIZES)}", field="tile_size")
    # ``field="projection"`` and not ``"view"``: the *form field* is still
    # called that, and a refusal has to name the control the user is looking at.
    # The vocabulary moved; the field key did not.
    resolved = LEGACY_VIEWS.get(str(view), str(view))
    if resolved not in VIEWS:
        raise Invalid(f"view must be one of {list(VIEWS)}", field="projection")
    expected_asset_type = {
        "top_down": "tileset_top_down",
        "three_quarter": "tileset_three_quarter",
        "isometric": "tileset_isometric",
    }[resolved]
    if asset_type is not None and asset_type != expected_asset_type:
        raise Invalid("asset_type does not match the tile-set view", field="asset_type")
    if asset_intent is not None and asset_intent != "tileset":
        raise Invalid("a tile set must use asset_intent='tileset'", field="asset_intent")
    if seed is not None:
        check_seed("seed", seed)

    # The geometry is derived once, here, the block's single writer -- so the
    # worker and anything reading the row later share one stored fact rather
    # than each re-deriving the table. Raises nothing at this point: every
    # input has just been checked against the same two lists the pipeline
    # validates, and the test suite pins the pair together.
    geom = tilesheet.geometry(size, resolved)

    # The sheet's identity is the pixel-art LoRA on a base that can take it, so
    # a mismatch is refused here rather than queued -- the worker would drop the
    # style and paint sixty-four photographs of gravel. Written as a guard on a
    # pair of constants deliberately, for ``create_pixel_sheet``'s reason: params
    # outlive today's UI, and the day the base becomes a parameter this refusal
    # is already at the door.
    base = models.BASE_MODELS[TILE_SHEET_BASE_MODEL]
    pixel_lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    if not models.lora_fits(base, pixel_lora):
        fitting = sorted(
            key
            for key, loras in models.loras_by_base().items()
            if models.PIXEL_SHEET_LORA in loras
        )
        raise Invalid(
            f"base_model {base.key!r} is {base.family} and the pixel-art LoRA "
            f"{pixel_lora.key!r} is fitted to {pixel_lora.family}; "
            f"pick one of {fitting}",
            field="base_model",
        )

    params: dict[str, Any] = {
        "seed": random_seed() if seed is None else int(seed),
        "base_model": TILE_SHEET_BASE_MODEL,
        "style_lora": models.PIXEL_SHEET_LORA,
        "control": "canny",
        "colors": SHEET_COLORS,
        "negative_prompt": negative,
        # One nested block rather than six loose keys, because it is a
        # *document description* rather than a settings vector: anything
        # reading the sheet back compares it whole against the file on disk,
        # and VECTOR_PARAMS deliberately does not carry it.
        "sheet": {
            # 2: the view vocabulary widened. The key below is still
            # ``projection`` and ``orthogonal`` still reads, so a version-1
            # block opens unchanged -- the number says which words to expect,
            # not whether the block can be read.
            "version": 2,
            "tile_w": geom.tile_w,
            "tile_h": geom.tile_h,
            # The key stays ``projection`` -- see ``tilesheet.sheet_sidecar``.
            "projection": geom.view,
            "columns": geom.columns,
            "rows": geom.rows,
        },
    }
    if asset_type in {
        "tileset_top_down", "tileset_three_quarter", "tileset_isometric"
    }:
        params["asset_type"] = asset_type
    if asset_intent == "tileset":
        params["asset_intent"] = asset_intent
    if reference is not None:
        # Only alongside the image it scales, so an unused adapter never
        # reaches params as a live setting -- ``settings_2d.submit_kwargs``'
        # rule for ip_scale, applied at the door.
        params["ip_adapter"] = "plus"

    # Presence as well as fit, and at the door: the family check above only asks
    # whether these weights belong together, not whether either is on this host.
    # Missing, the worker's own tolerance takes over -- it logs and paints bare
    # -- so the job would finish, look like one flat picture, and write a
    # sidecar naming a LoRA that never loaded.
    _check_weights(svc, with_reference=reference is not None)
    check_vram(svc, "tile_sheet", "tilesheet", params)

    normalized_ref: bytes | None = None
    if reference is not None:
        if len(reference) > MAX_UPLOAD_BYTES:
            raise TooLarge("reference upload is over 20 MB")
        try:
            normalized_ref = to_png(reference)
        except ImageTooLarge as exc:
            raise Invalid(str(exc), field="reference") from exc
        except Exception as exc:
            raise Invalid("could not decode uploaded reference", field="reference") from exc

    # The directory before the row, and the directory removed again if the row
    # write raises: ``next_queued`` can otherwise claim the job in the gap and
    # find no ``ref.png`` on disk. ``_jobs_create``'s invariant, and the reason
    # that module keeps its three doors together.
    new_id = uuid.uuid4().hex[:12]
    job_dir: Path | None = None
    try:
        if normalized_ref is not None:
            job_dir = svc.config.job_dir(new_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "ref.png").write_bytes(normalized_ref)
        svc.store.create("tile_sheet", text, params, new_id, stage="tilesheet")
    except Exception:
        if job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    # Named at creation rather than left to the library's fallback: a tile sheet
    # is not a subject, and a row called "mossy dungeon" among a list of assets
    # reads as a mesh somebody generated.
    # Truncated like every other auto-derived name (_jobs_create's two): the
    # prompt is capped at MAX_PROMPT (1000), far past what a name column holds.
    svc.store.set_meta(new_id, name=f"{text} tile sheet"[:MAX_JOB_NAME])
    svc.wake_worker()
    return {
        "id": new_id,
        "tiles": geom.tiles,
        "tile_w": geom.tile_w,
        "tile_h": geom.tile_h,
    }

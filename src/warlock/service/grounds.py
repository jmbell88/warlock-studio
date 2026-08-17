"""AI-painted ground sets: what one may ask for, and queuing the paint.

The door onto ``kind="ground_set"``. Modelled on ``sheets.create_pixel_sheet``
and for its reason: every refusal that can be made from the form belongs here,
before a row exists, because the alternative is a place in the queue and up to
sixteen SDXL generations spent on a request that was never going to produce a
usable atlas.

**The geometry rules are restated, not imported.** ``service/`` may not import
``studio/``, so the terrain count, the tile edge and the projection list are
literals here -- and ``tests/test_ground_service.py`` pins each of them to
``tilegen.MAX_TERRAINS`` / ``tilegen.MAX_TILE_EDGE`` / ``project.PROJECTIONS``,
so the copy cannot drift without a red test saying which one moved.
"""

from __future__ import annotations

import uuid
from typing import Any

from .. import models
from .core import WarlockService
from .errors import Invalid
from .validation import (
    MAX_PROMPT,
    check_base_model_weights,
    check_seed,
    check_vram,
    install_remedy,
    random_seed,
)

#: The base an AI ground set is pinned to. Not ``models.DEFAULT_BASE_MODEL``:
#: this path wants full CFG *and* the seamless-tile circular padding, and it has
#: to keep wanting them if the default ever moves. ``sheets.py``'s constant, for
#: ``sheets.py``'s reason.
GROUND_BASE_MODEL = "sdxl_cfg"

#: Every registry row a ground set needs on this host, in ``fetch.Entry``'s
#: ``row_key`` spelling -- so a pane can offer "install what this needs" without
#: knowing what a ground set is made of.
GROUND_ROWS: tuple[str, ...] = (
    f"base:{GROUND_BASE_MODEL}",
    f"lora:{models.PIXEL_SHEET_LORA}",
)

#: ``tilegen.MAX_TERRAINS`` and ``tilegen.MAX_TILE_EDGE``, restated. Two
#: textures per terrain, so eight terrains is sixteen generations -- which is
#: also the ceiling on how long one of these jobs can hold the card.
MAX_TERRAINS = 8
MIN_TILE_EDGE = 4
MAX_TILE_EDGE = 512

#: ``project.PROJECTIONS``, restated. The compositor draws a rectangle or a
#: diamond; an oblique map uses the rectangle, exactly as the flat generator
#: does.
PROJECTIONS: tuple[str, ...] = ("orthogonal", "isometric", "oblique")

#: What the palette select offers. ``sheets.PIXEL_COLOR_CHOICES``' shape, and
#: the same argument: one palette across the whole sheet is what makes N
#: terrains read as one tileset, so the count is a property of the set.
COLOR_CHOICES: tuple[int, ...] = (8, 16, 32, 64, 128)

#: How long a terrain's material description may be. Short on purpose: these are
#: *subjects* that ``guidance.compose_prompt`` wraps in the tile template, and a
#: paragraph here is a paragraph CLIP truncates.
MAX_MATERIAL = 120


def ground_options() -> dict[str, Any]:
    """What a ground-set request may ask for. One source for the form, so the
    pane never hardcodes a ceiling this module enforces."""
    return {
        "max_terrains": MAX_TERRAINS,
        "tile_edge_range": [MIN_TILE_EDGE, MAX_TILE_EDGE],
        "projections": list(PROJECTIONS),
        "colors": list(COLOR_CHOICES),
        "base_model": GROUND_BASE_MODEL,
        "rows": list(GROUND_ROWS),
        # 64, not 32: the measured occupancy at 32 was saturated for even a
        # two-terrain set (docs/measurements/2026-08-17-ground-reduction.md).
        "defaults": {"colors": 64, "border": 0},
    }


def _check_terrains(terrains: Any) -> list[dict[str, Any]]:
    """The terrain rows, validated and normalised into what params will hold.

    The colour comes in as the row's RGBA fill, which is also the palette hint
    the prompts are built from -- one value doing two jobs deliberately, since a
    separate "hint" control would be a second colour a user could set to
    something the map does not show.
    """
    rows = list(terrains or [])
    if not rows:
        raise Invalid("a ground set needs at least one terrain", field="terrains")
    if len(rows) > MAX_TERRAINS:
        raise Invalid(
            f"a ground set holds at most {MAX_TERRAINS} terrains", field="terrains"
        )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        entry = dict(row or {})
        name = str(entry.get("name") or "").strip()
        if not name:
            raise Invalid(f"terrain {index + 1} needs a name", field="terrains")
        if name.lower() in seen:
            raise Invalid(f"two terrains cannot both be called {name!r}", field="terrains")
        seen.add(name.lower())
        material = str(entry.get("material") or "").strip()
        edge = str(entry.get("edge") or "").strip()
        for label, text in (("material", material), ("edge", edge)):
            if len(text) > MAX_MATERIAL:
                raise Invalid(
                    f"terrain {index + 1}'s {label} must be at most "
                    f"{MAX_MATERIAL} characters",
                    field="terrains",
                )
        # Coerced inside the guard, not outside it: ``int()`` over a form value
        # raises ValueError on "green" and TypeError on a dict, and either one
        # would leave this door -- which exists to turn every bad request into a
        # sentence with a field on it -- propagating a bare builtin the pane
        # renders as an unhandled crash instead of a red field.
        try:
            colour = tuple(int(part) for part in (entry.get("color") or (128, 128, 128, 255)))
        except (TypeError, ValueError):
            raise Invalid(
                f"terrain {index + 1}'s colour must be four channels of 0..255",
                field="terrains",
            ) from None
        if len(colour) != 4 or any(part < 0 or part > 255 for part in colour):
            raise Invalid(
                f"terrain {index + 1}'s colour must be four channels of 0..255",
                field="terrains",
            )
        out.append(
            {"name": name, "material": material, "edge": edge, "color": list(colour)}
        )
    return out


def create_ground_set(
    svc: WarlockService,
    *,
    theme: str,
    terrains: Any,
    tile_w: int,
    tile_h: int,
    projection: str = "orthogonal",
    seed: int | None = None,
    colors: int = 64,
    border: int | None = None,
    negative_prompt: str | None = None,
) -> dict[str, Any]:
    """Queue an AI-painted forty-seven-case ground set.

    Nothing is written until every check has passed, which is the shape
    ``create_pixel_sheet`` uses: a form that asks for nine terrains, or that
    asks on a host with no pixel-art LoRA, costs the request rather than a
    minute of GPU and an atlas that merely got smaller.
    """
    from .. import fetch
    from .downloads import needed_keys

    text = str(theme or "").strip()
    if not text:
        raise Invalid("a ground set needs a theme", field="theme")
    if len(text) > MAX_PROMPT:
        raise Invalid(f"theme must be at most {MAX_PROMPT} characters", field="theme")
    # ``None`` means "the form said nothing", and what the absent value means
    # is the pipeline's own default -- an empty *string* is a user explicitly
    # asking for no negative prompt, and is honoured. Function-local import in
    # the allowed direction (service -> pipelines), ``sheets.py``'s precedent.
    from ..pipelines import ground

    negative = (
        ground.GROUND_NEGATIVE_PROMPT
        if negative_prompt is None
        else str(negative_prompt)
    )
    if len(negative) > MAX_PROMPT:
        raise Invalid(
            f"negative_prompt must be at most {MAX_PROMPT} characters",
            field="negative_prompt",
        )

    rows = _check_terrains(terrains)

    # Every numeric coercion below is guarded for ``_check_terrains``' reason:
    # these are form values, and a bare ``int()`` over one turns a typo into an
    # unhandled TypeError/ValueError rather than the refusal-with-a-field this
    # module exists to produce.
    def _whole(label: str, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            raise Invalid(f"{label} must be a whole number", field=label) from None

    width = _whole("tile_w", tile_w)
    height = _whole("tile_h", tile_h)
    for label, value in (("tile_w", width), ("tile_h", height)):
        if value < MIN_TILE_EDGE or value > MAX_TILE_EDGE:
            raise Invalid(
                f"{label} must be between {MIN_TILE_EDGE} and {MAX_TILE_EDGE} pixels",
                field=label,
            )
    if str(projection) not in PROJECTIONS:
        raise Invalid(
            f"projection must be one of {list(PROJECTIONS)}", field="projection"
        )
    if _whole("colors", colors) not in COLOR_CHOICES:
        raise Invalid(f"colors must be one of {list(COLOR_CHOICES)}", field="colors")

    # Zero, like ``None``, means "whatever suits this tile" -- the compositor
    # fills it in. Anything else has to leave some fill visible, or every one of
    # the forty-seven cases comes back looking alike.
    band = 0 if border is None else _whole("border", border)
    shorter = min(width, height)
    # Sign first. A negative band reaches the range check as a *false* negative
    # -- ``2 * -5 >= 32`` is False -- so it used to slip past and be caught by
    # the line below anyway; correct, but only by accident, and it read as
    # though the range message could fire for a negative value.
    if band < 0:
        raise Invalid("a border cannot be negative", field="border")
    if band and 2 * band >= shorter:
        raise Invalid(
            f"a {width}x{height} tile takes a border of 1..{(shorter - 1) // 2} pixels",
            field="border",
        )
    if seed is not None:
        check_seed("seed", seed)

    # The set's identity is the pixel-art LoRA on a base that can take it and
    # can tile, so a mismatch is refused here rather than queued -- the worker
    # would drop the style and paint sixteen photographs of gravel. Written as a
    # guard on a pair of constants deliberately, for ``create_pixel_sheet``'s
    # reason: params outlive today's UI, and the day the base becomes a
    # parameter this refusal is already at the door.
    base_key = GROUND_BASE_MODEL
    base = models.BASE_MODELS[base_key]
    if base_key not in models.tile_bases():
        # Seamlessness is circular padding over ``Conv2d``, which is an SDXL
        # fact rather than a general one -- a texture generated on a DiT looks
        # seamless in a thumbnail and seams in a painted field.
        raise Invalid(
            f"base_model {base.key!r} cannot generate a seamless tile; "
            f"pick one of {models.tile_bases()}",
            field="base_model",
        )
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
        "base_model": base_key,
        "style_lora": models.PIXEL_SHEET_LORA,
        "colors": int(colors),
        "negative_prompt": negative,
        # One nested block rather than eight loose keys, because it is a
        # *document description* rather than a settings vector: the adoption
        # step compares it whole against the map's geometry, and VECTOR_PARAMS
        # deliberately does not carry it.
        "ground": {
            "version": 1,
            "tile_w": width,
            "tile_h": height,
            "projection": str(projection),
            "border": band,
            # Computed here, the block's single writer, so the worker and the
            # adoption step read one stored fact rather than each re-deriving
            # the table. (This "version" is the params schema the adoption
            # reads, not ``ground.GROUND_VERSION`` -- the sidecar's number.)
            "phases": ground.variant_factor(width, height, str(projection)),
            "terrains": rows,
        },
    }

    # Presence as well as fit, and at the door: the family check above only asks
    # whether these weights belong together, not whether either is on this host.
    # Missing, the worker's own tolerance takes over -- it logs and paints bare
    # -- so the job would finish, look like sixteen ordinary textures, and write
    # a sidecar naming a LoRA that never loaded.
    check_base_model_weights(svc, base, rows=needed_keys(svc, GROUND_ROWS))
    if not fetch.present(svc.config, "lora", pixel_lora):
        remedy = fetch.download_text(svc.config, "lora", pixel_lora)
        raise Invalid(
            f"A ground set needs {pixel_lora.label!r}, which is not downloaded. "
            f"{install_remedy(pixel_lora.label, remedy)}",
            field="base_model",
            rows=needed_keys(svc, GROUND_ROWS),
        )
    check_vram(svc, "ground_set", "ground", params)

    new_id = svc.store.create(
        "ground_set", text, params, uuid.uuid4().hex[:12], stage="ground"
    )
    # Named at creation rather than left to the library's fallback: a ground set
    # is not a subject, and a row called "mossy dungeon" among a list of assets
    # reads as a mesh somebody generated.
    svc.store.set_meta(new_id, name=f"{text} ground set")
    svc.wake_worker()
    return {"id": new_id, "terrains": len(rows), "textures": 2 * len(rows)}

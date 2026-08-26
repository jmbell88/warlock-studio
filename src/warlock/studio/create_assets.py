"""The user-facing asset types in Studio's Create form.

The generation services still have three older switches (``output``,
``sheet_type`` and ``projection``).  They describe implementation doors, not
what somebody is trying to make.  This registry is the single translation
between the flat Create choice and those doors, and also gives finished jobs a
stable intent from which their next action can be chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssetType:
    key: str
    label: str
    intent: str
    create_label: str
    output: str
    sheet_type: str = "tile"
    projection: str = "top_down"
    sheet_layout: str = "turnaround"


_ORDERED = (
    AssetType("image_2d", "2D Image", "refine_2d", "Create image", "reference"),
    AssetType("model_3d", "3D Model", "reconstruct_3d", "Generate reference", "reference"),
    AssetType("seamless_tile", "Seamless Tile", "refine_2d", "Create tile", "tile"),
    AssetType(
        "tileset_top_down", "Top-Down Tileset", "tileset", "Create tile set",
        "sheet", projection="top_down",
    ),
    AssetType(
        "tileset_three_quarter", "3/4 Tileset", "tileset", "Create tile set",
        "sheet", projection="three_quarter",
    ),
    AssetType(
        "tileset_isometric", "Isometric Tileset", "tileset", "Create tile set",
        "sheet", projection="isometric",
    ),
    AssetType(
        "sprite_turnaround", "Sprite Turnaround", "sprite",
        "Draw character and make sheet", "sheet", sheet_type="sprite",
        sheet_layout="turnaround",
    ),
    AssetType(
        "sprite_walk", "Sprite Walk Cycle", "sprite",
        "Draw character and make sheet", "sheet", sheet_type="sprite",
        sheet_layout="walk",
    ),
)

ASSET_TYPES: dict[str, AssetType] = {item.key: item for item in _ORDERED}
ASSET_TYPE_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (item.key, item.label) for item in _ORDERED
)
DEFAULT_ASSET_TYPE = "model_3d"


def legacy_asset_type(form: Any) -> str:
    """Translate the pre-registry output fields without losing their intent."""
    if not isinstance(form, dict):
        return DEFAULT_ASSET_TYPE
    output = form.get("output")
    if output == "tile":
        return "seamless_tile"
    if output == "sheet":
        if form.get("sheet_type") == "sprite":
            return "sprite_walk" if form.get("sheet_layout") == "walk" else "sprite_turnaround"
        view = str(form.get("projection") or "top_down")
        if view == "orthogonal":
            view = "top_down"
        return {
            "top_down": "tileset_top_down",
            "three_quarter": "tileset_three_quarter",
            "isometric": "tileset_isometric",
        }.get(view, "tileset_top_down")
    # Old "Object" meant a reconstruction reference, not a standalone image.
    return "model_3d"


def selected(form: Any) -> AssetType:
    key = form.get("asset_type") if isinstance(form, dict) else None
    return ASSET_TYPES.get(str(key), ASSET_TYPES[DEFAULT_ASSET_TYPE])


def sync_legacy_fields(form: dict[str, Any]) -> AssetType:
    """Make the old service-door fields agree with authoritative asset_type.

    **The five fields written here are derived, not editable.** This runs from
    ``settings_2d._asset_type`` on *every frame*, so anything it writes is
    rewritten before the next draw -- which is correct while ``asset_type`` is
    the only control the user touches, and becomes a field nobody can type into
    the moment a widget is pointed at one of them. The pane's old per-type
    field groups that wrote ``sheet_type``, ``projection`` and ``count`` were
    unreachable from ``draw`` and are gone; wiring a control back onto one of
    these fields means deciding which of the two owns it first.
    """
    spec = selected(form)
    form["asset_type"] = spec.key
    form["output"] = spec.output
    form["sheet_type"] = spec.sheet_type
    form["projection"] = spec.projection
    form["sheet_layout"] = spec.sheet_layout
    if spec.output == "sheet":
        # A sheet is one operation.  In particular, prevent a count restored
        # from an ordinary image request from multiplying a sprite chain.
        form["count"] = 1
    return spec


def persisted_intent(form: Any) -> dict[str, str]:
    """The stable, deliberately small identity written into job params."""
    spec = selected(form)
    return {"asset_type": spec.key, "asset_intent": spec.intent}


def asset_type_from_params(params: Any, *, stage: str = "") -> str:
    """Read today's identity, with a conservative answer for legacy jobs."""
    if isinstance(params, dict) and "asset_type" in params:
        key = str(params.get("asset_type") or "")
        return key if key in ASSET_TYPES else DEFAULT_ASSET_TYPE
    if stage == "tile":
        return "seamless_tile"
    if isinstance(params, dict) and params.get("sprite_sheet"):
        layout = (params.get("sprite_sheet") or {}).get("sheet_type")
        return "sprite_walk" if layout == "walk" else "sprite_turnaround"
    if isinstance(params, dict) and params.get("sheet"):
        projection = (params.get("sheet") or {}).get("projection", "top_down")
        return legacy_asset_type({"output": "sheet", "projection": projection})
    return ""

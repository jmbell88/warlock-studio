"""Troupe's left-top pane: the cast, and which sheet of one you are watching.

A character is a mesh with at least one finished character sheet, which is a
narrower question than the library's "what have I made" -- see
``troupe_mode.characters``. The sheets under a selected character are listed
because a character can have several: the same mesh at 32 px and at 64, or a
second attempt at a palette.
"""

from __future__ import annotations

from typing import Any

from .. import controls, troupe_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = troupe_mode.ensure(ctx)
    widgets.section("Cast")
    manual_render.help_button(ctx, "troupe-characters")

    cast = troupe_mode.characters(ctx)
    if not cast:
        widgets.muted_wrapped(
            "No character sheets yet. Describe one below; you will approve the "
            "T-pose drawing in Create before anything is reconstructed."
        )
        return

    for character in cast:
        label = character["prompt"].strip() or character["id"]
        selected = character["id"] == state.job_id
        if controls.selectable_row(
            f"cast-{character['id']}",
            label,
            selected=selected,
            tooltip=character["id"],
        ):
            troupe_mode.select(ctx, character["id"])

    if not state.job_id:
        return
    records = troupe_mode.sheets(ctx, state.job_id)
    if len(records) < 2:
        # One sheet needs no chooser, and drawing a list of one is a control
        # that says "there is a choice here" when there is not.
        return
    imgui.dummy((0, 6))
    widgets.section("Sheets")
    for record in records:
        size = int(record.get("frame_size") or 0)
        name = str(record.get("name") or "").strip()
        label = f"{name or 'sheet'} - {size}px"
        selected = record["id"] == state.sheet_id
        if controls.selectable_row(
            f"sheet-{record['id']}", label, selected=selected
        ):
            troupe_mode.select(ctx, state.job_id, record["id"])

"""Troupe's right-bottom pane: the ways out.

Every one of them is an existing bridge rather than a new writer. A character
sheet is an ordinary sheet plus an ``animation`` block, so Inker's grid slicer,
Packwright's sheet adder and the sheet exporters all already read it -- and a
second path would be a second dialect of one format, which is the mistake
``sheet.sidecar``'s docstring spends a paragraph refusing.
"""

from __future__ import annotations

from typing import Any

from .. import troupe_mode, verbs, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = troupe_mode.ensure(ctx)
    widgets.section("Take it somewhere")
    manual_render.help_button(ctx, "troupe-bridge")

    ready = bool(state.job_id and state.sheet_id)
    if widgets.disabled_button(
        verbs.open_in("inker"),
        ready,
        (-1, 0),
        reason="Pick a character sheet first.",
        tooltip="Opens the sheet sliced on its own grid, with one tag per "
        "animation and direction. It opens unlinked: the first Ctrl+S is a "
        "Save As, so cleaning up frames cannot overwrite the render they came "
        "from.",
    ):
        troupe_mode.open_in_inker(ctx)
    imgui.dummy((0, 4))
    if widgets.disabled_button(
        verbs.add_to("packwright"),
        ready,
        (-1, 0),
        reason="Pick a character sheet first.",
        tooltip="One sprite per cell, packed beside everything else in the "
        "atlas.",
    ):
        troupe_mode.add_to_packwright(ctx)

    imgui.dummy((0, 8))
    widgets.muted_wrapped(
        "The sheet and its sidecar are already on disk beside the mesh. The "
        "Library's export list is where the files themselves are."
    )

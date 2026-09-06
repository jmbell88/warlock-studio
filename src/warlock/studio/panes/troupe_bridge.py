"""Troupe's right-bottom pane: the ways out.

Every one of them is an existing bridge rather than a new writer. A character
sheet is an ordinary sheet plus an ``animation`` block, so Inker's grid slicer,
Packwright's sheet adder and the sheet exporters all already read it -- and a
second path would be a second dialect of one format, which is the mistake
``sheet.sidecar``'s docstring spends a paragraph refusing.

Three of them now, and the third is the only one that produces *files*: Inker
and Packwright hand the sheet to another mode, and Export package copies the
PNG and its JSON out together for an engine. Together, because either one alone
is an asset nothing can interpret -- ``service.characters.export_package`` puts
that promise on ``export.staged_copy_all``.
"""

from __future__ import annotations

from typing import Any

from .. import tokens, troupe_mode, verbs, widgets
from ..manual import render as manual_render
from ..tokens import sp


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
    imgui.dummy((0, sp(tokens.SP_1)))
    if widgets.disabled_button(
        verbs.add_to("packwright"),
        ready,
        (-1, 0),
        reason="Pick a character sheet first.",
        tooltip="One sprite per cell, packed beside everything else in the "
        "atlas.",
    ):
        troupe_mode.add_to_packwright(ctx)
    imgui.dummy((0, sp(tokens.SP_1)))
    # **The third way out, and the only one that produces files.** The two
    # above hand the sheet to another mode; this one is for the engine, and it
    # copies the *pair* -- the PNG is the atlas and the JSON is what says which
    # cell is ``walk`` facing south-east, so a folder with one and not the
    # other holds an asset nothing can interpret.
    busy = ctx.busy(troupe_mode.export_key(state.job_id, state.sheet_id))
    if widgets.disabled_button(
        "Export package...",
        ready and not busy,
        (-1, 0),
        reason=(
            "That sheet is already being exported."
            if busy
            else "Pick a character sheet first."
        ),
        tooltip="Copies the PNG and its JSON sidecar together -- the pair an "
        "engine imports. Asks where to put them unless an export folder is "
        "configured.",
    ):
        troupe_mode.export_package(ctx)

    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.muted_wrapped(
        "The sheet and its sidecar are already on disk beside the mesh. The "
        "Library's export list is where the files themselves are."
    )

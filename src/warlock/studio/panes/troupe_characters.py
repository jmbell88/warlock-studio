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
    pending = troupe_mode.in_progress(ctx)
    if not cast:
        # Two registers, because "you have made none" and "the one you just
        # made is still on its way" are different facts and the second one used
        # to be told as the first -- a user who described a character here and
        # came back was informed that nothing had been started.
        widgets.muted_wrapped(
            "Nothing to play yet -- your first character is still on its way."
            if pending
            # No pose named: the form below is where one gets picked, and this
            # is drawn before it has been. It promised a T-pose for its whole
            # life, which stopped being true when A-pose became the default.
            else "No character sheets yet. Describe one below; you will approve "
            "the reference drawing in Create before anything is reconstructed."
        )
        _pending(ctx, pending)
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
        _row_menu(ctx, state, character)

    _pending(ctx, pending)

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


def _row_menu(ctx: Any, state: Any, character: dict[str, Any]) -> None:
    """A cast row's own verbs (B2).

    Four idioms across the app was never the defect -- *none* was. Troupe's
    rows were the one list in the tree with no way to act on a member without
    first selecting it and then finding a button somewhere else, and every
    action here is one that already exists behind that route.
    """
    from imgui_bundle import imgui

    if not imgui.begin_popup_context_item(f"cast-menu-{character['id']}"):
        return
    widgets.popup_chrome(_imgui=imgui)
    is_active = character["id"] == state.job_id
    if controls.menu_item_simple("Select"):
        troupe_mode.select(ctx, character["id"])
    imgui.separator()
    if controls.menu_item_simple(
        "Open the sheet in Inker",
        "",
        False,
        is_active,
        reason="Select this character first.",
    ):
        troupe_mode.open_in_inker(ctx)
    if controls.menu_item_simple(
        "Add the sheet to Packwright",
        "",
        False,
        is_active,
        reason="Select this character first.",
    ):
        troupe_mode.add_to_packwright(ctx)
    imgui.end_popup()


def _pending(ctx: Any, pending: list[dict[str, Any]]) -> None:
    """The characters that are on their way, and whose turn it is.

    **Not a ``selectable_row``**, and that is the honesty rather than the
    styling: selecting one would point the mode at a character with nothing to
    play, which is the empty preview this whole section exists to explain. A
    muted line cannot be mistaken for a member of the cast because it cannot be
    picked.
    """
    from imgui_bundle import imgui

    if not pending:
        return
    imgui.dummy((0, 6))
    widgets.section("In progress")
    for item in pending:
        widgets.muted_wrapped(str(item.get("prompt") or item["id"]))
        widgets.muted(str(item.get("phase") or ""))
        if item.get("waiting") and widgets.disabled_button(
            f"Approve in Create##troupe-approve-{item['id']}", True, (-1, 0)
        ):
            # The gate lives in Create by design -- a second promote button
            # here would be a second gate -- so this is a way *to* it, not
            # another one of it.
            from .. import create_stages
            from ..state import set_mode

            set_mode(ctx.state, "create")
            create_stages.go(ctx, "reference", select=item["id"])
        imgui.dummy((0, 4))

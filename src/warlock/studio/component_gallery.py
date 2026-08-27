"""Developer-only visual catalogue for the studio control contract."""

from __future__ import annotations

import os

from imgui_bundle import imgui

from . import controls, icons, widgets
from .tokens import sp

ENV_KEY = "WARLOCK_DEV_COMPONENTS"
POPUP = "Component gallery##developer"
_requested = False


def enabled(environ: dict[str, str] | None = None) -> bool:
    """Whether the developer surface is exposed in this process."""

    source = os.environ if environ is None else environ
    return str(source.get(ENV_KEY, "")).strip().lower() in {"1", "true", "yes", "on"}


def request() -> None:
    global _requested
    _requested = True


def _state_samples() -> None:
    widgets.section("Interaction states")
    states = tuple(controls.ControlState)
    if imgui.begin_table("gallery/states", 2):
        for state in states:
            imgui.table_next_column()
            widgets.muted(state.value.capitalize())
            imgui.table_next_column()
            controls.button(
                f"Button##gallery-state-{state.value}",
                role=controls.ButtonRole.SECONDARY,
                preview=state,
                reason="This sample is disabled.",
            )
        imgui.end_table()


def _buttons() -> None:
    widgets.section("Button roles")
    for index, role in enumerate(controls.ButtonRole):
        if index:
            imgui.same_line()
        controls.button(
            f"{role.value.capitalize()}##gallery-role-{role.value}",
            role=role,
            tooltip=f"{role.value.capitalize()} button",
        )
    widgets.muted("Regular 30 dp")
    controls.button("Regular##gallery-regular")
    imgui.same_line()
    controls.button(
        "Compact##gallery-compact", control_size=controls.ControlSize.COMPACT
    )


def _fields() -> None:
    widgets.section("Inputs and validation")
    imgui.set_next_item_width(sp(220))
    controls.input_text("##gallery-text", "Default")
    imgui.same_line()
    imgui.set_next_item_width(sp(220))
    controls.input_text(
        "##gallery-error", "Needs attention", error="Example validation message"
    )
    imgui.set_next_item_width(sp(220))
    controls.input_float("##gallery-number", 1.25, 0.0, 0.0, "%.2f")
    imgui.same_line()
    imgui.set_next_item_width(sp(220))
    controls.slider_float("##gallery-slider", 0.4, 0.0, 1.0, "%.0f%%")
    changed, _value = controls.combo(
        "##gallery-menu", "one", (("one", "One"), ("two", "Two"))
    )
    del changed


def _choices() -> None:
    widgets.section("Choices and selection")
    controls.switch("Immediate setting", True, control_id="gallery-switch")
    controls.checkbox("Included in export", True)
    controls.segmented_choice(
        "gallery-segments",
        (("one", "One"), ("two", "Two"), ("three", "Three")),
        "two",
    )
    controls.selectable_row("gallery-row-a", "Unselected row")
    controls.selectable_row("gallery-row-b", "Selected row", selected=True)


def draw() -> None:
    """Draw the gallery popup when requested by the developer build."""

    global _requested
    if not enabled():
        _requested = False
        return
    if _requested:
        imgui.open_popup(POPUP)
        _requested = False
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(
        viewport.get_center(), imgui.Cond_.appearing.value, (0.5, 0.5)
    )
    # Floored. The inset is subtracted unconditionally, so on a viewport
    # narrower than the margin itself the requested size goes *negative* and
    # imgui is asked for a window of impossible extent. 240 design px is small
    # enough to fit anything that can host a window at all and large enough to
    # still be a gallery rather than a sliver.
    inset = sp(64)
    floor = sp(240)
    imgui.set_next_window_size(
        (
            max(floor, min(sp(760), viewport.work_size.x - inset)),
            max(floor, min(sp(640), viewport.work_size.y - inset)),
        ),
        imgui.Cond_.appearing.value,
    )
    if not imgui.begin_popup(POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    widgets.pane_header(
        "Component gallery",
        help_text="Developer preview of shared control states.",
        actions=(("close", f"{icons.X} Close", imgui.close_current_popup),),
    )
    widgets.muted("Shared controls in the current theme and UI scale.")
    if imgui.begin_child("gallery/scroll", (0, 0)):
        # The gallery exists to show shared controls as a pane draws them, so it
        # asks for the blocks a pane gets -- a gallery that showed the headings
        # flat would be showing a look no pane has.
        with widgets.section_blocks():
            _state_samples()
            _buttons()
            _fields()
            _choices()
    imgui.end_child()
    imgui.end_popup()

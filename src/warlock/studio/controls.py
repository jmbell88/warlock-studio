"""The only presentational ImGui control layer used by studio panes.

The renderer still owns invisible canvas and drag targets.  Everything a user
recognises as a button, field, choice, row, or menu item comes through here so
height, roles, state paint, focus, disabled explanations, and tooltips cannot
drift pane by pane.

The low-level functions intentionally retain ImGui's return shapes.  That makes
them a safe migration boundary for old panes; the semantic helpers (``switch``,
``segmented_choice`` and ``selectable_row``) are the preferred API for new UI.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from imgui_bundle import imgui

from . import fonts, motion, theme, tokens
from .tokens import sp


class ButtonRole(StrEnum):
    """Visual hierarchy for an action."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    GHOST = "ghost"
    DESTRUCTIVE = "destructive"
    ICON = "icon"


class ControlSize(StrEnum):
    """The two supported control-density steps."""

    REGULAR = "regular"
    COMPACT = "compact"

    @property
    def design_height(self) -> float:
        return (
            tokens.CONTROL_HEIGHT_REGULAR
            if self is ControlSize.REGULAR
            else tokens.CONTROL_HEIGHT_COMPACT
        )


class ControlState(StrEnum):
    """Named states used by the developer gallery and visual smoke tests."""

    DEFAULT = "default"
    HOVER = "hover"
    PRESSED = "pressed"
    FOCUSED = "focused"
    SELECTED = "selected"
    DISABLED = "disabled"
    ERROR = "error"


DOCUMENT_ACTION_ORDER = ("New", "Open", "Save", "Save As", "Export")


def control_height(size: ControlSize | str = ControlSize.REGULAR) -> float:
    """Physical control height for ``size``; pure apart from DPI scale."""

    try:
        choice = size if isinstance(size, ControlSize) else ControlSize(size)
    except ValueError:
        choice = ControlSize.REGULAR
    return sp(choice.design_height)


def _role(value: ButtonRole | str) -> ButtonRole:
    try:
        return value if isinstance(value, ButtonRole) else ButtonRole(value)
    except ValueError:
        return ButtonRole.SECONDARY


def _vec(colour: int, alpha: float = 1.0) -> Any:
    return imgui.ImVec4(*theme.rgba(colour, alpha))


def _item_rect() -> tuple[Any, Any] | None:
    try:
        return imgui.get_item_rect_min(), imgui.get_item_rect_max()
    except (AttributeError, RuntimeError):
        return None


def _ring(colour: int, *, width: float, inset: float = 0.0) -> None:
    rect = _item_rect()
    if rect is None:
        return
    low, high = rect
    try:
        draw = imgui.get_window_draw_list()
        draw.add_rect(
            (low.x - inset, low.y - inset),
            (high.x + inset, high.y + inset),
            imgui.get_color_u32(theme.rgba(colour)),
            sp(tokens.RADIUS_M),
            sp(width),
            0,
        )
    except (AttributeError, RuntimeError):
        # Headless source/unit tests deliberately supply tiny ImGui stubs.
        return


def _leading_selection() -> None:
    rect = _item_rect()
    if rect is None:
        return
    low, high = rect
    try:
        x = low.x + sp(1)
        imgui.get_window_draw_list().add_line(
            (x, low.y + sp(2)),
            (x, high.y - sp(2)),
            imgui.get_color_u32(theme.rgba(theme.ACCENT)),
            sp(tokens.SELECTION_BOUNDARY_WIDTH),
        )
    except (AttributeError, RuntimeError):
        return


def _finish_item(
    *,
    tooltip: str = "",
    reason: str = "",
    enabled: bool = True,
    error: bool = False,
    selected: bool = False,
    force_focus: bool = False,
) -> None:
    """Apply the state treatment shared by every control."""

    try:
        hovered = imgui.is_item_hovered(
            imgui.HoveredFlags_.allow_when_disabled.value
        )
    except (AttributeError, RuntimeError, TypeError):
        try:
            hovered = imgui.is_item_hovered()
        except (AttributeError, RuntimeError):
            hovered = False
    note = reason if not enabled and reason else tooltip
    if hovered and note:
        imgui.set_tooltip(note)
    try:
        focused = imgui.is_item_focused()
    except (AttributeError, RuntimeError):
        focused = False
    if error:
        _ring(theme.ERR, width=tokens.FOCUS_RING_WIDTH)
    if selected:
        _ring(theme.ACCENT, width=tokens.SELECTION_BOUNDARY_WIDTH)
    # Focus is last so it remains distinct from validation and selection.
    if focused or force_focus:
        _ring(theme.ACCENT, width=tokens.FOCUS_RING_WIDTH, inset=sp(1))


@contextmanager
def _disabled(enabled: bool) -> Iterator[None]:
    if not enabled:
        imgui.begin_disabled()
    try:
        yield
    finally:
        if not enabled:
            imgui.end_disabled()


@contextmanager
def _button_colours(
    role: ButtonRole, selected: bool, preview: ControlState
) -> Iterator[None]:
    c = imgui.Col_
    if role is ButtonRole.PRIMARY:
        normal = theme.rgba(theme.ACCENT)
        hovered = theme.mix(theme.ACCENT, theme.TEXT, tokens.HOVER_WASH_ALPHA)
        active = theme.mix(theme.ACCENT, theme.BG, tokens.PRESSED_WASH_ALPHA)
    elif role is ButtonRole.DESTRUCTIVE:
        normal = theme.rgba(theme.ERR, 0.90)
        hovered = theme.rgba(theme.ERR)
        active = theme.mix(theme.ERR, theme.BG, tokens.PRESSED_WASH_ALPHA)
    elif role in (ButtonRole.GHOST, ButtonRole.ICON):
        normal = theme.rgba(theme.ACCENT, tokens.SELECTION_WASH_ALPHA) if selected else (0, 0, 0, 0)
        hovered = theme.rgba(theme.ELEV_2)
        active = theme.rgba(theme.ACCENT, tokens.PRESSED_WASH_ALPHA)
    else:
        normal = (
            theme.rgba(theme.ACCENT, tokens.SELECTION_WASH_ALPHA)
            if selected
            else theme.rgba(theme.ELEV_1)
        )
        hovered = theme.rgba(theme.ELEV_2)
        active = theme.rgba(theme.ACCENT, tokens.PRESSED_WASH_ALPHA)
    if preview is ControlState.HOVER:
        normal = hovered
    elif preview is ControlState.PRESSED:
        normal = active
    imgui.push_style_color(c.button.value, imgui.ImVec4(*normal))
    imgui.push_style_color(c.button_hovered.value, imgui.ImVec4(*hovered))
    imgui.push_style_color(c.button_active.value, imgui.ImVec4(*active))
    try:
        yield
    finally:
        imgui.pop_style_color(3)


def button(
    label: str,
    size: tuple[float, float] = (0, 0),
    *,
    role: ButtonRole | str = ButtonRole.SECONDARY,
    control_size: ControlSize | str = ControlSize.REGULAR,
    selected: bool = False,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
    _imgui: Any = None,
    preview: ControlState | str = ControlState.DEFAULT,
) -> bool:
    """Draw a role-aware button with one shared interaction contract."""

    # Interaction tests replace a canvas module's ImGui object with a minimal
    # mouse/popup harness.  Keeping that injection explicit lets renderer panes
    # use the shared API without making the global control backend mutable.
    if _imgui is not None and _imgui is not imgui:
        return bool(_imgui.button(label, size))
    try:
        visual = preview if isinstance(preview, ControlState) else ControlState(preview)
    except ValueError:
        visual = ControlState.DEFAULT
    selected = selected or visual is ControlState.SELECTED
    enabled = enabled and visual is not ControlState.DISABLED
    width, height = size
    height = height if height > 0 else control_height(control_size)
    resolved = _role(role)
    compact = control_size == ControlSize.COMPACT or control_size == ControlSize.COMPACT.value
    if compact:
        padding = imgui.get_style().frame_padding
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding.value, (padding.x, sp(6))
        )
    with _disabled(enabled), _button_colours(resolved, selected, visual), fonts.label(imgui):
        clicked = imgui.button(label, (width, height))
    if compact:
        imgui.pop_style_var()
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        error=visual is ControlState.ERROR,
        selected=selected,
        force_focus=visual is ControlState.FOCUSED,
    )
    return bool(clicked and enabled)


def small_button(
    label: str,
    *,
    role: ButtonRole | str = ButtonRole.GHOST,
    selected: bool = False,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> bool:
    """Compact button; unlike ImGui's small button it keeps a real hit box."""

    return button(
        label,
        role=role,
        control_size=ControlSize.COMPACT,
        selected=selected,
        enabled=enabled,
        reason=reason,
        tooltip=tooltip,
    )


@contextmanager
def _field_colours(error: bool) -> Iterator[None]:
    if not error:
        yield
        return
    c = imgui.Col_
    imgui.push_style_color(
        c.frame_bg.value, _vec(theme.ERR, tokens.ERROR_WASH_ALPHA)
    )
    imgui.push_style_color(c.frame_bg_hovered.value, _vec(theme.ERR, 0.18))
    imgui.push_style_color(c.frame_bg_active.value, _vec(theme.ERR, 0.22))
    try:
        yield
    finally:
        imgui.pop_style_color(3)


def _field_call(
    name: str,
    *args: Any,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
    error: str | bool = False,
    **kwargs: Any,
) -> Any:
    backend = kwargs.pop("_imgui", None)
    if backend is not None and backend is not imgui:
        return getattr(backend, name)(*args, **kwargs)
    with _disabled(enabled), _field_colours(bool(error)):
        result = getattr(imgui, name)(*args, **kwargs)
    _finish_item(
        tooltip=tooltip,
        reason=reason or (str(error) if isinstance(error, str) else ""),
        enabled=enabled,
        error=bool(error),
    )
    return result


def input_text(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_text", *args, **kwargs)


def input_text_with_hint(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_text_with_hint", *args, **kwargs)


def input_text_multiline(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_text_multiline", *args, **kwargs)


def input_int(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_int", *args, **kwargs)


def input_float(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_float", *args, **kwargs)


def input_float2(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_float2", *args, **kwargs)


def input_float3(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_float3", *args, **kwargs)


def input_float4(*args: Any, **kwargs: Any) -> Any:
    return _field_call("input_float4", *args, **kwargs)


def color_edit4(*args: Any, **kwargs: Any) -> Any:
    """Four-channel colour field with the shared field-state treatment."""

    return _field_call("color_edit4", *args, **kwargs)


def color_edit3(*args: Any, **kwargs: Any) -> Any:
    """Three-channel colour field with the shared field-state treatment."""

    return _field_call("color_edit3", *args, **kwargs)


def drag_int(*args: Any, **kwargs: Any) -> Any:
    return _field_call("drag_int", *args, **kwargs)


def drag_float(*args: Any, **kwargs: Any) -> Any:
    return _field_call("drag_float", *args, **kwargs)


def slider_int(*args: Any, **kwargs: Any) -> Any:
    return _field_call("slider_int", *args, **kwargs)


def slider_float(*args: Any, **kwargs: Any) -> Any:
    return _field_call("slider_float", *args, **kwargs)


def checkbox(
    label: str,
    value: bool,
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
    error: str | bool = False,
    _imgui: Any = None,
) -> tuple[bool, bool]:
    return _field_call(
        "checkbox",
        label,
        value,
        enabled=enabled,
        reason=reason,
        tooltip=tooltip,
        error=error,
        _imgui=_imgui,
    )


def radio_button(
    label: str,
    active: bool,
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> bool:
    with _disabled(enabled):
        clicked = imgui.radio_button(label, active)
    _finish_item(tooltip=tooltip, reason=reason, enabled=enabled, selected=active)
    return bool(clicked and enabled)


def selectable(
    label: str,
    selected: bool = False,
    flags: int = 0,
    size: tuple[float, float] = (0, 0),
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> Any:
    """ImGui-compatible selectable with the studio row-selection language."""

    c = imgui.Col_
    imgui.push_style_color(
        c.header.value,
        _vec(theme.ACCENT, tokens.SELECTION_WASH_ALPHA) if selected else _vec(theme.PANEL, 0.0),
    )
    imgui.push_style_color(c.header_hovered.value, _vec(theme.ELEV_2))
    imgui.push_style_color(c.header_active.value, _vec(theme.ACCENT, tokens.PRESSED_WASH_ALPHA))
    with _disabled(enabled):
        result = imgui.selectable(label, selected, flags, size)
    imgui.pop_style_color(3)
    if selected:
        _leading_selection()
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        selected=False,
    )
    return result


def selectable_row(
    row_id: str,
    label: str,
    *,
    selected: bool = False,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
    size: tuple[float, float] = (0, 0),
) -> bool:
    result = selectable(
        f"{label}##{row_id}",
        selected,
        0,
        size,
        enabled=enabled,
        reason=reason,
        tooltip=tooltip,
    )
    return bool(result[0] if isinstance(result, tuple) else result)


def collapsing_header(
    label: str,
    *args: Any,
    selected: bool = False,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
    **kwargs: Any,
) -> Any:
    with _disabled(enabled):
        result = imgui.collapsing_header(label, *args, **kwargs)
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        selected=selected,
    )
    return result


def menu_item(
    label: str,
    shortcut: str = "",
    selected: bool = False,
    enabled: bool = True,
    *,
    reason: str = "",
    tooltip: str = "",
) -> Any:
    result = imgui.menu_item(label, shortcut, selected, enabled)
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        selected=selected,
    )
    return result


def menu_item_simple(
    label: str,
    shortcut: str = "",
    selected: bool = False,
    enabled: bool = True,
    *,
    reason: str = "",
    tooltip: str = "",
) -> bool:
    """The binding's Boolean convenience shape, under the same state contract."""

    clicked = imgui.menu_item_simple(label, shortcut, selected, enabled)
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        selected=selected,
    )
    return bool(clicked and enabled)


def combo(
    control_id: str,
    current: str,
    options: Sequence[tuple[str, str]],
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
    error: str | bool = False,
) -> tuple[bool, str]:
    """A complete menu-backed choice control."""

    labels = dict(options)
    changed = False
    picked = current
    with _disabled(enabled), _field_colours(bool(error)):
        opened = imgui.begin_combo(control_id, labels.get(current, current))
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        error=bool(error),
    )
    if opened:
        for key, label in options:
            hit = selectable_row(
                f"{control_id}/{key}", label, selected=key == current
            )
            if hit:
                picked, changed = key, key != current
        imgui.end_combo()
    return changed, picked


def switch(
    label: str,
    value: bool,
    *,
    control_id: str | None = None,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> tuple[bool, bool]:
    """Immediate Boolean setting rendered as an animated switch."""

    key = control_id or label
    track_h = sp(18)
    track_w = sp(32)
    height = control_height(ControlSize.COMPACT)
    origin = imgui.get_cursor_screen_pos()
    top = origin.y + (height - track_h) * 0.5
    label_w = imgui.calc_text_size(label).x if label else 0.0
    with _disabled(enabled):
        clicked = imgui.invisible_button(
            f"##switch/{key}",
            (track_w + (sp(6) + label_w if label else 0), height),
        )
    if clicked and enabled:
        value = not value
    t = motion.value(
        f"switch/{key}", 1.0 if value else 0.0, duration=tokens.DUR_FAST
    )
    draw = imgui.get_window_draw_list()
    off, on = theme.rgba(theme.EDGE), theme.rgba(theme.ACCENT)
    fill = tuple(off[i] + (on[i] - off[i]) * t for i in range(3)) + (1.0,)
    draw.add_rect_filled(
        (origin.x, top),
        (origin.x + track_w, top + track_h),
        imgui.get_color_u32(fill),
        track_h * 0.5,
    )
    radius = track_h * 0.5 - sp(2)
    knob_x = origin.x + track_h * 0.5 + (track_w - track_h) * t
    draw.add_circle_filled(
        (knob_x, top + track_h * 0.5),
        radius,
        imgui.get_color_u32(theme.rgba(0xFFFFFF)),
        24,
    )
    if label:
        draw.add_text(
            (
                origin.x + track_w + sp(6),
                origin.y + (height - imgui.get_text_line_height()) * 0.5,
            ),
            imgui.get_color_u32(theme.rgba(theme.TEXT)),
            label,
        )
    _finish_item(tooltip=tooltip, reason=reason, enabled=enabled, selected=value)
    return bool(clicked and enabled), value


def segmented_choice(
    control_id: str,
    options: Sequence[tuple[str, str]],
    current: str,
    *,
    enabled: bool = True,
    reason: str = "",
    tooltips: dict[str, str] | None = None,
    compact: bool = False,
) -> tuple[bool, str]:
    """Short mutually-exclusive choice rendered as a selected pill group."""

    picked = current
    changed = False
    size = ControlSize.COMPACT if compact else ControlSize.REGULAR
    for index, (key, label) in enumerate(options):
        if index:
            imgui.same_line()
        if button(
            f"{label}##{control_id}/{key}",
            role=ButtonRole.GHOST,
            control_size=size,
            selected=key == current,
            enabled=enabled,
            reason=reason,
            tooltip=(tooltips or {}).get(key, ""),
        ):
            picked, changed = key, key != current
    return changed, picked

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

import contextlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from imgui_bundle import imgui

from . import fonts, motion, probe, theme, tokens
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
        # The ring has to trace the *button*, and buttons round at
        # ``frame_rounding`` (RADIUS_S, half RADIUS_M) -- drawing at RADIUS_M
        # bulged every selection and focus ring away from its fill at all four
        # corners. Read it off the live style rather than the token, because a
        # caller may have pushed its own rounding before this runs.
        draw.add_rect(
            (low.x - inset, low.y - inset),
            (high.x + inset, high.y + inset),
            imgui.get_color_u32(theme.rgba(colour)),
            imgui.get_style().frame_rounding + inset,
            sp(width),
            0,
        )
    except (AttributeError, RuntimeError):
        # Headless source/unit tests deliberately supply tiny ImGui stubs.
        return


def selection_ring() -> None:
    """Draw the shared selection boundary around the item just submitted.

    The public door onto ``_ring`` for the one caller that cannot go through
    :func:`button`: ``widgets._glyph_button`` draws its own square and pushes
    its own colours (the glyph-centring fix), so it takes the treatment rather
    than the drawing code. One spelling of "this control is the one in hand"
    is the whole point -- a hand-rolled ``push_style_color`` is invisible to
    ``probe`` and drifts from this by a pixel per site.
    """
    _ring(theme.ACCENT, width=tokens.SELECTION_BOUNDARY_WIDTH)


def leading_selection(rect: tuple[Any, Any] | None = None) -> None:
    """Draw the accent bar down the left edge of a selected row.

    Public for :func:`selection_ring`'s reason and by the same argument:
    ``widgets.list_row`` builds its own row surface over an invisible button,
    so it cannot go through :func:`selectable`, and "this row is the one in
    hand" must have one spelling. A hand-rolled ``add_line`` in a pane is
    invisible to ``probe`` and drifts from this by a pixel per site.

    ``rect`` defaults to the item just submitted, which is what
    :func:`selectable` wants. ``list_row`` passes its own, because by the time
    it can draw this the last submitted item is the caller's *content* -- the
    name, or a trailing percentage -- and the bar would land beside that rather
    than down the row's edge.
    """
    rect = _item_rect() if rect is None else rect
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


#: The private name kept as an alias: ``selectable`` calls it and there is no
#: reason to touch that call site to rename a function.
_leading_selection = leading_selection


#: What every slider and drag adds to its tooltip. Ctrl+click-to-type is
#: imgui's own default and has always worked here; nothing said so.
TYPED_ENTRY_HINT = "Ctrl+click to type a value."


def _finish_item(
    *,
    tooltip: str = "",
    reason: str = "",
    enabled: bool = True,
    error: bool = False,
    selected: bool = False,
    force_focus: bool = False,
    label: str = "",
    kind: str = "",
    trailing_label: bool = False,
) -> None:
    """Apply the state treatment shared by every control.

    ``label`` and ``kind`` are for :mod:`.probe` alone -- this is the one
    chokepoint every studio control passes through, so a census taken here is
    *derived* rather than hand-kept, and cannot fall behind the UI. The call is
    first because it reads the item rect, and the state treatment below
    submits draw commands rather than items; ordering is belt and braces
    either way. Unset ``WARLOCK_UI_PROBE`` and it is one attribute lookup.
    """

    if probe.ENABLED:
        probe.record(
            label=label,
            kind=kind,
            enabled=enabled,
            reason=reason,
            selected=selected,
            tooltip=tooltip,
            trailing_label=trailing_label,
        )
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
    if enabled and kind.startswith(("slider_", "drag_")):
        # Every slider says so, in one place. A rule that has to be remembered
        # at each of 166 call sites is one that will be forgotten at the next
        # -- the argument ``widgets.combo``'s tooltip-as-accessible-name makes.
        note = f"{note}\n{TYPED_ENTRY_HINT}" if note else TYPED_ENTRY_HINT
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
        label=label,
        kind="button",
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


#: Where ``flags`` sits in the positional signature of the two field families
#: that take :class:`imgui.SliderFlags_`, by name prefix.
#:
#: **Prefix-restricted deliberately.** ``input_int``'s sixth positional is an
#: ``InputTextFlags``, not a ``SliderFlags`` -- injecting a slider flag there
#: would silently switch on an unrelated text-field option. Sliders and drags
#: are the two families whose typed entry can land outside the range they draw.
_FLAGS_ARITY = {"slider_": 5, "drag_": 6}


def _clamp_typed_entry(name: str, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Add ``clamp_on_input`` to a slider/drag that has not spelled flags itself.

    Ctrl+click-to-type is imgui's own default and every one of this app's
    fields already has it -- what none of them had is a *bound* on what gets
    typed, so 9999 into a brush-size slider set an out-of-range brush. Done
    here rather than at 166 call sites for the reason the census is taken
    here: this is the one place every field passes through.

    **``clamp_on_input``, never ``always_clamp``.** ``always_clamp`` includes
    ``clamp_zero_range``, which clamps a drag whose ``v_min == v_max`` -- and
    that is exactly how ``settings_3d`` spells *unbounded*
    (``SIZE_NO_BOUND = (0.0, 0.0)``). ``always_clamp`` there would pin the
    asset's size to zero.
    """
    prefix = next((p for p in _FLAGS_ARITY if name.startswith(p)), None)
    if prefix is None or "flags" in kwargs:
        return args, kwargs
    index = _FLAGS_ARITY[prefix]
    if len(args) > index:
        return args, kwargs
    # Headless stubs supply the handful of functions a test drives and no
    # enums; a missing flag is a missing clamp, not a crash.
    with contextlib.suppress(AttributeError):
        kwargs["flags"] = imgui.SliderFlags_.clamp_on_input.value
    return args, kwargs


def _field_call(
    name: str,
    *args: Any,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
    error: str | bool = False,
    commit: bool = False,
    **kwargs: Any,
) -> Any:
    """One of imgui's fields, with this app's disabled/colour/tooltip treatment.

    ``commit`` changes *when* the field reports a change: off (the default) it
    is imgui's own answer, true on every keystroke, which is what a field
    driving live UI state wants. On, it is true only on the frame the field is
    left after an edit -- for the fields whose writes are **undoable**, where
    per-keystroke is not responsiveness but undo-stack spam: typing "120" into
    a frame's duration pushed one step per character, so a single Ctrl+Z took
    it back to "12" rather than to what it was.

    Opt-in rather than the default because most fields here are the first kind:
    onion-skin depth, the preview frame rate, an export wrap count. Gating those
    would make them feel broken to buy nothing, since none of them writes
    history.
    """
    backend = kwargs.pop("_imgui", None)
    if backend is not None and backend is not imgui:
        return getattr(backend, name)(*args, **kwargs)
    # After the escape hatch above, so a test stub still gets the signature it
    # was written against.
    args, kwargs = _clamp_typed_entry(name, args, kwargs)
    with _disabled(enabled), _field_colours(bool(error)):
        result = getattr(imgui, name)(*args, **kwargs)
    if commit:
        # Read here, while the field is still imgui's "last item" -- before
        # ``_finish_item`` below draws a tooltip or a marker over the answer.
        settled = imgui.is_item_deactivated_after_edit()
        if isinstance(result, tuple) and result:
            result = (settled, *result[1:])
    _finish_item(
        tooltip=tooltip,
        reason=reason or (str(error) if isinstance(error, str) else ""),
        enabled=enabled,
        error=bool(error),
        # imgui draws every field's label *beside* it and groups the two, so
        # the item rect runs past the widget onto the text. See ``probe.hit``.
        trailing_label=True,
        # The first positional is the label for every field imgui has; a call
        # that passed something else would name itself oddly in the census
        # rather than break.
        label=args[0] if args and isinstance(args[0], str) else "",
        kind=name,
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


# --- one gesture, one undo step ------------------------------------------------

#: The gesture a slider drag has open: ``(history, mark)``. One slot rather
#: than one per door because imgui has one active item at a time, so a second
#: drag cannot begin before the first has let go.
_gesture: tuple[Any, int] | None = None


def fold_undo(history: Any) -> None:
    """Fold a drag on the field just drawn into one undo step.

    Call it **immediately after** the field and **before** acting on its
    result. A slider reports a change on every frame the pointer moves, and a
    door that pushes a step per report turns one second of dragging into sixty
    steps -- past ``UNDO_MAX_DEPTH`` that evicts every earlier edit in the
    document. ``UndoStack.mark`` and ``collapse_since`` are the mechanism; this
    is the item-state plumbing around them, so a door reads as three lines:
    draw, fold, act.

    Before the act and not after: on the frame a drag begins imgui reports the
    activation *and* the first change together, and a mark taken after that
    first push would leave the first step outside the fold. Deactivation is
    the frame *after* the last active one, where nothing changes, so the order
    does not matter there.

    ``history`` may be ``None`` for a field whose value is session state and
    pushes nothing; the call is then a no-op, which lets a pane draw one kind
    of slider for a palette slot and a free colour alike. The only drag this
    cannot fold is one that ends in a *different* item's scope -- a popup
    colour picker's own sliders, say, which are their own items in their own
    window; those fold per component, not per drag.
    """
    global _gesture
    if imgui.is_item_activated():
        # A previous gesture still open means its item vanished mid-drag (a
        # pane closed, a tab switched); close it so the deferred eviction runs.
        close_gesture()
        if history is not None:
            _gesture = (history, history.mark())
    elif _gesture is not None and imgui.is_item_deactivated():
        close_gesture()


def close_gesture() -> None:
    """Close whatever ``fold_undo`` has open. Safe when nothing is."""
    global _gesture
    if _gesture is None:
        return
    history, mark = _gesture
    _gesture = None
    history.collapse_since(mark)


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
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        selected=active,
        label=label,
        kind="radio_button",
        trailing_label=True,
    )
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
        label=label,
        kind="selectable",
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
        label=label,
        kind="collapsing_header",
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
        label=label,
        kind="menu_item",
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
        label=label,
        kind="menu_item_simple",
    )
    return bool(clicked and enabled)


@contextmanager
def menu(
    label: str,
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> Iterator[bool]:
    """A submenu header. -> whether its body should be drawn.

    ``imgui.begin_menu`` is not in the AST guard's ``FORBIDDEN`` set -- which
    is how one pane came to call it directly -- but a greyed menu with no
    explanation is exactly the failure the disabled-reason contract exists to
    stop, and a menu header is the *one* row a user cannot hover a child of to
    find out why. So it is wrapped like every other row, and the ``end_menu``
    pairing becomes the context manager's problem rather than the caller's.
    """

    with _disabled(enabled):
        opened = bool(imgui.begin_menu(label, enabled))
    _finish_item(
        tooltip=tooltip, reason=reason, enabled=enabled, label=label, kind="menu"
    )
    try:
        yield opened
    finally:
        if opened:
            imgui.end_menu()


def menu_bar_id(bar_id: str, label: str) -> str:
    """The popup name a :func:`menu_button` opens. Pure, so tests can name it."""

    return f"{bar_id}/menu/{label}"


def menu_button(
    bar_id: str,
    label: str,
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> str:
    """One header of a workspace menu strip. -> the popup name to render.

    Not ``imgui.begin_menu_bar``: that belongs to a *window* via a flag, and
    the centre pane's flags are owned by :func:`layout.pane`. Changing the
    child's content-region arithmetic is what the canvas's height reservation
    depends on, and a negative child height silently kills the canvas. A strip
    of ghost buttons that each open an anchored popup is the same picture with
    none of that risk, and it collapses into :mod:`.toolbar`'s overflow for
    free.

    The name is returned rather than the popup opened, because a popup only
    renders inside the id stack of the window that opened it: the caller draws
    it in the same place it drew the button.
    """

    name = menu_bar_id(bar_id, label)
    hit = button(
        f"{label}##{bar_id}/menubutton/{label}",
        role=ButtonRole.GHOST,
        control_size=ControlSize.COMPACT,
        enabled=enabled,
        reason=reason,
        tooltip=tooltip,
    )
    if hit and enabled:
        imgui.open_popup(name)
    return name


@contextmanager
def menu_popup(name: str) -> Iterator[bool]:
    """The body of a menu opened by :func:`menu_button`. -> whether it is open."""

    opened = bool(imgui.begin_popup(name))
    if opened:
        from . import widgets

        widgets.popup_chrome(_imgui=imgui)
    try:
        yield opened
    finally:
        if opened:
            imgui.end_popup()


def menu_separator() -> None:
    """The rule between two groups of menu rows."""

    imgui.separator()


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
        label=control_id,
        kind="combo",
        trailing_label=True,
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
        imgui.get_color_u32(theme.rgba(theme.KNOB)),
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
    _finish_item(
        tooltip=tooltip,
        reason=reason,
        enabled=enabled,
        selected=value,
        label=label or key,
        kind="switch",
    )
    return bool(clicked and enabled), value


def segmented_flags(
    control_id: str,
    options: Sequence[tuple[str, str]],
    active: Any,
    *,
    enabled: bool = True,
    reason: str = "",
    tooltips: dict[str, str] | None = None,
    compact: bool = False,
    width: float = 0.0,
) -> str:
    """:func:`segmented_choice`'s multi-select sibling. -> the key hit, or "".

    The same pill group, and that is the point rather than a convenience: a set
    of toggles that *compose* -- Inker's four symmetry mirrors, where H and V
    together give four dabs -- drawn as four loose buttons reads as four
    unrelated things, which is exactly what made them hard to find. Drawn as
    one group they read as one control with several switches in it, which is
    what they are.

    Multi-select, so it returns the key pressed rather than a new value: the
    caller owns how the set composes (``brush.toggled`` here), and a control
    that guessed would be a second, quieter copy of that rule.
    """

    hit = ""
    size = ControlSize.COMPACT if compact else ControlSize.REGULAR
    for index, (key, label) in enumerate(options):
        if index:
            # **Contiguous, with no spacing between the segments.** That is what
            # a segmented control *is*: four buttons with the row's ordinary gap
            # between them read as four separate things, which is the complaint
            # Inker's symmetry mirrors existed as. It also gives the group back
            # three gaps of width, which is what lets it stay on the bar beside
            # a five-field tool instead of collapsing at the default size.
            imgui.same_line(0.0, 0.0)
        if button(
            f"{label}##{control_id}/{key}",
            (width, width) if width else (0, 0),
            role=ButtonRole.GHOST,
            control_size=size,
            selected=key in active,
            enabled=enabled,
            reason=reason,
            tooltip=(tooltips or {}).get(key, ""),
        ):
            hit = key
    return hit


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

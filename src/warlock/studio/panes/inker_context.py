"""The row under the tabs: what the tool in your hand is set to.

Aseprite's context bar, and the reason the toolbox can be a 90 px rail: the
options that were nine ``labeled_slider``s and five ``labeled_combo``s down the
left column -- two rows each, a label line and a full-width control, ~812 px of
sidebar -- are one 38 px row above the canvas, beside the drawing they are
about.

**Which widgets appear is a table**, ``inker_state.CONTEXT_WIDGETS``, not a
chain of ifs here: that is what lets a test assert both directions, and in
particular that *every* key in ``TOOL_OPTION_DEFAULTS`` is reachable from some
tool's bar. An option that quietly became unreachable when it left the sidebar
would be the one real risk of this move.

**Four state bars take precedence over the tool bar**, checked in order, and
the order is the modality: a transform, then a floating buffer, then a
half-finished gesture, then a selection. Each is about something the user is in
the middle of, and while they are in the middle of it that is what the row is
for. The selection bar is the only one that also draws under a tool bar,
because a mask outlives the gesture that made it.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, inker, inker_mode, inker_state, toolbar, widgets
from ..tokens import sp

#: How wide a context field is, in design px, per widget kind. A number rather
#: than a measurement: a field's natural size is a decision, and ``toolbar``
#: needs it *before* it can choose tiers.
WIDE = 130.0
NARROW = 92.0
COMPACT = 54.0

DYNAMICS_POPUP = "inker-dynamics"
OPTIONS_POPUP = "inker-tool-panels"

#: The options that are a box rather than a number or a menu.
_CHECKS = frozenset(
    {"pixel_perfect", "shape_filled", "wand_contiguous", "sample_layer", "aa", "use_stamp"}
)

NIB_LABELS = [
    ("soft", "Soft"),
    ("pixel", "Pixel"),
    ("square", "Square"),
    ("line", "Line"),
]
INK_LABELS = inker_state.INK_LABELS
ALIGN_LABELS = [("free", "Free"), ("origin", "Origin"), ("tile", "Tile")]
COMBINE_LABELS = [
    ("replace", "Replace"),
    ("add", "Add"),
    ("subtract", "Subtract"),
    ("intersect", "Intersect"),
]


def draw(ctx: Any, state: Any, tab: Any) -> None:
    """The bar. Called by the canvas, between the tab bar and the image."""

    if tab is None:
        return
    if state.transforming:
        _transform_bar(ctx, state, tab)
        return
    if tab.doc.floating is not None:
        _float_bar(ctx, state, tab)
        return
    if state.gesture_pts:
        _gesture_bar(ctx, state, tab)
        return
    _tool_bar(ctx, state, tab)
    if tab.doc.mask is not None:
        _selection_bar(ctx, state, tab)


# --- the state bars ---------------------------------------------------------


def _transform_bar(ctx: Any, state: Any, tab: Any) -> None:
    """Deliberately empty here: the transform row is the canvas's own.

    ``inker_canvas._transform_row`` draws it, because the numeric handles read
    and write the floating buffer the canvas is already holding. Named as a bar
    all the same, so the precedence order above is the whole truth about what
    this row shows.
    """


def _float_bar(ctx: Any, state: Any, tab: Any) -> None:
    """A pasted or lifted buffer, and the two ways out of it."""

    widgets.text_colored(_accent(), "Floating")
    imgui.same_line()
    items = [
        toolbar.Item("drop", "Drop", icons.CHECK, role=controls.ButtonRole.PRIMARY, pinned=True),
        toolbar.Item("cancel", "Cancel", icons.X, pinned=True),
    ]
    hit = toolbar.toolbar("inker-float", items)
    if hit == "drop":
        tab.doc.commit_floating()
    elif hit == "cancel":
        tab.doc.cancel_floating()
    imgui.separator()


def _gesture_bar(ctx: Any, state: Any, tab: Any) -> None:
    """A half-drawn polygon, its vertex count, and how to end it.

    It closes a real hole: Enter has always finished a multi-click shape and
    only the manual said so, which is a gesture with no way out on screen.
    """

    count = len(state.gesture_pts)
    widgets.text_colored(_accent(), f"{inker_state.tool_label(state.tool)}: {count} points")
    imgui.same_line()
    items = [
        toolbar.Item(
            "finish",
            "Finish",
            icons.CHECK,
            tooltip="Close the shape (Enter)",
            role=controls.ButtonRole.PRIMARY,
            enabled=count >= 2,
            reason="A shape needs at least two points.",
            pinned=True,
        ),
        toolbar.Item("cancel", "Cancel", icons.X, tooltip="Drop it (Esc)", pinned=True),
    ]
    hit = toolbar.toolbar("inker-gesture", items)
    if hit == "finish":
        inker_mode.commit_gesture(state, tab)
    elif hit == "cancel":
        state.clear_gesture()
    imgui.separator()


def _selection_bar(ctx: Any, state: Any, tab: Any) -> None:
    """The combine mode, shown whenever a mask exists.

    Not only under a select tool, which is the change: ``state.combine`` has
    always been sampled at press and has never been visible anywhere, so
    "Shift adds, Alt subtracts" was a line of prose in a panel rather than a
    control with a state. A mask outlives the tool that made it, so the bar
    that says what the next one will do to it does too.
    """

    imgui.same_line()
    changed, mode = controls.segmented_choice(
        "inker-combine", COMBINE_LABELS, state.combine, compact=True
    )
    if changed:
        state.combine = mode


# --- the tool bar -----------------------------------------------------------


def _tool_bar(ctx: Any, state: Any, tab: Any) -> None:
    tool = state.tool
    keys = inker_state.widgets_for(tool, tab.doc, state)
    inline = [key for key in keys if not inker_state.context_group(key)]
    behind = [key for key in keys if inker_state.context_group(key) == "dynamics"]
    fields = [_field(ctx, state, tab, key) for key in inline]
    fields = [field for field in fields if field is not None]
    items = []
    from . import inker_tools

    if inker_tools._has_panels(state, tab):
        # The panels a 38 px row is the wrong shape for -- a list of gradient
        # stops, the document's slices, the image brush's verbs, the named
        # presets -- behind one button rather than in a column that would have
        # to exist all session for the tools that have one.
        items.append(
            toolbar.Item(
                "panels",
                inker_state.tool_label(tool),
                icons.SETTINGS,
                tooltip=f"{inker_state.tool_label(tool)} panels",
                pinned=True,
            )
        )
    if behind:
        items.append(
            toolbar.Item(
                "dynamics",
                "Dynamics",
                icons.SLIDERS,
                tooltip="Spacing, smoothing, taper -- how the stroke is laid down",
                pinned=True,
            )
        )
    hit = toolbar.toolbar("inker-context", items, fields=fields)
    if hit == "dynamics":
        imgui.open_popup(DYNAMICS_POPUP)
    elif hit == "panels":
        imgui.open_popup(OPTIONS_POPUP)
    _dynamics_popup(ctx, state, tab, behind)
    _panels_popup(ctx, state, tab)
    imgui.separator()


def _dynamics_popup(ctx: Any, state: Any, tab: Any, keys: list[str]) -> None:
    """The settings of a *sitting* rather than of a stroke, behind one glyph.

    Aseprite's own arrangement. These four are reached when a way of working
    changes and then left for an hour, so a row that showed them permanently
    would spend its width on the controls least often touched.
    """

    if not keys:
        return
    with controls.menu_popup(DYNAMICS_POPUP) as opened:
        if not opened:
            return
        for key in keys:
            widgets.field_label(inker_state.context_label(key))
            imgui.set_next_item_width(sp(WIDE))
            field = _field(ctx, state, tab, key)
            if field is not None:
                field.draw(False)


def _panels_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The tool's panels, in a popover off the bar."""

    with controls.menu_popup(OPTIONS_POPUP) as opened:
        if not opened:
            return
        from . import inker_tools

        imgui.push_item_width(sp(200))
        inker_tools.panels(ctx, state, tab)
        imgui.pop_item_width()


def _field(ctx: Any, state: Any, tab: Any, key: str) -> Any:
    """One option as a :class:`toolbar.Field`, or None if it has no widget.

    A ``Field`` rather than a drawn control, so the row can measure the whole
    bar before it draws any of it -- ``same_line`` past the content region
    clips rather than wraps, and a clipped option is one the user cannot reach
    and cannot see to ask about.
    """

    options = state.options_for(state.tool)
    label = inker_state.context_label(key)

    def slider_int(low: int, high: int, fmt: str = "%d", width: float = NARROW) -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.slider_int(
                f"##ctx/{key}", int(options[key]), low, high, fmt
            )
            if changed:
                options[key] = int(value)

        return toolbar.Field(key, label, draw, width=width, compact=COMPACT)

    def slider_float(low: float, high: float, fmt: str = "%.2f") -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.slider_float(
                f"##ctx/{key}", float(options[key]), low, high, fmt
            )
            if changed:
                options[key] = float(value)

        return toolbar.Field(key, label, draw, width=NARROW, compact=COMPACT)

    def combo(choices: list[tuple[str, str]], width: float = NARROW) -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.combo(f"##ctx/{key}", str(options[key]), choices)
            if changed:
                options[key] = value

        # **A combo does not compact.** Its content is a word, and a combo
        # narrower than its own longest option shows "So" where it means
        # "Soft" -- which a screenshot at 1.0 caught. A slider can shrink
        # because its content is a number and the number stays whole.
        return toolbar.Field(key, label, draw, width=width, compact=width)

    def check() -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.checkbox(f"{label}##ctx/{key}", bool(options[key]))
            if changed:
                options[key] = value

        return toolbar.Field(key, label, draw, width=WIDE, compact=WIDE, priority=1)

    # **Every slider carries its own name in its format string.** A row of
    # bare numbers -- 12, 0.85, 1.00 -- is three settings the user has to
    # count along the bar to identify, which is what the screenshot pass
    # found. The names are short because the field is 92 px: this is the
    # place a four-letter word is worth more than a correct one.
    if key == "brush_size":
        return slider_int(inker.MIN_BRUSH, inker.MAX_BRUSH, "%d px")
    if key == "nib":
        # Not COMPACT: "Soft" drawn in a 54 px combo reads "So", which the
        # screenshot pass at 1.0 caught. A combo is as wide as its longest
        # word or it is lying about what is selected.
        return combo(NIB_LABELS)
    if key == "hardness":
        return slider_float(0.0, 1.0, "hard %.2f")
    if key == "opacity":
        return slider_float(0.05, 1.0, "opa %.2f")
    if key == "paint_ink":
        def draw(compact: bool) -> None:
            changed, value = controls.combo(
                "##ctx/paint_ink", state.ink, list(INK_LABELS)
            )
            if changed:
                # Through ``set_ink``, which is where "same in all tools" is
                # honoured -- writing ``options`` directly would set the ink on
                # the tool in hand and leave the others behind.
                state.set_ink(value)
            if imgui.is_item_hovered():
                imgui.set_tooltip(inker_state.ink_hint(state.ink))

        return toolbar.Field(key, label, draw, width=WIDE, compact=NARROW)
    if key == "spray_rate":
        return slider_int(5, 400, "%d/s")
    if key in ("spacing", "stabilise", "speed_taper", "strength"):
        return slider_float(0.0, 1.0, f"{label[:5].lower()} %.2f")
    if key == "wand_tolerance":
        return slider_int(0, 255, "tol %d")
    if key == "brush_angle":
        return slider_int(0, 180, "%d deg")
    if key == "text_size":
        return slider_int(4, 256, "%d px")
    if key == "shade_dir":
        return combo([("1", "Lighter"), ("-1", "Darker")], COMPACT)
    if key == "stamp_align":
        return combo(ALIGN_LABELS)
    if key == "gradient_dither":
        from ..inker import dither

        return combo([("none", "None")] + [(m, m.title()) for m in dither.ORDERED])
    if key == "font":
        return combo(inker_mode.font_choices(), WIDE)
    if key in _CHECKS:
        return check()
    return None


def _accent() -> Any:
    from .. import theme

    return theme.ACCENT

"""What the viewport says without being asked: the axis ball and the hint line.

Chrome drawn *over* and *under* the model rather than beside it, which is what
makes both of them worth having: an orientation widget in a sidebar is a
diagram, and one in the corner of the viewport is a control you reach for
without looking away from what you are turning.

The arithmetic is :mod:`~warlock.studio.clay_hints` and is pure. This file is
only the drawing, which is the split every pane in this app makes and the
reason "does the +X ball sit on the right" is a headless assertion.

No help button, on purpose: this is chrome with no heading to hang one beside,
which is the ``clay_menu`` / ``inker_context`` exemption and is recorded in
``tests/manual/test_coverage.py``. The manual documents both under the Clay
chapter's viewport section.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_hints, clay_mode, controls, theme, widgets
from ..tokens import sp

#: The widget's box, in design pixels. Blender's is about this and the size is
#: doing work: the six balls have to be far enough apart to click without
#: reading the label, and small enough not to sit on the model.
WIDGET = 84.0

#: How far in from the viewport's top-right corner the box sits.
INSET = 8.0

#: How tall the hint line is, for the viewport's reservation.
HINT_H = 20.0


def axis_widget(ctx: Any, view: Any, rect: tuple[float, float, float, float]) -> None:
    """The six axis ends, in the viewport's top-right corner.

    Each ball is a real button rather than a hit test against the draw list, and
    zero-alpha rather than invisible: ``controls.button`` is the chokepoint
    ``probe`` can see, so a ball that stopped working is catchable by a test
    that presses it -- which a hand-rolled ``is_mouse_clicked`` inside a bounds
    check is not. The picture is drawn under it by the draw list, which is what
    ``widgets.list_row`` does for exactly the same reason.
    """

    if view is None:
        return
    size = sp(WIDGET)
    inset = sp(INSET)
    x0 = rect[0] + rect[2] - size - inset
    y0 = rect[1] + inset
    draw_list = imgui.get_window_draw_list()
    balls = clay_hints.axis_layout(view.camera.view(), size)
    radius = sp(9.0)

    # The spokes first, so every ball sits over its own line rather than under
    # the next one's.
    centre = (x0 + size * 0.5, y0 + size * 0.5)
    for ball in balls:
        if not ball.positive:
            continue
        draw_list.add_line(
            centre,
            (x0 + ball.x, y0 + ball.y),
            imgui.get_color_u32(theme.rgba(theme.MUTED)),
            sp(1.5),
        )

    for ball in balls:
        cx, cy = x0 + ball.x, y0 + ball.y
        colour = theme.rgba(theme.ACCENT if ball.positive else theme.MUTED)
        draw_list.add_circle_filled(
            (cx, cy), radius, imgui.get_color_u32(colour), 16
        )
        if ball.label:
            offset = imgui.calc_text_size(ball.label)
            draw_list.add_text(
                (cx - offset.x * 0.5, cy - offset.y * 0.5),
                imgui.get_color_u32((0.0, 0.0, 0.0, 0.85)),
                ball.label,
            )
        imgui.set_cursor_screen_pos((cx - radius, cy - radius))
        # Zero alpha on every colour the button paints with, so the picture
        # above is what the reader sees and the button is only the hit target.
        imgui.push_style_color(imgui.Col_.button.value, (0.0, 0.0, 0.0, 0.0))
        imgui.push_style_color(imgui.Col_.button_hovered.value, (1.0, 1.0, 1.0, 0.2))
        imgui.push_style_color(imgui.Col_.button_active.value, (1.0, 1.0, 1.0, 0.35))
        hit = controls.button(
            f"##clay-axis/{ball.view}",
            (radius * 2.0, radius * 2.0),
            role=controls.ButtonRole.GHOST,
            tooltip=f"Look along {ball.view}",
        )
        imgui.pop_style_color(3)
        if hit:
            view.camera.look_along(ball.view)


def stats_overlay(ctx: Any, rect: tuple[float, float, float, float]) -> None:
    """The statistics line, in the viewport's top-left corner.

    Drawn straight to the window's draw list rather than as a control: it is a
    readout, nothing about it is clickable, and an imgui item at this position
    would sit over the render and eat the click that was meant for the mesh
    under it.

    Opposite corner from the navigation widget, which is the only placement
    rule it needs -- both are corner chrome and two in one corner is one of
    them unreadable.
    """

    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None or not state.overlays.get("stats", False):
        return
    line = clay_hints.stats(tab.doc)
    draw_list = imgui.get_window_draw_list()
    inset = sp(INSET)
    draw_list.add_text(
        (rect[0] + inset, rect[1] + inset),
        imgui.get_color_u32(theme.rgba(theme.MUTED, 0.9)),
        line,
    )


def hint_line(ctx: Any) -> None:
    """One line of what the mouse and the keyboard do right now.

    Under the viewport rather than over it: it is read when you are stuck, and
    a line over the model is a line covering the thing you are stuck on. Muted,
    for the same reason -- it must be legible and must not compete with the
    render.
    """

    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None:
        return
    widgets.muted(
        clay_hints.hint(
            tab.doc.element_mode,
            state.tool,
            dragging=bool(state.drag_kind),
            drag_kind=state.drag_kind,
        )
    )

"""The colour picker: the sliders under the palette, in four colour spaces.

Aseprite's colour bar has a picker at its foot -- a tab strip over RGB, HSV,
HSL and Gray, a slider per channel and a hex field -- and Warlock had none.
The only way to set a channel by number was ``color_edit4``'s own popup inside
the Colour panel, which is a *modal* surface: it closes on the next click, so
nudging a value and looking at the canvas meant reopening it every time.

**It writes through the same doors the rest of the app does.** ``state.set_fg``
for the foreground (which is what clears ``fg_slot`` -- see its docstring),
plain assignment for the background, and ``doc.recolour_slot`` when the
document is indexed and the brush is holding a slot. That last case is
deliberate and is what Aseprite does: in an indexed sprite the picker edits the
*entry*, because the pixels are numbers and a colour that was not in the table
is not a colour the document can hold.

Conversions come from ``colorsys``, the way ``inker.indexed.harmony`` already
does. ``inker.filters._from_hsl`` is not reused: it is array-shaped for a
whole-layer filter and would be the wrong tool for one scalar triple.
"""

from __future__ import annotations

import colorsys
from typing import Any

from imgui_bundle import imgui

from .. import anchors, controls, inker_mode, widgets
from ..manual import render as manual_render

#: The least this pane may be squeezed to, in design px: the heading, the
#: target row, the tab strip, four sliders and the hex field.
#:
#: **Measured, not estimated.** This said 200 px while naming all of that, and
#: the two never agreed: at the app's own default 1600x950 the pane was allotted
#: 363 px -- well clear of a 200 px floor, so ``give_way`` saw no reason to act
#: -- and drew the hex field at y=909 against a pane ending at 902. imgui
#: clipped it away, which made it the one control in Inker nobody could click.
#: A floor shorter than the content it enumerates protects nothing; this is the
#: 400 px that content actually occupies, and the Colour pane above gives way
#: to it. On a window too short for both, ``give_way``'s own ``SHARE_MIN``
#: ceiling still stops this pane starving the one above.
PICKER_FLOOR = 400.0

#: The two things the sliders can be pointed at. The foreground is what a
#: left-drag writes with and the background what a right-drag does, which is
#: the same pair the toolbox's two chips draw.
TARGETS: tuple[tuple[str, str], ...] = (("fg", "Foreground"), ("bg", "Background"))

#: Rec. 601 luma, which is what ``inker`` grades and sorts a palette by. The
#: Gray tab has to pick *a* number to call the grey of a colour, and picking a
#: different one from the rest of the app would make "Brightness" in the
#: palette sort and "Gray" here disagree about the same swatch.
LUMA = (0.299, 0.587, 0.114)


def clamp8(value: float) -> int:
    return max(0, min(255, int(round(value))))


def draw(ctx: Any) -> None:
    anchors.mark_window("inker/picker")
    state = inker_mode.ensure(ctx)
    widgets.section("Picker")
    # After the heading, never before it: ``help_button`` is a ``same_line``,
    # which returns to the previous row unconditionally -- called first it
    # lands on whatever the pane above drew.
    manual_render.help_button(ctx, "inker-picker")
    tab = state.active

    changed, picked = controls.segmented_choice(
        "inkpickertarget", TARGETS, target_of(state), compact=True
    )
    if changed:
        state.picker_target = picked
    slot = slot_of(state, tab)
    if slot is not None:
        widgets.muted(f"Editing palette slot {slot + 1}")

    colour = read(state, tab, slot)
    widgets.tab_bar(
        "inkpickerspace",
        [
            ("RGB", lambda: _rgb(ctx, state, tab, slot, colour)),
            ("HSV", lambda: _hsv(ctx, state, tab, slot, colour)),
            ("HSL", lambda: _hsl(ctx, state, tab, slot, colour)),
            ("Gray", lambda: _gray(ctx, state, tab, slot, colour)),
        ],
    )
    _hex(ctx, state, tab, slot, colour)


def target_of(state: Any) -> str:
    """Which colour the sliders point at. ``fg`` unless asked otherwise."""

    return "bg" if getattr(state, "picker_target", "fg") == "bg" else "fg"


def slot_of(state: Any, tab: Any) -> int | None:
    """The palette entry the sliders edit, or ``None`` for a free colour.

    Only the foreground can be holding a slot -- ``set_fg`` is the one door
    that records one -- and only an indexed document has entries to edit.
    """
    if tab is None or target_of(state) != "fg":
        return None
    doc = tab.doc
    if not getattr(doc, "is_indexed", False):
        return None
    index = state.fg_slot
    if index is None or not (0 <= int(index) < len(doc.palette)):
        return None
    return int(index)


def read(state: Any, tab: Any, slot: int | None) -> tuple[int, int, int, int]:
    """The colour the sliders are showing right now."""

    if slot is not None:
        source: Any = tuple(tab.doc.palette[slot])
    else:
        source = state.fg if target_of(state) == "fg" else state.bg
    channels = [int(channel) for channel in tuple(source)[:4]]
    while len(channels) < 4:
        channels.append(255)
    return (channels[0], channels[1], channels[2], channels[3])


def write(ctx: Any, state: Any, tab: Any, slot: int | None, colour: Any) -> None:
    """One door, so the four spaces and the hex field cannot disagree.

    Nothing is persisted: the two colours are session state, exactly as they
    are in ``inker_colors``, and writing the settings file on every frame of a
    slider drag would be a file write per pixel of travel. ``ctx`` is taken
    anyway so the signature does not have to change the day one of these grows
    a door that needs it.
    """

    value = tuple(clamp8(channel) for channel in tuple(colour)[:4])
    if slot is not None:
        if tab.doc.recolour_slot(slot, value):
            state.palette_usage = None
            state.set_fg(value, slot)
        return
    if target_of(state) == "fg":
        state.set_fg(value)
    else:
        state.bg = value


def _alpha(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    """The one channel every space shares, so it is drawn once."""

    changed, alpha = widgets.labeled_slider_int("Alpha", int(colour[3]), 0, 255)
    if changed:
        write(ctx, state, tab, slot, (colour[0], colour[1], colour[2], alpha))


def _rgb(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    out = list(colour)
    touched = False
    for index, label in enumerate(("Red", "Green", "Blue")):
        changed, value = widgets.labeled_slider_int(label, int(colour[index]), 0, 255)
        if changed:
            out[index] = value
            touched = True
    if touched:
        write(ctx, state, tab, slot, tuple(out))
    _alpha(ctx, state, tab, slot, colour)


def _wheel(
    ctx: Any,
    state: Any,
    tab: Any,
    slot: int | None,
    colour: tuple,
    *,
    labels: tuple[str, str, str],
    to_space: Any,
    from_space: Any,
) -> None:
    """HSV and HSL are the same three sliders over two different triples."""

    r, g, b = (channel / 255.0 for channel in colour[:3])
    first, second, third = to_space(r, g, b)
    values = [
        int(round(first * 360.0)) % 360,
        int(round(second * 100.0)),
        int(round(third * 100.0)),
    ]
    highs = (359, 100, 100)
    out = list(values)
    touched = False
    for index, label in enumerate(labels):
        changed, value = widgets.labeled_slider_int(label, values[index], 0, highs[index])
        if changed:
            out[index] = value
            touched = True
    if touched:
        rr, gg, bb = from_space((out[0] % 360) / 360.0, out[1] / 100.0, out[2] / 100.0)
        write(ctx, state, tab, slot, (rr * 255.0, gg * 255.0, bb * 255.0, colour[3]))
    _alpha(ctx, state, tab, slot, colour)


def _hsv(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    _wheel(
        ctx,
        state,
        tab,
        slot,
        colour,
        labels=("Hue", "Saturation", "Value"),
        to_space=colorsys.rgb_to_hsv,
        from_space=colorsys.hsv_to_rgb,
    )


def _to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """``colorsys`` orders H, L, S; this pane's sliders read H, S, L."""

    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    return hue, sat, light


def _from_hsl(hue: float, sat: float, light: float) -> tuple[float, float, float]:
    return colorsys.hls_to_rgb(hue, light, sat)


def _hsl(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    _wheel(
        ctx,
        state,
        tab,
        slot,
        colour,
        labels=("Hue", "Saturation", "Lightness"),
        to_space=_to_hsl,
        from_space=_from_hsl,
    )


def _gray(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    level = sum(weight * channel for weight, channel in zip(LUMA, colour[:3], strict=True))
    changed, value = widgets.labeled_slider_int("Gray", clamp8(level), 0, 255)
    if changed:
        write(ctx, state, tab, slot, (value, value, value, colour[3]))
    _alpha(ctx, state, tab, slot, colour)


def _hex(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    """``RRGGBB`` or ``RRGGBBAA``, which is what a palette is written down as.

    Applied on enter rather than per keystroke: half a typed hex triple is a
    colour, and applying it would repaint the brush three times on the way to
    the one the user meant.
    """
    text = "".join(f"{clamp8(channel):02X}" for channel in colour)
    widgets.field_label("Hex")
    imgui.set_next_item_width(-1)
    changed, typed = controls.input_text(
        "##inkpickerhex", text, imgui.InputTextFlags_.enter_returns_true.value
    )
    if not changed:
        return
    parsed = parse_hex(typed)
    if parsed is not None:
        write(ctx, state, tab, slot, parsed)


def parse_hex(text: str) -> tuple[int, int, int, int] | None:
    """``#rgb``/``rgba``/``rrggbb``/``rrggbbaa`` -> RGBA, or ``None``.

    Public and pure so the parsing is a plain assertion rather than a
    screenshot: a hex field that silently ignores what was typed is
    indistinguishable from one that is not wired up.
    """
    raw = str(text).strip().lstrip("#")
    if not raw or any(char not in "0123456789abcdefABCDEF" for char in raw):
        return None
    if len(raw) in (3, 4):
        raw = "".join(char * 2 for char in raw)
    if len(raw) == 6:
        raw += "FF"
    if len(raw) != 8:
        return None
    return (
        int(raw[0:2], 16),
        int(raw[2:4], 16),
        int(raw[4:6], 16),
        int(raw[6:8], 16),
    )

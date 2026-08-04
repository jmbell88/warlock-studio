"""Small imgui pieces used by more than one pane.

Nothing here holds state. Each function draws and returns what the user did,
which keeps every pane's read of "what happened this frame" in one place rather
than spread across callbacks.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any

from imgui_bundle import imgui

from . import fonts, motion, theme, tokens
from .tokens import sp

# Every artifact the downloads section offers, in the order it offers them:
# what the mesh *is* first, then what it can be turned into.
ARTIFACTS = (
    ("model.glb", "GLB"),
    ("source.glb", "Source GLB"),
    ("model.stl", "STL"),
    ("model_obj.zip", "OBJ (zip)"),
    ("model.fbx", "FBX"),
    ("collision.glb", "Collision"),
    ("textures.zip", "Textures"),
    ("rig.glb", "Rigged GLB"),
    # The pixels the mesh was reconstructed from. Last because it is an input
    # rather than an output, but present because a promoted job copies it and
    # then had no way to give it back: the inspector showed it as a 96px
    # thumbnail and offered no path to the file.
    ("input.png", "Reference image"),
)


def texture_ref(texture: Any) -> Any:
    """A moderngl texture as something ``imgui.image`` will accept.

    Two things happen here, and skipping the second is the bug where every
    image in the UI renders as imgui's font atlas: imgui 1.92 wants an
    ImTextureRef rather than a bare id, *and* the renderer has to be told which
    moderngl object that id belongs to, because it binds through moderngl and
    an unknown id leaves whatever was bound last in place.
    """
    from . import imgui_backend

    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.register_texture(texture)
    return imgui.ImTextureRef(texture.glo)


def text_colored(value: int, text: str, alpha: float = 1.0) -> None:
    imgui.text_colored(imgui.ImVec4(*theme.rgba(value, alpha)), text)


def muted(text: str) -> None:
    text_colored(theme.MUTED, text)


def section(label: str) -> None:
    """A small heading with breathing room above it."""
    imgui.dummy((0, sp(tokens.SP_1)))
    with fonts.small(imgui):
        text_colored(theme.MUTED, label.upper())
    imgui.separator()


def status_pill(status: str) -> None:
    """A rounded chip: colour *and* a glyph, because a pill that differs only
    by hue is unreadable to a chunk of people and useless in a screenshot."""
    glyph = theme.STATUS_GLYPHS.get(status, "?")
    label = f"{glyph} {status}"
    colour = theme.status_color(status)
    pad_x, pad_y = sp(8), sp(3)
    with fonts.small(imgui):
        size = imgui.calc_text_size(label)
        pos = imgui.get_cursor_screen_pos()
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(
            pos,
            (pos.x + size.x + pad_x * 2, pos.y + size.y + pad_y * 2),
            imgui.get_color_u32((colour[0], colour[1], colour[2], 0.16)),
            (size.y + pad_y * 2) * 0.5,
        )
        imgui.set_cursor_screen_pos((pos.x + pad_x, pos.y + pad_y))
        imgui.text_colored(imgui.ImVec4(*colour), label)
        imgui.set_cursor_screen_pos((pos.x, pos.y))
        imgui.dummy((size.x + pad_x * 2, size.y + pad_y * 2))


def quality_badge(job: dict[str, Any]) -> None:
    """The mesh's verdict, from mesh_report if there is one.

    mesh_report wins over mesh_audit because they answer different questions
    and only the report's answer is about whether an importer will accept it;
    the audit is a silhouette check and its thresholds are what the badge falls
    back to when no report exists.
    """
    params = job.get("params") or {}
    report = params.get("mesh_report")
    if isinstance(report, dict) and report.get("verdict"):
        verdict = str(report["verdict"])
        colour = {"good": theme.OK, "usable": theme.WARN}.get(verdict, theme.ERR)
        text_colored(colour, verdict)
        return
    audit = params.get("mesh_audit")
    if isinstance(audit, dict) and audit.get("hole_ratio") is not None:
        ratio = float(audit["hole_ratio"])
        colour = theme.OK if ratio < 0.02 else theme.WARN if ratio < 0.08 else theme.ERR
        text_colored(colour, f"{ratio * 100:.1f}% open")


def progress_bar(percent: float, width: float = -1.0, height: float = 0.0) -> None:
    """A bar, or a marquee when there is no percentage to show.

    A determinate bar sitting at zero reads as "stuck"; the marquee reads as
    "working on something it cannot measure", which is exactly what a cold
    model load is.
    """
    if height <= 0:
        height = sp(5)
    draw = imgui.get_window_draw_list()
    avail = imgui.get_content_region_avail().x if width < 0 else width
    pos = imgui.get_cursor_screen_pos()
    imgui.dummy((avail, height))
    radius = height * 0.5
    draw.add_rect_filled(
        pos, (pos.x + avail, pos.y + height), imgui.get_color_u32(theme.rgba(theme.EDGE)), radius
    )
    fill = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    if percent > 0:
        end = pos.x + avail * min(percent, 100.0) / 100.0
        draw.add_rect_filled(pos, (end, pos.y + height), fill, radius)
    else:
        span = avail * 0.25
        offset = (time.monotonic() * 0.6 % 1.0) * (avail + span) - span
        draw.add_rect_filled(
            (pos.x + max(offset, 0.0), pos.y),
            (pos.x + min(offset + span, avail), pos.y + height),
            fill,
            radius,
        )


def spinner(radius: float = 0.0, thickness: float = 0.0) -> None:
    """An indeterminate arc. Draws in place and advances the cursor."""
    radius = radius or sp(7)
    thickness = thickness or sp(2.5)
    draw = imgui.get_window_draw_list()
    pos = imgui.get_cursor_screen_pos()
    centre = (pos.x + radius, pos.y + radius)
    imgui.dummy((radius * 2, radius * 2))
    start = time.monotonic() * 3.0
    draw.path_clear()
    for i in range(24):
        angle = start + i / 24.0 * math.pi * 1.5
        draw.path_line_to(
            (centre[0] + math.cos(angle) * radius, centre[1] + math.sin(angle) * radius)
        )
    draw.path_stroke(imgui.get_color_u32(theme.rgba(theme.ACCENT)), thickness=thickness)


def combo(label: str, value: str, options: list[tuple[str, str]], width: float = -1.0):
    """A combo over (key, label) pairs. -> the (possibly unchanged) key.

    Keys rather than indices because every one of these is a guidance taxonomy
    whose order is free to change; an index would silently become a different
    option the next time a table gained an entry.
    """
    keys = [key for key, _ in options]
    labels = [text for _, text in options]
    current = keys.index(value) if value in keys else 0
    if width:
        imgui.set_next_item_width(width)
    changed, index = imgui.combo(label, current, labels)
    return keys[index] if changed else value


def input_text(label: str, value: str, *, max_length: int = 1000, hint: str = "") -> str:
    """A single-line field, clamped after the fact.

    imgui's Python binding grows its own buffer, so the cap is applied to what
    comes back rather than to what can be typed -- which also means a paste
    over the cap keeps its first N characters instead of being refused.
    """
    if hint:
        changed, out = imgui.input_text_with_hint(label, hint, value)
    else:
        changed, out = imgui.input_text(label, value)
    return out[:max_length] if changed else value


def multiline(label: str, value: str, height: float, max_length: int) -> str:
    """A wrapping text area.

    Wrapping is not imgui's default: without the flag a long prompt scrolls off
    to the right in a box three lines tall, which hides most of what was typed
    behind a horizontal scrollbar.
    """
    changed, out = imgui.input_text_multiline(
        label, value, (-1, height), imgui.InputTextFlags_.word_wrap.value
    )
    return out[:max_length] if changed else value


def field_options(ctx: Any, field: str) -> list[tuple[str, str]]:
    """(key, label) pairs for one taxonomy field, with a blank first entry.

    Blank because every guidance field is optional: an empty select means "say
    nothing about this", which is a different prompt from any of the choices.
    """
    entries = (ctx.guidance.get("fields") or {}).get(field) or []
    return [("", f"{field.replace('_', ' ')}...")] + [
        (e["key"], e["label"]) for e in entries
    ]


# Set only by the pane smoke test, which needs every section's contents built
# rather than a column of collapsed headings. Here rather than as an argument
# because the point is to override what the *user* left closed, which no caller
# of header() can know.
FORCE_SECTIONS_OPEN = False

# The Settings the persisted headers write through; injected at setup because
# widgets stay stateless functions and a pane never carries the settings just
# to draw a heading. None (tests, early frames) means default_open decides.
_SETTINGS: Any = None


def attach_settings(settings: Any) -> None:
    global _SETTINGS
    _SETTINGS = settings


def header(label: str, default_open: bool = True, persist_key: str | None = None) -> bool:
    """A collapsing section. Open by default, because these *are* the panel.

    The inspector's sections are its content, not extras: an asset opened with
    every section collapsed shows a column of headings and nothing to act on.
    With ``persist_key``, the open state survives a restart -- imgui's own ini
    is disabled, so this rides the app's Settings instead. FORCE_SECTIONS_OPEN
    still wins: the smoke test needs every section built regardless of what
    last session left closed.
    """
    stored: dict[str, Any] = {}
    if persist_key and _SETTINGS is not None:
        stored = _SETTINGS.get("panels_open") or {}
    if FORCE_SECTIONS_OPEN:
        imgui.set_next_item_open(True, imgui.Cond_.always.value)
    elif persist_key and persist_key in stored:
        imgui.set_next_item_open(bool(stored[persist_key]), imgui.Cond_.once.value)
    flags = imgui.TreeNodeFlags_.default_open.value if default_open else 0
    with fonts.label(imgui):
        opened = imgui.collapsing_header(label, flags)
    if (
        persist_key
        and _SETTINGS is not None
        and not FORCE_SECTIONS_OPEN
        and stored.get(persist_key) != opened
    ):
        _SETTINGS.set("panels_open", {**stored, persist_key: opened})
    return opened


def tab_bar(bar_id: str, tabs: list[tuple[str, Any]]) -> None:
    """Labelled tabs over draw callables.

    Under FORCE_SECTIONS_OPEN every tab's content is drawn sequentially, each
    inside its own ``push_id`` -- the smoke test exists to build every section,
    and a tab bar that only builds the selected tab would quietly exempt the
    other two from it.
    """
    if FORCE_SECTIONS_OPEN:
        for label, draw_fn in tabs:
            imgui.push_id(f"{bar_id}/{label}")
            draw_fn()
            imgui.pop_id()
        return
    if not imgui.begin_tab_bar(bar_id):
        return
    for label, draw_fn in tabs:
        opened, _ = imgui.begin_tab_item(label, None, 0)
        if opened:
            draw_fn()
            imgui.end_tab_item()
    imgui.end_tab_bar()


def disabled_button(label: str, enabled: bool, size: tuple[float, float] = (0, 0)) -> bool:
    """A button that is visibly unavailable rather than absent.

    Absent controls make a UI feel like it is hiding things; a greyed one with
    a tooltip says why.
    """
    if not enabled:
        imgui.begin_disabled()
    clicked = imgui.button(label, size)
    if not enabled:
        imgui.end_disabled()
    return clicked and enabled


def help_marker(text: str) -> None:
    imgui.same_line()
    text_colored(theme.MUTED, "(?)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


def segmented_control(seg_id: str, options: list[tuple[str, str]], current: str) -> str:
    """One rounded track with a sliding highlight; the macOS mode switch.

    Each segment is an ``invisible_button`` with a stable id
    (``{seg_id}/{key}``), which is what keeps this drivable from tests. The
    highlight's position animates; the click is honoured immediately.
    """
    draw = imgui.get_window_draw_list()
    pad_x, pad_y = sp(14), sp(6)
    with fonts.label(imgui):
        widths = [imgui.calc_text_size(label).x + pad_x * 2 for _, label in options]
        height = imgui.get_text_line_height() + pad_y * 2
        origin = imgui.get_cursor_screen_pos()
        draw.add_rect_filled(
            origin,
            (origin.x + sum(widths), origin.y + height),
            imgui.get_color_u32(theme.rgba(theme.ELEV_1)),
            height * 0.5,
        )
        # The sliding pill, at last frame's animated position.
        offsets = [sum(widths[:i]) for i in range(len(options))]
        keys = [key for key, _ in options]
        index = keys.index(current) if current in keys else 0
        x = motion.value(f"{seg_id}/x", offsets[index], duration=tokens.DUR_FAST)
        w = motion.value(f"{seg_id}/w", widths[index], duration=tokens.DUR_FAST)
        draw.add_rect_filled(
            (origin.x + x + sp(2), origin.y + sp(2)),
            (origin.x + x + w - sp(2), origin.y + height - sp(2)),
            imgui.get_color_u32(theme.rgba(theme.ELEV_2)),
            (height - sp(4)) * 0.5,
        )
        selected = current
        for (key, label), width, offset in zip(options, widths, offsets, strict=True):
            imgui.set_cursor_screen_pos((origin.x + offset, origin.y))
            if imgui.invisible_button(f"{seg_id}/{key}", (width, height)):
                selected = key
            hovered = imgui.is_item_hovered()
            active = key == current
            alpha = motion.value(
                f"{seg_id}/{key}/text",
                1.0 if active else (0.85 if hovered else 0.55),
                duration=tokens.DUR_FAST,
            )
            text_size = imgui.calc_text_size(label)
            draw.add_text(
                (
                    origin.x + offset + (width - text_size.x) * 0.5,
                    origin.y + (height - text_size.y) * 0.5,
                ),
                imgui.get_color_u32(theme.rgba(theme.TEXT, alpha)),
                label,
            )
        imgui.set_cursor_screen_pos((origin.x, origin.y))
        imgui.dummy((sum(widths), height))
    return selected


def toggle(label: str, value: bool, *, tag: str | None = None) -> tuple[bool, bool]:
    """An animated switch. -> (changed, value).

    ``tag`` keys the animation when the visible label alone would collide.
    """
    key = tag or label
    height = sp(18)
    width = sp(32)
    origin = imgui.get_cursor_screen_pos()
    text_w = imgui.calc_text_size(label).x if label else 0.0
    clicked = imgui.invisible_button(
        f"##toggle/{key}", (width + (sp(6) + text_w if label else 0), height)
    )
    if clicked:
        value = not value
    t = motion.value(f"toggle/{key}", 1.0 if value else 0.0, duration=tokens.DUR_FAST)
    draw = imgui.get_window_draw_list()
    on = theme.rgba(theme.ACCENT)
    off = theme.rgba(theme.EDGE)
    track = tuple(off[i] + (on[i] - off[i]) * t for i in range(3)) + (1.0,)
    draw.add_rect_filled(
        origin, (origin.x + width, origin.y + height), imgui.get_color_u32(track), height * 0.5
    )
    radius = height * 0.5 - sp(2)
    knob_x = origin.x + height * 0.5 + (width - height) * t
    draw.add_circle_filled(
        (knob_x, origin.y + height * 0.5), radius, imgui.get_color_u32(theme.rgba(0xFFFFFF)), 24
    )
    if label:
        draw.add_text(
            (origin.x + width + sp(6), origin.y + (height - imgui.get_text_line_height()) * 0.5),
            imgui.get_color_u32(theme.rgba(theme.TEXT)),
            label,
        )
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
    return clicked, value


def icon_button(
    icon: str, tooltip: str = "", *, danger: bool = False, enabled: bool = True
) -> bool:
    """A square glyph button with its meaning in the tooltip."""
    side = imgui.get_frame_height()
    pushed = 0
    if danger:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.push_style_color(
            imgui.Col_.button_hovered.value, imgui.ImVec4(*theme.rgba(theme.ERR, 0.8))
        )
        pushed = 2
    if not enabled:
        imgui.begin_disabled()
    clicked = imgui.button(icon, (side, side))
    if not enabled:
        imgui.end_disabled()
    imgui.pop_style_color(pushed)
    if tooltip and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip(tooltip)
    return clicked and enabled


def field_label(label: str) -> None:
    """The small caps line above a control; what makes a combo answerable."""
    with fonts.small(imgui):
        text_colored(theme.MUTED, label.upper())


def labeled_combo(label: str, value: str, options: list[tuple[str, str]], width: float = -1.0):
    """A combo that keeps saying what it is after a value is chosen."""
    field_label(label)
    return combo(f"##{label}", value, options, width)


def primary_button(label: str, size: tuple[float, float] = (0, 0), *, enabled: bool = True) -> bool:
    """The accent-filled call to action; one per pane."""
    imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(*theme.rgba(theme.ACCENT)))
    imgui.push_style_color(
        imgui.Col_.button_hovered.value, imgui.ImVec4(*theme.rgba(theme.ACCENT, 0.85))
    )
    imgui.push_style_color(
        imgui.Col_.button_active.value, imgui.ImVec4(*theme.rgba(theme.ACCENT, 0.7))
    )
    with fonts.label(imgui):
        clicked = disabled_button(label, enabled, size)
    imgui.pop_style_color(3)
    return clicked


def destructive_button(label: str, size: tuple[float, float] = (0, 0)) -> bool:
    """Red where it acts, not where it cancels."""
    imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(*theme.rgba(theme.ERR, 0.85)))
    imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
    imgui.push_style_color(
        imgui.Col_.button_active.value, imgui.ImVec4(*theme.rgba(theme.ERR, 0.7))
    )
    with fonts.label(imgui):
        clicked = imgui.button(label, size)
    imgui.pop_style_color(3)
    return clicked


@contextmanager
def card(card_id: str, size: tuple[float, float]):
    """An elevated child: soft shadow, raised fill, hover lift.

    The shadow is two translucent rounded rects on the parent's draw list,
    drawn *before* the child so they sit beneath it; hover state comes from
    last frame via the motion dict, because this frame's hover is not known
    until the child has been drawn.
    """
    origin = imgui.get_cursor_screen_pos()
    draw = imgui.get_window_draw_list()
    radius = sp(tokens.RADIUS_M)
    lift = motion.value(f"card/{card_id}/lift", 0.0, duration=tokens.DUR_FAST)
    for grow, alpha in ((sp(3), tokens.SHADOW_OUTER), (sp(1), tokens.SHADOW_INNER)):
        draw.add_rect_filled(
            (origin.x - 0, origin.y + grow),
            (origin.x + size[0], origin.y + size[1] + grow),
            imgui.get_color_u32((0, 0, 0, alpha * (0.6 + 0.4 * lift))),
            radius + grow * 0.5,
        )
    imgui.push_style_color(
        imgui.Col_.child_bg.value,
        imgui.ImVec4(*theme.rgba(theme.ELEV_1 if lift < 0.5 else theme.ELEV_2)),
    )
    visible = imgui.begin_child(card_id, size, imgui.ChildFlags_.borders.value)
    try:
        yield visible
    finally:
        imgui.end_child()
        imgui.pop_style_color()
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_blocked_by_active_item.value)
        motion.value(f"card/{card_id}/lift", 1.0 if hovered else 0.0, duration=tokens.DUR_FAST)


def empty_state(icon: str, title: str, hint: str = "") -> None:
    """A centred nothing-here: what this region is for, and what to do next."""
    avail = imgui.get_content_region_avail()
    imgui.dummy((0, max(avail.y * 0.5 - sp(40), 0)))

    def centred(text: str) -> None:
        width = imgui.calc_text_size(text).x
        imgui.set_cursor_pos_x(max((imgui.get_content_region_avail().x - width) * 0.5, 0))
        text_colored(theme.MUTED, text)

    with fonts.title(imgui):
        centred(icon)
    imgui.dummy((0, sp(4)))
    with fonts.label(imgui):
        centred(title)
    if hint:
        with fonts.small(imgui):
            centred(hint)


def toasts(state: Any, viewport_size: tuple[float, float]) -> None:
    """Stacked bottom-right, newest lowest; born sliding up, dying fading out.

    Info toasts stay ``no_inputs`` -- they never mattered enough to click.
    Error toasts take input so their close button works: eight seconds is
    enough to read a sentence, not always enough to act on one.
    """
    state.expire_toasts()
    if not state.toasts:
        return
    now = time.monotonic()
    margin = sp(16)
    y = viewport_size[1] - margin
    dismissed: list[Any] = []
    for toast in reversed(state.toasts[-5:]):
        age = now - toast.born
        fade_in = min(age / 0.18, 1.0)
        fade_out = min(max(toast.ttl - age, 0.0) / 0.3, 1.0)
        alpha = min(fade_in, fade_out)
        rise = (1.0 - (1.0 - fade_in) ** 3) * sp(10) - sp(10)  # eased slide-up
        error = toast.level == "error"
        colour = theme.ERR if error else theme.ELEV_2
        imgui.set_next_window_bg_alpha(0.96 * alpha)
        imgui.set_next_window_pos(
            (viewport_size[0] - margin, y - rise), imgui.Cond_.always.value, (1, 1)
        )
        imgui.set_next_window_size((sp(320), 0))
        imgui.push_style_color(imgui.Col_.window_bg.value, imgui.ImVec4(*theme.rgba(colour)))
        imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
        )
        if not error:
            flags |= imgui.WindowFlags_.no_inputs.value
        if imgui.begin(f"##toast{id(toast)}", None, flags)[0]:
            if error:
                if imgui.small_button("x"):
                    dismissed.append(toast)
                imgui.same_line()
            imgui.text_wrapped(toast.text)
        height = imgui.get_window_height()
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color()
        y -= height + sp(8)
    for toast in dismissed:
        if toast in state.toasts:
            state.toasts.remove(toast)

"""Small imgui pieces used by more than one pane.

Nothing here holds state. Each function draws and returns what the user did,
which keeps every pane's read of "what happened this frame" in one place rather
than spread across callbacks.
"""

from __future__ import annotations

import math
import time
from typing import Any

from imgui_bundle import imgui

from . import theme

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
    imgui.dummy((0, 4))
    text_colored(theme.MUTED, label.upper())
    imgui.separator()


def status_pill(status: str) -> None:
    """Colour *and* a glyph: a pill that differs only by hue is unreadable to
    a chunk of people and useless in a screenshot."""
    glyph = theme.STATUS_GLYPHS.get(status, "?")
    imgui.text_colored(imgui.ImVec4(*theme.status_color(status)), f"[{glyph}] {status}")


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


def progress_bar(percent: float, width: float = -1.0, height: float = 6.0) -> None:
    """A bar, or a marquee when there is no percentage to show.

    A determinate bar sitting at zero reads as "stuck"; the marquee reads as
    "working on something it cannot measure", which is exactly what a cold
    model load is.
    """
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


def spinner(radius: float = 7.0, thickness: float = 2.5) -> None:
    """An indeterminate arc. Draws in place and advances the cursor."""
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
    draw.path_stroke(imgui.get_color_u32(theme.rgba(theme.ACCENT)), 0, thickness)


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
    changed, out = imgui.input_text_multiline(label, value, (-1, height))
    return out[:max_length] if changed else value


# Set only by the pane smoke test, which needs every section's contents built
# rather than a column of collapsed headings. Here rather than as an argument
# because the point is to override what the *user* left closed, which no caller
# of header() can know.
FORCE_SECTIONS_OPEN = False


def header(label: str, default_open: bool = True) -> bool:
    """A collapsing section. Open by default, because these *are* the panel.

    The inspector's sections are its content, not extras: an asset opened with
    every section collapsed shows a column of headings and nothing to act on.
    """
    if FORCE_SECTIONS_OPEN:
        imgui.set_next_item_open(True, imgui.Cond_.always.value)
    flags = imgui.TreeNodeFlags_.default_open.value if default_open else 0
    return imgui.collapsing_header(label, flags)


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


def toasts(state: Any, viewport_size: tuple[float, float]) -> None:
    """Stacked bottom-right, newest lowest."""
    state.expire_toasts()
    if not state.toasts:
        return
    margin = 16.0
    y = viewport_size[1] - margin
    for toast in reversed(state.toasts[-5:]):
        colour = theme.ERR if toast.level == "error" else theme.PANEL
        imgui.set_next_window_bg_alpha(0.96)
        imgui.set_next_window_pos((viewport_size[0] - margin, y), imgui.Cond_.always.value, (1, 1))
        imgui.set_next_window_size((320, 0))
        imgui.push_style_color(imgui.Col_.window_bg.value, imgui.ImVec4(*theme.rgba(colour)))
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_inputs.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
        )
        if imgui.begin(f"##toast{id(toast)}", None, flags)[0]:
            imgui.text_wrapped(toast.text)
        height = imgui.get_window_height()
        imgui.end()
        imgui.pop_style_color()
        y -= height + 8

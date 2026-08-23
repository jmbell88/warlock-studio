"""Compact application status shared by every workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATUS_H = 24.0


@dataclass(frozen=True)
class StatusItem:
    key: str
    text: str
    warning: bool = False


def items(ctx: Any) -> list[StatusItem]:
    """Current status as data so the shell and tests share one account."""

    from . import modes

    mode = str(getattr(ctx.state, "mode", "home"))
    label = next((name for key, name, _icon in modes.MODES if key == mode), mode.title())
    out = [StatusItem("workspace", label)]

    if mode == "inker":
        from . import inker_mode, inker_state

        state = inker_mode.ensure(ctx)
        tab = state.active
        if tab is not None:
            dirty = " *" if bool(getattr(tab, "dirty", False)) else ""
            out.append(StatusItem("document", f"{tab.label}{dirty}"))
            out.append(StatusItem("tool", inker_state.tool_label(state.tool)))
            out.append(StatusItem("zoom", f"{tab.view.zoom * 100:.0f}%"))
    elif mode in ("clay", "plotter", "packwright", "poser"):
        try:
            from importlib import import_module

            module = import_module(f".{mode}_mode", __package__)
            tab = module.active(ctx)
            if tab is not None:
                label = str(getattr(tab, "label", "Untitled"))
                dirty = " *" if bool(getattr(tab, "dirty", False)) else ""
                out.append(StatusItem("document", f"{label}{dirty}"))
        except (AttributeError, ImportError):
            pass

    jobs = list(getattr(getattr(ctx, "cache", None), "jobs", []) or [])
    queued = sum(1 for job in jobs if job.get("status") == "queued")
    running = sum(1 for job in jobs if job.get("status") in ("running", "processing"))
    if queued or running:
        out.append(StatusItem("queue", f"Queue {running} active / {queued} waiting"))

    checks = list(getattr(getattr(ctx, "runtime", None), "checks", []) or [])
    failures = sum(1 for check in checks if not getattr(check, "ok", False))
    errors = len(getattr(ctx.state, "errors", []) or [])
    if failures or errors:
        out.append(StatusItem("health", f"{failures + errors} issue(s)", True))
    return out


def draw(ctx: Any) -> None:
    """Draw a flat one-line status surface; excess secondary items elide."""

    from imgui_bundle import imgui

    from . import fonts, rail, theme, tokens

    pad_x = tokens.sp(tokens.SP_2)
    imgui.push_style_var(
        imgui.StyleVar_.window_padding.value,
        (pad_x, max((tokens.sp(STATUS_H) - imgui.get_text_line_height()) * 0.5, 0.0)),
    )
    imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.PANEL)))
    visible = imgui.begin_child("##global-status", (0, tokens.sp(STATUS_H)))
    imgui.pop_style_color()
    imgui.pop_style_var()
    if visible:
        room = imgui.get_content_region_avail().x
        used = 0.0
        with fonts.small(imgui):
            for index, item in enumerate(items(ctx)):
                text = item.text if index == 0 else f"  |  {item.text}"
                width = imgui.calc_text_size(text).x
                if used + width > room:
                    break
                if index:
                    imgui.same_line(0.0, 0.0)
                if item.warning:
                    imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.WARN)), text)
                    if imgui.is_item_clicked():
                        rail.request("diagnostics")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Open health details")
                else:
                    imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.MUTED)), text)
                used += width
    imgui.end_child()

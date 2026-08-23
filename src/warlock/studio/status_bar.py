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
            # Split at ``##``: an Inker tab's label carries imgui's id suffix
            # (``Untitled##pd1``), which is markup for the widget that draws
            # the tab and was being printed verbatim down here.
            name = str(tab.label).split("##")[0]
            out.append(StatusItem("document", f"{name}{dirty}"))
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


def resource_item(ctx: Any) -> StatusItem | None:
    """The machine's own line, or None when it is off or unreadable.

    **Not in** :func:`items`. That list is unit-tested on its key set and is
    drawn left to right with the tail elided, so putting the meter in it would
    make it the *first* thing dropped as the window narrows -- which is
    backwards for the one item that has to be readable while a generation is
    being decided on. It is right-anchored in :func:`draw` instead, following
    ``overlay.doctor_banner``'s rule: reserve the trailing item before
    trimming the leading detail.
    """
    if not getattr(ctx.state, "show_resources", False):
        return None
    sampler = getattr(ctx, "resources", None)
    if sampler is None:
        return None
    text = sampler.reading.text()
    return StatusItem("resources", text) if text else None


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
        rows = items(ctx)
        meter = resource_item(ctx)
        with fonts.small(imgui):
            meter_w = imgui.calc_text_size(meter.text).x + pad_x if meter else 0.0
            # **The reservation is conditional.** Room is taken for the meter
            # only while the *first* left-hand item still fits beside it;
            # below that the meter is dropped whole rather than truncated. A
            # status bar reading "VRAM 9.2/32" and nothing about which
            # workspace you are in would have the priority exactly backwards.
            if meter and rows and meter_w + imgui.calc_text_size(rows[0].text).x > room:
                meter, meter_w = None, 0.0
            for index, item in enumerate(rows):
                text = item.text if index == 0 else f"  |  {item.text}"
                width = imgui.calc_text_size(text).x
                if used + width + meter_w > room:
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
            if meter:
                imgui.same_line(room - meter_w + pad_x, 0.0)
                imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.MUTED)), meter.text)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "What this machine has left right now. VRAM is read "
                        "from the driver, so it counts every process -- a "
                        "generation is refused when there is not enough of it "
                        "free. Turn it off in Settings."
                    )
    imgui.end_child()

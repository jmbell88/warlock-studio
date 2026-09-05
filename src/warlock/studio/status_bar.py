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


def _document_name(tab: Any) -> str:
    """A tab's label with imgui's id suffix taken off.

    **Every mode, not just Inker.** A tab label carries the id the widget that
    draws it is keyed on -- ``Untitled##pd1`` in Inker, ``Untitled###pl1`` in
    Plotter -- and that is markup, not a name. Inker learned to split it and
    the branch covering the other five modes did not, so a Plotter map read
    ``Untitled###pl1 *`` down here for as long as the status bar has existed.
    Found by looking at a screenshot; no test compared the two branches,
    because both were only ever asserted on their *keys*.

    Split on ``##`` and take the head, which handles ``###`` too -- the third
    hash lands at the start of the discarded tail.
    """
    return str(getattr(tab, "label", "Untitled")).split("##")[0] or "Untitled"


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
            out.append(StatusItem("document", f"{_document_name(tab)}{dirty}"))
            out.append(StatusItem("tool", inker_state.tool_label(state.tool)))
            out.append(StatusItem("zoom", f"{tab.view.zoom * 100:.0f}%"))
    elif mode in ("clay", "plotter", "packwright", "poser", "sirens"):
        try:
            from importlib import import_module

            module = import_module(f".{mode}_mode", __package__)
            tab = module.active(ctx)
            if tab is not None:
                dirty = " *" if bool(getattr(tab, "dirty", False)) else ""
                out.append(StatusItem("document", f"{_document_name(tab)}{dirty}"))
        except (AttributeError, ImportError):
            pass

    jobs = list(getattr(getattr(ctx, "cache", None), "jobs", []) or [])
    queued = sum(1 for job in jobs if job.get("status") == "queued")
    running = sum(1 for job in jobs if job.get("status") in ("running", "processing"))
    if queued or running:
        out.append(StatusItem("queue", f"Queue {running} active / {queued} waiting"))

    checks = list(getattr(getattr(ctx, "runtime", None), "checks", []) or [])
    # Downloads not made yet are not issues. Counting them put "28 issue(s)"
    # in the status bar of a fresh install with nothing wrong with it -- the
    # same false alarm the startup banner used to raise, in the one place a
    # user glances at to decide whether the app is healthy.
    failures = sum(
        1
        for check in checks
        if not getattr(check, "ok", False)
        and not getattr(check, "pending_install", False)
    )
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

    from . import fonts, theme, tokens

    pad_x = tokens.sp(tokens.SP_2)
    # **The face is pushed before the padding is measured, and that ordering is
    # the whole of the vertical centring.** The padding centres one line of
    # text in ``STATUS_H``, so it has to be half of what is left over after
    # *the line that will actually be drawn* -- and every item below is drawn
    # at ``TEXT_SMALL``. Measured outside this block it was ``TEXT_BODY``'s
    # line instead, which reserved too much above and left the remainder
    # below: 5px over and 7px under at scale 1, and 10 over / 14 under at
    # scale 2, because the error is half the gap between the two faces and
    # both scale. The bar read as top-aligned, worse the larger the UI.
    with fonts.small(imgui):
        line = imgui.get_text_line_height()
        imgui.push_style_var(
            imgui.StyleVar_.window_padding.value,
            (pad_x, max((tokens.sp(STATUS_H) - line) * 0.5, 0.0)),
        )
        imgui.push_style_color(
            imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.PANEL))
        )
        visible = imgui.begin_child("##global-status", (0, tokens.sp(STATUS_H)))
        imgui.pop_style_color()
        imgui.pop_style_var()
        if visible:
            room = imgui.get_content_region_avail().x
            used = 0.0
            rows = items(ctx)
            meter = resource_item(ctx)
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
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Health checks need attention")
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
                        "free. The frame rate is the loop's own: it settles at "
                        "12 while nothing on screen can change, which is the "
                        "idle clamp doing its job rather than a stall. Turn it "
                        "off in Settings."
                    )
        imgui.end_child()

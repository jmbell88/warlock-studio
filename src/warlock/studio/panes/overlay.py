"""What is drawn on top of the viewport: progress, doctor warnings, toolbar.

The progress card is the one piece of UI the whole app is judged by -- it is
what a user looks at for two minutes at a time -- so it says four things at
once: what stage, how far, how long it has taken, and how long is left. The
last of those is suppressed until it means something, because a wrong estimate
is worse than none.
"""

from __future__ import annotations

import time
from typing import Any

from imgui_bundle import imgui

from ...service import jobs as svc_jobs
from .. import fonts, icons, theme, widgets
from ..state import format_duration


def toolbar(ctx: Any) -> None:
    """The viewer's own controls, along the top of the viewport."""
    from .. import inker_mode

    state = ctx.state
    viewer = ctx.viewer
    if viewer is None:
        return
    job = ctx.job()
    if ctx.state.mode == "2d" and inker_mode.can_edit_job(ctx, job):
        # First, and only in 2D: the reference is the thing on screen, and the
        # camera controls beside it do not apply to it at all.
        if imgui.button(f"{icons.BRUSH} Open in Inker"):
            inker_mode.open_job_reference(ctx, job)
        imgui.same_line()
    if widgets.icon_button(icons.MAXIMIZE, "Frame the model (F)"):
        viewer.frame()
    imgui.same_line()
    changed, state.wireframe = widgets.toggle("Wireframe", state.wireframe, tag="wireframe")
    if changed:
        viewer.set_wireframe(state.wireframe)
    imgui.same_line()
    changed, state.turntable = widgets.toggle("Turntable", state.turntable, tag="turntable")
    if changed:
        viewer.set_turntable(state.turntable)
    imgui.same_line()
    if widgets.icon_button(icons.CAMERA, "Screenshot...", enabled=viewer.has_model):
        _screenshot(ctx)
    if viewer.has_model:
        # The wheel already dollies; these exist so the control is findable at
        # all. "Frame" beside them is the reset.
        imgui.same_line()
        if widgets.icon_button(icons.ZOOM_IN, "Zoom in (wheel also dollies)"):
            viewer.camera.dolly(1)
        imgui.same_line()
        if widgets.icon_button(icons.ZOOM_OUT, "Zoom out"):
            viewer.camera.dolly(-1)
    if state.comparing:
        imgui.same_line()
        if imgui.button(f"{icons.X} Exit comparison"):
            state.comparing = None
            viewer.exit_compare()


def _screenshot(ctx: Any) -> None:
    from .. import dialogs

    image = ctx.viewer.screenshot()

    def run():
        dest = dialogs.save_file(
            "Save screenshot", f"warlock_{ctx.state.selected or 'view'}.png", dialogs.PNG_FILTER
        )
        if dest is None:
            return None
        image.convert("RGB").save(dest)
        return dest

    ctx.submit("screenshot", run)


def fps_meter(ctx: Any, meter: Any) -> None:
    """The frame rate, bottom-left, when F10 has asked for it.

    Bottom-left rather than bottom-right because the toasts stack up that
    corner, and a fixed width rather than auto-resize because Inter is
    proportional -- an auto-sized box breathes with every digit that changes.
    """
    from ..tokens import sp

    if not ctx.state.show_fps:
        return
    fps = meter.fps
    if fps >= 58.0:
        colour = theme.OK
    elif fps >= 30.0:
        colour = theme.WARN
    else:
        colour = theme.ERR

    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(
        (
            viewport.work_pos.x + sp(16),
            viewport.work_pos.y + viewport.work_size.y - sp(16),
        ),
        imgui.Cond_.always.value,
        (0.0, 1.0),
    )
    imgui.set_next_window_size((sp(210), 0))
    imgui.set_next_window_bg_alpha(0.85)
    imgui.push_style_color(imgui.Col_.window_bg.value, imgui.ImVec4(*theme.rgba(theme.ELEV_2)))
    flags = (
        imgui.WindowFlags_.no_decoration.value
        | imgui.WindowFlags_.no_saved_settings.value
        | imgui.WindowFlags_.no_focus_on_appearing.value
        # Never takes input: it is a readout, and it sits over the library
        # list, whose cards must stay clickable through it.
        | imgui.WindowFlags_.no_inputs.value
    )
    if imgui.begin("##fps-meter", None, flags)[0]:
        with fonts.small(imgui):
            imgui.text_colored(imgui.ImVec4(*theme.rgba(colour)), meter.label())
    imgui.end()
    imgui.pop_style_color()


def progress_card(ctx: Any, eta: Any) -> None:
    """The running job's narration, floating bottom-centre, or nothing.

    A window rather than a child so it overlays whichever mode is on screen --
    the viewport keeps its full height, and a trellis run stays visible from
    Paint. Drawn after the host window, so its Cancel wins hit-testing over the
    image beneath it.
    """
    from ..tokens import sp

    job_id = ctx.runtime.current_job_id
    snapshot = ctx.runtime.progress()
    if snapshot is None or job_id is None:
        return
    percent = float(snapshot.get("percent") or 0.0)
    started = snapshot.get("started_at")
    elapsed = max(time.time() - float(started), 0.0) if started else 0.0
    cold = bool(snapshot.get("cold"))

    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(
        (
            viewport.work_pos.x + viewport.work_size.x * 0.5,
            viewport.work_pos.y + viewport.work_size.y - sp(18),
        ),
        imgui.Cond_.always.value,
        (0.5, 1.0),
    )
    imgui.set_next_window_size((sp(430), 0))
    imgui.set_next_window_bg_alpha(0.94)
    imgui.push_style_color(imgui.Col_.window_bg.value, imgui.ImVec4(*theme.rgba(theme.ELEV_2)))
    flags = (
        imgui.WindowFlags_.no_decoration.value
        | imgui.WindowFlags_.no_saved_settings.value
        | imgui.WindowFlags_.always_auto_resize.value
        | imgui.WindowFlags_.no_focus_on_appearing.value
    )
    if imgui.begin("##progress-card", None, flags)[0]:
        widgets.spinner()
        imgui.same_line()
        with fonts.label(imgui):
            imgui.text(str(snapshot.get("label") or "Working..."))
        widgets.progress_bar(percent)

        detail = str(snapshot.get("detail") or "")
        stage, stages = snapshot.get("stage"), snapshot.get("stages")
        if stage and stages:
            detail = f"stage {stage}/{stages}" + (f" - {detail}" if detail else "")
        widgets.muted(detail)

        line = format_duration(elapsed)
        remaining = eta.update(job_id, percent, elapsed, cold)
        if remaining is not None:
            line += f" - about {format_duration(remaining)} left"
        widgets.muted(line)

        if widgets.disabled_button("Cancel", not ctx.busy(f"cancel:{job_id}")):
            # No confirmation: this button says exactly what it does, sits on
            # the thing it acts on, and a blocking dialog would freeze the very
            # bar behind it.
            ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)
    imgui.end()
    imgui.pop_style_color()


def doctor_banner(ctx: Any) -> None:
    """Only the checks that failed, and only until they are dismissed.

    Drawn from the top strip so it is visible in every mode -- it used to live
    inside the viewport child, which made a dead worker invisible from Paint.
    The text wraps: multiple failures are a paragraph, not one clipped line.
    """
    if ctx.state.last_error is None:
        return
    imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.ERR, 0.25)))
    flags = imgui.ChildFlags_.borders.value | imgui.ChildFlags_.auto_resize_y.value
    if imgui.begin_child("doctor", (-1, 0), flags):
        if imgui.small_button("Dismiss"):
            ctx.state.last_error = None
        imgui.same_line()
        if imgui.small_button("Copy details"):
            imgui.set_clipboard_text(str(ctx.state.last_error))
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(str(ctx.state.last_error))
        imgui.pop_style_color()
    imgui.end_child()
    imgui.pop_style_color()


def placeholder(ctx: Any) -> None:
    """What the viewport says when there is nothing in it."""
    text = {
        "2d": "Describe something and press Generate.",
        "clay": "Add a primitive to start blocking something out.",
        "review": "Pick a sweep run to review.",
    }.get(ctx.state.mode, "Pick a finished reference, or open an image.")
    avail = imgui.get_content_region_avail()
    imgui.dummy((0, max(avail.y * 0.5 - 20, 0)))
    width = imgui.calc_text_size(text).x
    imgui.set_cursor_pos_x(max((avail.x - width) * 0.5, 0))
    widgets.muted(text)

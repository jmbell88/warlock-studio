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
from .. import controls, fonts, icons, rail, theme, tokens, widgets
from ..manual import render as manual_render
from ..state import format_duration


def offers_inker(ctx: Any, job: Any) -> bool:
    """Whether this toolbar is the thing offering Open in Inker.

    Named rather than inlined because the inspector's copy of the button is
    defined as *the complement of this* -- one control per action per object, and
    the toolbar wins wherever it exists because it sits against the pixels the
    button edits. Two spellings of "is it 2D" is how that guarantee would rot
    back into two buttons; see ``inspector.offers_inker``.
    """
    from .. import create_stages, inker_mode

    return create_stages.at(ctx.state, "reference") and inker_mode.can_edit_job(ctx, job)


# How many times across and down the tiled preview repeats. Two: enough to put
# all four wrap edges on screen at once, which is the whole question, and few
# enough that the texture is still drawn at half size rather than a ninth.
TILE_REPEAT = 2


def shows_tiled(ctx: Any, job: Any) -> bool:
    """Whether the tiled-preview toggle belongs on this toolbar.

    Named for ``offers_inker``'s reason: ``main._draw_reference`` has to ask
    the same question to decide what to draw, and two spellings of "is this a
    tile on screen in 2D" is how a toggle comes to be shown for something it
    does not affect -- or, worse, to affect something that does not show it.
    """
    from .. import create_stages

    return (
        create_stages.at(ctx.state, "reference")
        and bool(job)
        and job.get("stage") == "tile"
        and job.get("status") == "done"
    )


def toolbar(ctx: Any) -> None:
    """The viewer's own controls, along the top of the viewport."""
    from .. import inker_mode

    state = ctx.state
    viewer = ctx.viewer
    if viewer is None:
        return
    job = ctx.job()
    # Every continuation below wraps rather than running off the edge. This
    # toolbar carries up to ten controls and is drawn over the *viewport*, so
    # its width is whatever the side panes have left -- narrow the window, or
    # widen the inspector, and a bare same_line() chain puts the last few
    # buttons past the content edge where imgui clips them away and they cannot
    # be clicked at all. Zoom and "Exit comparison" went first, which are also
    # the ones with no keyboard route to fall back on.
    def _wrap(label: str) -> None:
        widgets.same_line_or_wrap(widgets.button_width(label))

    if offers_inker(ctx, job):
        # First, and only in 2D: the reference is the thing on screen, and the
        # camera controls beside it do not apply to it at all.
        if controls.button(f"{icons.BRUSH} Open in Inker"):
            inker_mode.open_job_reference(ctx, job)
        _wrap(f"Tiled {TILE_REPEAT}x{TILE_REPEAT}")
    if shows_tiled(ctx, job):
        # Only for a tile, and only in 2D: it is the one asset for which
        # "repeated" is a true picture of the thing rather than a duplicate of
        # it. The label carries the count so the scale change is stated -- the
        # texture is drawn at half size, and a toggle that only said "Tiled"
        # would leave the user wondering why the image shrank.
        _changed, ctx.state.tile_preview = widgets.toggle(
            f"Tiled {TILE_REPEAT}x{TILE_REPEAT}", ctx.state.tile_preview, tag="tile_preview"
        )
        _wrap(icons.MAXIMIZE)
    if widgets.icon_button(icons.MAXIMIZE, "Frame the model (F)"):
        viewer.frame()
    # The viewport's own way into its chapter. It had none: this toolbar was
    # exempt from the coverage gate for having no titled section to hang a (?)
    # beside, which was true and left the largest thing on screen with no
    # route into the manual at all.
    _wrap(icons.INFO)
    manual_render.help_button_inline(ctx, "overlay")
    _wrap("Wireframe")
    changed, state.wireframe = widgets.toggle("Wireframe", state.wireframe, tag="wireframe")
    if changed:
        viewer.set_wireframe(state.wireframe)
    _wrap("Turntable")
    changed, state.turntable = widgets.toggle("Turntable", state.turntable, tag="turntable")
    if changed:
        viewer.set_turntable(state.turntable)
    _wrap(icons.CAMERA)
    if widgets.icon_button(icons.CAMERA, "Screenshot...", enabled=viewer.has_model):
        _screenshot(ctx)
    if viewer.has_model:
        # The wheel already dollies; these exist so the control is findable at
        # all. "Frame" beside them is the reset.
        _wrap(icons.ZOOM_IN)
        if widgets.icon_button(icons.ZOOM_IN, "Zoom in (wheel also dollies)"):
            viewer.camera.dolly(1)
        _wrap(icons.ZOOM_OUT)
        if widgets.icon_button(icons.ZOOM_OUT, "Zoom out"):
            viewer.camera.dolly(-1)
    if state.comparing:
        _wrap(f"{icons.X} Exit comparison")
        if controls.button(f"{icons.X} Exit comparison"):
            state.comparing = None
            viewer.exit_compare()
    if _has_content(ctx, viewer) and ctx.clear_viewport is not None:
        # Last, and only when there is something to clear: a "Clear" offered
        # over an empty canvas is a button whose whole effect is nothing. It
        # sits at the end because the tiering rule sheds from the right and
        # this is the one control here with no urgency at all.
        _wrap(f"{icons.ERASER} Clear")
        if controls.button(
            f"{icons.ERASER} Clear",
            tooltip="Clears the canvas; reselect the asset to bring it back.",
        ):
            ctx.clear_viewport()
    _texture_losses(viewer)


def _has_content(ctx: Any, viewer: Any) -> bool:
    """Whether this stage's canvas is showing anything.

    Asked per stage rather than as one ``has_model or reference``, because the
    two stages draw from two different fields and a Clear offered on the
    Reference stage because a *mesh* happened to be loaded would appear to do
    nothing at all.
    """
    from .. import create_stages

    if create_stages.at(ctx.state, "reference"):
        return viewer.reference is not None
    return viewer.has_model


def _texture_losses(viewer: Any) -> None:
    """Say when the mesh on screen is missing maps the file carried (D42).

    The loader's stated policy is that an image is a cosmetic loss and never a
    reason to refuse a file, which is right and left the loss reported only in
    the log -- so a mesh that came out grey looked exactly like a mesh that was
    never textured, and the difference is the difference between "this
    reconstruction failed" and "this GLB references its images externally".

    On the toolbar rather than in the inspector because it is a fact about what
    is *rendered*, not about the job: the same asset compared against another
    can lose textures on one side only.
    """
    model = getattr(viewer, "model", None)
    skipped = getattr(model, "skipped_textures", 0) if model is not None else 0
    if not skipped:
        return
    message = (
        f"{skipped} texture could not be loaded"
        if skipped == 1
        else f"{skipped} textures could not be loaded"
    )
    # Wrapped like the rest of the toolbar, and this one is the longest thing
    # on it: a whole sentence appended after ten controls, so it is the first
    # to be clipped -- and it is the only item here that exists solely to be
    # read.
    widgets.same_line_or_wrap(imgui.calc_text_size(message).x)
    widgets.text_colored(theme.WARN, message)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "This file references images the viewer could not read -- stored in a "
            "separate file, in an unreachable buffer, or corrupt. The mesh is "
            "intact; see the log for which. Exports are unaffected: they are "
            "derived from the file, not from what is on screen."
        )


def _screenshot(ctx: Any) -> None:
    from .. import atomic, dialogs

    image = ctx.viewer.screenshot()

    def run():
        dest = dialogs.save_file(
            "Save screenshot", f"warlock_{ctx.state.selected or 'view'}.png", dialogs.PNG_FILTER
        )
        if dest is None:
            return None
        atomic.save_image(dest, image.convert("RGB"))
        return dest

    ctx.submit("screenshot", run)


def fps_meter(ctx: Any, meter: Any) -> None:
    """The frame rate in full, bottom-left, when F10 has asked for it.

    Kept alongside the always-on strip beside the mode switch, and deliberately
    not replaced by it: the strip is a summary and has room for one number,
    while the two that actually diagnose a stall are the mean frame time and
    the *worst* frame in the window -- a single 100 ms hitch moves a 60 fps
    mean to 59.5, so the summary cannot show it and this can.

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


# The last thing the progress card drew, so it has something to fade *out*
# (UX.md Phase 4). Module state, and one slot rather than a history: there is
# one running job at a time by construction, and what a fade-out needs is the
# frame it is a fade of.
_LAST_PROGRESS: dict[str, Any] = {}

# How present the card is, 0 to 1. Keyed here rather than passed through,
# exactly as every other animated value in the app is.
_PROGRESS_KEY = "progress-card/present"


def progress_card(ctx: Any, eta: Any) -> None:
    """The running job's narration, floating bottom-centre, or nothing.

    A window rather than a child so it overlays whichever mode is on screen --
    the viewport keeps its full height, and a trellis run stays visible from
    Paint. Drawn after the host window, so its Cancel wins hit-testing over the
    image beneath it.

    It is "the one piece of UI the whole app is judged by", so it takes the same
    depth treatment as every other floating surface (UX.md Phase 4): the surface
    radius, the raised shadow, and an eased arrival and departure rather than a
    rectangle appearing over the viewport between one frame and the next. The
    fade-*out* is the half that needs state: the job is gone by then, so the
    card is redrawn from the last snapshot it had, with its Cancel disabled --
    a button that acts on a job that has finished is worse than no button.
    """
    from .. import motion, tokens
    from ..tokens import sp

    job_id = ctx.runtime.current_job_id
    snapshot = ctx.runtime.progress()
    live = snapshot is not None and job_id is not None
    if live:
        _LAST_PROGRESS["snapshot"], _LAST_PROGRESS["job"] = snapshot, job_id
    present = motion.value(_PROGRESS_KEY, 1.0 if live else 0.0, duration=tokens.DUR_BASE)
    if not live:
        if present <= 0.01:
            return
        snapshot = _LAST_PROGRESS.get("snapshot")
        job_id = _LAST_PROGRESS.get("job")
        if snapshot is None or job_id is None:
            return
    percent = float(snapshot.get("percent") or 0.0)
    started = snapshot.get("started_at")
    elapsed = max(time.time() - float(started), 0.0) if started else 0.0
    cold = bool(snapshot.get("cold"))

    viewport = imgui.get_main_viewport()
    # The rise is the departure read backwards: it comes up out of the bottom
    # edge and goes back down into it, which is where a bottom-anchored surface
    # belongs. Under reduce-motion ``present`` is already at its target, so the
    # offset is zero and the card simply is or is not there.
    imgui.set_next_window_pos(
        (
            viewport.work_pos.x + viewport.work_size.x * 0.5,
            viewport.work_pos.y + viewport.work_size.y - sp(18) + sp(14) * (1.0 - present),
        ),
        imgui.Cond_.always.value,
        (0.5, 1.0),
    )
    imgui.set_next_window_size((sp(430), 0))
    imgui.set_next_window_bg_alpha(0.94 * present)
    imgui.push_style_color(imgui.Col_.window_bg.value, imgui.ImVec4(*theme.rgba(theme.ELEV_2)))
    imgui.push_style_var(imgui.StyleVar_.alpha.value, present)
    radius = widgets.push_surface_rounding()
    flags = (
        imgui.WindowFlags_.no_decoration.value
        | imgui.WindowFlags_.no_saved_settings.value
        | imgui.WindowFlags_.always_auto_resize.value
        | imgui.WindowFlags_.no_focus_on_appearing.value
    )
    opened = imgui.begin("##progress-card", None, flags)[0]
    widgets.pop_surface_rounding()
    if opened:
        # Raised, and the shadow rides the same fade: a shadow left at full
        # strength under a card on its way out is a dark rectangle hanging over
        # the viewport with nothing in it.
        widgets.window_shadow("raised", radius=radius)
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
        # Only while the job is live: the estimator is keyed on a job id and
        # feeding it samples from a fade-out would teach it about a job that
        # has already stopped moving.
        remaining = eta.update(job_id, percent, elapsed, cold) if live else None
        if remaining is not None:
            line += f" - about {format_duration(remaining)} left"
        widgets.muted(line)

        if widgets.disabled_button("Cancel", live and not ctx.busy(f"cancel:{job_id}")):
            # No confirmation: this button says exactly what it does, sits on
            # the thing it acts on, and a blocking dialog would freeze the very
            # bar behind it.
            ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)
    imgui.end()
    imgui.pop_style_var()
    imgui.pop_style_color()


def doctor_summary(errors: list[str]) -> tuple[str, str]:
    """Compact issue count and the leading action/detail for the shell strip."""

    count = len(errors)
    noun = "issue" if count == 1 else "issues"
    first = " ".join(str(errors[0]).split()) if errors else ""
    return f"{count} setup {noun}", first


def doctor_banner(ctx: Any) -> None:
    """One quiet shell summary; full commands and details stay in Issues."""

    if not ctx.state.errors:
        return
    title, leading = doctor_summary(ctx.state.errors)
    imgui.push_style_color(
        imgui.Col_.child_bg.value,
        imgui.ImVec4(*theme.rgba(theme.WARN, tokens.ERROR_WASH_ALPHA)),
    )
    flags = imgui.ChildFlags_.auto_resize_y.value
    if imgui.begin_child("doctor", (-1, 0), flags):
        widgets.text_colored(theme.WARN, f"{icons.TRIANGLE_ALERT} {title}")
        imgui.same_line()
        # Reserve the two trailing actions before trimming the leading detail.
        action_width = widgets.button_width("Review") + widgets.button_width("Dismiss")
        action_width += imgui.get_style().item_spacing.x * 2
        detail_width = max(imgui.get_content_region_avail().x - action_width, 0)
        widgets.muted(widgets.fit_text(leading, detail_width))
        imgui.same_line()
        if controls.small_button("Review", role=controls.ButtonRole.GHOST):
            rail.request("diagnostics")
        imgui.same_line()
        if controls.small_button("Dismiss", role=controls.ButtonRole.GHOST):
            ctx.state.dismiss_errors()
    imgui.end_child()
    imgui.pop_style_color()


# What an empty viewport says, per mode: icon, title, and what to do about it
# (H74). Upgraded from one muted sentence to the icon+title+hint form every
# other empty list in the app uses, and given the ``inker`` entry it never had
# -- Inker fell through to the mesh sentence, so an empty canvas pane advised
# picking a finished reference.
#
# Create is keyed ``create/{stage}`` rather than on the mode alone: one mode
# with two viewports would otherwise have to pick one of the two sentences and
# be wrong half the time. The slash is not a path -- it is there so a mode key
# and a stage key can never collide in one table.
PLACEHOLDERS: dict[str, tuple[str, str, str]] = {
    "create/reference": (
        icons.IMAGE,
        "Nothing generated yet",
        "Describe something and press Generate.",
    ),
    "create/mesh": (
        icons.BOX,
        "No mesh on screen",
        "Pick a finished reference, or open an image.",
    ),
    "create/rig": (
        icons.BONE,
        "No mesh to rig",
        "Pick a finished mesh; the skeleton is fitted to it.",
    ),
    "create/pose": (
        icons.PERSON_STANDING,
        "No rig on screen",
        "Rig the mesh first, then press Edit pose.",
    ),
    "create/export": (
        icons.DOWNLOAD,
        "Nothing to export",
        "Pick a finished asset; its files are listed on the left.",
    ),
    "inker": (icons.PEN_TOOL, "No drawing open", "Ctrl+N starts one, Ctrl+O opens a file."),
    "clay": (icons.RULER, "Empty document", "Add a primitive to start blocking something out."),
    "poser": (
        icons.PERSON_STANDING,
        "No skeleton on screen",
        "Pick a skeleton; the preview builds itself.",
    ),
    "review": (icons.CIRCLE_CHECK, "No unit on screen", "Pick a sweep run to review."),
    "plotter": (icons.GRID, "No map open", "Ctrl+N starts one, Ctrl+O opens a file."),
    "packwright": (
        icons.LAYERS,
        "Nothing to pack",
        "Add images, or pull the frames out of a document open in Inker.",
    ),
    "sirens": (
        icons.AUDIO_WAVEFORM,
        "No song open",
        "Ctrl+N starts one, Ctrl+O opens a file.",
    ),
    "troupe": (
        icons.PERSON_STANDING,
        "No character on screen",
        "Pick one on the left, or describe a new one below it.",
    ),
}


def centred_empty(icon: str, title: str, hint: str) -> None:
    """An :func:`widgets.empty_state` centred in the viewport it is drawn in.

    Hoisted out of :func:`placeholder` so a viewport with a *transient* empty
    state -- the Poser while Blender builds an armature, or after it failed --
    can look like the nine that have a permanent one, instead of two lines of
    muted text in the top-left corner.
    """
    from ..tokens import sp

    avail = imgui.get_content_region_avail()
    # Centred vertically by hand: ``empty_state`` centres its own text
    # horizontally but knows nothing about the height it is sitting in.
    imgui.dummy((0, max(avail.y * 0.5 - sp(48), 0)))
    widgets.empty_state(icon, title, hint)


def placeholder(ctx: Any) -> None:
    """What the viewport says when there is nothing in it."""
    from .. import create_stages

    key = (
        f"{create_stages.MODE}/{ctx.state.create_stage}"
        if create_stages.in_create(ctx.state)
        else ctx.state.mode
    )
    icon, title, hint = PLACEHOLDERS.get(key, PLACEHOLDERS["create/mesh"])
    centred_empty(icon, title, hint)

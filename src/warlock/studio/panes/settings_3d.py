"""The 3D pane: the mesh-side decisions, and where a mesh comes from.

No prompt controls at all. A 3D job starts from a finished 2D asset (whose
reference is promoted) or from an uploaded image, and everything this pane
holds is an *override* on what that source already recorded -- which is why
every one of them is optional and "unset" is a real value rather than a
default in disguise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import vectors
from ...bench import findings as findings_lib
from ...service import jobs as svc_jobs
from ...service.errors import Invalid
from ...service.validation import MAX_MESH_CANDIDATES, MAX_UPLOAD_BYTES, random_seed
from .. import controls, dialogs, focus, forms, matte_preview, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp

MATTE_TITLE = "Check the cutout"

# This pane's key in the focus ring; see ``settings_2d.FOCUS_PANE``.
FOCUS_PANE = "3d"

# What each of ``pipelines/matting``'s three sources is called on screen. The
# distinction matters to the user: the corner fill is a guess a plain
# background makes work, and BiRefNet is a model -- and "this image already
# carries one" is the answer that means their own edit is what will be used.
MATTE_SOURCES = {
    "alpha": "The reference's own alpha",
    "birefnet": "BiRefNet cutout",
    "flood": "Corner fill (BiRefNet's weights are not installed)",
}

# The only tier the UI offers. gltfpack is vendored now, so the named tiers can
# run -- but none of them has been qualified (kept UVs, both PBR maps and
# material assignment on a chest, a sword and a rock), and an unqualified tier
# on a generate form is a button that silently degrades a mesh. The retarget
# control in the inspector is the qualification path: it offers the whole list
# once the binary is present, so a tier can be exercised before it is exposed
# here.
PROFILES = [("raw", "Raw (no decimation)")]


def draw(ctx: Any) -> None:
    """Draw the mesh form, including its "Mesh resolution" choice."""

    # Form.errors replaces field_error(ctx.state, "platform") and keeps the
    # service's field key attached to the shared control's ring and error copy.
    with forms.Form("create-3d", errors=ctx.state.field_errors) as form_ui:
        _draw_form(ctx, form_ui, "Mesh resolution")


def _draw_form(
    ctx: Any, form_ui: forms.Form, mesh_resolution_label: str
) -> None:
    state = ctx.state
    form = state.form_3d

    # The keyboard ring (UX.md Phase 3), over this pane's own controls. Shorter
    # than 2D's because the pane is: everything here is an override on what the
    # source reference recorded, so the path to a mesh is pick a source, press
    # the button -- and both ends of that are in the ring.
    focus.pump(state, FOCUS_PANE)
    focus.begin(state, FOCUS_PANE)
    widgets.section("Source")
    manual_render.help_button(ctx, "settings-3d")
    _source(ctx)

    widgets.section("Mesh")
    # Labels above rather than beside: a combo here is drawn at -1 width, and
    # imgui puts a widget's label to its *right* -- so every one of these was
    # a full-width select with its name clipped off the edge of the panel, and
    # "Detail", "Budget" and "Background" were invisible. ``labeled_combo`` is
    # the widget that already answers this, and the 2D pane's guidance grid
    # uses the same small-caps line above each control.
    # "Mesh resolution", not "Detail" (UX.md Phase 3). It and the 2D pane's own
    # ``platform`` control were called "Detail" and "platform detail", two
    # near-identical names for a geometry resolution and a prompt fragment, and
    # the whole of what kept them apart was a tooltip on each apologising for
    # the other. A name that says what the control *is* ends that; the tooltip
    # below keeps only the half that is still worth saying.
    before = form["platform"]
    with focus.item(ctx.state, FOCUS_PANE, "platform"):
        _changed, form["platform"] = form_ui.combo(
            "platform",
            mesh_resolution_label,
            form["platform"],
            _platform_options(ctx),
            help_text=(
                "How much geometry trellis is asked for. Higher costs more GPU "
                "and more triangles."
            ),
        )
    if form["platform"] != before:
        ctx.state.clear_field_error("platform")
    _hint(ctx, "platform", form["platform"])
    _budget(ctx, form)

    _size(form)
    # Deliberately unhinted, unlike every other control here: size_m is
    # continuous, so its buckets are keyed on "0.35" and "0.36" separately and
    # a threshold of five would essentially never be met.
    widgets.help_marker("0 keeps whatever the reference recorded.")

    with focus.item(ctx.state, FOCUS_PANE, "bg_removal"):
        _changed, form["bg_removal"] = form_ui.combo(
            "bg_removal",
            "Background",
            form["bg_removal"],
            _bg_options(ctx),
        )
    _hint(ctx, "bg_removal", form["bg_removal"])

    with focus.item(ctx.state, FOCUS_PANE, "mesh_seed"):
        changed, seed = form_ui.number(
            "mesh_seed", "Mesh seed", int(form["mesh_seed"])
        )
    if changed:
        form["mesh_seed"] = max(0, seed)
    if controls.button("Reroll##mesh", role=controls.ButtonRole.GHOST):
        form["mesh_seed"] = random_seed()

    changed, prep = form_ui.switch(
        "reference_prep",
        "Normalise the reference",
        bool(form["reference_prep"]),
        help_text=(
            "Recentre the subject and scale it to fill the frame before the mesh "
            "engine sees it."
        ),
        helper=(
            "Off by default: the engine does its own cropping, and whether doing "
            "it twice helps has not been measured."
        ),
    )
    if changed:
        form["reference_prep"] = prep
    _hint(ctx, "reference_prep", form["reference_prep"])

    _rig(ctx, form)
    _reset_row(ctx)
    _submit(ctx, form)


# --- pieces -----------------------------------------------------------------


def _reset_row(ctx: Any) -> None:
    """The 2D pane's *Reset...* had no counterpart here, and the asymmetry was
    the whole of the reason: both panes accumulate overrides across a session
    and only one of them offered a way back.

    Above the submit rather than below it, exactly as 2D places it: a
    destructive control under the primary action is one the hand reaches by
    accident.
    """
    if controls.button("Reset...", role=controls.ButtonRole.GHOST):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Reset the model settings?",
                message=(
                    "The mesh resolution, size, background removal, seed, "
                    "candidate count and rig controls go back to their "
                    "defaults. The chosen source is kept, and the image form "
                    "is untouched."
                ),
                confirm_label="Reset",
                cancel_label="Cancel",
                on_confirm=lambda: _reset(ctx),
            )
        )


def _reset(ctx: Any) -> None:
    """The 3D form back to first-launch defaults.

    A fresh copy of ``DEFAULT_FORM_3D`` rather than a field-by-field clear, for
    ``settings_2d._reset``'s reason: a field added later is reset by having been
    added rather than by somebody remembering this function. A *copy*, because
    the module-level dict is the default and a form aliased onto it would edit
    it for the rest of the process -- which is the shape ``AppState.form_3d``'s
    own default_factory already takes.

    The source job is deliberately not cleared. It is not part of this form at
    all (it lives on ``state.source_job``), and a reset that silently dropped
    what the user had picked to work from would be a different, larger action
    than the one the button offers.
    """
    from ..state import DEFAULT_FORM_3D

    ctx.state.form_3d = dict(DEFAULT_FORM_3D)
    ctx.toast("The model settings are back to their defaults.")


def _findings_hint(ctx: Any, param: str, value: Any) -> str | None:
    """Same lookup as the 2D pane's -- see ``settings_2d._findings_hint``.

    The subject comes from the source asset rather than from a form, because
    this pane owns no prompt controls at all: a 3D job starts from a finished
    2D reference and inherits its prompt, so that reference's prompt *is* the
    subject the mesh will be of. With no source picked yet there is no subject
    to scope by and the pooled corpus answers, unlabelled -- which is honest:
    nothing has been chosen for a hint to be about.
    """
    doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    source = ctx.cache.get(ctx.state.source_job)
    subject = vectors.prompt_hash(source.get("prompt")) if source else ""
    return findings_lib.hint(doc, param, value, prompt_hash=subject or None)


def _hint(ctx: Any, param: str, value: Any) -> None:
    """Draw the findings hint for the control just drawn, if there is one.

    This pane used to hint one control out of five, which put the evidence
    furthest from where it applies: an observation measures *geometry* -- hole
    fraction, watertightness, triangle count -- so the settings it can speak
    about most directly are exactly these, and they were the ones showing
    nothing. Every param here is in ``vectors.VECTOR_PARAMS``, so every one of
    them is something a verdict and an observation are already filed against.
    """
    hint = _findings_hint(ctx, param, value)
    if hint is None:
        return
    widgets.hint_text(hint)


# "Size (m)" as a drag rather than a slider (K96), and the *ceiling* is why: a
# slider needs a maximum and there is no largest asset -- a wall section is
# legitimately 8 m. A drag has the same feel, still honours a double-click for
# typed entry, and has no upper bound to be wrong about. The speed is what
# makes it usable: 1 cm per pixel, so the range a prop actually lives in (0.1
# to 2 m) is two hundred pixels of travel rather than four.
SIZE_DRAG_SPEED = 0.01
# ``v_min >= v_max`` is how imgui's drag widgets spell "unbounded". Stated as a
# named pair rather than as a literal ``0.0, 0.0`` so the intent survives
# somebody "fixing" it into a range.
SIZE_NO_BOUND = (0.0, 0.0)


def _size(form: dict[str, Any]) -> None:
    """Metres, as a drag with the unit *in* the readout.

    "Size (m)" put the unit in the label and the number in the box, so a value
    read at a glance said "0.35" and the label it belonged to was a separate
    thing to look at. The format string carries it now, and 0 says what it
    means rather than showing a measurement of zero metres.
    """
    value = float(form["size_m"])
    fmt = "unset - keeps the reference's" if value <= 0.0 else "%.2f m"
    changed, size = controls.drag_float("Size", value, SIZE_DRAG_SPEED, *SIZE_NO_BOUND, fmt)
    if changed:
        # Floored here rather than by the widget: unbounded means unbounded in
        # both directions, and a negative size is not a smaller asset.
        form["size_m"] = max(0.0, size)


def _budget(ctx: Any, form: dict[str, Any]) -> None:
    """The triangle budget -- drawn only when there is a choice to make.

    While :data:`PROFILES` has one entry there is nothing here a user can do.
    It used to be drawn anyway, disabled, with three lines explaining why: the
    argument was that a combo with a single entry looks broken, so saying
    "unqualified tier, not missing binary" beats saying nothing. But the
    control and its note are five lines of the densest form in the app, spent
    entirely on explaining their own inertness -- and the note's own answer is
    that the *inspector's* retarget control is where a tier gets tried. Send
    the user there by not putting a dead affordance in front of them here.

    The form key is untouched either way, so the door (``profile`` at submit)
    is unchanged: this stops drawing a control, it does not stop sending one.
    """
    if len(PROFILES) == 1:
        return
    form["profile"] = widgets.labeled_combo("Budget", form["profile"], PROFILES)
    widgets.field_error(ctx.state, "profile")
    _hint(ctx, "profile", form["profile"])
    if form["profile"] == "custom":
        # The same control the retarget panel draws, appearing under exactly
        # the same condition (K95). It is the widget ``custom_triangles`` never
        # had: the field was submitted, validated and recorded with no way to
        # set it, which is a form field that exists only for the API.
        imgui.set_next_item_width(sp(140))
        changed, value = controls.input_int("Triangles", int(form["custom_triangles"]), 0, 0)
        if changed:
            form["custom_triangles"] = max(0, value)


def _platform_options(ctx: Any) -> list[tuple[str, str]]:
    entries = (ctx.guidance.get("fields") or {}).get("platform") or []
    return [("", "keep the reference's")] + [(e["key"], e["label"]) for e in entries]


def _bg_options(ctx: Any) -> list[tuple[str, str]]:
    return [("", "keep the reference's")] + [
        (key, key) for key in (ctx.guidance.get("bg_removal") or [])
    ]


def _source(ctx: Any) -> None:
    """The 2D asset this job starts from, or an upload.

    The whole block is one drag-and-drop target (I83): a card dragged out of
    the library lands here. A *group* rather than a child window, because a
    child clips and this content grows a line whenever a source is picked --
    the target has to be exactly the area the user is aiming at, at every
    display scale.
    """
    from . import library

    state = ctx.state
    source = ctx.cache.get(state.source_job)
    dragging = library.dragged_job(ctx)
    imgui.begin_group()
    origin = imgui.get_cursor_screen_pos()
    if source is not None:
        imgui.text_wrapped(source.get("name") or source.get("prompt") or source["id"])
        widgets.muted(f"reference - {source['id']}")
        if controls.button("Clear"):
            state.source_job = None
    elif dragging is not None:
        # The invitation replaces the instruction only while something is in
        # the air: a line about dropping, with nothing to drop, is noise.
        widgets.muted("Drop it here to use it as the source.")
    else:
        widgets.muted("Pick a finished reference in the library, or:")
    busy = ctx.busy("upload")
    if widgets.disabled_button(
        "Open an image...", not busy, reason="A file picker is already open."
    ):
        ctx.submit("upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER)
    widgets.muted("...or drop an image on the window.")
    imgui.end_group()
    end = imgui.get_item_rect_max()
    if dragging is not None:
        # A target the pointer is over says so, and one that is merely
        # *available* says that too but more quietly. Drawn after the group so
        # the outline is not clipped by it.
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_blocked_by_active_item.value)
        widgets.ring(
            origin,
            end,
            theme.ACCENT if hovered else theme.MUTED,
            0.9 if hovered else 0.4,
            2.0 if hovered else 1.0,
        )
    else:
        # The same ring, fading, for a file dropped from Explorer (H70): the
        # two arrivals look the same because they are the same event.
        widgets.ring(origin, end, theme.ACCENT, widgets.drop_flash(state, "3d-source"))
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(library.DRAG_JOB)
        if payload is not None and state.dragging_job:
            # Through ``library.select`` rather than by assigning ``source_job``
            # here: that function is what also moves the selection, so a
            # dropped card is the selected card and the inspector on the right
            # is showing the thing the form now names.
            library.select(ctx, state.dragging_job)
            state.source_job = state.dragging_job
            state.dragging_job = None
        imgui.end_drag_drop_target()


def _rig(ctx: Any, form: dict[str, Any]) -> None:
    if not ctx.rigging_available:
        # Hidden rather than disabled: without bpy the whole feature is absent,
        # and a greyed control implies it could be turned on from here.
        return
    widgets.section("Rig")
    changed, rig = controls.checkbox("Rig when the mesh lands", bool(form["rig"]))
    if changed:
        form["rig"] = rig
    if form["rig"]:
        options = [(t["key"], t["label"]) for t in ctx.rig_templates]
        if options:
            form["rig_template"] = widgets.labeled_combo(
                "Skeleton", form["rig_template"] or ctx.rig_default, options
            )
            # Three doors refuse on this exact field -- ``validation.py``,
            # ``service/rig.py`` and ``service/poses.py`` -- and without this
            # the one dropdown at fault was the only control in the form never
            # outlined, unlike a bad ``profile`` or ``base_model``.
            widgets.field_error(ctx.state, "rig_template")


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    imgui.dummy((0, sp(8)))
    imgui.separator()
    state = ctx.state
    source = ctx.cache.get(state.source_job)
    problems = validate(source)
    for problem in problems:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(problem)
        imgui.pop_style_color()
    _candidates(form)
    count = candidate_count(form)
    widgets.muted(
        "Roughly two minutes of GPU."
        if count == 1
        else f"Roughly {count * 2} minutes of GPU - {count} attempts, one queue."
    )
    busy = ctx.busy("submit")
    enabled = not problems and not busy
    with focus.item(ctx.state, FOCUS_PANE, "make3d") as focused:
        pressed = widgets.primary_button("Make 3D", (-1, sp(34)), enabled=enabled)
        # Enter on the ring's last stop; see ``settings_2d._submit``.
        if focused and enabled and (
            imgui.is_key_pressed(imgui.Key.enter)
            or imgui.is_key_pressed(imgui.Key.keypad_enter)
        ):
            pressed = True
    if pressed:
        promote(ctx, source, form)
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip("Ctrl+Enter")


def candidate_count(form: dict[str, Any]) -> int:
    """How many meshes this form asks for, clamped to what the service admits.

    Clamped here as well as refused there because the form is persisted: a
    settings file written when the ceiling was higher (or edited by hand) would
    otherwise send a number ``promote_candidates`` refuses, and the refusal
    would arrive as an error toast on a control the user cannot see is wrong.
    """
    try:
        count = int(form.get("candidates", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, min(count, MAX_MESH_CANDIDATES))


def _candidates(form: dict[str, Any]) -> None:
    """The Candidates control: how many attempts one press buys.

    A row of radio-style buttons rather than a combo, because there are three
    values and the number is the label -- and because it sits directly above
    Make 3D, where the cost sentence under it changes with the choice. It is
    the *only* control in this pane that multiplies what the button spends, so
    putting it anywhere else in the form would hide that.
    """
    widgets.field_label("Candidates")
    current = candidate_count(form)
    for count in range(1, MAX_MESH_CANDIDATES + 1):
        if count > 1:
            imgui.same_line()
        # Never drawn past the panel edge: three 40 px buttons and two spacings
        # fit inside the 300 px sidebar with room to spare, and the guard in
        # tests/test_studio_smoke.py measures rather than trusts that.
        if controls.radio_button(f"{count}##candidates", current == count):
            form["candidates"] = count
    widgets.help_marker(
        "Reconstruct the same reference more than once and keep the best. The "
        "engine is deterministic in its seed, so each attempt draws a new one; "
        "the rest are hidden from the library until you keep one."
    )


def validate(source: dict[str, Any] | None) -> list[widgets.Problem]:
    """``settings_2d.validate``'s counterpart, and its field rule.

    All three of these are about the *reference*, not about a control in this
    form, so none carries a field: they are refusals the library answers, and
    ringing a widget in the promotion form would point at the wrong thing.
    ``note_field_error`` already treats an empty field as "keep going to the
    toast", which is the behaviour that leaves.
    """
    if source is None:
        return [widgets.Problem("Choose a reference first.")]
    if source.get("status") != "done":
        return [widgets.Problem(f"That reference is {source.get('status')}.")]
    if "input.png" not in (source.get("files") or []):
        return [widgets.Problem("That reference has no image.")]
    return []


def promote_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The overrides, with "unset" left out entirely.

    Omitted means "keep what the reference recorded", and that is not the same
    as sending the reference's value back. ``promote_to_model`` drops the
    inherited resolution unconditionally -- re-deriving it from a platform
    override, or from the default 3D platform when there is none -- so this
    pane sends no ``resolution`` at all: an override here would pin a number
    the platform no longer implies.
    """
    out: dict[str, Any] = {}
    if form["platform"]:
        out["platform"] = form["platform"]
    if float(form["size_m"]) > 0:
        out["size_m"] = float(form["size_m"])
    if form["bg_removal"]:
        out["bg_removal"] = form["bg_removal"]
    if form["profile"]:
        out["profile"] = form["profile"]
        if int(form["custom_triangles"]) > 0:
            out["custom_triangles"] = int(form["custom_triangles"])
    if int(form["mesh_seed"]) > 0:
        out["mesh_seed"] = int(form["mesh_seed"])
    # An explicit False, not an omission: it has to clear a rig request the
    # reference inherited, or a reference generated with rigging on would rig
    # every promotion of it whatever this pane says.
    out["rig"] = bool(form["rig"])
    if form["rig"] and form["rig_template"]:
        out["rig_template"] = form["rig_template"]
    # Explicit like rig, and for the same reason: an omission would let the
    # promotion inherit whatever the reference recorded, and this checkbox is
    # the 3D pane's decision, not the reference's.
    out["reference_prep"] = bool(form["reference_prep"])
    return out


def promote(ctx: Any, source: dict[str, Any] | None, form: dict[str, Any]) -> None:
    """Put the matte in front of the two minutes of GPU.

    The button no longer submits: it opens the preview, which shows the cutout
    trellis will reconstruct from and offers Accept / Fix matte / Cancel. The
    matte is the single decision that most often turns a good reference into a
    solid slab, and it used to be made inside the exe *after* the user had
    committed. The composition gate's own verdict moves into the same panel for
    the same reason -- one place, before the spend, rather than a confirm here
    and a surprise there.
    """
    problems = validate(source)
    if problems:
        # ``settings_2d.generate``'s reason exactly: Ctrl+Enter in 3D mode and
        # the palette's promote both land here, and this used to return in
        # silence -- so pressing Ctrl+Enter with nothing selected did nothing
        # at all, which reads as a broken shortcut rather than as a refusal.
        from . import settings_2d

        settings_2d.refuse(ctx, problems)
        return
    # ``count`` rides with the overrides because the preview captures the form
    # as it stood when the button was pressed -- the whole point of that
    # capture is that Accept submits what the user pressed with, and how many
    # of it is part of that. ``promote_candidates`` takes it as a kwarg, so
    # nothing downstream has to unpack it back out.
    matte_preview.open_for(
        ctx, source["id"], {**promote_kwargs(form), "count": candidate_count(form)}
    )


def submit_promotion(ctx: Any, job_id: str, kwargs: dict[str, Any], force: bool) -> None:
    """Queue the mesh job (or the candidate group) the preview was about.

    Always through ``promote_candidates``, count included: at 1 it *is*
    ``promote_to_model`` with no group minted, so there is one call path here
    rather than a branch that could send the two halves different overrides.
    """
    # The rings from the last refusal describe a request that no longer exists;
    # see ``settings_2d.generate``, which clears them for the same reason.
    ctx.state.clear_field_errors()
    if ctx.state.filters.kind not in ("all", "model"):
        # Otherwise a filter left on "reference" (the natural way to find the
        # source image before promoting it) permanently hides the model job
        # this creates.
        ctx.state.filters.kind = "all"
    # The return is checked, and the reason is the shared ``"submit"`` key.
    # ``TaskRunner.submit`` refuses a key that is already in flight -- which is
    # right, because a second click almost always means "I did not see the
    # first one work". But this call arrives from the matte modal's Accept, and
    # the modal closes on its own regardless: an Accept landing while an earlier
    # create was still running was dropped, silently, with the modal closing and
    # looking exactly like success. Nothing queued, nothing said (UX-26).
    if not ctx.submit(
        "submit", svc_jobs.promote_candidates, ctx.svc, job_id, force=force, **kwargs
    ):
        ctx.toast("Still submitting the last one - try again in a moment.")


def matte_modal(ctx: Any) -> None:
    """The promote preview. Drawn beside the confirms, because it is a modal.

    Everything expensive happened elsewhere: ``matte_preview.pump`` submits the
    cutout to the TaskRunner and the frame thread only uploads the pixels it
    gets back, through ``ThumbnailCache.from_pixels`` so the texture inherits
    the deferred-release rule every other image in the UI has.
    """
    state = matte_preview.pump(ctx)
    if state is None:
        return
    appearing = not state._open
    if appearing:
        imgui.open_popup(MATTE_TITLE)
        state._open = True
    centre = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
    alpha, rise = widgets.popover_enter("matte-preview", appearing)
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    opened, _ = imgui.begin_popup_modal(
        MATTE_TITLE, None, imgui.WindowFlags_.always_auto_resize.value
    )
    widgets.pop_surface_rounding()
    if not opened:
        # Escape dismisses a modal without going through any of the buttons,
        # and imgui will not reopen a popup whose id it thinks is already open:
        # without this the modal would vanish once and never come back, with
        # ``job_id`` still set and every later press of Make 3D doing nothing.
        imgui.pop_style_var()
        state._open = False
        matte_preview.close(ctx)
        return
    widgets.window_shadow("overlay", radius=radius)
    if frosted:
        widgets.window_backdrop(radius=radius)
    if rise > 0.0:
        imgui.dummy((0, rise))
    _matte_body(ctx, state)
    imgui.end_popup()
    imgui.pop_style_var()


def _matte_body(ctx: Any, state: Any) -> None:
    preview = state.preview
    if preview is None:
        widgets.muted("Cutting the subject out...")
    else:
        _matte_image(ctx, preview)
        widgets.muted(f"{MATTE_SOURCES.get(preview.source, preview.source)} - "
                      f"keeps {preview.coverage * 100:.0f}% of the frame")
        if preview.approved:
            widgets.muted("This reference already carries this matte; it will be kept.")
        for reason in preview.reasons:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
            imgui.text_wrapped(reason)
            imgui.pop_style_color()
        for warning in preview.warnings:
            imgui.text_wrapped(warning)
    imgui.dummy((0, sp(6)))
    ready = preview is not None
    refused = bool(preview is not None and preview.reasons)
    label = "Build anyway" if refused else "Accept"
    # Both buttons in this popup wait on the same thing, and it is a state the
    # user can see happening -- so the sentence says what is being waited for
    # rather than restating that the button is off.
    preview_why = "The cutout is still being prepared."
    role = controls.ButtonRole.DESTRUCTIVE if refused else controls.ButtonRole.PRIMARY
    if controls.button(
        label,
        (sp(150), 0),
        role=role,
        enabled=ready,
        reason=preview_why,
    ):
        imgui.close_current_popup()
        state._open = False
        # Read *before* ``accept``, which closes the state before it calls
        # back: reading it inside the callback would name the empty string.
        job_id = state.job_id
        matte_preview.accept(
            ctx,
            lambda kwargs, force: submit_promotion(ctx, job_id, kwargs, force or refused),
        )
        return
    imgui.same_line()
    if controls.button(
        "Fix matte", (sp(150), 0), enabled=ready, reason=preview_why
    ):
        imgui.close_current_popup()
        state._open = False
        matte_preview.fix(ctx)
        return
    imgui.same_line()
    if controls.button(
        "Cancel", (sp(100), 0), role=controls.ButtonRole.GHOST
    ):
        imgui.close_current_popup()
        state._open = False
        matte_preview.close(ctx)


def _matte_image(ctx: Any, preview: Any) -> None:
    if ctx.textures is None:
        return
    texture = ctx.textures.from_pixels(
        f"matte:{preview.job_id}",
        float(preview.stamp or 0),
        (preview.width, preview.height),
        preview.rgb,
    )
    if texture is None:
        return
    imgui.image(widgets.texture_ref(texture), (preview.width, preview.height))


def upload_bytes(ctx: Any, data: bytes) -> None:
    """Start a mesh job from pixels that are already in memory.

    The path ``upload`` takes for a file, for a caller that has rendered the
    picture rather than read it -- Clay's "send to 3D", which draws the
    document offscreen on the frame thread and hands the bytes over. The form
    values are read here for the same reason ``upload`` reads them here: they
    are UI state, and the task thread has no business touching them.
    """
    kwargs = _upload_kwargs(ctx)

    def run():
        return svc_jobs.create_job(ctx.svc, image=data, **kwargs)

    ctx.submit("submit", run)


def _upload_kwargs(ctx: Any) -> dict[str, Any]:
    """The 3D form as create_job keyword arguments.

    Shared by both upload paths so a form field cannot be honoured for a
    dropped file and quietly ignored for a rendered one.
    """
    form = ctx.state.form_3d
    kwargs: dict[str, Any] = {"kind": "image"}
    if form["platform"]:
        kwargs["guidance_fields"] = {"platform": form["platform"]}
    if float(form["size_m"]) > 0:
        kwargs["size_m"] = float(form["size_m"])
    if form["bg_removal"]:
        kwargs["bg_removal"] = form["bg_removal"]
    if form["profile"]:
        kwargs["profile"] = form["profile"]
    if int(form["mesh_seed"]) > 0:
        kwargs["mesh_seed"] = int(form["mesh_seed"])
    kwargs["reference_prep"] = bool(form["reference_prep"])
    if form["rig"]:
        kwargs["rig"] = True
        if form["rig_template"]:
            kwargs["rig_template"] = form["rig_template"]
    return kwargs


def upload(ctx: Any, path: Path) -> None:
    """Start a mesh job from an image on disk (a picker, or a dropped file)."""
    kwargs = _upload_kwargs(ctx)

    # The form values are read here, on the frame thread, because they are UI
    # state; the *file* is read in the task, because a large one would freeze
    # the window for as long as the disk took. Only MAX_UPLOAD_BYTES + 1 bytes
    # are ever read -- create_job's contract -- so an enormous file is refused
    # rather than allocated.
    def run():
        try:
            with path.open("rb") as fh:
                data = fh.read(MAX_UPLOAD_BYTES + 1)
        except OSError as exc:
            # ``field=`` for ``settings_2d``'s reason: the upload control is
            # what is wrong, and a bare refusal points at nothing.
            raise Invalid(f"could not read {path.name}: {exc}", field="image") from exc
        return svc_jobs.create_job(ctx.svc, image=data, **kwargs)

    ctx.submit("submit", run)

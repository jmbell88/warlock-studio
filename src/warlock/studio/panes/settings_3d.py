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
from .. import dialogs, matte_preview, theme, widgets
from ..manual import render as manual_render

MATTE_TITLE = "Check the cutout"

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
    state = ctx.state
    form = state.form_3d

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
    form["platform"] = widgets.labeled_combo("Detail", form["platform"], _platform_options(ctx))
    _hint(ctx, "platform", form["platform"])
    widgets.help_marker(
        "The geometry resolution sent to trellis. The 2D pane's platform is a "
        "separate thing -- a hint in the prompt."
    )
    form["profile"] = widgets.labeled_combo("Budget", form["profile"], PROFILES)
    _hint(ctx, "profile", form["profile"])

    changed, size = imgui.input_float("Size (m)", float(form["size_m"]), 0.0, 0.0, "%.2f")
    if changed:
        form["size_m"] = max(0.0, size)
    # Deliberately unhinted, unlike every other control here: size_m is
    # continuous, so its buckets are keyed on "0.35" and "0.36" separately and
    # a threshold of five would essentially never be met.
    widgets.help_marker("0 keeps whatever the reference recorded.")

    form["bg_removal"] = widgets.labeled_combo("Background", form["bg_removal"], _bg_options(ctx))
    _hint(ctx, "bg_removal", form["bg_removal"])

    imgui.set_next_item_width(120)
    changed, seed = imgui.input_int("Mesh seed", int(form["mesh_seed"]), 0, 0)
    if changed:
        form["mesh_seed"] = max(0, seed)
    imgui.same_line()
    if imgui.button("Reroll##mesh"):
        form["mesh_seed"] = random_seed()

    changed, prep = imgui.checkbox("Normalise the reference", bool(form["reference_prep"]))
    if changed:
        form["reference_prep"] = prep
    _hint(ctx, "reference_prep", form["reference_prep"])
    widgets.help_marker(
        "Recentre the subject and scale it to fill the frame before the mesh "
        "engine sees it. Off by default: the engine does its own cropping, and "
        "whether doing it twice helps has not been measured."
    )

    _rig(ctx, form)
    _submit(ctx, form)


# --- pieces -----------------------------------------------------------------


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


def _platform_options(ctx: Any) -> list[tuple[str, str]]:
    entries = (ctx.guidance.get("fields") or {}).get("platform") or []
    return [("", "keep the reference's")] + [(e["key"], e["label"]) for e in entries]


def _bg_options(ctx: Any) -> list[tuple[str, str]]:
    return [("", "keep the reference's")] + [
        (key, key) for key in (ctx.guidance.get("bg_removal") or [])
    ]


def _source(ctx: Any) -> None:
    """The 2D asset this job starts from, or an upload."""
    state = ctx.state
    source = ctx.cache.get(state.source_job)
    if source is not None:
        imgui.text_wrapped(source.get("name") or source.get("prompt") or source["id"])
        widgets.muted(f"reference - {source['id']}")
        if imgui.button("Clear"):
            state.source_job = None
    else:
        widgets.muted("Pick a finished reference in the library, or:")
    busy = ctx.busy("upload")
    if widgets.disabled_button("Open an image...", not busy):
        ctx.submit("upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER)
    widgets.muted("...or drop an image on the window.")


def _rig(ctx: Any, form: dict[str, Any]) -> None:
    if not ctx.rigging_available:
        # Hidden rather than disabled: without bpy the whole feature is absent,
        # and a greyed control implies it could be turned on from here.
        return
    widgets.section("Rig")
    changed, rig = imgui.checkbox("Rig when the mesh lands", bool(form["rig"]))
    if changed:
        form["rig"] = rig
    if form["rig"]:
        options = [(t["key"], t["label"]) for t in ctx.rig_templates]
        if options:
            form["rig_template"] = widgets.labeled_combo(
                "Skeleton", form["rig_template"] or ctx.rig_default, options
            )


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    imgui.dummy((0, 8))
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
    if widgets.primary_button("Make 3D", (-1, 34), enabled=not problems and not busy):
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
        if imgui.radio_button(f"{count}##candidates", current == count):
            form["candidates"] = count
    widgets.help_marker(
        "Reconstruct the same reference more than once and keep the best. The "
        "engine is deterministic in its seed, so each attempt draws a new one; "
        "the rest are hidden from the library until you keep one."
    )


def validate(source: dict[str, Any] | None) -> list[str]:
    if source is None:
        return ["Choose a reference first."]
    if source.get("status") != "done":
        return [f"That reference is {source.get('status')}."]
    if "input.png" not in (source.get("files") or []):
        return ["That reference has no image."]
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
    if validate(source):
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
    if ctx.state.filters.kind not in ("all", "model"):
        # Otherwise a filter left on "reference" (the natural way to find the
        # source image before promoting it) permanently hides the model job
        # this creates.
        ctx.state.filters.kind = "all"
    ctx.submit("submit", svc_jobs.promote_candidates, ctx.svc, job_id, force=force, **kwargs)


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
    if not state._open:
        imgui.open_popup(MATTE_TITLE)
        state._open = True
    centre = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
    opened, _ = imgui.begin_popup_modal(
        MATTE_TITLE, None, imgui.WindowFlags_.always_auto_resize.value
    )
    if not opened:
        # Escape dismisses a modal without going through any of the buttons,
        # and imgui will not reopen a popup whose id it thinks is already open:
        # without this the modal would vanish once and never come back, with
        # ``job_id`` still set and every later press of Make 3D doing nothing.
        state._open = False
        matte_preview.close(ctx)
        return
    _matte_body(ctx, state)
    imgui.end_popup()


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
    imgui.dummy((0, 6))
    ready = preview is not None
    refused = bool(preview is not None and preview.reasons)
    label = "Build anyway" if refused else "Accept"
    if widgets.disabled_button(label, ready, (150, 0)):
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
    if widgets.disabled_button("Fix matte", ready, (150, 0)):
        imgui.close_current_popup()
        state._open = False
        matte_preview.fix(ctx)
        return
    imgui.same_line()
    if imgui.button("Cancel", (100, 0)):
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
            raise Invalid(f"could not read {path.name}: {exc}") from exc
        return svc_jobs.create_job(ctx.svc, image=data, **kwargs)

    ctx.submit("submit", run)

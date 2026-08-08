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
from ...service.validation import MAX_UPLOAD_BYTES, random_seed
from .. import dialogs, theme, widgets
from ..manual import render as manual_render

# The only tier the UI offers. Every named tier needs a gltfpack that is not
# vendored yet, so offering one would be offering a button that can only fail.
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
    widgets.muted("Roughly two minutes of GPU.")
    busy = ctx.busy("submit")
    if widgets.primary_button("Make 3D", (-1, 34), enabled=not problems and not busy):
        promote(ctx, source, form)
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip("Ctrl+Enter")


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
    as sending the reference's value back -- a platform override drops the
    resolution it implied, so sending one unnecessarily re-derives it.
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
    if validate(source):
        return
    kwargs = promote_kwargs(form)

    def go(force: bool = False) -> None:
        if ctx.state.filters.kind not in ("all", "model"):
            # Otherwise a filter left on "reference" (the natural way to find
            # the source image before promoting it) permanently hides the
            # model job this creates.
            ctx.state.filters.kind = "all"
        ctx.submit(
            "submit", svc_jobs.promote_to_model, ctx.svc, source["id"], force=force, **kwargs
        )

    report = (source.get("params") or {}).get("reference_report") or {}
    if report.get("ok") is False:
        # A confirm rather than a refusal: the rules are heuristics about
        # composition, and the user can see the image the pane is arguing
        # about. What they must not do is spend two minutes of GPU by
        # accident.
        ctx.confirms.ask(
            dialogs.Confirm(
                title="This reference may not reconstruct",
                message=" ".join(report.get("reasons") or []) + "\n\nBuild it anyway?",
                confirm_label="Build anyway",
                cancel_label="Cancel",
                on_confirm=lambda: go(force=True),
            )
        )
        return
    go()


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

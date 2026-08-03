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

from ...service import jobs as svc_jobs
from ...service.validation import random_seed
from .. import dialogs, theme, widgets

# The only tier the UI offers. Every named tier needs a gltfpack that is not
# vendored yet, so offering one would be offering a button that can only fail.
PROFILES = [("raw", "Raw (no decimation)")]


def draw(ctx: Any) -> None:
    state = ctx.state
    form = state.form_3d

    widgets.section("Source")
    _source(ctx)

    widgets.section("Mesh")
    form["platform"] = widgets.combo(
        "Detail", form["platform"], _platform_options(ctx)
    )
    widgets.help_marker(
        "The geometry resolution sent to trellis. The 2D pane's platform is a "
        "separate thing -- a hint in the prompt."
    )
    form["profile"] = widgets.combo("Budget", form["profile"], PROFILES)

    changed, size = imgui.input_float("Size (m)", float(form["size_m"]), 0.0, 0.0, "%.2f")
    if changed:
        form["size_m"] = max(0.0, size)
    widgets.help_marker("0 keeps whatever the reference recorded.")

    form["bg_removal"] = widgets.combo(
        "Background", form["bg_removal"], _bg_options(ctx)
    )

    imgui.set_next_item_width(120)
    changed, seed = imgui.input_int("Mesh seed", int(form["mesh_seed"]), 0, 0)
    if changed:
        form["mesh_seed"] = max(0, seed)
    imgui.same_line()
    if imgui.button("Reroll##mesh"):
        form["mesh_seed"] = random_seed()

    _rig(ctx, form)
    _submit(ctx, form)


# --- pieces -----------------------------------------------------------------


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
            form["rig_template"] = widgets.combo(
                "Skeleton", form["rig_template"] or ctx.rig_default, options
            )


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    imgui.dummy((0, 8))
    imgui.separator()
    state = ctx.state
    source = ctx.cache.get(state.source_job)
    problems = validate(source)
    for problem in problems:
        widgets.text_colored(theme.ERR, problem)
    widgets.muted("Roughly two minutes of GPU.")
    busy = ctx.busy("submit")
    if widgets.disabled_button("Make 3D", not problems and not busy, (-1, 34)):
        promote(ctx, source, form)


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
    return out


def promote(ctx: Any, source: dict[str, Any] | None, form: dict[str, Any]) -> None:
    if validate(source):
        return
    ctx.submit(
        "submit", svc_jobs.promote_to_model, ctx.svc, source["id"], **promote_kwargs(form)
    )


def upload(ctx: Any, path: Path) -> None:
    """Start a mesh job from an image on disk (a picker, or a dropped file)."""
    form = ctx.state.form_3d
    kwargs: dict[str, Any] = {"kind": "image", "image": path.read_bytes()}
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
    if form["rig"]:
        kwargs["rig"] = True
        if form["rig_template"]:
            kwargs["rig_template"] = form["rig_template"]
    ctx.submit("submit", svc_jobs.create_job, ctx.svc, **kwargs)

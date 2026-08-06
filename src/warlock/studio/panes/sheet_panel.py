"""Sprite sheets: the direction preview, the grid, and queuing a render.

The preview strip is a *direction* preview -- it cannot pose the mesh, so
drawing one row per pose would draw the same row N times. The grid the worker
will actually produce is stated as a summary line instead, and that line is the
part that has to agree with ``pipelines.sheet.plan``.
"""

from __future__ import annotations

import math
from typing import Any

from imgui_bundle import imgui

from ... import rigging
from ...service import sheets as svc_sheets
from .. import dialogs, widgets
from ..manual import render as manual_render
from ..viewer import sheet as sheetlib

# 4/8/16: the direction counts every 2D engine's facing conventions already
# use. Not free-form, because a count that does not divide 360 into the
# directions an engine indexes by is a sheet nothing can address.
YAW_CHOICES = (4, 8, 16)


def draw(ctx: Any, job: Any) -> None:
    if "model.glb" not in (job.get("files") or []):
        return
    if not widgets.header("Sprite sheet", default_open=False):
        return
    manual_render.help_button(ctx, "sheet")

    form = _form(ctx, job["id"])
    _preview(ctx, form)
    _controls(ctx, job, form)
    _summary(ctx, form)
    _submit(ctx, job, form)
    _saved(ctx, job)


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    """Lazily created and kept on the app state, so it survives a reselect.

    Rebuilt when the selection moves to a different job, because half of it is
    pose ids: those belong to the rig they were fitted to, and carrying them
    across a selection change submits another job's poses -- which ``validate``
    cannot catch, since it only checks that *a* rig exists.
    """
    form = ctx.state.preview.get("sheet_form")
    if form is None or form.get("job_id") != job_id:
        defaults = (ctx.sheet_options or {}).get("defaults") or {}
        form = {
            "job_id": job_id,
            "yaws": 8,
            "elevation": float(defaults.get("elevation") or 0.0),
            "frame_size": int(defaults.get("frame_size") or 128),
            "lighting": defaults.get("lighting") or "flat",
            "poses": set(),
            "clip": False,
            "clip_from": "",
            "clip_to": "",
            "clip_frames": 8,
            "name": "",
        }
        ctx.state.preview["sheet_form"] = form
    return form


def _preview(ctx: Any, form: dict[str, Any]) -> None:
    viewer = ctx.viewer
    if viewer is None or not viewer.has_model:
        return
    key = "sheet-preview"
    if widgets.disabled_button("Refresh preview", not ctx.busy(key)):
        yaws = [i * 360.0 / form["yaws"] for i in range(form["yaws"])]
        try:
            strip = viewer.render_sheet_strip(
                yaws, math.radians(form["elevation"]), form["lighting"] == "flat"
            )
        except Exception:
            ctx.toast("Could not render the preview.", "error")
            return
        ctx.state.preview["sheet_strip"] = strip
    strip = ctx.state.preview.get("sheet_strip")
    if strip is not None and ctx.textures is not None:
        texture = _strip_texture(ctx, strip)
        if texture is not None:
            width = imgui.get_content_region_avail().x
            imgui.image(widgets.texture_ref(texture), (width, width * strip.height / strip.width))


def release_strip_texture(ctx: Any) -> None:
    """Forget-then-release the cached strip texture. Also called at teardown."""
    cached = ctx.state.preview.pop("sheet_texture", None)
    if cached is None:
        return
    from .. import imgui_backend

    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.forget_texture(cached)
    cached.release()


def _strip_texture(ctx: Any, strip: Any) -> Any:
    """One reusable texture for the strip, replaced when its size changes."""
    cached = ctx.state.preview.get("sheet_texture")
    if cached is not None and cached.size != strip.size:
        # Registered with the imgui backend by texture_ref, so it must be
        # forgotten before the driver can reuse its GL name.
        release_strip_texture(ctx)
        cached = None
    if cached is None:
        cached = ctx.viewer.ctx.texture(strip.size, 4, strip.tobytes())
        cached.filter = (ctx.viewer.ctx.LINEAR, ctx.viewer.ctx.LINEAR)
        ctx.state.preview["sheet_texture"] = cached
    else:
        cached.write(strip.tobytes())
    return cached


def _controls(ctx: Any, job: Any, form: dict[str, Any]) -> None:
    imgui.text("Directions")
    for count in YAW_CHOICES:
        imgui.same_line()
        if imgui.radio_button(f"{count}##yaws", form["yaws"] == count):
            form["yaws"] = count

    options = ctx.sheet_options or {}
    sizes = [(str(s), f"{s} px") for s in (options.get("frame_sizes") or [64, 128, 256])]
    picked = widgets.combo("Frame", str(form["frame_size"]), sizes)
    form["frame_size"] = int(picked)
    form["lighting"] = widgets.combo(
        "Lighting",
        form["lighting"],
        [(key, key) for key in (options.get("lighting") or ["flat", "lit"])],
    )
    changed, elevation = imgui.slider_float("Elevation", form["elevation"], -60.0, 60.0, "%.0f deg")
    if changed:
        form["elevation"] = elevation
    # The service validates and stores this and the saved list displays it;
    # without an input there was no way to ever set it, so every sheet showed
    # its raw id.
    form["name"] = widgets.input_text(
        "##sheet-name", form["name"], max_length=rigging.MAX_SHEET_NAME, hint="Name (optional)"
    )

    poses = (ctx.state.preview or {}).get("poses") or []
    if poses:
        imgui.text("Rows")
        for pose in poses:
            checked = pose["id"] in form["poses"]
            hit, value = imgui.checkbox(f"{pose.get('name') or pose['id']}##row", checked)
            if hit:
                form["poses"].symmetric_difference_update({pose["id"]})
                del value
        changed, clip = imgui.checkbox("Animated clip", form["clip"])
        if changed:
            form["clip"] = clip
        if form["clip"]:
            # A clip replaces the pose rows rather than adding to them: its
            # rows *are* the animation, and mixing static poses in would give
            # an importer no way to tell which rows loop.
            names = [(p["id"], p.get("name") or p["id"]) for p in poses]
            form["clip_from"] = widgets.combo("From", form["clip_from"], names)
            form["clip_to"] = widgets.combo("To", form["clip_to"], names)
            changed, frames = imgui.slider_int("Frames", form["clip_frames"], 2, 32)
            if changed:
                form["clip_frames"] = frames
    del job


def _summary(ctx: Any, form: dict[str, Any]) -> None:
    rows = (
        form["clip_frames"]
        if form["clip"]
        else max(len(form["poses"]), 1)
    )
    widgets.muted(sheetlib.summary(rows, form["yaws"], form["frame_size"], form["clip"]))
    del ctx


def _submit(ctx: Any, job: Any, form: dict[str, Any]) -> None:
    job_id = job["id"]
    busy = ctx.busy(f"sheet:{job_id}")
    problems = validate(job, form)
    for problem in problems:
        widgets.muted(problem)
    if widgets.disabled_button("Render sheet", not problems and not busy, (-1, 0)):
        ctx.submit(
            f"sheet:{job_id}",
            svc_sheets.create_sheet,
            ctx.svc,
            job_id,
            poses=sorted(form["poses"]),
            elevation=form["elevation"],
            frame_size=form["frame_size"],
            lighting=form["lighting"],
            name=form["name"],
            clip_from=form["clip_from"] if form["clip"] else None,
            clip_to=form["clip_to"] if form["clip"] else None,
            clip_frames=form["clip_frames"],
            yaws=form["yaws"],
        )


def validate(job: Any, form: dict[str, Any]) -> list[str]:
    """Refusals stated before the button, matching what the service checks."""
    problems: list[str] = []
    files = job.get("files") or []
    if job.get("status") != "done" or "model.glb" not in files:
        problems.append("This job has no finished mesh.")
    rigged = "rig.glb" in files
    if form["poses"] and not rigged:
        problems.append("Posed sheets need a rigged mesh.")
    if form["clip"]:
        if not (form["clip_from"] and form["clip_to"]):
            problems.append("A clip needs both ends.")
        elif not rigged:
            problems.append("An animated clip needs a rigged mesh.")
    return problems


def _saved(ctx: Any, job: Any) -> None:
    sheets = (ctx.state.preview or {}).get("sheets") or []
    if not sheets:
        return
    widgets.section("Rendered sheets")
    job_id = job["id"]
    for sheet in sheets:
        sheet_id = sheet["id"]
        imgui.push_id(sheet_id)
        imgui.text(sheet.get("name") or sheet_id)
        widgets.muted(
            f"{len(sheet.get('cells') or [])} cells - {sheet.get('frame_size')} px"
        )
        if imgui.small_button("Save PNG..."):
            _save(ctx, job_id, sheet_id)
        imgui.same_line()
        if imgui.small_button("Save JSON..."):
            _save_sidecar(ctx, job_id, sheet_id)
        imgui.same_line()
        if imgui.small_button("Delete"):
            ctx.submit(
                f"sheet-del:{job_id}:{sheet_id}",
                svc_sheets.delete_sheet,
                ctx.svc,
                job_id,
                sheet_id,
            )
        imgui.pop_id()


def _save(ctx: Any, job_id: str, sheet_id: str) -> None:
    def run():
        source = svc_sheets.sheet_png(ctx.svc, job_id, sheet_id)
        dest = dialogs.save_file(
            "Save sprite sheet", f"{job_id}_{sheet_id}.png", dialogs.PNG_FILTER
        )
        if dest is None:
            return None
        dest.write_bytes(source.read_bytes())
        return dest

    ctx.submit(f"sheet-save:{job_id}:{sheet_id}", run)


def _save_sidecar(ctx: Any, job_id: str, sheet_id: str) -> None:
    """Export the cell map beside the PNG.

    The sheet's contract with an importer is which cell holds which yaw, pose
    and frame -- a bare PNG is a grid nothing can address. The record was
    written, read back and displayed, but had no way out of the app.
    """
    import json

    def run():
        record = svc_sheets.get_sheet(ctx.svc, job_id, sheet_id)
        dest = dialogs.save_file(
            "Save sheet data", f"{job_id}_{sheet_id}.json", dialogs.filters_for("x.json")
        )
        if dest is None:
            return None
        dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return dest

    ctx.submit(f"sheet-save:{job_id}:{sheet_id}:json", run)

"""Sprite sheets: the direction preview, the grid, and queuing a render.

The preview strip is a *direction* preview -- it cannot pose the mesh, so
drawing one row per pose would draw the same row N times. The grid the worker
will actually produce is stated as a summary line instead, and that line is the
part that has to agree with ``pipelines.sheet.plan``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from imgui_bundle import imgui

from ... import models, rigging
from ...service import sheets as svc_sheets
from ...service import validation
from .. import dialogs, icons, widgets
from ..manual import render as manual_render
from ..viewer import sheet as sheetlib
from . import model_gate, stamps

log = logging.getLogger(__name__)

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
        # The rendered strip belongs to the same rig the pose ids do. Left
        # cached across a selection change it kept showing the *previous*
        # asset's turnaround under the new asset's controls, with nothing in
        # the panel to say the picture was stale.
        ctx.state.preview.pop("sheet_strip", None)
        # Both are keyed by sheet id, and sheet ids belong to the asset whose
        # directory holds them: carried across a selection change they would
        # describe another job's sheets.
        ctx.state.preview.pop("pixel_forms", None)
        ctx.state.preview.pop("pixel_sidecars", None)
        release_strip_texture(ctx)
    return form


def _preview(ctx: Any, form: dict[str, Any]) -> None:
    """The direction strip, rendered one cell per frame.

    Every cell is a draw *and* a synchronous GPU-to-CPU readback, and the
    control offers up to sixteen -- so doing them all in the frame the button
    was pressed was a visible freeze. Spread out it is a quarter of a second
    nobody notices. It cannot leave the frame thread: this is the one GL
    context, so what changes is how much happens per frame, not where.

    (There used to be a ``ctx.busy("sheet-preview")`` here guarding exactly
    this, against a key nothing ever submitted -- scaffolding for the version
    that never arrived. The progress is real state on the viewer now.)
    """
    viewer = ctx.viewer
    if viewer is None or not viewer.has_model:
        return
    if widgets.disabled_button("Refresh preview", not viewer.stripping):
        yaws = [i * 360.0 / form["yaws"] for i in range(form["yaws"])]
        try:
            viewer.begin_sheet_strip(
                yaws, math.radians(form["elevation"]), form["lighting"] == "flat"
            )
        except Exception:
            # Logged as well as toasted, like every comparable site: an
            # offscreen render that fails leaves nothing in warlock.log
            # otherwise, and "could not render" is not a diagnosis.
            log.exception("could not render the sheet preview")
            ctx.toast("Could not render the preview.", "error")
            viewer.cancel_sheet_strip()
    if viewer.stripping:
        imgui.same_line()
        widgets.spinner()
        # L104: a spinner alone says "something is happening" for as many
        # frames as there are cells, which at 32 yaws is half a second of no
        # information. The count is already on the renderer.
        done, total = viewer.strip_progress
        if total:
            imgui.same_line()
            widgets.muted(f"{done} of {total}")
        try:
            finished = viewer.step_sheet_strip()
        except Exception:
            log.exception("could not render the sheet preview")
            ctx.toast("Could not render the preview.", "error")
            viewer.cancel_sheet_strip()
            finished = None
        if finished is not None:
            ctx.state.preview["sheet_strip"] = finished
    strip = ctx.state.preview.get("sheet_strip")
    # No ``ctx.textures`` gate: that is the *thumbnail cache*, and nothing on
    # this path touches it. ``_strip_texture`` allocates through
    # ``ctx.viewer.ctx`` and ``widgets.texture_ref`` registers with the imgui
    # backend, so the cache's absence said nothing about whether this could
    # draw -- it just silently left the preview blank.
    if strip is not None:
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
    picked = widgets.labeled_combo("Frame", str(form["frame_size"]), sizes)
    widgets.help_marker(
        "The size of one cell in the atlas, in pixels. The grid is worked out "
        "from it and the number of directions, and wraps to stay inside a "
        "texture an engine will accept."
    )
    form["frame_size"] = int(picked)
    form["lighting"] = widgets.labeled_combo(
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
            # The pose id, not a fixed "##row": two poses may share a name, and
            # two checkboxes with one imgui id are one checkbox.
            # The returned value is deliberately unread: the set *is* the
            # state, so the toggle is applied to it rather than copied out of
            # the widget.
            hit, _value = imgui.checkbox(
                f"{pose.get('name') or pose['id']}##row-{pose['id']}", checked
            )
            if hit:
                form["poses"].symmetric_difference_update({pose["id"]})
        changed, clip = imgui.checkbox("Animated clip", form["clip"])
        widgets.help_marker(
            "Interpolates between two saved poses. Each frame becomes more "
            "cells in the same atlas rather than a second file."
        )
        if changed:
            form["clip"] = clip
        if form["clip"]:
            # A clip replaces the pose rows rather than adding to them: its
            # rows *are* the animation, and mixing static poses in would give
            # an importer no way to tell which rows loop.
            names = [(p["id"], p.get("name") or p["id"]) for p in poses]
            form["clip_from"] = widgets.labeled_combo("From", form["clip_from"], names)
            form["clip_to"] = widgets.labeled_combo("To", form["clip_to"], names)
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
    # The line above says what comes out; this says what it costs (S138). The
    # cell count is the honest unit -- each one is a separate Blender render --
    # and it is already computed here, so the sentence carries a real number
    # rather than a wall-clock guess about someone else's machine.
    widgets.cost_note(
        f"Queued like a generation: {rows * form['yaws']} Blender renders, "
        "waiting behind anything already running."
    )
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
    widgets.section("Rendered sheets")
    if not sheets:
        # H73. Returning early left the section itself absent, so a user who
        # had rendered a sheet yesterday and could not find it had no way to
        # tell "none for this asset" from "the panel forgot how".
        widgets.empty_state(
            icons.GRID,
            "No sheets for this asset",
            "Render one above; it is saved beside the mesh.",
        )
        return
    job_id = job["id"]
    for sheet in sheets:
        sheet_id = sheet["id"]
        imgui.push_id(sheet_id)
        imgui.text(sheet.get("name") or sheet_id)
        widgets.muted(
            f"{len(sheet.get('cells') or [])} cells - {sheet.get('frame_size')} px"
        )
        # Three buttons in a sidebar, two of them long. Chained with a bare
        # same_line they ran off the content edge on a narrow panel and Delete
        # was clipped away -- pose_panel's row, with the same fix.
        if imgui.small_button("Save PNG..."):
            _save(ctx, job_id, sheet_id)
        widgets.same_line_or_wrap(widgets.button_width("Save JSON..."))
        if imgui.small_button("Save JSON..."):
            _save_sidecar(ctx, job_id, sheet_id)
        widgets.same_line_or_wrap(widgets.button_width("Delete"))
        if imgui.small_button("Delete"):
            _ask_delete(ctx, job_id, sheet)
        _pixelate(ctx, job_id, sheet)
        imgui.pop_id()


def _ask_delete(ctx: Any, job_id: str, sheet: dict[str, Any]) -> None:
    """Confirm before a rendered sheet goes.

    The cost is the part that is not obvious from a row that reads like a list
    entry: a sheet is one Blender render per cell, so re-rendering it is
    minutes of CPU rather than a click, and any restyled copy derived from it
    goes with it.

    Takes the sheet alone: the id used to come in beside it as a second
    argument, which let a caller pass a sheet and *another* sheet's id and
    delete the wrong one.
    """
    sheet_id = sheet["id"]
    cells = len(sheet.get("cells") or [])
    name = sheet.get("name") or sheet_id
    dialogs.ask_delete(
        ctx,
        title="Delete this sheet?",
        message=(
            f'"{name}" and its {cells} rendered cell(s) are deleted, along with '
            "anything restyled from it. This cannot be undone.\n\n"
            "Rendering it again is one Blender render per cell."
        ),
        on_confirm=lambda: ctx.submit(
            f"sheet-del:{job_id}:{sheet_id}",
            svc_sheets.delete_sheet,
            ctx.svc,
            job_id,
            sheet_id,
        ),
    )


def _pixel_form(ctx: Any, sheet_id: str) -> dict[str, Any]:
    """One restyle form per sheet, kept on the app state like the render's.

    Keyed by sheet id rather than by job, because a job holds several sheets
    and each one is restyled on its own terms -- and dropped when the selection
    moves, since a sheet id names a grid that another asset does not have.
    """
    forms = ctx.state.preview.setdefault("pixel_forms", {})
    form = forms.get(sheet_id)
    if form is None:
        defaults = (svc_sheets.pixel_sheet_options() or {}).get("defaults") or {}
        form = {
            "logical_size": int(defaults.get("logical_size") or 32),
            "colors": int(defaults.get("colors") or 32),
            "strength": float(defaults.get("strength") or 0.45),
            "structure_lock": bool(defaults.get("structure_lock", True)),
            "seed": validation.random_seed(),
        }
        forms[sheet_id] = form
    return form


def _pixel_sizes(frame_size: int) -> list[int]:
    """The offered logical sizes that actually divide this sheet's cells.

    Filtered rather than refused after the fact: a size that does not divide
    the frame exactly cannot be reduced without resampling across cell
    boundaries, and offering it would be offering a button that fails.
    """
    return [
        size
        for size in svc_sheets.PIXEL_LOGICAL_SIZES
        if frame_size % size == 0 and size <= frame_size
    ]


def _pixelate(ctx: Any, job_id: str, sheet: Any) -> None:
    """The restyle controls for one rendered sheet.

    A disclosure rather than a second panel: it belongs to the sheet it
    restyles, and the pair it produces lives beside that sheet's own files.
    """
    from ...pipelines import pixelsheet

    sheet_id = sheet["id"]
    if not imgui.tree_node("Pixelate"):
        return
    try:
        frame_size = int(sheet.get("frame_size") or 0)
        columns = int(sheet.get("columns") or 0)
        width = frame_size * columns
        if width > pixelsheet.BAND_PX:
            # Said here rather than under a disabled button with no reason:
            # the remedy is to re-render, which is a different control.
            widgets.muted(
                f"{width}px wide -- a restyle generates "
                f"{pixelsheet.BAND_PX}px at a time. Re-render at "
                f"{pixelsheet.BAND_PX // max(columns, 1)}px or smaller."
            )
            return
        sizes = _pixel_sizes(frame_size)
        if not sizes:
            widgets.muted(f"no pixel size divides {frame_size}px cells exactly")
            return

        form = _pixel_form(ctx, sheet_id)
        if form["logical_size"] not in sizes:
            form["logical_size"] = sizes[-1]
        form["logical_size"] = int(
            widgets.labeled_combo(
                "Pixel size",
                str(form["logical_size"]),
                [(str(s), f"{s} px") for s in sizes],
            )
        )
        form["colors"] = int(
            widgets.labeled_combo(
                "Colours",
                str(form["colors"]),
                [(str(n), f"{n} colours") for n in svc_sheets.PIXEL_COLOR_CHOICES],
            )
        )
        changed, value = imgui.slider_float(
            "Strength",
            form["strength"],
            models.IMG2IMG_STRENGTH_MIN,
            models.IMG2IMG_STRENGTH_MAX,
        )
        if changed:
            form["strength"] = float(value)
        toggled, locked = imgui.checkbox("Lock silhouettes", form["structure_lock"])
        if toggled:
            form["structure_lock"] = bool(locked)
        imgui.text(f"Seed {form['seed']}")
        imgui.same_line()
        if imgui.small_button("Reroll"):
            form["seed"] = validation.random_seed()

        key = f"sheet-pixel:{job_id}:{sheet_id}"
        busy = ctx.busy(key)
        # Ahead of the button, for ``model_gate``'s reason: the restyle is
        # pinned to one base and one style LoRA, and neither is in the core
        # download.
        locked = model_gate.draw(
            ctx, svc_sheets.PIXEL_SHEET_ROWS, what="A pixel restyle"
        )
        if busy:
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button("Pixelate", not busy and not locked):
            ctx.submit(
                key,
                svc_sheets.create_pixel_sheet,
                ctx.svc,
                job_id,
                sheet_id,
                logical_size=form["logical_size"],
                colors=form["colors"],
                strength=form["strength"],
                seed=form["seed"],
                structure_lock=form["structure_lock"],
            )
        _pixel_result(ctx, job_id, sheet_id)
    finally:
        # In a finally because every early return above is inside the node.
        imgui.tree_pop()


def pixel_record(ctx: Any, job_id: str, sheet_id: str) -> dict[str, Any] | None:
    """This sheet's pixel sidecar, parsed once per version of the file.

    Keyed on the sidecar, never on the PNG: the worker writes the image first,
    so the PNG alone can be a file still being written.

    One stat per frame and a parse only when the file changes -- the same idiom
    ``inspector._manifest`` uses, and for the same reason: this is drawn sixty
    times a second while the worker rewrites the file underneath it. Which is
    also why the remembering goes through :mod:`.stamps`: "Pixelate" pressed
    twice rewrites *this* sidecar, and a rewrite landing inside the stamped
    mtime's own 15.6 ms tick matches that stamp forever -- so the panel would
    advertise the previous restyle's cell size and palette against the new PNG,
    with no way back short of restarting the app.

    Split from the drawing half so the cache is assertable without a GL context,
    which is the split ``blender_worker._rig_bones`` is: the rule worth pinning
    is the one an imgui frame cannot reach.
    """
    path = rigging.sheet_pixel_path(ctx.job_dir(job_id), sheet_id)
    stamp = stamps.stamp_ns(path)
    if stamp is None:
        return None
    cache = ctx.state.preview.setdefault("pixel_sidecars", {})
    cached = cache.get(sheet_id)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    record = rigging.read_sheet_pixel(ctx.job_dir(job_id), sheet_id)
    if stamps.storable(stamp):
        cache[sheet_id] = (stamp, record)
    return record


def _pixel_result(ctx: Any, job_id: str, sheet_id: str) -> None:
    """The finished restyle, if the sidecar has landed."""
    record = pixel_record(ctx, job_id, sheet_id)
    if record is None:
        return
    widgets.muted(
        f"{record.get('frame_size')} px cells - "
        f"{len(record.get('palette') or [])} colours"
    )
    if imgui.small_button("Save pixel PNG..."):
        _save_pixel(ctx, job_id, sheet_id)
    imgui.same_line()
    if imgui.small_button("Save pixel JSON..."):
        _save_pixel_sidecar(ctx, job_id, sheet_id)


def _save_pixel(ctx: Any, job_id: str, sheet_id: str) -> None:
    def run():
        source = svc_sheets.sheet_pixel_png(ctx.svc, job_id, sheet_id)
        dest = dialogs.save_file(
            "Save pixel sheet", f"{job_id}_{sheet_id}_pixel.png", dialogs.PNG_FILTER
        )
        if dest is None:
            return None
        dest.write_bytes(source.read_bytes())
        return dest

    ctx.submit(f"sheet-save:{job_id}:{sheet_id}:pixel", run)


def _save_pixel_sidecar(ctx: Any, job_id: str, sheet_id: str) -> None:
    import json

    def run():
        record = svc_sheets.get_pixel_sheet(ctx.svc, job_id, sheet_id)
        dest = dialogs.save_file(
            "Save pixel sheet data",
            f"{job_id}_{sheet_id}_pixel.json",
            dialogs.filters_for("x.json"),
        )
        if dest is None:
            return None
        dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return dest

    ctx.submit(f"sheet-save:{job_id}:{sheet_id}:pixeljson", run)


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

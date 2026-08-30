"""Remesh a finished mesh to a quad budget and rebake its surface.

The inspector's third rework panel, between the triangle budget (which keeps
the reconstruction's surface and simplifies it) and the surface texture (which
keeps the geometry and repaints it). This one replaces both: quadriflow to a
face count, a fresh unwrap, and the old colour, roughness and normals baked
onto the new mesh. It is the step that turns a reconstruction into something an
engine budgets for, and it is what every commercial generator sells as
"game-ready".

Two things it does not hide, on ``retarget_panel``'s model. It runs in Blender,
so without the ``rig`` extra the panel says so and offers nothing. And it makes
a rig, its poses and its sheets describe a mesh that no longer exists -- the
service reports them rather than deleting them, and the warning is drawn
*before* the button.
"""

from __future__ import annotations

from typing import Any

from ...pipelines import remesh
from ...service import jobs as svc_jobs
from .. import controls, forms, theme, widgets
from ..manual import render as manual_render

#: ``remesh.FACE_PROFILES`` first, then the free-form entry -- membership
#: derived, so a budget added to the pipeline appears here without an edit.
PROFILES: tuple[tuple[str, str], ...] = tuple(
    (key, remesh.profile_label(key)) for key in remesh.FACE_PROFILES
) + (("custom", remesh.profile_label("custom")),)

TEXTURES: tuple[tuple[str, str], ...] = (("", "Match the mesh"),) + tuple(
    (str(s), f"{s} px") for s in remesh.TEXTURE_SIZES
)


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "model.glb" not in files:
        return
    if not widgets.header("Game-ready remesh", default_open=False):
        return
    manual_render.help_button(ctx, "remesh")

    if not _blender_available(ctx):
        widgets.muted("A remesh runs in Blender, which is not installed (the rig extra).")
        return

    job_id = job["id"]
    form = _form(ctx, job_id)
    with forms.Form("remesh-settings", errors=ctx.state.field_errors) as form_ui:
        _changed, form["remesh_profile"] = form_ui.combo(
            "remesh_profile",
            "Quads",
            form["remesh_profile"],
            PROFILES,
            help_text="Rebuild the surface as quads at this budget, then bake the "
            "old colour, roughness and normals onto it.",
            helper="A reconstruction is ~300k triangles; a prop ships at 2–8k quads.",
        )
        if form["remesh_profile"] == "custom":
            changed, value = form_ui.number(
                "custom_faces",
                "Faces",
                int(form["custom_faces"]),
                helper=f"{remesh.FACES_MIN:,} to {remesh.FACES_MAX:,}",
            )
            if changed:
                form["custom_faces"] = value
        _changed, form["texture_size"] = form_ui.combo(
            "texture_size",
            "Bake at",
            form["texture_size"],
            TEXTURES,
            help_text="Resolution of the rebaked textures.",
        )
        _changed, form["close_holes"] = controls.checkbox(
            "Close holes first", form["close_holes"]
        )
        widgets.help_marker(
            "A voxel pass before the remesh seals the gaps a reconstruction "
            "leaves, at the cost of slightly rounding sharp edges."
        )
        _warn_stale(ctx, job)
        _submit(ctx, job_id, form)

    line = remesh.report_line((job.get("params") or {}).get("remesh"))
    if line:
        widgets.muted(f"Last remesh: {line}")


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    form = ctx.state.preview.get("remesh_form")
    if form is None or form.get("job_id") != job_id:
        form = {
            "job_id": job_id,
            "remesh_profile": remesh.DEFAULT_PROFILE,
            "custom_faces": remesh.FACE_PROFILES[remesh.DEFAULT_PROFILE],
            "texture_size": "",
            "close_holes": False,
        }
        ctx.state.preview["remesh_form"] = form
    return form


def _blender_available(ctx: Any) -> bool:
    """The rig door's own answer, unprobed: ``blender_check(probe=False)``
    returns the cached verdict or a pending row, and a pending row draws as
    "not yet" rather than blocking a frame on a subprocess."""
    try:
        from ... import doctor

        return bool(doctor.blender_check(probe=False).ok)
    except Exception:  # noqa: BLE001 - no doctor answer means no panel
        return False


def _warn_stale(ctx: Any, job: Any) -> None:
    files = set(job.get("files") or [])
    if "rig.glb" in files:
        widgets.text_colored(
            theme.WARN, "The rig, its poses and its sheets will describe the old mesh."
        )


def validate(form: dict[str, Any]) -> list[str]:
    """The refusals stated before the button; ``remesh.resolve`` is the rule."""
    try:
        remesh.resolve(
            form["remesh_profile"],
            int(form["custom_faces"]) if form["remesh_profile"] == "custom" else None,
        )
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments ``remesh_job`` is called with, from the form."""
    return {
        "profile": form["remesh_profile"],
        "custom_faces": (
            int(form["custom_faces"]) if form["remesh_profile"] == "custom" else None
        ),
        "texture_size": int(form["texture_size"]) if form["texture_size"] else None,
        "close_holes": bool(form["close_holes"]),
    }


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"remesh:{job_id}"
    busy = ctx.busy(key)
    problems = validate(form)
    for problem in problems:
        widgets.muted(problem)
    if busy:
        widgets.busy("Queueing the remesh")
    if widgets.disabled_button(
        "Remesh and rebake",
        not problems and not busy,
        (-1, 0),
        reason="A remesh is already being queued for this asset."
        if busy
        else "; ".join(problems),
    ):
        ctx.submit(key, svc_jobs.remesh_job, ctx.svc, job_id, **submit_kwargs(form))

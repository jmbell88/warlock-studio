"""Give a finished mesh a new surface, from a prompt.

The retarget panel's sibling and it is placed beside it deliberately: both are
"an operation on a selected done job", which is the shape the mode list is
closed against. A re-texture is not a mode, it is a button on an asset.

Where the two differ is what they cost and what they invalidate, and the panel
says both before the button rather than after. A retarget is a couple of seconds
of gltfpack; this queues six SDXL passes around two Blender runs, so it is a
job with a place in the queue. And a retarget makes a rig, its poses and its
sheets describe a mesh that no longer exists, where a re-texture makes none of
them stale -- a rig references geometry, not pixels. What it *does* invalidate
is the exports that carry the skin, and the service names them.

The one limitation it will not paper over is the bake's own: there is no
occlusion test, so an overhang's front-view colours are smeared onto whatever
hides behind it, and the coverage figure a finished run reports is how much of
the atlas any view could speak about at all. Both are on screen, because that
is the finding a user needs in order to judge the result rather than be
surprised by it.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ... import models
from ...pipelines import retexture
from ...service import jobs as svc_jobs
from ...service.validation import MAX_PROMPT
from .. import theme, widgets
from ..manual import render as manual_render


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "model.glb" not in files:
        # Nothing to re-skin. A reference, a tile and a job that never produced
        # a mesh are all in this state.
        return
    if not widgets.header("Surface texture", default_open=False):
        return
    manual_render.help_button(ctx, "retexture")

    job_id = job["id"]
    form = _form(ctx, job_id)

    form["prompt"] = widgets.multiline(
        "##retexture_prompt", form["prompt"], 66, MAX_PROMPT
    )
    widgets.hint_text("Describe the surface: rusted iron, mossy stone, painted wood.")

    _changed, form["strength"] = widgets.labeled_slider_float(
        "Restyle strength",
        float(form["strength"]),
        models.IMG2IMG_STRENGTH_MIN,
        models.IMG2IMG_STRENGTH_MAX,
    )
    widgets.help_marker(
        "How far each rendered view is taken from the mesh's current look. Low "
        "keeps the shapes and recolours them; high reinterprets them."
    )
    form["texture_size"] = widgets.combo(
        "Atlas size",
        form["texture_size"],
        [("", "Match the mesh")]
        + [(str(s), f"{s} px") for s in retexture.TEXTURE_SIZES],
    )
    widgets.help_marker(
        "The default keeps the mesh's current atlas resolution. A smaller one "
        "would also shrink the parts no view covers, which keep their old "
        "colour."
    )

    _warn(ctx, job)
    _submit(ctx, job_id, form)
    _report(job)


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    """Kept on app state so it survives a frame, rebuilt per job so a prompt
    typed against one mesh is never submitted against another -- the retarget
    panel's rule, and for the sharper version of its reason: a surface
    description is a sentence somebody wrote, not a tier they clicked."""
    form = ctx.state.preview.get("retexture_form")
    if form is None or form.get("job_id") != job_id:
        form = {
            "job_id": job_id,
            "prompt": "",
            "strength": models.DEFAULT_IMG2IMG_STRENGTH,
            "texture_size": "",
        }
        ctx.state.preview["retexture_form"] = form
    return form


def _warn(ctx: Any, job: Any) -> None:
    """What this will invalidate, and what it deliberately will not."""
    stale = svc_jobs.stale_surface_artifacts(ctx.svc.job_dir(job["id"]))
    if stale:
        widgets.text_colored(
            theme.WARN, "These exports will be rebuilt: " + ", ".join(stale)
        )
    if "rig.glb" in set(job.get("files") or []):
        # Said out loud rather than left as an absence: the retarget panel two
        # headers up warns about exactly these, so silence here would read as an
        # oversight rather than as the answer.
        widgets.muted("The rig, its poses and its sheets are unaffected.")
    widgets.muted(
        "No occlusion test: colours from a facing view can reach surfaces "
        "hidden behind an overhang."
    )


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"retexture:{job_id}"
    busy = ctx.busy(key)
    problems = validate(form)
    for problem in problems:
        widgets.muted(problem)
    if busy:
        widgets.spinner()
        imgui.same_line()
    if widgets.disabled_button("Re-texture mesh", not problems and not busy, (-1, 0)):
        ctx.submit(
            key,
            svc_jobs.retexture_job,
            ctx.svc,
            job_id,
            form["prompt"].strip(),
            strength=float(form["strength"]),
            texture_size=int(form["texture_size"]) or None,
        )


def _report(job: Any) -> None:
    """What the last re-texture of this mesh actually managed.

    The coverage figure is the honest half of the missing occlusion test: a low
    number means most of the mesh kept its old skin, which is a thing to be told
    rather than to discover by looking for it.
    """
    report = (job.get("params") or {}).get("retexture")
    if not isinstance(report, dict):
        return
    coverage = report.get("coverage")
    views = report.get("views")
    if coverage is None or views is None:
        return
    widgets.muted(
        f"Last run: {float(coverage) * 100:.0f}% of the atlas covered by "
        f"{int(views)} views."
    )


def validate(form: dict[str, Any]) -> list[str]:
    """The refusals stated before the button, matching what the service checks.

    Not the service's numeric ranges: those are enforced by the controls above,
    which cannot express an out-of-range value. What a control *can* express is
    an empty prompt, so that is what is stated.
    """
    if not form["prompt"].strip():
        return ["Describe the surface you want."]
    return []

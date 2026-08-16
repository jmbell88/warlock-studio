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

The bake's one limitation is stated per run rather than papered over: an
*un-anchored* run has no occlusion test, so an overhang's front-view colours
smear onto whatever hides behind it -- that is the warning, shown exactly when
it is true. The "anchor to geometry" checkbox turns on the depth pass: every
restyle is held to the mesh's own rendered depth through a ControlNet, and the
backprojection depth-tests each texel, so the smear goes and coverage comes
from what the cameras genuinely saw. The report line says which kind of run
produced it, because the two coverage figures do not mean the same thing.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ... import models
from ...pipelines import retexture
from ...service import jobs as svc_jobs
from ...service.validation import MAX_PROMPT
from .. import controls, theme, widgets
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
        models.RETEXTURE_STRENGTH_MIN,
        # The re-texture's own ceiling, not the sheets': see models.py.
        models.RETEXTURE_STRENGTH_MAX,
        # A denoising fraction, drawn as the percentage the reader thinks in.
        # The range does not start at zero, so the inference cannot see it.
        percent=True,
    )
    widgets.help_marker(
        "How far each rendered view is taken from the mesh's current look. Low "
        "keeps the shapes and recolours them; high reinterprets them. The top "
        "of the range is only worth it with the geometry anchor on."
    )
    _changed, form["depth"] = controls.checkbox(
        "Anchor to geometry (depth)",
        bool(form["depth"]),
        tooltip=(
            "Renders the mesh's own depth from every view. Each restyle is "
            "held to it (a depth ControlNet), and colours stop reaching "
            "surfaces hidden behind overhangs (a per-texel depth test). "
            "Needs the depth ControlNet from Settings -> Models."
        ),
    )
    if form["depth"]:
        _changed, form["control_scale"] = widgets.labeled_slider_float(
            "Anchor strength",
            float(form["control_scale"]),
            models.CONTROL_SCALE_MIN,
            models.CONTROL_SCALE_MAX,
        )
        widgets.help_marker(
            "How firmly each restyle is held to the rendered depth. The "
            "default is the model's own."
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

    _warn(ctx, job, form)
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
            # Off until the retexture-visibility measurement flips it: the
            # anchored run is the new path, and a default the corpus has no
            # numbers for yet is a default on somebody else's evidence.
            "depth": False,
            "control_scale": models.CONTROLNETS["depth"].default_scale,
        }
        ctx.state.preview["retexture_form"] = form
    return form


def occlusion_note(depth_on: bool) -> str | None:
    """The overhang warning, exactly when it is true.

    A property of the run being configured rather than of the feature: an
    anchored run depth-tests every texel, and warning anyway would teach the
    user the checkbox does nothing.
    """
    if depth_on:
        return None
    return (
        "No occlusion test: colours from a facing view can reach surfaces "
        "hidden behind an overhang. Anchoring to geometry removes this."
    )


def _warn(ctx: Any, job: Any, form: dict[str, Any]) -> None:
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
    note = occlusion_note(bool(form["depth"]))
    if note:
        widgets.muted(note)


def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments ``retexture_job`` is called with, from the form.

    A pure function because the conversion had a crash in it nothing could
    see: ``int("")`` raises, and ``""`` is exactly what the default "Match the
    mesh" combo option holds -- so the untouched form blew up in the draw
    frame before ``ctx.submit`` ever ran. The control keys are *absent* rather
    than ``None`` when the anchor is off, matching what the door stores.
    """
    kwargs: dict[str, Any] = {
        "strength": float(form["strength"]),
        "texture_size": int(form["texture_size"]) if form["texture_size"] else None,
    }
    if form.get("depth"):
        kwargs["control"] = "depth"
        kwargs["control_scale"] = float(form["control_scale"])
    return kwargs


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
            **submit_kwargs(form),
        )


def report_line(report: Any) -> str | None:
    """One sentence about what the last run measured, or None.

    Anchored and un-anchored coverage do not mean the same thing -- one is
    "what the cameras genuinely saw", the other counts smear -- so the line
    says which kind of run produced it rather than leaving one number to be
    read as the other.
    """
    if not isinstance(report, dict):
        return None
    coverage = report.get("coverage")
    views = report.get("views")
    if coverage is None or views is None:
        return None
    line = (
        f"Last run: {float(coverage) * 100:.0f}% of the atlas covered by "
        f"{int(views)} views"
    )
    effective = report.get("coverage_effective")
    if report.get("occlusion_tested") and effective is not None:
        return f"{line}, {float(effective) * 100:.0f}% repainted (depth-tested)."
    return line + " (no occlusion test)."


def _report(job: Any) -> None:
    """What the last re-texture of this mesh actually managed.

    The coverage figure is the honest half: a low number means most of the
    mesh kept its old skin, which is a thing to be told rather than to
    discover by looking for it.
    """
    line = report_line((job.get("params") or {}).get("retexture"))
    if line:
        widgets.muted(line)


def validate(form: dict[str, Any]) -> list[str]:
    """The refusals stated before the button, matching what the service checks.

    Not the service's numeric ranges: those are enforced by the controls above,
    which cannot express an out-of-range value. What a control *can* express is
    an empty prompt, so that is what is stated.
    """
    if not form["prompt"].strip():
        return ["Describe the surface you want."]
    return []

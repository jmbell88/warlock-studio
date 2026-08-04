"""Rebuild a finished mesh at a different triangle budget.

``source.glb`` is the reconstruction and nothing ever overwrites it, which is
the whole reason a retarget is cheap: ``model.glb`` is derived from it by
optimize-then-normalize, so a new budget costs a two-second gltfpack run rather
than another two minutes of trellis. The service has done this correctly for a
long time -- the 409 gating, the derived-artifact sweep, the reapplied grounding
transform, the stale-rig report -- and had no caller once the HTTP API was
removed. This is that caller.

Two things it refuses to hide. Every named tier needs ``vendor/gltfpack``,
which is not shipped, so without the binary only ``raw`` is offered and the
reason is on screen. And a retarget makes a rig, its poses and its sheets
describe a mesh that no longer exists; the service reports them rather than
deleting them, so the report is shown *before* the button, not after.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...pipelines import optimize
from ...service import jobs as svc_jobs
from .. import theme, widgets
from ..manual import render as manual_render

# Named tiers coarsest-first, then the free-form one. "raw" is the identity --
# the full reconstruction density -- and is the only entry that needs no
# binary, which is why it is always offered.
TIERS = (
    ("raw", "Raw (full density)"),
    ("draft", "Draft (20k)"),
    ("standard", "Standard (50k)"),
    ("detailed", "Detailed (100k)"),
    ("custom", "Custom..."),
)


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "source.glb" not in files:
        # No reconstruction to rebuild from. Older jobs and rig jobs are both
        # in this state, and neither can be retargeted at any budget.
        return
    if not widgets.header("Triangle budget", default_open=False):
        return
    manual_render.help_button(ctx, "retarget")

    job_id = job["id"]
    form = _form(ctx, job_id)
    available = _gltfpack_available(ctx)

    options = list(TIERS) if available else [TIERS[0]]
    if not available:
        form["profile"] = "raw"
        widgets.muted("Only full density is available: gltfpack is not installed.")
    form["profile"] = widgets.combo("Budget", form["profile"], options)

    if form["profile"] == "custom":
        imgui.set_next_item_width(140)
        changed, value = imgui.input_int("Triangles", int(form["custom_triangles"]), 0, 0)
        if changed:
            form["custom_triangles"] = value
        widgets.muted(f"{optimize.CUSTOM_MIN:,} to {optimize.CUSTOM_MAX:,}")

    _warn_stale(ctx, job)
    _submit(ctx, job_id, form)


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    """Kept on app state so it survives a frame, rebuilt per job so a budget
    typed against one mesh is not submitted against another."""
    form = ctx.state.preview.get("retarget_form")
    if form is None or form.get("job_id") != job_id:
        form = {
            "job_id": job_id,
            "profile": "raw",
            "custom_triangles": optimize.PROFILES["standard"],
        }
        ctx.state.preview["retarget_form"] = form
    return form


def _gltfpack_available(ctx: Any) -> bool:
    try:
        return bool(ctx.svc.config.gltfpack_exe.exists())
    except Exception:
        return False


def _warn_stale(ctx: Any, job: Any) -> None:
    """Name the user work a retarget will invalidate, before it happens.

    The service reports these rather than deleting them -- a rig and its poses
    are minutes of work and must not be destroyed over a triangle count -- but
    reporting after the fact is only half of it.
    """
    files = set(job.get("files") or [])
    if "rig.glb" not in files:
        return
    widgets.text_colored(
        theme.WARN, "The rig, its poses and its sheets will describe the old mesh."
    )


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"retarget:{job_id}"
    busy = ctx.busy(key)
    problems = validate(form)
    for problem in problems:
        widgets.muted(problem)
    if busy:
        widgets.spinner()
        imgui.same_line()
    if widgets.disabled_button("Rebuild mesh", not problems and not busy, (-1, 0)):
        ctx.submit(
            key,
            svc_jobs.optimize_job,
            ctx.svc,
            job_id,
            profile=form["profile"],
            custom_triangles=(
                int(form["custom_triangles"]) if form["profile"] == "custom" else None
            ),
        )


def validate(form: dict[str, Any]) -> list[str]:
    """The refusals stated before the button, matching what the service checks.

    ``optimize.resolve`` is the authority on the range; restating the numbers
    here rather than the rule would let the two drift.
    """
    if form["profile"] != "custom":
        return []
    try:
        optimize.resolve("custom", int(form["custom_triangles"]))
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []

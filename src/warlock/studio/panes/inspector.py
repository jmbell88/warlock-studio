"""The selected asset: what it is, what it cost, and what you can take away.

The downloads section is the interesting part. Eight artifacts, three states
each -- ready, derivable, impossible -- and the difference matters: a greyed
"FBX" with "needs Blender" under it is information, while a missing one is a
mystery.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...service import derive as svc_derive
from ...service import jobs as svc_jobs
from ...service import system as svc_system
from .. import theme, widgets
from ..state import format_duration
from . import pose_panel, sheet_panel

# Matches the library card's thumbnail, so the two read as the same kind of
# object rather than as two different image widgets.
THUMB_SIZE = 96


def draw(ctx: Any) -> None:
    job = ctx.job()
    if job is None:
        widgets.muted("Select an asset.")
        return

    _header(ctx, job)
    _meta(ctx, job)
    if job.get("status") == "error":
        _error(ctx, job)
    _settings(ctx, job)
    _reference(ctx, job)
    if ctx.state.mode == "3d":
        _quality(ctx, job)
        pose_panel.draw(ctx, job)
        sheet_panel.draw(ctx, job)
    _downloads(ctx, job)


# --- pieces -----------------------------------------------------------------


def _header(ctx: Any, job: Any) -> None:
    job_id = job["id"]
    name = widgets.input_text("##name", job.get("name") or "", max_length=120, hint="Name")
    if name != (job.get("name") or ""):
        ctx.submit(f"name:{job_id}", svc_jobs.update_job, ctx.svc, job_id, {"name": name})
    tags = widgets.input_text("##tags", job.get("tags") or "", max_length=400, hint="tags, comma")
    if tags != (job.get("tags") or ""):
        ctx.submit(
            f"tags:{job_id}", svc_jobs.update_job, ctx.svc, job_id, {"tags": tags.split(",")}
        )
    widgets.status_pill(job["status"])
    imgui.same_line()
    favourite = bool(job.get("favorite"))
    if imgui.small_button("Unfavourite" if favourite else "Favourite"):
        ctx.submit(
            f"fav:{job_id}", svc_jobs.update_job, ctx.svc, job_id, {"favorite": not favourite}
        )


def _meta(ctx: Any, job: Any) -> None:
    widgets.muted(f"{job['id']} - {job.get('kind')} - {job.get('stage')}")
    if job.get("created_at"):
        widgets.muted(str(job["created_at"]))
    if ctx.viewer is not None and ctx.viewer.has_model and ctx.state.mode == "3d":
        stats = ctx.viewer.stats()
        if stats:
            size = stats["size"]
            # Counted off what is actually loaded, not read from mesh_report:
            # the viewer may be showing rig.glb or a baked pose, and the number
            # under the model should describe the model under it.
            widgets.muted(
                f"{stats['triangles']:,} tris - {stats['vertices']:,} verts - "
                f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m"
            )


def _error(ctx: Any, job: Any) -> None:
    widgets.text_colored(theme.ERR, job.get("error") or "It failed.")
    if not imgui.tree_node("Details"):
        return
    key = "trellis-log"
    if widgets.disabled_button("Read the trellis log", not ctx.busy(key)):
        ctx.submit(key, svc_system.trellis_log, ctx.svc)
    log_text = ctx.state.preview.get("trellis_log") if ctx.state.preview else None
    if log_text:
        imgui.input_text_multiline(
            "##log", log_text[-4000:], (-1, 160), imgui.InputTextFlags_.word_wrap.value
        )
    if "error.log" in (job.get("files") or []) and imgui.button("Save error.log..."):
        ctx.save_artifact(job["id"], "error.log")
    imgui.tree_pop()


def _settings(ctx: Any, job: Any) -> None:
    params = job.get("params") or {}
    if not params:
        return
    if not widgets.header("Generation settings", default_open=False):
        return
    rows = [
        ("seed", params.get("seed")),
        ("reference seed", params.get("reference_seed")),
        ("mesh seed", params.get("mesh_seed")),
        ("platform", params.get("platform")),
        ("resolution", params.get("resolution")),
        ("size (m)", params.get("size_m")),
        ("model", params.get("base_model")),
        ("style LoRA", params.get("style_lora")),
        ("appearance ref", params.get("ip_adapter")),
        ("appearance strength", params.get("ip_scale")),
        ("structure", params.get("control")),
        ("structure strength", params.get("control_scale")),
        ("background", params.get("bg_removal")),
        ("profile", params.get("profile")),
    ]
    for label, value in rows:
        if value in (None, ""):
            continue
        widgets.muted(f"{label}: {value}")
    composed = params.get("composed_prompt")
    if composed and imgui.tree_node("Prompt sent"):
        imgui.text_wrapped(str(composed))
        imgui.tree_pop()


def _reference(ctx: Any, job: Any) -> None:
    """What the mesh engine was actually handed, and what it made of it.

    The three images answer the question a bad mesh always raises -- was it
    the reconstruction, or was it what we sent? -- which nothing in the UI
    could answer before: trellis was streamed input.png verbatim and no
    measurement of it was kept.
    """
    params = job.get("params") or {}
    report = params.get("reference_report")
    files = set(job.get("files") or [])
    shown = [n for n in ("input.png", "ref.png", "reference.png", "control.png") if n in files]
    if not isinstance(report, dict) and len(shown) <= 1:
        return
    if not widgets.header("Reference", default_open=False):
        return

    if isinstance(report, dict):
        for reason in report.get("reasons") or []:
            widgets.text_colored(theme.ERR, reason)
        for warning in report.get("warnings") or []:
            widgets.text_colored(theme.WARN, warning)
        if report.get("occupancy"):
            widgets.muted(f"the subject fills {float(report['occupancy']) * 100:.0f}% of the frame")
        if report.get("measured") is False:
            widgets.muted("not measured")
        elif report.get("normalised"):
            widgets.muted("recentred and rescaled before upload")

    hint = params.get("control_hint")
    if isinstance(hint, dict) and hint.get("edge_fraction") is not None:
        # The number that answers "my silhouette lock did nothing": near zero
        # means the detector found no structure, not that it was ignored.
        widgets.muted(f"{hint['kind']} hint: {float(hint['edge_fraction']) * 100:.1f}% edges")

    if ctx.textures is None:
        return
    for name in shown:
        # Keyed by job *and* file: the cache keys on (id, mtime), so sharing
        # the bare job id across four images would show whichever one was
        # decoded first for all of them.
        texture = ctx.textures.get(f"{job['id']}:{name}", ctx.job_dir(job["id"]) / name)
        if texture is None:
            continue
        widgets.muted(name)
        imgui.image(widgets.texture_ref(texture), (THUMB_SIZE, THUMB_SIZE))


def _quality(ctx: Any, job: Any) -> None:
    params = job.get("params") or {}
    report = params.get("mesh_report")
    audit = params.get("mesh_audit")
    if not (report or audit):
        return
    if not widgets.header("Mesh quality"):
        return
    if isinstance(report, dict):
        widgets.quality_badge(job)
        for reason in report.get("reasons") or []:
            widgets.muted(f"- {reason}")
        # Only the report may use the word watertight: the audit is a
        # silhouette check and never proved it.
        for label, key in (
            ("triangles", "triangles"),
            ("materials", "materials"),
            ("watertight", "watertight"),
            ("pivot at feet", "grounded"),
        ):
            if key in report:
                widgets.muted(f"{label}: {report[key]}")
    if isinstance(audit, dict) and audit.get("hole_ratio") is not None:
        widgets.muted(f"visible openings: {float(audit['hole_ratio']) * 100:.1f}%")


def _downloads(ctx: Any, job: Any) -> None:
    if not widgets.header("Downloads"):
        return
    job_id = job["id"]
    files = set(job.get("files") or [])
    has_mesh = "model.glb" in files
    for name, label in widgets.ARTIFACTS:
        # job["files"] is the sanctioned answer; a raw exists() check here used
        # to re-enable buttons the service would then refuse.
        ready = name in files
        derivable = has_mesh and svc_derive.derivable(name)
        blocked = _why_blocked(ctx, name, ready, derivable)
        key = f"save:{job_id}:{name}"
        busy = ctx.busy(key)
        if busy:
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button(f"{label}##{name}", not blocked and not busy, (-1, 0)):
            ctx.save_artifact(job_id, name)
        if blocked:
            widgets.muted(f"   {blocked}")


def _why_blocked(ctx: Any, name: str, ready: bool, derivable: bool) -> str | None:
    if ready:
        return None
    if name == "model.fbx" and not ctx.rigging_available:
        return "needs Blender"
    if derivable:
        return None
    return "not available for this asset"


def duration(job: Any) -> str:
    return format_duration(job.get("duration"))

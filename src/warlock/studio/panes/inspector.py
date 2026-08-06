"""The selected asset: what it is, what it cost, and what you can take away.

The downloads section is the interesting part. Each artifact has three states
-- ready, derivable, impossible -- and the difference matters: a greyed "FBX"
with "needs Blender" under it is information, while a missing one is a mystery.
*Which* artifacts appear is a function of the job: a reference was being shown
the mesh grid, so all eight of its buttons were permanently greyed, which reads
as a broken asset rather than as a 2D one.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...service import derive as svc_derive
from ...service import jobs as svc_jobs
from ...service import system as svc_system
from .. import theme, widgets
from ..manual import render as manual_render
from ..state import format_duration
from . import pose_panel, retarget_panel, sheet_panel

# Matches the library card's thumbnail, so the two read as the same kind of
# object rather than as two different image widgets.
THUMB_SIZE = 96


def draw(ctx: Any) -> None:
    from .. import icons

    job = ctx.job()
    if job is None:
        widgets.empty_state(icons.BOX, "Select an asset.", "Its details and exports live here.")
        return

    # Identity stays above the tabs: whatever tab is open, the header answers
    # "what am I looking at". Everything below it moved from one nine-header
    # scroll column into three tabs, because reaching the sprite sheet on a
    # rigged mesh meant scrolling past the whole pose editor.
    _header(ctx, job)
    manual_render.help_button(ctx, "inspector")
    _meta(ctx, job)
    if job.get("status") == "error":
        _error(ctx, job)

    if ctx.state.mode == "3d":
        widgets.tab_bar(
            "inspector-tabs",
            [
                ("Details", lambda: _details_tab(ctx, job)),
                ("Rig && Pose", lambda: _rig_tab(ctx, job)),
                ("Export", lambda: _downloads(ctx, job)),
            ],
        )
    else:
        widgets.tab_bar(
            "inspector-tabs",
            [
                ("Details", lambda: _details_tab(ctx, job)),
                ("Export", lambda: _downloads(ctx, job)),
            ],
        )


def _details_tab(ctx: Any, job: Any) -> None:
    _settings(ctx, job)
    _reference(ctx, job)
    _seam(ctx, job)
    if ctx.state.mode == "3d":
        _quality(ctx, job)
        _verdict(ctx, job)


def _rig_tab(ctx: Any, job: Any) -> None:
    retarget_panel.draw(ctx, job)
    pose_panel.draw(ctx, job)
    sheet_panel.draw(ctx, job)


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


def _attempt_verdict(attempt: Any) -> str:
    """What one reroll attempt is to be called in the attempts line.

    Three words rather than two, because "not measured" is a third state and
    not a flavour of "refused": the composition report can fail to run at all
    (queue.py records the attempt with measured False, following
    reference.unmeasured), and calling that a refusal would put a verdict in
    the UI that nothing ever reached. Written against .get throughout, so an
    attempt row from before the third state existed still renders.
    """
    if not isinstance(attempt, dict):
        return "?"
    if attempt.get("measured") is False:
        return "not measured"
    return "kept" if attempt.get("ok") else "refused"


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

    attempts = params.get("reference_attempts")
    if isinstance(attempts, list) and len(attempts) > 1:
        # Only ever present on a job the reroll actually ran for, so the line
        # doubles as the answer to "why is this not the seed I asked for".
        widgets.muted(
            f"redrawn {len(attempts) - 1} time(s): "
            + "; ".join(f"seed {a.get('seed')} {_attempt_verdict(a)}" for a in attempts)
        )

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


def seam_verdict(report: Any) -> tuple[int, str] | None:
    """The one line that says whether the tile actually tiles, or None.

    A pure function of the stored report, so the wording is assertable without
    a GL context -- and so the number and the word never come from two places.
    The ratio is "how much worse than the picture's own grain the wrap seam is",
    which is why the threshold is quoted alongside it: a bare 3.4 is a number
    nobody can calibrate against, while "3.4, over 2.00" says which way it went
    and by how far.
    """
    if not isinstance(report, dict):
        return None
    worst = float(report.get("worst") or 0.0)
    if report.get("seamless"):
        return (theme.OK, f"seamless -- edge/grain {worst:.2f}")
    threshold = float(report.get("threshold") or 0.0)
    return (theme.WARN, f"visible seam -- edge/grain {worst:.2f}, over {threshold:.2f}")


def _seam(ctx: Any, job: Any) -> None:
    """Whether the tile actually tiles, and what it looks like wrapped.

    The ratio alone is a number nobody can calibrate against by eye, which is
    what the wrapped view is for: rolled by half, what was the wrap seam runs
    through the middle of the frame, where a discontinuity is obvious. It is a
    derived artifact like every other export rather than something this pane
    writes -- the roll reads and rewrites a full-size PNG, which is emphatically
    not frame-thread work, and going through ``derive.get_file`` means the
    button here and the Export tab's produce the same file under the same lock.
    """
    verdict = seam_verdict((job.get("params") or {}).get("seam_report"))
    if verdict is None:
        return
    if not widgets.header("Seam"):
        return
    report = job["params"]["seam_report"]
    widgets.text_colored(*verdict)
    widgets.muted(
        f"left/right {float(report.get('horizontal') or 0.0):.2f} - "
        f"top/bottom {float(report.get('vertical') or 0.0):.2f}"
    )
    if ctx.textures is None:
        return
    job_id = job["id"]
    # The cache answers "is it on disk yet" as a side effect of answering "give
    # me the texture": a missing file is a None, which is exactly the state in
    # which the button should be offered.
    texture = ctx.textures.get(f"{job_id}:wrap", ctx.job_dir(job_id) / "wrap_preview.png")
    key = f"wrap:{job_id}"
    if texture is None:
        busy = ctx.busy(key)
        if busy:
            # The same idiom the Export grid uses beside a busy artifact:
            # rolling a 1024-square PNG is several hundred milliseconds, and a
            # button that only greys out reads as one that did nothing.
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button("Show it wrapped", not busy):
            ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, "wrap_preview.png")
        return
    imgui.image(widgets.texture_ref(texture), (THUMB_SIZE * 2, THUMB_SIZE * 2))


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
    if isinstance(audit, dict) and audit.get("worst") is not None:
        widgets.muted(f"visible openings: {float(audit['worst']) * 100:.1f}%")

    attempts = params.get("mesh_attempts")
    if isinstance(attempts, list) and len(attempts) > 1:
        # Only ever present on a job the remesh actually ran for, so the line
        # doubles as the answer to "why is this not the mesh seed I asked for".
        widgets.muted(
            f"remeshed {len(attempts) - 1} time(s); kept the best of "
            + ", ".join(
                "unmeasured" if a.get("worst") is None else f"{float(a['worst']) * 100:.1f}%"
                for a in attempts
            )
        )


# --- verdicts ---------------------------------------------------------------
#
# The same Accept/Reject a sweep unit gets in Review, on any finished asset --
# which is the point: daily use and a deliberate sweep feed one findings pool,
# and the vast majority of assets anyone ever looks at were never part of a
# sweep. The two halves that are easy to get wrong (when the reject is armed,
# and what a recorded verdict does) are plain functions so they can be asserted
# without a GL context.


def verdict_armed(state: Any, job_id: str) -> bool:
    return state.inspector_reject_armed == job_id


def arm_verdict(state: Any, job_id: str | None) -> None:
    state.inspector_reject_armed = job_id


def record_verdict(ctx: Any, job_id: str, verdict: str, reasons: tuple[str, ...] = ()) -> bool:
    """File a verdict and queue the findings recompute. -> whether it landed.

    Inline on the frame thread for the reason Review's is: one INSERT under the
    store's lock. The recompute that follows reads every verdict and writes a
    file, so it is a task.
    """
    from ...service import findings as svc_findings
    from ...service import verdicts as svc_verdicts
    from ...service.errors import ServiceError

    try:
        svc_verdicts.record_verdict(ctx.svc, job_id, verdict=verdict, reasons=reasons)
    except (ServiceError, OSError):
        ctx.toast("Could not record that verdict.", "error")
        return False
    arm_verdict(ctx.state, None)
    ctx.submit("review-findings", svc_findings.refresh, ctx.svc)
    ctx.toast(f"Recorded: {verdict}.")
    return True


def _verdict(ctx: Any, job: Any) -> None:
    from ...service import verdicts as svc_verdicts

    if job.get("status") != "done" or job.get("stage") != "model":
        return
    if not widgets.header("Was this any good?", default_open=False):
        return
    job_id = job["id"]
    widgets.muted("Feeds the same findings a sweep does.")
    if widgets.primary_button("Accept"):
        record_verdict(ctx, job_id, "accept")
    imgui.same_line()
    if imgui.button("Reject"):
        arm_verdict(ctx.state, job_id)
    if verdict_armed(ctx.state, job_id):
        widgets.muted("Why?")
        for reason in svc_verdicts.REASONS:
            if imgui.button(f"{reason}##verdict-{reason}"):
                record_verdict(ctx, job_id, "reject", (reason,))
        if imgui.button("Cancel##verdict-cancel"):
            arm_verdict(ctx.state, None)


def _downloads(ctx: Any, job: Any) -> None:
    """The Export tab: a two-column grid of artifacts.

    A blocked artifact keeps its button and explains itself in a tooltip --
    the old layout wrote the reason as an indented line under each of eight
    disabled buttons, which for a plain reference job was a column of noise.
    """
    from .. import icons

    widgets.section("Downloads")
    job_id = job["id"]
    files = set(job.get("files") or [])
    # Both notes under the grid are about cutouts -- which matte cut them, and
    # what each one came out as -- so they belong to the stage that has any. A
    # tile's manifest holds the recipe and no artifact entries at all.
    cutouts = job.get("stage") == "reference"
    if not imgui.begin_table("downloads", 2):
        return
    for name, label in widgets.artifacts_for(job):
        # job["files"] is the sanctioned answer; a raw exists() check here used
        # to re-enable buttons the service would then refuse.
        ready = name in files
        blocked = _why_blocked(ctx, job, name, ready, _derivable(job, files, name))
        key = f"save:{job_id}:{name}"
        busy = ctx.busy(key)
        imgui.table_next_column()
        if busy:
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button(
            f"{icons.DOWNLOAD} {label}##{name}", not blocked and not busy, (-1, 0)
        ):
            ctx.save_artifact(job_id, name)
        if blocked and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(blocked)
    imgui.end_table()
    if cutouts:
        # Read once and handed to both: the two notes answer halves of the same
        # question, and asking for the manifest twice in a frame would make the
        # cheap-stat property depend on how many readers there happen to be.
        manifest = _manifest(ctx, job_id)
        _matte_note(ctx, manifest)
        _manifest_summary(manifest)


def _derivable(job: Any, files: set[str], name: str) -> bool:
    """Whether this job could produce ``name`` if the button were pressed.

    A 2D export comes from the reference's own pixels, so it is gated on those
    and not on a ``model.glb`` a reference will never have -- which is the old
    gate, and which greyed out every one of them. But it *is* gated on the job
    having finished: the mesh half got that for free, because model.glb is not
    listed until the worker is done with it, whereas ``derivable_2d`` answers a
    question about the name alone and would light six buttons on a queued job
    whose only outcome is an error toast. The conditions are deliberately the
    same three ``service.files.ready`` applies, restated here in terms of the
    listing rather than of the disk, because the frame thread may not stat --
    ``ready`` takes a job dir and would be a stat per artifact per frame. The
    two are pinned equal across the whole stage x status x files matrix by
    ``test_the_pane_agrees_with_the_service_about_every_artifact``, which is
    what stops the restatement drifting.

    Which names each image stage may ask for is deliberately *not* restated: a
    reference has the cutouts and a tile has the wrapped view, and that split is
    a fact about the artifacts rather than about the disk, so it is read
    straight off ``derivable_2d`` at no cost.
    """
    stage = job.get("stage")
    if stage in ("reference", "tile"):
        return (
            job.get("status") == "done"
            and "input.png" in files
            and svc_derive.derivable_2d(name, stage)
        )
    return "model.glb" in files and svc_derive.derivable(name)


def _why_blocked(ctx: Any, job: Any, name: str, ready: bool, derivable: bool) -> str | None:
    if ready:
        return None
    if name == "model.fbx" and not ctx.rigging_available:
        return "needs Blender"
    if derivable:
        return None
    if job.get("status") in ("queued", "running"):
        # Correctly disabled, and it used to be misleadingly explained: every
        # export on an unfinished job read "not available for this asset",
        # which says the asset cannot have the file rather than that it has not
        # got it yet. An errored job keeps the other wording, because there
        # "not yet" would promise something that is not coming.
        return "not finished yet"
    return "not available for this asset"


def _matte_note(ctx: Any, manifest: Any) -> None:
    """Why the cutouts will look the way they do, before any of them exist.

    Not a reason a button is blocked, which is why it sits under the grid
    rather than in ``_why_blocked``: the exports always work. The question a
    user has when the edges come out ragged is whether that is the model or the
    fallback, and nothing in the UI could answer it -- the doctor row says the
    same thing in a place nobody opens mid-export.

    Deliberately only for the case where nothing has been derived yet. Once
    artifacts exist, each manifest entry records the matte that actually cut
    it, and ``_manifest_summary`` says so per artifact -- which is the truthful
    answer where this one is merely the current one: an icon cut by the corner
    fill before the weights were installed would, on this note alone, claim
    nothing at all the moment they were.
    """
    from ...pipelines import matting

    if manifest is not None and (manifest.get("artifacts") or {}):
        return
    if matting.available(ctx.svc.config):
        return
    widgets.muted(
        "Cutouts use the corner fill -- edges are rougher than the matting "
        "model's. See the matting row under Settings for the one-time download."
    )


def _manifest_summary(manifest: Any) -> None:
    """The pivot, the alpha QA and the matte, read off the manifest as written.

    Read from the file rather than recomputed: the manifest is the thing an
    importer will consume, so showing anything else here would let the two
    disagree about the asset. That is also why the matte is reported per
    artifact -- ``derive`` records which one cut each file, and a file on disk
    was cut by whatever was installed when it was made, not by what is
    installed now.

    The hand-edited caveat travels with the recipe for the reason
    ``derive._write_manifest`` records it: the recipe hash names a seed and a
    model, which after a hand edit is no longer the whole story of the pixels.
    """
    if manifest is None:
        return
    if manifest.get("hand_edited"):
        widgets.muted("hand-edited -- the recipe describes the generated image, not these pixels")
    for name, entry in sorted((manifest.get("artifacts") or {}).items()):
        if not isinstance(entry, dict):
            continue
        bits = [name]
        if entry.get("pivot"):
            bits.append(f"pivot {entry['pivot'][0]:.0f},{entry['pivot'][1]:.0f}")
        alpha = entry.get("alpha") or {}
        if alpha.get("islands", 0) > 1:
            bits.append(f"{alpha['islands']} separate pieces")
        if entry.get("matte") == "flood":
            bits.append("corner fill")
        widgets.muted(" - ".join(bits))


def _manifest(ctx: Any, job_id: str) -> dict[str, Any] | None:
    """The job's parsed manifest.json, or None if there isn't a readable one.

    A stat every frame and a parse only when the file has changed. That split
    is the whole point: the export tab redraws sixty times a second, a
    derivation running on the TaskRunner rewrites the manifest underneath it,
    and re-reading on an mtime is how ``ThumbnailCache`` already reconciles
    those two facts. The slot lives on ``AppState`` rather than in a module
    global so the pane itself stays stateless.

    A failure to parse is cached too, as ``(key, None)``. "Unreadable" is an
    answer about that version of the file exactly as a parsed dict is, and a
    truncated or hand-mangled manifest is the one state that would otherwise be
    read and parsed *every* frame, for as long as it stayed mangled -- which is
    the per-frame read this cache exists to remove. A missing file needs no
    sentinel: the stat fails, so nothing is read either way.
    """
    import json

    path = ctx.job_dir(job_id) / "manifest.json"
    try:
        key = (job_id, path.stat().st_mtime_ns)
    except OSError:
        return None
    cached = ctx.state.manifest
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        manifest = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        manifest = None
    if not isinstance(manifest, dict):
        manifest = None
    ctx.state.manifest = (key, manifest)
    return manifest


def duration(job: Any) -> str:
    return format_duration(job.get("duration"))

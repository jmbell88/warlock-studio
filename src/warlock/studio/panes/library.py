"""The asset library: filters, cards, bulk actions, storage.

A card shows what the job *is* and offers exactly one primary action -- the
obvious next step, from :func:`~warlock.studio.state.primary_action` -- with
everything else behind the overflow menu. That ladder is the browser's, ported
verbatim, because "the button is always the thing I wanted" is most of what
made the old library usable.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...service import export as svc_export
from ...service import jobs as svc_jobs
from ...service import rig as svc_rig
from .. import dialogs, theme, widgets
from ..state import ACTIONS, primary_action

CARD_HEIGHT = 92.0
THUMB_SIZE = 72.0


def draw(ctx: Any) -> None:
    _filters(ctx)
    imgui.separator()
    jobs = ctx.cache.visible(ctx.state.filters)
    if ctx.cache.error:
        widgets.text_colored(theme.ERR, "Could not read the job list.")
    if not jobs:
        widgets.muted("Nothing here yet." if not ctx.cache.jobs else "Nothing matches.")
    # Leave room for the footer below: the bulk bar only exists when something
    # is checked, so the reservation has to change with it or the list either
    # overlaps the footer or floats above a gap.
    height = -96 if ctx.state.checked else -34
    if imgui.begin_child("library-list", (0, height)):
        for job in jobs:
            _card(ctx, job)
        _load_more(ctx)
    imgui.end_child()
    _bulk(ctx)
    _storage(ctx)


def _load_more(ctx: Any) -> None:
    """The window is the newest N of M -- and the filters above apply only to
    that window, so a history longer than it needs to say so rather than let a
    search quietly miss what it never loaded."""
    loaded = len(ctx.cache.jobs)
    total = ctx.cache.total
    if total <= loaded:
        return
    widgets.muted(f"Showing the newest {loaded} of {total}.")
    if imgui.button("Load older##library-more", (-1, 0)):
        ctx.cache.load_more()


# --- filters ----------------------------------------------------------------


def _filters(ctx: Any) -> None:
    filters = ctx.state.filters
    imgui.set_next_item_width(-1)
    filters.text = widgets.input_text("##filter", filters.text, max_length=120, hint="Filter...")
    filters.status = widgets.combo(
        "##status",
        filters.status,
        [
            ("all", "any status"),
            ("done", "done"),
            ("running", "running"),
            ("error", "failed"),
        ],
        width=110,
    )
    imgui.same_line()
    filters.kind = widgets.combo(
        "##kind",
        filters.kind,
        [
            ("all", "any kind"),
            ("reference", "references"),
            ("model", "meshes"),
            ("rig", "rigs"),
            ("sheet", "sheets"),
        ],
        width=110,
    )
    imgui.same_line()
    changed, favourites = imgui.checkbox("*", filters.favorites_only)
    if changed:
        filters.favorites_only = favourites
    widgets.help_marker("Favourites only")


# --- cards ------------------------------------------------------------------


def _card(ctx: Any, job: Any) -> None:
    state = ctx.state
    job_id = job["id"]
    selected = state.selected == job_id
    imgui.push_id(job_id)
    if selected:
        imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.EDGE)))
    if imgui.begin_child("card", (0, CARD_HEIGHT), imgui.ChildFlags_.borders.value):
        _card_body(ctx, job)
    imgui.end_child()
    if selected:
        imgui.pop_style_color()
    # The whole card selects, not just a title: the card *is* the affordance.
    if imgui.is_item_clicked():
        select(ctx, job_id)
    imgui.pop_id()


def _card_body(ctx: Any, job: Any) -> None:
    job_id = job["id"]
    texture = None
    if ctx.textures is not None and "thumb.png" in (job.get("files") or []):
        texture = ctx.textures.get(job_id, ctx.job_dir(job_id) / "thumb.png")
    if texture is not None:
        imgui.image(widgets.texture_ref(texture), (THUMB_SIZE, THUMB_SIZE))
    else:
        imgui.dummy((THUMB_SIZE, THUMB_SIZE))
    imgui.same_line()

    # begin_group returns nothing -- the pair is unconditional, and wrapping it
    # in an `if` is how a frame ends up one end_group short.
    imgui.begin_group()
    name = job.get("name") or job.get("prompt") or job_id
    imgui.text_wrapped(name if len(name) <= 46 else name[:43] + "...")
    widgets.status_pill(job["status"])
    imgui.same_line()
    widgets.quality_badge(job)
    if job.get("parent_id"):
        imgui.same_line()
        widgets.muted("| from a reference")

    progress = ctx.runtime.progress(job_id)
    if progress is not None:
        widgets.progress_bar(float(progress.get("percent") or 0.0))
        widgets.muted(str(progress.get("label") or ""))
    else:
        _card_actions(ctx, job)
    imgui.end_group()


def _card_actions(ctx: Any, job: Any) -> None:
    action = primary_action(job, rigging_available=ctx.rigging_available)
    if action is not None and imgui.small_button(ACTIONS[action]):
        run_action(ctx, job, action)
    imgui.same_line()
    if imgui.small_button("..."):
        imgui.open_popup("more")
    imgui.same_line()
    checked = job["id"] in ctx.state.checked
    changed, value = imgui.checkbox("##pick", checked)
    if changed and value != checked:
        ctx.state.toggle_check(job["id"])
    imgui.same_line()
    if imgui.small_button("*" if job.get("favorite") else "-"):
        ctx.submit(
            f"fav:{job['id']}",
            svc_jobs.update_job,
            ctx.svc,
            job["id"],
            {"favorite": not job.get("favorite")},
        )
    _overflow(ctx, job)


def _overflow(ctx: Any, job: Any) -> None:
    if not imgui.begin_popup("more"):
        return
    job_id = job["id"]
    files = job.get("files") or []
    if imgui.menu_item("Rename...", "", False)[0]:
        ctx.prompts.ask(
            dialogs.Prompt(
                title="Rename",
                label="Name",
                value=job.get("name") or "",
                on_accept=lambda value: ctx.submit(
                    f"rename:{job_id}", svc_jobs.update_job, ctx.svc, job_id, {"name": value}
                ),
            )
        )
    if job["status"] in ("done", "error", "cancelled"):
        if imgui.menu_item("Reroll", "", False)[0]:
            ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll")
        if "input.png" in files and imgui.menu_item("Remesh", "", False)[0]:
            ctx.submit(f"remesh:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="remesh")
    if "model.glb" in files:
        if imgui.menu_item("Compare with selected", "", False)[0]:
            compare(ctx, job_id)
        if ctx.rigging_available and imgui.menu_item("Rig", "", False)[0]:
            ctx.submit(f"rig:{job_id}", svc_rig.create_rig, ctx.svc, job_id)
    imgui.separator()
    if imgui.menu_item("Delete", "", False)[0]:
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Delete this asset?",
                message="The job and everything derived from it are removed from disk.",
                confirm_label="Delete",
                cancel_label="Keep",
                on_confirm=lambda: _delete(ctx, job_id),
            )
        )
    imgui.end_popup()


# --- actions ----------------------------------------------------------------


def select(ctx: Any, job_id: str) -> None:
    ctx.state.select(job_id)
    job = ctx.cache.get(job_id)
    if job is None:
        return
    # Promotion's source follows the selection when the selection is something
    # that can be promoted; anything else leaves it alone, so switching to a
    # mesh to look at it does not silently change what Make 3D would submit.
    if job.get("stage") == "reference" and job.get("status") == "done":
        ctx.state.source_job = job_id


def run_action(ctx: Any, job: Any, action: str) -> None:
    job_id = job["id"]
    if action == "cancel":
        ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)
    elif action == "retry":
        mode = "remesh" if "input.png" in (job.get("files") or []) else "reroll"
        ctx.submit(f"retry:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode=mode)
    elif action == "promote":
        ctx.state.source_job = job_id
        ctx.state.mode = "3d"
    elif action == "rig":
        ctx.submit(f"rig:{job_id}", svc_rig.create_rig, ctx.svc, job_id)
    elif action == "open":
        select(ctx, job_id)
        ctx.state.mode = "3d"


def compare(ctx: Any, job_id: str) -> None:
    if ctx.state.comparing == job_id:
        ctx.state.comparing = None
        if ctx.viewer is not None:
            ctx.viewer.exit_compare()
        return
    ctx.state.comparing = job_id
    if ctx.viewer is not None:
        ctx.viewer.compare(ctx.job_dir(job_id) / "model.glb")


def _delete(ctx: Any, job_id: str) -> None:
    ctx.state.checked.discard(job_id)
    if ctx.state.selected == job_id:
        ctx.state.select(None)
    ctx.submit(f"delete:{job_id}", svc_jobs.delete_job, ctx.svc, job_id)


# --- bulk and storage -------------------------------------------------------


def _bulk(ctx: Any) -> None:
    picked = sorted(ctx.state.checked)
    if not picked:
        return
    imgui.separator()
    imgui.text(f"{len(picked)} selected")
    imgui.same_line()
    if imgui.small_button("Clear"):
        ctx.state.checked.clear()
    if imgui.button("Export zip..."):
        _export_zip(ctx, picked)
    if ctx.export_dir:
        # Only when one is configured: the feature is off unless
        # WARLOCK_EXPORT_DIR is set, and a button that can only fail is worse
        # than no button.
        imgui.same_line()
        if imgui.button("Save to project"):
            ctx.submit(
                "export-folder", svc_export.export_to_folder, ctx.svc, picked, ["model.glb"]
            )
    imgui.same_line()
    if imgui.button("Delete##bulk"):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Delete these assets?",
                message=f"{len(picked)} jobs and everything derived from them.",
                confirm_label="Delete",
                cancel_label="Keep",
                on_confirm=lambda: [_delete(ctx, j) for j in picked],
            )
        )


def _export_zip(ctx: Any, ids: list[str]) -> None:
    def run():
        dest = dialogs.save_file("Export selection", "warlock_export.zip", dialogs.ZIP_FILTER)
        if dest is None:
            return None
        return svc_export.bulk_export(ctx.svc, ids, ["model.glb"], dest)

    ctx.submit("export-zip", run)


def _storage(ctx: Any) -> None:
    storage = ctx.cache.storage
    if storage:
        from ..state import format_bytes

        widgets.muted(f"{storage['job_dirs']} jobs - {format_bytes(storage['bytes'])}")
    imgui.same_line()
    if imgui.small_button("Prune..."):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Prune old assets?",
                message="Everything but the newest 20 jobs is deleted. Running jobs are kept.",
                confirm_label="Prune",
                cancel_label="Cancel",
                on_confirm=lambda: ctx.submit("prune", svc_jobs.prune_jobs, ctx.svc, 20),
            )
        )

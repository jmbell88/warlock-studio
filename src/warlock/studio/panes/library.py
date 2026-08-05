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
from .. import dialogs, icons, theme, widgets
from ..manual import render as manual_render
from ..state import ACTIONS, primary_action
from ..tokens import sp

CARD_HEIGHT = 92.0
THUMB_SIZE = 72.0


def draw(ctx: Any) -> None:
    _filters(ctx)
    manual_render.help_button(ctx, "library")
    imgui.separator()
    jobs = ctx.cache.visible(ctx.state.filters)
    if ctx.cache.error:
        widgets.text_colored(theme.ERR, "Could not read the job list.")
    # Leave room for the footer below: the bulk bar only exists when something
    # is checked, so the reservation has to change with it or the list either
    # overlaps the footer or floats above a gap.
    height = -sp(96) if ctx.state.checked else -sp(34)
    if imgui.begin_child("library-list", (0, height)):
        if not jobs:
            widgets.empty_state(
                icons.IMAGE,
                "Nothing here yet." if not ctx.cache.jobs else "Nothing matches.",
                "Generated references and meshes appear here."
                if not ctx.cache.jobs
                else "Loosen the filters above.",
            )
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
    filters.sort = widgets.combo(
        "##sort",
        filters.sort,
        [("newest", "newest first"), ("best", "best first")],
        width=110,
    )
    imgui.same_line()
    # A star that lights up, not a checkbox labelled "*".
    lit = filters.favorites_only
    if lit:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.WARN)))
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(*theme.rgba(theme.WARN, 0.2)))
    if widgets.icon_button(icons.STAR, "Favourites only"):
        filters.favorites_only = not filters.favorites_only
    if lit:
        imgui.pop_style_color(2)


# --- cards ------------------------------------------------------------------


def _card(ctx: Any, job: Any) -> None:
    state = ctx.state
    job_id = job["id"]
    selected = state.selected == job_id
    imgui.push_id(job_id)
    if selected:
        imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.ELEV_2)))
    origin = imgui.get_cursor_screen_pos()
    if imgui.begin_child("card", (0, sp(CARD_HEIGHT)), imgui.ChildFlags_.borders.value):
        _card_body(ctx, job)
    imgui.end_child()
    if selected:
        imgui.pop_style_color()
        # The accent edge is the selection mark; a raised fill alone reads as
        # hover, not choice.
        imgui.get_window_draw_list().add_rect_filled(
            origin,
            (origin.x + sp(3), origin.y + sp(CARD_HEIGHT)),
            imgui.get_color_u32(theme.rgba(theme.ACCENT)),
            sp(2),
        )
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
        imgui.image(widgets.texture_ref(texture), (sp(THUMB_SIZE), sp(THUMB_SIZE)))
    else:
        imgui.dummy((sp(THUMB_SIZE), sp(THUMB_SIZE)))
    imgui.same_line()

    # begin_group returns nothing -- the pair is unconditional, and wrapping it
    # in an `if` is how a frame ends up one end_group short.
    imgui.begin_group()
    name = job.get("name") or job.get("prompt") or job_id
    imgui.text_wrapped(name if len(name) <= 46 else name[:43] + "...")
    widgets.status_pill(job["status"])
    imgui.same_line()
    widgets.quality_badge(job)
    rank = (job.get("params") or {}).get("rank")
    if isinstance(rank, dict) and rank.get("score") is not None:
        imgui.same_line()
        widgets.muted(f"{float(rank['score']) * 100:.0f}%")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "How well framed this reference is, and -- when the profile "
                "has a style anchor -- how close it looks to it. Advisory."
            )
    if job["status"] == "queued":
        # Where in line it is: the queue used to be invisible past the pill.
        waiting = [j["id"] for j in reversed(ctx.cache.jobs) if j.get("status") == "queued"]
        if job_id in waiting:
            imgui.same_line()
            widgets.muted(f"#{waiting.index(job_id) + 1} in queue")
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
    if imgui.small_button(icons.ELLIPSIS):
        imgui.open_popup("more")
    if imgui.is_item_hovered():
        imgui.set_tooltip("More actions")
    imgui.same_line()
    checked = job["id"] in ctx.state.checked
    changed, value = imgui.checkbox("##pick", checked)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Select for bulk actions")
    if changed and value != checked:
        ctx.state.toggle_check(job["id"])
    imgui.same_line()
    favourite = bool(job.get("favorite"))
    if favourite:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.WARN)))
    if imgui.small_button(icons.STAR):
        ctx.submit(
            f"fav:{job['id']}",
            svc_jobs.update_job,
            ctx.svc,
            job["id"],
            {"favorite": not favourite},
        )
    if favourite:
        imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Unfavourite" if favourite else "Favourite")
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
    if job.get("params") and imgui.menu_item("Copy settings to form", "", False)[0]:
        _copy_settings(ctx, job)
    if job["status"] in ("done", "error", "cancelled"):
        # A hand-made reference has no generator behind it, so there is nothing
        # a new seed could change; the service refuses it, and offering the
        # menu item anyway only buys the user an error toast.
        rerollable = not (job["kind"] == "image" and job.get("stage") == "reference")
        if rerollable and imgui.menu_item("Reroll", "", False)[0]:
            ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll")
        if "input.png" in files and imgui.menu_item("Remesh", "", False)[0]:
            ctx.submit(f"remesh:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="remesh")
    if "model.glb" in files:
        if imgui.menu_item("Compare with selected", "", False)[0]:
            compare(ctx, job_id)
        if ctx.rigging_available and imgui.menu_item("Rig", "", False)[0]:
            ctx.submit(
                f"rig:{job_id}",
                svc_rig.create_rig,
                ctx.svc,
                job_id,
                template=_skeleton(ctx),
            )
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


def _copy_settings(ctx: Any, job: Any) -> None:
    """Load a job's recipe back into the 2D form, so it can be varied.

    Reroll re-runs a job as it was; this is the other half -- start from what
    it used and change one thing. Prompt history only ever restored the prompt
    text, which left every guidance field, the model, the LoRA and the
    conditioning strengths behind.
    """
    from ..state import form_from_params

    ctx.state.form_2d = form_from_params(job.get("params") or {})
    ctx.state.mode = "2d"
    ctx.toast("Settings copied to the form.")


def _skeleton(ctx: Any) -> str | None:
    """The skeleton the 3D pane's combo is showing.

    Rigging an existing mesh used to pass nothing, so the combo applied only
    when a rig was requested as part of generating -- picking "quadruped" and
    then rigging a finished mesh silently used the config default. None means
    "no explicit choice", which is what the service already resolves.
    """
    return (ctx.state.form_3d or {}).get("rig_template") or None


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
        ctx.submit(f"rig:{job_id}", svc_rig.create_rig, ctx.svc, job_id, template=_skeleton(ctx))
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

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
    # Resolved before the filter row rather than after it, because the row's
    # select-all acts on exactly this list and computing it twice a frame to
    # keep the old order would be paying for the same filter pass twice.
    jobs = ctx.cache.visible(ctx.state.filters)
    # Draws the (?) too, on the sort row that reserves the width for it --
    # ``render.help_button`` right-aligns with an unconditional ``same_line``,
    # so where it lands is a property of the row above it and belongs with the
    # code that lays that row out. Chapter 08's body has no other way in from
    # here: the Profiles panel's marker anchors at #profiles.
    _filters(ctx, jobs)
    imgui.separator()
    if ctx.cache.error:
        widgets.text_colored(theme.ERR, "Could not read the job list.")
    # Leave room for the footer below: the bulk bar only exists when something
    # is checked, so the reservation has to change with it or the list either
    # overlaps the footer or floats above a gap.
    height = -sp(96) if ctx.state.checked else -sp(34)
    # Hoisted out of the card loop (B20): each queued card used to rebuild the
    # whole reversed-list scan to learn its own position.
    queue_pos = {
        job_id: i + 1
        for i, job_id in enumerate(
            j["id"] for j in reversed(ctx.cache.jobs) if j.get("status") == "queued"
        )
    }
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
            _card(ctx, job, queue_pos)
        _load_more(ctx)
    imgui.end_child()
    _bulk(ctx, jobs)
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


def _filters(ctx: Any, jobs: list[Any]) -> None:
    """The three selects, the star and the tick, on two rows rather than one.

    Measured, not guessed: three 110 px combos plus two square buttons come to
    417 px, and the sidebar this pane lives in is a fixed 300 (``layout.
    SIDEBAR_W``), which leaves 290 inside the padding. A child window *clips*
    rather than wraps, so the star spent its whole life drawn past the right
    edge -- neither visible nor clickable -- and the tick would have joined it.
    Widths come off the live content region for the same reason a constant was
    the bug: ``sp`` scales the sidebar with the monitor and 110 did not scale
    with anything.
    """
    filters = ctx.state.filters
    imgui.set_next_item_width(-1)
    filters.text = widgets.input_text("##filter", filters.text, max_length=120, hint="Filter...")
    spacing = imgui.get_style().item_spacing.x
    half = (imgui.get_content_region_avail().x - spacing) * 0.5
    filters.status = widgets.combo(
        "##status",
        filters.status,
        [
            ("all", "any status"),
            ("done", "done"),
            ("running", "running"),
            ("error", "failed"),
        ],
        width=half,
    )
    imgui.same_line()
    filters.kind = widgets.combo(
        "##kind",
        filters.kind,
        [
            ("all", "any kind"),
            ("reference", "references"),
            ("tile", "tiles"),
            ("model", "meshes"),
            ("rig", "rigs"),
            ("sheet", "sheets"),
        ],
        width=half,
    )
    # The three square buttons share the sort row, so what is left for the
    # combo is what they and their gaps do not take. Three, not two: the (?)
    # right-aligns itself onto whatever line is current, so a row that reserved
    # room for only the star and the tick put it exactly on top of the tick --
    # same pixels, and the later item takes the click, so the select-all opened
    # the manual instead.
    buttons = 3 * (imgui.get_frame_height() + spacing)
    filters.sort = widgets.combo(
        "##sort",
        filters.sort,
        [("newest", "newest first"), ("best", "best first")],
        width=imgui.get_content_region_avail().x - buttons,
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
    imgui.same_line()
    _select_all(ctx, jobs)
    # Before ``_failures``, which starts a line of its own: the (?) joins the
    # line that is current when it is called, and joining that one would put it
    # on a row that is not always there.
    manual_render.help_button(ctx, "library")
    _failures(ctx)


def _failures(ctx: Any) -> None:
    """A way to the failed jobs, when there are any and they are not shown.

    A failed job says why it failed in the inspector and nowhere else, so
    before this the only route to the reason was to already know which card to
    click -- and after a sweep or an overnight batch that is the one thing the
    user does not know. Its own full-width row rather than a fourth control on
    the filter rows: three combos and two square buttons already overrun the
    fixed 300 px sidebar, and a child window clips rather than wraps.
    """
    filters = ctx.state.filters
    if filters.status == "error":
        return
    count = ctx.cache.failures(filters)
    if not count:
        return
    label = "1 job failed" if count == 1 else f"{count} jobs failed"
    imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
    clicked = imgui.small_button(f"{label} - show##library-failures")
    imgui.pop_style_color()
    if clicked:
        filters.status = "error"


def _select_all(ctx: Any, jobs: list[Any]) -> None:
    """Tick every card the filters are showing -- the bulk bar's other half.

    Deliberately *shown*, not "every job": the list is a window onto the newest
    N of M (see ``_load_more``), so a control claiming to select everything
    would silently leave the older ones out of the delete that follows. It
    flips to Deselect once the shown set is fully ticked, so the same button
    undoes it rather than leaving Clear as the only way back.
    """
    shown = [job["id"] for job in jobs]
    checked = ctx.state.checked
    picked = bool(shown) and all(job_id in checked for job_id in shown)
    icon = icons.SQUARE_DASHED if picked else icons.CHECK
    tip = "Deselect the assets shown" if picked else "Select every asset shown, for bulk actions"
    if widgets.icon_button(icon, tip, enabled=bool(shown)):
        if picked:
            checked.difference_update(shown)
        else:
            checked.update(shown)


# --- cards ------------------------------------------------------------------


def _card(ctx: Any, job: Any, queue_pos: dict[str, int] | None = None) -> None:
    state = ctx.state
    job_id = job["id"]
    selected = state.selected == job_id
    imgui.push_id(job_id)
    if selected:
        imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.ELEV_2)))
    origin = imgui.get_cursor_screen_pos()
    if imgui.begin_child("card", (0, sp(CARD_HEIGHT)), imgui.ChildFlags_.borders.value):
        _card_body(ctx, job, queue_pos)
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


def _card_body(ctx: Any, job: Any, queue_pos: dict[str, int] | None = None) -> None:
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
    # Between the two, and that order is the rule rather than a preference:
    # ``quality_badge`` may draw nothing and owns its own ``same_line`` for it,
    # so anything unconditional has to be ahead of it -- a badge placed after
    # would inherit the ``same_line`` that call did not spend.
    widgets.stage_badge(job, inline=True)
    widgets.quality_badge(job, inline=True)
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
        if queue_pos is None:
            queue_pos = {
                jid: i + 1
                for i, jid in enumerate(
                    j["id"] for j in reversed(ctx.cache.jobs) if j.get("status") == "queued"
                )
            }
        position = queue_pos.get(job_id)
        if position is not None:
            imgui.same_line()
            widgets.muted(f"#{position} in queue")
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
    if widgets.small_icon_button(icons.ELLIPSIS, "More actions"):
        imgui.open_popup("more")
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
    if widgets.small_icon_button(icons.STAR, "Unfavourite" if favourite else "Favourite"):
        ctx.submit(
            f"fav:{job['id']}",
            svc_jobs.update_job,
            ctx.svc,
            job["id"],
            {"favorite": not favourite},
        )
    if favourite:
        imgui.pop_style_color()
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
        if _remeshable(job) and imgui.menu_item("Remesh", "", False)[0]:
            ctx.submit(f"remesh:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="remesh")
    # The 2D half of the same pair as Edit in Clay, and gated the same way: on
    # a predicate the mode owns, answered from the cached row alone so the
    # frame thread never stats anything. Above the mesh block because the two
    # are mutually exclusive -- a reference has no ``model.glb`` -- and the
    # loader reuses an already-open tab rather than forking a second one over
    # the same file.
    from .. import inker_mode

    if inker_mode.can_edit_job(ctx, job) and imgui.menu_item("Open in Inker", "", False)[0]:
        inker_mode.open_job_reference(ctx, job)
    if "model.glb" in files:
        # Clay prefers the ``build.wblk`` sidecar when the asset was authored
        # here, and imports ``model.glb`` -- the optimized, grounded, served
        # mesh -- when it was not. Never ``source.glb``: that is the raw
        # reconstruction, and nothing downstream of the pipeline uses it.
        if imgui.menu_item("Edit in Clay", "", False)[0]:
            from .. import clay_mode

            clay_mode.edit_asset_in_clay(ctx, job)
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
                on_confirm=lambda: delete_asset(ctx, job_id),
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

    form = form_from_params(job.get("params") or {})
    # The output switch is restored from the *stage*, because that is where a
    # job's tile-ness lives -- params never carries it, so a form filled from
    # params alone would open in Object mode and quietly offer to make a mesh
    # of a texture. The whole job row is in hand here; form_from_params is not
    # given it, because it is the params allowlist and must stay one.
    form["output"] = "tile" if job.get("stage") == "tile" else "reference"
    ctx.state.form_2d = form
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


def _remeshable(job: Any) -> bool:
    """Whether "keep this image, rebuild the mesh" is offered for this job.

    One predicate rather than the same expression at each call site, because
    the two have already drifted apart once. The retry ladder learned that a
    tile has no subject to reconstruct and the context menu did not, so
    right-click -> Remesh on a finished tile reached ``rerun_job``, was refused
    and came back as an error toast. That is the rule ``rerollable`` two lines
    above the menu item states in prose: never offer an action the service will
    refuse. Stated once, it cannot be honoured in one place and not the other.
    """
    return "input.png" in (job.get("files") or []) and job.get("stage") != "tile"


def run_action(ctx: Any, job: Any, action: str) -> None:
    job_id = job["id"]
    if action == "cancel":
        ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)
    elif action == "retry":
        mode = "remesh" if _remeshable(job) else "reroll"
        ctx.submit(f"retry:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode=mode)
    elif action == "promote":
        ctx.state.source_job = job_id
        ctx.state.mode = "3d"
    elif action == "rig":
        ctx.submit(f"rig:{job_id}", svc_rig.create_rig, ctx.svc, job_id, template=_skeleton(ctx))
    elif action == "open":
        select(ctx, job_id)
        # A job that stops at an image opens in the pane that made it. A tile
        # has no mesh at all, so opening it in 3D would show an empty viewport.
        ctx.state.mode = "2d" if job.get("stage") in ("reference", "tile") else "3d"


def compare(ctx: Any, job_id: str) -> None:
    if ctx.state.comparing == job_id:
        ctx.state.comparing = None
        if ctx.viewer is not None:
            ctx.viewer.exit_compare()
        return
    ctx.state.comparing = job_id
    if ctx.viewer is not None:
        ctx.viewer.compare(ctx.job_dir(job_id) / "model.glb")


def delete_asset(ctx: Any, job_id: str) -> None:
    """Remove one asset: the tick, the selection, then the job and its files.

    Public because the candidate picker offers exactly this for the attempts
    the user did not keep -- through this function rather than a second
    ``svc_jobs.delete_job`` call, so a loser is removed by the one path that
    also clears the selection and the tick set.
    """
    ctx.state.checked.discard(job_id)
    if ctx.state.selected == job_id:
        ctx.state.select(None)
    ctx.submit(f"delete:{job_id}", svc_jobs.delete_job, ctx.svc, job_id)


# --- bulk and storage -------------------------------------------------------


def _bulk(ctx: Any, jobs: list[Any]) -> None:
    """The actions for what is ticked, and how much of it is off screen.

    ``state.checked`` is not pruned when the filters change, and that is worth
    keeping -- ticking a few meshes, then switching to references to tick a few
    more, is a real way to use this. What is not defensible is doing it
    silently: tick everything shown, narrow the filter, press Delete, and the
    wider set goes while the confirm names only a count. So the count says how
    much of itself is no longer on screen, and the confirm repeats it, because
    the destructive path must never describe a smaller act than it performs.

    *Not shown* rather than *filtered out*: an id can also be here because its
    job fell off the newest-N window or was deleted from somewhere else, and
    the honest word covers all three.
    """
    picked = sorted(ctx.state.checked)
    if not picked:
        return
    shown = {job["id"] for job in jobs}
    hidden = sum(1 for job_id in picked if job_id not in shown)
    imgui.separator()
    imgui.text(f"{len(picked)} selected" + (f" ({hidden} not shown)" if hidden else ""))
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
                message=_delete_message(len(picked), hidden),
                confirm_label="Delete",
                cancel_label="Keep",
                on_confirm=lambda: [delete_asset(ctx, j) for j in picked],
            )
        )


def _delete_message(total: int, hidden: int) -> str:
    """What the confirm says, as a function rather than inline so the wording
    the destructive path uses is something a test can read."""
    message = f"{total} jobs and everything derived from them."
    if hidden:
        message += f" {hidden} of them are not in the list you can see."
    return message


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
        # Inside the branch: the measurement arrives on a task thread, so
        # ``storage`` is empty for every frame between launch and the first
        # reply -- and a ``same_line`` there attached to the *list child* above,
        # which is full width, putting Prune 82 px past the panel's right edge.
        # Unclickable exactly while a new install has nothing else on screen.
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

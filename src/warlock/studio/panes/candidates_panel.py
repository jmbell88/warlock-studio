"""The mesh-candidate picker: three attempts at one asset, and which to keep.

Drawn at the top of the 3D inspector, above the identity header, because it is
a question the user is being asked rather than a description of what they are
looking at -- and because the rows it lists are hidden from the library, so
this is the only way back to them.

Two rules are worth stating here rather than in a comment at the call site.

**Selecting a candidate changes the selection and nothing else.** It calls
``state.select``, and the frame loop's ``_sync_viewer`` does the rest -- which
is what keeps the pose-mode guard, the ``viewer.pending`` drop rule and the
off-thread parse applying to a candidate exactly as they apply to any other
asset. A shortcut into ``viewer.load_model`` from here would have to reproduce
all three, and would get one of them wrong.

**Keep never deletes.** It settles the group (every member becomes an ordinary
asset) and then *offers* the losers to the ordinary delete path behind the
ordinary confirm. Deleting two meshes on a single click, because the user
pressed the button that means "I like this one", is not a trade anybody agreed
to.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...service import jobs as svc_jobs
from .. import candidates as candidates_mod
from .. import dialogs, widgets
from ..manual import render as manual_render
from . import library


def draw(ctx: Any) -> None:
    """Draw the picker for the newest undecided group, if there is one."""
    group = candidates_mod.pending(ctx.cache.jobs)
    if group is None:
        return
    widgets.section("Candidates")
    manual_render.help_button(ctx, "candidates")
    if group.finished:
        widgets.muted(f"{len(group.members)} meshes from one reference. Keep one.")
    else:
        widgets.muted(
            f"{group.done_count} of {len(group.members)} finished. "
            "Keep becomes available once they all have."
        )
    selected = ctx.state.selected
    for member in group.members:
        _member(ctx, group, member, selected == member["id"])
    imgui.separator()


def _member(ctx: Any, group: Any, member: dict[str, Any], current: bool) -> None:
    job_id = member["id"]
    label = candidates_mod.label(member)
    if current:
        if widgets.primary_button(f"{label}##candidate-{job_id}", (44, 0)):
            select(ctx, job_id)
    elif imgui.button(f"{label}##candidate-{job_id}", (44, 0)):
        select(ctx, job_id)
    # Safe against the pane edge: a 44 px button plus one item spacing inside a
    # 300 px sidebar leaves most of the line. The smoke-test guard measures it.
    imgui.same_line()
    widgets.muted(candidates_mod.status_line(member))
    # Keep is offered on the selected candidate only, and only once every
    # member has settled: keeping one dissolves the group, so a member still
    # queued would quietly become an asset nobody chose.
    if current:
        ready = group.finished and member.get("status") == "done"
        if widgets.disabled_button(
            f"Keep this one##keep-{job_id}",
            ready,
            (-1, 0),
            # Keeping one dissolves the group, so a member still queued would
            # quietly become an asset nobody chose -- which is why the gate is
            # about the *group* even though the button is on one candidate.
            reason="The other attempts have not finished yet."
            if not group.finished
            else "This one did not finish, so there is nothing to keep.",
        ):
            keep(ctx, group, job_id)
        if not ready and member.get("status") != "done":
            widgets.hint_text("This one did not finish; keep another.")


def select(ctx: Any, job_id: str) -> None:
    """Show a candidate. The viewer follows the selection, not this click."""
    ctx.state.select(job_id)


def keep(ctx: Any, group: Any, job_id: str) -> None:
    """Settle the group, then offer the losers to the delete path."""
    losers = group.losers(job_id)
    try:
        svc_jobs.keep_candidate(ctx.svc, job_id)
    except Exception as exc:
        # Inline rather than on a task thread: it is one UPDATE against the
        # store the frame loop already reads directly, and the answer decides
        # what the confirm below says.
        ctx.toast(f"That candidate could not be kept: {exc}", "error", action="log")
        return
    ctx.cache.invalidate()
    ctx.toast("Kept. The other attempts are in the library now.")
    if not losers:
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title=f"Delete the other {len(losers)}?",
            message=(
                "The attempts you did not keep are ordinary assets now. "
                "Deleting removes them and everything derived from them."
            ),
            confirm_label="Delete",
            cancel_label="Keep them",
            # The library's own delete, so a loser goes by the one path that
            # clears the selection and the tick set as well as the row.
            on_confirm=lambda: [library.delete_asset(ctx, i) for i in losers],
        )
    )

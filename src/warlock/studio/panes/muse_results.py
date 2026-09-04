"""Muse's results tray: the takes this mode has made, newest first.

The centre pane. What it draws is *not* a viewport -- Muse is deliberately
absent from ``modes.VIEWPORT_MODES``, because there is no asset to frame -- but a
grid of cards, one per music job row, read from the same ``ctx.cache.jobs`` the
Library reads.

**The card is an audio card**, which is the one place this departs from the
mesh-shaped candidate grid it is otherwise modelled on. Where that one offers
"Make 3D" it offers **Open in Sirens**, and where it shows a thumbnail it shows
a Play/Stop toggle -- because a picture of a waveform tells a listener nothing
that pressing play does not tell them better, and a thumbnail nobody can read is
worse than a control they can.

Filtering to this mode's own rows rather than showing every job: a tray that
listed meshes would be a second Library, and the Library is one implementation
that already exists.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, muse_mode, widgets
from ..tokens import sp

#: A card's size in design pixels. Wide enough for two lines of tags at a
#: readable width and for the two buttons on one row beneath them.
CARD_W = 260.0
CARD_H = 132.0


def plan_for(ctx: Any) -> list[dict[str, Any]]:
    """This mode's rows, newest first.

    Off ``ctx.cache.jobs`` rather than a query of its own: the cache is already
    refreshed on the app's own schedule, and a second poll would be a second
    answer to "what has finished" that could disagree with the Library's.
    """
    return [job for job in reversed(ctx.cache.jobs) if job.get("kind") == "music"]


def should_draw(ctx: Any) -> bool:
    """Whether there is anything to show. -> False for a first visit."""
    return bool(plan_for(ctx))


def draw(ctx: Any) -> None:
    # Once per frame, before anything reads ``playing_job``: a take that ran to
    # its end stops being the playing one without any card having to notice.
    muse_mode.sync(ctx)
    jobs = plan_for(ctx)
    if not jobs:
        widgets.empty_state(
            icons.MUSIC,
            "No takes yet",
            "Describe the music you want above and press Generate. Each take is "
            "its own row -- keep the ones you like, delete the rest.",
        )
        return
    _grid(ctx, jobs)


def _grid(ctx: Any, jobs: list[dict[str, Any]]) -> None:
    """Cards, wrapped to the pane's width."""
    width = sp(CARD_W)
    gap = imgui.get_style().item_spacing.x
    avail = imgui.get_content_region_avail().x
    per_row = max(1, int((avail + gap) // (width + gap)))
    for index, job in enumerate(jobs):
        if index % per_row:
            imgui.same_line()
        _card(ctx, job, width)


def _card(ctx: Any, job: dict[str, Any], width: float) -> None:
    job_id = str(job["id"])
    state = muse_mode.ensure(ctx)
    with widgets.card(f"muse-take/{job_id}", (width, sp(CARD_H))):
        if imgui.is_item_clicked():
            state.selected_job = job_id
        widgets.stage_badge(job, inline=True)
        imgui.same_line()
        widgets.status_pill(str(job.get("status") or ""))
        prompt = str(job.get("prompt") or "")
        widgets.muted_wrapped(widgets.fit_text(prompt, width) if prompt else "(no tags)")
        params = job.get("params") or {}
        duration = params.get("actual_duration") or params.get("duration")
        if duration:
            widgets.secondary(f"{float(duration):.0f}s")
        _actions(ctx, job, job_id)


def _actions(ctx: Any, job: dict[str, Any], job_id: str) -> None:
    """Play/Stop and Open in Sirens, both dead until the WAV exists.

    Gated on the row's *status* rather than on the file: a queued take has no
    audio yet, and a button that reads the disk every frame to find that out
    would be a stat per card per frame for an answer the row already carries.
    """
    ready = str(job.get("status") or "") == "done"
    playing = muse_mode.is_playing(ctx, job_id)
    if widgets.ghost_button(
        "Stop" if playing else "Play",
        enabled=ready,
        reason="" if ready else "this take has not finished yet",
    ):
        if playing:
            muse_mode.stop(ctx)
        else:
            muse_mode.play(ctx, job_id)
    imgui.same_line()
    if widgets.ghost_button(
        "Open in Sirens",
        enabled=ready,
        reason="" if ready else "this take has not finished yet",
        tooltip="Import this track into the tracker as a sample instrument.",
    ):
        muse_mode.open_in_sirens(ctx, job_id)


__all__ = ["CARD_H", "CARD_W", "draw", "plan_for", "should_draw"]

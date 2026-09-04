"""The in-context generation plan and results tray for Create.

Create used to send people from a form, to a floating progress card, to the
Library.  This small surface keeps the work that a submit starts in the same
place: the plan says what a press costs, and the tray becomes the place to
watch, compare, keep, and vary the result.  It deliberately reads the existing
job cache and services; it does not introduce another generation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

from ..service import jobs as svc_jobs
from ..service import sprites as svc_sprites
from . import asset_open, controls, create_assets, widgets
from . import candidates as candidates_mod
from .panes import thumbs
from .tokens import sp

#: How many finished results the tray shows, and the width of its grid. One
#: number because they are one fact: the tray is a fixed-height strip, so the
#: row it can draw whole is the row it should hold.
_RESULT_COLUMNS = 3


@dataclass(frozen=True)
class Plan:
    """The human-readable work implied by one Create press."""

    candidates: int
    generations: int
    duration: str
    stages: str
    recipe: str

    @property
    def count_line(self) -> str:
        noun = "candidate" if self.candidates == 1 else "candidates"
        return f"{self.candidates} {noun} · {self.generations} image generation"


def plan_for(form: dict[str, Any], resolved: Any = None) -> Plan:
    """Describe the actual work using the same form values the door receives.

    Kept presentation-only and safe for partially restored forms, so it can be
    called every frame before the service performs its final validation.
    """
    spec = create_assets.selected(form)
    candidates = max(1, _integer(form.get("count"), 1))
    generations = candidates
    duration = "a few seconds"
    stages = "Generate image"
    if spec.key == "3d_model":
        stages = "Generate reference → choose or make 3D"
    elif spec.key == "seamless_material":
        stages = "Generate seamless material → verify wrap"
    elif spec.key == "tileset":
        candidates = 1
        generations = 1
        duration = "about a minute"
        stages = "Generate tile sheet → inspect cells"
    elif spec.key == "sprite_sheet":
        # The sprite follow-up has one preliminary character plus one sheet
        # image per planned cell/candidate.  Import lazily: settings_2d owns
        # the UI layout vocabulary and importing it at module load cycles.
        from .panes import settings_2d

        sprite = settings_2d.sprite_plan(form)
        candidates = int(sprite["candidates"])
        generations = 1 + int(sprite["generations"])
        duration = svc_sprites.generation_time_phrase(generations)
        stages = (
            f"1 character reference + {int(sprite['generations'])} sheet generation"
            + ("s" if int(sprite["generations"]) != 1 else "")
        )
    recipe = "Automatic recipe"
    if resolved is not None:
        recipe = str(getattr(resolved, "base_model", "") or "Automatic recipe")
    return Plan(candidates, generations, duration, stages, recipe)


def should_draw(ctx: Any) -> bool:
    """Whether Create has work worth reserving central space for.

    **The same question :func:`draw` answers**, which is the fix: this asked
    "is there any queued, running or done job in the first twelve rows" while
    the tray showed a running job, a candidate group, or ``_recent_results``
    -- which excludes candidate members. So the two disagreed in both
    directions: a corpus of nothing but candidate rows reserved a strip and
    drew the empty state into it, and the viewer lost ``tray_height`` for a
    tray with nothing in it from the first finished job onward, permanently.
    """

    cache = getattr(ctx, "cache", None)
    if cache is None:
        return False
    if getattr(cache, "active", None) is not None:
        return True
    if candidates_mod.pending(cache.jobs) is not None:
        return True
    return bool(_recent_results(ctx))


def draw(ctx: Any, height: float = 0.0) -> None:
    """Draw the persistent results-and-iteration tray in the Create canvas."""
    if height > 0 and not imgui.begin_child("generation-results", (0, height), False):
        imgui.end_child()
        return
    widgets.pane_header("Generations")
    widgets.muted(_brief_caption(ctx))
    active = getattr(ctx.cache, "active", None)
    if active is not None:
        _progress(ctx, active)
    group = candidates_mod.pending(ctx.cache.jobs)
    if group is not None:
        _candidate_grid(ctx, group)
    else:
        jobs = _recent_results(ctx)
        if jobs:
            _result_grid(ctx, jobs)
        elif active is None:
            # Only reachable from a caller that draws the tray without asking
            # ``should_draw`` first; the shell always asks.
            widgets.muted(
                "Your completed generations will appear here for comparison "
                "and variation."
            )
    if height > 0:
        imgui.end_child()


def _brief_caption(ctx: Any) -> str:
    form = getattr(ctx.state, "form_2d", {})
    prompt = str(form.get("prompt") or "").strip()
    if not prompt:
        return "The current brief stays editable at left."
    return (prompt[:78] + "…") if len(prompt) > 79 else prompt


def _progress(ctx: Any, job: dict[str, Any]) -> None:
    status = str(job.get("status") or "queued")
    name = str(job.get("name") or job.get("prompt") or "Current generation")
    widgets.secondary("Working now")
    imgui.text_wrapped(name)
    if status == "queued":
        position = queue_position(ctx, str(job.get("id") or ""))
        widgets.muted(f"Queued{f' · position {position}' if position else ''}")
        _cancel(ctx, str(job.get("id") or ""))
        return
    progress = ctx.runtime.progress(str(job.get("id") or ""))
    if progress is not None:
        widgets.progress_bar(float(progress.get("percent") or 0.0))
        widgets.muted(str(progress.get("label") or "Generating…"))
    else:
        widgets.muted("Starting generation…")
    _cancel(ctx, str(job.get("id") or ""))


def _cancel(ctx: Any, job_id: str) -> None:
    """The progress card's own Cancel, on the tray's copy of its narration.

    This block says what the floating card says and used to say it without the
    one control the card carries, so the duplicate was strictly worse than the
    thing it duplicated. No confirmation, for the card's reason: the button
    says exactly what it does and sits on the thing it acts on.
    """

    if not job_id:
        return
    busy = ctx.busy(f"cancel:{job_id}")
    if widgets.disabled_button(
        f"Cancel##tray-cancel-{job_id}", not busy, reason="Cancelling..."
    ):
        ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)


def _candidate_grid(ctx: Any, group: Any) -> None:
    """Every candidate, in a strip that scrolls.

    Unlike the results grid this one cannot be trimmed to a row: a count of 8
    means eight candidates and choosing between them is the entire purpose, so
    the grid is put in a scrolling child rather than being cut short.
    """
    widgets.secondary("Compare candidates")
    widgets.muted(
        "Choose one when every candidate settles. Seeds and scores stay with each result."
    )
    if not imgui.begin_child("generation-candidate-scroll", (0, 0), False):
        imgui.end_child()
        return
    if imgui.begin_table("generation-candidates", 2, imgui.TableFlags_.sizing_stretch_same.value):
        for member in group.members:
            imgui.table_next_column()
            _result_card(ctx, member, group=group)
        imgui.end_table()
    imgui.end_child()


def _result_grid(ctx: Any, jobs: list[dict[str, Any]]) -> None:
    widgets.secondary("Compare and refine")
    if imgui.begin_table(
        "generation-results", _RESULT_COLUMNS, imgui.TableFlags_.sizing_stretch_same.value
    ):
        for job in jobs:
            imgui.table_next_column()
            _result_card(ctx, job)
        imgui.end_table()


def _result_card(ctx: Any, job: dict[str, Any], group: Any = None) -> None:
    job_id = str(job["id"])
    thumbs.job_thumb(ctx, job, sp(72))
    imgui.same_line()
    imgui.begin_group()
    widgets.status_pill(str(job.get("status") or "queued"))
    params = job.get("params") or {}
    seed = params.get("seed", params.get("mesh_seed"))
    if seed is not None:
        widgets.muted(f"seed {seed}")
    rank = params.get("rank")
    score = rank.get("score") if isinstance(rank, dict) else None
    if score is not None:
        # Named for what it is. "score 72%" reads as a measurement of the
        # picture; it is the trained probe's *probability that you would keep
        # this one* (``judge.score``), which is a guess about the reader and
        # advisory by construction -- the job records nothing when the probe is
        # missing or the embedding width has changed.
        widgets.muted(f"judge: {float(score) * 100:.0f}% likely a keeper")
    imgui.end_group()
    # **Two per row, not one per row.** Four full-width buttons stacked under a
    # 72 dp thumbnail make a card taller than the tray that holds it, and the
    # tray is the bottom of a column whose height is a fraction of the window --
    # so the last two fell outside it and could not be pressed at all.
    # ``/exercise-mode create`` reported fifteen clipped controls, every one of
    # them one of these; no test saw it, because a clipped button is drawn.
    half = (_half_width(), 0.0)
    status = str(job.get("status") or "")
    done = status == "done"
    not_ready = _why_not_finished(job, status)

    if controls.button(f"Open##result-open-{job_id}", half):
        # **The one door** (``asset_open.open_asset``), not a bare ``select``:
        # selecting alone leaves ``source_job`` stale on a reference, and a mesh
        # result opened this way showed ``input.png`` on the Reference stage.
        asset_open.open_asset(ctx, job)
    imgui.same_line()
    if widgets.disabled_button(f"Vary##result-vary-{job_id}", done, half, reason=not_ready):
        _vary(ctx, job)

    if group is not None:
        ready = group.finished and done
        if widgets.disabled_button(
            f"Keep##result-keep-{job_id}",
            ready,
            half,
            reason=(
                "Wait for every candidate to finish."
                if not group.finished
                else "This result did not finish."
            ),
        ):
            from .panes import candidates_panel

            candidates_panel.keep(ctx, group, job_id)
        imgui.same_line()

    # **Rerun is live on a failure.** ``rerun_job`` needs only the brief and the
    # reference the row already has, and the library card has always offered
    # "Try again" on exactly these rows; disabling it here with "not ready yet"
    # was both the wrong reason and the wrong answer.
    can_rerun = done or status in _FAILED
    if widgets.disabled_button(
        f"Rerun##result-rerun-{job_id}", can_rerun, half, reason=not_ready
    ):
        ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll")
    if group is None:
        imgui.same_line()
    is_reference = job.get("stage") == "reference" and "input.png" in (job.get("files") or [])
    if widgets.disabled_button(
        f"Make 3D##result-3d-{job_id}",
        done and is_reference,
        half,
        reason="A finished reference image is required.",
    ):
        _make_3d(ctx, job)


#: Statuses a job can end in without producing artifacts.
_FAILED = frozenset({"error", "cancelled", "failed"})


def _why_not_finished(job: dict[str, Any], status: str) -> str:
    """Why a control that needs a finished result is greyed, for *this* row.

    "This result is not ready yet." is true of a queued job and false of one
    that failed an hour ago; a disabled control that explains itself has to
    tell the two apart.
    """

    if status in ("error", "failed"):
        detail = str(job.get("error") or "").strip()
        return f"This generation failed: {detail}" if detail else "This generation failed."
    if status == "cancelled":
        return "This generation was cancelled."
    return "This result is not ready yet."


def _half_width() -> float:
    """Half the cell, less the gap between the two buttons that share it."""
    return (imgui.get_content_region_avail().x - imgui.get_style().item_spacing.x) * 0.5


def _make_3d(ctx: Any, job: dict[str, Any]) -> None:
    from .panes import settings_3d

    ctx.state.source_job = str(job["id"])
    settings_3d.promote(ctx, job, ctx.state.form_3d)


def _vary(ctx: Any, job: dict[str, Any]) -> None:
    """Copy a result's recorded brief back to the live form for a controlled edit."""
    from . import create_stages
    from .panes import library

    library.copy_settings(ctx, job)
    create_stages.go(ctx, "reference", follow=False)
    ctx.toast("Loaded this brief. Change one thing, then generate a variation.")


def _recent_results(ctx: Any) -> list[dict[str, Any]]:
    """The most recent finished results. **One row of the grid, not two.**

    Six filled the tray's three columns twice over, and the tray is a
    fixed-height strip -- so the second row's cards were drawn with their
    actions below the fold, where nothing can press them. Three whole cards
    beat six half-drawn ones, and the library beside them holds the rest.
    """
    return [
        job
        for job in ctx.cache.jobs
        if job.get("status") in ("done", "error", "cancelled") and not job.get("candidate_group")
    ][:_RESULT_COLUMNS]


def queue_position(ctx: Any, job_id: str) -> int | None:
    """Where a queued job sits in line, or None if it is not queued.

    Public: the plan footer in ``panes.settings_2d`` asks the same question,
    and was reaching for the private name to do it.
    """

    queued = [job for job in reversed(ctx.cache.jobs) if job.get("status") == "queued"]
    for index, job in enumerate(queued, start=1):
        if job.get("id") == job_id:
            return index
    return None


def _integer(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback

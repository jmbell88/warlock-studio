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
from . import candidates as candidates_mod
from . import controls, create_assets, widgets
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
        duration = settings_2d.svc_sprites.generation_time_phrase(generations)
        stages = (
            f"1 character reference + {int(sprite['generations'])} sheet generation"
            + ("s" if int(sprite["generations"]) != 1 else "")
        )
    recipe = "Automatic recipe"
    if resolved is not None:
        recipe = str(getattr(resolved, "base_model", "") or "Automatic recipe")
    return Plan(candidates, generations, duration, stages, recipe)


def should_draw(ctx: Any) -> bool:
    """Whether Create has work worth reserving central space for."""
    jobs = getattr(getattr(ctx, "cache", None), "jobs", ()) or ()
    return any(job.get("status") in ("queued", "running", "done") for job in jobs[:12])


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
            widgets.muted(
            "Your completed generations will appear here for comparison and variation."
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
        position = _queue_position(ctx, str(job.get("id") or ""))
        widgets.muted(f"Queued{f' · position {position}' if position else ''}")
        return
    progress = ctx.runtime.progress(str(job.get("id") or ""))
    if progress is not None:
        widgets.progress_bar(float(progress.get("percent") or 0.0))
        widgets.muted(str(progress.get("label") or "Generating…"))
    else:
        widgets.muted("Starting generation…")


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
        widgets.muted(f"score {float(score) * 100:.0f}%")
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
    not_ready = "This result is not ready yet."

    if controls.button(f"Open##result-open-{job_id}", half):
        ctx.state.select(job_id)
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

    if widgets.disabled_button(f"Rerun##result-rerun-{job_id}", done, half, reason=not_ready):
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

    library._copy_settings(ctx, job)
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


def _queue_position(ctx: Any, job_id: str) -> int | None:
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

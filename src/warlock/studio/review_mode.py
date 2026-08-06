"""Review mode's controller: launching sweeps, and the verdict loop over them.

The ``clay_mode.py`` pattern -- state and logic here, drawing in ``main.py``,
no imgui anywhere under this import -- so the part that is easy to get wrong is
assertable without a GL context.

What it is for. A sweep queues a family of settings vectors as ordinary jobs;
this mode is where they are looked at, one at a time, with the reference image
beside the mesh and two keys to say whether it worked. Those verdicts compile
into ``findings.json``, which is what puts an "accept 6/8" under a control in
the generate panes and what a saved vector preset is built from. The loop
closes here.

Four rules shape it.

**A verdict is filed against a job id, and carries a snapshot of that job's
whole config vector.** Not against a run-directory unit key, which only meant
anything inside one run and evaporated when the run was pruned. The snapshot is
what lets ``prune_jobs`` delete the assets without deleting what was learned
from them -- see ``service/verdicts.py``.

**Anything finished is reviewable.** The first bucket is not a sweep at all: it
is the recent finished meshes nobody has judged, which is how ordinary daily
use feeds the same findings pool a deliberate sweep does.

**Reject waits for a reason.** ``R`` arms; the verdict is not written until one
of the five reason keys is pressed. A bare rejection is a row the findings can
count and nothing else, and the tally of *why* things fail is the one thing a
sweep exists to produce. Accept has no such second step, because there is only
one way for a mesh to be right.

**The advance is to the next thing to do, not the next row.** A session is
resumed far more often than it is started, so opening a sweep lands on its
first unverdicted unit and recording steps past everything already answered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..service import verdicts as verdicts_mod

log = logging.getLogger(__name__)

# Task keys. Prefixed, because the app claims results by prefix: a key without
# one is a result delivered nowhere.
SCAN_KEY = "review-scan"
DELETE_KEY = "review-delete"
FINDINGS_KEY = "review-findings"

# What this reviewer is called. A free string by design -- verdicts are keyed
# on (job_id, source), so a future judge writing "ai:<model>" sits beside a
# human's verdict rather than overwriting it.
SOURCE = verdicts_mod.SOURCE_HUMAN

# 1-5, in the order verdicts.REASONS lists them, so the pane can number the
# buttons off the same table and the two can never disagree.
REASON_KEYS = {str(i + 1): reason for i, reason in enumerate(verdicts_mod.REASONS)}

# The synthetic first bucket: finished meshes from ordinary use that nobody has
# judged. Not a sweep row, so it has no spec and cannot be deleted.
RECENT_ID = "recent"
RECENT_LABEL = "Recent, unreviewed"

# The reference image a unit was generated from, in the order to look. A text
# job writes reference.png (what trellis actually saw); an upload has only the
# input it was given.
REFERENCE_NAMES = ("reference.png", "input.png")


@dataclass
class SweepForm:
    """The New sweep form. Axis rows are (param, comma-separated values) so the
    control is a text field and the parsing is one place."""

    prompt: str = ""
    seeds: str = "42"
    axes: list[dict[str, str]] = field(default_factory=lambda: [{"param": "", "values": ""}])
    label: str = ""
    # Fixed at "model" with no widget on purpose: a sweep exists to judge
    # meshes, and service.sweeps keeps "reference" expressible for a future
    # cheap-stage form rather than for this one.
    stage: str = "model"
    # The settings the units start from, captured from the 2D/3D forms the user
    # has already tuned. Empty until "start from current settings" is pressed --
    # a sweep off an unstated baseline is not reproducible.
    base: dict[str, Any] = field(default_factory=dict)
    base_note: str = ""
    submitting: bool = False


@dataclass
class ReviewState:
    """One review session: which sweeps exist, which one is open, where in it.

    Nothing here is persisted. A stored sweep id would outlive the sweep it
    names -- deleting one, or a fresh data dir, both make it an id for nothing
    -- and a mode that opens on an error is worse than one that opens on a list.
    """

    # [{id, label, prompt, units, todo}], the recent-unreviewed bucket first
    # then sweeps newest-first.
    sweeps: list[dict[str, Any]] = field(default_factory=list)
    sweep_id: str | None = None
    # The open sweep's units. The dicts are the same objects the matching
    # ``sweeps`` entry holds, so recording a verdict updates both the row and
    # the list's remaining count.
    units: list[dict[str, Any]] = field(default_factory=list)
    index: int = 0
    # Whether R has been pressed and a reason key is what the mode is waiting
    # for. Cleared by anything that moves, because the armed state belongs to
    # the unit that was on screen when it was armed.
    pending_reject: bool = False
    # There is deliberately no "which unit is loaded" field here. What the
    # shared viewer is showing is ``viewer.path``, and a second copy of that
    # answer is a way for the two to disagree -- see ``main._review_viewport``.
    scanning: bool = False
    form: SweepForm = field(default_factory=SweepForm)


def ensure(ctx: Any) -> ReviewState:
    """The mode's state, built on first use -- lazy for the reason Clay's is:
    a session that never reviews anything should not pay for it."""
    state = ctx.state.review
    if state is None:
        state = ReviewState()
        ctx.state.review = state
    return state


# --- scanning ----------------------------------------------------------------


def _unit(job: dict[str, Any], recorded: dict[tuple[str, str], dict[str, Any]],
          job_dir: Path) -> dict[str, Any]:
    seen = recorded.get((job["id"], SOURCE)) or {}
    return {
        "job_id": job["id"],
        "label": job.get("sweep_unit") or job.get("name") or job.get("prompt") or job["id"],
        "status": job.get("status"),
        "params": job.get("params") or {},
        "dir": job_dir,
        "verdict": seen.get("verdict"),
        "reasons": list(seen.get("reasons") or ()),
    }


def _collect(svc: Any) -> list[dict[str, Any]]:
    """Every sweep and its units, plus the recent-unreviewed bucket.

    Task thread only: several DB reads behind one serialized connection, and
    the frame loop must never queue behind them.
    """
    out: list[dict[str, Any]] = []

    recent = svc.store.unverdicted_models(source=SOURCE)
    out.append(
        {
            "id": RECENT_ID,
            "label": RECENT_LABEL,
            "prompt": "Finished meshes from ordinary use that nobody has judged yet.",
            "units": [_unit(job, {}, svc.job_dir(job["id"])) for job in recent],
            "todo": len(recent),
        }
    )

    for sweep in svc.store.list_sweeps():
        jobs = svc.store.sweep_jobs(sweep["id"])
        recorded = svc.store.verdicts_for([j["id"] for j in jobs], source=SOURCE)
        units = [_unit(job, recorded, svc.job_dir(job["id"])) for job in jobs]
        out.append(
            {
                "id": sweep["id"],
                "label": sweep["label"],
                "prompt": sweep["prompt"],
                "units": units,
                "todo": sum(1 for u in units if u["verdict"] is None),
            }
        )
    return out


def scan(ctx: Any) -> None:
    """Re-read the sweeps and their units, off the frame thread."""
    state = ensure(ctx)
    if state.scanning:
        return
    state.scanning = True
    if not ctx.submit(SCAN_KEY, _collect, ctx.svc):
        # The runner refuses a key already in flight. Leaving the flag set here
        # is what makes the mode permanently inert after a double click.
        state.scanning = False


def delete(ctx: Any, sweep_id: str) -> bool:
    """Delete a sweep's jobs and meshes, off the frame thread.

    The verdicts stay: each carries the config vector it was filed against, so
    what the sweep taught survives the assets it taught it with.
    """
    from ..service import sweeps as sweeps_mod

    if sweep_id == RECENT_ID:
        return False
    return bool(ctx.submit(DELETE_KEY, sweeps_mod.delete_sweep, ctx.svc, sweep_id))


def refresh_findings(ctx: Any) -> None:
    """Recompute findings.json after a verdict, off the frame thread.

    Fire-and-forget: the panes read the file through an mtime cache, so a
    refresh that is refused because one is already in flight simply means the
    next verdict's refresh picks up both.
    """
    from ..service import findings as findings_mod

    ctx.submit(FINDINGS_KEY, findings_mod.refresh, ctx.svc)


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``review-`` key."""
    state = ensure(ctx)
    if done.key == DELETE_KEY:
        removed = 0
        if isinstance(done.result, dict):
            removed = int(done.result.get("deleted") or 0)
        ctx.toast(f"Deleted {removed} job(s). Verdicts and findings kept.")
        # The counts are now wrong and the viewer may be showing a mesh that no
        # longer exists -- both are fixed by the rescan.
        if state.sweep_id is not None and state.sweep_id != RECENT_ID:
            state.sweep_id = None
        scan(ctx)
        return
    if done.key == FINDINGS_KEY:
        return
    if done.key != SCAN_KEY:
        return
    state.scanning = False
    if not isinstance(done.result, list):
        return
    state.sweeps = done.result
    # Whatever was open stays open if the rescan still finds it; otherwise the
    # first bucket, which is what an empty session wants and a deleted one
    # needs.
    wanted = state.sweep_id
    if wanted is not None and any(s["id"] == wanted for s in state.sweeps):
        open_sweep(ctx, wanted)
    elif state.sweeps:
        open_sweep(ctx, state.sweeps[0]["id"])
    else:
        state.sweep_id, state.units, state.index = None, [], 0


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed scan must not leave the mode inert: ``scanning`` gates every
    button and every key. A failed launch must not leave the form inert either.
    """
    state = ctx.state.review
    if state is None:
        return
    if done.key == SCAN_KEY:
        state.scanning = False
    if done.key == DELETE_KEY:
        ctx.toast("Could not delete that sweep.", "error")


# --- the open sweep ----------------------------------------------------------


def open_sweep(ctx: Any, sweep_id: str) -> None:
    """Show a sweep, starting on the first unit with no verdict.

    Resuming is the common case: landing back on unit one every time is what
    makes it useless.
    """
    state = ensure(ctx)
    entry = next((s for s in state.sweeps if s["id"] == sweep_id), None)
    state.sweep_id = sweep_id
    state.units = list(entry["units"]) if entry is not None else []
    state.pending_reject = False
    state.index = next(
        (i for i, unit in enumerate(state.units) if unit["verdict"] is None), 0
    )


def current(state: ReviewState) -> dict[str, Any] | None:
    if 0 <= state.index < len(state.units):
        return state.units[state.index]
    return None


def step(state: ReviewState, delta: int) -> None:
    """Move by hand, clamped at both ends. Disarms, because the armed state
    belongs to the unit that was on screen when R was pressed."""
    if not state.units:
        return
    state.index = min(max(state.index + delta, 0), len(state.units) - 1)
    state.pending_reject = False


def advance(state: ReviewState, *, unverdicted_only: bool = False) -> None:
    """Forward one, or on to the next thing left to do.

    ``unverdicted_only`` is what a recorded verdict uses, and it means exactly
    what it says: the next unit **with no verdict**, searched forward and then
    wrapping to the start, and staying put when there is none anywhere. Falling
    through to a plain +1 would put the cursor on a unit that had already been
    answered and ask the same question twice -- the one thing the flag exists to
    prevent. Wrapping is the honest completion of the same idea.
    """
    if not state.units:
        return
    state.pending_reject = False
    if unverdicted_only:
        order = list(range(state.index + 1, len(state.units))) + list(range(state.index))
        ahead = next((i for i in order if state.units[i]["verdict"] is None), None)
        if ahead is not None:
            state.index = ahead
        return
    state.index = min(state.index + 1, len(state.units) - 1)


def record(ctx: Any, verdict: str, reasons: Any = ()) -> None:
    """Write one verdict for the unit on screen, then move on.

    Inline on the frame thread on purpose: one INSERT under the store's RLock,
    the same IO class as ``settings.set``, and putting it on a task thread would
    mean a keypress whose effect arrives some frames later -- which, at the rate
    a reviewer presses A, reorders verdicts against navigation. The findings
    recompute that follows *is* a task, because it reads every verdict and
    writes a file.
    """
    from ..service.errors import ServiceError

    state = ensure(ctx)
    unit = current(state)
    if unit is None or state.scanning:
        return
    reasons = list(reasons)
    try:
        verdicts_mod.record_verdict(
            ctx.svc, unit["job_id"], verdict=verdict, reasons=reasons, source=SOURCE
        )
    except (ServiceError, OSError):
        log.exception("could not record a verdict for %s", unit["job_id"])
        ctx.toast("Could not record that verdict.", "error")
        return
    unit["verdict"] = verdict
    unit["reasons"] = reasons
    _recount(state)
    refresh_findings(ctx)
    advance(state, unverdicted_only=True)


def _recount(state: ReviewState) -> None:
    """Keep the sweep list's remaining count true. The unit dicts are shared
    with the ``sweeps`` entry, so only the tally needs recomputing."""
    for sweep in state.sweeps:
        if sweep["id"] == state.sweep_id:
            sweep["todo"] = sum(1 for unit in sweep["units"] if unit["verdict"] is None)


# --- launching a sweep -------------------------------------------------------


def capture_base(ctx: Any) -> dict[str, Any]:
    """The settings vector the two generate forms currently describe.

    Reusing the forms the user has already tuned is the whole point: a sweep is
    "this, but vary that", and re-picking a checkpoint and eleven taxonomy
    selects inside a second form would be its own small hell.
    """
    from .. import guidance

    state = ctx.state
    form_2d, form_3d = state.form_2d, state.form_3d
    known = set(guidance.form_fields())
    base: dict[str, Any] = {
        k: v for k, v in form_2d.items() if k in known and v not in ("", None)
    }
    if form_2d.get("style_lora"):
        base["lora_weight"] = float(form_2d["lora_weight"])
    if form_2d.get("negative_prompt"):
        base["negative_prompt"] = form_2d["negative_prompt"]
    # The 3D pane's platform is the geometry resolution and wins over the 2D
    # pane's prompt-fragment one, because a sweep unit runs to a mesh.
    if form_3d.get("platform"):
        base["platform"] = form_3d["platform"]
    if form_3d.get("bg_removal"):
        base["bg_removal"] = form_3d["bg_removal"]
    if form_3d.get("profile"):
        base["profile"] = form_3d["profile"]
    if float(form_3d.get("size_m") or 0) > 0:
        base["size_m"] = float(form_3d["size_m"])
    base["reference_prep"] = bool(form_3d.get("reference_prep"))
    return base


def parse_seeds(text: str) -> tuple[int, ...]:
    """A comma-separated seed list. Raises ValueError, which the caller turns
    into a toast -- an unparseable seed is a typo, not a crash."""
    out: list[int] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(int(raw))
    if not out:
        raise ValueError("a sweep needs at least one seed")
    return tuple(out)


def _coerce(value: str) -> Any:
    """An axis value typed as text, as the type the param wants.

    int before float before bool-ish before string, because "8" must reach
    ``trellis_band`` as 8 and not as "8" -- ``check_trellis_band`` demands a
    whole number, and a string would be refused with a message about the type
    rather than about the value.
    """
    text = value.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def build_plan(state: ReviewState) -> Any:
    """The form as a ``SweepPlan``. Raises ValueError for a malformed form."""
    from ..service.sweeps import Axis, SweepPlan

    form = state.form
    axes = []
    for row in form.axes:
        param = (row.get("param") or "").strip()
        raw = (row.get("values") or "").strip()
        if not param and not raw:
            continue
        if not param or not raw:
            raise ValueError("every axis needs a parameter and at least one value")
        values = tuple(_coerce(v) for v in raw.split(",") if v.strip())
        if not values:
            raise ValueError(f"axis {param} has no values")
        axes.append(Axis(param=param, values=values))
    return SweepPlan(
        label=form.label.strip() or (form.prompt.strip()[:40] or "sweep"),
        prompt=form.prompt.strip(),
        base=dict(form.base),
        seeds=parse_seeds(form.seeds),
        axes=tuple(axes),
        stage=form.stage,
    )


def launch(ctx: Any) -> bool:
    """Validate and queue the form's sweep. -> whether anything was queued.

    Inline on the frame thread, like ``settings_2d._submit``: it is N validated
    inserts against the same store the panes already write to, and the
    validation pass is what makes the whole thing all-or-nothing.
    """
    from ..service import sweeps as sweeps_mod
    from ..service.errors import ServiceError

    state = ensure(ctx)
    if state.form.submitting or state.scanning:
        return False
    try:
        plan = build_plan(state)
    except ValueError as exc:
        ctx.toast(str(exc), "error")
        return False
    state.form.submitting = True
    try:
        result = sweeps_mod.create_sweep(ctx.svc, plan)
    except ServiceError as exc:
        ctx.toast(exc.message, "error")
        return False
    except Exception:
        log.exception("could not launch the sweep")
        ctx.toast("Could not launch that sweep.", "error")
        return False
    finally:
        state.form.submitting = False
    ctx.toast(f"Queued {result['units']} unit(s).")
    state.sweep_id = result["id"]
    scan(ctx)
    return True


def preview_units(state: ReviewState) -> int:
    """How many jobs the form would queue, or -1 if it cannot be planned yet."""
    from ..service import sweeps as sweeps_mod

    try:
        return len(sweeps_mod.expand(build_plan(state)))
    except Exception:
        return -1


# --- what the pane reads -----------------------------------------------------


def model_path(unit: dict[str, Any]) -> Path:
    return Path(unit["dir"]) / "model.glb"


def reference_path(unit: dict[str, Any]) -> Path | None:
    """What the unit was generated from, or None if neither exists."""
    for name in REFERENCE_NAMES:
        path = Path(unit["dir"]) / name
        if path.exists():
            return path
    return None


def cache_id(unit: dict[str, Any]) -> str:
    """The id the thumbnail cache files this unit's reference under.

    A job id is globally unique, which is what the old run-qualified unit key
    was working around: unit keys repeated across runs of one sweep spec, so
    the second run's references were served the first run's pixels.
    """
    return f"review:{unit['job_id']}"


def mesh_lines(unit: dict[str, Any]) -> list[str]:
    """The mesh verdicts the worker already computed, as text.

    Read off the job row's params rather than an archived ``job.json``: the row
    *is* the record now, so there is nothing to cache and nothing to go stale.

    Deliberately the two measurements kept apart everywhere else: the report
    ("will an importer accept it, will it sit on the floor") and the audit
    ("can you see through it"). Merging them into one badge is what made the old
    one claim watertight about a silhouette check.
    """
    params = unit.get("params") or {}
    lines: list[str] = []
    report = params.get("mesh_report")
    if isinstance(report, dict):
        triangles = report.get("triangles")
        if isinstance(triangles, (int, float)):
            lines.append(f"{int(triangles):,} triangles")
        if "watertight" in report:
            lines.append(f"watertight: {'yes' if report.get('watertight') else 'no'}")
        materials = report.get("materials")
        if isinstance(materials, (int, float)):
            lines.append(f"{int(materials)} material(s)")
    audit = params.get("mesh_audit")
    if isinstance(audit, dict):
        verdict = audit.get("verdict")
        if verdict:
            lines.append(f"silhouette: {verdict}")
    return lines


def label(unit: dict[str, Any]) -> str:
    """One line naming this unit -- its sweep label, or whatever an ordinary
    asset is called."""
    return str(unit.get("label") or unit.get("job_id") or "")


# --- keys --------------------------------------------------------------------


def handle_key(ctx: Any, event: Any) -> bool:
    """Review's shortcuts. -> whether the key was consumed.

    The caller returns unconditionally either way, for the reason Inker's and
    Clay's do: Review has replaced the viewport and the forms, so a key falling
    through here would toggle a wireframe or submit a job against a pane that is
    not on screen. The return value is still honest, because it is what the
    tests read.
    """
    import pygame

    state = ctx.state.review
    if state is None or state.scanning:
        return False
    if event.type != pygame.KEYDOWN:
        return False

    name = pygame.key.name(event.key)
    if event.key == pygame.K_ESCAPE:
        state.pending_reject = False
        return True
    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        step(state, -1 if event.key == pygame.K_LEFT else 1)
        return True
    if current(state) is None:
        # Nothing on screen to judge: every key below acts on a unit.
        return False
    if name == "a":
        record(ctx, "accept")
        return True
    if name == "r":
        state.pending_reject = True
        return True
    if name == "s":
        advance(state)
        return True
    if name in REASON_KEYS and state.pending_reject:
        record(ctx, "reject", (REASON_KEYS[name],))
        return True
    return False

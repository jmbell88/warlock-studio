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

**Blinding hides the arm, and that means the order too.** The review that
produced the ``bg_removal`` signal was unblinded and single-reviewer, which is
why `TODO.md` §2 asks for a small blind confirm before anything is leaned on.
``blind`` is therefore a property of the *session*, not of a sweep: it renames
every unit to a neutral id prefix and presents them in an order derived from a
stable digest of the job id. Hiding the label alone would not be blinding at
all -- ``sweeps.expand`` enqueues the baseline first and then one unit per axis
value, so in a two-arm sweep position names the arm as plainly as the label
does. It is not persisted, for the reason nothing else here is: a review
resumed blinded without saying so is worse than one that starts unblinded.
"""

from __future__ import annotations

import hashlib
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
LABELS_KEY = "review-labels"
TRAIN_KEY = "review-train"
SCORE_KEY = "review-scores"

# What this reviewer is called. A free string by design -- verdicts are keyed
# on (job_id, source), so a future judge writing "ai:<model>" sits beside a
# human's verdict rather than overwriting it.
SOURCE = verdicts_mod.SOURCE_HUMAN

# The name the judge will file under when it files anything, declared now so it
# is decided once. **Nothing writes it yet, and that is a decision rather than
# an omission**: filing a verdict needs a probability-to-accept threshold, and
# a threshold is a constant the stored corpus is then keyed on, which owes a
# ``docs/measurements/`` document first (`TODO.md` §10 says what it must
# contain). The ``(job_id, source, stage)`` seam is already built and tested, so
# the day that measurement exists this is one call.
SOURCE_AI = "ai:dino-probe"

# Which probe scores a review unit. A sweep unit is a *model*-stage job, and the
# mesh probe does not exist -- but the blank question's declared population is
# exactly these jobs' reference images (``db.LABEL_POPULATION`` maps
# ``blank -> model``), so this is not a stage mismatch: it is the one question
# there is evidence for, asked about the picture the mesh was reconstructed
# from. What is on screen says so, because a number that does not name its
# question will be read as the one the reader wanted.
SCORE_STAGE = "blank"
SCORE_QUESTION = "will this reconstruct"

# 1-5, in the order verdicts.REASONS lists them, so the pane can number the
# buttons off the same table and the two can never disagree.
REASON_KEYS = {str(i + 1): reason for i, reason in enumerate(verdicts_mod.REASONS)}

# The synthetic first bucket: finished meshes from ordinary use that nobody has
# judged. Not a sweep row, so it has no spec and cannot be deleted.
RECENT_ID = "recent"
RECENT_LABEL = "Recent, unreviewed"

# Where a unit's reference image is, in the order to look, is
# ``verdicts.IMAGE_NAMES`` and is deliberately *not* restated here. This module
# used to carry an identical copy and then read both of them -- ``_label_rows``
# through the service's, ``reference_path`` through the local one -- which is
# the ``judge.STAGES``/``verdicts.STAGES`` hazard exactly: two spellings of one
# fact, agreeing right up until a name is added to one of them.


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
class LabelPass:
    """One labelling pass over the images no probe-question has reached yet.

    A separate loop from the verdict one and deliberately so: a mesh verdict
    takes ~15 s of orbiting and carries a reason, while an image label is ~2 s
    and one bit. Same corpus, same table, different pace -- which is why this is
    a grid with two keys rather than a second copy of the verdict panel.
    """

    stage: str
    # [{job_id, prompt, image, verdict}] -- ``image`` is the path to show. Rows
    # keep their place once answered rather than being removed: a shrinking grid
    # renumbers itself under the cursor mid-pass, which is how the wrong image
    # gets judged at the rate this is meant to be worked through.
    rows: list[dict[str, Any]] = field(default_factory=list)
    index: int = 0
    # How many thumbnails have been handed to the GPU. One per frame -- see
    # ``next_thumbnail``.
    uploaded: int = 0
    loading: bool = False
    # ``judge.status``' answer as of the last task that read it -- how many labels
    # this question has, how many it needs, whether a probe exists. A snapshot
    # rather than a live call: ``status`` is a ``latest_verdicts`` scan plus a
    # file stat, and the frame loop may not do that per frame. Kept current by
    # arithmetic here and replaced wholesale when a training run reports back.
    status: dict[str, Any] = field(default_factory=dict)


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
    # Whether the arm each unit ran is hidden -- see the module docstring. A
    # session flag, never persisted, and it reorders as well as renames.
    blind: bool = False
    form: SweepForm = field(default_factory=SweepForm)
    # The open labelling pass, or None for "the verdict loop owns the keyboard".
    # None rather than a mode flag beside a always-present LabelPass, so there is
    # one answer to "which loop is A pressing" and it cannot be two.
    labels: LabelPass | None = None
    # The job ids a scoring run was asked about, so its answer lands on exactly
    # those rows. Not "the units that are open": the user can open another sweep
    # while the pass runs. Empty means nothing is in flight.
    score_request: list[str] = field(default_factory=list)


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
    """Ask for findings.json to be recomputed. Cheap, and safe to spam.

    Marks rather than submits, and the two halves are `pump_findings` below.
    The old version submitted straight away and treated a refusal as harmless
    on the grounds that "the next verdict's refresh picks up both" -- which is
    only true while a *verdict* is the sole trigger. It is not: the worker
    appends an observation on every finished model job, and the request that
    matters most is the one following the last unit of a sweep, with nothing
    after it to pick anything up.
    """
    ctx.state.findings_dirty = True


def pump_findings(ctx: Any) -> None:
    """Submit the pending recompute if there is one and nothing is in flight.

    Called every frame from the app's refresh step. The flag is cleared only
    when the submit is *accepted*, so a request arriving while a recompute runs
    survives to the next frame rather than being dropped -- and because the
    recompute reads the DB when it starts, one pass absorbs however many
    requests piled up behind it.
    """
    from ..service import findings as findings_mod

    if not ctx.state.findings_dirty:
        return
    if ctx.submit(FINDINGS_KEY, findings_mod.refresh, ctx.svc):
        ctx.state.findings_dirty = False


def request_scores(ctx: Any) -> None:
    """Ask for the open sweep's units to be scored. Cheap, and safe to spam."""
    ctx.state.review_scores_dirty = True


def unscored(units: list[dict[str, Any]]) -> list[str]:
    """The job ids with no score yet, in presentation order."""
    return [u["job_id"] for u in units if "score" not in u]


def pump_scores(ctx: Any) -> None:
    """Score whatever is open, if anything is missing one and nothing is running.

    ``pump_findings``' shape, for ``pump_findings``' reason: ``submit`` refuses
    a key already in flight and nothing re-arms it, so a direct submit drops the
    request that arrives while a run is going -- and the request that matters
    most is the last one.

    Scoring is a DINOv2 forward pass per row, so it is a task and never a frame.
    A ``blind`` session asks for nothing at all: see ``open_sweep``.
    """
    from ..service import judge as judge_mod

    if not ctx.state.review_scores_dirty:
        return
    state = ctx.state.review
    wanted = [] if state is None or state.blind else unscored(state.units)
    if not wanted:
        ctx.state.review_scores_dirty = False
        return
    if ctx.submit(SCORE_KEY, judge_mod.score_jobs, ctx.svc, wanted, SCORE_STAGE):
        # What was *asked*, so the answer lands on those rows and no others. The
        # user can open another sweep while the pass runs, and marking whatever
        # happens to be open when it returns would tick off units nobody scored
        # -- which then never get asked about again, since ``unscored`` reads the
        # same key.
        state.score_request = list(wanted)
        ctx.state.review_scores_dirty = False


def adopt_scores(state: ReviewState, scores: dict[str, Any]) -> None:
    """Merge scores onto the rows that were asked about, **reordering nothing**.

    The unit dicts are shared with the ``sweeps`` entries, so a row is found and
    written once wherever it lives -- which is what makes an answer arriving
    after the user has moved on still count. The order is deliberately left
    alone: a list that resorts under the cursor is how the wrong thing gets
    judged, which is the lesson ``LabelPass`` rows keeping their place already
    carries. Order is applied once, when a sweep is opened.

    An empty dict is "no probe", not "no scores" -- and every asked row is still
    written, with None, or the pump requests them again on every frame forever.
    """
    asked, state.score_request = state.score_request, []
    if not asked:
        return
    index = {
        unit["job_id"]: unit
        for entry in state.sweeps
        for unit in entry.get("units") or ()
    }
    index.update({unit["job_id"]: unit for unit in state.units})
    for job_id in asked:
        unit = index.get(job_id)
        if unit is not None:
            unit["score"] = scores.get(job_id)


def clear_scores(state: ReviewState) -> None:
    """Forget every score. For a retrain: the probe that produced them is gone,
    and a stale number is the ``warlockc`` hazard -- an absent one is obvious,
    one silently computing last week's answer is not."""
    for entry in state.sweeps:
        for unit in entry.get("units") or ():
            unit.pop("score", None)
    for unit in state.units:
        unit.pop("score", None)
    # The run in flight was scored by the probe that has just been replaced, so
    # its answer must not be adopted when it lands.
    state.score_request = []


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``review-`` key."""
    state = ensure(ctx)
    if done.key == DELETE_KEY:
        removed = remaining = kept = 0
        if isinstance(done.result, dict):
            removed = int(done.result.get("deleted") or 0)
            remaining = int(done.result.get("remaining") or 0)
            kept = int(done.result.get("kept") or 0)
        if kept and not remaining:
            # Said apart from the transient case, and without "delete again":
            # pressing again will never remove these, so an invitation to
            # retry would be a lie that renews itself every press. What the
            # user can still do is named, because the per-asset delete is the
            # deliberate escape hatch this guard leaves open.
            ctx.toast(
                f"Deleted {removed} job(s); kept {kept} you reviewed. "
                "Delete those from the library if you want them gone."
            )
        elif remaining:
            # A unit the worker is still inside is cancelled but not deleted --
            # its directory is being written to. Say so and say what to do,
            # rather than reporting a deletion that did not happen.
            # Plain info: the toast vocabulary is info|error (widgets.py reads
            # nothing else), and a partial delete is neither a failure nor a
            # surprise -- it is what cancelling something mid-run looks like.
            ctx.toast(
                f"Deleted {removed} job(s); {remaining} still finishing. "
                "Delete again in a moment."
            )
        else:
            ctx.toast(f"Deleted {removed} job(s). Verdicts and findings kept.")
        # The counts are now wrong and the viewer may be showing a mesh that no
        # longer exists -- both are fixed by the rescan.
        if state.sweep_id is not None and state.sweep_id != RECENT_ID:
            state.sweep_id = None
        scan(ctx)
        return
    if done.key == FINDINGS_KEY:
        return
    if done.key == SCORE_KEY:
        adopt_scores(state, done.result if isinstance(done.result, dict) else {})
        return
    if done.key == LABELS_KEY:
        if state.labels is not None:
            state.labels.loading = False
            if isinstance(done.result, dict):
                state.labels.rows = list(done.result.get("rows") or ())
                state.labels.status = dict(done.result.get("status") or {})
                state.labels.index = 0
                state.labels.uploaded = 0
        return
    if done.key == TRAIN_KEY:
        summary = done.result if isinstance(done.result, dict) else {}
        if state.labels is not None and summary.get("stage") == state.labels.stage:
            # The authoritative counts, from the run that just read them.
            state.labels.status.update(
                {
                    "trained": bool(summary.get("trained")),
                    "trained_labels": summary.get("usable", 0),
                    "labels": summary.get("labels", 0),
                }
            )
        if summary.get("trained"):
            ctx.toast(
                f"Trained the {summary.get('stage')} probe on "
                f"{summary.get('usable')} label(s)."
            )
            if summary.get("stage") == SCORE_STAGE:
                # Every score on screen was produced by the probe this just
                # replaced. Dropping them and asking again is the only honest
                # option; keeping them would show last week's answer under a
                # judge that has changed its mind.
                clear_scores(state)
                request_scores(ctx)
        # Silent otherwise: "12 more labels to go" is the ordinary state of a
        # corpus being built, and a toast per keypress saying so is noise. The
        # panel carries the count.
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
    if done.key == LABELS_KEY and state.labels is not None:
        # ``loading`` gates the grid's empty state; leaving it set is what makes
        # a failed listing look like a pass that is still starting, forever.
        state.labels.loading = False
    if done.key == TRAIN_KEY:
        ctx.toast("Could not train the probe.", "error")
    if done.key == SCORE_KEY:
        # Silent, and the rows are left *unmarked* rather than marked as scored
        # with nothing. The judge is advisory, so a failure to have an opinion is
        # not a failure of the review and a toast would make it look like one --
        # but writing None would tell ``unscored`` these rows had been answered,
        # and nothing would ever ask again. Unmarked, the next trigger (a
        # retrain, reopening the sweep) picks them up; and since nothing re-arms
        # the flag by itself, this cannot become a retry loop either.
        state.score_request = []


# --- the open sweep ----------------------------------------------------------


def blind_order(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The units in an order that says nothing about how they were queued.

    Sorted by a digest of the job id: independent of enqueue order (which names
    the arm -- ``expand`` puts the baseline first), and the *same* order in
    every process. ``hash()`` is salted per interpreter run, so a session
    resumed tomorrow would present the same units in a different order under
    different names, which is not a blind review of anything.
    """
    return sorted(units, key=lambda u: hashlib.sha1(str(u["job_id"]).encode()).digest())


def set_blind(ctx: Any, blind: bool) -> None:
    """Turn blinding on or off, and re-present whatever is open under it."""
    state = ensure(ctx)
    if state.blind == bool(blind):
        return
    state.blind = bool(blind)
    if state.sweep_id is not None:
        open_sweep(ctx, state.sweep_id)


def open_sweep(ctx: Any, sweep_id: str) -> None:
    """Show a sweep, starting on the first unit with no verdict.

    Resuming is the common case: landing back on unit one every time is what
    makes it useless.
    """
    state = ensure(ctx)
    entry = next((s for s in state.sweeps if s["id"] == sweep_id), None)
    state.sweep_id = sweep_id
    units = list(entry["units"]) if entry is not None else []
    # Presentation order only: the ``sweeps`` entry keeps its own list, which
    # is what ``_recount`` tallies, and the dicts are shared either way.
    # Blind wins wholesale, and hides the score as well as choosing the order.
    # A score-derived order is a quality channel that re-identifies the arms a
    # blind review exists to hide, and a judge's opinion on screen anchors the
    # independent human judgement it exists to collect.
    state.units = blind_order(units) if state.blind else by_score(units)
    state.pending_reject = False
    state.index = next(
        (i for i, unit in enumerate(state.units) if unit["verdict"] is None), 0
    )
    request_scores(ctx)


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


# --- the labelling pass ------------------------------------------------------


def _label_rows(svc: Any, stage: str) -> dict[str, Any]:
    """The images this question has not reached, plus the probe's state.

    Both in one task and one result, because both are DB reads and the pane needs
    them together -- and because ``judge.status`` on the frame thread would be a
    ``latest_verdicts`` scan and a file stat *per frame*, which is precisely the
    class of work the frame loop may not do. The counts are kept current
    afterwards by arithmetic on this snapshot rather than by re-reading.
    """
    from ..service import judge as judge_mod
    from ..service import verdicts as verdicts_mod

    out: list[dict[str, Any]] = []
    for job in svc.store.unlabelled_references(stage=stage, source=SOURCE):
        job_dir = svc.job_dir(job["id"])
        image = next(
            (job_dir / name for name in verdicts_mod.IMAGE_NAMES if (job_dir / name).exists()),
            None,
        )
        if image is None:
            # A refused job whose picture never landed, or a pruned directory.
            # There is nothing to look at, so there is nothing to label.
            continue
        out.append(
            {
                "job_id": job["id"],
                "prompt": job.get("name") or job.get("prompt") or job["id"],
                "status": job.get("status"),
                "image": image,
                "verdict": None,
            }
        )
    return {"rows": out, "status": judge_mod.status(svc, stage, source=SOURCE)}


def open_labels(ctx: Any, stage: str) -> None:
    """Begin a labelling pass over one question.

    The listing walks a directory per row, so it goes on a task thread -- but the
    pass itself is created immediately, because a mode that shows nothing until a
    task returns reads as broken.
    """
    state = ensure(ctx)
    state.labels = LabelPass(stage=stage, loading=True)
    if not ctx.submit(LABELS_KEY, _label_rows, ctx.svc, stage):
        state.labels.loading = False


def close_labels(ctx: Any) -> None:
    """End the pass and give the keyboard back to the verdict loop."""
    state = ensure(ctx)
    state.labels = None


def current_label(state: ReviewState) -> dict[str, Any] | None:
    labels = state.labels
    if labels is None or not (0 <= labels.index < len(labels.rows)):
        return None
    return labels.rows[labels.index]


def record_label(ctx: Any, verdict: str) -> bool:
    """Record one image label and step to the next unanswered row.

    Named ``record_label`` rather than ``label`` because ``label(state, unit)``
    below is the *display* name of a unit -- one module cannot hold both, and the
    collision is silent: the later definition simply wins, and a keypress then
    calls a formatter.

    Inline on the frame thread, exactly as ``record`` is and for the same reason:
    one INSERT under the store's lock, and a keypress whose effect arrives some
    frames later reorders labels against navigation at the rate these are pressed.
    """
    from ..service.errors import ServiceError

    state = ensure(ctx)
    row = current_label(state)
    if row is None or state.labels is None:
        return False
    try:
        verdicts_mod.record_verdict(
            ctx.svc, row["job_id"], verdict=verdict, source=SOURCE,
            stage=state.labels.stage,
        )
    except (ServiceError, OSError):
        log.exception("could not label %s", row["job_id"])
        ctx.toast("Could not record that label.", "error")
        return False
    row["verdict"] = verdict
    # The snapshot, kept current by arithmetic rather than by a re-read: this runs
    # on the frame thread and ``status`` is a whole-table scan.
    key = "positives" if verdict == "accept" else "negatives"
    status = state.labels.status
    status[key] = int(status.get(key, 0)) + 1
    status["labels"] = int(status.get("labels", 0)) + 1
    advance_labels(state.labels)
    # A flag, never a submit. ``TaskRunner.submit`` refuses a key already in
    # flight and nothing re-arms it, so a burst of labels trained once on the set
    # as it stood at the first press and silently dropped the rest -- the
    # ``findings_dirty`` bug, in a loop designed to be pressed even faster.
    ctx.state.judge_dirty = state.labels.stage
    return True


def advance_labels(labels: LabelPass) -> None:
    """Forward to the next row with no answer, wrapping, then staying put."""
    order = list(range(labels.index + 1, len(labels.rows))) + list(range(labels.index))
    ahead = next((i for i in order if labels.rows[i]["verdict"] is None), None)
    if ahead is not None:
        labels.index = ahead
    else:
        labels.index = min(labels.index + 1, max(len(labels.rows) - 1, 0))


def pump_judge(ctx: Any) -> None:
    """Submit a pending retrain if there is one and nothing is in flight.

    The ``pump_findings`` shape, for the same reason: the flag is cleared only
    when the submit is *accepted*, so a request that arrives while a training run
    is going survives to the next frame instead of being dropped -- and because
    training reads the labels when it starts, one pass absorbs however many
    presses piled up behind it.
    """
    from ..service import judge as judge_mod

    stage = ctx.state.judge_dirty
    if not stage:
        return
    if ctx.submit(TRAIN_KEY, judge_mod.train, ctx.svc, stage):
        ctx.state.judge_dirty = None


def next_thumbnail(labels: LabelPass) -> dict[str, Any] | None:
    """The next row whose thumbnail may be uploaded this frame, or None.

    **One per frame**, which is ``viewer/sheet.StripRender``'s lesson at a larger
    scale: a draw plus a synchronous upload, sixteen times in one frame, is a
    visible freeze, and this grid is a hundred cells. The already-uploaded ones
    keep drawing from the cache; the rest simply appear over the next second.
    """
    if labels.uploaded >= len(labels.rows):
        return None
    row = labels.rows[labels.uploaded]
    labels.uploaded += 1
    return row


def by_score(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same units, best-scoring first. **Sorted, never filtered.**

    The filter-bubble guard, and it is structural rather than a promise: if a
    judge hid what it disliked, its mistakes would become invisible and nobody
    would ever learn it was wrong -- the failure factories manage by auditing
    *passed* parts on a schedule. Sorting shows the same set in a more useful
    order and costs nothing if the judge is wrong. Unscored rows sort last rather
    than as 0.0, because "no opinion" is not "bad".
    """
    return sorted(
        units,
        key=lambda u: (u.get("score") is None, -(u.get("score") or 0.0)),
    )


def cache_id_for_label(row: dict[str, Any]) -> str:
    return f"label:{row['job_id']}"


# --- launching a sweep -------------------------------------------------------


def capture_base(ctx: Any) -> dict[str, Any]:
    """The settings vector the two generate forms currently describe.

    Reusing the forms the user has already tuned is the whole point: a sweep is
    "this, but vary that", and re-picking a checkpoint and twelve taxonomy
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
        # -1 is "cannot be planned yet", which a half-filled form legitimately
        # is -- but a *bug* in expand reads the same on screen, so it is logged
        # at debug rather than dropped: this runs every frame the form is open,
        # and an exception level would fill the log with the ordinary case.
        log.debug("could not plan the sweep preview", exc_info=True)
        return -1


# --- what the pane reads -----------------------------------------------------


def model_path(unit: dict[str, Any]) -> Path:
    return Path(unit["dir"]) / "model.glb"


def reference_path(unit: dict[str, Any]) -> Path | None:
    """What the unit was generated from, or None if neither exists."""
    for name in verdicts_mod.IMAGE_NAMES:
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
        # ``verdict`` was read here and nothing has ever written it -- the
        # worker stores worst/mean/faces/resolution, so this branch was dead
        # from the day it was typed, the same way ``report["verdict"]`` was in
        # ``widgets.quality_badge``. Replaced with the reading that exists,
        # worded so it cannot be read as a quality score (P120): a low figure
        # is what a solid slab measures, and AUC(worst -> reject) over the
        # reviewed corpus is 0.115, which is backwards rather than weak.
        worst = audit.get("worst")
        if isinstance(worst, (int, float)):
            lines.append(f"see-through at worst view: {float(worst) * 100:.1f}%")
    return lines


def label(state: ReviewState, unit: dict[str, Any]) -> str:
    """One line naming this unit -- its sweep label, or whatever an ordinary
    asset is called. Under blinding, an id prefix instead.

    It takes the state rather than a keyword flag on purpose: a call site that
    forgot to pass the flag would draw an unblinded label inside a blind review
    and nothing would say so, whereas a missing positional argument is a
    TypeError the first frame Review is drawn.
    """
    if state.blind:
        return f"#{str(unit.get('job_id') or '')[:6]}"
    return str(unit.get("label") or unit.get("job_id") or "")


def score_line(state: ReviewState, unit: dict[str, Any]) -> str:
    """What the judge thinks, or "" for nothing to say.

    It names its question. A bare percentage beside a mesh will be read as an
    opinion about the mesh, and this probe has never seen one -- it was fitted
    to reference images labelled "would this reconstruct", which is the only
    question the corpus has evidence for.

    Takes the state positionally for ``label``'s reason, and answers "" under
    blinding for ``open_sweep``'s: an AI opinion on screen anchors the
    independent human judgement a blind review exists to collect.

    Basic-Latin only (imgui's default atlas), so ``·`` and no arrows.
    """
    if state.blind:
        return ""
    score = unit.get("score")
    if not isinstance(score, (int, float)):
        return ""
    return f"judge: {round(float(score) * 100)}% - {SCORE_QUESTION}"


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
    if state.labels is not None:
        # The labelling pass owns the keyboard while it is open, and it is a
        # *different* loop: two keys, no reason step, and its own cursor. Handled
        # before anything below so a label can never be mistaken for a mesh
        # verdict -- which would file an accept about a mesh from a keypress
        # about a picture.
        return _label_key(ctx, state, event, name)
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


def _label_key(ctx: Any, state: ReviewState, event: Any, name: str) -> bool:
    """The labelling pass's own keys. -> whether the key was consumed.

    Two answers and no reason step: reasons are a mesh-stage concept, and five
    classes is far more than a first corpus can support. What a blank probe
    learns from a label is one bit, so one bit is what the keyboard offers.
    """
    import pygame

    labels = state.labels
    if labels is None:
        return False
    if event.key == pygame.K_ESCAPE:
        close_labels(ctx)
        return True
    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        if labels.rows:
            delta = -1 if event.key == pygame.K_LEFT else 1
            labels.index = min(max(labels.index + delta, 0), len(labels.rows) - 1)
        return True
    if name == "a":
        return record_label(ctx, "accept")
    if name == "r":
        return record_label(ctx, "reject")
    if name == "s":
        advance_labels(labels)
        return True
    # Everything else is swallowed, not passed down: the verdict loop's keys act
    # on a mesh nobody is looking at while this grid is on screen.
    return False

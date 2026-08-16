"""Review mode's controller: launching sweeps, and the verdict loop over them.

The ``clay_mode.py`` pattern -- state and logic here, drawing in ``main.py``,
no imgui anywhere under this import -- so the part that is easy to get wrong is
assertable without a GL context.

What it is for. A sweep queues a family of settings vectors as ordinary jobs;
this mode is where they are looked at, one at a time, with the reference image
beside the mesh and a keypress to say how well it worked. Those verdicts compile
into ``findings.json``, which is what puts a "usable 6/8" under a control in
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

**A mesh verdict is a grade, and the sign is armed rather than typed.** ``1``-
``5`` file +1..+5, ``R`` arms the negative so the next digit files -1..-5, and
``0`` is its own key. Eleven values inside six keys, which is what keeps a pass
at the pace the binary loop had.

The old rule this replaces said that reject waits for a reason while "accept has
no such second step, because there is only one way for a mesh to be right".
The second half was the mistake, and the corpus showed it: 3 accepts against 81
rejects, in which a slab, a smeared texture and a mesh a modeller would fix in
five minutes were the same row. There are eleven ways to be almost right, the
grade now carries what a bare reject could not, and **tags are optional at every
grade** -- Ctrl+1-5 for the good vocabulary, Shift+1-5 for the bad, positional
in both, staged until a grade is pressed and dropped by anything that moves.
They stopped being *reasons a reviewer rejected*, so a ``clean-shape`` on a -4 is
exactly the pair of facts worth having. ``A`` is gone from this loop and is
deliberately *not* remapped onto +3: silently filing the mildest usable grade for
somebody reaching for the old key is a wrong number rather than a missing one.
The labelling pass keeps its own A/R, because an image label is still one bit.

**The guided pass is the second holder of that licence, and it is binary on
purpose.** ``JudgingPass`` walks every bucket with work left, in one run, with
Accept/Reject filing ``vectors.BINARY_GRADES`` (+-3, which is what migration 10
backfilled a binary row to and what a reviewer with no stronger key was
asserting). It does not replace the scale -- the grade row and the digits keep
working inside a pass, and a grade pressed there files and advances the pass
like anything else. It exists for the case the scale is bad at: twenty units, no
strong opinions, and a reviewer who will not start at all if the price of the
first unit is choosing between -2 and -3. Nothing about it is silent: it is
entered from a button, its panel labels its own keys, and ``a`` stays unbound
outside it. Two shifts inside a pass are worth stating because they are not the
outer loop's: ``R`` is *reject* rather than "arm the negative sign" (a binary
loop has no magnitude to arm; negative grades stay on the digits and the pane's
buttons), and ``Esc`` ends the pass rather than disarming.

**A sweep every unit of which has been judged has its assets removed, with a
toast and no dialog.** That is a deliberate, user-chosen override of
``jobs.retained_job_ids`` -- see ``sweeps.cleanup_sweep``, which carries the
argument -- and the warning is given up front on the entry card rather than as a
modal after the fact. It fires on ``todo`` reaching zero whether or not a pass
was running, because the requirement is about the sweep rather than about how it
was judged, and it refuses ``RECENT_ID``, whose units are ordinary library rows.

**The advance is to the next thing to do, not the next row.** A session is
resumed far more often than it is started, so opening a sweep lands on its
first unverdicted unit and recording steps past everything already answered.

**Blinding hides the arm, and that means the order too.** The review that
produced the ``bg_removal`` signal was unblinded and single-reviewer, which is
why the campaign's rule is a small blind confirm before anything is leaned
on -- the pre-registration in ``docs/measurements/2026-08-09-rebaseline.md``,
whose confirm is the one that ran and failed against its own rule.
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
CLEANUP_KEY = "review-cleanup"
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
# ``docs/measurements/`` document first --
# ``docs/measurements/2026-08-09-judge-threshold.md`` is that document, and it
# says exactly what the run must produce before a cut may be adopted.
# The ``(job_id, source, stage)`` seam is already built and tested, so
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

# The grade keyboard. 1-5 are the magnitudes; ``R`` arms the negative sign and
# 1-5 then read as -1..-5; ``0`` is its own key, because zero has no sign to
# arm and is a real answer rather than a refusal to give one.
#
# The sign-then-digit shape is what keeps eleven values inside six keys, and it
# is deliberately the *same* ``R`` the binary loop used to arm a reject with:
# the muscle memory is "R then say more", and what "more" means got richer.
GRADE_KEYS = {str(i): i for i in range(1, verdicts_mod.GRADE_MAX + 1)}

# Digit -> tag, positional, which is the whole reason each vocabulary is five
# long. Modifiers pick the polarity: Ctrl for good, Shift for bad. Derived from
# the vocabularies rather than written out, so a reordered tuple cannot leave
# the keyboard naming the old order -- the ``palette._mode_commands`` rule.
GOOD_TAG_KEYS = {str(i + 1): tag for i, tag in enumerate(verdicts_mod.GOOD_TAGS)}
BAD_TAG_KEYS = {str(i + 1): tag for i, tag in enumerate(verdicts_mod.BAD_TAGS)}

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
class JudgingPass:
    """One guided run through everything left to judge.

    A *pass* rather than a mode flag, for ``LabelPass``' reason: while one is
    open the keyboard means something different, and "which loop is A pressing"
    must have exactly one answer rather than two flags that can disagree.

    It is **binary on purpose** and that is the whole of what it adds. The
    eleven-point scale is not going anywhere -- it is still in the verdict pane,
    still on the digits, and a grade pressed during a pass files and advances
    like everything else. What the pass is for is the case the scale is bad at:
    twenty units, no strong opinions, and a reviewer who will not start at all
    if the price of the first unit is choosing between -2 and -3. Accept/Reject
    files +3/-3 through ``vectors.BINARY_GRADES``, which is exactly what
    migration 10 backfilled a binary row to and exactly what a reviewer with no
    stronger key was asserting.

    ``total`` is the count at the moment the pass started, so "3 of 20" counts
    up to a number that does not move under the reader. ``filed`` counts what
    this pass filed rather than what is left, so a unit re-graded twice does not
    make the header count down.
    """

    total: int = 0
    filed: int = 0
    # Bucket ids with work to do when the pass started, RECENT_ID first and then
    # sweeps in list order. Fixed at the start rather than recomputed: a bucket
    # that gains a unit mid-pass (a job finishing) would otherwise extend a pass
    # the user was told the length of.
    order: list[str] = field(default_factory=list)


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
    # Whether R has been pressed, so the next digit reads as a negative grade.
    # Cleared by anything that moves, because the armed state belongs to the
    # unit that was on screen when it was armed.
    pending_negative: bool = False
    # Tags toggled for the unit on screen, ready to ride along with whatever
    # grade is pressed next. Cleared by exactly the same four things, and for
    # exactly the same reason: a tag chosen while looking at one mesh must never
    # be filed against the next one.
    pending_tags: list[str] = field(default_factory=list)
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
    # The guided pass, or None for "the ordinary loop owns the keyboard".
    # Session-only like everything else here, and for the same reason: a stored
    # pass would resume against buckets that have since been judged or deleted.
    judging: JudgingPass | None = None
    # What the pass that just ended did, as a plain dict ready to draw. Built
    # once at the end from what is in memory -- never a DB read on the frame
    # thread -- and cleared by the report card's Dismiss.
    judging_report: dict[str, Any] | None = None


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
        # The grade the row carries, or None. ``verdict`` stays beside it and
        # stays the field ``advance``/``_recount``/``open_sweep`` key on: their
        # question is "has this been answered", which a grade of 0 answers yes
        # and a falsy check would answer no.
        "grade": seen.get("grade"),
        "tags": list(seen.get("reasons") or ()),
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
                # What the sweep actually varied. ``list_sweeps`` has always
                # parsed it and this bucket has always dropped it, which left
                # the list identifying a run by whatever the user typed as its
                # name -- routinely "test2", days after the fact. See
                # ``spec_summary``.
                "spec": sweep.get("spec") or {},
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


def _removal_toast(ctx: Any, done: Any) -> None:
    """What a delete or a cleanup says afterwards. -> nothing.

    One block for both keys, because the *result* is the same shape and the
    three cases mean the same three things whichever button produced them. Only
    the verb differs, and it comes off the key rather than from a second copy of
    this branch -- which is exactly how the "delete again in a moment" sentence
    would have come to be missing from one of the two.
    """
    cleaning = done.key == CLEANUP_KEY
    # Named, when the caller said which sweep. A cleanup fires without anybody
    # pressing anything for it, so "3 asset folders removed" on its own is a
    # sentence about nothing the reader can place.
    named = str(getattr(done, "tag", None) or "")
    verb = f"Cleaned up {named}:" if cleaning and named else (
        "Cleaned up" if cleaning else "Deleted"
    )
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
            f"{verb} {removed} asset folder(s); kept {kept} you reviewed. "
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
            f"{verb} {removed} asset folder(s); {remaining} still finishing. "
            "Delete again in a moment."
        )
    else:
        ctx.toast(
            f"{verb} {removed} asset folder(s). Verdicts and findings kept."
        )


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``review-`` key."""
    state = ensure(ctx)
    if done.key in (DELETE_KEY, CLEANUP_KEY):
        _removal_toast(ctx, done)
        # The counts are now wrong and the viewer may be showing a mesh that no
        # longer exists -- both are fixed by the rescan.
        #
        # The selection is dropped only when **no pass is running**. A manual
        # delete removes the sweep you were looking at, so forgetting it is
        # right; a pass's cleanup fires for a sweep the pass has already walked
        # *past*, and clearing the id there sent the rescan's fallback to the
        # first bucket -- which is how a pass ended up parked on an emptied
        # recent bucket with two units still outstanding elsewhere.
        if state.judging is None and state.sweep_id not in (None, RECENT_ID):
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
    # And let a running pass re-decide where it is. A scan is the one thing
    # that can empty the open bucket without a keypress -- a cleanup's rescan
    # does exactly that -- and the pass's other two pumps both need a unit on
    # screen, so without this it has no way out of a bucket with none.
    _pump_judging(ctx, state)


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
    if done.key == CLEANUP_KEY:
        # Toasted rather than swallowed, even though nobody pressed a button for
        # it: the entry card promised the assets would go, and silence would
        # leave the user believing a promise that had failed.
        ctx.toast("Could not clean up that sweep's assets.", "error")
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
    _disarm(state)
    state.index = next(
        (i for i, unit in enumerate(state.units) if unit["verdict"] is None), 0
    )
    request_scores(ctx)


def current(state: ReviewState) -> dict[str, Any] | None:
    if 0 <= state.index < len(state.units):
        return state.units[state.index]
    return None


def step(state: ReviewState, delta: int) -> None:
    """Move by hand, clamped at both ends. Disarms and drops any toggled tags,
    because both belong to the unit that was on screen when they were set."""
    if not state.units:
        return
    state.index = min(max(state.index + delta, 0), len(state.units) - 1)
    _disarm(state)


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
    _disarm(state)
    if unverdicted_only:
        order = list(range(state.index + 1, len(state.units))) + list(range(state.index))
        ahead = next((i for i in order if state.units[i]["verdict"] is None), None)
        if ahead is not None:
            state.index = ahead
        return
    state.index = min(state.index + 1, len(state.units) - 1)


def _disarm(state: ReviewState) -> None:
    """Drop the armed sign and the toggled tags together.

    One function rather than two assignments at four call sites: they are one
    piece of state about one unit, and the way this goes wrong is that a fifth
    site clears the sign and forgets the tags, which then ride onto the next
    mesh silently.
    """
    state.pending_negative = False
    state.pending_tags = []


def toggled(tags: Any, tag: str) -> list[str]:
    """``tags`` with ``tag`` added or removed. Pure, and the one owner of the
    rule -- the inspector stages tags against a job id rather than against a
    review unit, so it holds a different list and calls this with it.

    A toggle rather than a set, because the same key has to be able to undo an
    accidental press. An unknown tag is ignored rather than staged: it would
    make ``record_verdict`` refuse the whole verdict, which loses a grade to a
    typo in a key map.
    """
    staged = list(tags)
    if tag not in verdicts_mod.TAGS:
        return staged
    if tag in staged:
        staged.remove(tag)
    else:
        staged.append(tag)
    return staged


def toggle_tag(state: ReviewState, tag: str) -> None:
    """Add or remove a tag from what the next grade will carry."""
    state.pending_tags = toggled(state.pending_tags, tag)


def grade_text(grade: Any) -> str:
    """A grade as it is written everywhere in the UI -- ``"+4"``, ``"-2"``,
    ``"0"`` -- or ``""`` for a unit with no grade.

    Signed for everything but zero: ``+0`` reads as a magnitude that lost its
    meaning, while a bare ``4`` beside a ``-2`` reads as a different scale.
    ``bool`` is excluded because it is an ``int`` and ``True`` would render +1.
    """
    if not isinstance(grade, int) or isinstance(grade, bool):
        return ""
    return "0" if grade == 0 else f"{grade:+d}"


def record(ctx: Any, grade: int, tags: Any = ()) -> None:
    """Write one graded verdict for the unit on screen, then move on.

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
    tags = list(tags)
    try:
        result = verdicts_mod.record_verdict(
            ctx.svc, unit["job_id"], grade=grade, reasons=tags, source=SOURCE
        )
    except (ServiceError, OSError):
        log.exception("could not record a verdict for %s", unit["job_id"])
        ctx.toast("Could not record that verdict.", "error")
        return
    unit["grade"] = grade
    # Read back off the service rather than re-derived here: the cut has one
    # owner, and a second application of it in the UI is how the list comes to
    # disagree with the row it is drawing.
    unit["verdict"] = result["verdict"]
    unit["tags"] = tags
    _recount(state)
    refresh_findings(ctx)
    # Said *before* the advance, and about the unit that is about to leave the
    # screen. The sidebar's "Recorded:" line describes whatever is on screen
    # now, so after the jump it is answering about the next mesh -- which reads
    # like the verdict landed on the wrong one. It also names the way back,
    # because there is no undo here and the route (Left, then grade again,
    # which overwrites) is not guessable.
    filed = grade_text(grade) or str(grade)
    if tags:
        filed += " - " + ", ".join(tags)
    ctx.toast(f"Filed {filed} for {label(state, unit)}. Left arrow to re-grade it.")
    if state.judging is not None:
        state.judging.filed += 1
    advance(state, unverdicted_only=True)
    # After the advance, so a pass that has just emptied its last bucket ends
    # with the cursor already parked rather than being moved by a report.
    # Unconditional rather than gated on ``judging``: a sweep hand-graded to
    # zero outside a pass has been judged just as thoroughly, and the
    # requirement is about the sweep rather than about how it was reached.
    if state.judging is not None:
        _pump_judging(ctx, state)
    elif state.sweep_id is not None and _todo_of(state, state.sweep_id) == 0:
        auto_cleanup(ctx, state.sweep_id)


def _recount(state: ReviewState) -> None:
    """Keep the sweep list's remaining count true. The unit dicts are shared
    with the ``sweeps`` entry, so only the tally needs recomputing."""
    for sweep in state.sweeps:
        if sweep["id"] == state.sweep_id:
            sweep["todo"] = sum(1 for unit in sweep["units"] if unit["verdict"] is None)


# --- the guided judging pass -------------------------------------------------


def todo_total(state: ReviewState) -> int:
    """How many units, across every bucket, have no verdict. Pure."""
    return sum(int(sweep.get("todo") or 0) for sweep in state.sweeps)


def start_judging(ctx: Any) -> bool:
    """Open a guided pass over everything left to judge. -> whether one opened.

    It opens the first bucket with work rather than leaving the user where they
    were: the pass is a claim about *all* the outstanding units, and starting it
    in the middle of one sweep would make the "3 of 20" header describe a walk
    the user cannot see the shape of.
    """
    state = ensure(ctx)
    if state.scanning or state.judging is not None:
        return False
    order = [
        str(sweep["id"]) for sweep in state.sweeps if int(sweep.get("todo") or 0) > 0
    ]
    if not order:
        return False
    state.judging = JudgingPass(total=todo_total(state), order=order)
    state.judging_report = None
    _disarm(state)
    # ``open_sweep`` already lands on the first unverdicted unit and already
    # honours blinding, so there is nothing for the pass to re-decide here.
    open_sweep(ctx, order[0])
    return True


def judging_skip(ctx: Any) -> None:
    """Leave this unit unjudged and move on, staying inside the pass."""
    state = ensure(ctx)
    if state.judging is None:
        return
    advance(state, unverdicted_only=True)
    _pump_judging(ctx, state)


def _pump_judging(ctx: Any, state: ReviewState) -> None:
    """Move the pass on when the open bucket runs out.

    Called after every file, after every skip, **and after every scan lands** --
    and that third caller is not belt-and-braces. The first two both need a
    unit on screen to have been reached at all, so a pass parked on an empty
    bucket could not pump itself out of one: `record` returns early with no
    current unit, so nothing ever ran this again. A cleanup's own rescan lands
    on the recent bucket when it has emptied, which is exactly that state, and
    the pass stopped dead with work still outstanding in another sweep.

    The bucket-to-bucket step is what makes this a pass over the *session* and
    not over one sweep -- which is the whole difference between it and simply
    pressing digits, and the reason the entry card can promise a number.
    """
    pass_ = state.judging
    if pass_ is None:
        return
    open_id = state.sweep_id
    entry = next((s for s in state.sweeps if s["id"] == open_id), None)
    if entry is not None and entry.get("units") and _todo_of(state, open_id) == 0:
        # Judged out. The cleanup is fired from here rather than from
        # ``finish_judging`` so a sweep's assets go as soon as it is finished
        # with, rather than all of them at the end of a long pass.
        #
        # Gated on the bucket still *having* units, because this now runs after
        # a rescan too: a sweep whose cleanup already landed is a row that is
        # gone, and an emptied recent bucket is not a sweep at all -- neither is
        # something to submit a second removal for.
        auto_cleanup(ctx, open_id)
    remaining = [sid for sid in pass_.order if _todo_of(state, sid) > 0]
    if not remaining:
        finish_judging(ctx)
        return
    if open_id not in remaining:
        open_sweep(ctx, remaining[0])


def _todo_of(state: ReviewState, sweep_id: str) -> int:
    entry = next((s for s in state.sweeps if s["id"] == sweep_id), None)
    return int((entry or {}).get("todo") or 0)


def end_judging(ctx: Any) -> None:
    """Stop early -- Esc, or the Finish button. Still writes the report.

    A pass abandoned halfway is exactly the case a summary is most useful for:
    what it says is how much was done and how much is left, and hiding that
    because the user stopped would be punishing them for stopping.
    """
    finish_judging(ctx)


def finish_judging(ctx: Any) -> None:
    """Close the pass and build its report. Frame thread, from memory only.

    **Never a DB read.** ``_collect`` fetched every unit and ``record`` keeps
    the dicts current as verdicts are filed, so the numbers are already here --
    and this runs on the frame thread, behind the store's one serialized
    connection, at the moment a reviewer is expecting the pass to end.
    """
    state = ensure(ctx)
    pass_ = state.judging
    if pass_ is None:
        return
    state.judging = None
    per_sweep = []
    filed = accepted = rejected = remaining = 0
    for sweep in state.sweeps:
        if str(sweep["id"]) not in pass_.order:
            continue
        units = sweep.get("units") or []
        graded = [u for u in units if isinstance(u.get("grade"), int)]
        row = {
            "id": sweep["id"],
            # Blind sessions get blind labels here too. A report naming the arms
            # at the end of a blinded pass would un-blind every unit in it after
            # the fact, which is most of what blinding was protecting.
            "label": (
                f"#{str(sweep['id'])[:6]}"
                if state.blind and sweep["id"] != RECENT_ID
                else str(sweep.get("label") or sweep["id"])
            ),
            "total": len(units),
            "accepted": sum(1 for u in units if u.get("verdict") == "accept"),
            "rejected": sum(1 for u in units if u.get("verdict") == "reject"),
            "no_opinion": sum(1 for u in units if u.get("grade") == 0),
            "todo": int(sweep.get("todo") or 0),
            "mean_grade": (
                round(sum(int(u["grade"]) for u in graded) / len(graded), 1)
                if graded
                else None
            ),
        }
        per_sweep.append(row)
        accepted += row["accepted"]
        rejected += row["rejected"]
        remaining += row["todo"]
    filed = pass_.filed
    state.judging_report = {
        "sweeps": per_sweep,
        "filed": filed,
        "accepted": accepted,
        "rejected": rejected,
        "remaining": remaining,
    }


def dismiss_report(ctx: Any) -> None:
    state = ensure(ctx)
    state.judging_report = None


def auto_cleanup(ctx: Any, sweep_id: str) -> bool:
    """Remove a fully judged sweep's assets. -> whether it was submitted.

    Fired from ``record`` and from the pass, both on ``todo`` reaching zero:
    the requirement is "once a sweep has been judged", which a hand-graded sweep
    meets exactly as a pass-driven one does.

    ``RECENT_ID`` is refused, and not as a special case: the recent bucket is
    ordinary library rows that happen not to carry a verdict yet, so deleting
    them because they were judged would delete the user's own assets. Pruning
    the library is prune's job and has its own confirmation.

    No dialog, per the user's explicit choice -- the entry card carries the
    warning up front instead, and the toast says what went and what was kept.
    """
    from ..service import sweeps as sweeps_mod

    if sweep_id == RECENT_ID:
        return False
    state = ensure(ctx)
    entry = next((s for s in state.sweeps if s["id"] == sweep_id), None)
    # Captured now and carried on the tag, because by the time this lands the
    # rescan will have dropped the row this label came from.
    named = (
        f"#{str(sweep_id)[:6]}"
        if state.blind
        else str((entry or {}).get("label") or sweep_id)
    )
    return bool(
        ctx.submit(
            CLEANUP_KEY, sweeps_mod.cleanup_sweep, ctx.svc, sweep_id, tag=named
        )
    )


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
        # Framed rather than forwarded, the ``packwright_mode`` house rule: a
        # bare ``str(exc)`` toast is library text with no subject in front of
        # it, so the user reads "axis 'seed' has no values" and has to guess
        # what was being attempted. ``action="log"`` because the traceback is
        # the only place the rest of the story is.
        ctx.toast(f"That sweep could not be planned: {exc}", "error", action="log")
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


# One line per axis param, for the params whose *name* is the whole of what the
# form says about them. Only the ones that resolve to a free-text field need one
# -- an enumerated param draws its own options and a numeric one draws its range
# -- but the test asserts the complement, so a param that stops resolving is
# caught by having no help line rather than by somebody noticing the blank box.
AXIS_HELP: dict[str, str] = {
    "custom_triangles": (
        "Triangle budget, used only when profile is 'custom'. Ignored by 'raw'."
    ),
    "negative_prompt": (
        "What to steer away from. Only reaches models that take a real CFG "
        "scale; commas separate the values, so avoid them inside one prompt."
    ),
}


def _catalog_options(ctx: Any, field: str) -> list[dict[str, str]]:
    entries = (getattr(ctx, "guidance", None) or {}).get("fields") or {}
    return [
        {"key": str(o.get("key") or ""), "label": str(o.get("label") or o.get("key") or "")}
        for o in entries.get(field) or []
    ]


def axis_options(ctx: Any) -> list[dict[str, Any]]:
    """What each sweep axis param may be set to. -> one row per param.

    ``kind`` is what the form draws: ``options`` gets a checkbox per legal
    value, ``number`` a hint naming its range, ``bool`` an on/off pair, and
    ``text`` is the free field every row used to be. Whatever the kind, the row
    is written back as the same comma-separated string, so ``build_plan`` and
    ``_coerce`` are untouched -- this is a control over an existing field, not a
    second representation of it.

    Headless, and built from ``ctx.guidance`` rather than from the service: the
    catalog is fetched once at startup and this runs per frame the form is open.
    A param the catalog cannot resolve falls through to ``text``, which is
    exactly today's behaviour -- so a param added to ``axis_params`` without
    being described here is *less* discoverable, never broken.
    """
    from ..service import sweeps as sweeps_mod

    guidance = getattr(ctx, "guidance", None) or {}
    sweeps_extra = guidance.get("sweeps") or {}
    defaults = guidance.get("defaults") or {}
    rows: list[dict[str, Any]] = []
    for param in sweeps_mod.axis_params():
        row: dict[str, Any] = {
            "param": param,
            "kind": "text",
            "options": [],
            "range": None,
            "default": defaults.get(param),
            "help": AXIS_HELP.get(param, ""),
        }
        options = _catalog_options(ctx, param)
        if param == "bg_removal":
            options = [{"key": k, "label": k} for k in guidance.get("bg_removal") or []]
        elif param in ("resolution", "profile"):
            options = [
                {"key": str(v), "label": str(v)} for v in sweeps_extra.get(param) or []
            ]
        # Only when the list is non-empty, and that is not defensiveness: an
        # ``options`` row with nothing in it draws a label and no controls at
        # all, which is strictly worse than the free-text field it replaced.
        if options:
            row["kind"], row["options"] = "options", options
        elif param == "reference_prep":
            row["kind"] = "bool"
        elif (bounds := _axis_range(guidance, sweeps_extra, param)) is not None:
            row["kind"], row["range"] = "number", bounds
            if row["default"] is None:
                row["default"] = bounds[0]
        rows.append(row)
    return rows


def _axis_range(
    guidance: dict[str, Any], sweeps_extra: dict[str, Any], param: str
) -> tuple[Any, Any] | None:
    """``(low, high)`` for a numeric axis, or None.

    Two spellings because the catalog grew two: the guidance half named its
    ranges ``<param>_range`` beside ``size_range_m``, and the sweeps sub-dict
    follows the first of those. Looked up rather than tabulated so a range added
    to either half is offered here by having been added.
    """
    for source, key in (
        (guidance, f"{param}_range"),
        (sweeps_extra, f"{param}_range"),
        (guidance, "size_range_m" if param == "size_m" else ""),
    ):
        bounds = source.get(key) if key else None
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            return (bounds[0], bounds[1])
    return None


def spec_summary(spec: Any) -> str:
    """What a stored sweep varied, in one line. Pure.

    Drawn under a sweep in the list, because a sweep is judged days after it is
    launched and its *name* is whatever the user typed at the time -- routinely
    "test2". The spec is the only record of what the run was actually asking.

    Basic-Latin only (imgui's default atlas), so " - " and not an en dash.
    """
    if not isinstance(spec, dict):
        return ""
    parts = []
    for axis in spec.get("axes") or ():
        if not isinstance(axis, dict):
            continue
        param = str(axis.get("param") or "")
        values = ", ".join(str(v) for v in axis.get("values") or ())
        if param and values:
            parts.append(f"varies {param} ({values})")
    seeds = spec.get("seeds") or ()
    if seeds:
        parts.append(f"{len(seeds)} seed(s)")
    if spec.get("vectors"):
        parts.append(f"{len(spec['vectors'])} explicit vector(s)")
    return " - ".join(parts)


def preview_line(state: ReviewState, labels: dict[str, str] | None = None) -> str:
    """What the form would queue, said in words rather than as a bare count.

    "12 jobs" is a number the user has to reverse-engineer the form to check.
    Naming the multiplication -- baseline, then each axis by its values, then by
    the seeds -- is what makes a mis-typed axis visible *before* two hours of
    GPU rather than after.

    ``labels`` is a param -> human-name map the *caller* supplies, because label
    resolution is ``panes.settings_2d.field_label``'s and this module may not
    import a pane. Absent, the param names are used, which is what a headless
    caller wants anyway.

    Basic-Latin only, so " x " and " - " rather than the typographic forms.
    """
    planned = preview_units(state)
    if planned <= 0:
        return ""
    labels = labels or {}
    parts = ["baseline"]
    for row in state.form.axes:
        param = (row.get("param") or "").strip()
        count = len([v for v in (row.get("values") or "").split(",") if v.strip()])
        if param and count:
            parts.append(f"{labels.get(param, param)} x {count} value(s)")
    seeds = len(state.form.seeds.split(",")) if state.form.seeds.strip() else 0
    tail = f" x {seeds} seed(s)" if seeds > 1 else ""
    return (
        f"{planned} jobs: {' + '.join(parts)}{tail} - everything else from your "
        "captured settings."
    )


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
        # worded so it cannot be read as a quality score (P120).
        #
        # ``hole_worst`` is corpus-dependent and is never presented as a quality
        # scale: it measured AUC 0.115 against reject on the 2026-08-07 corpus
        # and 0.756 after the matte fix
        # (``docs/measurements/2026-08-09-rebaseline.md``). A low figure is
        # still what a solid slab measures, which is why the caveat below is
        # about the *direction of inference* rather than about either number.
        worst = audit.get("worst")
        if isinstance(worst, (int, float)):
            lines.append(f"see-through at worst view: {float(worst) * 100:.1f}%")
            # The same caveat the inspector carries on the same number, and it
            # matters more here: Review is where the corpus judgements are
            # filed, so an unqualified low reading beside a mesh is a figure a
            # reviewer will read as "no holes -- good" while grading. A high
            # reading is real evidence and needs no caveat; a low one is what a
            # solid, featureless slab measures. Imported inside the function
            # because ``widgets`` pulls in imgui and nothing else in this
            # module needs it.
            from .widgets import AUDIT_UNINFORMATIVE

            if float(worst) < AUDIT_UNINFORMATIVE:
                lines.append("(a solid, featureless mesh scores this too)")
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
    # Modifiers are read off ``event.mod`` rather than off the key name, because
    # ``pygame.key.name`` returns "1" whether or not Ctrl or Shift is held --
    # the digit is the same key, and only the modifier says which of three
    # things it means. There is no collision with a global binding: the
    # positional Alt+digit switch was deleted with ``mode_for_digit``, and
    # Ctrl+K is answered above the modes either way.
    #
    # Read *above* the Escape/arrow branches because the guided pass below needs
    # to know a bare key is bare, and those two do not care either way.
    ctrl = bool(event.mod & pygame.KMOD_CTRL)
    shift = bool(event.mod & pygame.KMOD_SHIFT)
    if state.judging is not None:
        # The guided pass owns A/R/S and Esc while it is open. This is the
        # ``LabelPass`` licence extended: the objection to reviving ``a`` was
        # that a *silent* remap files the mildest usable grade for somebody
        # reaching for a key that no longer exists. Here nothing is silent --
        # the pass is entered deliberately, its panel draws "Accept (A)" and
        # "Reject (R)" above the grade row, and the mapping to +-3 is what
        # ``vectors.BINARY_GRADES`` already means. Outside a pass ``a`` stays
        # unbound, which is what the fall-through at the bottom is still for.
        #
        # ``R`` is *reject* here rather than "arm the negative sign": a binary
        # loop has no magnitude to arm, and the negative grades stay reachable
        # through the digits and the pane's own buttons, which keep working.
        # Esc **ends the pass** rather than disarming, because leaving is the
        # thing a user in a modal-feeling flow reaches Esc for.
        if event.key == pygame.K_ESCAPE:
            end_judging(ctx)
            return True
        if not ctrl and not shift and current(state) is not None:
            if name == "a":
                record(ctx, verdicts_mod.BINARY_GRADES["accept"], state.pending_tags)
                return True
            if name == "r":
                record(ctx, verdicts_mod.BINARY_GRADES["reject"], state.pending_tags)
                return True
            if name == "s":
                judging_skip(ctx)
                return True
        # Everything else -- the digits, the tag modifiers, the arrows -- falls
        # through to the ordinary branches below, so a reviewer who wants to say
        # "-5, holes" mid-pass still can and the pass still counts it.
    if event.key == pygame.K_ESCAPE:
        _disarm(state)
        return True
    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        step(state, -1 if event.key == pygame.K_LEFT else 1)
        return True
    if current(state) is None:
        # Nothing on screen to judge: every key below acts on a unit.
        return False

    # Tags before grades, or Ctrl+1 would file a +1 and stage nothing.
    if ctrl and name in GOOD_TAG_KEYS:
        toggle_tag(state, GOOD_TAG_KEYS[name])
        return True
    if shift and name in BAD_TAG_KEYS:
        toggle_tag(state, BAD_TAG_KEYS[name])
        return True
    if not ctrl and not shift:
        if name == "0":
            # Its own key rather than a magnitude: zero has no sign to arm, and
            # "no opinion either way" is an answer rather than a refusal to give
            # one. Deliberately reachable with the sign armed too -- -0 is 0.
            record(ctx, 0, state.pending_tags)
            return True
        if name in GRADE_KEYS:
            magnitude = GRADE_KEYS[name]
            record(ctx, -magnitude if state.pending_negative else magnitude,
                   state.pending_tags)
            return True
        if name == "r":
            state.pending_negative = True
            return True
        if name == "s":
            advance(state)
            return True
    # ``a`` deliberately falls through unconsumed *outside a pass*. It was
    # Accept, and a mesh verdict is a grade now: silently mapping it onto +3
    # would file the mildest usable grade every time somebody reached for the
    # old key, which is a wrong number rather than a missing one. The label pass
    # keeps its own A/R, and the guided judging pass above is the second holder
    # of that same licence -- in both cases the key is bound because a loop the
    # user explicitly entered says on screen what it does.
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

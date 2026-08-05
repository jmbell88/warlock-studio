"""Review mode's controller: the verdict loop over a sweep's units.

The ``build_mode.py`` pattern -- state and logic here, drawing in ``main.py``,
no imgui anywhere under this import -- so the part that is easy to get wrong is
assertable without a GL context.

What it is for. ``bench/sweep.py`` varies one parameter at a time and leaves a
run directory full of meshes; ``bench/report.py`` turns verdicts on those meshes
into a findings table. Nothing sat between the two except a directory listing
and a mesh viewer the app already had, which is what this mode is: the units of
one sweep run, one at a time, with the reference image beside the mesh and two
keys to say whether it worked.

Three rules shape it.

**A verdict is filed under the param and value the *run* recorded**, read off
the item record rather than off the unit key or the axis spec.
``report.aggregate`` groups on ``(param, str(value))``, and a verdict filed
under anything else is not wrong so much as invisible -- it lands in a bucket
nothing joins to and the findings table simply never mentions it.

**Reject waits for a reason.** ``R`` arms; the verdict is not written until one
of the five reason keys is pressed. A bare rejection is a row the report can
count and nothing else, and the tally of *why* things fail is the one thing a
sweep exists to produce. Accept has no such second step, because there is only
one way for a mesh to be right.

**The advance is to the next thing to do, not the next row.** A session is
resumed far more often than it is started, so opening a run lands on its first
unverdicted unit and recording steps past everything already answered.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..bench import verdicts as verdicts_mod

log = logging.getLogger(__name__)

# The one task key. Prefixed, because the app claims results by prefix: a key
# without one is a result delivered nowhere.
SCAN_KEY = "review-scan"

# What this reviewer is called in verdicts.jsonl. A free string by design --
# ``latest`` keys on (unit, source), so a future judge writing "ai:<model>"
# sits beside a human's verdict rather than overwriting it.
SOURCE = "human"

# 1-5, in the order verdicts.REASONS lists them, so the pane can number the
# buttons off the same table and the two can never disagree.
REASON_KEYS = {str(i + 1): reason for i, reason in enumerate(verdicts_mod.REASONS)}

# The reference image a unit was generated from, in the order to look. A text
# job writes reference.png; an upload has only the input it was given.
REFERENCE_NAMES = ("reference.png", "input.png")


@dataclass
class ReviewState:
    """One review session: which runs exist, which one is open, where in it.

    Nothing here is persisted. A stored run directory would outlive the sweep
    it names -- ``prune`` and a hand-deleted bench dir both make it a path to
    nothing -- and a mode that opens on an error is worse than one that opens
    on a list.
    """

    # [{dir, label, units, todo}], newest first.
    runs: list[dict[str, Any]] = field(default_factory=list)
    run_dir: Path | None = None
    # The open run's units, in the order the run recorded them. The dicts are
    # the same objects the matching ``runs`` entry holds, so recording a
    # verdict updates both the row and the list's remaining count.
    units: list[dict[str, Any]] = field(default_factory=list)
    index: int = 0
    # Whether R has been pressed and a reason key is what the mode is waiting
    # for. Cleared by anything that moves, because the armed state belongs to
    # the unit that was on screen when it was armed.
    pending_reject: bool = False
    # The unit key whose model.glb the shared viewer is currently showing, so
    # the pane loads a mesh on a change of unit rather than every frame.
    loaded_key: str | None = None
    scanning: bool = False


def ensure(ctx: Any) -> ReviewState:
    """The mode's state, built on first use -- lazy for the reason Build's is:
    a session that never reviews a sweep should not pay for it."""
    state = ctx.state.review
    if state is None:
        state = ReviewState()
        ctx.state.review = state
    return state


# --- scanning ----------------------------------------------------------------


def _unit_records(run_dir: Path) -> list[dict[str, Any]]:
    """One row per unit, joined to whatever verdict it already carries.

    ``latest_items`` is the join target rather than the sweep spec: it is what
    was actually run, it survives a resume, and its ``param``/``value`` are the
    pair the report groups on.
    """
    from ..bench import runner as runner_mod

    recorded = verdicts_mod.latest(run_dir)
    out: list[dict[str, Any]] = []
    for key, record in runner_mod.latest_items(run_dir).items():
        seen = recorded.get((key, SOURCE))
        out.append(
            {
                "key": key,
                "param": record.get("param"),
                "value": record.get("value"),
                "status": record.get("status"),
                "seconds": record.get("seconds"),
                "dir": run_dir / "items" / key,
                "verdict": (seen or {}).get("verdict"),
                "reasons": list((seen or {}).get("reasons") or ()),
                # The archived job.json, read lazily on navigation and cached
                # here. None means "not read yet", {} means "there is none".
                "job": None,
            }
        )
    return out


def _collect(config: Any) -> list[dict[str, Any]]:
    """Every sweep run and its units. Task thread only: this walks the whole
    bench directory and reads two JSONL files per run."""
    from ..bench import report as report_mod

    out: list[dict[str, Any]] = []
    for run_dir in report_mod.sweep_runs(config):
        try:
            units = _unit_records(run_dir)
        except OSError:
            log.exception("could not read the sweep run at %s", run_dir)
            continue
        out.append(
            {
                "dir": run_dir,
                "label": run_dir.name,
                "units": units,
                "todo": sum(1 for unit in units if unit["verdict"] is None),
            }
        )
    # Newest first: run directories are named by their start timestamp, and the
    # sweep worth reviewing is almost always the one that just finished.
    out.reverse()
    return out


def scan(ctx: Any) -> None:
    """Re-read the bench directory, off the frame thread."""
    state = ensure(ctx)
    if state.scanning:
        return
    state.scanning = True
    if not ctx.submit(SCAN_KEY, _collect, ctx.runtime.config):
        # The runner refuses a key already in flight. Leaving the flag set here
        # is what makes the mode permanently inert after a double click.
        state.scanning = False


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``review-`` key."""
    state = ensure(ctx)
    if done.key != SCAN_KEY:
        return
    state.scanning = False
    if not isinstance(done.result, list):
        return
    state.runs = done.result
    # Whatever was open stays open if the rescan still finds it; otherwise the
    # newest run, which is what an empty session wants and a pruned one needs.
    wanted = state.run_dir
    if wanted is not None and any(run["dir"] == wanted for run in state.runs):
        open_run(ctx, wanted)
    elif state.runs:
        open_run(ctx, state.runs[0]["dir"])
    else:
        state.run_dir, state.units, state.index = None, [], 0


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed scan must not leave the mode inert: ``scanning`` gates every
    button and every key."""
    state = ctx.state.review
    if state is not None and done.key == SCAN_KEY:
        state.scanning = False


# --- the open run ------------------------------------------------------------


def open_run(ctx: Any, run_dir: Path) -> None:
    """Show a run, starting on the first unit with no verdict.

    Resuming is the common case: landing back on unit one every time is what
    makes it useless.
    """
    state = ensure(ctx)
    entry = next((run for run in state.runs if run["dir"] == run_dir), None)
    state.run_dir = run_dir
    state.units = list(entry["units"]) if entry is not None else []
    state.pending_reject = False
    state.index = next(
        (i for i, unit in enumerate(state.units) if unit["verdict"] is None), 0
    )
    _touch(state)


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
    _touch(state)


def advance(state: ReviewState, *, unverdicted_only: bool = False) -> None:
    """Forward one, or forward to the next thing left to do.

    ``unverdicted_only`` is what a recorded verdict uses: walking back onto
    work already answered asks the same question twice. It falls back to a
    plain step when there is nothing left ahead, so the pane keeps showing the
    unit the verdict was just given to rather than jumping to the start.
    """
    if not state.units:
        return
    state.pending_reject = False
    if unverdicted_only:
        ahead = next(
            (
                i
                for i in range(state.index + 1, len(state.units))
                if state.units[i]["verdict"] is None
            ),
            None,
        )
        if ahead is not None:
            state.index = ahead
            _touch(state)
            return
        if all(unit["verdict"] is not None for unit in state.units):
            _touch(state)
            return
    state.index = min(state.index + 1, len(state.units) - 1)
    _touch(state)


def record(ctx: Any, verdict: str, reasons: Any = ()) -> None:
    """Write one verdict for the unit on screen, then move on.

    Inline on the frame thread on purpose: one flushed line appended to a small
    file, the same IO class as ``settings.set``, and putting it on a task thread
    would mean a keypress whose effect arrives some frames later -- which, at
    the rate a reviewer presses A, reorders verdicts against navigation.
    """
    state = ensure(ctx)
    unit = current(state)
    if unit is None or state.run_dir is None or state.scanning:
        return
    reasons = list(reasons)
    try:
        verdicts_mod.append_verdict(
            state.run_dir,
            unit=unit["key"],
            source=SOURCE,
            verdict=verdict,
            reasons=reasons,
            param=unit.get("param"),
            value=unit.get("value"),
        )
    except (OSError, ValueError):
        log.exception("could not record a verdict for %s", unit["key"])
        ctx.toast("Could not record that verdict.", "error")
        return
    unit["verdict"] = verdict
    unit["reasons"] = reasons
    _recount(state)
    advance(state, unverdicted_only=True)


def _recount(state: ReviewState) -> None:
    """Keep the run list's remaining count true. The unit dicts are shared with
    the ``runs`` entry, so only the tally needs recomputing."""
    for run in state.runs:
        if run["dir"] == state.run_dir:
            run["todo"] = sum(1 for unit in run["units"] if unit["verdict"] is None)


# --- what the pane reads -----------------------------------------------------


def _touch(state: ReviewState) -> None:
    unit = current(state)
    if unit is not None:
        load_job(unit)


def load_job(unit: dict[str, Any]) -> None:
    """The archived ``job.json``, read once and cached on the unit.

    Once per navigation on the frame thread is the same bargain every other
    pane makes with a small file; once per *frame* is not, which is what the
    cache is for. A missing or unreadable file caches ``{}`` rather than None,
    so a broken unit is not re-read every frame either.
    """
    if unit.get("job") is not None:
        return
    path = Path(unit["dir"]) / "job.json"
    try:
        doc = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        doc = {}
    unit["job"] = doc if isinstance(doc, dict) else {}


def model_path(unit: dict[str, Any]) -> Path:
    return Path(unit["dir"]) / "model.glb"


def reference_path(unit: dict[str, Any]) -> Path | None:
    """What the unit was generated from, or None if neither was copied."""
    for name in REFERENCE_NAMES:
        path = Path(unit["dir"]) / name
        if path.exists():
            return path
    return None


def mesh_lines(unit: dict[str, Any]) -> list[str]:
    """The mesh verdicts the worker already computed, as text.

    Deliberately the two measurements kept apart everywhere else: the report
    ("will an importer accept it, will it sit on the floor") and the audit
    ("can you see through it"). They answer different questions and merging
    them into one badge is what made the old one claim watertight about a
    silhouette check.
    """
    params = (unit.get("job") or {}).get("params") or {}
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
    """One line naming what was varied -- ``lora_weight = 0.9``, or "baseline"
    for the control unit, whose item record carries no param at all."""
    param = unit.get("param")
    if not param:
        return "baseline"
    return f"{param} = {unit.get('value')}"


# --- keys --------------------------------------------------------------------


def handle_key(ctx: Any, event: Any) -> bool:
    """Review's shortcuts. -> whether the key was consumed.

    The caller returns unconditionally either way, for the reason Inker's and
    Build's do: Review has replaced the viewport and the forms, so a key
    falling through here would toggle a wireframe or submit a job against a
    pane that is not on screen. The return value is still honest, because it is
    what the tests read.
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

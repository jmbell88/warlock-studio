"""Human (and future AI) Accept/Reject verdicts on a sweep or suite run's
units, stored the way ``runner.py`` stores ``items.jsonl`` -- an append-only
JSONL file, flushed per line so a Ctrl-C mid-write leaves at worst one torn
last record, read back tolerant of exactly that.

A verdict is not derived from anything the run already recorded: ``items.jsonl``
says what happened (did the job finish, how long did it take), this file says
what a reviewer thought of the result. The two never merge, which is why this
is a second file rather than a field bolted onto ``items.jsonl``.

**The AI-judge seam.** A verdict's ``source`` is a free string, not an enum of
"human" vs "ai" -- ``append_verdict`` only requires it be non-empty. A future
``bench/judge.py`` is expected to read, per unit directory
(``<run_dir>/items/<unit_key>/``): ``views/*.png`` and ``views.json`` (the
rendered turntable), ``reference.png`` (what the item asked for), and
``job.json`` (whose ``params["mesh_report"]``/``params["mesh_audit"]`` carry
the topology/silhouette verdicts the worker already computed) -- score them
with a model, and write the result through this same ``append_verdict`` with
``source=f"ai:{model_name}"`` (e.g. ``"ai:gpt-4-vision"``). Nothing else about
storage changes: ``latest`` keys on ``(unit, source)``, so a human's verdict on
a unit and an AI's sit side by side rather than one overwriting the other, and
``unverdicted(..., source="ai:...")`` lets a judge run resume exactly like a
human review session does.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FILENAME = "verdicts.jsonl"

VERDICTS = ("accept", "reject")

# Rejection reasons a reviewer picks from -- short and mesh/render-specific,
# not free text, so a later report can tally them. Meaningful on reject only,
# but not refused on accept: a reviewer who mis-clicks a reason before
# switching to Accept should not be blocked by it.
REASONS = ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def append_verdict(
    run_dir: Path,
    *,
    unit: str,
    source: str,
    verdict: str,
    reasons: Iterable[str] = (),
    param: str | None,
    value: Any,
) -> None:
    """Append one verdict record, flushed immediately -- mirrors
    ``runner.append_item`` exactly, so the same Ctrl-C-mid-write story applies.

    Raises ``ValueError`` for an unknown ``verdict``, an unknown reason
    (refused regardless of ``verdict`` -- kept simple rather than gating on
    accept/reject), or an empty ``source``.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}; expected one of {VERDICTS}")
    reason_list = list(reasons)
    unknown = [r for r in reason_list if r not in REASONS]
    if unknown:
        raise ValueError(f"unknown reason(s) {unknown}; expected one of {list(REASONS)}")
    if not source:
        raise ValueError("source must be non-empty")

    record = {
        "unit": unit,
        "source": source,
        "verdict": verdict,
        "reasons": reason_list,
        "param": param,
        "value": value,
        "created_at": _now(),
    }
    # append_item assumes run_dir already exists (plan_run/plan_sweep always
    # create it first); a verdict can be the first thing written to a run
    # directory in a test or a standalone review session, so this mkdir has
    # no analogue there.
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def read_verdicts(run_dir: Path) -> list[dict[str, Any]]:
    """Every recorded verdict, oldest first, torn lines skipped -- mirrors
    ``runner.read_items``."""
    path = run_dir / FILENAME
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            # A torn last line is what a Ctrl-C mid-write leaves.
            continue
        if isinstance(record, dict) and "unit" in record and "source" in record:
            out.append(record)
    return out


def latest(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """One record per ``(unit, source)`` -- the newest, so a changed-my-mind
    re-review supersedes the earlier verdict rather than sitting beside it."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for record in read_verdicts(run_dir):
        out[(record["unit"], record["source"])] = record
    return out


def unverdicted(
    run_dir: Path, unit_keys: Iterable[str], *, source: str = "human"
) -> list[str]:
    """The subset of ``unit_keys`` with no verdict from ``source``, in the
    order given -- what a review session (or a judge run) still has to do."""
    seen = {u for (u, s) in latest(run_dir) if s == source}
    return [k for k in unit_keys if k not in seen]

"""Accept/Reject verdicts on a job, and the one function that records them.

Verdicts used to be an append-only JSONL file beside a sweep run directory,
keyed by a unit key that only meant anything inside that run. They are rows
now, keyed by job id, which is what lets *any* finished asset be judged --
daily use and sweep units feeding one pool.

**The vector is snapshotted onto the row on purpose.** It is denormalized and
that is the point: ``prune_jobs`` deletes job rows and their directories, and
the learning corpus has to outlive the assets it was learned from. A verdict
that only pointed at a job id would evaporate the first time a workshop was
tidied up -- taking with it every finding the user spent an afternoon
recording.

**The AI-judge seam is unchanged.** ``source`` is a free string, not an enum:
``latest_verdicts`` keys on (job_id, source), so a future judge writing
``ai:<model>`` sits beside a human's verdict rather than overwriting it, and
``unverdicted_models(source="ai:...")`` lets a judge run resume exactly as a
human review session does.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import findings
from .core import WarlockService
from .errors import Invalid

VERDICTS = ("accept", "reject")

# Rejection reasons a reviewer picks from -- short and mesh/render-specific,
# not free text, so the findings table can tally them. Meaningful on reject
# only, but not refused on accept: a reviewer who mis-clicks a reason before
# switching to Accept should not be blocked by it.
REASONS = ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")

SOURCE_HUMAN = "human"


def record_verdict(
    svc: WarlockService,
    job_id: str,
    *,
    verdict: str,
    reasons: Iterable[str] = (),
    source: str = SOURCE_HUMAN,
) -> dict[str, Any]:
    """File one verdict against a job, with a snapshot of its config vector."""
    if verdict not in VERDICTS:
        raise Invalid(f"verdict must be one of {list(VERDICTS)}", field="verdict")
    reason_list = [str(r) for r in reasons]
    unknown = [r for r in reason_list if r not in REASONS]
    if unknown:
        raise Invalid(
            f"unknown reason(s) {unknown}; expected one of {list(REASONS)}", field="reasons"
        )
    if not source:
        raise Invalid("source must be non-empty", field="source")

    job = svc.require_job(job_id)
    row_id = svc.store.add_verdict(
        job_id,
        source=source,
        verdict=verdict,
        reasons=reason_list,
        vector=findings.config_vector(job),
    )
    return {"id": row_id, "job": job_id, "verdict": verdict, "reasons": reason_list}

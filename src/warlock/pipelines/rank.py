"""Which of N reference candidates to look at first.

Nothing here rejects anything. A fan-out of eight candidates arrives in
submission order, which is the order the seeds happened to be drawn in, and
every one of them already carries a measurement nobody reads -- the
composition report the reference stage takes anyway. This turns that report,
plus an optional DINOv2 similarity against the profile's style anchor, into
one number the gallery can sort by.

Pure: no torch, no I/O, no imports from service/queue/studio. The cosine is
computed by the caller (bench.metrics) and handed in, so this module stays
testable without weights and the weighting stays a single readable formula
rather than something buried in the worker.
"""

from __future__ import annotations

from typing import Any

from .reference import DEFAULT_OCCUPANCY

# How the two halves trade off. Composition dominates because it is the one
# that predicts whether the mesh stage can succeed at all; the anchor is about
# whether this candidate belongs with the others, which only matters among
# candidates that can all reconstruct.
COMPOSITION_WEIGHT = 0.6
ANCHOR_WEIGHT = 0.4

# What an unmeasured reference scores. Deliberately mid-range: a job whose
# report is missing is unknown, not bad, and scoring it zero would sort it
# below a candidate that was measured and refused.
UNMEASURED = 0.5

# Per-defect costs, all applied to a base of 1.0.
WARNING_COST = 0.10
COMPONENT_COST = 0.15
TOUCH_COST = 0.10
# The most the occupancy distance can take off, and it is charged in proportion
# to that distance: |occupancy - DEFAULT_OCCUPANCY| straight, not normalised by
# anything, so a candidate at 0.70 against a target of 0.78 loses 0.4 * 0.08 --
# very little -- and one at 0.10 loses a quarter of its score. The min(1.0, ..)
# beside it cannot fire on a measured report (occupancy is subject pixels over
# frame pixels, so the distance is at most DEFAULT_OCCUPANCY) and is kept as a
# guard on a report read back off disk, which is JSON some other build wrote.
OCCUPANCY_COST = 0.40


def composition_score(report: dict[str, Any] | None) -> float:
    """0..1 for how well framed one reference is, from its own report."""
    if not isinstance(report, dict):
        return UNMEASURED
    if report.get("ok") is False:
        # The report already said this cannot reconstruct. Nothing below can
        # rescue it, and a candidate that is going to be refused at promotion
        # belongs last.
        return 0.0
    score = 1.0
    try:
        occupancy = float(report.get("occupancy") or 0.0)
    except (TypeError, ValueError):
        occupancy = 0.0
    score -= OCCUPANCY_COST * min(1.0, abs(occupancy - DEFAULT_OCCUPANCY))
    score -= WARNING_COST * len(report.get("warnings") or ())
    score -= COMPONENT_COST * max(0, int(report.get("components") or 1) - 1)
    score -= TOUCH_COST * min(1, len(report.get("touches") or ()))
    return max(0.0, min(1.0, score))


def score(
    report: dict[str, Any] | None, anchor_cosine: float | None = None
) -> dict[str, Any]:
    """The whole verdict for one candidate, as it is stored in params.

    ``anchor_cosine`` is a DINOv2 cosine in -1..1 and is rescaled to 0..1 here
    rather than by its producer, so the raw number stays comparable with every
    other cosine in the codebase.
    """
    composition = composition_score(report)
    if anchor_cosine is None:
        return {"score": composition, "composition": composition, "anchor": None}
    anchor = max(0.0, min(1.0, (float(anchor_cosine) + 1.0) / 2.0))
    combined = COMPOSITION_WEIGHT * composition + ANCHOR_WEIGHT * anchor
    return {
        "score": max(0.0, min(1.0, combined)),
        "composition": composition,
        "anchor": float(anchor_cosine),
    }

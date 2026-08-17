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

# How much of the final score the human-preference term (PickScore, an
# optional download) takes when it was measured. Folded in as a blend over
# the composition/anchor score rather than as a third weight beside them, so
# a candidate with no preference measurement scores *exactly* what it always
# did -- the same absence-changes-nothing rule the anchor follows. 0.25 keeps
# composition dominant (0.45 effective with an anchor), which is right for
# the same reason COMPOSITION_WEIGHT leads: aesthetics only matter among
# candidates that can all reconstruct. The pre-registered labelling session
# (docs/measurements/2026-08-09-judge-threshold.md) is the natural place to
# calibrate this against real picks; until it runs, the weight is the
# literature's "best single human-preference predictor" claim applied
# conservatively.
PREFERENCE_WEIGHT = 0.25

# The affine window that maps PickScore's raw logit (logit_scale * cosine)
# onto 0..1 for blending. Probed on real weights 2026-08-17: a clean subject
# scored ~24.7 against its own description and ~16.7 against an unrelated
# one, so the window brackets that observed spread. Values outside clamp:
# below the floor everything is equally disliked, above the ceiling equally
# liked, and the ranking only needs resolution in between.
PREFERENCE_LOW = 16.0
PREFERENCE_HIGH = 25.0


def preference_component(raw: float) -> float:
    """A raw PickScore logit -> the 0..1 term the blend uses."""
    span = PREFERENCE_HIGH - PREFERENCE_LOW
    return max(0.0, min(1.0, (float(raw) - PREFERENCE_LOW) / span))

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
    report: dict[str, Any] | None,
    anchor_cosine: float | None = None,
    preference: float | None = None,
) -> dict[str, Any]:
    """The whole verdict for one candidate, as it is stored in params.

    ``anchor_cosine`` is a DINOv2 cosine in -1..1 and is rescaled to 0..1 here
    rather than by its producer, so the raw number stays comparable with every
    other cosine in the codebase. ``preference`` is PickScore's raw logit,
    stored raw for the same reason and blended via ``preference_component``.
    Every term is optional and its absence changes nothing about the rest --
    a candidate measured under fewer models scores what it always did.
    """
    composition = composition_score(report)
    if anchor_cosine is None:
        base = composition
        out: dict[str, Any] = {
            "score": composition, "composition": composition, "anchor": None,
        }
    else:
        anchor = max(0.0, min(1.0, (float(anchor_cosine) + 1.0) / 2.0))
        base = max(
            0.0,
            min(1.0, COMPOSITION_WEIGHT * composition + ANCHOR_WEIGHT * anchor),
        )
        out = {
            "score": base,
            "composition": composition,
            "anchor": float(anchor_cosine),
        }
    if preference is not None:
        out["preference"] = float(preference)
        blended = (1.0 - PREFERENCE_WEIGHT) * base + (
            PREFERENCE_WEIGHT * preference_component(preference)
        )
        out["score"] = max(0.0, min(1.0, blended))
    return out

"""What a mesh measurement is allowed to *say*. Headless, and one wording.

``hole_worst`` is the silhouette audit's reading and it is **corpus-dependent**:
it is never a quality scale, and INVARIANTS forbids presenting it as a ranking.
Three surfaces show it -- the quality badge, the inspector's remesh line and
Review's mesh lines -- and each had spelled the caveat itself, one of them by
importing imgui-bearing ``widgets`` from inside a per-frame function.

Nothing here imports imgui, so a test can assert the sentence without a GL
context and ``review_mode`` can read the threshold without dragging the whole
widget layer into a frame.

See ``docs/measurements/2026-08-09-rebaseline.md`` and ``judge.py``'s module
docstring for the measurement itself.
"""

from __future__ import annotations

from typing import Any

# Below this the silhouette audit has found nothing, and that is the whole of
# what it means (P120). It used to be the boundary of a *green* verdict, which
# is the claim that had to go.
AUDIT_UNINFORMATIVE = 0.02

#: The one wording, for the one thing a low reading means.
UNINFORMATIVE_CAVEAT = "(a solid, featureless mesh scores this too)"


def caveat_for(ratio: float | None) -> str:
    """The caveat a reading needs, or ``""``.

    A high reading is real evidence of a hole and needs no caveat; a low one is
    what a solid, featureless slab measures, and unqualified beside a mesh it
    reads as "no holes -- good" to whoever is grading.
    """

    if ratio is None:
        return ""
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return ""
    return UNINFORMATIVE_CAVEAT if value < AUDIT_UNINFORMATIVE else ""


def remesh_line(attempts: list[Any]) -> list[str]:
    """What a remesh compared, and what it kept. Never a ranking.

    The old sentence was "kept the best of 12.3%, 4.5%", which is
    ``hole_worst`` presented as a quality scale in the two ways INVARIANTS
    names: "best" is a ranking word, and the figures stand unqualified. What is
    said instead is the *measurement* -- silhouette openness -- and the rule
    that actually chose, which is "the lowest reading", plus the caveat when
    the reading that won is one the audit cannot tell from a solid slab.
    """

    if not isinstance(attempts, list) or len(attempts) < 2:
        return []
    readings = [None if a.get("worst") is None else float(a["worst"]) for a in attempts]
    shown = ", ".join("unmeasured" if r is None else f"{r * 100:.1f}%" for r in readings)
    measured = [r for r in readings if r is not None]
    kept = min(measured) if measured else None
    line = (
        f"remeshed {len(attempts) - 1} time(s); silhouette openness measured {shown}"
    )
    if kept is not None:
        line += f" -- kept the lowest ({kept * 100:.1f}%)"
    lines = [line, "Openness is corpus-dependent; it is not a quality score."]
    caveat = caveat_for(kept)
    if caveat:
        lines.append(caveat)
    return lines

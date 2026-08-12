"""Training the quality probes from the labels on disk, and scoring with them.

`TODO.md` §7's service half: the seam between the verdict table, which owns the
labels, and ``judge.py``, which owns the arithmetic and may not import anything
from here. Everything slow lives behind ``train``, which is a TaskRunner task --
a DINOv2 forward pass over a hundred images plus a logistic fit is seconds, and
seconds on the frame thread is a freeze.

**A label whose pixels are gone cannot train anything, and the count says so.**
``verdicts.vector`` is denormalized exactly so the corpus outlives ``prune_jobs``
deleting the assets it was learned from -- which is what keeps the *marginals*
computable forever. A probe is trained on pixels, so for this one consumer a
pruned row is unusable. Reporting ``labels`` and ``usable`` separately is the
difference between "9 of 90 labels still have their image" and a probe that
claims ninety.

**Nothing here decides anything.** ``score_job`` returns a probability or
``None``; there is no threshold, because a threshold is a constant the stored
corpus is keyed on and owes a measurement document first. ``None`` rather than
0.5 for "no opinion", because 0.5 reads like one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import judge as judge_mod
from . import verdicts as verdicts_mod
from .core import WarlockService
from .errors import Invalid

log = logging.getLogger(__name__)

# The stages this half may act on, **imported rather than restated**: a second
# spelling of the same list is how the two come to disagree.
#
# ``judge.STAGES`` declares three questions and ``model`` is the one that is
# declared and unbuilt. It matters here and not in ``judge.py`` because this is
# the half that knows where the pixels are: ``_image_for`` answers with
# ``reference.png``/``input.png``, so ``train(svc, "model")`` would quietly fit
# the *mesh* probe to 2D reference images -- a probe pointed at a population it
# was not fitted to, which is exactly what one file per stage exists to prevent.
# ``db.LABEL_POPULATION`` already refuses the same question on the listing side;
# this is the other half of that guard.
TRAINABLE_STAGES = verdicts_mod.IMAGE_STAGES


def _check_stage(stage: str) -> str:
    if stage not in TRAINABLE_STAGES:
        # ``errors.Invalid`` with a ``field``, like every other refusal in this
        # layer. A bare ``ValueError`` was the single exception, and the
        # refusal contract is what lets the UI put the message beside the
        # control it is about -- a ValueError escaping the service boundary is
        # a 500 rather than a highlighted select (SVC-04).
        raise Invalid(
            f"stage must be one of {list(TRAINABLE_STAGES)}; {stage!r} is declared "
            "and unbuilt -- the mesh probe pools over rendered views and cannot "
            "be fitted to a reference image",
            field="stage",
        )
    return stage


# Where a probe lives: beside ``findings.json``, because it is the same kind of
# thing -- a derived artifact of the verdict corpus, rebuilt from the DB and
# disposable. Not in the data dir: that holds assets, and a probe is not one.
def probe_dir(svc: WarlockService) -> Path:
    return Path(svc.config.bench_dir)


def _image_for(svc: WarlockService, job_id: str) -> Path | None:
    job_dir = svc.job_dir(job_id)
    for name in verdicts_mod.IMAGE_NAMES:
        path = job_dir / name
        if path.exists():
            return path
    return None


def _labels(svc: WarlockService, stage: str, source: str) -> list[dict[str, Any]]:
    """Every latest label for one question, by one labeller.

    Filtered here rather than in a DB query because ``latest_verdicts`` is the
    one place the latest-wins rule is spelled out, and a second query with its
    own GROUP BY would be a second chance to get that rule subtly different.
    """
    return [
        row
        for row in svc.store.latest_verdicts()
        if row.get("stage") == stage and row.get("source") == source
    ]


def train(
    svc: WarlockService,
    stage: str,
    *,
    source: str = verdicts_mod.SOURCE_HUMAN,
) -> dict[str, Any]:
    """Fit and write the probe for one question. Task thread only.

    -> a summary the pane can render verbatim, including every row it could not
    use. It never raises for an ordinary shortage: "not enough labels yet" is the
    normal state of a corpus being built.
    """
    _check_stage(stage)
    rows = _labels(svc, stage, source)
    embeddings: list[Any] = []
    labels: list[float] = []
    missing = unreadable = 0
    for row in rows:
        image = _image_for(svc, row["job_id"])
        if image is None:
            # Pruned, or a row about a job whose directory is gone. Counted, not
            # skipped silently -- see the module docstring.
            missing += 1
            continue
        vector = judge_mod.embed(image, svc.config)
        if vector is None:
            # No DINOv2 on this machine, or an unreadable PNG. Either way one bad
            # row must not cost the other eighty.
            unreadable += 1
            continue
        embeddings.append(vector)
        labels.append(1.0 if row.get("verdict") == "accept" else 0.0)

    summary: dict[str, Any] = {
        "stage": stage,
        "source": source,
        "labels": len(rows),
        "usable": len(labels),
        "missing": missing,
        "unreadable": unreadable,
        "positives": int(sum(labels)),
        "needed": judge_mod.MIN_PER_CLASS,
        "trained": False,
    }
    if not embeddings:
        return summary

    probe = judge_mod.fit(embeddings, labels, stage=stage, corpus=source)
    if probe is None:
        return summary
    judge_mod.save(probe, judge_mod.probe_path(probe_dir(svc), stage))
    summary["trained"] = True
    summary["positives"] = probe.positives
    return summary


def probe(svc: WarlockService, stage: str) -> Any:
    """The probe on disk for one question, or None.

    Not stage-checked: reading a file that is not there is already None, and a
    caller asking whether the mesh probe exists deserves that answer rather than
    an exception.
    """
    return judge_mod.load(judge_mod.probe_path(probe_dir(svc), stage))


def score_job(svc: WarlockService, job_id: str, stage: str) -> float | None:
    """This job's image, scored by the probe for ``stage``. None for no opinion.

    The single-job form, and it is deliberately not what the app calls: every
    screen that scores anything is scoring a grid, so ``score_jobs`` -- which
    loads the probe once for the whole set -- is the path the panes take, and
    this one is for a caller that has exactly one job in hand (and for the tests
    that pin one score at a time). Kept rather than folded into ``score_jobs``
    because the two differ in the thing that costs: a probe load per call is
    fine for one row and is a file read per row for a hundred.

    Every way of having nothing to say collapses to None: no probe, no image, no
    weights, an unreadable file. A judge failure must never fail its caller --
    ``Worker._record_observation``'s rule, and here the caller is a frame.
    """
    _check_stage(stage)
    fitted = probe(svc, stage)
    if fitted is None:
        return None
    image = _image_for(svc, job_id)
    if image is None:
        return None
    vector = judge_mod.embed(image, svc.config)
    if vector is None:
        return None
    return judge_mod.score(fitted, vector)


def score_jobs(
    svc: WarlockService, job_ids: Any, stage: str
) -> dict[str, float | None]:
    """Many jobs, one probe load. Task thread only.

    ``score_job`` reads the probe off disk and rebuilds it per call, which is
    fine for one job and is a file read per row for a review grid of a hundred.
    The result is a plain dict so the caller can merge it onto whatever it is
    holding; a missing entry and a ``None`` entry mean the same thing and both
    are honest -- ``{}`` when there is no probe at all.

    No threshold and no verdict, exactly as ``score_job`` files none. A score is
    an ordering, and turning one into an accept needs a constant the stored
    corpus would then be keyed on.
    """
    _check_stage(stage)
    fitted = probe(svc, stage)
    if fitted is None:
        return {}
    scores: dict[str, float | None] = {}
    for job_id in job_ids:
        image = _image_for(svc, job_id)
        vector = None if image is None else judge_mod.embed(image, svc.config)
        scores[job_id] = None if vector is None else judge_mod.score(fitted, vector)
    return scores


def status(
    svc: WarlockService,
    stage: str,
    *,
    source: str = verdicts_mod.SOURCE_HUMAN,
) -> dict[str, Any]:
    """What the pane says about one question, without training anything.

    Cheap on purpose -- one DB read and one stat -- because it is what a panel
    draws. ``trained_at`` is the probe file's mtime: the ``warlockc`` staleness
    rule says a probe should say *when*, and the file already knows.
    """
    _check_stage(stage)
    fitted = probe(svc, stage)
    path = judge_mod.probe_path(probe_dir(svc), stage)
    rows = _labels(svc, stage, source)
    accepts = sum(1 for row in rows if row.get("verdict") == "accept")
    return {
        "stage": stage,
        "labels": len(rows),
        "positives": accepts,
        "negatives": len(rows) - accepts,
        "needed": judge_mod.MIN_PER_CLASS,
        "trained": fitted is not None,
        "trained_labels": fitted.labels if fitted is not None else 0,
        "trained_at": path.stat().st_mtime if fitted is not None and path.exists() else None,
    }

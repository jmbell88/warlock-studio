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

log = logging.getLogger(__name__)

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
    if stage not in judge_mod.STAGES:
        raise ValueError(f"stage must be one of {list(judge_mod.STAGES)}")

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
    """The probe on disk for one question, or None."""
    return judge_mod.load(judge_mod.probe_path(probe_dir(svc), stage))


def score_job(svc: WarlockService, job_id: str, stage: str) -> float | None:
    """This job's image, scored by the probe for ``stage``. None for no opinion.

    Every way of having nothing to say collapses to None: no probe, no image, no
    weights, an unreadable file. A judge failure must never fail its caller --
    ``Worker._record_observation``'s rule, and here the caller is a frame.
    """
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

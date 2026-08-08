"""A quality judge: linear probes over frozen DINOv2 embeddings.

Named for the analogy that produced it (`TODO.md` §7): an industrial vision
system is shown good parts and bad parts, learns the boundary, then inspects on
its own. What is built here is the boundary and nothing else -- **advisory
only**. A probe files a verdict beside a human's and sorts Review; it never
refuses, deletes or retries. That is the only mode in which its accuracy can be
measured before it is trusted, and it mirrors ``native.py``'s doctrine that an
optimisation never replaces the reference it is checked against.

**Pure in the ``vram.py`` / ``memlog.py`` sense.** Stdlib plus numpy plus
``bench.metrics`` for the embedding; nothing from ``service``, ``queue`` or
``studio``; ``None`` rather than an exception when weights, a probe file or torch
are missing. The one dependency is deliberate: ``bench.metrics`` already owns the
DINOv2 loader, its ``local_files_only=True`` flag and its per-device cache, and a
second copy here would be a second answer to "which weights".

**Three probes, separated by intent rather than by artifact type.**
``image-as-product`` (is this a good 2D asset), ``image-as-blank`` (will this
reconstruct) and ``mesh``. The first two read the same *kind* of file and mean
opposite things by good: in 2D mode the image is the deliverable, so a dramatic
plate with pillars and a cast shadow is a better asset and a worse blank. A
single probe trained across both label sets learns the average of two opposed
objectives and is useless for each -- which is why a probe file records the stage
it was fitted to and refuses to be anything else. The mesh probe is `TODO.md` §8
and is not built here: it is blocked on a corpus with positives in it, and its
pooling over eight views is a decision §8 says to make with data.

**Why the probe is trained on pixels and never on the audit scalars.** The first
calibration result is that the obvious feature runs backwards:
AUC(``hole_worst`` -> reject) = 0.115 over the existing corpus, because a slab
has no holes and ``meshaudit`` scores the dominant failure mode as perfect. A
probe fitted to those numbers would have to be sign-flipped to beat a coin, and
would then be fitting the slab artefact rather than quality. It also disqualifies
``hole_worst`` as a sanity check *on* a probe: a probe that disagrees with it is,
on this evidence, more likely to be right.

**Labels must be human.** It is tempting to treat the composition gate's own
refusals as free labels, but those labels *are* the rule's output, so a probe
fitted to them can at best reproduce ``reference.py`` exactly, blind spots
included -- and ``baseline s23`` passed those rules and is still a poor blank,
which is precisely the case a learned judge exists to catch.

**No threshold lives here.** ``score`` returns a probability. A threshold is a
constant the stored corpus is keyed on, so by this repo's own rule it gets a
document under ``docs/measurements/`` before it is baked in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Bumped whenever the stored arrays change meaning. A probe from another schema
# is refused rather than read: ``vendor/warlockc``'s hazard exactly -- an absent
# artifact is obvious, a stale one silently computing last week's answer is not.
SCHEMA_VERSION = 1

# The questions a probe may be fitted to, **spelled exactly as
# ``verdicts.STAGES`` spells them**. Not "mesh": the label column says ``model``,
# and a probe whose stage string did not match the column it is trained from
# would need a translation table between two names for one thing, which is how
# the two drift. ``model`` is declared and left unbuilt -- see the module
# docstring on why §8 is blocked on a corpus rather than on code.
STAGES = ("reference", "blank", "model")

# Below this, per class, there is no boundary to learn -- only the class balance.
# Stated as a constant so a caller can say "12 more labels to go" rather than
# reporting a silent None. Deliberately modest: a linear probe over a 768-d
# frozen embedding needs tens of examples per class, not thousands, and the
# gate that actually matters is `TODO.md` §8's "positives in the tens".
MIN_PER_CLASS = 8

# Fit hyperparameters. Full-batch gradient descent rather than an optimiser from
# scipy or sklearn, for two reasons: neither is a dependency, and a fixed
# iteration count from a zero start is exactly reproducible -- two probes trained
# on one corpus must be the same probe, or "the judge changed its mind" is
# indistinguishable from "the judge was retrained".
ITERATIONS = 2000
LEARNING_RATE = 0.5
L2 = 1e-3


@dataclass(frozen=True, slots=True)
class Probe:
    """One fitted boundary, plus the provenance that makes it interpretable."""

    weights: np.ndarray
    bias: float
    stage: str
    labels: int
    positives: int
    corpus: str = ""
    schema: int = SCHEMA_VERSION

    @property
    def negatives(self) -> int:
        return self.labels - self.positives


def probe_path(directory: Path, stage: str) -> Path:
    """Where a stage's probe lives. One file per question, never one file with a
    stage field inside it: two questions in one artifact is how a probe gets
    pointed at the population it was not fitted to."""
    return Path(directory) / f"probe-{stage}.npz"


# --- fitting -----------------------------------------------------------------


def fit(
    embeddings: Any, labels: Any, *, stage: str, corpus: str = ""
) -> Probe | None:
    """Fit one probe, or None when the corpus cannot support one.

    ``None`` rather than a raise for the same reason the rest of this module
    returns it: "not enough labels yet" is the ordinary state of a corpus being
    built, and the caller's response is a sentence on screen rather than an
    error path.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("embeddings must be (n, dim) and labels (n,)")
    positives = int((y > 0.5).sum())
    negatives = int(len(y) - positives)
    if positives < MIN_PER_CLASS or negatives < MIN_PER_CLASS:
        # One class, or nearly one. A probe fitted to 3 accepts against 81
        # rejects learns "reject" and scores 96% accuracy doing it, which is
        # worse than no probe because it is believed.
        return None

    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    n = float(len(y))
    for _ in range(ITERATIONS):
        p = _sigmoid(x @ w + b)
        error = p - y
        w -= LEARNING_RATE * ((x.T @ error) / n + L2 * w)
        b -= LEARNING_RATE * (error.sum() / n)
    return Probe(
        weights=w.astype(np.float64),
        bias=float(b),
        stage=stage,
        labels=int(len(y)),
        positives=positives,
        corpus=corpus,
    )


def score(probe: Probe | None, embedding: Any) -> float | None:
    """The probe's probability that this embedding is a good one, or None.

    None for a width mismatch as well as for a missing probe: a DINOv2 swap
    changes the embedding width and the probe file would still *load*, so
    scoring is where that has to be caught.
    """
    if probe is None:
        return None
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if vector.shape != probe.weights.shape:
        log.warning(
            "probe expects %d dimensions, got %d", len(probe.weights), len(vector)
        )
        return None
    return float(_sigmoid(np.array([float(vector @ probe.weights + probe.bias)]))[0])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Split by sign so neither branch overflows: exp(1000) warns and returns inf,
    # and inf/inf is a nan score rather than the 1.0 it should be.
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


# --- persistence -------------------------------------------------------------


def save(probe: Probe, path: Path) -> Path:
    """Write a probe, via a temp file and a rename.

    Staged because the UI reads the probe on the frame thread while a retrain
    writes it from a task thread, which is the ``optimize.staged_copy``
    situation: a plain write truncates the file a reader is inside.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``.tmp.npz`` rather than ``.npz.tmp``: ``np.savez`` appends ``.npz`` to any
    # name that does not already end in it, so the obvious spelling writes
    # ``probe.npz.tmp.npz`` and the rename then fails on a file that is not there.
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        weights=probe.weights,
        bias=np.array(probe.bias),
        stage=np.array(probe.stage),
        labels=np.array(probe.labels),
        positives=np.array(probe.positives),
        corpus=np.array(probe.corpus),
        schema=np.array(probe.schema),
    )
    # np.savez appends .npz to a path that lacks it; with_suffix already gave us
    # one, so the written name is exactly ``tmp``.
    tmp.replace(path)
    return path


def stage_is_known(stage: str) -> bool:
    """Whether a probe can be fitted for this label stage at all.

    ``verdicts.STAGES`` and ``judge.STAGES`` hold the same three words and must
    keep holding them; ``tests/test_judge.py`` asserts the two sets are equal
    rather than trusting a comment to keep them so.
    """
    return stage in STAGES


def load(path: Path) -> Probe | None:
    """Read a probe, or None -- missing, unreadable, or from another schema."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        # allow_pickle stays off: a probe file is numbers and short strings, and
        # unpickling one would make it executable.
        with np.load(path, allow_pickle=False) as data:
            schema = int(data["schema"])
            if schema != SCHEMA_VERSION:
                log.warning(
                    "%s was written by schema %d, this is %d -- ignoring it",
                    path.name, schema, SCHEMA_VERSION,
                )
                return None
            return Probe(
                weights=np.asarray(data["weights"], dtype=np.float64),
                bias=float(data["bias"]),
                stage=str(data["stage"]),
                labels=int(data["labels"]),
                positives=int(data["positives"]),
                corpus=str(data["corpus"]),
                schema=schema,
            )
    except Exception:
        log.exception("could not read the probe at %s", path)
        return None


# --- the embedding seam ------------------------------------------------------


def embed(image_path: Path, config: Any = None, device: str | None = None):
    """One image's DINOv2 CLS embedding, or None.

    None covers every way this can be unavailable -- weights never downloaded,
    no torch, an unreadable PNG, a driver in a bad mood -- because the caller is
    the UI and a judge failure must never be an exception at the call site. That
    is ``Worker._record_observation``'s rule applied to the frame thread.

    The availability probe precedes the torch import, which is the ordering
    ``test_offline.py`` pins everywhere else: a checkout with no text2image
    extra gets "unavailable" rather than an ImportError about a dependency that
    was never the problem.
    """
    from .bench import metrics

    if not metrics.dino_available(config):
        return None
    try:
        return metrics.cls_embedding(Path(image_path), config=config, device=device)
    except Exception:
        log.exception("could not embed %s", image_path)
        return None

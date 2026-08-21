"""The quality judge: a linear probe over frozen DINOv2 embeddings.

The quality-judge programme. Every test here runs with no weights, no GPU and
no torch, which is the point of the module being pure: ``judge.py`` is stdlib
plus numpy, imports nothing from ``service``/``queue``/``studio``, and returns
``None`` rather than raising when a probe file or DINOv2 is absent. That is the
``vram.py`` / ``memlog.py`` doctrine, and it is what makes the arithmetic below
assertable from synthetic vectors.

Two design points are pinned here rather than left to the caller.

**A probe carries its own provenance.** Corpus size, label count and a schema
version, in the ``.npz`` -- the ``vendor/warlockc`` staleness hazard exactly: an
absent probe is obvious, a stale one quietly scoring last week's opinion is not.

**Advisory means advisory.** ``score`` returns a number and nothing else. There
is no threshold in this module, because the threshold is a constant the stored
corpus is keyed on and by this repo's own rule it gets a measurement document
before it is baked in.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import judge


def _separable(n: int = 40, dim: int = 8, seed: int = 7):
    """Two clouds a linear boundary can separate, and their labels."""
    rng = np.random.default_rng(seed)
    positive = rng.normal(1.0, 0.3, size=(n, dim))
    negative = rng.normal(-1.0, 0.3, size=(n, dim))
    x = np.vstack([positive, negative]).astype(np.float32)
    y = np.array([1] * n + [0] * n, dtype=np.float32)
    return x, y


# --- fitting and scoring -----------------------------------------------------


def test_a_probe_separates_what_it_was_fitted_on(tmp_path):
    x, y = _separable()

    probe = judge.fit(x, y, stage="blank")
    scores = np.array([judge.score(probe, row) for row in x])

    assert probe is not None
    assert ((scores > 0.5) == (y > 0.5)).all()


def test_a_score_is_a_probability_and_nothing_more(tmp_path):
    """No threshold in this module: the threshold is a corpus-keyed constant and
    owes a measurement document before it is baked in anywhere."""
    x, y = _separable()
    probe = judge.fit(x, y, stage="blank")

    for row in x:
        assert 0.0 <= judge.score(probe, row) <= 1.0


def test_fitting_is_deterministic(tmp_path):
    """Two probes trained on one corpus must be the same probe, or "the judge
    changed its mind" is indistinguishable from "the judge was retrained"."""
    x, y = _separable()

    first = judge.fit(x, y, stage="blank")
    second = judge.fit(x, y, stage="blank")

    assert np.array_equal(first.weights, second.weights)
    assert first.bias == second.bias


def test_a_corpus_with_one_class_in_it_is_not_a_probe(tmp_path):
    """The state the existing corpus is actually in: 3 accepts against 81
    rejects for meshes. A probe fitted to one class learns the constant and
    scores 96% doing it, which is worse than no probe because it is trusted."""
    x, _ = _separable()
    y = np.zeros(len(x), dtype=np.float32)

    assert judge.fit(x, y, stage="blank") is None


def test_a_corpus_too_small_to_mean_anything_is_not_a_probe(tmp_path):
    x, y = _separable(n=2)

    assert judge.fit(x, y, stage="blank") is None


def test_the_minimum_is_stated_rather_than_implied():
    """So a caller can say "12 more labels to go" instead of a silent None."""
    assert judge.MIN_PER_CLASS >= 5


# --- provenance and staleness ------------------------------------------------


def test_a_probe_records_the_corpus_it_was_fitted_to(tmp_path):
    x, y = _separable(n=25)

    probe = judge.fit(x, y, stage="blank", corpus="rogue")

    assert probe.labels == 50
    assert probe.positives == 25
    assert probe.stage == "blank"
    assert probe.corpus == "rogue"
    assert probe.schema == judge.SCHEMA_VERSION


def test_a_probe_survives_a_round_trip_through_a_file(tmp_path):
    x, y = _separable()
    probe = judge.fit(x, y, stage="blank")
    path = tmp_path / "blank.npz"

    judge.save(probe, path)
    loaded = judge.load(path)

    assert loaded is not None
    assert np.array_equal(loaded.weights, probe.weights)
    assert (loaded.bias, loaded.stage, loaded.labels) == (
        probe.bias, probe.stage, probe.labels
    )
    assert judge.score(loaded, x[0]) == judge.score(probe, x[0])


def test_a_missing_probe_is_none_rather_than_an_error(tmp_path):
    assert judge.load(tmp_path / "nothing.npz") is None


def test_a_corrupt_probe_is_none_rather_than_an_error(tmp_path):
    path = tmp_path / "blank.npz"
    path.write_bytes(b"not an npz")

    assert judge.load(path) is None


def test_a_probe_from_another_schema_is_refused(tmp_path):
    """The ``warlockc`` rule: ``vendor/`` is gitignored, so a checkout routinely
    holds an artifact built from older sources -- an absent one is obvious, a
    stale one silently computing last week's answer is not."""
    x, y = _separable()
    path = tmp_path / "blank.npz"
    judge.save(judge.fit(x, y, stage="blank"), path)
    data = dict(np.load(path, allow_pickle=False))
    data["schema"] = np.array(judge.SCHEMA_VERSION + 1)
    np.savez(path, **data)

    assert judge.load(path) is None


def test_a_probe_scored_against_the_wrong_width_is_none_not_a_crash(tmp_path):
    """A DINOv2 swap would change the embedding width, and the probe file would
    still load. Scoring it is the point at which that becomes wrong."""
    x, y = _separable(dim=8)
    probe = judge.fit(x, y, stage="blank")

    assert judge.score(probe, np.zeros(16, dtype=np.float32)) is None


# --- the embedding seam ------------------------------------------------------


def test_embedding_without_the_weights_is_none_rather_than_an_error(tmp_path, monkeypatch):
    """A machine that never downloaded DINOv2 gets "unavailable", not a stack
    trace -- and never an import of torch, which is the ordering
    ``test_offline.py`` pins everywhere else."""
    from warlock.bench import metrics

    monkeypatch.setattr(metrics, "dino_available", lambda config=None: False)
    image = tmp_path / "a.png"
    image.write_bytes(b"png-not-really")

    assert judge.embed(image) is None


def test_the_judge_imports_nothing_it_is_not_allowed_to():
    """Pure in the ``vram.py`` sense, so every rule above is assertable
    headlessly. ``bench.metrics`` is the one dependency and it is deliberate:
    the DINOv2 loader, its offline flag and its per-device cache already exist
    there, and a second copy would be a second answer to "which weights"."""
    from pathlib import Path

    source = Path(judge.__file__).read_text("utf-8")
    for banned in ("from .service", "from .queue", "from .studio", "import torch"):
        assert banned not in source, banned


def test_a_judge_failure_is_never_an_exception_at_the_call_site(tmp_path, monkeypatch):
    """``Worker._record_observation``'s rule: a diagnostic must never fail the
    thing it is diagnosing. Here the caller is the UI, and the cost of a raise
    is a frame that does not draw."""
    from warlock.bench import metrics

    monkeypatch.setattr(metrics, "dino_available", lambda config=None: True)

    def boom(*a, **k):
        raise RuntimeError("cuda is on fire")

    monkeypatch.setattr(metrics, "cls_embedding", boom)
    image = tmp_path / "a.png"
    image.write_bytes(b"png-not-really")

    assert judge.embed(image) is None


# --- what a stage means ------------------------------------------------------


def test_a_probe_names_which_question_it_answers(tmp_path):
    """One image, two opposed objectives -- ``baseline s23`` is a better 2D asset
    than a flat T-pose on grey and a worse blank. A probe file that did not say
    which question it was fitted to could be pointed at the other one."""
    x, y = _separable()

    with pytest.raises(ValueError):
        judge.fit(x, y, stage="mesh")


def test_the_judge_and_the_verdict_table_use_one_vocabulary(tmp_path):
    """Both name the same three questions, and they must name them with the same
    words: a probe stage that needed translating to a ``verdicts.stage`` would be
    a table two files could disagree about, and the disagreement would show up as
    a probe silently trained on the wrong population."""
    from warlock.service import verdicts as svc_verdicts

    assert set(judge.STAGES) == set(svc_verdicts.STAGES)


def test_the_probe_filenames_are_one_per_question(tmp_path):
    names = {stage: judge.probe_path(tmp_path, stage) for stage in judge.STAGES}

    assert len(set(names.values())) == len(judge.STAGES)
    assert all(path.parent == tmp_path for path in names.values())

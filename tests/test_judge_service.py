"""Training a probe from the labels on disk, and scoring one image with it.

The quality-judge programme's service half: the seam between the verdict table
(which owns the labels) and ``judge.py`` (which owns the arithmetic). Three
properties matter here and none of them is about the maths.

**A label whose pixels are gone cannot train anything.** The verdict corpus is
denormalized precisely so it outlives ``prune_jobs`` deleting the assets -- which
is what makes the *marginals* survivable. A probe is trained on pixels, so for
this consumer a pruned row is simply unusable, and the count it reports has to
say so rather than quietly training on nine rows while claiming ninety.

**Training never raises into its caller.** It runs on the TaskRunner and its
failure is a toast, not a dead task thread.

**Nothing is scored against a probe that does not exist.** Absent weights,
absent probe, too few labels: all of them are ``None``, which the UI renders as
"no opinion" rather than as a neutral 0.5 that reads like one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from warlock import judge
from warlock.service import judge as svc_judge
from warlock.service import verdicts as svc_verdicts
from warlock.service.errors import Invalid


def _labelled(svc, verdict, *, stage="blank", image=True, prompt="a rogue"):
    """A model-stage job with a reference image and one image label."""
    job_id = svc.store.create("image", prompt, {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    if image:
        (job_dir / "reference.png").write_bytes(b"png-not-really")
    if stage == "model":
        # A mesh verdict is graded; only the image stages take a word. The
        # backfill table is what "accept" meant, so a mesh row planted here is
        # the same evidence it always was.
        from warlock.vectors import BINARY_GRADES

        svc_verdicts.record_verdict(svc, job_id, grade=BINARY_GRADES[verdict])
    else:
        svc_verdicts.record_verdict(svc, job_id, verdict=verdict, stage=stage)
    return job_id


def _fake_embeddings(monkeypatch, mapping):
    """``judge.embed``, replaced by a lookup: the arithmetic is tested in
    test_judge.py, and what is under test here is which rows reach it."""
    seen: list[Path] = []

    def embed(path, config=None, device=None):
        seen.append(Path(path))
        return mapping.get(Path(path).parent.name)

    monkeypatch.setattr(judge, "embed", embed)
    return seen


def _vector(value: float):
    return np.full(8, value, dtype=np.float32)


# --- gathering ---------------------------------------------------------------


def test_training_reads_the_labels_for_one_stage_only(svc, monkeypatch):
    blank = _labelled(svc, "accept", stage="blank")
    product = _labelled(svc, "accept", stage="reference")
    mesh = _labelled(svc, "accept", stage="model")

    seen = _fake_embeddings(monkeypatch, {})
    svc_judge.train(svc, "blank")

    assert [p.parent.name for p in seen] == [blank]
    assert product not in [p.parent.name for p in seen]
    assert mesh not in [p.parent.name for p in seen]


def test_a_label_whose_image_was_pruned_is_reported_not_silently_dropped(
    svc, monkeypatch
):
    """The corpus outlives the assets on purpose -- but a probe is trained on
    pixels, so for *this* consumer a pruned row is unusable. Saying so is the
    difference between "9 of 90 labels usable" and a probe that claims 90."""
    kept = _labelled(svc, "accept")
    pruned = _labelled(svc, "reject")
    (svc.job_dir(pruned) / "reference.png").unlink()

    _fake_embeddings(monkeypatch, {kept: _vector(1.0)})
    result = svc_judge.train(svc, "blank")

    assert result["labels"] == 2
    assert result["usable"] == 1
    assert result["missing"] == 1


def test_too_few_labels_is_a_reported_state_not_a_probe(svc, monkeypatch):
    ids = {_labelled(svc, "accept"): None for _ in range(3)}
    _fake_embeddings(monkeypatch, {job_id: _vector(1.0) for job_id in ids})

    result = svc_judge.train(svc, "blank")

    assert result["trained"] is False
    assert result["needed"] == judge.MIN_PER_CLASS
    assert svc_judge.probe(svc, "blank") is None


def test_a_probe_is_written_once_there_are_enough_of_both(svc, monkeypatch):
    mapping = {}
    for _ in range(judge.MIN_PER_CLASS):
        mapping[_labelled(svc, "accept")] = _vector(1.0)
        mapping[_labelled(svc, "reject")] = _vector(-1.0)
    _fake_embeddings(monkeypatch, mapping)

    result = svc_judge.train(svc, "blank")

    assert result["trained"] is True
    assert result["positives"] == judge.MIN_PER_CLASS
    written = svc_judge.probe(svc, "blank")
    assert written is not None
    assert written.stage == "blank"
    assert written.labels == 2 * judge.MIN_PER_CLASS


def test_a_probe_lives_beside_the_findings_it_was_learned_from(svc, monkeypatch):
    mapping = {}
    for _ in range(judge.MIN_PER_CLASS):
        mapping[_labelled(svc, "accept")] = _vector(1.0)
        mapping[_labelled(svc, "reject")] = _vector(-1.0)
    _fake_embeddings(monkeypatch, mapping)
    svc_judge.train(svc, "blank")

    assert judge.probe_path(Path(svc.config.bench_dir), "blank").exists()


def test_an_unembeddable_image_is_skipped_rather_than_failing_the_run(svc, monkeypatch):
    """``judge.embed`` returns None for a machine with no DINOv2 and for a
    corrupt PNG alike, and one bad row must not cost the other eighty."""
    mapping = {}
    for _ in range(judge.MIN_PER_CLASS):
        mapping[_labelled(svc, "accept")] = _vector(1.0)
        mapping[_labelled(svc, "reject")] = _vector(-1.0)
    broken = _labelled(svc, "accept")
    mapping[broken] = None
    _fake_embeddings(monkeypatch, mapping)

    result = svc_judge.train(svc, "blank")

    assert result["trained"] is True
    assert result["usable"] == 2 * judge.MIN_PER_CLASS
    assert result["unreadable"] == 1


# --- scoring -----------------------------------------------------------------


def test_scoring_without_a_probe_is_no_opinion(svc, monkeypatch):
    job_id = _labelled(svc, "accept")
    _fake_embeddings(monkeypatch, {job_id: _vector(1.0)})

    assert svc_judge.score_job(svc, job_id, "blank") is None


def test_scoring_uses_the_probe_for_the_question_being_asked(svc, monkeypatch):
    mapping = {}
    for _ in range(judge.MIN_PER_CLASS):
        mapping[_labelled(svc, "accept")] = _vector(1.0)
        mapping[_labelled(svc, "reject")] = _vector(-1.0)
    subject = _labelled(svc, "accept")
    mapping[subject] = _vector(1.0)
    _fake_embeddings(monkeypatch, mapping)
    svc_judge.train(svc, "blank")

    score = svc_judge.score_job(svc, subject, "blank")
    assert score is not None and score > 0.5
    # No product probe has been trained, so there is no opinion to give.
    assert svc_judge.score_job(svc, subject, "reference") is None


def test_a_job_with_no_image_scores_nothing(svc, monkeypatch):
    """Not reachable through a label -- ``record_verdict`` refuses an image label
    with no image -- but reachable through pruning, which is the ordinary way a
    scored row loses its pixels."""
    job_id = _labelled(svc, "accept")
    (svc.job_dir(job_id) / "reference.png").unlink()
    _fake_embeddings(monkeypatch, {})

    assert svc_judge.score_job(svc, job_id, "blank") is None


def test_the_probes_state_is_readable_without_training_anything(svc, monkeypatch):
    """What the pane draws: how many labels are in, how many are needed, and
    whether a probe exists at all. The staleness half of the ``warlockc`` rule --
    a probe should say when it was trained."""
    _labelled(svc, "accept")
    _fake_embeddings(monkeypatch, {})

    state = svc_judge.status(svc, "blank")

    assert state["stage"] == "blank"
    assert state["labels"] == 1
    assert state["trained"] is False
    assert state["needed"] == judge.MIN_PER_CLASS


# --- the boundary the mesh probe does not cross ------------------------------


def test_the_mesh_probe_cannot_be_trained_on_reference_pixels(svc, monkeypatch):
    """``_image_for`` answers with ``reference.png``/``input.png``, so nothing
    in this module can produce a rendered view -- and ``train(svc, "model")``
    would quietly fit the mesh probe to 2D images and write it under the name
    the mesh question will want. ``db.LABEL_POPULATION`` refuses the same
    question on the listing side; this is the other half.
    """
    _fake_embeddings(monkeypatch, {})
    for call in (
        lambda: svc_judge.train(svc, "model"),
        lambda: svc_judge.status(svc, "model"),
        lambda: svc_judge.score_job(svc, "whatever", "model"),
        lambda: svc_judge.score_jobs(svc, ["whatever"], "model"),
    ):
        with pytest.raises(Invalid, match="declared"):
            call()


def test_the_trainable_stages_are_the_verdict_tables_image_stages(svc):
    """Imported, never restated: a second spelling of one list is how the two
    come to disagree, and the drift would show up as a probe trained on the
    wrong population."""
    assert svc_judge.TRAINABLE_STAGES is svc_verdicts.IMAGE_STAGES
    assert set(svc_judge.TRAINABLE_STAGES) < set(judge.STAGES)


def test_an_unknown_stage_is_refused_rather_than_writing_a_stray_probe(svc):
    with pytest.raises(Invalid):
        svc_judge.train(svc, "mesh")


# --- scoring many rows -------------------------------------------------------


def test_scoring_many_jobs_loads_the_probe_once(svc, monkeypatch):
    """``score_job`` rebuilds the probe from disk per call, which is a file read
    per row on a review grid of a hundred."""
    mapping = {}
    for _ in range(judge.MIN_PER_CLASS):
        mapping[_labelled(svc, "accept")] = _vector(1.0)
        mapping[_labelled(svc, "reject")] = _vector(-1.0)
    _fake_embeddings(monkeypatch, mapping)
    svc_judge.train(svc, "blank")

    loads: list[int] = []
    real_load = judge.load
    monkeypatch.setattr(judge, "load", lambda path: (loads.append(1), real_load(path))[1])

    scores = svc_judge.score_jobs(svc, list(mapping), "blank")

    assert len(loads) == 1
    assert set(scores) == set(mapping)
    assert all(isinstance(v, float) for v in scores.values())


def test_scoring_many_jobs_with_no_probe_is_an_empty_answer(svc, monkeypatch):
    job_id = _labelled(svc, "accept")
    _fake_embeddings(monkeypatch, {job_id: _vector(1.0)})

    assert svc_judge.score_jobs(svc, [job_id], "blank") == {}


def test_a_row_with_no_pixels_scores_none_beside_rows_that_do(svc, monkeypatch):
    """One pruned row must not cost the other eighty their scores -- the rule
    ``train`` already follows for its own corpus."""
    mapping = {}
    for _ in range(judge.MIN_PER_CLASS):
        mapping[_labelled(svc, "accept")] = _vector(1.0)
        mapping[_labelled(svc, "reject")] = _vector(-1.0)
    good = next(iter(mapping))
    pruned = _labelled(svc, "accept")
    (svc.job_dir(pruned) / "reference.png").unlink()
    _fake_embeddings(monkeypatch, mapping)
    svc_judge.train(svc, "blank")

    scores = svc_judge.score_jobs(svc, [good, pruned], "blank")

    assert scores[pruned] is None
    assert isinstance(scores[good], float)


def test_the_stage_refusal_names_the_control_it_is_about():
    """SVC-04: this was the one service-layer refusal outside the
    ``service.errors`` hierarchy.

    The contract is what lets the UI put a message beside the control it
    concerns -- a bare ``ValueError`` escaping the boundary is a 500 rather
    than a highlighted select.
    """
    from warlock.service import judge as svc_judge

    with pytest.raises(Invalid) as caught:
        svc_judge._check_stage("mesh")
    assert caught.value.field == "stage"

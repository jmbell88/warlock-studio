"""The verdict and sweep tables: latest-wins, sources coexisting, the one
property the denormalized vector exists for -- surviving the job it names -- and
the grade scale that replaced the binary mesh verdict."""

from __future__ import annotations

import pytest

from warlock import db as db_mod
from warlock import vectors
from warlock.service import findings as svc_findings
from warlock.service import jobs as svc_jobs
from warlock.service import verdicts as svc_verdicts
from warlock.service.errors import Invalid, NotFound


def _model_job(store, **kwargs):
    return store.create("image", "a chest", {"lora_weight": 0.9}, stage="model", **kwargs)


def test_latest_verdict_per_job_and_source_wins(store):
    job_id = _model_job(store)
    store.add_verdict(job_id, source="human", verdict="reject", reasons=["holes"], vector={})
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})

    latest = store.latest_verdicts()
    assert len(latest) == 1
    assert latest[0]["verdict"] == "accept"


def test_a_human_and_an_ai_verdict_sit_side_by_side(store):
    job_id = _model_job(store)
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})
    store.add_verdict(job_id, source="ai:demo", verdict="reject", reasons=["broken"], vector={})

    latest = {r["source"]: r for r in store.latest_verdicts()}
    assert latest["human"]["verdict"] == "accept"
    assert latest["ai:demo"]["verdict"] == "reject"

    for_job = store.verdicts_for([job_id])
    assert set(for_job) == {(job_id, "human"), (job_id, "ai:demo")}
    only_human = store.verdicts_for([job_id], source="human")
    assert set(only_human) == {(job_id, "human")}


def test_a_verdicts_vector_outlives_the_job_row(store):
    job_id = _model_job(store)
    store.add_verdict(
        job_id, source="human", verdict="accept", reasons=[], vector={"lora_weight": 0.9}
    )
    store.delete(job_id)

    # This is the whole reason the vector is denormalized: prune_jobs deletes
    # rows, and the corpus has to survive it.
    assert store.get(job_id) is None
    latest = store.latest_verdicts()
    assert latest[0]["vector"] == {"lora_weight": 0.9}


def test_unverdicted_models_skips_sweep_units_and_judged_jobs(store):
    plain = _model_job(store)
    judged = _model_job(store)
    unit = _model_job(store, sweep_id="deadbeefcafe", sweep_unit="lora_weight=0.6 s42")
    store.create("text", "a picture", {}, stage="reference")
    store.add_verdict(judged, source="human", verdict="accept", reasons=[], vector={})
    for job_id in (plain, judged, unit):
        store.set_status(job_id, "done")

    ids = [job["id"] for job in store.unverdicted_models()]
    assert ids == [plain]


def test_unverdicted_models_is_per_source(store):
    job_id = _model_job(store)
    store.set_status(job_id, "done")
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})

    assert [j["id"] for j in store.unverdicted_models(source="human")] == []
    assert [j["id"] for j in store.unverdicted_models(source="ai:demo")] == [job_id]


def test_sweep_rows_carry_their_unit_counts(store):
    sweep_id = store.create_sweep("lora weight", "a chest", {"axes": []})
    running = store.create("text", "a chest", {}, sweep_id=sweep_id, sweep_unit="baseline s1")
    finished = store.create("text", "a chest", {}, sweep_id=sweep_id, sweep_unit="baseline s2")
    store.set_status(finished, "done")
    store.create("text", "unrelated", {})

    listed = store.list_sweeps()
    assert len(listed) == 1
    assert listed[0]["id"] == sweep_id
    assert listed[0]["label"] == "lora weight"
    assert listed[0]["spec"] == {"axes": []}
    assert (listed[0]["units"], listed[0]["done"], listed[0]["todo"]) == (2, 1, 1)
    assert [j["id"] for j in store.sweep_jobs(sweep_id)] == [running, finished]


def test_deleting_a_sweep_leaves_its_jobs_to_the_caller(store):
    sweep_id = store.create_sweep("s", "p", {})
    job_id = store.create("text", "a chest", {}, sweep_id=sweep_id, sweep_unit="u")
    store.delete_sweep(sweep_id)

    assert store.list_sweeps() == []
    assert store.get(job_id) is not None


def test_next_queued_breaks_created_at_ties_on_id(store):
    """A sweep submits N rows in one loop, so time.time() genuinely ties."""
    ids = sorted(store.create("text", "a chest", {}, f"{i:012x}") for i in range(5))
    for row in store.list(10):
        store._conn.execute(
            "UPDATE jobs SET created_at = 100.0 WHERE id = ?", (row["id"],)
        )
    store._conn.commit()

    assert store.next_queued()["id"] == ids[0]


def test_record_verdict_snapshots_the_jobs_vector(svc):
    job_id = svc.store.create(
        "image", "a chest", {"lora_weight": 0.9, "seed": 12, "composed_prompt": "x"},
        stage="model", status="done",
    )
    svc_verdicts.record_verdict(svc, job_id, grade=-3, reasons=["holes"])

    row = svc.store.latest_verdicts()[0]
    assert row["verdict"] == "reject"
    assert row["grade"] == -3
    assert row["reasons"] == ["holes"]
    # The allowlist, not "everything in params": no seed, no derived value.
    assert row["vector"] == {"lora_weight": 0.9, "stage": "model"}


def test_record_verdict_refuses_unknown_tags_and_jobs(svc):
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    with pytest.raises(Invalid):
        svc_verdicts.record_verdict(svc, job_id, grade=-3, reasons=["ugly"])
    with pytest.raises(NotFound):
        svc_verdicts.record_verdict(svc, "ffffffffffff", grade=3)
    assert svc.store.latest_verdicts() == []


# --- the grade scale --------------------------------------------------------


def test_the_two_tag_vocabularies_are_disjoint_and_the_bad_spellings_are_frozen():
    """One namespace is only sound while polarity is recoverable from the tag.

    And the bad five are the pre-migration ``REASONS`` tuple verbatim: the stored
    corpus carries those exact strings and there is no alias table, so a rename
    would split the evidence rather than migrate it.
    """
    assert set(vectors.GOOD_TAGS).isdisjoint(vectors.BAD_TAGS)
    assert vectors.BAD_TAGS == ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")
    assert vectors.TAGS == vectors.GOOD_TAGS + vectors.BAD_TAGS
    # Five per polarity is what makes Ctrl+1..5 / Shift+1..5 map positionally.
    assert len(vectors.GOOD_TAGS) == len(vectors.BAD_TAGS) == 5
    assert len(set(vectors.TAGS)) == len(vectors.TAGS)


def test_the_backfill_value_and_the_usable_cut_are_one_decision():
    """A cut above the backfill would silently demote every row migration 10
    wrote; a cut below it would claim a grade nobody recorded was usable."""
    assert vectors.BINARY_GRADES == {"accept": 3, "reject": -3}
    assert vectors.BINARY_GRADES["accept"] == vectors.USABLE_GRADE
    assert vectors.BINARY_GRADES["reject"] == -vectors.USABLE_GRADE
    for word, grade in vectors.BINARY_GRADES.items():
        assert vectors.verdict_for_grade(grade) == word
    assert (vectors.GRADE_MIN, vectors.GRADE_MAX) == (-5, 5)


def test_the_usable_cut_lives_in_exactly_one_place():
    assert vectors.verdict_for_grade(vectors.USABLE_GRADE) == "accept"
    assert vectors.verdict_for_grade(vectors.USABLE_GRADE - 1) == "reject"
    assert vectors.verdict_for_grade(vectors.GRADE_MAX) == "accept"
    assert vectors.verdict_for_grade(vectors.GRADE_MIN) == "reject"
    assert vectors.verdict_for_grade(0) == "reject"


def test_migration_10s_literals_still_equal_the_backfill_constants():
    """The migration hard-codes +-3 on purpose -- a shipped migration must mean
    the same thing forever, so interpolating a constant somebody may retune would
    rewrite history. This is what catches the divergence that permits."""
    statements = db_mod.MIGRATIONS[9]
    assert "ALTER TABLE verdicts ADD COLUMN grade INTEGER" in statements[0]
    for word, grade in vectors.BINARY_GRADES.items():
        assert any(
            f"SET grade = {grade}" in stmt and f"verdict = '{word}'" in stmt
            for stmt in statements
        ), word


def test_record_verdict_refuses_a_grade_outside_the_scale_or_a_bool(svc):
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    for bad in (6, -6, 100, None, "3", 3.0, True, False):
        with pytest.raises(Invalid) as excinfo:
            svc_verdicts.record_verdict(svc, job_id, grade=bad)
        assert excinfo.value.field == "grade"
    assert svc.store.latest_verdicts() == []


def test_a_mesh_verdict_refuses_a_caller_supplied_verdict(svc):
    """One door in: the derived column can never disagree with its grade."""
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    with pytest.raises(Invalid) as excinfo:
        svc_verdicts.record_verdict(svc, job_id, verdict="accept", grade=-5)
    assert excinfo.value.field == "verdict"
    with pytest.raises(Invalid):
        svc_verdicts.record_verdict(svc, job_id, verdict="accept")
    assert svc.store.latest_verdicts() == []


def test_the_whole_scale_records_and_derives_its_verdict(svc):
    for grade in range(vectors.GRADE_MIN, vectors.GRADE_MAX + 1):
        job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
        result = svc_verdicts.record_verdict(svc, job_id, grade=grade)
        assert result["grade"] == grade
        assert result["verdict"] == vectors.verdict_for_grade(grade)
        row = {r["job_id"]: r for r in svc.store.latest_verdicts()}[job_id]
        assert (row["grade"], row["verdict"]) == (grade, result["verdict"])


def test_a_good_tag_on_a_negative_grade_is_legal(svc):
    """Tags stopped being *reasons a reviewer rejected*. A mesh can have a clean
    shape and be unusable, and that is the pair of facts worth recording."""
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    svc_verdicts.record_verdict(svc, job_id, grade=-4, reasons=["clean-shape", "holes"])

    row = svc.store.latest_verdicts()[0]
    assert row["reasons"] == ["clean-shape", "holes"]
    assert row["grade"] == -4


def test_an_image_label_stays_binary_and_keeps_its_grade_null(svc, tmp_path):
    job_id = svc.store.create("text", "a chest", {}, stage="reference", status="done")
    (svc.job_dir(job_id)).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "reference.png").write_bytes(b"x")

    result = svc_verdicts.record_verdict(svc, job_id, verdict="accept", stage="reference")
    assert result["grade"] is None
    assert svc.store.latest_verdicts()[0]["grade"] is None

    with pytest.raises(Invalid) as excinfo:
        svc_verdicts.record_verdict(svc, job_id, grade=4, stage="blank")
    assert excinfo.value.field == "grade"


def test_record_verdict_snapshots_the_sweep_context(svc):
    """The matched-pair columns are denormalized at record time -- once the
    sweep's job rows are deleted, the verdict row is the only place the
    pairing context (sweep, seed, prompt) survives."""
    from warlock import vectors

    unit = svc.store.create(
        "image", "a chest", {"lora_weight": 0.9, "seed": 42},
        stage="model", status="done",
        sweep_id="deadbeefcafe", sweep_unit="lora_weight=0.9 s42",
    )
    plain = svc.store.create(
        "image", "another chest", {"seed": 7}, stage="model", status="done"
    )
    svc_verdicts.record_verdict(svc, unit, grade=3)
    svc_verdicts.record_verdict(svc, plain, grade=3)
    svc.store.delete(unit)

    rows = {r["job_id"]: r for r in svc.store.latest_verdicts()}
    assert rows[unit]["sweep_id"] == "deadbeefcafe"
    assert rows[unit]["sweep_unit"] == "lora_weight=0.9 s42"
    assert rows[unit]["seed"] == 42
    assert rows[unit]["prompt_hash"] == vectors.prompt_hash("a chest")
    # An ordinary job records the absence honestly rather than inventing one.
    assert rows[plain]["sweep_id"] is None
    assert rows[plain]["sweep_unit"] == ""
    assert rows[plain]["seed"] == 7
    assert rows[plain]["prompt_hash"] == vectors.prompt_hash("another chest")


def test_record_verdict_tolerates_a_job_with_no_seed(svc):
    job_id = svc.store.create("image", None, {}, stage="model", status="done")
    svc_verdicts.record_verdict(svc, job_id, grade=3)

    row = svc.store.latest_verdicts()[0]
    assert row["seed"] is None
    assert row["prompt_hash"] == ""


def test_a_verdict_on_a_pruned_job_still_feeds_the_findings(svc):
    job_id = svc.store.create(
        "image", "a chest", {"lora_weight": 0.9}, stage="model", status="done"
    )
    svc_verdicts.record_verdict(svc, job_id, grade=3)
    svc_jobs.prune_jobs(svc, keep=0)

    doc = svc_findings.aggregate(svc.store)
    assert doc["params"]["lora_weight"]["0.9"]["accepts"] == 1

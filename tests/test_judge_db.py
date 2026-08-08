"""The judge's half of the data model: ``verdicts.stage`` and the labelling query.

`TODO.md` §7. Two additions, and both of them are shaped by things that have
already gone wrong once.

**``verdicts.stage`` has three values, not two, and the third is the point.**
The instinct is a two-value column mirroring ``jobs.stage`` (which is already
exactly ``'reference' | 'model'``) and recovering the intent by joining back to
the job. That breaks: ``prune_jobs`` deletes job rows, and the sole reason
``verdicts.vector`` is denormalized is that the corpus must outlive the assets it
was learned from. A label whose meaning depends on a row that no longer exists is
uninterpretable the moment it matters. The three values are the two *intents* an
image can be judged under -- ``reference`` (is this a good 2D asset?) and
``blank`` (will this reconstruct?) -- plus ``model`` for a mesh. One image, two
opposed objectives: ``baseline s23`` is a better 2D asset than a flat T-pose on
grey and a worse blank, and a single probe trained across both label sets learns
the average of two opposed things and is useless for each.

**``unlabelled_references`` cannot be ``unverdicted_models`` with a parameter.**
That query filters ``status = 'done' AND stage = 'model' AND sweep_id IS NULL``,
which excludes the two things a labelling pass most needs: errored jobs (a
reference refused for multi-object is the most informative negative available)
and sweep units (the entire corpus this work is built on).
"""

from __future__ import annotations

import sqlite3

from warlock.db import MIGRATIONS, JobStore


def _reference(store, **kwargs):
    return store.create("text", "a rogue", {}, stage="reference", **kwargs)


def _mesh(store, **kwargs):
    return store.create("image", "a chest", {}, stage="model", **kwargs)


# --- the column --------------------------------------------------------------


def test_a_verdict_records_which_question_it_answers(store):
    job_id = _reference(store, status="done")
    store.add_verdict(
        job_id, source="human", verdict="accept", reasons=[], vector={}, stage="blank"
    )

    assert store.latest_verdicts()[0]["stage"] == "blank"


def test_a_verdict_with_no_stage_named_is_about_a_mesh(store):
    """Every verdict to date is, and every existing caller passes nothing."""
    job_id = _mesh(store, status="done")
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})

    assert store.latest_verdicts()[0]["stage"] == "model"


def test_the_same_image_can_be_labelled_under_both_intents_at_once(store):
    """The two questions are not a refinement of each other, so one answer must
    never overwrite the other -- ``latest_verdicts`` keys on (job, source) and
    would do exactly that if intent lived in ``source``."""
    job_id = _reference(store, status="done")
    store.add_verdict(
        job_id, source="human", verdict="accept", reasons=[], vector={}, stage="reference"
    )
    store.add_verdict(
        job_id, source="human", verdict="reject", reasons=[], vector={}, stage="blank"
    )

    by_stage = {r["stage"]: r["verdict"] for r in store.latest_verdicts()}
    assert by_stage == {"reference": "accept", "blank": "reject"}


def test_latest_wins_within_one_stage_and_leaves_the_other_alone(store):
    job_id = _reference(store, status="done")
    store.add_verdict(
        job_id, source="human", verdict="accept", reasons=[], vector={}, stage="blank"
    )
    store.add_verdict(
        job_id, source="human", verdict="reject", reasons=[], vector={}, stage="blank"
    )
    store.add_verdict(
        job_id, source="human", verdict="accept", reasons=[], vector={}, stage="reference"
    )

    by_stage = {r["stage"]: r["verdict"] for r in store.latest_verdicts()}
    assert by_stage == {"blank": "reject", "reference": "accept"}


def test_verdicts_for_can_be_asked_about_one_stage(store):
    """What the labelling grid needs: which of these hundred images have I
    already answered *this* question about."""
    first, second = _reference(store, status="done"), _reference(store, status="done")
    store.add_verdict(
        first, source="human", verdict="accept", reasons=[], vector={}, stage="blank"
    )
    store.add_verdict(
        second, source="human", verdict="accept", reasons=[], vector={}, stage="reference"
    )

    blanks = store.verdicts_for([first, second], source="human", stage="blank")
    assert set(blanks) == {(first, "human")}


# --- the labelling query -----------------------------------------------------


def test_each_intent_asks_about_the_population_it_is_a_question_about(store):
    """§7's probe table, which is not a detail: ``image-as-product`` judges a
    **reference**-stage job's image (that image *is* the deliverable), while
    ``image-as-blank`` judges a **model**-stage job's -- the image trellis
    actually consumed. Pointing one probe at the other's population would train
    it on the average of two opposed objectives."""
    reference = _reference(store, status="done")
    mesh = _mesh(store, status="done")

    assert [r["id"] for r in store.unlabelled_references(stage="reference")] == [reference]
    assert [r["id"] for r in store.unlabelled_references(stage="blank")] == [mesh]


def test_unlabelled_references_includes_what_unverdicted_models_excludes(store):
    """Errored jobs and sweep units, which are most of the corpus.

    The errored one is the sharpest case: a reference refused at the composition
    gate for multi-object is the most informative negative a blank probe can
    have, and it is a *model*-stage job that failed -- exactly what
    ``unverdicted_models``' ``status = 'done'`` throws away.
    """
    sweep_id = store.create_sweep("rogue", "a rogue", {})
    plain = _mesh(store, status="done")
    errored = _mesh(store, status="error")
    unit = _mesh(store, status="done", sweep_id=sweep_id, sweep_unit="baseline s23")

    ids = {row["id"] for row in store.unlabelled_references(stage="blank")}
    assert ids == {plain, errored, unit}
    # And the sibling still excludes two of the three, unchanged.
    assert {r["id"] for r in store.unverdicted_models()} == {plain}


def test_unlabelled_references_skips_what_has_already_been_labelled(store):
    first, second = _mesh(store, status="done"), _mesh(store, status="done")
    store.add_verdict(
        first, source="human", verdict="accept", reasons=[], vector={}, stage="blank"
    )

    ids = {row["id"] for row in store.unlabelled_references(stage="blank")}
    assert ids == {second}


def test_a_mesh_verdict_does_not_count_as_having_labelled_its_blank(store):
    """The same job row genuinely carries both: "this mesh is good" and "this
    image was a good thing to reconstruct" are different claims, and the second
    is the one a blank probe learns from."""
    job_id = _mesh(store, status="done")
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})

    assert [r["id"] for r in store.unlabelled_references(stage="blank")] == [job_id]


def test_unlabelled_references_is_per_source_like_its_sibling(store):
    """So a judge run resumes where it stopped rather than re-scoring the set."""
    job_id = _mesh(store, status="done")
    store.add_verdict(
        job_id, source="ai:demo", verdict="accept", reasons=[], vector={}, stage="blank"
    )

    assert store.unlabelled_references(stage="blank", source="ai:demo") == []
    assert [r["id"] for r in store.unlabelled_references(stage="blank")] == [job_id]


def test_unlabelled_references_never_offers_a_queued_job(store):
    """There is nothing to look at yet."""
    _mesh(store, status="queued")
    _mesh(store, status="running")

    assert store.unlabelled_references(stage="blank") == []


def test_unlabelled_references_is_bounded_and_newest_first(store):
    ids = [_mesh(store, status="done") for _ in range(5)]

    rows = store.unlabelled_references(stage="blank", limit=2)
    assert [r["id"] for r in rows] == ids[::-1][:2]


def test_the_labelling_query_refuses_a_stage_it_does_not_serve(store):
    """``model`` is the mesh question, and ``unverdicted_models`` is its query.
    Answering it here would be a second, subtly different spelling of the same
    listing -- which is exactly the drift ``verdicts.stage`` exists to avoid."""
    import pytest

    with pytest.raises(ValueError):
        store.unlabelled_references(stage="model")


# --- the migration -----------------------------------------------------------


def test_a_v6_db_gains_the_verdict_stage_and_backfills_it(tmp_path):
    """Additive over a v6 database, and every existing row is about a mesh.

    Written as migration 6 in the plan and it is not: Night 2's candidate
    columns took 6 by landing first. Migrations are append-only and never edited
    once shipped, which is why the entry is confirmed against
    ``len(MIGRATIONS)`` rather than against the number in a document.
    """
    path = tmp_path / "v6.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            prompt TEXT, params TEXT NOT NULL DEFAULT '{}', error TEXT,
            created_at REAL NOT NULL, started_at REAL, finished_at REAL,
            stage TEXT NOT NULL DEFAULT 'model', parent_id TEXT,
            name TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',
            favorite INTEGER NOT NULL DEFAULT 0,
            sweep_id TEXT, sweep_unit TEXT NOT NULL DEFAULT '',
            candidate_group TEXT, candidate_index INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE sweeps (
            id TEXT PRIMARY KEY, label TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '', spec TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE TABLE verdicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
            source TEXT NOT NULL, verdict TEXT NOT NULL,
            reasons TEXT NOT NULL DEFAULT '[]', vector TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL, sweep_id TEXT,
            sweep_unit TEXT NOT NULL DEFAULT '', seed INTEGER,
            prompt_hash TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
            sweep_id TEXT, sweep_unit TEXT NOT NULL DEFAULT '', seed INTEGER,
            prompt_hash TEXT NOT NULL DEFAULT '', vector TEXT NOT NULL DEFAULT '{}',
            metrics TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO verdicts (job_id, source, verdict, reasons, vector, created_at)"
        " VALUES ('aaaaaaaaaaaa', 'human', 'accept', '[]', '{\"lora_weight\": 0.9}', 1.0)"
    )
    conn.execute("PRAGMA user_version = 6")
    conn.commit()
    conn.close()

    store = JobStore(path)
    try:
        record = store.latest_verdicts()[0]
        assert record["vector"] == {"lora_weight": 0.9}
        assert record["stage"] == "model"
    finally:
        store.close()

    fresh_path = tmp_path / "fresh.sqlite"
    fresh = JobStore(fresh_path)
    fresh.close()

    def shape(p):
        c = sqlite3.connect(p)
        info = {
            table: c.execute(f"PRAGMA table_info({table})").fetchall()
            for table in ("jobs", "sweeps", "verdicts", "observations")
        }
        version = c.execute("PRAGMA user_version").fetchone()[0]
        c.close()
        return info, version

    migrated, fresh_shape = shape(path), shape(fresh_path)
    assert migrated == fresh_shape
    assert migrated[1] == len(MIGRATIONS)

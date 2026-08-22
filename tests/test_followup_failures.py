from __future__ import annotations

from warlock import followups
from warlock.db import JobStore
from warlock.service.validation import DERIVED_PARAMS
from warlock.studio.panes import inspector, library


def test_persist_keeps_parent_params_and_each_followup(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    try:
        job_id = store.create("image", "a ranger", {"seed": 7})
        assert followups.persist(store, job_id, "rig", OSError("read-only"))
        assert followups.persist(store, job_id, "charsheet", "rig row was not queued")

        params = store.get(job_id)["params"]
        assert params["seed"] == 7
        assert set(params[followups.PARAM_KEY]) == {"rig", "charsheet"}
        assert params[followups.PARAM_KEY]["rig"]["error_type"] == "OSError"
    finally:
        store.close()


def test_records_rejects_malformed_entries_and_bounds_messages():
    record = followups.failure_record("rig", "x" * 900, recorded_at=12.0)
    params = {
        followups.PARAM_KEY: {
            "rig": record,
            "bad": "not an object",
            "empty": {"message": ""},
        }
    }
    assert len(record["message"]) == followups.MAX_MESSAGE
    assert followups.records(params) == [
        {
            "kind": "rig",
            "label": "Automatic rig",
            "message": "x" * followups.MAX_MESSAGE,
            "error_type": "Unavailable",
            "recorded_at": 12.0,
        }
    ]


def test_studio_wording_names_the_missing_followup_and_reason():
    job = {
        "params": {
            followups.PARAM_KEY: {
                "sprite_synthesis": followups.failure_record(
                    "sprite_synthesis", RuntimeError("database is locked")
                )
            }
        }
    }
    assert inspector.followup_failure_lines(job) == [
        ("Sprite sheet was not queued.", "database is locked")
    ]
    assert library.followup_failure_tooltip(job) == (
        "Sprite sheet was not queued: database is locked"
    )


def test_a_rerun_cannot_inherit_an_earlier_followup_failure():
    assert followups.PARAM_KEY in DERIVED_PARAMS

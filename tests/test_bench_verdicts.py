"""Verdict storage: append-only, torn-line-tolerant, latest-per-(unit, source)
wins -- exactly the idioms ``runner.append_item``/``read_items`` already
established for ``items.jsonl``, applied to a second file so a human's or an
AI judge's Accept/Reject never contends with the run's own record of what
happened.
"""

from __future__ import annotations

import json

import pytest

from warlock.bench import verdicts as verdicts_mod


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return d


def test_append_writes_a_record_with_a_stamped_created_at(run_dir):
    verdicts_mod.append_verdict(
        run_dir, unit="prop-01--s42", source="human", verdict="accept",
        param=None, value=None,
    )
    records = verdicts_mod.read_verdicts(run_dir)
    assert len(records) == 1
    record = records[0]
    assert record["unit"] == "prop-01--s42"
    assert record["source"] == "human"
    assert record["verdict"] == "accept"
    assert "created_at" in record and record["created_at"]


def test_an_unknown_verdict_is_refused(run_dir):
    with pytest.raises(ValueError):
        verdicts_mod.append_verdict(
            run_dir, unit="a--s1", source="human", verdict="maybe",
            param=None, value=None,
        )


def test_an_unknown_reason_is_refused_even_on_accept(run_dir):
    with pytest.raises(ValueError):
        verdicts_mod.append_verdict(
            run_dir, unit="a--s1", source="human", verdict="accept",
            reasons=("not-a-real-reason",), param=None, value=None,
        )


def test_a_known_reason_is_accepted_on_reject(run_dir):
    reason = next(iter(verdicts_mod.REASONS))
    verdicts_mod.append_verdict(
        run_dir, unit="a--s1", source="human", verdict="reject",
        reasons=(reason,), param=None, value=None,
    )
    records = verdicts_mod.read_verdicts(run_dir)
    assert records[0]["reasons"] == [reason]


def test_an_empty_source_is_refused(run_dir):
    with pytest.raises(ValueError):
        verdicts_mod.append_verdict(
            run_dir, unit="a--s1", source="", verdict="accept",
            param=None, value=None,
        )


def test_latest_wins_per_unit_and_source(run_dir):
    verdicts_mod.append_verdict(
        run_dir, unit="a--s1", source="human", verdict="reject",
        param=None, value=None,
    )
    verdicts_mod.append_verdict(
        run_dir, unit="a--s1", source="human", verdict="accept",
        param=None, value=None,
    )
    latest = verdicts_mod.latest(run_dir)
    assert latest[("a--s1", "human")]["verdict"] == "accept"


def test_a_torn_last_line_is_skipped(run_dir):
    path = run_dir / verdicts_mod.FILENAME
    path.write_text(
        json.dumps({"unit": "a--s1", "source": "human", "verdict": "accept"})
        + '\n{"unit": "a--s',
        encoding="utf-8",
    )
    records = verdicts_mod.read_verdicts(run_dir)
    assert len(records) == 1
    assert records[0]["unit"] == "a--s1"


def test_human_and_ai_verdicts_coexist_on_the_same_unit(run_dir):
    verdicts_mod.append_verdict(
        run_dir, unit="a--s1", source="human", verdict="accept",
        param=None, value=None,
    )
    verdicts_mod.append_verdict(
        run_dir, unit="a--s1", source="ai:some-model", verdict="reject",
        reasons=(next(iter(verdicts_mod.REASONS)),), param=None, value=None,
    )
    latest = verdicts_mod.latest(run_dir)
    assert latest[("a--s1", "human")]["verdict"] == "accept"
    assert latest[("a--s1", "ai:some-model")]["verdict"] == "reject"
    assert len(latest) == 2


def test_unverdicted_returns_the_subset_with_no_verdict_from_that_source(run_dir):
    verdicts_mod.append_verdict(
        run_dir, unit="a--s1", source="human", verdict="accept",
        param=None, value=None,
    )
    remaining = verdicts_mod.unverdicted(run_dir, ["a--s1", "b--s1", "c--s1"])
    assert remaining == ["b--s1", "c--s1"]


def test_unverdicted_preserves_input_order(run_dir):
    remaining = verdicts_mod.unverdicted(run_dir, ["c--s1", "a--s1", "b--s1"])
    assert remaining == ["c--s1", "a--s1", "b--s1"]


def test_unverdicted_is_scoped_to_source(run_dir):
    verdicts_mod.append_verdict(
        run_dir, unit="a--s1", source="ai:some-model", verdict="accept",
        param=None, value=None,
    )
    # An AI verdict does not count as a human one.
    assert verdicts_mod.unverdicted(run_dir, ["a--s1"], source="human") == ["a--s1"]
    assert verdicts_mod.unverdicted(run_dir, ["a--s1"], source="ai:some-model") == []


def test_append_records_param_and_value(run_dir):
    verdicts_mod.append_verdict(
        run_dir, unit="lora_weight=0.8--s1", source="human", verdict="accept",
        param="lora_weight", value=0.8,
    )
    record = verdicts_mod.read_verdicts(run_dir)[0]
    assert record["param"] == "lora_weight"
    assert record["value"] == 0.8


def test_read_verdicts_on_a_missing_file_is_empty(run_dir):
    assert verdicts_mod.read_verdicts(run_dir) == []
    assert verdicts_mod.latest(run_dir) == {}

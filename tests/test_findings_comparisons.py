"""The conclusive half of findings: Wilson-ranked vectors, matched-pair axis
comparisons from sweep structure, and machine metrics joining both."""

from __future__ import annotations

from warlock.service import findings as svc_findings
from warlock.service import verdicts as svc_verdicts


def _v(store, job_id, vector, verdict="accept", *, sweep=None, seed=42,
       source="human", prompt="p1"):
    store.add_verdict(
        job_id, source=source, verdict=verdict,
        reasons=["holes"] if verdict == "reject" else [],
        vector={**vector, "stage": "model"},
        sweep_id=sweep, sweep_unit="", seed=seed, prompt_hash=prompt,
    )


def _o(store, job_id, vector, metrics, *, sweep="sw1", seed=42, prompt="p1"):
    store.add_observation(
        job_id, sweep_id=sweep, sweep_unit="", seed=seed, prompt_hash=prompt,
        vector={**vector, "stage": "model"}, metrics=metrics,
    )


# --- the lower bound ---------------------------------------------------------


def test_wilson_lower_bound_matches_the_textbook_values():
    assert svc_findings.wilson_low(5, 5) == 0.566
    assert svc_findings.wilson_low(19, 20) == 0.764
    assert svc_findings.wilson_low(6, 8) == 0.409
    assert svc_findings.wilson_low(0, 0) == 0.0
    assert svc_findings.wilson_low(0, 3) == 0.0


def test_vectors_rank_by_the_lower_bound_not_the_raw_rate(store):
    """The stated inversion: a 5/5 must no longer outrank a 19/20."""
    for i in range(5):
        _v(store, f"a{i}", {"lora_weight": 0.6})
    for i in range(19):
        _v(store, f"b{i}", {"lora_weight": 0.9})
    _v(store, "b19", {"lora_weight": 0.9}, "reject")

    doc = svc_findings.aggregate(store)
    assert doc["version"] == 2
    top = doc["vectors"][0]
    assert top["vector"]["lora_weight"] == 0.9
    assert (top["accept_rate"], top["wilson_low"]) == (0.95, 0.764)
    assert doc["vectors"][1]["wilson_low"] == 0.566


def test_the_marginals_carry_the_bound_and_no_dead_metric_keys(store):
    _v(store, "a", {"platform": "pc"})
    entry = svc_findings.aggregate(store)["params"]["platform"]["pc"]
    assert entry["wilson_low"] == svc_findings.wilson_low(1, 1)
    assert "mean_silhouette_iou" not in entry
    assert "mean_dino_cosine" not in entry


# --- matched pairs -----------------------------------------------------------


def test_a_baseline_pairs_with_an_axis_unit_via_the_absent_key(store):
    """The baseline never set lora_weight, so the diff is one *absent* key --
    rendered "unset" -- and no sweep_unit label is ever parsed."""
    _v(store, "base", {}, "reject", sweep="sw1")
    _v(store, "unit", {"lora_weight": 0.6}, "accept", sweep="sw1")

    entry = svc_findings.aggregate(store)["comparisons"]["lora_weight"][0]
    assert (entry["a"], entry["b"]) == ("0.6", "unset")
    assert (entry["pairs"], entry["a_wins"], entry["b_wins"], entry["ties"]) == (1, 1, 0, 0)
    assert (entry["sweeps"], entry["prompts"]) == (1, 1)


def test_two_values_on_one_axis_pair_across_seeds(store):
    """The deliverable sentence: 0.6 beat 0.9 in 2 of 2 matched pairs."""
    for seed in (42, 1337):
        _v(store, f"lo{seed}", {"lora_weight": 0.6}, "accept", sweep="sw1", seed=seed)
        _v(store, f"hi{seed}", {"lora_weight": 0.9}, "reject", sweep="sw1", seed=seed)

    entry = svc_findings.aggregate(store)["comparisons"]["lora_weight"][0]
    assert (entry["a"], entry["b"]) == ("0.6", "0.9")
    assert (entry["pairs"], entry["a_wins"], entry["b_wins"]) == (2, 2, 0)


def test_units_differing_in_two_params_do_not_pair(store):
    _v(store, "a", {"lora_weight": 0.6}, sweep="sw1")
    _v(store, "b", {"lora_weight": 0.9, "platform": "pc"}, sweep="sw1")
    assert svc_findings.aggregate(store)["comparisons"] == {}


def test_pairs_never_straddle_sweeps_seeds_or_sources(store):
    _v(store, "a", {}, sweep="sw1")
    _v(store, "b", {"lora_weight": 0.6}, sweep="sw2")
    _v(store, "c", {"lora_weight": 0.6}, sweep="sw1", seed=1337)
    _v(store, "d", {"lora_weight": 0.6}, sweep="sw1", source="ai:demo")
    _v(store, "e", {"lora_weight": 0.6})  # no sweep context at all
    assert svc_findings.aggregate(store)["comparisons"] == {}


def test_ties_are_counted_but_decide_nothing(store):
    _v(store, "a", {}, "accept", sweep="sw1")
    _v(store, "b", {"lora_weight": 0.6}, "accept", sweep="sw1")
    entry = svc_findings.aggregate(store)["comparisons"]["lora_weight"][0]
    assert (entry["pairs"], entry["a_wins"], entry["b_wins"], entry["ties"]) == (1, 0, 0, 1)


def test_findings_accumulate_across_sweeps_and_count_them(store):
    for i, (sweep, prompt) in enumerate((("sw1", "p1"), ("sw2", "p2"))):
        _v(store, f"base{i}", {}, "reject", sweep=sweep, prompt=prompt)
        _v(store, f"unit{i}", {"lora_weight": 0.6}, "accept", sweep=sweep, prompt=prompt)

    entry = svc_findings.aggregate(store)["comparisons"]["lora_weight"][0]
    assert (entry["pairs"], entry["a_wins"]) == (2, 2)
    assert (entry["sweeps"], entry["prompts"]) == (2, 2)


def test_metric_deltas_come_from_observation_pairs(store):
    """Machine evidence pairs the moment the sweep finishes, before any
    review: an entry can carry deltas with zero human pairs."""
    _o(store, "base", {}, {"hole_worst": 0.10, "watertight": False})
    _o(store, "unit", {"lora_weight": 0.6}, {"hole_worst": 0.06, "watertight": True})

    entry = svc_findings.aggregate(store)["comparisons"]["lora_weight"][0]
    assert entry["pairs"] == 0
    assert entry["deltas"]["hole_worst"] == {"mean": -0.04, "pairs": 1}
    assert entry["deltas"]["watertight"] == {"mean": 1.0, "pairs": 1}


def test_pairs_survive_the_sweep_and_job_deletion(svc):
    """The corpus argument, end to end: the context is on the verdict rows, so
    the flagship comparison outlives the designed cleanup path."""
    base = svc.store.create(
        "image", "a chest", {"seed": 42}, stage="model", status="done",
        sweep_id="sw1", sweep_unit="baseline s42",
    )
    unit = svc.store.create(
        "image", "a chest", {"seed": 42, "lora_weight": 0.6},
        stage="model", status="done",
        sweep_id="sw1", sweep_unit="lora_weight=0.6 s42",
    )
    svc_verdicts.record_verdict(svc, base, verdict="reject", reasons=["holes"])
    svc_verdicts.record_verdict(svc, unit, verdict="accept")
    svc.store.delete(base)
    svc.store.delete(unit)
    svc.store.delete_sweep("sw1")

    entry = svc_findings.aggregate(svc.store)["comparisons"]["lora_weight"][0]
    assert (entry["pairs"], entry["a_wins"], entry["b_wins"]) == (1, 1, 0)


# --- machine metrics in marginals and vectors --------------------------------


def test_observations_feed_the_marginals_before_any_verdict(store):
    _o(store, "j0", {"lora_weight": 0.6},
       {"hole_worst": 0.02, "watertight": True, "triangles": 24000, "ready": True})
    _o(store, "j1", {"lora_weight": 0.6},
       {"hole_worst": 0.04, "watertight": False, "triangles": 24002, "ready": True})

    entry = svc_findings.aggregate(store)["params"]["lora_weight"]["0.6"]
    assert (entry["n"], entry["accepts"], entry["accept_rate"]) == (0, 0, 0.0)
    assert entry["metrics"] == {
        "n": 2,
        "hole_worst": 0.03,
        "watertight_rate": 0.5,
        "triangles": 24001,
        "ready_rate": 1.0,
        "counts": {
            "hole_worst": 2, "watertight_rate": 2, "triangles": 2, "ready_rate": 2
        },
    }


def test_every_mean_carries_the_count_it_actually_rests_on(store):
    """``meshreport.build`` is logged and swallowed, and can also return
    ``status: "invalid"`` with no watertight flag at all -- so a bucket's size
    and a metric's sample size genuinely differ, and the bucket size was being
    printed beside both."""
    _o(store, "j0", {"lora_weight": 0.6},
       {"hole_worst": 0.02, "watertight": True})
    for i in range(1, 4):  # the audit landed; the report did not
        _o(store, f"j{i}", {"lora_weight": 0.6}, {"hole_worst": 0.04})

    metrics = svc_findings.aggregate(store)["params"]["lora_weight"]["0.6"]["metrics"]

    assert metrics["n"] == 4
    assert metrics["counts"] == {"hole_worst": 4, "watertight_rate": 1}


def test_vector_metrics_join_on_the_exact_configuration(store):
    vector = {"lora_weight": 0.6}
    for i in range(2):
        _v(store, f"v{i}", vector)
    _o(store, "v0", vector, {"hole_worst": 0.02})
    _o(store, "other", {"lora_weight": 0.9}, {"hole_worst": 0.5})

    doc = svc_findings.aggregate(store)
    top = next(v for v in doc["vectors"] if v["vector"].get("lora_weight") == 0.6)
    assert top["metrics"] == {"n": 1, "hole_worst": 0.02, "counts": {"hole_worst": 1}}
    # A configuration nobody judged is evidence, not a recommendation: it
    # feeds marginals and comparisons but is not listed as a vector.
    assert all(v["vector"].get("lora_weight") != 0.9 for v in doc["vectors"])


def test_a_payload_that_is_not_json_at_all_costs_its_row_and_no_more(store):
    """The tolerance the next test asserts stopped at the *type* check in
    aggregate -- the decode happens in the store, and a blob that was not JSON
    raised there, taking the whole recompute (and Review's own list) with it.
    """
    _v(store, "good", {"lora_weight": 0.6})
    store.add_verdict(
        "bad", source="human", verdict="accept", reasons=[], vector={},
        sweep_id=None, sweep_unit="", seed=None, prompt_hash="",
    )
    with store._lock:  # a hand-edited DB is the only way to get here
        store._conn.execute("UPDATE verdicts SET vector = 'not json' WHERE job_id = 'bad'")
        store._conn.execute("UPDATE observations SET vector = '{' WHERE 1")
        store._conn.commit()

    doc = svc_findings.aggregate(store)

    assert doc["params"]["lora_weight"]["0.6"]["n"] == 1
    # And the unreadable row is gone rather than ranking as the empty config.
    assert [v["vector"] for v in doc["vectors"]] == [{"lora_weight": 0.6, "stage": "model"}]


def test_aggregate_skips_rows_with_malformed_json_payloads(store):
    """A hand-edited DB or a partial write must cost the row, not the
    refresh: non-dict vectors are dropped, and a non-dict metrics blob can
    reach neither the marginals nor the observation-pair loop."""
    store.add_verdict(
        "j1", source="human", verdict="accept", reasons=[], vector=["broken"],
    )
    # These two would form a matched pair; the broken metrics must not crash it.
    store.add_observation(
        "j2", sweep_id="sw1", sweep_unit="", seed=42, prompt_hash="p",
        vector={"lora_weight": 0.6, "stage": "model"}, metrics="broken",
    )
    store.add_observation(
        "j3", sweep_id="sw1", sweep_unit="", seed=42, prompt_hash="p",
        vector={"stage": "model"}, metrics="broken",
    )

    doc = svc_findings.aggregate(store)
    assert doc["params"] == {}
    assert doc["vectors"] == []
    assert doc["comparisons"] == {}

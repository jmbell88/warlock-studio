"""Canonicalizing a config vector, aggregating verdicts into findings, and the
compatibility contract with the readers that predate this module."""

from __future__ import annotations

import numpy as np

from warlock.bench import findings as bench_findings
from warlock.service import findings as svc_findings
from warlock.service import verdicts as svc_verdicts
from warlock.service.validation import DERIVED_PARAMS


def test_the_vector_is_the_allowlist_and_nothing_else():
    job = {
        "stage": "model",
        "params": {
            "lora_weight": 0.9,
            "platform": "pc",
            # Seeds are the thing a repeat is for.
            "seed": 12, "reference_seed": 12, "mesh_seed": 99,
            # Derived measurements of one run's artifacts.
            "composed_prompt": "a chest, fantasy", "mesh_report": {"triangles": 10},
            # Provenance, not config.
            "hand_edited": True, "built": True, "rerun_of": "aaaaaaaaaaaa",
            # Unset fields the forms send for every untouched select.
            "genre": "", "mood": None,
        },
    }
    assert svc_findings.config_vector(job) == {
        "lora_weight": 0.9,
        "platform": "pc",
        "stage": "model",
    }


def test_no_derived_param_can_ever_reach_a_vector():
    """A structural assertion rather than a sampled one: adding a derived value
    to queue.py must not silently widen what a verdict is filed against."""
    assert set(svc_findings.VECTOR_PARAMS) & set(DERIVED_PARAMS) == set()


def test_a_float32_slider_value_lands_in_the_same_bucket_as_the_literal():
    """imgui hands back the float32 rounding of 0.6; a spec file writes 0.6."""
    slider = float(np.float32(0.6))
    assert slider != 0.6
    from_slider = svc_findings.config_vector({"params": {"lora_weight": slider}})
    from_literal = svc_findings.config_vector({"params": {"lora_weight": 0.6}})
    assert from_slider == from_literal
    assert svc_findings.vector_key(from_slider) == svc_findings.vector_key(from_literal)


def test_the_key_does_not_depend_on_dict_order():
    a = {"platform": "pc", "lora_weight": 0.9, "stage": "model"}
    b = {"stage": "model", "lora_weight": 0.9, "platform": "pc"}
    assert svc_findings.vector_key(a) == svc_findings.vector_key(b)


# --- aggregation -------------------------------------------------------------


def _judged(svc, verdict, reasons=(), source="human", **params):
    job_id = svc.store.create("image", "a chest", params, stage="model", status="done")
    svc_verdicts.record_verdict(
        svc, job_id, verdict=verdict, reasons=reasons, source=source
    )
    return job_id


def test_a_verdict_credits_every_param_in_its_vector(svc):
    """The documented semantics change: marginals are confounded full
    marginals now, which is what lets an ordinary asset's verdict feed the
    hints at all."""
    _judged(svc, "accept", lora_weight=0.9, platform="pc")
    _judged(svc, "reject", ("holes",), lora_weight=0.9, platform="mobile")

    doc = svc_findings.aggregate(svc.store)
    assert doc["params"]["lora_weight"]["0.9"] == {
        "n": 2,
        "accepts": 1,
        "accept_rate": 0.5,
        "wilson_low": svc_findings.wilson_low(1, 2),
        "sources": {"human": {"accept": 1, "reject": 1}},
        "top_reasons": [["holes", 1]],
    }
    assert doc["params"]["platform"]["pc"]["accepts"] == 1
    assert doc["params"]["platform"]["mobile"]["accepts"] == 0
    # ``stage`` is in the vector but is not a knob anyone sets on a form.
    assert "stage" not in doc["params"]


def _judged_about(svc, prompt, verdict, source="human", **params):
    job_id = svc.store.create("image", prompt, params, stage="model", status="done")
    svc_verdicts.record_verdict(svc, job_id, verdict=verdict, reasons=(), source=source)
    return job_id


def test_the_marginals_are_also_broken_out_per_subject(svc):
    """TODO item 4. Confounding across *settings* is documented and accepted --
    the price of letting daily use feed the hints. Confounding across
    *subjects* is not the same bargain: what makes a good wooden crate says
    very little about what makes a good character.

    Demonstrated rather than predicted, too. The regenerated findings.json of
    2026-08-07 reported ``bg_removal: auto`` at n=84, but only 80 of those were
    rogue-sweep units -- the other four were old wood-prompt rejects, pooled
    into a marginal about a character.
    """
    from warlock import vectors

    _judged_about(svc, "a wooden crate", "reject", platform="pc")
    _judged_about(svc, "a wooden crate", "reject", platform="pc")
    _judged_about(svc, "a snes rogue", "accept", platform="pc")

    doc = svc_findings.aggregate(svc.store)
    # The global marginal is unchanged: pooled, as it always was.
    assert doc["params"]["platform"]["pc"]["n"] == 3
    assert doc["params"]["platform"]["pc"]["accepts"] == 1

    rogue = doc["prompts"][vectors.prompt_hash("a snes rogue")]["params"]["platform"]["pc"]
    crate = doc["prompts"][vectors.prompt_hash("a wooden crate")]["params"]["platform"]["pc"]
    assert (rogue["n"], rogue["accepts"]) == (1, 1)
    assert (crate["n"], crate["accepts"]) == (2, 0)
    # Everything ``bench.findings.hint`` reads, and nothing else: a scoped
    # bucket exists per distinct prompt, so the two fields only the Review
    # pane's whole-vector view uses would be paid for once per subject.
    assert set(rogue) == {"n", "accepts", "accept_rate", "wilson_low"}


def test_a_verdict_with_no_prompt_joins_no_subject_scope(svc):
    """``prompt_hash`` is "" for a job with no prompt -- an upload, a painted
    reference. That is an absence, not a subject, and a bucket keyed on it
    would pool every unrelated one of them into a single fake subject."""
    _judged_about(svc, None, "accept", platform="pc")
    doc = svc_findings.aggregate(svc.store)
    assert doc["params"]["platform"]["pc"]["n"] == 1
    assert doc["prompts"] == {}


def test_the_per_subject_scope_carries_machine_evidence_too(svc):
    """An observation is scoped for the same reason a verdict is, and it is the
    tier that fires without anyone reviewing anything."""
    from warlock import vectors

    for i in range(5):
        svc.store.add_observation(
            f"{i:012x}",
            sweep_id=None,
            sweep_unit="",
            seed=1,
            prompt_hash=vectors.prompt_hash("a snes rogue"),
            vector={"base_model": "sdxl_cfg", "stage": "model"},
            metrics={"refused": 1.0},
        )

    doc = svc_findings.aggregate(svc.store)
    scoped = doc["prompts"][vectors.prompt_hash("a snes rogue")]
    assert scoped["params"]["base_model"]["sdxl_cfg"]["metrics"]["refused_rate"] == 1.0


def test_vectors_are_ranked_and_carry_their_jobs(svc):
    good = [_judged(svc, "accept", lora_weight=0.6) for _ in range(2)]
    _judged(svc, "reject", ("bad-shape",), lora_weight=1.2)

    doc = svc_findings.aggregate(svc.store)
    top = doc["vectors"][0]
    assert top["vector"] == {"lora_weight": 0.6, "stage": "model"}
    assert (top["n"], top["accepts"], top["accept_rate"]) == (2, 2, 1.0)
    assert sorted(top["jobs"]) == sorted(good)
    assert doc["vectors"][-1]["accept_rate"] == 0.0
    # Two verdicts is under the preset threshold.
    assert svc_findings.presets(doc) == []
    assert len(svc_findings.presets(doc, min_n=2)) == 1


def test_an_ai_source_sits_beside_the_human_one(svc):
    job_id = svc.store.create("image", "a chest", {"platform": "pc"},
                              stage="model", status="done")
    svc_verdicts.record_verdict(svc, job_id, verdict="accept")
    svc_verdicts.record_verdict(svc, job_id, verdict="reject", source="ai:demo")

    entry = svc_findings.aggregate(svc.store)["params"]["platform"]["pc"]
    assert entry["n"] == 2
    assert entry["sources"] == {
        "human": {"accept": 1, "reject": 0},
        "ai:demo": {"accept": 0, "reject": 1},
    }


# --- the compatibility contract ----------------------------------------------


def test_the_written_file_is_what_the_existing_readers_expect(svc):
    """``bench.findings.load``/``hint`` and both settings panes are unchanged;
    the doc this writes has to keep working with them."""
    for _ in range(6):
        _judged(svc, "accept", lora_weight=0.9)
    _judged(svc, "reject", ("holes",), lora_weight=0.9)
    _judged(svc, "accept", lora_weight=0.6)

    path = svc_findings.refresh(svc)
    assert path == svc.config.bench_dir / "findings.json"

    doc = bench_findings.load(path)
    assert bench_findings.hint(doc, "lora_weight", 0.9) == "accept 6/7 (49%+)"
    # A float32 slider value still finds the bucket.
    assert (
        bench_findings.hint(doc, "lora_weight", float(np.float32(0.9)))
        == "accept 6/7 (49%+)"
    )
    # A thin bucket is noise, not a finding.
    assert bench_findings.hint(doc, "lora_weight", 0.6) is None


def test_refresh_is_written_even_with_nothing_recorded(svc):
    path = svc_findings.refresh(svc)
    doc = bench_findings.load(path)
    assert doc["params"] == {}
    assert doc["vectors"] == []


def test_the_machine_evidence_reaches_the_file_with_no_verdict_filed(svc):
    """The end-to-end half of the observation corpus. Every other test here
    goes through ``aggregate`` directly, which is exactly how a recompute that
    nothing triggered survived review -- the aggregation was proven and the
    file was never written. ``test_studio_frame`` pins the trigger; this pins
    that a triggered refresh actually puts the measurements on disk.
    """
    for i in range(5):
        svc.store.add_observation(
            f"{i:012x}",
            sweep_id=None,
            sweep_unit="",
            seed=42,
            prompt_hash="p1",
            vector={"platform": "pc", "stage": "model"},
            metrics={"hole_worst": 0.03, "watertight": i > 0},
        )

    path = svc_findings.refresh(svc)
    doc = bench_findings.load(path)

    assert svc.store.latest_verdicts() == [], "no verdict was filed"
    assert doc["params"]["platform"]["pc"]["metrics"]["n"] == 5
    # And a value nobody has reviewed still says something true under the
    # control it belongs to.
    assert bench_findings.hint(doc, "platform", "pc") == "holes 3% · watertight 80% (5 meshes)"


def test_a_refusal_rate_survives_into_the_document(svc):
    """TODO item 3's other half, and the half that decides whether it was worth
    doing: an appended observation is not a finding until something aggregates
    it. ``_metric_summary`` names the keys it averages, so a metric added to
    ``vectors.observation_metrics`` and not to it is written to the DB, read
    back out, and silently dropped on the floor."""
    for i in range(5):
        svc.store.add_observation(
            f"{i:012x}",
            sweep_id=None,
            sweep_unit="",
            seed=42,
            prompt_hash="p1",
            vector={"base_model": "sdxl_cfg", "stage": "model"},
            # Three of five refused, all of them for the same rule.
            metrics={
                "refused": 1.0 if i < 3 else 0.0,
                "refused_multi_object": 1.0 if i < 3 else 0.0,
                "refused_occupancy": 0.0,
            },
        )

    doc = bench_findings.load(svc_findings.refresh(svc))
    metrics = doc["params"]["base_model"]["sdxl_cfg"]["metrics"]
    assert metrics["refused_rate"] == 0.6
    assert metrics["refused_multi_object_rate"] == 0.6
    assert metrics["refused_occupancy_rate"] == 0.0
    assert metrics["counts"]["refused_rate"] == 5


def test_a_refusal_rate_is_shown_where_the_setting_is_chosen(svc):
    """It is a fact about the *reference*, so it belongs under the prompt and
    model controls -- which is exactly where a mesh-only hint said nothing,
    because a refused job never reached a mesh."""
    for i in range(6):
        svc.store.add_observation(
            f"{i:012x}",
            sweep_id=None,
            sweep_unit="",
            seed=1,
            prompt_hash="p1",
            vector={"base_model": "sdxl_cfg", "stage": "model"},
            metrics={"refused": 1.0 if i < 3 else 0.0},
        )

    doc = bench_findings.load(svc_findings.refresh(svc))
    assert bench_findings.hint(doc, "base_model", "sdxl_cfg") == "refused 50% (6 references)"

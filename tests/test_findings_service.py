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
        "sources": {"human": {"accept": 1, "reject": 1}},
        "top_reasons": [["holes", 1]],
        "mean_silhouette_iou": None,
        "mean_dino_cosine": None,
    }
    assert doc["params"]["platform"]["pc"]["accepts"] == 1
    assert doc["params"]["platform"]["mobile"]["accepts"] == 0
    # ``stage`` is in the vector but is not a knob anyone sets on a form.
    assert "stage" not in doc["params"]


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
    assert bench_findings.hint(doc, "lora_weight", 0.9) == "accept 6/7"
    # A float32 slider value still finds the bucket.
    assert bench_findings.hint(doc, "lora_weight", float(np.float32(0.9))) == "accept 6/7"
    # A thin bucket is noise, not a finding.
    assert bench_findings.hint(doc, "lora_weight", 0.6) is None


def test_refresh_is_written_even_with_nothing_recorded(svc):
    path = svc_findings.refresh(svc)
    doc = bench_findings.load(path)
    assert doc["params"] == {}
    assert doc["vectors"] == []
    assert svc_findings.summary_lines(doc) == ["no verdicts recorded yet"]

"""``warlock.vectors`` -- the vector/evidence vocabulary queue.py is allowed
to import, and the extraction of machine measurements from a finished job."""

from __future__ import annotations

from warlock import vectors


def test_observation_metrics_reads_both_measurements():
    params = {
        "mesh_audit": {"worst": 0.034, "mean": 0.011, "faces": 267360, "resolution": 512},
        "mesh_report": {"status": "ready", "watertight": True, "triangles": 24120},
    }
    assert vectors.observation_metrics(params) == {
        "hole_worst": 0.034,
        "hole_mean": 0.011,
        "watertight": True,
        "triangles": 24120,
        "ready": True,
    }


def test_observation_metrics_with_only_an_audit():
    params = {"mesh_audit": {"worst": 0.1, "mean": 0.05}}
    assert vectors.observation_metrics(params) == {"hole_worst": 0.1, "hole_mean": 0.05}


def test_observation_metrics_with_only_a_report():
    params = {"mesh_report": {"status": "review", "watertight": False, "triangles": 9}}
    assert vectors.observation_metrics(params) == {
        "watertight": False,
        "triangles": 9,
        "ready": False,
    }


def test_the_watertight_metric_is_the_welded_one_when_the_report_has_it():
    """``meshreport`` loads unwelded, so the raw flag counts every UV-seam
    split as a boundary edge. The corpus wants the welded answer under the key
    it already uses; a row written before the welded analysis existed still
    contributes its raw flag rather than nothing."""
    welded = {"status": "ready", "watertight": False, "welded_watertight": True}
    assert vectors.observation_metrics({"mesh_report": welded})["watertight"] is True

    legacy = {"status": "ready", "watertight": True}
    assert vectors.observation_metrics({"mesh_report": legacy})["watertight"] is True


def test_observation_metrics_omits_keys_rather_than_guessing():
    """The audit is advisory and can be absent, partial, or -- after a JSON
    round trip of something that went wrong -- the wrong shape entirely. Every
    malformed piece costs its own key and nothing else; garbage never raises,
    because this runs on the worker's completion path."""
    assert vectors.observation_metrics({}) == {}
    assert vectors.observation_metrics({"mesh_audit": "broken", "mesh_report": 7}) == {}
    partial = {
        "mesh_audit": {"worst": "0.1", "mean": 0.05},  # a str worst is dropped
        "mesh_report": {"status": 3, "watertight": "yes", "triangles": 1.5},
    }
    assert vectors.observation_metrics(partial) == {"hole_mean": 0.05}
    # A bool is an int in Python; it must not impersonate a triangle count.
    assert vectors.observation_metrics({"mesh_report": {"triangles": True}}) == {}


def test_a_refused_reference_is_a_measurement_like_any_other():
    """Observations on refusal. The 17 refusals in the 2026-08-07 sweep produced zero
    observations rows, so "this checkpoint draws character sheets 60% of the
    time" -- the single most useful thing that sweep measured about
    ``base_model`` -- lived only in the jobs table and died with
    ``prune_jobs``."""
    params = {"reference_report": {"ok": False, "codes": ["multi_object"]}}
    metrics = vectors.observation_metrics(params)
    assert metrics["refused"] == 1.0
    assert metrics["refused_multi_object"] == 1.0
    assert metrics["refused_occupancy"] == 0.0
    assert metrics["refused_edge"] == 0.0


def test_a_reference_that_passed_the_gate_is_the_denominator():
    """A rate needs both halves. If only refused jobs carried ``refused``, its
    mean over any bucket would be 1.0 -- the reader would show "refused 100%"
    for a checkpoint that almost never is."""
    metrics = vectors.observation_metrics({"reference_report": {"ok": True}})
    assert metrics["refused"] == 0.0
    assert metrics["refused_multi_object"] == 0.0


def test_a_refusal_with_no_codes_still_counts_as_one():
    """A report stored before ``codes`` existed. The aggregate rate survives;
    the per-reason keys are simply absent rather than guessed at, which is the
    rule every other malformed piece follows."""
    metrics = vectors.observation_metrics({"reference_report": {"ok": False}})
    assert metrics["refused"] == 1.0
    assert not any(k.startswith("refused_") for k in metrics)


def test_a_missing_or_malformed_report_contributes_no_refusal_keys():
    assert vectors.observation_metrics({}) == {}
    assert vectors.observation_metrics({"reference_report": "broken"}) == {}
    # ``ok`` absent is not "it passed": nothing is known, so nothing is said.
    assert vectors.observation_metrics({"reference_report": {"codes": []}}) == {}


def test_the_refusal_keys_are_the_gate_s_own_vocabulary():
    """One list, so a rule added to reference.py cannot record a rate under a
    name nothing aggregates."""
    from warlock.pipelines import reference

    metrics = vectors.observation_metrics({"reference_report": {"ok": True}})
    assert set(metrics) == {"refused"} | {
        f"refused_{code}" for code in reference.REFUSAL_CODES
    }


def test_prompt_hash_is_short_stable_and_empty_for_no_prompt():
    a = vectors.prompt_hash("a wooden chest")
    assert a == vectors.prompt_hash("a wooden chest")
    assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)
    assert vectors.prompt_hash("a different chest") != a
    assert vectors.prompt_hash(None) == ""
    assert vectors.prompt_hash("") == ""


def test_the_service_module_re_exports_the_moved_names():
    """Every existing import path (service.verdicts, studio, the tests) keeps
    working; the move exists only so queue.py can import the vocabulary
    without importing ``service``."""
    from warlock.service import findings as svc_findings

    assert svc_findings.config_vector is vectors.config_vector
    assert svc_findings.vector_key is vectors.vector_key
    assert tuple(svc_findings.VECTOR_PARAMS) == tuple(vectors.VECTOR_PARAMS)


def test_importing_vectors_pulls_in_no_service_and_no_torch():
    """queue.py deliberately imports no ``service`` module; the vocabulary it
    needs at completion time must not smuggle one in. Same subprocess shape as
    test_offline's torch-free pins."""
    import subprocess
    import sys

    code = (
        "import warlock.vectors, sys; "
        "bad = [m for m in sys.modules if m.startswith('warlock.service')"
        " or m in ('torch', 'imgui_bundle')]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""

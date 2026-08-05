"""The candidate score: what the reference report already knew, as a number.

Advisory by construction -- nothing rejects a candidate. Its whole job is to
put the most likely one first in a strip of eight.
"""

from __future__ import annotations

from warlock.pipelines import rank, reference


def _report(**kwargs):
    return reference.Report(**kwargs).as_dict()


def test_a_refused_reference_scores_zero():
    assert rank.composition_score(_report(ok=False, reasons=("too small",))) == 0.0


def test_a_clean_reference_at_the_target_occupancy_scores_one():
    assert (
        rank.composition_score(_report(occupancy=reference.DEFAULT_OCCUPANCY, components=1))
        == 1.0
    )


def test_missing_the_target_occupancy_costs_something_but_not_everything():
    near = rank.composition_score(_report(occupancy=0.70, components=1))
    far = rank.composition_score(_report(occupancy=0.20, components=1))
    assert 0.0 < far < near < 1.0


def test_a_second_object_is_penalised():
    one = rank.composition_score(_report(occupancy=0.78, components=1))
    two = rank.composition_score(_report(occupancy=0.78, components=2))
    assert two < one


def test_running_off_the_edge_is_penalised():
    clean = rank.composition_score(_report(occupancy=0.78, components=1))
    cropped = rank.composition_score(
        _report(occupancy=0.78, components=1, touches=("left",))
    )
    assert cropped < clean


def test_warnings_cost_less_than_reasons():
    warned = rank.composition_score(
        _report(occupancy=0.78, components=1, warnings=("close to the edge",))
    )
    clean = rank.composition_score(_report(occupancy=0.78, components=1))
    refused = rank.composition_score(_report(ok=False, reasons=("too small",)))
    assert refused < warned < clean


def test_no_report_at_all_is_a_middling_score_not_a_zero():
    # A job whose measurement failed is unknown, not bad -- scoring it zero
    # would sort it below a candidate that was actually measured and refused.
    assert 0.0 < rank.composition_score(None) < 1.0


def test_the_score_is_the_composition_when_there_is_no_anchor():
    out = rank.score(_report(occupancy=0.78, components=1))
    assert out["score"] == out["composition"] == 1.0
    assert out["anchor"] is None


def test_an_anchor_cosine_moves_the_score_and_is_recorded():
    report = _report(occupancy=0.78, components=1)
    close = rank.score(report, anchor_cosine=0.9)
    far = rank.score(report, anchor_cosine=0.1)
    assert close["score"] > far["score"]
    assert close["anchor"] == 0.9


def test_every_score_stays_inside_zero_and_one():
    for cosine in (-1.0, 0.0, 1.0):
        for report in (None, _report(ok=False), _report(occupancy=0.01, components=5)):
            out = rank.score(report, anchor_cosine=cosine)
            assert 0.0 <= out["score"] <= 1.0

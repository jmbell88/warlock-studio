"""``findings.py`` is the studio-side reader of ``findings.json``: pure
stdlib, importable with neither torch nor imgui, so a generate pane's per-
frame hint lookup costs one ``stat()`` and nothing else on the common path.

The cache is keyed on path and re-reads only when mtime changes -- these
tests pin that the second call is a true no-op (no re-read) and that a
changed mtime is honoured.
"""

from __future__ import annotations

import json

from warlock.bench import findings as findings_mod


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _doc(entries):
    return {"version": 1, "generated": "x", "params": entries}


# --- load ---------------------------------------------------------------------


def test_load_missing_file_returns_none(tmp_path):
    findings_mod._CACHE.clear()
    assert findings_mod.load(tmp_path / "findings.json") is None


def test_load_missing_file_caches_the_miss(tmp_path, monkeypatch):
    findings_mod._CACHE.clear()
    path = tmp_path / "findings.json"
    calls = []
    orig_stat = type(path).stat

    def counting_stat(self, *a, **k):
        calls.append(self)
        return orig_stat(self, *a, **k)

    monkeypatch.setattr(type(path), "stat", counting_stat)

    assert findings_mod.load(path) is None
    assert findings_mod.load(path) is None
    # One stat() per call is fine (that's the contract); what matters is no
    # exception path and no repeated directory scan -- both calls return None.
    assert len(calls) >= 2


def test_load_corrupt_json_returns_none(tmp_path):
    findings_mod._CACHE.clear()
    path = tmp_path / "findings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert findings_mod.load(path) is None


def test_load_corrupt_json_caches_by_mtime_not_none(tmp_path, monkeypatch):
    """A corrupt-but-present file's stat() succeeds every time, so caching
    the miss as (None, None) would never match the real mtime and every call
    would re-read and re-parse -- exactly what the mtime gate exists to
    avoid. The second call on an unchanged corrupt file must not re-read."""
    findings_mod._CACHE.clear()
    path = tmp_path / "findings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    calls = []
    orig_read_text = type(path).read_text

    def counting_read_text(self, *a, **k):
        calls.append(self)
        return orig_read_text(self, *a, **k)

    monkeypatch.setattr(type(path), "read_text", counting_read_text)

    assert findings_mod.load(path) is None
    assert findings_mod.load(path) is None
    assert len(calls) == 1


def test_load_reads_a_valid_file(tmp_path):
    findings_mod._CACHE.clear()
    path = tmp_path / "findings.json"
    doc = _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}})
    _write(path, doc)

    loaded = findings_mod.load(path)

    assert loaded == doc


def test_load_mtime_cache_rereads_a_changed_file(tmp_path):
    findings_mod._CACHE.clear()
    path = tmp_path / "findings.json"
    _write(path, _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}}))
    first = findings_mod.load(path)
    assert first["params"]["lora_weight"]["0.6"]["n"] == 8

    # Bump mtime forward so a coarse filesystem clock still registers a change.
    import os
    import time

    _write(path, _doc({"lora_weight": {"0.6": {"n": 3, "accepts": 1}}}))
    future = time.time() + 5
    os.utime(path, (future, future))

    second = findings_mod.load(path)

    assert second["params"]["lora_weight"]["0.6"]["n"] == 3


def test_load_unchanged_mtime_returns_cached_doc_object(tmp_path):
    findings_mod._CACHE.clear()
    path = tmp_path / "findings.json"
    _write(path, _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}}))

    first = findings_mod.load(path)
    second = findings_mod.load(path)

    assert first is second


# --- hint -----------------------------------------------------------------------


def test_hint_at_threshold_reports_accepts_over_n(tmp_path):
    doc = _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}})

    assert findings_mod.hint(doc, "lora_weight", 0.6) == "accept 6/8"


def test_hint_below_threshold_is_none(tmp_path):
    doc = _doc({"lora_weight": {"0.6": {"n": 4, "accepts": 3}}})

    assert findings_mod.hint(doc, "lora_weight", 0.6) is None


def test_hint_respects_custom_min_n(tmp_path):
    doc = _doc({"lora_weight": {"0.6": {"n": 4, "accepts": 3}}})

    assert findings_mod.hint(doc, "lora_weight", 0.6, min_n=4) == "accept 3/4"


def test_hint_value_is_str_keyed(tmp_path):
    doc = _doc({"platform": {"desktop": {"n": 10, "accepts": 9}}})

    assert findings_mod.hint(doc, "platform", "desktop") == "accept 9/10"


def test_hint_unknown_param_is_none(tmp_path):
    doc = _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}})

    assert findings_mod.hint(doc, "mystery", 0.6) is None


def test_hint_unknown_value_is_none(tmp_path):
    doc = _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}})

    assert findings_mod.hint(doc, "lora_weight", 1.2) is None


def test_hint_on_none_doc_is_none(tmp_path):
    assert findings_mod.hint(None, "lora_weight", 0.6) is None


def test_hint_missing_accepts_key_does_not_raise(tmp_path):
    """A parseable but malformed findings.json (``n`` present, ``accepts``
    missing) must not crash the frame thread with a KeyError -- ``n`` is
    already read with ``.get``, ``accepts`` must be too."""
    doc = _doc({"lora_weight": {"0.6": {"n": 8}}})

    assert findings_mod.hint(doc, "lora_weight", 0.6) == "accept 0/8"


def _scoped_doc(pooled, per_prompt):
    return {"version": 3, "generated": "x", "params": pooled, "prompts": per_prompt}


def test_hint_prefers_this_subject_and_says_so(tmp_path):
    """Per-subject scoping. A hint that silently mixes subjects is worse than no
    hint,
    because it is trusted -- so when a subject-scoped answer exists it wins,
    and either way the line says which corpus it came from."""
    doc = _scoped_doc(
        {"base_model": {"turbo": {"n": 84, "accepts": 3}}},
        {"abc123": {"params": {"base_model": {"turbo": {"n": 8, "accepts": 6}}}}},
    )

    assert findings_mod.hint(doc, "base_model", "turbo", prompt_hash="abc123") == (
        "accept 6/8 · this subject"
    )


def test_hint_falls_back_to_every_subject_and_says_that_too(tmp_path):
    """A thin subject scope is not a reason to say nothing -- the pooled answer
    is still evidence, and the label is what stops it being mistaken for a
    finding about this prompt."""
    doc = _scoped_doc(
        {"base_model": {"turbo": {"n": 84, "accepts": 3}}},
        {"abc123": {"params": {"base_model": {"turbo": {"n": 2, "accepts": 2}}}}},
    )

    assert findings_mod.hint(doc, "base_model", "turbo", prompt_hash="abc123") == (
        "accept 3/84 · all subjects"
    )
    # A subject with nothing recorded at all takes the same path.
    assert findings_mod.hint(doc, "base_model", "turbo", prompt_hash="nothing") == (
        "accept 3/84 · all subjects"
    )


def test_hint_without_a_subject_is_unlabelled_and_pooled(tmp_path):
    """Every caller that does not know its subject -- and every findings.json
    written before ``prompts`` existed -- behaves exactly as it did. The label
    would be a claim the caller never made."""
    doc = _scoped_doc(
        {"base_model": {"turbo": {"n": 84, "accepts": 3}}},
        {"abc123": {"params": {"base_model": {"turbo": {"n": 8, "accepts": 6}}}}},
    )

    assert findings_mod.hint(doc, "base_model", "turbo") == "accept 3/84"


def test_hint_with_a_subject_on_a_v2_file_still_answers(tmp_path):
    doc = _doc({"base_model": {"turbo": {"n": 84, "accepts": 3}}})

    assert findings_mod.hint(doc, "base_model", "turbo", prompt_hash="abc123") == (
        "accept 3/84 · all subjects"
    )


def test_hint_with_a_malformed_prompts_section_never_raises(tmp_path):
    """The file crossed a disk, and this runs on the frame thread."""
    for broken in ("nonsense", [], {"abc123": 7}, {"abc123": {"params": "no"}}):
        doc = _scoped_doc({"base_model": {"turbo": {"n": 8, "accepts": 6}}}, broken)
        assert findings_mod.hint(doc, "base_model", "turbo", prompt_hash="abc123") == (
            "accept 6/8 · all subjects"
        )


def test_hint_matches_a_float32_slider_value_against_a_string_key(tmp_path):
    """``findings.json`` keys a bucket by ``str()`` of the sweep's JSON
    value ("0.6"), but imgui's float32 ``slider_float`` hands back
    0.6000000238418579 -- the float32 rounding of 0.6, not the float
    ``str(0.6)`` would parse back to. A literal ``str(value)`` lookup misses
    every such slider; ``hint`` must fall back to a rounded float
    comparison."""
    doc = _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}})

    assert findings_mod.hint(doc, "lora_weight", 0.6000000238418579) == "accept 6/8"


# --- hint v2: the bound, and machine evidence -----------------------------------


def test_hint_appends_the_confidence_floor_when_the_writer_provides_it(tmp_path):
    doc = _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6, "wilson_low": 0.409}}})

    assert findings_mod.hint(doc, "lora_weight", 0.6) == "accept 6/8 (41%+)"


def test_hint_falls_back_to_the_bare_count_on_a_v1_file(tmp_path):
    """A findings.json written before the bound existed still hints -- the
    reader never requires what an older writer could not have known."""
    doc = _doc({"lora_weight": {"0.6": {"n": 8, "accepts": 6}}})

    assert findings_mod.hint(doc, "lora_weight", 0.6) == "accept 6/8"


def test_hint_shows_machine_evidence_below_the_verdict_threshold(tmp_path):
    """The inline face of evidence accumulating on every generation: a value
    generated with twenty-one times but reviewed twice still says something
    true."""
    doc = _doc({
        "lora_weight": {"0.6": {
            "n": 2, "accepts": 2,
            "metrics": {"n": 21, "hole_worst": 0.034, "watertight_rate": 0.71},
        }},
    })

    assert (
        findings_mod.hint(doc, "lora_weight", 0.6)
        == "holes 3% \u00b7 watertight 71% (21 meshes)"
    )


def test_a_thin_measurement_is_dropped_rather_than_carried_by_a_fat_one(tmp_path):
    """The regression: ``n`` is the bucket size and every mean was printed
    beside it, so one watertight reading among twenty-one observations was
    advertised as twenty-one. The threshold is per measurement."""
    doc = _doc({
        "lora_weight": {"0.6": {
            "n": 0, "accepts": 0,
            "metrics": {
                "n": 21, "hole_worst": 0.034, "watertight_rate": 1.0,
                "counts": {"hole_worst": 21, "watertight_rate": 1},
            },
        }},
    })

    assert findings_mod.hint(doc, "lora_weight", 0.6) == "holes 3% (21 meshes)"


def test_a_ranked_recipe_states_its_rate_and_its_floor():
    assert findings_mod.vector_line(
        {"n": 20, "accept_rate": 0.8, "wilson_low": 0.607}
    ) == "80% of 20 (61%+)"


def test_a_v1_recipe_states_no_floor_rather_than_a_floor_of_zero():
    """The regression: the pane defaulted a missing bound to 0.0, so every
    recipe in an existing user's Review pane read "(0%+)" -- under a heading
    claiming the Wilson ranking -- until some verdict rewrote the file."""
    assert findings_mod.vector_line({"n": 20, "accept_rate": 0.8}) == "80% of 20"
    assert findings_mod.vector_line({}) == "0% of 0"
    assert findings_mod.vector_line(None) == ""


def test_measurements_of_different_weight_each_say_their_own(tmp_path):
    """Both above the threshold but not equally supported: one trailing count
    would be a claim about whichever the reader assumed it belonged to."""
    line = findings_mod.metrics_line(
        {"n": 21, "hole_worst": 0.034, "watertight_rate": 0.71,
         "counts": {"hole_worst": 21, "watertight_rate": 18}}
    )
    assert line == "holes 3% worst (21 meshes) · watertight 71% (18 meshes)"


def test_hint_prefers_verdicts_once_they_reach_the_threshold(tmp_path):
    doc = _doc({
        "lora_weight": {"0.6": {
            "n": 8, "accepts": 6, "wilson_low": 0.409,
            "metrics": {"n": 21, "hole_worst": 0.034},
        }},
    })

    assert findings_mod.hint(doc, "lora_weight", 0.6) == "accept 6/8 (41%+)"


def test_hint_with_malformed_metrics_never_raises(tmp_path):
    doc = _doc({
        "lora_weight": {
            "0.6": {"n": 1, "accepts": 1, "metrics": {"n": "x", "hole_worst": 0.1}},
            "0.7": {"n": 1, "accepts": 1, "metrics": "broken"},
            "0.8": {"n": 1, "accepts": 1, "metrics": {"n": 21, "hole_worst": "bad"}},
        },
    })

    assert findings_mod.hint(doc, "lora_weight", 0.6) is None
    assert findings_mod.hint(doc, "lora_weight", 0.7) is None
    # A well-formed count with no readable measurement is still nothing to say.
    assert findings_mod.hint(doc, "lora_weight", 0.8) is None


# --- comparison_lines -----------------------------------------------------------


def _comparison_doc(entries):
    return {"version": 2, "generated": "x", "params": {}, "vectors": [],
            "comparisons": entries}


def test_comparison_lines_lead_with_the_winner(tmp_path):
    doc = _comparison_doc({
        "lora_weight": [{
            "a": "0.9", "b": "unset", "pairs": 8, "a_wins": 1, "b_wins": 7,
            "ties": 0, "sweeps": 2, "prompts": 2, "deltas": {},
        }],
    })

    assert findings_mod.comparison_lines(doc) == [
        "lora_weight: unset beat 0.9 in 7 of 8 matched pairs (2 sweeps, 2 prompts)",
    ]


def test_comparison_lines_append_metric_deltas(tmp_path):
    doc = _comparison_doc({
        "lora_weight": [{
            "a": "0.6", "b": "0.9", "pairs": 8, "a_wins": 7, "b_wins": 1,
            "ties": 0, "sweeps": 2, "prompts": 2,
            "deltas": {
                "hole_worst": {"mean": -0.041, "pairs": 12},
                "watertight": {"mean": 0.25, "pairs": 12},
                "triangles": {"mean": -12400.0, "pairs": 2},
            },
        }],
    })

    assert findings_mod.comparison_lines(doc) == [
        "lora_weight: 0.6 beat 0.9 in 7 of 8 matched pairs (2 sweeps, 2 prompts)",
        "    worst-hole -4.1% over 12 paired runs",
        "    watertight +25.0% over 12 paired runs",
        # triangles is below min_pairs and stays quiet
    ]


def test_comparison_lines_report_a_tie_honestly(tmp_path):
    doc = _comparison_doc({
        "platform": [{
            "a": "desktop", "b": "mobile", "pairs": 6, "a_wins": 3, "b_wins": 3,
            "ties": 0, "sweeps": 1, "prompts": 1, "deltas": {},
        }],
    })

    assert findings_mod.comparison_lines(doc) == [
        "platform: desktop vs mobile - tied over 6 matched pairs (1 sweep, 1 prompt)",
    ]


def test_comparison_lines_show_machine_evidence_before_any_review(tmp_path):
    doc = _comparison_doc({
        "trellis_band": [{
            "a": "8", "b": "unset", "pairs": 0, "a_wins": 0, "b_wins": 0,
            "ties": 0, "sweeps": 1, "prompts": 1,
            "deltas": {"hole_worst": {"mean": 0.012, "pairs": 6}},
        }],
    })

    assert findings_mod.comparison_lines(doc) == [
        "trellis_band: 8 vs unset - machine evidence only",
        "    worst-hole +1.2% over 6 paired runs",
    ]


def test_comparison_lines_stay_quiet_below_the_pair_threshold(tmp_path):
    doc = _comparison_doc({
        "lora_weight": [{
            "a": "0.6", "b": "0.9", "pairs": 2, "a_wins": 2, "b_wins": 0,
            "ties": 0, "sweeps": 1, "prompts": 1,
            "deltas": {"hole_worst": {"mean": -0.04, "pairs": 2}},
        }],
    })

    assert findings_mod.comparison_lines(doc) == []
    assert findings_mod.comparison_lines(doc, min_pairs=2) == [
        "lora_weight: 0.6 beat 0.9 in 2 of 2 matched pairs (1 sweep, 1 prompt)",
        "    worst-hole -4.0% over 2 paired runs",
    ]


def test_comparison_lines_tolerate_a_doc_without_the_section(tmp_path):
    assert findings_mod.comparison_lines(None) == []
    assert findings_mod.comparison_lines({"params": {}}) == []


def test_metrics_line_renders_what_a_recipe_measured(tmp_path):
    line = findings_mod.metrics_line(
        {"n": 21, "hole_worst": 0.034, "watertight_rate": 0.71, "triangles": 24120}
    )
    assert line == "holes 3% worst \u00b7 watertight 71% \u00b7 24,120 tri (21 meshes)"


def test_metrics_line_skips_what_was_not_measured(tmp_path):
    assert findings_mod.metrics_line({"n": 3, "hole_worst": 0.1}) == "holes 10% worst (3 meshes)"
    assert findings_mod.metrics_line({"n": 3}) is None
    assert findings_mod.metrics_line(None) is None
    assert findings_mod.metrics_line({"n": "x", "hole_worst": 0.1}) is None


def test_comparison_lines_orient_deltas_to_the_winner(tmp_path):
    """The stored delta is a-minus-b, but the header leads with the winner --
    an unattributed delta line under "0.9 beat 0.6" must describe the winner,
    or the sign silently flips meaning on a lexicographic accident."""
    doc = _comparison_doc({
        "lora_weight": [{
            "a": "0.6", "b": "0.9", "pairs": 8, "a_wins": 1, "b_wins": 7,
            "ties": 0, "sweeps": 2, "prompts": 2,
            "deltas": {"hole_worst": {"mean": 0.041, "pairs": 12}},
        }],
    })

    assert findings_mod.comparison_lines(doc) == [
        "lora_weight: 0.9 beat 0.6 in 7 of 8 matched pairs (2 sweeps, 2 prompts)",
        "    worst-hole -4.1% over 12 paired runs",
    ]


def test_comparison_lines_never_raise_on_malformed_entries(tmp_path):
    """This renders every frame; a hand-corrupted findings.json must cost the
    entry, never the frame thread."""
    doc = _comparison_doc({
        "lora_weight": [
            {"a": "0.6", "b": "0.9", "pairs": "8", "a_wins": 7, "b_wins": 1,
             "ties": 0, "sweeps": 1, "prompts": 1, "deltas": {}},
            {"a": "0.6", "b": "0.9", "pairs": 8, "a_wins": "x", "b_wins": 1,
             "ties": 0, "sweeps": 1, "prompts": 1, "deltas": []},
            "not-a-dict",
        ],
    })

    assert findings_mod.comparison_lines(doc) == [
        # The one salvageable half: entry 2's counts collapse to 0-vs-1.
        "lora_weight: 0.9 beat 0.6 in 1 of 8 matched pairs (1 sweep, 1 prompt)",
    ]

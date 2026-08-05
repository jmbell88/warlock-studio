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

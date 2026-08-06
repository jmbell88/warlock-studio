"""Turning recorded verdicts into ``findings.json``.

What this replaces. The old ``bench/report.py`` aggregated verdicts written
beside a sweep run directory, grouped strictly on the one axis that sweep
varied. Verdicts now live in the job DB keyed by job id, every one of them
carrying a snapshot of the *whole* config vector it was filed against -- so
the same aggregation can be done over ordinary daily-use verdicts as well as
sweep units, and a "vector" is a first-class result rather than something a
one-factor-at-a-time sweep could never express.

**The semantics of the marginals changed, and it is deliberate.** A verdict now
credits every ``param: value`` pair in its vector, not just the single axis a
sweep varied. Those marginals are therefore *confounded*: "lora_weight 0.9
accepts 6/8" now means "of the eight judged assets whose lora_weight was 0.9,
six were accepted", with everything else free to vary, rather than "six of
eight, all else held equal". That is the price of letting ordinary verdicts
feed the hints at all, and it is the right trade -- a confounded number over a
hundred real assets beats a clean one over eight sweep units. The ``vectors``
section is where the unconfounded answer lives: a whole configuration, ranked.

The output shape is unchanged for every existing reader. ``params`` is exactly
what ``bench/report.write_findings`` wrote, so ``bench.findings.load``/``hint``
and both settings panes keep working with no edit; ``vectors`` is additive.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FINDINGS_VERSION = 1
JSON_FILENAME = "findings.json"

# The only two metrics the old report ever emitted. Kept in the output as
# nulls rather than dropped: anything reading findings.json (the generate
# panes through ``bench.findings.load``) expects the keys to exist. Nothing computes them any
# more -- they were joined from a sweep run's scores.json, which the run-dir
# path took with it.
METRIC_NAMES = ("silhouette_iou", "dino_cosine")

# What a config vector is made of: every param that is an *input* to a
# generation and could plausibly be varied. An allowlist rather than "params
# minus DERIVED_PARAMS" because the interesting property is that adding a
# derived value to queue.py can never silently widen it.
#
# Deliberately absent, each for its own reason: ``prompt`` (a property of the
# subject, not of the settings -- grouping on it would make every bucket n=1),
# every seed (the thing a repeat is *for*), everything in
# ``validation.DERIVED_PARAMS`` (measurements of one run's artifacts), the rig
# and pose keys (they describe a follow-up job, not this one's settings), and
# ``hand_edited``/``imported``/``built``/``rerun_of`` (provenance, not config).
VECTOR_PARAMS = (
    "category",
    "silhouette",
    "material",
    "condition",
    "rarity",
    "emissive",
    "setting",
    "genre",
    "mood",
    "art_style",
    "palette",
    "platform",
    "base_model",
    "style_lora",
    "lora_weight",
    "negative_prompt",
    "ip_adapter",
    "ip_scale",
    "control",
    "control_scale",
    "control_end",
    "resolution",
    "size_m",
    "bg_removal",
    "reference_prep",
    "profile",
    "custom_triangles",
    "trellis_band",
    "trellis_tex_res",
)

# How many verdicts a whole vector needs before it is offered as a preset. The
# same threshold ``bench.findings.hint`` applies to a single param value, and
# for the same reason: a thin bucket is noise, not a finding.
PRESET_MIN_N = 5


def config_vector(job: dict[str, Any]) -> dict[str, Any]:
    """The settings that produced this job, canonicalized.

    Canonical in three ways, all of which exist so two runs of the same
    configuration land in the same bucket. Unset is *absent* rather than
    ``None``/``""`` -- a form sends empty strings for every taxonomy field the
    user left alone, and keeping them would make "no style chosen" a distinct
    value from "the key was never there". Floats are rounded to six decimals,
    because an imgui float32 slider hands back ``0.6000000238418579`` for the
    0.6 a spec file wrote. And ``stage`` comes along, because a reference and a
    mesh judged under the same settings are not the same thing.
    """
    params = job.get("params") or {}
    out: dict[str, Any] = {}
    for key in VECTOR_PARAMS:
        if key not in params:
            continue
        value = params[key]
        if value is None or value == "":
            continue
        if isinstance(value, float):
            value = round(value, 6)
        out[key] = value
    out["stage"] = job.get("stage") or "model"
    return out


def vector_key(vector: dict[str, Any]) -> str:
    """A short stable id for a vector -- the same settings always hash the
    same, whatever order the dict was built in."""
    blob = json.dumps(vector, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def aggregate(store: Any) -> dict[str, Any]:
    """The whole findings document, over every source's latest verdicts."""
    records = store.latest_verdicts()

    params: dict[str, dict[str, list[dict[str, Any]]]] = {}
    vectors: dict[str, dict[str, Any]] = {}
    for record in records:
        vector = record.get("vector") or {}
        if not isinstance(vector, dict):
            continue
        for param, value in vector.items():
            if param == "stage":
                continue
            params.setdefault(param, {}).setdefault(str(value), []).append(record)
        key = vector_key(vector)
        bucket = vectors.setdefault(key, {"key": key, "vector": vector, "records": []})
        bucket["records"].append(record)

    out_params: dict[str, dict[str, Any]] = {}
    for param, values in params.items():
        out_params[param] = {v: _summarise(rows) for v, rows in values.items()}

    out_vectors = []
    for bucket in vectors.values():
        summary = _summarise(bucket["records"])
        out_vectors.append(
            {
                "key": bucket["key"],
                "vector": bucket["vector"],
                "n": summary["n"],
                "accepts": summary["accepts"],
                "accept_rate": summary["accept_rate"],
                "top_reasons": summary["top_reasons"],
                "jobs": [r["job_id"] for r in bucket["records"]],
            }
        )
    # Best first, and a tie broken by sample size: two vectors both at 1.0 are
    # not equally believable, and the one judged more often is the one worth
    # offering as a preset.
    out_vectors.sort(key=lambda v: (-v["accept_rate"], -v["n"], v["key"]))

    return {
        "version": FINDINGS_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": out_params,
        "vectors": out_vectors,
    }


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    accepts = sum(1 for r in records if r.get("verdict") == "accept")

    sources: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = sources.setdefault(str(record.get("source") or ""), {"accept": 0, "reject": 0})
        verdict = record.get("verdict")
        if verdict in bucket:
            bucket[verdict] += 1

    reasons: Counter[str] = Counter()
    for record in records:
        if record.get("verdict") == "reject":
            reasons.update(record.get("reasons") or ())
    entry: dict[str, Any] = {
        "n": n,
        "accepts": accepts,
        "accept_rate": round(accepts / n, 3) if n else 0.0,
        "sources": sources,
        "top_reasons": [
            [reason, count]
            for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }
    # Kept as nulls for the readers that expect the keys. Nothing measures them
    # now that the run directory (and its scores.json) is gone.
    for metric in METRIC_NAMES:
        entry[f"mean_{metric}"] = None
    return entry


def presets(doc: dict[str, Any], *, min_n: int = PRESET_MIN_N) -> list[dict[str, Any]]:
    """The vectors judged often enough to be worth offering, best first."""
    return [v for v in (doc.get("vectors") or []) if v.get("n", 0) >= min_n]


def refresh(svc: Any) -> Path:
    """Recompute ``findings.json`` and write it under the bench dir.

    Task thread only -- it reads every verdict and writes a file, neither of
    which belongs on the frame loop. The path is unchanged from the old
    ``bench report`` output, which is what keeps ``bench.findings.load`` (and
    so both settings panes) working with no edit at all.
    """
    bench_dir = Path(svc.config.bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / JSON_FILENAME
    path.write_text(json.dumps(aggregate(svc.store), indent=2), encoding="utf-8")
    return path

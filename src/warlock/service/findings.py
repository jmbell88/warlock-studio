"""Turning recorded verdicts and observations into ``findings.json``.

Three sections, three kinds of answer. ``params`` is the confounded marginal:
a verdict credits every ``param: value`` pair in its vector, so "lora_weight
0.9 accepts 6/8" means "of the eight judged assets whose lora_weight was 0.9,
six were accepted", everything else free to vary. That is deliberate -- the
price of letting ordinary daily-use verdicts feed the hints at all -- and the
machine ``metrics`` sub-objects share exactly those semantics: an observation
(hole fraction, watertight, triangles, recorded by the worker on every
finished model job) credits every pair in *its* vector too, which is what
lets a value the user has generated with twenty times but never reviewed
still say something true.

Two consequences of that being the *same* rule for both, neither of which is
an oversight. An observation is only ever written for a model-stage job, but
a model job's vector carries the whole 2D taxonomy it was promoted with -- so
"art_style snes" accumulates the hole fraction of the meshes reconstructed
from SNES-era references, and the hint appears under a prompt control in the 2D
pane. That is a real finding (whether a style reconstructs badly is exactly
what someone choosing it wants to know), which is why the reader says
"(21 meshes)" rather than "(21 runs)": the number has to name what it
measured, since the control it sits beside does not. And a reference that was
never promoted contributes nothing, so these marginals describe the meshes
that were made, not the images that were drawn.

``vectors`` is the whole-configuration ranking --
verdict-driven, because a recipe nobody judged is evidence but not a
recommendation -- ordered by the Wilson score lower bound rather than the raw
rate, so a 19/20 outranks a 5/5 instead of the other way round.

``comparisons`` is the unconfounded answer, recovered from sweep structure.
Two rows form a matched pair iff they share a sweep, a source and a seed and
their vectors differ in exactly one key -- the diff is computed on the
vectors themselves, never parsed out of a unit label, and an absent key is a
side of the contrast ("unset"), which is how a baseline pairs with an axis
unit. Win counts come from human verdict pairs; per-metric deltas come from
observation pairs, so an axis shows machine evidence the moment its sweep
finishes, before anyone reviews a thing. The pairing context (sweep_id, seed,
prompt_hash) is denormalized onto both row kinds precisely so these pairs
survive ``delete_sweep`` and ``prune_jobs``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The vector vocabulary lives in ``warlock.vectors`` so the worker can import
# it without importing ``service``; re-exported here because this module is
# where every service-side caller (and test) learned the names.
from ..vectors import VECTOR_PARAMS, config_vector, vector_key

__all__ = [
    "VECTOR_PARAMS",
    "config_vector",
    "vector_key",
    "aggregate",
    "presets",
    "refresh",
]

# 2: vectors rank by wilson_low, the dead mean_* nulls are gone, and the
# metrics / comparisons sections exist. Readers tolerate v1 files (hint falls
# back to the bare "accept 6/8" when wilson_low is absent).
# 3: the ``prompts`` section -- the same marginals again, one set per subject
# (TODO item 4) -- and the ``refused*`` metrics the composition gate now
# records (item 3). Both are additive: every v2 key keeps its meaning and its
# value, so a reader that knows nothing about either is correct on a v3 file,
# and ``hint`` asked for a subject it cannot find simply falls back to the
# pooled answer.
FINDINGS_VERSION = 3
JSON_FILENAME = "findings.json"

# How many verdicts a whole vector needs before it is offered as a preset. The
# same threshold ``bench.findings.hint`` applies to a single param value, and
# for the same reason: a thin bucket is noise, not a finding.
PRESET_MIN_N = 5

# 95% score interval. Used for *ordering* and as a "floor of confidence" in
# display; the raw rate is always shown beside it, and PRESET_MIN_N keeps
# gating, so the conservatism costs nothing where it is not needed.
WILSON_Z = 1.96


def wilson_low(accepts: int, n: int, z: float = WILSON_Z) -> float:
    """Lower bound of the Wilson score interval for ``accepts``/``n``.

    The recognisable "how not to sort by average rating" bound: harsh exactly
    where the raw rate misleads -- thin buckets -- which is what fixes a 5/5
    outranking a 19/20 (lower bounds 0.566 and 0.764).
    """
    if n <= 0:
        return 0.0
    p = accepts / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return round(max(0.0, (centre - spread) / (1 + z2 / n)), 3)


def aggregate(store: Any) -> dict[str, Any]:
    """The whole findings document, over every source's latest verdicts and
    every job's latest observation."""
    # **Mesh verdicts only, and the filter is here rather than in _marginals.**
    # §7 asks for it in ``_marginals`` on the grounds that the per-subject
    # ``prompts`` section is built by the same helper -- true, and not enough.
    # Two other consumers read this same list. ``vectors`` would advertise an
    # image label as a ranked mesh configuration, and ``_comparisons`` pairs rows
    # sharing a sweep and a seed whose vectors differ in one key -- which a blank
    # label and a mesh verdict on the *same unit* satisfy, yielding a matched
    # "pair" that compares two different questions. One filter at the top covers
    # all four; four filters are four chances to forget one.
    #
    # Migration 7 is what makes this necessary at all: before it, every row was a
    # mesh verdict and ``stage`` did not exist. Rows written before it default to
    # ``model``, so the filter is a no-op over the historical corpus.
    records = [
        r
        for r in store.latest_verdicts()
        if isinstance(r.get("vector"), dict) and r.get("stage", "model") == "model"
    ]
    observations = [
        o for o in store.latest_observations() if isinstance(o.get("vector"), dict)
    ]

    vectors: dict[str, dict[str, Any]] = {}
    for record in records:
        key = vector_key(record["vector"])
        bucket = vectors.setdefault(
            key, {"key": key, "vector": record["vector"], "records": []}
        )
        bucket["records"].append(record)

    vector_metrics: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        metrics = obs.get("metrics")
        if isinstance(metrics, dict) and metrics:
            vector_metrics.setdefault(vector_key(obs["vector"]), []).append(metrics)

    out_params = _marginals(records, observations, full=True)

    out_vectors = []
    for bucket in vectors.values():
        summary = _summarise(bucket["records"])
        entry = {
            "key": bucket["key"],
            "vector": bucket["vector"],
            "n": summary["n"],
            "accepts": summary["accepts"],
            "accept_rate": summary["accept_rate"],
            "wilson_low": summary["wilson_low"],
            "top_reasons": summary["top_reasons"],
            "jobs": [r["job_id"] for r in bucket["records"]],
        }
        if bucket["key"] in vector_metrics:
            entry["metrics"] = _metric_summary(vector_metrics[bucket["key"]])
        out_vectors.append(entry)
    # Best first by the lower bound, not the raw rate -- the ordering the
    # bound exists for -- with sample size breaking exact ties.
    out_vectors.sort(key=lambda v: (-v["wilson_low"], -v["n"], v["key"]))

    return {
        "version": FINDINGS_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": out_params,
        "prompts": _per_prompt(records, observations),
        "vectors": out_vectors,
        "comparisons": _comparisons(records, observations),
    }


def _marginals(
    records: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    full: bool,
) -> dict[str, dict[str, Any]]:
    """The confounded ``param: value`` marginals over one pool of rows.

    Factored out so the global document and each per-prompt scope are computed
    by the same code rather than by two that can drift -- the whole value of
    scoping is that a scoped number means exactly what the pooled one means,
    over a narrower set.

    ``full`` is what the global one gets. A scoped bucket keeps only the four
    fields ``bench.findings.hint`` reads, because there is one of them per
    distinct prompt and ``sources``/``top_reasons`` are read by the Review
    pane's whole-vector view, which stays global.
    """
    judged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        for param, value in record["vector"].items():
            if param == "stage":
                continue
            judged.setdefault(param, {}).setdefault(str(value), []).append(record)

    # The machine-evidence twin: the same confounded crediting, so a value
    # generated with twenty times and never reviewed still says something.
    measured: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for obs in observations:
        metrics = obs.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            continue
        for param, value in obs["vector"].items():
            if param == "stage":
                continue
            measured.setdefault(param, {}).setdefault(str(value), []).append(metrics)

    # Sorted so the written file is stable and diffable across refreshes --
    # set-union order would churn with hash randomization on every launch.
    out: dict[str, dict[str, Any]] = {}
    for param in sorted(set(judged) | set(measured)):
        rows, metric_rows = judged.get(param, {}), measured.get(param, {})
        out[param] = {}
        for value in sorted(set(rows) | set(metric_rows)):
            entry = _summarise(rows.get(value, []))
            if not full:
                entry = {
                    k: entry[k] for k in ("n", "accepts", "accept_rate", "wilson_low")
                }
            if value in metric_rows:
                entry["metrics"] = _metric_summary(metric_rows[value])
            out[param][value] = entry
    return out


def _per_prompt(
    records: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """The same marginals again, one set per subject.

    The global ones pool across subjects, which is a confound of a different
    kind from the accepted one. A verdict crediting every ``param: value`` in
    its vector is the price of letting daily use feed the hints at all, and
    everything under it was still generated for the same thing. Two subjects
    are not: what makes a good wooden crate says very little about what makes
    a good character, and a hint that silently mixes them is worse than no hint
    -- it is trusted.

    A row with no prompt (an upload, a painted reference) has ``prompt_hash``
    ``""``, and is skipped rather than bucketed: that is an absence, and a
    bucket keyed on it would pool every unrelated one of them into a single
    subject that does not exist.
    """
    by_prompt: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for pool, rows in ((0, records), (1, observations)):
        for row in rows:
            key = row.get("prompt_hash")
            if not isinstance(key, str) or not key:
                continue
            by_prompt.setdefault(key, ([], []))[pool].append(row)
    return {
        key: {"params": _marginals(scoped_records, scoped_obs, full=False)}
        for key, (scoped_records, scoped_obs) in sorted(by_prompt.items())
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
    return {
        "n": n,
        "accepts": accepts,
        "accept_rate": round(accepts / n, 3) if n else 0.0,
        "wilson_low": wilson_low(accepts, n),
        "sources": sources,
        "top_reasons": [
            [reason, count]
            for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over the observations where each metric was present -- the audit
    can succeed while the report fails, so per-metric counts genuinely differ
    and an absent reading must not drag a mean toward zero.

    Which is exactly why every mean is accompanied by its own count in
    ``counts``. ``n`` is the size of the *bucket*, and a reader that printed it
    beside a mean was making a claim about the wrong number: one job in
    twenty-one whose ``meshreport.build`` failed (logged and swallowed,
    queue.py) or came back ``status: "invalid"`` carries no ``watertight`` at
    all, so a single reading would be advertised as twenty-one.
    """
    counts: dict[str, int] = {}
    out: dict[str, Any] = {"n": len(rows), "counts": counts}

    def numbers(key: str) -> list[float]:
        return [
            float(r[key])
            for r in rows
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
        ]

    def flags(key: str) -> list[bool]:
        return [r[key] for r in rows if isinstance(r.get(key), bool)]

    for key in ("hole_worst", "hole_mean"):
        values = numbers(key)
        if values:
            out[key] = round(sum(values) / len(values), 4)
            counts[key] = len(values)
    for key, name in (("watertight", "watertight_rate"), ("ready", "ready_rate")):
        values = flags(key)
        if values:
            out[name] = round(sum(values) / len(values), 3)
            counts[name] = len(values)
    # The composition gate's verdicts, averaged the same way. Stored as 0.0/1.0
    # numbers rather than bools precisely so this is a mean: a job that passed
    # contributes a zero, which is what makes the result a *rate* rather than
    # the tautology "every refused job was refused". Enumerated off the rows
    # rather than off a fixed list so a rule added to reference.py needs no
    # edit here -- but still prefixed, so nothing else can land in the sum.
    for key in sorted({k for row in rows for k in row if str(k).startswith("refused")}):
        values = numbers(key)
        if values:
            out[f"{key}_rate"] = round(sum(values) / len(values), 3)
            counts[f"{key}_rate"] = len(values)
    values = numbers("triangles")
    if values:
        out["triangles"] = int(round(sum(values) / len(values)))
        counts["triangles"] = len(values)
    return out


_MISSING = object()


def _one_key_diff(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """The param two vectors disagree on, iff they disagree on exactly one.

    Absence is a value: the baseline never set the axis param, so the diff
    against an axis unit is one *absent* key. ``stage`` alone is not a
    contrast anyone chose. This -- never the sweep_unit label -- is what makes
    two rows a matched pair, so hand-picked ``vectors`` units pair whenever
    they happen to be one-key neighbours and mislabeled units cannot lie.
    """
    diff = [k for k in a.keys() | b.keys() if a.get(k, _MISSING) != b.get(k, _MISSING)]
    if len(diff) == 1 and diff[0] != "stage":
        return diff[0]
    return None


def _comparisons(
    records: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Matched-pair contrasts, accumulated across sweeps.

    Win counts come from verdict pairs grouped by (sweep, source) -- a human
    and an AI judge disagreeing is not an axis finding -- and metric deltas
    from observation pairs grouped by sweep alone. Both bucket by seed within
    the group: same prompt (same sweep), same seed, one param differing is
    the whole definition of all-else-equal here.

    Sign convention the reader must honour: every delta is *a-side minus
    b-side*, where ``a`` is the lexicographically lower value string. The
    winner of the human pairs is unrelated to that ordering, so a display
    that leads with the winner has to re-orient the sign
    (``bench.findings._entry_lines`` does).
    """
    stats: dict[tuple[str, str, str], dict[str, Any]] = {}

    def entry_for(param: str, low: str, high: str) -> dict[str, Any]:
        return stats.setdefault(
            (param, low, high),
            {
                "pairs": 0, "a_wins": 0, "b_wins": 0, "ties": 0,
                "sweeps": set(), "prompts": set(),
                "obs_sweeps": set(), "obs_prompts": set(),
                "deltas": {},
            },
        )

    def oriented(param: str, row_a: dict[str, Any], row_b: dict[str, Any]):
        def side(row: dict[str, Any]) -> str:
            vector = row["vector"]
            return str(vector[param]) if param in vector else "unset"

        a_val, b_val = side(row_a), side(row_b)
        if a_val == b_val:
            # The one-key diff compares values, this compares their spellings,
            # and 0.6 against "0.6" passes the first and collapses under the
            # second -- which would print "0.6 beat 0.6 in 3 of 4 matched
            # pairs". Not reachable through the sweep builder (guidance
            # coerces), and a contrast with one side is not a contrast.
            return None
        if a_val > b_val:
            return b_val, a_val, row_b, row_a
        return a_val, b_val, row_a, row_b

    def grouped(rows: list[dict[str, Any]], with_source: bool):
        groups: dict[Any, dict[Any, list[dict[str, Any]]]] = {}
        for row in rows:
            if not row.get("sweep_id") or row.get("seed") is None:
                continue
            key = (row["sweep_id"], row["source"]) if with_source else row["sweep_id"]
            groups.setdefault(key, {}).setdefault(row["seed"], []).append(row)
        return groups

    for group in grouped(records, with_source=True).values():
        for rows in group.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    param = _one_key_diff(rows[i]["vector"], rows[j]["vector"])
                    if param is None:
                        continue
                    sides = oriented(param, rows[i], rows[j])
                    if sides is None:
                        continue
                    low_val, high_val, low, high = sides
                    entry = entry_for(param, low_val, high_val)
                    entry["pairs"] += 1
                    low_accept = low.get("verdict") == "accept"
                    high_accept = high.get("verdict") == "accept"
                    if low_accept and not high_accept:
                        entry["a_wins"] += 1
                    elif high_accept and not low_accept:
                        entry["b_wins"] += 1
                    else:
                        entry["ties"] += 1
                    entry["sweeps"].add(low["sweep_id"])
                    if low.get("prompt_hash"):
                        entry["prompts"].add(low["prompt_hash"])

    for group in grouped(observations, with_source=False).values():
        for rows in group.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    param = _one_key_diff(rows[i]["vector"], rows[j]["vector"])
                    if param is None:
                        continue
                    sides = oriented(param, rows[i], rows[j])
                    if sides is None:
                        continue
                    low_val, high_val, low, high = sides
                    low_m = low.get("metrics")
                    high_m = high.get("metrics")
                    if not isinstance(low_m, dict) or not isinstance(high_m, dict):
                        continue
                    deltas = {}
                    for metric in low_m.keys() & high_m.keys():
                        a_val, b_val = low_m[metric], high_m[metric]
                        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                            deltas[metric] = float(a_val) - float(b_val)
                    if not deltas:
                        continue
                    entry = entry_for(param, low_val, high_val)
                    for metric, delta in deltas.items():
                        entry["deltas"].setdefault(metric, []).append(delta)
                    entry["obs_sweeps"].add(low["sweep_id"])
                    if low.get("prompt_hash"):
                        entry["obs_prompts"].add(low["prompt_hash"])

    out: dict[str, list[dict[str, Any]]] = {}
    for (param, low_val, high_val), e in stats.items():
        # Breadth counts describe the human pairs when there are any; a
        # delta-only entry honestly describes its observation pairs instead.
        sweeps = e["sweeps"] or e["obs_sweeps"]
        prompts = e["prompts"] or e["obs_prompts"]
        out.setdefault(param, []).append(
            {
                "a": low_val,
                "b": high_val,
                "pairs": e["pairs"],
                "a_wins": e["a_wins"],
                "b_wins": e["b_wins"],
                "ties": e["ties"],
                "sweeps": len(sweeps),
                "prompts": len(prompts),
                "deltas": {
                    metric: {
                        "mean": round(sum(values) / len(values), 4),
                        "pairs": len(values),
                    }
                    for metric, values in sorted(e["deltas"].items())
                },
            }
        )
    for param in out:
        out[param].sort(key=lambda item: (-item["pairs"], item["a"], item["b"]))
    return dict(sorted(out.items()))


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
    # Staged, like every other file something else reads -- ``postprocess._staged``
    # is the same idiom, written out here rather than imported because that
    # module pulls trimesh in. Both settings panes load this through
    # ``bench.findings.load`` on the frame thread, and a bare ``write_text``
    # lets one of them observe a half-written document. It self-heals on the
    # next recompute, which is exactly why it survived as the one unstaged
    # writer.
    fd, raw = tempfile.mkstemp(dir=bench_dir, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(raw)
    try:
        tmp.write_text(json.dumps(aggregate(svc.store), indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return path

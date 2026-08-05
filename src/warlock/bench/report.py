"""Turning a sweep's verdicts into a findings table.

A sweep answers "does this one parameter matter", one axis at a time, against
a fixed baseline (``sweep.py``). ``score.py`` puts a number on a unit and
``verdicts.py`` puts a human's (or an AI judge's) Accept/Reject on it; neither
answers the sweep's own question, which is "grouped by the value this axis
was set to, what fraction of units were accepted, and what did the score
metrics say". That is this module's one job.

Grouping is keyed off ``items.jsonl``, not the verdict's own ``param``/
``value`` fields: a verdict already carries a copy of them (``append_verdict``
requires it), but the item record is what the sweep actually planned and run,
so a join against it is the same "the run directory is the source of truth"
rule ``score.py`` follows for status. A baseline unit's item record carries
``param: None`` -- grouped under the literal strings ``"baseline"``/
``"baseline"`` rather than the string ``"None"``, so the control group reads
as one in the table instead of looking like a fourth axis value.

Scores join is best-effort and null-tolerant: ``scores.json`` may not exist
yet (nobody ran ``bench score``), and even when it does, ``dino_cosine`` is
silently absent whenever DINOv2 was never downloaded (the torchvision
caveat in TODO.md). A unit with no score for a metric is simply excluded from
that metric's mean rather than turning the whole group's mean into ``None``.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import manifest as manifest_mod
from . import runner as runner_mod
from . import score as score_mod
from . import verdicts as verdicts_mod

FINDINGS_VERSION = 1
JSON_FILENAME = "findings.json"
MD_FILENAME = "findings.md"

# The control group's synthetic param/value -- a baseline unit's item record
# carries param=None, and grouping on the literal string "None" would put it
# in a bucket that looks like a fourth axis value rather than the control.
BASELINE_PARAM = "baseline"
BASELINE_VALUE = "baseline"

# The only two metrics metrics.score_view ever produces (metrics.available).
# Named explicitly, not discovered from scores.json's "metrics" list, because
# a run scored without the text2image extra never has dino_cosine at all and
# findings.json's schema must stay the same shape either way.
METRIC_NAMES = ("silhouette_iou", "dino_cosine")


def sweep_runs(config: Any) -> list[Path]:
    """Run directories under ``bench_dir/runs`` whose manifest carries a
    ``"sweep"`` marker -- a plain suite run (``runner.run``) has no such key
    and is silently skipped, exactly as an unreadable manifest is."""
    root = Path(config.bench_dir) / "runs"
    if not root.exists():
        return []
    out: list[Path] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            doc = manifest_mod.read_manifest(path)
        except (OSError, ValueError):
            continue
        if "sweep" in doc:
            out.append(path)
    return out


def _scores_by_key(run_dir: Path) -> dict[str, dict[str, Any]]:
    try:
        doc = score_mod.read_scores(run_dir)
    except (OSError, ValueError):
        return {}
    return {unit["key"]: unit.get("scores") or {} for unit in doc.get("units") or ()}


def aggregate(runs: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{param: {value_str: {n, accepts, accept_rate, sources,
    mean_silhouette_iou, mean_dino_cosine, top_reasons}}}`` over every verdict
    in ``runs``.

    A verdict with no matching item record (a unit key that was never planned
    -- should not happen, but the join is defensive the same way ``score.py``
    is about a status it doesn't recognise) is skipped rather than guessed at.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for run_dir in runs:
        items = runner_mod.latest_items(run_dir)
        scores = _scores_by_key(run_dir)
        for record in verdicts_mod.latest(run_dir).values():
            item = items.get(record["unit"])
            if item is None:
                continue
            param = item.get("param")
            value = item.get("value")
            key = (
                BASELINE_PARAM if param is None else str(param),
                BASELINE_VALUE if param is None else str(value),
            )
            entry = dict(record)
            entry["_scores"] = scores.get(record["unit"], {})
            groups.setdefault(key, []).append(entry)

    out: dict[str, dict[str, Any]] = {}
    for (param, value), records in groups.items():
        out.setdefault(param, {})[value] = _summarise_group(records)
    return out


def _summarise_group(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    top_reasons = [
        [reason, count]
        for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    entry: dict[str, Any] = {
        "n": n,
        "accepts": accepts,
        "accept_rate": round(accepts / n, 3) if n else 0.0,
        "sources": sources,
        "top_reasons": top_reasons,
    }
    for metric in METRIC_NAMES:
        values = [
            float(v)
            for record in records
            for v in [record["_scores"].get(metric)]
            if isinstance(v, int | float)
        ]
        entry[f"mean_{metric}"] = round(statistics.fmean(values), 4) if values else None
    return entry


def write_findings(config: Any, doc: dict[str, dict[str, dict[str, Any]]]) -> tuple[Path, Path]:
    """Write ``findings.json`` and ``findings.md`` under ``bench_dir`` from
    ``aggregate``'s output, returning both paths."""
    bench_dir = Path(config.bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)
    full = {
        "version": FINDINGS_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": doc,
    }
    json_path = bench_dir / JSON_FILENAME
    json_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    md_path = bench_dir / MD_FILENAME
    md_path.write_text("\n".join(summary_lines(full)) + "\n", encoding="utf-8")
    return json_path, md_path


def summary_lines(doc: dict[str, Any]) -> list[str]:
    """One line per param value, mirroring ``score.summary_lines``'s style: a
    header, then indented figures, "nothing to score" when a metric has no
    number rather than a bare ``None``."""
    params = doc.get("params") or {}
    if not params:
        return ["no verdicts recorded yet"]
    out: list[str] = []
    for param in sorted(params):
        out.append(f"{param}:")
        values = params[param]
        for value in sorted(values):
            entry = values[value]
            out.append(
                f"  {value}: {entry['accepts']}/{entry['n']} accepted"
                f" (rate {entry['accept_rate']})"
            )
            for metric in METRIC_NAMES:
                mean = entry.get(f"mean_{metric}")
                out.append(
                    f"    {metric}: nothing to score"
                    if mean is None
                    else f"    {metric}: mean {mean}"
                )
            if entry["top_reasons"]:
                reasons = ", ".join(f"{reason} x{count}" for reason, count in entry["top_reasons"])
                out.append(f"    top reject reasons: {reasons}")
    return out

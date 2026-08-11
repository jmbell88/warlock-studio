"""Turn a finished run's rendered views into numbers.

This is the half the package docstring always claimed and never had. ``--render``
produced eight views per mesh "for measurement" and nothing read them;
``metrics.score_view`` existed and was called only by ``calibrate``; the
``scores.json`` that manifest.py's own docstring names was written by nothing.
So an A/B -- the thing recipes exist for -- could be generated but not decided.

Scoring is separated from running for the reason the module docstring gives for
keeping the manifest: it is re-appliable. A metric change can be run against
every past run without regenerating a pixel, because the views are on disk and
the manifest says what they were made with.

The comparison itself is one pair per unit: the reference the mesh was built
from, against the camera-matched view of the mesh. Column 0 is that view by
construction (``views.reference_view``), which is the same pairing ``calibrate``
sweeps to find.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

from . import manifest as manifest_mod
from . import runner as runner_mod
from . import views as views_mod

log = logging.getLogger(__name__)

SCORES_VERSION = 1
FILENAME = "scores.json"


def score_run(
    run_dir: Path, config: Any = None, *, on_event: Any = None
) -> dict[str, Any]:
    """Score every finished unit in ``run_dir`` and write scores.json."""
    from . import metrics as metrics_mod

    say = on_event or (lambda _msg: None)
    run_dir = Path(run_dir)
    records = runner_mod.latest_items(run_dir)
    if not records:
        raise ValueError(f"{run_dir.name}: no items.jsonl to score")

    provenance = _provenance(run_dir)
    # A reference-stage run has no mesh and therefore no views, which used to
    # mean it could not be scored at all. It can: the pixel metrics measure the
    # generated picture itself, which is the only thing that stage produces.
    reference_stage = provenance.get("stage") == "reference"
    names = (
        list(metrics_mod.REFERENCE_METRICS)
        if reference_stage
        else list(metrics_mod.available(config))
    )
    units: list[dict[str, Any]] = []
    for key in sorted(records):
        record = records[key]
        entry = {
            "key": key,
            "item": record.get("item"),
            "category": record.get("category"),
            "seed": record.get("seed"),
            "status": record.get("status"),
        }
        pair = _pair(run_dir, key, record)
        if record.get("status") != "done":
            entry["skipped"] = f"status {record.get('status')}"
        elif reference_stage:
            reference = run_dir / "items" / key / "input.png"
            if not reference.exists():
                entry["skipped"] = "no reference image"
            else:
                say(f"scoring {key}")
                try:
                    entry["scores"] = metrics_mod.score_reference(reference)
                except Exception as exc:
                    # The same rule the view path states below: a measurement
                    # that fails costs its unit a score, never the run.
                    log.warning("could not score %s: %s", key, exc)
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    say(f"could not score {key}: {exc}")
        elif pair is None:
            # Not a failure: a reference-stage run has no mesh to render, and a
            # model-stage run without --render has no views. Saying which is
            # the difference between "rerun with --render" and "this is fine".
            entry["skipped"] = "no rendered views"
        else:
            reference, render = pair
            say(f"scoring {key}")
            try:
                entry["scores"] = metrics_mod.score_view(reference, render, config)
            except Exception as exc:
                # The same rule the runner states for a render: this costs the
                # item its automatic score, not the run. A CUDA OOM 900 units
                # into 1280 used to discard every one of them and write no
                # scores.json at all.
                log.warning("could not score %s: %s", key, exc)
                entry["error"] = f"{type(exc).__name__}: {exc}"
                say(f"could not score {key}: {exc}")
        units.append(entry)

    doc = {
        "version": SCORES_VERSION,
        "run": run_dir.name,
        "metrics": names,
        # Enough of the manifest to make a pasted scores.json self-describing;
        # the manifest itself stays the authority next to it.
        "measured": provenance,
        "units": units,
        "summary": summarise(units, names),
        "by_category": {
            category: summarise(rows, names)
            for category, rows in _by_category(units).items()
        },
    }
    write_scores(run_dir, doc)
    return doc


def _pair(
    run_dir: Path, key: str, record: dict[str, Any] | None = None
) -> tuple[Path, Path] | None:
    """(reference, camera-matched render) for one unit, or None.

    A record that says its render *failed* wins over the directory. A unit
    re-run under --resume against a different mesh, whose second render wrote
    nothing, still has the first mesh's frames on disk -- and scoring those
    attributes an old silhouette to a new reconstruction, which is worse than
    not scoring it at all.

    Only ``views is False`` refuses. The key is absent both on runs made before
    it existed and on runs made without ``--render``, and in the second case
    there is no views/ directory to be misled by anyway.
    """
    if record is not None and record.get("views") is False:
        return None
    item_dir = run_dir / "items" / key
    reference = item_dir / "input.png"
    views_dir = item_dir / "views"
    if not reference.exists() or not views_dir.is_dir():
        return None
    render = views_mod.reference_view(views_dir)
    return None if render is None else (reference, render)


# What compare_manifests actually reads. Kept as one list rather than spelled
# out inline, because the two used to disagree: _provenance carried four keys,
# compare_manifests read six, and the two it could not see -- config and models
# -- are precisely the ones an A/B is most likely to have moved. Both sides
# missing a key compare equal, so the drift report was silent about a changed
# band and a changed checkpoint.
PROVENANCE_KEYS = ("stage", "suite", "recipe", "versions", "config", "models")


def _provenance(run_dir: Path) -> dict[str, Any]:
    try:
        doc = manifest_mod.read_manifest(run_dir)
    except (OSError, ValueError):
        return {}
    return {key: doc.get(key) for key in PROVENANCE_KEYS}


def _by_category(units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        out.setdefault(str(unit.get("category") or ""), []).append(unit)
    return out


def summarise(units: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """Mean, median and count per metric over the units that produced one.

    ``n`` is reported alongside every figure on purpose: a mean over three of
    a hundred units is not a run's score, and without the count the two are
    indistinguishable in the output.
    """
    out: dict[str, Any] = {}
    for name in names:
        values = [
            float(v)
            for unit in units
            for v in [(unit.get("scores") or {}).get(name)]
            if isinstance(v, int | float)
        ]
        out[name] = {
            "n": len(values),
            "mean": round(statistics.fmean(values), 4) if values else None,
            "median": round(statistics.median(values), 4) if values else None,
        }
    out["units"] = len(units)
    # A unit counts as scored when at least one metric produced a *number*.
    # ``score_view`` always returns a dict -- {"silhouette_iou": None} when the
    # render has no subject -- and a non-empty dict is truthy, so counting
    # dicts reported "N of N scored" for a run in which nothing was measurable
    # and suppressed the one line telling the user to rerun with --render.
    out["scored"] = sum(
        1
        for u in units
        if any(isinstance(v, int | float) for v in (u.get("scores") or {}).values())
    )
    return out


def write_scores(run_dir: Path, doc: dict[str, Any]) -> Path:
    path = Path(run_dir) / FILENAME
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def read_scores(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / FILENAME).read_text("utf-8"))


# --- reporting ---------------------------------------------------------------


def summary_lines(doc: dict[str, Any]) -> list[str]:
    names = list(doc.get("metrics") or ())
    summary = doc.get("summary") or {}
    out = [
        f"{doc.get('run')}: {summary.get('scored', 0)} of "
        f"{summary.get('units', 0)} unit(s) scored"
    ]
    for name in names:
        stats = summary.get(name) or {}
        if not stats.get("n"):
            out.append(f"  {name}: nothing to score")
            continue
        out.append(
            f"  {name}: mean {stats['mean']}  median {stats['median']}  (n={stats['n']})"
        )
    return out


def compare_lines(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Two scored runs, metric by metric -- the A/B recipes exist for.

    Comparability is not asserted here: the two runs' manifests are what decide
    that, and ``manifest.compare_manifests`` is the thing that knows. This
    reports the difference and names the drift, so a deliberate A/B (which
    changes the recipe, and therefore *is* drift) still prints.
    """
    out: list[str] = []
    drift = manifest_mod.compare_manifests(
        {**(old.get("measured") or {})}, {**(new.get("measured") or {})}
    )
    for line in drift:
        out.append(f"  differs: {line}")
    for name in sorted(set(old.get("metrics") or ()) | set(new.get("metrics") or ())):
        a = ((old.get("summary") or {}).get(name) or {}).get("mean")
        b = ((new.get("summary") or {}).get(name) or {}).get("mean")
        if a is None or b is None:
            out.append(f"  {name}: not measured on both runs")
            continue
        out.append(f"  {name}: {a} -> {b}  ({b - a:+.4f})")
    return out

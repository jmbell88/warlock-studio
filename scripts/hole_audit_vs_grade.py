"""Tabulate the silhouette audit against the human grade for a graded corpus.

Does the four-view see-through audit (``meshaudit``, stored on every finished
mesh as ``params["mesh_audit"]["worst"]``) see the holes a reviewer tags? The
question decides whether ``Config.mesh_retries`` -- the reroll-and-keep-best
loop, shipping at 0 since its 0.07 trigger was retired on a 41-unit
single-subject corpus -- fires on the corpus that matters. Three outcomes, and
the measurement document that records them is
``docs/measurements/2026-09-02-hole-audit-vs-grade.md``:

- the reviewer's holed meshes measure *above* the trigger: the retired trigger
  fires on this corpus, and the default is worth re-testing with a re-run;
- they measure *below* it: the audit does not see what the reviewer sees, which
  is a finding about ``meshaudit``, and the trigger stays retired;
- a mix, in which case the table itself is the evidence and the threshold is
  what to argue about.

A reader, not a submitter: it opens the job database, reads the rows carrying
the corpus tag, joins the latest human mesh verdict and the corpus file's
difficulty class (on the prompt -- the class is deliberately never written to
the job, so a blind grading pass stays blind), and prints one table. Nothing
is written, no GPU is touched, and the app's own lock discipline is respected
by going through ``JobStore``.

    uv run python scripts/hole_audit_vs_grade.py
    uv run python scripts/hole_audit_vs_grade.py --tag props-v1 --threshold 0.07
    uv run python scripts/hole_audit_vs_grade.py --csv > audit.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import campaign_props  # noqa: E402

from warlock import vectors  # noqa: E402
from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402
from warlock.service.core import WarlockService  # noqa: E402

#: The only source and stage a mesh grade lives under. ``verdicts_for`` keys
#: on ``(job_id, source)`` and takes ``stage`` as a filter for exactly this
#: one-question-at-a-time reason.
SOURCE = "human"
STAGE = "model"


def tagged_jobs(store: JobStore, tag: str) -> list[dict[str, Any]]:
    """Every finished mesh-stage job carrying ``tag``, oldest first.

    Paged with ``JobStore.list``'s keyset cursor rather than one enormous
    limit, so a library with years of rows costs the same per page as a
    small one. The tag column is comma-separated and normalised lowercase,
    which is what the split-and-compare below relies on.
    """
    wanted = tag.strip().lower()
    out: list[dict[str, Any]] = []
    before: tuple[float, str] | None = None
    while True:
        page = store.list(limit=500, before=before)
        if not page:
            break
        for job in page:
            tags = [t for t in (job.get("tags") or "").split(",") if t]
            if wanted in tags and job.get("stage") == STAGE:
                out.append(job)
        last = page[-1]
        before = (last["created_at"], last["id"])
    out.reverse()
    return out


def classes_by_prompt(corpus: Path) -> dict[str, str]:
    """``{prompt: class}`` from the corpus file; empty if the file is absent,
    so a corpus queued from somewhere else still tabulates."""
    if not corpus.is_file():
        return {}
    return {s.prompt: s.cls for s in campaign_props.read_corpus(corpus)}


def _mib(path: Path) -> float | None:
    """Size in MiB, or None when the file is not there (a failed unit)."""
    try:
        return round(path.stat().st_size / (1024 * 1024), 1)
    except OSError:
        return None


def tabulate(
    store: JobStore,
    jobs: list[dict[str, Any]],
    classes: dict[str, str],
    svc: WarlockService | None = None,
) -> list[dict[str, Any]]:
    verdicts = store.verdicts_for([j["id"] for j in jobs], source=SOURCE, stage=STAGE)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        audit = (job.get("params") or {}).get("mesh_audit") or {}
        verdict = verdicts.get((job["id"], SOURCE))
        reasons = list(verdict.get("reasons") or []) if verdict else []
        # The detail sweep's machine evidence: what the two GLBs weigh. Read
        # from disk rather than params because nothing records it, and the
        # source/model split is exactly the exe-decimation question.
        job_dir = svc.job_dir(job["id"]) if svc is not None else None
        rows.append(
            {
                "job_id": job["id"],
                "status": job.get("status"),
                "unit": job.get("sweep_unit") or "",
                "source_mib": _mib(job_dir / "source.glb") if job_dir else None,
                "model_mib": _mib(job_dir / "model.glb") if job_dir else None,
                "seconds": (
                    round(job["finished_at"] - job["started_at"], 1)
                    if job.get("finished_at") and job.get("started_at")
                    else None
                ),
                "class": classes.get(job.get("prompt") or "", "?"),
                "prompt": job.get("prompt") or "",
                "worst": audit.get("worst"),
                "mean": audit.get("mean"),
                "faces": audit.get("faces"),
                "grade": verdict.get("grade") if verdict else None,
                "holes": "holes" in reasons,
                "tags": ",".join(reasons),
            }
        )
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]], threshold: float) -> None:
    header = (
        f"{'class':8} {'worst':>7} {'mean':>7} {'faces':>8} {'srcMiB':>7} {'glbMiB':>7} "
        f"{'secs':>6} {'grade':>5} {'holes':5}  prompt"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['class']:8} {_fmt(r['worst']):>7} {_fmt(r['mean']):>7} "
            f"{_fmt(r['faces']):>8} {_fmt(r['source_mib'], 1):>7} {_fmt(r['model_mib'], 1):>7} "
            f"{_fmt(r['seconds'], 0):>6} {_fmt(r['grade']):>5} "
            f"{'yes' if r['holes'] else 'no':5}  {r['prompt'][:48]}"
            + (f"  [{r['unit']}]" if r['unit'] else "")
        )
    print()
    summarise(rows, threshold)


def summarise(rows: list[dict[str, Any]], threshold: float) -> None:
    """The three-outcome question, answered in numbers rather than adjectives."""
    audited = [r for r in rows if isinstance(r["worst"], (int, float))]
    graded = [r for r in audited if r["grade"] is not None]
    holed = [r for r in graded if r["holes"]]
    clean = [r for r in graded if not r["holes"]]
    fires = [r for r in audited if r["worst"] > threshold]
    print(f"rows: {len(rows)}   audited: {len(audited)}   graded: {len(graded)}")
    print(f"trigger {threshold}: fires on {len(fires)} of {len(audited)} audited")
    if holed:
        above = sum(1 for r in holed if r["worst"] > threshold)
        print(
            f"holes-tagged: {len(holed)}, of which {above} above the trigger"
            f" (worst {min(r['worst'] for r in holed):.4f}..{max(r['worst'] for r in holed):.4f})"
        )
    if clean:
        above = sum(1 for r in clean if r["worst"] > threshold)
        print(
            f"not holes-tagged: {len(clean)}, of which {above} above the trigger"
            f" (worst {min(r['worst'] for r in clean):.4f}..{max(r['worst'] for r in clean):.4f})"
        )
    usable = [r for r in graded if r["grade"] >= vectors.USABLE_GRADE]
    if graded:
        print(f"usable (grade >= {vectors.USABLE_GRADE}): {len(usable)} of {len(graded)}")
        if usable:
            print(
                f"  usable worst: {min(r['worst'] for r in usable):.4f}"
                f"..{max(r['worst'] for r in usable):.4f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=campaign_props.DEFAULT_TAG)
    parser.add_argument("--corpus", type=Path, default=campaign_props.DEFAULT_CORPUS)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="the reroll trigger to test against; default Config.mesh_hole_max",
    )
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of the table")
    args = parser.parse_args()

    config = Config()
    threshold = config.mesh_hole_max if args.threshold is None else args.threshold
    db_path = Path(config.db_path)
    if not db_path.exists():
        print(f"no job database at {db_path}", file=sys.stderr)
        return 1
    store = JobStore(db_path)
    try:
        jobs = tagged_jobs(store, args.tag)
        svc = WarlockService(config, store)
        rows = tabulate(store, jobs, classes_by_prompt(args.corpus), svc)
    finally:
        store.close()
    if not rows:
        print(f"no mesh-stage jobs tagged {args.tag!r} in {db_path}", file=sys.stderr)
        return 1
    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_table(rows, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Qualify the gltfpack tiers: does draft/standard/detailed keep what it must?

`TODO.md` §3. ``vendor/gltfpack/gltfpack.exe`` arrived on 2026-08-07, so
``pipelines/optimize.py``, the config field, the doctor check and the retarget
panel's full tier list are all live. What is *not* done is the qualification, and
until it is, ``Config.mesh_profile`` stays ``raw`` and the generate forms offer
``raw`` alone. The bar: a tier stays unqualified until it has been run against a
chest, a sword and a rock and shown to keep UVs, both PBR maps and material
assignment.

The measurable half is ``warlock.tiercheck``, which reads both GLBs' JSON chunks
and reports every loss; this script is the driver, the corpus selection and the
before/after picture. Three things about how it runs are deliberate.

**It never writes inside the corpus.** Every output lands under ``--out``. The
plan says to run the tiers "through the existing ``optimize.py`` +
``service.jobs.optimize_job`` path", and the *binary and flags* half of that is
exactly what happens -- ``tiercheck.qualify`` defaults to ``optimize.run``, so
there is no second spelling of the invocation. What is deliberately not driven
is ``optimize_job`` itself: it rewrites ``model.glb`` in place and deletes the
derived artifacts describing the old mesh, so it would consume the accepted
meshes this needs. A harness that destroys its inputs can be run once.

**The corpus defaults to the accepted meshes and says so.** Whether a tier
*preserves* something cannot be judged on output that is already broken, and 80
of the rogue sweep's 83 meshes were rejects. With no subjects named, this reads
the verdict table and picks every mesh a human accepted. It prints the list, and
prints a warning when there are fewer than three subjects or when they are not
recognisably a chest, a sword and a rock -- because a pass over one blob is not
the qualification the bar describes.

**The render sheet is a picture, and nothing scores it.** ``--sheets`` renders
each subject before and after through the one sheet pipeline
(``sheet.plan``/``pack`` around ``blender_worker.op_sheet``), which needs the
``rig`` extra. The automated checks are the pass/fail; the sheet is for the eye,
because "the silhouette survived" is not a claim any number here earns.

    uv run python scripts/qualify_tiers.py --out bench/tiers
    uv run python scripts/qualify_tiers.py --glb a.glb b.glb --tier draft
    uv run python scripts/qualify_tiers.py --sheets            # + a picture
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from warlock import tiercheck  # noqa: E402
from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402

# The named tiers, in budget order. ``raw`` is not one: it applies no budget, so
# there is nothing to preserve and nothing to compare against.
TIERS = ("draft", "standard", "detailed")

# What the bar names. Matched loosely against a job's name and prompt, because
# the point is to notice a corpus of three chests rather than to police wording.
WANTED_SHAPES = ("chest", "sword", "rock")


def _accepted_sources(store: JobStore, job_dir) -> list[Path]:
    """Every accepted mesh's ``source.glb``.

    ``source.glb`` rather than ``model.glb``: it is the immutable
    reconstruction, and ``model.glb`` is already the output of an
    optimize-then-normalize pass -- simplifying a simplified mesh measures the
    second pass, not the tier.
    """
    out: list[Path] = []
    for row in store.latest_verdicts():
        if row.get("verdict") != "accept":
            continue
        source = job_dir(row["job_id"]) / "source.glb"
        if source.exists():
            out.append(source)
    return out


def _describe(store: JobStore, sources: list[Path]) -> list[str]:
    """A human name per subject, for the warning about corpus shape."""
    names: list[str] = []
    for source in sources:
        job = store.get(source.parent.name)
        names.append(str((job or {}).get("name") or (job or {}).get("prompt") or source.stem))
    return names


def _warn_about_the_corpus(names: list[str]) -> None:
    if len(names) < 3:
        print(
            f"warning: {len(names)} subject(s). The bar names three -- a sword is thin, "
            "a rock is a blob, a chest has flat panels and a lid, and a simplifier's "
            "failure modes are shape-dependent.",
            file=sys.stderr,
        )
    lowered = " ".join(names).lower()
    missing = [shape for shape in WANTED_SHAPES if shape not in lowered]
    if missing:
        print(
            f"warning: nothing in this corpus looks like a {', '.join(missing)}. "
            "A pass here is weaker than the bar in TODO.md section 3 describes.",
            file=sys.stderr,
        )


def _table(report: tiercheck.Report) -> str:
    lines = ["| subject | tier | triangles | verdict |", "|---|---|---|---|"]
    for row in report.rows:
        stats = row.stats or {}
        triangles = (
            f"{stats.get('source_triangles') or '?'} -> {stats.get('achieved') or '?'}"
        )
        detail = "pass" if row.verdict.ok else "; ".join(row.verdict.failures)
        lines.append(f"| {row.subject} | {row.tier} | {triangles} | {detail} |")
    return "\n".join(lines)


def _render_sheets(sources: list[Path], report: tiercheck.Report, out: Path) -> None:
    """A before/after sheet per subject, through the one sheet pipeline."""
    from warlock import doctor, rigging
    from warlock.pipelines import sheet as sheetlib

    check = doctor.blender_check()
    if not check.ok:
        print(f"skipping sheets: {check.detail}", file=sys.stderr)
        return
    wanted: list[tuple[str, Path]] = [(f"{p.parent.name}-source", p) for p in sources]
    wanted += [
        (f"{row.subject}-{row.tier}", Path(row.output))
        for row in report.rows
        if row.output
    ]
    # One row, the unposed mesh: ``plan`` takes an empty pose list for exactly
    # this, and a prop has no rig to pose anyway.
    plan = sheetlib.plan([], frame_size=256, lighting="lit")
    for label, glb in wanted:
        with tempfile.TemporaryDirectory() as scratch:
            frames = Path(scratch)
            spec = rigging.sheet_spec(
                glb,
                frames,
                [asdict(cell) for cell in plan.cells],
                frame_size=plan.frame_size,
                elevation=plan.elevation,
                lighting=plan.lighting,
            )
            try:
                rigging.run_worker(spec, timeout=600.0)
                # ``pack`` takes an index -> path mapping, not the directory;
                # the worker names each frame ``{index:04d}.png``.
                frame_map = {c.index: frames / f"{c.index:04d}.png" for c in plan.cells}
                sheetlib.pack(plan, frame_map, out / f"{label}.png")
            except Exception as exc:  # noqa: BLE001 -- a picture is not the verdict
                print(f"sheet for {label} failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glb", nargs="*", type=Path, help="subjects, as GLB paths")
    parser.add_argument("--job", nargs="*", help="subjects, as job ids (uses source.glb)")
    parser.add_argument(
        "--tier", action="append", choices=TIERS, help="repeatable; default all three"
    )
    parser.add_argument("--out", type=Path, default=Path("bench/tiers"))
    parser.add_argument(
        "--sheets", action="store_true", help="also render before/after sheets (needs bpy)"
    )
    args = parser.parse_args()

    config = Config()
    store = JobStore(Path(config.db_path))
    try:
        sources = [Path(p) for p in (args.glb or ())]
        for job_id in args.job or ():
            sources.append(Path(config.data_dir) / job_id / "source.glb")
        chose_corpus = not sources
        if chose_corpus:
            sources = _accepted_sources(store, config.job_dir)
        missing = [p for p in sources if not p.exists()]
        if missing:
            print(f"no such GLB: {missing[0]}", file=sys.stderr)
            return 1
        if not sources:
            print(
                "no accepted meshes to qualify against. TODO.md section 3: whether "
                "a tier preserves something cannot be judged on output that is "
                "already broken -- review some meshes first, or name subjects "
                "with --glb.",
                file=sys.stderr,
            )
            return 1

        names = _describe(store, sources)
        print(f"{len(sources)} subject(s): {', '.join(names)}")
        if chose_corpus:
            _warn_about_the_corpus(names)

        tiers = tuple(args.tier or TIERS)
        out = Path(args.out)
        report = tiercheck.qualify(sources, tiers, out, exe=Path(config.gltfpack_exe))
    finally:
        store.close()

    table = _table(report)
    print()
    print(table)
    for row in report.rows:
        for note in row.verdict.notes:
            print(f"note: {row.subject}/{row.tier}: {note}")

    (out / "report.json").write_text(
        json.dumps(
            {
                "tiers": list(report.tiers),
                "subjects": list(report.subjects),
                "qualified": list(report.qualified),
                "rows": [
                    {
                        "subject": row.subject,
                        "tier": row.tier,
                        "ok": row.verdict.ok,
                        "failures": list(row.verdict.failures),
                        "notes": list(row.verdict.notes),
                        "error": row.error,
                        "stats": row.stats,
                        "before": asdict(row.before) if row.before else None,
                        "after": asdict(row.after) if row.after else None,
                    }
                    for row in report.rows
                ],
            },
            indent=2,
        ),
        "utf-8",
    )
    (out / "report.md").write_text(table + "\n", "utf-8")

    if args.sheets:
        _render_sheets(sources, report, out)

    print()
    if report.qualified:
        print(f"qualified on every subject: {', '.join(report.qualified)}")
        print(
            "That is the automated half. Look at the sheets before exposing a tier in "
            "panes/settings_3d.PROFILES, and leave Config.mesh_profile at raw -- the "
            "default flip is a separate decision."
        )
    else:
        print("nothing qualified. PROFILES stays as it is.")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

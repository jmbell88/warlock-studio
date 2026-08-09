"""Run a re-texture against real meshes and lay the result out to be looked at.

The qualification harness for Tier 2's bake, and it is `qualify_tiers.py`'s
shape for `qualify_tiers.py`'s reason: **it may not consume its own corpus.**
``service.jobs.retexture_job`` rewrites the served ``model.glb`` in place, so
driving it would destroy the meshes the next comparison needs -- and a harness
that eats its inputs can be run once. This copies each mesh into its own output
directory and works there; ``assets/`` is never written to.

It drives the same four stages the worker does, in the same order, through the
same functions: ``op_views`` renders the mesh flat from ``retexture.VIEWS``,
one ``Text2Image.generate`` img2img pass restyles each render,
``op_project`` bakes each restyled view back into the mesh's own atlas with a
weight image, and ``retexture.assemble`` combines them. It is deliberately not
the *queue* path -- that is unit-tested -- because the question this exists to
answer is about pixels, and reaching them through a job row would add a worker,
an event loop and a database to a question none of them bears on.

The deliverable is the contact sheet: the six flat views before, the six after,
and the two atlases side by side, per subject. A verdict about whether a
multi-view projection bake is good enough is a verdict somebody reaches by
looking, so the harness's job is to make looking cheap and to record the
numbers that would otherwise be an impression -- coverage, and how much of the
atlas actually changed.

    uv run python scripts/retexture_probe.py --out docs/measurements/data/retexture \
        --subject 3e32ea1a8533="rusted iron, oxidised metal plating"
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _render_views(source_glb: Path, out_dir: Path, size: int) -> None:
    from warlock import rigging
    from warlock.pipelines import retexture

    out_dir.mkdir(parents=True, exist_ok=True)
    rigging.run_worker(
        rigging.views_spec(source_glb, out_dir, list(retexture.VIEWS), size=size),
        timeout=1800.0,
    )


def _restyle(t2i, prompt: str, views_dir: Path, count: int, *, strength: float, seed: int) -> None:
    from warlock.pipelines.conditioning import Conditioning

    for index in range(count):
        init = views_dir / f"view_{index:02d}.png"
        if not init.exists():
            continue
        t0 = time.monotonic()
        t2i.generate(
            prompt,
            views_dir / f"restyled_{index:02d}.png",
            seed=seed,
            conditioning=Conditioning(init_image=init, strength=strength),
        )
        print(f"    view {index}: {time.monotonic() - t0:.1f}s")


def _contact_sheet(rows: list[list[Path]], dest: Path, cell: int) -> None:
    """One row per stage, one column per view. Nothing clever: a grid."""
    from PIL import Image

    width = cell * max(len(r) for r in rows)
    sheet = Image.new("RGB", (width, cell * len(rows)), (24, 24, 28))
    for y, row in enumerate(rows):
        for x, path in enumerate(row):
            if not path.exists():
                continue
            with Image.open(path) as im:
                im.load()
                tile = im.convert("RGB").resize((cell, cell), Image.LANCZOS)
            sheet.paste(tile, (x * cell, y * cell))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, "PNG")


def _atlas_delta(before: Path, after: Path) -> dict[str, float] | None:
    """How much of the atlas actually moved, and by how much.

    Coverage says how much of it a view could *speak* about; this says how much
    it then had anything to say. The two differ wherever a low restyle strength
    gave a view back nearly what it was handed, which is exactly the case where
    a run looks like it did nothing and the coverage figure alone would not
    explain it.
    """
    import numpy as np
    from PIL import Image

    if not before.exists() or not after.exists():
        return None
    with Image.open(before) as a, Image.open(after) as b:
        a.load()
        b.load()
        first = np.asarray(a.convert("RGB"), dtype=np.float32)
        second = np.asarray(b.convert("RGB"), dtype=np.float32)
    if first.shape != second.shape:
        return None
    diff = np.abs(first - second).mean(axis=2)
    return {
        "mean_abs_diff": float(diff.mean()),
        "changed_fraction": float((diff > 8.0).mean()),
    }


def run_subject(
    job_dir: Path, prompt: str, out_dir: Path, *, args: argparse.Namespace, t2i
) -> dict:
    from warlock.pipelines import retexture

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "model.glb"
    shutil.copyfile(job_dir / "model.glb", work)

    views = list(retexture.VIEWS)
    before_dir = out_dir / "before"
    print("  rendering before views")
    _render_views(work, before_dir, args.view_px)

    print("  restyling")
    t0 = time.monotonic()
    _restyle(
        t2i, prompt, before_dir, len(views), strength=args.strength, seed=args.seed
    )
    restyle_s = time.monotonic() - t0

    print("  projecting")
    from warlock import rigging

    rigging.run_worker(
        rigging.project_spec(
            work, before_dir, before_dir, views,
            size=args.view_px, texture_size=args.texture_px,
        ),
        timeout=1800.0,
    )

    base_png = out_dir / "atlas_before.png"
    has_base = retexture.extract_base_colour(work, base_png, size=args.texture_px)
    atlas = out_dir / "atlas_after.png"
    report = retexture.assemble(
        before_dir, base_png if has_base else None, atlas, count=len(views)
    )
    if report is None:
        raise SystemExit(f"no projection could be read back for {job_dir.name}")

    swapped = out_dir / "retextured.glb"
    if not retexture.swap_base_colour(work, atlas, swapped):
        raise SystemExit(f"{job_dir.name} has no base-colour texture to replace")

    # The after views come off the *swapped* mesh rather than being the restyled
    # renders: a restyled render is what the projection was asked to bake, and
    # showing it as the result would be grading the bake on its own input.
    after_dir = out_dir / "after"
    print("  rendering after views")
    _render_views(swapped, after_dir, args.view_px)

    _contact_sheet(
        [
            [before_dir / f"view_{i:02d}.png" for i in range(len(views))],
            [before_dir / f"restyled_{i:02d}.png" for i in range(len(views))],
            [after_dir / f"view_{i:02d}.png" for i in range(len(views))],
        ],
        out_dir / "contact.png",
        args.cell_px,
    )
    _contact_sheet([[base_png, atlas]], out_dir / "atlas.png", args.texture_px // 2)

    report = dict(report)
    report["restyle_seconds"] = round(restyle_s, 1)
    report["delta"] = _atlas_delta(base_png, atlas)
    report["prompt"] = prompt
    report["had_base"] = has_base
    print(f"  coverage {report['coverage'] * 100:.1f}%  delta {report['delta']}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--subject",
        action="append",
        default=[],
        metavar="JOB_ID=PROMPT",
        help="repeatable; a job directory under assets/ and the surface to ask for",
    )
    ap.add_argument("--assets", type=Path, default=ROOT / "assets")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-model", default="sdxl_cfg")
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--view-px", type=int, default=1024)
    ap.add_argument("--texture-px", type=int, default=1024)
    ap.add_argument("--cell-px", type=int, default=256)
    args = ap.parse_args()
    if not args.subject:
        ap.error("at least one --subject JOB_ID=PROMPT")

    from warlock import models
    from warlock.config import get_config
    from warlock.pipelines.text2image import Text2Image

    config = get_config()
    spec = models.BASE_MODELS[args.base_model]
    t2i = Text2Image(spec, config.t2i_model_root, None)

    results: dict[str, dict] = {}
    try:
        for entry in args.subject:
            job_id, _, prompt = entry.partition("=")
            job_dir = args.assets / job_id
            if not (job_dir / "model.glb").exists():
                raise SystemExit(f"no mesh at {job_dir / 'model.glb'}")
            print(f"[{job_id}] {prompt}")
            results[job_id] = run_subject(
                job_dir, prompt, args.out / job_id, args=args, t2i=t2i
            )
    finally:
        t2i.unload()

    summary = {
        "base_model": args.base_model,
        "strength": args.strength,
        "seed": args.seed,
        "view_px": args.view_px,
        "texture_px": args.texture_px,
        "subjects": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

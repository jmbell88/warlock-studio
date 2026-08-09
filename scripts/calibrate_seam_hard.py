"""The second half of the seam calibration: hard-structured tiles, tiled only.

The first batch (``calibrate_seam.py``) establishes the *contrast* -- what a
circular-padded generation scores against an identical un-padded one. It leaves
one question open, and it is the question that decides the constant: the
tiled arm's ceiling there is a wood-plank tile at 1.87, and the thing pushing it
up is hard linear structure, not a seam. So the ratio's denominator -- the
picture's mean adjacent-pixel difference -- is small relative to the individual
plank edges the wrap column may land on.

This batch is nothing but that failure mode: eight materials chosen for hard
edges, straight lines and large flat cells, generated with ``tile=True`` only.
If a legitimately seamless tile scores above the threshold here, the threshold
is wrong and has to move up. If none does, the observed tiled ceiling stands.

There is no plain arm because there is no contrast to establish -- the first
batch did that, and a second copy of it would only widen an already-settled gap.

Run:  uv run python scripts/calibrate_seam_hard.py --out docs/measurements/data/seam

The same ``--out`` as ``calibrate_seam.py`` on purpose: the two batches are one
corpus and the measurement document reads them together. This half writes
``results-hard.json`` beside the other's ``results.json``, so neither overwrites
the other and re-running one does not invalidate the other. See
``docs/measurements/2026-08-08-seam-threshold.md``, which both reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from warlock import models  # noqa: E402
from warlock.pipelines import seam, text2image  # noqa: E402

HARD: dict[str, str] = {
    "checker": "black and white checkerboard floor tiles",
    "mosaic": "small square ceramic mosaic tiles with grout",
    "hex": "hexagonal concrete paving stones",
    "panel": "sci-fi metal hull panels with rivets and seams",
    "chain": "galvanised chain link fence mesh",
    "circuit": "printed circuit board traces and pads",
    "herring": "herringbone parquet wood floor",
    "corrugate": "corrugated ribbed metal sheeting",
}

SEEDS = (11, 12, 13)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=models.DEFAULT_BASE_MODEL)
    ap.add_argument("--model-root", type=Path, default=Path("models"))
    args = ap.parse_args()

    spec = models.BASE_MODELS[args.base]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    t2i = text2image.Text2Image(spec, args.model_root)

    rows: list[dict[str, object]] = []
    for material, text in HARD.items():
        for seed in SEEDS:
            name = f"{material}-s{seed}-tiled.png"
            path = out / name
            t2i.generate(text, path, seed=seed, tile=True)
            rep = seam.report(path)
            seam.wrap_preview(path, out / f"{material}-s{seed}-wrap.png")
            rows.append(
                {
                    "material": material,
                    "seed": seed,
                    "arm": "tiled",
                    "file": name,
                    "horizontal": rep["horizontal"],
                    "vertical": rep["vertical"],
                    "worst": rep["worst"],
                }
            )
            print(
                f"{material:10s} s{seed} h={rep['horizontal']:.3f} "
                f"v={rep['vertical']:.3f} worst={rep['worst']:.3f}",
                flush=True,
            )

    (out / "results-hard.json").write_text(
        json.dumps({"base_model": args.base, "materials": HARD, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

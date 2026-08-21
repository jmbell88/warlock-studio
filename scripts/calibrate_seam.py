"""Generate a two-class tile corpus and measure ``seam.report`` over it.

``SEAM_MAX`` is a corpus-keyed constant, so it owes a measurement document
before it moves (repo rule). This is the harness that produces the corpus that
document reads.

The design is a *contrast*, not a survey. One prompt set is generated with
``tile=True`` -- the circular-padding path, which makes the wrap seam a
non-event by construction -- and the identical prompts, seeds and checkpoint
are generated again with ``tile=False``. The threshold's whole job is to
separate those two populations, so measuring only the first would say what a
seamless tile scores and nothing at all about what it has to be distinguished
from.

Writes one PNG per unit plus a rolled-by-half ``wrap_preview`` for the tiled
half (the only way to *see* the failure the ratio measures), and a
``results.json`` carrying every ratio. It scores nothing and moves nothing:
picking the constant is a judgement made against this output, by eye, in the
measurement document.

Run:  uv run python scripts/calibrate_seam.py --out docs/measurements/data/seam
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from warlock import models  # noqa: E402
from warlock.pipelines import prompt as prompt_lib  # noqa: E402
from warlock.pipelines import seam, text2image  # noqa: E402

# The four names the seam work started from, plus four more that stress the
# ratio's denominator from both ends: the statistic divides the wrap seam by the
# picture's own grain, so a texture with almost no grain (plaster) and one that
# is nothing but grain (gravel) are the two places it can misbehave.
MATERIALS: dict[str, str] = {
    "stone": "rough grey granite stone surface",
    "plaster": "smooth off-white painted plaster wall",
    "gravel": "loose grey gravel and small pebbles",
    "fabric": "woven linen fabric weave",
    "brick": "weathered red brick wall with mortar",
    "wood": "worn oak wood planks",
    "grass": "dense green grass turf",
    "metal": "scratched brushed steel panel",
}

SEEDS = (11, 12, 13)


def _compose(text: str) -> str:
    """What the queue composes for a tile, minus the taxonomy fields.

    ``guidance.compose_prompt`` with no params contributes nothing, so the
    bare material phrase is exactly what a user typing it into the 2D form
    with every dropdown unset would get. ``generate`` applies TILE_TEMPLATE
    itself.
    """
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=models.DEFAULT_BASE_MODEL)
    ap.add_argument("--model-root", type=Path, default=Path("models"))
    args = ap.parse_args()

    spec = models.BASE_MODELS[args.base]
    if args.base not in models.tile_bases():
        print(f"{args.base} cannot generate a seamless tile", file=sys.stderr)
        return 2

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    t2i = text2image.Text2Image(spec, args.model_root)

    rows: list[dict[str, object]] = []
    for material, text in MATERIALS.items():
        for seed in SEEDS:
            for tiled in (True, False):
                arm = "tiled" if tiled else "plain"
                name = f"{material}-s{seed}-{arm}.png"
                path = out / name
                started = time.monotonic()
                t2i.generate(
                    _compose(text),
                    path,
                    seed=seed,
                    tile=tiled,
                )
                elapsed = time.monotonic() - started
                rep = seam.report(path)
                if tiled:
                    seam.wrap_preview(path, out / f"{material}-s{seed}-wrap.png")
                rows.append(
                    {
                        "material": material,
                        "seed": seed,
                        "arm": arm,
                        "file": name,
                        "horizontal": rep["horizontal"],
                        "vertical": rep["vertical"],
                        "worst": rep["worst"],
                        "seconds": round(elapsed, 2),
                    }
                )
                print(
                    f"{material:8s} s{seed} {arm:5s} "
                    f"h={rep['horizontal']:.3f} v={rep['vertical']:.3f} "
                    f"worst={rep['worst']:.3f}  ({elapsed:.1f}s)",
                    flush=True,
                )

    (out / "results.json").write_text(
        json.dumps(
            {
                "base_model": args.base,
                "prompt_version": prompt_lib.PROMPT_VERSION,
                "threshold_at_capture": seam.SEAM_MAX,
                "materials": MATERIALS,
                "seeds": list(SEEDS),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

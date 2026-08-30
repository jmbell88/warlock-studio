"""Training a style LoRA on the user's own art, offline: the host-side half.

The one 2D capability every commercial generator sells -- Scenario, Layer,
Leonardo, Krea, Recraft all train a style adapter from 5-100 images -- and
every one of them does it in the cloud. Here it is the same mechanism (a rank-r
LoRA on the SDXL UNet's attention projections, DreamBooth-style, one trigger
phrase) run in a child process on the user's card, so the art never leaves the
machine. The product is an ordinary ``STYLE_LORAS`` entry, registered through
``generation.import_lora`` exactly as a downloaded adapter would be, so nothing
downstream -- the picker, ``lora_fits``, ``_ensure_adapter`` -- learns a second
path.

This module holds the numbers and the spec; the training loop is
``lora_train_worker.py`` and runs under ``rigging.run_worker``'s generic
child-process contract (spec on stdin, ``[train] frac label`` on stdout,
result JSON at ``result_path``). Pure: stdlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: How many training images a run needs and may take. Three is the floor at
#: which a rank-16 adapter learns a *style* rather than memorising a picture;
#: a hundred is where an afternoon on a 5090 stops being an afternoon.
MIN_IMAGES = 3
MAX_IMAGES = 100

#: Steps. 800 at batch 1 is the commercial tools' "20-40 minutes" on a 5090
#: (~1.5 it/s at 1024 with gradient checkpointing), and where a ~20-image set
#: stops improving in the pixel-art-xl and pixelklein qualification grids'
#: idiom of "the lowest setting that gives a clean result".
DEFAULT_STEPS = 800
MIN_STEPS = 100
MAX_STEPS = 3000

#: Adapter rank and learning rate. Rank 16 with ``lora_alpha == rank`` keeps
#: the adapter's natural strength at slider 1.0, which is why a trained entry
#: is registered at ``TRAINED_WEIGHT`` rather than the module default: its
#: scale is known by construction, not measured after the fact.
RANK = 16
LEARNING_RATE = 1e-4
RESOLUTION = 1024
TRAINED_WEIGHT = 1.0

#: The file the trainer writes, in diffusers' own layout so
#: ``pipe.load_lora_weights`` reads it with no conversion.
WEIGHTS_NAME = "pytorch_lora_weights.safetensors"

#: The child's progress marker and its deadline. Four hours: the ceiling for
#: ``MAX_STEPS`` on a slow card, not a budget for the default.
MARKER = "train"
TIMEOUT = 4 * 3600.0

MAX_LABEL = 64
MAX_TRIGGER = 64


def train_spec(
    base_dir: Path,
    images: list[Path],
    out_dir: Path,
    result_dir: Path,
    *,
    trigger: str,
    steps: int = DEFAULT_STEPS,
    rank: int = RANK,
    learning_rate: float = LEARNING_RATE,
    resolution: int = RESOLUTION,
    seed: int = 0,
) -> dict[str, Any]:
    """The worker spec. Every path is a string, every number resolved: the
    child reads it and asks nothing of the host's config."""
    return {
        "op": "train",
        "base_dir": str(base_dir),
        "images": [str(p) for p in images],
        "out_dir": str(out_dir),
        "result_path": str(result_dir / ".train_result.json"),
        "trigger": str(trigger),
        "steps": int(steps),
        "rank": int(rank),
        "learning_rate": float(learning_rate),
        "resolution": int(resolution),
        "seed": int(seed),
    }


def check_steps(steps: Any) -> int:
    """``MIN_STEPS..MAX_STEPS`` or a ``ValueError`` naming the range."""
    try:
        value = int(steps)
    except (TypeError, ValueError) as exc:
        raise ValueError("steps must be a whole number") from exc
    if not MIN_STEPS <= value <= MAX_STEPS:
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    return value


def report_line(report: Any) -> str | None:
    """"800 steps over 24 images, final loss 0.081" -- or None."""
    if not isinstance(report, dict):
        return None
    steps = report.get("steps")
    images = report.get("images")
    if steps is None or images is None:
        return None
    line = f"{steps} steps over {images} images"
    loss = report.get("loss")
    if isinstance(loss, int | float):
        line += f", final loss {loss:.3f}"
    return line

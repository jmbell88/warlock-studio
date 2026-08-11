"""Queue the FLUX.2 klein half of the SNES/Zelda prop run: same subjects, no axes.

**This is a sibling of ``sweep_zelda_style.py``, not an arm of it, and the
distinction is forced rather than stylistic.** ``models.lora_fits`` is
family-gated, so a ``base_model=flux_klein_distilled`` arm against that run's
base is *refused* by ``guidance.normalize`` inside ``_check_unit`` -- "style_lora
'render3d' is fitted to sdxl and base_model 'flux_klein_distilled' is
flux2klein" -- which under all-or-nothing admission would take the whole 60-unit
campaign with it. And with the adapter dropped, a klein arm still moves
``base_model``, ``style_lora`` and ``lora_weight`` at once, so
``findings._one_key_diff`` could never pair it. There is no way to ask this
question inside that sweep.

**What it can do instead is share a bucket.** Both prompts here are imported
from ``sweep_zelda_style`` rather than restated, so they hash to the same two
``prompt_hash``es, and ``findings._per_prompt`` therefore files these rows in the
*same* per-subject buckets as the playground run. The comparison is between two
``params.base_model`` marginals inside one subject's scope -- confounded in the
way every marginal is, and explicitly **not** a matched pair, since
``_comparisons`` groups on ``sweep_id`` and no pair spans two sweeps. That is a
weaker instrument than the axes next door and it is the honest one available.

**Why it is worth ten units at all.** The single accepted row in the live corpus
is ``flux_klein_distilled`` at ``art_style=snes`` with no style adapter, graded
+4 and tagged ``on-style`` -- the only ``on-style`` accept anybody has recorded.
It is n=1 and it was a *character* at ``platform=2d``, so it is not a claim about
props; it is a reason to spend half an hour finding out. The rest of the
depiction vector is character-for-character the playground run's baseline, so
the two differ in the checkpoint and in the adapter's absence and in nothing
else.

**Ten is chosen, not left over.** The playground run's *baseline* arm is five
units per subject, so this is exactly matched to the only thing it can be read
against. Extra seeds here would be units with no counterpart on the other side
of that comparison.

**No adapter, deliberately.** ``pixelklein`` is the only LoRA that fits this
family, it is a *pixel-art* adapter qualified at 0.0625 on 2026-08-09, and flat
pixel art is poor trellis input -- the reading of "SNES-era" that
``sweep_rogue`` put in precisely to measure rather than assume, and which has
never once won. The +4 above also ran without one. Adding it here would move a
second key against the very marginal this run exists to read.

**No axes.** ``expand`` on an axis-less plan gives one baseline unit per seed and
``_check_unit``'s canonical key includes the seed, so five is five distinct jobs.
The deliverable is the meshes and the two marginals; a contrast would need pairs
this design cannot form.

Run it **after** ``sweep_zelda_style.py`` has drained, with the app closed:
``_campaign.require_no_live_writer`` guards the dangerous case, and two
processes writing one rollback-journal sqlite file is how a submit gets lost.

    uv run python scripts/sweep_zelda_klein.py --dry-run   # plan and validate
    uv run python scripts/sweep_zelda_klein.py             # write the rows
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _campaign  # noqa: E402
from sweep_rogue import SEEDS  # noqa: E402
from sweep_zelda_style import BASE as SDXL_BASE  # noqa: E402
from sweep_zelda_style import SUBJECTS  # noqa: E402

from warlock.service import sweeps as sweeps_mod  # noqa: E402

# The depiction half is taken from the playground run rather than retyped: these
# two sweeps are only comparable while that half is identical, and a second copy
# of it is how the two come to disagree by one word six weeks from now. The
# render half is what differs, and it is written out here in full.
_RENDER_HALF = ("base_model", "style_lora", "lora_weight")

BASE = {
    **{k: v for k, v in SDXL_BASE.items() if k not in _RENDER_HALF},
    # ~10 GiB at peak under `residency="offload"` -- it streams one submodule at
    # a time rather than going resident, so it is slower per job and displaces
    # nothing. vram.estimate charges 10.0 + 16.0 against a 32 GB card's budget.
    "base_model": "flux_klein_distilled",
    # No style_lora and therefore no lora_weight; see the module docstring.
}


PLANS = tuple(
    sweeps_mod.SweepPlan(
        label=f"zelda klein - {label}",
        prompt=prompt,
        base={**BASE, **fields},
        # The same five seeds, though nothing here pairs: it keeps the two runs'
        # noise drawn from the same places, which is the most a marginal
        # comparison can be given.
        seeds=SEEDS,
        axes=(),
        stage="model",
    )
    for label, prompt, fields in SUBJECTS
)


if __name__ == "__main__":
    raise SystemExit(_campaign.main(PLANS, __doc__.splitlines()[0]))

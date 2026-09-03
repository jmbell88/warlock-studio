"""Queue the trellis detail sweep over the detail-v1 subjects.

The instrument ``docs/measurements/2026-09-03-trellis-detail-sweep.md``
pre-registers: for each subject, one sweep whose units are the *vectors* below
rather than an OFAT fan-out, because two of the rungs are pairs (``--decim 0``
is only a question beside a gltfpack budget, and a 4096 atlas only beside a
1024 texture). ``expand`` adds its own ``baseline`` unit per sweep, which *is*
the shipped rung, so no vector restates it -- ``_validate`` would refuse the
duplicate by canonical key.

Two passes, because ``resolution=1536`` puts trellis at 24 GiB beside the
7 GiB image pipe and on a 32 GB card that is a WDDM spill into host commit
waiting to happen. ``--exclusive-pass`` plans one unit per subject -- the
sweep's *base* is resolution 1536 and there are no vectors, so ``expand``'s
own ``baseline`` unit is the 1536 rung and nothing restates the shipped one
-- under its own tag, and is submitted and drained under
``WARLOCK_VRAM_EXCLUSIVE=1``. The default pass emits the five res-1024
vectors beside the shipped baseline.

One sweep *per subject* because a ``SweepPlan`` carries exactly one prompt
(``campaign_props.py`` says why a corpus is not a sweep). Every unit is also
tagged, so ``scripts/hole_audit_vs_grade.py --tag <tag> --corpus <corpus>``
tabulates the whole campaign with the unit label beside each row.

A submitter, not a runner, like ``_campaign.py``.

    uv run python scripts/campaign_detail.py --dry-run
    uv run python scripts/campaign_detail.py
    WARLOCK_VRAM_EXCLUSIVE=1 uv run python scripts/campaign_detail.py --exclusive-pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _campaign  # noqa: E402
import campaign_props  # noqa: E402

from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402
from warlock.service import jobs as jobs_mod  # noqa: E402
from warlock.service import sweeps as sweeps_mod  # noqa: E402
from warlock.service.core import WarlockService  # noqa: E402
from warlock.service.errors import ServiceError  # noqa: E402

DEFAULT_CORPUS = _ROOT / "docs" / "measurements" / "corpora" / "detail-v1.txt"
DEFAULT_TAG = "detail-060"
EXCLUSIVE_TAG = "detail-060-1536"

#: The exe's own quadric target at res 1024, so the decim0-300k rung asks
#: gltfpack for the same count the shipped rung gets from the exe.
EXE_FACE_TARGET = 300_000

#: The res-1024 rungs, beside the implicit ``baseline`` (the shipped default:
#: decim omitted, raw, tex_res 512, atlas omitted). Label first, because the
#: unit label is what the tabulator prints beside each row.
VECTORS: tuple[dict[str, object], ...] = (
    {
        "label": "decim0-300k",
        "trellis_decim": 0,
        "profile": "custom",
        "custom_triangles": EXE_FACE_TARGET,
    },
    {
        "label": "decim0-1M",
        "trellis_decim": 0,
        "profile": "custom",
        "custom_triangles": 1_000_000,
    },
    {"label": "decim0-raw", "trellis_decim": 0},
    {"label": "tex1024", "trellis_tex_res": 1024},
    {"label": "tex1024-atlas4096", "trellis_tex_res": 1024, "trellis_atlas": 4096},
)

#: The exclusive-mode pass: geometry resolution 1536, everything else shipped,
#: as the sweep's base so its one unit is ``expand``'s own baseline.
EXCLUSIVE_BASE: dict[str, object] = {"resolution": 1536}


def plan_for(prompt: str, seed: int, *, exclusive: bool) -> sweeps_mod.SweepPlan:
    if exclusive:
        return sweeps_mod.SweepPlan(
            label=f"detail-1536: {prompt[:40]}",
            prompt=prompt,
            base=dict(EXCLUSIVE_BASE),
            seeds=(seed,),
        )
    return sweeps_mod.SweepPlan(
        label=f"detail: {prompt[:40]}",
        prompt=prompt,
        seeds=(seed,),
        vectors=VECTORS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--tag", default=None, help=f"default {DEFAULT_TAG}, or {EXCLUSIVE_TAG} for the 1536 pass"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclusive-pass",
        action="store_true",
        help="queue only the res-1536 rung; run under WARLOCK_VRAM_EXCLUSIVE=1",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tag = args.tag or (EXCLUSIVE_TAG if args.exclusive_pass else DEFAULT_TAG)

    subjects = campaign_props.read_corpus(args.corpus)
    plans = [plan_for(s.prompt, args.seed, exclusive=args.exclusive_pass) for s in subjects]

    config = Config()
    db_path = Path(config.db_path)
    _campaign.require_no_live_writer(db_path)
    store = JobStore(db_path)
    svc = WarlockService(config, store)
    try:
        total = 0
        for plan in plans:
            try:
                total += _campaign.plan_and_validate(svc, plan)
            except ServiceError as exc:
                print(f"refused: {plan.label}: {exc.message}", file=sys.stderr)
                return 1
        print(f"{len(plans)} subjects, {total} units -> {db_path}")
        if args.dry_run:
            print("dry run: nothing written")
            return 0
        for plan in plans:
            result = sweeps_mod.create_sweep(svc, plan)
            for job in store.sweep_jobs(result["id"]):
                jobs_mod.update_job(svc, job["id"], {"tags": [tag]})
            print(f"queued {result['units']} units as sweep {result['id']} ({plan.label})")
        print(f"\n{total} jobs queued, tagged {tag!r}.")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

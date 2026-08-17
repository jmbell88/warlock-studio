# Reference-source bench: sdxl_cfg vs sdxl_cfg_pag vs flux_klein_distilled

**Date:** 2026-08-17 (evening; same day the PAG recipe and the PickScore
ranking term landed).
**Question:** does either of the two new reference sources — the PAG sampling
recipe over the default's weights, or the FLUX.2 klein distilled architecture
— make better TRELLIS references than the shipped default `sdxl_cfg`?
**Runs:** `~/.warlock/bench/runs/20260817-{170827,171816,174055}-*-core-v2`,
recipes `sdxl-cfg-raw` / `sdxl-cfg-pag-raw` / `klein-distilled-raw` (all
committed), suite core-v2, seeds 42 and 1337 → 80 units per arm, reference
stage only. Zero errors in any arm. The DB was backed up first
(`jobs.sqlite.bak-2026-08-17-bench`).

## Instruments

Per unit, off the job's own params: the composition gate
(`reference_report.ok`, occupancy, components) and the candidate ranker's
terms — including the PickScore preference logit that landed the same day, so
every unit carries a human-preference proxy scored against its composed
prompt. Two caveats stated up front:

* **The components count is noise at this image class.** Median 15–18
  connected alpha regions per image across *all three* arms (speckle, not
  subjects), which floors `composition_score` everywhere. The comparison
  below therefore leans on the gate rate and the preference term; the
  composition means are reported but carry little signal between arms.
* **PickScore is a proxy for "which would a person pick", not for "which
  reconstructs better".** The downstream half — mesh quality — is exactly
  what this bench does *not* measure; see "What is owed".

## Results

| arm | ok rate | preference mean | occupancy mean | median s/unit |
| --- | --- | --- | --- | --- |
| sdxl_cfg (control) | 75/80 (93.75%) | 22.66 | 0.31 | 4.6 |
| sdxl_cfg_pag | 76/80 (95.0%) | 22.47 | 0.31 | 6.1 |
| flux_klein_distilled | 78/80 (97.5%) | 22.71 | 0.20 | 17.6 |

Paired per-unit (same item, same seed):

* **PAG vs control: the control wins.** 55 of 80 pairs prefer the control's
  image (mean diff −0.185); gate flips are a wash (3 fixed, 2 newly refused);
  +34% sampling time. Visually the PAG arm adds *intricacy* — the spot-checked
  crate grew wire mesh and a handle — which is the wrong direction for a
  single-simple-object reference. PAG 3.0 + rescale 0.7 was the literature's
  pairing, not a tuned one; a lower `pag_scale` or dropping the rescale might
  land differently, but that sweep has to earn its GPU time against this
  result.
* **Klein vs control: parity on preference, better at the gate.** 43–37 on
  preference (mean diff +0.05, noise); all three gate flips in klein's favour
  (78/80 vs 75/80). Occupancy is markedly smaller (0.20 vs 0.31 — klein frames
  subjects further from the camera), which costs composition points as scored
  but is fixable by `reference_prep`. The spot check is the striking part:
  klein's crate is a **closed, solid, convex form with a clean silhouette and
  exact adherence** ("bound with iron" actually bound with iron), where both
  SDXL arms drew open slatted crates — thin structures and interior voids,
  the classic TRELLIS failure feed. Cost: ~4× the sampling time (17.6 s vs
  4.6 s median) and offload residency.

## Decisions

* **`DEFAULT_BASE_MODEL` stays `sdxl_cfg`.** PAG loses its own comparison, so
  `sdxl_cfg_pag` remains what it shipped as: an opt-in arm. No flip, and this
  document is the measurement the flip rule would have required.
* **Klein-distilled is promoted to the candidate worth the expensive test.**
  Reference-stage evidence is parity-or-better everywhere except speed. The
  decisive question is downstream mesh quality, which only the model-stage
  bench (≈6–8 h GPU per arm) plus graded verdicts can answer.

## What is owed

1. The model-stage paired run (`--stage model`) of `klein-distilled-raw` vs
   `sdxl-cfg-raw`, graded — the actual gate on any conditioning-parity
   investment in the klein backend (img2img first, per the phase plan).
2. If a PAG retune is ever wanted: sweep `pag_scale` at rescale 0, not the
   3.0+0.7 pairing this run measured.

## Observed in passing

* Every *resident*-pipe bench run (both SDXL arms, and a pre-session
  `playground-fidelity` control run at 609b2be) **hangs at worker shutdown**
  after `[80/80]` — `runtime.shutdown` times out, prints "the worker did not
  shut down cleanly", and the process never exits. The offloaded klein arm
  exits cleanly. Reproduced on pre-session code, so it is a pre-existing
  bench/teardown defect, not this day's regression; the results are unaffected
  (items.jsonl and every job row are complete before the hang). Worth its own
  chase.

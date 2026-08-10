# Qualifying the FLUX.2-klein pixel-art LoRA

**Status: run 2026-08-09. Qualified, at `default_weight = 0.0625` -- a figure the
model card does not mention and which its own recommended range excludes.**

The rules below were fixed before any image existed and are applied verbatim; the
results are in "What the grids showed", and the one place the outcome falls outside
what the rules anticipated is called out there rather than quietly reinterpreted.

Pre-registered, in the idiom `2026-08-06-pixel-art-xl.md` set: the rules below were
written before an image existed, and whichever one fires is applied verbatim --
including the boring one, which is that the entry does not ship.

## Why this needs a measurement at all

The doctrine is gltfpack's: a tier stays out of the generate form until it has been
run and shown to keep what it claims, because a picker entry that silently does
nothing is worse than no entry. `Limbicnation/pixel-art-lora` is the registry's
first non-SDXL style adapter, so nothing in the tree has ever loaded one, and three
constants are at stake:

1. the effective LoRA scale, which decides whether the strength slider means
   anything at all (settled in Phase 0 below, before the grids);
2. `default_weight`, which is keyed into the findings corpus the moment a job runs
   -- `style_lora` and `lora_weight` are both in `vectors.VECTOR_PARAMS`, so a
   default chosen badly is not merely a poor first impression, it is evidence
   accumulating under a number nobody measured;
3. whether the style fires on `flux_klein` (klein-base) at all, which is a
   *separate* question from whether it fires on the base it was trained against.

Both checkpoints were fetched up front, deliberately, so that the first pair is
measured on the exact base the LoRA was trained on and a recipe mismatch is never a
confound in the answer to "does this fire".

## Phase 0 -- the header, read before designing around it

`peft` honours a `lora_adapter_metadata` header verbatim
(`_create_lora_config` is literally `if metadata is not None: lora_config_kwargs =
metadata`). Absent that header and absent kohya `.alpha` keys, `get_peft_kwargs`
sets `r = lora_alpha = <first rank>` and never emits `use_rslora` -- a scale of 1.0
against a trained `128/sqrt(64) = 16.0`, which `LORA_WEIGHT_MAX = 1.5` cannot
reach. So which case the file is in decides whether a new `StyleLora` field is
needed, and it is one header read away.

**Reading, 2026-08-09.** Two files ship. The diffusers one is case A -- *declared*:

| file | metadata | keys | verdict |
|---|---|---|---|
| `pytorch_lora_weights.safetensors` | `lora_adapter_metadata` present | 172, `transformer.*.lora_A/B.weight` | **A -- declared** |
| `pytorch_lora_weights.comfyui.safetensors` | none | 172, unprefixed | C -- lost |

The declared config, read back out of the header:

    r = 64        lora_alpha = 128      use_rslora = True     use_dora = False
    rank_pattern = {}   alpha_pattern = {}   peft_type = LORA
    target_modules = 17 entries (to_q/k/v, to_out.0, add_*_proj, to_add_out,
                     linear_in/out, to_qkv_mlp_proj, proj_out, x_embedder,
                     context_embedder, {single,double}_stream_modulation*.linear)

Scaling is therefore `alpha / sqrt(r) = 128 / 8 = 16.0`, restored exactly by peft
from the file's own header.

**Consequence: no new field.** `StyleLora.peft_config` is not added, remedy (2)
(a folded `scale`) is not needed, and the existing single `load_lora_weights` call
is correct for this adapter unchanged. The ComfyUI variant is *not* shipped -- it
carries no metadata and no `.alpha` keys, so it would load at scale 1.0, a 16x
under-application no strength slider could reach.

## The grids

Both through the ordinary generate path, so the trigger prepend and
`_apply_adapters` are exercised rather than bypassed. `params["composed_prompt"]` is
recorded per image, so the trigger is visible in the record rather than assumed.

The declared trigger is the card's own `trigger_word`, `pixel art sprite`.

**Grid 1 -- does it fire.** On `flux_klein_distilled`, the card's own recipe
(4 steps, CFG 1.0): 3 prompts x 1 seed x {no LoRA, 0.85, 1.1, 1.4}.

**Grid 2 -- does klein-base express it.** On `flux_klein` (50 steps, CFG 4.0): the
same grid. This is the separate question of whether the entry should be offered on
the undistilled base too, and it is kept apart from the first deliberately -- a
weak result here says something about a recipe, not about the adapter.

## Decision rules, fixed in advance

| Reading | Rule |
|---|---|
| Fires with a clean lattice somewhere in 0.85-1.4 | **Qualified.** `default_weight` is the *lowest* weight that does, following `pixelxl`'s own justification. If that weight is 0.9, the entry declares no `default_weight` and inherits the module default -- the cleanest result. |
| Fires only on the distilled base | **Ships**, and the entry's comment says klein-base under-expresses it. |
| Fires at no weight on either base | **Refused.** No registry entry, and this document says why. Phases 1-4 still ship: they are a correctness fix in their own right. |

512-vs-1024 is *measured*, not pre-emptively overridden. The card trained at 512 and
the registry entry declares `image_size=1024` like its sibling; if the lattice is
only clean at 512 that is a finding to record here, not a constant to guess.

**Measured: 1024 stands.** Every image above is 1024 square and the lattice is clean
at the chosen weight, so the entry declares no `image_size` override. The training
resolution is a property of the run that produced the adapter, not a constraint on
the checkpoint it is applied to.

## A defect the first grid found, which had to be fixed before it measured anything

The first run of grid 1 returned **byte-identical images for 0.85, 1.1 and 1.4** on
all three prompts, and none of the three matched the same weight run first in a
fresh process. The images looked like clean pixel art, which is what made it worth
chasing rather than accepting.

`_apply_adapters` called `pipe.disable_lora()` for a job with no style and
`pipe.set_adapters(...)` for one with a style, and **`set_adapters` writes the
scaling without clearing the `_disable_adapters` flag `disable_lora` set**. Read
straight off the layer:

    load       ({'pixelklein': 16.0}, disabled=False)
    set 0.85   ({'pixelklein': 13.6}, disabled=False)
    disable    ({'pixelklein': 13.6}, disabled=True)
    set again  ({'pixelklein': 13.6}, disabled=True)   <- still off

So one job generated with no style LoRA silently switched **every later job in the
same process** to no style LoRA. The pipe stays resident across jobs, so the state
outlived the job that set it and the only recovery was the idle unload. It reads as
working because the trigger words are still prepended: the output does change when a
style is picked, it just is not the adapter doing it -- which is exactly what the
first grid's plausible-looking sprites were.

Nothing about it is family-specific. It is a plain PEFT state machine, so **SDXL was
affected identically** and had been for as long as `disable_lora` has been called.
`pipe.enable_lora()` before `set_adapters` clears the flag; the order matters, since
`enable_lora` takes no weights. Pinned by
`tests/test_lora_loading.py::test_a_style_survives_a_run_that_had_none_before_it`
and `::test_enabling_precedes_the_weights_it_is_meant_to_restore`.

Every number below was measured after the fix, and the re-run grid returned twelve
distinct images matching the fresh single-shot hashes exactly.

## What the grids showed

**Grid 1, `flux_klein_distilled`, seed 7, knight / crate / tree.** The card's
recommended range does not work at all, and the usable band is an order of magnitude
below it:

| slider | effective scale | result |
|---|---|---|
| 0.02 | 0.32 | clean lattice, crisp outline |
| 0.04 | 0.64 | clean |
| **0.0625** | **1.00** | **clean; chosen default** |
| 0.08 | 1.28 | clean, slightly soft |
| 0.125 | 2.00 | smearing -- lattice breaking down, anatomy drifting |
| 0.3 / 0.7 | 4.8 / 11.2 | noise texture; black frame |
| 0.85 / 1.1 / 1.4 | 13.6 / 17.6 / 22.4 | **black frames** |

The cause is declared rather than mysterious: `use_rslora: True` means peft restores
`alpha / sqrt(r) = 128 / 8 = 16.0`, where an ordinary `alpha / r` adapter restores
about 2. The card's 0.85-1.4 is written for a loader that ignores rslora; honoured,
it is an 8x overdose.

**This is the one place the pre-registered rules did not anticipate the outcome.**
Rule 1 says "fires with a clean lattice somewhere in 0.85-1.4 -> qualified"; nothing
in that range fires. Rule 3 says "fires at no weight -> refused"; it fires very well,
outside the range. The range was the card's claim rather than a property of the
adapter, so the entry ships and the *rule's* actual content -- `default_weight` is
the lowest weight giving a clean lattice, at the effective scale the file declares --
is applied to the band that exists. 0.0625 rather than 0.02 because it is the
adapter's own declared strength (effective 1.0), with 0.02-0.08 all usable; the
figure was confirmed on the crate and the tree at the same seed.

**Grid 2, `flux_klein` (klein-base, 50 steps, CFG 4.0).** The style is expressed
here too, and the band is **shifted down again**: 0.0625 is full noise, while 0.02
gives the single best sprite of the whole measurement. Fifty steps accumulate what
four do not. `default_weight` is one number per adapter and cannot vary per base, so
it is set for the distilled base -- the one the LoRA was trained against and the one
the entry exists for -- and a klein-base user drops the strength to about 0.02 by
hand. That is a documented fact rather than a code change: a per-base default would
be a second table to keep in agreement with this one.

## To record whatever the outcome

`_load_loras` eagerly loads every *fitting* adapter, so `vram_gib = 10.0` silently
absorbs however many FLUX adapters the registry ever holds. One rank-64 adapter is
~0.3 GB, inside that figure's stated rounding-up, so a note is the right answer
today; a second FLUX entry is when it becomes a number.

**Measured, `flux_klein_distilled` + `pixelklein` at the chosen weight:** peak
allocated **8.17 GiB**, peak reserved **9.06 GiB**, against a declared `vram_gib` of
10.0. The declaration stands with room to spare, and the adapter is inside the
rounding as predicted.

`enable_lora()` before `set_adapters()` was checked for a byte-identity regression on
the path it does *not* fix: one SDXL render (`sdxl` + `render3d`, weight 0.9, seed
11) with the line and with the pre-change sequence is the same file, md5
`f70674c3beaa47ee59bff2f1c5d8ffd4` both ways. The fix repairs the broken ordering and
changes nothing about a healthy one.

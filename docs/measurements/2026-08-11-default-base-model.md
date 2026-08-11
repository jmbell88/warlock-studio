# The shipped default base model, 2026-08-11

**Status: decision, taken on already-recorded data.** Nothing new was run for
this document. It reads the arms of
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) -- which this follows and
does not supersede -- and turns them into a distribution choice. It is not
corpus-keyed: no stored verdict, grade or threshold is computed from the value
decided here, so changing it later costs a re-read of this page and nothing on
disk. That is why it may be taken without a fresh sweep, unlike `trellis_band`
or the grade scale.

## The decision

`models.DEFAULT_BASE_MODEL` moves from `turbo` to `sdxl_cfg`, and
`stabilityai/stable-diffusion-xl-base-1.0` becomes the one base download the
onboarding documents tell a new host to fetch. The full ten-entry registry is
unchanged and stays as the in-app install catalog.

`models.T2I_DIR_MODEL = "turbo"` was added first, in a separate step, so that
the legacy `WARLOCK_T2I_DIR` override keeps pointing at the entry it was
documented against. Tying that redirect to whichever key happens to be the
default would have moved an existing setup's override onto a different
checkpoint silently, which is the one way this change could have corrupted a
working install.

## What the measurement says

From the 2026-08-09 re-baseline's marginal accept rates (confounded and
underpowered, and reported there as evidence for choosing settings rather than
as a winner claim):

| Arm | Accept rate |
| --- | --- |
| `base_model=sdxl_cfg` | 3/3 · 100% |
| **baseline** (`sdxl`, no LoRA) | 2/4 · 50% |
| `base_model=turbo` | 1/5 · 20% |

`base_model=turbo` is named in that document as *the weakest checkpoint that
produced data*. `sdxl_cfg` is the best survivor with data. n is tiny on both
sides and no significance is claimed; what is claimed is that nothing in the
record argues for keeping the weakest arm as the default.

### The opposition, stated out loud

Refusal rate and mesh quality rank the two checkpoints **oppositely**. Both of
the re-baseline's two composition refusals fell on SDXL-family arms at full CFG,
and the 2026-08-07 rogue sweep's 17-of-100 refusals were the same mechanism at
larger n. A full-CFG SDXL base produces more images the composition gate turns
away, and better meshes out of the ones it lets through.

This picks quality, deliberately. A refusal is a sentence naming a thing the
user can change, delivered in seconds. A poor mesh is two minutes of the serial
GPU worker followed by a discard, and -- before the birefnet matte -- was the
failure mode `hole_worst` scored as perfect. Trading refusals for survivors is
the cheap direction.

## Why this one is also the distribution answer

The registry entry chosen as default decides what a new install is told to
download, and the SDXL 1.0 weights are the only entry where that download is
shared:

* **7.0 GiB, four recipes.** `sdxl`, `sdxl_cfg`, `pixel` and `lightning` all
  name `dir_name="sdxl-base-1.0"` and the same `_SDXL_BASE_1_0` fetch record.
  `fetch.plan` dedupes on `(repo_id, destination)`, so having the default
  installed leaves each of the other three one small LoRA away: Hyper-SD 0.8 GB
  for `sdxl`, LCM 0.4 GB for `pixel`, Lightning 0.4 GB for `lightning`.
* **Pixel sheets: +0.2 GB.** The restyle is pinned to `sdxl_cfg` plus the
  `pixelxl` style LoRA, and that LoRA is the only missing piece on a host that
  has the default.
* **Sprite sheets: +3.5 GB adapter, +2.5 GB ControlNet.** IP-Adapter Plus and
  the Canny ControlNet, over the same base -- about 13.2 GB total on a fresh
  host once the shared 7 GiB is counted once, not the 28 GB a key-keyed sum
  would quote.

SDXL-Turbo, by contrast, is 7 GiB that unlocks exactly one recipe: its
`dir_name="sdxl-turbo"` is claimed by no other entry. It remains registered, is
still the right answer when four-step latency matters more than fidelity, and
keeps its own documented download command in `docs/MODELS.md`.

## What changes for an existing host

A host that has only SDXL-Turbo downloaded and has not set
`WARLOCK_T2I_MODEL` will have its first defaulted job **refused**, by
`service.validation.check_base_model_weights`, naming SDXL 1.0. That refusal is
correct and actionable -- it leads with Settings → Models and carries the
`hf download` line -- and it is recoverable in two ways that cost nothing:
install the 7 GiB, or pick "SDXL-Turbo (fast)" in the base-model picker, which
pins `base_model` into the job's params and never consults the default again.

Nothing on disk is invalidated. Jobs already in the store carry a resolved
`base_model` in their params, so no stored row's provenance changes meaning, and
no re-run of an existing row silently switches checkpoints.

## Not decided here

The VRAM cost is unchanged: both entries declare `vram_gib = 7.0` under
`RESIDENT`, so admission control (`vram.estimate`) charges a defaulted job
exactly what it charged before and no host gains or loses a fit. What *does*
change is wall-clock -- 30 steps at 1024px with real CFG against 4 steps at
512px -- and that is a cost this document accepts without measuring, because it
is a property of the recipes as registered and was never in question.

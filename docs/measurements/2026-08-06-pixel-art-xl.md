# Pixel-art profile: the comparison, 2026-08-06

**Status: procedure written, run not yet taken.** Everything below the "What
will be run" heading is a pre-registration -- the decision rules were written
*before* any image was generated, which is the only thing that makes the answer
worth anything. When the run happens, the numbers go under "Results" and
whichever rule fired is applied verbatim, including if it is the boring one.

This document exists for the same reason
[`2026-08-04-hole-rate-baseline.md`](2026-08-04-hole-rate-baseline.md) does: a
constant the stored corpus is keyed on gets a measurement before it changes, not
an argument. Two things here are that kind of constant.

## The two questions

1. **Which arm the `pixel_sprite` preset should name.** `pixel-art-xl` is a
   style LoRA, so it can run on any SDXL-class base. Two are plausible and both
   ship:

   - `pixel-lcm-xl` -- SDXL 1.0 under an LCM adapter at 8 steps, guidance 1.0.
     This is the recipe the LoRA's author documents, and it is what the preset
     names today. It is the **null hypothesis**: a new profile starts at the
     author's own settings and has to be beaten off them.
   - `pixel-hyper-xl` -- the same LoRA on the Hyper-SD backend every other style
     LoRA in the registry runs on. Stacking a 1.0-weight step-distillation LoRA
     under a 1.2-weight style LoRA is genuinely unproven, which is exactly why
     it is an arm rather than an assumption.

   A third arm, `baseline-turbo-raw`, is the control: no pixel LoRA at all,
   which is what "Warlock generates a picture and downscales it" scored before
   any of this existed.

2. **Where `pipelines.pixel.GRID_RESIDUAL_MAX` belongs.** It currently sits at
   0.05 and that number is a guess. It is the threshold that decides whether an
   export reduces on the generator's own lattice or falls back to the legacy
   crop-then-scale path, so it is read on every pixel export of every asset --
   and it is keyed into every manifest entry written since. It should be set
   from the observed separation between the two populations, not from taste.

## What will be run

```
uv run python -m warlock.bench run --suite pixel-v1 --recipe pixel-lcm-xl --stage reference
uv run python -m warlock.bench run --suite pixel-v1 --recipe pixel-hyper-xl --stage reference
uv run python -m warlock.bench run --suite pixel-v1 --recipe baseline-turbo-raw --stage reference
uv run python -m warlock.bench score <each run dir>
uv run python -m warlock.bench score <run B> --against <run A>
```

The A/B is `score --against`, not a `compare` subcommand: there is no such
subcommand and there never was. `python -m warlock.bench --help` lists the whole
surface — `suites`, `recipes`, `suite`, `run`, `score`, `calibrate`, `prune`,
`purge`. Two `--against` calls give the two comparisons this run needs, each
pixel arm against the `baseline-turbo-raw` control.

Reference stage, not model: the question is entirely about the picture, and a
TRELLIS reconstruction of a 32-colour sprite would add two minutes per unit and
answer nothing that was asked. `pixel-v1` is 10 items x 4 seeds = 40 units per
arm, 120 in total.

The metrics are `bench/metrics.REFERENCE_METRICS`, measured on the raw
`input.png` at full resolution:

| Metric | What it says |
| --- | --- |
| `pixel_grid_residual` | How much of the image's change happens *between* cells rather than inside them. 0 is a perfect lattice; ~1 is a smooth render. **The headline number.** |
| `pixel_grid_scale` | The detected cell size, or null when nothing passed the threshold. |
| `pixel_colors_128` / `_64` | Distinct colours surviving a reduction to 128 / 64 px. |
| `pixel_orphans_128` | Fraction of pixels with no neighbour of their own colour at 128 px -- reduction noise, the thing that reads as mush. |

They are measured on the **generation**, deliberately never on the exported
pixel artifact: the export's own palette mapping and orphan cleanup would
launder precisely the failures being measured, and all three arms would score
the same.

## Decision rules, written in advance

**On the preset's base.** The `pixel_sprite` preset keeps naming `pixel` (the
LCM arm) unless `pixel-hyper-xl` wins on *both* of:

- lower mean `pixel_grid_residual`, and
- no worse mean `pixel_orphans_128` (within 1 standard error).

Ties go to the LCM arm, because it is the author's documented recipe and the
one already shipped. If Hyper wins both, `guidance.PRESETS["pixel_sprite"]`
changes its `base_model` to `sdxl` in the same commit as this document's
Results section, and the registry keeps both entries either way -- a losing arm
is still a base a user may pick.

**On `GRID_RESIDUAL_MAX`.** Both pixel arms should cluster near 0 and the
control near 1. The threshold is set to the midpoint of the observed gap,
rounded to one significant figure, provided the gap is at least 5x the wider
population's standard deviation. If the two populations *overlap*, the
threshold is not moved: an overlapping distribution means the residual is not
separating them and a re-tuned number would be false precision. It stays at
0.05 and this document says why.

**On dither and cleanup defaults.** Both stay off regardless of what the run
shows. They are aesthetic choices per asset, and this run measures the
generator rather than anyone's taste. A default change would need its own
document.

**On the control arm.** If `baseline-turbo-raw` scores a detected grid on any
unit, that is a bug in `detect_grid` and not a finding about SDXL-Turbo -- the
threshold work above is suspended until it is explained.

## Results

Not yet taken. The run needs the weights from
[Installation](../manual/39-installation.md#optional-image-models-and-style-loras):
`nerijs/pixel-art-xl`, `latent-consistency/lcm-lora-sdxl` (renamed), and the
SDXL 1.0 base, which is likely already present.

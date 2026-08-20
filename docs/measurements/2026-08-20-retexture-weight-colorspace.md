# The re-texture facing weights were sRGB-encoded, 2026-08-20

## What was wrong

`blender_worker.op_project` bakes three targets per view: `bake` (the recoloured
albedo), `weight` (the facing ratio) and, when depth testing is on, `depthpair`.
All three are created with `bpy.data.images.new(...)`, which defaults an 8-bit
image's colourspace to **sRGB**. Only `depthpair` was corrected to `Non-Color`,
and its comment states the rule in as many words:

> The bake target's colorspace decides how `save()` encodes the PNG. Non-Color
> writes the pair's linear values raw; the default would sRGB-encode them and
> the host's subtraction would compare two bent curves.

`weight` carries `max(0, N·V) × mask_alpha` — a linear data value, not a colour —
and was left on the default. `retexture.assemble` reads it straight back as
linear (`_read(..., "L") / 255.0`) and thresholds it against
`MIN_FACING = 0.15`, which the constant's own docstring calls a *facing ratio*.

`bake` is a colour and is correctly left sRGB. The omission was specific to
`weight`.

## The size of it

Computed from the sRGB transfer function, not measured on a card — the bend is
exact arithmetic, and what it does to the threshold follows from it:

| quantity | intended | as it behaved |
|---|---|---|
| `MIN_FACING` as a linear facing ratio | 0.15 | **0.0196** |
| the angle off-normal that floor drops at | 81.37° | **88.88°** |

So a floor written to discard views more than ~81° off-normal was discarding
only those past ~89° — which is to say, discarding almost nothing. The edge-on
"smear" views the constant exists to reject were contributing to the blend.

The curve also compresses the working range, so every surviving view was
weighted wrongly relative to the others:

| true linear weight | value stored and read back | inflation |
|---|---|---|
| 0.15 | 0.424 | ×2.82 |
| 0.26 | 0.547 | ×2.10 |
| 0.50 | 0.735 | ×1.47 |
| 0.87 | 0.941 | ×1.08 |

The inflation is largest exactly where the weight is smallest, so grazing views
were pulled toward parity with head-on ones: at the extreme, a view contributing
0.15 of a texel was mixed in as though it contributed 0.42. `assemble` does a
weighted mean with no per-view renormalisation, so this lands directly in the
output colour.

## The fix

`weight` joins `depthpair` on `Non-Color`. One line, and the tuple is spelled
out rather than inverted so that `bake` staying sRGB reads as a decision:

```python
if suffix in ("depthpair", "weight"):
    image.colorspace_settings.name = "Non-Color"
```

`MIN_FACING` itself is **unchanged at 0.15**. That is the point: the constant
was always the right number and was never the number being applied.

## What this owes

`docs/measurements/2026-08-08-retexture-bake.md` computes its coverage table
from these same `weight_NN.png` files, so **that table is keyed on the bent
values** and its numbers are not comparable to anything produced after this
change. Coverage should read *lower* and be more honest: views that were being
counted at ~89° off-normal are now dropped at ~81°, so texels whose only
contributor was a grazing view will fall back to the base colour rather than
taking a smeared one.

Re-measuring it needs a real card, real weights and the retexture corpus, so it
is owed rather than done here. Until it is run, treat the 2026-08-08 coverage
figures as describing the old behaviour only.

`tests/test_retexture.py` feeds `assemble` synthetic float weights directly and
so never saw this; the gap is between Blender's save and the host's read, and
nothing in the suite crosses it. That seam is only exercised by the gpu lane.

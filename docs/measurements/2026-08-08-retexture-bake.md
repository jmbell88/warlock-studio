# Is a multi-view projection bake good enough? — 2026-08-08

**Question.** Phase 5's Tier 2 gives a finished mesh a new surface by rendering it flat from six
directions, restyling each render with SDXL img2img, projecting the restyled views back onto the
mesh's own UV atlas and combining them by a facing-weighted mean. Tier 3 — a dedicated UV-space
texture model — is written down as conditional: *only if Tier 2's bake proves insufficient*. This is
the measurement that decides it.

**Verdict, up front.** The bake is **sufficient as a recolour and insufficient as a re-texture**, and
the cause is measured rather than guessed: **coverage**. On four real reconstructions only 36–37% of
the atlas is seen by any of the six views, and the remaining 63% keeps its old colour by design. The
limit is *not* atlas resolution (1024 and 2048 are indistinguishable) and *not* the number of views
(front and back reach 36.1%; the other four add 1.0 percentage point between them). So the indicated
next step is **not** the Tier 3 written in the roadmap — a better texture *model* does not address
coverage. See "What this changes" below.

A second, unrelated finding fell out of the control case and is the more actionable of the two:
**Clay-authored meshes cannot be re-textured at all today**, because `clay/uv.py`'s box projection
stacks every face on the same unit square rather than packing them.

## Harness

`scripts/retexture_probe.py`, run on this machine (RTX 5090, bpy 5.2.0, `sdxl_cfg`, seed 42). It
drives the same four stages the worker does, through the same functions, into a copy of each mesh —
it never writes to `assets/`, for `qualify_tiers.py`'s reason: `retexture_job` rewrites the served
`model.glb`, so driving *it* would consume the corpus the next comparison needs.

Per subject it writes `contact.png` (six flat views before, the six restyled renders, then six flat
views **of the re-textured mesh** — not of the restyled renders, which would be grading the bake on
its own input), the two atlases side by side, and `results.json`. Corpora are under
`docs/measurements/data/retexture-{trellis,clay,2048}/`; the PNGs are gitignored and the JSON is not.

Subjects: the four trellis reconstructions in `assets/`, all of the same prompt ("a weathered wooden
crate bound with iron") at 223k–248k triangles, re-textured with *"rusted iron, oxidised metal
plating, heavy corrosion"* at strength 0.55. One subject was repeated at a 2048 atlas. A
Clay-authored crate was run as an unwrap control at strength 0.85.

## What was measured

| Subject | Triangles | Coverage | Mean abs. change vs old atlas | Texels changed |
|---|---|---|---|---|
| `3e32ea1a8533` | 228 508 | 37.1% | 15.9 / 255 | 70.8% |
| `8f20f18dee70` | 248 278 | 35.8% | 13.5 / 255 | 66.3% |
| `ab98133935cb` | 223 044 | 37.3% | 15.9 / 255 | 73.3% |
| `f0a5fe72492e` | 245 968 | 37.2% | 16.4 / 255 | 77.8% |
| `f0a5fe72492e` at 2048 | 245 968 | 37.2% | 16.5 / 255 | 77.6% |

Coverage against view count, cumulative, on all four (the figures below are `f0a5fe72492e`; the
other three agree to within 0.3 pp at every step):

| Views | front | +back | +right | +left | +top | +bottom |
|---|---|---|---|---|---|---|
| Coverage | 18.2% | 36.2% | 36.7% | 37.2% | 37.2% | 37.2% |

Each individual view covers 18.0–18.2% with a mean facing weight of 0.121–0.125 — six near-identical
figures, and after the second view the union stops growing.

## Reading it

**The mechanism behind 37% is the silhouette mask, not the facing test.** A view's weight is
`max(0, N·view) × the render's own alpha`, and that alpha is what says "this texel was actually
visible in this picture". These reconstructions are heavily perforated (the corpus that produced
`hole_worst`'s inversion), so most interior wall area projects onto a *hole* in the render and is
masked out. That is why the third through sixth views add almost nothing: they are not blocked by
each other, they are blocked by the same holes.

**That also explains what the contact sheets show.** The restyle is vivid and inventive — it puts
gears, rings and bolt heads into every view — and the re-textured mesh keeps almost none of it. Two
things happen at once: 63% of the atlas never hears the restyle at all and keeps the old wood, and
the detail the restyle invented is *geometry-shaped*, so the third of the atlas that does get
repainted receives it as high-frequency noise averaged across grazing projections. The net effect
reads as a desaturation of the original: a mean absolute change of 13–16/255 across roughly 70% of
the texels, which is a shift in tone rather than a new material.

**Atlas resolution is not the limit.** Baking `f0a5fe72492e` at 2048 — its own native atlas size —
produced a contact sheet indistinguishable from the 1024 run and moved the delta by 0.1/255. Higher
resolution cannot help a texel no view spoke about.

It did, however, expose a defect in the default. `TEXTURE_PX` was a flat 1024 while a trellis atlas
is 2048, so every run resampled the **base** atlas down by half — and the base is what the uncovered
63% keeps. Invisible in these renders and wrong in principle: a re-texture must leave the part it
did not touch exactly as it found it. The default is now the mesh's own atlas size
(`retexture.atlas_size`), with the fixed sizes offered as an override.

**The unwrap control was invalid, and finding out why is the more useful half.** A Clay-authored
crate was run to separate "the bake is weak" from "trellis's UVs defeat it". Its coverage came back
100%, split as 86.8% from one view and 13.2% from another with the other four at exactly zero — a
perfect partition, which no facing-weighted bake produces. `clay/uv.py`'s box projection is
documented as allowing overlapping islands, and its own docstring says the primitives that can pack
cheaply "answer it in their own generator instead". They do not: `primitives.box()` emits 24 corners
with **4 distinct UV values** — all six faces stacked on the same unit square. Every face bakes into
the same texels and the last fragment wins, per view.

So the control measured Clay's unwrap, not the bake. It is worth stating plainly what that means for
the roadmap: Phase 5's preamble names Clay25 (generator UVs + box unwrap) as the hard prerequisite
for texturing authored geometry, and the prerequisite is only half met. A `.wblk` asset has UVs and
cannot carry a texture.

Two smaller observations from the same runs, recorded because each cost a run to learn. img2img over
a **flat, untextured** render does almost nothing at strength 0.55 — the Clay control came back
pixel-for-pixel white — and at 0.85 it invents plausible structure (planks, corner brackets, rust
streaks) while keeping the albedo white, because a flat white init has no colour information to
depart from. And the bake itself is faithful where it has something to bake: in that 0.85 run every
plank seam and bolt mark from the restyled views is legible in the re-rendered mesh, which is the one
positive control available and says the projection, the weight masking and the atlas swap are sound.

## What this changes

Tier 3 as written in `NEXT_ROADMAP.md` — "a UV-space diffusion or material-decomposition model" — is
**not** indicated by this measurement. It would produce a better restyle for the third of the atlas
that is already being painted and would do nothing about the two thirds that are not. Three things
are indicated instead, in this order:

1. **Pack Clay's box unwrap.** It is the smallest change, it is a prerequisite the roadmap already
   names, and it turns a feature that silently does nothing on authored meshes into one that works.
   The 0.85 control shows the bake is ready for it.
2. **Attack coverage, not fidelity.** More axis views are measured not to help. What would: weighting
   by *visibility* rather than by facing (a depth test from the camera, which also removes the stated
   overhang smear), and views chosen from the mesh rather than from a fixed basis.
3. **Only then reconsider a texture model**, and against a corpus whose coverage is high enough for
   the difference to be attributable to the model.

None of the above blocks shipping Tier 2. A recolour is a real thing to be able to do, the panel
reports its own coverage so a user is told when most of a mesh kept its old skin, and the
`occlusion_tested: false` flag rides in the record. What it must not do is be described as more than
it is.

## Reproducing

```
uv run python scripts/retexture_probe.py --out docs/measurements/data/retexture-trellis \
    --subject <job-id>="rusted iron, oxidised metal plating, heavy corrosion"
```

Fixed seed 42, `sdxl_cfg`, 1024 px views. `--texture-px` overrides the atlas size and `--strength`
the restyle. The coverage-versus-view-count table is computed from the `weight_NN.png` files a run
leaves behind, with no GPU.

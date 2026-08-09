# `SEAM_MAX` on a CFG base, 2026-08-09

**Status: procedure written, run not yet taken.** A pre-registration, and
deliberately a thin one. The *method* is inherited wholesale from
[`2026-08-08-seam-threshold.md`](2026-08-08-seam-threshold.md) and restating it
here would be ceremony. Two paragraphs are genuinely new -- the refusal rule and
the reconciliation rule -- and they are exactly what an after-the-fact write-up
would fudge.

## The question

`pipelines/seam.SEAM_MAX` was closed at **3.5** on a 72-unit corpus, and the
document that closed it names its own limitation: **one checkpoint only**,
sdxl-turbo at 4 steps. A CFG base draws harder edges, and the metric is a ratio
of edge energy across the wrap seam to edge energy inside the tile. So the
question is not "is 3.5 right" but "does the ratio still separate seamless tiles
from seamed ones when the tiles themselves are sharper".

## What will be run

The same two harnesses, on `sdxl_cfg` (30 steps, guidance 7.0, the same
`sdxl-base-1.0` weights as `sdxl`, and in `models.tile_bases()` so
`calibrate_seam.py`'s own gate admits it):

```
uv run python scripts/calibrate_seam.py      --out docs/measurements/data/seam-cfg --base sdxl_cfg
uv run python scripts/calibrate_seam_hard.py --out docs/measurements/data/seam-cfg --base sdxl_cfg
```

8 materials x 3 seeds x tiled/plain = 48, plus 8 hard-structured x 3 seeds tiled
= 24. **72 units, mirroring the turbo corpus exactly.**

**`--out .../seam-cfg`, never `.../seam`.** The two scripts share an output
directory on purpose -- the hard batch is a second half of one corpus, which is
what `calibrate_seam_hard.py`'s docstring licenses -- but they write
`{material}-s{seed}-{arm}.png` with **identical filenames across checkpoints**,
because the base model appears only inside the results JSON and never in a path.
Writing this run into `data/seam` overwrites all 125 turbo files and both result
JSONs, and the measurement that closed `SEAM_MAX` at 3.5 becomes unreproducible
with nothing on screen to say so.

## Decision rules, written in advance

**The method, restated only as far as it binds.** Sort both populations, locate
the empty band between them, take its **geometric** centre (the metric is a
ratio, so the midpoint is multiplicative), and pick a round value inside it.
Every tiled unit scoring above the incumbent 3.5 is eyeballed through
`seam.wrap_preview` -- rolled by half -- **before** it counts as a false alarm.
That is the turbo run's own procedure and it is why `mosaic-s13` at 2.500 and
`corrugate-s11` at 2.237 were correctly read as legitimately seamless grout and
ridge cases rather than misses.

**The refusal rule.** If the CFG populations *overlap* -- any tiled unit scoring
above the lowest **visibly seamed** plain unit -- **the constant does not move**,
and this document records that the ratio does not separate the two populations on
a CFG base. That is a finding about the metric rather than about the threshold,
and it is a more useful one. A re-tuned number in the middle of one population is
false precision, which is `2026-08-06-pixel-art-xl.md`'s rule applied to a
different metric.

**The reconciliation rule, and it is the reason this needs pre-registering at
all.** Two outcomes, decided now:

- If the turbo empty band and the CFG empty band **overlap**, `SEAM_MAX` moves to
  a round value in the intersection.
- If they are **disjoint**, the threshold does **not** become per-checkpoint. One
  number, one stored `threshold` field, one meaning of `seamless` -- a
  per-checkpoint threshold is two spellings of one fact and drifts the first time
  a base is added. It takes the **larger** of the two values, on the stated ground
  that for an advisory gate a false alarm on a good tile is worse than passing a
  marginal one, and this document states the cost: seamed tiles on the softer
  base go unflagged.

**Null, stated in advance.** A higher tiled ceiling on a CFG base is the
*expected* outcome and is not on its own evidence to move anything. The number
moves only if a **wrap-preview-confirmed seamless** CFG tile lands above 3.5.

**Nothing on disk is reinterpreted either way.** `seam.report` stores the
`threshold` it judged against beside `seamless`, and `inspector.seam_verdict`
reads the stored one, so a change is prospective by construction.

## Results

Not yet taken.

# Does visibility weighting fix the re-texture, and what defaults does it earn? — 2026-08-15

**Question.** The 2026-08-08 measurement's prescribed next step was "attack coverage, not fidelity:
weighting by *visibility* rather than by facing (a depth test from the camera, which also removes
the stated overhang smear), and views chosen from the mesh rather than from a fixed basis". Retexture
v2 built the depth test (a per-view depth render, a depth-pair bake through the same projection as
the colour bake, and a host-side compare), a depth-ControlNet anchor for the restyle passes, and a
10-view basis. This measurement decides the defaults it ships with: the panel checkbox, the strength
default, the view set, and the `DEPTH_EPS_*` / `FEATHER_*` constants.

**Verdict, up front.** The depth test **works and ships on by default**: on a fixture whose hidden
region is known exactly, smear contamination fell from **99.5% to 0.17%** (0.05% with the anchor) —
the pre-registered hard bar was <1%. On real meshes it revealed that the shipped coverage figure was
partly fiction: **~16 percentage points of the old "coverage" was smear** — surfaces painted by
cameras that never saw them. Honest coverage on this corpus is 38% from six views, 41% from ten. The
pre-registered coverage aspiration (≥ +15 pp) is **not met and was mis-aimed**: it assumed occlusion
weighting would *unmask* interior walls that the 2026-08-08 fixtures' render-alpha masking hid, but
those fixtures were deleted in the 2026-08-15 cleanup, and on the character-class subjects measured
here the missing coverage is surface no outside camera can see at all. That finding retires the
greedy view search (step 2's second half) for this corpus and puts UV-space inpainting (Tier 3) back
on the table as the only route past ~45% — this time for the right reason. The strength ladder is
smooth and pathology-free to 0.85 under the anchor; the default moves to **0.65**, inside the
pre-registered expected landing zone.

## Harness

`scripts/retexture_probe.py` on this machine (RTX 5090, bpy 5.2.0, `sdxl_cfg`, seed 42, 1024 px
views, atlas at the mesh's own size), same four stages as the worker through the same functions,
never writing into `assets/`. New for this measurement: `--views 6|10`, `--occlusion` (depth-pair
bake + depth-tested assembly), `--control`/`--control-scale` (depth-ControlNet-anchored restyles),
`--glb` (file subjects), `--occluded-rect` (contamination of a known-hidden region), and a per-view
zread/zsurf agreement diagnostic that doubles as the colorspace canary.

**The 2026-08-08 subjects no longer exist** (repo cleanup, 2026-08-15), so the un-anchored baseline
was re-measured (arm `a0`) on four library meshes at the same seed, strength and restyle prompt
("rusted iron, oxidised metal plating, heavy corrosion"): a hooded adventurer `1e45c9c5172b`
(293 652 tris), a mystical jewel `4a468143c330` (296 862), a stylized dragonborn `61265ce477d7`
(291 312), and a SNES-era rogue `950f84d9247d` (284 294). Character-class geometry, so the corpus
answers a *harder* coverage question than the 2026-08-08 crates did — self-occlusion by cloaks and
limbs rather than perforation. The overhang fixture (`scripts/make_overhang_fixture.py`: a plate
floating in front of a wall, hidden region a known 12.5% of the atlas, run at `--views 6` so exactly
one view speaks) is the controlled case. Data under
`docs/measurements/data/retexture-visibility/`; PNGs gitignored, `results.json` not.

## What was measured

**The overhang fixture (the hard bar).** Contamination = fraction of the known-hidden region whose
colour moved more than 8/255.

| Arm | Contamination | Reported coverage |
|---|---|---|
| Facing-only (shipped behaviour until today) | **99.5%** | 100% |
| + depth test | **0.17%** | 87.5% |
| + depth test + ControlNet anchor | **0.05%** | 87.5% |

The plate hides exactly 12.5% of the atlas, and the depth-tested coverage is exactly 100% − 12.5%.
Depth agreement on the speaking view: mean |zread − zsurf| = 5.7 × 10⁻⁵ of the encoded range —
three orders of magnitude inside `DEPTH_EPS_LO` (2/255 ≈ 7.8 × 10⁻³), so the compare is limited by
the eps band, not by precision, colorspace or registration. **`DEPTH_EPS_LO/HI` and the feather
constants are hereby pinned at their shipped values** (2/255, 6/255, `FEATHER_TOTAL` 0.5, 2 blur
passes); the band is generous by ~100× and produced zero false occlusion at the frontier.

**Real meshes, coverage by arm** (per subject: adventurer / jewel / dragonborn / rogue; mean):

| Arm | Coverage | Mean |
|---|---|---|
| `a0` facing-only, 6 views | 54.3 / 57.8 / 51.5 / 54.0% | **54.4%** |
| `a1` depth-tested, 6 views | 42.1 / 32.3 / 37.4 / 41.1% | **38.2%** |
| `b` depth-tested, 10 views | 45.1 / 36.0 / 40.3 / 44.3% | **41.4%** |

`a0` − `a1` = **16.2 pp of the shipped coverage figure was smear**, per-subject 12.9–25.5 pp — worst
on the jewel, whose glow-shell geometry is exactly the overhang case. The four upper 3/4 diagonals
(`b` − `a1`) buy **+3.2 pp of genuine coverage** (range +2.9 to +3.7) for four more SDXL passes
(~+65% restyle wall-time). Their per-view mean visibility (0.37–0.57) is *higher* than the
side/top/bottom axis views' (0.25–0.43) — they are good cameras whose union simply overlaps
front/back heavily. `coverage_effective` (post-feather) tracks coverage at −5.2 ± 0.3 pp everywhere.

**Anchor scale** (`c1` 0.5 vs `c2` 0.8, strength 0.55): atlas-level deltas identical to within
0.1/255 and 0.6 pp changed-fraction on every subject. The registry default (0.65) stands; the knob
matters to the look of individual restyles (contact sheets kept), not to any figure measured here.

**Strength ladder, anchored, 10 views** (mean over the four subjects):

| Strength | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 |
|---|---|---|---|---|---|
| Mean abs. change /255 | 11.2 | 11.8 | 12.4 | 13.1 | 14.3 |
| Texels changed | 41.9% | 44.6% | 48.0% | 51.4% | 54.6% |
| Restyle seconds (10 views) | 46.8 | 51.2 | 55.5 | 60.6 | 65.6 |

Monotone in both restyle metrics, no discontinuity, no subject diverging from the pack at any rung —
the multi-view mud that made 0.65 the un-anchored ceiling does not appear when every view is held to
the same rendered depth. The anchored pass costs ~2.2× the un-anchored (ControlNet per step).

## Reading it

**What the depth test bought is correctness, not reach.** The pre-registered coverage bar
(≥ +15 pp) assumed the 2026-08-08 mechanism — interior walls masked by the render's own alpha
wherever they projected onto a perforation — would be *unmasked* by visibility weighting. These
subjects have few perforations; their unreached surface is inside hoods, between limbs, under
cloaks, where no outside camera at any angle has line of sight. The per-view visibility means make
the ceiling visible: every candidate direction lands at 0.25–0.62 mean visibility, and adding four
well-chosen diagonals moved the union 3 pp. A greedy mesh-driven view search optimises within that
same ceiling; on this corpus it is **not indicated**. What the old figure called 54% coverage was
41% truth and 13–16 pp of paint on surfaces the camera never saw — and on the fixture, where the
lie is isolated, it was 99.5% wrong exactly where it claimed to be covering.

**Keep-base makes the low number livable.** The unreached 59% keeps TRELLIS.2's own bake — a real
texture, not a void — so honest 41% coverage means "the restyle reaches 41% of the skin", not "59%
of the mesh is broken". That is also why Tier 3 (UV-space inpainting of the unreached texels,
conditioned on the reached ones) is now the *only* remaining coverage lever, and this time the
measurement supports it for the right reason: the 2026-08-08 doc rejected it because coverage was
the bottleneck a texture model cannot fix; today the bottleneck is line of sight, which is exactly
what UV-space synthesis does not need.

**Why the default strength is 0.65 and not more.** The ladder is clean to 0.85, and 0.85 stays
available at the door (its 2026-08-08 positive control is unchanged). But the changed-texel fraction
climbs only ~6.6 pp per rung while wall-time climbs ~9%/rung, the pre-registration expected
0.60–0.70, and above 0.65 the difference is an aesthetic judgement the contact sheets should settle
per asset rather than a default settling it for every asset. 0.65 is also the old *un-anchored*
ceiling — the anchor's job was to make that ceiling comfortable, and it measurably is.

**One number to watch in future runs**: the depth-agreement diagnostic (`visibility[].agree_mean`).
It is the colorspace canary — a Blender upgrade that changes image colorspace defaults would show
here as agreement jumping from ~10⁻⁵ toward 10⁻¹ on every view at once, long before anyone
attributes blotchy output to the wrong cause.

## What this changes

1. **The panel's "Anchor to geometry (depth)" checkbox defaults on** (`texture_panel._form`). The
   un-anchored path stays reachable (uncheck it) and byte-identical, pinned by test — it is this
   document's own baseline arm.
2. **`models.RETEXTURE_DEFAULT_STRENGTH = 0.65`**; the door and worker fall back to it. The shared
   `DEFAULT_IMG2IMG_STRENGTH` (0.45) is untouched — pixel sheets have no anchor.
3. **The 10-view basis stands** as shipped; greedy view selection is retired for this corpus.
4. **`DEPTH_EPS_LO/HI` (2/255, 6/255) and `FEATHER_TOTAL`/blur (0.5, 2) are pinned** by the fixture's
   agreement and contamination figures above.
5. **Tier 3 (UV-space inpaint of unreached texels) is re-opened** as the named next coverage step,
   superseding the 2026-08-08 doc's rejection — the bottleneck it could not fix has been fixed, and
   the one that remains is the kind it exists for.

## Reproducing

```
uv run python scripts/make_overhang_fixture.py --out docs/measurements/data/overhang
uv run python scripts/retexture_probe.py --out <dir> --views 6 [--occlusion] [--control] \
    --glb docs/measurements/data/overhang/model.glb="rusted iron, oxidised metal plating, heavy corrosion" \
    --occluded-rect 0.125,0.25,0.375,0.75
uv run python scripts/retexture_probe.py --out <dir> --assets <data-dir> \
    --subject <job-id>="rusted iron, oxidised metal plating, heavy corrosion" \
    [--views 6|10] [--occlusion] [--control [--control-scale S]] [--strength S]
```

Fixed seed 42, `sdxl_cfg`. The nine subject arms are `a0-baseline6, a1-occlusion6, b-occlusion10,
c1-anchored-s05, c2-anchored-s08, d-ladder-{045,065,075,085}` under
`docs/measurements/data/retexture-visibility/`; the fixture arms are `overhang-{plain,occlusion,anchored}`.

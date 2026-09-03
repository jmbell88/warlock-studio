# Does the shipped default throw away detail the reference shows? — pre-registration, 2026-09-03

**Status: pre-registered 2026-09-03; the machine run follows the same day, the
blind grade is owed.** Everything under "What will be run" and "Decision
rules" is fixed first; numbers go under "Results" afterwards and whichever
rule fires is applied verbatim
([`2026-08-30-art-verdicts-preregistration.md`](2026-08-30-art-verdicts-preregistration.md)).

## Why

Every graded corpus so far has asked whether a mesh is *usable*
([`2026-09-02-trellis-060-props.md`](2026-09-02-trellis-060-props.md): 11 of
22 at the shipped default). None has asked whether it keeps what the
reference shows. Reading the pipeline for that question found that at the
shipped defaults Warlock's own half removes nothing — `mesh_profile=raw`,
`reference_prep=False`, `mesh_retries=0` — and that every loss between
`input.png` and `source.glb` happens inside `trellis-server.exe`, three of
them behind launch flags the app never passed:

- **The exe decimates before Warlock sees the mesh.** A cold res-1024 run
  (`tests/fixtures/trellis_1024_v060.log`) reads `remesh_dc: … F=15210192`
  then `decimate_qem_gpu(target=300000): … F 15210192->297896`. So "Raw (no
  decimation)" was never that; it is the exe's quadric simplify to 300K faces
  at res 1024 (150K at 512). `--decim 0` turns the pass off; a positive grid
  selects the legacy cluster-grid pass. Never passed.
- **The texture is decoded at 512 on a 1024 mesh.** `Config.trellis_tex_res
  = 512` is a pin taken against v0.5.4's auto-tex-res noise; the props
  document's own first decision rule scheduled its re-examination on v0.6.0
  and `TODO.md` P3 carries it as the one surviving human item.
- **The UV atlas is fixed at the exe's default** (2048 at res 1024). `--atlas`
  never passed.
- **Geometry resolution 1536** is priced (`vram.TRELLIS_RES_MULT`) and
  admitted (`validation.ALLOWED_RESOLUTIONS`) but reachable from no form:
  `guidance.PLATFORMS` offers 2d/3d only.

As of 2026-09-03 the first two flags are `Config.trellis_decim` and
`trellis_atlas`, `None` omits the flag, and both are `service.sweeps.SERVER_AXES`
members; `optimize.CUSTOM_MAX` rose from 200k to 2M so a gltfpack budget can
be asked for a million faces once the exe stops throwing them away.

What this does **not** re-ask: the guidance strengths and the token budget
([`2026-09-02-trellis-guidance-sweep.md`](2026-09-02-trellis-guidance-sweep.md),
negative), the band ladder (`config.py`, widening is measured harmful), the
reroll and the hole-closing remesh
([`2026-09-02-hole-audit-vs-grade.md`](2026-09-02-hole-audit-vs-grade.md), 0
of 5 and <0.012). Those are settled instruments for a different question.

## What will be run

**Subjects** — `docs/measurements/corpora/detail-v1.txt`, five props-v1
prompts chosen on two facts the 2026-09-02 runs established: the reference
PNG is byte-identical across submits on this machine (so every delta below is
the reconstruction's, not SDXL's), and each carries fine surface detail with
no open form (so the silhouette audit is a reproducibility check here, not
the question). Treasure chest with iron banding (+4 on v0.6.0), barrel with
iron hoops (+3), weathered skull (+3), knight's helmet with the visor raised
(+3), carved pillar capital with acanthus leaves (1 — the detail-heaviest
subject in the corpus). Seed 42, `text → sdxl_cfg → TRELLIS` at the shipped
defaults except where a rung says otherwise, band auto, gss/gsh/max_tokens
omitted. Submitted by `scripts/campaign_detail.py`, tag `detail-060`,
drained headlessly through the real `studio.runtime.Runtime`.

**Rungs, per subject** — vectors, not OFAT, because two of them are pairs:

| unit | `trellis_decim` | profile | `trellis_tex_res` | `trellis_atlas` | resolution | what it asks |
|---|---|---|---|---|---|---|
| baseline | omit | raw | 512 | omit | 1024 | the shipped default, the control |
| decim0-300k | 0 | custom 300,000 | 512 | omit | 1024 | does gltfpack's simplify keep more than the exe's QEM at the same count |
| decim0-1M | 0 | custom 1,000,000 | 512 | omit | 1024 | does 3× the faces show |
| decim0-raw | 0 | raw | 512 | omit | 1024 | the ceiling: ~15M faces — size, load time, viewer viability |
| tex1024 | omit | raw | 1024 | omit | 1024 | the P3-owed pin re-examination |
| tex1024-atlas4096 | omit | raw | 1024 | 4096 | 1024 | the texture ceiling |
| res1536 | omit | raw | 512 | omit | 1536 | the geometry ceiling (second pass, exclusive mode) |

Six units per subject in the first pass (30, four server-restart groups,
tag `detail-060`), one in the second (5, tag `detail-060-1536`, where the
sweep's *base* is resolution 1536 so its single unit is `expand`'s own
`baseline`). The second pass is separate because at 1536 trellis is priced
at 24 GiB beside the 7 GiB image pipe, and on the 32 GB card that is a WDDM
spill into host commit — the 2026-08-03 crash — rather than a clean refusal,
so those five are submitted and drained under `WARLOCK_VRAM_EXCLUSIVE=1`.
Roughly three hours of card time.

**Machine evidence, free, first** — `scripts/hole_audit_vs_grade.py --tag
detail-060 --corpus docs/measurements/corpora/detail-v1.txt`, extended for
this run with the size of `source.glb` and `model.glb` in MiB beside the
audited face count and the whole-job seconds:

1. `mesh_audit.worst` on every unit, as a reproducibility check: on these
   closed forms it must stay within 0.02 of the subject's corpus value on
   the baseline and must not *rise* past 0.07 on any rung (a rung that opens
   the silhouette is a regression whatever it does for detail).
2. Faces, bytes and seconds per rung. `decim0-raw`'s GLB is expected to be
   hundreds of MiB; whether the viewer opens it is recorded, not assumed.
3. `tiercheck.compare` for the two gltfpack rungs against their own
   `source.glb`: UVs, both PBR maps and the material must survive, the same
   bar `tests/test_generation_tiers.py` and the tier programme set.
4. The trellis log per unit: the first `decim0-*` unit must show no
   `decimate_qem_gpu` line; the first `tex1024` unit `[6/7] res-1024
   texture`; the first `atlas4096` unit `uv_bake: atlas 4096x4096`. If any of
   those is absent the flag did not reach the exe and the run is void.

**Human evidence** — one blind pass in Review on the −5..+5 scale with tags,
and per subject one pairwise call per rung against the baseline, judged in
the viewer at fit-to-view and at 4× on the busiest surface: **more** of the
reference's detail, **same**, or **less**. The pairwise call is the
instrument; the grade is there so a rung that gains detail and loses
usability is seen as both.

## Decision rules

- **`tex1024`**: clean texture (no per-texel noise, the v0.5.4 defect) on 5 of
  5 **and** "more" on ≥3 of 5 → `Config.trellis_tex_res` becomes 1024, citing
  this document, and P3's surviving item is struck. Noise reproduces on any
  subject → the pin stays and the reproduction is recorded here.
- **`tex1024-atlas4096`**: graded only if `tex1024` is clean; becomes the
  default only if "more" over `tex1024` on ≥3 of 5 **and** whole-job time
  ≤1.5× baseline. Otherwise it stays an axis.
- **`decim0-*`**: `trellis_decim = 0` becomes the default **only paired with
  a gltfpack budget** that passes `tiercheck.compare` on 5 of 5 and is "more"
  on ≥3 of 5; that budget then enters `settings_3d.PROFILES` as a named tier
  and becomes `Config.mesh_profile`, with this corpus standing in for the
  chest/sword/rock trio the tier bar names (the chest is here; the document
  says so where the bar is restated). `decim0-raw` is never a default
  candidate — it is recorded for size and load time, and graded as the
  ceiling only if the viewer opens it. No decim rung wins → the flag stays
  `None`, remains an axis, and the negative is recorded.
- **`res1536`**: "more" on ≥3 of 5 and no new audit failure → a third
  `guidance.PLATFORMS` row (`3d_high`, resolution 1536), which the Mesh
  column's platform combo picks up with no other code, and the manual says it
  forces exclusive mode. Time cap 2× baseline. Otherwise recorded, no row.
- Whatever wins, the rig path's ~300k-face constraint and the remesh panel
  are unaffected: both read `model.glb`, and a budget is what a winning decim
  rung ships with.
- The noise floor for "same" on faces is `trellis_band`'s 0.3 %; for the
  audit it is the 0.02 above. Neither is a floor for the pairwise call,
  which has none — it is one reviewer's eye, and the single-reviewer caveat
  of [`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) applies.

## Results — machine evidence

*(appended after the run)*

## Results — grades

*(owed: the blind pass and the pairwise calls)*

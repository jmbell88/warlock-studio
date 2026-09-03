# trellis.cpp v0.6.0 against props-v1 — pre-registration, 2026-09-02

**Status: concluded 2026-09-02 — pre-registered, run on both exe versions and graded blind the same day; the first decision rule fired.** Everything under
"What will be run" and "Decision rules" is fixed first; numbers go under
"Results" afterwards and whichever rule fired is applied verbatim
([`2026-08-30-art-verdicts-preregistration.md`](2026-08-30-art-verdicts-preregistration.md)).

## Why

[`2026-08-30-sdxl-cfg-props.md`](2026-08-30-sdxl-cfg-props.md): 21 of 22
references passed the 2D gate, 16 of those 21 became unusable meshes, and
`holes` was the reviewer tag on 10 of 22. The reconstruction half is where the
corpus is lost, and that baseline was taken on trellis.cpp **v0.5.4**
(`doctor.TRELLIS_EXE_VERSION`, 2026-07-27), whose own release note fixed only
res-1024 speckle. Upstream **v0.6.0** (2026-08-19) is titled "CUDA geometry
fixes, robust BiRefNet, legacy NVIDIA support" and names holes and corrupted
geometry at high resolution as what it fixes. It is the cheapest lever that
targets the failing stage directly, and it has not been tried.

## The bump itself (human-owed, because it is a download)

`vendor/trellis/` is not git-tracked, so replacing it is not reversible from
the tree; the steps are written out so the swap is a copy, not a decision.

1. Fetch `trellis-cuda-windows-x64.zip` from the v0.6.0 release (695 MB). The
   GitHub-published digest, read from the release page on 2026-09-02, is
   `4d08ab27e83094035fd8349aaf34d3460738df0466ef9c4991ddd958c0344bc2`. Verify
   before unpacking. Note that v0.6.0 also publishes a separate
   `trellis-cuda12-windows-x64.zip`; the pinned asset name is unchanged and is
   the one the dev card runs.
2. `src/warlock/doctor.py`: `TRELLIS_EXE_VERSION = "v0.6.0"` and
   `TRELLIS_EXE_SHA256` = the digest above. The comment there is the rule:
   both move together or neither.
3. Replace `trellis-server.exe`, `trellis-cli.exe`, the ggml DLLs and, if the
   release changed them, the CUDA redistributables in `vendor/trellis/`.
4. Regenerate `installer/runtime-manifest.json` (sizes and sha256 per file;
   `installer/verify_runtime.py` consumes it).
5. Version string in `README.md`, `../manual/38-installation.md` and
   `installer/build.ps1`.
6. Capture one cold res-1024 run's stdout as
   `tests/fixtures/trellis_1024_v060.log` and run `tests/test_progress*.py`
   against it: `progress.py`'s `RE_*` are verified against the exe's format
   strings, and a changed stage list moves `TRELLIS_STAGES`.
7. `CHANGELOG.md`.

## What will be run

The 2026-08-30 protocol, unchanged: `docs/measurements/corpora/props-v1.txt`
at seed 42 through `scripts/campaign_props.py`, `mesh_profile=raw`,
`trellis_tex_res=512`, band auto, then a blind grading pass on the −5..+5
scale. The 2D stage is byte-identical across the two exe versions (the
2026-08-13 regression-check protocol), so any delta is the reconstruction.

Retention: the corpus stays in the store until this document's Results are
written. The 2026-08-30 rows were cleaned before
`scripts/hole_audit_vs_grade.py` existed, which is why that script's table is
still empty.

## Decision rules

Baseline: **4 of 16** usable on easy+medium; **10 of 22** tagged `holes`.

- Usable rate on easy+medium rises by at least 4 (to 8/16 or better) **and**
  the `holes` count falls by at least 4: v0.6.0 stays, and the two pins set
  against v0.5.4 behaviour are re-examined next — `trellis_tex_res = 512`
  (reproduce the auto-tex-res noise with `trellis-cli.exe --tex-res 1024` on
  one reference; if clean, a measurement document lifts the pin) and the
  `trellis_band` ladder (left alone; the heuristic won on v0.5.4 and v0.5.4
  already carried the band remesh).
- Either number moves by less than that, or moves the wrong way: v0.6.0 still
  stays (the fixes are upstream's and a same-or-better result is not a reason
  to pin an older binary), but the props-v1 loss is declared **not** an exe
  regression, and the guidance sweep
  ([`2026-09-02-trellis-guidance-sweep.md`](2026-09-02-trellis-guidance-sweep.md))
  is the next instrument.
- Any subject that v0.5.4 reconstructed and v0.6.0 fails outright (server
  error, empty mesh) is listed by name; three or more is a regression that
  blocks the bump regardless of the aggregate.

## Results — machine audit, 2026-09-02

**Both versions were run the same day on the same card** (RTX 5090, 32 GB,
coexist mode), seed 42, `scripts/campaign_props.py` at the shipped defaults,
tags `props-v1-054` and `props-v1-060`, drained headlessly through the real
`studio.runtime.Runtime`. The v0.5.4 pass is therefore a *re-run* of the
2026-08-30 baseline on the same binary, not that run's rows (those were
cleaned before this document existed). The numbers below are
`params["mesh_audit"]["worst"]` — the four-view see-through fraction — and
wall-clock seconds per job (SDXL plus TRELLIS plus optimise), read back by
`scripts/hole_audit_vs_grade.py`. This section is the machine evidence; the
grades and the rules follow it.

| class | subject | v0.5.4 worst | v0.6.0 worst | v0.5.4 s | v0.6.0 s |
|---|---|---|---|---|---|
| easy | a mossy granite rock, rounded and weathered | 0.000 | 0.000 | 202 | 182 |
| easy | a cast iron cauldron with three stubby legs | 0.231 | 0.222 | 259 | 277 |
| easy | a ceramic jug glazed in deep blue | 0.160 | 0.000 | 235 | 303 |
| easy | a wooden barrel bound with iron hoops | 0.450 | 0.005 | 294 | 416 |
| easy | a large ripe pumpkin with a curled stem | 0.396 | 0.000 | 250 | 336 |
| easy | a weathered human skull, bleached bone | 0.305 | 0.017 | 352 | 406 |
| easy | a round loaf of crusty dark bread | 0.000 | 0.000 | 171 | 191 |
| easy | a terracotta amphora with a rounded belly | 0.119 | 0.039 | 237 | 283 |
| medium | a knight's steel helmet with the visor raised | 0.004 | 0.009 | 179 | 185 |
| medium | a blacksmith's iron anvil on a thick wooden block | 0.000 | 0.000 | 167 | 169 |
| medium | a leather drawstring pouch spilling gold coins | — | 0.091 | 20 | 209 |
| medium | a wooden tree stump with thick gnarled roots | 0.170 | 0.014 | 215 | 326 |
| medium | a brass hand bell with a turned wooden handle | 0.000 | 0.000 | 86 | 86 |
| medium | a carved stone pillar capital with acanthus leaves | 0.275 | 0.000 | 678 | 772 |
| medium | a hanging oil lantern with glass panes | 0.232 | 0.245 | 86 | 93 |
| medium | a stone well head with a small shingled roof | 0.224 | 0.204 | 233 | 318 |
| hard | a wooden treasure chest with iron banding | 0.530 | 0.008 | 500 | 667 |
| hard | a bundle of dry branches tied with coarse twine | 0.367 | 0.215 | 95 | 154 |
| hard | a three-legged wooden milking stool | 0.000 | 0.002 | 146 | 149 |
| humanoid | a stout dwarf blacksmith in a leather apron | 0.024 | 0.023 | 146 | 150 |
| humanoid | a hooded traveller in a long heavy cloak | 0.000 | 0.000 | 162 | 166 |
| humanoid | an armoured knight standing at attention | 0.009 | 0.025 | 115 | 127 |

The v0.5.4 pouch row is empty because its reference failed the 2D gate ("the
subject runs off the edge of the frame"), as one subject did on 2026-08-30.
On v0.6.0 the same prompt and seed produced a reference that passed. That
was first read as the gate's threshold sitting on a boundary; the
reproducibility pass in
[`2026-09-02-trellis-guidance-sweep.md`](2026-09-02-trellis-guidance-sweep.md)
then showed the pouch's reference PNG differs on *every* submit at the same
seed, as does the branches', while the other subjects' are byte-identical.
So "the 2D stage is byte-identical" does not hold for every prompt, and the
per-subject comparison below is exact only where `input.png` hashed the same
across the two runs. Checked: it did for **18 of 22**. The four that differ
are the pouch and the branches (the two unstable prompts) plus the rock and
the loaf, which audit 0.000 under both versions either way. Every one of the
seven subjects that dropped from 0.12–0.45 to under 0.04 was reconstructed
from a byte-identical reference, so that improvement is the exe's.

| | v0.5.4 | v0.6.0 |
|---|---|---|
| audited | 21 | 22 |
| `worst` above the 0.07 trigger | **12** | **5** |
| median `worst` | 0.160 | 0.008 |
| median seconds per job | 202 | 200 |

Seven subjects that measured 0.12–0.45 on v0.5.4 (jug, barrel, pumpkin,
skull, amphora, tree stump, pillar capital) measure under 0.04 on v0.6.0.
The five still above the trigger — cauldron, lantern, well head, dry
branches, pouch — are within noise of their v0.5.4 values, so the release
fixed one failure class and left another; those five are the guidance
sweep's subjects ([`2026-09-02-trellis-guidance-sweep.md`](2026-09-02-trellis-guidance-sweep.md)).
Time is unchanged. Face counts are unchanged to within the 0.3 % noise floor
`trellis_band`'s sweep established.

**What this does and does not say.** The audit is the silhouette test, and
[`2026-09-02-hole-audit-vs-grade.md`](2026-09-02-hole-audit-vs-grade.md) is
the open question of whether it tracks the reviewer's `holes` tag at all. On
2026-08-30 the reviewer tagged 10 of 22; on the same binary the audit fires on
12 of 21. Until the v0.6.0 corpus is graded, "12 → 5" is a claim about the
audit, not about usable meshes. The pin bump itself stands on the release's
own terms (upstream's fixes, unchanged time, unchanged parser).

## Results — grades, 2026-09-02

One blind pass in Review over both tagged corpora, −5..+5 with tags, read
back by `scripts/hole_audit_vs_grade.py`; the per-mesh join with the audit is
in [`2026-09-02-hole-audit-vs-grade.md`](2026-09-02-hole-audit-vs-grade.md).
Usable is grade ≥ +3.

| class | subject | v0.5.4 grade | v0.6.0 grade |
|---|---|---|---|
| easy | a mossy granite rock, rounded and weathered | 3 | 5 |
| easy | a cast iron cauldron with three stubby legs | −4 | 2 |
| easy | a ceramic jug glazed in deep blue | −5 `holes` | 5 |
| easy | a wooden barrel bound with iron hoops | −5 `holes` | 3 |
| easy | a large ripe pumpkin with a curled stem | −5 `holes` | 4 |
| easy | a weathered human skull, bleached bone | −5 `holes` | 3 |
| easy | a round loaf of crusty dark bread | 5 | 5 |
| easy | a terracotta amphora with a rounded belly | −5 `holes` | 5 |
| medium | a knight's steel helmet with the visor raised | 2 | 3 |
| medium | a blacksmith's iron anvil on a thick wooden block | 3 | 0 |
| medium | a leather drawstring pouch spilling gold coins | — | 1 |
| medium | a wooden tree stump with thick gnarled roots | −5 `holes` | 1 |
| medium | a brass hand bell with a turned wooden handle | 5 | 3 |
| medium | a carved stone pillar capital with acanthus leaves | −5 `holes` | 1 |
| medium | a hanging oil lantern with glass panes | 1 | −3 |
| medium | a stone well head with a small shingled roof | −5 `holes` | 2 |
| hard | a wooden treasure chest with iron banding | −5 `holes` | 4 |
| hard | a bundle of dry branches tied with coarse twine | 2 | −4 |
| hard | a three-legged wooden milking stool | 1 | 3 |
| humanoid | a stout dwarf blacksmith in a leather apron | 2 | 2 |
| humanoid | a hooded traveller in a long heavy cloak | 1 | 1 |
| humanoid | an armoured knight standing at attention | 2 | 1 |

| | v0.5.4 | v0.6.0 |
|---|---|---|
| graded | 21 | 22 |
| usable, all classes | 4 | **11** |
| usable, easy+medium | 4 of 15 | **8 of 16** |
| tagged `holes` | 9 | **0** |
| median grade | −4 | 3 |

Every one of the nine `holes` tags on v0.5.4 is on a subject the audit
measured 0.12–0.53, and all nine grade +3 or better on v0.6.0 except the
tree stump and the pillar capital (1 and 1 — no longer perforated, still not
usable). The three humanoids grade 1–2 on both binaries: the release did
nothing for the single-view figure, and
[`2026-08-30-art-verdicts-preregistration.md`](2026-08-30-art-verdicts-preregistration.md)
says the usability grade is not the instrument for that class anyway. The
anvil is the one subject that fell (3 → 0) from a byte-identical reference
and a 0.000 audit on both — one row, untagged, and not a holes story.

**Decision rule one fires.** Usable on easy+medium rose from 4 to 8 of 16
(bar: at least 8) and the `holes` count fell from 9 to 0 (bar: fall by at
least 4). No subject that v0.5.4 reconstructed fails outright on v0.6.0 (the
third rule's list is empty). v0.6.0 stays, and the two pins are re-examined
per the rule:

**Pins re-examined.** `trellis_band`'s ladder is left alone, as the rule
itself says — the heuristic won on v0.5.4 and the face counts are unchanged
on v0.6.0 to within its 0.3 % floor. `trellis_tex_res = 512` is the one pin
that still owes a run: the rule asks for the auto-tex-res noise to be
reproduced with `trellis-cli.exe --tex-res 1024` on one reference from this
corpus, and a measurement document lifts the pin if the texture is clean.
That is a GPU afternoon and a judgement, and is recorded in `TODO.md` rather
than here; until it runs the pin stands.

**What the number means for the product.** 11 of 22 usable at the shipped
defaults on a representative prop corpus, 8 of 16 on the easy and medium
classes, is the first graded figure ever recorded for the path that ships
(`text → sdxl_cfg → TRELLIS`, `mesh_profile=raw`); the only earlier graded
run was 0 of 20 on deliberately hard subjects through `playground`
([`2026-08-13-tier-qualification.md`](2026-08-13-tier-qualification.md)).
The same afternoon's [`2026-09-02-fantasy-v1.md`](2026-09-02-fantasy-v1.md)
graded 10 of 20 on a second corpus at the same defaults.


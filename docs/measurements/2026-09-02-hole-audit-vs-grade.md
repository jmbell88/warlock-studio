# Does the silhouette audit see the reviewer's holes? — pre-registration, 2026-09-02

**Status: concluded 2026-09-02.** Pre-registered the same morning with an
empty table — the 2026-08-30 props-v1 rows had already been cleaned from the
store when `scripts/hole_audit_vs_grade.py` was written — then run against the
re-run corpus of [`2026-09-02-trellis-060-props.md`](2026-09-02-trellis-060-props.md)
once its blind grading pass was in. The first decision rule fired on v0.5.4,
its own revert clause fired on the reroll re-run, and `Config.mesh_retries`
stays 0 with a graded negative behind it.

## Why

`Config.mesh_retries` — the reroll-and-keep-best loop in `_q_generate` — ships
at 0. Its trigger, `mesh_hole_max = 0.07`, was set by
[`2026-08-04-hole-rate-baseline.md`](2026-08-04-hole-rate-baseline.md) on a
41-unit, single-subject corpus whose hole rate was sharply bimodal, and the
retry was then retired because a reroll is two minutes the user did not ask
for. props-v1 does not resemble that corpus, and its reviewer tagged `holes`
on 10 of 22. Two levers are already built and unmeasured against it: the
retry, and the `close_holes` voxel pass reachable only through a default-off
checkbox on the Remesh panel.

Whether either is worth turning on hinges on one question that costs no GPU:
does `meshaudit`'s four-view see-through fraction, stored on every finished
mesh as `params["mesh_audit"]["worst"]`, agree with the human `holes` tag?

## What will be run

`uv run python scripts/hole_audit_vs_grade.py --tag props-v1` against a store
holding a graded props-v1 pass. It joins the latest human `model`-stage
verdict and prints, per mesh: class, `worst`, `mean`, faces, grade, whether
`holes` was tagged. Nothing is written.

## Decision rules

Let H be the holes-tagged meshes and C the rest, both graded.

- **At least 8 of 10 in H measure above 0.07, and at most 3 of C do**: the
  audit sees what the reviewer sees, and the retired trigger fires on the
  corpus that matters. `Config.mesh_retries` defaults to 1 with this document
  as the reason, and H is re-run with `WARLOCK_MESH_RETRIES=2` to count how
  many the keep-best loop converts to usable (grade ≥ +3). Three or more of
  ten keeps the default; fewer reverts it and records the negative.
- **Fewer than 5 of H measure above 0.07**: the silhouette audit does not see
  the reviewer's holes — the "weak signal" result of the June 2026 VLM-judge
  work reproduced locally. The trigger stays retired, the 24-view audit
  extension stays deferred (more views of a signal that does not correlate
  is not an improvement), and the `close_holes` remesh question is answered
  by grading rather than by audit.
- **In between**: the table is published as the finding, the threshold is
  re-derived from it the way 0.07 was derived from the 2026-08-04 gap (the
  midpoint of the largest empty interval between H and C, if one exists),
  and the first rule is re-applied at that threshold.

## The "Seal holes" question, beside this one

Queue a remesh with `close_holes=True` on the ten in H via
`service.jobs.remesh_job`, regrade blind. `tiercheck.compare` records UV, PBR
and material survival per mesh. If at least 3 of 10 convert to usable, the
Review badge grows a one-click "Seal holes" that submits the same rework
(`widgets.py` mesh badge, through `service/_jobs_rework.py` — no new door);
`tests/test_evidence_clarity.py` pins that the badge's colour semantics do
not change. Fewer than 3: the checkbox stays where it is.

## Results

**Graded 2026-09-02.** One blind pass in Review over the retained rows,
−5..+5 with tags, then `uv run python scripts/hole_audit_vs_grade.py --tag
props-v1-054` and `--tag props-v1-060`. Grade is the human `model`-stage
verdict; `holes` is whether that tag was set; usable is grade ≥ +3
(`vectors.USABLE_GRADE`).

### v0.5.4 — the binary the 2026-08-30 reviewer graded

| class | worst | mean | faces | s | grade | holes | subject |
|---|---|---|---|---|---|---|---|
| easy | 0.0000 | 0.0000 | 287368 | 202 | 3 | | a mossy granite rock, rounded and weathered |
| easy | 0.2315 | 0.0886 | 283664 | 259 | −4 | | a cast iron cauldron with three stubby legs |
| easy | 0.1601 | 0.1214 | 228516 | 235 | −5 | yes | a ceramic jug glazed in deep blue |
| easy | 0.4500 | 0.2666 | 236178 | 294 | −5 | yes | a wooden barrel bound with iron hoops |
| easy | 0.3959 | 0.3474 | 272582 | 250 | −5 | yes | a large ripe pumpkin with a curled stem |
| easy | 0.3050 | 0.2935 | 233632 | 352 | −5 | yes | a weathered human skull, bleached bone |
| easy | 0.0000 | 0.0000 | 289880 | 171 | 5 | | a round loaf of crusty dark bread |
| easy | 0.1191 | 0.0745 | 287278 | 237 | −5 | yes | a terracotta amphora with a rounded belly |
| medium | 0.0041 | 0.0012 | 280040 | 179 | 2 | | a knight's steel helmet with the visor raised |
| medium | 0.0000 | 0.0000 | 287844 | 167 | 3 | | a blacksmith's iron anvil on a thick wooden block |
| medium | — | — | — | 20 | — | | a leather drawstring pouch spilling gold coins (2D gate) |
| medium | 0.1703 | 0.1646 | 241934 | 215 | −5 | yes | a wooden tree stump with thick gnarled roots |
| medium | 0.0000 | 0.0000 | 298232 | 86 | 5 | | a brass hand bell with a turned wooden handle |
| medium | 0.2750 | 0.2151 | 197502 | 678 | −5 | yes | a carved stone pillar capital with acanthus leaves |
| medium | 0.2325 | 0.1294 | 282688 | 86 | 1 | | a hanging oil lantern with glass panes |
| medium | 0.2237 | 0.1510 | 242834 | 233 | −5 | yes | a stone well head with a small shingled roof |
| hard | 0.5303 | 0.3343 | 266810 | 500 | −5 | yes | a wooden treasure chest with iron banding |
| hard | 0.3671 | 0.3476 | 288486 | 95 | 2 | | a bundle of dry branches tied with coarse twine |
| hard | 0.0001 | 0.0000 | 277342 | 146 | 1 | | a three-legged wooden milking stool |
| humanoid | 0.0239 | 0.0061 | 299422 | 146 | 2 | | a stout dwarf blacksmith in a leather apron |
| humanoid | 0.0004 | 0.0002 | 279582 | 162 | 1 | | a hooded traveller in a long heavy cloak |
| humanoid | 0.0093 | 0.0056 | 282406 | 115 | 2 | | an armoured knight standing at attention |

| | |
|---|---|
| audited / graded | 21 / 21 |
| H (`holes` tagged) | **9**, of which **9** above 0.07 (worst 0.119..0.530) |
| C (not tagged) | 12, of which **3** above 0.07 — cauldron, lantern, branches |
| usable (grade ≥ +3) | **4 of 21** (rock, loaf, anvil, bell — every one audits 0.000) |

### v0.6.0 — the shipped binary

| class | worst | mean | faces | s | grade | holes | subject |
|---|---|---|---|---|---|---|---|
| easy | 0.0000 | 0.0000 | 297896 | 182 | 5 | | a mossy granite rock, rounded and weathered |
| easy | 0.2210 | 0.0830 | 103640 | 277 | 2 | | a cast iron cauldron with three stubby legs |
| easy | 0.0000 | 0.0000 | 276440 | 303 | 5 | | a ceramic jug glazed in deep blue |
| easy | 0.0055 | 0.0032 | 238724 | 416 | 3 | | a wooden barrel bound with iron hoops |
| easy | 0.0000 | 0.0000 | 290570 | 336 | 4 | | a large ripe pumpkin with a curled stem |
| easy | 0.0166 | 0.0107 | 285012 | 406 | 3 | | a weathered human skull, bleached bone |
| easy | 0.0000 | 0.0000 | 284684 | 191 | 5 | | a round loaf of crusty dark bread |
| easy | 0.0391 | 0.0118 | 293834 | 283 | 5 | | a terracotta amphora with a rounded belly |
| medium | 0.0091 | 0.0025 | 289208 | 185 | 3 | | a knight's steel helmet with the visor raised |
| medium | 0.0000 | 0.0000 | 298832 | 169 | 0 | | a blacksmith's iron anvil on a thick wooden block |
| medium | 0.0921 | 0.0342 | 185340 | 209 | 1 | | a leather drawstring pouch spilling gold coins |
| medium | 0.0141 | 0.0066 | 294522 | 326 | 1 | | a wooden tree stump with thick gnarled roots |
| medium | 0.0000 | 0.0000 | 288986 | 86 | 3 | | a brass hand bell with a turned wooden handle |
| medium | 0.0004 | 0.0001 | 295406 | 772 | 1 | | a carved stone pillar capital with acanthus leaves |
| medium | 0.2408 | 0.1339 | 136588 | 93 | −3 | | a hanging oil lantern with glass panes |
| medium | 0.2049 | 0.0801 | 349592 | 318 | 2 | | a stone well head with a small shingled roof |
| hard | 0.0075 | 0.0019 | 271884 | 667 | 4 | | a wooden treasure chest with iron banding |
| hard | 0.2033 | 0.1316 | 69424 | 154 | −4 | | a bundle of dry branches tied with coarse twine |
| hard | 0.0020 | 0.0005 | 293276 | 149 | 3 | | a three-legged wooden milking stool |
| humanoid | 0.0227 | 0.0058 | 296408 | 150 | 2 | | a stout dwarf blacksmith in a leather apron |
| humanoid | 0.0000 | 0.0000 | 289204 | 166 | 1 | | a hooded traveller in a long heavy cloak |
| humanoid | 0.0251 | 0.0120 | 283416 | 127 | 1 | | an armoured knight standing at attention |

| | |
|---|---|
| audited / graded | 22 / 22 |
| H (`holes` tagged) | **0** |
| C above 0.07 | 5 — cauldron, pouch, lantern, well head, branches; graded 2, 1, −3, 2, −4 |
| usable (grade ≥ +3) | **11 of 22**; easy+medium **8 of 16**; usable meshes audit 0.000..0.039 |

The cauldron, lantern, well head and branches (and on v0.6.0 the pouch) are
the five the reviewer graded low *without* tagging `holes` on either binary:
what is wrong with them is not a perforated skin. Their face counts on v0.6.0
are the `close_holes` remesh's, at half the source (see below); the reviewer
graded the sealed meshes, as that section said they would.

### Applying the rules

**Rule one fires, on v0.5.4.** 9 of 9 in H above the trigger against a bar
of 8 of 10; 3 of 12 in C above it against a bar of at most 3. The silhouette
audit sees what the reviewer calls `holes`, to the mesh — the June 2026
"weak signal" result does *not* reproduce here, and the 24-view extension
question is moot because four views already separate H from C with three
false positives, all of them open forms (legs, panes, loose sticks).

**Rule one's second clause then reverts it.** The clause reruns H with
`WARLOCK_MESH_RETRIES=2` and keeps the default at 1 only if at least three
of ten convert to usable. H on the shipped binary is *empty* — v0.6.0 fixed
all nine — so the reroll was run on what the trigger still fires on there,
the five open forms (tag `retry-060`, table below): the keep-best loop moved
the audit by 0.006–0.090 and the reviewer graded the kept meshes **1, 1, 1,
1, −1 — 0 of 5 usable**. Fewer than three reverts, and **`Config.mesh_retries`
stays 0**, now on a graded negative rather than the audit-only one recorded
below the same morning. The threshold 0.07 itself is confirmed rather than
re-derived: the largest empty interval between H and C on v0.5.4 runs from
0.0239 to 0.1191, and 0.07 sits inside it.

What the join adds to the retirement argument: the trigger is a good
detector of the failure v0.6.0 no longer produces, and on the failure that
remains — an open form the reconstruction reads as solid-with-gaps — it fires
on exactly the subjects a reroll cannot repair. [`2026-09-02-fantasy-v1.md`](2026-09-02-fantasy-v1.md)
adds that on open forms it also fires on usable meshes (market stall 0.231
and fountain 0.097, both graded +3): as a *reroll* trigger it would now cost
six minutes on subjects that are either fine or unfixable, which is the
retirement, restated with grades.

### The reroll re-run, done ahead of the grades — 2026-09-02

Rather than wait, the five v0.6.0 survivors were re-submitted (tag
`retry-060`) and drained with `WARLOCK_MESH_RETRIES=2`, so the keep-best
loop ran for real on the corpus that matters. Worst-view audit per attempt
(seed 42, then two random mesh seeds), and what the loop kept:

| subject | attempt 1 | attempt 2 | attempt 3 | kept | grade of kept |
|---|---|---|---|---|---|
| cauldron | 0.222 | 0.229 | 0.216 | 0.216 | 1 |
| lantern | 0.245 | 0.213 | 0.221 | 0.213 | 1 |
| well head | 0.204 | 0.186 | 0.167 | 0.167 | 1 |
| pouch | 0.036 (under trigger; no retry) | — | — | 0.036 | 1 |
| branches | 0.201 | 0.111 | 0.145 | 0.111 | −1 |

(The grade column was added when the `retry-060` rows were graded in the
same pass as the corpus: 0 of 5 usable.)

The loop works exactly as documented — it fires past 0.07, rerolls the
TRELLIS stage only, keeps the best — and it buys **0.006 to 0.090** for six
extra minutes per subject (the 0.090 is the branches, the one subject whose
reference is itself unstable). None of the four crosses back under the
trigger. Read with the guidance sweep's reproducibility table (the same
three subjects reproduce their audit to four decimals from the same image),
this says the mesh seed moves the see-through fraction on an open form by a
few hundredths and never closes it, which is what one would expect if the
openings are in the reference.

### The close-holes remesh, also done ahead of the grades — 2026-09-02

`service.jobs.remesh_job(close_holes=True, profile="custom",
custom_faces=<half the source's triangle count>)` on the same five v0.6.0
survivors (Blender, quadriflow, voxel pre-pass on). `model.glb` was backed
up first and re-audited with `meshaudit.hole_fraction` before and after:

| subject | before | after (close_holes remesh) |
|---|---|---|
| cauldron | 0.2217 | 0.2210 |
| lantern | 0.2450 | 0.2408 |
| well head | 0.2039 | 0.2049 |
| branches | 0.2148 | 0.2033 |
| pouch | 0.0912 | 0.0921 |

**The voxel pass closes nothing the audit sees.** Every subject moves by
less than 0.012, the face count halves as asked, and the silhouette's
enclosed gaps survive the remesh intact — which is the strongest evidence
yet that they are openings in the form (legs, panes, posts, loose sticks)
rather than pinholes in a skin, since `VOXEL_FRACTION` is sized for the
latter. The "Seal holes" affordance from the Review badge therefore has no
measured case on this corpus and is **not built**; the decision rule above
(at least 3 of 10 converting) cannot be met by a pass that changes the audit
by a hundredth. The remeshed `model.glb` files are left in place on the five
rows, so the grading pass sees the sealed mesh; the pre-remesh files are
retained beside this session's scratch for comparison.

**Applied to the decision rules (written before the grades, kept as the
record):** the `mesh_retries` default stays 0. Not because the audit does not
see holes — on v0.5.4 it fired on 12 of 21 where the reviewer tagged 10 of
22 — but because on v0.6.0 what it still fires on is not something a reroll
repairs. The grading pass above confirmed both halves: the audit is the
reviewer's `holes` tag to the mesh on v0.5.4, and the five rerolled meshes
graded 0 of 5 usable.

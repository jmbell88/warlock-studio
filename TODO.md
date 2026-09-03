# TODO.md — everything still owed, and who has to do it

Written 2026-08-21, consolidating the four plan files that had accumulated at
the root and in `docs/`; rewritten 2026-08-25 as a priority list after a
whole-tree review closed everything on it that code could close. Git holds
every earlier version and every deleted plan (`git log --all --diff-filter=D`).

**This file has two kinds of entry and no others.**

1. **Work only a human can do**: art direction, authoring keyframes, opening a
   file in real Aseprite or real Tiled, running a card, buying a certificate,
   making a decision. None of it is derivable from the tree and none of it can
   be closed by writing code.
2. **Work that is fully specified and deliberately unstarted** — today that is
   Troupe's phases 7 and 8 alone (P13), each here with the argument that
   makes it actionable, not as a title. Phase 6 closed on 2026-09-03 with the
   re-render merge, its last unbuilt item.

**The moment an item could be built, it is built and struck out rather than
tracked.** A plan whose boxes disagree with the tree is worse than no plan, and
that is why every other roadmap file in this repository's history was deleted
rather than ticked. Nothing here is blocked on finding time. Every claim in
this file was verified against the tree on 2026-08-25, and P7, P10 and P15–P18
against the tree on 2026-08-29.

**This file has no `§N` API.** `tests/test_ux_todo_fixes.py` refuses any
citation of this filename from `src/` or `scripts/`. What a module needs to
explain, it explains where it is — or in `docs/INVARIANTS.md`, or in a
`docs/measurements/` document, both of which outlive any plan.

**How to read an entry.** Highest priority first. *Why it is yours* names the
human dependency. *Do* is the concrete steps. *Expected outcome* is what
changes when it is done — the verdict it produces or the thing it unblocks —
so that "done" is recognisable without re-deriving it.

---

## P1. Run the installer end to end, then on a clean machine

**Why it is yours:** hardware. `installer/build.ps1`, `installer/warlock.iss`,
the runtime manifest and `verify_runtime.py` were built on 2026-08-22. **They
were executed for the first time on 2026-08-26** and the first two bullets
below are settled; everything from the relocated install down is still owed,
and none of it has been seen on a machine that is not this one.

**Do:**
- ~~`pwsh installer\build.ps1` on a Windows machine with `uv`, a uv-managed
  CPython 3.13 and Inno Setup 6. Confirm the staged smoke test and the
  `torch.version.cuda == "12.8"` assert pass; record payload size and compile
  time.~~ **Done 2026-08-26.** Staged smoke test and CUDA assert both pass, but
  the assert only passes on the *retry* path: the default index does not serve
  cu128, so the first `uv pip sync` always fails and the pinned-index retry is
  load-bearing rather than a fallback. `iscc` was not on PATH (it is under
  `%LOCALAPPDATA%\Programs\Inno Setup 6`), so `-Iscc` was needed. **2.91 GB
  payload, 855 s compile, 6.16 GB installed.**
- ~~Settle single-exe versus disk spanning.~~ **Decided 2026-08-26: single
  exe.** The ~4 GB payload compresses to one 2.91 GB executable, so
  `DiskSpanning=no`. `DiskSliceSize` stays in `warlock.iss`, inert, so the
  decision is one line to reverse. `INSTALL.md` and `tests/test_installer.py`
  were both rewritten off the three-file assumption.
- Install per-user, then once with `/DIR=C:\Temp\WarlockApp /SILENT`. A default
  per-user `/SILENT` install was proved on 2026-08-26 (shortcuts, uninstall
  entry, and `warlock doctor` exiting 0 from `{app}` against the real model
  library) — but a *silent* install shows no wizard, so the relocated `/DIR`
  case and every wizard-page claim below remain unseen. With
  `$env:WARLOCK_HOME` at a scratch directory: the Start Menu shortcut launches
  under `pythonw`, the checkout-shape gate passes, the fatal banners name the
  two missing-weights rows, the first-run overlay shows correct GPU verdicts,
  ~23 GB and the disk check, and the wizard's first page renders `LICENSE`.
- Prove the fetch pipeline under the bundled interpreter: one small row end to
  end (dinov2, 0.4 GB), then SDXL and one reference generation. For TRELLIS,
  copy an existing `~/.warlock/models/trellis2-gguf` into the scratch home to
  prove the engine launches from `{app}\vendor`; start the engine download and
  cancel it mid-flight to prove staging cleanup.
- Upgrade over a scratch version (clean slate, data intact); uninstall
  (`{app}` gone, `~/.warlock` intact). Then a clean Windows VM with no Python
  and no CUDA toolkit: install, generate one asset, uninstall, reinstall.
- Then the laptop — the one machine whose GPU is unknown.

Two things the code review could not check: `pythonw.exe` has no console, so a
crash *before* `_setup_logging` attaches closes silently — if that happens, run
`python -m warlock` from a terminal to see it. And an unsigned exe means
SmartScreen's "More info → Run anyway" (see P9).

**Expected outcome:** ~~a recorded build (size, time, one-exe-or-spanning
decision)~~ — recorded above — and a first **non-developer** install that
generated an asset. The build half is done; the half that matters is not. Until
a machine without `uv`, Python or a CUDA toolkit has installed this and made
something, the project has no shippable artifact, whatever the tree says.

## P2. Purge `examples/` from git history before the repository goes public

**Why it is yours:** it rewrites every commit SHA and needs a force-push, and
the moment is yours to pick — **before** publishing, never after; a public
repo's history is cloned within minutes and a later rewrite cannot recall them.

`examples/` was untracked and gitignored on 2026-08-24 and
`tests/test_release_hygiene.py` refuses it coming back, but the blobs remain in
every earlier commit: `light_world.png` and `zelda_1.jpg` (Nintendo material),
`*_base.obj`, `*_base.blend`, `*_spritesheet.png` (ULPC-derived, CC-BY-SA/GPL,
no attribution file anywhere).

**Do:** `git filter-repo --path examples/ --invert-paths`, force-push, have
every collaborator re-clone. If the ULPC files are wanted back, they come back
with an attribution file naming source and licence —
`src/warlock/pipelines/birefnet/ATTRIBUTION.md` is the shape. The Nintendo pair
does not come back.

**Expected outcome:** `git log --all -- examples/` is empty and the repository
can be made public without redistributing art it may not.

## P3. ~~One GPU afternoon: a graded mesh run at the shipped default~~

**Done 2026-09-02.** Two corpora through `text → sdxl_cfg → TRELLIS` at the
shipped defaults, graded blind in Review on the −5..+5 scale the same
afternoon: props-v1 on trellis.cpp v0.6.0 is **11 of 22 usable** (8 of 16 on
easy+medium; `docs/measurements/2026-09-02-trellis-060-props.md`), fantasy-v1
is **10 of 20** (`docs/measurements/2026-09-02-fantasy-v1.md`). The same
grades closed `docs/measurements/2026-09-02-hole-audit-vs-grade.md` (the silhouette audit is
the reviewer's `holes` tag to the mesh on v0.5.4; the reroll converted 0 of 5;
`mesh_retries` stays 0) and settled the guidance sweep's open-form question.
The README states the figure. The library was cleaned on 2026-09-03 once the
open sweeps were reviewed; the verdict rows stay. P10's decisions are unblocked.

What survived as a human item, from the props document's first decision rule:

- **Re-examine the `trellis_tex_res = 512` pin.** Reproduce the auto-tex-res
  noise with `trellis-cli.exe --tex-res 1024` on one reference from props-v1
  (a byte-stable one — the rock, jug or loaf, not the pouch or branches). If
  the texture is clean on v0.6.0, a measurement document lifts the pin; if
  not, it records the reproduction and the pin stays. One reference, one
  judgement, well under an hour of card time.

## P4. A textured, rigged humanoid `.glb` — one file, three jobs

**Why it is yours:** art. Both `examples/*_base.obj` carry no texture, so every
Troupe frame to date quantises into the pale end of whatever ramp it is given.
The palette ramps are installed (`~/.warlock/palettes/cosmos`, `light_world`)
and proven on 2D; **this file is the only thing between them and a verdict on
Troupe.** The same file is the base mesh the Troupe manual chapter assumes, and
the tutorial sample for chapter 11.

**Constraints a base mesh must satisfy:** GLB/glTF (`blender_worker._import_glb`
is the only importer on this path); T-pose or A-pose; +Z up, −Y forward; if it
ships rigged, bone names mapping onto the 19-bone template (Mixamo/Rigify
naming needs a mapping table); no very short bones (Blender silently deletes a
bone below a fraction of the mesh's largest dimension *and takes its children*
— fingers and toes are the usual casualties); under ~300k faces; male and
female variants; a licence permitting commercial redistribution of rendered
sprites. `examples/` cannot supply it (ULPC, CC-BY-SA/GPL: reference and
validation only).

**Do:** author or commission it; put it through **Send to Troupe** (library
menu, inspector, or the picker inside Troupe) with `palette=cosmos`.

**Partially unblocked 2026-08-30.** `tests/fixtures/humanoid/cesium_man.glb`
(CesiumMan, CC-BY 4.0 Cesium — see `tests/fixtures/humanoid/ATTRIBUTION.md`)
is textured, rigged,
+Z up, A-pose-ish and 4,672 polys, so **P5 is now runnable**. It does *not*
close this entry: it is a 3,273-vertex specification sample with a small JPEG
and no female variant, so a ramp verdict taken on it is a claim about
CesiumMan rather than about character art anyone would ship. The authored or
commissioned mesh is still owed, and this entry stays open until it exists.

Putting a rigged mesh through the path found three silent defects that no
TRELLIS reconstruction could ever have exposed — a skin guard defeated by the
incoming weights, two skeletons in the export, and a doubled Y-up→Z-up rotation
that fitted the skeleton to an arm span under half its real width. All three
are fixed (`_strip_incoming_rig`) and pinned by
`tests/test_rig_supplied_mesh.py`; the argument is in
`docs/measurements/2026-08-30-art-verdicts-preregistration.md` Q5.

**Expected outcome:** the first Troupe sheet with real colour, and a verdict on
whether the ramp works at sprite scale. Unblocks P5 and P11.

## P5. Run a `charsheet` job end to end against real Blender

**Why it is yours:** a card. Troupe Phase 4's job has never run on hardware.
The pieces either side of it have (Phase 0d), and the render call is
`rigging.sheet_spec` + `run_worker` exactly as `_sheet` makes it — but the
end-to-end run is owed, and it is how you find out that the clip edits from P8
reach a rendered sheet.

**Do:** with P4's mesh (or any rigged humanoid), **Send to Troupe**, wait for
the rig and the sheet, open the sheet in Troupe.

**Expected outcome:** a rendered sheet from the shipped clips, or the first
real defect in the chain. Either is worth more than the tests.

## P6. Open a Warlock-written `.aseprite` in real Aseprite

**Why it is yours:** an app this repository does not have. A green test proves
this reader and this writer agree with *each other*; a round trip through our
own two halves cannot catch an error both halves make together
(`docs/COMPAT.md`, top).

**Do:** `tests/inker/fixtures/aseprite/FIXTURES.md` names the four fixtures
worth authoring first. **Start with the tilemap ones** — `tilemap-rgb`,
`tilemap-indexed`, `spare-tileset`: their chunk field order was written by
inverting the *reader*, field for field, and has never been checked against a
file Aseprite itself wrote. That is the highest-value five minutes on this
list. In the same sitting: every RGB and grayscale file now carries a palette
chunk derived from the art's own colours, including a 1-entry transparent
palette on a blank document — check Aseprite is happy with that rather than
replacing its default with a single swatch.

**Expected outcome:** either the tilemap chunk order is confirmed and
`docs/COMPAT.md`'s Aseprite rows can say "opened in Aseprite 1.3.x", or a
field-order bug is found that no test could have — and it gets a fixture from
the real app, which is what makes the corpus test worth having.

## P7. Author `.tmx`/`.tsx` fixtures in real Tiled, then lift the version pin

**Why it is yours:** the same rule as P6. Every map under
`tests/plotter/fixtures/tiled/` was produced by this editor, so every
`round-trips` row in `docs/COMPAT.md`'s Tiled part is a round trip against
ourselves. `tests/plotter/fixtures/tiled/FIXTURES.md` lists what is owed and
in what order.

**Do, in order:**
1. Author the fixtures in Tiled 1.12.x and drop them in. **Still owed**, and
   `basic-ortho` is still the first one: 8×8 orthogonal at 16 px, one external
   tileset, two layers with the second at 0.5 opacity, and one tile flipped
   each way (`X`/`Y`/`Z` while stamping). Save it twice — `.tmx` and an
   exported `.tmj` — because the manifest keys on stems having both.
2. ~~Only then move `tsx.TILED_VERSION` (`studio/plotter/tsx.py`) from `1.10.2`
   to `1.12.2`.~~ **Done 2026-08-29: it is `1.12.2`.** The gate was "a real
   Tiled 1.12.2 opens one of our exports without complaint", and both halves
   were exercised against files in the `D:\Projects\RPG` repository: a Plotter
   export (150×150 orthogonal, three CSV layers, external `tilesets/*.tsx`)
   was opened and worked on in Tiled 1.12.x, and a map Tiled 1.12.2 itself
   wrote (640×360, two external tilesets with a `firstgid` split) reads in
   Plotter. `docs/COMPAT.md` records both and what they do not cover.
3. Re-check a grid pack's `.tsx` geometry: pow2 rounding is off by default
   now, so the standing verification is stale. **Still owed** — the maps
   checked in step 2 came from image tilesets, not from a Packwright grid
   pack, so nothing about `tsxout`'s margin/spacing/columns arithmetic was
   exercised in Tiled.

**Expected outcome:** the Tiled rows of `docs/COMPAT.md` become claims about
Tiled rather than about ourselves. ~~and `TILED_VERSION` says what it means.~~
The version half is closed; the corpus half is the one that matters and it is
untouched — the 2026-08-29 check was done on files outside this repository, so
it moved one attribute and left no golden the suite can re-run. Steps 1 and 3
are what turn the rows themselves.

## P8. Author the 22 keyframes

**Why it is yours:** animation is art. The shipped 22 are provisional. Moving
authoring from frames to keyframes made a bad clip cheap to fix, not good; a
bad clip reproduces exactly the "stiff posing" flaw being escaped.

**Do:** Poser → **Clips** in the left sidebar. Pick a key, pose the skeleton
with the normal gizmos, **Update key from pose**. Onion skin ghosts the keys
either side; **Play** scrubs the real interpolation. **Save clips** writes to
your data folder and never touches what the build ships, so **Revert to
shipped clips** is always available. Manual: *Poser → Editing clips*. Decide
first whether it is you or an animator.

Two things to know before starting. **Easing does nothing at the current
segment lengths**: it reshapes where inside a step frames land, so it needs a
step of ≥3 frames, and every shipped step is 1 or 2 — `ease` is a smoothstep
whose only interior sample is exactly 0.5, so `idle`'s `ease` renders
identically to `linear` today (`ease_in`/`ease_out` do differ; the panel says
so). And **the arms hang slightly forward** on the shipped keys.

**The brief, per clip.** Judge each at 16–32 px through the Troupe preview,
not as a 3D pose; the preview's heatmap (built 2026-09-02, `troupe/qa.py`)
flags silhouette pops, foot-line jitter, a loop seam and cross-direction drift
per cell, so use it as the measuring tool and click a flagged square to land
on the frame.

- **Idle** (cyclic): a breath — one or two pixels of vertical bob, shoulders
  and chest, nothing else moves. The seam must be invisible; the heatmap's
  `seam` flag is the check.
- **Walk** (cyclic): two contacts, two passing poses, the bob passing through
  zero between them; arms counter-swing the legs. Feet stay on the ground
  line at the contacts (`foot` flag) and the silhouette changes smoothly
  (`shape` flag).
- **Run** (cyclic): the same four poses with a flight phase — both feet off the
  ground for one frame — a forward lean, and a larger arm swing. Read it in
  profile first: the knee drive is where the current clip crumples.
- **Attack** (one-shot): anticipation (a wind-up, 1–2 frames), the hit (the
  frame with the most silhouette change, and the one to draw first), recovery
  back to idle's first pose so the return does not pop.
- **Jump** (one-shot): crouch, launch, apex (held), fall, land in a crouch,
  recover. The apex is the readable frame; the landing is the second.

Once the clips are authored, calibrate `qa.THRESHOLDS` against the rendered
sheet and record the values in
`docs/measurements/2026-09-02-troupe-qa-thresholds.md`, which says so.

**Expected outcome:** clips that look like movement at 32 px, verified through
P5. This is the most important art task in the programme.

## P9. Decide: code signing

**Why it is yours:** money. No `SignTool=` in `warlock.iss`, no signing step in
`build.ps1`. Every public install trips SmartScreen's "unrecognized app" wall,
the single largest install-abandonment cause for a free tool outside a store.
An OV certificate is a few hundred dollars a year; reputation accrues from
there.

**Expected outcome:** a yes (then a certificate, a `SignTool=` line and a
signing step — code, once the certificate exists) or a recorded no.

## P10. Decide: what the model picker offers

**Why it is yours:** editorial calls that P3's number should inform.

- `juggernaut` and `dreamshaper` have **no hits anywhere in
  `docs/measurements/`** and sit in the picker as peers of the default at
  6.9 GB each. Hide them behind an Advanced toggle, or measure them.
- `sdxl_cfg_pag` is offered as an equal and **lost its own bench**: the control
  won 55 of 80 paired units and PAG cost +34% sampling time
  (`docs/measurements/2026-08-17-reference-source-bench.md`).
- `turbo` is labelled non-commercial (the disclosure half). Whether it is also
  labelled *draft* is an editorial call.
- ~~Tile sheets: `docs/measurements/2026-08-18-tile-sheet-grid.md` says the
  mechanism works and the output is "one continuous brick wall" or
  "near-identical grey mush", and names the answer — "N materials, one grid".
  Ship that, mark the feature experimental, or hold it.~~ **Done 2026-08-29:
  shipped.** The Layout control offers Materials and Terrain set alongside the
  old path, which is labelled *Grid (legacy)* and kept only because a 3/4 or
  isometric tile cannot wrap and so cannot be seamless. What is owed now is the
  verdict, which is P15.

**Expected outcome:** each of the four is a decision recorded in the model
registry's own comments (or a measurement document), after which the picker
stops offering what has not earned its place.

## P11. Decide: two Troupe design questions

**Why it is yours:** design, not implementation.

- **Where does the "judge clips as pixels" preview live?** The plan asked for a
  live low-res sprite preview in Poser. **It cannot go there as built**:
  `template_preview` (`service/poses.py`) builds an armature-only GLB, so
  Poser's preview is a meshless armature and there is nothing to pixelise.
  Either Poser learns to load a rigged asset for preview, or the pixel verdict
  stays in Troupe where the mesh is. The scrubber shipped as the fast loop,
  which is right either way.
- ~~**Phase 6, the cleanup workflow** (P13) — the hard item, "re-render one
  animation without discarding hand edits", is a design problem that should be
  a conversation before it is a commit.~~ **Decided 2026-09-02, in
  conversation:** the merge happens **in Inker**, three-way, because Inker is
  the only place the hand edits exist and Troupe holds no document by
  invariant. (1) When a sheet opens in Inker, the importer records a digest of
  each cell's rendered pixels in the document as an additive `animation.json`
  key written only when set, the way `groups` is. (2) The character-sheet job
  gains a `subset` parameter — a list of `(animation, direction)` runs — and
  the worker renders only those, copying every other cell from the previous
  atlas through a staged write, published as a new sheet id. (3) Inker gets a
  **Merge re-render** op: per cell it compares the recorded base digest, the
  current pixels and the new render; untouched cells take the render, cells
  the user edited where the render did not change keep the edit, and cells
  where both changed are **conflicts, marked in the timeline**. **Default on
  conflict: keep the hand edit and flag the cell** — nothing painted is ever
  overwritten silently; the user resolves per cell or per run. Deferred to its
  own plan; the two cheaper phase-6 items were built first (P13) and the
  merge's cell addressing is theirs (`inker/sheetscope.py`).
- **`plotter-wave-2`.** The branch last moved 2026-08-14 and holds 52 unmerged
  commits; master has moved several hundred since. It is gated on P7's Tiled
  fixtures and on a whole-branch review. Three outcomes: rebase and finish it,
  cherry-pick what still applies, or delete it and let history hold it. **A
  branch delete needs an explicit ask.**

**Expected outcome:** three recorded decisions; the first and second turn into
buildable specs, the third into a branch operation.

## P12. ~~Troupe Phase 0e — judge humanoid reconstruction from a single image~~

**Answered 2026-08-30: no. The generated-character path is not viable at the
shipped default.** Three humanoids went through `text → sdxl_cfg → TRELLIS` as
part of the props-v1 corpus and were judged on this entry's own rubric — limb
separation and silhouette, not face fidelity. The verdict: **limbs are bent and
stretched**. The pre-registered bottom rule fires, and the exact count does not
matter, since the viable rule needed 2 or 3 of 3.

The `_init_frame` distortion bug is ruled out as a confound: these jobs carried
no init image and no conditioning, so that path was never entered. It is the
reconstruction's own geometry.

**What it decides:** Phase 7 is not worth planning on the generated-character
path, which is the decision this entry existed to take before the investment
rather than after. The **supplied-base-mesh path is the one to build on**, and
it became runnable the same day (see P4/P5). This says nothing about whether a
better reconstruction — a multi-view backend, say — could carry characters; it
is a verdict on the shipped single-view default.

Recorded in `docs/measurements/2026-08-30-sdxl-cfg-props.md`, which also carries
the corpus finding this is consistent with: 16 of 21 good references became
unusable meshes, with `holes` the dominant defect tag.

## P13. Troupe phases 6, 7 and 8 — fully specified, deliberately unstarted

The second kind of entry. Phases 0a–0d and 1–5 are implemented and verified;
what they established is in `docs/INVARIANTS.md`, and the measured ULPC facts
are passing oracles in `studio/troupe/ulpc.py`.

**Phase 6 — the cleanup workflow.** Export exists (`inker/sheetout.py`,
`aseout.py`, `packwright/tsxout.py`); the work is the loop:
- ~~*Propagate a correction* across frames / direction / animation. Inker's
  ranged ops (`_doc_ranges.py`) are most of the machinery; through the write
  funnel, addressed by uid.~~ **Built 2026-09-02**: `inker/sheetscope.py`
  (the addressing), `inker/_doc_sheet.py` (one funnel, five verbs), the strip
  under the timeline transport, the **Sheet** menu.
- ~~*Mirror-assisted cleanup.* Measured on the reference sheets: W/E mirroring
  leaves 36–37 differing pixels confined to the face, and every non-zero shift
  is far worse (443 px at ±1) — real facial asymmetry, not a centring offset.
  A fix on one side can be offered on the other, face excluded.~~ **Built
  2026-09-02**: `inker/mirror.py`, the face box at 30 % of the alpha bbox by
  default, a live diff on the canvas, apply per cell or per run.
- ~~*Re-render one animation without discarding hand edits.*~~ **Built
  2026-09-03**, as designed in P11. Three parts: `inker/sheetmerge.py` holds the
  digest and the five-verdict comparison, and the base rides in `animation.json`
  as an additive `sheet` key the way `groups` does; `charsheet.check_subset` /
  `subset_indices` plus `sheet.pack(only=)` and `sheet.compose_cells` are the
  subset arithmetic, and `service.troupe.rerender_charsheet` is the door — it
  copies its pixel settings from the row that made the sheet, so the new cells
  match the ones they land beside. `_doc_sheet.merge_render` is the sixth verb,
  and on conflict the hand edit stands and the cell is flagged. Two ordering
  traps are recorded where they bite: the pixel-art pass is not idempotent, so
  the compose happens *after* the quantise, and the palette is pinned off the
  base atlas or a subset would derive its own.

**Phase 7 — layered equipment (deferred until whole-character generation
works).** *Multi-GLB scene composition*: `op_sheet` takes one `source_glb` and
equipment items are separate assets by construction, so the task is composing
N GLBs under a shared camera, not splitting one (`op_rig` joins every mesh into
one object, which is why splitting is a dead end). *Per-part passes with depth*
give correct per-direction occlusion for free; the depth machinery is proven in
`blender_worker._depth_material` and can move onto the sheet path. *Garment
fitting*: skin-weight transfer by proximity (Blender Data Transfer) — hugging
garments first; capes and long skirts are a separate problem.

**Phase 8 — reconsider only against a working system.** *AI restyle*:
`create_pixel_sheet` with `structure_lock` over a rendered sheet; note
`structure_lock` is only a Canny-ControlNet toggle and what keeps silhouettes
exact is `pixelsheet.remask()` stamping the render's own alpha back, and
`check_restylable` refuses `frame_size × columns > 1024`. Opt-in, measured,
never default. *A learned pixel refiner*: once cleanup is routine,
`(render, hand-cleaned)` pairs accumulate for free, perfectly registered, over
a fixed palette — a well-posed supervised problem that automates the cleanup
step rather than relearning pose transfer. *More animations*: hurt, death,
cast, climb are additive. *Natural-language character description*: only over
a working catalog, only local weights through `fetch_worker`, following the
`expand.py` precedent.

**Expected outcome:** none until P11 and P12 say the programme continues; the
value of this entry is that nobody re-plans it.

**P12 answered on 2026-08-30, and it answered no** for the generated-character
path: limbs came back bent and stretched. Phase 7's own parenthesis already
defers it "until whole-character generation works", so that phase stays where it
is — the gate did not need moving, it needed measuring, and now it has been.

What this does *not* do is close the entry. Phases 6 and 8 never depended on
generated characters, and the **supplied-base-mesh path is untouched by the
verdict** — a user's own rigged humanoid reaches Troupe without the
reconstruction being involved at all, which is the path P4/P5 opened the same
day. P11's two design questions are still yours and still open.

## P14. Listen to Sirens, on a machine with a sound card

**Why it is yours:** hardware, and the plainest instance of it in this file.
Sirens landed complete on 2026-08-27 across six landings and **nobody has ever
heard it**. Every box it was built on is headless and silent, so the synthesis
is proved the only way it could be — a byte-identical render corpus, a
`wavout` reader that is exactly its writer's inverse, a perf budget, and a
pane-draw test with no GPU behind it — and none of that is the same as a person
saying "that is a pulse wave and it is in tune". The device path
(`studio/sirens_audio.py`, `pygame.mixer`) has been exercised by tests that
assert it *degrades* when there is no device, which is the opposite half of the
question. It belongs beside P5, P6 and P7 — the entries that are "run
this against the real thing", as do P15 and P16 below — and it is numbered here
rather than inserted in place only because inserting it would renumber twelve
entries and thirty-seven cross-references, in a file whose own rule is that a
plan disagreeing with the tree is worse than none. Everything appended after it
is numbered by the same rule and read by priority, not by number.

**Do:** open Sirens on a machine with audio and a display, and go through the
tutorial (`docs/manual/14-making-a-soundtrack.md`) as written, out loud:

- Write a bar on the triangle and press Space. Is it in tune against a
  reference pitch? Is the tempo the BPM the transport claims?
- Drag a decay into a volume envelope and hear the shape change. Drag the
  release marker and hold a long note — the tail should take over audibly.
  Then write `Shift+Backtick` (`~~~`) under one note and a backtick (`===`)
  under another: the first should let go into that tail, the second should stop
  dead. That pair is the whole argument for the release half of an envelope and
  the difference is not something a test can hear.
- Type into the other four columns, which only became possible in the sixth
  landing. A volume digit should make one row quieter; an `F` and two digits in
  the effect and parameter columns should change the tempo *from that row on*
  and not from the top; an arpeggio or a vibrato should sound like the thing it
  is named after rather than merely different.
- Drop a `.wav` on the window, point a sample instrument at it, and play it
  from the grid at three different pitches.
- Audition a sound effect. Confirm the song's own buffer is untouched: play the
  song again straight afterwards and hear the song rather than the effect.
- Export into an empty folder. Open `song.wav`, a stem and an `sfx/` file in
  something that is **not** this app. Confirm the loop points in `song.wav`
  actually loop in a player that reads `smpl`, and that a stem lines up
  sample-for-sample with the mix.
- Save, close, reopen. Confirm the song is the song.

**Expected outcome:** either the mode is what the manual says it is, or the
first defect that only a listener could find — a panning error, a tuning
error, a click at a loop point, a mixer that opens at the wrong rate. The tests
cannot produce either verdict, which is exactly why this is here.

## P15. Judge a generated terrain set, and open it in real Tiled

**Why it is yours:** a card, then eyes, then an application this repository does
not have. The seamless path has been through real weights exactly once — four
isolated 1024px materials in `tests/test_tileset_gpu.py` on 2026-08-29 — and
**nothing has generated a terrain set end to end.** The blob-47 construction is
proved by a map round-trip identity on real AI texture, which is the strongest
evidence available without a person looking, and it is still evidence about the
compositing rather than about the art.

**Do:** Create → Sheet → **Terrain set**, two surfaces that ought to meet (grass
into dirt is the honest first try), at 32px. Then take the sheet into Plotter,
paint with the **Terrain** tool, and look at the joins where the brush turns a
corner and where two strokes meet. Then export the map and **open it in Tiled**.

That last step is P6 and P7's argument, not a new one: a round trip through our
own two halves cannot catch an error both halves make together, and a terrain
set is the case where that matters most, because our writer and our reader agree
on the 47-case ordering by construction.

**Expected outcome:** either the first generated tileset that a person would
actually paint a map with, or the first defect in the chain that only a painted
map shows — a coverage field that is right and reads wrong, a boundary that is
too soft at 16px, two materials whose scales disagree.

## P16. Judge an eight-direction action sprite sheet at 32px

**Why it is yours:** art. The seven pose guides are verified as *guides* — the
mirror rule is pinned, every authored row was rendered and looked at, and four
were rewritten because of what that showed — but **nothing has been through SDXL
with them.** Whether a guide that reads correctly as a stick figure produces a
character that reads correctly as an attack is not a question the tests can
reach.

**Do:** one character, one reference, then `attack8` at 32px (eight generations,
about three minutes) and `walk8` at 32px. Play the walk at 10fps in Inker and
turn the character through all eight directions. Judge three things separately:
does one identity survive all eight bands; does the action read as the action;
and does the front row, which is a literal copy of the back row with a different
prompt clause, look like a different picture.

Three limits are already recorded rather than hidden, so they are not the
finding: front and back rows are copies (the convention `walk.json` and
`idle8.json` already set), `run8`'s knee-drive frames read a little like a
crumple in profile, and `cast8`'s release is weaker head-on than in profile
because a forward thrust has nowhere to go in an orthographic front view.

**Expected outcome:** the art verdict on whether the whole action set is worth
having — which is the decision that says whether the remaining guides are worth
authoring, and whether P17 is a question at all.

## P17. Decide whether four-direction guides are wanted

**Why it is yours:** editorial. `SPRITE_DIRECTION_COUNTS` is `(4, 8)` and the
action table plans a `walk4` as readily as a `walk8`, but the menu is discovered
from the guide files on disk and only the `*8.json` files ship — so the
Directions control has exactly one option today, which is a control that cannot
be operated.

The case for four is real: it is half the generations and half the wait, and the
two legacy kinds (`turnaround`, the four-frame `walk`) already cover part of that
ground. The case against is that a four-direction sheet and a legacy walk are
near-neighbours that would need explaining, and that the guides are art — six
more files of authored joint coordinates, each rendered and looked at.

**Do:** decide. If yes, the work after the decision is authoring, not coding:
the loader, the planner, the door and the form already take a four-direction
kind, and the legacy row order (front/left/right/back, not the preset's
front/left/back/right) is already pinned so a four-direction sheet lands in the
vocabulary every draft on disk already uses.

**The one step that is code, whichever way the decision goes:** until `*4.json`
guides exist on disk the Directions control is a one-item menu, and a control
with one option is a control that cannot be operated. Make it honest — show
the eight-direction count as a fact (a label, not a combo) whenever the
discovery in `spritesynth.py` (`TEMPLATE_DIR`, the `<type>.json` loader) finds
a single count, and let the combo reappear on its own the day a second count
ships. `SPRITE_DIRECTION_COUNTS` in `generation.py` stays `(4, 8)`, since the
vocabulary is right; it is the *offer* that is wrong.

**If yes, the six guides to author** (`src/warlock/templates/sprite_guides/`,
same joint-coordinate shape as the `*8.json` beside them, each rendered and
looked at): `idle4`, `walk4`, `run4`, `attack4`, `cast4`, `hurt4`, and
`jump4` if the eight-direction jump survives P16's verdict. Four views each —
front, left, right, back in the legacy row order — with the same keyframe
brief as P8, since a four-direction sheet is the same motion seen from fewer
places rather than a simpler motion.

**Expected outcome:** either six authored guides and a Directions control with
two options, or the control removed and the eight-direction count stated as a
fact rather than offered as a choice.

## P18. ~~Decide what `style_lock` should look like, or remove it~~

**Decided and built 2026-08-30:** the small answer. A *Keep one style across the
list* checkbox on the Materials arm, with the sentence about what it costs (the
IP-Adapter, ~1.2 GB) beside it, and the door now requires the encoder's weights
when it is set. The verdict on the output is P15's. The original entry follows for
the record.

**Why it was yours:** design. It was built and it was unreachable. In the service
and the worker, a materials sheet with `style_lock` set generates the first
material and then uses it as the style reference for every one after it, so that
N surfaces read as one artist's set rather than as N generations; `vram.py` even
accounts for the extra adapter it loads. **No pane offers it and no route sets
it**, so today it is a field that is always False and a branch that never runs —
exactly the defect this session spent itself removing everywhere else.

**Do:** decide one of three. A checkbox on the Materials arm ("keep one style
across the list"), which is the small answer and needs a sentence saying what it
costs. Folding it into the profile, beside the palette and the style LoRA, which
is where "two sheets of one character match" already lives. Or deleting it, on
the argument that a shared seed and a shared prompt template already do most of
the work and an unmeasured second mechanism is not worth a control.

Whichever way it goes, it is a decision first: shipping a control for a
capability nobody has looked at the output of would only move the problem.

**Expected outcome:** a reachable control with a sentence beside it, or 40-odd
lines deleted and the fact recorded — and either way, no more unreachable
branch.

---

## Also owed, smaller

- **Tutorial sample assets** (art): a 32×32 `.ora` sprite with a few layers
  and frames for the Inker chapters; a 16 px tileset of sixteen to twenty-four
  tiles with a terrain set for Plotter and Packwright; a low-poly `crate.glb`
  for Clay; the humanoid from P4 for Troupe. Original and project-licensed —
  nothing from `examples/`, nothing procedurally generated. If they land:
  `src/warlock/assets/tutorial/`, added to the hatchling force-include beside
  `docs/manual/`, under about a megabyte, used by the last section of
  chapters 05, 07, 09, 10 and 11. **Expected outcome:** the "Try it" sections
  gain a starting file for readers who would rather learn the tools than
  design a character in the same ten minutes.
- **Read the first 3.12 CI run.** The `floor` job in
  `.github/workflows/windows-ci.yml` runs the suite under Python 3.12 without
  the `rig` extra; it was added 2026-08-24 and 3.13 changed docstring
  dedenting, which several source-scanning tests depend on. **Expected
  outcome:** green, or a raised floor to 3.13 — not a deleted leg.

## Closed records (kept so nobody re-derives them)

- **Host commit (D1/D2/D3).** A model load never gave its host memory back:
  `flux_klein_distilled` charged +21.1 GiB and returned 0.1. Closed 2026-08-22
  by the t2i child process (`pipelines/text2image_worker.py`,
  `t2i_client.Text2ImageClient`; `WARLOCK_T2I_IN_PROCESS=1` restores the old
  arrangement): in the child, 24.08 GiB charged / 24.08 returned. The child-pid
  reporting defect was closed by reading the kill-on-close job
  (`winjob.job_pids`), and the orphaning claim was refuted by measurement.
  `docs/measurements/2026-08-22-trampoline-child-pids.md` and
  `docs/INVARIANTS.md` hold the figures and the stdin-reader rule.
- **Release audit (2026-08-24).** `REPORT.md` (written at `bd41c75`, deleted
  2026-08-25 once every code-closable finding was closed): GPL-3.0,
  sdist allowlist, licence disclosure in the picker, `THIRD-PARTY-NOTICES.md`
  staged into the installer, the zip ceiling as a subclass. What it left to a
  human is P1, P2, P3, P9 and P10 above.
- **GPU lane.** `uv run pytest -m gpu -n 0`: 26 passed, 0 errors on
  2026-08-21, top-down tile sheets included; the isometric guide and the 3/4
  clause remain unproven on a card and are part of P3's afternoon.
- **Art direction and palettes.** References supplied and ramps installed
  2026-08-21 (`~/.warlock/palettes/cosmos`, `light_world`); the derivation
  drops any colour filling complete scanlines before ranking, because the map
  rip's canvas ranked first by coverage.

## Not on this list on purpose

Decisions with arguments beside them, not backlog:

- **Scale and crop of a tilemap layer** stay refused, permanently. They
  resample, and a tileset cannot follow a resample — there is no permutation
  to teach, only a re-cut, which is a different operation.
- **A hexagonal 120° tile rotation** stays refused: not a permutation of the
  pixel grid, and the standing bar for a tile transform is that it invents no
  colour. `docs/COMPAT.md` carries the argument.
- **Pen/tablet pressure, ICC colour, per-frame palettes, ~~per-cel opacity~~,
  ~~per-cel z-index~~** and the rest of the Aseprite parity programme's named
  non-goals, in `docs/INVARIANTS.md`. Per-cel opacity was struck out on
  2026-08-30 and per-cel z-index the same day: each could be built, so each was
  built (divergences #1 and #12, retired in place there), and this file's own
  rule is that an item which could be built is struck through rather than left
  standing as a decision it is no longer. The z-index is the one that cost
  something to build — it turns the compositor's below-cache off for a
  document that uses it, measured in
  `docs/measurements/2026-08-30-cel-z-below-cache.md`.
- **The Aseprite P1 backlog** was waved after all, on 2026-08-22, by the UX
  refactor's Wave 6. The rule that said it never would be was reversed in
  `docs/INVARIANTS.md` on the same day, and this line said the opposite until
  2026-08-30. What that wave shipped, and what it deliberately left, lives in
  `docs/INVARIANTS.md`.
- **An LLM director for Troupe.** No LLM infrastructure exists, a local HTTP
  endpoint would be the first socket in the app besides the trellis client, and
  it would break `HF_HUB_OFFLINE=1`. The user approves a *picture*, which is a
  better interface than a manifest.

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
   Troupe's phases 6, 7 and 8 alone (P13), each here with the argument that
   makes it actionable, not as a title.

**The moment an item could be built, it is built and struck out rather than
tracked.** A plan whose boxes disagree with the tree is worse than no plan, and
that is why every other roadmap file in this repository's history was deleted
rather than ticked. Nothing here is blocked on finding time. Every claim in
this file was verified against the tree on 2026-08-25.

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
the runtime manifest and `verify_runtime.py` were built on 2026-08-22 and have
**never been executed**. The installer is the only path a non-developer will
ever take, and the release audit rated this the single blocker.

**Do:**
- `pwsh installer\build.ps1` on a Windows machine with `uv`, a uv-managed
  CPython 3.13 and Inno Setup 6. Confirm the staged smoke test and the
  `torch.version.cuda == "12.8"` assert pass; record payload size and compile
  time.
- Settle single-exe versus disk spanning: `warlock.iss` sets `DiskSpanning=yes`
  with a 2.1 GB slice for a ~4 GB payload. Decide at the first compile; drop the
  spanning if one exe compiles.
- Install per-user, then once with `/DIR=C:\Temp\WarlockApp /SILENT`. With
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

**Expected outcome:** a recorded build (size, time, one-exe-or-spanning
decision) and a first non-developer install that generated an asset. Until
then the project has no shippable artifact, whatever the tree says.

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

## P3. One GPU afternoon: a graded mesh run at the shipped default

**Why it is yours:** a card and a judgement. **The number the product will be
judged on does not exist.** The only completed graded run scored 0 usable of 20
(`docs/measurements/2026-08-13-tier-qualification.md`) on deliberately hard
subjects with `playground`; the superseded binary-era number was 19/41. **No
graded run has ever targeted `sdxl_cfg`, which is what ships.**

**Do:** a representative corpus (props, not the hard set) through
`text → sdxl_cfg → TRELLIS` at the shipped defaults, graded on the
`docs/measurements/2026-08-09-grade-scale.md` scale, written up as a
measurement document. `scripts/qualify_tiers.py` is the harness.

**Expected outcome:** a usable-of-N figure the README can state — or, if it is
bad, the honest position that it makes no claim. Either closes the release
audit's "no positive quality evidence" item. It also unblocks P10's three
decisions.

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
1. Author the fixtures in Tiled 1.12.x and drop them in.
2. Only then move `tsx.TILED_VERSION` (`studio/plotter/tsx.py`) from `1.10.2`
   to `1.12.2`. The constant is pinned below the target deliberately; it was
   bumped once without the gate being satisfied and reverted.
3. Re-check a grid pack's `.tsx` geometry: pow2 rounding is off by default
   now, so the standing verification is stale.

**Expected outcome:** the Tiled rows of `docs/COMPAT.md` become claims about
Tiled rather than about ourselves, and `TILED_VERSION` says what it means.

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
- Tile sheets: `docs/measurements/2026-08-18-tile-sheet-grid.md` says the
  mechanism works and the output is "one continuous brick wall" or
  "near-identical grey mush", and names the answer — "N materials, one grid".
  Ship that, mark the feature experimental, or hold it.

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
- **Phase 6, the cleanup workflow** (P13) — the hard item, "re-render one
  animation without discarding hand edits", is a design problem that should be
  a conversation before it is a commit.
- **`plotter-wave-2`.** The branch last moved 2026-08-14 and holds 52 unmerged
  commits; master has moved several hundred since. It is gated on P7's Tiled
  fixtures and on a whole-branch review. Three outcomes: rebase and finish it,
  cherry-pick what still applies, or delete it and let history hold it. **A
  branch delete needs an explicit ask.**

**Expected outcome:** three recorded decisions; the first and second turn into
buildable specs, the third into a branch operation.

## P12. Troupe Phase 0e — judge humanoid reconstruction from a single image

**Why it is yours:** a card and eyes. Untested. Run a prompt → reference →
TRELLIS → `fit_template` pass on a humanoid and judge **limb separation and
silhouette** — at sprite scale those two matter enormously and face-level
fidelity barely at all. This is the largest unproven assumption in the
automated chain, compounded by a known property: reconstruction is
single-image, so **the back is hallucinated**, and a humanoid with separable
limbs is a harder subject than a prop.

It matters only for the generated-character path; the supplied-base-mesh path
(P4) works without it and is reachable from the UI.

**Expected outcome:** a verdict on whether generated characters are viable at
all, before anyone invests in Phase 7.

## P13. Troupe phases 6, 7 and 8 — fully specified, deliberately unstarted

The second kind of entry. Phases 0a–0d and 1–5 are implemented and verified;
what they established is in `docs/INVARIANTS.md`, and the measured ULPC facts
are passing oracles in `studio/troupe/ulpc.py`.

**Phase 6 — the cleanup workflow.** Export exists (`inker/sheetout.py`,
`aseout.py`, `packwright/tsxout.py`); the work is the loop:
- *Propagate a correction* across frames / direction / animation. Inker's
  ranged ops (`_doc_ranges.py`) are most of the machinery; through the write
  funnel, addressed by uid.
- *Mirror-assisted cleanup.* Measured on the reference sheets: W/E mirroring
  leaves 36–37 differing pixels confined to the face, and every non-zero shift
  is far worse (443 px at ±1) — real facial asymmetry, not a centring offset.
  A fix on one side can be offered on the other, face excluded.
- *Re-render one animation without discarding hand edits.* The hardest
  workflow problem in the programme — designed deliberately (P11), not on
  contact.

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

## P14. Listen to Sirens, on a machine with a sound card

**Why it is yours:** hardware, and the plainest instance of it in this file.
Sirens landed complete on 2026-08-27 across five phases and **nobody has ever
heard it**. Every box it was built on is headless and silent, so the synthesis
is proved the only way it could be — a byte-identical render corpus, a
`wavout` reader that is exactly its writer's inverse, a perf budget, and a
pane-draw test with no GPU behind it — and none of that is the same as a person
saying "that is a pulse wave and it is in tune". The device path
(`studio/sirens_audio.py`, `pygame.mixer`) has been exercised by tests that
assert it *degrades* when there is no device, which is the opposite half of the
question. It belongs beside P5, P6 and P7 — the three entries that are "run
this against the real thing" — and it sits last only because inserting it in
place would renumber twelve entries and thirty-seven cross-references, in a file
whose own rule is that a plan disagreeing with the tree is worse than none.

**Do:** open Sirens on a machine with audio and a display, and go through the
tutorial (`docs/manual/14-making-a-soundtrack.md`) as written, out loud:

- Write a bar on the triangle and press Space. Is it in tune against a
  reference pitch? Is the tempo the BPM the transport claims?
- Drag a decay into a volume envelope and hear the shape change. Drag the
  release marker and hold a long note — the tail should take over audibly.
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
- **Pen/tablet pressure, ICC colour, per-frame palettes, per-cel opacity** and
  the rest of the Aseprite parity programme's named non-goals, in
  `docs/INVARIANTS.md`.
- **The Aseprite P1 backlog** stays unscheduled by design — items are pulled
  into sessions individually, never waved. It lives in `docs/INVARIANTS.md`.
- **An LLM director for Troupe.** No LLM infrastructure exists, a local HTTP
  endpoint would be the first socket in the app besides the trellis client, and
  it would break `HF_HUB_OFFLINE=1`. The user approves a *picture*, which is a
  better interface than a manifest.

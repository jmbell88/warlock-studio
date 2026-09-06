# TODO.md — everything still owed, and who has to do it

Written 2026-08-21, consolidating the four plan files that had accumulated at
the root and in `docs/`; rewritten 2026-08-25 as a priority list after a
whole-tree review closed everything on it that code could close; purged
2026-09-04 of every entry that had since closed. Git holds every earlier
version and every deleted plan (`git log --all --diff-filter=D`).

**This file has three kinds of entry and no others.**

1. **Work only a human can do**: art direction, authoring keyframes, opening a
   file in real Aseprite or real Tiled, running a card, listening, making a
   decision. None of it is derivable from the tree and none of it can be
   closed by writing code.
2. **Work that is fully specified and deliberately unstarted** — today that is
   Troupe's phases 7 and 8 alone (P13), each here with the argument that
   makes it actionable, not as a title.
3. **Open findings** (the section at the end): code work a review or a real
   run turned up and did not fix, numbered `F<N>` so it cannot be confused
   with the `P<N>` entries above. Each is buildable and is struck out the day
   it is built; the section is deleted when it is empty.

**The moment an item could be built, it is built and struck out rather than
tracked.** A plan whose boxes disagree with the tree is worse than no plan, and
that is why every other roadmap file in this repository's history was deleted
rather than ticked. Entry numbers are stable: a closed entry's number is not
reused, and what it decided is one line under *Closed records* at the bottom.

**This file has no `§N` API.** `tests/test_ux_todo_fixes.py` refuses any
citation of this filename from `src/` or `scripts/`. What a module needs to
explain, it explains where it is — or in `docs/INVARIANTS.md`, or in a
`docs/measurements/` document, both of which outlive any plan.

**How to read an entry.** Highest priority first. *Why it is yours* names the
human dependency. *Do* is the concrete steps. *Expected outcome* is what
changes when it is done — the verdict it produces or the thing it unblocks —
so that "done" is recognisable without re-deriving it.

---

## P1. Generate an asset from a clean-machine install

**Why it is yours:** hardware. The installer was built for the first time on
2026-08-26 and has been rebuilt since; two things it reproduced every time, and
which are the path rather than an edge case: `iscc` is not on PATH (`-Iscc` is
needed) and the default index does not serve cu128, so the pinned-index retry in
`build.ps1` is load-bearing.

**The install half of this entry closed on 2026-09-05**
(`docs/measurements/2026-09-05-clean-machine-install.md`).
`WarlockSetup-v0.0.35.exe` was carried on a USB drive to a second Windows PC
with no Python, no `uv` and no CUDA toolkit, and installed there with no network
involved at any point. The app launched, every mode was browsable, and a drawing
was made in Inker, saved and reopened. So the sentence this entry used to end
with — that the project has no shippable artifact — is retired: it installs on a
machine that is not this one, and it draws. What it has still never done is
*make* anything, because that machine's card is 8 GiB and `vram.TRELLIS_GIB` is
16.0, so a reconstruction is out of reach there by design.

Note also that the figures this entry used to carry (2.91 GB payload, 6.61 GB
installed) were the all-extras shape and are superseded. P26 took the installer
to `--extra studio` alone with the three heavy extras arriving as packs, and
`INSTALL.md` now measures that build: **810 MB download (846,950,916 bytes),
about 1.4 GB installed base**, SHA-256 `254b3af9…`, at v0.0.35.

**Do:**

1. ~~**Install on a machine that is not this one, and reach the window.**~~
   Proved 2026-09-05. Offline from USB onto a clean Windows PC: the shortcut
   launched, the checkout-shape gate passed, the welcome dialog offered its
   three doors, the hardware scan named the real GPU, and the fatal banners
   named the missing-weights rows. The base install reached the window with no
   torch and no `bpy` — P26's added step — and Inker round-tripped a document.
2. ~~**Prove the installer stages the packs.**~~ Proved 2026-09-05. Settings →
   Packs drew a real row with a size and a live Install button rather than
   `app_settings.pack_blocked`'s "this build carries no packs" fallback, which
   is what a source checkout gets. `packs.json` and the bundled wheels are
   reaching `{app}\packs` on a machine that has never had `uv`.
3. ~~**Install a pack from Settings on a machine that has never had `uv`.**~~
   Proved 2026-09-05, on the reinstall. All three went on — Image Generation,
   Rigging and Music — which is P26's programme validated outside the test
   suite for the first time, `music`'s sdist build path and the bundled-wheel
   branch of `pack_worker.collect` included. Still unchecked: whether Poser
   opens without a restart afterwards.
4. **Fix the model downloads, then prove the fetch pipeline under the bundled
   interpreter.** The one thing that stopped both runs. Her log named the cause
   and it is ours: a failed fetch deletes its staging tree, resume state and
   all, so a 16.1 GB model cannot be retried in over a line that resets — F1.
   The resets themselves hit several hosts and are still undiagnosed (F2), but
   F1 is what makes them fatal rather than annoying. When it works: one small row end to end
   (dinov2, 0.4 GB), then SDXL and one reference generation. For TRELLIS, copy
   an existing `~/.warlock/models/trellis2-gguf` into the scratch home to prove
   the engine launches from `{app}\vendor`; start the engine download and cancel
   it mid-flight to prove staging cleanup.
5. **The three recovery paths, which have only ever run against fakes** (the
   beta audit's H02/M01/M02). Quit the app *during* a pack install and watch
   which half you are in — a download must cancel on the worker's own
   acknowledgement, and a commit must hold the quit until pip is done rather
   than confirm one it cannot honour. Then press **Repair** on a pack that
   installed cleanly (it should reinstall the pinned wheels and come back green)
   and on one you have damaged by hand, by emptying a `.dist-info` or a
   top-level module. Finally upgrade over a version with packs installed and
   take the **Restore packs** button the banner offers: the selection is
   recorded beside the wheel cache under `WARLOCK_HOME`, so it is supposed to
   survive the installer wiping `site-packages`. **These are reachable now**:
   they need a pack, not a weight or a card, and that machine has three. The
   cheapest step left, and the only one that does not wait on the fetch path.
6. **Upgrade over a scratch version** (clean slate, data intact). Uninstall and
   reinstall are done — that is how the second 2026-09-05 run began, and
   `~/.warlock` survived it — but an upgrade *over* an existing install is
   not, and that is the one carrying the Restore packs banner.
7. **Then the laptop** — the one machine whose GPU is unknown.

Two things the code review could not check, one of which is now half-answered:
`pythonw.exe` has no console, so a crash *before* `_setup_logging` attaches
closes silently — if that happens, run `python -m warlock` from a terminal to
see it. And an unsigned exe means SmartScreen's "More info → Run anyway" (code
signing was answered no for the closed beta; see Closed records) — though on
2026-09-05 SmartScreen did not appear at all, so that page is still unwitnessed.

**Expected outcome:** a **non-developer install that generated an asset**. The
install itself is no longer the question; a working download path and a card
big enough to reconstruct on are.

## P3. Re-examine the `trellis_tex_res = 512` pin

**Why it is yours:** a card and a judgement. The graded mesh run closed on
2026-09-02 (props-v1 11 of 22 usable, fantasy-v1 10 of 20; see Closed
records), and this is what survived from its first decision rule.

**Do:** reproduce the auto-tex-res noise with `trellis-cli.exe --tex-res 1024`
on one byte-stable reference from props-v1 (the rock, jug or loaf — not the
pouch or branches). If the texture is clean on v0.6.0, a measurement document
lifts the pin; if not, it records the reproduction and the pin stays. One
reference, one judgement, well under an hour of card time.

## P4. A textured, rigged humanoid `.glb` — one file, three jobs

**Why it is yours:** art. Every Troupe frame to date quantises into the pale
end of whatever ramp it is given because no textured base mesh exists. The
palette ramps are installed (`~/.warlock/palettes/cosmos`, `light_world`) and
proven on 2D; **this file is the only thing between them and a verdict on
Troupe.** The same file is the base mesh the Troupe manual chapter assumes, and
the tutorial sample for chapter 11.

**Constraints a base mesh must satisfy:** GLB/glTF (`blender_worker._import_glb`
is the only importer on this path); T-pose or A-pose; +Z up, −Y forward; if it
ships rigged, bone names mapping onto the 19-bone template (Mixamo/Rigify
naming needs a mapping table); no very short bones (Blender silently deletes a
bone below a fraction of the mesh's largest dimension *and takes its children*
— fingers and toes are the usual casualties); under ~300k faces; male and
female variants; a licence permitting commercial redistribution of rendered
sprites.

**Do:** author or commission it; put it through **Send to Troupe** (library
menu, inspector, or the picker inside Troupe) with `palette=cosmos`.

**Partially unblocked 2026-08-30.** `tests/fixtures/humanoid/cesium_man.glb`
(CesiumMan, CC-BY 4.0 — `tests/fixtures/humanoid/ATTRIBUTION.md`) is textured,
rigged, +Z up, A-pose-ish and 4,672 polys, so the chain was runnable from that
day. It does not close this entry: a ramp verdict taken on a 3,273-vertex sample
with a small JPEG and no female variant is a claim about CesiumMan, not about
character art anyone would ship. Putting it through the path found and fixed
three silent rig defects (`_strip_incoming_rig`, `tests/test_rig_supplied_mesh.py`,
`docs/measurements/2026-08-30-art-verdicts-preregistration.md` Q5).

**Narrowed 2026-09-05.** This entry is no longer what stands between Troupe and
a verdict, and it is no longer chapter 11's tutorial sample. Create's Character
type builds a textured, rigged body from an authored family — thirty-one species
over four body plans — so there is a base mesh with real colour on it in the
build, and the ramp verdict is P28's. What is still owed here is what it always
was underneath: a *human-authored* character anyone would ship, as the thing a
generated species is judged against and as the mesh a user brings of their own.
It no longer blocks anything.

**Expected outcome:** the first Troupe sheet from art rather than from a
generator, and a verdict on whether the ramp works at sprite scale on it.
Unblocks P11.

## P6. Open a Warlock-written `.aseprite` in real Aseprite

**Why it is yours:** an app this repository does not have. A green test proves
this reader and this writer agree with *each other*; a round trip through our
own two halves cannot catch an error both halves make together
(`docs/COMPAT.md`, top).

**Do:** `tests/inker/fixtures/aseprite/FIXTURES.md` names the four fixtures
worth authoring first. **Start with the tilemap ones** — `tilemap-rgb`,
`tilemap-indexed`, `spare-tileset`: their chunk field order was written by
inverting the *reader*, field for field, and has never been checked against a
file Aseprite itself wrote. In the same sitting: every RGB and grayscale file
carries a palette chunk derived from the art's own colours, including a
1-entry transparent palette on a blank document — check Aseprite is happy with
that rather than replacing its default with a single swatch.

**Expected outcome:** either the tilemap chunk order is confirmed and
`docs/COMPAT.md`'s Aseprite rows can say "opened in Aseprite 1.3.x", or a
field-order bug is found that no test could have — and it gets a fixture from
the real app.

## P7. Author `.tmx`/`.tsx` fixtures in real Tiled

**Why it is yours:** the same rule as P6. Every map under
`tests/plotter/fixtures/tiled/` was produced by this editor, so every
`round-trips` row in `docs/COMPAT.md`'s Tiled part is a round trip against
ourselves. `TILED_VERSION` already moved to `1.12.2` on 2026-08-29 against
files outside this repository; that moved one attribute and left no golden the
suite can re-run.

**Do:**
1. Author the fixtures in Tiled 1.12.x per `tests/plotter/fixtures/tiled/FIXTURES.md`.
   `basic-ortho` first: 8×8 orthogonal at 16 px, one external tileset, two
   layers with the second at 0.5 opacity, one tile flipped each way. Save it
   twice — `.tmx` and an exported `.tmj` — because the manifest keys on stems
   having both.
2. Re-check a grid pack's `.tsx` geometry in Tiled: pow2 rounding is off by
   default now, and the 2026-08-29 maps came from image tilesets, so nothing
   about `tsxout`'s margin/spacing/columns arithmetic has been exercised there.

**Expected outcome:** the Tiled rows of `docs/COMPAT.md` become claims about
Tiled rather than about ourselves.

## P8. Author the 22 keyframes

**Why it is yours:** animation is art. The shipped 22 are provisional. Moving
authoring from frames to keyframes made a bad clip cheap to fix, not good; a
bad clip reproduces exactly the "stiff posing" flaw being escaped.

**Wider since 2026-09-05, and easier at the same time.** There are now four
authored clip libraries, not one — `humanoid`, `quadruped`, `bird` and `blob`,
each carrying all five movements because `charsheet.resolve_layout(None)` asks
for five and `expand_clips` raises on a missing one. The 22 below are the
humanoid's and are the ones to start with, but a four-beat lateral-sequence
walk and a wing beat are their own problems and neither is a humanoid walk with
different bone names. Easier because the thing that was missing is here: every
species is a body you can build in one press with no card, so a clip edit can
be judged on a *rendered* sheet within minutes instead of waiting on P4. That
is also what unblocks the calibration below — `qa.THRESHOLDS` was chosen
against synthetic sheets and explicitly not against rendered motion, and
rendered motion now exists for four body plans. Calibrate it as part of P28
rather than separately; one sitting judging four sheets is where the numbers
come from.

**Do:** Poser → **Clips** in the left sidebar. Pick a key, pose the skeleton
with the normal gizmos, **Update key from pose**. Onion skin ghosts the keys
either side; **Play** scrubs the real interpolation. **Save clips** writes to
your data folder and never touches what the build ships, so **Revert to
shipped clips** is always available. Manual: *Poser → Editing clips*. Decide
first whether it is you or an animator.

Two things to know before starting. **Easing does nothing at the current
segment lengths**: it needs a step of ≥3 frames and every shipped step is 1 or
2, so `idle`'s `ease` renders identically to `linear` today. And **the arms
hang slightly forward** on the shipped keys.

**The brief, per clip.** Judge each at 16–32 px through the Troupe preview,
whose heatmap (`troupe/qa.py`) flags silhouette pops, foot-line jitter, a loop
seam and cross-direction drift per cell; click a flagged square to land on the
frame.

- **Idle** (cyclic): a breath — one or two pixels of vertical bob, shoulders
  and chest, nothing else. The seam must be invisible (`seam` flag).
- **Walk** (cyclic): two contacts, two passing poses, the bob passing through
  zero between them; arms counter-swing the legs. Feet on the ground line at
  the contacts (`foot`), silhouette changing smoothly (`shape`).
- **Run** (cyclic): the same four poses with a flight phase, a forward lean and
  a larger arm swing. Read it in profile first: the knee drive is where the
  current clip crumples.
- **Attack** (one-shot): anticipation (1–2 frames), the hit (the frame with the
  most silhouette change, and the one to draw first), recovery back to idle's
  first pose so the return does not pop.
- **Jump** (one-shot): crouch, launch, apex (held), fall, land in a crouch,
  recover. The apex is the readable frame; the landing is the second.

Once the clips are authored, calibrate `qa.THRESHOLDS` against the rendered
sheet and record the values in
`docs/measurements/2026-09-02-troupe-qa-thresholds.md`, which says so.

**Expected outcome:** clips that look like movement at 32 px, verified through
P28's rendered sheets. This is the most important art task in the programme.

## P10. Decide: what the model picker offers

**Why it is yours:** editorial calls that the graded run's numbers now inform.

- `juggernaut` and `dreamshaper` have **no hits anywhere in
  `docs/measurements/`** and sit in the picker as peers of the default at
  6.9 GB each. Hide them behind an Advanced toggle, or measure them.
- `sdxl_cfg_pag` is offered as an equal and **lost its own bench**: the control
  won 55 of 80 paired units and PAG cost +34% sampling time
  (`docs/measurements/2026-08-17-reference-source-bench.md`).
- `turbo` is labelled non-commercial (the disclosure half). Whether it is also
  labelled *draft* is an editorial call.

**Expected outcome:** each is a decision recorded in the model registry's own
comments (or a measurement document), after which the picker stops offering
what has not earned its place.

## P11. Decide: two Troupe questions

**Why it is yours:** design, not implementation.

- **Where does the "judge clips as pixels" preview live?** It cannot go in
  Poser as built: `template_preview` (`service/poses.py`) builds an
  armature-only GLB, so there is nothing to pixelise. Either Poser learns to
  load a rigged asset for preview, or the pixel verdict stays in Troupe where
  the mesh is.
- **`plotter-wave-2`.** The branch last moved 2026-08-14 and holds 52 unmerged
  commits; master has moved several hundred since. Gated on P7's fixtures and
  a whole-branch review. Three outcomes: rebase and finish it, cherry-pick what
  still applies, or delete it. **A branch delete needs an explicit ask.**

**Expected outcome:** two recorded decisions; the first turns into a buildable
spec, the second into a branch operation.

## P13. Troupe phases 7 and 8 — fully specified, deliberately unstarted

The second kind of entry. Phases 0a–0d and 1–6 are implemented and verified;
what they established is in `docs/INVARIANTS.md`, and the measured ULPC facts
are passing oracles in `studio/troupe/ulpc.py`. Phase 6 closed on 2026-09-03
with the three-way re-render merge (`inker/sheetmerge.py`,
`service.troupe.rerender_charsheet`, `_doc_sheet.merge_render`; on conflict the
hand edit stands and the cell is flagged).

**Phase 7 — layered equipment (deferred until whole-character generation
works, which P12 measured it does not at the shipped default).** *Multi-GLB
scene composition*: `op_sheet` takes one `source_glb` and equipment items are
separate assets by construction, so the task is composing N GLBs under a
shared camera, not splitting one (`op_rig` joins every mesh into one object,
which is why splitting is a dead end). *Per-part passes with depth* give
correct per-direction occlusion for free; the depth machinery is proven in
`blender_worker._depth_material`. *Garment fitting*: skin-weight transfer by
proximity (Blender Data Transfer) — hugging garments first; capes and long
skirts are a separate problem. The supplied-base-mesh path (P4) and the
authored-family path Create's Character type shipped on 2026-09-05 are both
untouched by P12's verdict, and are the two to build on.

**Phase 8 — reconsider only against a working system.** *AI restyle*:
`create_pixel_sheet` with `structure_lock` over a rendered sheet; note
`structure_lock` is only a Canny-ControlNet toggle and what keeps silhouettes
exact is `pixelsheet.remask()` stamping the render's own alpha back, and
`check_restylable` refuses `frame_size × columns > 1024`. Opt-in, measured,
never default. *A learned pixel refiner*: once cleanup is routine,
`(render, hand-cleaned)` pairs accumulate for free, perfectly registered, over
a fixed palette. *More animations*: hurt, death, cast, climb are additive.
*Natural-language character description*: only over a working catalog, only
local weights through `fetch_worker`.

**Expected outcome:** none until P11 says the programme continues; the value
of this entry is that nobody re-plans it.

## P14. Listen to Sirens, on a machine with a sound card

**Why it is yours:** hardware, and the plainest instance of it in this file.
Sirens landed complete on 2026-08-27 and **nobody has ever heard it**. Every
box it was built on is headless, so the synthesis is proved the only way it
could be — a byte-identical render corpus, a `wavout` reader that is its
writer's inverse, a perf budget — and none of that is a person saying "that is
a pulse wave and it is in tune".

**Do:** open Sirens on a machine with audio and go through
`docs/manual/14-making-a-soundtrack.md` as written, out loud:

- Write a bar on the triangle and press Space. In tune against a reference
  pitch? Is the tempo the BPM the transport claims?
- Drag a decay into a volume envelope and hear the shape change. Drag the
  release marker and hold a long note. Then write `~~~` under one note and
  `===` under another: the first should let go into the tail, the second stop
  dead.
- Type into the other four columns. A volume digit should make one row
  quieter; an `F` and two digits should change the tempo *from that row on*;
  an arpeggio or vibrato should sound like the thing it is named after.
- Drop a `.wav` on the window, point a sample instrument at it, play it from
  the grid at three pitches.
- Audition a sound effect, then play the song again and hear the song.
- Export into an empty folder. Open `song.wav`, a stem and an `sfx/` file in
  something that is **not** this app. Confirm the `smpl` loop points loop, and
  that a stem lines up sample-for-sample with the mix.
- Save, close, reopen.

**Expected outcome:** either the mode is what the manual says it is, or the
first defect only a listener could find — a panning error, a tuning error, a
click at a loop point, a mixer that opens at the wrong rate.

## P15. Judge a generated terrain set, and open it in real Tiled

**Why it is yours:** a card, then eyes, then an application this repository
does not have. The seamless path has been through real weights exactly once
(four isolated 1024px materials, `tests/test_tileset_gpu.py`, 2026-08-29) and
**nothing has generated a terrain set end to end.** This is also where the
*Keep one style across the list* checkbox (P18, built 2026-08-30) gets its
verdict.

**Do:** Create → Sheet → **Terrain set**, two surfaces that ought to meet
(grass into dirt), at 32px. Take the sheet into Plotter, paint with the
**Terrain** tool, look at the joins where the brush turns a corner and where
two strokes meet. Export the map and **open it in Tiled** — our writer and
reader agree on the 47-case ordering by construction, so only Tiled can catch
an error both make.

**Expected outcome:** either the first generated tileset a person would
actually paint a map with, or the first defect only a painted map shows.

## P16. Judge an eight-direction action sprite sheet at 32px

**Why it is yours:** art. The seven pose guides are verified as *guides*, but
**nothing has been through SDXL with them.**

**Do:** one character, one reference, then `attack8` at 32px and `walk8` at
32px. Play the walk at 10fps in Inker through all eight directions. Judge
three things separately: does one identity survive all eight bands; does the
action read as the action; does the front row, a literal copy of the back row
with a different prompt clause, look like a different picture.

Known and recorded, so not the finding: front and back rows are copies,
`run8`'s knee-drive reads a little like a crumple in profile, and `cast8`'s
release is weaker head-on than in profile.

**Expected outcome:** the art verdict on whether the whole action set is worth
having, and whether P17 is a question at all.

## P17. Decide whether four-direction guides are wanted

**Why it is yours:** editorial. `SPRITE_DIRECTION_COUNTS` is `(4, 8)` but only
`*8.json` guides ship, so the Directions control has exactly one option. The
case for four is half the generations and half the wait; the case against is
that a four-direction sheet and the legacy `walk` are near-neighbours, and the
guides are art.

**Do:** decide. If yes, author `idle4`, `walk4`, `run4`, `attack4`, `cast4`,
`hurt4` (and `jump4` if the eight-direction jump survives P16) in
`src/warlock/templates/sprite_guides/`, four views each in the legacy row order
front/left/right/back, with P8's brief. The loader, planner, door and form
already take a four-direction kind.

**The one step that is code, whichever way it goes:** while discovery finds a
single count, show the eight-direction count as a label rather than a
one-item combo, and let the combo reappear the day a second count ships.

**Expected outcome:** either six authored guides and a two-option control, or
the count stated as a fact.

## P19. Measure the generated Flourish texture against the procedural one

**Why it is yours:** a GPU afternoon, and a judgement. The texture door is
built and defaults to nothing: every preset is procedural. Whether a generated
flame or ember *beats* the procedural core at 128 px is a measurement, and the
earlier prompt expander was deleted for shipping without one.

**Do:** generate five textures with the shipped prompt template (flame, ember,
rune, skull, shard); put each on the fireball's *Sparks* and on a *sprite*
layer at 128 px, painterly and pixel. Judge beside the procedural version and
write a document under `docs/measurements/` with the verdict, the prompts, and
whether the black key or the matting model made the better cutout. If a
texture wins, the preset gets it as a file beside its JSON; the door stays
opt-in either way.

**Expected outcome:** one measurement document; presets changed only if it
says so.

## P20. Pick the Flourish prompt's text model, measure it, pin it

**Why it is yours:** a CPU afternoon and a judgement, and the one entry that
touches the offline invariant's stated exception. The door is gated on a
directory (`inker_flourish.TEXT_MODEL_DIR`), not a registry row, because every
`models.py` entry carries a revision pin and the pin comes from this
measurement.

**Do:** candidates Qwen2.5-0.5B-Instruct, 1.5B-Instruct, SmolLM2-1.7B-Instruct,
one at a time in `text-instruct/`. Twenty fixed sentences, half inside the
keyword vocabulary and half outside. For each, the `[model]` toast versus the
`[keywords]` toast, whether the effect did what the sentence said, and CPU time
per prompt. Write a document under `docs/measurements/`. If a model beats the
vocabulary on the outside half without losing the inside half, add a pinned
`TextModel` row to `models.py`; if none does, delete `recipe_worker.py` and the
door.

**Expected outcome:** one measurement document and one of the two edits.

## P21. Judge restyled keyframes against the procedural frames

**Why it is yours:** a GPU afternoon. `Flourish → Restyle keyframes…` is
built and opt-in; a crossfade of two diffusion frames is exactly where the plan
expected it to fail.

**Do:** the fireball's *explosion* and the portal's *loop*, three and five
keyframes, strengths 0.4 and 0.7, "oil painting" and "ink woodcut". Play beside
the procedural layer. Write a document under `docs/measurements/`: does the
in-between read as motion or a fade; does the model keep the silhouette at 0.4;
is five keyframes enough for a twelve-frame phase. If it never reads as motion,
delete `keyframes.py`, the door and its popup.

**Expected outcome:** one measurement document and one of the two edits.

## P22. Write what the closed beta is told it has not seen

**Why it is yours:** it is a claim about the product, made in your name, to
people you invited. Only Troupe carries an **Experimental** chip. Four other
surfaces have evidence gaps and no chip: Sirens has never been heard (P14),
Muse has never been heard (P23), Plotter's Tiled interop has only round-tripped
against itself (P7), no character sheet has been judged by an eye at sprite
scale (P28), and Warlock-written `.aseprite` files have never been opened in Aseprite
(P6). One more belongs here that is not a mode: on a base install, Create and
Muse send you to Settings → **Models**, and the weights are only half of what
they need — the matching **pack** is the other half, and nothing at the door
says so (F4). An invitee who downloads 23 GB and still cannot generate has hit a
known gap, not a broken build.

**Do:** name them, by mode, in whatever the invite is — a note beside the
download. One sentence each: what runs, what has never been checked against the
other implementation, what to report if it breaks. **Not by adding chips**: a
chip is a permanent statement about design; these are temporary statements
about evidence, and the beta is what removes them.

**Expected outcome:** an invited user who hits one of these knows they hit a
known gap, and reports the right thing.

## P23. Hear Muse, and give its two VRAM figures real numbers

**Why it is yours:** a card big enough for an 8.3 GB model, and ears.
Everything provable without a card is proved; what none of it answers is
whether the model produces music, whether a cancel interrupts a real sampling
loop, and what the thing costs.

**Do:**

- `uv sync --extra music`, download the weights from Settings → Models, confirm
  `uv run warlock doctor` flips the ACE-Step row.
- `uv run pytest tests/test_music_gpu.py -m gpu -n 0`. The cancel test is the
  only proof `WARLOCK 1/5` reaches the loop. Its first job is confirming on
  hardware that a take the *model* produced (44.1 kHz 16-bit PCM, `WARLOCK 5/5`)
  opens in Sirens.
- The derived tasks: retake at 0.2 and 0.8, extend a 60 s take, repaint one
  phrase, edit the tags alone. Cancel each mid-run — the `edit` especially,
  whose cancel hook is a second loop and has never been exercised.
- In the app: two takes at 60 s from style tags, and listen. Is it music? Do
  the tags do anything? Does a `[verse]`/`[chorus]` lyric block get sung?
- Cancel mid-generation: child dies, row reads **cancelled**, next take runs
  without a restart. Kill the app mid-generation: no `music_worker` survives.
- **Open in Sirens** on a take; play the sample from the grid at three pitches.

**The measurement.** `models.MusicModel.vram_gib` (10.0) and `host_peak_gib`
(12.0) are documented estimates. Take them from a real run, publish
`docs/measurements/<date>-ace-step-vram.md`, replace the constants with cited
numbers. `cpu_offload` and `overlapped_decode` stay class attributes on
`music_worker._Server` until a measurement says they need to be knobs.

**Expected outcome:** either the mode is what chapters 16 and 35 say it is, or
the first defect only a listener could find — and two constants measured
rather than guessed.

## P24. Judge the loop finder, and hear a stem split

**Why it is yours:** ears, again, and a card.

**The loop finder.** `studio/muse/loops.py`'s `W_CONTEXT` (1.5), `W_LEVEL`
(0.6) and `W_LENGTH` (2.0) were **chosen by ear and ship saying so**. Generate
a dozen takes across styles, run **Find loop points**, listen to the best
candidate looping four or five times, then try the numbered alternatives. Does
the top candidate usually win? Does it favour quiet moments (`W_LEVEL` too
strong) or merely spectrally similar ones (`W_CONTEXT` too weak)? Is `MIN_SPAN`
(0.35) too generous for a two-minute take? Does a 0 ms crossfade click and does
500 ms audibly duck? Either write `docs/measurements/<date>-loop-weights.md` or
leave the constants with their honest "unmeasured" comments — **an honest
unmeasured constant beats a measured-sounding one**.

**Stem separation** has never been run. Download it from Settings → Models and
confirm the red non-commercial marker appears at the moment you agree. Split
three or four takes (percussive, vocal, ambient) and listen to each stem for
**bleed**; chapter 35 promises "a little", which a listener has to confirm or
correct. Cancel a split mid-run: row cancelled, no child, the take reads as
unsplit rather than partly split.

**The measurements.** `SeparationModel.vram_gib` (4.0), `host_peak_gib` (4.0)
and `vram.MUSIC_SOURCE_GIB` (1.0) are guesses. Take all three from real runs —
the third from an `extend` of a 240 s take — and publish
`docs/measurements/<date>-hdemucs-separation.md` with wall clock beside them.
`segment_seconds` (10.0) is the knob if separation is slower than about a
minute for a four-minute take.

**Expected outcome:** three measured constants, a verdict on the loop weights,
and either a confirmation of chapter 35's bleed sentence or a better one.

## P25. Decide: is a non-commercial stem model worth shipping at all

**Why it is yours:** a licensing judgement about what this app is *for*.
Hybrid Demucs ships **labelled** `commercial=False` with a `license_note` and
the red marker; the reasoning is in `docs/MODELS.md`. It is the second
non-commercial entry beside SDXL-Turbo, and unlike Turbo it is optional.

**The question:** for an app whose purpose is making assets people sell, is
"labelled and optional" the right answer, or should the feature not be offered?

**If remove:** the surface is the `SeparationModel` table,
`pipelines/separation_worker.py`, `separate_job`, the `separate` arms in
`_q_music`/`_q_jobs`/`vram`/`validation`/`progress`, the four `files.MEDIA`
keys, the tray button, and chapter 35's Stems section. The `url`/`sha256`
transport in `models.Fetch` **stays** either way.

## P26. Decide whether the dependency packs ship, and wire them up if so

**Why it is yours:** a product decision with a cost attached, and it gates
buildable work that is otherwise ready. The pure and performing halves are
built and proven (`0975721f`, `2556cb6d`, `26b40d8a`, and this session's
commit); what is left is a *choice* about the shipped installer, plus one
packaging question with a real weight in megabytes.

**Where it stands.** `warlock/packs.py` is the registry and planner,
`scripts/make_packs.py` the generator, `pipelines/pack_worker.py` the child
that downloads and installs, `service/packs.py` the parent. Measured against
the real lock: base+studio is 30 distributions; `rig` adds 7, `text2image` 34,
`music` 74, with 27 shared between the two torch packs — 85 wheels, 82 fetched
and 3 built. The rig pack has been collected and installed end to end into a
base-only runtime (0.32 GiB download, 0.63 GiB installed, `import bpy` → 5.2.0
LTS). The second offline exception is recorded in `docs/INVARIANTS.md`.

**The decision was taken on 2026-09-04: they ship.** The installer now stages
`--extra studio` alone and the three heavy extras arrive from Settings, which
takes the base download to roughly a third of 2.91 GB and lets a user who only
draws pixel art never download torch. What that decision costs — a second
install path to support — is real, and P1 is where it is met: every figure in
that entry was measured against the all-extras installer and none of them is
this build's any more.

**Do** — steps 2, 3 and 4 were built the day the decision was taken; what is
left needs a person:

1. **Decide where the three built wheels live.** `docopt`, `mojimoji` and
   `unidic-lite` publish no Windows wheel, so the build compiles them and they
   are marked `bundled` in the manifest: there is no URL to fetch them from, so
   the installer must carry them. `unidic-lite` is ~47 MB of that, all of it in
   the base download, for a pack the user may never install. The alternatives
   are hosting the three built wheels as release assets (a publishing step, and
   the first URL in this project that is ours) or dropping `cutlet`/`fugashi`
   and losing Japanese lyric romanisation. **This is the one open design
   question in the programme.**
2. ~~**Wire `installer/build.ps1`.**~~ Built 2026-09-04. The build syncs the
   full resolution first, collects the packs against it (unpacked sizes exist
   nowhere but an installed tree, and the CUDA 12.8 assertion is what proves
   the collected wheels are the cu128 ones), then syncs the staged runtime
   down to `--extra studio`. `packs.json` is staged beside `pyproject.toml`
   and the bundled wheels into `{app}\packs`, which is
   `service.packs.bundled_dir`. `runtime-manifest.json` deliberately does
   **not** gain them: it is verified against the *checkout* before anything is
   built, and these three files do not exist at that point — they are pinned
   by digest in `packs.json` instead, by the generator that made them, and
   `pack_worker` refuses a bundled wheel that does not match before it goes
   near site-packages.
3. ~~**The Settings pane.**~~ Built 2026-09-04 as its own category beside
   Models: models are weights and packs are the code that reads them. Rows
   carry both volumes' figures; Cancel is offered while it downloads and
   withdrawn once pip starts writing into the running site-packages.
4. ~~**Say what a finished install means.**~~ Answered 2026-09-04, and it is
   both, in that order: the landing re-runs `doctor.run_checks(force=True)`
   the way a finished fetch does, and a module that still will not resolve in
   *this* process (`service.packs.unresolved`, after the import caches are
   invalidated) asks for a restart out loud rather than leaving a mode grey.

5. **Collect the other two packs once, on a real line**, and record the
   figures. Only `rig` has ever been collected; `text2image` and `music` are
   multi-gigabyte and `music` is the only one that exercises the sdist build
   path (three compiles, one of them a C extension). It is also the only one
   that will ever exercise the bundled-wheel branch of `pack_worker.collect`,
   which is today covered by tests alone.

**Expected outcome:** an installer whose base download is around a gigabyte,
three packs a user chooses, and a figure for each of the two that have never
been collected.

## P28. Judge the character render benchmark — four verdicts, not one

**Why it is yours:** art, and it is the entry the whole character programme was
built to reach. Everything up to the pixels is measured, tested and structural:
thirty-one species over four body plans, four clip libraries, union framing
against a table (`docs/measurements/2026-09-05-union-framing.md`), a structural
check that flags every clipped and blank cell. None of that is the question. The
question is whether a 64-pixel sprite of a wolf reads as a wolf, and no test
this repository can write answers it.

**Why it is four verdicts.** The original scope was one fire ogre, and that was
written when there was one body plan. A convincing humanoid walk tells you
nothing about a quadruped: a four-beat lateral-sequence gait has diagonal pairs
moving out of phase, and at 64 px the legs are two pixels wide and overlap for
most of the cycle — it either reads as walking or reads as a smear, and which
one is not derivable from the humanoid's result. A wing beat and a blob's surge
are two more separate questions. So this entry does not close until it has four
answers, and it may well close as *proven, proven, repair, proven*.

**Do**, once per archetype — `humanoid`, `quadruped`, `winged`, `amorphous`:

1. **A representative species at the ladder's middle.** Suggested: `ogre` (it
   carries the fire theme, so it judges the effects pass at the same time),
   `wolf`, `dragon`, `slime`. 64 px, 32 colours, 8 directions, all five
   movements, **seed locked** — the same seed across the whole sitting, so a
   difference between two sheets is the thing you changed.
2. **Each body slider at both bounds, three seeds.** Six channels for humanoid,
   quadruped and winged, five for amorphous (`family.ARCHETYPES[key].channels`
   is the list; every range is −1 to +1 and every default 0). That is the
   generator's actual span, and the failure it is looking for is a bound that
   produces a body the rig no longer fits — an arm inside the ribcage at
   `limb_length = -1`, a neck the clips swing through at `neck_length = +1`.
3. **Look at every direction and every animation, at zoom 1 and at zoom 4.**
   Zoom 1 is the size a player sees; zoom 4 is where you find out *why*. Troupe's
   heatmap is the reading order, not the verdict — click the flagged squares
   first, then watch the whole thing play.

**What to judge, and it is the same seven questions each time:**

- **Recognisable from the front, and from the back.** The back is the half a
  generator has no reason to get right and the half a player looks at while
  walking away.
- **One identity across the eight directions.** The same creature turned, not
  eight creatures. Cross-direction size drift is what `qa.py`'s `drift_*`
  scores are pointed at.
- **A readable attack.** One frame that is unmistakably the hit, at 64 px, with
  no colour cue.
- **Walk contact.** Feet on the ground at the contact frames, and a bob that
  passes through zero between them.
- **Clean loops.** Idle, walk and run hand their last frame back to their first
  without a pop. Note that a fire theme's flame is *known* not to loop
  seamlessly on a short idle (the noise field does not come back round); judge
  the body separately from the flame.
- **Effects on their sockets.** For the fire species, the flame at the crown or
  the core, rising in world space in every direction, occluded when the socket
  is behind the body.
- **Feet on one line.** Across every direction of one animation, the ground
  line is the same row of pixels — a sprite that floats in three of eight
  directions is unusable however good it looks in the other five.

**Outcome:** a dated character-benchmark document under `docs/measurements/`
that records, **per archetype**, a verdict and the evidence for it. (Named here
by its directory rather than by a `YYYY-MM-DD-` placeholder path:
`tests/test_external_doc_links.py` reads a cited path as a claim that the file
exists, and a placeholder is a citation of a document nobody can open.) That document is
what lifts chapter 11's *"Built, awaiting the render benchmark"* to **Proven —
per archetype, not in one line** — or names the repair, in which case the
repair is code and comes back here as a finding. In the same sitting, calibrate
`qa.THRESHOLDS` against those sheets and record the numbers in
`docs/measurements/2026-09-02-troupe-qa-thresholds.md`, which was written
against synthetic sheets and says in as many words that it is waiting for
rendered motion. P8's authored keyframes are judged through the same sheets;
if the clips change afterwards, the thresholds are re-taken, not patched.

**No card is needed for any of this.** The character route is mesh generation
in-process, Blender on the CPU for the rig and the render, and numpy for the
reduction — which is what makes "run it again at both slider bounds" a
reasonable instruction rather than an afternoon of GPU time.

## P30. Judge the 2D walk cycle — an ogre and a humanoid

**Why it is yours:** art. The motion is *correct* and that is all a test can
say: `tests/inker/walk/` pins that no limb ever changes length, that the stance
foot is on the ground line on every frame it should be, that the stance foot
travels backwards and never forwards, that the cycle hands its last frame back
to its first, and that one rig renders the same bytes twice. None of that is the
question. The question is whether a drawing cut into fourteen rigid pieces and
turned about its joints reads as a body walking or as a paper puppet rotating,
and no assertion this repository can write answers it.

**Why an ogre and a humanoid, and not one of them.** They fail differently. A
humanoid has thin limbs whose silhouette is mostly outline, so the failure to
look for is joint gaps and the seam where an upper arm's end leaves its lower
arm's start. An ogre is mass: thick limbs overlap through most of the cycle, so
the failure is a limb reading as a flat card sliding over another flat card,
which is exactly what a rigid cut-out is. A verdict taken on one is a claim
about that build, not about the feature.

**Do:**

1. **Draw or open one side-view ogre and one side-view humanoid.** Layers per
   body part if you have them; otherwise one layer and the marquee, which is the
   path most users will take and therefore the one worth walking.
2. **Set both up and record how long it takes**, honestly, including the part
   that is not the tool: repairing artwork that was never drawn to be cut. A
   torso with an arm painted over it has no torso underneath, and the panel
   cannot invent one. If the preparation dominates, that is the finding.
3. **Play each at native size and at 4x**, and judge seven things separately:
   readable steps; the knee bending forward and not backward; gaps at the
   joints; limb overlap where two pieces cross; silhouette stability from frame
   to frame; the loop seam between frame eight and frame one; and whether the
   arms read as opposing the legs rather than merely moving.
4. **Bake both, and export a sheet**, to confirm the timing survives the trip
   and the frames are the ones the preview showed.
5. **Try the far-limb shading slider at 1.0 and at 0.6.** Whether a copied far
   limb needs shading to read as behind the body is the one control decision
   taken here without evidence.

**If it reads as a rotating paper puppet, write that down before anything is
expanded.** That is the milestone, and a negative answer is a real one: it
would mean the next move is deformation -- head and torso counter-motion,
squash on the contact frame -- rather than more directions or more actions, and
it is much cheaper to learn that from two drawings than from a second workflow
built on top of the first.

**Deferred on purpose, and not to be built until this closes:** saved rigs,
regenerating a baked walk from its rig, other directions, other actions,
automatic segmentation, and any AI reconstruction of occluded parts. Each of
them multiplies whatever this verdict says, in whichever direction it says it.

**Expected outcome:** one sentence per figure saying whether the walk is
convincing, a recorded preparation time, and — if the answer is no — the
specific thing that broke the illusion.

## P29. Prove the update path against a real release

**Why it is yours:** it needs a release published under your account and an
older build to offer it to, and neither is something this repository can do to
itself. Everything up to that point is built and tested: `service.updates`,
`pipelines/update_worker.py`, Settings -> Updates, and
`scripts/make_update_manifest.py`. The real check runs today against the public
`jmbell88/warlock-studio` and correctly reports "up to date", because that
repository has published no releases at all -- which is exactly why the
interesting half is unproven.

**Do**, once:

1. Build the installer (`pwsh scripts\rebuild.ps1`), then
   `uv run python scripts/make_update_manifest.py`.
2. Publish a GitHub Release carrying **both** `dist\WarlockSetup-v<version>.exe`
   and `dist\update-manifest.json`.
3. On a machine (or a build) whose `__version__` is lower, open Settings ->
   Updates and go Check -> Download -> Run Installer.

**What would fail:** an asset name that does not match what the manifest pins
(the check refuses, correctly, and says which); a release published without the
manifest (the app says "up to date" and offers nothing, which is the designed
behaviour and needs to be seen once so it is not mistaken for a bug); and the
installer refusing to upgrade an install it is running beside, which is the one
thing no test in this repository can reach.

## Also owed, smaller

- **Tutorial sample assets** (art): a 32×32 `.ora` sprite with a few layers
  and frames for the Inker chapters; a 16 px tileset of sixteen to twenty-four
  tiles with a terrain set for Plotter and Packwright; a low-poly `crate.glb`
  for Clay; ~~the humanoid from P4 for Troupe~~ — struck 2026-09-05: chapter 11
  now opens on Create's Character type, and the thirty-one shipped species *are*
  its samples, in the build, needing no file and no licence. Original and
  project-licensed. If they land: `src/warlock/assets/tutorial/`, added to the
  hatchling force-include, under about a megabyte, used by the last section of
  chapters 05, 07, 09 and 10.
- **Delete the pre-purge mirror** at `D:/Projects/_archive/warlock-pre-purge.git`
  once you have worked in the rewritten repository long enough to be
  satisfied. Keeping it indefinitely means keeping the problem indefinitely.

## P30. Decide which of Inker's four right-hand panes gives up height

**Why it is yours:** art direction. Four panes want the same 750 px and the
question is which one matters least while drawing, which is a judgement about
how the mode is used rather than a fact about the code.

**Where it stands.** Inker's five export doors moved out of the timeline's
second toolbar row into the bridge on 2026-09-05 (`3476f114`), because that row
was measured overflowing at 1280x800 scale 1.0: three of the five collapsed into
a `...` menu, "Skip empty" was clipped mid-word, and the row beneath it was cut
off by the pane's bottom edge. The exports the row existed for were the part of
it a user could not see.

The move fixed that and moved the pressure. `inker-generate` now stacks Drawing
file, Export and the exits, under Preview and Tools in the same column, and at
1280x800 the last three doors, the collapsed **Sheet options** header and the
four exit buttons are below the fold. **Nothing is unreachable** -- the pane is
an ordinary scrolling child -- and the exercise harness reports `clipped: 5`
against a measured baseline of `clipped: 1`, that one being **Revert to
original**, which was already below the fold before this work.

This is a soft overflow where there was a hard one, which is why it shipped. It
is still worse than it should be.

**Do** -- one of these, and it is a choice, not a defect to grind at:

1. **Give Preview less.** It is the largest of the four and it is a playback
    surface; a shorter one may cost nothing while drawing.
2. **Compact the doors to two columns.** Three rows instead of five, about
    76 px back. It costs the labels: "Export per layer..." does not fit ~130 px,
    so they would have to shorten, and the label is the door.
3. **Put Export behind a collapsed header**, as Sheet options already is. Keeps
    the exits visible and adds a click to every export -- which is the thing the
    move just removed, so this is the weakest of the three.
4. **Decide it is fine.** A sidebar that scrolls is a sidebar that scrolls, and
    1280x800 is the floor rather than the common case.

Whichever is taken, `scripts/exercise_mode.py inker` reports the clipped count,
so the result is measurable rather than a matter of opinion about a screenshot.

## Open findings

Code work a review or a real run turned up. Each is buildable and is struck out
the day it is built; this section is deleted when it is empty. All four below
came out of the 2026-09-05 clean-machine install
(`docs/measurements/2026-09-05-clean-machine-install.md`) — two installs, four
app sessions, and her `warlock.log` read — and all four were built the same day.

1. ~~**F1. A failed fetch discarded everything it downloaded.**~~ Built
   2026-09-05. `fetch_one`'s unwind was `except BaseException:
   rmtree(staging)`, and `huggingface_hub` keeps its resume bookkeeping in
   `.cache/` *inside* `local_dir`, so a failure threw away the ability to
   resume along with the bytes — one engine attempt ran eight minutes and
   several GB and was discarded whole. The tree is now kept and keyed by a
   `.warlock-resume.json` marker holding the entire spec; a later fetch resumes
   only on an exact match, a terminal failure (digest mismatch, missing rename
   source, no digest) still drops it, both sweeps spare a marked tree, and both
   transports retry with backoff over the same tree. The destination is
   untouched by any of it — nothing moves until the tree is whole and verified.
   `docs/INVARIANTS.md` carries the reasoning; five tests in
   `tests/test_fetch.py` carry the claims.
2. ~~**F2. A socket error reached a non-developer verbatim.**~~ Built
   2026-09-05. `download.describe_failure` translates by `winerror`/`errno`,
   walking the `__cause__` chain because the transports bury the number, and
   falls back on class name for `hf_xet`'s Rust errors which carry none.
   Offline, reset, timeout, refused and DNS are five remedies where there was
   one stringified exception; a sentence this project wrote itself is passed
   through untouched; anything unrecognised names itself and points at the log.
   The raw exception now goes to `warlock.log` beside the friendly one, because
   this incident was diagnosed from a log and the translation must not have
   made that harder. It lives in `download.py` rather than `fetch_worker.py` so
   it can be imported without setting `HF_HUB_OFFLINE=0`.
   `tests/test_fetch_messages.py`.
3. ~~**F3. The health poll imported torch while pip was writing it.**~~ Built
   2026-09-05. `vram.probe` and `doctor._cuda_check` caught `ImportError`
   only, and a half-written torch raises from the DLL loader — `OSError
   [WinError 126]`, then `PermissionError [WinError 32]` — so the whole health
   task died, five tracebacks in twenty-one seconds. `probe` now falls back to
   NVML on any import failure and the CUDA row reports "installed but will not
   load" with the likely cause, non-fatal, since it clears on the next launch.
   `tests/test_torch_import_failures.py`.
4. ~~**F4. The pack gate at a mode's door was never written.**~~ Built
   2026-09-05. `Pack.modes` was read in one place, a Settings label, so a base
   install sent the user to Models to fetch ~23 GB that could not run without
   `torch`. `model_gate.mode_gate` now answers packs-or-models for a mode and
   **packs come first**, since weights with nothing to read them buy nothing;
   the rail, its tooltip and `set_mode`'s refusal all read that one answer, and
   the library escape applies to both halves so nobody is locked out of
   finished work. `tests/test_pack_gate.py`.

**What is left on that machine is not code**: whether the resets stop once a
retry can outlast them (F1 and F2 together should turn "never finishes" into
"finishes eventually"), and the still-unrun `HF_HUB_DISABLE_XET=1` experiment.
Both belong to P1 step 4 now rather than here.

---

## Closed records (kept so nobody re-derives them)

- **P2, purge `examples/` from history.** Done 2026-09-03: `git filter-repo`,
  then the remote deleted and recreated rather than force-pushed, because
  GitHub keeps unreachable objects fetchable by SHA. 963 commits and both tags
  survive; `git log --all -- examples/` is empty; `master`'s tree hash did not
  move. Three traps for the next rewrite: `filter-repo` deletes `origin`; it
  migrates remote-tracking refs into local branches and rewrites those too;
  `gh repo delete` needs the `delete_repo` scope. The mirror is the item under
  *Also owed*.
- **P3, the graded mesh run.** Done 2026-09-02: props-v1 on trellis.cpp
  v0.6.0 is 11 of 22 usable, fantasy-v1 10 of 20
  (`docs/measurements/2026-09-02-trellis-060-props.md`,
  `docs/measurements/2026-09-02-fantasy-v1.md`); the hole audit closed
  (`docs/measurements/2026-09-02-hole-audit-vs-grade.md`). The tex-res pin
  survived as the new P3.
- **P5, the end-to-end `charsheet` run.** Struck 2026-09-05, absorbed into P28
  rather than answered. Its premise was "a card": the sheet job had never run on
  hardware and P4's mesh was what it was waiting for. Neither holds now — the
  character route builds mesh, rig and sheet with no GPU at all and no supplied
  file, so the run is a press rather than a scheduled event, and P28 makes it
  four times over with something to judge at the end of each. A one-line "it
  ran" verdict would have been strictly less than that.
- **P9, code signing.** Answered no for the closed beta on 2026-09-03. The
  revisit triggers and the priced option (Azure Trusted Signing) are in
  `docs/INVARIANTS.md`.
- **P27, the release candidate and its figures.** Built and measured
  2026-09-05 into `INSTALL.md`: 810 MB download (846,950,916 bytes), about
  1.4 GB installed base, SHA-256 `254b3af9...`, at v0.0.35. The placeholders it
  existed to replace are gone. Installing that build on a machine that is not
  this one is the surviving half of P1.
- **P10's tile-sheet half.** Shipped 2026-08-29 as Materials and Terrain set,
  with the old path labelled *Grid (legacy)*; the verdict is P15.
- **P11's phase-6 design.** Decided 2026-09-02 and built 2026-09-03: the merge
  happens in Inker, three-way, and on conflict the hand edit stands.
- **P12, humanoid reconstruction from a single image.** Answered no on
  2026-08-30: limbs come back bent and stretched at the shipped default
  (`docs/measurements/2026-08-30-sdxl-cfg-props.md`). Phase 7 stays deferred;
  the supplied-base-mesh path is untouched.
- **P18, `style_lock`.** Built 2026-08-30 as the *Keep one style across the
  list* checkbox on the Materials arm, with its cost beside it.
- **The 3.12 CI leg.** Read 2026-09-03: fourteen failures, six of them rig
  paths that fail rather than skip without `bpy`. The floor was raised to 3.13
  (`bpy` is 3.13-only and the installer packs its own 3.13 runtime).
- **Host commit (D1/D2/D3).** Closed 2026-08-22 by the t2i child process;
  `docs/measurements/2026-08-22-trampoline-child-pids.md` and
  `docs/INVARIANTS.md` hold the figures and the stdin-reader rule.
- **Release audit (2026-08-24).** `REPORT.md` deleted 2026-08-25 once every
  code-closable finding closed: GPL-3.0, sdist allowlist, licence disclosure
  in the picker, notices staged into the installer.
- **GPU lane.** 26 passed, 0 errors on 2026-08-21; the isometric guide and the
  3/4 clause remain unproven on a card.
- **Art direction and palettes.** Ramps installed 2026-08-21
  (`~/.warlock/palettes/cosmos`, `light_world`).

## Not on this list on purpose

Decisions with arguments beside them, not backlog:

- **Scale and crop of a tilemap layer** stay refused, permanently. They
  resample, and a tileset cannot follow a resample.
- **A hexagonal 120° tile rotation** stays refused: not a permutation of the
  pixel grid. `docs/COMPAT.md` carries the argument.
- **Pen/tablet pressure, ICC colour, per-frame palettes** and the rest of the
  Aseprite parity programme's named non-goals, in `docs/INVARIANTS.md`.
  Per-cel opacity and z-index were built on 2026-08-30 and struck from this
  list (`docs/measurements/2026-08-30-cel-z-below-cache.md`).
- **An LLM director for Troupe.** No LLM infrastructure exists, a local HTTP
  endpoint would be the first socket in the app besides the trellis client,
  and it would break `HF_HUB_OFFLINE=1`. The user approves a *picture*, which
  is a better interface than a manifest.

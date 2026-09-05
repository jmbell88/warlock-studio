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
3. **Open audit findings** (the section at the end): code work the 2026-09-04
   static audit found and did not fix. Each is buildable and is struck out the
   day it is built; the section is deleted when it is empty.

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

## P1. Run the installer on a clean machine

**Why it is yours:** hardware. The installer was built for the first time on
2026-08-26 and rebuilt at 0.0.31 on 2026-09-03 (2.91 GB payload, ~800 s
compile, 6.61 GB installed; single exe, `DiskSpanning=no`). Two things it
reproduced both times and which are the path, not an edge case: `iscc` is not
on PATH (`-Iscc` is needed) and the default index does not serve cu128, so the
pinned-index retry in `build.ps1` is load-bearing. A default per-user `/SILENT`
install was proved the same day. Everything below has never been seen on a
machine that is not this one.

**Do:**
- Install once with `/DIR=C:\Temp\WarlockApp /SILENT`. With `$env:WARLOCK_HOME`
  at a scratch directory: the Start Menu shortcut launches under `pythonw`, the
  checkout-shape gate passes, the fatal banners name the two missing-weights
  rows, the first-run overlay shows correct GPU verdicts, ~23 GB and the disk
  check, and the wizard's first page renders `LICENSE`.
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
SmartScreen's "More info → Run anyway" (code signing was answered no for the
closed beta; see Closed records).

**The installer changed under this entry on 2026-09-04** (P26): it stages
`--extra studio` alone and the three heavy extras arrive as packs from
Settings, so the 2.91 GB payload and 6.61 GB installed above are the *old*
shape and one of the things this run measures is the new one. Two steps join
the list: the base install must reach the window with no torch and no `bpy`
(Create, Poser, Troupe and Muse present and saying what they need), and one
pack must be installed from Settings on that machine — `rig` is the cheap one
at 0.32 GiB — after which Poser opens without a restart.

**Expected outcome:** a first **non-developer** install that generated an
asset. Until a machine without `uv`, Python or a CUDA toolkit has installed
this and made something, the project has no shippable artifact, whatever the
tree says.

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
rigged, +Z up, A-pose-ish and 4,672 polys, so **P5 is runnable**. It does not
close this entry: a ramp verdict taken on a 3,273-vertex specification sample
with a small JPEG and no female variant is a claim about CesiumMan, not about
character art anyone would ship. Putting it through the path found and fixed
three silent rig defects (`_strip_incoming_rig`, `tests/test_rig_supplied_mesh.py`,
`docs/measurements/2026-08-30-art-verdicts-preregistration.md` Q5).

**Expected outcome:** the first Troupe sheet with real colour, and a verdict on
whether the ramp works at sprite scale. Unblocks P5 and P11.

## P5. Run a `charsheet` job end to end against real Blender

**Why it is yours:** a card. Troupe Phase 4's job has never run on hardware.
The pieces either side of it have, and the render call is `rigging.sheet_spec`
+ `run_worker` exactly as `_sheet` makes it — but the end-to-end run is owed,
and it is how you find out that the clip edits from P8 reach a rendered sheet.

**Do:** with P4's mesh (or CesiumMan), **Send to Troupe**, wait for the rig and
the sheet, open the sheet in Troupe.

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
P5. This is the most important art task in the programme.

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
skirts are a separate problem. The supplied-base-mesh path (P4/P5) is untouched
by P12's verdict and is the one to build on.

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
against itself (P7), Troupe's sheet job has never run against real Blender
(P5), and Warlock-written `.aseprite` files have never been opened in Aseprite
(P6).

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

## Also owed, smaller

- **Tutorial sample assets** (art): a 32×32 `.ora` sprite with a few layers
  and frames for the Inker chapters; a 16 px tileset of sixteen to twenty-four
  tiles with a terrain set for Plotter and Packwright; a low-poly `crate.glb`
  for Clay; the humanoid from P4 for Troupe. Original and project-licensed. If
  they land: `src/warlock/assets/tutorial/`, added to the hatchling
  force-include, under about a megabyte, used by the last section of chapters
  05, 07, 09, 10 and 11.
- **Delete the pre-purge mirror** at `D:/Projects/_archive/warlock-pre-purge.git`
  once you have worked in the rewritten repository long enough to be
  satisfied. Keeping it indefinitely means keeping the problem indefinitely.

---

## Audit 2026-09-04 — open findings

The third kind of entry. A ten-slice static audit on 2026-09-04 (after the two
sweeps that landed as `b19a9d47` and `cb019390`) found these and did not fix
them. Each is buildable; strike it out the day it lands, with the regression
test the audit rule requires for Medium and above. Delete the section when it
is empty. No Critical was found.

**High**

- **Muse: Play after Stop discards the loop region, crossfade and position.**
  `panes/muse_player.py` `_transport` calls `seek` (a no-op when nothing is
  sounding) then `muse_mode.play`, which always re-submits the decode and
  replaces the whole `Player` with defaults. Resume in place with `_play_from`
  when the loaded player already holds this job's PCM. Test in
  `tests/test_muse_player.py`: loop + xfade survive a stop/play cycle.
- **Doctor: the fatal TRELLIS GGUF row is green on a zero-byte file.**
  `doctor._gguf_check` uses `fetch.present`, which is `is_file()`; only the
  base-model branch calls `fetch.suspect_files`. Fold the engine kind in,
  mirroring the base branch.
- **Three undo doors push a step per drag frame** — the class `cb019390`
  fixed in seven siblings: Plotter's layer-stack Opacity (`plotter_layers.py`
  `_opacity_row`), Packwright's Padding and Extrude
  (`panes/packwright_settings.py`), and the tileset Terrain tab's Wang-colour
  swatch and probability (`panes/plotter_tileset_editor.py` →
  `MapDoc.replace_tileset`, which pushes unconditionally). Add
  `controls.fold_undo` between field and write; add each door to
  `tests/test_undo_gesture_doors.py`. The swatch is a popup picker, whose
  sliders fold per component — verify that case rather than assume it.

**Medium**

- **Doctor: every other model row has the same zero-byte blind spot** (LoRA,
  IP-Adapter, ControlNet, music, separation, pose, matting loops). One shared
  helper, one zero-byte test per kind.
- **Doctor: `_exe_check` and `_gltfpack_check` accept a directory.** `exists()`
  → `is_file()`. A directory named `trellis-server.exe` must fail the fatal row.
- **Inker: the layer-opacity drag can orphan its undo step.** `inker_menu.py`
  seeds the "before" value with `setdefault` and pops only on deactivation; an
  interrupted drag leaves the change unrecorded and poisons the next drag's
  "before". Refresh on `is_item_activated()`, or use `fold_undo`.
- **Troupe QA: a blank frame can lose its "worst" ranking.** `qa.score_sheet`'s
  blank branch never raises `worst_ratio`, so a later marginal warn overwrites
  it. Give blank a ratio above any metric.

**Low**

- **Clay: `ObjectPropsEdit`, `MaterialEdit`, `MaterialListEdit` carry no
  `cost`**, so they weigh zero to eviction.
- **`service.files.job_dir_file` trusts its `name` argument by docstring
  alone.** Reject separators and `..`, or check against `MEDIA`.

**Docs** (each a one-line edit unless noted)

- `README.md` lists twelve modes and omits Muse.
- `docs/manual/20-overview.md` lists Sirens twice and out of rail order.
- `THIRD-PARTY-NOTICES.md` omits Hybrid Demucs and claims every checkpoint
  comes from Hugging Face.
- `CONTRIBUTING.md` says "three extras" under a four-extra command, and
  "~12,000 tests" where the suite is 16,000+.
- `CHANGELOG.md` 0.0.32 never announces Muse as a mode.
- `docs/manual/39-installation.md`'s extras table omits `music`.
- `docs/manual/40-configuration.md` does not list `WARLOCK_T2I_IN_PROCESS`.
- `docs/MODELS.md` has an unclosed code fence swallowing the rigging heading.
- `SECURITY.md`'s untrusted-file list omits `.wsng`.
- `docs/manual/31-plotter.md` says "Choose image…" for a button that reads
  "Choose..."; `docs/manual/08-rigging-and-posing.md` capitalises "Adjust
  Joints".

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
- **P9, code signing.** Answered no for the closed beta on 2026-09-03. The
  revisit triggers and the priced option (Azure Trusted Signing) are in
  `docs/INVARIANTS.md`.
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

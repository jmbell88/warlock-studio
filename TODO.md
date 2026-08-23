# TODO.md — everything still owed, and who has to do it

Written 2026-08-21, consolidating the four plan files that had accumulated at
the root and in `docs/`. `MY_TODO.md`, `LPC_ALT.md` and `EXE_PLAN.md` were deleted
in that change, along with the two interop ledgers whose owed passes are now in
§4; git holds every one of them and `git log --all --diff-filter=D` finds them.

**This file has two kinds of entry and no others.**

1. **Work only a human can do**: art direction, authoring keyframes, opening a
   file in real Aseprite or real Tiled, running a card. None of it is derivable
   from the tree and none of it can be closed by writing code.
2. **Work that is fully specified and deliberately unstarted** — today that is
   the open Troupe phases and the three host-commit defects in §8. Each is here
   with the argument that makes it actionable, not as a title. The installer was
   in this class until 2026-08-22; it is built now, and what §7 still carries is
   the verification, which is the first kind of entry.

**The moment an item could be built, it is built and struck out rather than
tracked.** A plan whose boxes disagree with the tree is worse than no plan, and
that is why every other roadmap file in this repository's history was deleted
rather than ticked. Nothing here is blocked on finding time.

**This file has no `§N` API.** The old `TODO.md §N` citations resolved against
a *different*, long-deleted file; all fourteen were rewritten on 2026-08-21 to
name the programme or the measurement document they meant, and
`tests/test_ux_todo_fixes.py` now refuses any citation of this filename from
`src/` or `scripts/`. Do not mint new ones. What a module needs to explain, it
explains where it is — or in `docs/INVARIANTS.md`, or in a
`docs/measurements/` document, both of which outlive any plan.

---

## 1. Art direction — blocks the whole Troupe pipeline

**The single highest-value thing on this list.** Everything downstream of it is
built and waiting.

- [x] ~~**Two or three pixel-art references** whose look you want.~~ **Supplied
      2026-08-21**, in `examples/`: `light_world.png` (the LTTP light world map
      — 4110×5136, indexed, 204 colours, the only one of the four that carries a
      real palette), `zelda_1.jpg` and `player_male_spritesheet.png` (the
      character bar: four directions, five rows, chunky limbs, dark outline).
      **The two character references are colour-destroyed and can only ever be
      look references** — the JPEG carries 19,909 colours and bakes transparency
      in as a checkerboard, and the sheet is a smooth upscale at 53,088. A ramp
      pulled from either would be JPEG ringing and bicubic interpolation
      wearing the name of art direction. A clean PNG rip is a minute's work if a
      character ramp is wanted.
- [x] ~~**Palette ramps**~~ **Installed 2026-08-21 — the directory had been
      empty**, which is *why* this blocked: `colors.gpl` sat in `examples/`
      where nothing reads it, so every pixel export and every Troupe sheet to
      date ran on median cut. `~/.warlock/palettes/` now holds **`cosmos`** (The
      Cosmos, 64, the chosen bar — pass it as Troupe's `palette` param) and
      **`light_world`** (64, derived from the map's own index table). Both
      verified through `service.palettes.load` and `pixel.map_palette`.
      They stay *user data and out of git*, per the standing rule the manual
      states: nothing ships with the app, because a palette is a decision about
      which colours exist.
      One finding worth keeping, recorded in the derived file's own header: a
      coverage rule alone would have kept the map rip's **canvas** — a flat grey
      covering 14.29% of the sheet, ranking *first* by coverage — so the
      derivation drops any colour filling complete scanlines before it ranks
      anything. No map feature spans 4110 px; padding does.
- [ ] **A textured base mesh** (or a run through the Phase 4 reference chain).
      Both `examples/*_base.obj` carry no texture, so every frame still
      quantises into the pale end of whatever ramp it is given — **and now that
      the ramps exist, this is the only thing between them and a verdict.** The
      palette is provable on 2D today (a tile sheet has real colour) and stays
      unproven *on Troupe* until this lands.

## 1b. Tutorial sample assets — art, so they are yours

The tutorial chapters (`docs/manual/01-13`) were deliberately written not to
*need* shipped starting material: every exercise either starts from nothing
("Ctrl+N, draw something", "place a box and a sphere") or from a file the
reader already has ("import a tileset image"). That is why the series ships
complete without this item, and why this is an improvement rather than a
blocker.

What would improve them is a small set of **original, project-licensed** files,
and each is a piece of art rather than a build step:

- [ ] **A 32x32 sprite** (`.ora`, a few layers, two or three frames) for the
      Inker chapters, so "draw something" has an alternative for a reader who
      would rather learn the tools than design a character in the same ten
      minutes.
- [ ] **A 16 px tileset** for Plotter's and Packwright's exercises. Small --
      sixteen or twenty-four tiles with a terrain set among them, so the
      terrain brush has something to resolve.
- [ ] **A low-poly `crate.glb`** for Clay's import exercise.
- [ ] **A textured, rigged humanoid `.glb`** for Troupe's no-GPU path. This is
      the same file as the base mesh in section 1 and should be authored once
      for both.

Two constraints that are not negotiable. **`examples/` cannot supply any of
them**: it is ULPC-derived, CC-BY-SA/GPL, and reference-and-validation only, and
the manual would be shipping it. And nothing procedurally generated should
stand in -- a generated 32x32 "sprite" would be a picture of nothing, shipped in
the wheel, teaching a reader that this is what the tool produces.

If they land, `src/warlock/assets/tutorial/` is the place (add it to the
hatchling force-include beside `docs/manual/`), keep the set under about a
megabyte, and the "Try it" sections that would use them are the last section of
chapters 05, 07, 09, 10 and 11.

## 2. Author the 22 keyframes — the editor is built for exactly this

Open Poser → **Clips** in the left sidebar. Pick a key, pose the skeleton with
the normal gizmos, **Update key from pose**. Onion skin ghosts the keys either
side; **Play** scrubs the real interpolation. **Save clips** writes to your own
data folder and never touches what the build ships, so **Revert** is always
available and an update cannot overwrite you. Manual: *Poser → Editing clips*.

- [ ] **Author the keys.** The shipped 22 are provisional. This is the most
      important art task in the programme, and the one thing that cannot be
      automated away — moving authoring from frames to keyframes does not make
      animation good, it makes it cheap to fix. A bad clip reproduces exactly
      the "stiff posing" flaw being escaped.
- [ ] Decide **who does it** — you, or an animator. Still a scheduling
      question.

Two things worth knowing before you start:

- **Easing does nothing at the current segment lengths.** It reshapes *where
  inside a step* frames land, so it needs a step of ≥3 frames. Every shipped
  step is 1 or 2, and `ease` is a smoothstep whose value at the only interior
  sample is exactly 0.5 — so `idle`'s `ease` renders identically to `linear`
  right now. `ease_in`/`ease_out` do differ. The panel says so; it is mentioned
  here because otherwise you would change it, see nothing, and assume the field
  is ignored.
- **The arms hang slightly forward** on the shipped keys (noted at the Troupe
  0d spike, still true). Worth an art pass while you are in there.

## 3. Decisions owed

- [ ] **Where does the "judge clips as pixels" preview live?** The Troupe plan
      asked for a live low-res *sprite* preview in Poser. **It cannot go
      there**: `template_preview` builds an armature-only GLB over the canonical
      unit box, so Poser's preview is a *meshless armature* and there is nothing
      to pixelise — a sprite preview would show reduced bone lines. Either Poser
      learns to load a rigged asset for preview, or the pixel verdict stays in
      Troupe where the mesh is. The scrubber shipped instead as the fast loop,
      which is right either way. **Your call which direction.**
- [ ] **Troupe Phase 6 — the cleanup workflow.** See §6 below. It is three
      features and the hard one is a genuine design problem, not an
      implementation task. Worth a conversation before anyone writes code.
- [ ] **Decide what happens to `plotter-wave-2`.** It last moved 2026-08-14 and
      `master` has advanced 262 commits since; it holds 52 unmerged commits.
      Per the Plotter Wave 2 record it is gated on the user-authored Tiled
      fixtures in §4 and on a final whole-branch review — so §4 has to land
      first either way. The three outcomes are: rebase and finish it, cherry-pick
      what still applies, or delete it and let git history hold it. **A branch
      delete needs an explicit ask**, so nothing happens here without one.

## 4. The interop passes — need an app this repo does not have

Both follow the standing rule, now stated once at the top of `docs/COMPAT.md`:
a green test proves this app's reader and this app's writer agree with *each
other*, and a round trip through our own two halves cannot catch an error both
halves make together. **The claim only strengthens once a human with the app
has looked.**

- [ ] **Open a Warlock-written `.aseprite` in real Aseprite.**
      `tests/inker/fixtures/aseprite/FIXTURES.md` names the four fixtures worth
      authoring first. **Start with the tilemap ones** — `tilemap-rgb`,
      `tilemap-indexed`, `spare-tileset`. Their chunk field order was written by
      inverting the *reader*, field for field, and has never been checked
      against a file Aseprite itself wrote. That is the highest-value five
      minutes on this whole list.
      In the same sitting: every RGB and grayscale file now carries a palette
      chunk derived from the art's own colours (divergence #23) — including a
      1-entry transparent palette on a blank document. Check Aseprite is happy
      with that rather than replacing its own default with a single swatch.
- [ ] **Author a `.tmx`/`.tsx` fixture in real Tiled.** Every map under
      `tests/plotter/fixtures/tiled/` was produced by this editor, so every
      `round-trips` row in `docs/COMPAT.md`'s Tiled part is currently a round
      trip *against ourselves*. `tests/plotter/fixtures/tiled/FIXTURES.md` lists what
      authoring is owed and in what order.
- [ ] **Move `tsx.TILED_VERSION` from `1.10.2` to `1.12.2`** — but only after
      the above. The constant is pinned below the 1.12.2 target deliberately;
      nothing in this repo can satisfy the gate, and it has been wrongly bumped
      once already (the bump was reverted rather than the gate satisfied).
- [ ] **Re-check a grid pack's `.tsx` geometry.** pow2 rounding is off by
      default now, so the standing verification is stale.

## 5. GPU runs — need your card

- [x] ~~**`uv run pytest -m gpu -n 0`.**~~ **Run 2026-08-21: 21 passed, 5
      errors**, and the five were one stale attribute rather than a defect on
      the card — the three-views change renamed `SheetGeometry.projection` to
      `.view` and `tests/test_tilesheet_gpu.py` was the one caller it did not
      reach, because that file only runs when someone asks for it. Fixed the
      same day. Serial is still enforced, and this is still the only lane that
      sees real weights and the real `~/.warlock`.
- [x] ~~**Re-run the one file the errors ate**~~ **Run 2026-08-21 as part of a
      full `uv run pytest -m gpu -n 0`: 26 passed, 0 errors, 104 s**, with
      `tests/test_tilesheet_gpu.py` contributing all five. Its two load-bearing
      claims — the model puts its seams where the guide drew them, and the
      reduction keeps a cell's contrast — are asserted again for the first time
      since the view vocabulary widened. Top-down only; the isometric diamond
      guide and the 3/4 clause remain unproven on a card, and that is still a
      separate ask.
- [ ] **Run a `charsheet` job end to end against real Blender.** Troupe Phase
      4's job has *never* been run on a card. The pieces either side of it have
      (Phase 0d), and the render call is `rigging.sheet_spec` + `run_worker`
      exactly as `_sheet` makes it — but the end-to-end run is owed, and it is
      how you would find out that the clip edits from §2 actually reach a
      rendered sheet.

## 6. Troupe — the open phases

Phases 0a–0d, 1, 2, 3, 4 and 5 are **implemented and verified**; what they
established lives in `docs/INVARIANTS.md`, and the measured ULPC facts the
programme rested on are passing oracles in `studio/troupe/ulpc.py`. What
follows is what is left.

### Phase 0e — humanoid reconstruction from a single image

Untested. Run a prompt → reference → TRELLIS → `fit_template` pass on a humanoid
and judge **limb separation and silhouette** — at sprite scale, face-level
fidelity barely matters and those two matter enormously. This is the largest
unproven assumption in the automated chain, and it is compounded by a known
property rather than a bug: Warlock is single-image-only for reconstruction, so
**the back is hallucinated**, and a humanoid with separable limbs is a harder
subject than a prop.

It matters **only for the generated-character path**. The supplied-base-mesh
path works without it, which is why the programme did not stop for it.

### Phase 6 — the cleanup workflow

Export mostly exists (`inker/sheetout.py`, `aseout.py`, `packwright/tsxout.py`).
The work is the loop:

- **Propagate a correction** across frames / direction / animation — fix a pixel
  once, apply it everywhere compatible. Inker's ranged ops (`_doc_ranges.py`)
  are most of the machinery. Must go through the write funnel (`_commit_patch`)
  and address by uid, per `docs/INVARIANTS.md`.
- **Mirror-assisted cleanup.** The W/E mirror property was measured on the
  reference sheets: mirroring leaves 36–37 differing pixels, confined to the
  face, and every non-zero shift is far worse (443px at ±1) — so it is real
  facial asymmetry rather than a centring offset. A fix on one side can be
  offered on the other, face excluded.
- **Re-render one animation without discarding hand edits.** The hardest
  workflow problem in the programme and the one most likely to be discovered
  too late. **Design it deliberately rather than on contact** — that is the
  standing instruction, and it is why this phase is a conversation before it is
  a commit.

### Phase 7 — layered equipment (deferred)

Swappable gear, once whole-character generation works.

- **Multi-GLB scene composition.** `op_sheet` takes one `source_glb`, and
  equipment items are separate assets by construction — so the task is
  *composing* N GLBs under a shared camera, not splitting one. `op_rig` joins
  every mesh into a single object, which is exactly why splitting is a dead end
  and composition is the right shape.
- **Per-part passes with depth**, giving correct per-direction occlusion for
  free — no z-order table to maintain. The depth machinery is proven in
  `op_views._depth_material` and can move onto the sheet path.
- **Garment fitting**: skin-weight transfer from the weighted body by proximity
  (Blender `Data Transfer`). Tractable for hugging garments; capes and long
  skirts are a separate problem — scope to hugging first.

### Phase 8 — reconsider only against a working system

- **AI restyle** — `create_pixel_sheet` with `structure_lock` over a rendered
  sheet. Note `structure_lock` is only a Canny-ControlNet toggle; what actually
  keeps silhouettes exact is `pixelsheet.remask()` stamping the render's own
  alpha back unconditionally. Also note `check_restylable` refuses
  `frame_size × columns > 1024`, so a 512-render 8-column sheet cannot be
  restyled without banding. Opt-in, measured, never default.
- **A learned pixel refiner.** Once cleanup is routine, `(render, hand-cleaned)`
  pairs accumulate for free, perfectly registered, over a fixed palette. A
  well-posed supervised problem — and it automates the Dead Cells cleanup step
  rather than trying to relearn pose transfer, which the geometry already
  solved.
- **More animations.** The spec is extensible by construction; hurt, death, cast
  and climb are additive.
- **Natural-language character description** — only over a working catalog, only
  local weights through `fetch_worker`, following the `expand.py` precedent.

### Inputs the repo cannot produce

Beyond the art direction in §1, a base mesh must satisfy: **GLB/glTF**
(`blender_worker._import_glb` is the only importer on this path); **T-pose or
A-pose** (both `fit_template`'s bbox-proportional fit and automatic-weight
skinning degrade badly on a dynamically posed mesh); **+Z up, −Y forward**;
bone names mapping onto the 19-bone template if it ships rigged (Mixamo or
Rigify naming is fine but needs a mapping table); **no very short bones** —
Blender silently deletes a bone below a minimum fraction of the mesh's largest
dimension *and takes its children with it*, a quiet failure, with fingers and
toes the usual casualties; **under ~300k faces**, because automatic weights on
a 300k-face mesh is minutes of CPU; **male and female variants**; and a
**licence permitting commercial redistribution of derived rendered sprites**.

The reference sheets in `examples/` are ULPC-derived and CC-BY-SA/GPL:
**reference and validation only** — keep them out of shipped art and out of any
training set, and the question stays closed.

## 7. The installer — built, unverified on real hardware

Written 2026-08-16 as a six-phase plan. **Phases 1 through 4 landed in `594619e`
(2026-08-22)** and phase 5 is all but the manual pass. What is left is phase 6,
and every step of it needs hardware, so none of it can be struck out from here.

What exists now, so nobody re-plans it: `src/warlock/__main__.py` (the pythonw
null-stdio shim); the `engine` fetch kind — `models.EngineModel`/`ENGINE_MODELS`
with `trellis_gguf`, `fetch.KINDS`' first entry, and doctor probing through the
registry; `src/warlock/studio/panes/first_run.py`, the hardware-preflight and
required-download overlay; `installer/build.ps1`, `installer/warlock.iss`, and —
beyond the original plan — `installer/runtime-manifest.json` plus
`installer/verify_runtime.py`, which pin every vendored executable and DLL by
path, size and SHA-256 and reject an unlisted file before ISCC runs. The
checkout-shaped contract is recorded in `docs/INVARIANTS.md` and in
`config.source_checkout()`'s docstring; `installer/README.md` is the build
instructions. `tests/test_installer.py` covers the manifest.

### What a human has to do

**Run the build.** `pwsh installer\build.ps1` on a Windows machine with `uv`, a
uv-managed CPython 3.13 and Inno Setup 6. Confirm the staged smoke test and the
`torch.version.cuda == "12.8"` assert both pass, and record the payload size and
compile time — nothing here has ever been run end to end.

**Settle single-exe versus disk spanning.** `warlock.iss` currently sets
`DiskSpanning=yes` with a 2.1 GB slice for a ~4 GB compressed payload. Whether
one exe compiles is empirical; decide it at the first compile and drop the
spanning if it is not needed.

**Install and exercise it.** Per-user default, then once with
`/DIR=C:\Temp\WarlockApp /SILENT`. In a shell with `$env:WARLOCK_HOME` pointed at
a scratch directory: the Start Menu shortcut launches under pythonw, the
checkout-shape gate passes, the fatal banners name the two missing-weights rows,
and the first-run overlay shows correct GPU verdicts, ~23 GB and the disk check.

**Prove the fetch pipeline under the bundled interpreter.** Download one *small*
row end to end (dinov2, 0.4 GB) — that alone proves worker spawn, verify, publish
and banner refresh. Then SDXL and one reference generation. For TRELLIS, copy an
existing `~/.warlock/models/trellis2-gguf` into the scratch home to prove the
engine launches from `{app}\vendor` and reconstructs, and separately start the
engine download and cancel it mid-flight to prove staging cleanup — that avoids a
second 16 GB pull.

**Upgrade and uninstall.** Over-install a scratch version: clean slate, data
intact. Uninstall: `{app}` gone, `~/.warlock` intact.

**Then the laptop**, which is the reason the programme exists and the one machine
whose GPU is unknown.

### Notes that survive

- An unsigned exe means SmartScreen's "More info → Run anyway". Accepted.
- An old hand-downloaded partial GGUF set reads "missing" under the named-ten
  probe. That is correct by the partial-reads-absent rule, not a bug.
- The pythonw shim runs before any pygame or imgui import; a bare `print(` in
  studio would still be fine, but it is worth a grep if stdio ever misbehaves.

## 8. Host commit — a model load never gives its host memory back

Measured 2026-08-21 on a live session, from the app's own `warlock.log` plus
`Get-CimInstance Win32_Process`. It was here rather than built because the first
defect is a **design question** with three possible shapes, and the other two
are the instrumentation that decides which one is right.

**State on 2026-08-22.** D2 is built and D3 is refuted; both were settled by a
second session that reproduced D1 on a larger machine, plus three direct probes
(`docs/measurements/2026-08-22-trampoline-child-pids.md`). The instrument is now
right, and the answer it gives is worse than the figure this section was written
against: **`flux_klein_distilled` charged +21.1 GiB of host commit and gave
0.1 GiB back**, against the `host_peak_gib=16.0` the registry ships, on a run
that ended with the app refusing its own sprite-sheet follow-up at 94% commit.
D1's shape was chosen — **option 1, the t2i child process** — and it is now
built. **Nothing in this section remains open**; it is kept as the record of a
defect that took two sessions and three measurements to close, and the figures
below are what the app was doing before it was.

**The session that produced this.** Two reference generations, ten minutes
apart, on a 63.46 GiB machine with a 12.6 GiB pagefile — a 76.06 GiB commit
limit:

| time | event | app private | commit |
| --- | --- | --- | --- |
| 18:17:54 | idle at startup | **2.3 GiB** | 52.3/73.0 (72%) |
| 18:18:48 | `sdxl-base-1.0` done, VRAM 6.58 → 0.01 | **11.3 GiB** | 61.3 (84%) |
| 18:19:43 | `dreamshaper-xl` done, VRAM 6.74 → 0.02 | **13.6 GiB** | 63.6 (84%) |
| 18:25:54 | idle, six minutes later | **13.5 GiB** | 70.9 (93%) |

The card was empty throughout the tail (1.9 of 32.6 GiB). The session ended in
the app refusing its own work:

```
CRITICAL warlock.queue: host commit at 92% before unloading dreamshaper
  -- at or past the ceiling the 2026-08-03 crash hit; further jobs will be refused
RuntimeError: host memory is 92% committed before loading SDXL 1.0
  (full CFG, structural control), at or past the 90% ceiling.
```

- [x] ~~**D1 — the t2i load path pays an unreturnable host cost in the app
      process.**~~ **Built 2026-08-22 as option 1, the t2i child process.**
      `pipelines/text2image_worker.py` holds the checkpoint,
      `pipelines/t2i_client.Text2ImageClient` is the app-side handle presenting
      the surface `Text2Image` presented, and `WARLOCK_T2I_IN_PROCESS=1`
      restores the old arrangement for debugging. Measured on both sides, same
      machine, same checkpoint, load plus one 1024x1024 sample: **in process
      24.26 GiB charged / 0.26 returned; in the child 24.08 charged / 24.08
      returned**, with system commit back at its baseline. Both klein entries'
      `host_peak_gib` went 16.0 → 24.0, because the old figure priced the
      weights and forgot the sample.

      Three things the build learned that the design below did not know. The
      boundary was **narrower than feared** — `generate()` already wrote to an
      `output_path` and `Conditioning` is a frozen dataclass of paths and
      floats, so no pixels cross the pipe and no call site changed. The
      resident-pipe design was **not** the cost it was billed as: the child is
      persistent, so a warm pipe survives between jobs exactly as before. And
      the real cost was somewhere nobody was looking — a concurrent stdin
      reader deadlocks the child's next native-extension import on Windows,
      which took most of the session to find and is written up in the
      measurement and in `docs/INVARIANTS.md`.

      Original design note follows.

      **D1 — the t2i load path pays an unreturnable host cost in the app
      process.** The `host_peak_gib` comment in `models.py` states the
      assumption this breaks: *"under RESIDENT the weights are read and handed
      to the device, so the host charge is **transient** and roughly the
      checkpoint's size."* It is not transient. `unload()` returns the VRAM
      exactly as it claims (6.58 → 0.01 GiB) and the host figure does not move
      — +9.0 GiB for the first checkpoint, +2.3 for the second, flat across six
      minutes idle. **Each distinct checkpoint pays its own**, so the ceiling is
      reached by switching models, not by running many jobs.

      The mechanism is already measured and already written down:
      `docs/measurements/2026-08-08-load-probe-memory.md` found that dropping
      every reference plus `gc.collect()` recovered 422 MB of BiRefNet's 1475 —
      **1053 MB, 71%, held by the allocator's arenas.** That document is *why*
      `loadprobe` and `matting_worker` are child processes. The t2i loader is
      the one path paying the same cost in the process that has to keep
      running, where nothing but exit reclaims it.

      **The decision owed is which fix**, and it is a conversation before it is
      a commit:
      1. **A t2i child process** — the trade `matting_worker` and `loadprobe`
         already made, and the only option that genuinely returns the memory.
         It costs the resident-pipe design: the pipe object the idle sweep
         deliberately keeps would live across a process boundary, and every
         LoRA / ControlNet / IP-Adapter handle with it. Large; do not start it
         casually.
      2. **Raise `host_peak_gib` to the truth and let admission refuse
         earlier.** Cheap and honest. Makes the app *correct* rather than
         better — the second checkpoint of a session becomes a refusal instead
         of a crash. Worth doing regardless of 1, because the shipped figures
         price a load that is released, and this one never is.
      3. **Accept it and recycle the process** — an explicit "restart to
         reclaim" affordance, which is what the user does today unprompted.

      ~~**Do not rank these until D2 lands.**~~ D2 landed 2026-08-22 and the
      corrected number chose for us: at 21.1 GiB retained per checkpoint,
      option 2 alone turns klein-then-anything into a refusal at the door on a
      77 GiB machine, and option 3 is that refusal with a restart attached.
      **Option 1 is the shape being built.**

      Two things the second measurement changed about its cost. The boundary is
      narrower than feared: `generate()` already writes to an `output_path` and
      `Conditioning` is a frozen dataclass of paths and floats, so pixels cross
      as files — `matting_worker`'s rule — and only `on_state`, `on_step` and
      `cancel_event` need a protocol. And `vram.device_memory()` reads through
      `sys.modules.get("torch")`, so moving the one thing that imports torch in
      the app process out of it leaves that reading `None`; admission needs
      device-wide free VRAM from somewhere that is not a parent torch import.

- [x] ~~**D2 — every child pid the app records is the wrong process, so the
      idle-tick reports `children 0.0 GiB` against a 6.56 GiB child.**~~
      **Built 2026-08-22.** It reproduced exactly on a second session, on the
      tick that then refused the next job. The fix is *not* the one sketched
      below: no spawn site changed. The kill-on-close job already holds the
      whole tree — a process created by a process in a job is assigned to that
      job at creation — so the register existed and was simply never read.
      `winjob.job_pids()` reads it, `winjob.measured_pids()` falls back to
      `tracked()` when the job is unarmed, and the two `memlog.summary` call
      sites take that instead. Measured [trampoline, real, …] against
      [trampoline] alone; it is also correct under §7's installer layout, where
      the job holds exactly one pid per worker.
      Sketch kept below for the reasoning, which was right about the cause:
      `sys.executable` under this uv venv is a **trampoline**, not an
      interpreter: it spawns the real CPython as its own child and stays alive
      as a ~0.8 MB parent. Demonstrated directly —

      ```
      sys.executable : D:\Projects\Warlock\.venv\Scripts\python.exe
      Popen.pid      : 3144
      child says pid : 2960      <- not the same process
      ```

      So `winjob.track(proc.pid)` records the shim. `memlog.children_private`
      dutifully opens it, reads 0.8 MB and rounds to zero — while the matting
      worker beside it holds **6.56 GiB**, which is the exact figure
      `children_private`'s own docstring cites as its reason for existing.
      Admission is **not** affected (`_require_commit_headroom` reads
      system-wide commit, which counts the grandchild), so this is a reporting
      defect and not a safety one. It is also the instrument D1 will be judged
      with, which is why it goes first.

      The fix is to record the pid that holds the memory rather than the one
      `Popen` returned, and it touches every worker spawn — `blender_worker`,
      `fetch_worker`, `matting_worker`, `loadprobe` — not just matting. Note
      that **the installer removes this by construction**: §7's layout gives
      `sys.executable` as a real `{app}\python\python.exe`, so the fix must not
      assume a trampoline is always there.

- [x] ~~**D3 — `matting.unload()` would orphan the process it means to kill.**~~
      **Refuted 2026-08-22 by direct measurement — no fix is owed.** A 200 MB
      grandchild behind the trampoline, killed exactly the way `unload()` kills
      it, died with its parent within 0.5 s. `TerminateProcess` does not
      cascade, as stated; the uv trampoline does the cascading itself, holding
      its child in a kill-on-close job of its own. A true premise carried a
      false conclusion, so the refutation is written down
      (`docs/measurements/2026-08-22-trampoline-child-pids.md`) rather than
      left to be re-derived. The caveat that survives: this is a property of
      *uv's* trampoline and not of Windows, so the guarantee to rely on remains
      the job object rather than the kill. Original reasoning:
      Same root cause as D2, latent rather than observed. `unload()` calls
      `proc.kill()` on the trampoline, and Windows `TerminateProcess` does not
      cascade to children, so the 6.56 GiB grandchild would survive — reparented
      — until the app exits and the kill-on-close job takes it. That is the
      precise opposite of the module's stated purpose, that *"only a process
      that ends can"* return the memory. It has not fired in a session yet (both
      pids were alive when this was measured), so there is no incident to point
      at; it is waiting for the first idle sweep that calls it. The
      kill-on-close guarantee is unaffected either way — job objects *do*
      cascade — so nothing outlives the app.

**Not a repo item, but it is what the machine looked like.** `mysqld` held
8.40 GiB of commit with a *zero* working set, untouched since 2026-08-17; the
21.4 GiB standby cache is reclaimable and was never the problem; and the commit
limit is only 76.06 GiB because the pagefile is 12.6 GiB on a 63 GiB machine.
Warlock's own ~20 GiB was the largest movable share. Any measurement of a fix
has to name the other tenants, or it will credit itself with their departure.

## 9. Not on this list on purpose

These are decisions with arguments beside them, not backlog items waiting for
time.

- **Scale and crop of a tilemap layer** stay refused, permanently. They
  *resample*, and a tileset cannot follow a resample — there is no permutation
  to teach, only a re-cut, which is a different operation.
- **A hexagonal 120° tile rotation** stays refused. A 120° rotation of a square
  raster is not a permutation of the pixel grid, and the standing bar for a tile
  transform is that it invents no colour. `docs/COMPAT.md` carries the full
  argument, including what would have to change to re-open it.
- **Pen/tablet pressure, ICC colour, per-frame palettes, per-cel opacity** and
  the rest of the Aseprite parity programme's named non-goals, recorded in
  `docs/INVARIANTS.md`.
- **The Aseprite P1 backlog** stays unscheduled *by design* — items are pulled
  into sessions individually, never waved. It lives in `docs/INVARIANTS.md`.
- **An LLM director for Troupe.** Warlock has no LLM infrastructure, an
  OpenAI-compatible endpoint over local HTTP would be the first socket in the
  app besides the trellis subprocess client, and it would break
  `HF_HUB_OFFLINE=1`. Under the chosen flow it is also unnecessary: the user
  approves a *picture*, which is a better interface than a manifest.

# Troupe — an automated character sprite factory

*An alternative to the Universal LPC Spritesheet. Named to match Warlock's other modes, which are
all craft nouns: Inker, Clay, Poser, Plotter, Packwright.*

## Decisions taken

| | |
|---|---|
| **Name** | Troupe |
| **Target sizes** | 16px minimum, 128px maximum |
| **Animations** | Designed fresh — **Idle, Walk, Run, Attack, Jump**. Not LPC's set. |
| **Directions** | 8 |
| **Bodies** | Male and female variants; no other fixed body set |
| **Base mesh** | Supplied by the user |
| **Human input** | One approval gate on the 2D reference; hand-cleanup in Inker afterwards |

---

## Context

The goal is an alternative to Universal LPC, not a consumer of it. The two sheets in `examples/`
are reference and validation material only.

**The product:** a user writes a prompt, is shown a 2D character reference, and **approves it**.
Everything after that gate is automatic — reconstruction, rigging, posing, rendering at 8
directions, pixelisation, and layout into a sprite sheet. The result is then hand-authored and
cleaned in Inker. Long build time is acceptable if the result justifies it.

The four failings of LPC this must beat: everyone's game looks identical; 64×64 is too low-res to
carry detail; the art is flat and muddy; the proportions and posing are stiff.

### What the reference sheets actually are (measured, not assumed)

- **832×3456** — a 13×54 grid of 64×64 cells, 352 occupied. Not 493×2048. This is the ULPC "full"
  layout exactly: `(7,8,9,6,13)×4 dirs` + `6 + 6` (hurt, climb, single-direction) +
  `(2,5,3,3,8,2,13,6)×4` = `172 + 12 + 168 = 352`.
- Direction order is **N, W, S, E**, confirmed empirically: walk rows 9 and 11 are near-exact
  mirrors (mean diff 0.25 vs 17+ for every other pairing); row 8 has zero iris-coloured pixels;
  row 10 has exactly twice as many as the sides.
- **16 unique colours, every pixel alpha 255.** No antialiasing anywhere — losslessly indexable.
- **W and E mirror losslessly except the head.** Mirroring leaves exactly 36–37 differing pixels,
  confined to rows 17–25 / cols 24–40 (the face). Every non-zero shift is far worse (443px at ±1),
  so this is real facial asymmetry, not a centring offset. `sit` and `backslash` are handed
  exceptions (173–185px).
- Male and female differ in 325 of 352 cells and ~22k silhouette pixels.

These become free regression oracles in Phase 0, and the mirror property is directly useful in
Phase 6 cleanup.

## Why 3D-derived

Hand-authoring is impossible at this scale, and per-frame AI generation is impossible for a
different reason — at hard-edged, alpha-255, low-colour pixel art a single drifted pixel is
visible in every frame, and there is no averaging to hide it.

Every one of the four complaints is expensive in 2D and nearly free from geometry:

| Requirement | 2D-native | 3D-derived |
|---|---|---|
| 8 directions instead of 4 | ×2 all authoring | one render parameter (already the default) |
| Larger canvas | ×(area) all authoring | one render parameter |
| Non-stiff posing | redraw every frame | fix one clip, every character inherits it |
| Per-user variety | a bigger fixed library | a different prompt |
| Occlusion per direction | hand-maintained z-order table | the depth buffer |

**ULPC's real constraint is that it threw the geometry away** — hence a decade of work, art
flattened toward a contributor-average, and "one design → hundreds of poses" as its hardest
problem. Warlock did not throw the geometry away.

The precedent is **Dead Cells**: 3D models animated in 3D, rendered orthographically to sprites,
palette-reduced, then hand-touched-up. A tiny team produced an enormous volume of animation that
reads as hand-drawn. That is exactly the arrangement chosen here — automate the volume, hand-clean
the result.

### The authoring-cost inversion

`rigging.py:178-182` states the property that makes this work:

> *"a pose authored against one humanoid rig applies to every other humanoid rig — which is what
> makes a shipped library possible at all"*

| | 2D-native | 3D-derived |
|---|---|---|
| Authoring unit | one frame of pixel art | one keyframe pose |
| Full set, per character | 256 frames | **0** — inherited |
| Full set, once ever | 256 × library size | **~22 keyframes** |

---

## The frame table

Proposed, to be fixed in Phase 0:

| Animation | Frames | Loop | Unique keyframes |
|---|---|---|---|
| Idle | 4 | yes | 3 |
| Walk | 8 | yes | 4 (contact A, passing A, contact B, passing B) |
| Run | 8 | yes | 4 |
| Attack | 6 | one-shot | 5 |
| Jump | 6 | one-shot | 6 |
| **Total** | **32 per direction** | | **~22** |

`32 × 8 directions = 256 frames per character.`

**At 128px this is 1024×4096 in an 8-column layout — comfortably under `MAX_ATLAS_PX = 8192`
(`sheet.py:39`), so one atlas per character works.** Per-animation atlases remain an export
option, not a requirement. For contrast: the LPC set at 8 directions would have been 692 frames
and 1024×11136, which does not fit.

Loop and one-shot map onto Inker's `Tag` model directly — `loop`, `direction`, `repeat`
(`animation.py:268-320`).

### Size ladder

`pixelsheet.downscale` is integer-stride only. From a 512px render, **16 / 32 / 64 / 128 divide
exactly**; 24 / 48 / 96 do not — which is precisely why `spritesynth.reduce_atlas` exists as a
single NEAREST resize. The ladder is therefore exact on the power-of-two rungs and uses that
documented fallback elsewhere. `PIXEL_LOGICAL_SIZES` is currently `(16,24,32,48,64)` and needs 128
added.

---

## What already exists

This is not a from-scratch problem. **Every stage of the chain ships today** as a separate, tested
feature; the work is orchestration plus four real gaps.

| Stage | Exists as | Anchor |
|---|---|---|
| Prompt → 2D reference | `reference` job stage, SDXL + ControlNet + IP-Adapter | `pipelines/reference.py` |
| Reference → mesh | `trellis-server.exe` subprocess | `pipelines/trellis.py` |
| Mesh → rigged | `fit_template` + Blender automatic weights | `rigging.py:251`, `blender_worker.py:683` |
| Interactive posing | GPU-skinned `PoseEditor` with undo, mirror, presets | `studio/viewer/pose.py:114` |
| Shipped rig-portable poses | `idle, walk contact A/B, run, attack, hit, death` | `templates/poses/humanoid.json` |
| Pose × yaw → atlas | orthographic EEVEE, native-resolution, v1 sidecar | `pipelines/sheet.py`, `blender_worker.py:776` |
| 8 directions at 30° | **already the default** | `sheet.py:29`, `:33` |
| Atlas → layered timeline | `document_from_atlas` with tags + `DirectionalLayout` | `studio/inker/sheetin.py:67` |
| Frame editing | full Aseprite-parity editor: layers, cels, tags, indexed colour, undo by uid | `studio/inker/` |
| Export | PNG+sidecar, `.ase`, `.ora`, GIF, Tiled `.tsx`, TexturePacker | `inker/sheetout.py`, `aseout.py`, `packwright/tsxout.py` |

**This call works today, unmodified:**

```python
create_sheet(svc, job_id, yaws=8, frame_size=256, elevation=30.0,
             lighting="flat", clip_from=<pose id>, clip_to=<pose id>, clip_frames=9)
```

It yields an 8-column atlas rendered natively at 256px by an orthographic camera framed once from
the rest bbox (`blender_worker.py:800-811`), with a sidecar naming every cell's
pose/yaw/frame/pivot/trim — openable in Inker, packable in Packwright.

The shipped pose library already contains `idle`, `walk contact A`, `walk contact B`, `run` and
`attack` — five of the poses this program needs, already authored.

**The Frame Editor from the original concept is Inker.** Pencil, eraser, fill, colour-replace,
layer move, mirror, per-frame layered editing, undo — all built. Troupe produces a document; it
does not rebuild an editor.

### The four real gaps

1. **Clips are 2-key.** `sheet.interpolate` (`:179`) takes exactly `pose_a, pose_b` and linearly
   slerps. Eight frames from contact A to contact B is *half a stride played straight* — no
   passing pose, no vertical bob, no foot plant. It also **explicitly refuses any root
   translation** (`:196-201`), which is what a bob requires. Ironically the 2D path already knows
   better: `templates/sprite_guides/walk.json` describes contact/passing/contact/passing and body
   bob. That knowledge never reached the 3D path.
2. **No AI-free pixeliser.** `create_pixel_sheet` always queues an SDXL img2img job. All the pure
   primitives exist — `pixel.downscale_grid` (`:278`), `map_palette` (`:344`, nearest in Oklab
   with spacing-scaled ordered dither), `snap_alpha` (`:395`), `clean_orphans` (`:410`) — and
   nothing wires them to a sheet.
3. **`lighting="flat"` is not toon shading.** `_make_flat` (`blender_worker.py:487`) rewires Base
   Color into an Emission node — unlit albedo. No ramp, no Shader-to-RGB, no Freestyle, no outline
   pass anywhere. Also `_setup_render` never sets `taa_render_samples` for a sheet, so a native
   low-res render comes back antialiased and soft — not crisp pixel art.
4. **Rendered sidecars carry no timing.** The `animation` block (`frames[{cell_index,
   duration_ms}]`, `tags`) is defined at `sheet.py:412` but written **only** by the Inker exporter;
   `_q_rig.py:423-432` passes just `pivot=` and `trims=`. An engine gets frame indices with no fps
   and no loop tags. `sheetin.walk_tags()` is hardcoded to the 4×4 spritesynth layout.

### What is explicitly cut

**The LLM director.** Warlock has no LLM infrastructure. The only text model is
`pipelines/expand.py` — a 124M CPU GPT-2 whose generation is constrained to a shipped whitelist so
it can only append aesthetic vocabulary. A repo-wide grep for
`openai|ollama|llama_cpp|langchain|chatcompletion` over `src/`, `pyproject.toml` and `docs/`
returns exactly one hit: the word "completions" in that file's docstring. An OpenAI-compatible
endpoint over local HTTP would be the first socket in the app besides the trellis subprocess
client and would break `HF_HUB_OFFLINE=1` (`src/warlock/__init__.py:9`, pinned by
`tests/test_offline.py`). Under the chosen flow it is also unnecessary — the user approves a
*picture*, which is a better interface than a manifest.

**PySide6 / Qt.** Warlock is one process, one ModernGL context, imgui drawn through it.

**A parts catalog in V1.** Because each character is reconstructed whole, swappable layered
equipment is not a precondition for shipping. It moves to Phase 7.

---

## Architecture

```
  prompt (+ optional image)
          |
   [ 2D reference stage ]  SDXL + Canny guide -> orthographic T-pose character
          |
      ***  USER APPROVAL GATE  ***        <- the only human step before cleanup
          |
   [ reconstruct ]  trellis-server.exe -> GLB
          |
   [ auto-rig ]  fit_template -> Blender automatic weights -> rig.glb
          |          (fallback: hand joint correction in Poser)
          |
   [ clip library ]  ~22 keyframes, authored once against the base mesh
          |
   [ render ]  orthographic EEVEE, 8 yaws x 32 frames, 1 sample, flat
          |
   [ pixelise ]  supersample-reduce -> designed palette -> alpha snap -> outline
          |
   [ lay out ]  frame table -> atlas + sidecar with tags & timing
          |
   [ Inker ]  hand-author and clean; layers, tags, undo, propagate
          |
   [ export ]  PNG+JSON, .ase, Godot, Tiled
```

Four new units, each with one job:

- **`studio/troupe/`** — headless engine: layout spec, frame table, clip model, character
  document, deterministic composition. Imports no imgui/moderngl/pygame/`service`, pinned by
  `tests/troupe/test_troupe_imports.py` following `tests/packwright/test_packwright_imports.py`.
- **`pipelines/charsheet.py`** — pure planning: clip × yaw → frame table → `Plan`. No torch, no
  `service`.
- **`pipelines/pixelize.py`** — the AI-free pixeliser (gap 2).
- **`service/characters.py`** — the doors and the chain orchestrator, modelled on
  `service/tilesheets.py`, the cleanest recent exemplar of a new job kind.

---

## Status, 2026-08-19

**Phases 0b, 0c, 1, 2 and 3 are implemented and verified.** 10,605 passing, ruff clean;
`tests/troupe/` is 103 of those. What shipped:

| | |
|---|---|
| `studio/troupe/spec.py` + `data/layout.json` | the frame table as versioned data; import-pinned to an empty outward set |
| `studio/troupe/ulpc.py` | the 352-cell reader; every measurement in this file is now a passing oracle |
| `pipelines/pixelize.py` | gap 2, the AI-free pixeliser, byte-identical run to run |
| `blender_worker._setup_render(taa_samples=1)` | gap 3's crispness half, on the flat sheet path only |
| `sheet.interpolate_clip` + root translation | gap 1; the refusal is lifted and `service.sheets` no longer routes it |
| `templates/clips/humanoid.json` | the 22 keyframes and five clips, expanding to exactly 4/8/8/6/6 |
| `pipelines/charsheet.py` | clip x yaw -> `Plan`, and gap 4's `animation` block |
| `sheetin.span_tags` / `document_from_sheet` | the handoff, generalised off the 4x4 layout |

**Not done, and why:** 0a/0d/0e need the base mesh, which has not been supplied; Phase 1's
palette and Phase 2's keyframe judgement need the art direction and a render to judge; Phases
4-8 are downstream of 0d, which the plan calls a real go/no-go gate. The clip library is
authored to the frame table's *shape* and has never been seen as pixels -- treat it as a
starting point, not a result.

## Phases

Phases 0–2 are gates. Everything after is construction.

### Phase 0 — Spec, oracles, and the honest spike

**0a. Receive and qualify the base mesh.** Check it against the requirements in *Inputs needed*
below; rig it; confirm no bones are silently dropped.

**0b. The layout spec as data.** `studio/troupe/spec.py` + a versioned data file: the five
animations, their frame counts and loop flags, 8 directions and their order, the size ladder,
camera elevation (30°, already `DEFAULT_ELEVATION`), and anchors. Render at 512 and reduce.

**0c. A ULPC reader, for validation only.** ~60 lines decoding the example sheets into 352 named
`(animation, direction, frame)` cells. Cheap, and it turns every measurement above into a
regression test. It also lets a user bring LPC art in as filler. No ULPC art ships.

**0d. The spike that decides the program.** Rig the base mesh, author one 2-key walk, render 8
yaws at 512, reduce and quantise with the existing pure functions, and *look at it*. The whole
program rests on the claim that 3D-derived pixel art can be crisp rather than mushy — and mushy is
exactly the "flat and muddy" failure being escaped. **A real go/no-go gate.** Cost: days. Cost of
discovering it in Phase 5: months.

**0e. Spike humanoid reconstruction, in parallel.** Run a prompt → reference → TRELLIS →
`fit_template` pass on a humanoid and judge limb separation and silhouette. This is the largest
unproven assumption in the automated chain and it is cheap to test now.

*Deliverable:* spec module, ULPC reader, passing oracle tests, and two spike results to judge.

### Phase 1 — The pixeliser (gaps 2 and 3)

The cheapest high-value work in the program, and it decides whether output looks crisp.

- `pipelines/pixelize.py`: supersample-reduce → **designed palette** → alpha snap → orphan cleanup
  → outline pass.
- **A designed palette, not median-cut.** `pixelsheet.quantize_shared` uses MEDIANCUT, which is
  what produces muddy output. Use `pixel.map_palette` against authored ramps loaded through
  `service/palettes.py` (`.hex`/`.gpl`). This is the highest-leverage art input in the program.
- **An outline pass.** Hard outlines are most of what separates crisp pixel art from a shrunk
  render, and nothing in the repo does this.
- `taa_render_samples = 1` on the sheet render path (`_setup_render`, `blender_worker.py:447`) —
  the depth pass already does this at `:1069`.
- Consider a cel ramp in `_make_flat`. Unlit albedo plus a designed palette may be enough; measure
  before adding shader complexity.

*Deliverable:* a non-AI pixelise path from any rendered sheet. Re-judge the Phase 0 spike.

### Phase 2 — Multi-key clips (gap 1)

- Extend `interpolate` from `(pose_a, pose_b)` to an **ordered keyframe list with per-segment
  frame counts and easing**. This is genuinely small: the slerp, the cell/frame plumbing,
  `op_sheet`'s `(pose, frame)` pose cache (`blender_worker.py:837`) and the sidecar's flat cell
  list are all already frame-aware.
- **Lift the root-translation refusal** (`sheet.py:196-201`) by interpolating the offset. The
  machinery exists everywhere else — `_apply_root_translation` (`blender_worker.py:368`),
  `rigging.root_offset_world` (`:1274`), `_sheet_root_offsets` (`queue.py:576`), and `op_sheet`
  already applies a per-cell `root_offset` (`:841-846`). The refusal is a correctness guard, not
  an absence of capability.
- Clip format in `templates/clips/humanoid.json`, loaded by the existing
  `rigging._load_pose_library` machinery. Cyclic clips (Idle, Walk, Run) close the loop; one-shots
  (Attack, Jump) do not.
- Clip authoring **in Poser**, not a new mode: keyframe list, onion-skinning, and a **live
  low-res sprite preview** so clips are judged as pixels, not as viewport playback.
- Author the ~22 keyframes against the base mesh. Five usable poses already ship in
  `templates/poses/humanoid.json`. **This is the most important art task in the program** —
  "stiff posing" was one of the four complaints, and a bad clip reproduces exactly the flaw being
  escaped.

> `MAX_POSES = 500` (`rigging.py:748`) and `MAX_SHEETS = 200` (`:912`) are comfortable, but the
> clip library must be shipped, not stored one pose per job.

*Deliverable:* real walk and run cycles. Re-judge again.

### Phase 3 — Sheet layout and handoff (gap 4)

- `pipelines/charsheet.py`: clip × yaw → the Phase 0 frame table → `Plan`.
- **Write the `animation` block into rendered sidecars.** It is defined at `sheet.py:412` and
  simply never populated by the 3D path (`_q_rig.py:423-432`). Durations come from the clip; tags
  come from `(animation, direction)` with loop and repeat set per the frame table.
- Generalise `sheetin.walk_tags()` beyond the hardcoded 4×4 spritesynth layout to any
  `(animation, direction)` frame table.
- Handoff into Inker with tags and directional layout already set.

> **Two `DIRECTION_ORDER` tables already exist** — `pipelines/spritesynth.py:64` and
> `studio/inker/animation.py:93` (`DirectionalLayout`, `SHEET_KINDS`) — deliberately duplicated
> because inker imports nothing outward, with `tests/test_sprite_geometry_agreement.py` as sole
> owner of the agreement. **A new 8-direction sheet kind must move both tables and that test
> together.**

*Deliverable:* a spec-conformant, correctly-tagged, engine-ready sheet from a rigged mesh.

### Phase 4 — The automated chain

One job kind that orchestrates the whole run, with a single human gate.

- **Reference generation shaped for reconstruction.** The approved image should be an
  **orthographic T-pose/A-pose character reference**, not a dynamic illustration — it makes both
  reconstruction and the bbox-proportional `fit_template` far more reliable. Reuse the existing
  ControlNet + guide mechanism with a new `templates/sprite_guides/tpose.json`, exactly as
  `spritesynth` already does for turnaround and walk. Male and female variants get their own
  guides.
- **The approval gate is a real UI stage**, not a parameter — the user sees candidates, picks one,
  and only then does the expensive chain start.
- **Auto-rig unattended, with a fallback.** `fit_template` is bbox-proportional and approximate by
  design (`rigging.py:251`), with `adjust_joints` for correction. A constrained T-pose reference
  makes the bbox fit far more predictable. Add a confidence check and, on failure, route the user
  into Poser's existing joint-correction mode rather than failing the job.
- Chain: `reference` → gate → `trellis` → `rig` → `charsheet` → `pixelise` → assemble. Resumable
  per stage, cancellable, with real progress.

> **Known risk, stated plainly:** Warlock is single-image-only for reconstruction and **the back is
> hallucinated**. A humanoid with separable limbs is harder than a prop. At sprite scale
> face-level fidelity barely matters, but *limb separation and silhouette matter enormously*.
> Phase 0e exists to find this out before it can sink the schedule.

*Deliverable:* prompt → approve → finished, tagged, pixelised sprite sheet.

### Phase 5 — The mode

A studio mode following the `packwright_mode.py` skeleton (sidebar / centre pane / inspector),
plus the full registration sweep:

- `modes.py` (`MODES`, `RAIL_GROUPS`, `WORK_MODES`, `WORKSPACE_MODES`, `NAV_KEY_MODES`)
- `state.py` slot; `main.py` workspace dispatch — the dispatch ends in a **bare `else` that draws
  Inker**, so an unregistered mode silently draws the wrong workspace
- `palette.py` `_DOC_MODES`, `panes/landing.py`, `panes/overlay.py` `PLACEHOLDERS`
- `journal.py` `_PROVIDER_MODULES` for crash recovery
- `manual/targets.py` + a numbered `docs/manual/NN-troupe.md` chapter — **adding a chapter is a
  renumbering**, gated both ways by `tests/manual/`

Plus the continuously-animating preview (bad frames become obvious immediately) and **"Open in
Inker"**.

### Phase 6 — Cleanup workflow and export

Export mostly exists (`inker/sheetout.py`, `aseout.py`, `packwright/tsxout.py`). The work is the
cleanup loop:

- **Propagate-correction** across frames / direction / animation — fix a pixel once, apply it
  everywhere compatible. Inker's ranged ops (`_doc_ranges.py`) are most of the machinery. Must go
  through the write funnel (`_commit_patch`) and address by uid, per `docs/INVARIANTS.md`.
- **Mirror-assisted cleanup**: the measured W/E mirror property means a fix on one side can be
  offered on the other, face excluded.
- **Re-render one animation without discarding hand edits** — the hardest workflow problem here,
  and worth designing deliberately rather than discovering on contact.

### Phase 7 — Layered equipment (deferred)

Swappable gear, once whole-character generation works.

- **Multi-GLB scene composition**: `op_sheet` takes one `source_glb`; equipment items are separate
  assets by construction, so the task is *composing* N GLBs with a shared camera, not splitting
  one. Note `op_rig` joins every mesh into one object (`blender_worker.py:59`), which is why
  splitting is a dead end and composition is the right shape.
- **Per-part passes with depth**, giving correct per-direction occlusion for free — no z-order
  table to maintain. The depth machinery is proven in `op_views._depth_material` (`:966`) and can
  move onto the sheet path.
- **Garment fitting**: skin-weight transfer from the weighted body by proximity (Blender `Data
  Transfer`). Tractable for hugging garments; capes and long skirts are a separate problem — scope
  to hugging first.

### Phase 8 — Deferred, to reconsider only against a working system

- **AI restyle** — `create_pixel_sheet` with `structure_lock` over a rendered sheet. Note
  `structure_lock` is only a Canny-ControlNet toggle (`_q_sprite.py:139-152`); what actually keeps
  silhouettes exact is `pixelsheet.remask()` stamping the render's own alpha back unconditionally.
  Also note `check_restylable` refuses `frame_size × columns > 1024` (`pixelsheet.py:77`), so a
  512-render 8-column sheet cannot be restyled without banding. Opt-in, measured, never default.
- **Learned pixel refiner.** Once cleanup is routine, `(render, hand-cleaned)` pairs accumulate for
  free, perfectly registered, over a fixed palette. A well-posed supervised problem — and it
  automates the Dead Cells cleanup step rather than trying to relearn pose transfer, which the
  geometry already solved.
- **More animations.** The spec is extensible by construction; hurt, death, cast and climb are
  additive.
- **Natural-language character description** — only over a working catalog, only local weights
  through `fetch_worker`, following the `expand.py` precedent.

---

## New job kinds

At least `character` (the chain) and `char_sheet` (one animation's render). Per
`docs/INVARIANTS.md:119`, a new kind must sweep **nine stage-keyed tables**, each of which falls
through *silently* for an unknown kind — a button that refuses, or no button, never a failing
test. Four of nine were missed the last time a kind landed.

`files.ready`/`unready_reason` · `files.derived_2d_for` · `progress._PHASES_BY_KIND` ·
`state.primary_action`/`card_kind` · `panes/library._remeshable` · `palette.rerollable` ·
`create_stages.IMAGE_STAGES`/`available` · `widgets.STAGE_BADGES`/`thumbs.thumb_glyph` ·
`validation.DERIVED_PARAMS`

Plus `_q_generate.py` dispatch, `vram.py` per-kind cost, `_q_jobs.py` artifact lists, and
`_jobs_resubmit.py`. Trace with `grep -rn '"tile_sheet"' src/` (12 hits) as the template. Anything
the worker records about artifacts must join `DERIVED_PARAMS`, or a reroll wears a stale verdict.

---

## Risks, honestly

1. **3D-derived pixel art can look mushy — the exact failure being escaped.** The largest risk.
   Mitigated by flat shading, supersample-and-reduce, a designed palette, an outline pass, and
   Inker cleanup. **Phase 0d is a real gate.**
2. **Humanoid reconstruction from a single image is unproven here** — and the back is hallucinated
   by design. Limb separation and silhouette are what matter at sprite scale. Phase 0e.
3. **Clip quality is the new stiffness risk.** Moving authoring from frames to keyframes does not
   make animation good; it makes it cheap to fix. Judge clips as pixels, not as viewport playback.
4. **Auto-rig reliability unattended.** `fit_template` is approximate by design. Constraining the
   reference to a T-pose is the main mitigation; a Poser fallback is the safety net.
5. **Re-rendering after hand edits** is the workflow problem most likely to be discovered too
   late. Design it in Phase 6, not on contact.
6. **The reference sheets are ULPC-derived and CC-BY-SA/GPL.** Reference and validation only — keep
   them out of shipped art and out of any training set, and the question stays closed.
7. **Base mesh licence** must permit commercial redistribution of derived rendered sprites.

## Verification

- **Per phase:** `uv run pytest` (parallel; never edit `src/` while it runs — several tests read
  module source). `uv run ruff check .`
- **Phase 0:** decode both `examples/*.png` into 352 named cells; assert the frame table, direction
  order, and the W/E mirror property measured above.
- **Phase 1:** golden-image tests — fixed render + fixed palette must reduce byte-identically. The
  chain is deterministic, so that is the bar.
- **Phases 2–3:** `uv run pytest -m gpu -n 0` (real card, serial-enforced). Assert a clip's frame
  count, tag ranges, loop flags and durations round-trip through the sidecar into Inker and back
  out.
- **Phase 4:** end-to-end on a fixed seed — prompt → approve → sheet, asserting each stage's
  artifacts and that cancellation at any stage leaves no partial served file (writes onto served
  files are staged, never in place).
- **Phase 5:** run the app, build a character, drive the preview, hand off to Inker, export,
  reopen. The repo's history is explicit that screenshots catch defects tests do not.
- **Throughout:** the import-pinning test for `studio/troupe/`, and a `docs/INVARIANTS.md` entry
  for every invariant this program establishes.

---

## Inputs needed from the user

The base mesh is being supplied. These are the remaining things the repo cannot produce.

**Blocks Phase 0 — mesh specifics:**

- **Format**: GLB/glTF. `blender_worker._import_glb` is the only importer on this path.
- **Pose**: T-pose or A-pose. Both `fit_template`'s bbox-proportional fit and automatic-weight
  skinning degrade badly on a dynamically posed mesh.
- **Axes and scale**: +Z up, −Y forward, per the `humanoid.json` comment.
- **Skeleton**: if it ships rigged, bone names ideally map onto the 19-bone template —
  `hips, spine, chest, neck, head, shoulder/upper_arm/forearm/hand .L/.R,
  thigh/shin/foot .L/.R`. Mixamo or Rigify naming is fine but needs a mapping table. If unrigged,
  `op_rig` fits and skins it.
- **No very short bones.** `rigging.py:61-63` records that Blender silently deletes a bone below a
  minimum fraction of the mesh's largest dimension **and takes its children with it** — a quiet
  failure, not a loud one. Fingers and toes are the usual casualties; they are not needed here.
- **Poly count** under ~300k faces. `rigging.py:65`: automatic weights on a 300k-face mesh is
  minutes of CPU.
- **Male and female variants**, since the spec commits to both.
- **Licence** permitting commercial redistribution of *derived rendered sprites*.

**Blocks Phase 1 — art direction:**

- **Style references.** "Too generic" is the complaint, so Phase 1 needs a target: two or three
  examples of pixel art whose look is wanted. Every palette, outline and shading decision is judged
  against these, and without them Phase 1 has no bar to clear.
- **Palette ramps**, or approval of ones derived from those references, as `.hex`/`.gpl` through
  `service/palettes.py`. Median-cut quantisation is *why* sheets come out muddy; authored ramps are
  the fix.

**Blocks Phase 2 — animation:**

- **Who authors the ~22 keyframes.** This is the one art task that cannot be automated away. Five
  usable poses already ship in `templates/poses/humanoid.json`. If the user authors them, Phase 2
  delivers the Poser clip editor and they drive it; otherwise the schedule needs an animator.

**Not needed:** a parts library, LPC assets, any network service, or a body set beyond male and
female.

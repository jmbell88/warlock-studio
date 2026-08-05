# Asset-Consistency Program — Design Spec

**Date:** 2026-08-04
**Status:** approved design, not yet implemented
**Scope:** two phases — (1) close the measurement loops, (2) standalone 2D assets

---

## Context

`REPORT.md` surveys ComfyUI-era controlled-generation techniques. Exploration of the
codebase showed Warlock already implements most of what that report recommends:

- IP-Adapter appearance conditioning
- ControlNet-Canny structure conditioning (CFG-capable bases only)
- per-stage recipes with model fingerprints
- user style profiles
- host-side reference normalization with enforced composition gates
- a bench harness carrying silhouette-IoU and DINOv2 metrics

So this program is not "add the missing techniques". Four things are structurally
absent, and they are what it fills:

1. **Measurements never act.** `meshaudit` / `meshreport` are display-only; the
   candidates produced by `count > 1` are unranked; nothing ever retries.
2. **No project-level style anchor.** IP-Adapter references are manual per-job
   attachments; style profiles carry only words.
3. **Two UI footguns.** The negative prompt is silently inert on CFG-0 bases
   (`turbo`, `sdxl`), and structure conditioning is only available on
   `playground` / `sdxl_cfg` with no explanation of why it is missing elsewhere.
4. **No standalone 2D output.** 2D exists only as the concept stage feeding
   trellis (`output=reference`).

### Decisions taken during brainstorming

- All four failure modes matter: style drift, run-to-run roulette, 2D→3D fidelity
  loss, structural defects.
- New model weights are allowed where they earn it — one-time manual downloads,
  graceful degradation on the `bpy` precedent.
- Standalone 2D is in scope: icons, sprites, tileables, pixel art, packaged as
  files plus a JSON manifest. No engine adapters.
- Architecture for 2D: derive object assets from references (per-artifact
  derivation idiom); generate tiles as a new output kind (`output="tile"`, with
  circular-padding convolution for native seamlessness).
- One design, two phases: loop-closing first, standalone 2D second.

---

## Phase 1 — Close the measurement loops

### 1.1 UI honesty fixes (footguns)

- In `src/warlock/studio/panes/settings_2d.py`: when the selected base model has
  `guidance_scale <= 1.0`, disable and annotate the negative-prompt field — "this
  model runs CFG 0 — negative prompts have no effect". The inert path is
  `pipelines/text2image.py:509` (`if self.spec.guidance_scale > 1.0`).
- Where the Structure (ControlNet) group is hidden for non-CFG bases, show a
  one-line hint naming which bases support it instead of hiding silently.

### 1.2 Style anchor image in user profiles

- Extend `studio/profiles.py` user profiles (stored in `studio_settings.json`
  under `user_profiles`) with an optional anchor image plus `ip_scale`. Store the
  image as a **file** in a stable per-profile location — not base64 inside the
  settings JSON; the profile records the path.
- When a profile carrying an anchor is applied, auto-populate the IP-Adapter
  reference (the `ref.png` conditioning path resolved at `queue.py:566-626`) on
  every generation under that profile. A manual per-job attachment still
  overrides.
- UI in `studio/panes/profiles_panel.py`: "use current reference as style anchor"
  plus a clear button.
- This is the single highest-leverage anti-drift feature: every asset in a set
  shares one visual anchor, not merely shared taxonomy words.

### 1.3 Candidate auto-ranking (`count > 1`)

- After a candidate fan-out completes, compute a rank score per candidate **on the
  TaskRunner**, never the frame loop: a base score from the existing reference
  report (composition ok, occupancy, aspect), plus DINOv2 cosine against the
  profile's style anchor when one exists. `bench/metrics.py:dino_cosine` already
  exists — reuse it, do not rewrite; A-vs-A comparisons only, per its docstring.
- Store as a param, which means it joins `service.validation.DERIVED_PARAMS` — it
  describes artifacts, per the CLAUDE.md invariant. The gallery / candidate strip
  sorts by it. The score is advisory; the user still picks.

### 1.4 Opt-in auto-reroll on failed reference report

- When a generated reference fails its own composition report (the gate at
  `queue.py:789`), automatically reroll **once** with a fresh `reference_seed`
  before surfacing the failure. Bounded to 1 retry, config-gated (settings + env),
  and recorded in params so provenance shows both attempts.

### 1.5 Opt-in remesh-on-bad-audit

- **Prerequisite: measure first.** Run the bench suite to establish the current
  hole-rate baseline. Project memory records that the old "7–31% holes" claim is
  stale — do not build retry machinery around a phantom.
- If defects still warrant it: after `mesh_audit` (`queue.py:1164-1174`), when the
  worst `hole_fraction` exceeds a threshold, retry in-worker with a fresh
  `mesh_seed`, up to N attempts (default 1), keeping the GLB with the best audit.
  Record every attempt's audit and seed in params. Config-gated, off by default.
- In-worker rather than a new linked job, so the user sees one job that
  self-healed. It must respect the cancel event and the exclusive-mode
  stop-before-load / unload-before-next-start ordering in `Worker._generate`.

### 1.6 Fidelity score (calibration-gated)

- **Step 1:** run `bench/calibrate.py` to answer its own open question — does a
  stable camera-matched view of the mesh exist (a stable yaw/elevation argmax
  against `input.png`)?
- **Step 2, only if the answer is yes:** request-path fidelity measurement —
  render the calibrated view at low resolution, silhouette IoU against the
  reference subject mask (reuse `bench/metrics.py:silhouette_iou`, bbox-cropped so
  it measures shape rather than framing). Store like `mesh_audit` (summary only),
  display in the inspector. Advisory, on the same philosophy as `meshreport`:
  nothing rejects a mesh whose GLB is already on disk.
- If calibration says no stable view exists, this item stops there and the finding
  is recorded.

---

## Phase 2 — Standalone 2D assets

### 2.1 Host-side matting (new optional weight)

- A vendored-weights matting model, one-time manual `hf download`, under the
  models root. **BiRefNet preferred** — it is what `trellis-server` uses
  internally for `bg_removal="birefnet"`, so 2D exports and 3D inputs see the same
  subject boundary.
- Loaded transiently per derivation on the TaskRunner and torn down, on the same
  per-call attach/teardown rationale as ControlNet at
  `pipelines/text2image.py:262-275`.
- Missing weights are non-fatal: a doctor row reports it, and the export UI
  degrades to the existing flood-fill `subject_mask`
  (`pipelines/reference.py:129-171`) with visibly rougher edges — or the user
  downloads the weights. Never a runtime network call.
- This deliberately does **not** change `reference.py`'s rule that the mask is
  never written back as alpha for the 3D path. Matting stays trellis's job there.

### 2.2 Derived 2D exports: icon / sprite / pixel

- New pure module `src/warlock/pipelines/asset2d.py`, under the same purity
  contract as `pipelines/sheet.py` — no imgui / service / queue imports, headlessly
  testable:
  - matte (BiRefNet or the fallback mask) → trim to subject bbox → pad per profile
    → resize to target canvas
  - **icon:** square canvas, centered, configurable padding
  - **sprite:** alpha cutout plus pivot (default bottom-center, recorded in the
    manifest)
  - **pixel:** nearest-neighbour downsample to a target size (32 / 64 / 128) →
    palette quantization (Pillow median-cut, or a fixed-palette param) → optional
    isolated-pixel cleanup
- Served on demand from `done` reference jobs under the existing per-artifact
  `_convert_locks` idiom, exactly as STL/OBJ derive from `model.glb`. Artifacts:
  `icon.png`, `sprite.png`, `pixel_<n>.png`, `manifest.json` in the job dir. Any
  existing reference retroactively gains these exports.
- `manifest.json` records pivot, trim bounds, canvas, palette, alpha-QA results
  (island count, halo check — advisory), and the source recipe hash.
- UI: export controls in the selected-asset inspector for reference-stage assets,
  mirroring the 3D export buttons.
- Sprite stance consistency across a set is the existing ControlNet-Canny with an
  outline template plus the Phase-1 style anchor. A shipped stance-template
  library is future work, not this program.

### 2.3 Tile generation (`output="tile"`)

- A new output kind alongside `reference` / `model`. Generation differences:
  - Patch the resident pipe's UNet/VAE `Conv2d` layers to
    `padding_mode="circular"` for the duration of the job, and restore afterwards
    — the same pipe serves normal jobs. This makes SDXL output natively seamless
    with no inpainting model.
  - A tile-specific prompt template — the current `PROMPT_TEMPLATE`'s
    single-object framing would fight it — with the relevant guidance-field subset
    (material, palette, condition, setting).
  - Skip reference normalization and the subject-composition gates: there is no
    subject. Tile jobs are not promotable to 3D.
- **Seam QA:** offset the result by half its width and height, measure the
  opposing-edge difference, store a seam report on the job — advisory and
  displayed. A wrap preview in the inspector would be ideal but is optional.
- UI: tile as a choice in the 2D pane, which owns all prompt controls per the mode
  invariants.

---

## Critical files

**Phase 1:** `studio/profiles.py`, `studio/panes/profiles_panel.py`,
`studio/panes/settings_2d.py`, `service/jobs.py`, `service/validation.py`
(`DERIVED_PARAMS`), `queue.py` (retry loops, audit hook), `bench/metrics.py`
(reuse), `bench/calibrate.py` (run, then extend).

**Phase 2:** new `pipelines/asset2d.py`, a new matting wrapper module,
`pipelines/prompt.py` (tile template), `pipelines/text2image.py` (circular-padding
patch), the artifact routes in `service/files.py` / the `app.py` successor
(`_convert_locks`), `guidance.py` (tile field subset), `studio/panes/settings_2d.py`
plus inspector panes, `doctor.py` (matting check).

## Invariants to preserve (from CLAUDE.md)

- The frame loop never blocks — all scoring and derivation on the TaskRunner.
- Anything recorded about a finished job's artifacts joins `DERIVED_PARAMS`.
- Fully offline; missing weights fail with download instructions, and are
  non-fatal where optional (the `bpy` precedent).
- Advisory-first QA: nothing fails a job whose artifact is already on disk; gates
  act before expensive work, or as bounded retries.
- Derived artifacts are pure functions of their source, under `_convert_locks`.
- VRAM: transient loads (matting, DINO) respect coexist/exclusive budgeting;
  retries respect the `Worker._generate` ordering.

## Verification

- `uv run pytest` and `uv run ruff check .` throughout. New pure modules
  (`asset2d`, the seam metric, ranking) get headless unit tests with synthetic
  images, as `sheet.py`'s tests do.
- Phase 1 baseline and proof: bench runs before and after (suite `core-v1.json`) —
  a hole-rate baseline for 1.5, a calibrate sweep for 1.6, an A/B with the style
  anchor for 1.2 / 1.3.
- Tile seamlessness: the automated opposing-edge metric plus a manual 2×2 wrap
  render.
- End-to-end: run the app; generate a set of 4 assets under one anchored profile;
  export icon / sprite / pixel from an old reference; generate a tile and check the
  seam report.

## Out of scope (explicitly)

ComfyUI integration (the report's orchestration layer does not apply — Warlock is
its own pipeline), engine adapters (Unity / Unreal), inpainting and img2img, sprite
animation, PBR map derivation for tiles, rigging-ready character work.

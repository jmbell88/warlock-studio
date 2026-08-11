# Changelog

Hand-written, newest first. Nothing here is derived from git: the commit
subjects are `Warlock vN.N.N` and carry no detail, so this file is the only
record of what a version actually changed. The top heading's version must
match `pyproject.toml` — a test asserts it, so a release bump cannot leave this
file behind.

## 0.0.18 — 2026-08-10

- **A sprite sheet from a single drawing.** A finished 2D reference can now be
  turned into a sprite sheet without ever becoming a mesh. A queued
  `sprite_synthesis` job draws two candidate 1024px atlases from two seeds —
  turnaround (2x2: front, left, right, back) or a 4-direction walk cycle (4x4)
  — in one SDXL pass each, conditioned on the reference through the IP-Adapter
  and on a per-cell stick-figure pose guide through the canny ControlNet, with
  the pixel-art LoRA on top. Each candidate is matted per cell, put on a shared
  baseline, reduced to 32/48/64px cells in one NEAREST pass and quantized to one
  palette across the whole atlas. Drafts accumulate under the inspector's
  **Sprite sheet** header with a thumbnail and per-cell notes for each
  candidate; nothing is overwritten and nothing is chosen for you.
- **Turnarounds keep your own drawing.** When the reference's proportions match
  what the model drew, it is pasted back into the front cell *before* the
  reduction and the palette, so the one view that is definitely right shares the
  sheet's colours rather than standing out from them. The sidecar records
  whether it was, and why not when it was not.
- **Drafts open in Inker as editable animations.** **Edit in Inker** slices a
  candidate on the sidecar's own rectangles — one frame per cell, never
  re-detected from pixels — and attaches a *directional layout*: a walk sheet
  arrives with a looping tag per direction, so Play loops one direction at a
  time with no change to the animation engine. The layout is saved in the
  `.ora` (additive; the format version is unchanged) and drives **Export sheet**,
  which then writes the sheet's own fixed grid instead of wrapping, with each
  cell's direction and frame in the sidecar. A timeline that no longer fills the
  grid is refused by name rather than exported with a hole in it.
- The pose guides ship as data (`src/warlock/templates/sprite_guides/*.json`),
  so pose quality is iterable without touching code.
- Manual: a new **From a single drawing** section in
  [Sprite sheets](docs/manual/06-sprite-sheets.md), and the layout paragraph in
  [Inker](docs/manual/07-inker.md).

## 0.0.17 — 2026-08-10

- **The Poser: a workspace for reusable poses.** Author a pose once against any
  of the seven skeleton templates — on a bare armature preview, built by the
  same Blender path as a real rig — and it lives in a global library rather
  than inside one asset. Every rigged asset on the same skeleton offers the
  library in its Pose panel; applying one copies it onto the asset, so later
  library edits never change what an asset already carries.
- **A pose can move its root.** Select the root joint in the Poser and tick
  Move root to offset the whole pose — a crouch that actually lowers, a leap
  that leaves the ground. The offset is stored in character heights and scaled
  onto each asset's own rig at bake time, in posed GLBs and sprite sheet rows
  alike. An animated clip cannot interpolate one yet and says so by name.
- The pose editor draws the skeleton's bone lines between the joint markers,
  so a bare armature reads as a skeleton rather than a cloud of dots.

## 0.0.16 — 2026-08-09

- **A style LoRA now declares the architecture it was fitted to**, so the picker
  offers each model the styles that fit it rather than being disabled wholesale
  off SDXL. Adds a FLUX.2 klein pixel-art LoRA and a distilled FLUX.2-klein-4B
  base for it to run on at the 4-step recipe it was trained against.
- **Fixed: one job generated without a style LoRA silently disabled the adapter
  for every later job in the process.** `disable_lora()` sets a flag that
  `set_adapters()` never clears, and the pipe stays resident across jobs, so the
  state outlived the job that set it. It read as working because the trigger
  words are still prepended — the output changed when a style was picked, it just
  was not the adapter doing it. SDXL was affected identically.
- **A mesh verdict is a grade, not a bit.** Review files −5..+5 instead of
  accept/reject, and the five rejection reasons became optional tags — legal at
  any grade, with five good ones added beside them — because a solid slab and a
  mesh a modeller would fix in five minutes were the same row. Image labels stay
  one keypress; they feed binary probes and a grade would be thresholded straight
  back to a bit.
- The re-baseline campaign ran. `bg_removal=birefnet` is the baseline —
  `auto` took 0 accepts in 90 units — and `hole_worst` reads the right way round
  again now the solid-slab failure mode is gone. Written up under
  `docs/measurements/`.
- **Home is a two-column screen.** What's new and the machine's status on one
  side, everything you were working on down the other. The tile grid is gone:
  seven of its nine tiles were the mode switch again, one click either way.
- **Library and Profiles are modes**, in the switch beside everything else,
  rather than sub-views of Home with nowhere else to live.
- **One Resume list across every document mode.** Inker, Clay, Plotter and
  Packwright each kept their own recent-files list; they are now one list,
  newest first, with your assets folded in and each row badged with the mode
  that opens it.
- Positional Alt+digit mode switching is gone. Twelve modes and ten digits was
  never going to fit; the palette (Ctrl+K) is the keyboard route.

## 0.0.14 — 2026-08-09

- **Plotter**: a tilemap editor, with Tiled `.tmx` import and export.
- **Packwright**: a deterministic sprite-atlas packer with a TexturePacker
  sidecar and a `.tsx` tileset for Plotter.
- Inker learned animation: a sparse track x frame grid, onion skinning,
  playback and a sprite-sheet export.
- A light palette, an accessibility reduce-motion switch, and three sidebar
  widths.

## 0.0.13 — 2026-08-08

- **Clay**: block a model out from primitives by hand and export it as an
  ordinary asset.
- Native kernels for the compositor, the contour tracer and the mesh audit,
  built on demand and always beside the numpy reference they are checked
  against.
- A download button for model weights, in a subprocess so the app itself stays
  offline.
- The manual moved into the app as its own mode.

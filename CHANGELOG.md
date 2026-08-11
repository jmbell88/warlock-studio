# Changelog

Hand-written, newest first. Nothing here is derived from git: the commit
subjects are `Warlock vN.N.N` and carry no detail, so this file is the only
record of what a version actually changed. The top heading's version must
match `pyproject.toml` — a test asserts it, so a release bump cannot leave this
file behind.

## 0.0.20 — 2026-08-11

- **Groundwork for Plotter's ground tile sets.** The engine for generated
  terrain tilesets — blob autotiling, both map projections, and the generator
  itself — but **no user interface yet**, so nothing in this release is
  reachable from the app. Four new pure modules: the 47-case blob collapse,
  cell-to-pixel placement for orthogonal *and* isometric maps, the terrain
  model, and a procedural generator that emits a flat-fill-plus-outline base
  set from a list of named terrains. A terrain is read back off the tile's own
  id rather than stored beside it, so there is nothing that can disagree with
  the picture; and a cell where three terrains meet resolves to one tile,
  because a terrain's list position is its precedence and a cell's neighbours
  count only from its own rank upward. The generator is deterministic to the
  byte — no randomness and no floating point in any coverage decision — so a
  polished set can be diffed against the base it was painted over.
- **A drag in Plotter is one undo step.** It used to be one *per cell*: a stamp
  pulled across forty cells pushed forty steps and forty entries against the
  history's byte budget. Painting now opens a stroke session, writes the live
  layer with no history while the button is down, and pushes a single patch
  over everything that moved when it is released — the same three calls the
  raster editor has always used, over tile ids instead of pixels.
- A tileset's art can be replaced in place, keeping its ids, its position and
  its declared terrains. A replacement with a different tile count is refused
  by name rather than silently renumbering every cell already painted.

## 0.0.19 — 2026-08-11

- **Indexed colour in Inker.** A document can now carry a palette, and every
  write snaps onto it — strokes, fills, shapes, gradients, filters and pastes
  alike. Index to the swatch row or to a `.gpl`, then edit a slot and every
  pixel painted in that colour is repainted across every layer and every frame
  as **one** undo step; delete a slot and its pixels merge into the nearest
  survivor; reorder freely, because the order is what an exported `.gpl` and an
  exported GIF colour table carry. **Count usage** reports how many pixels sit
  on each slot, so a slot showing zero is one you can safely drop. Alpha is
  never snapped, so a soft brush still fades — it just bands, which is what the
  mode is for. The pixels stay full-colour RGBA underneath: "indexed" means the
  writes are constrained, not that the file stores palette indices, so nothing
  about layers, blending or export changes shape and turning the mode off leaves
  every pixel exactly where it is. The table is saved inside the `.ora` as a
  `palette.gpl` member, and an editor that does not know about it opens the file
  as the ordinary image it already is.
- **An indexed GIF exports your table verbatim.** Slot *n* is the same colour in
  every frame of the clip, instead of each frame being quantised on its own.
- **Clay documents can be closed.** Clay could open documents and never shut
  one: there was no tab bar, so Ctrl+Tab switched between documents with nothing
  on screen saying there was more than one, and a dirty-quit prompt asked about
  documents you had no way to reach. Clay now has the same tab bar as every
  other workspace, with Ctrl+W and a close button, and a dirty document asks
  before it goes. Plotter and Packwright mark a dirty tab with imgui's own dot
  now rather than a `"* "` prefix, matching Inker.
- **Space-to-pan in Plotter no longer latches.** The key-up was filtered out
  before it was ever seen, so the first press left panning on for the rest of
  the session and every left-drag panned instead of drawing.
- **Global shortcuts work while a text field has focus.** Ctrl+K and the F-keys
  were dead the moment you clicked into the 2D prompt box — which is exactly
  where you are when you want to jump somewhere else. Plain keys still belong to
  the field, and so do imgui's own Ctrl+Z/Y/X/C/V/A inside it.
- **Two toasts asked for a level that does not exist** ("warning" rather than
  `warn`) and silently arrived as grey, non-sticky info notices. A test now
  pins every toast level in the source against the ones that exist.
- **Deleting a saved pose or a rendered sheet asks first.** Both sat one pixel
  from a save button and deleted on the click; a sheet costs one Blender render
  per cell to recreate. **Reset all** and the joint **Revert** in the pose
  editor now take the same unsaved-changes guard the preset path already did.
- **Greyed controls can say why.** `disabled_button` promised a tooltip in its
  docstring and drew none across 92 call sites; it takes a `reason` now, and the
  command palette's greyed rows explain themselves instead of swallowing Enter
  in silence.
- **The Review loop says what it did.** An armed negative grade is now visible
  on screen, filing a verdict says what was filed and how to get back to it
  before it advances, the grade and tag buttons carry their keys, and the
  see-through reading carries the same caveat the inspector gives it — a solid,
  featureless mesh scores it too.

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

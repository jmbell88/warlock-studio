# Overview

## What Warlock Studio is

Warlock Studio generates game-ready 3D assets on your own machine. You give it a text prompt or an
image; it gives you back a textured GLB — a base colour texture plus a combined
metallic/roughness texture, with surface detail carried on vertex normals rather than a normal map
— ready to import into Godot, Blender, Unity or Unreal.

Two models do the work. An image model (SDXL 1.0 by default) draws the reference picture from
your prompt. A reconstruction engine, Microsoft TRELLIS.2-4B running natively through
`trellis-server.exe`, turns that picture into a mesh. Both run on your GPU.

The app is **fully offline**. Model weights are downloaded once, by hand, before you start; after
that Warlock Studio never touches the network. There is no provider API, no account, no upload of
your prompts or your images. If a set of weights is missing, the app tells you the exact command
to fetch it rather than fetching anything itself.

It is also a single desktop window. There is no server to start, no browser tab, no `localhost`
address. Everything described in this manual happens in one process.

## The two-stage pipeline

Making a mesh is expensive — roughly two minutes of GPU per attempt — and the single biggest
factor in how good that mesh is turns out to be the picture it was made from. TRELLIS can only be
as good as the image it is handed.

So the pipeline is deliberately split in two, and the split is visible in the app:

1. **The reference stage.** A text job draws an image and stops. This takes a few seconds. The
   image is shown to you full size, and you can generate several candidates at once from different
   seeds before choosing one.
2. **The mesh stage.** Once you approve a reference, you promote it, and only then does the
   reconstruction run.

A text job never falls through to a mesh by accident: the Reference stage always submits with the
output set to `reference`. Going straight from a prompt to a mesh would spend two minutes of GPU on an
image nobody has looked at.

If you already have a picture, you can skip the first stage entirely and upload it — see
[Starting from an upload](23-generating-meshes.md#starting-from-an-upload).

## The modes

A rail down the left edge of the window chooses between thirteen modes, and that rail is the single
thing that decides what the panes show. It is drawn in every mode, so there is no screen you cannot
leave. There is no per-mode keyboard shortcut — the command palette (`Ctrl+K`) is the keyboard
route, see [Keyboard shortcuts](38-shortcuts.md).

The rail shows glyphs by default and expands to show the labels beside them; **Window → Navigation
labels** toggles that, and the choice is remembered. In icon-only form every item names itself in a tooltip.
A window too narrow to hold the labelled rail *and* three usable columns draws the collapsed one
until there is room again — what you chose and what fits are two different facts, so dragging the
window wider brings the labels back.

It is drawn in three sections, and the list below is in that order. The first is where an asset
**begins** — what you have, and making another one. The second is the **creative workspaces**. The
third is the **footer**, the last group in the column, carrying no caption: the two destinations
where you are not making something.

- **Home.** What the app opens on: what changed in this build, what the machine is doing, and a
  single list of everything you were recently working on. Returning here is never destructive.
- **Library.** Every asset that has ever been generated, filtered, sorted and searched, with the
  trash and the prune. Covered in [The library and jobs](36-library-and-jobs.md).
- **Create.** One mode for the whole asset pipeline, drawn as five **stages** on a rail above the
  settings column. **Reference** owns the prompt and every control that composes it — the
  negative prompt, the image model and style LoRA, the seed and the candidate count.
  **Mesh** owns no prompt controls at all: a mesh job starts from a finished reference or from an
  uploaded image, and the column holds only the reconstruction decisions. **Rig** fits a skeleton,
  **Pose** edits one, and **Export** is what you can take away. A stage you cannot enter yet is
  drawn dimmed with the reason on hover rather than hidden. Covered in
  [Generating references](22-generating-references.md),
  [Generating meshes](23-generating-meshes.md) and
  [Rigging and posing](25-rigging-and-posing.md).

Then the eight workspaces:

- **Inker.** A layered raster editor, wired into the pipeline in both directions. Covered in
  [Inker](28-inker.md), with the timeline in [Inker: animation](29-inker-animation.md).
- **Clay.** Modelling from primitives: transforms, a material palette, and two ways out —
  export a `.glb` or import the document as an asset. Covered in [Clay](30-clay.md).
- **Poser.** Authoring reusable poses against a skeleton template, kept in a global pose library
  rather than belonging to any one asset. Covered in [Poser](26-poser.md).
- **Troupe.** A character-sprite factory: a prompt becomes a reference, a mesh, a fitted rig and
  then a rendered, pixelised sprite sheet of the clips a character walks and swings through.
  Experimental — the chain runs end to end, but the shipped keyframes are provisional and the
  prompt-to-character half does not currently produce usable humanoids (measured 2026-08-30), so
  the route worth using is a mesh you supply. Covered in [Troupe](33-troupe.md).
- **Plotter.** A tile-map editor: a grid, a layer stack, one or more tilesets, and the objects an
  engine reads as spawn points and trigger volumes — where a sheet of tiles becomes a level. It
  speaks Tiled's formats in both directions. Covered in [Plotter](31-plotter.md).
- **Packwright.** A sprite-atlas packer: many images in, one atlas out, with a sidecar that says
  where everything landed. Covered in [Packwright](32-packwright.md).
- **Muse.** Generated music: a comma-separated style-tag string and an optional lyric block become a
  finished track, one job row per take, auditioned in the mode and openable in Sirens as a sample
  instrument. Covered in [Muse](35-muse.md).
- **Sirens.** A chiptune tracker: NES-era pulse, triangle, noise and sample voices written into a
  pattern grid, stitched into a song by an order list, and saved as a `.wsng`. Instruments carry
  four envelope sequences you drag into shape, a `.wav` dropped on the window becomes a sample, and
  the whole thing exports as a mix, one WAV per channel and one per sound effect. Covered in
  [Sirens](34-sirens.md).

And in the footer:

- **Review.** Judging finished meshes — one at a time or as a parameter sweep — and the "what
  works" findings the verdicts add up to. Covered in [Review](37-review.md).
- **Settings.** The app's own preferences — UI scale, the frame-rate readout, layout resets, and the
  list of models it loaded, from which a missing one can be downloaded. See
  [In-app settings](40-configuration.md#in-app-settings).

This documentation used to be a mode and is not, for the reason nothing here is: it is *about* the
screen you are on rather than a place to go. It opens over the window (`F1`, or any pane's (?)
button) instead of replacing it, so the control you were asking about is still there when you have
the answer.

The **guided tour** is a second overlay, for the same reason: it points at the controls of whatever
mode you are in, so taking that mode away to run it would leave nothing to point at. It never
clicks anything for you. Home offers it on a fresh install and the palette carries it thereafter —
see [New here?](21-home.md#new-here).

Each generation control belongs to exactly one stage. The one setting both Reference and Mesh need
is **platform**, and it is deliberately two separate controls: at the Reference stage it is a hint
that goes into the prompt ("how much fine detail should be drawn"), and at the Mesh stage it is the
geometry resolution sent to the reconstruction engine. One control cannot be owned by two stages,
so there are two.

Switching modes is never destructive. Inker keeps its open documents when you leave it, a queued
job keeps running whichever mode you are in, and the progress card floats over every mode but Home.

## The window

The app opens on Home, every launch: no mode is remembered between runs, because none of them is
what you want to be dropped into before you have said what you are doing. **Home**
is the first entry in the rail described above, and returns there at any time.

Once you are in the workspace, the window is three columns:

- **The left sidebar** is the settings form for the current mode, and nothing else — there is
  nothing left to split against, so it is one scrolling column with no divider. In Create a **stage
  rail** sits above it, naming the five steps an asset goes through and switching the column
  between them. Its width is not draggable; it is one of three named sizes chosen in Settings.
- **The middle column** is the viewport: the interactive 3D preview, or the reference image at the
  Reference stage, or the canvas in Inker mode. A small toolbar sits over it with the framing,
  wireframe and turntable toggles, and — on a finished reference at the Reference stage — the
  **Open in Inker** button.
- **The right column** is two stacked panels, not one scrolling column. The upper panel is the
  inspector: everything about the selected asset. In Create it carries no tabs — the stage rail is
  what switches it, so it shows the evidence for the stage you are on. Everywhere else it is three
  tabs, **Details**, **Rig & Pose** and **Export**. The lower panel is the asset library — every
  job you have ever run, with its filters. The divider between them can be dragged; the sidebar's
  own width is not draggable, only chosen from the three named sizes in Settings.

Above the columns is the menu bar and below them is the status bar, and both are described next.

## The menu bar

One menu bar across the top of the window, drawn in every mode. Its roots are **File**, **Edit**,
**View**, **Workspace**, **Window** and **Help**, and between Edit and View sits whatever the
current workspace contributes. For most of them that is a single menu under the mode's own name —
*Clay*, *Plotter*, *Troupe* — holding the actions that belong to that mode alone. Inker, which has
far more of them, contributes several: **Sprite**, **Layer**, **Frame** and **Select**, and it adds
rows to File, Edit and View as well. Either way a mode's actions get their own place rather than
being filed into File or Edit, which would turn the two menus everybody already understands into a
list of everything.

**Nothing in the menu is a second implementation of anything.** Every row is an adapter over the
same command registry the palette searches and the same operation registry the keys dispatch
through, so the menu, `Ctrl+K` and the keyboard cannot disagree about what an action does, whether
it is available, or why it is not. A row you cannot use is greyed with the reason on hover — the
same reason the palette gives — and a row with a keyboard binding prints it on the right.

**Workspace** is the one to know about: it holds all thirteen modes, so it is a third way — beside
the rail and the palette — to change what the window is showing.

## The status bar

One line along the foot of the window, also in every mode. Left to right: the workspace you are in,
then the open document and whether it has unsaved changes — plus the current tool and zoom in Inker
— then the queue when anything is running or waiting, then an amber **N issue(s)** when a startup
check has failed. Clicking that last one opens the Issues list; it is **Issues** in the command
palette too, which is how you reach it when nothing is failing and there is no count to click.
There is no green "all well" state, because a healthy install has nothing to report.

When the window is too narrow to hold all of it, items drop from the *right* end, so the answer to
"where am I" is the last thing to go. The one item anchored to the right instead is the optional
system-resource meter — see [App settings](41-app-settings.md#appearance) — which is
reserved before the rest is trimmed, because it is read while a generation is being decided on.

The keyboard shortcut list is `Ctrl+/`, **Help → Keyboard shortcuts**, or **Keyboard shortcuts** in
the command palette, and it is reproduced in [Keyboard shortcuts](38-shortcuts.md).

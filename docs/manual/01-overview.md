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
[Starting from an upload](04-generating-meshes.md#starting-from-an-upload).

## The modes

A rail down the left edge of the window chooses between ten modes, and that rail is the single
thing that decides what the panes show. It is drawn in every mode, so there is no screen you cannot
leave. There is no per-mode keyboard shortcut — the command palette (`Ctrl+K`) is the keyboard
route, see [Keyboard shortcuts](16-shortcuts.md).

The rail shows glyphs by default and expands to show the labels beside them; the chevron at its foot
toggles that, and the choice is remembered. In icon-only form every item names itself in a tooltip.
A window too narrow to hold the labelled rail *and* three usable columns draws the collapsed one
until there is room again — what you chose and what fits are two different facts, so dragging the
window wider brings the labels back.

It is drawn in three sections, and the list below is in that order. The first is the **asset
pipeline** — start something, take it through its stages, find it again, judge it. The second is the
**creative workspaces**. The third is the **footer**, drawn against the bottom edge: a badge that
appears only when something is failing its startup check, then the manual, Settings, and the
collapse toggle.

- **Home.** What the app opens on: what changed in this build, what the machine is doing, and a
  single list of everything you were recently working on. Returning here is never destructive.
- **Create.** One mode for the whole asset pipeline, drawn as five **stages** on a rail above the
  settings column. **Reference** owns the prompt and every control that composes it — the
  negative prompt, the image model and style LoRA, the seed and the candidate count.
  **Mesh** owns no prompt controls at all: a mesh job starts from a finished reference or from an
  uploaded image, and the column holds only the reconstruction decisions. **Rig** fits a skeleton,
  **Pose** edits one, and **Export** is what you can take away. A stage you cannot enter yet is
  drawn dimmed with the reason on hover rather than hidden. Covered in
  [Generating references](03-generating-references.md),
  [Generating meshes](04-generating-meshes.md) and
  [Rigging and posing](05-rigging-and-posing.md).
- **Library.** Every asset that has ever been generated, filtered, sorted and searched, with the
  trash and the prune. Covered in [The library and jobs](13-library-and-jobs.md).
- **Review.** Judging finished meshes — one at a time or as a parameter sweep — and the "what
  works" findings the verdicts add up to. Covered in [Review](15-review.md).

Then the five workspaces:

- **Inker.** A layered raster editor, wired into the pipeline in both directions. Covered in
  [Inker](08-inker.md), with the timeline in [Inker: animation](09-inker-animation.md).
- **Clay.** Modelling from primitives: transforms, a material palette, and two ways out —
  export a `.glb` or import the document as an asset. Covered in [Clay](10-clay.md).
- **Poser.** Authoring reusable poses against a skeleton template, kept in a global pose library
  rather than belonging to any one asset. Covered in [Poser](06-poser.md).
- **Plotter.** A tile-map editor: a grid, a layer stack, one or more tilesets, and the objects an
  engine reads as spawn points and trigger volumes — where a sheet of tiles becomes a level. It
  speaks Tiled's formats in both directions. Covered in [Plotter](11-plotter.md).
- **Packwright.** A sprite-atlas packer: many images in, one atlas out, with a sidecar that says
  where everything landed. Covered in [Packwright](12-packwright.md).

And in the footer:

- **Settings.** The app's own preferences — UI scale, the frame-rate readout, layout resets, and the
  list of models it loaded, from which a missing one can be downloaded. See
  [In-app settings](18-configuration.md#in-app-settings).

Two things that used to be modes are not, and both moved for the same reason: they are *about* the
screen you are on rather than places to go. This documentation opens over the window (`F1`, or any
pane's (?) button) instead of replacing it, so the control you were asking about is still there when
you have the answer; and the style-profile manager opens as a sheet from the profile picker at the
Reference stage — see [Profiles](14-profiles.md).

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

There is no top bar. The keyboard shortcut list is `Ctrl+/` or **Keyboard shortcuts** in the command
palette, and it is reproduced in [Keyboard shortcuts](16-shortcuts.md). The **health badge** sits in
the rail's footer and appears only when a startup check is failing — amber for a non-fatal one
(missing optional weights, no gltfpack, no CUDA), red when something fatal failed or the worker
died. Hovering it names the failing checks; clicking it opens the full Issues list, which is also
**Issues** in the palette when everything is passing and there is no badge to click.

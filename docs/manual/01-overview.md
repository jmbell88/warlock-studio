# Overview

## What Warlock Studio is

Warlock Studio generates game-ready 3D assets on your own machine. You give it a text prompt or an
image; it gives you back a textured GLB — a base colour texture plus a combined
metallic/roughness texture, with surface detail carried on vertex normals rather than a normal map
— ready to import into Godot, Blender, Unity or Unreal.

Two models do the work. An image model (SDXL-Turbo by default) draws the reference picture from
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

A text job never falls through to a mesh by accident: the 2D pane always submits with the output
set to `reference`. Going straight from a prompt to a mesh would spend two minutes of GPU on an
image nobody has looked at.

If you already have a picture, you can skip the first stage entirely and upload it — see
[Starting from an upload](03-generating-meshes.md#starting-from-an-upload).

## The modes

A switch at the top of the window chooses between seven modes, and that switch is the single thing
that decides what the panes show. It is drawn in every mode, so there is no screen you cannot leave.

- **Home.** The chooser the app opens on: start a 2D reference, start a 3D asset, open something
  already made, or manage profiles. Returning here is never destructive.
- **Manual.** This documentation, embedded in the window rather than floating over it. `F1` and
  every pane's (?) button come here.

- **2D reference.** Owns the prompt and every control that composes it: the guidance selects, the
  negative prompt, the image model and style LoRA, the seed and the candidate count. Covered in
  [Generating references](02-generating-references.md).
- **3D asset.** Owns no prompt controls at all. A 3D job starts either from a finished 2D asset or
  from an uploaded image, and this pane holds only the mesh, rig, pose and sprite-sheet decisions.
  Covered in [Generating meshes](03-generating-meshes.md).
- **Inker.** A layered raster editor, wired into the pipeline in both directions. Covered in
  [Inker](06-inker.md).
- **Clay.** Modelling from primitives: transforms, a material palette, and two ways out —
  export a `.glb` or import the document as an asset. Covered in [Clay](07-clay.md).
- **Settings.** The app's own preferences — UI scale, the frame-rate readout, layout resets, and a
  read-only list of the models it loaded. See
  [In-app settings](11-configuration.md#in-app-settings).

Each generation control belongs to exactly one mode. The one setting both the 2D and the 3D pane need is
**platform**, and it is deliberately two separate controls: in the 2D pane it is a hint that goes
into the prompt ("how much fine detail should be drawn"), and in the 3D pane it is the geometry
resolution sent to the reconstruction engine. One control cannot be owned by two panes, so there
are two.

Switching modes is never destructive. Inker keeps its open documents when you leave it, a queued
job keeps running whichever mode you are in, and the progress card floats over every mode but Home.

## The window

The app opens on Home, every launch: no mode is remembered between runs, because none of them is
what you want to be dropped into before you have said what you are doing. **Home**
is the first entry in the mode switch described above, and returns there at any time.

Once you are in the workspace, the window is three columns:

- **The left sidebar** is two stacked panels, not one scrolling column. The upper panel is the
  settings form for the current mode; the lower panel is the asset library — every job you have
  ever run, with its filters. The divider between them can be dragged, as can the divider between
  the sidebar and the middle column.
- **The middle column** is the viewport: the interactive 3D preview, or the reference image in 2D
  mode, or the canvas in Inker mode. A small toolbar sits over it with the framing, wireframe and
  turntable toggles, and — on a finished reference in 2D mode — the **Open in Inker** button.
- **The right column** is the inspector: everything about the selected asset. In 3D mode it is
  three tabs, **Details**, **Rig & Pose** and **Export**; in 2D mode, **Details** and **Export**.

At the far right of the top bar are two small controls. The **?** button opens the keyboard
shortcut list, which is also reproduced in [Keyboard shortcuts](09-shortcuts.md). Beside it is the
**health dot**: green when every startup check passed, amber when a non-fatal check failed (missing
optional weights, no gltfpack, no CUDA), and red when something fatal failed or the worker died.
Clicking it opens the full diagnostics list.

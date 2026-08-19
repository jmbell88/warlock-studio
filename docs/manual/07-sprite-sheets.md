# Sprite sheets

Any finished mesh can be baked into a 2D sprite sheet: a grid of rendered views, plus an
engine-neutral JSON sidecar describing what each cell contains. The controls are in the inspector
at the **Pose** stage, under the collapsed **Sprite sheet** header — a sheet is made of poses, so it
sits beside the editor that makes them.

A 2D reference that never became a mesh has its own way in, described under
[From a single drawing](#from-a-single-drawing) — the same kind of sheet, generated rather than
rendered.

## The grid

The grid is **poses down, compass directions across**.

Each row is one pose. Ticking poses in the **Rows** list adds rows; ticking none gives you a single
row of the mesh's rest pose. Each column is one yaw — one direction the subject is seen from.
**Directions** offers 4, 8 or 16, defaulting to 8. The choice is not free-form on purpose: a count
that does not divide 360 into the facings an engine indexes by produces a sheet nothing can address.

**Column 0 is the front view.** Yaw increases from there in even steps, so an eight-direction sheet
runs 0, 45, 90, 135, 180, 225, 270 and 315 degrees across.

The other controls:

- **Frame** is the pixel size of one square cell — 64, 128 or 256.
- **Lighting** is `flat` or `lit`. Flat is the usual choice for sprites.
- **Elevation** tilts the camera, from -60 to +60 degrees. Zero looks at the subject level-on.
- **Name** is an optional label for the finished sheet, so the saved list shows something more
  useful than an identifier.

Ticking **Animated clip** replaces the pose rows with an interpolated sequence: pick a **From** pose,
a **To** pose and a number of **Frames** (2 to 32), and the rows become the animation. A clip
replaces the static rows rather than adding to them, because its rows *are* the animation and mixing
static poses in would leave an importer no way to tell which rows loop.

The line under the controls states the grid you will actually get — how many render cells, and the
output image's pixel dimensions.

A sheet is a queued job, started with the **Render sheet** button and rendered in Blender's EEVEE
with a transparent film. A rigged mesh is
required for posed rows and for clips; a plain mesh can still be rendered, as described under
[Unrigged props](#unrigged-props).

## Previewing

**Refresh preview** renders the chosen directions live, in the app's own viewport renderer, and
shows them as a horizontal strip above the controls. It is there so the framing, the camera
elevation and the flat-versus-lit choice can be judged before committing a job to the queue.

It is a *direction* preview, not a sheet preview: the app's viewport cannot pose the mesh, so
drawing one row per pose would draw the same row several times. That is why the grid itself is
stated as a summary line instead.

The preview and the final render agree on framing by construction. Both put yaw 0 in the same place,
and both size the camera to the subject's worst-case silhouette with the same 12% margin.

One detail matters more than it looks: the camera is framed **once**, from the mesh's rest bounding
box, and every cell in the sheet uses that same framing. Reframing per pose would make the subject
jump in size between rows, which is exactly what a sprite sheet must not do.

## The sidecar

Each finished sheet is a PNG plus a JSON sidecar. The sidecar is the sheet's contract with your
engine — a bare PNG is a grid nothing can address — and it is deliberately engine-neutral, with no
Godot `AtlasTexture` or Unity `SpriteMetaData` opinions baked in.

```json
{"version": 1, "columns": 8, "rows": 2, "frame_size": 128,
 "yaws": [0, 45, 90, 135, 180, 225, 270, 315],
 "cells": [{"index": 0, "x": 0, "y": 0, "w": 128, "h": 128,
            "pose": "751cf6147291", "pose_name": "idle", "yaw": 0, "frame": 0}]}
```

The header states the grid; `cells` describes every cell in it. Each cell carries its pixel
rectangle and what that rectangle shows: which pose, that pose's name, which yaw, and which frame.

`cells` is a **flat list**, not a nested grid, and that is the format's one piece of foresight. An
animated clip is not a different file format — it is simply more cells whose `frame` is above zero.
A reader that walks the flat list handles both without knowing which it has.

Rendered sheets are listed under **Rendered sheets** with their cell count and frame size. Each
offers **Save PNG...**, **Save JSON...** and **Delete**. Save both: the PNG without its sidecar is
just a picture. **Edit in Inker** opens the sheet as an animation — one frame per cell, cut on the
grid the sidecar records — as an *unlinked* document, so the first Ctrl+S asks where to put it and
the render on disk is never overwritten. **Add to Packwright** sends it to the atlas packer with
the cell size already filled in, and the pixel restyle below offers the same two.

## Pixel art from the sheet

Each rendered sheet carries a **Pixelate** disclosure, and it is the one thing here that generates
rather than renders: it restyles the finished atlas into pixel art and writes a second image and
sidecar beside the render's, leaving the render itself untouched.

What makes it work is that the eight directions are not eight generations that have to be talked
into agreeing. They are exact orthographic renders of one mesh, so the geometry already agrees
perfectly and the only thing being generated is how it looks. Three consequences follow, and each is
a property rather than a hope:

- **One denoise per band.** Eight directions at 128 px is exactly 1024 pixels wide, which is one
  frame — so every direction is drawn in a single latent under a single seed, and there is one
  identity rather than eight. Whole rows are grouped; a row is never split across two.
- **Exact silhouettes.** The generated colours are given the render's own alpha, verbatim. Whatever
  the model invented outside the subject is background it was never asked for, so the silhouette
  cannot drift.
- **One palette.** The reduction and the colour quantization run once, across the whole atlas —
  never per cell, which is how the same shirt comes out two shades in two directions.

The controls are **Pixel size** (only the sizes that divide this sheet's cells exactly are offered;
anything else would resample across cell boundaries), **Colours**, **Strength** — how far the
denoise is taken, 0.30 to 0.65, with lower keeping more of the render — **Lock silhouettes**, which
adds an edge hint from the flat render, and a seed with a **Reroll**. The result is saved with
**Save pixel PNG...** and **Save pixel JSON...**; the pixel sidecar is the render's, with every
rectangle divided by the reduction, plus the palette it chose and the settings that produced it.

A restyle is a queued job, not an instant export: it needs the image model, so it waits its turn
behind whatever else is generating. Cancelling one leaves the render completely intact — the restyle
deletes only its own pair. Deleting a sheet deletes its restyle too, since a pixel sheet of a render
that is gone depicts nothing.

A sheet wider than 1024 pixels cannot be restyled, and the panel says so with the frame size to
re-render at rather than offering a button that fails.

## From a single drawing

Everything above starts from a mesh. A finished **2D reference** has a second way in: the
inspector's **Sprite sheet** header offers to invent a sheet from the one drawing you have. It is a
different bargain and worth understanding before you press it. A rendered sheet is exact, because
the eight views are eight photographs of one object. Here there is no object — only a picture of
one — so the front view is the drawing and the other three are the image model's guess at what the
subject looks like from the side and the back.

That is why it produces **two candidates every time**, from two different seeds, side by side. You
pick. Nothing is chosen for you, nothing is overwritten, and drafts accumulate until you delete
them.

**Type** is `turnaround` (a 2x2 grid: front, left, right, back) or `walk` (a 4x4 grid: one row per
direction, four frames of a walk cycle across). **Cell size** is the finished pixel size of one
cell, 32, 48 or 64. **Palette** is how many colours the whole sheet is reduced to. Each candidate
has its own seed with a **Reroll** beside it; the two must differ, or you would be asking for the
same picture twice.

Three things make the result hang together rather than being four unrelated drawings:

- **One generation per candidate.** The whole atlas is drawn in a single 1024-pixel pass under a
  single seed, so there is one character wearing one shirt, not four.
- **Pose guides.** Each cell is drawn over a stick figure fed to the edge ControlNet, so where the
  limbs go is imposed rather than requested — and the walk rows get a real contact-passing cycle
  rather than four poses that happen to differ.
- **One palette and one baseline.** The colour reduction runs once across the whole atlas, and
  every cell's subject is moved onto a shared floor line. Feet that move between frames is what
  reads as a broken animation, far more than a slightly wrong arm does.

For a turnaround, your own drawing is pasted back into the front cell when its proportions match
what the model drew — so the one view that is definitely right is definitely right. The panel says
whether it was, and why not when it was not.

Under each candidate is a line per cell that came out doubtful: empty, running off the edge of its
cell, or far off the size of the rest of the sheet. These are notes, never refusals — a warning
costs you a sentence to read, and throwing a candidate away would leave you comparing one draft
against nothing.

**Edit in Inker** opens that candidate as an animation, one frame per cell, and this is the point of
the whole feature: what arrives is editable, not final. A walk sheet arrives with a tag per
direction, so pressing Play loops one direction at a time. The document is unsaved and belongs to no
file — the first `Ctrl+S` asks where to put it, and the draft on disk is left alone. **Export
sheet** from there writes the atlas back out on the sheet's own fixed grid rather than wrapping it,
so a walk cycle stays four rows of four and each cell's sidecar entry carries the direction and the
frame. See [Inker: animation](09-inker-animation.md).

A synthesis is a queued job that runs two full image generations, so it waits its turn behind
whatever else is generating. **Delete draft** removes a pair; a reference keeps at most 50.

## Unrigged props

A sheet does not require a rig. A crate, a rock or a sword has no poses and needs none — render it
with no rows ticked and you get a **turnaround of its rest pose**: one row, one cell per direction,
which is exactly what a prop needs for a 2D game.

Only posed rows and animated clips need a rigged mesh, and the panel says so before the button if
you ask for one without a rig. Sheets themselves still need Blender, so they live behind the same
optional extra as rigging — see
[When rigging is unavailable](05-rigging-and-posing.md#when-rigging-is-unavailable).

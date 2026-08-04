# Sprite sheets

Any finished mesh can be baked into a 2D sprite sheet: a grid of rendered views, plus an
engine-neutral JSON sidecar describing what each cell contains. The controls are in the inspector's
**Rig & Pose** tab, under the collapsed **Sprite sheet** header.

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

A sheet is a queued job, rendered in Blender's EEVEE with a transparent film. A rigged mesh is
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
just a picture.

## Unrigged props

A sheet does not require a rig. A crate, a rock or a sword has no poses and needs none — render it
with no rows ticked and you get a **turnaround of its rest pose**: one row, one cell per direction,
which is exactly what a prop needs for a 2D game.

Only posed rows and animated clips need a rigged mesh, and the panel says so before the button if
you ask for one without a rig. Sheets themselves still need Blender, so they live behind the same
optional extra as rigging — see
[When rigging is unavailable](04-rigging-and-posing.md#when-rigging-is-unavailable).

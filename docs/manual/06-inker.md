# Inker

Inker is the top-level drawing mode: a layered raster editor, built into the app rather than bolted
onto it. It exists because the reference image is the biggest lever on mesh quality, and the fastest
way to fix a nearly-right reference is usually to paint over it.

It is a mode, not a takeover. Switching away leaves every open document exactly where it was, and a
reconstruction job started before you switched keeps running with its progress card floating over
the canvas. Only quitting the app and closing a tab can lose pixels, and both ask first.

The layout follows the rest of the app: tools and their options on the left, the canvas in the
middle, layers and the pipeline panel on the right. Several documents stay open at once, as tabs.

## Tools

The toolbox is an icon grid; hovering a tool shows its name and its letter. Every tool is listed in
[Keyboard shortcuts](09-shortcuts.md).

| Tool | Key | What it does |
| --- | --- | --- |
| Brush | `B` | A soft round brush. |
| Eraser | `E` | The same brush, cutting alpha instead of adding colour. |
| Fill | `G` | Flood-fills from where you click. |
| Gradient | `U` | Drag to lay a gradient. |
| Blur | `R` | Softens what you drag over. |
| Smudge | `N` | Pushes pixels along the drag. |
| Line | `P` | A straight line. |
| Rect | `K` | A rectangle, outlined or filled. |
| Ellipse | `J` | An ellipse, outlined or filled. |
| Marquee | `M` | Rectangular selection. |
| Ellipse select | `S` | Elliptical selection. |
| Lasso | `Q` | Freehand selection. |
| Wand | `W` | Selects a region of similar colour. |
| Move | `V` | Moves the floating selection. |
| Pick | `I` | Samples a colour from the canvas. |

Options appear for the selected tool only, rather than as one long form — a brush's hardness means
nothing while the wand is active.

The painting tools share **Size**, **Hardness**, **Opacity** and **Spacing**; blur and smudge add
**Strength**. The shape tools share the size slider and, except for the line, a **Filled**
checkbox. Fill and the wand share **Tolerance** (0 to 255) and **Contiguous** — turning contiguous
off acts on every similar pixel in the image, not just the ones touching where you clicked. The
gradient tool chooses its **Shape** and whether it fades **To transparent**.

Two canvas-wide aids sit below the tool options. **Symmetry** mirrors every stroke — off,
left/right, top/bottom, or both. **Grid** overlays a grid at a spacing you set, from 2 to 512 pixels.

## Layers

The layers panel shows the stack top-first, the way every editor shows it.

Each layer has a visibility checkbox, a thumbnail, a name, an **Opacity** slider and a **Blend**
mode: `normal`, `multiply`, `screen`, `overlay` or `add`. Above the list are **Add**, **Copy**,
**Delete**, **Merge down** and **Flatten**. Dragging the opacity slider previews live but records a
single undo step when you let go, rather than one step per pixel of drag.

One rule about erasing is worth stating plainly, because it differs from some editors. **The eraser
makes pixels transparent.** It does not paint the background colour. The background colour — the
matte — is a property of the document, applied once when the image is flattened for export. So what
you erase through is genuinely a hole until the moment you export, and changing the document's
matte changes what every erased area exports onto without touching a single stroke.

Undo is unlimited within a memory budget and is addressed by layer identity rather than by position
in the stack — an undo issued after you reorder layers still lands on the layer the edit was
actually made to. The document is "dirty" when it differs from the last save, which means undoing
back to the saved state marks it clean again rather than leaving it unsaved forever.

The pipeline panel on the right also shows **Undo** and **Redo** buttons and the current history
depth, alongside canvas operations: **Flip H**, **Flip V**, **Rotate**, **Fit view**, and
**Resize...**. The resize popup deliberately offers two different operations — **Scale image**
resamples the picture, **Resize canvas** changes how much room it has around the picture.

## Selections and transform

Four tools make selections: the rectangular marquee, the ellipse, the lasso and the wand. Hold
**Shift** while dragging to add to the current selection and **Alt** to subtract from it.

With a selection live, the **selection** section offers **All**, **None**, **Invert**, a
**Feather** radius slider up to 32 pixels with a **Feather** button, and **Crop to selection**. The
same actions have keyboard shortcuts: `Ctrl+A`, `Ctrl+D` and `Ctrl+Shift+I`.

Cutting or copying a selection puts it on the clipboard; pasting brings it back as a **floating
buffer** which you move with the Move tool and commit by doing anything else. `Esc` cancels a
floating buffer, and `Delete` clears the selected pixels.

Cancelling a lift — where the buffer was cut out of a layer — puts the pixels back and removes that
step from history entirely, rather than leaving it on the redo stack where `Ctrl+Y` could replay the
cut with no buffer left to restore.

**Free transform** (`Ctrl+T`, or the button in the tool options) rotates and scales the selection,
or the whole layer when there is no selection. It is modal: while transforming, **Enter** applies
and **Esc** cancels, and nothing else can change the tool out from under a half-finished transform.

## Saving

Inker saves natively as **OpenRaster** (`.ora`) — a zip of layer PNGs that both Krita and GIMP read
and write. That is the format that keeps your layers, their blend modes and their opacities.

- `Ctrl+S` saves. A document that has never been written anywhere asks where to put it first.
- `Ctrl+Shift+S` is Save As.
- `Ctrl+Shift+E` exports a **flattened PNG**. This is an export, not a save: it does not change what
  the tab points at, so the document stays unsaved against its own file.

Saving is a background operation, and it shows: while a save is in flight the layer panel and the
structural shortcuts are disabled, because a save is encoding the layer stack on another thread and
restructuring it underneath would corrupt the file. Brush strokes are still allowed, since they
write pixels in place. If a save fails, the tab is released again and a toast says so.

Closing a tab or quitting with unsaved changes asks first. Every dialog in Inker runs off the frame
thread, so the window never freezes behind one.

## Pipeline bridges

Inker is wired into the pipeline in both directions. The **document** panel on the right states
which direction you are in: a document is either **linked to a job** or **not part of a job**.

**Into Inker.** With a finished reference selected in 2D mode, **Open in Inker** appears on the
viewport toolbar. It opens that reference as a linked document. If a layered working file already
exists for the job and is current, you get your layers back; otherwise you get the flat image.

Saving a linked document (`Ctrl+S`) writes the reference back in place. Two files are written: the
flat `input.png`, through exactly the same path the app has always used — so the reference is
re-measured and marked as hand-edited — and the layered source beside it as `paint.ora`.

`paint.ora` is internal working state. It is never served, never exported, and never counted among a
job's files. It is treated as **stale whenever it is older than `input.png`**, which is the rule
that keeps it honest: reverting or regenerating a reference rewrites the flat image without touching
the layers, and resurrecting layers that describe pixels which no longer exist would be worse than
losing them.

**Revert to original** puts the generated image back. The untouched original is kept once, as
`input.orig.png`, the first time you edit a reference — so revert always works — and reverting also
discards the layered file, because it describes an edit that no longer exists.

**Out of Inker.** Two buttons in the **pipeline** section:

- **Save as reference** (on an unlinked document) adds what you painted to the library as a
  finished reference. It is measured on the way in, so the quality gate has real data, and it can
  then be meshed, promoted and rerun exactly like a generated one. The document becomes linked to
  the new job immediately, so the next `Ctrl+S` saves in place rather than minting a second job from
  the same pixels.
- **Send to 3D** queues the mesh stage from the flattened image. A linked document promotes the
  reference it already is — and refuses if you have unsaved changes, so the mesh is made from what
  you can see. An unlinked one becomes an ordinary image job, the same call the 3D pane's upload
  button makes. Either way, if the quality report is unhappy you get a confirm naming the reasons
  rather than a refusal.

A painted reference is a real job row that never ran on the worker: the image already exists, so
queueing a run to reproduce what you just drew would be two minutes of GPU for nothing. It is
created finished, at the reference stage, which is exactly what promotion consumes. It cannot be
rerolled — there is no generator behind it for a new seed to change — but it can be remeshed. See
[Rerun and promotion](08-library-and-jobs.md#rerun-and-promotion).

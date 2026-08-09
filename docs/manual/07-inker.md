# Inker

Inker is the top-level drawing mode: a layered raster editor, built into the app rather than bolted
onto it. It exists because the reference image is the biggest lever on mesh quality, and the fastest
way to fix a nearly-right reference is usually to paint over it.

It is a mode, not a takeover. Switching away leaves every open document exactly where it was, and a
reconstruction job started before you switched keeps running with its progress card floating over
the canvas. Only quitting the app and closing a tab can lose pixels, and both ask first.

The layout follows the rest of the app: tools and their options on the left, the canvas in the
middle, layers and the pipeline panel on the right. Several documents stay open at once, as tabs.

## Starting a canvas

**New** offers three square presets and, under them, width and height fields for anything else —
1920 × 1080, a tall banner, a tile. Sizes are clamped rather than refused, up to 8192 px a side: the
fields are being typed into, and there is nothing useful to show halfway through a number. The
figure you get is the one printed on the Create button.

Changing the size of a document you already have is a different pair of operations, in the document
panel on the right: **Scale image** resamples the picture, and **Resize canvas** changes how much
room it has, with a 3 × 3 anchor saying where the old picture sits in the new space.

## Turning the page

Two buttons at the end of the file row change how the canvas is *shown*, and nothing else — no pixel
moves, so there is nothing to undo and nothing to save.

- **Rotate the view** (`Ctrl+4`, `Ctrl+Shift+4` the other way) turns the canvas a quarter at a time.
- **Flip the view** (`Ctrl+5`) mirrors it left to right. This is the oldest check there is on a
  drawing: errors you have stopped seeing are obvious in the mirror.

Quarter turns rather than a free angle, deliberately: at a quarter turn every overlay — the grid, the
marching ants, the symmetry lines, the transform box — stays exactly as accurate as it was, and a
free angle would put all of them slightly wrong. While either is on, a small button beside them says
so and sets the view back upright, because a mirrored canvas you have forgotten about quietly
teaches the wrong hand.

Do not confuse these with **Flip H**, **Flip V** and **Rotate** in the document panel's *canvas*
section. Those move pixels: they are edits, they are one undo step each, and they change what a save
writes. These two change nothing at all, which is also why they keep working while a save is in
flight — an editor that would not let you *look* at your drawing while it writes a file would be a
strange one.

## Tools

The toolbox is an icon grid; hovering a tool shows its name and its letter. Every tool is listed in
[Keyboard shortcuts](14-shortcuts.md).

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

The painting tools have **Size**, **Hardness**, **Opacity** and **Spacing**; blur and smudge add
**Strength**. The shape tools have the size slider and, except for the line, a **Filled**
checkbox. Fill and the wand have **Tolerance** (0 to 255) and **Contiguous** — turning contiguous
off acts on every similar pixel in the image, not just the ones touching where you clicked. The
gradient tool chooses its **Shape** and whether it fades **To transparent**. Pick has **This layer
only** — off, it reads the colour you can see, which is the blend of every visible layer; on, it
reads the active layer's own pixels, before its opacity and blend mode. The second is what you want
picking a line colour off lineart with flats underneath it, because the blend of the two is a
colour that exists nowhere in the document.

**Each tool remembers its own.** Sizing the eraser to 60 to clean up a corner leaves the brush at
whatever you had it, and switching back finds each tool exactly as you left it. **Reset _tool_**
under the options puts the one in your hand back to its defaults and touches nothing else. The
settings below the tool options — symmetry, the grid, the colours — are canvas-wide and shared,
because they are properties of what you are drawing rather than of what you are drawing with.

A gradient runs from the foreground colour to the background one by default. **Add stops** turns
that preset into an editable list: each stop has a position, a colour and an alpha, and a gradient
with three of them can fade out in the middle and back in. **Use fg / bg** goes back to the preset,
which is a live reading of the two colours rather than a copy — so `X` still changes the next
gradient you draw.

The painting tools also have **Smoothing** and **Taper**. Smoothing makes the brush follow the
cursor at a distance instead of exactly, which turns a shaky line into a smooth one; it catches up
when you stop moving, so a stroke still ends where you left it. Taper thins a fast stroke, for a
pen-like flick. Both are off by default, and with both off a stroke is the same stroke this app has
always drawn.

Two canvas-wide aids sit below the tool options. **Symmetry** mirrors every stroke — off,
left/right, top/bottom, both, or **radial**, which repeats it around a circle a set number of ways
(2 to 32) for snowflakes and mandalas. With any symmetry on you can set the **axis** the mirrors
reflect about, in image coordinates; **Centre** puts it back, and "centred" means exactly that even
after the canvas is resized. **Grid** overlays a grid at a spacing you set, from 2 to 512 pixels,
and **Snap to grid** lands shapes, lines and the marquee on its intersections. Freehand strokes
never snap: quantising a brush to a lattice is a different tool, not a drawing aid.

## Colour

Two colours, not one: **Foreground** and **Background**. The gradient tool needs both ends, and `X`
swaps them — universal muscle memory from every other raster editor. Both carry an alpha bar, so a
semi-transparent brush is a colour rather than a separate mode.

Both swatches show their hex value inline and open a full picker — wheel, HSV, hex, alpha — when
clicked, so a colour somebody sent you as `#3b4252` can be typed straight in.

Below them is a row of **swatches**. Clicking one makes it the foreground; the row is saved with
your settings rather than reset each session, because a project has a palette and retyping it every
time is the kind of small friction that makes a tool feel unfinished.

**Import .gpl** and **Export .gpl** move that row in and out as a GIMP palette, which is the format
GIMP, Krita, Aseprite and Inkscape all read. Two things about it: the format has no alpha channel,
so exported swatches are written opaque, and an import **adds** to the row rather than replacing it
— unwanted colours are a right-click each, where a palette silently wiped has no way back.

The `I` **Pick** tool samples a colour from the canvas into the foreground.

## Layers

The layers panel shows the stack top-first, the way every editor shows it.

Each layer has a visibility checkbox, a thumbnail, a name, an **Opacity** slider and a **Blend**
mode. Above the list are **Add**, **Copy**, **Delete**, **Merge down** and **Flatten**. Dragging the
opacity slider previews live but records a single undo step when you let go, rather than one step
per pixel of drag.

The blend modes are the twelve separable ones, listed in the order every editor groups them —
darkening, then lightening, then contrast, then comparison:

| | |
|---|---|
| `normal` | the layer, over what is under it |
| `darken`, `multiply`, `color-burn` | can only darken the backdrop |
| `lighten`, `screen`, `color-dodge`, `add` | can only lighten it |
| `overlay`, `hard-light`, `soft-light` | darken the dark half and lighten the light half |
| `difference` | the distance between the two colours |

These are the W3C formulas, which is what OpenRaster's composite operators are defined against — so
a document saved here and reopened in Krita or GIMP composites identically rather than approximately.
The four *non-separable* modes (hue, saturation, colour, luminosity) are not implemented; a file
that arrives using one opens with that layer set to `normal` rather than being refused.

**Lock alpha** paints inside what is already on a layer and never past its edge: colours change,
transparency does not. It is how you recolour lineart or shade a shape without selecting it first —
and it makes the eraser a no-op on that layer, because erasing *is* changing transparency. The lock
is saved with the document; other editors ignore it and open the layer as an ordinary one.

One rule about erasing is worth stating plainly, because it differs from some editors. **The eraser
makes pixels transparent.** It does not paint the background colour. The background colour — the
matte — is a property of the document, applied once when the image is flattened for export. So what
you erase through is genuinely a hole until the moment you export, and changing the document's
matte changes what every erased area exports onto without touching a single stroke.

Undo keeps up to 64 steps, and fewer than that when they are large — the stack is bounded by bytes
(192 MB) first and by that depth ceiling second, so a document of small dabs gets all 64 and a
document of full-canvas crops gets a handful. Steps are addressed by layer identity rather than by position
in the stack — an undo issued after you reorder layers still lands on the layer the edit was
actually made to. The document is "dirty" when it differs from the last save, which means undoing
back to the saved state marks it clean again rather than leaving it unsaved forever.

The pipeline panel on the right also shows **Undo** and **Redo** buttons and the current history
depth, alongside canvas operations: **Flip H**, **Flip V**, **Rotate**, **Fit view**, and
**Resize...**, and **Filter...**. The resize popup deliberately offers two different operations —
**Scale image** resamples the picture, **Resize canvas** changes how much room it has around the
picture.

The 3×3 **anchor** grid says where the old image sits in the new canvas, and it belongs to Resize
canvas only: scaling has no slack to put anywhere. Growing a canvas anchored centre adds room on
all four sides; anchored top-left it adds room right and below, which is what the button did before
there was a grid. Shrinking works the same way and crops from the opposite sides.

## Filters

**Filter…** in the document panel opens five whole-layer adjustments: brightness/contrast,
hue/saturation, levels, blur and sharpen. Every one previews live on the canvas as you drag, and
the whole session — however many sliders you moved and however many times — records as a single
undo step when you press Apply. Cancel, or clicking away from the popup, puts the pixels back and
records nothing at all.

Three things about what they do to a layer. They apply to the **selection** if there is one, faded
by a feathered edge exactly as a brush would be, and to the whole layer if there is not. The colour
filters never change transparency; blur does, because blurring a layer's edge is most of the reason
to blur one. And a layer with **Lock alpha** on keeps its transparency under all five.

Every filter opens at settings that change nothing, so the preview is safe to start immediately —
the picture only moves once you move a slider.

## Selections and transform

Four tools make selections: the rectangular marquee, the ellipse, the lasso and the wand. Hold
**Shift** while dragging to add to the current selection and **Alt** to subtract from it.

With a selection live, the **selection** section offers **All**, **None**, **Invert**, a
**Feather** radius slider up to 32 pixels with a **Feather** button, and **Crop to selection**. The
same actions have keyboard shortcuts: `Ctrl+A`, `Ctrl+D` and `Ctrl+Shift+I`.

**This layer** selects what is painted on the active layer, at the coverage it is painted at — a
soft brush edge becomes a soft selection rather than a jagged one. It reads the layer's own pixels,
so its opacity and blend mode do not enter into it.

**Grow**, **Shrink** and **Border** move the edge by a whole number of pixels, which is a different
thing from feathering: feather *softens* an edge where these *move* it, so they have their own
control. Border replaces the selection with the band that many pixels either side of its edge —
fill it and you have stroked the outline.

Cutting or copying a selection puts it on the clipboard; pasting brings it back as a **floating
buffer** which you move with the Move tool and commit by doing anything else. `Esc` cancels a
floating buffer, and `Delete` clears the selected pixels.

Cancelling a lift — where the buffer was cut out of a layer — puts the pixels back and removes that
step from history entirely, rather than leaving it on the redo stack where `Ctrl+Y` could replay the
cut with no buffer left to restore.

**Free transform** (`Ctrl+T`, or the button in the tool options) rotates and scales the selection,
or the whole layer when there is no selection. It is modal: while transforming, **Enter** applies
and **Esc** cancels, and nothing else can change the tool out from under a half-finished transform.

## Animation

A drawing can become a frame-by-frame animation. Press **Animate** in the document panel: the
layers you have become *tracks*, the drawing becomes frame one, and a second empty frame is added.
It is a single edit — one `Ctrl+Z` turns the document back into the still image it was.

Once a document is animated a **timeline** strip appears under the canvas: frames across, tracks
down, one square per cell. A filled square is a drawing, `=` is a linked cel, and an empty outline
is a frame that track has nothing on. Clicking a square selects that track and moves to that frame;
right-clicking one offers link, unlink and clear. Right-clicking a frame number offers insert,
duplicate, reorder and delete. `,` and `.` step back and forward a frame.

**Cels are created by drawing on them.** There is no "add cel" button: the grid is empty until you
paint, and the first stroke on a blank frame creates the cel it needs. That is still one undo step,
and a stroke that changes nothing leaves nothing behind.

**Linked cels are one drawing in several frames.** *+ Link* adds a frame that shares the current
one's cels rather than copying them, so a background held across twenty frames is stored — and
edited — once. Painting on any of them paints on all of them, which is the point. **Unlink** gives
that one frame a private copy from then on. *+ Copy* is the other choice: an independent duplicate
you can diverge immediately.

**Durations are per frame**, in milliseconds, in the box on the transport row — so a held pose and
a fast blink live in the same clip without anything having to be a frame rate.

**Onion skin** shows the neighbouring frames beneath the one you are drawing, the previous tinted
red and the next tinted green. Toggle it on the transport row; while it is on, **back**, **ahead**
and **fade** set how many frames either side are drawn and how strongly. Both counts may be zero,
which is how you see only what is behind or only what is ahead.

**Playback** is the Play button or `Enter`; `Esc` or Play again stops it, leaving the playhead where
you last saw it. While playing, the document is read-only — the canvas is showing a cached picture
of another frame, so a stroke would land somewhere you cannot see. If a **tag** covers the current
frame, playback loops inside that tag rather than over the whole timeline.

**Tags** name a span of frames — "walk", "idle", "hit". Right-click a frame number and choose
**New tag here** to make a one-frame tag, then right-click the tag's name in the band under the grid
to rename it, to set either end to wherever the playhead is, to turn its looping off, or to delete
it. Tags may overlap, and playback follows the innermost one containing the playhead — which is what
makes a short **hit** inside a long **combat** the useful arrangement rather than an ambiguous one.
Tags are saved with the document and written into a sprite sheet's sidecar.

Two things are unavailable while a document is animated: **merge down** and **flatten**. Both are
defined over one layer stack and an animated document has one per frame, so rather than guess which
frame you meant, the buttons say so.

Moving the playhead is not an edit. It pushes no undo step and does not make the document unsaved —
looking at another frame is looking, not drawing.

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

An animated document saves into the same `.ora`. The frames are written as nested groups, so Krita
and GIMP open the file and show frame one rather than refusing it; the timeline itself — durations,
tags and which cels are shared — rides along in a member those editors ignore. Opening such a file
in anything that does not understand that member, **including an older build of Warlock**, and then
saving it, writes the file back flat and loses the animation.

**Export sheet** packs an animated document into one PNG atlas plus a JSON sidecar, one cell per
frame, wrapping into rows when a single row would be wider than an engine will accept as a texture.
The sidecar names each cell, its duration and any tags, in the same format the 3D sprite sheets use.
Two things about it are worth knowing. A cel linked across three frames becomes three identical
cells, because the engine playing it back knows nothing about links. And the cells keep their
transparency rather than being flattened onto the document's matte, which is what a sheet wants
almost always — a matte is what a *flattened* export puts behind transparency, and an atlas is
composited over whatever is behind it in the game.

## Autosave and recovery

Every open document with unsaved changes is copied to `assets/autosave/` every two minutes. This is
crash safety and nothing else: an autosave is **not** a save. It does not mark the document saved,
it does not choose a location, and it does not touch a linked job — all it promises is that a crash
costs you minutes rather than an afternoon. Saving or closing a document removes its copy, because
an autosave that outlived its document is exactly the file that turns up later and confuses you.

If Warlock finds copies left over from a previous session, it offers to reopen them once, at
startup. Recovered documents open **untitled and unsaved**, deliberately: the file each was copied
from may still be on disk with its own contents, and adopting that path would arm `Ctrl+S` to
overwrite something you have not looked at yet. Declining keeps the copies — "not now" is not
"delete my work" — and they are cleared once you save or close whatever you recover.

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

### Fixing a matte

**Fix matte**, in the [Check the cutout](04-generating-meshes.md#checking-the-cutout) panel, is the
same hand-off with one difference: the document opens with the cutout already folded into its alpha,
as a single undoable step. Every layer keeps its own pixels — the cut is multiplied into each of
them — so a layered reference stays layered, and the eraser and the brush now edit the matte itself.
Erase to cut more away; paint on a layer to put it back.

The tab opens **unsaved**, because the cutout is on screen and in no file. Saving writes it exactly
as any other linked save does, and the reference then carries that alpha. Promoting it afterwards
records the matte as approved and tells the reconstruction engine to keep it rather than cutting its
own.

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
[Rerun and promotion](11-library-and-jobs.md#rerun-and-promotion).

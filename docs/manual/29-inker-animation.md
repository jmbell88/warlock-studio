# Inker: animation

A drawing in the [Inker](28-inker.md) can become a frame-by-frame animation. Everything in that
chapter still applies — the tools, the layers, the colour modes, the selections — and this chapter is
the second axis the document grows: frames across, layers down, a grid of cels between them.

Nothing here exists until you ask for it. A still drawing is a one-frame sprite: the strip under
the canvas is there and lists your layers, but it has one column and no transport, and the controls
that would act on a clip are greyed rather than hidden — so the shape of the mode is visible before
you enter it.

## Making a document animated

Press **Animate** in the document panel. The layers you have become *tracks*, the drawing becomes
frame one, and a second empty frame is added.

It is a single edit — one `Ctrl+Z` turns the document back into the still image it was.

## The timeline

The strip under the canvas is always there — it is where the layers live, see
[Layers](28-inker.md#layers) — and animating a document is what gives it more than one column and a
transport row above it: frames across, tracks down, one square per cell. A filled square is a
drawing, `=` is a linked cel, and an empty outline is a frame that track has nothing on.

Clicking a square selects that track and moves to that frame; right-clicking one offers link, unlink
and clear. Right-clicking a frame number offers insert, duplicate, reorder and delete. `,` and `.`
step back and forward a frame, `Home` and `End` jump to the first and last, `Alt+N` adds a frame and
`Alt+D` duplicates the one you are on.

Clicking a track's **name** selects that layer without moving the playhead, and `Shift`+clicking one
stretches the selection across the tracks between it and the layer you were on. A layer that is in a
folder draws under a **folder header** with a fold triangle: pressing it hides everything inside,
and the header carries the folder's own eye and its right-click menu. Folding is a view setting —
neither saved nor undoable — for the same reason the playhead is.

The strip's controls are two rows. The top one is the frame you are on — the transport, the frame
operations, the counter and that frame's duration — and the bottom one is what leaves the app: the
three exports, with the onion-skin and thumbnail switches and the export magnification beside them.
Both rows give up their labels for their icons before they give up any button, and drop what they
can into a **…** menu after that, so nothing is ever pushed off the right-hand edge.

Moving the playhead is not an edit. It pushes no undo step and does not make the document unsaved —
looking at another frame is looking, not drawing.

### Cel thumbnails

The **Thumbs** switch on the second row of the timeline draws each cel's picture in its cell and
grows the cells to fit. Linked cels share one thumbnail, so a link looks like the same drawing
standing in several columns.

## Cels

**Cels are created by drawing on them.** There is no "add cel" button: the grid is empty until you
paint, and the first stroke on a blank frame creates the cel it needs. That is still one undo step,
and a stroke that changes nothing leaves nothing behind.

### Continuous layers

**Continuous layers** change what that new cel starts as. The **Cels** toggle at the top of the
layers panel makes the active track continuous: drawing on an empty frame of it starts from a
*copy* of the nearest earlier drawing on that track rather than from nothing, which is how you
carry a held pose or a background forward and then change it.

It is a copy and not a link, so editing it leaves the frame it came from alone — if you want them to
stay the same drawing, use **Link** below. The flag is saved with the document, and an `.aseprite`
layer marked "prefer linked cels" opens as a continuous one (Aseprite links there where this
copies). It is a track setting, so a still image has nothing to be continuous about and the toggle
is greyed until you Animate.

### Linked cels

**Linked cels are one drawing in several frames.** *Link* adds a frame that shares the current one's
cels rather than copying them, so a background held across twenty frames is stored — and edited —
once. Painting on any of them paints on all of them, which is the point.

**Unlink** gives that one frame a private copy from then on. *Copy* is the other choice: an
independent duplicate you can diverge immediately.

### Cel opacity

**Each cell has an opacity of its own**, on the same right-click menu, and it is a *multiplier* on
the layer's own — a layer at 50% with a cel at 50% draws at 25%. That is what makes it useful for
a fade: leave the layer where it is and dim the cells across a few frames.

**A linked cel can be dimmed on one frame without dimming the others.** The drawing is still one
drawing — paint on it and every frame changes, exactly as before — but the opacity belongs to the
*slot* rather than to the picture, so the same held pose can fade out over ten frames without
being unlinked into ten copies first. A cell with no drawing in it has no opacity to set, and the
slider is not offered there. The value is saved with the document, and survives a round trip
through `.aseprite` in both directions.

### Cel z

**A cell can also be lifted above its own layer**, with the **Z** slider beside the opacity one on
the same right-click menu. It is an *offset*, not a reorder: `+1` draws that one cell a row higher
in the stack than its layer sits, `-1` a row lower, and the layer itself does not move — the
timeline order, the layer panel and everything you address by picking a row all go on meaning what
they meant. A hand that has to pass in front of the body on three frames of a walk and behind it on
the other five is the case this is for; it used to mean two layers and an eye toggle on every
frame.

Two things are worth knowing. **A tie keeps layer order**, so an offset that lands exactly on
another row's height sits *under* it — to clear the top of a three-layer stack from the bottom, use
3 and not 2. And **a lift belongs to the slot, not the drawing**: a linked cel can be in front on
one frame and behind on the next while still being one drawing you paint once. A cell with nothing
in it has nothing to lift, and the slider is not offered there.

While any cell on the frame you are drawing on is lifted, the editor stops caching the layers
underneath the one you are painting on, because a lifted cel can be one of them. A dab costs a
little more and the *first* dab of a stroke costs a great deal less; you are unlikely to notice
either. The numbers are in `docs/measurements/2026-08-30-cel-z-below-cache.md`.

### Colours and notes

**A layer, a cel and a tag can each carry a colour and a line of text.** Right-click any of the
three and the menu ends with **Properties...** for the text and a row of swatches for the colour;
the last swatch, marked with an ×, takes the colour off again. The colour is drawn where you set
it — a stripe down the layer's name, a stripe down the cel, and the tag's own band under the grid —
so a long clip can be read at a glance: the anticipation frames in one colour, the holds in another,
the tags for each attack in a third. The text shows in the layer row's tooltip and is there for the
next person to open the file, or for you in a month.

**A linked cel takes its own colour and its own note**, exactly as it takes its own opacity. The
drawing is still one drawing shared across the frames, but the label belongs to the *slot*, so the
same held pose can be marked "start of the swing" on one frame and "hold" on the next. A cell with
no drawing in it has nothing to label, and the block is not offered there.

Notes are saved with the document and survive a round trip through `.aseprite` in both directions —
they are Aseprite's own *user data*, which is where its layer, cel and tag colours live too. Two
things are worth knowing about the edges. A **still** drawing has no timeline, so it has no layers
in the timeline sense, no cels and no tags to hang a note on; opening a one-frame `.aseprite` that
carries user data says so rather than losing it silently. And the swatch row is seven fixed
colours rather than a full picker — a colour read across a grid of small cells has to be
*distinguishable* above all else — though a colour that arrives in a file is kept and drawn exactly
as the file stored it.

The layer row's older **Layer properties...** entry is unchanged and is a different thing: it is the
blend mode, the opacity and the locks.


## Timing

**Durations are per frame**, in milliseconds, in the box on the transport row — so a held pose and
a fast blink live in the same clip without anything having to be a frame rate.

## Onion skin

**Onion skin** shows the neighbouring frames beneath the one you are drawing, the previous tinted
red and the next tinted green. Toggle it on the strip's second row or with `F3`; while it is on, **back**,
**ahead** and **fade** appear on a row of their own and set how many frames either side are drawn
and how strongly. Both counts may be zero, which is how you see only what is behind or only what is
ahead.

**Current layer only**, on the row beneath them, ghosts just the layer you have selected on those
neighbouring frames rather than the whole frame. That is what you want when the drawing sits over a
static background: without it, the background is repeated in red and green under every ghost and
hides the line you are trying to see.

## Playback

**Playback** is the Play button or `Enter`; `Esc` or Play again stops it, leaving the playhead where
you last saw it. While playing, the document is read-only — the canvas is showing a cached picture
of another frame, so a stroke would land somewhere you cannot see. If a [tag](#tags) covers the
current frame, playback loops inside that tag rather than over the whole timeline.

## Tags

**Tags** name a span of frames — "walk", "idle", "hit". Right-click a frame number and choose
**New tag here** to make a one-frame tag, then right-click the tag's name in the band under the grid
to rename it, to set either end to wherever the playhead is, to turn its looping off, to choose
which way it plays, or to delete it.

Clicking a tag's name in the band jumps the playhead to that tag's first frame, and
double-clicking it opens the rename in place — the same field the menu's **Rename** opens.

Tags may overlap, and playback follows the innermost one containing the playhead — which is what
makes a short **hit** inside a long **combat** the useful arrangement rather than an ambiguous one.
Tags are saved with the document and written into a sprite sheet's sidecar.

### Direction

A tag plays **forward**, **reverse** or **ping-pong**, and that is a separate question from whether
it loops: direction is the path through the span, looping is whether reaching the end of that path
starts it again. A non-looping ping-pong swings out and back once.

Ping-pong is the one worth having — a torch flicker or an idle breath drawn as frames costs the
whole span again in cels, every one of them a duplicate of a drawing already in the file, and every
edit to the middle of the swing then has to be made twice.

### Repeat counts

A tag can also **repeat** a fixed number of times. Set **repeat** in the tag's menu to 3 and the
span plays three times and stops; leave it at 0 — the default, and what every document written
before this carries — and the Loop tick decides as it always has.

While a count is set the Loop tick is disabled, because the count is the more specific answer to the
same question — a tag that was set to play *once* and is then given a count of three plays three
times, and clearing the count back to 0 gives it its "once" back.

When the count runs out playback **stops inside the tag**, on the frame the span ends at; it does
not fall through into the frames after it, which is where Aseprite would carry on. A ping-pong
counts one out-and-back as one play. Repeats are saved with the document and written into a sprite
sheet's sidecar, and a tag exported on its own as a GIF carries its count into the file.

## Ranges

**Ranges** are how every command in this chapter reaches more than one cel. Drag across the grid to
select a rectangle of cells — press, drag, release; `Shift`+click extends the selection from where
you last pressed, and `Esc` clears it. The selection is drawn as one accent outline and is not part
of the document: it pushes no undo step and is not saved.

Right-clicking any cell opens the **Range** section of the cell menu, which acts on the whole
selection: copy and paste cels (the clipboard is shared between tabs, and a link inside the copied
block survives the paste as one drawing again), clear, link, unlink, duplicate the frames copied or
linked, reverse them, delete them, and set every frame's duration at once. Every one of them is a
single `Ctrl+Z`, and each is refused rather than half-applied — deleting *every* frame is the one
range delete that is not allowed, because a timeline keeps at least one frame.

### Drawing on a whole range

The same section carries the verbs that **draw on** every cel of the selection rather than
rearranging them: **Flip horizontal** and **Flip vertical**, **Rotate 90 clockwise**,
**anticlockwise** and **180**, the four **Shift** directions, and **Fill with foreground**. One
`Ctrl+Z` each, a linked cel touched once however many frames it appears on, an empty cel left
empty.

The flips, turns and shifts act on the whole cel — a selection weights a *fill*, but there is no
such thing as half-mirroring something — and they move pixels without inventing any, so on an
indexed document two palette slots holding the same colour stay two slots. A shift carries what goes
off one edge round to the other.

The quarter turns need a square canvas and are greyed otherwise: a cel is the size of the canvas,
and turning the cels without turning the canvas would leave the grid holding drawings of two
different shapes. To turn the whole document instead, use **Rotate** in the Canvas panel.

### A transform applies to the range

With a range selected, committing a free transform (Enter, or a click outside it) replays the same
move, turn, scale and slant on *every* cel in the selection — each one transforming its own
contents, not a copy of the one you were watching. That is Aseprite's timeline-target behaviour, and
the range outline is what tells you how far the commit will reach.

The preview shows only the cel you are on, because that is the only one that has moved yet.
Cancelling (`Esc`) is never ranged: nothing but that cel was ever touched. A *pasted* buffer commits
to the one cel it was aimed at whatever the range says — there is nothing to replay, since it was
never cut from anywhere.

### Filters over a range

With a range selected, the filter popup gains **Apply to range**: the filter runs over every cel in
the selection as one undo step. A linked cel is filtered *once* however many frames it appears on,
an empty cel stays empty rather than becoming a filtered blank, and the selection on the canvas is
honoured as a weight exactly as it is for a single layer — so a feathered selection fades the filter
in on every frame at once.

### Sheet corrections

A character sheet from Troupe opens with one tag per animation and direction, named
`walk_left`, `idle_back` and so on, and every frame is one cell of the same size. That is enough
for the timeline to know what the sheet *is*, and on such a document a strip appears between the
transport and the grid -- on an ordinary animation there is nothing there.

The strip's first row is **where a correction goes**: this frame in every direction, every frame of
this direction, every cell of this animation, the whole sheet, or the range you have selected. The
count beside it says how many cells that reaches. Three verbs send one:

- **Propagate patch** copies what you changed on this cell since it was *marked* onto every cell in
  the scope. A cell is marked the moment the playhead lands on it, so the usual case is: step to a
  frame, paint the fix, press the button. If you want to start measuring from now, **Re-mark**. A
  selection narrows what is sent; it is never required.
- **Replace across scope** recolours a colour pair, with a tolerance, on this cell and every cell in
  the scope -- the filter panel's replace, sent to cells that are not on screen.
- **Shift selection across scope** moves the selected pixels by whole pixels on every cell, clearing
  where they were and clipping at the edge rather than wrapping.

The second row is the **mirror**. Every direction but front and back has an opposite, and a fix
drawn on the west view can be offered to the east one flipped. Switch on **Preview diff** and the
canvas shows which pixels the other side would take; the count beside it says how many, and how many
the **face** box is holding back. The face is excluded because a face is not symmetric -- measured on
the reference sheets, mirroring left onto right differs only there -- and the slider decides how much
of the sprite, from the top, counts as face. **Apply to right** writes one cell; **Apply whole run**
writes every frame of the direction onto its counterpart.

Every one of these is a single `Ctrl+Z` however many cells it touched, a linked cel is written once,
an empty cell stays empty, and a press that would change nothing pushes nothing.

### Merging a re-render

The corrections above fix a sheet you have. This is what happens when the sheet itself changes
underneath you.

Say you clean up a walk cycle, then find the *run* clip was wrong, fix it in Poser and re-render. A
whole new sheet would throw away every cleanup you made to the walk. **Sheet ▸ Merge re-render**
brings the new render into the document you already have, cell by cell, and decides each one against
what the renderer gave you the first time:

- Cells you never touched take the new render.
- Cells you painted, where the render did not change, keep your work.
- Cells where **both** changed are conflicts. Your paint stays and the cell is flagged.

Nothing you painted is ever overwritten without asking. That is the rule, and it is not a setting:
a cell wrongly kept costs you one click to re-take, and a cell wrongly taken costs you the
afternoon. The whole merge is a single `Ctrl+Z`.

**Go to the next conflicted cell** walks the flagged cells and wraps at the end, and **Keep the hand
edit on this cell** clears a flag once you have looked. Keeping writes nothing — your paint is
already what is on the canvas — so it too undoes in one step.

The merge is offered only on a document opened from a rendered character sheet, because it needs a
third picture to compare against: what the renderer gave you when the document was made. A sheet
opened as a plain image has no such record, and the menu row says so rather than going grey with no
reason. That record travels in the `.ora` file, so it survives closing the document and coming back
to it a week later.

To make the sheet to merge, use **Re-render some runs** in Troupe: tick the animations and
directions you want rebuilt, and the rest are copied from the sheet you are re-rendering, at that
sheet's own settings.

### A palette per frame

On an **indexed** drawing each frame can carry a colour table of its own, which is how palette
cycling is done: the drawing does not move and the colours do. **Frame ▸ Give this frame its own
palette** starts one off as a copy of the drawing's, and every edit you make to the palette from
then on applies to that frame alone. **Use the drawing's palette here** takes the override away
again.

Nothing about the pixels changes — slot 4 stays slot 4 and becomes a different colour — so this is
not an edit you can lose track of: undo puts the table back in one step, and clearing the override
returns the frame to whatever the drawing's palette says today rather than to a snapshot of it.

The row is offered on indexed drawings only, and that is a real distinction rather than caution. On
an ordinary drawing with a palette set, the palette is a *rule applied to new strokes* — it snaps
what you paint onto the nearest colour in the table — so swapping it repaints nothing that is
already there. Only an indexed drawing stores slot numbers for a new table to re-colour.

Frames with their own palette survive a save to `.aseprite` and to `.ora` and come back on the same
frames. Aseprite writes these as a change applied from one frame onward; the same file opened here
shows each frame the colours it was given.

### The layers panel is the other half of a range

Its rows are the timeline's tracks, so a row inside the selection draws highlighted there too,
`Shift`+clicking a row stretches the selection across the tracks between it and the active one, and
clicking the eye on a row of a multi-track range hides or shows the whole range as one step.

`Ctrl`+clicking a row adds it to the selection or takes it out again, so the layers you act on do
not have to be next to each other — background, character and effects with two rows untouched
between them is three `Ctrl`+clicks. It builds on whatever is already selected rather than starting
over, so you can drag a range and then add one more row to it. Each selected row draws its own
outline: a selection with a gap in it has no single block to draw round, and one box round the whole
span would claim rows nothing is going to happen to.

The two kinds of selection are exclusive, because they are answering different questions. `Ctrl`+
clicking a layer clears any cell range you had, and clicking a cell — or plain-clicking a layer name
— clears the layers you had picked. `Esc` clears whichever you are holding.

A row's right-click menu acts on the whole block as well, and says so: with three rows selected it
reads **Duplicate 3 layers**, **Merge down 3 layers**, **Delete 3 layers** and **Group 3 layers**
rather than naming one. Each is a single `Ctrl+Z` however many rows it touched, and deleting every
layer is refused for the reason deleting every frame is.

**Merge down** and **flatten** are not unavailable here: both run across the whole grid, every frame
at once, and the links you have survive them — see [Layers](28-inker.md#layers).

## Preview

While a document is animated, a **Preview** pane sits at the top of the right column and plays the
clip in a corner of the screen. It is a second playhead, not the timeline's: pressing Play here does
not lock the document, does not move the frame you are drawing on, and does not stop when you paint.
That is the whole point — an animator runs the cycle and keeps working, and the frame under the
brush updates in the preview within a quarter of a second of each stroke.

The transport is Play/Stop, a frame counter, a **speed** multiplier from 0.25x to 4x, and a **scope**:
*Whole clip* runs the entire timeline round and round, *Active tag* follows the tag under the
preview's own frame — its direction, its looping and its repeat count. The preview always draws the
picture upright, ignoring any rotation or mirroring you have put on the canvas view, because those
are aids for drawing and this is a check on the result.

## Exporting part of a clip

The Range section ends with **Export range → sheet** and **→ GIF**, which write only the selected
frames. A tag's own menu has **Export tag → sheet** and **→ GIF**, which use that tag's span and its
looping — a tag with a repeat count writes a GIF that plays that many times.

In a partial export the tags are renumbered against the exported frames, tags that fall entirely
outside it are dropped, and a directional layout is not carried over: half a walk sheet is a clip,
not a smaller walk sheet.

### Splitting one export into several files

The export row also offers **Export sheet per tag** and **Export sheet per layer**, which write a
whole set of files from one press. You pick one name and each output is that name plus what it holds:
`hero_walk.png` and `hero_idle.png` for the tags, `hero_Background.png` and `hero_ink.png` for the
layers, each with its own sidecar beside it.

Per tag is exactly **Export tag → sheet** run for every tag in turn — same frames, same renumbered
tags, same looping — so a file from the batch and a file exported on its own are the same file.

Per layer writes one sheet per row of the layers panel, holding only that layer's own pixels. A group
is one file, not one per layer inside it, because a group is what the panel shows as a single row and
its layers composite as a unit. Hidden layers and hidden groups are left out entirely rather than
exported as sheets of nothing.

Both verbs stay available while the document is open and are greyed out rather than hidden when there
is nothing to split — no tags, or a single visible layer. A name a file cannot hold is cleaned up
(`A/B` becomes `A-B`), but two tags that would end up sharing a filename are **refused** rather than
numbered apart: a second `walk` quietly becoming `walk_2.png` is a file claiming to be a clip you
never named. Rename one and press again.

Splitting by slice is not offered; slices are exported as their own PNGs from the document panel
instead — see [Slices](28-inker.md#slices).

## Importing frames

**Import sprite sheet** in the document panel goes the other way: pick any image and give it a cell
size, an offset, the padding between cells and how many frames to take, and it becomes one frame per
cell, read row by row.

The popup counts the frames your numbers produce as you type them, and says what is wrong rather
than importing a short sheet — the last column and the last row carry no trailing padding, which is
the arithmetic that otherwise quietly drops a frame.

An imported sheet is an ordinary animation with no directions and no tags: a *layout* is something
the generator knows about its own output, not something a cell size can imply.

**Opening a GIF** needs no cell size at all: `Ctrl+O` an animated GIF and it arrives as one frame
per frame, on one layer, carrying each frame's own duration. GIF stores time in hundredths of a
second, so the durations you get back are the rounded ones — a clip exported at 33 ms comes back at
30. A GIF with a single frame opens as an ordinary still drawing. Inker cannot save a GIF in place,
so `Ctrl+S` on one offers an `.ora` beside it; **Export GIF** is the way back out.

### Directional layouts

A document opened from a generated sprite sheet carries a **directional layout** as well: its frames
are that sheet's cells in order — four or eight directions, with as many frames each as the action
has — and a sheet of an action arrives with a looping tag per direction, named for that action, so
playback loops one direction at a time. An `idle8` sheet is tagged `idle_front` and not
`walk_front`: a tag that names the wrong action is worse than no tag, because an importer believes
it. A turnaround is untagged, since four still views are not a cycle.

The layout is saved with the document and survives a round trip through `.ora`. You can draw on it,
repaint it and retime it like any other animation; adding or removing a frame is allowed and simply
means the timeline no longer fills the grid, which **Export sheet** then says rather than writing a
sheet with a hole in it. See [From a single
drawing](27-sprite-sheets.md#from-a-single-drawing).

### From an Aseprite file

**Import Aseprite file** opens an `.aseprite` or `.ase` and rebuilds it here: the layers with their
opacities, blend modes and locks, the layer groups with their nesting, every frame with its own
duration, the tags with their spans, directions and repeat counts, and the slices with their pivots
and nine-slice centres. A cel that Aseprite shares between frames arrives shared here too, so
editing it changes every frame it appears on — the same *linked cel* it was in the file it came
from. An indexed file brings its palette across and the document opens indexed; a greyscale one is
converted exactly, since grey is only a colour with its three channels equal. Tilemap layers and
their tilesets open too, as tilemap layers here — a tile you paint still edits the shared tileset
strip rather than becoming ordinary pixels.

The import itself is still **reading only** — it does not decide where a save goes, only what the
document is — so it opens as an **unsaved** document and the first `Ctrl+S` asks where to put it,
the same as any import. What has changed is the answer that dialog can give: Save As can now write
`.aseprite` itself, deliberately made from the save dialog once you have looked at the document,
never assumed by the door that read the file in. See [Saving](28-inker.md#saving) for that half.

Two kinds of thing do not come across, and they are told apart on purpose. Anything that would
change what the pixels *mean* is a refusal that names itself and opens nothing: a colour depth this
build cannot read, a canvas too small to draw on, a cel that will not decompress, a tilemap cel this
build cannot align to its own tile grid, a cel type nobody here knows, a tileset that links an
external file rather than carrying its own pixels, or a cel linked to a frame that holds none.
Anything cosmetic is a message and the file still opens: colour profiles (this app assumes sRGB
throughout). Per-frame palettes, per-cel opacity, a cel's **z-index**
and the **colours and notes** on layers, cels and tags *are* all kept now — except on a one-frame
file, which opens as a still image with no timeline for any of them to live on, and there each is a
message. What is still a message about user data even on an animated
file: a note on a *slice* or on a *tileset*, a note on an individual *tile*, and Aseprite's custom
**properties** — a typed key/value tree, which is a document format of its own inside the file. A
reference layer opens hidden, which is what exporting from Aseprite would do with it.

The full list of what comes across, what is only a message, and what a save back out to `.aseprite`
drops in turn, is kept in `docs/COMPAT.md`.

## A walk cycle from a still drawing

**Sprite → Create walk cycle...** turns a side-view drawing into an eight-frame walk. It is a
*prototype*, and it is narrow on purpose: one figure seen from the side, one walk on the spot, and
rigid cut-out limbs that turn rather than bend. What it saves you is the eight frames; what it
cannot do is draw.

Your drawing is **never edited**. Every part is copied out of it, the joints live with the setup,
and Bake writes a new document — so Cancel throws away the setup and leaves you exactly where you
started, with nothing to undo.

While a walk is being set up the right sidebar belongs to it: the toolbox and the drawing-file panel
stand down and the Preview pane shows the walk instead of the document. The paint tools genuinely do
nothing during a session — the canvas is placing joints, not painting — and everything in the file
panel is a File or Sheet menu row as well. Both come straight back when you bake or cancel.

### Assigning the parts

The panel on the right lists fourteen parts in five groups: the body, then the near arm and leg —
the ones on your side of the figure — then the far ones behind it. Each row takes either a layer of
your drawing, chosen from the list, or the pixels inside the current selection, taken with **Cut**.
Cut is the one to use on a drawing that is all on one layer: marquee or lasso the thigh, press Cut,
and it is copied into the rig without a mark on the original.

The body is required and so is one leg. A limb you leave out entirely is fine — a figure in profile
may genuinely have one arm hidden — but a *half*-assigned limb is refused by name, because a shin
with no thigh is always a mistake rather than a choice.

**Copy near arm across** and **Copy near leg across** start the far limb from the near one, art and
joints together, darkened by the **Far-limb shading** slider above them. That shading matters more
than it sounds: a far arm drawn identically to the near one in front of it reads as one arm. Adjust
the copy from there — repaint it, or re-cut it from a different selection.

### Placing the joints

The joints go on the **canvas**, over the drawing, at whatever zoom you are reading it at. The row
above the canvas names the joint a click will place; click the drawing to place it and the row moves
on to the next one that is missing. Drag any dot to adjust it — adjusting one already placed does
not move you on, since after the first pass every drag is a correction.

The dashed line across the canvas is the **ground**: where the feet stand. Drag it like a joint. It
starts under the lower of the two ankles, which is usually right for a figure drawn standing.

The lengths of the limbs are measured **once**, from where you put the joints, and nothing after that
changes them. A pose the leg cannot reach makes the *step* shorter; it never stretches a thigh.

### The four numbers

**Stride** is how far apart the feet are at a contact. Its slider stops where the leg can no longer
reach, and that ceiling moves when you drag the hip or the ground line, because it is geometry rather
than a fixed number.

**Foot lift** is how high the swinging foot clears the ground. **Arm swing** is how far the arms
swing, in degrees; they swing against the legs on the same side, which is what stops a walk reading
as a march.

**Body bob** is extra vertical travel. The body already sinks at the contacts without it — a figure
drawn standing has its hip a full leg above the ground, so it has to come down to reach a step at
all — and this deepens that sink rather than creating it.

**Frame duration** is milliseconds per frame. Eight frames at 100 ms is about a step and a bit per
second.

### Preview, and baking

The preview plays underneath at a whole-number zoom, because a pixel-art walk judged through a
fractional resample is a walk judged through a filter you will never ship it with. Step a frame at a
time with the arrows to check a pose.

If the cycle runs off the canvas the panel says so, and by how much, **before** you bake — the bake
crops silently, and a foot lost to the edge looks like a rig error rather than a framing one.

**Bake to new animation** — or Enter — opens the result in a new tab: one layer per part, eight
frames at your duration, and a looping `walk` tag. Every cel is its own drawing rather than a linked
one, so you can paint on frame three without touching frame five. It is an ordinary animated
document from that moment on: retime it, onion-skin it, [export it as a
sheet](#exporting-part-of-a-clip) or a GIF.

`Esc` cancels. The setup is not saved with anything — reopening the tool starts a fresh rig — so
finish a figure in one sitting, and keep the baked document rather than the setup that made it.

## Effects

**Flourish → Insert effect...** puts a procedural effect — a fireball, a shockwave, a heal, a
portal, one of about thirty — into the document as a **layer group** above the active layer, with
one **tag per phase** (*cast*, *projectile*, *impact*, *explosion*, *dissipate*, or whatever the
effect's phases are called). Pick the effect, whether it should look **painterly** (soft, many
colours, one layer per ingredient: core, flame, glow, sparks, smoke, trail, heat) or like **pixel
art** (hard alpha, one palette across every frame, one layer holding the finished composite), and
how many **facings** it should be rendered in. With four or eight facings the simulation itself is
turned — a spark stream that fires right fires down at 90° with the same spread, the same gravity —
and each phase gets a tag per facing, `impact/E`, `impact/S` and so on. Frames are added to the
timeline if the effect needs more than the document has.

Nothing about an effect is a picture until it is rendered, and everything about it is a number you
can change. With a layer of the effect active, the **inspector** appears under the timeline's
transport: the seed, the frame rate, the look, the frames and loop flag of every phase, and — for
whichever of the effect's layers you pick — every parameter its ingredient has: colours, radii,
counts, speeds, turbulence, gravity, lifetimes. A parameter that animates across a phase shows two
sliders, what it starts at and what it ends at. Change anything and the effect renders again in
the background a moment after you stop dragging; the timeline updates when it lands, and the whole
render is **one undo step**. The frame loop never waits on a render.

**Nothing you painted is overwritten silently.** Every render remembers what it put in each cel.
When a regenerate arrives, a cel that still holds the last render takes the new one; a cel you have
painted on keeps your paint and is **flagged**, and the inspector says how many. **Keep painted
cels** clears the flags and leaves your work; **Replace painted cels** renders over them. This is
the same three-way rule a re-rendered character sheet is merged with, for the same reason: a cel
wrongly kept is one click to re-take, and a cel wrongly taken is an afternoon gone.

**Detach effect** forgets the recipe and leaves the layers as ordinary layers. The pixels do not
change; only the ability to regenerate goes. An effect's recipe is saved in the `.ora` beside the
group it belongs to and comes back when the file is opened, so a regenerate is still possible next
week; a save to `.aseprite` keeps the layers and tags and drops the recipe, because that format has
nowhere to hold one — see `docs/COMPAT.md`.

The inspector's **prompt field** takes words — *colder, more sparks, no smoke, green flames,
bigger, faster, brighter, longer* — and turns them into parameter changes; **Apply words** or
Enter. Two things can be reading them, and the label beside the field says which. **keywords**
is a fixed vocabulary of colour names, kind names (sparks, smoke, flames, core, ring, trail,
glow, heat) and adjectives, deterministic and offline, and it names in the toast every word it
acted on. **model** is a small local instruct model, used when one has been placed in the
`text-instruct` directory under the model root (doctor's *text model* row says whether it was
found); it is asked for a list of parameter changes and *only* that, and whatever it answers goes
through the same clamp every slider goes through — an unknown layer or parameter is dropped and
named, a value outside its range is pulled inside it — so the model can narrow what is already
legal and never widen it. If it answers nonsense or takes too long the words fall back to the
vocabulary, and the toast says so. Either way the change is a pending edit rendered like any
other: one undo step, nothing painted overwritten.

Two ingredients are pictures rather than arithmetic: a **sprite** layer stamps a texture (scaled,
turned, faded, flickering), and **particles** stamp one instead of a disc when given one. A texture
is an *asset* of the effect, held beside its recipe and saved with the `.ora`. **Flourish → Use
selection as texture** makes one from the selected pixels of the active layer — draw an ember or a
rune, select it, and the layer the inspector is showing takes it if it can. **Generate texture...**
asks the image model for a single centred ingredient on black from a few words, keys the black out
into alpha (through the matting model when the machine has one), and lands it the same way; it needs
an SDXL-family model like every generation, and one runs at a time per document. A layer whose
texture is missing renders nothing rather than a placeholder.

**Restyle keyframes...** is the third and least reliable AI door, and it says so. It sends a few
frames of one phase — the effect's own composite — through the image model as img2img with your
words and a strength, and fills the frames between by crossfading the returned keyframes under the
effect's own motion field, so the blend has a direction rather than a ghost. The result lands as a
**new layer inside the group** with the procedural layers left in place; a regenerate leaves it
alone, because what the image model painted is not the recipe's to redo. Opt-in, one undo step,
and unmeasured: judge it against the procedural frames before you keep it.

**Flourish → Export effect sheets...** writes one sprite sheet per phase — it is the per-tag export
([Exporting part of a clip](#exporting-part-of-a-clip)), since a phase is a tag — each with its
sidecar. **Engine snippet...** shows the few lines that load one exported phase in Pygame-CE, Godot,
Unity or Phaser, with a Copy button; it assumes the export's filenames and an origin at the canvas
centre, which is where an effect is placed.

An effect is deterministic: the same recipe with the same seed renders the same bytes on any
machine, so a preset you tune and save renders the same after an upgrade. Change the **seed** for a
different arrangement of the same sparks.

## Slices on an animated document

[Slices](28-inker.md#slices) are the same rectangles they are on a still drawing, and by default one
slice is the same rectangle on every frame.

**Key this frame** gives the current frame its own rectangle, pivot and centre; every other frame
goes on using the slice's own. Keys are always explicit — dragging a slice moves it everywhere,
because a drag that silently keyed whichever frame you happened to be on is how a clip ends up with
forty slightly different rectangles nobody meant.

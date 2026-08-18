# Inker: animation

A drawing in the [Inker](08-inker.md) can become a frame-by-frame animation. Everything in that
chapter still applies — the tools, the layers, the colour modes, the selections — and this chapter is
the second axis the document grows: frames across, layers down, a grid of cels between them.

Nothing here exists until you ask for it. A still drawing has no timeline, and the controls that
would act on one are greyed rather than hidden, so the shape of the mode is visible before you enter
it.

## Making a document animated

Press **Animate** in the document panel. The layers you have become *tracks*, the drawing becomes
frame one, and a second empty frame is added.

It is a single edit — one `Ctrl+Z` turns the document back into the still image it was.

## The timeline

Once a document is animated a **timeline** strip appears under the canvas: frames across, tracks
down, one square per cell. A filled square is a drawing, `=` is a linked cel, and an empty outline
is a frame that track has nothing on.

Clicking a square selects that track and moves to that frame; right-clicking one offers link, unlink
and clear. Right-clicking a frame number offers insert, duplicate, reorder and delete. `,` and `.`
step back and forward a frame.

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

## Timing

**Durations are per frame**, in milliseconds, in the box on the transport row — so a held pose and
a fast blink live in the same clip without anything having to be a frame rate.

## Onion skin

**Onion skin** shows the neighbouring frames beneath the one you are drawing, the previous tinted
red and the next tinted green. Toggle it on the strip's second row; while it is on, **back**,
**ahead** and **fade** appear on a row of their own and set how many frames either side are drawn
and how strongly. Both counts may be zero, which is how you see only what is behind or only what is
ahead.

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

### The layers panel is the other half of a range

Its rows are the timeline's tracks, so a row inside the selection draws highlighted there too,
`Shift`+clicking a row stretches the selection across the tracks between it and the active one, and
clicking the eye on a row of a multi-track range hides or shows the whole range as one step.

**Merge down** and **flatten** are not unavailable here: both run across the whole grid, every frame
at once, and the links you have survive them — see [Layers](08-inker.md#layers).

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

## Importing frames

**Import sprite sheet** in the document panel goes the other way: pick any image and give it a cell
size, an offset, the padding between cells and how many frames to take, and it becomes one frame per
cell, read row by row.

The popup counts the frames your numbers produce as you type them, and says what is wrong rather
than importing a short sheet — the last column and the last row carry no trailing padding, which is
the arithmetic that otherwise quietly drops a frame.

An imported sheet is an ordinary animation with no directions and no tags: a *layout* is something
the generator knows about its own output, not something a cell size can imply.

### Directional layouts

A document opened from a generated sprite sheet carries a **directional layout** as well: its frames
are that sheet's cells in order, four directions with one or four frames each, and a walk sheet
arrives with a tag per direction so playback loops one direction at a time.

The layout is saved with the document and survives a round trip through `.ora`. You can draw on it,
repaint it and retime it like any other animation; adding or removing a frame is allowed and simply
means the timeline no longer fills the grid, which **Export sheet** then says rather than writing a
sheet with a hole in it. See [From a single
drawing](07-sprite-sheets.md#from-a-single-drawing).

### From an Aseprite file

**Import Aseprite file** opens an `.aseprite` or `.ase` and rebuilds it here: the layers with their
opacities, blend modes and locks, the layer groups with their nesting, every frame with its own
duration, the tags with their spans, directions and repeat counts, and the slices with their pivots
and nine-slice centres. A cel that Aseprite shares between frames arrives shared here too, so
editing it changes every frame it appears on — the same *linked cel* it was in the file it came
from. An indexed file brings its palette across and the document opens indexed; a greyscale one is
converted exactly, since grey is only a colour with its three channels equal.

Reading only, and it shows in one place: the import opens as an **unsaved** document, so the first
`Ctrl+S` asks where to put it and writes an `.ora`. Nothing this app does can write back over the
`.aseprite`, which is deliberate — a format read by one program and written by another is how a
day's work goes missing.

Two kinds of thing do not come across, and they are told apart on purpose. Anything that would
change what the pixels *mean* is a refusal that names itself and opens nothing: a tilemap layer,
whose pixels live in a tileset; a colour depth this build cannot read; a cel linked to a frame that
holds none. Anything cosmetic is a message and the file still opens: colour profiles (this app
assumes sRGB throughout), user data and timeline colours, a per-cel opacity (opacity is a layer
property here), a cel's z-index (layer order is stacking order). A reference layer opens hidden,
which is what exporting from Aseprite would do with it.

## Slices on an animated document

[Slices](08-inker.md#slices) are the same rectangles they are on a still drawing, and by default one
slice is the same rectangle on every frame.

**Key this frame** gives the current frame its own rectangle, pivot and centre; every other frame
goes on using the slice's own. Keys are always explicit — dragging a slice moves it everywhere,
because a drag that silently keyed whichever frame you happened to be on is how a clip ends up with
forty slightly different rectangles nobody meant.

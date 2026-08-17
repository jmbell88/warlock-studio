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

**New**, **Open**, **Save**, **Save as** and **Export PNG** are in the **file** section at the top of
the right-hand panel, where Plotter's and Packwright's have always been. The canvas's own row keeps
only what acts on the drawing in front of you: undo and redo, the two view turns, the tiling control
and the one word that says whether there is anything unsaved.

**New** offers three square presets and, under them, width and height fields for anything else —
1920 × 1080, a tall banner, a tile. Sizes are clamped rather than refused, up to 8192 px a side: the
fields are being typed into, and there is nothing useful to show halfway through a number. The
figure you get is the one printed on the Create button.

Changing the size of a document you already have is a different pair of operations, in the document
panel on the right: **Scale image** resamples the picture, and **Resize canvas** changes how much
room it has, with a 3 × 3 anchor saying where the old picture sits in the new space.

## Turning the page

Three buttons on the canvas row change how the canvas is *shown*, and nothing else — no pixel
moves, so there is nothing to undo and nothing to save. They are the one group on that row that stays
live while a save is running: refusing to let you *look* at a drawing while it writes a file would
be an odd kind of care.

- **Rotate the view** (`Ctrl+4`, `Ctrl+Shift+4` the other way) turns the canvas a quarter at a time.
- **Flip the view** (`Ctrl+5`) mirrors it left to right. This is the oldest check there is on a
  drawing: errors you have stopped seeing are obvious in the mirror.
- **Center the page** puts the canvas back under the middle of the pane, keeping the zoom you are
  already at. Use it when a pan has taken the drawing off screen and you do not want to lose the
  magnification you were working at — which is what **Fit view** (`Ctrl+0`) would do instead.

Quarter turns rather than a free angle, deliberately: at a quarter turn every overlay — the grid, the
marching ants, the symmetry lines, the transform box — stays exactly as accurate as it was, and a
free angle would put all of them slightly wrong. While either is on, a small button beside them says
so and sets the view back upright, because a mirrored canvas you have forgotten about quietly
teaches the wrong hand.

Do not confuse these with **Flip H**, **Flip V** and **Rotate** in the document panel's *canvas*
section. Those move pixels: they are edits, they are one undo step each, and they change what a save
writes. These three change nothing at all, which is also why they keep working while a save is in
flight — an editor that would not let you *look* at your drawing while it writes a file would be a
strange one.

## Zooming

The wheel zooms in steps of 5%, and it rounds to that step first: come out of a **Fit view** at some
awkward 83% and the first notch takes you to 85, not to 88. That is what makes 100% a place you can
reach from either direction rather than a number you have to type. The zoom stops at 25% and at
1000% — far enough out to see a large page whole, far enough in to place single pixels, and no
further in either direction, because past those the canvas is either unreadable or unusable.

One consequence worth knowing: an image too large to fit at 25% is centred at 25% and runs off the
edges of the pane rather than shrinking to meet it. Every size this app makes fits comfortably; it
takes a hand-opened file to run into.

## Tiled mode

Beside those is **Tiled**, with four positions: off, X, Y and X+Y. It is the one control on that
row that changes what a stroke *does*, and that is the point — a canvas showing its neighbours while
the brush went on stopping at the edge would be a picture of a seamless tile you cannot paint.

With it on, the canvas draws the neighbouring tiles around the one you are working on — three across
for X, three down for Y, the full nine for X+Y — so all four seams are visible at once and the
middle one is still the document. Everything that lays down colour wraps with it: the brush and the
eraser, the spray, the fill (a region that runs off one edge and continues on the other is one
region, so a tile's background is a single click), the shape tools, and the magic wand.

Three things deliberately do not. **Smudge** falls back to stopping at the edge, because its pickup
trails the brush and "the pixels it just passed over" has no answer when the brush is in two places
at once. **Blur** wraps, but each piece blurs its own side of the seam rather than reading across
it. And a **gradient** never wraps: a ramp has two ends, and wrapping one puts the last colour
against the first — a hard edge exactly where tiled mode exists to remove one.

Pasting and the floating selection stay on the middle tile: a pasted chunk hanging over the edge is
cropped there rather than appearing on the far side. What floats is a rectangle you are positioning,
not colour being laid down, so it lands where the transform box says it will.

Drag across a seam and the stroke carries on in a straight line; it is the *document* that wraps,
not the cursor. Selections, the grid, the symmetry guides and the marching ants are drawn on the
middle tile only. Nothing about this is saved: it is how you are looking at the file this afternoon,
not a property of the picture.

## Tools

The toolbox is an icon grid; hovering a tool shows its name and its letter. Every tool is listed in
[Keyboard shortcuts](14-shortcuts.md).

| Tool | Key | What it does |
| --- | --- | --- |
| Brush | `B` | A soft round brush. |
| Spray | `A` | An airbrush: scattered dabs for as long as you hold the button. |
| Eraser | `E` | The same brush, cutting alpha instead of adding colour. |
| Fill | `G` | Flood-fills from where you click. |
| Gradient | `U` | Drag to lay a gradient. |
| Blur | `R` | Softens what you drag over. |
| Smudge | `N` | Pushes pixels along the drag. |
| Shading | `H` | Moves what you drag over one swatch along a palette ramp. |
| Line | `P` | A straight line. |
| Curve | `F` | A smooth curve through the points you click. |
| Rect | `K` | A rectangle, outlined or filled. |
| Ellipse | `J` | An ellipse, outlined or filled. |
| Polyline | `L` | A chain of straight segments, one click per corner. |
| Polygon | `O` | The same, closed, and fillable. |
| Marquee | `M` | Rectangular selection. |
| Ellipse select | `S` | Elliptical selection. |
| Lasso | `Q` | Freehand selection. |
| Poly lasso | `D` | Selection from clicked corners, one click per vertex. |
| Wand | `W` | Selects a region of similar colour. |
| Move | `V` | Moves the floating selection, or the whole layer if there is none. |
| Pick | `I` | Samples a colour from the canvas. |
| Text | `T` | Stamps typed text onto the canvas as pixels. |
| Slice | `C` | Names a rectangle on the canvas. |

Options appear for the selected tool only, rather than as one long form — a brush's hardness means
nothing while the wand is active.

The painting tools have **Size**, a **Nib**, **Opacity** and **Spacing**; blur and smudge add
**Strength**.

The nib is soft or one of two pixel nibs. **Soft** is the antialiased disc, and it has a
**Hardness** slider shaping its falloff — that is what a painted reference wants, and it is what
every stroke this editor drew before the other two existed. **Pixel (round)** and **pixel
(square)** lay down whole pixels only: no partial coverage anywhere, so an edge stays an edge and
a drawing's colour count does not grow along every line. They have no hardness, because coverage
that is only ever fully on or fully off has no falloff to shape. With a pixel nib you also get
**Pixel perfect**, which drops the doubled corner pixel a freehand diagonal leaves at every step,
so the line comes out one pixel wide the whole way.

The brush alone has an **ink**: **Blend** composites the colour over what is already there, which is
what every stroke this editor drew before the option existed. **Replace** writes the colour exactly
— alpha included — so it can paint transparency back *down* as well as up, which is what recolouring
flat pixel art wants and what a normal brush cannot do at all. A soft nib still feathers either way:
in this app feathering means one thing everywhere, so opacity, a soft rim and a feathered selection
all soften a replace stroke exactly as they soften a paint one. (Aseprite's copy ink has a hard edge
instead; with a pixel nib the two agree exactly, because coverage that is only ever 0 or 1 has
nothing to feather.)

The spray has a **Rate** — dabs a second, for as long as the button is held, whether or not the
cursor is moving. Its **Size** is the width of the *cloud* rather than of one dab, which is what the
brush ring shows and what "spray width" means elsewhere; the dabs themselves are a quarter of it, so
a wide spray builds up slowly and a narrow one quickly. Spacing, smoothing and pixel-perfect are
hidden for it on purpose: all three are about a line, and a spray does not draw one.

**Shading** does not paint a colour at all. It moves every pixel you drag over **one swatch along a
ramp**, so what it writes is decided by what is already there — which is how shading is actually
done in pixel art, and it is the difference between deepening a shadow you already drew and
covering it over with a colour you picked out of a menu.

The ramp is the **selection in the Colour panel**, walked in palette order: Ctrl+click a few
swatches or Shift+click a run, and those are the colours the tool steps between. They do not have to
be next to each other in the table — slots 2, 5 and 9 are three adjacent steps of a three-colour
ramp — which is what lets you pull one character's ramp out of a palette holding several. With
fewer than two swatches selected the whole palette is the ramp, which is the right answer for a
table that *is* one ramp. Nothing about the ramp is saved in the file: the document keeps its
palette, and which run of it you were shading along is a property of what you were doing.

**Direction** is the tool's one setting. **Forward** moves each pixel toward the end of the ramp,
**Back** toward its start, and either way a pixel already at the end stays there — the ramp does not
wrap round, because sending the deepest shadow to the brightest highlight in one dab reads as damage
rather than as a tool.

Two rules make it controllable. **One step per stroke**: scrubbing back and forth over a shoulder
moves it one swatch and stops, and you take a second step by lifting the button and dragging again.
And **only ramp colours move** — a pixel painted in a colour that is not one of the selected
swatches is left exactly as it is, as is anything transparent, so shading a face does not disturb
the outline around it. Alpha is never touched, which also means "preserve transparency" on the layer
changes nothing about what a shade stroke does.

Shading needs an indexed document: with no palette, or with only one colour in it, the tool is
greyed out in the toolbox and says why. [Indexed colour](#indexed-colour) is how to give a drawing
one. There is no **Opacity** for it either, for the same reason there is no hardness on a pixel
nib — a shift lands on the next swatch exactly or it does not happen, and there is no partial
version of it to scale.

Three of the shape tools are **clicked rather than dragged**, the way the poly lasso is. The
**polyline** (`L`) drops a corner per click and joins them with straight segments; the **polygon**
(`O`) does the same and closes the shape; the **curve** (`F`) runs a smooth curve **through every
point you click** — not near them, through them, so a curve is placed by putting its points where
you want the line to go. All three finish on a double-click or `Enter`, and the polygon also closes
if you click back on its first point. `Esc` abandons the shape, and so does switching tools or tabs;
nothing is painted until it lands, and when it does it is one undo step however many clicks it took.
While you are drawing, the line from the last point follows the cursor — for the curve that preview
is the curve itself, resampled as you move, so what is on screen is what will be painted.

The shape tools have the size slider and, except for the open ones — the line, the polyline and the
curve, which have no inside to fill — a **Filled**
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

**Dither** on the gradient throws the blend away. Instead of mixing between two stops it gives every
pixel one of them, chosen against an ordered 2×2, 4×4 or 8×8 matrix — so the ramp lands on exactly
the colours you chose and nothing in between, which is what makes it usable on an indexed document
and on pixel art generally. It is off by default, because on a painted reference it is noise. A
selection's soft edge is *not* dithered: feathering means one thing across every tool here, and a
soft edge chopped into a chequer is not that thing.

The painting tools also have **Smoothing** and **Taper**. Smoothing makes the brush follow the
cursor at a distance instead of exactly, which turns a shaky line into a smooth one; it catches up
when you stop moving, so a stroke still ends where you left it. Taper thins a fast stroke, for a
pen-like flick. Both are off by default, and with both off a stroke is the same stroke this app has
always drawn.

Two canvas-wide aids sit below the tool options. **Symmetry** mirrors every stroke — off,
left/right, top/bottom, both, or **radial**, which repeats it around a circle a set number of ways
(2 to 32) for snowflakes and mandalas. With any symmetry on you can set the **axis** the mirrors
reflect about, in image coordinates; **Centre** puts it back, and "centred" means exactly that even
after the canvas is resized. **Grid** overlays a grid at a spacing you set, from 2 to 512 pixels —
32 by default, the most common sprite and tile cell — and **Snap to grid** lands shapes, lines and
the marquee on its intersections. Freehand strokes never snap: quantising a brush to a lattice is a
different tool, not a drawing aid. **Rulers** draws pixel rulers along the canvas's top and left
edges, with a marker shadowing the cursor on each; tick labels follow the decimal 1/2/5 ladder, so
the numbers you read are always round ones. The grid and the rulers remember how you left them
across sessions.

Two modifiers apply while you drag a line, a rectangle or an ellipse. **Shift** constrains it — a
square, a circle, or a line at a multiple of 45° — and **Alt** grows it from the point you pressed
rather than from a corner. Both can be held at once, and both change the preview as you hold them,
so you can decide halfway through. They belong to the shape tools alone: on the four selection
tools Shift and Alt already mean add and subtract. **Alt** over a painting tool picks the colour
under the cursor, which saves a trip to the eyedropper.

**The right button paints with the background colour.** It drives the brush, the eraser and the fill
— the three where "the other colour" is unambiguous — and Alt with it picks *into* the background,
so the button means one thing in both directions. On every other tool the right button does nothing
at all, deliberately: the selection tools may want it later, and a button left free can still be
given a meaning where a wrong one cannot be taken back. Middle-drag still pans.

**Arrow keys nudge by a pixel**, with Shift for eight. They move the floating selection if there is
one, and otherwise the whole layer while the Move tool is in your hand — each press is one undo
step. Nudging is gated on the Move tool rather than global because quietly translating a layer
because somebody pressed Right with the brush selected is not a trade worth making.

### Image brushes

Select part of your drawing and press **Capture from selection** in the brush's options: what you
selected becomes the brush tip, and the brush picks it up straight away. It reads the active layer
and cuts nothing, so what you captured from is untouched, and a lasso or feathered selection makes
a tip of that shape rather than of its bounding box.

The tip is not a tool of its own — it replaces the *tip* of the tool in your hand, so everything
that tool already does comes with it: symmetry mirrors where the stamps land, the spray scatters
them, tiled mode wraps them at the seam, the selection clips them and a layer's transparency lock
holds against them. It works on the brush, the eraser and the spray; the eraser with a tip loaded
cuts a hole exactly the shape of the picture.

**Rotate**, **Flip H** and **Flip V** give the variants. They cycle — four presses of Rotate is
where you started — and each is taken from the capture rather than from the last variant, so a
rotate and a flip give the same tip whichever order you press them in. Nothing is resampled: a
variant is the captured pixels reindexed, to the byte. **Forget** drops the tip and puts the round
brush back everywhere.

**Placing** decides where a dab lands. **Free** puts the picture under the cursor, which is what a
brush does. **Aligned to a grid** snaps every dab to a lattice of the tip's own size anchored on the
canvas, so neighbouring stamps line up into a pattern instead of meeting wherever the mouse happened
to be — and going over the same square twice changes nothing at all. The lattice belongs to the
canvas rather than to the stroke, so a pattern laid down in two sittings still tiles.

**A stroke never builds up on itself, either way.** Dragging a half-transparent tip slowly over one
spot leaves exactly what a single dab leaves, and a stroke drawn on a slow machine is the same
picture as the same stroke on a fast one. Lift the button and stamp again to put one picture over
another — that is what makes a second pass a second pass. The whole drag is one undo step, however
many stamps it laid.

Two things a tip does not do. It is not scaled by the **Size** slider and has no **Hardness**:
those shape a generated falloff, and a captured picture has none, so scaling it per dab would mean
resampling your own pixels several times a frame. And the brush's **ink** control disappears while
a tip is loaded, because a captured tip's transparency *is* its shape and there is nothing left for
a copy ink to write.

### Tool presets

Type a name beside **Save** in the presets section and the options of the tool in your hand are
stored under it; clicking the name later picks that tool back up with those settings, and the `x`
beside it removes it. Presets are remembered between sessions.

A preset is one tool's options and nothing else. The colours, the symmetry, the grid and the onion
skin are not in it — they belong to the canvas or to the sitting rather than to a tool, and a preset
that dragged them along would turn "my inking pen" into "my inking pen, and also switch the grid
off". The captured image tip is not in one either: it is pixels rather than a setting, and it is
captured again from the drawing, which is where it came from. What *is* stored is the tick that says
to use one.

## Text

The `T` **Text** tool puts a word on the canvas. Click where you want it and a box opens: type the
text (Enter starts a new line), choose a **font** and a **size** in pixels, and decide whether it is
**antialiased**. It is stamped in the foreground colour.

**There are no text objects and no text layers.** What you get is pixels — a floating selection, the
same thing a paste gives you — so the Move tool is in your hand when the box closes, you drag the
word where you want it, and it lands on the active layer as one undo step. Every tool and every
filter then applies to it, with nothing to flatten first. The other side of that trade is that
**re-editing text is retyping it**: the box remembers what you last typed, so a second stamp at a
different size is a size change and an OK.

The font list is every face in `C:\Windows\Fonts`, with the one that ships with Warlock (Inter) at
the top and chosen by default — so the tool behaves the same on a machine with no fonts installed.
The list is read the first time you open the box; a font installed while Warlock is running shows up
after a restart.

**Antialias** off renders whole pixels only, with no partial coverage anywhere — which is what pixel
art wants, and what stops a stamp adding colours to an indexed drawing. On an indexed document it
starts off for that reason, and on every other document it starts on. That is a starting point
rather than a rule: set the box yourself once and it stays where you put it, on every document you
open afterwards, until you press **Reset text** in the tools panel.

If there is nothing to render — an empty box, or a font file the system lists but cannot be read —
the stamp is refused with a message rather than putting an empty buffer on your canvas. A locked
layer refuses the click before the box opens.

## Colour

Two colours, not one: **Foreground** and **Background**. The gradient tool needs both ends, and `X`
swaps them — universal muscle memory from every other raster editor. Both carry an alpha bar, so a
semi-transparent brush is a colour rather than a separate mode.

Both swatches show their hex value inline and open a full picker — hue bar, HSV, hex, alpha — when
clicked, so a colour somebody sent you as `#3b4252` can be typed straight in.

Below them is a row of **swatches**. Clicking one makes it the foreground; the row is saved with
your settings rather than reset each session, because a project has a palette and retyping it every
time is the kind of small friction that makes a tool feel unfinished.

**Import palette** and **Export palette** move that row in and out as a GIMP `.gpl` — the format
GIMP, Krita, Aseprite and Inkscape all read — or a JASC `.pal`, which Paint Shop Pro, GraphicsGale
and most pixel-art palette sites write. The format follows the suffix you save under, and an import
reads whichever it was handed. Two things about both: neither has an alpha channel, so exported
swatches are written opaque, and an import **adds** to the row rather than replacing it — unwanted
colours are a right-click each, where a palette silently wiped has no way back.

The `I` **Pick** tool samples a colour from the canvas into the foreground.

### Indexed colour

Under the swatch row is the **palette** section, and it is a different thing from the swatches above
it. A swatch is a colour you keep reaching for this session; a palette slot is a colour *this file
is made of*. **Index to the swatches**, **Index to a palette file...**, **Palette from an image...**
or **Convert...** turns the mode on, and from then on every write — a stroke, a fill, a shape, a
gradient, a filter, a paste — lands on the nearest colour in the table. Alpha is never snapped, so a
soft brush still fades; it just bands, which is what the mode is for.

The pixels stay full-colour RGBA underneath. "Indexed" here means the writes are constrained, not
that the document stores palette indices — so nothing about layers, blending or export changes shape,
and turning the mode off leaves every pixel exactly where it is.

### Converting a drawing onto a palette

**Convert...** is the entry point when the drawing came from somewhere else — a photo, a render, a
sketch with a thousand near-identical greys in it. It builds a palette out of the drawing's own
colours (a **Colours** slider decides how many, 2 to 64) and shows you the result before you commit
to anything. On a document that is already indexed the button reads **Re-convert...** and keeps the
table you have, so what you are choosing is only how the pixels reach it.

That choice is the **Dither**:

- **nearest** puts every pixel on the closest swatch. Clean, and it bands — a smooth sky becomes
  three stripes.
- **floyd-steinberg** scatters each pixel's error into its neighbours, which keeps the average tone
  a nearest conversion loses. Best on photographic input; on a sprite at 4× zoom the scattered noise
  reads as dirt.
- **bayer2**, **bayer4** and **bayer8** threshold against an ordered matrix of that size. The noise
  is *regular*, which at pixel-art sizes reads as texture, and it is what every palette-first editor
  since Deluxe Paint has offered. A smaller matrix is a coarser, more visible weave.

The preview covers the frame you are on, because converting forty frames to show you one is a wait
for nothing. **Apply** converts the whole document — every layer and every frame — as one undo step,
and clicking away from the popup cancels rather than applies: a preview you did not answer is not a
yes.

A conversion **ignores any selection**, and that is deliberate. Indexing is a change of mode rather
than a write: the table it installs constrains every write afterwards, everywhere, so converting
only the marquee would leave the pixels outside it off the palette they are now declared to be on.
Aseprite does the same.

**Palette from an image...** is the other half of the same idea: point it at any image and its
colours become this document's table. An image with more than 256 distinct colours is reduced to 256
rather than refused — a photograph would fail every time otherwise — and a toast tells you what it
came down from.

### Editing the table

Click a slot to select it and load it into the foreground. **Ctrl-click** adds a slot to the
selection or takes it out again, and **Shift-click** selects the range from the last plain click to
where you clicked. The outlined slot is the *anchor*: every single-slot control below acts on it,
and **Sort** and **Insert** act on the whole selection.

With a slot selected:

- the **Slot** picker edits it — which repaints every pixel painted in that colour, across every
  layer and every frame, as **one** undo step;
- **+ from colour** adds the current foreground as a new slot, repainting nothing;
- **Remove** drops a slot and merges its pixels into the nearest surviving colour, because the
  pixels are the picture and a palette edit is a statement about the table;
- **&lt;** and **&gt;** reorder, which changes no pixel — the order is what an exported `.gpl` and an
  exported GIF colour table carry;
- **Count usage** walks the document once and reports how many pixels sit on each slot, so a slot
  showing zero is one you can delete without losing anything. It is a button rather than a live
  figure because counting is a pass over every pixel of every frame.

**Sort** reorders the table by hue, saturation, brightness, red, green, blue, alpha or usage, and
**Down** reverses the direction. With slots selected it sorts *those slots in place* — they keep the
positions they occupy and only which colour sits in each of them changes — so you can straighten out
one ramp in the middle of a hand-arranged table without the rest of it moving. Sorting by usage uses
the last **Count usage** figures, and takes them itself if you have not asked for any.

**Insert** fills the gap between two selected slots with an interpolated run, as many colours as the
slider beside it says. Colours already in the table are skipped rather than added twice. Select the
two ends with Ctrl-click or Shift-click first.

Neither sorting nor inserting is an undo step, and neither changes a pixel. Order is presentation in
an indexed document — it is what an exported palette and an exported GIF colour table carry — and a
new swatch is a colour you *may* paint with rather than a claim about what is already on the canvas.
Editing or removing a slot does repaint, and those are one Ctrl+Z each, table and pixels together.

The table is saved inside the `.ora` as a `palette.gpl` member, so it comes back when the file does;
an editor that does not know about it opens the file as an ordinary image, which is exactly what the
pixels already are. **Export palette** writes it out as a GIMP `.gpl` or a JASC `.pal`, whichever
suffix you give the file; both are plain text and neither has an alpha channel, so exported swatches
are opaque. **Export animated GIF** on an indexed document writes your table verbatim instead of
quantising each frame, so slot *n* is the same colour in every frame of the clip.

## Layers

The layers panel shows the stack top-first, the way every editor shows it, and is laid out the way
Photoshop and Krita lay theirs out. At the top are the **Blend** mode, the **Opacity** slider and
the two lock toggles, and they always describe the *active* layer. Each row in the list is an eye
(visibility), a thumbnail and the layer's name — hovering a row shows its blend, opacity and locks,
and a locked layer wears a small padlock beside its name. Under the list is the action strip:
**add**, **duplicate**, **group**, **merge down**, **flatten** and **delete**, as icon buttons whose
names are in their tooltips. Dragging the opacity slider previews live but records a single undo
step when you let go, rather than one step per pixel of drag.

There are nineteen blend modes, listed in the order every editor groups them — darkening, then
lightening, then contrast, then comparison, then the arithmetic and colour ones:

| | |
|---|---|
| `normal` | the layer, over what is under it |
| `darken`, `multiply`, `color-burn` | can only darken the backdrop |
| `lighten`, `screen`, `color-dodge`, `add` | can only lighten it |
| `overlay`, `hard-light`, `soft-light` | darken the dark half and lighten the light half |
| `difference`, `exclusion` | the distance between the two colours; exclusion is the softer of them |
| `subtract`, `divide` | arithmetic, clamped at black and at white rather than wrapping |
| `hue`, `saturation`, `color`, `luminosity` | take one attribute from the layer and the rest from underneath |

These are the W3C formulas, which is what OpenRaster's composite operators are defined against — so
a document saved here and reopened in Krita or GIMP composites identically rather than approximately.
The last four are the *non-separable* ones: `color` paints over a drawing without changing how light
or dark it is, and `luminosity` is that trade the other way round. `subtract` and `divide` are the
two the W3C set has no name for, so they go into the file under Krita's names — a mode that arrives
from another editor and has no equivalent here still opens, with that layer set to `normal`.

**Lock alpha** paints inside what is already on a layer and never past its edge: colours change,
transparency does not. It is how you recolour lineart or shade a shape without selecting it first —
and it makes the eraser a no-op on that layer, because erasing *is* changing transparency. The lock
is saved with the document; other editors ignore it and open the layer as an ordinary one.

**Lock layer** is the stronger one. A locked layer refuses every tool: no strokes, fills,
gradients, filters, lifts or pastes land on it, and the canvas says so once per press rather than
swallowing the click. What it deliberately does *not* stop is managing the layer — renaming,
hiding, reordering and deleting all still work — or anything that is a statement about the whole
document, so a rotate, a crop, a canvas resize and Flatten apply regardless. Undo also works
underneath the lock: locking a layer after an edit does not wedge the history that already holds
it. Like Lock alpha it is saved with the document and ignored by other editors.

**Merge down** and **Flatten** work on an animated document too, across every frame at once. A
merge is worked out per pair of cels, so two frames that shared a drawing go on sharing the merged
one rather than each getting a copy — the links you have survive the merge. Flatten does the same
for whole frames: frames whose layers were all the same drawings come back sharing one flattened
cel. Frame durations and tags are untouched either way.

### Groups

**Group** wraps the active layer in a folder. Folders are for organising a stack that has grown
past what you can read at a glance, and they fold three things down onto everything inside them:
visibility (hidden folder, hidden contents), opacity (a folder at 50% holding a layer at 50% shows
it at 25%) and the layer lock (locking a folder locks its layers).

Drag a layer onto a folder's header to move it in, and use the layer's right-click menu to take it
back out. A folder's own menu offers **Ungroup**, **Rename** and **Lock**; the button beside it
folds it shut, which is a view setting and is neither saved nor undoable.

Two rules are worth knowing because they are visible. **There are no empty folders** — a folder is
made *around* layers that are already next to each other, and the last layer leaving one dissolves
it. And a folder's layers stay together in the stack: grouping layers that are not adjacent is
refused rather than silently reordering your drawing, and moving a layer into a folder moves it in
the stack too, as one undo step.

Folders composite **pass-through**: each layer inside still blends with everything beneath it,
exactly as it did before you grouped them, with the folder's opacity and visibility folded in. What
that means in practice is that a folder has no blend mode of its own — *isolated* group
compositing, where the folder is rendered to its own buffer and blended once, is not implemented in
this build.

Folders are saved as OpenRaster's own nested stacks, so one made here opens as a group layer in
Krita, and a Krita file's groups open as folders here.

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

**Resample** says how a scale decides what each new pixel holds, and it applies to the free
transform's scale and rotate as well. **Smooth** filters, which is right for a photograph or a
generated reference. **Nearest** copies each source pixel whole, which is the only correct answer
for a drawing whose pixels *are* the artwork — a filtered scale of a 32×32 sprite comes back
blurred and with thousands of colours in it that were never drawn.

**RotSprite** is the third, and it is about *rotation*. Nearest neighbour is right for scaling
pixel art and wrong for turning it: a hard-edged diagonal turned by copying whole pixels comes out
as a staircase with a different tread on every step. RotSprite upscales eight times with an
edge-preserving filter, turns that with nearest neighbour, and samples the middle of each block
back down — so the result is still made only of colours you drew, but the staircase is decided on a
finer lattice. Nothing is interpolated at any stage. It only affects turning, so a scale asked for
it behaves as Nearest, and above roughly 512×512 pixels a rotate falls back to Nearest and says so,
because the eight-times upscale costs sixty-four times the memory on every frame of a drag.

The 3×3 **anchor** grid says where the old image sits in the new canvas, and it belongs to Resize
canvas only: scaling has no slack to put anywhere. Growing a canvas anchored centre adds room on
all four sides; anchored top-left it adds room right and below, which is what the button did before
there was a grid. Shrinking works the same way and crops from the opposite sides.

## Filters

**Filter…** in the document panel opens nine whole-layer adjustments: brightness/contrast,
hue/saturation, levels, blur, sharpen, invert, replace colour, outline and despeckle. Every one
previews live on the canvas as you drag, and the whole session — however many sliders you moved and
however many times — records as a single undo step when you press Apply. Cancel, or clicking away
from the popup, puts the pixels back and records nothing at all.

The last four are the pixel-art staples. **Invert** has a checkbox per channel, all three on when
the popup opens. **Replace colour** takes a From and a To colour — the **use FG** button beside
each fills in the colour you are painting with — and a tolerance, which is a distance in colour
space rather than a per-channel slack; at zero it replaces exactly the colour you named, which is
the same set of pixels that editing a palette slot would recolour. **Outline** draws a ring along
the edge of whatever is painted: choose its colour and thickness, whether it goes outside or inside
the shape, whether corners are rounded (4) or square (8), and **wrap** if the drawing is a tile, so
the outline carries across the seam instead of stopping at the border. **Despeckle** is a median
filter — it deletes stray pixels outright and leaves hard lines hard, which is what a blur cannot
do; its **speck** slider is how big a stray thing it removes, and it stops at 4 because past that a
median takes out detail rather than specks — and takes long enough doing it to stall the preview.

Three things about what they do to a layer. They apply to the **selection** if there is one, faded
by a feathered edge exactly as a brush would be, and to the whole layer if there is not. The colour
filters never change transparency; blur and despeckle do, because softening a layer's edge is most
of the reason to blur one, and an outline placed *outside* does, because it draws where the drawing
is not. And a layer with **Lock alpha** on keeps its transparency under all of them.

Every filter opens at settings that change nothing — except Invert, whose three channel boxes open
ticked, because inverting no channels is not a thing anybody wants. So the preview is safe to start
immediately, and the picture only moves once you move a slider.

## Selections and transform

Five tools make selections: the rectangular marquee, the ellipse, the lasso, the poly lasso and the
wand. Hold **Shift** while dragging to add to the current selection, **Alt** to subtract from it,
and both together to keep only the overlap.

The **poly lasso** (`D`) is clicked rather than dragged: each click drops a corner, a line follows
the cursor from the last one, and the shape closes when you double-click, press `Enter`, or click
back on the first corner — the small ring drawn around it is how near you have to land. Three
corners is the minimum; fewer than that closes nothing and leaves the selection you already had.
`Esc` abandons the polygon, and so does switching tools or tabs. The modifier is read at the
**first** click and held for the whole shape, so letting go of Shift halfway through does not turn
an add into a replace. With **Snap to grid** on, every corner lands on an intersection. However many
clicks it took, the finished selection is one undo step.

With a selection live, the **selection** section offers **All**, **None**, **Invert**, a
**Feather** radius slider up to 32 pixels with a **Feather** button, and **Crop to selection**. The
same actions have keyboard shortcuts: `Ctrl+A`, `Ctrl+D` and `Ctrl+Shift+I`.

**Reselect** (`Ctrl+Shift+D`) brings back the selection you last dismissed, which is the other half
of `Ctrl+D` and saves redrawing a lasso you only meant to step outside of for a moment. A selection
from before a resize or a crop cannot come back: it describes a canvas that no longer exists, and
placing it somewhere would be a guess.

Dragging *inside* an existing selection with a selection tool and no modifier moves its **edges**
rather than replacing it — the marching ants follow the cursor and the pixels underneath do not
move at all. That is the difference between this and the Move tool, which moves the pixels. Shift
and Alt still start the add and subtract drags they always did, even when the drag starts inside
the selection.

**Layer from selection** promotes the selection onto a layer of its own, lined up with what it came
from. `Ctrl+J` copies it and leaves the original where it was; `Ctrl+Shift+J` moves it, cutting it
out of the layer it was on. Either way it is one undo step, a feathered selection makes a feathered
layer rather than a hard-edged crop of one, and the new layer joins whatever folder the one it came
from is in.

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

The Move tool has a third answer, and it is the one it used to have nothing for: with no floating
buffer and nothing selected, dragging moves **the whole active layer**. Pixels pushed past the edge
are cropped rather than wrapped round — a layer's edge is the canvas edge — and the whole drag is
one undo step, measured from where you pressed, so a slow drag and a fast one land in the same
place. On an animated document a *linked* cel moves on every frame it appears in, because a link
means one drawing shared rather than two copies that happen to match.

Cancelling a lift — where the buffer was cut out of a layer — puts the pixels back and removes that
step from history entirely, rather than leaving it on the redo stack where `Ctrl+Y` could replay the
cut with no buffer left to restore.

**Free transform** (`Ctrl+T`, or the button in the tool options) rotates, scales and slants the
selection, or the whole layer when there is no selection. It is modal: while transforming, **Enter**
applies and **Esc** cancels, and nothing else can change the tool out from under a half-finished
transform.

The box has eight grab points. The four corners scale both axes at once and the four edge midpoints
scale one, so a sprite can be made taller without being made wider; hold **Shift** on any of them
to keep the scale uniform. The arm above the box rotates, and Shift snaps that to 15°. The row
above the canvas carries the same numbers — **Angle**, **Scale X** and **Scale Y** with a **Link**
toggle — because a drag cannot express "exactly 90 degrees" and a rotation that is nearly square is
worse than either.

The tool options panel adds typed **X**, **Y**, **W**, **H**, **Angle** and **Slant** fields while
a transform is running. Slant is an italic: two numbers in degrees, horizontal then vertical. It is
applied after the scale and before the rotation, so the two slant axes are always the page's, and
in this build it has numeric fields only — there are no slant handles on the box. Two large slants
the same way fight each other — at 45° each they would squash the picture onto a line — so a pair
that extreme comes back unslanted rather than as a sliver.

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

**Linked cels are one drawing in several frames.** *Link* adds a frame that shares the current
one's cels rather than copying them, so a background held across twenty frames is stored — and
edited — once. Painting on any of them paints on all of them, which is the point. **Unlink** gives
that one frame a private copy from then on. *Copy* is the other choice: an independent duplicate
you can diverge immediately.

The strip's controls are two rows. The top one is the frame you are on — the transport, the frame
operations, the counter and that frame's duration — and the bottom one is what leaves the app: the
three exports, with the onion-skin and thumbnail switches and the export magnification beside them.
Both rows give up their labels for their icons before they give up any button, and drop what they
can into a **…** menu after that, so nothing is ever pushed off the right-hand edge.

**Durations are per frame**, in milliseconds, in the box on the transport row — so a held pose and
a fast blink live in the same clip without anything having to be a frame rate.

**Onion skin** shows the neighbouring frames beneath the one you are drawing, the previous tinted
red and the next tinted green. Toggle it on the strip's second row; while it is on, **back**,
**ahead** and **fade** appear on a row of their own and set how many frames either side are drawn
and how strongly. Both counts may be zero,
which is how you see only what is behind or only what is ahead.

**Playback** is the Play button or `Enter`; `Esc` or Play again stops it, leaving the playhead where
you last saw it. While playing, the document is read-only — the canvas is showing a cached picture
of another frame, so a stroke would land somewhere you cannot see. If a **tag** covers the current
frame, playback loops inside that tag rather than over the whole timeline.

**Tags** name a span of frames — "walk", "idle", "hit". Right-click a frame number and choose
**New tag here** to make a one-frame tag, then right-click the tag's name in the band under the grid
to rename it, to set either end to wherever the playhead is, to turn its looping off, to choose
which way it plays, or to delete it. Tags may overlap, and playback follows the innermost one
containing the playhead — which is what makes a short **hit** inside a long **combat** the useful
arrangement rather than an ambiguous one. Tags are saved with the document and written into a
sprite sheet's sidecar.

A tag plays **forward**, **reverse** or **ping-pong**, and that is a separate question from
whether it loops: direction is the path through the span, looping is whether reaching the end of
that path starts it again. A non-looping ping-pong swings out and back once. Ping-pong is the one
worth having — a torch flicker or an idle breath drawn as frames costs the whole span again in
cels, every one of them a duplicate of a drawing already in the file, and every edit to the middle
of the swing then has to be made twice.

A tag can also **repeat** a fixed number of times. Set **repeat** in the tag's menu to 3 and the
span plays three times and stops; leave it at 0 — the default, and what every document written
before this carries — and the Loop tick decides as it always has. While a count is set the Loop
tick is disabled, because the count is the more specific answer to the same question — a tag that
was set to play *once* and is then given a count of three plays three times, and clearing the count
back to 0 gives it its "once" back. When the
count runs out playback **stops inside the tag**, on the frame the span ends at; it does not fall
through into the frames after it, which is where Aseprite would carry on. A ping-pong counts one
out-and-back as one play. Repeats are saved with the document and written into a sprite sheet's
sidecar, and a tag exported on its own as a GIF carries its count into the file.

**Ranges.** Drag across the grid to select a rectangle of cells — press, drag, release; `Shift`+click
extends the selection from where you last pressed, and `Esc` clears it. The selection is drawn as
one accent outline and is not part of the document: it pushes no undo step and is not saved.
Right-clicking any cell opens the **Range** section of the cell menu, which acts on the whole
selection: copy and paste cels (the clipboard is shared between tabs, and a link inside the copied
block survives the paste as one drawing again), clear, link, unlink, duplicate the frames copied or
linked, reverse them, delete them, and set every frame's duration at once. Every one of them is a
single `Ctrl+Z`, and each is refused rather than half-applied — deleting *every* frame is the one
range delete that is not allowed, because a timeline keeps at least one frame.

**Cel thumbnails.** The **Thumbs** switch on the second row of the timeline draws each cel's picture
in its cell and grows the cells to fit. Linked cels share one thumbnail, so a link looks like the
same drawing standing in several columns.

**Exporting part of a clip.** The Range section ends with **Export range → sheet** and **→ GIF**,
which write only the selected frames. A tag's own menu has **Export tag → sheet** and **→ GIF**,
which use that tag's span and its looping — a tag with a repeat count writes a GIF that plays that
many times. In a partial export the tags are renumbered against the exported frames, tags that fall
entirely outside it are dropped, and a directional layout is not carried over: half a walk sheet is
a clip, not a smaller walk sheet.

**Filters over a range.** With a range selected, the filter popup gains **Apply to range**: the
filter runs over every cel in the selection as one undo step. A linked cel is filtered *once*
however many frames it appears on, an empty cel stays empty rather than becoming a filtered blank,
and the selection on the canvas is honoured as a weight exactly as it is for a single layer — so a
feathered selection fades the filter in on every frame at once.

**Merge down** and **flatten** are not unavailable here: both run across the whole grid, every frame
at once, and the links you have survive them — see [Layers](#layers).

**Import sprite sheet** in the document panel goes the other way: pick any image and give it a cell
size, an offset, the padding between cells and how many frames to take, and it becomes one frame per
cell, read row by row. The popup counts the frames your numbers produce as you type them, and says
what is wrong rather than importing a short sheet — the last column and the last row carry no
trailing padding, which is the arithmetic that otherwise quietly drops a frame. An imported sheet is
an ordinary animation with no directions and no tags: a *layout* is something the generator knows
about its own output, not something a cell size can imply.

A document opened from a generated sprite sheet carries a **directional layout** as well: its frames
are that sheet's cells in order, four directions with one or four frames each, and a walk sheet
arrives with a tag per direction so playback loops one direction at a time. The layout is saved with
the document and survives a round trip through `.ora`. You can draw on it, repaint it and retime it
like any other animation; adding or removing a frame is allowed and simply means the timeline no
longer fills the grid, which **Export sheet** then says rather than writing a sheet with a hole in
it. See [From a single drawing](06-sprite-sheets.md#from-a-single-drawing).

Moving the playhead is not an edit. It pushes no undo step and does not make the document unsaved —
looking at another frame is looking, not drawing.

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

## Slices

The `C` **Slice** tool names a rectangle on the canvas. A slice carries no pixels — it is a note
about the drawing that travels with it into an exported sprite sheet and into a Packwright atlas,
where a game engine reads it.

Drag on empty canvas to make one. Click a slice to select it, drag its middle to move it and drag a
corner to resize it; each gesture is one `Ctrl+Z`. The tools panel lists what the document has and
gives the selected one a name, two switches and a Delete.

- **Pivot** is the point an engine places the sprite by — the one that stays put as a character
  turns. Switch it on and it appears as a small crosshair you can drag. When a document has several
  slices, the first one with a pivot is the one an exported sheet uses.
- **Nine-slice** marks the stretchable middle of a panel: the four corners keep their own size and
  the edges repeat, which is how a UI frame scales to any size. It draws as a dashed rectangle
  inside the slice, with its own corner handles.

On an animated document a slice is the same rectangle on every frame until you say otherwise.
**Key this frame** gives the current frame its own rectangle, pivot and centre; every other frame
goes on using the slice's own. Keys are always explicit — dragging a slice moves it everywhere,
because a drag that silently keyed whichever frame you happened to be on is how a clip ends up with
forty slightly different rectangles nobody meant.

Slices survive everything the canvas does to them: a flip, a quarter turn, a scale, a crop and a
canvas resize all carry them — and their pivots and centres — along, and undoing puts them back. A
crop that misses a slice entirely leaves it as a single pixel rather than deleting it; its name and
its settings are not something a crop should be able to throw away.

They are saved in the `.ora`, in a member of their own. A file with no slices in it is written
exactly as it always was, and an `.ora` from any other program opens here with none.

**Show with other tools** keeps the overlay up while you paint; the slice tool turns it on for you.

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

## Saving

Inker saves natively as **OpenRaster** (`.ora`) — a zip of layer PNGs that both Krita and GIMP read
and write. That is the format that keeps your layers, their blend modes and their opacities.

- `Ctrl+S` saves. A document that has never been written anywhere asks where to put it first.
- `Ctrl+Shift+S` is Save As.
- A drawing you opened from a **JPG, WebP or BMP** also asks where to put it, every time. Inker can
  read those formats but cannot write them, so `Ctrl+S` offers you an `.ora` beside the original
  rather than either putting PNG bytes into a file named `.jpg` — unreadable by its own extension —
  or re-encoding your original to JPEG and losing pixels on a keystroke that means "keep what I
  have". The original file is never touched. `.png` and `.ora` save in place.
- `Ctrl+E` adds the drawing to the library as a finished reference, which is what plain `Ctrl+E`
  does in Clay, Plotter and Packwright too. It does nothing for a document you opened *from* the
  library — that one is already there, and `Ctrl+S` is the write it wants.
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
A document with a directional layout skips the wrapping entirely and uses that sheet's own fixed
grid, with each cell's direction and frame number in the fields a rendered sheet puts a pose and a
frame in.
Two things about it are worth knowing. A cel linked across three frames becomes three identical
cells, because the engine playing it back knows nothing about links. And the cells keep their
transparency rather than being flattened onto the document's matte, which is what a sheet wants
almost always — a matte is what a *flattened* export puts behind transparency, and an atlas is
composited over whatever is behind it in the game.

**Export PNGs** writes one numbered PNG per frame — `name_0000.png`, `name_0001.png` and so on,
beside whatever name you pick. No atlas to slice and no sidecar to read, which is what an engine
with its own importer wants.

The **scale** box on the timeline's second row magnifies every export by a whole number, nearest
neighbour: each pixel is drawn N times and nothing is resampled, so a 32×32 sprite at 8× is the
artwork at 256×256 rather than a blurred version of it. A sheet's sidecar is built on the scaled
size, so its cells and trims describe the file that is actually written.

**Export GIF** writes the same clip as an animated GIF, looping. That one is for showing the
animation to a person rather than to an engine: it plays anywhere, on its own, with nothing needed
to read a sidecar. The format costs it two things and both are worth knowing before you send one
out. A GIF pixel is either fully there or fully gone, so soft edges become hard ones at the
halfway mark — which is no loss at all with a pixel nib and is very visible with a soft brush. And
a GIF times its frames in hundredths of a second, so each duration is rounded to the nearest 10 ms;
a 15 ms frame becomes 20, and anything under 10 ms becomes 10.

## Autosave and recovery

Every open document with unsaved changes is copied to `~/.warlock/assets/autosave/` every two minutes. This is
crash safety and nothing else: an autosave is **not** a save. It does not mark the document saved,
it does not choose a location, and it does not touch a linked job — all it promises is that a crash
costs you minutes rather than an afternoon. Saving or closing a document removes its copy, because
an autosave that outlived its document is exactly the file that turns up later and confuses you.

If Warlock finds copies left over from a previous session, they are listed under **Unsaved work**
at the top of the Home screen — one row per document, each with its own **Recover** button, and
**Discard all** underneath. Recovering one opens it and takes you to the editor it belongs in;
the others stay listed until you deal with them. Recovered documents open **untitled and unsaved**,
deliberately: the file each was copied from may still be on disk with its own contents, and adopting
that path would arm `Ctrl+S` to overwrite something you have not looked at yet. Ignoring the list
keeps the copies — "not now" is not "delete my work" — and they are cleared once you save or close
whatever you recover, or when you press **Discard all**.

The list is read **once, at startup**, and does not refresh while you work. That is deliberate and
not a limitation: the autosave directory is also where *this* session writes its copies, so a list
that re-read it would start offering your own open documents back to you.

**This is no longer only Inker's.** The same two minutes now cover a Clay model, a Plotter map, a
Packwright atlas, a pose you are authoring and a profile draft you have typed into and not saved —
whatever was open, whichever modes they were in, each on its own row. Each copy is written in that
mode's own format (`.ora`, `.wblk`, `.wmap`, `.wpack`, and small JSON files for a pose and a
draft), so anything recovered can also just be opened by hand. A document of a kind this build has
no editor for is listed as **unavailable** rather than hidden, and its files are left alone.

A pose is the one that can decline to come back, and it says so when it does. A pose is a set of
rotations for a skeleton rather than a document of its own, so putting one back needs that rig
loaded — if the asset it was authored against is not open, or a different one is, Warlock keeps the
copy and tells you to open the right one. It would otherwise apply somebody else's rotations to
whatever bones happened to share a name.

Nothing is ever deleted on age. A copy sits there until you recover it and save, until you close
what you recovered, or until you press **Discard all**; leaving the list alone keeps everything
exactly where it was.

## Pipeline bridges

Inker is wired into the pipeline in both directions. The **document** panel on the right states
which direction you are in: a document is either **linked to a job** or **not part of a job**.

**Into Inker.** With a finished reference selected at the Reference stage, **Open in Inker** appears on the
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
- **Make 3D** queues the mesh stage from the flattened image. A linked document promotes the
  reference it already is — and refuses if you have unsaved changes, so the mesh is made from what
  you can see. An unlinked one becomes an ordinary image job, the same call the Mesh stage's upload
  button makes. Either way, if the quality report is unhappy you get a confirm naming the reasons
  rather than a refusal.

A painted reference is a real job row that never ran on the worker: the image already exists, so
queueing a run to reproduce what you just drew would be two minutes of GPU for nothing. It is
created finished, at the reference stage, which is exactly what promotion consumes. It cannot be
rerolled — there is no generator behind it for a new seed to change — but it can be remeshed. See
[Rerun and promotion](11-library-and-jobs.md#rerun-and-promotion).

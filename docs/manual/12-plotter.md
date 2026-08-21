# Plotter

Plotter is the top-level tile-map mode: a grid, a stack of layers, one or more tilesets, and the
objects an engine reads as spawn points and trigger volumes. It is where a sheet of tiles becomes a
level.

It exists because the rest of Warlock makes *assets* and stops there. A sprite sheet or a packed
atlas is a pile of pictures; a map is the thing a game actually loads. Plotter closes that gap
without leaving the app, and it speaks Tiled's file formats in both directions so nothing you build
here is trapped.

It is a mode, not a takeover. Switching away leaves every open map exactly where it was, and a
reconstruction started before you switched keeps running with its progress card floating over the
canvas. Only quitting the app and closing a tab can lose unsaved work, and both ask first.

The layout follows the rest of the app: tools and the tileset palette on the left, the map in the
middle, the layer stack and the file panel on the right. Several maps stay open at once.

## Starting a map

With nothing open, the middle column offers **New map...** and **Open a file...**, and lists the maps
you had open recently — clicking one reopens it, and hovering it shows the full path. The map panel
on the right offers the same two buttons. Once a map is open, `Ctrl+N` and `Ctrl+O` do the same two
things from the keyboard.

**New map** asks what the map is before making it. Five presets start you off — 16 px tiles, 32 px
tiles, a 2:1 isometric cell, staggered, or hexagonal — and each one fills in every field below,
which you can then change.
The fields are the projection, the map size in tiles, and the tile size in pixels; a line under them
says what that comes to in pixels overall, because a map is authored in tiles and exported in pixels
and the two numbers are easy to be surprised by. Choosing *Isometric* with a cell that is not twice
as wide as it is tall says so, and still lets you do it.

**Infinite** is the one other answer on that dialog. An infinite map has no fixed edge: painting past
the side you can see grows it, cells can sit at negative coordinates, and the size you typed above
becomes the window it starts with rather than a rectangle it is stuck inside. Leave it off for a
level with a known shape — most maps — and turn it on when you are laying out a world and do not yet
know how far it goes. It is not a decision you are stuck with either way; see *Resize* below.

Two of those answers are worth getting right at this point. The **projection** is fixed once anything
is painted, because a tileset drawn for one lattice paints the wrong shape into every cell drawn for
the other. The **tile size** is what a plain image is sliced at when you add it, so a tileset added
to a 32 px map is a 32 px tileset for good. Neither is a trap you can fall into silently any more,
which is what the dialog is for.

Last, **Then** picks what happens once the map exists: add a tileset from a file, or nothing yet.
A map cannot be painted until it has a tileset, so the dialog offers the door rather than leaving
you to find it. Tile sheets are made in **Create** — see *Sheets* — and reach a map from the library
like any other asset. A sheet records the view it was drawn for, and Plotter reads it: an isometric
sheet added to a square map you have already painted on asks first, because a map's lattice is fixed
once anything is on it. On an empty map the sheet simply brings its lattice with it. Top-down and
3/4 sheets are both square, so neither ever asks about the other.

Every new map arrives with one tile layer called *Ground* already on it. Both sizes can be changed
later, under **Resize** in the tools pane. The grid fields grow or crop the map and move every object
with the content, so a trigger volume drawn around a doorway stays around that doorway; the
**Offset** fields decide where the old content lands, which is how you add rows at the *top* rather
than the bottom. On an infinite map that section is called **Size** instead and has no width and
height to type: the rectangle is whatever you have painted, so what it offers is moving the content,
**Shrink to content** — which throws away the empty space an erase left behind — and **Give the map
a fixed size**, which crops to what is painted and asks first, because the cells outside that
rectangle are gone. A finite map's section offers **Make infinite**, which asks nothing: nothing is
lost going that way, and every cell stays exactly where it is. Either conversion is one undo step.
Below them, **Tile size** redraws every cell at a new size and scales the objects
with it — nothing painted is lost, because a tile keeps its identity whatever size the cell under it
is. Tilesets already on the map keep the slicing they arrived with, which is why the size you start
with still matters.

## Tilesets

A map paints nothing until it has a tileset. **Add from a file...** in the tileset pane takes either
a plain image, which is sliced into a grid at the map's own tile size, or a Tiled `.tsx`, which
carries its own slicing, spacing and margin. Dropping an image or a `.tsx` on the window does the
same thing.

You can also send a library asset straight in — right-click any reference card and choose **Add to
Plotter as a tileset**. If no map is open, the **New map** dialog appears first and the asset is
added to whatever you make there. A [Packwright](13-packwright.md) grid pack is the natural source:
it writes a `.tsx` beside its atlas precisely so it can be used here with no conversion.

A generated tilesheet — the kind an image model produces when you ask it for a tileset — usually
comes back as one image with dark lines ruled between its cells, and cells that are not quite the
same size. Add one and Plotter says so: **Detected a 6 × 6 tile grid with separator lines**, with
**Import** to strip the rules and redraw every cell at the map's tile size, and **Slice at 16 × 16
instead** to do what it would have done without the detector. It is always a question, never an
answer applied for you — dark art can rule itself off convincingly, and the second button is there
for exactly that. Nothing is added to the map until you pick one.

Tilesets are *added*, never removed. Every tile in the map is numbered from the tileset it belongs
to, so dropping one from the middle would either renumber everything above it — invalidating every
cell already painted — or leave a hole. Undo takes back a tileset you have only just added.

### What one tile carries

Pick a single tile and a **Tile** header appears under the palette. A tile can carry a class, custom
properties of the same typed kind everything else here does, a **probability** that weights it for a
random brush, an **animation** and a set of **collision shapes** — all of it Tiled's own model, read
from and written to `.tsx`, `.tmx`, `.tmj` and `.wmap`.

Probability 0 is worth stating on its own: such a tile is never chosen by a random brush and is
always placeable by hand. It is how a set marks a tile that belongs to the palette but not to the
scatter, and it is Tiled's rule rather than ours.

An animated tile **plays on the canvas** and is drawn as its first frame in every export, on the
minimap, and by anything that composites the map flat. That disagreement is deliberate: an export is
a still. The document's own cells never move while it plays — a clock that wrote tile numbers would
mark a saved map dirty sixty times a second.

Collision shapes are drawn as outlines and are **never** hit-tested against the map. They are
metadata an engine reads, exactly as an object layer is.

### Picking a tile

The palette under the tileset combo is the atlas itself. Click a tile to pick it; drag across
several and you get a multi-tile brush, which stamps as one block. The white outline shows what is
selected, and the cursor on the canvas shows the footprint the brush is about to cover.

### Polishing an atlas in Inker

**Polish in Inker** opens a tileset's atlas as an ordinary drawing — one flat layer, not sliced into
cells, because keeping an outline consistent *across* neighbouring cases is exactly what the pass is
for and 235 separate frames would hide it.

When you are done, **Back onto...** under *from Inker* returns it to the same tileset. Every painted
cell keeps its tile and simply redraws with the new art, because the numbering is untouched. An
atlas whose size changed is refused by name rather than accepted — the roles are positional, so a
cropped atlas is one whose tile 93 is no longer the tile the map thinks it is.

### Tilesets from Inker

Inker has tilesets and tilemap layers of its own, and they are the *same type* — no conversion
happens in either direction. **Use in Plotter** in Inker's Tiles panel hands the tileset it is
showing to the open map, and a `.tsx` exported there is a `.tsx` **Add from a file...** reads here.

Either way it is a **snapshot, not a link**. The map holds the tileset as it stood when it arrived;
painting on it back in Inker afterwards mints a new one there and leaves the map's copy exactly as
it was, and you send it across again to bring the changes over. That is deliberate: a live link
between two documents with two undo stacks is a synchronisation feature, and a snapshot is the
honest thing without it. [Tilemap layers](09-inker.md#tilemap-layers) is the Inker side of this.

## Isometric and oblique maps

A map is drawn on one of five lattices, and which one is a property of the map. On an **isometric**
map a cell is a 2:1 diamond, the grid follows the two lattice directions rather than the screen
axes, and the status line under the canvas shows the cell under the pointer — which is the only
thing that reliably answers "am I about to click the diamond I mean".

An **oblique** map is the square lattice sheared: each cell keeps its size and the grid leans by the
map's skew. It is Plotter's own projection rather than one of Tiled's, so an oblique map exported to
`.tmx` will not open in Tiled — see "What Plotter writes that Tiled does not read" below.

Everything else is unchanged. The same tools paint, the same layers stack, and the flat render an
export produces places a cell exactly where the canvas does.

Tiled measures an isometric object's position differently from the way it measures a cell, and the
conversion is applied in both directions, so a spawn point exported to `.tmx` opens in Tiled where
you left it.

## Tools

| Key | Tool | What it does |
| --- | --- | --- |
| `B` | Stamp | Puts the brush down, following the drag |
| `E` | Erase | Clears cells, following the drag; a terrain cell re-fits its neighbours |
| `F` | Fill | Floods the connected run under the cursor, or the terrain field with a terrain in hand |
| `T` | Terrain | Paints a terrain and re-fits the eight cells around it |
| `P` | Shape | Fills a rectangle or an ellipse between press and release |
| `R` | Select | Drags a rectangular selection; a plain click clears it |
| `W` | Wand | Selects the connected run of one tile; Ctrl+click selects every cell of it |
| `I` | Pick | Takes the tile under the cursor as the brush, or drag to capture a block |
| `S` | Objects | Selects and draws objects (see below) |

The letters are [Tiled](https://www.mapeditor.org/)'s, because that is the editor whose files this
one reads and writes, and so the one you are most likely arriving from.

**Pick** takes one tile with a click and a whole *block* with a drag — Tiled's capture. Drag a
rectangle across the map and what was in it becomes the brush, flags and all, and the tool switches
to Stamp so you can put it straight back down. A drag across empty map is refused rather than arming
a brush that erases everything it touches.

**Wand** selects by content instead of by rectangle: click and you get the connected run of the tile
you clicked, Ctrl+click and you get every cell of that tile anywhere on the map. Hold **Shift** to
add to what is already selected and **Alt** to subtract — on the wand and on the marquee both. A
selection that is no longer a rectangle still constrains every tool exactly as a rectangle does, and
a fill inside a concave one cannot escape around the outside and come back in.

**Fill** and **Shape** take the whole brush, not just its first tile. A multi-tile brush lays down a
*pattern*, anchored to map coordinates so two overlapping fills continue one pattern rather than each
restarting it. With **Random** on they scatter instead, choosing per cell from the brush's tiles, the
same way Stamp already does.

**Offset** and **Autocrop to content** live in the *Resize* section. Offset moves cells by whole
cells — the whole map or just the active layer — and wrapping is an exact permutation, so offsetting
back puts everything where it was. Autocrop shrinks the grid to the cells that hold something,
moving objects with them; a map with nothing painted on it is refused.

**Shape** is one tool with two modes rather than two tools — Tiled's Shape Fill. The buttons for
rectangle and ellipse appear beside the tools while Shape is in hand. Either way the gesture, the
outline that previews it and the single undo step are the same; only the cells differ. An ellipse
is measured from the box you drag and then clipped to the map, so dragging half of one off the edge
does not reshape the half still on it.

**The brush itself can be transformed before it lands.** `X` mirrors it across, `Y` mirrors it
down, and `Z` turns it a quarter clockwise (`Shift+Z` turns it back). These move the arrangement
*and* each tile in it, so a flipped brush stamps a mirrored picture rather than a mirrored
arrangement of unmirrored tiles.

**Shift+click stamps a line** from the last cell you painted to the one you clicked, and the whole
line is one undo step. A fast drag no longer skips cells either: the run between one frame's cell
and the next is filled in.

The fill matches on the tile *and how it is flipped*, so a mirrored wall tile bounds a fill of its
unmirrored twin. That is deliberate: two cells that draw differently are two different cells, and a
fill that spilled through the mirrored ones would cross exactly the seam you drew them to make. It
is four-connected, so it cannot leak diagonally through a corner where two walls only touch at a
point.

The **Terrain** tool needs a terrain set on the map: a tileset that carries terrain rows, which a
Tiled `.tsx` with Wang sets does and a plain image does not. It sets the cell you touch and then
re-fits that cell and its eight neighbours, so edges, outer corners and inner corners follow as you
draw rather than being placed one at a time.

Terrain painting is **refused on a staggered or hexagonal map**, by name. Both terrain paths read a
cell's eight neighbours off a square lattice, and on an offset one those are not the neighbouring
cells — every other row is pushed sideways, and a hexagon has six neighbours rather than eight.
Painting anyway would fit every edge against the wrong cell and look almost right, which is exactly
what this editor refuses by name instead.

A **Tiled Wang set** works too, and no longer has to be one of Plotter's own.
A corner set, an edge set, or a mixed one with a table of its own opens and
paints: each of its colours gets a row in the Terrain list, and a stroke picks the
tile whose wangid matches what its neighbours already say — ties broken by the
colour's probability and then by the lowest tile id. Where nothing matches, the
cell is **left alone** rather than given the closest tile: a wrong tile is a
silent mistake you have to notice, and a hole is one you can see.

A plain image can become one. Add a sheet holding all forty-seven blob cases — drawn by hand, or
asked of a model — and Plotter says so: *These 47 tiles look like a complete 47-case terrain set*,
with **Import as terrain set** to reorder them into the canonical layout and turn terrain painting
on. The reorder is not cosmetic: a terrain set's roles are decided by *position*, so the tiles have
to be put in the right order for any of them to mean anything. **Import as plain tiles** is there
for the coincidence — a sprite sheet whose silhouettes happen to cover every case — and nothing is
added to the map until you choose.

Two other tools know about terrains, and both decide per cell rather than per map. **Erase** clears
a plain tile as a plain tile, and clears a terrain cell by cutting a hole and re-fitting everything
that now borders it — otherwise the ring around the hole keeps the edge art of a neighbour that is
no longer there. **Fill** floods the tile you picked, as it always has; with no tile picked and a
terrain in hand it floods the *terrain field* instead, which crosses a terrain's own forty-seven
cases rather than stopping at the first edge tile. With neither, it still says so.

A whole drag is **one** undo step. That is true of Stamp and Erase now too: a stroke is one gesture,
so it is one thing to take back.

Painting lands on the *active* layer — the highlighted row in the layers pane. Painting with an
object layer active says so rather than doing nothing.

## The minimap

A small view of the whole map sits in the bottom-right corner of the canvas, one pixel per cell,
with a box showing where the pane is currently looking. Click or drag inside it to jump the view
there. The **Minimap** toggle beside *Grid* turns it off.

It is drawn from one average colour per tile rather than by shrinking a full render — a 512-square
map of 32-pixel tiles composites to over 250 million pixels, which is not something that can happen
while you drag. It shows layers as you have them set, so hiding a layer hides it here too.

## Selection

`R` is the rectangular select tool. Drag a marquee; a plain click with no drag clears it. Ctrl+A
selects the whole map and Ctrl+D deselects (Ctrl+Shift+A does too, which is what Inker uses).

A selection is **not** part of the document. It is not undoable, it is not saved, and dragging one
across a map does not mark it as changed — Tiled treats selections the same way, and the
alternative is a map that asks to be saved because you looked at part of it. It is dropped when you
switch to another map, because it names cells in the one you left.

Esc clears it, but only once there is nothing else to cancel: a drag first, then the selected
object, then the selection.

**A selection constrains what the tools may write.** Stamp, Erase, Fill and Shape all land only
inside it, so you can paint freely against an edge you drew once. The Fill is bounded as it
spreads rather than trimmed afterwards — otherwise it could leave the selection, travel around the
outside and come back in, and the trimming would hide the trip.

The exception is a **terrain re-fit**. Painting a terrain, or erasing a terrain cell, re-fits the
eight cells around what you touched, and that ring is allowed to reach past the marquee. Cutting it
off would leave those neighbours drawn as though they still bordered something that is no longer
there, which is a broken field rather than a constrained one.

### The clipboard

Ctrl+C copies the selected cells, Ctrl+X copies and clears them in one undo step, and Delete clears
them without copying. With the Objects tool in hand, Delete removes the selected object instead —
which one you get is decided by the tool you are holding, so a marquee left over from earlier
cannot quietly erase tiles.

**Ctrl+V loads the copy into the brush and switches to Stamp** rather than dropping it somewhere
straight away. You then place it like any other brush, which is why it clips at the map edges, costs
one undo step per stroke and obeys the selection — all rules the stamp already had. One difference
worth knowing: the stamp replaces wholesale, so empty cells in a pasted block *erase* what they land
on rather than letting it show through.

Pasting into a different map is **refused by name**. Tile numbers are assigned per map, so the same
number means a different tile elsewhere and the block would come out silently redrawn. The message
says which map the copy came from.

## Layers

Two kinds, and they are genuinely different.

A **tile layer** is a rectangle of cells the tools paint into. A map usually has several: a floor, a
wall layer above it, decoration above that. Each carries its own opacity and visibility, and they
composite bottom-first — which is why the list is drawn top-first, the way every layered editor
does it.

An **object layer** holds named rectangles and points. Nothing on one is drawn in an export; they
are metadata an engine reads. With the **Objects** tool active, click empty space and drag to draw a
rectangle, or click without dragging to drop a point. Click an existing object to select it, and its
form appears under the layer list.

**Selected objects can be moved and resized on the canvas.** Drag an object's body to move it, or
one of the four corner handles to resize it — the opposite corner stays pinned, and dragging a
corner past it flips the rectangle rather than giving it a negative size. Hold Ctrl while dragging
to snap to the grid. A whole drag is one undo step, and a click that moves nothing costs none at
all. Delete removes the selected object.

Handles appear only on the selected rectangle, and not at all on a locked layer, where the drag
would be refused anyway.

A **polygon** or **polyline** gets a handle per point instead. Drag one to move it — computed in the
object's own frame, so a vertex on a rotated shape lands where the outline says rather than where the
maths would put it if the rotation were ignored. Ctrl+click a segment to insert a point on it, and
Alt+click a point to remove it; a polygon floors at three points and a polyline at two, and the floor
is refused by name rather than silently.

**Ctrl+J** duplicates the selected object one cell down and right. Ctrl+C and Ctrl+V with the Objects
tool in hand copy and paste one, and pasting into another map is refused whole — a tile object carries
a gid and an object property may name an id, and both mean something else elsewhere. (Ctrl+D stays
deselect here, unlike Tiled, because every other editor in this app deselects on it.) Selecting
several objects at once is deliberately not offered yet.

An **image layer** created here starts empty, and **Choose image…** on its row attaches a picture.
`.wmap` already stores an image layer's pixels, so the file needs nothing new to hold it.

Right-click a layer for **Duplicate** and **Merge down** beside Delete. A duplicate copies the whole
subtree with fresh identities — nothing is shared with the original, so painting on one does not paint
the other. Merge down folds the layer onto the tile layer directly below it, cell by cell, with a
painted cell above winning; it is a merge of the *data* rather than a picture, so opacity, tint and
blend stay where they were and the result is what the renderer was already drawing. Merging onto
anything that is not a tile layer is refused by name.

**Highlight current layer**, beside Grid and Minimap, dims everything but the layer you are painting
on. It changes the canvas only — an export is unaffected, because exports composite from the same
resolver and a dim living there would export dimmed.

### Locking a layer

The padlock beside the eye locks a layer. A locked layer cannot be painted on, erased, cut from,
deleted out of, or have objects added to or removed from it — the usual reason being that you are
working above a finished floor and keep catching it by accident.

A lock stops **content** changes and nothing else. You can still rename the layer, hide it, change
its opacity, move it up and down the stack, delete the whole layer, and of course unlock it. You can
also still *select* an object on a locked layer and read its properties, and still copy cells from
one — a lock is not a reason to lose sight of your own work.

Locks are saved in `.wmap`, and in `.tmx`/`.tmj` exports where Tiled understands them. A map written
before this existed opens with everything unlocked.

### Layer and map properties

Layers and the map itself carry typed custom properties, the same kind objects do. A layer's live
under a collapsed **Properties** header in the layer's expanded row; the map's under one in the
tools pane, below the size readout.

Both have been part of the file format — and survived every Tiled round trip — since Plotter
shipped. What was missing was any way to set one without opening the file in a text editor. Both are
undoable, and both are saved.

### Object properties

An object carries a name, a class, and any number of typed custom properties: string, int, float,
bool, colour, file, object and class. The type is stored rather than guessed, so a colour stays a
colour on the way out to Tiled and back. Those properties are the whole point of an object layer —
they are how a map says "this door needs the brass key" to code that has never heard of Warlock.

A **file** property is a path Plotter carries verbatim and never resolves; an **object** property is
a Tiled object id, where 0 means none; and a **class** property holds a block of properties of its
own, under a type name declared in a Tiled project. All three arrive from Tiled and survive the trip
back. The new-property row offers the seven you can fill in on one line, and a class or a list opens
an **Edit** disclosure holding an editor of its own — nested classes and lists included, each member
edited by its stored type. One editor serves the map's properties, a layer's and an object's, because
all three are the same model.

## Files

`Ctrl+S` saves. A map's own format is `.wmap`, a single file that embeds its tileset images, so it
can be moved or sent without a folder of dependencies. Two saves of an unchanged map produce
byte-identical files, which means a content hash means something and a diff shows only what you
actually changed.

A map *opened* from a `.tmx` or `.tmj` saves back to that format instead. Warlock does not silently
convert a file you brought from Tiled into one Tiled cannot open.

## Working with Tiled

**Export .tmx** and **Export .tmj** write a Tiled map beside one `.tsx` and one `.png` per tileset,
which is the layout Tiled and every engine importer expects. TMX has no portable way to embed an
image, which is why an export is several files rather than one.

A `.wmap` carries the map's projection and its terrain sets. A `.tmx` carries the projection and
describes the terrain sets as Tiled Wang sets, so an atlas made here opens in Tiled with a working
terrain brush.

Exporting deliberately does *not* retarget `Ctrl+S`. The `.wmap` holds things the `.tmx` cannot, so
making the export the document's home would lose them on the next save.

Import goes the other way: `Ctrl+O` opens a `.tmx` or `.tmj`, in CSV, base64, zlib, gzip or the
older XML tile form. Everything Plotter models survives the trip exactly, transform flags included.

### What Plotter refuses

Tiled's format is much larger than an orthogonal stamp-and-fill editor. A file using something
Plotter does not model is **refused by name** rather than loaded with half of it quietly dropped —
because the drop is invisible right up to the moment you save, at which point the other half is
gone. The message says which feature and what to do about it.

Refused: the hexagonal 120° tile-rotation flag; object templates; a tileset
with no pixels at all; an orientation or a compression Plotter has no name for; Tiled's older
terrain types and its deprecated per-tile terrain assignment, both of which Tiled is itself
retiring; deprecated tile-space layer coordinates and image-layer transparent colours; and custom
properties of a type outside the set Plotter models.

Several things that were on that list have left it, each in the change that taught the editor to
model them: staggered and hexagonal maps, image-collection tilesets, external `.tsj` tilesets,
generic Wang sets, everything a tile can carry (class, properties, probability, animation,
collision), the tileset presentation fields, colour-key transparency, infinite maps, and
zstd-compressed layer data (read; writes stay zlib, because every Tiled reads zlib).

An **infinite map** opens with its chunks assembled into one window and its corner remembered, so a
map authored in Tiled at negative coordinates comes back out at the same coordinates it went in at.
Empty chunks are not written, which is the format's own shape: a map painted in two clusters saves
two clusters, not the rectangle between them, and erasing an area shrinks the file.

An **image-collection tileset** — one where every tile is its own file — opens like any other. Its
ids may have gaps, and a tile larger than the map's grid draws at its own size anchored by its
bottom-left, so a 32 px map full of 48 px trees looks the way it does in Tiled.

Group layers, image layers, layer offsets, tints, parallax and classes, every one of the eight
object shapes, object rotation and object templates' *contents* all load — several of those were on
the refusal list in earlier versions and left it as the editor learned to draw them. A *hidden*
object is modelled rather than refused: hiding something changes nothing about where it is.

### What Plotter writes that Tiled does not read

The document model has grown a handful of things Tiled has no spelling for, and they go into a
`.tmx` or `.tmj` anyway because losing them on export would be worse. They are **Warlock dialect**:
an oblique projection with its skew, a per-layer blend mode, a per-object opacity, the capsule
object shape, and list-valued custom properties. A file carrying any of them opens in Plotter and
will not open cleanly in Tiled.

If a map has to go to Tiled, keep it to the features in the round-trip list. If it only has to come
back here, `.wmap` holds everything without qualification — which is why exporting deliberately
does not retarget `Ctrl+S`.

The full list of what Plotter reads, refuses, preserves and writes as dialect is kept in
`docs/COMPAT.md`, checked against the code on every test run.

## Sending a map to the library

**Export to the library** (`Ctrl+E`) renders the map flat and mints it as an ordinary reference
asset. It joins the same library every other asset is in, with a thumbnail, a card and every export
the library offers.

The map itself is kept beside it as `map.wmap`, which is what lets **Open in Plotter** on the card
reopen the real document — layers, objects and all — rather than a single flattened picture. It
follows the same precedent as Inker's `paint.ora` and Clay's `build.wblk`: never served, never
downloadable, and gone with the job directory for free.

Hidden layers are not rendered. One flag decides both what you see and what comes out.

## Where the files go

| File | What it is |
| --- | --- |
| `<name>.wmap` | The map, with its tileset images embedded. Warlock's own format. |
| `<name>.tmx` / `.tmj` | A Tiled export, beside its `tilesets/` folder. |
| `~/.warlock/assets/<job>/input.png` | The flat render, for a map exported to the library. |
| `~/.warlock/assets/<job>/map.wmap` | The map behind that render. Not served; reopened by **Open in Plotter**. |

See [Keyboard shortcuts](18-shortcuts.md) for every binding, and
[Packwright](13-packwright.md) for building the tilesets this mode consumes.

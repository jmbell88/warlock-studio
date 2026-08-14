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

With nothing open, the middle column offers **New map** and **Open a file...**, and lists the maps
you had open recently — clicking one reopens it, and hovering it shows the full path. The map panel
on the right offers the same two buttons. Once a map is open, `Ctrl+N` and `Ctrl+O` do the same two
things from the keyboard.

A new map is 32 by 32 cells of 32 by 32 pixels, with one tile layer called *Ground* already on it.
Both sizes can be changed later; the grid under **Resize** in the tools pane grows or crops the map
and moves every object with the content, so a trigger volume drawn around a doorway stays around
that doorway. The **Offset** fields decide where the old content lands, which is how you add rows at
the *top* rather than the bottom.

## Tilesets

A map paints nothing until it has a tileset. **Add from a file...** in the tileset pane takes either
a plain image, which is sliced into a grid at the map's own tile size, or a Tiled `.tsx`, which
carries its own slicing, spacing and margin. Dropping an image or a `.tsx` on the window does the
same thing.

You can also send a library asset straight in — right-click any reference card and choose **Use as a
tileset in Plotter**. A [Packwright](10-packwright.md) grid pack is the natural source: it writes a
`.tsx` beside its atlas precisely so it can be used here with no conversion.

Tilesets are *added*, never removed. Every tile in the map is numbered from the tileset it belongs
to, so dropping one from the middle would either renumber everything above it — invalidating every
cell already painted — or leave a hole. Undo takes back a tileset you have only just added.

### Picking a tile

The palette under the tileset combo is the atlas itself. Click a tile to pick it; drag across
several and you get a multi-tile brush, which stamps as one block. The white outline shows what is
selected, and the cursor on the canvas shows the footprint the brush is about to cover.

### Generating a ground set

**Generate a ground set** in the tileset pane builds a tileset instead of loading one: flat terrain
colours with a one-pixel darker outline, laid out for **blob autotiling**. Each terrain gets 47
cells — every combination of neighbours that looks different — so grass meeting dirt has a real
inner corner rather than a staircase.

Name the terrains and pick a colour each. **Order is precedence**: where two terrains meet, the one
lower in the list is the one that gets the outline, and the one above it runs underneath unbroken.
That is what makes a three-way meeting work — grass, then sand, then water gives you a beach, with
sand outlined against the grass and water outlined against both.

This is the *base* set, and it is meant to be painted over: it is deliberately plain so that the
shapes are unambiguous, and it generates identically every time so you can compare a polished set
against the one it started from.

The **projection** is chosen here, and only while the map is still empty. Generating an isometric
set is what makes a map isometric, and it happens in the same undo step as the tileset arriving.
Once anything is painted the choice is fixed, because a set drawn for one lattice would paint the
wrong shape into every cell already drawn for the other.

### Polishing an atlas in Inker

**Polish in Inker** opens a tileset's atlas as an ordinary drawing — one flat layer, not sliced into
cells, because keeping an outline consistent *across* neighbouring cases is exactly what the pass is
for and 235 separate frames would hide it.

When you are done, **Back onto...** under *from Inker* returns it to the same tileset. Every painted
cell keeps its tile and simply redraws with the new art, because the numbering is untouched. An
atlas whose size changed is refused by name rather than accepted — the roles are positional, so a
cropped atlas is one whose tile 93 is no longer the tile the map thinks it is.

## Isometric maps

A map is drawn on one of two lattices, and which one is a property of the map. On an **isometric**
map a cell is a 2:1 diamond, the grid follows the two lattice directions rather than the screen
axes, and the status line under the canvas shows the cell under the pointer — which is the only
thing that reliably answers "am I about to click the diamond I mean".

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
| `I` | Pick | Takes the tile under the cursor as the brush |
| `S` | Objects | Selects and draws objects (see below) |

The letters are [Tiled](https://www.mapeditor.org/)'s, because that is the editor whose files this
one reads and writes, and so the one you are most likely arriving from.

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

The **Terrain** tool needs a terrain set on the map — see *Generating a ground set* above. It sets
the cell you touch and then re-fits that cell and its eight neighbours, so edges, outer corners and
inner corners follow as you draw rather than being placed one at a time.

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
back. The new-property row offers the seven you can fill in on one line — a class arriving from a
file is shown as a read-only summary until the recursive editor lands.

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
describes the terrain sets as Tiled Wang sets, so a generated atlas opens in Tiled with a working
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

Refused: staggered and hexagonal maps; infinite (chunked) maps; group and image layers;
ellipse, polygon, polyline and text objects; tile objects and object templates; rotated objects;
image-collection tilesets; Wang sets that are not one of Plotter's own terrain sets, and Tiled's
older terrain types; per-tile animation, properties and collision shapes; zstd-compressed
layer data; layer pixel offsets; and custom properties of a type outside Tiled's own set.

A *hidden* object is modelled rather than refused — hiding something changes nothing about where it
is. A *rotated* one is refused, because an unrotated outline drawn for a rotated object is a wrong
picture, and a wrong picture is worse than a refusal.

The full list of what Plotter reads, refuses and preserves is kept in `docs/PLOTTER_COMPAT.md`,
checked against the code on every test run.

## Sending a map to the library

**Export to the library** (`Ctrl+E`) renders the map flat and mints it as an ordinary reference
asset. It joins the same library every other asset is in, with a thumbnail, a card and every export
the library offers.

The map itself is kept beside it as `map.wmap`, which is what lets **Edit in Plotter** on the card
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
| `~/.warlock/assets/<job>/map.wmap` | The map behind that render. Not served; reopened by **Edit in Plotter**. |

See [Keyboard shortcuts](14-shortcuts.md) for every binding, and
[Packwright](10-packwright.md) for building the tilesets this mode consumes.

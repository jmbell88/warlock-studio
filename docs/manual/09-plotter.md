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

## Tools

| Key | Tool | What it does |
| --- | --- | --- |
| `B` | Stamp | Puts the brush down, following the drag |
| `E` | Erase | Clears cells, following the drag |
| `G` | Fill | Floods the connected run under the cursor |
| `R` | Rect | Fills a rectangle between press and release |
| `I` | Pick | Takes the tile under the cursor as the brush |
| `O` | Objects | Selects and draws objects (see below) |

The fill matches on the tile *and how it is flipped*, so a mirrored wall tile bounds a fill of its
unmirrored twin. That is deliberate: two cells that draw differently are two different cells, and a
fill that spilled through the mirrored ones would cross exactly the seam you drew them to make. It
is four-connected, so it cannot leak diagonally through a corner where two walls only touch at a
point.

Painting lands on the *active* layer — the highlighted row in the layers pane. Painting with an
object layer active says so rather than doing nothing.

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

### Object properties

An object carries a name, a class, and any number of typed custom properties: string, int, float,
bool or colour. The type is stored rather than guessed, so a colour stays a colour on the way out to
Tiled and back. Those properties are the whole point of an object layer — they are how a map says
"this door needs the brass key" to code that has never heard of Warlock.

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

Exporting deliberately does *not* retarget `Ctrl+S`. The `.wmap` holds things the `.tmx` cannot, so
making the export the document's home would lose them on the next save.

Import goes the other way: `Ctrl+O` opens a `.tmx` or `.tmj`, in CSV, base64, zlib, gzip or the
older XML tile form. Everything Plotter models survives the trip exactly, transform flags included.

### What Plotter refuses

Tiled's format is much larger than an orthogonal stamp-and-fill editor. A file using something
Plotter does not model is **refused by name** rather than loaded with half of it quietly dropped —
because the drop is invisible right up to the moment you save, at which point the other half is
gone. The message says which feature and what to do about it.

Refused: isometric, staggered and hexagonal maps; infinite (chunked) maps; group and image layers;
ellipse, polygon, polyline and text objects; tile objects and object templates; rotated objects;
image-collection tilesets; Wang sets and terrains; per-tile animation, properties and collision
shapes; zstd-compressed layer data; layer pixel offsets; and custom properties outside the five
supported types.

A *hidden* object is modelled rather than refused — hiding something changes nothing about where it
is. A *rotated* one is refused, because an unrotated outline drawn for a rotated object is a wrong
picture, and a wrong picture is worse than a refusal.

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
| `assets/<job>/input.png` | The flat render, for a map exported to the library. |
| `assets/<job>/map.wmap` | The map behind that render. Not served; reopened by **Edit in Plotter**. |

See [Keyboard shortcuts](14-shortcuts.md) for every binding, and
[Packwright](10-packwright.md) for building the tilesets this mode consumes.

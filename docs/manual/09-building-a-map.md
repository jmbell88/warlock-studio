# Building a map

Plotter is a tile-map editor. If you have used Tiled, almost everything here will be familiar — it
speaks Tiled's formats and borrows Tiled's keyboard deliberately. If you have not, this chapter
starts from the beginning.

No GPU, no weights.

## Starting a map

`Ctrl+N` opens the New Map dialogue, and it asks for two things you cannot change later.

**Tile size** is how big one cell is in pixels. **Projection** is the lattice — orthogonal,
isometric, staggered or hexagonal.

Projection is fixed the moment anything is painted, and that is stricter than Tiled, which lets you
change orientation afterwards. The reason is that projection here decides the lattice rather than
just the drawing: cells mean different things in a diamond grid than in a square one, and
reinterpreting a painted map under a different lattice would silently move every tile.

Tile size, by contrast, *can* be changed on a painted map. Nothing re-slices and no tile is
renumbered — a tile keeps naming the same artwork whatever size the cell under it now is.

## Tilesets

A tileset is an image cut into tiles. Bring one in from the Tileset menu: a `.tsx` file carries its
own slicing, and any other image is cut at the map's current tile size.

You can also generate one. Create's **Tileset** asset type lands a finished sheet in the library
that can be used as a tileset here, and its **Tile layout** decides what kind. *Materials* draws one
seamless tile per surface you name, so the tiles genuinely repeat. *Terrain set* draws two surfaces
and composites them into a complete forty-seven-case blob set, which arrives here with the
[Terrain](#terrain) tool already working — it says what it is, so nothing is
guessed and nothing is asked. *Grid (legacy)* is the older single-generation 8×8 sheet, kept because
it is the only one that can be drawn 3/4 or isometric.

Note that this lives in Create, not in Plotter — an in-Plotter "paint with AI" existed once and
was removed in favour of it. If a map is still unpainted it will adopt the sheet's own projection; if
it is painted and the lattices disagree, you get a question rather than a silently mis-sliced sheet.

Per-tile metadata is edited in the tileset editor, in three tabs:

- **Tiles** — a class name, custom properties, and a random-paint **probability**. Setting
  probability to zero means "never chosen at random", but the tile stays placeable by hand. That is
  Tiled's own rule.
- **Collision** — rectangles, ellipses and polygons drawn on an enlarged view of one tile. Plotter
  never hits-tests these itself; they are metadata for your engine to read.
- **Animation** — an ordered list of tiles and durations. The canvas plays them; every export writes
  the first frame.

## Painting

The tools, and their letters, are Tiled's:

| Key | Tool |
| --- | --- |
| `B` | Stamp — the brush. Drag to paint a stroke. |
| `E` | Erase. `Shift`-drag clears a rectangle. |
| `F` | Fill — flood fill. |
| `T` | Terrain — the autotiling brush. |
| `P` | Shape — filled rectangle or ellipse. |
| `R` | Select — a rectangular marquee. |
| `W` | Wand. `Ctrl`-click folds in Tiled's *Select Same Tile*. |
| `I` | Pick — the tile eyedropper. |
| `S` | Objects. |

Six of those letters mean something different on an object layer, which is again Tiled's
arrangement: the letter belongs to the gesture, and which gesture depends on what you are painting.
On an object layer `R`, `I`, `E`, `P`, `L`, `T` and `X` insert a rectangle, point, ellipse, polygon,
polyline, tile and text respectively.

`X`, `Y` and `Z` flip and rotate the *brush* — the tiles about to be placed, not the map.

The Stamp has a **Random** mode that scatters from the non-empty tiles of the current stamp, which
is how you break up a floor without placing every variant by hand.

One thing about flood fill that surprises people: it matches on the tile *including its flip flags*,
so a mirrored tile bounds a fill of its unmirrored twin. That is usually what you want and
occasionally not.

## Terrain

The Terrain brush is the feature most worth learning. Paint a terrain type and the edges resolve
themselves — grass meets sand with the right corner tiles, automatically. It needs a tileset that
carries terrain rows: a Tiled `.tsx` with Wang sets, a sheet you recognised on the way in, or a
*Terrain set* generated in Create, which carries the rows in its own record and needs no import step
at all.

Plotter reads the 47-case blob layout, and it resolves conflicts by **list position**: a terrain
ranked higher in the list always owns the outline where two meet. That is a real design choice with
a real cost, and it is worth knowing rather than discovering. It guarantees that a three-terrain
junction resolves to exactly one tile per cell. It also means a terrain's edge art cannot vary
depending on what is on the other side of it.

Generic Wang sets — corner-only, edge-only or mixed, as authored in Tiled — are read as data and
painted by matching. A cell with no matching tile is **left untouched** rather than given a near
miss, so a gap in your Wang set shows up as a gap rather than as a wrong tile.

## Objects

Object layers hold things that are not tiles: spawn points, trigger volumes, collision shapes,
labels. Shapes available are rectangle, point, ellipse, capsule, polygon, polyline, tile and text.

Object coordinates are **pixels, not tiles** — Tiled's convention. Objects are placed exactly where
the mouse reports, with no snap-to-grid toggle, which is again Tiled's default.

**Rotation is a numeric field in the properties panel, not a drag handle on the canvas.** There is no
rotate gesture here. Type degrees, clockwise, about the object's own origin. Resizing a rotated
object does work correctly in the object's own frame.

Both objects and layers carry typed custom properties, which is how anything you invent reaches your
engine.

## Layers

Four kinds: tile, object, group and image. All four share five decorations — a class name, blend
mode, tint, pixel offset and parallax factors — and groups nest, with a child's offsets summing and
opacities multiplying.

Blend modes are the one thing the live canvas cannot always show. Above a size budget, mid-stroke, or
with no GL context, it falls back to an unblended draw and **says so on screen** rather than quietly
disagreeing with what the export will produce.

`H` highlights the current layer by dimming the others. `Ctrl+G` toggles the grid. Space-drag or
middle-drag pans; the wheel zooms; `1` returns to 100%. There is a minimap in the corner, one pixel
per cell.

## Infinite maps

An infinite map is not "a map with no edges". It is a **dense window over the populated area, plus an
origin**. Painting outside the window grows it; the origin slides to keep coordinates meaning the
same thing.

That model explains two behaviours which otherwise look like bugs. Offsetting the whole map slides
the origin rather than moving any content — nothing to see, and correct. And offsetting with *wrap*
is **refused by name** on an infinite map, because a wrap needs a fixed boundary to wrap around and
an infinite map does not have one.

Introduce infinite maps to yourself after the fixed-size model is comfortable. It is a genuinely
different mental model, not a switch that removes a limit.

## Tiled interoperability

Plotter saves `.wmap` natively and exports `.tmx` / `.tmj` with `.tsx` tilesets for Tiled.

Most things round-trip: every Tiled projection, infinite maps, all four layer kinds with their
decorations, every Tiled object shape, object rotation, external and embedded and image-collection tilesets, per-tile
animation, collision, properties, class and probability, and both blob and generic Wang sets.

Some things are **Warlock dialect** — modelled and written, but no Tiled release reads them back:
oblique projection with skew, per-layer blend modes, the capsule object shape, per-object opacity,
and the recursive list property type. Use them freely if Warlock is your only editor; avoid them if
the map has to survive a round trip through Tiled.

A few things are refused by name rather than half-supported: object templates, Automapping, projects
and worlds, plugins, and 120° hex tile rotation. Those are non-goals, not gaps.

One honest caveat, smaller than it was. Plotter now writes `tiledversion` 1.12.2, because on
2026-08-29 a real Tiled 1.12.x opened a Plotter export and a real Tiled 1.12.2 map was read back
here. But that was one orthogonal map each way — **the compatibility test corpus is still authored
by Plotter itself**, every fixture a file this editor wrote. A green test proves that Warlock's
reader and Warlock's writer agree with each other. Beyond plain CSV ground — flipped tiles, objects,
properties, infinite maps — it still does not prove that Tiled agrees with either.

## Try it

1. `Ctrl+N` with 32 px tiles, orthogonal.
2. Import a tileset image, then paint a room with the Stamp and floor it with Fill. Undo a couple of
   strokes and note that one drag is one undo step.
3. Add a second tile layer for decoration and an object layer for logic. Put a point object down as a
   spawn, a rectangle as a trigger, give each a custom property, and rotate the trigger 30° with the
   numeric field.
4. Open the tileset editor: draw a collider on one tile, chain three tiles into an animation, and set
   one tile's probability to zero.
5. With two terrains in a tileset, paint them meeting, then add a third across the boundary and watch
   precedence resolve it.
6. Export `.tmx`. If you have Tiled installed, open it there — that is the exercise that actually
   tests the claims above.

## What to read next

[Packing an atlas](10-packing-an-atlas.md) — Packwright, and the setting that can make an atlas
bigger than the images that went into it.

# Packwright

Packwright is the top-level packing mode: many images in, one atlas out, with a sidecar that says
where everything landed. It is the step between drawing sprites and shipping them.

It exists because every 2D engine wants an atlas and nothing else in Warlock made one. A sprite
sheet baked from a 3D model is a regular grid by construction; a folder of hand-drawn frames is not,
and packing it by hand is exactly the sort of arithmetic a person should never do.

There are four images in `examples/` in the source checkout to try it on — `player.png`, two sprite
sheets and a tileset. They are inputs, not fixtures: nothing in the app or the test suite reads
them, so they are safe to delete if you would rather not carry them.

It is a mode, not a takeover. Switching away leaves every open document where it was. Several
atlases stay open at once, and the layout follows the rest of the app: sources and settings on the
left, the packed atlas in the middle, the placement list and the file panel on the right.

## Starting an atlas

With nothing open, the middle column offers **New atlas** and **Open a file...**, and lists what you
had open recently. `Ctrl+N` and `Ctrl+O` do the same from the keyboard. The document's own format is
`.wpack`.

A document that cannot be opened is refused with the reason, not a generic failure. A file over the
read ceiling is refused before a byte of it is read, and an archive that claims more than a
gigabyte unpacked, names more than 4096 sources, or carries a source image past 16 megapixels is
named as the problem it is — those are the marks of a damaged or hostile file, not a big atlas. A
recent document that refuses to open also drops off the Resume list, so a moved or corrupted file
does not sit there failing every time you look at it.

## Sources

Four ways in.

**Add an image...** takes one file. Dropping images on the window adds them too — several at once,
and one that is already in the atlas is skipped rather than refusing the whole batch.

**Add a tile set...** takes a sheet that is already a tileset and slices it back into tiles. You
give the tile size — 32×32 until you change it — and the popup answers with the grid it makes and
how many cells that keeps, live as you type. A cell with no opaque pixel anywhere in it is dropped,
which is the point: a sparse sheet re-packs into a smaller one. A strip left over where the sheet's
size is not a multiple of the tile is outside the grid and ignored, which is what Tiled does with
the same sheet, and a sheet with no transparency at all simply keeps every cell. Re-importing the
same file cut at a different tile size adds a second set rather than being skipped as duplicates of
the first cut.

**From Inker** is the reason this mode sits beside the raster editor. Every document open in
[Inker](08-inker.md) gets a button here: an animated document contributes one sprite per frame, and
a still one contributes one sprite per layer. A packed frame is pixel-identical to what the timeline
plays, because it goes through the same flatten the playback and the onion skin use.

**From the library** — right-click any reference card and choose **Add to a Packwright atlas**.

Each source keeps a stable identity derived from where it came from, not from what it is called. So
renaming a sprite changes what the sidecar calls it and nothing else: two layers legitimately called
"Layer 1" stay two sprites, and the pack order does not move under you when you rename something.

Selecting a source highlights it in the atlas preview and in the placement list, and offers a rename
box and a **Remove** button. `Delete` removes the selected source.

A rename that lands re-packs automatically, so the next export's sidecar carries the new name. One
past 64 characters, or containing a path separator or control character, is refused — the name is
written verbatim into the sidecar, where a consumer may treat it as a filename.

## The two modes

**Grid** puts every sprite in a uniform cell the size of the largest one, row-major. The result is a
*tileset* — a regular atlas an engine can slice by arithmetic — so it exports a `.tsx` as well as
the JSON, and can be used directly as a tileset in [Plotter](11-plotter.md) or in Tiled.

**MaxRects** packs tightly and irregularly. The atlas comes out considerably smaller, but the cells
are not a grid, so an importer has to read the JSON to find anything.

Pick grid when the output is a tileset or an animation strip; pick MaxRects when every sprite is
addressed by name.

## Settings

**Trim** cuts each sprite down to where its alpha actually stops before packing it, and records the
offset so an engine can put it back exactly where you drew it. It is on by default and is usually
free space. A fully transparent sprite is packed as a single pixel and marked blank rather than
dropped — a blank frame in the middle of a clip is a real frame, it is the pause, and removing it
would renumber everything after it.

**Columns** (grid mode only) fixes how many cells wide the grid is, for a tileset you index by column
— an animation strip cut at a known width, say. Zero is auto: the packer searches for a near-square
grid that fits the sprite count. An explicit count is honoured exactly, including through
power-of-two rounding — the atlas may still round up, but the grid does not follow it there, so a
rounded atlas can carry dead space past the last column rather than a column nothing placed. That is
also why an explicit count occasionally refuses a `.tsx` export (see Exporting): Tiled derives its
own column count from the image, and this pack will not let the two disagree.

**Padding** is the gap between neighbours and around the edge. Two is enough for most things.

**Extrude** repeats each sprite's border pixels outward into that gap, so a filtered texture
sampling just past an edge finds the sprite's own colour rather than its neighbour's. It is the fix
for the thin seams that appear between tiles at some zoom levels. Padding must be at least twice
extrude, because two neighbours extrude into one shared gutter; a combination that would bleed is
refused with the numbers rather than quietly clamped. Both are capped at 256 — past that a gutter
is not padding, and the only thing the arithmetic could compute is a refusal.

**Power-of-two** rounds the atlas up to the next power of two in each direction. Older hardware and
some engines require it; leaving it off gives a tighter atlas.

**Max size** is the ceiling the packer grows to. Past 8192 pixels engines start refusing a texture
outright, so that is the hard limit whatever this says.

**Sidecar schema** picks TexturePacker's *Array* or *Hash* shape for the exported JSON — see
Exporting.

## The preview

The middle pane shows the packed atlas over a checkerboard, so transparency is visible rather than
guessed. The wheel zooms, the middle button pans, `Ctrl+0` fits and `Ctrl+1` goes to 100%. Every
placement is outlined; the selected one is highlighted. Clicking a sprite selects it.

Packing happens automatically whenever something changes and runs off the frame thread, so a
hundred-sprite atlas does not stall the window. `R` forces a repack.

Below the atlas size, a line compares the packed area to the pile of source pixels it came from —
"Packed to 41% of the source pixels' area", say. It says shrink or growth honestly rather than
implying one: a *sparse* source (mostly empty, from a tile-set import that dropped its blank cells)
reliably packs smaller, but power-of-two rounding can round a tightly-fitting grid up past its
source, and this is where that shows up rather than being discovered at export. Turning
power-of-two off, or switching to MaxRects, is the fix either line points at.

## When it does not fit

A pack that cannot fit says so in the placement list, with the number and the remedy — raise the max
size, turn trimming on, or split it into two atlases. It never produces a partial atlas: half an
atlas is much harder to notice than none of one, because it looks like success.

## Exporting

**Atlas + JSON** (`Ctrl+Shift+E`) writes the atlas PNG and a sidecar beside it in one of
TexturePacker's two JSON schemas — the Settings pane's **Sidecar schema** picks which. *Array*
(the default, and what every export wrote before there was a choice) lists frames in the packer's
own order; *Hash* keys the same frames by filename instead, for a loader that looks one up by name
rather than scanning for it. Two sources sharing a name pack fine under Array; Hash refuses instead
of silently keeping only one of them, since a dict has room for only one key of that name. When the
pack is a grid it writes a `.tsx` as well — unless an explicit **Columns** count and power-of-two
rounding have put the grid and the image geometry out of agreement (see Settings), in which case the
`.tsx` is refused by name rather than written wrong; the PNG and JSON still export.

The sidecar is engine-neutral: pixel rectangles and nothing else. Each frame records where it landed
in the atlas, whether it was trimmed, where the trimmed rectangle sat inside the original image, and
what the original's size was — which together are what let a consumer place a sprite where you drew
it rather than flush against its own bounding box.

Note that this is deliberately *not* the sidecar a [sprite sheet](07-sprite-sheets.md) writes. That
format is Warlock's own and describes poses and view directions; this one describes an arbitrary
pile of pictures. They answer different questions and have one writer each.

An unchanged document exports byte-identical files however its sources happen to be ordered, because
the packer is deterministic all the way down.

## Sending an atlas to the library

**Export to the library** (`Ctrl+E`) mints the atlas as an ordinary reference asset, with the
document kept beside it as `pack.wpack` — which is what lets **Edit in Packwright** on the card
reopen the real document rather than a flat picture. It follows the same precedent as Inker's
`paint.ora` and Clay's `build.wblk`.

## Where the files go

| File | What it is |
| --- | --- |
| `<name>.wpack` | The document: sources and settings. The atlas is derived, not stored. |
| `<name>.png` | An exported atlas. |
| `<name>.json` | Its sidecar, in TexturePacker's Array or Hash JSON schema. |
| `<name>.tsx` | A Tiled tileset, written for a grid pack whose geometry agrees with Tiled's own. |
| `~/.warlock/assets/<job>/input.png` | The atlas, for one exported to the library. |
| `~/.warlock/assets/<job>/pack.wpack` | The document behind it. Not served; reopened by **Edit in Packwright**. |

See [Keyboard shortcuts](16-shortcuts.md) for every binding, and [Plotter](11-plotter.md) for the
mode that consumes a grid pack as a tileset.

# Packing an atlas

Packwright takes a pile of images and produces one atlas plus a sidecar that says where everything
landed. It is the step between having art and shipping it, and it is entirely offline — no GPU, no
weights, nothing to download.

## Getting sprites in

Six doors, and they all produce the same kind of thing:

- **Add an image** — a file picker.
- **Drag and drop** — several files at once. A file already in the atlas is *skipped*, not refused,
  so dropping twenty of which one is a duplicate adds nineteen.
- **Add a tile set** — an existing sheet, cut on a cell size you type, with a live count as you type
  it. Empty cells are dropped. There is an optional **drop duplicate tiles**, and a sub-option to
  treat flipped and rotated copies as duplicates too.
- **From Inker** — one button per open document. An animated document contributes one sprite per
  frame, pixel-identical to the timeline; a still one contributes one sprite per layer, including
  hidden ones.
- **From the library** — right-click a card, *Add to a Packwright atlas*.
- **From Troupe** — a rendered character sheet contributes one sprite per cell.

A sprite's identity comes from where it came from, never from its display name, so renaming one is
cosmetic and cannot collide with anything.

## Grid or MaxRects

Two packers, and the choice is about what you are making.

**Grid** puts every sprite in a uniform cell, row-major. It wastes space on sprites smaller than the
cell, and it is what a tile set is — so a grid pack is the one that can also export a `.tsx` for
Tiled.

**MaxRects** packs tightly and irregularly. It is smaller, and there is no cell to speak of, so
there is no tileset to export.

Both are deterministic: the same inputs give the same atlas, every time. That matters more than it
sounds — a repack that shuffled your sprites would invalidate every sidecar anyone had already
shipped against it.

Nothing is ever rotated to fit. The sidecar's `rotated` field is always false, honestly.

## The settings

**Trim** cuts each sprite to its non-transparent bounds and records the offset. On by default, and
almost always right. One deliberate exception: a fully transparent sprite packs as a 1×1 frame
rather than being dropped, because a blank frame in an animation is a real pause and deleting it
would shift everything after it.

**Padding** is the gutter between neighbours and around the edge. Two pixels by default.

**Extrude** replicates each sprite's border pixels outward into that gutter, which is how you stop a
filtered texture bleeding its neighbour in at small scales. Zero by default.

These two interact, and the app enforces it: **padding must be at least twice extrude**. Two
neighbours extrude into one shared gutter, so anything less means they extrude into each other. Set
extrude larger and the refusal names the rule rather than quietly clamping.

**Columns**, on a grid pack, defaults to an automatic near-square search. Give it an explicit number
and it is honoured exactly.

**Max size** caps the atlas — 2048 by default, and the hard ceiling is 8192.

**Sidecar schema** is Array (an ordered list, the default) or Hash (keyed by filename). Hash refuses
on a filename collision, which is the correct behaviour and worth knowing before you hit it.

## The power-of-two trap

**Power-of-two** rounds the atlas up to the next power of two in each dimension. It is on by default
for MaxRects and **off by default for grid packs**, and that asymmetry is the interesting part.

Rounding a grid atlas up buys nothing. A grid pack already has a predictable shape, so the rounding
just adds dead space — and near a size boundary it can nearly double the image. It was measured at
1.6× wasted area on average and 3.6× at worst, which is why the default flipped.

The pane gives you the instrument to see this: a line under the atlas size comparing the packed area
to the total pixels of the sources. At the current defaults a grid pack reliably comes out *smaller*
than its inputs. Turn power-of-two back on and that number can go the other way — an atlas larger
than the pile of images that went into it.

It is worth doing on purpose once, so that the line means something to you afterwards.

One related corner: with an explicit column count, power-of-two rounding can leave dead space past
the last column, and Tiled would then derive a different column count from the rounded image than
the pack actually used. Packwright refuses the `.tsx` in that case by name rather than writing a
tileset that reads back wrong. The PNG and JSON are still written — a tileset-geometry mismatch says
nothing about whether those two are trustworthy.

## Getting things out

`Ctrl+Shift+E` exports the atlas, its JSON sidecar, and a `.tsx` if it is a grid pack. Every file for
one export is staged first and moved into place only once all of them exist, so a failure partway
through never leaves you with a new PNG beside a stale sidecar.

`Ctrl+E` exports to the library instead, which mints an ordinary asset — and keeps the document
beside it, so **Open in Packwright** on that card later gives you back the real editable atlas rather
than a flattened picture.

`Ctrl+S` saves the document itself as `.wpack`. The atlas and the layout are never stored in it, only
derived, which is why an unchanged save is byte-identical.

## Shortcuts

`R` repacks now — packing is automatic, so this is a convenience. `Delete` removes the selected
source. `Ctrl+Z` / `Ctrl+Y` undo and redo. `Ctrl+N`, `Ctrl+O`, `Ctrl+W` and `Ctrl+Tab` handle
documents and tabs. `Ctrl+0` fits, `Ctrl+1` goes to 100%. Middle-drag pans and the wheel zooms.

## Try it

1. `Ctrl+N`, then drag in several PNGs.
2. Pack them as MaxRects, note the atlas size, then switch to grid and note it again.
3. On the grid pack, turn power-of-two on and watch the "packed to N% of source" line cross 100%.
   Turn it off again.
4. Set extrude larger than half the padding and read the refusal.
5. Export with the Array schema, then with Hash, and diff the two JSON files.
6. `Ctrl+E` to the library, then **Open in Packwright** from the new card and confirm your sources
   came back.

## What to read next

[A character sprite sheet](11-a-character-sprite-sheet.md) — Troupe, which is the most ambitious
thing in the app and the one with the most honest caveats.

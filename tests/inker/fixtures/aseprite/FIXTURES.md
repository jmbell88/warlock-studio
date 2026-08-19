# The Aseprite fixture corpus

Two kinds of file can live here, and **the difference is the entire value of
the directory**, so every entry below is labelled with which it is.

**Aseprite-authored.** Written by Aseprite itself and checked in as a golden.
This is what the corpus is *for*: `test_aseout.py` already proves our writer
and our reader agree with each other on documents built in Python, which
cannot catch either of them spelling something in a way real Aseprite would
reject, or a field real Aseprite writes that our reader has never seen. A
round trip over one of these is evidence about Aseprite.

**aseout-synthesized.** Written by our own `aseout.aseprite_bytes`, from a
document built by hand in Python. A round trip over one of these is evidence
about *us*: it proves the code path runs, is stable under a second trip, and
matches the model that produced it — and it proves nothing whatsoever about
whether real Aseprite would open the file or would have written the same
bytes for the same picture. Useful — a regression in our own writer or reader
still fails it — but it must never be quoted as compatibility.

> **Every file in this directory is currently aseout-synthesized.** No human
> has yet opened one of our exports in real Aseprite, or authored a fixture
> in the app for our reader to prove itself against. That pass is a debt this
> repository cannot pay on its own — it needs a human with Aseprite
> installed; see "What is owed" at the end. This is exactly the state the
> Tiled corpus (`tests/plotter/fixtures/tiled/`) was in before its own
> human-authored pass, and this file follows its conventions for the same
> reason: the gate only strengthens when a human with the app has looked.

## Building an Aseprite-authored fixture

- Aseprite **1.3.x** (the current stable release line as of this corpus;
  record the exact build number in the fixture's heading below when one is
  built), official download, default settings.
- Build the sprite in the app, then **Save As** a `.aseprite` file directly
  into this directory. Nothing is exported or converted — the format this
  corpus tests is the native one.
- Keep canvases tiny — 4x4 to 8x8 is plenty, matching the synthesized
  fixtures beside it. These are checked into git and read on every suite run.
- Match one of the matrix cells below (a still RGB document, an indexed
  document with a repeated palette colour, a slice with a pivot, a tilemap
  layer, and so on) so the new fixture displaces a synthesized one of the
  same shape rather than adding an unrelated case.

A fixture built this way replaces the synthesized one of the same stem:
delete the synthesized file, drop the Aseprite-authored one in with the same
stem, and move its entry below from "aseout-synthesized" to
"Aseprite-authored". Nothing else changes — `MANIFEST` and the gate test key
on the stem, not on how the bytes were produced.

One thing does change for an Aseprite-authored fixture and not for a
synthesized one: **it is read only, never rewritten and reread for the fixed-
point check** the way a synthesized fixture is (see `test_aseprite_corpus.py`
for why a synthesized fixture's committed bytes are a fixed point rather than
a builder's raw output). An Aseprite-authored file instead wants a *read,
write, read* pair — parse it, write what our own encoder makes of that model,
parse that back, and assert the second model equals the first — because its
own bytes are never going to be *our* fixed point; only a subsequent write of
ours is.

## The manifest

`_asecorpus.py` carries `MANIFEST`, the stems the gate requires. Adding a
fixture means adding its stem there in the same commit — a file in this
directory that nothing lists is a file nothing tests.

## The fixtures

Each entry is one `.aseprite` file.

### `rgb-still` — **aseout-synthesized**
4x4, two layers: an opaque `Background` and an `Ink` layer at 0.6 opacity,
`multiply` blend, hidden and locked. The baseline still-document case — no
animation, no groups, no palette.

### `rgb-animated-linked-tags` — **aseout-synthesized**
4x4, two tracks over three frames. `Background` is one linked cel across all
three frames; `Ink` is painted differently per frame. Two tags cover
overlapping spans with different `direction`s (`pingpong`, `reverse`) and one
carries a finite `repeat`. Exercises the frame grid, the link chunk, and the
tag chunk's direction/repeat byte together.

### `grayscale-animated` — **aseout-synthesized**
8x8, two frames, converted to grayscale *after* an eraser stroke was drawn on
each frame. The eraser cuts alpha and leaves the colour it was drawn in
behind it, so both frames carry the funnel-painted shape this corpus exists
to pin: dead RGB sitting under alpha 0, which a grayscale cel's two-channel
storage (`value`, `alpha`) can only preserve on the visible half of the
picture. The other half of this same construct is `tilemap-indexed`, below,
where the same shape shows up in a tileset strip instead of a raster cel.

### `indexed-duplicate-colours` — **aseout-synthesized**
4x4, indexed. The palette holds **two identical browns** in different slots
and a transparent index that is not slot 0. Every pixel is placed by hand
across all four slots, so the round trip has to keep each pixel in the slot
it started in rather than collapsing the duplicate pair — the whole reason
index planes exist rather than pixels being re-quantised on the way out.

### `indexed-transparent-nonzero` — **aseout-synthesized**
4x4, indexed, four colours, `transparent=2` — the hole is neither slot 0 nor
the last slot, which is the header byte a writer could most easily get
off-by-one on.

### `groups-nested` — **aseout-synthesized**
4x4, three layers, the top two grouped and that group nested inside a second
group (`Art` inside `Ink`). The inner group is hidden and locked; each layer
is painted a distinct colour so a cel-index shift from the two group rows
would be visible in the wrong layer's pixels, not just in the names.

### `slices-pivot-ninepatch` — **aseout-synthesized**
6x6, three frames, two slices. `Whole` carries both a pivot and a nine-patch
centre and is never keyed. `Head` carries **no** base pivot but frame 1's key
does — the key-only-pivot case: the format stores pivot presence once per
slice, not per key, so the value a key alone introduces has to become the
slice's own for every unkeyed frame, never a fabricated zero. Frame 2's key
returns `Head` to its base rectangle, which is the run-length case where a
key exists purely to end the previous one early.

### `tilemap-rgb` — **aseout-synthesized**
8x8 at 4x4 tiles, RGB, two frames. A raster `Background` under a `Tiles`
tilemap layer bound to a three-tile strip. Frame 0 places a plain tile and a
horizontally-flipped one; the tilemap cel is linked into frame 1, which then
adds a diagonally- *and* vertically-flipped placement on a second row — so
the fixture carries a link, a flip combination, and a grid grown past its
first placement all at once.

### `tilemap-indexed` — **aseout-synthesized**
8x8 at 4x4 tiles, indexed, one frame. The strip is painted with a soft
(0.2-hardness) eraser stroke in **auto** tile-behaviour, which is what
reaches the tileset with the funnel-painted alpha shapes this corpus pins:
the fixture asserts its own strip carries a partially-erased, sub-255-alpha
pixel before it is written, so the committed file is guaranteed to exercise
`index_plane.resolve`'s alpha-first placement rather than an incidentally
clean strip.

### `spare-tileset` — **aseout-synthesized**
`rgb-still` plus one tileset (`spare`, two colours) that no tilemap layer
ever binds to. A tileset the user has built but not placed yet is document
state, not garbage, and this is the fixture that would catch a writer
dropping it because no layer chunk points at it.

### `palette-constrained-rgb` — **aseout-synthesized**
`rgb-still` plus a four-colour palette set on an RGB document (not indexed
mode). **A pinned loss, not a defect** (`ASEPRITE_PARITY.md` divergence 19):
the chunks are written — a file Aseprite opens has its colour table — but
`document_from_aseprite` installs a palette only at indexed depth, so the
read halfway back drops the constraint. That means this fixture's committed
bytes are **not** this document's first-pass output; they are the *second*
pass — see `test_aseprite_corpus.py`'s docstring for why every synthesized
fixture's committed bytes are a write-read-write fixed point rather than a
builder's raw bytes, and this is the one fixture in the corpus where that
distinction is load-bearing rather than incidental.

## What is owed

An Aseprite-authored corpus, which only a human with the app installed can
produce. In rough order of what it would buy, each replacing the
synthesized fixture of the same shape:

1. **A still RGB document with layer opacity and a non-normal blend mode** —
   the `rgb-still` shape. The single most valuable one to author, for the same
   reason `basic-ortho` tops the Tiled list: it is the plainest file and the
   one most likely to surface a spelling our writer and reader silently agree
   on between themselves and nowhere else.
2. **An indexed document with a duplicate palette colour** — the
   `indexed-duplicate-colours` shape, to confirm real Aseprite's index-plane
   storage is bit-for-bit what `asein._decode_indices` assumes.
3. **An animated document with a tag and a linked cel** — the
   `rgb-animated-linked-tags` shape.
4. **A tilemap layer with flipped placements** — the `tilemap-rgb` shape,
   which is the one fixture whose gid flip bits ride the top three bits of
   every cell and are invisible until the map is open in an editor.
5. Once any Aseprite-authored fixture opens one of our exports without
   complaint, the debt blockquote above comes out and each displaced entry's
   heading changes from "aseout-synthesized" to "Aseprite-authored".

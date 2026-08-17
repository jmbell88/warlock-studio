# The Tiled fixture corpus

Two kinds of file live here, and **the difference is the entire value of the
directory**, so every entry below is labelled with which it is.

**Tiled-authored.** Written by **Tiled 1.12.2 itself** and checked in as a
golden. This is what the corpus is *for*: every other test in `tests/plotter/`
builds a document in Python and asserts our reader agrees with our writer,
which cannot catch Tiled spelling something in a way we never thought to emit.
A round trip over one of these is evidence about Tiled.

**Synthesized.** Written by our own exporter, or by hand against our own
reader. A round trip over one of these is evidence about *us*: it proves the
code path runs and is stable, and it proves nothing whatsoever about what Tiled
does with the file. Useful — a regression in our own encoder still fails — but
it must never be quoted as compatibility.

> **Every map fixture in this directory is currently synthesized.** The
> corpus was empty before Wave 2; these four pairs were produced by this
> editor. The rule this file used to state — "never synthesize a file here" —
> was the right rule and was overtaken rather than repealed. It now reads:
> *never synthesize a file here and label it Tiled-authored*, and the
> Tiled-authored corpus is a debt this repository cannot pay on its own. It
> needs a human with Tiled 1.12.2 installed; see "What is owed" at the end.

`docs/PLOTTER_COMPAT.md` marks the rows that rest only on a synthesized
fixture, so the ledger and this file cannot drift into disagreeing about what
has been proved.

## Building a Tiled-authored fixture

- Tiled **1.12.2**, official download, default settings.
- Save each map twice: once as `.tmx` (File → Save As) and once as `.tmj`
  (File → Export As → JSON map file). Both go in this directory, same stem.
- Tilesets are **external** (`.tsx` + `.png` in this directory), which is what
  our exporter writes and what the loaders resolve.
- Keep atlases tiny — 2×2 tiles at 16×16 is plenty. These are checked into git
  and read on every suite run.
- In Preferences → General, leave "Export files as read-only" off.

A fixture built this way replaces the synthesized one of the same stem: delete
the synthesized pair, drop the Tiled pair in, and move its entry below from
"Synthesized" to "Tiled-authored". Nothing else changes — the manifest and the
tests key on the stem.

## The manifest

`_corpus.py` carries `MANIFEST`, the stems the gate requires. Adding a fixture
means adding its stem there in the same commit — a file in this directory that
nothing lists is a file nothing tests. `MANIFEST` currently holds `basic-iso`,
`core-112`, `oblique-112` and `typed-embedded-112`.

`basic.png`, `big.png`, `blob.png` and `iso.png` are atlases the fixtures and
several unit tests load directly; they are not stems and are not in `MANIFEST`.

## The fixtures

Each entry is one map, saved as both `.tmx` and `.tmj`.

### `core-112` — **synthesized**
Orthogonal, 4×4. Combines recursive groups, an image layer, layer
class/tint/offset/parallax/blend fields, all eight object shapes, object
rotation and opacity, index draw order, map class/parallax origin/render
order/background colour, flipped gids, and a recursively nested list property.
`core-1.12.tsx` is its external atlas.

Several of those constructs are **Warlock dialect** rather than Tiled features
— layer blend modes, object opacity, the `capsule` shape and the `list`
property type. See the dialect section of `docs/PLOTTER_COMPAT.md`. This
fixture therefore *cannot* be re-authored in Tiled as it stands; a
Tiled-authored replacement would cover the Tiled half and a separate
synthesized fixture would keep the dialect half honest.

### `basic-iso` — **synthesized**
Isometric, 8×8, 32×16 tiles, one tileset, one tile layer with a few tiles
painted. Isometric is the projection that left the refusal list, so this is
exactly the fixture that most wants to be a real Tiled golden and is not one
yet. Nothing in it is dialect, so it can be re-authored as-is.

### `oblique-112` — **synthesized**
Oblique, 4×3 at 16×16 with `skewx=8`, `skewy=-2`, and left-down render order.

**Tiled has no oblique orientation and no `skewx`/`skewy` map attributes.**
This is dialect from top to bottom and cannot be authored in Tiled at all. It
is a fixture about our own affine projection and its negative-coordinate
bounding box, and it is honest as long as nothing quotes it as compatibility.

### `typed-embedded-112` — **synthesized**
Orthogonal, 2×2, with an embedded atlas and properties authored at map, layer,
tileset and object levels. Carries every scalar kind, an object reference, a
file path, a class value, and a recursively nested typed list — the last of
which is dialect. The JSON class deliberately has no custom type name, because
such names live in a Tiled project schema this corpus does not use.

## What is owed

A Tiled-authored corpus, which only a human with Tiled installed can produce.
In rough order of what it would buy:

1. **`basic-ortho`** — orthogonal, 8×8, 16×16 tiles, one external tileset
   (`basic.png`), two tile layers (`Ground`, `Detail`), `Detail` at 0.5
   opacity, and on `Detail` at least one horizontally flipped tile, one
   vertically flipped and one diagonally flipped (`X`/`Y`/`Z` while stamping).
   The flip bits are the top three of every cell and a lost one is invisible
   until the map is in an engine. This file was described in this document
   before it existed; it still does not exist, and it is the single most
   valuable one to author.
2. **`basic-iso`** — re-author the existing synthesized pair in Tiled.
3. **A Tiled-authored `core`** — everything `core-112` covers *minus* the
   dialect constructs listed under its entry.
4. Once any Tiled-authored fixture opens one of our exports without complaint,
   `TILED_VERSION` in `src/warlock/studio/plotter/tsx.py` may move to `1.12.2`
   and the gate paragraph in `docs/PLOTTER_COMPAT.md` comes out.

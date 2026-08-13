# The Tiled fixture corpus

Files in this directory are authored in **Tiled 1.12.2 itself** and checked in
as goldens. That is the whole point of them: every other test in
`tests/plotter/` builds a document in Python and asserts our reader agrees
with our writer, which cannot catch Tiled spelling something in a way we never
thought to emit.

**Never synthesize a file here.** A fixture written by our own exporter and
checked in as a golden records our assumptions and calls them Tiled's.

## The build

- Tiled **1.12.2**, official download, default settings.
- Save each map twice: once as `.tmx` (File → Save As) and once as `.tmj`
  (File → Export As → JSON map file). Both go in this directory, same stem.
- Tilesets are **external** (`.tsx` + `.png` in this directory), which is what
  our exporter writes and what the loaders below resolve.
- Keep atlases tiny — 2×2 tiles at 16×16 is plenty. These are checked into git
  and read on every suite run.
- In Preferences → General, leave "Export files as read-only" off.

## The manifest

`_corpus.py` carries `MANIFEST`, the stems the gate requires. Adding a fixture
means adding its stem there in the same commit — a file in this directory that
nothing lists is a file nothing tests.

## The fixtures

Each entry is one map, saved as both `.tmx` and `.tmj`.

### `basic-ortho`
Orthogonal, 8×8, 16×16 tiles. One tileset (`basic.tsx`, a 2×2 atlas of 16×16
tiles). Two tile layers, `Ground` and `Detail`; paint a handful of tiles on
each, and on `Detail` include at least one horizontally flipped tile, one
vertically flipped, and one diagonally flipped (the `X`/`Y`/`Z` keys while
stamping). Set `Detail`'s opacity to 0.5. Flips are the reason this fixture
exists: they live in the top three bits of every cell and a lost bit is
invisible until the map is in an engine.

### `basic-iso`
Isometric, 8×8, 32×16 tiles, one tileset, one tile layer with a few tiles
painted. Isometric is the projection that left the refusal list, so it needs a
golden that is not ours.

### `two-tilesets`
Orthogonal, 6×6. Two external tilesets with different tile sizes (16×16 and
32×32). Paint from **both** onto one layer. This is the firstgid fixture: the
second set's ids start above the first's, and getting that wrong is silent.

### `objects-rect-point`
Orthogonal, 6×6. One tile layer and one object layer holding: a named
rectangle with a class set, a point, and a rectangle with `visible` unchecked.
Give one object a custom property. Do **not** add ellipses, polygons,
polylines, text, tile objects, or any rotation — those are refused today, and
they arrive as fixtures in M3 when they stop being.

### `typed-props`
Orthogonal, 4×4, one tile layer. Custom properties of every type Plotter
models today — `string`, `int`, `float`, `bool`, `color` — at all three
levels: on the map, on the layer, and on the tileset. Include a `float` with a
long decimal expansion (e.g. `0.30000000000000004`) — that value is why the
comparator rounds.

### `locked-layers`
Orthogonal, 4×4, two tile layers, one of them locked and one hidden. Lock and
visibility are document state, not view state, and this proves we read both.

### `blob-terrain`
Orthogonal, 8×8, one tileset carrying **one** Wang set shaped exactly as
Plotter's blob preset (47 tiles for one terrain colour). Paint a few cells
with the terrain brush. This is the fixture that pins the recognise-or-refuse
boundary at the shape we actually accept — a second colour or a differently
shaped set is an M4 fixture, not this one.

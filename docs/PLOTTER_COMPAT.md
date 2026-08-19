# Plotter ↔ Tiled compatibility

Target: **Tiled 1.12.2**. This ledger records document semantics, not byte
spelling: Plotter canonicalizes layer data to CSV and bundles external assets
under collision-free paths.

**The `tiledversion` this build writes is `1.10.2`** — see
`src/warlock/studio/plotter/tsx.py`'s `TILED_VERSION` — and it moves to
`1.12.2` only once a real Tiled 1.12.2 has been confirmed to open one of our
exports without complaint. Until then the two numbers disagreeing is expected,
not a bug. This gate was briefly deleted and the constant bumped in the same
change; the bump was reverted rather than the gate satisfied, because nothing
in this repo can satisfy it — it needs a human with Tiled installed.

**Two kinds of positive row, and the difference matters.** A `round-trips` row
is a claim about *Tiled*: the feature is one Tiled has, and a file carrying it
survives the trip in both directions. A `warlock-dialect` row is a claim only
about this editor: Plotter reads and writes the construct, no Tiled release
does, and a `.tmx`/`.tmj` carrying one is a file only Plotter opens. The
dialect rows are listed together in their own section at the end; they exist
because the document model grew features Tiled has no spelling for, and the
alternative — inventing syntax and *calling* it Tiled — is what the
`tiledversion` gate above exists to stop. `.wmap` is the format that holds all
of them without qualification.

States mean:

- **round-trips** — read, modeled and written without semantic loss, against
  Tiled;
- **warlock-dialect** — modeled and written, but no Tiled release reads it
  back; see "Warlock dialect" below;
- **refused** — stopped by name before a partial document can be edited;
- **preserved-verbatim** — retained for export but not interpreted by Plotter;
- **silently-dropped** — compatibility debt. There are no rows in this state.

The refused rows are checked in both directions against every
`TiledUnsupported` site. Positive rows are checked for a real fixture pair by
`tests/plotter/test_compat_matrix.py`; the corpus then exercises Tiled XML →
Plotter → Tiled XML, Tiled XML → Plotter JSON, and `.wmap` round trips.

**What the corpus does and does not prove.** A fixture pair proves the code
path runs and is stable across the trip. It proves compatibility with Tiled
only when the fixture was *authored in Tiled*.

**At present no fixture in the corpus is Tiled-authored** — every map under
`tests/plotter/fixtures/tiled/` was produced by this editor, so every
`round-trips` row below is currently a round trip *against ourselves*. That is
worth having and it is not the claim the word makes on its own, which is why
it is said once here rather than appended to thirty rows.
`tests/plotter/fixtures/tiled/FIXTURES.md` labels each fixture and lists what
authoring is owed. As Tiled-authored fixtures land, this paragraph shrinks to
name the rows still waiting.

Tiled 1.12.2 itself writes native `version="1.10"` while identifying the
writer with `tiledversion="1.12.2"`. Plotter follows that spelling rather than
inventing a `1.12` format-version value.

## The `M{n}` citations

A milestone number used to be cited as `M{n}` from comments under
`src/warlock/studio/plotter/`, from `docs/INVARIANTS.md` and from the table
below. **They referred to `docs/PLOTTER_PLAN.md`, which was deleted in
`09c64b4`** — chase it with `git log --all --diff-filter=D --
'*PLOTTER_PLAN.md'`, the same way a `TODO.md §N` citation is chased. This is the
plotter's instance of the rule `CLAUDE.md` already states for the deleted
roadmap: the numbering was a citable API while the file existed, so a citation
is left pointing at history rather than renumbered, and **no new `M{n}` citation
is minted**. Write what the deferred work is instead of a number for it.

**No `M{n}` is cited from code any more.** The last one was `M5`, infinite
(chunked) map storage, and it shipped: those comments now say what arrived
rather than what was owed. The seams it had held open are worth naming because
each did its job — `project.Lattice`'s `stagger_axis`/`stagger_index`/`hex_side`
fields, `.wmap`'s reserved `infinite` key, `scene.resolve` never asking a layer
for a dense `(h, w)` rectangle, and the two `WmapUnstorable` handlers that keep
a writer-door refusal from reaching the frame thread as a crash. The one
reservation deliberately left unused is `.wmap`'s `chunks`-beside-`data` layer
entry: the document holds a dense window plus an origin, so an infinite map's
tile layer is the same one array as any other's.

## Maps

| Feature | State | Notes |
|---|---|---|
| `orthogonal projection` | round-trips | Geometry, picking and rendering; fixture: `core-112`. |
| `isometric projection` | round-trips | Tiled object-coordinate conversion and depth order; fixture: `basic-iso`. |
| `oblique projection` | warlock-dialect | `orientation="oblique"` with `skewx`/`skewy`. **Tiled has no oblique orientation.** Modeled, drawn and written, including negative skew; fixture: `oblique-112`. |
| `map class and parallax origin` | round-trips | Map class plus both parallax-origin components; fixture: `core-112`. |
| `renderorder` | round-trips | All four orthogonal/oblique orders are modeled and rendered; fixture: `oblique-112`. |
| `backgroundcolor` | round-trips | Preserved and painted by the flat renderer; fixture: `core-112`. |
| `a {} map` | refused | An orientation outside the five this places. The list is `project.PROJECTIONS` itself, so the reader and the placement arithmetic cannot disagree about it. |
| `staggered and hexagonal maps` | round-trips | Both offset lattices place cells, and resolve a click by **exact containment** rather than by an affine inverse -- which is what the refusal they replaced demanded ("named rather than projected approximately"). `staggeraxis`, `staggerindex` and `hexsidelength` are read, written and stored. fixture: `hex-112`. |
| `an infinite map` | round-trips | No fixed rectangle: painting past the edge grows the map, cells may sit at negative coordinates, and both spellings read and write `<chunk>`/`chunks` in CSV and base64 with every compression. **The editor holds a dense window plus an origin rather than sparse chunks** -- chunking is translated at this codec's door and never leaks into the document, which is why every tool, both renderers and the terrain engine were untouched by this. Growth is an ordinary undoable `resize`; the origin rides that same step. fixture: `infinite-112`. |
| `chunked (infinite) layer data` | round-trips | The JSON spelling of the row above, and read and written by the same two functions (`tmx.chunks_from`/`chunks_of`). Empty chunks are dropped on write, which is the format's own shape and what makes an erase shrink a file. fixture: `infinite-112`. |
| `hexagonal 120-degree tile rotation` | refused | The hex-only fourth gid transform bit, and the one refusal on this page that is **not** waiting on effort. Two things block it. First, `gid.GID_MASK` is `0x1FFFFFFF` because that is *Aseprite's* default tile-id mask and the Inker writes it verbatim into `.aseprite` files (`aseout.py`), where the identity remap is what makes the output a file real Aseprite reads -- so Tiled's narrower `0x0FFFFFFF` cannot simply replace it; the plotter would need its own mask. That part is solvable. The second is not: **a 120° rotation of a square raster is not a permutation of the pixel grid.** `render.orient` is slices and transposes only, deliberately -- the standing bar for a tile transform is that it invents no colour -- and drawing the rotation faithfully in an export would mean resampling there. A canvas that rotated while the export did not would be a far larger lie than the parallax and animation-frame disagreements this editor already states. Re-open it if the flat renderer ever gains a resampler, or if Tiled's own hex tiles move to a lattice a permutation can turn. |

## Layers

| Feature | State | Notes |
|---|---|---|
| `tile layers` | round-trips | Persistent ids, visibility, lock, opacity and flagged gids; fixture: `core-112`. |
| `recursive group layers` | round-trips | Nested order and inherited decorations; fixture: `core-112`. |
| `image layers` | round-trips | Images, repeat flags and stacking; fixture: `core-112`. |
| `layer class, tint, offset and parallax` | round-trips | Common fields on every layer kind; fixture: `core-112`. |
| `layer blend modes` | warlock-dialect | A `mode` attribute on a layer. **Tiled has no per-layer blend mode.** The names and the compositing arithmetic are the W3C/SVG ones, so an engine that implements them agrees with our flat renderer; fixture: `core-112`. |
| `object-layer draw order and color` | round-trips | Both `topdown` and `index`, plus editor outline color; fixture: `core-112`. |
| `layer tile coordinates` | refused | Deprecated nonzero tile-space layer x/y cannot be confused with pixel offsets. |
| `an image layer transparent colour` | refused | Deprecated color-key transparency is named instead of discarded. |
| `layer data encoded as {}` | refused | XML, CSV and base64 read; unknown encodings stop. |
| `{}-compressed layer data` | refused | Raw, zlib, gzip and **zstd** read; anything else stops. zstd is read-only -- every Tiled reads zlib, so writing it would buy nothing and cost a reader. |
| `{} layers` | refused | Unknown JSON layer kinds stop by their Tiled type name. |

## Objects

| Feature | State | Notes |
|---|---|---|
| `rectangle and point objects` | round-trips | Geometry, visibility, ids, class and properties; fixture: `core-112`. |
| `ellipse objects` | round-trips | The `<ellipse/>` tag and its JSON `ellipse: true`; fixture: `core-112`. |
| `capsule objects` | warlock-dialect | A `<capsule/>` tag beside `<ellipse/>`. **Tiled has no capsule shape.** Modeled, hit-tested, drawn and written; fixture: `core-112`. |
| `polygon and polyline objects` | round-trips | Ordered floating-point vertices; fixture: `core-112`. |
| `tile and text objects` | round-trips | Gid transforms and complete Tiled text styling fields; fixture: `core-112`. |
| `object rotation` | round-trips | Tiled's clockwise degrees about the object origin, editable and undoable; fixture: `core-112`. |
| `object opacity` | warlock-dialect | An `opacity` attribute on an object. **Tiled has per-*layer* opacity, not per-object.** Modeled, editable, undoable and written; fixture: `core-112`. |
| `object templates` | refused | Templates are an explicit project/workflow non-goal. |

## Tilesets and terrain

| Feature | State | Notes |
|---|---|---|
| `external atlas tilesets` | round-trips | Atlas slicing, margin, spacing, firstgid and TSX properties; fixture: `core-112`. |
| `embedded atlas tilesets` | round-trips | Map-local atlas definitions and properties are read and modeled without loss. **Written back as an external `.tsx`**, not re-embedded: both exporters emit the portable `tilesets/` bundle for every tileset, so an embedded atlas comes home beside the map rather than inside it. Semantics survive; the embedding does not. fixture: `typed-embedded-112`. |
| `tileset class` | round-trips | Custom class names on external and embedded atlases; fixture: `core-112`. |
| `tileset object grid` | round-trips | Orthogonal/isometric collision-authoring grid metadata; fixture: `basic-iso`. |
| `tileset transformations` | round-trips | Allowed flips/rotation and untransformed preference; fixture: `core-112`. |
| `an image-collection tileset` | round-trips | Per-tile images with sparse ids, composed into one backing atlas on the way in -- `ids` and `sizes` keep every fact the composition could lose, and an oversized tile draws at its own size anchored bottom-left in **both** renderers. fixture: `collection-112`. |
| `an embedded tileset image` | refused | Embedded image payloads or missing source paths are not decoded. |
| `tileset image transparent colour` | round-trips | Applied at decode, so nothing downstream sees the key colour. The deprecated image-layer twin stays refused. fixture: `presentation-112`. |
| `an external .tsj tileset` | round-trips | Read through the same JSON tileset definition an embedded one uses; which spelling a reference names is the host's question. fixture: `tsj-112`. |
| `Wang sets / terrain brushes` | round-trips | Generic corner/edge/mixed sets are read as data and painted by wangid constraint matching (ties by colour probability, then lowest id; **no match leaves the cell untouched** rather than writing a near-miss). The blob preset is still recognised first and keeps its positional terrain rows and a byte-identical export. fixture: `wang-112`. A blob set exported with phase variants (`phases > 1`, declared by an int `phases` tileset property) writes every phase sub-row with the same wangid per case: Tiled's terrain brush treats equal wangids as random alternatives and keeps working, at the accepted cost that it *randomises* phases where Plotter derives them from cell coordinates. Exported layers carry concrete gids per cell, so position-baked phases travel exactly. |
| `terrain types` | refused | Deprecated pre-Wang terrain syntax. |
| `per-tile animation` | round-trips | Ordered frames of local ids and durations; the canvas plays them and every export draws frame 1. fixture: `tilemeta-112`. |
| `per-tile collision shapes` | round-trips | Rect, ellipse and polygon outlines, stored on the tile and drawn in the collision editor. Never hit-tested against the map: collision is metadata an engine reads. fixture: `tilemeta-112`. |
| `per-tile custom properties` | round-trips | The same typed property model layers, objects and the map use. fixture: `tilemeta-112`. |
| `per-tile class` | round-trips | Read, written and editable under the tileset palette. fixture: `tilemeta-112`. |
| `per-tile probability` | round-trips | Weights a random brush; 0 is never chosen at random and always placeable by hand (Tiled's rule). fixture: `tilemeta-112`. |
| `per-tile terrain assignment` | refused | Deprecated terrain indices are not inferred as Wang data. |
| `tileset object alignment` | round-trips | The tile-object anchor, through `project.object_to_pixels`; fixture: `presentation-112`. |
| `tileset render size` | round-trips | Grid-sized or own-size drawing of an oversized tile; fixture: `presentation-112`. |
| `tileset fill mode` | round-trips | How a grid-fitted tile fills its cell; fixture: `presentation-112`. |
| `tileset background colour` | round-trips | Palette presentation, preserved on the trip; fixture: `presentation-112`. |
| `tileset tile offset` | round-trips | A draw offset in the canvas and the flat renderer; the minimap ignores it by its one-pixel-per-cell rule, as it does layer offsets. fixture: `presentation-112`. |

## Properties

| Feature | State | Notes |
|---|---|---|
| `scalar, file and object properties` | round-trips | Tiled scalar values, paths and persistent object ids; fixture: `typed-embedded-112`. |
| `recursive class properties` | round-trips | XML keeps self-describing member types; JSON keeps the values its schema contains; fixture: `typed-embedded-112`. |
| `recursive list properties` | warlock-dialect | A `list` property type, spelled as nested `<item>` elements in XML and typed records in JSON. **Tiled has no list property type**; its eight are string, int, float, bool, color, file, object and class. Modeled recursively, including lists inside classes and inside other lists; fixture: `typed-embedded-112`. |
| `a custom property of type {}` | refused | Types outside the eight Tiled kinds and the one dialect kind stop by name. |

JSON class members are bare values in Tiled's map format; their declared member
types live in a project schema. Projects are out of scope, so Plotter infers
the JSON-native bool/int/float/string/container kind while preserving every
value. XML class members are self-describing and retain their exact types.

## Permanent non-goals

| Feature | State | Notes |
|---|---|---|
| `projects and worlds` | refused | Plotter opens portable maps rather than Tiled workspace graphs. |
| `plugins and JavaScript extensions` | refused | No extension runtime is embedded. |
| `object templates and Automapping` | refused | Authoring workflows, not core map document data. |
| `custom exporter APIs` | refused | Plotter exports its built-in portable formats. |

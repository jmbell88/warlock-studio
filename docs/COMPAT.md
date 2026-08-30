# Interop ledgers

Two foreign file formats this app reads and writes, and exactly what each
direction of each trip keeps. **Plotter ↔ Tiled** is the first part;
**Inker ↔ Aseprite** is the second. Each part carries its own vocabulary of
states, because the two trips lose things in different shapes — Tiled's ledger
is about whether a *construct* survives, Aseprite's about what a *document*
drops on the way out and on the way back in — and one merged state list would
blur them.

**The rule both parts obey, stated once.** A green test proves this app's
reader and this app's writer agree with *each other*. It does not prove the
real application agrees with either of them: a round trip through our own two
halves cannot catch an error both halves make together. So every positive claim
below is a claim about this editor until a human with the real app has opened
one of our exports or authored a fixture for our reader to prove itself
against. `TODO.md` holds the passes that are owed, and the two fixture
inventories — `tests/plotter/fixtures/tiled/FIXTURES.md` and
`tests/inker/fixtures/aseprite/FIXTURES.md` — name what to author first.

**One part is executable and the other is prose, and the difference is
load-bearing.** `tests/plotter/test_compat_matrix.py` parses the Tiled tables
below *as data*: six tests read their rows and check the `refused` ones in both
directions against every `TiledUnsupported` site in the engine, so a row edited
carelessly fails the suite. That parser is scoped to this file's Tiled part by
name — the Aseprite tables use a different state vocabulary and would parse as
unknown states — so a heading added under `## Plotter ↔ Tiled` joins the gate
and one added under `## Inker ↔ Aseprite` does not. The Aseprite half has no
corpus-as-data test of that kind; the fixed-point gate
`tests/inker/test_aseprite_corpus.py` runs beside it instead.

## Plotter ↔ Tiled

Target: **Tiled 1.12.2**. This ledger records document semantics, not byte
spelling: Plotter canonicalizes layer data to CSV and bundles external assets
under collision-free paths.

**The `tiledversion` this build writes is `1.12.2`** — see
`src/warlock/studio/plotter/tsx.py`'s `TILED_VERSION` — and it says what it
means as of **2026-08-29**. It was held at `1.10.2` behind a gate that only a
human with Tiled installed could open: a real Tiled 1.12.2 had to be confirmed
to open one of our exports without complaint. (The gate was briefly deleted and
the constant bumped in the same change; that bump was reverted rather than the
gate satisfied.) **Both directions have now been exercised against Tiled
1.12.x**, and each covers a different half:

- **Plotter → Tiled.** A Plotter-exported `.tmx` — 150×150 orthogonal, three
  CSV tile layers, one external tileset under `tilesets/` — was opened and
  worked on in Tiled 1.12.x. That is the gate, and it is what moved the
  constant.
- **Tiled → Plotter.** A map Tiled 1.12.2 itself wrote — 640×360 orthogonal,
  one CSV layer, *two* external `.tsx` tilesets with a `firstgid` split — reads
  here without complaint, into the right dimensions, layer and tileset
  structure.

**What that does not cover, and it is most of the table.** Both files are
orthogonal CSV ground and nothing more: no flipped tile in either (checked —
zero cells with a transform bit set), no objects, no properties, no groups, no
infinite chunks, no image collection, no Wang set, and no `.tmj` on the Tiled
side at all. So the two bullets above are a claim about the map header, the
external-tileset reference and the CSV layer payload. Every other
`round-trips` row below still rests on a fixture this editor wrote, which is
the paragraph after next.

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
authoring is owed, and `TODO.md` carries the pass itself. As Tiled-authored
fixtures land, this paragraph shrinks to name the rows still waiting.

The 2026-08-29 verification above did **not** change this: it was done against
files living outside this repository, so it moved the `tiledversion` gate — a
claim about one number — without adding a golden the suite can re-check. A
verification nothing re-runs decays; a fixture does not. That is why the corpus
debt is still open with the gate closed.

Tiled 1.12.2 itself writes native `version="1.10"` while identifying the
writer with `tiledversion="1.12.2"`. Plotter follows that spelling rather than
inventing a `1.12` format-version value.

### The `M{n}` citations

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

### Maps

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

### Layers

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

### Objects

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

### Tilesets and terrain

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
| `per-tile collision shapes` | round-trips | Rect, ellipse and polygon outlines, stored on the tile. Authored in the **tileset editor's Collision tab** (Tileset > Edit tileset), one tile drawn large with its shapes over it, and written through the same undoable `set_tile_meta` the class and probability fields use. A box or an ellipse is added covering the whole tile, which is the one obviously editable starting size. Shapes authored in Tiled survive the round trip intact. Never hit-tested against the map: collision is metadata an engine reads. Two things Tiled's collision editor can author are **not modeled and are dropped at import, by name into the log**: point/polyline collision members, and a shape's `rotation`. fixture: `tilemeta-112`. |
| `per-tile custom properties` | round-trips | The same typed property model layers, objects and the map use. fixture: `tilemeta-112`. |
| `per-tile class` | round-trips | Read, written and editable under the tileset palette. fixture: `tilemeta-112`. |
| `per-tile probability` | round-trips | Weights a random brush; 0 is never chosen at random and always placeable by hand (Tiled's rule). fixture: `tilemeta-112`. |
| `per-tile terrain assignment` | refused | Deprecated terrain indices are not inferred as Wang data. |
| `tileset object alignment` | round-trips | The tile-object anchor, through `project.object_to_pixels`; fixture: `presentation-112`. |
| `tileset render size` | round-trips | Grid-sized or own-size drawing of an oversized tile; fixture: `presentation-112`. |
| `tileset fill mode` | round-trips | How a grid-fitted tile fills its cell; fixture: `presentation-112`. |
| `tileset background colour` | round-trips | Palette presentation, preserved on the trip; fixture: `presentation-112`. |
| `tileset tile offset` | round-trips | A draw offset in the canvas and the flat renderer; the minimap ignores it by its one-pixel-per-cell rule, as it does layer offsets. fixture: `presentation-112`. |

### Properties

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

### Permanent non-goals

| Feature | State | Notes |
|---|---|---|
| `projects and worlds` | refused | Plotter opens portable maps rather than Tiled workspace graphs. |
| `plugins and JavaScript extensions` | refused | No extension runtime is embedded. |
| `object templates and Automapping` | refused | Authoring workflows, not core map document data. |
| `custom exporter APIs` | refused | Plotter exports its built-in portable formats. |

## Inker ↔ Aseprite

Target: the **Aseprite file-format specification** as published upstream in the
`aseprite/aseprite` repository, as *ase-file-specs* under its own `docs/`
directory rather than anything in this one; read and
written against the chunks Aseprite 1.3.x itself writes (old-format palette
`0x0004` included, for files older than that). This ledger records what each
direction of the trip drops, not byte spelling — `asein.py` reads the format,
`aseout.py` writes it, and this part is the explicit lossy-interop report
Wave 5 of the Aseprite parity programme promised alongside them.

**Two readers, two writers, one ledger.** ORA is this editor's native format
and the direction Aseprite interop travels *through*: an `.aseprite` file
opens into the same `Document` an `.ora` does, and Save As can now put that
document back into either format (`manual/28-inker.md#saving`). The two
tables below are therefore about the two directions a document can leave
Inker's own model — **ORA → aseprite** is what `aseout.py` drops writing a
document out; **aseprite → ORA** is what `asein.py` drops reading one in —
and both are read against the same `Document`, which is why a construct lost
on the way in (say, a colour profile) never shows up as a loss on the way
out: it was never modeled to lose in the first place.

States mean:

- **dropped** — silently absent from the far side; nothing warns because
  there is no warning channel out of a pure `bytes → Document` or
  `Document → bytes` function (see `aseout.py`'s own module docstring for why
  the write direction has no toast to raise).
- **warned** — the reader opens the file and returns a message alongside the
  document (`asein.document_from_aseprite`'s second return value); whoever
  opened the file decides how to say it. `sheetin`'s precedent, restated.
- **refused** — the reader stops before a partial or misleading document can
  be edited, naming the thing by name.
- **n/a on write** — nothing to lose because this side never has the
  construct to begin with.

Every row cites its `docs/INVARIANTS.md` divergence number where one exists.
Not every row has one: a divergence number marks a standing decision about
this editor's *document model*, cited by code and tests across the package,
where several of the rows below are narrower — a single field's mapping
between two file formats — and are recorded here instead. That is the same
argument Wave 5 made for the two divergences it did number: a decision about
the model earns a number, a decision about one field's spelling earns a row.

**Two surfaces here are riskier than the rest**, and they are named so the
owed pass knows where to look first. The **tilemap and tileset chunks** had
their field order written from the *reader*, inverted field for field, and it
has never been checked against a file Aseprite itself wrote — the one error a
round trip through our own two halves structurally cannot catch. And the
**derived palette chunk** (#23) changes the bytes of every RGB and grayscale
file this build writes; the corpus proves it stable and lossless here, not that
Aseprite likes the table it finds. `TODO.md` names the three fixtures that
settle the first question in a minute.

### ORA → aseprite (what `aseout.py` drops writing a document out)

| What | State | Divergence | Notes |
|---|---|---|---|
| Cel opacity | dropped | #1 | Opacity is a track/layer property here; every cel writes at 255. |
| Cel z-index | n/a on write | #12 | This build has none — track order *is* stack order, so there is nothing to write per cel. |
| User data (layer/cel/tileset/tile) | dropped | #14 | Not modeled; nothing is written to the `0x2020` chunk this format offers for it. |
| Per-frame palettes | n/a on write | #20 | One table per document; there is only ever one palette to write. |
| Colour profile | dropped | #3 | No `0x2007` chunk is written; a real Aseprite opening the file assumes sRGB, which is what this editor already assumes throughout. |
| Group opacity | dropped | — | Aseprite's UI offers a group none, and `asein._group_tree` hands every group back `1.0` whatever byte is stored — any value but 255 would be a number nobody could ever read again, so every group row writes opacity 255. |
| `Track.alpha_lock` | dropped | — | An editing aid, not picture data; the format has no bit for it. |
| An empty group | dropped | — | A group is a *run* of the layer list here, so one with no members has no run to write (`_install_groups` prunes the same shape on read). |
| a background layer | round-trips | -- | **Divergence #6 is retired** (6.5). A background layer is a flag on the bottom layer, written into the layer chunk's own flags (`0x08`) and into `stack.xml` as `warlock-background`, and read back from both. What it means is that the layer composites opaque -- so erasing on it reveals the colour under the eraser rather than a hole. `Document.matte` remains as the *stand-in* for a document that has no background layer, and converting one folds the matte into the pixels and clears it: the flatten-time overlay becomes a layer every format can store. |
| the flatten matte (`Document.matte`) | dropped | -- | `.aseprite` has no field for what a flattened export puts behind transparency, so a document saved here and reopened *infers* one with `matte_for`: white if every pixel is opaque, off otherwise. The user's own answer survives only in `.ora`, as `warlock-matte` on the `<image>` root -- and that attribute is the one `warlock-*` attribute written **unconditionally**, because the setting is tri-state (on / off / nobody said) and absence has to keep meaning "infer" for Krita files and for files written before it existed. Sheet imports infer for the same reason. |
| a reference layer | round-trips | -- | Read since the reader landed (opened hidden, because Aseprite's own export omits one) and **kept as a layer type** since 6.5 rather than folded into `visible`: `Layer.reference` is written into the chunk flags (`0x40`) and into `stack.xml` as `warlock-reference`, and it refuses every tool write through the door the content lock already uses. |
| Palette-constrained RGB's own palette | dropped | #19 | The chunks *are* written — a file Aseprite opens carries its colour table — but the constraint itself has nowhere to live in the format; see the aseprite → ORA row below for why re-opening it does not bring the constraint back either. Pinned, not fixed: `tests/inker/fixtures/aseprite/palette-constrained-rgb.aseprite` in the corpus. |
| Grayscale storage | normalized | #2 | `(v, v, v, a)` writes as the format's own `(value, alpha)` pair — lossless for every *visible* pixel; see the next row for the one place it is not. |
| Dead colour under a grayscale pixel's alpha 0 | dropped | #2 | The funnel deliberately leaves whatever colour an eraser stroke exposed alone rather than rewriting it (a no-op write should stay a no-op), so an invisible pixel's RGB is real per-channel data this format's two-channel storage cannot carry — it is written as its own red channel alone and reads back `(v, v, v, 0)`. |
| A document with no palette of its own | derived, not omitted | #23 | Aseprite writes a colour table into every file it saves and this writer used to omit the chunk entirely when `doc.palette` was empty. It now writes one built from the document's own pixels — every entry a colour actually painted somewhere in the file, ranked by pixel count, capped at 256 and emitted in colour order; a document with no visible pixel gets the single transparent entry. **Nothing is invented**: writing Aseprite's own default table instead would mean reciting thirty-two colours from memory, which is the unmeasured claim this repository refuses to make. Indexed documents are unaffected — there a missing palette is a refusal, never a derivation. |
| RGBA tileset strips in an indexed document | resolved, exact-match | — (Wave 3 divergence, unnumbered) | Every strip this package stores is RGBA regardless of document colour mode; an indexed document's strip is resolved back through its palette on the way out (`index_plane.resolve`'s own rule), exact match only — a strip pixel with no slot to place it in refuses by name rather than being nearest-matched into somebody else's atlas. |
| A slice key's pivot/nine-patch *presence*, per key | widened, never invented | — | The format declares presence once per slice, not per key (`_slice_chunk`'s `_first_set`): a key that lacks what the chunk declares inherits the *first* value the slice carries anywhere — its own where it has one — and the zero branch is never reached, so nothing is fabricated. What is lost is the distinction between "this key has no pivot" and "this key has the slice's pivot"; both read back to the same rectangle through `Slice.at`. |
| A slice's fractional pivot | rounded | — | The format's field is a signed DWORD; a fractional pivot loses at most half a pixel. |
| A tag's ping-pong-**reverse** direction | narrowed to ping-pong | — | This document's own `DIRECTIONS` model has three values (`forward`/`reverse`/`pingpong`), not Aseprite's four — a document can never *hold* a ping-pong-reverse tag to write, because `asein` already opens one as ordinary ping-pong with a warning (see the matching row below). Not a Wave 5 decision; recorded here because it is the writer's mirror of that reader behaviour. |
| A `loop=False, repeat=0` tag ("play once", the timeline Loop menu's own "once") | translated to `repeat=1` | #16 | This model's own zero (`Tag.repeat`) means "the loop flag decides"; Aseprite's own zero means "forever", and `asein._read_tags` hard-codes `loop=True` on the way back in — so a bare 0 would round-trip a tag set to play once into one that never stops. `aseout._tags_chunk` writes Aseprite's own "play once" spelling (`repeat=1`) instead, which reads back `loop=True, repeat=1`: different field values, but `animation.advance` forces `loop` True under any positive repeat anyway and still stops after the count, so playback is identical on both sides of the trip. `loop=True` tags are unaffected — their `repeat` byte is written verbatim. |
| A TilemapCel's tileset binding, in an indexed document, if a strip pixel is genuinely unrepresentable | refused by name | — | The one hole `index_plane.resolve` and `indexed.snap` disagree about: a visible colour only the *transparent* slot holds. Recorded as a refusal rather than a drop because nothing invented could be honest here — see `aseout.py`'s own module docstring for the full argument. |

### aseprite → ORA (the `asein.py` reader's warning table)

| What | State | Divergence | Notes |
|---|---|---|---|
| Cel opacity | warned | #1 | "per-cel opacity is not kept; the layer's opacity is." |
| Cel z-index | warned | #12 | "a cel's z-index was dropped; layer order is stacking order." |
| User data (layer or tag/timeline colour) | warned | #14 | "user data and timeline colours are not kept; the drawing is" — raised once for a layer's own user-data chunk and again for a tag whose colour bytes are non-zero. |
| Per-tile user data | warned, tiles kept | #14 | Its own, narrower sentence ("the tiles are") — the picture survives; only the metadata about individual tiles does not. |
| Per-frame palettes | warned | #20 | "per-frame palettes are not kept; the final table is used." |
| Colour profile | warned | #3 | "a colour profile was dropped; this app assumes sRGB." |
| A reference layer | warned, visibility as stated | #6-adjacent | Aseprite's own export leaves a reference layer out; this reader keeps the pixels and the layer. It used to force the layer hidden as well, on the grounds that Aseprite opens them that way — but `aseout` writes `visible` verbatim beside the REFERENCE flag, so a file can say one is showing, and the override silently discarded the user's own toggle on the next load. The VISIBLE bit now wins; only the warning remains. |
| An unknown blend mode | warned, opens as normal | — | A future Aseprite mode this build has no number for falls back to `normal` rather than refusing the whole file. |
| A tileset's `base_index != 1` | warned | — | Aseprite's own display-only numbering in its tileset panel; no id this reader stores is affected. |
| A cel's precise (cropped) bounds | warned | — | "a cel's precise bounds were dropped; its pixels were not" — this package's cels are always canvas-sized, so a tight Aseprite cel is expanded to the canvas, matching `aseout`'s own full-canvas write convention. |
| References to external files (an external-link tileset) | dropped, refused | — | An external-file tileset is **refused by name** rather than merely warned about: its pixels are not merely elsewhere in this reader's model, they are not in the file at all, and a tilemap layer bound to it would draw nothing. |
| A saved mask or path (a selection) | dropped | — | "a saved mask or path was dropped; selections do not travel" — Aseprite selections are session state this format happens to persist; this reader has no selection-on-disk concept to receive it into. |
| An unrecognised chunk type | warned | — | Names the chunk kind (`0x{kind:04x}`) so a future format addition is visible rather than silently absorbed. |
| A cel on a group layer | warned, dropped | — | "a cel on a group layer was dropped; a group holds no pixels" — malformed input; a group row has nowhere to put pixels. |
| A linked cel drawn at its own offset | warned, unlinked | — | Aseprite shares a cel's *position* along with its pixels; a link whose declared offset disagrees with its source is unlinked into an independent copy so nothing draws in the wrong place. |
| A background layer painted in the transparent index, with a full palette | warned | — | The read path duplicates the transparent slot to give the background layer somewhere unambiguous to point at; when the palette is already full at 256, there is no slot to spare and the pixels read back as ordinary transparency instead. |
| A slice starting partway through the timeline | warned | — | "the slice starts partway through the timeline; it is shown from the first frame here" — this model's slices exist from frame 0. |
| A slice hidden on some frames (Aseprite's zero-size key) | warned, stays visible | — | There is no "hidden" state for a slice here — a slice is a note about the drawing, not part of it — so the rectangle from the zero-size key is kept as-is and the user is told. |
| A file declaring a size other than its own | warned | — | "this .aseprite declares a size other than the file's; it may be truncated" — an integrity signal, not a modeled construct. |
| A file with no drawable (non-group) layers | warned, opens empty | — | Becomes one empty `Background` layer rather than a refusal, matching every other empty-document convention this package already has. |
| A tag's ping-pong-**reverse** direction | warned, opens as ping-pong | — | This document's `DIRECTIONS` has no fourth value; the tag opens playing ordinary ping-pong from its near end instead of its far one. |
| A tag over a document's only frame | warned, dropped | #22 | New at Wave 5: a one-frame file opens as a **still** document (`Document.anim is None`), which has nowhere to hold a tag at all — "there is nothing to play." |
| Palette-constrained RGB's write-constraint (reading it back) | dropped | #19 | An RGB-depth file's palette chunk is a real colour table Aseprite wrote, but installing it as a *constraint* on an ordinary RGB document would silently put the whole editor into palette-locked mode over a table nobody asked to be limited by — so it is read and then set aside; only an *indexed*-depth file's palette is installed. |
| Everything refused outright | refused, named | — | A colour depth this build does not read; a file with no frames; a canvas smaller than 1×1; a cel that will not decompress; a tilemap cel that is not 32 bits per tile or whose bit-mask offset is not tile-aligned; a cel linking to a frame that holds none; a cel type nobody here knows. Each names the thing rather than failing generically — `sheetin`'s own argument for refusing a mis-registered atlas, restated. |

## Inker ↔ ORA (Krita, GIMP, MyPaint)

`ora.py` is both halves — reader and writer — so unlike the Aseprite pair there
is no asymmetry to tabulate, only what the format carries that this document
model has nowhere to put. Everything here is a **round-trip loss**: open in
Warlock, save, reopen in Krita, and the column is gone for good. That is worth
stating plainly rather than leaving in code comments, because "the pixels are
fine" is true of every row and is not the question a user asks.

| What | State | Notes |
|---|---|---|
| Canvas resolution (`xres`/`yres`) | **kept** | Read into `Document.dpi` and written straight back. Carried, not used: nothing here renders at a physical size. A document that never stated one still does not — no guessed 72 is invented on first save. This was a silent loss until the 2026-08-23 audit. |
| A group's `composite-op` (blend mode on a group as a unit) | dropped | The document model has no group-level blend: a group is a membership fact, not a compositing stage. A Krita group set to Multiply therefore renders pass-through on load — visibly wrong before anything is saved — and the attribute is gone after. The drop is logged. Modelling it is a real feature (`composite.py` would need to composite a group's members into a scratch plane first), not a line of parsing. |
| ICC colour profiles | dropped | This app is sRGB throughout — the same non-goal the Aseprite table records for colour profiles, stated here so its absence is not read as an oversight. Pixel bytes are unchanged; a colour-managed viewer will reinterpret them against sRGB. |
| Foreign per-layer attributes (Krita `uuid`, `colorlabel`, `selected`, `collapsed`; GIMP `edit-mask`) | dropped | Editor state about a layer rather than the layer, and there is no round-trip guarantee for any of it. Unlike the group case these are dropped without a log, deliberately: they appear on every layer of every Krita file, so logging them would be noise on every open rather than a signal. |
| `isolation` | written as `auto` always | The writer states the ORA default rather than preserving what it read, for the same reason as `composite-op`: nothing in the model distinguishes an isolated group from a pass-through one. |

# Plotter ↔ Tiled compatibility

**Target: Tiled 1.12.2.** (The `tiledversion` this build actually writes into
an export is still `1.10.2` — see `src/warlock/studio/plotter/tsx.py`'s
`TILED_VERSION` — and moves to `1.12.2` only once a real Tiled 1.12.2 has been
confirmed to open one of our exports without complaint. Until then the two
numbers disagreeing is expected, not a bug.) One row per feature, each in
exactly one state:

- **round-trips** — read, modelled, and written back without loss. The note
  names the corpus fixture that proves it, written as `` fixture: `stem` ``
  (several, comma-separated, if more than one fixture backs the row) — that
  marker, not any other backticked word in the note, is what
  `tests/plotter/test_compat_matrix.py` looks for when it checks the fixture
  exists.
- **refused** — the reader stops by name and says what to remove. Never
  half-loaded: a feature silently dropped on read is a feature deleted on the
  next save, and the user finds out when the map is already gone.
- **preserved-verbatim** — carried through a round trip but not honoured by
  the editor. Written back exactly as it arrived.
- **silently-dropped** — read, but neither modelled nor written back. This is
  the state the other three exist to make unnecessary, and every row under it
  is a debt: something Tiled says about a map that this editor currently
  forgets between open and save. See "Read but not modelled" below.

This table is checked by `tests/plotter/test_compat_matrix.py`, in both
directions, against the `TiledUnsupported` strings in
`src/warlock/studio/plotter/`. A refusal with no row fails the suite, and so
does a row for a refusal that no longer exists — which is how a parity
milestone is forced to update this file in the same commit that changes the
behaviour. `silently-dropped` rows are not checked against a refusal — there
is none to check, precisely because nothing stops the reader for them — so
they are asserted only by inspection; keeping them honest is on whoever edits
this file.

Feature names are the refusal's own words, normalised: an interpolated part of
the message is written `{}`, so one row covers one refusal rather than one row
per value it can name.

## Maps

| Feature | State | Notes |
|---|---|---|
| `a {} map` | refused | Staggered and hexagonal. Orthogonal and isometric are drawn; see M5. |
| `an infinite map` | refused | Fixed-size maps only; see M5. |
| `chunked (infinite) layer data` | refused | The JSON spelling of the same thing. |
| `hexagonal 120-degree tile rotation` | refused | The gid bit that only a hex map can set. |

## Layers

| Feature | State | Notes |
|---|---|---|
| `group layers` | refused | Flatten in Tiled first; see M2/M3. |
| `image layers` | refused | See M2/M3. |
| `{} layers` | refused | Any layer kind the JSON reader does not model. |
| `layer pixel offsets` | refused | See M2/M3. |
| `layer data encoded as {}` | refused | CSV and base64 are read; anything else is refused. |
| `{}-compressed layer data` | refused | zlib and gzip are read; zstd is not. |

## Objects

| Feature | State | Notes |
|---|---|---|
| `object templates` | refused | See M7. |
| `tile objects` | refused | See M3. |
| `rotated objects` | refused | An unrotated outline drawn for a rotated object is a wrong picture. |
| `ellipse objects` | refused | See M3. |
| `polygon objects` | refused | See M3. |
| `polyline objects` | refused | See M3. |
| `text objects` | refused | See M3. |

## Tilesets

| Feature | State | Notes |
|---|---|---|
| `an image-collection tileset` | refused | One sliced atlas per tileset; see M4. |
| `an embedded tileset image` | refused | An `<image source=…>` path is required. |
| `an external .tsj tileset` | refused | Re-save as `.tsx`; see M4. |
| `Wang sets / terrain brushes` | refused | One blob-shaped set is recognised; anything else is refused. See M4. |
| `terrain types` | refused | Tiled's pre-1.5 spelling. |
| `per-tile animation` | refused | See M4. |
| `per-tile collision shapes` | refused | See M4. |
| `per-tile custom properties` | refused | See M4. |

## Properties

Eight of Tiled's nine property types are modelled: `string`, `int`, `float`,
`bool`, `color`, `file`, `object` and `class` — the last three since M2, with
`class` recursive and its `propertytype` (the name of a class or enum declared
in a Tiled *project*, which Plotter never reads) carried verbatim. They are
**not** listed as `round-trips` because that state's note has to name a corpus
fixture that proves it, and the corpus is still empty — `FIXTURES.md` forbids
synthesizing one from our own exporter, so the row would have to name a file
that does not exist. What backs them today is
`tests/plotter/test_props.py`, `test_tsx.py` and `test_tmx_refusals.py`
(read → export → re-read, in both syntaxes); the rows below move to
`round-trips` with a `fixture:` marker as soon as a Tiled-authored fixture
carrying them lands.

The losses, stated rather than hidden — both of them Tiled's **JSON** only, and
both from one cause: a `.tmj` writes a class property's members as bare values
inside the parent's value object rather than as property records.

1. **Member types.** Tiled recovers them from the project's
   `propertytypes.json`, which Plotter does not read, so a `color` or a `file`
   member of a class comes back from a `.tmj` as a `string`.
2. **A nested class's `propertytype`.** The type name is an attribute of a
   property *record* and a member is not one, so Tiled's schema has nowhere to
   put it. The outermost class keeps its name; every class nested inside it
   comes back with an empty one, its own members intact.

The XML spelling writes a real `<property>` per member — with its `type` and
its `propertytype`, at every depth — so a `.tmx` loses neither. Both are pinned
by `tests/plotter/test_props.py`
(`test_a_colour_member_of_a_class_returns_as_a_string_through_json`,
`test_a_class_nested_in_a_class_loses_its_type_name_through_json`).

| Feature | State | Notes |
|---|---|---|
| `a custom property of type {}` | refused | Any type outside Tiled's nine — `file`, `object` and `class` no longer refuse. |
| `a list-valued custom property` | refused | Modelled in the document and stored in `.wmap`; refused at the Tiled door because Tiled 1.12.2 has no list-valued property to write it as. See M2. |

## Preserved but not honoured

| Feature | State | Notes |
|---|---|---|
| `renderorder` | preserved-verbatim | Written back as it arrived; the renderer draws right-down. M5 honours it. |
| `backgroundcolor` | preserved-verbatim | Round-tripped; not painted. |

## Read but not modelled

These attributes are present in a Tiled file and the reader parses far enough
to see them, but nothing in `MapDoc`, `TileLayer`, `ObjectLayer` or
`MapObject` carries a place to put them, so they never reach the document and
are gone the moment the map is exported again. This is not the `refused`
state — there is no name-and-stop for any of these, so a map holding them
loads cleanly and looks, until compared closely with the original, like it
loaded completely. `PLOTTER_PLAN.md` § Milestone 2 is where each of these
gets an actual home: layer decorations (`parallaxx`/`parallaxy`,
`tintcolor`), `class` and `id` on both layers and objects.

`draworder` is the one entry here that is not merely lost but actively wrong
on a round trip: `tmx_export` writes every object layer's `<objectgroup>`
with `draworder="topdown"` regardless of what the source file said, so a
Tiled map authored with `draworder="index"` (objects drawn in list order
rather than sorted by `y`) changes what it *means* — not just what it
carries — the moment it passes through this editor, even though nothing
about the shapes or their positions changed.

| Feature | State | Notes |
|---|---|---|
| `layer parallaxx` / `layer parallaxy` | silently-dropped | Per-layer parallax factor. See M2. |
| `layer tintcolor` | silently-dropped | Per-layer colour multiply. See M2. |
| `layer class` | silently-dropped | Tiled's per-layer custom type. See M2. |
| `layer id` | silently-dropped | Tiled's own layer id; a fresh one is minted on every export. See M2. |
| `object id` | silently-dropped | Read only far enough to name the object in a refusal message; never stored. A fresh one is minted on every export. See M2. |
| `object-layer draworder` | silently-dropped | Not read at all, and always written back as `"topdown"` -- an `"index"`-ordered layer changes meaning across the round trip, not just loses an attribute. See M2. |

## Permanent non-goals

| Feature | State | Notes |
|---|---|---|
| `oblique projection` | refused | Not a Tiled feature. Tiled 1.12.2 has four orientations and oblique is not one of them, so there is nothing to be compatible with. |

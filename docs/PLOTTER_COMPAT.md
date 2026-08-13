# Plotter ↔ Tiled compatibility

**Target: Tiled 1.12.2.** One row per feature, each in exactly one state:

- **round-trips** — read, modelled, and written back without loss. The note
  names the corpus fixture that proves it.
- **refused** — the reader stops by name and says what to remove. Never
  half-loaded: a feature silently dropped on read is a feature deleted on the
  next save, and the user finds out when the map is already gone.
- **preserved-verbatim** — carried through a round trip but not honoured by
  the editor. Written back exactly as it arrived.

This table is checked by `tests/plotter/test_compat_matrix.py`, in both
directions, against the `TiledUnsupported` strings in
`src/warlock/studio/plotter/`. A refusal with no row fails the suite, and so
does a row for a refusal that no longer exists — which is how a parity
milestone is forced to update this file in the same commit that changes the
behaviour.

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

| Feature | State | Notes |
|---|---|---|
| `a custom property of type {}` | refused | `file`, `object`, `class` and `list`; see M2. |

## Preserved but not honoured

| Feature | State | Notes |
|---|---|---|
| `renderorder` | preserved-verbatim | Written back as it arrived; the renderer draws right-down. M5 honours it. |
| `backgroundcolor` | preserved-verbatim | Round-tripped; not painted. |

## Permanent non-goals

| Feature | State | Notes |
|---|---|---|
| `oblique projection` | refused | Not a Tiled feature. Tiled 1.12.2 has four orientations and oblique is not one of them, so there is nothing to be compatible with. |

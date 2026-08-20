# Inker ↔ Aseprite compatibility

Target: the **Aseprite file-format specification** as published in
`docs/ase-file-specs.md` of the `aseprite/aseprite` repository, read and
written against the chunks Aseprite 1.3.x itself writes (old-format palette
`0x0004` included, for files older than that). This ledger records what each
direction of the trip drops, not byte spelling — `asein.py` reads the format,
`aseout.py` writes it, and this file is the explicit lossy-interop report
Wave 5 of the Aseprite parity programme promised alongside them.

**Two readers, two writers, one ledger.** ORA is this editor's native format
and the direction Aseprite interop travels *through*: an `.aseprite` file
opens into the same `Document` an `.ora` does, and Save As can now put that
document back into either format (`docs/manual/09-inker.md#saving`). The two
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
between two file formats — and are recorded here instead, exactly as this
file's closing paragraph argues for the two Wave 5 introduced.

## ORA → aseprite (what `aseout.py` drops writing a document out)

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
| `Document.matte` (the background-layer stand-in) | dropped | #6 | Matte is a flatten-time overlay, not a stored layer; nothing Aseprite-side records it, because this build has no real background-layer type to carry the flag on. |
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

## aseprite → ORA (the `asein.py` reader's warning table)

| What | State | Divergence | Notes |
|---|---|---|---|
| Cel opacity | warned | #1 | "per-cel opacity is not kept; the layer's opacity is." |
| Cel z-index | warned | #12 | "a cel's z-index was dropped; layer order is stacking order." |
| User data (layer or tag/timeline colour) | warned | #14 | "user data and timeline colours are not kept; the drawing is" — raised once for a layer's own user-data chunk and again for a tag whose colour bytes are non-zero. |
| Per-tile user data | warned, tiles kept | #14 | Its own, narrower sentence ("the tiles are") — the picture survives; only the metadata about individual tiles does not. |
| Per-frame palettes | warned | #20 | "per-frame palettes are not kept; the final table is used." |
| Colour profile | warned | #3 | "a colour profile was dropped; this app assumes sRGB." |
| A reference layer | warned, opens hidden | #6-adjacent | Aseprite's own export leaves a reference layer out; this reader keeps the pixels but opens the layer hidden rather than dropping it outright — a cheaper way back than reopening in Aseprite. |
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

## The user-owed manual pass

Every green test above proves this reader and this writer agree with *each
other* — the automated gate is `aseout` writing a document, `asein` reading
it back, and the corpus at `tests/inker/fixtures/aseprite/` doing the same
over eleven documents rather than one (`FIXTURES.md` in that directory names
each fixture and what it exercises). None of that proves real Aseprite
agrees with either of them. That is exactly the shape
`docs/PLOTTER_COMPAT.md`'s `TILED_VERSION` rule states for Tiled interop, and
the same rule applies here without qualification: **this report's claims only
strengthen once a human with a real copy of Aseprite has opened one of this
editor's exports, or authored a fixture in the app for this reader to prove
itself against.** `tests/inker/fixtures/aseprite/FIXTURES.md`'s "What is
owed" section names the four fixtures worth building first, in priority
order, and states plainly that every file in the corpus today is
aseout-synthesized — written and read by this package alone, proving the
code path runs and is stable, and proving nothing about Aseprite itself.
Until that pass happens, every "round-trips" claim above is a claim about
this editor's own two halves agreeing with themselves.

**Two parts of the surface are riskier than the rest and are named here so the
manual pass knows where to look first.**

1. **The tilemap and tileset chunks.** Their field order was written from the
   *reader*, inverted field for field, and has never been checked against a file
   Aseprite itself wrote. A round trip through our own two halves cannot catch
   an order both halves get wrong together, and nothing else in this corpus can
   either. `tilemap-rgb`, `tilemap-indexed` and `spare-tileset` are the three
   fixtures that exercise it; opening any one of them in Aseprite settles the
   question in a minute, and is the single highest-value item on the owed list.
2. **The derived palette chunk (#23).** New in this pass, and it changes the
   bytes of every RGB and grayscale file this build writes. The round trip and
   the corpus both prove it is stable and lossless *here*; what they cannot
   prove is that Aseprite likes the table it finds — in particular that a
   1-entry palette on a blank document opens without complaint rather than
   replacing Aseprite's own default with a single swatch. Worth a look in the
   same sitting.

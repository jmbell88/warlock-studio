# Changelog

Hand-written, newest first. Nothing here is derived from git: the commit
subjects are `Warlock vN.N.N` and carry no detail, so this file is the only
record of what a version actually changed. The top heading's version must
match `pyproject.toml` — a test asserts it, so a release bump cannot leave this
file behind.

## 0.0.23 — 2026-08-16

- **Review judges in one guided pass.** **Start judging** at the top of the
  sweep list walks every unit that has no verdict yet, across every bucket, one
  at a time: `A` accepts, `R` rejects, `S` skips, `Esc` finishes early. The
  header counts your position in the whole run rather than in the sweep on
  screen. The eleven-point scale has not moved — the grade row and the digits
  keep working inside a pass, and a grade pressed there files and advances just
  as an accept does. When the pass ends a card reports what it did, per sweep
  and overall.
- **A fully judged sweep cleans itself up.** Once every unit of a sweep has a
  verdict, its images and meshes are removed automatically, with a toast and no
  dialog — the offer on the entry card is the warning. Every verdict,
  observation and finding survives, because each one carries its own copy of the
  settings it judged. This is the one place a delete overrides the rule that
  your accepted assets are never touched in bulk, and it applies whether the
  sweep was judged in a pass or one grade at a time.
- **The sweep form says what it wants.** Each axis is drawn as whatever the
  parameter actually accepts — tick boxes for a taxonomy field or a switch, a
  hint naming the legal range for a number — and a line under the form spells
  out what will be queued rather than showing a bare count. Every sweep in the
  list now says what it varied, so a run called "test2" is still legible a week
  later.
- **Inker: a zoom you can land on.** The wheel steps 5% at a time and rounds to
  that step first, so coming out of a fit at 83% takes you to 85 rather than to
  88, and 100% is reachable from either direction. The zoom stops at 25% and at
  1000%. A new **Center view** button puts the page back under the middle of the
  pane at the zoom you are already at — which is what **Fit view** cannot do.
- **Create: clearing what is on screen.** A **Clear** button on the viewport
  toolbar empties the canvas for the stage you are on and it stays empty;
  reselect the asset to bring it back. The Mesh settings gained the **Reset…**
  the image settings have always had.
- **Clay: Merge Faces and Union Objects.** *Merge Faces* is the face-mode
  dissolve under the name most people look for. *Union Objects* is new: a real
  boolean that removes the geometry inside an overlap, where *Merge Objects*
  welds and keeps it. A union costs the texture coordinates and turns quads into
  triangles, and needs every object to be a closed solid, which is why both
  operations are here.
- **Panels have room to breathe.** Every pane's content now sits inside the
  12 px inset it was always meant to have; it had been running flush to the
  column edge.
- **New map asks what the map is.** *New map...* now opens a setup dialog
  instead of silently making a 32 x 32 grid of 32 px orthogonal cells: three
  presets, then the projection, the map size in tiles and the tile size in
  pixels, with a line saying what that comes to overall and a note when an
  isometric cell is not 2:1. A last question — add a tileset from a file,
  generate a ground set, or nothing yet — means a new map can arrive paintable.
  All five doors go through it: the canvas, the map panel, Home, the command
  palette and `Ctrl+N`.
- **The map's tile size can be changed after the fact.** Under *Resize* in the
  tools pane, beside the grid fields. Cells are redrawn at the new size and
  objects scale with them; nothing painted is lost, because a tile keeps its
  identity whatever size the cell under it is. It is one undo step. The manual
  had claimed this worked for some time.
- **Fixed: the Resize button did nothing.** The *Resize* section and its
  *Resize* button had the same name, and imgui addresses an item by a hash of
  its label — so the button claimed the section's identity and never acted,
  which is what made a map look unresizable. Dear ImGui's own "conflicting ID"
  warning was the red panel some of you saw over the fields. Every Plotter pane
  is now held to one id per item by a test.
- **Fixed: three file pickers listed almost nothing.** Plotter's *Add from a
  file…* opened on a filter that claimed six formats and in fact showed only
  Tiled `.tsx` files, so a folder of PNGs looked empty. Plotter's *Open a map*
  could not see `.tmx` or `.tmj` at all, and Packwright's image picker showed
  only PNGs. All three now open on a row that lists everything it names, and
  Add a tileset offers narrower *Tiled tileset* and *Images* rows beneath it.
- **Fixed: seven blend modes made the whole drawing slow.** `exclusion`,
  `subtract`, `divide`, `hue`, `saturation`, `color` and `luminosity` were the
  only modes the fast compositor did not know, and because it composites a stack
  as one piece, a single layer in one of them put *every* layer back on the slow
  path. One `saturation` layer took a brush dab from 5 ms to 49 and a full
  recomposite from 0.26 s to 2.8. All seven are now composited natively, 4.6–7.7×
  faster, and the pixels are identical to what the old path produced.
- **Fixed: `divide` did its arithmetic at the wrong precision.** It was the one
  mode of nineteen computing in double rather than single, which made a divide
  layer's whole composite twice as wide for no benefit. Values move by at most
  one part in eight million — 27 bytes in 7.9 million changed in testing.
- **Converting to a palette with Floyd–Steinberg is ~70× faster.** It ran at
  about ten microseconds a pixel, so a 1024 x 1024 drawing took 10.6 seconds and
  a 2048 x 2048 one about 43 — long enough that changing the method in the
  preview looked like a hang. The same conversion is now 139 ms and 0.5 s. The
  dithering itself is unchanged, pixel for pixel.

## 0.0.22 — 2026-08-15

- **Plotter models most of what Tiled models.** Maps carry Tiled's richer
  property vocabulary — file paths, object references and nested classes
  round-trip through `.tmx`, `.tmj` and `.wmap` alike — and every layer and
  object keeps its persistent Tiled id across a round trip, so nothing gets
  renumbered behind an engine's back.
- **Layers form a real tree.** Groups nest, image layers carry a picture, and
  visibility, opacity, tint, pixel offsets and locks resolve through ancestors
  the same way in the editor and in an export. A lock on a group is a lock on
  everything inside it, including the tools.
- **Objects have real geometry.** Ellipse, polygon, polyline, text and tile
  shapes plus rotation and draw order are modelled; drawing them and Tiled
  interop for them arrive next. Rectangles and points behave exactly as
  before.
- **`.wmap` version 3.** The map file stores all of the above; version 1 and 2
  maps open unchanged. Anything a Tiled export cannot yet spell is refused by
  name rather than dropped in silence — the compatibility table in
  `docs/PLOTTER_COMPAT.md` says which door refuses what and why.
- **Two importer gaps closed.** A `.tsx`-less `.tsj` tileset named from a
  `.tmx` is refused with the right sentence, and per-tile features hidden in
  an embedded `.tmj` tileset no longer slip past the named refusals.

- **The timeline is edited as a range.** Drag across the grid to select a
  rectangle of cells — `Shift`+click extends it, `Esc` drops it — and the
  cell menu's **Range** section then acts on all of it at once: copy and paste
  cels (the clipboard is shared between tabs, and a link inside the copied
  block comes back as one drawing rather than as copies), clear, link, unlink,
  duplicate, reverse, delete, and set every frame's duration in one go. Each is
  a single `Ctrl+Z` and each is refused rather than half-applied. **Thumbs**
  draws each cel's picture in its cell, linked cels sharing one thumbnail, so a
  held background reads as the same drawing standing in several columns.
- **A preview pane that does not stop when you draw.** It plays the clip in the
  corner of the right column on a *second* playhead: it never locks the
  document, never moves the frame under your brush, and updates within a
  quarter second of each stroke — which is the thing an animator actually
  does, where the timeline's Play button ("watch this") was the only offer
  before. A **speed** multiplier from 0.25x to 4x and a **scope** — the whole
  clip, or the tag the preview is inside, with its direction, looping and
  repeat count. It always draws upright, ignoring a turned or mirrored canvas
  view, because those are aids for drawing and this is a check on the result.
- **Tags repeat a fixed number of times.** Set a tag's **repeat** to 3 and the
  span plays three times and stops; 0 — the default, and what every document
  written before this carries — leaves the Loop tick deciding as it always
  did. Playback stops *inside* the tag rather than falling through into the
  frames after it, which is a deliberate divergence from Aseprite. Counts are
  saved, ride into a sprite sheet's sidecar, and a tag exported on its own as a
  GIF carries its count into the file.
- **Filters run over a range.** With cells selected the filter popup gains
  **Apply to range** and runs over every cel in it as one undo step. A linked
  cel is filtered once however many frames it appears on, an empty cel stays
  empty rather than becoming a filtered blank, and a feathered selection fades
  the filter in on every frame at once.
- **The exports grew the options an engine actually asks for.** A whole-number
  **scale** box magnifies every export nearest-neighbour, so a 32×32 sprite at
  8× is the artwork at 256×256 rather than a blurred version of it — and a
  sheet's sidecar is built on the scaled size, so it describes the file that
  was written. **Export PNGs** writes one numbered frame per file for an engine
  with its own importer. **Export range → sheet / → GIF** and **Export tag →
  sheet / → GIF** write only part of a clip, renumbering the tags against the
  frames actually exported and dropping a directional layout, because half a
  walk sheet is a clip rather than a smaller walk sheet. **Import sprite
  sheet** goes the other way: any image, a cell size, an offset, the padding
  and a frame count become one frame per cell, with the arithmetic checked as
  you type instead of a short sheet imported in silence.
- **Slices, and they travel.** The `C` tool names a rectangle on the canvas —
  drag it, resize it, give it a name, a draggable **pivot** and a nine-slice
  centre, and key any of those per frame on an animated document. They carry
  through a flip, a turn, a scale, a crop and a canvas resize, and they ride
  into an exported sheet's sidecar, into Packwright's atlas and its
  TexturePacker JSON (in source-image coordinates, so trimming cannot move
  them), and into a `.wpack`. **Export slices as PNGs** cuts each one out on
  its own. They are stored in a member of the `.ora` written only when there
  are some, so a document with no slices saves byte-for-byte as it always did.
- **Seamless tiles are painted seamlessly.** **Tiled** has four positions —
  off, X, Y, X+Y — and draws the neighbouring tiles around the one you are
  working on, so all four seams are visible at once. Everything that lays down
  colour wraps with it: the brush, the eraser, the spray, the fill (a region
  running off one edge and continuing on the other is one region), the shapes
  and the magic wand. Three deliberately do not — smudge stops at the edge,
  blur wraps but does not read across the seam, and a gradient never wraps,
  because a ramp has two ends and joining them puts a hard edge exactly where
  this mode exists to remove one.
- **Three new ways to lay down colour.** The brush gains a **replace** ink that
  writes the colour exactly, alpha included, so it paints transparency *down*
  as well as up — what recolouring flat pixel art wants and what a normal
  brush cannot do at all. **Spray** (`A`) emits scattered dabs at a rate for as
  long as the button is held, its size being the width of the cloud. And
  **Shading** (`H`) paints no colour at all: it moves every pixel you drag over
  one swatch along the palette ramp you selected in the Colour panel, one step
  per stroke, leaving anything not on the ramp alone — which is how shading is
  actually done in pixel art.
- **Three clicked shapes and a polygonal lasso.** **Polyline** (`L`),
  **polygon** (`O`) and **curve** (`F`) drop a point per click, finish on a
  double-click or `Enter` and land as one undo step; the curve runs *through*
  every point rather than near it. **Poly lasso** (`D`) is the same gesture for
  a selection. The right mouse button now paints with the background colour on
  the brush, the eraser and the fill — the three where "the other colour" is
  unambiguous — and it stays inert everywhere else. The Move tool has a third
  answer as well: with nothing selected and nothing floating, dragging moves
  the whole active layer, one undo step, a linked cel moving on every frame it
  appears on.
- **A text tool.** `T`, click, type: a font, a size in pixels and an antialias
  switch that starts *off* on an indexed document. There are no text objects
  and no text layers — what you get is pixels, as a floating selection, so
  every tool and every filter applies with nothing to flatten first, and the
  other half of that trade is that re-editing text is retyping it (the box
  remembers what you last typed).
- **Image brushes and tool presets.** **Capture from selection** makes the tip
  of the tool in your hand out of part of your drawing — a tip rather than a
  tool of its own, so symmetry, the spray, tiled mode and the selection clip
  all come with it. Rotate and flip give the variants without resampling a
  pixel, **aligned** placing snaps dabs to a lattice so stamps tile, and a
  stroke never builds up on itself however slowly it is dragged. A **preset**
  stores one tool's options under a name between sessions — the colours, the
  grid and the symmetry deliberately not among them.
- **Layers got folders, a lock and animated merges.** **Group** wraps layers in
  a folder that folds visibility, opacity and the lock down onto its contents,
  composites pass-through, refuses to be made around layers that are not
  adjacent, dissolves when the last one leaves, and round-trips through
  OpenRaster's own nested stacks — so a folder made here opens as a group in
  Krita and vice versa. **Lock layer** refuses every tool on that layer while
  leaving renaming, hiding, reordering and deleting alone, and undo still works
  underneath it. **Merge down** and **Flatten** now work on an animated
  document, across every frame at once: a merge is worked out per pair of cels,
  so frames that shared a drawing go on sharing the merged one instead of each
  getting a copy.
- **Selections and transform go deeper.** `Ctrl+Shift+D` brings back the
  selection you last dismissed; dragging inside a selection moves its *edges*
  rather than its pixels; `Ctrl+J` and `Ctrl+Shift+J` promote a selection onto
  a layer of its own, copying or cutting. The transform box gains edge handles
  for non-uniform scale, a **slant** (an italic, in two degrees), and typed
  **X/Y/W/H/Angle/Slant** fields — because a drag cannot express "exactly 90
  degrees". **RotSprite** joins Smooth and Nearest as a resample mode: it turns
  pixel art on an eight-times lattice and samples back down, so a diagonal
  keeps its own colours instead of coming out as a staircase.
- **Seven more blend modes, four more filters.** `exclusion`, `subtract`,
  `divide`, `hue`, `saturation`, `color` and `luminosity` take the set to
  nineteen — the W3C formulas OpenRaster is defined against, with the two the
  W3C has no name for written under Krita's, so a file composites identically
  in Krita and GIMP rather than approximately. The filter popup gains
  **invert** (per channel), **replace colour** (a From, a To and a tolerance),
  **outline** (colour, thickness, inside or outside, rounded or square corners,
  and wrap for a tile) and **despeckle** (a median, so it deletes strays
  outright and leaves hard lines hard).
- **Palettes became a workflow rather than a switch.** **Convert...** builds a
  table out of a drawing's own colours (2 to 64) and shows you the result
  before you commit, with the dither picked from nearest, Floyd–Steinberg or
  an ordered 2×2/4×4/8×8 matrix; **Palette from an image...** takes any image's
  colours instead. The table itself now has multi-select, **Sort** by hue,
  saturation, brightness, a channel or usage (selected slots sorting in place,
  so one ramp straightens without the rest of the table moving), **Insert** for
  an interpolated run between two slots, **Count usage**, and export to a JASC
  `.pal` beside the GIMP `.gpl`. The gradient tool gained a **dither** of its
  own, which throws the blend away and gives every pixel one of the stops —
  the point being a ramp that lands only on colours you chose.
- **An Aseprite file opens here.** **Import Aseprite file** reads an
  `.aseprite`/`.ase` and rebuilds the layers with their opacities, blend modes
  and locks, the groups with their nesting, the frames with their durations,
  the tags with their spans, directions and repeat counts, the slices with
  their pivots and centres, and the shared cels as links. Reading only, and it
  shows: the import opens as an *unsaved* document, so the first `Ctrl+S` writes
  an `.ora` — nothing here can write back over the `.aseprite`, because a
  format read by one program and written by another is how a day's work goes
  missing. What cannot come across is told apart on purpose: anything that
  changes what the pixels mean (a tilemap layer, an unreadable colour depth) is
  a named refusal, and anything cosmetic (colour profiles, user data, per-cel
  opacity, a cel's z-index) is a message with the file still opening.
- **Tablet pressure was investigated and declined.** The pen route was priced
  properly rather than guessed at: pygame's vendored SDL2 exports no pen API at
  all, and the Windows Ink route is a `WndProc` subclass whose engine-side half
  is a brush-dynamics decision rather than two defaulted parameters. The
  velocity-driven taper already produces a pen-like stroke from a signal the app
  definitely has, so it stands, and nothing in the brush was touched.
  `docs/measurements/2026-08-15-tablet-pressure-spike.md` is the evidence and
  the order to revisit it in.

## 0.0.21 — 2026-08-11

- **The app can be driven from the keyboard.** Tab and Shift+Tab move between
  controls in every pane — Settings, Profiles, the library, the inspector, the
  mode switch — and Space or Enter operates the one you land on, with an
  accent-coloured ring showing where you are. Before, focus traversal existed
  in the 2D and 3D forms and nowhere else, while the shortcut sheet implied
  otherwise. The arrow keys stay with whatever binds them: Home and the library
  move their selection, Review steps units, Inker and Plotter pan on Space —
  those never also step the ring. Icon-only buttons now say their name when you
  arrive by keyboard, not just on hover.
- **Muted copy is readable.** Second-rank text was drawn with imgui's
  *disabled* styling — 3.20:1 against a dark panel and 2.55:1 on a light one,
  where body text needs 4.5:1. It is now drawn opaque, which clears the bar in
  both themes on every surface; disabled styling is reserved for controls that
  really cannot be operated. The contrast of every text and status colour is
  now checked by the test suite rather than by eye.
- **Typing works for everyone.** Text now comes from the operating system
  instead of being reconstructed from raw keypresses, so input methods,
  compose keys and dead keys produce what they should, and characters outside
  the Basic Multilingual Plane are no longer dropped. An input method's
  candidate window is placed against the field being typed into.
- **The window survives a move to another monitor.** Display scaling is
  re-read when the window changes display, and the interface — fonts included —
  is rebuilt at the new size without a restart. Your UI scale is re-applied
  against the new monitor, so a zoom that had to be capped on one display is
  offered in full again on a display with room for it.
- **Nothing is pushed off the edge at a high UI scale.** Three-column
  workspaces reserved two full-width sidebars and a 300px centre whatever the
  window size, which at 1.5x or 2x wanted more room than the window could be
  shrunk to — and the pane that fell off was always the inspector, for exactly
  the people who had enlarged the UI in order to read it. The sidebars now
  narrow first, then the centre, and every pane stays reachable.
- **Plotter edits the way Tiled does.** The tool letters are now Tiled's rather
  than the raster editor's — this is a tile-map editor, and Tiled is the one you
  are most likely arriving from. Fill moves to `F`, the old Rect tool becomes
  **Shape** on `P` and gains an ellipse mode, and `R` becomes **rectangular
  select**. `Ctrl+G` still toggles the grid and `Ctrl+S` still saves; a chord
  does not claim the bare letter.
- **A selection, and a clipboard.** Drag a marquee with `R`, `Ctrl+A` to take
  the whole map, `Ctrl+D` to drop it. Stamp, Erase, Fill and Shape all land only
  inside it. `Ctrl+C`/`Ctrl+X` copy and cut, `Delete` clears, and `Ctrl+V` loads
  the block into the brush and switches to Stamp — so pasting clips at the map
  edge, costs one undo step and obeys the selection, all rules the stamp already
  had. Pasting into a different map is refused by name, because tile numbers are
  per map and the block would come out silently redrawn.
- **The brush can be transformed before it lands.** `X` and `Y` mirror it, `Z`
  turns it a quarter clockwise (`Shift+Z` back). The arrangement *and* each tile
  in it move, so a flipped brush stamps a mirrored picture rather than a
  mirrored arrangement of unmirrored tiles.
- **Lines, and drags that no longer skip cells.** `Shift`+click stamps a line
  from the last cell painted, as one undo step. A fast drag now fills in the run
  between one frame's cell and the next instead of coming out dotted.
- **Layers can be locked.** The padlock beside the eye stops painting, erasing,
  cutting and object edits on that layer. It blocks *content* only: renaming,
  hiding, reordering, deleting the layer, copying from it and reading an
  object's properties all stay available. Locks are saved, and carried through
  `.tmx`/`.tmj`.
- **Custom properties have an editor.** Layers and the map itself have carried
  typed properties through every Tiled round trip since Plotter shipped; there
  was simply no way to set one without a text editor. Both are undoable.
- **Objects can be moved and resized on the canvas.** Drag the body to move,
  corner handles to resize — the opposite corner pins, and dragging past it
  flips the rectangle rather than going negative. `Ctrl` snaps to the grid, and
  a whole drag is one undo step.
- **A minimap** sits in the corner with a box showing where you are looking;
  click or drag it to jump. Built from one average colour per tile rather than a
  shrunk render, because a 512-square map composites to over 250 million pixels.
- Painting a large map got faster: which tileset owns a tile id is now memoised
  per document instead of being re-derived once per visible cell per layer.

- **Ground tile sets in Plotter.** Generate a tileset instead of loading one:
  name your terrains, pick a colour each, and get a full **blob autotiling**
  set — 47 cells per terrain, so every edge, outer corner and inner corner has
  its own tile. The new **Terrain** tool (`T`) paints a terrain and re-fits the
  eight cells around it, so coastlines and paths follow the brush instead of
  being placed one tile at a time. Order is precedence: where terrains meet the
  lower one gets the outline and the one above runs underneath unbroken, which
  is what makes a grass → sand → water beach resolve to one correct tile per
  cell even where all three touch. The base set is deliberately plain — flat
  fill, one-pixel darker outline — and generates identically every time, so a
  polished set can be diffed against the one it started from.
- **Polish an atlas in Inker and send it back.** **Polish in Inker** opens a
  tileset's atlas as an ordinary flat drawing — not sliced into cells, because
  keeping an outline consistent *across* neighbouring cases is the whole point
  of the pass. **Back onto...** returns it to the same tileset, and every
  painted cell keeps its tile and simply redraws. An atlas whose size changed
  is refused by name rather than quietly renumbering the map.
- **Isometric maps.** A map is drawn on one of two lattices. Cells are 2:1
  diamonds, the grid follows the lattice rather than the screen, and the status
  line shows the cell under the pointer — which is the only thing that reliably
  answers "am I about to click the diamond I mean". Generating an isometric
  ground set is what makes a map isometric, and only while it is still empty.
  Tiled's isometric maps now load and export instead of being refused, and
  object positions are converted in both directions so a spawn point opens in
  Tiled where you left it. Staggered and hexagonal maps are still refused.
- Plotter writes its terrain sets into an exported `.tmx` as Tiled Wang sets, so
  a generated atlas arrives in Tiled with a working terrain brush. Wang sets
  that are not one of Plotter's own are still refused by name.
- The status bar gained the projection, the hovered cell and the terrain under
  the cursor.

## 0.0.20 — 2026-08-11

- **Groundwork for Plotter's ground tile sets.** The engine for generated
  terrain tilesets — blob autotiling, both map projections, and the generator
  itself — but **no user interface yet**, so nothing in this release is
  reachable from the app. Four new pure modules: the 47-case blob collapse,
  cell-to-pixel placement for orthogonal *and* isometric maps, the terrain
  model, and a procedural generator that emits a flat-fill-plus-outline base
  set from a list of named terrains. A terrain is read back off the tile's own
  id rather than stored beside it, so there is nothing that can disagree with
  the picture; and a cell where three terrains meet resolves to one tile,
  because a terrain's list position is its precedence and a cell's neighbours
  count only from its own rank upward. The generator is deterministic to the
  byte — no randomness and no floating point in any coverage decision — so a
  polished set can be diffed against the base it was painted over.
- **A drag in Plotter is one undo step.** It used to be one *per cell*: a stamp
  pulled across forty cells pushed forty steps and forty entries against the
  history's byte budget. Painting now opens a stroke session, writes the live
  layer with no history while the button is down, and pushes a single patch
  over everything that moved when it is released — the same three calls the
  raster editor has always used, over tile ids instead of pixels.
- A tileset's art can be replaced in place, keeping its ids, its position and
  its declared terrains. A replacement with a different tile count is refused
  by name rather than silently renumbering every cell already painted.

## 0.0.19 — 2026-08-11

- **Indexed colour in Inker.** A document can now carry a palette, and every
  write snaps onto it — strokes, fills, shapes, gradients, filters and pastes
  alike. Index to the swatch row or to a `.gpl`, then edit a slot and every
  pixel painted in that colour is repainted across every layer and every frame
  as **one** undo step; delete a slot and its pixels merge into the nearest
  survivor; reorder freely, because the order is what an exported `.gpl` and an
  exported GIF colour table carry. **Count usage** reports how many pixels sit
  on each slot, so a slot showing zero is one you can safely drop. Alpha is
  never snapped, so a soft brush still fades — it just bands, which is what the
  mode is for. The pixels stay full-colour RGBA underneath: "indexed" means the
  writes are constrained, not that the file stores palette indices, so nothing
  about layers, blending or export changes shape and turning the mode off leaves
  every pixel exactly where it is. The table is saved inside the `.ora` as a
  `palette.gpl` member, and an editor that does not know about it opens the file
  as the ordinary image it already is.
- **An indexed GIF exports your table verbatim.** Slot *n* is the same colour in
  every frame of the clip, instead of each frame being quantised on its own.
- **Clay documents can be closed.** Clay could open documents and never shut
  one: there was no tab bar, so Ctrl+Tab switched between documents with nothing
  on screen saying there was more than one, and a dirty-quit prompt asked about
  documents you had no way to reach. Clay now has the same tab bar as every
  other workspace, with Ctrl+W and a close button, and a dirty document asks
  before it goes. Plotter and Packwright mark a dirty tab with imgui's own dot
  now rather than a `"* "` prefix, matching Inker.
- **Space-to-pan in Plotter no longer latches.** The key-up was filtered out
  before it was ever seen, so the first press left panning on for the rest of
  the session and every left-drag panned instead of drawing.
- **Global shortcuts work while a text field has focus.** Ctrl+K and the F-keys
  were dead the moment you clicked into the 2D prompt box — which is exactly
  where you are when you want to jump somewhere else. Plain keys still belong to
  the field, and so do imgui's own Ctrl+Z/Y/X/C/V/A inside it.
- **Two toasts asked for a level that does not exist** ("warning" rather than
  `warn`) and silently arrived as grey, non-sticky info notices. A test now
  pins every toast level in the source against the ones that exist.
- **Deleting a saved pose or a rendered sheet asks first.** Both sat one pixel
  from a save button and deleted on the click; a sheet costs one Blender render
  per cell to recreate. **Reset all** and the joint **Revert** in the pose
  editor now take the same unsaved-changes guard the preset path already did.
- **Greyed controls can say why.** `disabled_button` promised a tooltip in its
  docstring and drew none across 92 call sites; it takes a `reason` now, and the
  command palette's greyed rows explain themselves instead of swallowing Enter
  in silence.
- **The Review loop says what it did.** An armed negative grade is now visible
  on screen, filing a verdict says what was filed and how to get back to it
  before it advances, the grade and tag buttons carry their keys, and the
  see-through reading carries the same caveat the inspector gives it — a solid,
  featureless mesh scores it too.

## 0.0.18 — 2026-08-10

- **A sprite sheet from a single drawing.** A finished 2D reference can now be
  turned into a sprite sheet without ever becoming a mesh. A queued
  `sprite_synthesis` job draws two candidate 1024px atlases from two seeds —
  turnaround (2x2: front, left, right, back) or a 4-direction walk cycle (4x4)
  — in one SDXL pass each, conditioned on the reference through the IP-Adapter
  and on a per-cell stick-figure pose guide through the canny ControlNet, with
  the pixel-art LoRA on top. Each candidate is matted per cell, put on a shared
  baseline, reduced to 32/48/64px cells in one NEAREST pass and quantized to one
  palette across the whole atlas. Drafts accumulate under the inspector's
  **Sprite sheet** header with a thumbnail and per-cell notes for each
  candidate; nothing is overwritten and nothing is chosen for you.
- **Turnarounds keep your own drawing.** When the reference's proportions match
  what the model drew, it is pasted back into the front cell *before* the
  reduction and the palette, so the one view that is definitely right shares the
  sheet's colours rather than standing out from them. The sidecar records
  whether it was, and why not when it was not.
- **Drafts open in Inker as editable animations.** **Edit in Inker** slices a
  candidate on the sidecar's own rectangles — one frame per cell, never
  re-detected from pixels — and attaches a *directional layout*: a walk sheet
  arrives with a looping tag per direction, so Play loops one direction at a
  time with no change to the animation engine. The layout is saved in the
  `.ora` (additive; the format version is unchanged) and drives **Export sheet**,
  which then writes the sheet's own fixed grid instead of wrapping, with each
  cell's direction and frame in the sidecar. A timeline that no longer fills the
  grid is refused by name rather than exported with a hole in it.
- The pose guides ship as data (`src/warlock/templates/sprite_guides/*.json`),
  so pose quality is iterable without touching code.
- Manual: a new **From a single drawing** section in
  [Sprite sheets](docs/manual/06-sprite-sheets.md), and the layout paragraph in
  [Inker](docs/manual/07-inker.md).

## 0.0.17 — 2026-08-10

- **The Poser: a workspace for reusable poses.** Author a pose once against any
  of the seven skeleton templates — on a bare armature preview, built by the
  same Blender path as a real rig — and it lives in a global library rather
  than inside one asset. Every rigged asset on the same skeleton offers the
  library in its Pose panel; applying one copies it onto the asset, so later
  library edits never change what an asset already carries.
- **A pose can move its root.** Select the root joint in the Poser and tick
  Move root to offset the whole pose — a crouch that actually lowers, a leap
  that leaves the ground. The offset is stored in character heights and scaled
  onto each asset's own rig at bake time, in posed GLBs and sprite sheet rows
  alike. An animated clip cannot interpolate one yet and says so by name.
- The pose editor draws the skeleton's bone lines between the joint markers,
  so a bare armature reads as a skeleton rather than a cloud of dots.

## 0.0.16 — 2026-08-09

- **A style LoRA now declares the architecture it was fitted to**, so the picker
  offers each model the styles that fit it rather than being disabled wholesale
  off SDXL. Adds a FLUX.2 klein pixel-art LoRA and a distilled FLUX.2-klein-4B
  base for it to run on at the 4-step recipe it was trained against.
- **Fixed: one job generated without a style LoRA silently disabled the adapter
  for every later job in the process.** `disable_lora()` sets a flag that
  `set_adapters()` never clears, and the pipe stays resident across jobs, so the
  state outlived the job that set it. It read as working because the trigger
  words are still prepended — the output changed when a style was picked, it just
  was not the adapter doing it. SDXL was affected identically.
- **A mesh verdict is a grade, not a bit.** Review files −5..+5 instead of
  accept/reject, and the five rejection reasons became optional tags — legal at
  any grade, with five good ones added beside them — because a solid slab and a
  mesh a modeller would fix in five minutes were the same row. Image labels stay
  one keypress; they feed binary probes and a grade would be thresholded straight
  back to a bit.
- The re-baseline campaign ran. `bg_removal=birefnet` is the baseline —
  `auto` took 0 accepts in 90 units — and `hole_worst` reads the right way round
  again now the solid-slab failure mode is gone. Written up under
  `docs/measurements/`.
- **Home is a two-column screen.** What's new and the machine's status on one
  side, everything you were working on down the other. The tile grid is gone:
  seven of its nine tiles were the mode switch again, one click either way.
- **Library and Profiles are modes**, in the switch beside everything else,
  rather than sub-views of Home with nowhere else to live.
- **One Resume list across every document mode.** Inker, Clay, Plotter and
  Packwright each kept their own recent-files list; they are now one list,
  newest first, with your assets folded in and each row badged with the mode
  that opens it.
- Positional Alt+digit mode switching is gone. Twelve modes and ten digits was
  never going to fit; the palette (Ctrl+K) is the keyboard route.

## 0.0.15 — 2026-08-09

Backfilled on 2026-08-12 from commit `4504d7c`, which shipped under this version
and was the one release with no entry here. The file's own preamble says it is
the only record of what a version changed, so a gap in it is a defect rather
than a shrug.

- **The UI got its visual pass.** Elevation and depth are real drawing now:
  `shadows.py` strokes shadow bands instead of filling them (one recipe, any
  radius), `surfaces.py` owns panel/inspector surfaces, `ninepatch.py` supplies
  the scalable frames, and `vibrancy.py` blurs the last clean frame behind modal
  surfaces. Each of the GPU-tier effects sits behind its own switch in
  app-Settings and none of them is load-bearing: turning them all off changes
  how the app looks and nothing about what it does.
- **Motion, with a reduced-motion switch.** `motion.py` and `effects.py` add the
  small transitions — the progress card fading *out* after its job is already
  gone, the splash reporting the load it is actually waiting on.
- **Field-level errors and a keyboard focus ring.** A refusal now records the
  control it is about and that control draws the error, rather than the message
  arriving as an unaddressed toast; `focus.py` owns the traversal order for the
  primary 2D and 3D forms, written down rather than emergent.
- **The 2D form grew a common path and a fold.** The fields most jobs never
  touch moved behind "More", and the fold is guaranteed not to hide anything
  that can refuse a submit.
- Home's tile table stopped being the drawn list; one function owns what appears
  there.

## 0.0.14 — 2026-08-09

- **Plotter**: a tilemap editor, with Tiled `.tmx` import and export.
- **Packwright**: a deterministic sprite-atlas packer with a TexturePacker
  sidecar and a `.tsx` tileset for Plotter.
- Inker learned animation: a sparse track x frame grid, onion skinning,
  playback and a sprite-sheet export.
- A light palette, an accessibility reduce-motion switch, and three sidebar
  widths.

## 0.0.13 — 2026-08-08

- **Clay**: block a model out from primitives by hand and export it as an
  ordinary asset.
- Native kernels for the compositor, the contour tracer and the mesh audit,
  built on demand and always beside the numpy reference they are checked
  against.
- A download button for model weights, in a subprocess so the app itself stays
  offline.
- The manual moved into the app as its own mode.

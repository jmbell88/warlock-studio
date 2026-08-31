# Changelog

Hand-written, newest first. Nothing here is derived from git: the commit
subjects are `Warlock vN.N.N` and carry no detail, so this file is the only
record of what a version actually changed. The top heading's version must
match `pyproject.toml` — a test asserts it, so a release bump cannot leave this
file behind.

**A note on how this reads.** These entries are written for whoever maintains
this next, which means they name the measurement that made a default wrong, the
review score that condemned a mode, and what a crash actually was — in the
belief that a fix nobody can audit is a fix nobody should trust. A long run of
"fixed: X crashed" is therefore a record of things *found and closed*, usually
by the project's own test suite or its own audits, and not a weather report on
stability. If you want the short version, the app shows the opening sentence of
each entry under **All release notes...** on the Home screen, and only expands
the release you are actually running.

## 0.0.30 — 2026-08-30

- **Four palette formats, and a folder to keep them in.** The Inker read GIMP `.gpl` and
  JASC `.pal`; the pixel pipeline read `.gpl` and Lospec `.hex`; neither read Paint.NET's
  `.txt`, and the palette directory listed only two suffixes, so a `.pal` or a `.txt`
  dropped in it was not refused — it never appeared at all. All four formats are now read
  and written on both sides, and the palette directory offers all four. The duplication is
  deliberate and forced: `pipelines/pixel.py` may not import the studio, and the headless
  `studio/inker/gpl.py` may not import a pipeline, so each keeps its own parser —
  `tests/inker/test_palette_formats.py` feeds both the *same fixture bytes*, one real file
  per format, and asserts the same colours come back, including on a malformed row, so the
  two cannot drift quietly. `service/palettes.py` carried a long comment explaining why
  `.pal` and `.txt` were absent; it is rewritten rather than left to lie about the tree.
- **The palette folder is browsable from the Inker at last.** Every palette path in that
  workspace was an OS file dialog, which starts wherever it was last, so the palettes a
  user keeps were three navigations away every time. The Colour panel now lists the
  palette directory and loads one by name — through `inspector.palette_names`, so there is
  one remembering of that directory rather than two, and from a *pane*, because the
  headless Inker package may not import the service layer. It presets nothing: Warlock
  ships no palettes, and a palette is still just a file you dropped in a folder. Driven in
  test by clicking the real button at the rect imgui put it at, because a control that
  draws and does nothing is the defect this project has shipped most often.
- **A palette can leave as a picture, not only arrive as one.** *Image...* built a palette
  out of any image and there was no way back out; **Export image** writes the document's
  table as a PNG swatch strip, which reads back as the same table in the same order and is
  how a palette reaches a tool that reads no palette format at all.
- **A terrain set can be made here, which it could not be at all.** The tileset editor
  has a fourth tab, **Terrain**, and it is the author for Tiled's Wang sets. Everything
  around one already existed: `WangColour`/`WangSet` and `Tileset.wangsets` had
  round-tripped through `.tsx`, `.tsj` and `.wmap` since they landed, the Terrain tool's
  swatch list had offered a Wang colour beside a blob terrain, and the constraint matcher
  had painted with them — but the **only** way to get a set was to import a file Tiled
  wrote. The single `wangset` reference anywhere in `panes/` was a read-only swatch
  enumeration, which is to say the feature was a complete round trip with no entrance.
  The tab creates and deletes whole sets, adds, renames, recolours, weights and removes
  their colours, and writes a tile's eight slots by clicking the corner or edge itself —
  markers positioned from `wang.slot_points` and picked with Wave 7's
  `picking.nearest_region`, so what is drawn is what is clickable by construction rather
  than by two lists agreeing for a while. The tile's own art is drawn under the markers,
  because a Wang slot is a claim about *which corner of the picture is grass* and marking
  that up against a blank square is guessing.
- **No new data types, no new write door, and no plumbing between the author and the
  painter.** Every gesture is one pure edit from `tilegrid.wang` followed by one
  `MapDoc.replace_tileset` — the same undoable door the Inker polish trip and *Reload the
  image...* already come back through, which keeps the firstgid, the tile count and the
  declared blob terrains so nothing already painted moves. One click is one undo step,
  which is this editor's existing granularity; unlike a collision drag there is nothing
  continuous to batch. A set authored in the tab paints the map through the canvas's own
  dispatch with nothing added in between, and
  `tests/plotter/test_wang_authoring.py::test_a_set_authored_here_paints_the_map_through_the_canvas_dispatch`
  is that end-to-end assertion.
- **A Wang set cannot be authored onto a generated terrain set, and that combination is
  now refused at both doors.** The Terrain tab was the first producer of
  `Tileset.wangsets` other than the `.tsx` reader, and the reader had always enforced an
  exclusion the writer only *assumed*: `tsx._wang_model_of` drops the general Wang model
  whenever the blob preset is present, while `write_tsx` writes the preset's `<wangsets>`
  and the model's `<wangsets>` from two doors that each return early only on their own
  emptiness. So a set hand-authored onto a tileset carrying a generated preset exported
  **two** `<wangsets>` blocks — invalid against Tiled's schema and off the exporter's
  byte-identical pin — and was **silently dropped** when the file was read back, which is
  the one state `docs/COMPAT.md` claims it has no rows in. Merging the two into one block
  would round-trip to exactly the same loss, so the combination is stopped by name
  instead: the tab does not offer *Create* on a blob-preset tileset and says why in its
  place (a generated set already *is* a terrain brush, so nothing is lost), the
  `create_wangset` door refuses with the same sentence, and `write_tsx` refuses it as a
  named `TiledUnsupported` the save door already toasts — reachable only from a document
  a build between the Terrain tab and this refusal could write, and escapable by deleting
  the hand-authored set on that tab.
- **A set's `kind` is fixed when it is created**, and that is a refusal rather than an
  omission: `kind` decides which of the eight slots a set *uses*, so an editable one
  would either strand values in slots that no longer count — they still travel out to a
  `.tsx` and are read back by Tiled — or clear them on the switch, which is silent data
  loss one misclick away. Making a second set is the cheapest of the three.
- **Removing a Wang colour renumbers every wangid.** A slot is a 1-based *position* in
  the colour list, so dropping the second of three has to unset every slot naming it and
  shift every slot naming the third. Left undone the set would not even construct
  (`WangSet` refuses a slot pointing past the end); done by halves it would have quietly
  repainted every tile in the set. `wang.without_colour` is where that lives, and it is
  the reason the removal is not a list splice.
- **The Terrain tab is the only one of the four with its own tile strip**, and the choice
  was made rather than defaulted. Collision and Animation edit whichever tile the *Tiles*
  tab selected, which is right for them — a collision shape is a property of one tile you
  went looking for. A Wang set is not: authoring one means saying something about every
  tile in it in a row, forty-seven of them for a blob-shaped set, and under that
  convention each would cost a trip to the Tiles tab and back. The strip is the same
  `_tile_grid` the Tiles tab draws, writing the same `state.editing_tile`, so it is one
  selection shown in two places rather than a second one that can disagree with the first.
- **A tile's collision shapes can be moved, resized and reshaped, which they could not
  be at all.** The tileset editor's Collision tab could *add* a shape and *clear* the lot
  and nothing else: `_add_shape` hard-coded the geometry to `x=0, y=0, w=tile_w,
  h=tile_h`, and the view under it was an `imgui.dummy` — a rectangle of nothing, which
  imgui will not even report as hovered. So every collision shape this app had ever
  written was the same full-tile box, and `TilePolygon` — which has round-tripped through
  `.tsx` and `.wmap` since collision landed — had **no author at all**. The tile is now a
  real region: click a shape to select it, drag its body to move it, drag one of eight
  square handles to resize a box or an ellipse with the opposite edge pinned, and edit a
  polygon by its corners — drag one, **Ctrl+click** an edge to add one on that edge,
  **Alt+click** one to remove it, with the third refused by name because two points
  enclose nothing. Every gesture clamps to the tile, because the view *is* the tile and a
  shape dragged past its edge would be invisible and so unrecoverable rather than merely
  moved. Sizes stop one pixel short of collapsing, for the reason a shape is created
  full-tile in the first place: a zero-sized box has all eight handles in one place and
  can never be grabbed again.
- **One drag is one undo step.** A drag is sixty writes a second and `set_tile_meta`
  pushes a step per call, so writing it straight would have put sixty steps on the stack
  for one gesture. Rather than a second write path there is a session — `begin_tile_meta_edit`
  / `live_tile_meta` / `end_tile_meta_edit` — which is `begin_object_edit`'s shape applied
  to tile metadata and has its reason: the document moves live so the user can see what
  they are doing, and the *history* moves once, at the release. It calls the same
  `_apply_tile_meta` hook `TileMetaEdit` calls and pushes an ordinary `TileMetaEdit`, and
  it is closed at the same three chokepoints an open stroke and an open object drag are —
  undo, redo and a history jump — because a document whose shapes are ahead of its head is
  the defect those exist to prevent. A click that moved nothing pushes nothing.
- **The click-region scaffolding is `studio/tilegrid/picking.py`, and it is deliberately
  general.** `TileView` is the one place a screen coordinate becomes a tile pixel or the
  other way round — it *fits* the tile into the square rather than stretching it, so a
  16 x 32 tile's right edge is tile pixel 16 and not 32 — and `nearest_region` is a
  generic picker over `{key: point}` that takes the nearest within a radius, with ties
  going to the earlier key so the answer is stable. Handles and polygon corners are its
  two callers today; a per-tile Wang corner or edge marker is the same question about the
  same square and should ask this rather than grow a second picker. The module is
  headless and pure, and every write returns a new frozen shape — the caller decides
  whether it goes through the undoable door live or at the release.
- Every gesture above is tested through the **real** dispatch. `tests/plotter/_drive.py`
  gains a `TileScene` beside its existing `Scene`, reusing the same synthetic `Mouse`, and
  `tests/plotter/test_tile_collision.py` drives press / held frames / release through
  `_collision_input` one call per frame — a control that is drawn and does nothing is this
  codebase's most common historical defect, and a unit test of a helper is exactly what
  fails to catch it. One test drives the same dispatch over the shared `plotter_ctx`
  fixture so the fast harness and the real mode state are pinned to each other.

- **Plotter can select more than one object at a time, and drag the set as one.** The
  Select tool now sweeps a **marquee** over empty space — a rubber band that takes every
  object it *touches*, rotated ones included, because the box it tests is the object's
  turned corners rather than its `w`/`h` — and **Shift or Ctrl click** adds one object to
  the set or takes it back out. Dragging any member moves them all by a single offset, so
  the arrangement keeps its spacing exactly; with a snap on it is the **offset** that
  snaps rather than each object, since snapping every member's own corner would pull the
  group apart. The marquee is on the **Select tool alone**: every insert tool's
  empty-space drag still draws the shape it is named after, which is the one regression
  this change was able to cause and `test_multi_select.py` pins it in both directions.
  The hard half was undo. `begin_object_edit` stores one object and `end_object_edit`
  pushes exactly one `ObjectPropsEdit`, so a multi-object drag was **not representable**;
  rather than widening that session — four working gestures would have paid for a fifth —
  there is a parallel `begin_group_edit` / `move_group` / `end_group_edit` that closes
  into a `compound` of the *same* `ObjectPropsEdit`. One Ctrl+Z restores every object,
  there is no new kind of `Edit`, and because the steps are addressed by uid a Raise or
  Lower that reshuffles the layer's list underneath an open drag changes nothing about
  what the undo puts back. Delete over a set is one step too, through a new
  `remove_objects`. The Properties pane shows an **"N objects selected"** summary with
  move and delete; **editing one property across a whole selection is deliberately not in
  this pass**, because "blank means unchanged" versus "blank means cleared" is a real
  design question and a form that quietly wrote the first object's values onto the rest
  would answer it the worst way. Wave 5's rotation grip and the resize handles stay on a
  selection of exactly one — half a multi-object rotate would be worse than none — and a
  click inside a group that never moved narrows the set to that one object, which is
  Tiled's rule and the only way back out of a selection.
- **Four more Plotter gaps, all of them a gesture the mode did not have.**
  (1) **Canvas rulers**, along the top and left edges, on by default and on `Ctrl+R` or
  the sidebar's View block. They are Inker's bands — same thickness, same 1/2/5 tick
  ladder, same cursor shadow, and the ladder is *imported* from `inker_canvas` rather
  than copied so the two cannot drift — with the one deliberate difference that they
  count **cells**, not pixels. A band reading "512" where the user counts sixteen tiles
  would be a second coordinate system to hold in your head, which is exactly what the
  status line's cell readout already refuses to be. (2) **The tile Animation tab can
  reorder its frames and play them.** It could set a duration per frame and append, so
  moving a frame meant deleting every frame after it and re-adding them in order, and
  nothing ever played — the durations were numbers you typed and then went to the map to
  see. Up/down arrows per row (disabled at the ends rather than absent, so a row does not
  jump under the pointer) reorder in one undo step, and a Play button runs a preview
  through the *same* `tileset.frame_at` the canvas substitutes gids with, so what plays
  in the editor is what plays on the map. That function is new and is now the single
  implementation: `animated_gid` was walking the durations itself. (3) **A rotation grip
  on the selected object.** `MapObject.rotation` has round-tripped through all four
  codecs since objects landed and could only be authored by typing a number into the
  sidebar. The grip is a round handle on a stalk out of the top edge — round where the
  resize handles are square, because they do different things — and it **turns the object
  about `(obj.x, obj.y)`, its origin corner**, not about the middle of the rectangle you
  can see. That is not a preference: those two fields are the object's *unrotated* origin
  and `_rotated` turns every drawn point about them, so a gesture that rotated about the
  centre would have to write a position and an angle from the same pointer sample, and
  the object leaps on the first frame of the drag when they disagree. Writing only
  `rotation` is what makes the corner the fixed point by construction, and
  `test_object_rotate.py` pins the corner, the no-jump first frame and the one undo step.
  (4) **A snap setting**, `Off` / `Grid` / `Pixel`, on `Ctrl+Shift+G` and `Ctrl+Shift+P`,
  which are Tiled's own chords. Snapping was Ctrl-gated and per-gesture: a user who
  wanted every object on the grid held a modifier down for the length of every drag, and
  Tiled's pixel snap was unreachable. **Ctrl now inverts the setting** — the momentary
  opposite of whatever is set, which is what every editor with a persisted snap does —
  and the consequence worth stating is that the default is unchanged behaviour: at the
  default `Off`, Ctrl still means "snap this one gesture to the grid", exactly as it
  always did. Rotation reads the same setting and snaps to 15°.
- **Six Plotter capabilities that had no control in front of them now have one.** Each
  one was finished code with no entry point, which is the shape of defect this codebase
  is most prone to. (1) **The capsule object** was modelled, hit-tested, drawn and
  carried by all four codecs — and listed in `docs/COMPAT.md` as a Warlock dialect
  feature — but was in neither `OBJECT_TOOLS` nor `OBJECT_SHAPES`, so the only way to
  author one was to hand-edit a file. It is `C` on the object toolbox now.
  (2) **Tileset ▸ Reload the image...** brings an atlas back from a paint program that
  is not Inker, through the same `MapDoc.replace_tileset` door the Inker round trip uses,
  so the firstgid, the ids and the declared terrains are kept and every painted cell
  simply redraws. Re-importing the file was the only route before, and it made a *second*
  tileset that not one painted cell pointed at; a changed atlas size is still refused by
  name, because the roles are positional. (3) **The undo-history panel**: the step count
  in the *Map file* panel is a button now, and opens the stack's own list with the head
  marked — a popover rather than a tenth pane, Inker's precedent. It jumps through a new
  `MapDoc.step_history`, which commits an open stroke first; `doc.history.step_to` does
  not, and a jump made mid-drag would have left the cells ahead of the head. (4) **A
  filter box over the tilesets**, at `widgets.list_filter`'s own threshold, which is the
  count at which the strip starts scrolling. It can never hide the set you are painting
  with. (5) **Map ▸ Go to coordinate...** centres the view on a cell, in the same
  coordinates the status bar reads out, clamped rather than refused. The pan goes through
  a new `plotter_state.centre_pan`, which the minimap's click-to-recentre now shares
  rather than repeating. (6) **The tileset picker is a tab strip, not a combo** — one
  click per swap instead of two, Tiled's arrangement, with the selection written back
  through `state.tileset_index` rather than left to imgui so a filtered tab can never
  quietly become the tileset in hand. `docs/COMPAT.md`'s two cross-references to a
  "Warlock dialect" section at the end of the file are fixed: there is no such heading
  and never was — the five dialect rows are distributed and marked in place.
- **Plotter's Terrain tool no longer crashes the canvas on a map that imports a Tiled
  Wang set.** The Terrain picker offers a foreign Wang set's colours beside the blob
  preset — correctly, because to the person holding the tool there is no difference — and
  encodes a colour as a *negative* rank, `-1 - colour_index`. Nothing decoded it. The
  rank went straight into the blob painter, which reads it as a row of the positional
  layout, and `Tileset.local_for` raised `IndexError: terrain -1 is outside this set
  (0..-1)` out of the frame loop; the pane guard caught it, so what a user saw was the
  canvas replaced by a placeholder and then, after three retries, gone for the session.
  The bound check that should have stopped it was inverted for this input — `rank <
  len(terrains)` is true of every negative rank, so *the emptier the terrain list, the
  more certainly a bad one passed*, and a tileset carrying only Wang sets has an empty
  one. **No click was needed to reach it:** the section auto-selects its first row, so
  merely opening Terrain on such a map armed the bad value. Practical effect: any map
  whose only terrain-ish tileset was a genuine (non-blob) Tiled `.tsx` was unpaintable.
  `_terrain_ref` now dispatches on the rank's sign and returns which model it resolved
  to, and the terrain and fill branches route a Wang colour to `terrain.paint_wang` — the
  constraint matcher that has existed beside the blob path since it landed and had no
  production caller at all. `terrain.fill_wang` is new beside it, flooding over a Wang
  *colour* field for the reason `fill_terrain` floods over the rank field. There is
  deliberately no `erase_wang`: an empty cell constrains none of its neighbours, so a
  Wang hole has nothing to reconcile, which is not true of the blob collapse. The blob
  path is untouched and the existing terrain corpus is byte-identical.
- **The Hunyuan3D multi-view backend is removed.** It was opt-in, marked experimental,
  and had never been runnable by anyone: it needed an isolated Python 3.10/CUDA worker
  and weights named by `WARLOCK_HUNYUAN_PYTHON` / `WARLOCK_HUNYUAN_WEIGHTS`, which no
  install has ever had, so every path through it ended at its own admission refusal. It
  was not free while it sat there. The queue's generation stage carried three
  `if backend != "hunyuan3d_multiview"` guards around the TRELLIS half, and the remesh
  retry at the bottom of the same function called `trellis.generate` through none of
  them -- so its own comment, that the loop "re-enters only the trellis half", was false
  for such a job; deleting the backend makes it true. `service.jobs` decoded multi-view
  uploads through a `to_png` name bound inside an `if request.references:` block above
  it, which would have raised `NameError` on a request with no references. The 3D
  settings pane loses a one-option combo, a licence checkbox and a VRAM note for numbers
  nothing could reach. TRELLIS is now the only reconstruction backend and the queue has
  no backend concept at all. A row promoted before today can still carry `backend`,
  `texture_mode`, `view_assets` and `license_acknowledged`; `promote_to_model` now clears
  all four on every promotion, so those rows migrate silently instead of refusing.
- **A mesh you already have can now enter the library.** Home has an **Import mesh...**
  button beside **New...**, and a `.glb` dropped onto Home or onto the Library becomes an
  ordinary finished model row. This closes a gap rather than adding a feature:
  `service.jobs.import_mesh` was written, tested and reachable from *one* caller — Clay,
  which converts a file into an editable document and refuses a rigged one outright,
  because it has no skinning. So the supplied-base-mesh path that Troupe's whole intake
  assumes had no door at all. Nothing new happens to the file; every consumer inherits it
  at once — Send to Troupe, the Poser, the triangle retarget and every export — because all
  of them are pure functions of `model.glb`. Import is deliberately *not* one of the
  **New...** items: those are the things this app can start from nothing, and a file you
  already have is the opposite errand. Create does not take a dropped mesh either, because
  a mesh dropped mid-generation is ambiguous between "start from this" and "put this in my
  library".

- **The seam verdict is a different statistic.** The old one was crying wolf on the very
  output this project generates. `seam.report` divided the wrap seam by the picture's
  *mean* interior step; it now divides by the *maximum* — "is the wrap the worst join in
  this picture" rather than "is it worse than the average join" — and that number,
  `dominance`, decides. The difference is one word and it is the whole reason for the
  change: a texture of flat cells parted by thin hard lines (pixel art, ceramic grout,
  riveted panels) has an average that collapses toward zero while its maximum does not, so
  the mean-normalised ratio inflated on exactly the population the seamless-tileset track
  generates. On a held-out corpus of 144 tiled axes at seeds no published corpus contains,
  the old ratio called 18 of 72 confirmed-seamless tiles seamed — 15 of them under the
  pixel-art LoRA the track ships with — and dominance called 0 of 72. The threshold is
  `SEAM_DOMINANCE_MAX = 1.0`, fixed by construction rather than fitted: at exactly 1.0 the
  seam ties with the largest step the picture already contains, so it cannot drift with a
  corpus. What it costs is recorded rather than discovered later — dominance is a
  specificity instrument and misses 4 of 44 visibly seamed controls, against the ratio's 5
  — which is why the inspector still says "likely seamless" and still sends you to the
  wrapped view. `worst` is still reported beside it because three published corpora are
  keyed on it, a row now carries `metric` naming which statistic judged it, and a stored
  report *without* that field is worded as the edge/grain number it actually is rather than
  described in today's vocabulary. `docs/measurements/2026-08-30-seam-dominance.md`.

- **Fixed: a non-square selection sent to *Regenerate selection* came back stretched.**
  Two of the pipeline classes the conditioned path routes to —
  `StableDiffusionXLInpaintPipeline` and `StableDiffusionXLControlNetImg2ImgPipeline`, and
  their PAG siblings — *honour* width and height, and they honour them by resizing the init
  image and the mask to that frame. The spec's square frame was being passed regardless, so
  a 1024x448 crop was stretched to a square, denoised distorted, and squashed back into the
  box: a repaint that does not line up with the surroundings it was cut from, silently.
  `_init_frame` takes the frame from the init picture when there is one. Only an exactly
  square selection escaped, which is why the GPU test's square box never saw it. Omitting
  the pair is not the fix — diffusers defaults it to the UNet's own 1024 square.

- **Fixed: the remesh had been shipping triangles since it landed.** glTF cannot share a
  vertex position between two texture coordinates, so *every* GLB splits its vertices at
  each UV seam — which makes an imported mesh non-manifold before anything is actually
  wrong with it. Every input on the remesh path is a GLB, so quadriflow could never
  succeed and every remesh silently produced the decimate fallback instead. Measured on a
  UV sphere: 1,106 vertices before export, 4,512 after the round trip, and quadriflow
  answering "Remeshing failed"; welded back to 1,106 it returns 479 faces, all of them
  quads. The weld runs on the working copy only — the bake reads the original surface.

- **Fixed: three silent defects in rigging a mesh the user supplied.** Every mesh this path
  had ever seen was a TRELLIS reconstruction, which carries no armature and no vertex
  groups, so nothing noticed. A supplied humanoid usually arrives rigged, and then: the
  bone-heat guard stops working, because it asks whether *any* vertex group holds a weight
  and an incoming skin answers yes before the new armature is bound at all — so a bind that
  produced nothing was reported as a clean rig and the character did not deform; the old
  skeleton was exported beside the new one, since the export writes the whole scene; and
  the Y-up to Z-up rotation was applied *twice*, once baked into the vertex data and again
  through the armature the mesh is parented to, so the measured bounds came back rotated
  one turn too far — on CesiumMan an arm span of 0.505 against a true 1.138, with the
  height still plausible enough to look fine, and the skeleton fitted to that box.
  `_strip_incoming_rig` discards the incoming skin and skeleton before the bounds are
  taken. Discarding rather than adopting is deliberate: CesiumMan has 19 bones like the
  shipped template and still does not fit it (3 per arm and 4 per leg against 4 and 3), and
  a rig that does not map onto the one the clips are authored against is not evidence about
  where those joints go.

- **Fixed: Clay's refusals were reaching the user as "Something went wrong".** `OpError` is
  documented as a user-facing refusal whose message is the whole interface for it, and on a
  task thread it was not shown at all — the task layer kept a message only for
  `ServiceError`. Clay's four submissions are the only place a refusal is raised off the
  frame thread, so every one of them (a rigged GLB, an unreadable one, a mesh past the
  triangle ceiling) named its cause and its remedy to `warlock.log` alone. Found by a user
  who dropped a rigged GLB into Clay and had to read the log to learn why nothing happened.

- **The first graded run of the shipped recipe, and it is not a good result.** A
  pre-registered corpus of props went through text to `sdxl_cfg` to TRELLIS at the shipped
  defaults and was graded blind: **25.0% usable**, which clears the pre-registered bottom
  rule by one grade. The failure localises to the mesh half — 95% of the references were
  good and 16 of 21 were lost in reconstruction, with `holes` the dominant defect tag — and
  the difficulty prediction *inverted*, with the easy tier scoring worst. The protocol and
  the corpus were registered before any of it was drawn
  (`docs/measurements/2026-08-30-art-verdicts-preregistration.md`), the result is
  `docs/measurements/2026-08-30-sdxl-cfg-props.md`, and `scripts/campaign_props.py` is the
  harness.

- **TODO P12 is answered, and the answer is no.** Three humanoids in that corpus were
  judged on the entry's own rubric — limb separation and silhouette, not face fidelity —
  and came back with limbs bent and stretched. The `_init_frame` bug above is ruled out as
  a confound: those jobs carried no init image and no conditioning, so that path was never
  entered. Troupe's Phase 7 therefore stays deferred on the generated-character path, which
  is the decision that entry existed to take before the investment rather than after, and
  the supplied-base-mesh path is the one to build on — which is what the import door above
  opens. This says nothing about whether a better reconstruction could carry characters; it
  is a verdict on the shipped single-view default.

- **Four native-kernel candidates were measured and none of them was written in C** —
  which is the point, because the house rule is that an unbenched candidate must not be
  built, and two of these four turn out not to want C at all.
  `docs/measurements/2026-08-30-native-batch-6-candidates.md` has the numbers;
  `scripts/bench_native.py` gains the cases `tiles_materialize`, `oklab_fold` and
  `clay_bevel`, and `inker_grow` gains three variants where it had none. The two
  findings that change what should be built next are pure numpy. `inker/filters._grow`
  was **deferred to a kernel** by batch 5 on the strength of its linear cost curve; the
  constant factor it was assumed to be stuck with was never measured, and fusing the
  eight `_shift` calls — each of which allocates a whole-canvas `zeros_like`, fills it,
  then allocates the union — is **6.3×** (175 ms → 27.8 ms at r=32 on a 1024² mask),
  bit-identical, no ABI bump. `inker/tiles.materialize` was recorded as **not measured**
  by batch 5, on the strength of a failed vectorisation of a *different* function; it
  calls `oriented()` once per cell with no memo, where there are at most `tile_count × 8`
  distinct answers, and memoising on the raw gid is **3.1×** (444 ms → 142 ms on a 3200²
  canvas at 8 px tiles). Its cost is per *cell*, not per pixel — the same canvas at 8 px
  tiles instead of 32 px costs 8.9× more — so the small-tile pixel-art case, which is the
  ordinary one, is the expensive one. `pipelines/pixel._to_oklab` is now essentially the
  whole remaining cost of `map_palette` at 236 ms, the shipped kernel having taken the
  other half, and a uint8 LUT for sRGB→linear should be tried before any C.
  `clay/ops_bevel.bevel_edges` was benched for the first time and **fails its
  pre-registered gate by an order of magnitude** (579 ms with a quarter of a 40k-face
  mesh's edges selected, against 1.0 s) — and its cost tracks mesh size rather than
  selection size, because two loops walk all 40 000 faces and 160 000 corners to bevel
  ten edges. That is an algorithmic fix, not a kernel, exactly as `ops_topo.weld` was in
  batch 2. `retexture.combine` was left unbenched on purpose and no number is claimed
  for it.

## 0.0.29 — 2026-08-29

- **Generated tiles now actually tile.** Create → **Sheet** has a Layout control with three
  entries. *Materials* takes a list of surfaces and generates each one as its own seamless
  1024px tile through the circular-padding path, then lays the results out; *Terrain set*
  takes an inner and an outer surface and composites them into a blob-47 set that Plotter's
  Terrain tool paints with immediately; *Grid (legacy)* is the old path, one generation cut
  into sixty-four cells. The reason the old grid is legacy is measured:
  `docs/measurements/2026-08-18-tile-sheet-grid.md` ran four arms and found the mechanism
  sound and the art direction not — the sixty-four guide cells are *identical*, so the model
  has no per-cell signal to vary against and either ignores the guide and paints one
  continuous scene or obeys it and paints one tile sixty-four times. That document named the
  answer, "N materials, one grid", and this is it. Grid stays because it is the only route to
  a 3/4 or isometric sheet: a 3/4 tile has a visible front face and an isometric tile is a
  diamond, so neither of those wraps and neither can be seamless. A generated terrain set is
  recognised in Plotter from what `sheet.json` records rather than by looking at the pixels —
  role inference needs transparency or one dominant ring colour and a real set is two opaque
  textures, so it would answer nothing.

- **A tile sheet no longer records a structure it does not have, and a grid row no longer
  fails an hour in on a validation of pixels it was never going to draw.** The legacy grid
  path was still compiling a per-cell plan — per-cell prompts, per-cell seeds, sixteen Wang
  roles — and writing it into the job's params and into the sheet's sidecar, while drawing
  one generation sliced on a fixed lattice. That was the original defect the whole tileset
  programme started from: a compiled plan stored as metadata that changed nothing about the
  generation, so the record beside the PNG could contradict the PNG. It also raised inside the
  worker, after the row had been accepted and queued, for a stored request carrying a terrain
  block the grid arm never reads. Sheets already on disk keep their block and still open.

- **The pixel look reaches sprite sheets and tile sheets, not just Troupe.** An authored
  palette (Lospec/GIMP ramps from your palette folder), ordered dithering, and — on sprites — a
  hard outline are now on the Create form and in the request, where before only the Troupe
  render path could reach any of it. `pipelines/pixelize.py` has always said an authored ramp
  is the single highest-leverage art input in the program and that hard outlines are most of
  what separates crisp pixel art from a shrunk render; five generated-atlas paths existed and
  one of them was wired. The palette also joins the fields a **profile** captures, so two
  sheets of one character can be made to match without retyping it. Outlines are sprite-only
  and refused by name on a tile rather than quietly dropped: on an opaque tile the edge mask
  treats the atlas border as transparent, so an inner outline returns the outer ring of every
  cell — a grid line drawn around every tile. Dither is offered with or without a palette,
  because the tile path derives its own table by median cut and dithers against that.
  Sprite drafts move to version 3 and their bytes change: the reduction is the alpha-weighted
  box supersample rather than a nearest resize, the mapping is nearest-in-Oklab rather than
  the median cut's own assignment, and orphaned single pixels are cleaned. The default tile
  path deliberately still uses its old quantiser — measured on a synthetic atlas, the two
  produce the *same* 64-entry table and disagree on 27.66% of pixels, so routing the default
  through the new pair would have silently re-coloured every tile sheet in your library on its
  next reroll.

- **A sprite sheet can be one of seven actions in eight directions.** Idle, walk, run, attack,
  cast, hurt and jump, each with its own frame count, drawn one full direction per generation
  rather than one frame for the lot. The arithmetic forces that shape: the pixel-art LoRA
  spends about eight generation pixels on one art pixel, so an honest 32px sprite needs a
  256px cell and eight directions of eight frames is 4.2 megapixels — four SDXL frames, not
  one. A whole direction shares one denoise because drift between frames of one direction
  plays at 10fps and reads as flicker, while drift between directions reads as the character
  turning and is held by the shared reference, the shared seed and the IP-Adapter. One
  consequence is worth knowing before you pick a size: at 48px and 64px only four frames fit
  in a band, so only the four-frame actions (idle, hurt) can be drawn there and the size
  picker now says so instead of compiling a request the door refuses. The seven pose guides
  are mirror-derived — five directions authored, three exact reflections — and every one was
  rendered and looked at; four were rewritten because of what that showed, including a walk
  whose front row splayed the arms outward with the stride and so produced eight identical
  frames.

- **Fast and Quality are two different pictures now.** They were not: `image_fast` and
  `image_quality` both named `sdxl_cfg` at the same working resolution with the same
  downloads, so the tier selector resolved to two names and one checkpoint and changing it
  changed nothing you could see. Fast is SDXL 1.0 under Hyper-SD at four steps against
  Quality's thirty, on the same base weights, so the only extra download is the 0.8 GB adapter
  it already declared. This is a measured trade rather than a guess —
  `docs/measurements/2026-08-11-default-base-model.md` ran this exact arm and scored it 2 of 4
  against 3 of 3 — and the tier now says what it costs: four steps, no structure control, no
  negative prompt. It also stops refusing in a place you cannot see: the refusal for a
  structure image under Fast used to name `base_model`, a control automatic routing does not
  draw, and a stranded ControlNet or Avoid line is now cleared with a sentence when you change
  tier rather than keeping Generate off over a field that is off screen.

- **A reroll that runs out of attempts keeps its best draw rather than whichever was last.**
  Stopping on the first acceptable attempt is unchanged and right; the other exit was not.
  No quality score was invented for this, because the reference report carries none and its
  refusal codes are documented as the order the rules fire in and explicitly not as severity —
  the ranking is (empty last, then fewest refusals, then fewest warnings), which is the only
  ordering the measurement supports. Ties keep the last draw, so the common case where every
  attempt fails the same rule is byte-identical to before. The recorded seed and report are
  rewritten to the draw that survived, so provenance never names a seed that does not
  reproduce what is on disk.

- **Fixed: the structured request route had never once been able to queue a sprite sheet.**
  It mapped a sprite request onto an output name that `create_job` refuses by name, so every
  such request was turned away at the door. A sprite sheet is an ordinary reference job
  carrying a follow-up, and it is submitted as one now.

- **Fixed: an eight-direction sprite draft was complete on disk and invisible in the pane.**
  The listing demanded both candidate PNGs, but a sheet that large is drawn as a single
  candidate — so the finished draft, correct in its own sidecar, was never shown. The record's
  own candidate list is what says which images it claims, which is stricter as well as truer.

- **Fixed: the Palette combo advertised two file formats that silently never load.** Its
  helper said the palette folder takes `.hex`, `.gpl`, `.pal` and `.txt`; the loader takes the
  first two. Dropping a `.pal` in produced no error, no row and nothing to see. The sentence is
  now derived from the loader's own list rather than restated beside it, since a restated list
  is exactly the thing that drifted. Whether the two missing formats should load is recorded in
  the loader — both are real formats somebody meant to support, and Inker's own import already
  parses a JASC `.pal`.

- **Fixed: the sprite panel promised two drafts and two generations for sheets that are
  neither.** An eight-direction action generates once per direction, and above sixteen cells
  the door draws one draft rather than a pair — so an eight-direction attack at 32px is eight
  generations for one draft, wrong in both directions at once and wrong on cost by 4x. The two
  strings had also contradicted each other, a button offering "two drafts" above a note saying
  both land as one. Both now come from one service call that answers with the cells, bands,
  candidates, generations and duration for the selection in hand, and carries the door's own
  refusal as the button's disabled reason when the combination is one the door would turn away.

- **A seam measurement that refused to move its constant, which is the interesting result.**
  `SEAM_MAX` stays 3.5. The re-measurement was run because four real generated materials scored
  2.95–3.74 on the GPU lane, inside the band the original turbo corpus left empty — but the run
  found the instrument, not the threshold, is what fails on this population. Over 120 units
  with decision rules registered before any image existed, the ratio is **inverted** on
  LoRA'd pixel art: it flags twenty of twenty-four tiles that were then viewed through their
  seam crosses and confirmed to have no visible join, while passing plain, definitely-seamed
  images that scored as low as 0.857. No threshold beats chance there, so tuning one to make
  the assertion pass would have been tuning a constant to a test. The failing GPU assertion
  stays red and carries its own diagnosis. Two things came out worth more than a number: a
  better instrument exists — *is the wrap seam the largest discontinuity in the image?* scores
  1 of 48 on tiled and 19 of 48 on plain — and a reproduction control proved this path has not
  moved, with a maximum delta of 0.000000 against the 2026-08-13 run across all 72 shared
  units, so two rewrites of the image pipeline and three prompt versions provably never
  touched the tile path.

- **New mode: Sirens, a chiptune tracker.** Twelve modes now. NES-shaped pulse, triangle,
  noise and sample voices; a pattern grid you type into; an order list; instruments whose four
  envelope sequences you drag into shape; sound effects in the same document as the music; and
  an export that writes `song.wav`, one WAV per channel under `stems/` and one per effect under
  `sfx/`. It is offline, needs no GPU and no downloaded weights, and its document is a `.wsng`.
  Two departures from the hardware are deliberate and audible: tuning is equal-tempered rather
  than a period table, because a track from here has to sit under modern music without beating
  against it; and `Fxx` sets a tempo in BPM rather than ticks per row, because with a row fixed
  at a sixteenth note the FamiTracker spelling is a control that appears to do nothing. Voices
  are synthesised at four times the output rate and filtered back down — a naive square at
  44.1 kHz folds its harmonics down and reads to a listener as being *out of tune* rather than
  as brightness.

- **A machine with no sound card can still use the whole mode.** Writing, editing, saving,
  loading and exporting a WAV all work with no audio device; only playback does not, and the
  transport prints the reason beside the greyed button rather than leaving a dead control to
  read as a broken app. That is also what makes the mode testable: the audio device is confined
  to one module, so the tests do not all skip on a machine without one.

- **Playback is render-then-play rather than a real-time callback.** The song is synthesised
  into a buffer off the frame thread and the buffer is played, so what you hear is the same
  samples the export writes and a busy machine cannot click. The cost is that an edit is heard
  on the next render, which the transport says out loud; a stale buffer is refused rather than
  played, because hearing the version of a bar you have just replaced is the one outcome that
  makes everything else you heard suspect.

- **Fixed before shipping, in the pattern grid: a `TypeError` on every frame that drew a
  caret.** `add_rect` takes thickness before flags and the call had them the other way round —
  so every frame with a grid on screen, which is every frame in the mode. Nothing caught it,
  because every existing pane smoke test builds a real renderer and *skips* where there is
  none, which is CI and every remote shell. There is now a pane test that draws all seven
  Sirens panes through a bare imgui context with no backend at all, into a window of a stated
  size — at the default size the grid drew no channels and passed while the bug was there.

- **Fixed before shipping, in the envelope editor: one undo step per column crossed.**
  Painting a decay across twenty columns pushed twenty edits, so taking back one drag was
  twenty presses of Ctrl+Z. The drag is opened and closed as one collapsed step now, which is
  the mechanism a sound effect and its pattern already used.

- **Fixed before shipping, and it was data loss: the 256th object minted in a session
  rewrote a song's instruments.** A pattern cell is `int16` and its instrument column held a
  process-global uid, which the file reader then clipped to 255 across *every* column — so a
  song saved after enough objects had been created came back playing a different instrument, or
  none. Instrument ids are now a per-document space bounded at 128 and minted as the lowest
  free slot, which is also the number the grid shows and the user types; a file whose ids
  predate that is renumbered on read rather than refused, because refusing loses the song and
  clipping lands the note on whatever instrument is in slot 127.

- **The `.wsng` is the composition and every WAV is derived from it.** Exporting an untouched
  document twice writes byte-identical files — no timestamp, no writer string, no randomness —
  so an exported track is something a build script regenerates rather than an artefact to keep.
  Nothing is written until every file has been encoded, so a refusal partway through leaves no
  half-populated `stems/` to be mistaken for an export. Stems keep the other channels' effect
  column, because jumps, halts, breaks and tempo changes belong to the player rather than to a
  voice: a stem rendered from a wiped grid would run at a different tempo than the mix it is
  supposed to line up with.

- **Sirens is documented and toured, and writing the documentation is what found the gap the
  entry below closes.** Two manual chapters (a tutorial that walks one
  song from an empty document to an exported WAV, and the reference chapter) plus a
  `sirens-basics` guided tour that needs no GPU, no weights and no sound card to finish.
  Adding the reference chapter renumbered eleven chapters after it, which is what a chapter's
  number deciding its part costs and what the manual's tests check in both directions. The
  `Ctrl+/` sheet gains a Sirens group, which it had been silent about while the mode had
  twenty bindings.

- **Every column of the pattern grid can be typed into, and every effect the synthesiser
  implements is now reachable from the keyboard.** The instrument and parameter columns take
  two hex digits, entered left to right and replacing one nibble at a time, with the caret
  narrowed to the character still owed so a half-finished entry is visible; the volume column
  takes one digit; and the effect column takes an effect's letter, read out of the engine's own
  `EFFECT_NAMES` so that a letter it has no handler for writes nothing rather than a value the
  song cannot play. `Shift+Backtick` writes the release note `~~~` beside the backtick's
  `===`, which is what makes an instrument's release tail audible where it is written instead
  of only at the end of a song. `Delete` narrows to the column under the caret when nothing is
  selected, and still clears every column of a block. Every one of these goes through the same
  `write_cell` the note column always did, so each owes the same three things — the refusal
  framed as a toast, the renderer re-armed, the caret stepped by the edit step.

  Which column the caret is in decides what a key means, which is the rule that had to hold:
  `c` is a note in the first column and the hex digit twelve in the third, and the piano row
  still fires in the note column alone. The gap this closes was found on 2026-08-27 by writing
  the manual chapter, and it had survived three green landings because nothing in the suite
  asserted that a keystroke reached a non-note column. `tests/test_sirens_keys.py` closes that
  properly: parametrised over every column the document has, and ending in two assertions made
  against **rendered audio** rather than against a cell — a tempo effect typed through the key
  handler has to change the song's length, and a released note has to sound where a cut one is
  silent.

- **Sirens keeps its Experimental chip, narrowed again.** What is left is a block selection
  that can be transposed and cleared but not copied, cut or pasted, so a bar that repeats is
  retyped or is a second entry in the order list. The tooltip, the reference chapter's limits
  section and the tutorial's opening all name that and nothing else.

- **Closing Warlock while a picture was being generated no longer hangs it.** Shutdown
  closed the image worker's handle while the running sample still held it, so it waited for
  the sample -- up to fifteen minutes -- before it could even ask the job to stop. The close
  now stops the worker process first, which is what makes the sample return.

- **An IP-Adapter is refused at the door on a base model it does not fit.** A ControlNet or
  a style LoRA on the wrong base was refused before anything loaded; the IP-Adapter was not,
  so the refusal arrived with a 7 GB checkpoint already on the card. Same rule for all three
  now.

- **Rigging from the inspector's Pose tab uses the skeleton you picked.** The Rig stage and
  the library row already did; this one door passed the configuration default, so choosing
  "quadruped" and then pressing *Rig this mesh* in the Pose tab rigged with the humanoid.

- **Animated drawings keep their background and reference layers across a save.** The
  `.ora` writer recorded both flags and the reader never read them back, so every animated
  document opened with neither. Converting a layer to a tilemap and back dropped the flag
  the same way.

- **A cancel that hit a busy database is still a cancel.** The finished/failed verdict was
  retried when the one sqlite connection was mid-backup; the cancelled verdict was not, so
  a job you stopped could come back on the next launch reading "interrupted".

- **A re-rig whose second file cannot be published leaves no stale marker.** Publishing
  renames the mesh and then its rig record; if the record's rename failed after the mesh
  had landed, the asset advertised a skeleton it no longer had. It now reads as unrigged.

- **The sheet cap is one count at every door.** Sprite sheets, character sheets and *Send
  to Troupe* on an unrigged mesh all draw on the same 200-per-asset pool and each counted a
  different subset of the others, so the pool could be reserved past the cap. Troupe's
  unrigged path did not check it at all.

- **Clay: edge picking chooses the nearer edge.** With two objects' edges under the cursor
  the earlier one in the document always won; vertex picking already compared depth and
  edge picking now does too. Closing a Clay tab also no longer throws away every other open
  tab's manifold checks.

- **Plotter: a closed map releases its minimap.** The composited minimap (and a handful of
  smaller per-map form caches) stayed in memory for the life of the session after the tab
  closed.

- **Inker: the export sidecar switches are remembered.** *Tags* and *Slices* were recorded
  with the rest of the export options and silently reset to their defaults on the way back.
  A third switch that nothing ever read was removed.

- **Studio preferences that cannot be saved no longer litter the data folder.** A failed
  save left its staging file behind, and the once-a-second retry left another each time.

- **Housekeeping.** Dead code out: three unreachable field groups in the 2D settings pane,
  an unused per-tab job id in Troupe, an unread feather setting, a store method only tests
  called, an unreachable worker op. One helper each for the prompt-length check (seven
  copies) and the stage-everything-then-replace file write (two copies). The installer
  script's comment carried a literal backspace byte where `` was meant; the native build
  removes the import library the non-MSVC linkers leave beside the DLL, which the installer's
  manifest check would refuse. Three manual chapters and four docstrings brought back in
  line with the code, and a download-pin test that had been passing on zero matches now
  checks the rendered text. `REPORT.md` deleted; `TODO.md` rewritten as a priority list of
  what only a human can close.

## 0.0.28 — 2026-08-23

- **Saving a screenshot now says where it went.** The picture was written and nothing was
  said about it, which is exactly what a save that failed looks like.

- **The resize box in Inker no longer accepts a number that would take the machine down.**
  Typing 100000 into it asked for about 40 GB per layer, immediately, holding a drawing that
  had not been saved. Growing is now capped at 8192 a side (or at whatever the drawing
  already is, if it came in larger); shrinking is not capped at all, because shrinking is
  what the box is usually opened for.

- **Smooth in Clay refuses to multiply a mesh past what Clay works with.** Each press
  quadruples the face count, and there was no limit on it -- six presses on a mesh imported
  at the maximum is eight billion faces. It now says so and does nothing, rather than
  freezing. Packwright likewise refuses the sprite that would take a pack past its limit at
  the moment it is added, rather than at the moment you try to pack.

- **A large library is faster to scroll and uses far less video memory.** Only the cards
  actually on screen are drawn now, thumbnails are budgeted in megabytes rather than counted,
  and "Load older" stops offering itself once the whole loadable history is loaded -- it used
  to keep offering, and do nothing. Inker's animation frames got the same treatment: a long
  clip no longer holds every frame's picture on the graphics card at once.

- **A document file can no longer ask Warlock for more memory than the machine has.** Every
  format the app opens -- `.ora`, `.aseprite`, `.gif`, `.wblk`, `.wmap`, `.tmx`/`.tmj`,
  `.tsx`, and a hand-supplied `.glb` -- states sizes inside itself, and in a dozen places
  Warlock allocated from one of those numbers before anything checked it. A four-hundred-byte
  model file could ask for 64 GB; a two-kilobyte drawing could claim a 200,000-pixel canvas;
  a two-hundred-byte Aseprite header could claim 65535 square. All of them are now refused by
  name, with the size that was refused in the message, and none of the files anybody actually
  has are affected -- the ceilings sit above the largest thing Warlock itself produces.

- **Two ways of hiding a document type declaration inside a Tiled or OpenRaster file are
  closed.** Warlock refuses these because an XML declaration can expand a few hundred bytes
  into gigabytes as it is read. The check looked at the start of the file as bytes, which a
  file saved as UTF-16 and a file with a long comment in front of it both walked straight
  past; the parser itself now refuses them, whatever they are written in and wherever they
  appear. The same door caps how deeply a file may nest -- a deeply nested map used to open
  and then break the workspace on every frame afterwards.

- **A sprite sheet export over two tags that differ only in capitals no longer loses one of
  them.** `Walk` and `walk` are one filename on Windows and macOS, so the second export
  silently wrote over the first; both spellings are now refused before anything is written,
  which is what the check was there for. The same applies to accented names typed on
  different systems.

- **One bad byte in the preferences file no longer stops Warlock opening.** Several settings
  were read as though whatever was in the file had to be the right shape, so a hand-edited
  or half-written `studio_settings.json` could crash the app on the first frame of whichever
  workspace read it -- every launch, because nothing rewrote the offending file. Each of
  those reads now falls back to its default. Preferences written by a *different version* of
  Warlock were also silently thrown away; they are now kept under a timestamped name and the
  reset is reported, the same way an unreadable file already was.

- **The failures before the window exists now say something.** Warlock could not prepare its
  home directory, could not get an OpenGL 3.3 context, or was missing one of its own font
  files -- and in each case the app simply did not appear. All four now open a dialog naming
  what happened, and the two with a remedy say it: update the graphics driver, or reinstall.
  Changing the interface scale with a font file missing is no longer fatal either.

- **A damaged job database can be recovered from inside Warlock.** It is the library's index,
  and if it would not open there was no way past it: the same failure box on every launch.
  Warlock now offers to set the damaged file aside -- renamed and kept, never deleted -- and
  start with an empty index; your generated assets are folders beside it and are untouched.
  Diagnostics also gained a **job database** row, so a file that has started to go can be
  found while there is still time to back it up.

- **A crash-recovery file with a malformed timestamp no longer stops the app.** It was read
  on the first frame after a crash, which is exactly when it mattered that it did not.

- **Autosave stopped believing in copies it had not written.** The journal marked a
  document as backed up the instant it *queued* the write, not when the write landed, so a
  full disk, a removed drive or a permission change left the slot recorded as saved at its
  current state, the "has it changed" gate then saw nothing to do, and no retry ever came
  for the rest of the session — the app believed a crash copy existed when none did. The
  mark now waits for the write, the failure is retried, and it toasts by name ("Autosave
  could not write a recovery copy") rather than as the generic "something went wrong".
  Relatedly, a disk that filled *between* the two halves of a copy used to leave the file
  on disk with no marker beside it, which meant nothing would ever offer it back; both
  halves are now written before either is published.

- **The unsaved-work list on Home shows all of it.** It scanned for eight and silently
  dropped the rest, so a crash with twelve documents open buried four of them, and the
  count in the crash dialog under-reported by the same four. It still draws eight rows at a
  time; the rest are counted, and take their place as the listed ones are dealt with.

- **Cancelling a character sheet no longer deletes one you can already see.** The cancel
  landing after the sheet had been published recorded the row as cancelled and then removed
  the finished sheet with it. Re-texturing had the same shape from the other side: the mesh
  was repainted and the row said it never happened. Both now finish normally once the
  result is on disk — which is what a cancel there has always meant everywhere else.

- **Exports no longer truncate the file you picked.** Nine exports wrote straight onto the
  destination, so pointing one at a file that already existed and having the write fail
  partway through destroyed it and left nothing in its place. Sprite sheets and their data,
  posed GLBs, screenshots, palettes, PNG sequences, GIFs, sliced PNGs and the generic
  artifact export all write beside the destination and swap it in at the end, which is what
  Inker's document saves already did.

- **A broken job database now says the worker has stopped.** The dispatch loop retried
  forever and reported nothing, so a corrupt page or a full disk left every job queued with
  no explanation on screen. A finished job whose completion could not be written is also
  retried, rather than being reported as a crash on the next launch. And backing up the
  library no longer freezes the job list while it runs.

- **Layer groups have a header row in the timeline, and it folds.** The fold
  triangle reads `TabDoc.collapsed_groups`, which had been declared for as long
  as groups have existed and read by nothing — so a folder could be made and
  never shut. The header carries the folder's own eye and a menu offering
  Rename, Opacity, Locked and Ungroup, which gives `set_group_props` and
  `ungroup` their first callers; dragging a layer onto it moves the layer in.
  Folding is view state: not saved, not undoable, and swept when the group it
  names is dissolved so a uid cannot stay folded with no header to reopen it.

- **A timeline range is a selection the layer verbs honour.** `Shift`+clicking
  a track name stretches the range across tracks — the `extend_range` the
  panel has always had and never reached — and clicking a name selects the
  layer, which before this needed a click on a *cel*. The row menu then acts on
  the whole block and says so: **Duplicate 3 layers**, **Merge down 3 layers**,
  **Delete 3 layers**, **Group 3 layers**. Each is a single Ctrl+Z however many
  rows it touched, through a new `Document.one_gesture()` that folds a run of
  pushed steps into one rather than threading an edit list through six ops.

- **Inker opens animated GIFs.** It could write one and not read one, which
  made the export a one-way door — the file a user had just shared was a file
  this editor refused, and it was the only export in the app with that shape.
  Frames arrive one per frame with their own durations, read through Pillow's
  seek because a GIF frame is a *patch* with an offset and a disposal method
  rather than a picture; assembling them by hand is correct on frame one and
  increasingly wrong after it. A one-frame GIF opens as a still drawing. Saving
  in place is refused for the reason a JPG is: Ctrl+S offers an `.ora` beside
  it rather than flattening a layered document into 256 thresholded colours.

- **Entering indexed mode is a dither like every other conversion.** It took
  `"nearest"` with nothing on screen to say so, and was the only conversion in
  the app with no choice of matrix. The Indexed button opens the Convert popup
  now — same controls, same preview, same refusals — and Apply enters the mode.
  This also settles a name that had been wrong for months: the **Colour mode…**
  menu row opened that popup, and what the popup did was snap the pixels onto a
  table and leave the document in RGB. One popup answers both questions and
  which one it is answering is now a property of how it was opened.

- **The paint bucket has Aseprite's three options.** **Refer to** is Canvas or
  Layer — lineart on its own layer over painted flats is the case the second
  exists for, since referring to the canvas makes every fill stop at the paint
  under the line rather than at the line. **Diagonals** continues a region
  through a corner touch, without which a 45° pixel-art line is a wall; it is
  shared with the wand, so the two cannot disagree about what one region is.
  **Stop at grid** confines the fill to the cell you clicked in, which is how a
  single tile of a tileset gets filled without masking it first.

- **The animation verbs are bindings rather than context menus.** `Alt+N` adds
  a frame, `Alt+D` duplicates the current one, `Home` and `End` jump to the
  ends, `F3` toggles onion skin, `Ctrl+U` and `Ctrl+I` open the filter popup on
  hue/saturation and invert, and `Ctrl+Alt+C` is canvas size. Delete frame is
  deliberately unbound — Aseprite leaves it unbound too, and a one-key drop of
  the frame under the playhead is the one worth reaching for a menu. Because
  bindings here are data, all of them appear in the shortcut sheet and the
  remapper without either being told about them.

- **Clicking a tag jumps to it.** The band under the grid was right-click only,
  so the one thing a tag is for — "show me this animation" — took a menu it did
  not have. A double-click opens the rename in place, the same field the menu
  offers.

- **A mesh job with no matte setting no longer picks the one mode measured at
  zero.** The door resolves `bg_removal` against this host — `birefnet` when
  `birefnet.gguf` is on disk — and the worker was reaching past that for a flat
  `auto`, so the two disagreed about the same default and the worker held the
  worse half. `auto` is the mode the 2026-08-07 review scored at 0 accepts in
  80, with 58 of those rejects tagged `broken`: without the learned matte the
  server falls back to a threshold cutout, which leaves background attached for
  the reconstruction to turn into a slab. The worker asks the same question the
  door does now, and still answers `auto` on a host with no weights to load.
  This reaches any job the store was handed directly rather than through
  `service.jobs` — a seeded row, or one predating the key.

- **Inker is laid out the way Aseprite is.** Colour and the new picker on the
  left, the toolbox on the right with the preview, the tiles and a new
  Generation panel, and the timeline along the bottom. What was there before
  was Aseprite's *Mirrored Default* preset — a 90 px tool rail on the left,
  the palette on the right — and the rail cost the toolbox its heading, its
  help button and two clipped rows: 90 px minus the pane's own padding is
  74 px of content, which fitted two 34 px buttons and the gap between them
  exactly. The tool grid's column count follows the pane's width now.

- **The layers panel had been invisible since 0.0.26, and this is why.** The
  commit that merged it into the timeline said "the strip is always available,
  Tab toggles it", but the `doc.anim is not None` gate in the workspace
  predated that merge and was never lifted — so a still document had no layer
  list at all, no eye or lock toggles, and `Tab` toggled a flag nothing read.
  Every headless test passed throughout, because they all call the timeline's
  functions directly rather than walking the composition; the missing coverage
  class is a composition walk, and there is now an assertion over the
  workspace's own source that no condition in it mentions the animation.

- **The timeline is always on screen**, at a floor of 150 px with a draggable
  top edge, and `Tab` no longer hides it. That is the same argument one step
  further: this strip *is* the layer list, so every state in which it is off
  screen is a document whose layers cannot be seen or reached — and both of
  the pane's shipped defects were instances of exactly that, the second being
  a clip that ran on with its Stop button hidden while the canvas silently
  refused every gesture.

- **One bar above the canvas, and it is the context bar.** Brush type, size,
  ink, dynamics, pixel-perfect and the symmetry toggles — what a hand reaches
  for between strokes. The view row that used to sit above it is gone: Rotate,
  Flip and Fit were already View-menu rows with chords, so the buttons were a
  second door; Center the page, the four tiling modes (as checked rows) and
  Roll the seam to the middle are View-menu rows now; and the seam figure and
  the `unsaved` word moved to the status bar under the canvas, with the cursor
  position and the zoom, because they are readouts rather than settings.

- **Symmetry composes.** It was one mode at a time, which could not express
  "horizontal and top-left diagonal" — the pair an isometric tile is drawn
  with. The four mirrors are independent toggles on the context bar (`H`, `V`,
  `\` and `/`) with a Reset beside them, radial keeps its own checkbox and
  count in the canvas popover, and any combination is legal. The engine
  reflects by iterating to a fixed point rather than by enumerating cases,
  because two mirrors generate a third that neither of them names: horizontal
  and vertical give the diagonal one for free — which *was* the hardcoded
  third point of the old "both" mode — and horizontal with a 45-degree mirror
  generates all eight. Every legacy spelling still reads, so no stored setting
  and no existing call site changed, and the canvas draws a guide for each of
  the four rather than for two.

- **A colour picker, and a Generation panel.** The picker is RGB / HSV / HSL /
  Gray with a slider per channel and a hex field, as a panel rather than the
  popup that closed on the next click; on an indexed document holding a
  palette slot it edits that entry, which is what an indexed document means.
  The Generation panel surfaces the four pipeline verbs — Make 3D, Save as
  reference, Add to Packwright, Revert to original — that had been File-menu
  rows and nothing else, each greyed with its reason rather than hidden.

- **Fixed: a press-and-drag in the timeline crashed the frame loop.** The
  range gesture asked which cell it was over by splatting the grid's whole
  per-draw scratch dict into a keyword-only signature, so the two cached lists
  added to that scratch this cycle arrived as unexpected arguments. It went
  unnoticed because nothing headless drags and the pane's own tests call the
  function with exactly the arguments it wants; the mapping between the two is
  now named once and asserted against the real signature.

- **A keyboard-remap dialog and an Aseprite-compatible input registry.**
  Commands, tools and held modifiers share one many-to-many binding table, so
  a chord can be rebound, aliased or given a context, and the shortcut sheet
  is derived from it rather than written twice.

## 0.0.27 — 2026-08-22

- **Clay, Plotter and the pixel pipeline stop waiting on work that was never
  arithmetic.** Entering element mode on a 200k-vertex mesh spent three
  quarters of a second building one edge table; the picking tree rebuilt after
  every mesh edit cost a second; a 200x200 three-layer map took four seconds to
  composite; and counting which palette slots a document actually uses made one
  pass over the whole canvas *per palette entry* — eleven seconds on a
  2048-square drawing with a full palette, which is now 117 ms. Nothing about
  what any of them computes has changed: every replacement is asserted equal to
  the code it replaced, bit for bit, and the three new native kernels each keep
  a numpy path beside them that runs on a machine with no compiler.

- **The measurements are in the repo this time, rejections included.**
  `scripts/bench_native.py` is a standing harness rather than a script written
  and thrown away, and `docs/measurements/2026-08-22-native-batch-5.md` records
  every candidate with the bar it had to clear written down before it was
  measured — including the two that were rejected for being slower or less
  exact than what they would have replaced, and the one that turned out to have
  been fixed already.

## 0.0.26 — 2026-08-22

- **A whole-tree audit, written down rather than acted on.** `AUDIT.md` records
  what twelve parallel auditors found across the queue, the database, `service/`,
  the app shell, Inker, Plotter, Packwright, Clay, Poser, Troupe, the pipelines,
  the test suite and the docs — every finding re-verified against the source
  before it was written, and three of them re-graded when verification showed the
  reported severity was wrong. Nothing in it is fixed yet: the file is the work
  list for the next session, and it also records which subsystems were checked
  and found clean, so a later pass does not pay to re-derive them. Four HIGH
  findings lead it — Inker's Symmetry controls are unreachable in the shipped
  app, background/reference layer conversions push no undo step and lose the
  matte, the layers panel is missing the `busy` gate the invariants name it
  under, and pressing Tab during playback leaves the document silently
  read-only. No behaviour changed in this release.

## 0.0.25 — 2026-08-20

- **Warlock stops refusing jobs while the machine has memory free.** A job is
  admitted against a percentage of Windows' *commit* total, and the image model
  used to stay loaded between jobs — around 7 GB whose backing is charged
  against that commit for the whole idle timeout after the job that wanted it
  had finished. On a 64 GB machine with the default pagefile that reads as 96%
  committed while 24 GB of RAM is genuinely free, and every job is refused with
  advice — close other applications, restart Warlock — that cannot help. The
  image model is handed back when the stage that loaded it ends, so a run of
  jobs pays one reload each instead of the queue holding memory against a job
  that may never come. A refused job no longer counts as a finished one either:
  it used to restart the idle clock, so each retry pushed the cleanup that would
  have lifted the refusal another timeout away, and the app could not recover on
  its own from the one state it most needed to.
- **The memory readings now include Warlock's own background work.** Doctor has
  a **host memory** row that says how close the machine is to that wall and,
  when the pagefile is what is holding the limit down, says so — the two causes
  look identical from the error message and want opposite fixes. The session
  log's memory line counts child processes as well: the background cut-out
  worker holds about 6.5 GB while it is alive, which no line in the log had ever
  shown, and it is the largest single thing the idle sweep gives back.

- **Tile sheets come in three views.** The choice used to be *Orthogonal* or
  *Isometric*, and the first of those was spelled "flat top-down" in the prompt
  the model actually reads — so **3/4**, the tilted top-down that most 2D games
  use, was the one framing you could not ask for. There are three now: Top-down,
  3/4 and Isometric. 3/4 shares the square grid with top-down and differs in
  what the model is asked to draw, so it wants subjects with height — walls,
  crates, fences, a well — and on those the difference is unmistakable. Sheets
  made before this still open; the old *Orthogonal* reads as Top-down. Which
  guide 3/4 gets was measured rather than guessed: two interior marks were
  tried, both were obeyed, and both made the sheet worse, so 3/4 gets the same
  plain guide top-down does and the words carry the view.
- **Plotter checks a sheet's view against the map's.** A tile sheet has always
  recorded the view it was drawn for and nothing ever read it back, so an
  isometric sheet added to a square map was sliced into diamonds and laid on a
  square grid with nothing saying so. It asks now — and on an empty map it
  simply brings its lattice with it. Top-down and 3/4 are both square, so
  neither ever asks about the other.
- **A third theme, for the pixel side of the app.** Settings → Appearance now
  offers **Pixel** beside Dark and Light. It is a second dark palette in the
  register the pixel workspaces belong to: warm neutral greys against a
  near-black canvas surround, with amber where the other two spend indigo. The
  Inker is measured against the editor its users already know in what it reads,
  what it writes and how its tools behave — and, until now, in how it looks not
  at all, so somebody who lives in a pixel editor opened it into the cool
  register that belongs to the 3D half of the program. Pixel keeps the dark
  palette's direction, so a raised surface still reads as raised; what changes
  is temperature. It is not an eyedropper copy of anyone's chrome — those are
  typically grey on grey, and every colour in this program has to clear a
  measured contrast bar before it ships. The three themes are now spelled
  *Dark*, *Light* and *Pixel* in the menu rather than in lower case.
- **Clip editing in Poser.** The keyframes a Troupe character sheet animates —
  which keys, in what order, how many frames apart — could only be changed by
  hand-editing a file inside the app's own installation. Poser now has a
  **Clips** panel: pick a key and it loads onto the skeleton you already know
  how to pose, **Update key from pose** puts it back, and everything on top of
  that is timing. **Onion skin** ghosts the keys either side of the one you are
  editing, wrapping round on a looping clip so a walk's first key is judged
  against its last. **Play** scrubs the clip through the same interpolation the
  renderer uses, so what you see is what it will draw. Your clips are saved
  into your own data folder and the ones the app ships are left alone, so an
  update can never overwrite your work and **Revert** is always one click. A
  save that would produce something a character sheet could not be built from
  is refused by name, and leaves your previous clips exactly where they were.
  See the *Editing clips* section of the Poser chapter.
- **Flip and rotate now work on tilemap layers**, on the whole canvas and over a
  timeline range alike. Both mirror the arrangement *and* turn each cell's own
  flip flags, so a mirrored map is made of mirrored tiles rather than a
  mirrored layout of unmirrored ones. Two cases are refused by name because they
  genuinely cannot be drawn: a canvas that is not a whole number of tiles on the
  axis you are flipping, and a quarter turn of non-square tiles. A timeline
  shift works too when it wraps and moves whole tiles. Scale and crop stay
  refused — they resample, and a tileset has no way to follow a resample.
- **Drawing is faster, and ordinary strokes gained the most.** Every dab used to
  recomposite the bounding box of the whole stroke so far, so a long stroke got
  slower the further it went — 33× more work per dab at its end than at its
  start. With mirrored symmetry a single dab landed in four far-apart places and
  recomposited 95% of the canvas. Measured on a 512×512 canvas: **6.6× faster**
  for a plain stroke and 3.9× with X+Y symmetry, recompositing 64–107× less.
- **Every `.aseprite` this build writes now carries a colour palette**, built
  from the colours the drawing actually contains. Aseprite writes one into every
  file it saves and this did not, so a reader expecting the chunk found nothing.
  Nothing is invented — every swatch is a colour painted somewhere in the file.
- **Fixes.** A sprite-sheet export split per layer wrote a fully transparent
  sheet for a visible group whose layers were all hidden; a merged-away frame's
  own slice rectangles were dropped from the sidecar with nothing anywhere
  recording it, and are now noted per cell; and the export refusals that fire
  before the save dialog — padding against extrude, a frame count that no longer
  matches a directional sheet, Arrange or Merge over one — had no test holding
  them at the door they fire from.

## 0.0.24 — 2026-08-20

- **Troupe: character sheets from a 3D model.** A tenth mode, and a chain
  rather than a button: a prompt draws a reference with a T-pose guide, you
  approve it, and the same asset then goes through reconstruction, the auto-rig
  and a 256-cell render without being asked again. The sheet is five animations
  — idle, walk, run, attack and jump — across eight directions, rendered at
  512px per cell and reduced to the pixel size you asked for, then quantised
  against one palette so the same shirt cannot come out two shades in two
  directions. The sidecar carries a tag per animation and direction and a
  duration per frame, so **Edit in Inker** opens the whole sheet on its own
  timeline with the spans already set. See the *Troupe* chapter in the manual.
- **Fixes.** Delete in Packwright sent the *selected library asset* to the
  trash, because that mode's keyboard fell through to the library's;
  ``walk`` and ``run`` shared one internal identity, so every character
  sheet's walk cycle was really its run cycle; the Wrap ½ tool on an indexed
  drawing moved the picture without recording an undo step, and the document
  went on reporting itself saved; a character sheet's progress bar reached 100%
  when Blender finished and stayed there for the whole pixel-art pass; a matte
  preview that failed re-asked for itself every frame, filling the screen with
  the same error; and a re-texture's facing weights were being written
  sRGB-encoded and read back as linear, so the grazing views the blend is meant
  to discard were being kept and over-weighted.

- **Create makes sheets.** The **Output** switch is now *Object*, *Seamless
  tile* or **Sheet**, and a sheet is one of two things. A **tile grid** is 64
  tiles in an 8×8 arrangement — grass, path, water, cliffs, props — drawn as a
  single generation and cut up, so every tile shares one palette, one light
  direction and one style. Two settings and no more: the tile size (16/32/48/64)
  and the view (top-down, 3/4, or 2:1 isometric diamonds); the line
  under them says what the finished sheet comes to. The grid is *imposed* rather
  than asked for — the cell boundaries go to the structure control as a guide,
  so the tiles land where the app is about to cut. A **sprite sheet** turns the
  same prompt into a character and then into a sheet of it: it draws the
  character first and keeps it as its own asset, then imagines two candidate
  sheets from it, so a sheet you dislike still leaves you the drawing it was
  made from. Neither can be made into a mesh. A tile grid reaches a map through
  the library like any other asset — Plotter to paint with, Packwright to cut up.
- **Plotter no longer generates tilesets, and the ground set is gone.** Both
  generators in the tileset pane — the flat procedural one and *Paint with AI* —
  were retired along with the `ground_set` job kind, the new-map dialog's
  *Generate a ground set* option and their two manual sections. A map gets its
  tiles from a file, from Tiled, from Inker or from the library, and making them
  now happens in Create. The **Terrain** tool is untouched: a tileset that
  carries terrain rows still autotiles exactly as before.
- **The manual navigates to sections, not just chapters.** The contents list
  expands the chapter you are reading to show its own headings, indented;
  clicking one scrolls to it, and the heading you are inside stays lit as you
  scroll. Searching now lists the matching *sections* of every matching
  chapter rather than only the chapters, so a search for `gltfpack` lands on
  the passage instead of the top of the file. Arriving at a chapter also puts
  you at the top of it — the page kept the previous chapter's scroll offset,
  which could open a short chapter at its own footer.
- **Two new manual chapters.** **Poser** was a section inside Rigging and
  posing and is now its own chapter, the last workspace mode to get one; the
  Inker timeline moved out of a chapter that had reached 1110 lines into
  **Inker: animation**. Everything from the old chapter 06 onward is
  renumbered, and the manual now names the FLUX.2 klein models, PickScore,
  DINOv2, the prompt expander and ViTPose with the commands to fetch them.
- **Inker edits a whole range of the timeline at once.** With a range selected,
  the cell menu's Range section gains flips, quarter turns, wrapping shifts and
  Fill with foreground — each running over every cel in the selection as a
  single Ctrl+Z, touching a linked cel once however many frames it appears on,
  and leaving an empty cel empty. The turns, flips and shifts move pixels
  without inventing any, so on an indexed drawing two slots holding the same
  colour stay two slots. A quarter turn needs a square canvas and greys
  otherwise: a cel is canvas-sized, so turning the cels without the canvas
  would leave the grid holding two different shapes.
- **A free transform now lands on the whole range.** Committing a move, turn,
  scale or slant replays the same gesture on every cel in the selection, each
  transforming its own drawing rather than receiving a copy of the one you were
  watching. The preview still shows only the cel you are on, and cancelling
  still touches only that one. A pasted buffer lands where it was aimed,
  whatever the range says.
- **Continuous layers.** The new **Cels** toggle at the top of the layers panel
  makes drawing on an empty frame start from a copy of the last drawing on that
  track instead of from nothing — how you carry a held pose or a background
  forward and then change it. It is a copy, not a link, so editing it leaves
  the frame it came from alone. Saved with the document, and an `.aseprite`
  layer marked "prefer linked cels" opens as one.
- **The layers panel is the other half of a range.** Rows inside the selection
  draw highlighted, Shift-clicking a row stretches the selection across the
  tracks between it and the active one, and the eye on a row of a multi-track
  range hides or shows the whole range in one step.
- **Fixed: a filter over a range corrupted an indexed drawing.** It wrote
  colours straight onto the cels without telling the palette planes, so the
  document looked right until it was saved, undone or reordered — and then the
  slots won and the drawing changed. Range writes also read "preserve
  transparency" off the layer row now rather than off the cel, so the lock no
  longer applies to only the frame on screen.
- **Fixed: flipping a floating selection threw away the pixel-art setting.** A
  flip re-rendered the transform with the smooth filter regardless, so pressing
  it mid-gesture blurred a nearest-neighbour rotation.
- **Inker has three colour modes.** A **Mode** row at the top of the palette
  section switches between **RGB**, **Indexed** and **Grayscale**, each a
  whole-document conversion and each a single Ctrl+Z.
- **Indexed is real now: the file stores palette slots, not colours.** Editing
  a slot repaints its pixels across every layer and frame instantly, because it
  is a lookup table changing rather than pixels being rewritten. Two slots
  holding the same colour stay separate, so painting with the second brown and
  recolouring the first later does what you meant. Moving or sorting slots
  becomes an undo step, because it moves your pixels with it — and the
  transparent slot moves with them. One slot is transparent, marked with a
  corner notch and movable with **Make transparent**; it is the only way a
  pixel becomes a hole, so a painted black can no longer collapse into the
  black that means "nothing". Palettes cap at 256 and say so.
- **Grayscale flattens colour to brightness and keeps every later write grey.**
  Storage stays RGBA, which is sound because all nineteen blend modes preserve
  grayness — so the composite of a grey document is grey too.
- **`.aseprite` files keep what they knew.** An indexed file's slots arrive as
  slots rather than being flattened through its table, so a duplicate swatch
  survives the import; a grayscale file opens as a grayscale document. An
  Aseprite background layer painted in the transparent index keeps its colour:
  the slot is duplicated and the layer re-pointed at the copy, which is only
  possible because slots are now what the document stores.
- **Indexed `.ora` files carry the slots.** Each layer is written as a paletted
  PNG, so Krita, GIMP and a browser all open it correctly — and an exported GIF
  writes the slots you painted rather than re-deriving them from the colours.
  Opening such a file in an older build and saving it still writes plain colour
  back, the same cost an animated `.ora` has.
- **Palette-constrained RGB is still there and still works.** Every drawing
  indexed before this version opens in it, unchanged: RGBA pixels with the
  table applied as a write constraint, so soft edges stay legal. Only that mode
  can hold them, which is why it was kept rather than replaced.
- **Packwright imports tilesets.** A `.tsx` or a sliced tileset image is a
  source like any other: the tiles arrive as individual sprites, ready to
  re-pack. The source key carries the tile size, so two slicings of one image
  are two sources rather than one that quietly changed shape.
- **A drawing opened from a JPG, WebP or BMP can no longer be overwritten with
  PNG bytes.** Ctrl+S used to dispatch on the document's *format*, never on the
  file's name, so saving an opened `photo.jpg` wrote a PNG into it — unreadable
  by extension, unrecoverable, and silent. Those suffixes now route to Save As
  with a toast saying why; `.png` and `.ora` save in place exactly as before.
  Re-encoding to JPEG was the alternative and it is the worse one: a lossy
  write over an original, on a keystroke that means "keep what I have".
- **Inker: rulers, 32px grid presets, and a Photoshop-style layers pane.** The
  canvas draws rulers along both edges, the grid presets include 32, and the
  layers pane gets the compact PS7 row layout. The bridge's layer list is in
  stack order again — it had been drawing bottom-up against the canvas.
- **The Create Reference form is flat, and much smaller.** The twelve-select
  creative taxonomy (category, genre, material, era style and the rest), the
  shipped presets, the "found settings" picker, the detail brief and the
  "More options"/"Advanced" folds are gone. What remains is one column:
  Output, Profile, Prompt, References, Seed, Model, LoRA and Negative prompt,
  with Generate pinned below. Your prompt is the brief — no taxonomy axis
  ever measured a quality win (`docs/measurements/2026-08-17-taxonomy-
  retirement.md`), and assets generated under the old fields still reroll and
  promote, composing without the retired fragments.
- **The retirement goes all the way down.** `PROMPT_VERSION` is 5 (the
  empty-params composition is byte-identical; prompts of taxonomy-carrying
  jobs shorten), the findings vocabulary drops the twelve keys (evidence
  re-accumulates per configuration), profiles keep only model, LoRA, strength
  and negative prompt, the vector-preset save mechanism retires (Review's
  "Apply to forms" survives), bench suites are re-minted as `core-v2` /
  `pixel-v2`, and the taxonomy-bearing campaign scripts are deleted — their
  findings live on in `docs/measurements/`.
- **Create's section headings draw as full gray blocks again.** The fills
  were landing under the form's own opaque background, leaving only a sliver
  at the left edge; the block scope now opens inside the form child. The
  Profile row's buttons wrap instead of clipping at 1.5 scale, and the
  evidence hints under Model/LoRA sit on their own line instead of
  overflowing the pane.

## 0.0.23 — 2026-08-16

- **Paint a ground set with AI.** Plotter's tileset pane has a **Paint with
  AI** section: name a theme and say what each terrain is made of, and the
  queue generates two seamless textures per terrain — a surface and its rim —
  reduces each to exactly your tile size, and composites the same forty-seven
  blob cases the flat generator draws. The map stays editable while it paints,
  the progress bar tracks the whole set rather than resetting per texture, and
  the finished atlas is adopted straight onto the map. A set that did not
  finish is not offered as one.
- **Plotter reads and writes far more of Tiled.** Group layers, image layers,
  layer class/tint/offset/parallax, all eight object shapes, object rotation
  and index draw order now survive a round trip in both the `.tmx` and `.tmj`
  spellings — most of them were refusals before. Rotated objects can be
  selected, moved and resized on the canvas, and a resize now keeps the corner
  you pinned exactly where it was.
- **Five things Plotter writes that Tiled cannot read.** An oblique projection
  with its skew, per-layer blend modes, per-object opacity, the capsule object
  shape and list-valued custom properties are Plotter's own — a `.tmx`
  carrying one will not open in Tiled. They are listed as a **Warlock dialect**
  section in `docs/PLOTTER_COMPAT.md` and in the manual, and `.wmap` holds all
  of them without qualification. Exports no longer claim to be Tiled 1.12.2
  files.
- **Blend modes on the canvas, without the stall.** A map using a non-normal
  blend mode is drawn composited, up to a size budget; past it — or while you
  are mid-stroke — the canvas draws layers unblended and says so on its face
  rather than freezing a frame or, on a large map, taking the window down.
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
- **Clay: dragging vertices on an imported mesh is 4× faster.** Every frame of an
  element drag rebuilt the whole object from scratch — including re-triangulating
  it, which the viewport then ignored, since a drag rewrites vertex positions and
  keeps the triangles it already has. A 200 000-triangle import went from 368 ms a
  frame to 92. It is still not smooth at that size; what is left is on the list.
- **Clay: moving the cursor over an element no longer re-uploads the mesh.** The
  wireframe you see in vertex, edge and face mode was thrown away and rebuilt
  every time the cursor crossed onto the next element — several megabytes per
  mouse move on a large mesh. Only the highlight is rebuilt now: 5.6 ms to 0.07.
  Hover picking itself is unchanged and is still the slow part on a big import.
- **Plotter: painting terrain on a large map is up to 41× faster.** Each painted
  cell re-fitted the entire layer instead of the cell and its neighbours, so the
  cost followed the size of the map rather than the size of the brush. On a
  512 x 512 map a painted cell went from 7.9 ms to 0.36; small maps were already
  fine and are unchanged. The tiles chosen are identical, edges and corners
  included.

- **Panel sections are easier to tell apart.** Every heading in a panel now sits
  on its own softly tinted block that runs down to the next one, so which
  controls belong to which group is visible before you read a word. It was
  hardest to see where it mattered most — the Clay tools, outliner and
  properties panes, and Plotter's tools pane, where six headings share a 300
  pixel column. No lines were added; the grouping is the surface. The manual's
  own headings and the library's date groups are deliberately left as they were.
- **Plotter's tileset panel leads with the tiles.** The tile picker for the set
  already loaded is now the first thing in the panel, above **Add from a
  file...**; the palette you click on every stroke used to sit below two
  controls you use once per map. **Generate a ground set** has moved to the
  bottom, as the last resort of the three ways onto a map. On a map with no
  tileset yet nothing is buried: adding and generating are still the whole panel.
- **Plotter's view switches have their own heading.** *Grid*, *Show objects* and
  *Minimap* were three loose toggles that appeared to belong to whichever tool
  was in hand; they are grouped under **View**, which is what they have always
  been about.
- **Clay's Union Objects has a shortcut, and Merge points at it.** `Ctrl+Shift+J`
  next to `Ctrl+J` for the weld. Union was already there and did the job people
  wanted — it removes the walls buried inside an overlap, so two shapes pushed
  into each other come out as the single solid they look like — but it had no
  key and nothing pointed at it, so *Merge Objects* was the half everybody found
  and the buried walls stayed. The merge dialog now says which one you want.
- **Unsaved work is offered on the Home screen, not by a popup at startup.**
  Anything a crashed session left is listed under **Unsaved work** at the top of
  Home, one row per document with its own **Recover** button and **Discard all**
  underneath. Recovering one takes you to it and leaves the rest listed. The old
  dialog appeared before you had seen the app and took everything or nothing, so
  a session that crashed with one document worth keeping and two worth throwing
  away had no right answer. Nothing about what is kept has changed: the list is
  read once at startup, ignoring it keeps every copy, and a copy is cleared when
  you save or close what you recovered.

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
  [Sprite sheets](docs/manual/07-sprite-sheets.md), and the layout paragraph in
  [Inker](docs/manual/08-inker.md).

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

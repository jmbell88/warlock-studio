Build mode — a primitive-based 3D editor for Warlock Studio

 Context

 Warlock generates meshes from images. There is no way to author one. The gap
 is specific: SDXL cannot be prompted into a particular silhouette, and trellis
 reinterprets whatever it is given, so a user who knows the shape they want has
 no path to it short of installing Blender — which is exactly what this app
 exists to avoid.

 Build mode closes that gap with a small object-level modeller: place primitives,
 transform them with gizmos, assign flat PBR materials, and then take the result
 down either of two paths — export the exact geometry as a first-class library
 asset, or render it and feed it to trellis as a shaped reference. It is not a
 Blender replacement and is not trying to be; it is enough to prototype an asset
 in a couple of minutes.

 Scope is split across three phases (agreed with the user). This plan specifies
 Phase 1 in full; Phases 2 and 3 are sketched at the end so the Phase 1 data
 model can be built to receive them rather than rewritten for them.

 ┌─────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │    Phase    │                                                  Delivers                                                   │
 ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 1 (this     │ Object-level editor, end-to-end usable: primitives, selection, move/rotate/scale, materials, snap, undo,    │
 │ plan)       │ save/load, both export paths                                                                                │
 ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 2           │ Vertex/edge/face selection modes, element picking, extrude, inset                                           │
 ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 3           │ Bevel, loop cut, subdivide                                                                                  │
 └─────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 ---
 The shape of the design

 Four decisions carry the whole thing.

 1. A mesh is an immutable CSR n-gon array, and every op is a pure function
 Mesh -> Mesh. No half-edge structure, no mutation, no invariant maintenance.
 Blockout meshes are hundreds of faces, so rebuilding a few numpy arrays per
 operation is microseconds. Three things fall out for free: every op is testable
 as assert op(mesh_in) == expected; undo is a mesh snapshot of a few KB; and
 because the dataclass is frozen, mesh_a is mesh_b is a valid GPU-cache dirty
 check.

 2. One conversion function, three consumers. BuildDoc converts to
 viewer.gltf.Model — the type the loader already produces. That single Model is
 what the viewport uploads (scene.GpuModel), what the GLB exporter serialises,
 and what the send-to-trellis render draws. Verified: scene.py:112 already
 substitutes zero UVs for Primitive.uvs is None, and _face_normals synthesises
 missing normals, so an authored Model renders through the existing pipeline
 with no change to scene.py, render.py or programs.py.

 3. paint is the template for everything above the engine. A pure package
 (studio/build/), a state module, a build_mode.py controller, panes, and the
 ~10 dispatch sites in main.py. Same task-key prefix discipline, same
 dirty == history.head comparison, same uid-addressed undo.

 4. glbio.rebuild_glb is already a GLB writer. It takes arbitrary rest
 bytes, so building a BIN chunk in numpy and handing it over produces a valid
 GLB. tests/test_gltf_loader.py::_glb is the working 6-line proof, and
 tests/test_gltf_loader.py's skinned_glb fixture is a ~60-line worked example.

 ---
 New files

 Pure engine — src/warlock/studio/build/

 Imports numpy, ..viewer.gltf, ..viewer.math3d and ...glbio only. No
 imgui, moderngl, pygame or service — the same rule studio/paint/ follows,
 and what makes every geometric claim assertable headlessly.

 ┌───────────────┬─────────────────────────────────────────────────────────┐
 │     File      │                        Contents                         │
 ├───────────────┼─────────────────────────────────────────────────────────┤
 │ mesh.py       │ Mesh (frozen dataclass) + the pure functions over it    │
 ├───────────────┼─────────────────────────────────────────────────────────┤
 │ primitives.py │ Generator per primitive + the parameter registry        │
 ├───────────────┼─────────────────────────────────────────────────────────┤
 │ ops.py        │ Object-level operations: mirror, snap, transform baking │
 ├───────────────┼─────────────────────────────────────────────────────────┤
 │ document.py   │ Obj, BuildDoc, and to_model() — the one conversion      │
 ├───────────────┼─────────────────────────────────────────────────────────┤
 │ edits.py      │ The Edit subclasses for build                           │
 ├───────────────┼─────────────────────────────────────────────────────────┤
 │ serialize.py  │ .wblk read/write                                        │
 └───────────────┴─────────────────────────────────────────────────────────┘

 mesh.py — CSR storage, chosen so Phase 2 can add faces by concatenation:

 @dataclass(frozen=True)
 class Mesh:
     positions: np.ndarray   # (V, 3) f4
     loops:     np.ndarray   # (L,)   i4  vertex index per face corner
     starts:    np.ndarray   # (F+1,) i4  CSR offsets into loops
     material:  np.ndarray   # (F,)   i4  index into the document palette
     smooth:    np.ndarray   # (F,)   bool

 Functions: face_count, face(mesh, i), edges(mesh), triangulate(mesh) -> (tris, tri_face) (fan; tri_face maps each triangle back
 to its source face —
 Phase 2's face picking needs it, and Phase 1 needs it to group by material),
 render_arrays(mesh) (splits vertices per-corner on flat faces, shares them on
 smooth ones, and computes normals), transformed(mesh, matrix) (mirror bakes
 here rather than using a negative node scale, which flips glTF winding),
 bounds, validate.

 primitives.py — box, plane, cylinder, cone, uv_sphere, torus,
 each (**params) -> Mesh. A GENERATORS registry maps name → (defaults dict,
 builder) so the properties panel is generated from data. This mirrors the
 existing "add a skeleton by adding a JSON file, never by hardcoding bones"
 convention in rigging.py/templates/.

 document.py:

 @dataclass
 class Obj:
     uid: int                       # stable, never reused; the undo address
     name: str
     mesh: Mesh
     translation / rotation / scale # rotation is XYZW, as everywhere else
     generator: str | None          # "cylinder" while live, None once frozen
     params: dict[str, Any]         # the generator's parameters
     visible: bool
     material: int                  # default for new faces; per-face lives on Mesh

 class BuildDoc:
     objects: list[Obj]
     materials: list[gltf.Material]   # reused directly, not a parallel type
     selection: set[int]              # object uids
     history: UndoStack
     rev: int

 Materials are viewer.gltf.Material objects, not a new type. They are
 already pure, they already carry base_color_factor / metallic_factor /
 roughness_factor / emissive_factor / double_sided, and GpuMaterial
 de-duplicates by id(material) — so a shared palette de-duplicates on the GPU
 with no extra work. Unused texture slots stay None.

 to_model(doc) -> gltf.Model emits one Node per object carrying its TRS, and
 one Primitive per (object, material) group. to_primitives(obj, materials) is
 the per-object half, so the viewport can cache by object.

 Live-until-frozen primitives: an Obj with generator is not None shows its
 parameters in the properties panel and regenerates its Mesh on change (one
 MeshEdit). Phase 2's first topology edit sets generator = None and the panel
 switches to a vert/face count with a "frozen" note. Phase 1 never freezes
 anything, but the field exists from day one so Phase 2 adds a line, not a
 migration.

 edits.py — every edit addresses its object by uid, never by index, the
 rule paint/undo.py states and PatchEdit enforces:

 - MeshEdit(obj_uid, before, after) — cost is both meshes' nbytes
 - TransformEdit(obj_uid, before, after) — near-zero cost
 - ObjectAddEdit(index, obj) / ObjectRemoveEdit(index, obj) — hold the Obj
 itself, not a copy, so re-insertion keeps the uid (mirrors LayerAddEdit)
 - ObjectPropsEdit(obj_uid, before, after), MaterialEdit(index, before, after)

 Selection is deliberately not undoable. Paint makes it undoable because a
 lasso is laborious to redo; a 3D object click is not, and Blender's object mode
 agrees. It therefore does not touch history.head, which keeps
 dirty == history.head != saved_head honest — the same reason
 paint/selection.py refuses to push a no-op step.

 serialize.py — .wblk is a zip: scene.json (objects, TRS, generator
 params, material palette, doc meta) plus meshes/<uid>.npz per object. Mirrors
 inker.ora_bytes / write_ora. A zip rather than one JSON because a subdivided
 Phase-3 mesh should not be a megabyte of nested float lists.

 GLB writing — src/warlock/studio/viewer/glbwrite.py

 The inverse of gltf.py, in the same package, over the same types:
 write_glb(model: gltf.Model) -> bytes. Builds accessors, bufferViews and a BIN
 chunk in numpy, emits glTF materials from Material's factors, and finishes with
 glbio.rebuild_glb. No TEXCOORD_0 is written — Phase 1 has no textures and
 TEXCOORD_0 is optional in the spec.

 Placing it beside gltf.py rather than at package root keeps the layering right
 (service/ never imports studio/; the editor produces bytes and hands them
 over) and makes the round-trip test the obvious one to write.

 UI

 ┌────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┬─────────────────┐
 │              File              │                                  Role                                   │   Modelled on   │
 ├────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┼─────────────────┤
 │ studio/build_state.py          │ BuildDoc tabs, tool/snap settings, drag state                           │ paint_state.py  │
 ├────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┼─────────────────┤
 │ studio/build_mode.py           │ The only layer that knows about jobs and task threads                   │ paint_mode.py   │
 ├────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┼─────────────────┤
 │ studio/build_view.py           │ GL: Viewport, Camera, Renderer, GPU cache, gizmos, picking, pygame      │ viewer_embed.py │
 │                                │ events                                                                  │                 │
 ├────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┼─────────────────┤
 │ studio/panes/build_tools.py    │ Add-primitive buttons, transform mode, snap, duplicate/mirror/delete    │ paint_tools.py  │
 ├────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┼─────────────────┤
 │ studio/panes/build_props.py    │ Selected object: numeric TRS, generator params, material editor         │ paint_layers.py │
 ├────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┼─────────────────┤
 │ studio/panes/build_outliner.py │ Object list: select, rename, visibility, delete                         │ paint_layers.py │
 ├────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┼─────────────────┤
 │ studio/panes/build_bridge.py   │ Document facts + the pipeline buttons                                   │ paint_bridge.py │
 └────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┴─────────────────┘

 build_view.py reuses viewer/camera.py, render.py, glctx.py, grid.py,
 gizmo.py, programs.py and capture.py wholesale. It caches one GpuModel
 per object keyed on (uid, id(obj.mesh), materials_rev) — valid because Mesh
 is frozen — and rebuilds only what changed.

 Layout mirrors _paint_workspace exactly, including lay.settings_share:

 [ build_tools   ]           [ build_outliner ]
 [ build_props   ]  viewport [ build_bridge   ]

 ---
 Changes to existing files

 Undo extraction (agreed)

 Move Edit, CompoundEdit, UndoStack, _serials and the UNDO_BYTES /
 UNDO_MAX_DEPTH / UNDO_MIN_DEPTH constants from
 src/warlock/studio/paint/undo.py to a new src/warlock/studio/undo.py.
 paint/undo.py re-exports them and keeps PatchEdit, LayerAddEdit,
 LayerRemoveEdit, LayerMoveEdit, LayerPropsEdit, SelectionEdit,
 ReplayEdit, _pack/_unpack. No name any paint module imports changes, so
 tests/paint/ should pass untouched — that is the check that the move was clean.

 studio/viewer/picking.py — add ray-triangle

 ray_triangles(origin, direction, positions, tris) -> (t, tri_index) | None,
 vectorised Möller–Trumbore. Object picking transforms the ray into object space
 with the inverse TRS and prefilters on the object AABB. This is the one genuinely
 absent capability — the module is currently analytic spheres/planes/rings only,
 by design, because nothing clickable was ever geometry.

 studio/viewer/gizmo.py — add ScaleGizmo

 Three axis handles plus a uniform centre handle, built the same way
 TranslateGizmo is and reusing picking.closest_on_axis. RotateGizmo and
 TranslateGizmo are already documented as knowing nothing about bones, so they
 are used as-is.

 service/jobs.py — import_mesh

 Modelled line-for-line on import_reference (jobs.py:239), which is the
 established way to mint a done row with no worker run:

 def import_mesh(svc, glb: bytes, *, name=None, prompt=None, size_m=None) -> dict:
     # size + GLB validation at the door
     # job_dir.mkdir(); write source.glb; copy to model.glb   <- files BEFORE the row
     # postprocess.normalize_glb(model.glb, size_m)  — logged and swallowed on failure
     # meshreport.build(...) into params
     # params = {"built": True, ...}
     # store.create("image", prompt or "", params, job_id, stage="model", status="done")
     # except: shutil.rmtree(job_dir); raise
     # set_meta(name)

 source.glb is the authored mesh and model.glb derives from it, which is the
 existing invariant and is what keeps optimize_job's retarget working on a
 built asset. Grounding via normalize_glb applies to every job, built ones
 included. No new artifact names are needed — model.glb, source.glb and
 thumb.png are already in files.MEDIA.

 The payoff is disproportionate: a built asset immediately inherits rigging,
 posing, sprite sheets, triangle retarget, and STL/OBJ/FBX/collision export,
 because all of those are pure functions of model.glb.

 Guards. rerun_job already refuses both modes for a job with no input.png,
 but with a message about references. Add a params["built"] branch giving the
 right one. promote_to_model needs nothing — it already requires
 stage == "reference".

 service/validation.py / service/files.py

 MAX_MESH_BYTES plus a GLB magic + "has a mesh" check (belt-and-braces, since
 we author the bytes — but "inputs are bounded at the door" is a stated
 invariant). save_build_source / build_source_status in files.py, mirroring
 save_paint_working, for assets/<job_id>/build.wblk.

 build.wblk follows the paint.ora precedent: deliberately absent from
 MEDIA and LISTED, never served. It does not get the staleness rule —
 _save_linked's rule exists because a revert can rewrite input.png behind the
 layers, whereas a build export writes the GLB and the .wblk in one operation
 and a later optimize_job retarget does not make the authored source wrong.
 Order is still GLB first, .wblk second, so a crash between them leaves the
 sidecar absent rather than lying.

 studio/main.py — the ~10 dispatch sites

 Exactly the set paint occupies:

 1. _mode_switch (:899) — one more tuple: ("build", "Build")
 2. _build_ui (:732) — if mode == "build": self._build_workspace(); imgui.end(); self._overlays(viewport); return
 3. _build_workspace() — new, modelled on _paint_workspace (:793)
 4. _sync_viewer (:493) — early-return; build owns the centre pane
 5. _shortcut (:616) — first refusal via build_mode.handle_key(ctx, event) -> bool, both key edges
 6. _shortcuts_popup (:945) — rows derived from build_mode.TOOL_KEYS
 7. _collect_tasks / _on_task_done (:307, :353) — claim the build- prefix, both on_task_done and on_task_failed
 8. _on_drop (:658) — a .wblk opens; a .glb is refused with a message (mesh import is not Phase 1)
 9. _request_quit (:682) — join the guard chain alongside paint_mode.guard
 10. _persist / teardown (:1135) — persist settings, release GL

 Plus state.py (build: Any = None, lazily built), panes/landing.py (a tile +
 start_build), panes/overlay.py (placeholder text).

 Keyboard

 build_mode.TOOL_KEYS: G/R/S move/rotate/scale, Ctrl+D duplicate,
 Delete, F frame selection, Ctrl+Z/Ctrl+Y, 1/2/3 reserved for Phase
 2's element modes. Build takes first refusal, so the global F/W/S bindings
 (which act on the 3D viewer) are correctly shadowed.

 Docs

 A docs/manual/NN-build.md chapter plus HELP_TARGETS entries and
 help_button calls in each pane. This is gating: tests/manual/test_docs.py
 validates every target, every cross-link, every anchor, and that the index links
 each chapter exactly once. Also add the Build-mode invariants to CLAUDE.md,
 which is how this repo records load-bearing design rules.

 ---
 The two output paths

 Both live in build_bridge.py, both go through build_mode.py.

 Export to library — write_glb(to_model(doc)) on a task thread (pure numpy,
 no GL), then jobs.import_mesh, then save_build_source for the .wblk. On
 completion the frame thread calls the existing ctx.capture_thumbnail(job_id).
 The card appears in the library like any other asset.

 Send to 3D — render the doc offscreen at 1024², flat-shaded on a plain
 background with no grid, gizmos or overlays (Renderer.draw(..., flat=True, overlays=[]) into a dedicated Viewport, then
 capture.png_bytes). This runs
 synchronously on the frame thread because it needs the GL context — one
 offscreen draw, the same thing ctx.capture_thumbnail already does. The bytes
 then go to a task thread calling create_job(kind="image", output="model", image=png), which is precisely what
 paint_mode.send_to_3d does at
 paint_mode.py:398.

 ---
 Verification

 Unit — the pure engine, no GL, no window:

 - tests/build/test_mesh.py — CSR invariants, validate, fan triangulation and
 tri_face correctness, flat-vs-smooth vertex splitting, transformed
 - tests/build/test_primitives.py — per primitive: closed, expected vert/face
 counts, bounds, and outward winding (sum(dot(centroid - origin, normal)) > 0),
 which is the one that catches a flipped cap
 - tests/build/test_document.py — undo/redo lands on the right object after a
 reorder and after a delete (the uid rule); dirty goes false again on undoing
 back to the saved head; a no-op pushes no step
 - tests/build/test_serialize.py — .wblk round trip preserves uids, TRS,
 generator params and the material palette

 Round trip — the strongest available test:

 - tests/test_glbwrite.py — author a gltf.Model, write_glb, load it back
 with gltf.load, assert identical positions/indices/normals, material factors
 and node TRS. The writer is verified against the reader that already ships.

 Service:

 - tests/test_build_service.py — import_mesh yields status=done,
 stage=model, both GLBs on disk, a mesh report in params; it cannot be
 rerolled or remeshed; oversize and non-GLB bytes are refused; a failed insert
 leaves no job dir

 GL, via the existing gl() fixture and create_standalone():

 - tests/test_build_view.py — an authored doc uploads and renders non-empty
 pixels; the GPU cache rebuilds only the object whose frozen Mesh changed
 - ray-triangle picking against a known box from a known camera

 Regression:

 - tests/paint/ must pass unchanged after the undo extraction
 - tests/test_studio_smoke.py — build each new pane under FORCE_SECTIONS_OPEN
 - tests/test_studio_plumbing.py — source assertions pinning the UI→service wiring
 - tests/manual/test_docs.py — the new chapter and its HELP_TARGETS

 End to end, by hand (uv run warlock — the /run skill covers launching it):
 add a box and a cylinder, move/rotate/scale each, set two materials, save the
 .wblk, reopen it, export to the library, confirm the card appears with a
 thumbnail and a mesh report, download the GLB and reopen it in the 3D viewport,
 then rig it — proving a hand-built mesh is genuinely first-class. Separately,
 Send to 3D and confirm a trellis job queues from the render.

 Full gate: uv run pytest and uv run ruff check ..

 ---
 Phases 2 and 3 (not this plan)

 Recorded so Phase 1's data model is built to receive them.

 Phase 2 — element editing. BuildState.element_mode (object / vert /
 edge / face); element picking (ray-triangle → tri_face for faces, screen-space
 nearest for verts and edges); element selection sets on the document; and
 extrude / inset as pure Mesh -> (Mesh, new_faces) functions. The first such
 op sets Obj.generator = None. Nothing in Phase 1 needs to change: CSR append,
 tri_face, the frozen-mesh cache key and the generator field are all already
 there for this.

 Phase 3 — topology tools. bevel (vertex miters are the hard case),
 loop_cut (a quad-strip walk over the derived edge adjacency), subdivide,
 merge/weld.
# Inker: Aseprite-Inspired Roadmap — Phased Master Plan

## Context

Inker already covers Aseprite's core drawing loop (pixel brushes, layers, selections, palettes,
linked cels, onion skinning, tags, GIF/sheet export). The gap is production workflow: timeline
range editing, slice/pivot metadata that reaches Packwright, tiled painting, palette conversion
tooling, layer groups/locks, and interop. This plan covers the full roadmap (all six areas) for
execution **in one session using multiple agents**, on a single branch `inker-update` off
`master`; commits stay at the current version (`Warlock v0.0.21`) unless a bump is requested.
The design below was verified against the code and `docs/INVARIANTS.md` as of 2026-08-14.
The authoritative copy of this plan lives at `INKER_UPDATE.md` in the repo root — syncing that
file with this plan is the first execution step.

**Aseprite compatibility audit (2026-08-14):** a section-by-section sweep of the Aseprite 1.3
docs against Inker's actual surface (engine + panes + manual) produced Phase C below — the
parity gaps not already covered by phases T/S/P/L/Q — plus a Deliberate Divergences list.
Target per the user: Inker relatively compatible with Aseprite EXCEPT plugins/extensions and
advanced settings. Decisions taken: cel opacity skipped (divergence), simple text tool included,
FX staples only (no convolution matrix / color curves), full blend-mode set included.

## Execution model — one session, multiple agents

Five parallel tracks plus a serialized integration lane. Track contents reference the phase
sections below (T-A, S, T-B, P, L, Q).

| Track | Phases | Primary file territory |
|---|---|---|
| 1 | T-A A1–A5 | `_doc_ranges.py` (new), `anim_edits.py`, `sheetout.py`/`gifout.py`, `inker_timeline.py`, `inker_preview.py` (new), `inker_mode.py` export/preview |
| 2 | S S1–S5 | `slices.py`/`_doc_slices.py` (new), `ora.py` (warlock.json), `transform.py`, `pipelines/sheet.py`, `pixelsheet.py`, packwright engine |
| 3 | T-B B1–B5 | `tiling.py` (new), `brush.py` (`_stamp` wrap), `_doc_paint.py` (wrap kwargs), `selection.py`, `inker_canvas.py` view/input |
| 4 | P prereqs + P1–P7 | `dither.py` (new), `_doc_indexed.py`, `inker_colors.py`, `inker_bridge.py` |
| 5 | L L1–L3 | `layers.py`, `animation.py`, `_doc_layers.py`, `groups.py` (new), `inker_layers.py`, `ora.py` (locks/groups) |
| 6 | C6, C8, C14 | Blend modes (`composite.py` — clean territory), FX staples, text tool |
| — | Q d→a→c→b | `asein.py` (new) is safe to parallelize; a/c/b only if capacity remains (b starts with a spike doc) |

**Phase C (parity sweep) rides the tracks** — assignments: C1 fixes land on the branch root before
tracks fork; C7 tag repeat + C9 range filters + C13a-c export options → Track 1 (C7 before A5 —
`advance` signature widens); C15 tiled axes folds directly into T-B's build + C2 replace ink,
C3 spray, C10 move-without-selection, C12d right-click-BG → Track 3 (owns brush.py + canvas
input); C11 gradient dither + C12a-c palette extras → Track 4 (after P1); C5 selection/transform
depth → Track 5 (owns `_doc_selection.py`/`transform.py`/`_transform_input`; stated collision:
Track 3 owns `_press`/`_input` dispatch, Track 5 owns `_transform_input`/`_handles`); C6/C8/C14 →
Track 6; C13d slice export → integration lane after Track 2; C4 poly lasso → serialized lane
after Track 3, before Q-c (Q-c reuses its gesture infra).

**Known cross-track collisions — assign explicitly, do not discover at merge:**
- `brush.py`: Track 3 (B2 `_stamp` wrap) and Track 4 (P6 `_shade`). Rule: Track 3 owns `brush.py`;
  P6 shading is implemented **after** Track 3 merges (it also depends on P4 multi-slot).
- `ora.py`: Track 2 (S2 warlock.json) and Track 5 (L1 lock attr, L3 groups). Rule: Track 2 merges
  first; Track 5 rebases its ora.py work on the merged result.
- `document.py` (mixin list, `_ensure_cel_for`, `_replay`/`ReplayEdit` fields), `inker_state.py`,
  `undo.py`, `tests/inker/test_inker_imports.py` smoke list, `docs/manual/07-inker.md`,
  `HELP_TARGETS`, `CHANGELOG.md`, `docs/INVARIANTS.md`: every track touches at least one — treat
  as **merge hotspots**; each track keeps its edits additive and minimal (trailing defaulted
  fields, new list entries), and the integrator resolves them at each merge.

**Worktree + suite discipline** (hard-won repo rules): each agent works in its own git worktree —
worktrees need the 3 `WARLOCK_*` env vars and a venv synced with
`--extra studio --extra text2image --extra rig` (a bare `uv sync` breaks ~10 files at collection);
the git stash is repo-wide, never stash in a worktree; **never run two pytest suites
concurrently** (they fake failures) and never edit `src/` while one runs. Tracks run their
targeted lanes (`uv run pytest tests/inker tests/packwright tests/manual`) inside their own
worktree; the **full suite runs only on the merged tree**, once per integration step and once at
the end.

**Integration order** (onto `inker-update`, one track at a time, full suite between):
C1 → Track 1 → Track 2 → Track 3 → Track 5 → Track 4 (P6 last, after brush.py and multi-slot
exist) → Track 6 (C6 → C8 → C14) → C13d → C4 → Q-d (+ any Q remainder, Q-c consuming C4's
gesture). INVARIANTS.md updates land with their tracks (L2 retires the `can_restructure` refusal
in §127; S2/L3 extend §129; groups add pass-through prose); one CHANGELOG sweep at the end.
Tool shortcut letters claimed this plan: **A** spray, **C** slice, **D** poly lasso, **T** text —
integrator checklist item so two tracks never mint one letter twice. `TOOLS` rows and
`TOOL_OPTION_DEFAULTS` keys are appended trailing by each track; the integrator orders the
toolbox once at the end.

Cross-track design notes: `ReplayEdit` grows trailing defaulted fields (S1 `slices`) and P adds a
separate `PaletteEdit` type — no clash; L3 groups' lock resolution composes with L1's content
lock; nothing in Q blocks anything.

## Phase T-A: Timeline range editing (roadmap item 1) — Track 1

**A1. Engine: range ops mixin + `FrameOrderEdit`** — new `studio/inker/_doc_ranges.py` (`RangeOps`
joins the mixin list in `document.py:96`; joins the import-smoke list; `OUTWARD_IMPORTS` unchanged):
- Range = inclusive index rect `(t0,t1,f0,f1)`, **view state on the tab**
  (`InkerDoc.range_sel`), indices not uids (the `Tag` positional argument), clamped at use never at
  store (plotter selection rule). The engine never sees the selection — every op takes explicit
  indices, headlessly testable.
- New `FrameOrderEdit(before, after)` in `anim_edits.py`: uid sequences, cost 0, one
  `_set_frame_order` doc hook that re-derives the playhead **by uid**. One edit = one
  `_anim_changed` (vs 50 recomposites from chained `FrameMoveEdit`s).
- Ops, each = clamp → `commit_floating()` → mutate via existing raw hooks → exactly one
  `CompoundEdit` → bool; no-op pushes nothing:
  - `remove_range(f0,f1)` — refuse whole-timeline; drop **descending** (CompoundEdit undoes
    reversed, so re-insert indices are valid); partition released cel `id()`s across edits
    first-occurrence so linked cels are charged once.
  - `clear_range(t0,t1,f0,f1)` — one `CelSetEdit(before, None)` per occupied slot;
    first-occurrence pinning (CelSetEdit.pinned is a bool).
  - `duplicate_range(f0,f1, at=None, link=False)` — copy **per distinct object**
    (`dict[id(src) -> src.copy()]`): internal links preserved, no link back to source;
    `link=True` shares source objects, `pinned=False` (matches `add_frame(link=True)`).
  - `move_range(f0,f1,to)` / `reverse_range(f0,f1)` — one `FrameOrderEdit`; links survive
    trivially (cels keyed by uid).
  - `link_range` — earliest occupied cel per track becomes the shared object;
    `unlink_range` — copies minted **at op time**, held by their edits (redo returns the same
    identity — no stranded patches, the `unlink_cel` rule).
  - `set_range_duration(f0,f1,ms)` — one `FrameDurationEdit` per frame that actually changes.
  - `copy_cels -> CelClip` (pure read; planes deduped by distinct cel + `slots` index map — the
    `ReplayEdit.grid["slots"]` trick preserves internal links) / `paste_cels(clip,t0,f0)` —
    refuses size mismatch; pasting past the end appends blank `Frame`s **inside the same
    CompoundEdit** (autovivify pattern one level up), so one Ctrl+Z removes cels and conjured
    frames together. Clip lives app-level (`InkerState.cel_clip`) for cross-tab paste.
- Tests `tests/inker/test_timeline_ranges.py`: one step per op; undo restores frame uids,
  durations, tags AND link identity (`is` assertions); budget charged once for shared cels;
  ops after a reorder land right; playhead stays on the same frame uid; refusals.

**A2. Timeline UI marquee + range menus** (`panes/inker_timeline.py`): geometric drag tracking
(pressed imgui buttons suppress neighbour hover) using the `_tag_row` column arithmetic and the
reversed row order; press = 1×1 range + existing click behaviour; Shift+click extends from
`InkerState.timeline_anchor`; Esc clears; range drawn as one accent `add_rect`. `_cell_menu`
gains the range section (copy/paste/clear/link/unlink/duration/reverse/duplicate/delete/export),
all `tab.busy`-gated, disabled-not-hidden. Cell↔index mapping extracted as a pure function with a
unit test. Trap: map mouse through the scrolling grid child via captured screen rects.

**A3. Cel thumbnails**: per-**layer** stamps `Document._layer_stamps` bumped in `_stamp_layer` and
whole-grid paths (NOT in `invalidate_all` — the "stamps no frame" lesson);
`inker_textures.cel_thumb` keyed `inker_tex:{tab.uid}:cel{layer.uid}` — linked slots share one
texture; 0.25 s refresh throttle (B24 idiom); per-tab LRU capped `CEL_THUMB_CAP = 512` (~2.6 MB),
eviction via `docmodes.forget_texture`, tab close swept by the existing prefix sweep. Cells grow
to 36 px behind an app-level `timeline_thumbs` toggle; only `is_rect_visible` cells request
thumbs; off path byte-identical to today. Reach cel pixels through `anim.cels`, never
`stack[index]`; placeholders draw the empty button. Tests `tests/inker/test_layer_stamps.py`.

**A4. Range/tag export**: `sheetout.frame_uids(doc, span=None)` + `timing(doc, span=None)` with
pure `rebase_tags(tags, f0, f1)` (clamp/shift/drop); partial span forces `layout=None`;
`gifout.write_gif(..., loop=True)` — non-looping tag omits the netscape loop extension (the
docstring's deferred feature); whole-timeline output stays byte-identical (pin it).
`_Export.span` sliced at begin, `timing` sliced at submit with the same span — safe because the
tab is locked (`saving`) for the whole spread. UI: tag menu "Export tag → GIF/sheet"
(`loop=tag.loop`), range menu "Export range". Tests `tests/inker/test_range_export.py`.

**A5. Live preview pane** — new `panes/inker_preview.py`:
- Per-tab `preview_playing/index/accum_ms/forward` on `InkerDoc`; `inker_mode.tick_preview` clones
  `tick_playback` (pure `animation.advance`, `MAX_TICK_MS` clamp), **never** touches
  `tab.playing`/`busy` and never calls `set_current_frame` — edit-while-preview-plays needs zero
  gating changes (the preview only reads `frame_flat`, same as onion skinning; read-safe even
  during save, the `sheetout.snapshot` argument).
- Draws the existing `inker_textures.frame_texture` textures (zero new GPU state); 0.25 s
  re-flatten throttle while painting the shown frame; upright `add_image`, ignores `PaintView`
  orientation deliberately. Transport: play/pause + frame counter, **plus (parity audit) a
  playback-speed multiplier (×0.25–×4, scales dt before `tick_preview`) and a play-scope choice
  (whole clip / active tag)** — both per-tab preview state beside `preview_playing`.
- Placement: top of the right column in `main._inker_workspace`, `PREVIEW_H ≈ 180`, drawn only
  when animated. Manual: `help_button(ctx, "inker-preview")`, `HELP_TARGETS["inker-preview"] =
  ("07-inker", "preview")`, new `## Preview` section (no chapter renumbering).
- Tests: `tick_preview` headless (advances preview fields, never moves `anim.current`, never sets
  playing/saving, respects tags/pingpong, stops at non-looping tag end).

## Phase T-B: Seamless tiled painting (roadmap item 3) — Track 3

**B1. One shared wrap helper** — new `studio/inker/tiling.py` (pure numpy; joins smoke list):
`spans`, `pieces(rect, size)` (dest-rect + src-offset segments), `fold_coverage(mask, origin,
size)` (np.maximum fold → bbox + weight), `canonical(point, size)`. Overlap-safe on the paint
path because coverage is max-accumulated and `_resolve` recomputes from `before`.

**B2. Wrapped strokes**: `StrokeState.wrap`; `_stamp` (`brush.py:489`) is the single choke point —
per `tiling.pieces` piece, slice the stamp and run the existing body; `_mark` per piece.
Selection clipping unchanged (dest rects are in-canvas — selections stay scoped to the canonical
tile for free). Symmetry composes for free (mirroring precedes stamping). **Smudge excluded**
(seam-crossing pickup ill-defined; falls back to clamped); blur wraps piece-locally (stated
limitation). Undo: **one `PatchEdit` over the union bbox** — tiles are small canvases; a
multi-rect edit type isn't worth new eviction accounting. `begin_stroke(..., wrap=False)` threads
it; the flag comes from `InkerDoc.tiled` (per-tab view state, never in the file) passed
explicitly by the canvas — the engine stays stateless about UI. **Amendment (parity audit):
`tiled` is an axis enum `("off","x","y","xy")`, not a bool** — the roadmap and Aseprite's Tiled
Mode both offer per-axis wrap; `tiling.pieces/spans/canonical` take an `axes` parameter and the
canvas UV-repeat/checkerboard extent honour it per axis (details in Phase C15).

**B3. Wrapped fill and shape**: `magic_wand(..., wrap=)` → toroidal contiguity by iterating
Pillow's C flood to a fixpoint over seam seeds (typically ≤ 2 floods). `shape(..., wrap=)` draws
the PIL coverage at the gesture's offset then `fold_coverage` → the unchanged `write_colour`.
**Gradient deliberately does not wrap** (a ramp has endpoints — docstring states it). Floating
commit/paste stay canonical-only in v1 (noted, not silently wrong).

**B4. Canvas 3×3 view + input mapping**: UV-repeat one quad (the `overlay.TILE_REPEAT` /
`main.py:4241` precedent) — set `texture.repeat_x/y` on **both** branches every frame (the
one-way-door lesson: LINEAR sampling bleeds the far edge otherwise); extend `_blit` with
`uv0/uv1` and draw the composite over `(-W,-H)–(2W,2H)` with UVs `(-1,-1)–(2,2)` — canonical tile
centred, all four seams visible; rotation/flip free via `add_image_quad`; checkerboard box
extends to the 3× AABB; overlays stay canonical-only in v1. Input: click-tools fold points via
`tiling.canonical`; drag gestures compute the tile offset at `_press` and subtract it for the
whole gesture (folding per-point would jump the brush a full tile at the seam). Toggle "Tiled" in
`_view_row`, driving view and writes together.

**B5. Tests + manual**: `test_tiling.py` (helper properties); `test_tiled_paint.py` — **torus
translation-equivariance** (stroke at p vs p+delta equal under `np.roll`), seam-crossing stroke =
roll of centre stroke, off-path parity (wrap on/off identical for interior strokes), wrapped fill
connects across the seam, smudge/gradient ignore the flag, alpha-lock holds per piece. Manual
sections + CHANGELOG.

## Phase S: Slices, pivots, nine-slice → Packwright (roadmap item 2)

Purely additive; ships independently of everything else. Five sub-phases in order.

**S1. Slice model (engine)** — new `studio/inker/slices.py` + mixin `_doc_slices.py`:
- `Slice(name, bounds x0y0x1y1-exclusive, pivot: (float,float)|None from bounds origin,
  center: rect|None, keys: dict[frame_uid, SliceKey], uid=new_uid())`, `Slice.at(frame_uid)` resolves
  key-or-base. `SliceKey` frozen. Hangs off `Document.slices: list[Slice]` (slices matter on still docs too).
- Per-frame keys: **frame uid at runtime, index on disk** (the `cels` precedent); a key whose uid left the
  grid is skipped at save (accepted small leak, like `_placeholder_uids`).
- Undo: per-slice uid-addressed edits (`SliceAddEdit`/`SliceRemoveEdit` hold the object;
  `SliceChangeEdit(uid, before, after)` dicts), all cost 0 — NOT the Tag whole-list snapshot (a slice has
  durable exported identity; drags need O(1) steps). "A step that changes nothing is not pushed" holds.
- Ops: `add_slice/remove_slice/set_slice(uid, was=...)/set_slice_key`. Geometry ops map slices via new pure
  rect mappers in `transform.py` (flip/rotate90/scale/crop/resize; clip floors at 1×1, never deletes);
  `ReplayEdit` grows a trailing defaulted `slices` field (deep copies, uids preserved) restored by
  `restore_snapshot`. `doc.rev` ticks on slice changes; composite/dirty untouched.
- Tests `tests/inker/test_slices.py`: uid-addressed undo after reorder, no-op push, key survives
  frame-delete+undo, rotate90 exact, crop clamp.

**S2. `warlock.json` ORA member** (`inker/ora.py`): `WARLOCK_MEMBER="warlock.json"`, version 1, slices with
`{x,y,w,h}` rects, pivot/center/keys written only when set, keys sorted by frame index. Written through
`_member()` **only when slices exist** — slice-less archives stay byte-identical (determinism suite green).
Reader is `.get`-based; bad version or any entry failing on its own terms drops the whole member (never the
file); keys discarded (logged) when the grid read fell back flat. Separate member, not `animation.json`,
because slices exist on still docs and `animation.json` fails whole-grid. Tests: round-trip, old files →
`slices == []`, byte-determinism, malformed-member negative controls, journal recovery carries slices
(ORA bytes ride `ora_bytes` — assert, don't trust).

**S3. Canvas overlay + editing** — a **tool** (`("slice","Slice","C")` in `inker_state.TOOLS`; verify C
unbound), not a pane (no new HELP_TARGETS anchor needed; slice section appears in `panes/inker_tools.py`
when active: list, name, pivot/center toggles, "Key this frame", Delete). View state on `InkerState`:
`slice_uid` (0 = none, stale tolerated), `show_slices` (forced on with tool). Overlay `_slices()` in
`inker_canvas.py` draws via `_corners`/`_box` transformed corners (never `origin + x*zoom` — quarter-turn
invariant); handles/pivot crosshair/dashed center. Drag kinds `slice-new/move/resize/pivot/center`; hit-test
in image coords via `view.to_image`; live-mutate during drag, one `set_slice(was=...)` step at release.
Keys are **explicit** ("Key this frame"), never implicit. Gate on `tab.busy`; don't fall through to paint;
respect `state.transforming`. Tests: `_Lines` fake under rotation 90/flip; one edit per gesture.

**S4. Sheet sidecar** — **additive, no version bump** (the square-sidecar byte-equality pin is the negative
control): `pipelines/sheet.py::sidecar(..., pivots=None, slices=None)`; per-cell pivots override, falling
back to today's bottom-centre constant. Pivot source rule: **first slice in document order carrying a
pivot**, resolved per frame, canvas coords. `sheetout.py` grows `slices_snapshot(doc)`/`slices_block()`;
`snapshot()`/`compose()` grow a paired fifth element; `inker_mode._submit_export` passes both — `sheet.py`
stays the format's sole writer. `pipelines/pixelsheet.py::pixel_sidecar` rescales the slices block
(floor-origin/ceil-extent like trim; pivot as float). Tests: square pin untouched, non-square carries
slices, per-frame pivot lands on the right cell, pixelsheet rescale exact on divisible sizes.

**S5. Packwright plumbing** — metadata as plain data, import pins untouched:
- `sources.py`: frozen `SliceSpec` + `SpriteMeta(pivot, slices)`; `Sprite` gains trailing
  `meta: SpriteMeta = EMPTY_META`. `sprites_from_document` duck-types a new
  `Document.sprite_meta_for_frame(frame_uid) -> plain dicts` (keys resolved on the Inker side, coerced in
  `sources.py` — Packwright never learns frames-vs-keys). `PackDoc.sprites()` rename-rebuild site must copy
  `meta` through (test pins it).
- `layout.Frame` gains trailing defaulted `pivot`/`slices`; layout determinism untouched.
- `texturepacker._frame_entry`: `"pivot": {(px-tx)/w, (py-ty)/h}` normalized against the **trimmed frame
  rect** (exactly as `spriteSourceSize`); fallback 0.5/0.5 keeps schema stable; fully-trimmed 1×1 rect makes
  division safe (pin it). Replace the module docstring's "nowhere to come from" paragraph.
- Nine-slice: additive per-frame `"slices"` block in **source-image space** (no trim interaction), emitted
  only when non-empty — existing exports byte-identical.
- `wpack`: manifest source entries gain `pivot`/`slices` (version stays 1, `.get` reads); reader
  refuses malformed meta by name (wpack is recognise-or-refuse). Round-trip + byte-identity tests.
- Never name anything `ninepatch` (`studio/ninepatch.py` is an unrelated UI primitive).

## Phase L: Layer safety — content lock, animated restructure, groups (roadmap item 5)

Split so the two independent halves land before groups (the deep structural change).

**L1. Content lock** — `Layer.locked` + `Track.locked`: **the sixth track property** — fix every copy site
(`document.py::_ensure_cel_for` ~L533, `animation.py::Track.of/props/placeholder/layers_for`,
`layers.py::Layer.copy`, `ora.py` both stack-xml writers + both readers) and the three-places comment at
`document.py:531`. Enforcement at the engine op doors, refusing before any mutation (undo writes raw pixels
below the doors, so undo can't break): `begin_stroke` (before `_ensure_active_cel`), `write_colour`
(covers fill/shape), `gradient()` (top — **and fold in the known alpha-lock gap**: restore `out[...,3]`
after the ramp composite at `_doc_paint.py:441`, own regression test), `begin_filter`, `lift()` (copy stays
legal), `paste()` (`paste_as_layer` stays legal), `merge_down` (either participant). `commit_floating`
deliberately NOT refused (a float outlives lock toggles; wedging saves is worse — docstring states it).
Document-scope ops (geometry, matte, palette recolour, flatten) apply regardless — the lock guards
tool-level writes; one owning comment. UX: engine returns False silently; canvas raises one toast per press;
`disabled_button` for Merge down; property edits/rename/hide/reorder/delete stay legal. ORA:
`warlock-content-lock` attr (written only when set — byte-identity), `Track.props()` gains `locked`,
version stays 1. Tests `tests/inker/test_content_lock.py`: every door, undo on since-locked layer restores,
round-trips, foreign-ORA-unlocked negative control.

**L2. Animated merge-down/flatten** — retire the `can_restructure` refusal (update INVARIANTS §127 prose):
- `merge_down`: one `CompoundEdit`; memo keyed on `(id(lower), id(upper))` cel pairs so frames sharing both
  cels share **one** merged Layer (link preservation by identity, the `unique_cel_layers` doctrine); upper
  absent → no edit for that slot; merged cels are fresh Layers minted once at op time and held by
  `CelSetEdit` (redo returns the same object — no stranded patches); lower's in-place mutation forbidden
  (may be linked where upper differs); `TrackPropsEdit` bakes opacity/blend to defaults;
  `TrackRemoveEdit` last with `charged`-style pinning; `_stamp_all` + `_anim_changed`.
- `flatten_layers`: via `_replay`/`ReplayEdit` (whole grid changes anyway); link partition computed
  structurally at op time as frame-index groups, one uid minted per group + one track uid, closed over
  (replay purity). Durations/tags untouched.
- Tests `tests/inker/test_anim_restructure.py`: link preservation by `is`, absent-upper identity, undo
  restores link structure, redo lands on same uids, byte-budget charging.

**L3. Layer groups** — **parallel tree over the flat stack** (new `studio/inker/groups.py`); the flat
`LayerStack` stays authoritative for paint order (composite slicing, `_below` arithmetic, track==stack
index, `_materialize_frame`, native `stack_region` all untouched):
- `GroupNode(name, visible, opacity, locked, uid=new_uid())`; `Document.groups: dict[uid, GroupNode]` +
  `group_of: dict[member_uid, group_uid]` (absent = root). Invariant: a group's leaves are contiguous in
  stack order and spans nest — enforced by ops, checked by pure `groups.check()` in tests. **Empty groups
  disallowed** (created around selected layers; dissolve when last member leaves).
- Composite semantics: **pass-through** — `groups.resolve()` folds visibility (AND), opacity (multiply),
  lock (OR) down the ancestry; fed into the `_entries` walk. Isolated-group compositing deferred **by name**
  in the docstring + manual. Krita opacity numbers survive round-trip.
- **Placeholder-uid trap**: on animated docs resolve by `anim.tracks[i].uid` (materialised placeholders
  carry per-slot uids) — documented in `groups.py`, tested via hide-group-hides-empty-cel.
- Undo: uid-addressed cost-0 edits — `GroupAddEdit`, `GroupDissolveEdit`, `GroupPropsEdit`,
  `MembershipEdit` (in a CompoundEdit with the move that maintains contiguity). Move-into-own-subtree
  refused by name. `merge_down` across a boundary keeps the lower's membership.
- Collapse state = view: `InkerDoc.collapsed_groups: set[int]` on the tab, never persisted/undoable.
- ORA: still docs — nested `<stack>` elements are the record (reader builds tree while flattening into the
  flat list, replacing the `_layer_elements` flattener; re-export nests; unmodelled attrs dropped with a
  log). Animated — nested inside each frame-group for the picture; the record is an additive `"groups"`
  key in `animation.json` guarded like `layout` (malformed costs the grouping, never the grid). Ungrouped
  documents byte-identical (negative control).
- UI: `panes/inker_layers.py` header rows + indent + drop-onto-header; timeline v1 = indented labels only.
- Tests `tests/inker/test_groups.py` / `test_groups_ora.py` as designed above.

## Phase P: Pixel-art palette workflow (roadmap item 4)

**Prerequisite fixes (small, land first — both are real latent bugs found in exploration):**
- `Document._map_planes` maps the selection mask through the same `fn` as pixel planes
  (`src/warlock/studio/inker/document.py:669`), so `set_palette`/`recolour_slot` **raise today when a
  selection is active** (`indexed.snap` rejects a 2-D mask). Split into `_map_planes(fn, *, mask_fn=...)`;
  colour maps pass `mask_fn=None`. Test: `set_palette` with a live marquee neither raises nor moves the mask.
- The palette *table* is not in history (`ReplayEdit` restores pixels only). Add a tiny
  `PaletteEdit(before, after)` in `inker/undo.py`; pixel-rewriting palette ops push
  `CompoundEdit([PaletteEdit, ReplayEdit])`. Pin with new undo tests.

**P1. New engine module `studio/inker/dither.py`** (pure numpy/stdlib; joins the import-smoke list in
`tests/inker/test_inker_imports.py`; `OUTWARD_IMPORTS` unchanged):
- `METHODS = ("nearest", "floyd-steinberg", "bayer2", "bayer4", "bayer8")`, `convert(pixels, palette, method)`.
  Nearest delegates to `indexed.snap`. FS: serpentine, 7/16-3/16-5/16-1/16 in float32 straight RGB (Python
  loop is acceptable — preview is memoised like the blur filter). Bayer: two-candidate ordered dithering,
  fully vectorised via the `np.unique` packed-uint32 idiom. Alpha never touched; transparent pixels skipped.
- `build_palette(planes, max_colours)` — count-weighted median cut, exact set when distinct ≤ max,
  fully deterministic (fixed tie-breaks, output sorted by luma/packed).
- Docstring must state why this is deliberately separate from `pipelines/pixel.py::map_palette`
  (Oklab export restyle vs document conversion; engine cannot import pipelines).
- Tests `tests/inker/test_dither.py`: output ⊆ palette, alpha byte-equal, determinism + negative controls
  (bayer2 vs bayer8 differ on a ramp), identity on already-palettised input, median-cut weighting.

**P2. `convert_to_palette(colours, method)`** in `_doc_indexed.py`, shaped like `set_palette`
(`commit_floating` → assign palette → `_replay(_map_planes(dither.convert))`): one undo step across every
unique cel; links survive via `ReplayEdit.grid`. `set_palette` becomes the `method="nearest"` case.
Palette-from-document = `build_palette` over `unique_cel_layers()` then `convert_to_palette`. Conversion is
whole-document (mask ignored — global mode change, matching Aseprite); state this in the docstring.

**P3. Conversion preview session + popup** — generalise the filter-session mechanism
(`begin/preview/commit/cancel_convert` beside `begin_filter` in `_doc_paint.py`; snapshot the current
frame's real layers only, never autovivify; memoise per (palette, method)). UI: `CONVERT_POPUP` in
`panes/inker_bridge.py` cloned from `FILTER_POPUP`; entry buttons in `panes/inker_colors.py`; state
`convert_open/convert_method/convert_max` on `InkerState` (view state, not persisted). Commit restores the
preview then runs the ordinary one-undo `convert_to_palette`. Tests port the filter-session suite's shape.

**P4. Multi-slot selection** — `InkerState.palette_slots: list[int]` (ordered) beside the existing
`palette_slot` anchor; Ctrl+click toggle, Shift+click range; view state, reset by `index_to`.

**P5. Sort + ramp** in `_doc_indexed.py`, following the `move_slot` precedent (table-only, `rev += 1`,
no undo step — the existing rule for order-only palette edits):
- `sort_palette(key, *, indices=None, counts=None, descending=False)`,
  `key ∈ ("hue", "saturation", "luma", "red", "green", "blue", "alpha", "usage")` (parity audit:
  Aseprite's full sort-key set, ascending/descending), stable + deterministic tie-breaks; a
  multi-slot selection sorts in place within the selected positions.
- `insert_ramp(a, b, steps)` — straight-RGB interpolation between two selected slots, dupes skipped.
- UI in `_slots`; `SORT_LABELS` ↔ `SORT_KEYS` two-way table test (`test_ui_tables.py` pattern).

**P6. Shading ink** — new stroke mode `"shade"` in `brush.MODES` + tool row in `inker_state.TOOLS`:
- Ramp = palette-order adjacency of the multi-slot selection (whole palette fallback); no ramp metadata
  persisted. Requires `doc.palette`; tool disabled with a reason string otherwise.
- Per-dab read-modify-write like `_filter` (reads live target, not `before`); exact packed-RGB match against
  the ramp; shift one index toward direction, clamped at ends; non-members and transparent untouched.
- One-step-per-stroke via a `(H,W) bool _shifted` plane on `StrokeState`. Symmetry free (mirroring happens
  in `_dab` above `_stamp`); alpha untouched so alpha-lock trivially holds; indexed snap in `_commit_patch`
  is a no-op on exact members (confirmed). Undo = ordinary `PatchEdit`.
- Direction toggle in the tools pane. Tests `tests/inker/test_shading.py`: one-step, clamping, selection
  gating, symmetry, single undo step, snap no-op.

**P7.** Manual (`docs/manual/07-inker.md` Indexed-colour + Tools sections), CHANGELOG entry.

## Phase Q: Later parity (roadmap item 6) — outlines, ordered d → a → c → b

- **Q-d. Read-only `.aseprite` import** (`studio/inker/asein.py`, mirrors `sheetin.py`): pure-python chunk
  parser (zlib is stdlib, no PIL issue); Aseprite linked cels (type 1) map exactly onto the
  two-slots-one-object link model; tags → `Tag`; indexed mode → `Document.palette`. Refuse-by-name what
  changes pixel meaning, warn on cosmetic chunks, one named test per refusal. Fixtures byte-built in tests
  (or `tests/inker/fixtures/` + FIXTURES.md per the plotter precedent).
- **Q-a. Image brushes + presets**: `Stamp` source concept in `brush.py`; image stamps take the
  `_filter`-style live-RMW path with rotate/flip variants and aligned-pattern mode; presets = named
  `tool_options` bundles via `inker_mode.persist`. Risk: overlap accumulation semantics + undo cost.
- **Q-c. Curve/polygon/contour tools**: extend `PaintOps.shape`/`SHAPES` (PIL ImageDraw); the real work is
  the first multi-click gesture in `inker_canvas` (commit on double-click/Enter, Esc cancels) and its
  interaction with the busy gate.
- **Q-b. Tablet pressure — spike first** (`docs/measurements/` doc): pygame-ce/SDL2 has no pen API on
  Windows; routes are SDL3 pen events or Windows Ink via WndProc subclassing. Engine half is nearly free
  (`StrokeState.to()` optional pressure → existing `_taper`). The spike may legitimately conclude the
  velocity-taper fallback stands.

## Phase C: Aseprite parity sweep (compatibility audit gaps)

Design verified against code 2026-08-14. Three verification wins baked in: new blend modes need
**zero native changes** (`composite._MODE_IDS` absence ⇒ numpy fallback, and `_stack_native` is
all-or-nothing so a new-mode layer can't be silently mis-composited — documented contract at
`composite.py:52-59`); non-uniform scale **already exists in the engine**
(`FloatingBuffer.transform` takes per-axis scale — C5d is UI-only); outline-wrap is plain
`np.roll` (no tiling.py dependency).

**C1. Bug fixes (branch root, before tracks fork):**
- Fill contiguous: `inker_canvas.py:621` never passes `contiguous` — add
  `contiguous=state.wand_contiguous`; ensure the fill tool's options show the checkbox; engine
  regression test (disconnected same-colour region reached only when contiguous=False).
- Manual fix: `07-inker.md` says colour "wheel"; the picker is a hue **bar**
  (`picker_hue_bar`, deliberate) — fix the prose, not the flag.

**C2. Copy-colour ("replace") paint mode** — 5th entry in `brush.MODES`; branch in
`StrokeState._resolve`: `out = before + (colour - before) * alpha` (writes FG RGBA verbatim at
full coverage, incl. painting alpha *down*). Soft coverage lerps (repo doctrine: feathering means
one thing) — divergence from Aseprite's hard edge only visible on soft nibs; docstring states it.
alpha_lock/selection/indexed-snap all free by construction (lock restore runs after the branch).
UI: `paint_ink: "blend"|"replace"` radio in `TOOL_OPTION_DEFAULTS`, brush tool only (no per-tool
ink selector — divergence list). Tests `test_replace_ink.py`: exact bytes, overlap = single dab
(no compounding), alpha-lock, feather lerp, snap-RGB-only, undo, replace stays on the `_resolve`
side of the blur/smudge branch.

**C3. Spray tool** — row `("spray","Spray","A")`, brush mode `paint`, new *emission*:
`StrokeState.scatter/seed` + `self._rng = np.random.default_rng(seed)`; `spray(point, count)`
emits uniform-in-disc dabs (`sqrt` radius) through `_dab` (symmetry + selection free);
`Document.spray_at(point, count)`; `begin_stroke(..., scatter=0.0, seed=0)`. Determinism seam:
engine deterministic given (seed, call sequence) — tests inject seed; UI supplies
`random.getrandbits(32)` at press and `count = spray_rate × delta_time` (fractional accumulator;
`_drag` fires every held frame, so stationary spray keeps emitting). Options `spray_rate` +
reused `brush_size`; force pixel_perfect/stabilise off in the canvas call. One PatchEdit per
press-release. **Jumble: skipped by name** (divergence list; smudge covers the intent).

**C4. Polygonal lasso + shared multi-click gesture** (lands before Q-c; Q-c reuses):
`gesture_pts`/`gesture_combine` on `InkerState` (combine captured at first click); press appends
grid-snapped vertex; overlay polyline via `to_screen` (quarter-turn invariant) + rubber band;
commit on double-click/Enter/click-near-vertex-0 (≥3 vertices) → `commit_floating()` →
PIL polygon → mask → `doc.select(mask, op)`; cancel on Esc/tool switch/`tab.busy`/tab switch.
Tool `("lasso_poly","Poly lasso","D")` — must NOT take the `drag_kind="lasso"` path; extract
lasso's polygon→mask helper for sharing. One `select()` undo step per commit. Tests: extracted
helper + close-detection pure units.

**C5. Selection/transform depth** (Track 5):
- (a) **Reselect** Ctrl+Shift+D: `Document._last_mask` (plain field, not persisted/replayed) set
  on deselect; `reselect()` = ordinary `select()`; refuses a stale-shaped mask after resize
  (shape mismatch → no-op, pinned). `"d"` already in `_MUTATING_CTRL`.
- (b) **Move selection edges only**: `SelectionMask.translated(dx,dy)` (zero-fill, not roll);
  unmodified drag starting *inside* the mask sets `drag_kind="mask-move"`, live offset drawn by
  offsetting the ants points (view-only), ONE `select(translated)` step at release. Shift/Alt
  still start combine drags (modifier check first).
- (c) **Layer from selection** Ctrl+J cut / Ctrl+Shift+J copy: `layer_from_selection(cut)` —
  one CompoundEdit composed from `_masked_alpha` crop + a refactored `paste_as_layer` tail
  (`_add_layer_edit` returns the edit instead of pushing) + a `_cut_selection_patch()`
  extraction for the cut half. New layer's cel on the current frame only. `"j"` joins
  `_MUTATING_CTRL`. Tests `test_layer_from_selection.py`.
- (d) **Non-uniform scale** — UI only: per-axis ratios from handle drags, Shift constrains
  uniform, N/S/E/W edge handles, Scale X/Y sliders + link toggle; ratios computed against the
  press-time ref (anti-compounding rule at `inker_canvas.py:492`).
- (e) **Skew**: `FloatingBuffer.shear (sx,sy degrees)`; `transform.shear()` via PIL AFFINE with
  premultiply round-trip; applied scale → shear → rotate (documented order). v1 = numeric
  drag-floats only, no handles (stated).
- (f) **Numeric X/Y/W/H + angle + shear fields** in `inker_tools._transform_entry` while
  transforming (X/Y → offset; W/H → per-axis scale).
- (g) **RotSprite** (last, droppable): `RESAMPLES` gains `"rotsprite"` — 3 rounds of vectorised
  EPX/Scale2x in uint8 (exact equality on packed RGBA), nearest-rotate, `[4::8, 4::8]`
  downsample; area threshold with toast-fallback to nearest (named constant). Tests: hand-computed
  3×3 EPX fixture; 90° rotsprite == rotate90 exactly; determinism.

**C6. Full blend-mode set (12 → 19)** — Track 6, `composite.py` only:
`exclusion` + `subtract` (max(Cb−Cs,0)) + `divide` (Krita zero-convention, pinned in comment AND
test) + non-separable `hue/saturation/color/luminosity` (W3C Lum/ClipColor/SetLum/Sat/SetSat,
argsort form, on the straight-RGB (H,W,3) `blend()` already receives). Append after `difference`
in `BLEND_MODES`. **Do not extend `_MODE_IDS`** — absence = numpy fallback (add one sentence
naming the seven + the whole-stack-numpy consequence). `ORA_OPS`: `svg:exclusion/hue/saturation/
color/luminosity`; `subtract`/`divide` as `krita:*` — verify spelling against a Krita-authored
ORA at integration (read side is `.get`-safe either way). Check `ora.py` track-props reader's
unknown-blend fallback → soften to normal-with-log if it refuses. Verify the layer-pane blend
combo iterates `BLEND_MODES` (fix if it restates the list). Straight-alpha: nothing to do —
state it so nobody "fixes" it. Tests `test_blend_full_set.py`: per-mode properties (Lum
preservation, hue(C,C)=C, divide zero cases…), opacity-0 sanity, per-mode ORA round-trip,
unknown-op degradation control, mode-less byte-determinism pin.

**C7. Tag repeat count** — Track 1, before A5: `Tag.repeat: int = 0` trailing field (0 = loop
flag decides — today's semantics; N>0 = play span N times then stop; `loop` NOT folded in — no
migration). No fall-through past the span (playback is span-confined by `loop_range`; manual
states the difference from Aseprite). `advance(..., repeat=0, cycles=0)` returns cycles;
`_step`'s three wrap points count; `tab.play_cycles` reset on play start (pingpong: one cycle =
out-and-back, pinned). `animation.json` additive `"repeat"` written only when non-zero (the
`direction` precedent — determinism holds); sheet sidecar tags likewise. GIF:
`write_gif(loop: bool|int)` — True→0, False→omit, r≥2→r−1 (netscape counts additional), r==1→
omit; A4's tag export passes `tag.repeat or tag.loop`; pin the 19-byte netscape block for
r ∈ {True, 1, 3}. UI: `input_int` in `_tag_menu` via the existing `TagsEdit` snapshot (undo
free); Loop checkbox disabled while repeat>0. Q-d maps Aseprite repeat → `repeat=N, loop=True`.

**C8. FX staples** — Track 6, riding `filters.py` FILTERS + the session (defaults = identity
rule honoured):
- **invert**(red/green/blue 0/1 toggles as checkboxes; popup pre-sets all three on open, defaults
  table stays identity; alpha never touched).
- **replace_colour**(from, to, tolerance 0-255; RGB distance ∧ alpha>0; alpha kept; tolerance-0
  ≡ `indexed.remap` — parity test). New `COLOUR_PARAMS` set beside `RANGES`; FILTER_POPUP draws
  `color_edit4` + "use FG" button for members.
- **outline**(colour, size, place in/out, corners 4/8, wrap): dilate-ring; outside-outline ADDS
  alpha — the one stated exception to alpha-untouched; wrap = `np.roll` vs zero-padded shift
  (no tiling.py import; cross-reference comment only); clipped to the session rect (documented).
- **despeckle**(radius, shared 0-32 range accepted): premultiply pair + lazy PIL MedianFilter
  (radius→odd size), identity at 0.
  Tests `test_fx_filters.py` incl. the precise alpha-exception assert.

**C9. Filters over a timeline range** — Track 1, after A1. `filter_range(name, params, t0, t1,
f0, f1)` in `_doc_ranges.py`: **parallel to P3's convert session, not shared** (whole-doc
`_replay` vs bounded mask-weighted per-cel patches — docstring cites the contrast). Dedupe cels
by `id()` first-occurrence (linked cel filtered once); mask-weight via a `_masked_apply` helper
lifted from `preview_filter`; per-cel `PatchEdit`, drop no-changes, ONE CompoundEdit; stamp per
touched layer (not `_stamp_all`); never autovivifies (a filter on an empty cel is a no-op —
documented). UI: "Apply to range" in FILTER_POPUP, enabled when `range_sel` non-empty; cancels
the preview session first (no double-apply). Tests `test_filter_range.py`.

**C10. Move tool without selection** — Track 3. Translate-pixels-with-crop (NOT np.roll — layers
are canvas-sized by invariant; cel-offset model rejected; docstring notes a future tiled-wrap
variant is deliberately unwired). Move session mirroring the filter session:
`begin_layer_move` (commit_floating + `_ensure_active_cel` + snapshot; refuses content-locked —
joins L1's door list) / `preview_layer_move(dx,dy)` (re-render from snapshot, total offset,
anti-compounding) / `commit_layer_move` (patch rect = union of nonzero-alpha bboxes, one
PatchEdit) / `cancel_layer_move` (+`_discard_pending_cel`). Canvas third arm in the move-tool
press branch; arrow-key nudge (±1, Shift ×8) = inline begin+preview+commit, one step per press.
Tests `test_layer_move.py` incl. linked-cel move shows on every linked frame (`is` pin + manual
sentence).

**C11. Gradient dithering** — Track 4, after P1. Inside `gradient.render` (NOT post-remap):
threshold the ramp parameter between adjacent stops against `dither.bayer_matrix(n)` (P1
exposes the matrices publicly — the shared layer; convert's candidates come from palette search,
gradient's from adjacent stops). Kind-agnostic (linear + radial); coverage/weight NOT dithered
(selection softness stays soft); alpha dithers with its stop. `_doc_paint.gradient` +
`grad.render` gain `dither=`; tool option `gradient_dither: "none"`. Tests
`test_gradient_dither.py`: output ⊆ stop set, bayer2≠bayer8 control, dither=None byte-identical
(off-path pin), indexed compose (snap no-op on member stops).

**C12. Palette extras** — Track 4 (a-c), Track 3 (d):
- (a) Sort keys += saturation/red/green/blue/alpha + `reverse` flag (already amended into P5).
- (b) "Palette from image…" — file dialog + PIL decode on the task thread → `dither.build_palette`
  (median-cuts >256 with a toast, never refuses) → ordinary `set_palette` undo.
- (c) `.pal` **JASC only** — `gpl.py` grows `parse_jasc`/`dumps_jasc` (one swatch-text module;
  CRLF tolerated in, written out; pinned bytes); joins the palette dialogs' filters. Tests
  `test_pal_format.py`.
- (d) **Right-click paints BG**: `state.drag_button` set at press; button 1 for
  brush/eraser/fill only (selection tools stay inert, reserved); `_drag`/`_release` test the
  stored button; Alt+right-click picks into `state.bg`. Verified free: canvas has no
  right-click menu; button 2 pans.

**C13. Export options** — Track 1 (a-c), integration lane (d):
- (a) **Scale ×N (nearest)**: `transform.upscale(pixels, n)` (`np.repeat`, exact); applied on
  the task thread in `png_bytes(scale=)`, `run_gif` (upscale before quantise), `run_sheet`
  (upscale before `compose` — the plan builds on scaled size so cells/trims/sidecar stay
  self-consistent; `sheet.py` stays sole writer, zero new code in it). UI `export_scale` combo
  ×1/2/3/4/8, persisted; S4 Packwright-bound sidecars NOT scaled (tooltip says so).
- (b) **PNG sequence**: `_Export.kind = "pngs"`; `run_pngs` writes `{stem}_{i:04d}.png`; the
  pump spread is untouched.
- (c) **Import Sprite Sheet** (modest scope, no marquee preview): refactor `sheetin.py` —
  extract `_document_from_rects`; new `document_from_grid(atlas, cell, offset, padding, count)`
  (row-major, named refusals, layout=None); `SHEET_IMPORT_POPUP` (cell/offset/padding/count +
  computed frame count) → `_adopt` (unsaved-but-clean). Tests: grid arithmetic, refusals, the
  last-column padding off-by-one.
- (d) **Export slices as PNGs** (after S): per slice, resolve `at(current_frame_uid)`, crop the
  flatten, `{sanitised_name}.png` (+`_2` suffix on dupes unless S1 pins uniqueness), scale
  honoured; entry in the slice tool's pane section. Per-frame matrix export deferred by name
  (Packwright's job).

**C14. Text tool** — Track 6, after Track 3 merges. New engine `studio/inker/textstamp.py`
(joins smoke list, PIL lazy): `text_stamp(text, font_path, size_px, colour, antialias=True) ->
RGBA|None` — multiline via `multiline_textbbox` (+1px slack, crop-to-content); aa=False renders
the mode-"1" mask (honest aliasing, not a threshold); returns None on bad font (caller toasts).
Delivery through the floating buffer: new `_doc_selection.float_pixels(pixels, at)` (paste-rule
`forget_redo`; Q-a's image brushes will reuse it) — placement/move/commit/cancel/undo all free;
re-edit = new text (manual sentence, divergence: no text objects). UI: tool `("text","Text","T")`,
click → `TEXT_POPUP` (multiline input, font combo, size, AA, "colour = FG"); OK → float →
`state.tool = "move"` (the Ctrl+V precedent). Fonts: scan `C:\Windows\Fonts` at popup-open
(cached), prepend the vendored `studio/resources/fonts/Inter-Regular.ttf` (verified present via
`fonts.FONT_DIR` — UI may import `fonts` for the constant; engine takes a plain path). Indexed
docs default AA off. Options `text_size/font/aa` in `TOOL_OPTION_DEFAULTS`. No new HELP_TARGETS
key (rides the tools-pane anchor). Tests `test_text_stamp.py` — vendored font is the fixture,
never a system font; aa=False alpha ∈ {0,255}; float-commit-undo round trip.

**C15. Tiled-mode axes** — folded into T-B's build (Track 3): `InkerDoc.tiled: str` with
`TILED_AXES = ("off","x","y","both")` (string enum like `nib`); engine surfaces take
`axes: (bool, bool)` — `spans(lo, hi, size, wrap)`, `pieces(rect, size, axes)` (cartesian
product of per-axis spans), `canonical(point, size, axes)`, `fold_coverage(..., axes)`;
`StrokeState.wrap_axes`; wand seam seeds only on wrapped edges. Canvas: `repeat_x/y` set per
axis every frame on both branches; draw AABB + checkerboard extend only along wrapped axes.
UI: one four-state combo "Tiled: Off / X / Y / X+Y" in `_view_row`. B5 torus tests parameterized
over the three modes (x-only stroke crossing the y seam clamps — per-axis negative control).

## Deliberate divergences from Aseprite (documented, not gaps)

1. **Cel opacity** — opacity is a Track property (§127); per-cel skipped (user decision).
2. **Colour modes** — RGBA-only planes; indexed is a write constraint, no grayscale/index
   storage (§131).
3. **ICC / colour management** — none; sRGB-assumed bytes end to end.
4. **Workspace docking / layouts / split views** — fixed three-width sidebars, one canvas per tab.
5. **Zoom/hand as tools** — wheel-zoom + space/middle pan; toolbox slots not spent on navigation.
6. **Background layer type** — `Document.matte` applied at flatten instead; eraser always cuts
   alpha.
7. **Continuous-layer flag** — links are explicit (+Copy/+Link, two-slots-one-object, §127).
8. **In-editor sheet packing options** — row-wrapped grids only; real packing is Packwright's.
9. **Convolution matrix & colour curves** — FX staples only (user decision).
10. **Per-tool ink selector** — brush modes + C2 replace toggle + layer locks instead.
11. **Pattern fill via bucket** — flat colour only; patterns arrive with Q-a image brushes if
    ever.
12. **Cel z-index** — track order IS stack order (compositor + native kernel contract).
13. **Eraser replace-colour mode** — the eraser cuts alpha, period (§121).
14. **Timeline element colours / user-data** — tags are name+numbers+repeat; Q-d drops these as
    cosmetic chunks with a warning.
15. **Jumble ink** — skipped; smudge covers the intent.
16. **Finite tag repeat does not fall through** past its span (C7; manual states it).
17. **No text objects/layers** — text rasterizes on commit (C14).
18. **Playback speed / play-once are preview-pane options** (A5), not document playback modes.

## Verification (per integration step and at the end)

- `uv run pytest` full suite (never edit `src/` while it runs; if the venv was resynced, it must have been
  `uv sync --extra studio --extra text2image --extra rig`). Targeted lanes first:
  `uv run pytest tests/inker tests/packwright tests/manual`.
- `uv run ruff check .`
- Gates that must stay green by construction: `tests/inker/test_inker_imports.py` (new engine modules
  `_doc_ranges.py`, `tiling.py`, `slices.py`, `_doc_slices.py`, `dither.py`, `groups.py`, `asein.py` join
  the smoke list; `OUTWARD_IMPORTS` unchanged in every phase), `test_inker_ora_determinism.py` (feature-less
  documents stay byte-identical after S2/L1/L3), the square-sidecar byte-equality pin (S4),
  `tests/manual/test_docs.py` + `test_coverage.py` (new `inker-preview` help target; slice UI rides
  existing anchors), `tests/test_changelog.py`.
- In-app smoke after integration (via the `run` skill): open an animated ORA, exercise the new surface (drag a
  range, drag a slice, toggle Tiled, run a conversion preview), save, reopen, undo through the session.
- Round-trip checks with foreign tools where relevant: a Krita-authored grouped ORA (L3) **that also
  carries the seven new blend modes — the fixture that settles the `krita:subtract`/`krita:divide` op
  spelling (C6)**, an Aseprite-authored `.ase` fixture (Q-d), a TexturePacker-consuming engine reading
  the pivot JSON (S5 — at minimum assert against TexturePacker's published schema shape).
- Phase C gates: new engine modules `textstamp.py` (+ the C additions to existing modules) keep the
  import pins green; the netscape-extension byte pin (C7); the dither off-path byte pin (C11); the
  mode-less-document byte-determinism pin (C6); `test_fx_filters.py`'s precise alpha-exception assert
  (C8).

## Critical files

- `src/warlock/studio/inker/document.py`, `_doc_anim.py`, `_doc_paint.py`, `_doc_layers.py`,
  `_doc_indexed.py` — document ops core; new mixins `_doc_ranges.py`, `_doc_slices.py`
- `src/warlock/studio/inker/anim_edits.py`, `undo.py` — new edit types (`FrameOrderEdit`, slice edits,
  `PaletteEdit`); `brush.py` — wrap choke point (`_stamp`) + shading ink + replace ink + spray
- `src/warlock/studio/inker/composite.py` (blend set + ORA op tables), `filters.py` (FX staples),
  `gradient.py` (dither), `selection.py`/`transform.py` (mask translate, shear, RotSprite, upscale),
  new `textstamp.py`; `gpl.py` (JASC .pal)
- `src/warlock/studio/inker/ora.py` — `warlock.json`, content-lock attr, groups; `sheetout.py`/`gifout.py`
  — range export
- `src/warlock/studio/inker_mode.py` (`_Export`, `tick_preview`), `inker_state.py` (range/preview/tiled
  state), `panes/inker_timeline.py`, `panes/inker_canvas.py`, `panes/inker_colors.py`,
  `panes/inker_bridge.py`, new `panes/inker_preview.py`
- `src/warlock/pipelines/sheet.py` (sole sidecar writer), `pipelines/pixelsheet.py`
- `src/warlock/studio/packwright/sources.py`, `layout.py`, `texturepacker.py`, `wpack.py`
- `docs/manual/07-inker.md`, `docs/INVARIANTS.md`, `CHANGELOG.md`

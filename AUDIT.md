# Warlock repository audit — 2026-08-22

Audit of `master` at `05398ac` (v0.0.25). Twelve parallel auditors covered the
tree; every finding below was then **re-verified first-hand against the source**
before being written down, and three findings arrived with a severity that the
verification pass corrected (noted inline).

Nothing here is fixed. This file is the work list.

## Baseline

| Check | Result |
| --- | --- |
| `uv run pytest` | **11,650 passed, 20 skipped**, 61 s, exit 0 |
| `uv run ruff check .` | **All checks passed** |
| Version lockstep | `pyproject.toml` = `__init__.__version__` = `CHANGELOG.md` heading = 0.0.25 |
| Bare `except:` / `except Exception: pass` / mutable default args in `src/` | **none** |
| Banned citations of deleted plan files from `src/` or `scripts/` | **none** |
| TODO/FIXME/HACK/XXX comments in `src/` | **none** |

All 20 skips are environment gates — unbuilt `warlockc.dll`, undownloaded model
weights, or non-Windows. **No test is disabled, vacuous, or asserting nothing**:
an AST sweep of all 415 test files found zero pass-only or empty test bodies, and
every one of the ~103 assert-less candidates resolved to a smoke test, a helper
that asserts internally, or a deliberate "must not raise" call.

## Summary

| Sev | # | Findings |
| --- | --- | --- |
| CRITICAL | 0 | — |
| HIGH | 4 | A1–A4 |
| MEDIUM | 9 | B1–B9 |
| LOW | 9 | C1–C9 |

A3 and A4 compound: A4 can leave a document `busy` indefinitely, which turns
A3's ungated layers panel from a narrow race into an open window.

---

## HIGH

### A1 — Inker's Symmetry feature has no reachable control

`src/warlock/studio/panes/inker_tools.py:977` — `canvas_options()` has **zero
callers** anywhere in `src/`, `tests/` or `scripts/`, and no dynamic/`getattr`
dispatch reaches it.

It is the only writer of `state.symmetry`, `state.radial_count` and
`state.symmetry_axis` (lines 983, 990, and `_symmetry_axis` at 1028, itself
called only from the dead function). The default is `symmetry: str = "none"`
(`inker_state.py:1391`), so **symmetry is permanently off in the shipped app**.

The engine behind it is entirely live and tested: `inker/brush.py:278-300`
(`_mirror`, handling `radial`, `xy` and `diag`) and `brush.py:786`, which the
v0.0.25 stroke-performance work explicitly tuned for the `symmetry=xy` case.

Likely lost in the panel refactor documented at `inker_tools.py:491-499`.

**Scope check — grid, snap and rulers are NOT affected.** They look orphaned by
the same function but each has a reachable toggle op in `inker_ops.py`
(`toggle_grid` :1399, `toggle_grid_snap` :1410, `toggle_rulers` :1420, via
`_toggle`'s `setattr`). The only other casualty is the direct numeric
grid-size entry (`inker_tools.py:1004`); grid size remains settable indirectly
through "Selection as Grid" (`inker_ops.py:1440`).

**Fix:** wire `canvas_options(ctx, state)` back into `draw()`/`_options()`, or —
if Symmetry is deliberately retired — delete the function, the three state
fields, and the `_mirror` path together.

### A2 — Background/reference layer conversions push no undo step, and lose the matte

`src/warlock/studio/inker/_doc_layers.py:690-744`. Three methods mutate
**persisted** document data outside the undo funnel:

- `to_background()` (:690) sets `layer.background = True` and `self.matte = None`
  as plain attribute writes, then calls `self._commit_patch(...)` — which pushes
  a `PatchEdit` covering **pixels only**. It has no way to record the flag or the
  matte.
- `from_background()` (:723) sets `self.stack[0].background = False`, calls
  `invalidate_all()`, bumps `self.rev` — and **never calls `history.push` at all**.
- `set_reference()` (:737) has the identical gap for `self.stack[index].reference`.

Both fields are real persisted document state, not view state: `docs/COMPAT.md`
lines 274-275 record `background` round-tripping via the `.aseprite` layer-chunk
flag `0x08` and `stack.xml`'s `warlock-background`, and `reference` via `0x40` /
`warlock-reference` (`aseout.py:546`, `asein.py:1184`).

**Failure scenario:** a document with a matte set → "Convert to Background"
(matte folds into pixels, `background=True`, `matte=None`, one `PatchEdit`) →
Ctrl+Z. The pixels come back, but `background` is still `True`, so `_shown_pixels`
keeps forcing composited alpha to 255 and the canvas still shows an opaque
background. **The original matte is unrecoverable — no edit ever captured it.**
Separately, "Un-background" then Ctrl+Z silently does nothing while the document
still reports itself dirty.

**No test covers this.** `tests/inker/test_layer_types.py` exercises all three
methods but never calls `undo()`/`redo()` afterwards.

**Fix:** add a `LayerFlagEdit` (or extend the props edit) covering
`background`/`reference`/`matte`, and push it from all three methods — compounded
with the pixel `PatchEdit` in `to_background`'s case.

### A3 — The Inker layers panel is not gated on `busy`, against a named invariant

`docs/INVARIANTS.md` states the rule and enumerates the surfaces it covers:
"**every control that changes the document is gated on `saving`** — the canvas,
the keyboard path (`_MUTATING_CTRL`), **the layers panel**, `inker_bridge`'s
canvas ops and resize popup, and `inker_tools`' selection and transform
sections… That gate is now `InkerDoc.busy`, not `saving`."

The layers panel does not have it. `imgui.begin_disabled(tab.busy)` appears in
`panes/inker_timeline.py` at lines 437 (`_frame_trailing`), 956 (`_frame_menu`),
1326 and 1563 — and **nowhere** in the layer-row range. Ungated:

| Location | Control |
| --- | --- |
| `inker_timeline.py:924` `_toggle_all` | header show/hide-all and lock/unlock-all |
| `inker_timeline.py:1026` `_track_row` | per-row eye and padlock |
| `inker_timeline.py:1124` `_drag_toggle` | drag-across-rows visibility gesture |
| `inker_timeline.py:1143` `_reorder` | drag-and-drop layer reorder |
| `inker_timeline.py:1163` `_row_menu` | Rename / Properties / Move up-down / Group |
| `panes/inker_menu.py:202-282` `header_controls` | Layer ▸ Properties (blend, opacity, alpha-lock, layer-lock, continuous) |
| `inker_ops.py:978` `toggle_reference`, `:991` `solo_layer` | `enabled=has_doc`, not `enabled=ready` |

Every sibling control in the same modules gets this right — `_frame_menu`
(:956), `inker_tools.py:436,572`, `inker_colors.py:121`.

**Failure scenario.** `inker/ora.py:834` `write_ora` reads `doc.stack` twice: once
to build `stack.xml` (order, visibility, opacity, blend) and again, later, to
encode one PNG per layer. Both passes run on a task thread with `tab.saving`
`True` while the frame thread keeps drawing. Nothing stops a reorder, an eye
toggle or an opacity drag landing between the two passes, producing an archive
whose `stack.xml` disagrees with its own PNG members — the "parts disagree"
corruption the module's docstring calls out for rotate and undo.

The same gap applies for the whole of playback (`busy` is `saving or playing`,
`inker_state.py:1186`), where the canvas is showing a cached flatten of another
frame while these controls rewrite the live stack underneath it.

**No test covers the gating**; the one test naming `_toggle_all` checks its
any-hidden-means-show-all logic, not busy behaviour.

**Fix:** wrap the layer-row block and `header_controls` in
`imgui.begin_disabled(tab.busy)` as `_frame_menu` already is, and change
`solo_layer`/`toggle_reference` from `enabled=has_doc` to `enabled=ready`
(which is already `tab is not None and not tab.busy and not state.transforming`).

### A4 — Pressing Tab during playback wedges the document read-only

`panes/inker_timeline.py:163-171` — `draw()` early-returns on
`if not state.timeline_open` **before** it calls `_tick(tab)`, and `_tick` is the
only caller of `tick_playback`, which is the only thing that advances
`tab.play_index` or ends playback naturally.

`toggle()` (:196-205) flips `state.timeline_open` and never calls `stop_play`.
The Tab handler (`inker_mode.py:3265-3272`) calls it unconditionally, with no
check on `tab.playing`.

**Failure scenario.** Start playback, then press Tab to hide the timeline strip —
Aseprite's own binding for exactly that. `tick_playback` stops being called, so
`tab.playing` stays `True` forever and `tab.busy` with it. The canvas then
refuses every paint gesture **silently** — `inker_canvas.py:866-878` does
`state.clear_gesture(); return` with no toast and no tip. The Stop button and the
"Playback is running" explanation are drawn inside the strip the user just hid,
so nothing on screen says why the document stopped accepting edits. Recovery is
Escape or Ctrl+S, neither of which a user mid-playback has a reason to try.

**No test covers the interaction**: `timeline_open` is only ever asserted against
its default and the autoshow toggle, never combined with `playing`.

**Fix:** have `toggle()` call `stop_play(tab)` when hiding the strip while
playing, or move `_tick` above the early return so playback keeps advancing with
the strip collapsed.

---

## MEDIUM

### B1 — `offset()` ignores the origin on infinite maps, and destroys content

`src/warlock/studio/plotter/_map_geometry.py:115-219`.

Every other geometry op slides `origin_x`/`origin_y` by the same delta "so a
cell's *true* coordinate never moves" (`resize`'s own docstring at :40-46), and
rides a `ResizeEdit` carrying `before_origin`/`after_origin`. `offset()` **never
touches the origin**, and its `ResizeEdit` push (:208-217) omits both origin
fields entirely.

Two consequences on an infinite map, whose stored rectangle is a dense window
over the painted extent plus an origin:

- `wrap=True` does `np.roll` over the **window** — whose size and position are an
  artifact of painting history, not a map boundary. Two documents with identical
  true content but different framing offset differently.
- `wrap=False` clears vacated cells to gid 0, **discarding content shifted past
  the window edge — on a map that by definition has no edge.** This is data loss,
  not a clip.

Reachable: `panes/plotter_tools.py:483` calls `_offset_form` inside the
`if tab.doc.infinite:` branch, where it is one of only two operations offered.

**No test combines `infinite=True` with `.offset(...)`.**

**Fix:** for an infinite document, refuse `wrap` (a roll needs a fixed boundary)
and route the non-wrap path through the origin-sliding `resize` machinery so the
shift is a lossless translation; or hide Offset for infinite maps.

### B2 — `plotter_io._write` replaces files one at a time

`src/warlock/studio/plotter_io.py:308-342` stages to a dotfile and `os.replace`s
**inside the same loop iteration**, per file. `packwright_io._write`
(`packwright_io.py:141-176`) does the correct two-pass stage-all-then-replace-all,
and its docstring names this exact bug class as the reason, citing plotter's rule
as the weaker version it improved on. `docs/INVARIANTS.md` calls packwright's
"plotter's rule with the set-wide staging added" — plotter was never brought up
to it.

A Tiled export is `map.tmx` plus a `.tsx` and `.png` per tileset. A failure after
`map.tmx` lands but before a referenced tileset does leaves a map pointing at
stale or missing files. Violates CLAUDE.md's "writes onto served files are staged,
never in place".

**Fix:** mirror `packwright_io._write`'s two-pass shape.

### B3 — Blender retexture leaks every material and image datablock

`src/warlock/pipelines/blender_worker.py:1169-1360`. Verified by grep:
**`bpy.data.images.remove` and `bpy.data.materials.remove` appear nowhere in the
file** (only `bpy.data.meshes.remove`, at :200 and :257).

`_project_material` creates one material (:1169), loads up to three images
(:1180, :1185, :1215) and creates up to three baked targets (:1330) **per view**.
`mesh.data.materials.clear()` (:1318, :1362) only empties material slots and
`tree.nodes.remove` only detaches a node — the datablocks survive for the life of
the subprocess. With `retexture.VIEWS` = 10 and `texture_size` up to 2048 this is
hundreds of MB to multiple GiB of avoidable peak RSS, in the one subprocess whose
host-commit budget the rest of the codebase guards carefully.

**Fix:** `bpy.data.images.remove(...)` once each image is consumed, and
`bpy.data.materials.remove(...)` after the slot clear at :1362.

### B4 — `JobStore.transaction()` holds the sqlite lock across disk writes

`src/warlock/db.py:476-498` + `src/warlock/service/_jobs_create.py:530-568`.

`transaction()` deliberately holds the connection lock for its whole body —
unlike `deferred_commits`, whose docstring explains it does *not*, "because the
caller does file writes between inserts, and holding the store lock across those
would queue the frame thread's reads behind a disk". Its one caller, `create_job`,
does exactly those file writes inside the held lock: up to `MAX_REFERENCE_COUNT`
(8) iterations each writing `input.png` and `ref.png` of up to `MAX_UPLOAD_BYTES`
(20 MB, `validation.py:46`).

The frame thread reaches the same lock synchronously every tick —
`main.py:1775` → `JobsCache.tick()` (`jobs_cache.py:120`) → `list_jobs` →
`JobStore.list()`. So a submission with an uploaded image freezes the window for
the duration of the write. `settings_3d.py:740-743` shows the authors already
avoid precisely this hazard on the upload-read path.

**Fix:** write the candidate files before entering `transaction()` — only the row
inserts need the savepoint's atomicity — and let the existing `except` cleanup
keep removing `made_dirs` on failure.

### B5 — `rig_template` refusals never ring their control

`panes/settings_3d.py:382-384` and `panes/stage_rig.py:86-98` draw the Skeleton
combo with `widgets.labeled_combo(...)` and **no following
`widgets.field_error(ctx.state, "rig_template")`** — verified: zero such call
sites exist in `src/`.

Three doors refuse on exactly that field: `service/validation.py:512`,
`service/rig.py:52`, `service/poses.py:260`. Every sibling field in these same
forms follows the idiom (e.g. `settings_3d.py:281-282` for `profile`).

A generic toast still fires, so the failure is not swallowed — but the one
dropdown at fault is never outlined, unlike a bad `profile` or `base_model`.

**Fix:** add the `field_error` call after both combos.

### B6 — A Clay generator edit pushes two undo steps

`panes/clay_props.py:214-217` calls `doc.set_props(...)` then
`doc.set_mesh(..., keep_generator=True)` as two independent edits. Everywhere
else in Clay that must keep such a pair atomic wraps it in a `CompoundEdit`,
because a lone Ctrl+Z restoring half the pair reproduces "the exact state the
freeze exists to prevent" (`clay/document.py:369-372`).

Drag a cylinder's radius, press Ctrl+Z once: the viewport shows the old mesh
while the properties panel still reads the new radius. Because imgui's
`InputFloat` fires per keystroke, typing a multi-digit number pushes several such
pairs for one felt edit.

`tests/clay/test_document.py:390` walks this exact sequence but only asserts
after **two** `undo()` calls, so the intermediate state is untested.

**Fix:** push both as one `CompoundEdit`.

### B7 — Three `rigging` readers lack the guards their four siblings have

`src/warlock/rigging.py` — `read_sheet` (:1167), `read_sheet_pixel` (:1155) and
`read_sprite_draft` all do:

```python
return json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
```

`read_pose` (:1029-1043), `read_rig`, `read_record` and `read_preview_sidecar` all
(a) `stat()` and refuse over `MAX_RECORD_BYTES` before reading, (b) catch the
wider `(OSError, ValueError)`, and (c) check `isinstance(record, dict)`.

`UnicodeDecodeError` is a `ValueError` but **not** a `JSONDecodeError`, so a sheet
file with invalid UTF-8 raises straight out of `read_sheet`, through
`list_sheets`, into the calling pane — instead of costing one record. A valid-JSON
non-object (a JSON array) passes the `try` and fails later at the first
`record.get(...)`.

`tests/test_rigging.py` has the malformed-file tests for poses and rigs; there are
no analogues for these three.

**Fix:** apply the same three guards.

### B8 — `_derived_palette`'s cap does not honour its own tie-break rule

`src/warlock/studio/inker/aseout.py:1176-1186`. `np.argpartition` only guarantees
the pivot's position, so which colours tied at the cutoff land inside the
`[:MAX_COLOURS + 1]` window is unspecified — a lower-coded colour that should win
by the documented "ties broken by colour value" rule is sometimes dropped.
Measured at ~3.5% of randomized trials with boundary ties.

Impact is confined to the derived palette **chunk** (Aseprite's swatch panel), not
cel pixels, and it does not break byte-determinism (numpy's partition is
deterministic for fixed input).

`test_aseout.py::test_the_derived_palette_is_capped_and_keeps_the_most_used_colours`
gives every colour a distinct count, so it never exercises a multi-way tie.

**Fix:** take a full slice at the boundary count (`seen >= seen[keep].min()`)
before sorting and capping.

### B9 — "Hide every layer" costs one undo step per layer

`panes/inker_timeline.py:924-948`. `_toggle_all` loops
`doc.set_layer_props(index, visible=hidden)` once per layer, and
`Document.set_layer_props` (`inker/_doc_layers.py:344-388`) pushes its own
`LayerPropsEdit`/`TrackPropsEdit` per call that actually changes something.

One click of "hide every layer" on a ten-layer document therefore pushes up to
ten undo steps, and reversing that one gesture takes ten Ctrl+Z — against the
one-gesture-one-step rule the rest of the codebase follows (filters, palette
conversion, `apply_matte`, and the opacity drag's own `was=` pre-image).
`_drag_toggle` (:1124) has the same per-row shape.

**Fix:** batch the loop into a single history entry, reusing the `was=`
pre-image pattern `header_controls` already applies to the opacity drag.

---

## LOW

### C1 — Two refusals carry the wrong `field`

`service/_jobs_create.py:247` and `service/tilesheets.py:252` both raise
`Invalid` with a message entirely about `asset_intent` while passing
`field="asset_type"` — copied from the legitimate `asset_type` check on the line
above each. Reachable only via a direct `create_job`/`create_tile_sheet` call:
`asset_intent` has no independent UI control (it is derived alongside
`asset_type` in `studio/create_assets.py`), so no on-screen control is
mis-highlighted today. Fix for API-contract correctness.

### C2 — `_weld` orphans a mesh datablock when its operator raises

`pipelines/blender_worker.py:179` takes `original = mesh.data.copy()` **before**
the `try`. `_weld`'s own docstring notes `remove_doubles` raising is a real
failure mode; when it does, the exception propagates and `original` is never
freed. `_skin` catches the `RuntimeError` (:238-245) but its own `original` stays
`None`, so nothing reaches `bpy.data.meshes.remove`.

**Fix:** free `original` in an except-and-reraise, or defer the copy into the `try`.

### C3 — Two staged writes never clean up their temp on failure

`pipelines/control.py:108-138` (`write_hint`) and `pipelines/reference.py:418-454`
(`prepare`) stage to a dotfile and `os.replace`, but without the `try/finally`
unlink that `postprocess._staged`, `trellis._atomic_write` and `optimize.run` all
use. A raising `save`/`copyfile` leaves the dotfile behind. The served file is
never partially written, so this is consistency rather than correctness. Both
existing tests assert the staging name is gone on the success path only.

### C4 — `README.md:8` undercounts the base models

Says "Ten base models are registered". `models.py:531-847` registers **eleven**
(`turbo`, `sdxl`, `playground`, `sdxl_cfg`, `sdxl_cfg_pag`, `pixel`, `lightning`,
`juggernaut`, `dreamshaper`, `flux_klein`, `flux_klein_distilled`) — the
`flux_klein` / `flux_klein_distilled` split pushed it past ten.

### C5 — `CLAUDE.md:39` undercounts the unstarted programmes

Says "the two fully-specified unstarted programmes (the installer, and Troupe's
phases 0e/6/7/8)". `TODO.md:13-15` now names **three**, adding the host-commit
defects in its §8.

### C6 — Dead code

All verified to have zero callers, static or dynamic:

| Location | Note |
| --- | --- |
| `panes/settings_2d.py:275` `_output` + `OUTPUTS` | **Superseded, not missing** — `_asset_type` (:93) is the shipped control and `create_assets.sync_legacy_fields` sets `form["output"]` from the chosen spec. *An auditor rated this HIGH on the theory the output selector was unreachable; verification showed the selection works and only this pre-registry control is dead.* Note `tests/test_sheet_form.py:45` still asserts `OUTPUTS`' order — a test guarding a dead control. |
| `panes/inker_tools.py:893` `_transform_entry`, `:906` `_transform_numbers` | Superseded — the comment at :491-499 says the numbers moved to the context bar. `_transform_numbers` is called only from the dead `_transform_entry`. *Also rated HIGH by an auditor; the capability is reachable, so this is dead code only.* |
| `troupe_mode.py:644` `start_from_home` | Duplicate of `panes/landing.py:989` `start_troupe`, which is the one Home actually wires (:1023). |
| `service/troupe.py:299` `get_charsheet` | All three call sites (`packwright_mode.py:190`, `inker_mode.py:764`, `panes/sheet_panel.py:699`) call `svc_sheets.get_sheet` directly, bypassing the function whose docstring exists to spare them that. |
| `inker_state.py:204` `group_label` | Sibling `group_members` (:198) is live. |
| `inker_ops.py:327` `_state_doc` | Siblings `_mode_ctx`/`_doc` are used dozens of times. |
| `panes/app_settings.py:486` `_reset_measure`, `:600` `_reset_sweep` | Documented as test-only reset hooks for module flags `_MEASURED`/`_SWEPT`; no test or conftest calls them — latent pollution risk the authors themselves flagged. |
| `clay_view.py:382`, `viewer_embed.py:396` `thumbnail_png` | Identical one-liners, both dead; sibling `screenshot()` is live. |

### C7 — `vram.estimate_parts`' `retexture` branch is unreachable

`vram.py:241-246` adds `IP_ENCODER_GIB` when `params.get("ip_adapter")` is set,
but `service/_jobs_rework.retexture_job` — the only door that creates a
`retexture` job — never accepts or writes that param. The condition is always
false, so estimates are correct; the branch is a remnant.

### C8 — Stale type annotation

`studio/textures.py:31` annotates `_entries` with a 3-tuple key while `get()`
stores 4-tuple keys including `max_side`. No runtime effect.

### C9 — Repository hygiene

- `master` is **37 commits ahead of `origin/master`** — nothing has been pushed
  since the v0.0.25 work began.
- Branch `plotter-wave-2` last moved 2026-08-14 and `master` has advanced 262
  commits since; it holds 52 unmerged commits. Per the Plotter Wave 2 record it
  is gated on user-authored Tiled fixtures and a final whole-branch review.

---

## Verified clean

These areas were audited and no defect was found. Recorded so a later pass need
not re-derive them.

- **Queue, worker dispatch, VRAM.** Coexist/exclusive handoff (`_needs_handoff`),
  `unload()`-never-`trim()` teardown (the single `trim()` on the job-failure path
  at `queue.py:1360-1363` is deliberate and documented), admission-door vs
  dispatch re-check agreement, the resident-pipe credit, all nine job kinds
  present in every stage-keyed table, `DERIVED_PARAMS` complete against every
  worker param write, `VECTOR_PARAMS` undrifted, no `subprocess` call bypassing
  `winjob`, `ProgressBus` stale-write safety, `ModelLease` correctness.
  (`params["layout"]` at `_q_troupe.py:283` is *not* a `DERIVED_PARAMS` gap —
  `resolve_layout` is an idempotent normalization of a stored input.)
- **Database and persistence.** Every `JobStore` method takes the lock (B4 is
  about hold *duration*, not a missing acquisition); `merge_params` is a correct
  single-hold read-modify-write and every params mutator uses it; migrations are
  append-only, ordered and idempotent; the `~/.warlock` migration's
  copy→verify→delete ordering cannot lose data; every write onto a served or
  derived artifact goes through a unique temp + `os.replace`; `journal.py` writes
  the `.meta.json` sidecar last for all six document kinds, with no age-out.
- **`service/`.** No bare exceptions anywhere; every refusal carries a `field`
  (C1 and B5 aside); `DERIVED_PARAMS`/`CONDITIONING_PARAMS` stripped at every
  resubmit/promote/rework door; `check_vram`/`check_weights` re-applied at every
  re-entry rather than trusted from the source row; cap checks taken under a
  job-wide lock covering both read and write; no business logic found leaking
  into panes.
- **App shell and shared UI.** Every imgui-registered texture has a matching
  forget/release, with `ThumbnailCache` correctly deferring release a frame; no
  blocking I/O on the draw thread; `same_line`/`help_marker` overflow guards
  present at every site checked; one imgui context over one GL context; undo
  revoke matches by identity, never index; `_shortcut`'s per-mode dispatch
  returns unconditionally in all seven work modes, so the 2026-08-20
  Delete-falls-through-to-library-trash class has not regressed.
- **Pipelines.** Offline enforcement (`fetch_worker` is the sole
  `HF_HUB_OFFLINE=0` site, scoped to its own subprocess env); `bpy` confined to
  `blender_worker.py`, machine-enforced; optimize-then-normalize ordering and the
  `T·S·M_root` grounding composition; LoRA `default_weight` correctly read;
  `SEAM_MAX` matches its measurement document.
- **Troupe.** Node-local-vs-delta pose space handled correctly at every site; the
  8192 ceiling enforced twice (`charsheet.plan()` at request time and at render
  time) with per-frame reduction; the T-pose guide rendered by the worker
  (`_q_generate.py:126-136`); the walk-renders-as-run collision genuinely fixed
  via name-based clip ids, not merely claimed; `rig_joints="measured"` with a
  real `jointfit` fallback; clip keyframe counts in `templates/clips/humanoid.json`
  agree with `charsheet.ANIMATIONS` for all five animations.
- **Clay, Poser, viewer.** CSR mesh immutability; uid-addressed undo; drag state
  captured once at press and rebound rather than written through a live `trs()`;
  BVH keyed on mesh identity; every GL owner has a matching release; the
  glTF↔Blender root-delta converters are true inverses applied at exactly one
  boundary each.
- **Inker headless package.** All 19 blend modes verified against the W3C/SVG
  formulas; range ops funnel per cel and read `track.alpha_lock`; link doctrine
  honoured; identity-keyed caches use `is`, never `id()`; the 256-colour cap and
  transparent-index invariants hold; ORA/`.aseprite` round-trip including
  play-once `repeat=1` and tag renumbering.
- **Tests, native kernels, packaging.** No vacuous tests; no escape from the
  `WARLOCK_HOME` pin; `gpu`/`perf` markers registered and excluded, with serial
  enforcement for the gpu lane; all kernels have live numpy fallbacks and
  bit-parity tests; `native.py` `ABI = 9` matches `warlockc.h`; a missing DLL
  degrades to the fallback; extras vs groups correct in `pyproject.toml`; the
  installer's `verify_runtime.py` guards path traversal.
- **Inker UI glue.** The save flow's head/commit ordering and `mark_saved`
  semantics match the invariant exactly (rev captured before submit, `saving`
  cleared on both the success and failure paths, `_settle` commits the floating
  buffer before every read of `history.head`); GL texture lifecycle in
  `inker_textures.py` pairs every cache path with `docmodes.forget_texture` and
  `release_doc`'s prefix sweep covers all six texture families; shortcut dispatch
  returns explicitly on every branch with no fall-through to a destructive
  global; the quarter-turn screen↔image math is exact (orthonormal transpose,
  with the floor-then-compare pattern avoiding the `int(-0.5)` off-by-one);
  `inker_tiles.py`, `inker_tools.py`, `inker_colors.py`, `inker_context.py`,
  `inker_bridge.py` and `inker_preview.py` all gate their mutating controls on
  `tab.busy` correctly — the layers panel (A3) is the exception, not the rule.

## Coverage limits

Audited but not exhaustively, and worth a second pass if a symptom points here:

- `inker_state.py` beyond the `PaintView`/`InkerDoc`/`basis` sections — the
  tool-option and key-context machinery was not read in depth.
- `inker_ops.py`'s params/dialog ops and the undo-history popup, beyond the
  `has_doc`-vs-`ready` sweep that produced A3.
- `inker_mode.py`'s sheet-import, tileset-import and matte/reference flows
  (`_adopt`, `import_sheet`, `_cut_matte`), and the GIF/PNG-sequence export
  runners' own busy handling.
- `inker_canvas.py`'s marquee/ants/ruler/tile-outline drawing beyond the
  transform core and `tile_cell`/`_press`.
- `pipelines/birefnet/modeling.py` is vendored upstream code held to bit-identical
  parity by `tests/test_birefnet_parity.py`, and was deliberately not logic-reviewed.

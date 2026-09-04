# FINDINGS.md — Warlock Studio review, 2026-09-02

A whole-tree review of the Studio at v0.0.30 on `feature/plotter-tiled-clay-ergonomics`
(working tree, including the uncommitted trellis v0.6.0, engine-flag and Sirens
clipboard work). Seven parallel passes, one per subsystem, each verifying claims
against the code; the two Critical crashes and the Plotter export overwrite were
re-verified by hand. `uv run ruff check .` is clean and the targeted test files
touched by in-flight work pass (373 passed, 1 environmental skip).

This is a findings list, not a plan. Per the repository's rule, anything here
that can be built should be built and the entry removed, not ticked. Severity
tags: **Critical** (crash, data loss, or corrupts a document), **High** (wrong
result the user will hit, or a promise in the docs the code breaks), **Medium**,
**Low**. Line numbers are against the working tree at review time.

**Every functional High is closed as of 2026-09-03, and the T7 extractions on
2026-09-04.** What remains unstruck is section 8 (Sirens), which the user set
aside on 2026-09-04; nothing else in this document is open. Two of the entries
closed on 2026-09-03 were closed by *striking*
rather than by building: `review_mode.launch` and `_reload_linked` had both
been fixed as collateral of the T2 sweep and left standing here. An audit that
lists work already done overstates what is open, which is a defect in the
audit -- so a theme being struck is now a prompt to re-read the per-section
entries it touched.

---

## 0. Do these first

All twelve entries that stood here (the two Critical crashes, the Plotter export
overwrite, the shared undo-gesture bug and the eight High data-loss items) were
built on 2026-09-03 and struck; `tests/test_findings_section0.py` is the
regression for each. What remains below is everything else.

---

## 1. Cross-cutting themes

These recur in more than one mode and are cheaper to fix once.

**T1. ~~Slider and drag doors push one undo step per frame.~~** Built 2026-09-03:
`controls.fold_undo` (draw, fold, act) at the eight Sirens and Inker slider doors,
`commit=True` on the two Sirens name fields, the rule in `docs/INVARIANTS.md` and a
test per door in `tests/test_undo_gesture_doors.py`.

**T2. ~~Blocking work on the frame thread.~~** Built 2026-09-03: all nine sites are
task-thread halves with a frame-thread adopt, the rule in `docs/INVARIANTS.md` (the
three-threads paragraph) and a worker-thread guard per door in
`tests/test_frame_thread_doors.py`.

**T3. ~~Task-thread writes into UI state.~~** Built 2026-09-03: the journal write hands
its mark back through `Done.result` (`journal.on_task_done`, routed from `main` where
`drop` also runs, so the two can no longer interleave) and `jobs_cache.measure`/
`measure_one` are readings that `adopt_storage` publishes on the frame thread. Clay
`reserve_uid` was locked the same day. The rule is in `docs/INVARIANTS.md` and the guard
is `tests/test_task_thread_writes.py`, which runs each task half on a real worker thread
and asserts the UI state it must not touch did not move.

**T4. ~~Greyed controls with no or wrong reason.~~** Built 2026-09-03, each site as a
*pure* reason function beside the draw call, because a pane cannot be driven headlessly
and a sentence chosen inline cannot be tested: `overlay.cancel_reason`,
`plotter_menu._layer_reason` (which names the first gate that actually fails, and lets
busy win over the shape gates because busy is the one that passes on its own),
`generation_workspace._why_not_finished`, `inker_mode._no_document_reason`.
`tests/test_findings_themes.py`.

**T5. ~~Documented behaviour the code does not have.~~** Closed 2026-09-03, each the way
it deserved. Sirens' "instrument kind vs channel kind is a refusal": the *sentence* went --
the channel decides the voice and the instrument brings its envelopes, which is the tracker
convention and a working idiom, and the one silent combination (a non-`sample` instrument on
the sample channel) is now a line in the manual's "a typed note is silent" list. Inker's
Repeat Last Export "no dialog": the *code* went -- `run_pngs` honours `export.recorded` like
the two runners beside it, and `export_slices` both records `dest`/`export_kind` and takes
`repeat`, so `REPEATABLE`'s own "slices" row is reachable at last
(`tests/inker/test_export_templates.py`). `_refresh`'s "lazy import": the sentence, again --
the import is eager and would be even without that line, because `journal.snapshot` reaches
`ensure_providers`, which imports all six mode modules to register their kinds.
(The X-ray pick, the stale-buffer `play`, the imgui-nav sentences, `TILED_VERSION` and
"six workspaces" were settled on 2026-09-03 too.)

**T6. ~~Per-frame recomputation nobody looks at.~~** Built 2026-09-03, five sites; the
sixth was already fixed and is worth recording, since an audit that lists work already
done overstates what is open: Troupe's `sendable_meshes` had grown both halves the entry
asks for (a `files_cache` and `cast_and_pending`'s throttle) before the review was
written. The menu bar takes `specs(..., evaluate=False)` for the root names and evaluates
the gates only inside a menu that is open -- **nothing memoised**, so an open menu is
still rebuilt from live state every frame, which is what that docstring promises.
`landing.rows` memoises on the cache generation (`jobs_cache.visible`'s own key, B19) and
refuses to memoise at all for a cache that does not count generations, because such a key
cannot tell two job pages apart. `review_mode.reference_path` remembers a hit outright
and re-asks a miss once a second. `settings_2d.problems_for` is the Reference stage's one
evaluation, keyed on the new `AppState.frame_index` -- the bar and the plan block have to
*agree*, which one call guarantees and two only tend to. `Primitive.box` is the
measurement `Model.bounds`' docstring already claimed it did not repeat: the transform
was eight corners, but the min/max making them was a full pass over 443k vertices, every
frame. `tests/test_findings_themes.py`.

**T7. ~~`main.py` at 6,345 lines and `inker_mode.py` at 4,466.~~** Closed 2026-09-04, and
deliberately last: pure code motion over behaviour the sections below had just pinned, so
a move could not also be a change. `main.py` is 4,880 -- the Review panes, the Clay and
Poser viewports are mixins on `App` (`review_panes.py`, `clay_viewport.py`,
`poser_viewport.py`, the repository's own idiom, as `ClayView` is assembled) and the
shortcut data is `shortcuts.py`. `inker_mode.py` is 2,430, with `inker_export`,
`inker_keys`, `inker_palette_io`, `inker_playback` and `inker_open` beside it, and its
fifteen-arm `on_task_done` chain is now a prefix table (`_TASK_HANDLERS`) -- which is what
found the two palette-export keys that had been *forgotten* from the chain and fell
through reporting neither success nor failure. `inker_canvas.py` is 3,108 with
`inker_drag`, `inker_slices` and `inker_gestures` beside it.

Every moved name is still reachable at its old address, through a PEP 562 `__getattr__`
over a `_MOVED` table rather than an import at the bottom of the parent: each moved module
imports its parent as a *module object* (attributes resolved at call time), so a bottom
`from .inker_export import ...` would fail whenever something imported the pair the other
way round, and resolving on demand has no order at all. `sirens_mode.py` (1,250) is
untouched -- see section 8, which the user set aside.

**T8. ~~Test blind spots.~~** Closed 2026-09-04 in `tests/test_findings_blind_spots.py`,
which names every module on the list and asserts the *decidable* half of each -- a pure
helper, a cache key, a predicate, a piece of arithmetic. None of these panes can be
driven headlessly (this suite has no imgui harness), so a draw function stays covered by
the smoke pass and its **decisions** are covered here; that split is the pattern the rest
of the suite already follows and is why several of this review's fixes were built as pure
reason functions beside their draw calls. `screen_ray`'s orthographic branch, the GPU
cache key's mesh-not-transform rule, the drag cancel's overlay restore and the picked
element's per-mode shape are behaviour tests. `_layouts_popup` is covered by section 2's
guard entry; `_tileset_facts` was widened on 2026-09-03 (see section 7's Tests);
`test_generation_workspace.py`'s string assertions are answered by the behaviour tests in
`tests/test_findings_themes.py`. **The six Sirens panes are deliberately not covered** --
section 8 was left open on 2026-09-04 at the user's direction.

---

## 2. Core shell (`main.py`, layout, dialogs, palette, journal, Home)

### Correctness
- ~~**[Medium]** `_layouts_popup`, `component_gallery.draw()` and `_shortcuts_popup` all
  draw at host scope with no `guard.run`.~~ Built 2026-09-03, with
  `draw_placeholder=False`: a popup that failed has nowhere to put a placeholder except
  the host window beside the rail, where it would sit for the rest of the session.
- ~~**[High] `Layout.save()` can skip persisting rail/sidebar.**~~ Built 2026-09-03:
  the latch is gone rather than made accurate. `save` is the only writer of `rail` and
  `sidebar` -- which the library never persists -- so *no* early return there can be
  right, and its own comment already said so ("a key this method forgets is a preference
  that silently resets"). Rewriting the legacy share blob unchanged costs nothing: it is
  a migration seed, not a source of truth. `tests/test_layouts.py`.
- ~~**[Medium] Ctrl chords leak through the command palette.**~~ Built 2026-09-03 -- but
  **not** by adding `palette_open` to `modal_open`, which would have blocked Ctrl+K too
  and left no way out. The gate is in `_shortcut`, immediately below the Ctrl+K branch,
  so exactly one binding survives; Esc is swallowed with the rest because the palette
  reads its own Escape through imgui, where its query box holds the focus.
- ~~**[Medium] The Manual overlay does not own the keyboard.**~~ Built 2026-09-03, the
  palette gate's shape one branch lower: Esc passes because the branch immediately below
  is what answers it, and Ctrl+K/Ctrl+//F1 stay above because they are the three bindings
  that work everywhere.
- ~~**[Medium] Prompt dialog cannot be tabbed.**~~ Built 2026-09-03: `Prompt._focused`,
  `Confirm`'s one-shot. `is_any_item_active` is true of the *field*, so the old spelling
  re-grabbed on every frame after a Tab and the buttons were unreachable.
- ~~**[Medium]** `journal.write` sets three tab attributes from the task thread with no
  lock on the read side (`journal.py:427-446`). See T3.~~ Built 2026-09-03.
- ~~**[Low]** `_MODIFIER_MAP`/`_KEY_MAP` lack GUI/super and `K_APPLICATION`.~~ Built
  2026-09-03: left/right super in both tables, plus the menu key -- which is `K_MENU`,
  not `K_APPLICATION`; pygame-ce has no such name, which the entry assumed it did.

### UX
- ~~**[High] Settings → Advanced "Sidebar width" is dead.**~~ Built 2026-09-03: made
  live rather than removed, and write-through rather than "next launch". `Layout.
  set_sidebar_width` now calls `layouts.Library.set_width_seed`, which updates the seed
  *and* drops the per-workspace overrides -- a named width is a global preference and a
  splitter drag is a local override, so choosing one replaces them, and the drag is
  there again the moment one workspace wants a different answer.
- ~~**[High] "Reset pane sizes" does nothing for a split ever dragged.**~~ Built
  2026-09-03: `Layout.reset_sizes` -> `layouts.Library.reset_sizes`, which clears every
  arrangement's `widths` and `shares` and deliberately *not* `columns`/`hidden` --
  those are which panes are where, which is a different button's promise.
- ~~**[Medium]** Delete key is bound in the Create sidebar library but not in Library
  mode.~~ Built 2026-09-03, on the sidebar binding's own reasoning: the trash *is* the
  confirmation.
- ~~**[Medium]** The window title reflects only unsaved *pose* edits.~~ Built 2026-09-03
  exactly as prescribed: `docmodes.any_unsaved` is the quit chain's own predicate, so the
  caption and the question asked on the way out cannot disagree. Sampled per frame rather
  than pushed from the one callback that used to own it, and only a *change* is sent to
  the window manager.
- ~~**[Low]** Progress-card Cancel greys with no reason (`overlay.py:367`).~~ Built
  2026-09-03 (T4).

### Structure
- ~~**[High]** Extract `panes/review_*.py`, `panes/clay_viewport.py`,
  `panes/poser_viewport.py` and `shortcuts.py` from `main.py:259-554, 4400-5610`.~~ Built
  2026-09-04 (T7). The three pane groups are *mixins* rather than modules of free
  functions -- `class App(ClayViewport, PoserViewport, ReviewPanes)` -- because every one
  of them reads a dozen attributes off `self`, and threading those through a parameter
  list would have made the move a rewrite. The shell names they call are imported inside
  the methods, which is what keeps the import one-way.
- ~~**[Medium]** Panes import `main` for `modal_open` and `_version`.~~ Built 2026-09-03
  to those two destinations. `main.modal_open` and `main._version` stay as re-exports,
  where every existing caller names them.
- ~~**[Medium]** `layouts.py` re-spells `layout.SIDEBAR_WIDTHS`, `PANEL_MIN/MAX`.~~
  Built 2026-09-03: both live in `tokens` with a `clamp_panel` helper, and `layout` names
  them where its eight readers already look. `layouts` cannot import `layout` -- that is
  the direction the dependency runs -- which is why it had literals in the first place.
- ~~**[Medium]** The ~150 function-local `from . import` statements in `main.py`.~~
  Settled 2026-09-03 by saying so, which is the first of the two options offered: they
  are about *import order*, not laziness -- this module is imported before pygame has a
  display, and a mode module at its top would drag imgui and moderngl into that moment.
  The module docstring records it. Gating on `state.inker is not None` was rejected: that
  is state about a document, not about whether a module is loaded.
- ~~**[Low]** `App` reaches `viewer._grab`/`clay_view._grab`; three copies of the
  hover/grab routing rule.~~ Built 2026-09-03: `Viewer.dragging`/`ClayView.dragging` and
  one `main._takes_pointer`. The `tab.saving` press gate stays where it is -- it is a
  different rule, about a document rather than a pointer.
- ~~**[Low]** Surface sizes off the token scale.~~ Built 2026-09-03: four named
  floating-surface widths in `tokens` (tip / popover / card / sheet, plus the one height a
  surface is *given* rather than fitted to) and one `THUMB_CELL` for the picture a
  judgement is made about -- Review's cell and Home's Resume thumbnail were 132 and 136,
  two answers to one question and a difference nobody could see.

### Performance
- ~~**[Medium]** Menu bar spec rebuild per frame (T6).~~ ~~**[Medium]** Settings flush on
  the frame thread during drags (T2).~~ ~~**[Low]** `ctypes.string_at` copies vertex/index
  buffers per command list per frame.~~ Built 2026-09-03: `from_address`, so the upload
  reads imgui's own memory. ~~**[Low]** `guard.enter` allocates an `ErrorRecoveryState`
  per pane per frame.~~ Measured and kept 2026-09-03: 0.077 us to allocate, so ~25 guarded
  surfaces cost about 2 us a frame. Marks nest and each holds its own until it is landed,
  so pooling would mean threading the mark back through every caller's success path to buy
  back a thousandth of a frame.

---

## 3. Create, Review, Library, Home

### Correctness
- ~~**[High]** `review_mode.launch` runs `create_sweep` inline on the frame thread.~~
  Built 2026-09-03 exactly as prescribed, as part of T2 rather than as this entry:
  `LAUNCH_KEY = "review-launch"` (`review_mode.py:108`), `form.submitting` set before
  `ctx.submit` (`:1500-1503`), landing and failure at `:703`/`:781`. Struck late --
  it had been fixed and left standing here, which is how an audit comes to overstate
  what is open.
- ~~**[Medium]** the reference PNG is decoded on the frame thread.~~ Built 2026-09-03 --
  and half of the entry was already fixed, which is worth recording: `_sync_viewer` had
  the `parse`/`adopt` split when the review was written. What was real is
  `inker_mode._nudge_viewer`, which decoded inline on the frame a revert lands.
  `viewer_embed.request_reference` is that split as a function a mode can call, and
  `LOAD_KEY` moved there with it so no mode has to import the shell to show a picture.
- ~~**[Medium]** Results-tray "Open" bypasses `asset_open.open_asset`
  (`generation_workspace.py:200`).~~ Built 2026-09-03: the card calls the one router.
- ~~**[Medium]** `jobs_cache.measure_one` mutates `_dir_sizes` from a task (T3).~~ Built
  2026-09-03.
- ~~**[Low]** `create_stages._reached_export` imports imgui-bearing `widgets` every
  frame.~~ Built 2026-09-03 exactly as prescribed: `studio/artifacts.py`, four tables and
  one lookup, re-exported from `widgets` where every caller already names them.

### UX
- ~~**[Medium]** Error/cancelled rows in the tray say "not ready yet" and disable
  Rerun.~~ Built 2026-09-03 (T4): Rerun is live on a stopped row and the caption is the
  failure, which is what the library card has always said on the same rows.
- ~~**[Medium]** A VRAM door refusal is a fading toast while the plan block keeps saying
  "Ready to generate".~~ Built 2026-09-03 exactly as prescribed: `AppState.submit_refusal`,
  set where every task failure already passes through and only for a refusal that names no
  control, drawn above the form problems and cleared by the next accepted press.
- ~~**[Medium]** `hole_worst` is presented as a ranking in the inspector's remesh line.~~
  Built 2026-09-03: `studio/quality.py` owns the threshold, the caveat and
  `remesh_line`, which says the *measurement* ("silhouette openness measured 12.3%,
  4.5% -- kept the lowest") and the rule that chose, and never a ranking word. The
  "three wordings" half was stale: two of the three were already the same sentence and
  the badge carries none -- it mutes a colour. What was real is the per-frame
  `from .widgets import` inside `review_mode`, which is gone with the threshold.
- ~~**[Medium]** "Working now" in the tray duplicates the global progress card without
  its Cancel.~~ Built 2026-09-03 by giving it the Cancel rather than by deleting the
  block: the tray is where a Create user is looking, and a duplicate that cannot act was
  strictly worse than the thing it duplicated.
- ~~**[Medium]** Review's sweep-axis help covers 3 of 14 axes.~~ Built 2026-09-03: all
  eighteen, a parity test against `sweeps.KWARG_AXES` in both directions, and a "What you
  can vary" section in `36-review.md` grouping them by what they act on.
- ~~**[Low]** "Keep → Delete the others" raises one undo toast per loser.~~ Built
  2026-09-03: `library.delete_assets` is one toast with one Undo carrying the batch, and
  `restore_asset` takes it back. ~~**[Low]** `rank.score` shown as an unqualified
  "score N%".~~ Built 2026-09-03: it is the probe's probability that you would keep this
  one, so it says so -- "judge: 72% likely a keeper". ~~**[Low]** Toast "Show" from
  Sirens/Settings only selects, no mode change.~~ Built 2026-09-03: the fallback goes to
  the library, which is where a bare selection is visible and the one destination that is
  right for a row whose stage cannot be read.

### Structure / performance
- ~~**[Low]** Private reach-across.~~ Built 2026-09-03: `library.copy_settings` and
  `generation_workspace.queue_position` are public (both are questions two surfaces
  genuinely share), and `plan_for` imports `service.sprites` itself. ~~**[Low]**
  `generation_workspace.should_draw` is permanently true after the first done job.~~
  Built 2026-09-03: it asks the same three questions `draw` answers, so the strip is
  reserved exactly when there is something to put in it.

### Tests
`test_generation_workspace.py` is five source-string assertions; nothing drives
`_result_card`'s enable table, error-row inclusion, or Open routing. No test asserts a
refused `"submit"` is reported. Nothing pins where the reference PNG decode happens.

---

## 4. First run, doctor, docs, tour, accessibility

- ~~**[Low]** The tour welcome step omits music; the "click the rail" step has no
  keyboard path.~~ Built 2026-09-03: the welcome step names music, and Ctrl+K is in the
  rail step and in the one that waits for a click.
- ~~**[Medium] Clay HUD colours are untokenised**~~ and ~~**[Low]** the toggle knob is
  hard-coded white~~: built 2026-09-03 as two palette entries in all three themes --
  `WASH` (what a hover or press paints over a 3D viewport) and `KNOB` -- so
  `test_accessibility` can see them at all, which was the deeper half of the entry.
  ~~**[Low]** Raw px sizes at `candidates_panel.py` and `inspector.py`.~~ Built
  2026-09-03: both named and through `sp`. ~~**[Low]** Tour card has no
  Enter/Right/Left binding.~~ Built 2026-09-03, read through imgui like the palette's
  own keys, because the card is a floating surface and the keys belong to it.
- ~~**[Low]** No i18n layer exists; record the decision.~~ Done 2026-09-03, in
  `docs/INVARIANTS.md`, naming the two prose files (the manual and the tour script) that
  would have to move first so the cost is not rediscovered.
- ~~**[Low]** Stale `src/warlock/{sirens,plotter,packwright}` directories.~~ Deleted
  2026-09-03, and CLAUDE.md now says the shorthand drops a `studio/` prefix rather than
  spelling six paths that read as top-level packages.
- ~~**[Low]** Seven files carry `LF will be replaced by CRLF` warnings.~~ Re-checked
  2026-09-03 and gone: all six that still exist are uniformly CRLF in the worktree and LF
  in the index, which is exactly what `* text=auto` promises, `git add -n` over the whole
  tree emits no such warning, and `service/generation.py` no longer exists at all. Nothing
  to fix, and a whole-tree `eol=lf` flip would have been a large churn for a warning that
  is not being printed.

---

## 5. Inker

### Engine correctness (`studio/inker/`)
- *(Withdrawn 2026-09-03.)* "Mirror symmetry is one pixel off": measured through
  `StrokeState` on a 32-wide canvas, a diameter-4 pixel dab at x=6.0 covers columns
  5–8 and its mirror covers 23–26, which is exact about `default_axis` (column c ↔
  column 31−c). The review's arithmetic assumed a different axis.
- ~~**[Medium]** `animation.layers_for` mutates the shared `Layer.opacity` of a linked
  cel.~~ Built 2026-09-03: `layers_for(..., detach=True)` hands shallow copies (the pixel
  arrays are shared) to every read-only consumer, and `frame_stack` -- the onion skin and
  the playback cache -- is the one that needed it. The editable stack still gets the cel
  objects themselves, because a stroke has to land on them.
- ~~**[Medium]** `merge_down` drops per-cel opacity and z.~~ Built 2026-09-03, and the
  two halves get different answers because they are different problems. Opacity is folded
  in (it joins the memo key -- two frames sharing both cel objects but not both alphas do
  not merge to the same picture) and cleared on the lower slots afterwards, for the reason
  the track's opacity is. `cel_z` is a **refusal**: it reorders rows within one frame, so
  on a frame where a lift puts a third row between the pair, the two layers this op merges
  are not the two that frame composites -- there is no per-frame answer to give, so the
  frame is named and nothing happens. The "bakes then resets" half was already correct and
  is what the fix is modelled on.
- ~~**[Medium]** Interactive Move re-resolves palette indices by nearest colour.~~ Built
  2026-09-03: the session snapshots the index plane beside the pixels, translates both,
  and commits through `_commit_permuted_indices` -- `offset_layer`'s door, which exists
  for exactly this and had one caller.
- ~~**[Medium]** Bucket fill `refer="layer"` stops at erased pixels.~~ Built 2026-09-03
  in `colour_distance`, so the fill, the wand and Select Colour Range stay one predicate:
  two fully transparent pixels are the same pixel. Only when the *reference* is
  transparent -- an opaque seed still differs from an erased pixel by its alpha, which was
  already right.
- ~~**[Low]** `FloatingBuffer.flip` seeds `source` but not `source_offset`;
  `duplicate_layer` drops `cel_opacity/cel_z/cel_notes/note/continuous`; `paste()` with
  no selection lands at (0,0).~~ All three built 2026-09-03. The clipboard records an
  `origin` now, so a paste with no marquee lands where the copy was taken from
  (Aseprite's answer, and what makes copying a sprite between frames two keys); (0, 0)
  survives as the floor for content from outside the document.

### App-layer correctness
- ~~**[High]** The context-bar combine mode is overwritten by every press.~~ Built
  2026-09-03: `sticky_combine` is what the bar shows and what an unmodified drag does;
  `combine` is what *this* gesture does, which a held modifier still wins for.
- ~~**[High]** A click with a select tool leaves a 1-pixel selection instead of
  deselecting.~~ Built 2026-09-03: `is_click` asks the *pixels* whether the pointer ever
  left the one it pressed in, which is the question `marquee_rect`'s floor/ceil corners
  cannot answer.
- ~~**[High]** Regenerate selection on an empty animated cel captures the placeholder
  uid and is refused on landing.~~ Built 2026-09-03: `apply_pixels` was the one
  pixel-writing path that refused a placeholder instead of autovivifying it, and
  `_ensure_cel_for` keeps the slot's uid, so the uid captured at submit still names the
  cel it makes. The refusal survives for the slot that genuinely cannot be materialised
  (a tilemap track bound to no tileset), and a job landing on a frame the user has since
  left now says that rather than "the layer is gone".
- ~~**[High]** `fg_slot` goes stale after palette move/sort/remove/ramp insert.~~ Built
  2026-09-03: `InkerState.palette_moved()` at the six doors that reorder the table drops
  the brush's slot claim (and the usage counts keyed on it), so the next stroke paints
  the nearest slot exactly as a colour from the wheel does.
- ~~**[High]** Repeat Last Export re-opens the dialog for PNG sequences and can never
  repeat slices (`inker_mode.py:1666-1672, 2284, 1372`).~~ Built 2026-09-03 (T5).
- ~~**[High]** `_reload_linked` decodes on the frame thread.~~ Built 2026-09-03,
  also T2 collateral: `_reload_linked` takes an already-decoded document and owns only
  the textures and the tab; the decode moved to the revert task (`inker_mode.py:2626`).
- ~~**[Medium]** Space mid-stroke kills its own pan.~~ Built 2026-09-03 exactly as the
  entry's own sentence prescribes: `forget_held_keys` is called from the three tab doors,
  where the original reasoning (a release this pane will never see) actually applies, and
  `clear_drag` no longer touches it.
- ~~**[Medium]** `fill_selection`/`stroke_selection`/`shift_selected` skip the busy
  gate.~~ Built 2026-09-03 through `when_ready`, which exists for this and which they were
  the three writing ops not to use.
- ~~**[Medium]** Save As `.aseprite` marks clean and drops the journal though the write
  is lossy.~~ Built 2026-09-03, asked of *this* document rather than of the format:
  `aseout.dropped_by_aseprite` walks `docs/COMPAT.md`'s ORA -> aseprite table against what
  the document actually holds, so a plain drawing written here is still a real save and
  one carrying a Flourish recipe, a matte, an alpha lock or an empty group stays dirty,
  keeps its crash copy, and is told what has no place in the file.
- ~~**[Medium]** Four sliders push a step per frame (T1).~~ ~~**[Medium]** Cursor readout
  and shape commit truncate where the press floors.~~ Built 2026-09-03: `math.floor`, so
  the readout names the pixel a click would hit (`int` truncates towards zero, so a cursor
  just off the left edge read as pixel 0). ~~**[Medium]** `toggle_play` lacks `step_frame`'s
  open-gesture guard.~~ Built 2026-09-03, with *stopping* deliberately above the guard: a
  way out must always be available. ~~**[Medium]** `_settle` mutates before a save that may
  be refused.~~ Built 2026-09-03 at the reachable door -- "send to 3D" on a dirty linked
  tab now refuses *before* the float is committed and the preview thrown away, with the
  pending float folded into the dirty test rather than settled first, because a pending
  paste is exactly "what you see". ~~**[Medium]** Sixteen `ACTION_MODIFIERS` are remappable
  and never read.~~ Built 2026-09-03 in both directions: the nine with no behaviour at all
  were **deleted** rather than implemented (a remappable key that does nothing is worse
  than an absent one), the four that had hard-coded behaviour were wired to the registry,
  and the four spellings of "which modifiers are held" became one `held_chord()`. A test
  pairs the tuple and its readers in both directions.
- ~~**[Medium]** Frame-spread exports refuse silently.~~ Built 2026-09-03: all four
  refusals at that door say why -- no timeline, playback running, the tab being written,
  and an export already being set up.
- ~~**[Low]** Timeline prompt callbacks and drag-reorder carry stack indices into
  deferred paths.~~ Built 2026-09-03: both carry the layer's **uid** and resolve it when
  the answer or the drop arrives -- the package's own "undo is addressed by uid, never
  index" rule, one level up. ~~Fractional wheel notches leave the 5 % zoom lattice.~~
  Built 2026-09-03: `PaintView.zoom_carry` keeps the remainder, so a trackpad still zooms
  and 101.5% is unreachable. ~~"Queued a mesh…" toasted before the job exists.~~ Built
  2026-09-03: the toast is on the landing, where the job id is.

### File IO
- ~~**[High]** GIF export ignores per-frame palette overrides.~~ Built 2026-09-03.
  (The mechanism had drifted by the time it was fixed: the export passes exact per-frame
  index planes, so it was index-mapped, not nearest-matched, through the one document
  table.) The table is now read per frame beside the index plane it resolves --
  `_Leg.palettes` parallel to `_Leg.planes`, for that field's own reason -- and
  `gifout.write_gif` writes a *local* colour table for any frame that overrides.
  Asserted by decoding the file, which is the only way to tell a local table from a
  global one.
- ~~**[High]** `ora._read_tiles` never validates the refs grid against the canvas, and
  the "atomic apply" swaps cels before `materialize`.~~ Built 2026-09-03. Half of this
  entry was wrong and is worth recording: the apply *was* atomic with respect to
  validation -- everything that can raise already ran inside the `try`. What was real is
  both halves of the rest. The grid is now checked against `tiles.grid_shape(doc.size,
  ...)` and not merely against the blob's byte length, because `materialize` is tolerant
  by design and would have blanked whatever an undersized grid did not cover; and the
  rebuild moved *inside* the guard, before the swap, with `MemoryError` added to the
  caught set since that loop is the member's one large allocation.
- ~~**[Medium]** Hand-listed `Layer` fields drop `background`/`reference` on open.~~
  Built 2026-09-03 through `Track.props()`, which is now `CEL_PROPS` rather than a dict
  literal of its own -- so the copied-down set has one definition and the ORA reader takes
  it whole. (The `ora` list had caught up with the set by review time; what was real is
  that it was a *list*, and the fifth hand copy was `props` itself.)
- ~~**[Medium]** `gifin`/`sheetin` set `duration_ms` past the clamp.~~ Built 2026-09-03
  on `Frame.__setattr__` rather than at the two call sites: the clamp is a property of the
  field, so a third importer inherits it instead of being the next bug.
- ~~**[Medium]** `asein` decodes tilesets with the raw transparent index and cels with
  the clamped one.~~ Built 2026-09-03: `_transparent_slot` is the one answer, as its own
  docstring already argued for the other two readers. ~~**[Medium]**
  `write_ora`/`write_aseprite` hand-roll a `.tmp`.~~ Built 2026-09-03 through
  `studio/atomic.py`, which joins the engine's outward-import allowlist as a shared leaf
  for `zipguard`'s reason -- a third private copy of a staging rule is a copy that stops
  agreeing.
- ~~**[Low]** `asein` warns the reference layer "opens hidden" while reading VISIBLE
  verbatim.~~ Built 2026-09-03: the sentence went, not the behaviour -- it described an
  override the reader had already stopped doing. ~~Skipped header fields have no COMPAT
  row.~~ Two added 2026-09-03: the pixel aspect ratio and the grid rectangle, each with
  why it is dropped rather than read. ~~`textstamp` allocates unbounded.~~ Built
  2026-09-03: `pixelguard.check`, refused by name because this one is reachable by typing
  rather than by opening a hostile file.

### UX parity
- ~~**[High]** Max zoom is 1000 %.~~ Built 2026-09-03: 64× (Aseprite's 6400%), with the
  ladder extended and the wheel switching from 5% notches to one rung per notch above 8×
  — 1,260 notches to the ceiling is not a control.
- ~~**[Medium]** Brush cursor is a floating ring at raw mouse coordinates, not the
  footprint.~~ Both halves built now; the footprint on 2026-09-03. `brush.footprint` is
  `StrokeState._stamp`'s own anchoring arithmetic lifted out, so the box drawn is the box
  the dab will cover -- the two nib families anchor differently and one rule for both
  would be half a pixel out exactly where it matters. Above `PIXEL_CELL_MIN_ZOOM` the ring
  is **not** drawn at all: two cursors saying different things about one brush is worse
  than either.
- ~~**[Medium]** Grid, symmetry, layer-edge and pixel-cell lines at fractional screen
  coordinates antialias across two pixels.~~ Built 2026-09-03: `crisp` snaps an
  axis-aligned line into one device pixel. **The symmetry guide is deliberately not
  snapped** — a mirror axis genuinely lies on the boundary between two columns, and
  `test_the_symmetry_guide_is_drawn_where_the_engine_reflects` pins it there after a bug
  that had the guide and the reflection half a pixel apart.
- ~~**[Medium]** No chord for flip H/V, merge down, duplicate layer, or Ctrl+Shift+Z
  redo; `SHIFT_TOOL_KEYS`/`ALT_TOOL_CHORDS` a hand copy.~~ Built 2026-09-03: Shift+H /
  Shift+V (Aseprite's own), Ctrl+Shift+M for merge down (**not** Aseprite's Ctrl+E, which
  is Save as reference here and has no Aseprite counterpart to defer to), Ctrl+Shift+L for
  duplicate layer, Ctrl+Shift+Z as a second binding for redo, and the chord tables are
  read out of `inker_ops.BINDINGS`. **Delete layer and delete frame keep no chord**, which
  is `delete_frame`'s own recorded argument: every other verb there is undone by pressing
  it again, and a one-key drop of the thing under the cursor is worth a menu.
- ~~**[Medium]** Picker Hue/Saturation sliders are dead on greys and drift on dark
  colours.~~ Built 2026-09-03: `InkerState.picker_space` holds the triple, keyed on the
  bytes it wrote -- so it is a cache of *this gesture* and anything else changing the
  colour replaces it.
- ~~**[Low]** Onion skin on a two-frame clip draws the other frame twice~~ -- the span
  wraps, so -1 and +1 are one frame; a drawn-set fixes it and nearest-first ordering means
  the ghost kept is the closest. ~~`,`/`.`/Enter bound twice~~ -- the raw arms went; they
  were a workaround for a context bug in the `play` op that is fixed, and only the
  registry's binding is remappable or carries a refusal. ~~`_MUTATING_CTRL` duplicates
  `inker_ops.ready`'s gate~~ -- the duplicate check at the registry dispatch went, since
  `inker_ops.run` enforces the gate *and says why*: the second check refused the same
  presses silently. Every op bound to one of those chords is `when_ready`-gated, which is
  what makes the removal safe and is asserted.

### Structure
- ~~**[High]** Extract from `inker_mode.py` in payoff order: `inker_export.py`
  (`:1243-2329`), `inker_keys.py` (`:3206-3830`), `inker_palette_io.py`,
  `inker_playback.py`, `inker_open.py`; replace the fifteen-arm `on_task_done` chain
  with a prefix table.~~ Built 2026-09-04, all five and the table -- see T7. The table is
  built once behind `functools.cache` rather than written as a literal, because two of its
  keys are constants on `inker_flourish`, which imports this module back.
  `tests/test_ux_silent_refusals.py` and friends now name the module a function actually
  lives in; `test_every_task_key_inker_submits_is_answered` reads `_TASK_HANDLERS()`
  instead of parsing the chain's source.
- ~~**[Medium]** Split `inker_canvas.py` into input / transform / slices / overlays.~~
  Built 2026-09-04, on the seams the code actually had rather than the four this line
  guessed: `inker_drag` (the held pointer and its release -- the pane's one hard rule,
  that a drag never commits and the release is the only writer, is now a file), 
  `inker_slices` (a named rectangle on the canvas, so its own hit test, handles and drag)
  and `inker_gestures` (the multi-click gestures -- lasso, curve, text -- which are not
  drags at all, because nothing is held down to end them). Transform and the overlays stay
  in the canvas: they are what the draw order *is*, and splitting them would have put half
  a sequence in another file.
  ~~Move pure helpers (`marquee_rect`, `closes_gesture`, `_onion_index`) into
  `inker_state`~~ -- done 2026-09-03, with `is_click` beside them, and the canvas
  re-exports all four. ~~`0x1FFFFFFF` → `gid.GID_MASK`~~ and ~~one `held_chord()`~~ done
  the same day. The **split itself is the T7 work** and is deliberately last, so the
  moves are pure code motion over behaviour these tests now pin.
- ~~**[Medium]** Three `over`-style commits and three cut-arithmetic copies in
  `_doc_selection.py`.~~ Closed 2026-09-03. The `over` half was already done --
  `_composite_onto` is the one rule and its docstring says so -- and the cut half was two
  copies rather than three; `_cut_alpha` is now the one place the "it is a *subtraction*"
  argument lives.

### Performance
- ~~**[Medium]** `preview_filter` re-blends, invalidates and re-uploads the box every
  frame the popup is open.~~ Built 2026-09-03: `_filter_written` is the signature of the
  blend last written, so an idle popup costs a tuple compare. The mask is the one input it
  cannot describe cheaply, so it is dropped at `select`, the one writer.
- ~~**[Medium]** Every brush-down copies the whole layer twice.~~ Measured 2026-09-03
  and it is one copy, not two: `StrokeState.coverage` is `np.zeros`, 0.012 ms at 2048
  square because the pages are mapped lazily and a stroke touches a few of them. The real
  one is the `before` snapshot at 6.8 ms -- and 0.1 ms at 256 square, which is the size
  documents are drawn at; 2048 is `pixelguard`'s ceiling rather than a normal canvas.
  Kept, with the figures in the code: the alternative is a copy-on-write tile ledger
  inside the blend, which is the one loop where a mistake corrupts undo rather than
  dropping a frame.
- ~~**[Medium]** A brush-down that draws nothing triggers `_stamp_all` + full
  recomposite.~~ Built 2026-09-03 exactly as the entry prescribes: the autovivify was a
  swap in place, so `_discard_pending_cel` is the same swap back -- the placeholder is
  rebuilt with the uid it always had, and only its row is stamped and invalidated. The
  whole rebuild survives as the fallback for a row or column that has since gone.
- ~~**[Medium]** `_content_box` ... keyed on a never-pruned `id(doc)`.~~ Built
  2026-09-03: a weakref finalizer drops a document's whole block when it dies, which
  closes both halves at once -- the id cannot be reused while an entry for it exists, and
  the entries cannot outlive what they describe. (The rescan itself was already memoised
  on `(rev, layer_stamp)`.) ~~LRUs are Python lists with `remove()` per visible cell per
  frame.~~ Built 2026-09-03: dicts used as ordered sets, which is the ordering these need
  and a constant-time drop.
- **[Low]** `_below` cache is full-canvas float32 (64 MiB at 2048²) -- **kept**, and the
  only one of these four left: it is the base a stacked blend composites onto, and uint8
  there is banding on exactly the documents that have layers under the active one. One
  allocation, not per frame. ~~Mirror preview does up to 4096 `add_rect_filled` per
  frame~~ -- built 2026-09-03: one rect per *run*, split by the face box because the two
  sides are different colours. ~~`_adopt` writes the settings file on every tab open~~ --
  re-checked and stale: `Settings.set` compares before marking dirty and the write is
  debounced, so an identical block is a no-op. ~~`gifout.map_to_palette` is
  O(palette × pixels)~~ -- built 2026-09-03 over the region's *distinct* colours, which is
  `indexed.snap`'s own trick and at most 256 inputs after the snap above it.

---

## 6. Clay, Poser, Troupe, shared viewer

### Correctness
- ~~**[Medium]** Esc on a Clay element drag leaves the overlay VBO at the previewed
  positions.~~ Built 2026-09-03: `_restore_overlays` rewrites them from the mesh's own
  positions, at both doors. The overlay is keyed on `id(mesh)`, which does not change
  during a drag, so nothing else was going to.
- ~~**[Medium]** `reserve_uid` on the task thread races `new_uid` (T3).~~ ~~**[Medium]**
  Clay crash-recovery decode on the frame thread.~~ Re-checked 2026-09-03 and already
  fixed: `_journal_adopt` reads and parses on a task, `inker-recover`'s shape, and says
  so.
- ~~**[Medium]** Poser `_blend` interpolates a partial key from identity.~~ Built
  2026-09-03, and the answer differs by space because the problem does. In `delta` the
  identity *is* rest and the old behaviour was right. In `node` it is parent alignment,
  so the other end is **held**: the honest reading of a key that says nothing about a
  bone is that it says nothing, and the bone does not move rather than moving somewhere
  nobody authored. Blending from the bone's own rest would be better still and is not
  available -- `_blend` is handed two poses and no rig.
- ~~**[Medium]** Zero-sum or unnormalised skin weights collapse vertices in the
  viewer.~~ Built 2026-09-03 at the reader: renormalised (exact for a well-formed file --
  a sum of 1 divides by 1) and a zero-sum vertex pinned to its first joint, since the
  shader sums with no division and such a vertex went to the world origin as a spike.
- ~~**[Medium]** Clip-library key quaternions are never validated.~~ Built 2026-09-03
  through `validate_pose`'s own bone loop, split out as `validate_bones` -- a split
  rather than a flag, because an empty map is a mistake in a saved pose and is exactly
  what a clip's rest key is.
- ~~**[Medium]** Troupe atlas decoded and uploaded on the frame thread~~ -- fixed before
  the review by T2; both `_decode_atlas` and `_score_task` are task-thread halves and say
  so. The *double decode* stands and is deliberate: they are two tasks that need not both
  run, of a file the OS has just cached, and coupling their caches to save one decode on
  selection would tie the QA pass to the texture's lifetime.

### UX
- ~~**[Medium]** Poser right-click "Reset the whole pose" bypasses the guard the button
  has.~~ Built 2026-09-03: through `poser_mode.guard`, like the two buttons. A right-click
  menu is the easiest of the three doors to hit by accident.
- ~~**[Medium]** Poser and the shared viewer have no reframe key, no axis views, and a
  click on empty space orbits rather than deselecting.~~ All three built 2026-09-03.
  Poser reads **Clay's** `AXIS_VIEW_KEYS` rather than restating them, so the two
  viewports cannot come to disagree about which number is the front; `F` reframes through
  `frame_bounds`, since an armature has no mesh for `Model.bounds` to measure. A press
  that hits no marker still starts an orbit -- a drag from empty space must turn the
  model -- so the deselect is decided at the *release*, and `_motion` tells a click from a
  drag with a small threshold because a pen jitters on a tap.
- ~~**[Low]** No pan without a middle button in either viewer.~~ Built 2026-09-03:
  Alt+drag, which is what every DCC binds it to, beside the middle button. ~~Shade Auto's
  "whole document" branch is unreachable.~~ Built 2026-09-03: `any_object` is the gate
  that matches the op's own fallback. ~~`insert_key` has no caller.~~ Built 2026-09-03:
  the clip pane can add a key the library already holds, which it could not -- building a
  walk out of four poses meant authoring each of them twice. `PoserState.key_names` was
  written for exactly this picker and had no caller either.

### Structure / performance
- ~~**[Medium]** The node/rest rotation conversion lives only in `poser_mode` and is
  restated in `blender_worker`.~~ Built 2026-09-03: `rigging.node_from_delta` /
  `delta_from_node` are the one definition, beside the sentence that justifies them, and
  both ends call them -- the worker converting Blender's WXYZ rest at the boundary. They
  cannot share a *rest source* (one is bpy's, one the viewer's), which is why only the
  algebra moved. ~~`service.clips.preview_frames` has zero callers while `rebuild_frames`
  reimplements it.~~ Settled 2026-09-03 by saying so in its docstring: they share the
  interpolator, which is the part that must not differ, and they differ in *where the
  record comes from* -- the pane expands the clip being edited, and a scrubber showing the
  file instead would ignore the edit in front of it.
- ~~**[Medium]** Bevel rewrites every face in a Python loop~~ -- built 2026-09-03: a
  face the bevel does not touch is copied whole, so one edge on a 200k-face sculpt no
  longer runs the branch 200k times; the affected faces stay corner by corner, because
  which corners they grow genuinely is per corner. ~~Select More/Less reimplements the
  vectorised `_face_corner_mask`~~ -- built 2026-09-03, which also retires the second
  spelling of the rule those two verbs rest on being inverses of. ~~`Model.bounds()` per
  frame (T6)~~. ~~Hover motion redraws the MSAA scene~~ -- built 2026-09-03: only a
  *changed* gizmo hover is dirty. ~~Pose mode rebuilds every overlay per frame~~ -- closed
  by that: `_overlays` is only reached on a frame the renderer actually redraws.
- ~~**[Low]** `dissolve_edges` is O(selected × corners).~~ Built 2026-09-03: the corner
  list is sorted by edge once and each edge's pair of faces found by bisection.

### Tests
No Ctrl chord-during-drag beyond Ctrl+J and Ctrl+Z; nothing drives
`Viewer._press/_motion/_release`; `screen_ray` orthographic branch untested outside Clay.
`panes/troupe_bridge.py` and `panes/troupe_sheets.py` still have no test references.

---

## 7. Plotter and Packwright

### Correctness
- ~~**[High] Embedded XML tilesets drop presentation fields, Wang sets and `trans`.**~~
  Built 2026-09-03: `tsx.tileset_from_element` is the one definition both spellings go
  through, as `tileset_from_json` already was for the JSON pair. The comparator was
  widened *first* (below), so the fix is proved rather than asserted: reverting it fails
  `presentation-112` and `wang-112` in `test_fixture_corpus.py`.
- ~~**[High] A failed repack leaves the previous atlas on screen and exportable.**~~
  Built 2026-09-03: `PackTab.pack_stale_why` says why the atlas on screen is not the one
  the document describes; the preview marks it, both exports refuse it, and the bridge's
  two buttons grey with that sentence as the reason. The last good picture is still
  drawn -- it beats a blank pane -- it just no longer passes for current.
- ~~**[High] Grid packs relocate trimmed tiles to the cell's top-left.**~~ Built
  2026-09-03: a grid pack never trims, whatever the setting says, because the two things
  the setting could mean there -- a smaller atlas and aligned tiles -- cannot both be had
  and a mode whose purpose is arithmetic alignment answers to the second. MaxRects still
  trims and its sidecar still records the box. The settings pane says so where the toggle
  is.
- ~~**[High] Dropping several files onto Packwright keeps only the first.**~~ Built
  2026-09-03: the add key carries the batch (`inker-open`'s shape), so every distinct
  drop runs and a repeat of the same one still dedupes.
- ~~**[High]** Re-adding a changed source PNG is skipped as a duplicate
  (`packwright_mode.py:329`), contradicting `wpack.py:16`.~~ Built 2026-09-03:
  `PackDoc.replace_source` swaps one source's pixels in a single undo step, keeping its
  uid and its rename, and the toast names both halves of a mixed batch. Identical pixels
  are still a skip.
- ~~**[Medium]** `tileset_usage` misses grouped tile layers and tile objects.~~ Built
  2026-09-04: it walks `all_layers()` and counts a `TileShape`'s gid as the cell it is.
- ~~**[Medium]** Closing the active tab carries brush/palette/terrain/selection onto the
  neighbour.~~ Built 2026-09-04, and only when the *active* tab went: closing a background
  tab is not arriving anywhere.
- ~~**[Medium]** Image-layer texture stamped on `id(pixels)`.~~ Built 2026-09-04: the
  array is pinned beside the stamp, which is what makes an identity sound -- the
  ``_content_box`` fix in another mode. ~~**[Medium]** Ctrl+Shift+Up/Down reorders layers
  while saving.~~ Built 2026-09-04, gated inline rather than through `_MUTATING_CTRL`
  (which keys on the letter and would swallow a plain Ctrl+Up nudge too). ~~**[Medium]**
  The object clipboard is a live reference.~~ Built 2026-09-04: `_apply_object_props`
  writes straight onto the object, so the copy is taken at the press. ~~**[Medium]**
  Terrain re-fit treats an infinite map's window edge as the map edge.~~ Built
  2026-09-04, and the parameter for it already existed: `outside=not doc.infinite`. Past
  an infinite map's window the map is genuinely unpainted, so the terrain must grow an
  outline there rather than run seamlessly off into nothing. ~~**[Medium]** Foreign Wang
  set `class`/representative tile dropped.~~ Built 2026-09-04 on both spellings, read and
  written. ~~**[Medium]** Deprecated image-layer `x`/`y` folded on XML, dropped on JSON.~~
  Built 2026-09-04 -- one map read with its image in place from a `.tmx` and at the origin
  from the `.tmj` beside it. ~~**[Medium]** Absolute filesystem paths persisted as sprite
  keys.~~ Built 2026-09-04: `sources.file_key` is the stem plus a digest of where the file
  came from -- stable per file, which is what makes a re-add a replacement, and carrying
  no directory into a shared document.
- ~~**[Low]** Tile-object `gid` unchecked~~ -- said out loud now, once per read, naming
  the object and the layer. ~~`_image_layer_files` can be shadowed by the map name~~ --
  the bundle's own names are reserved, so a source that spells `map.tmx` takes the
  `images/` fallback instead of replacing the map with PNG bytes. ~~Unknown `staggeraxis`
  silently replaced~~ -- logged on both spellings; on a staggered map that fallback moves
  every other row half a tile. ~~MaxRects tries only POT sizes even with
  `power_of_two=False`~~ -- the limit itself is the last candidate, so a 1500 px ceiling
  is reachable. ~~`wpack` coerces `"false"` to True~~ -- `_json_bool`, because the file is
  hand-editable. All 2026-09-04.

### UX
- ~~**[Medium]** Tileset removal is modelled and undoable but has no UI.~~ Built
  2026-09-04: a Tileset menu row, with the model's own refusal (which names the count and
  the layer) passed on. ~~**[Medium]** "Add to Packwright" from Library/Troupe toasts
  "Start or open an atlas first".~~ The Library door already minted; Troupe's and the file
  *drop* did not, and do now (2026-09-04) -- an atlas has no numbers that cannot be taken
  back later, which is that rule's own argument. ~~**[Medium]** No pivot/anchor control or
  preview.~~ Built 2026-09-04: `PackDoc.set_pivot` (one undo step, `None` clears it and is
  deliberately not the same as the centre), a checkbox and two folded drag fields on the
  selected source, and a cross in the preview placed by the frame's trim exactly as
  `texturepacker._pivot` normalises it. ~~**[Medium]** Export encode errors reach the user
  as "see the log".~~ Built 2026-09-04: framed as a `ServiceError`, since only that text
  survives the task classifier -- and the sentence being thrown away was the actionable
  one.
- ~~**[Low]** Every repack resets the view~~ -- only a repack that changes the atlas's
  *shape* does now, so a rename no longer flings somebody working at 400% back to "fit".
  ~~Undo drops the object selection~~ -- it is pruned rather than cleared: what the clear
  was for is an object the step *removed*, and undoing a nudge deselecting the thing that
  just moved back was the cost. ~~Deleting a layer leaves `selected_objects` pointing into
  it~~ -- the same prune, at both delete doors. ~~Layer-menu reasons name the wrong
  cause~~ -- T4. ~~Bridge label always says "JSON (Array)"~~ -- it reads the document's
  schema. All 2026-09-04.

### Structure / performance
- ~~**[Low]** `tileset_from_inker` bypasses `land_tileset`.~~ Built 2026-09-04: that
  function is "the whole tail of the arrival" and this was a hand copy of its first line,
  so a drawing brought over from Inker landed without selecting itself, without clearing
  the stale brush, without arming its terrain and without refitting the view.
  ~~`_shift_layer` duplicates `shift_layer`.~~ Built 2026-09-04 -- and the two had already
  come apart at the ends, so the same press did different things from different
  controls.
- ~~**[Medium]** Save/Export encode on the frame thread (T2).~~ ~~**[Medium]**
  Per-visible-cell Python rebuilding an 11-field `Lattice` per call.~~ Built 2026-09-04 on
  both paths by hoisting the lattice out of the loop -- it cannot change inside one
  layer's draw. No LOD: the cell count is already bounded by the visible block, and a
  second picture at a second detail level is a second thing to keep agreeing with the
  export. ~~**[Low]** Objects drawn without culling~~ -- culled against the pane with a
  margin, which the tile layers already were. ~~Wang fill is per-cell Python~~ -- the
  assert-write is one fancy index now; the *re-fit* is unavoidably sequential (each cell
  reads neighbours the loop may already have re-chosen) and that is the algorithm rather
  than the implementation, which the code now says. ~~Source/packed lists have no
  `list_clipper` at 1024 rows~~ -- both clipped; the sources list's selected row is taller
  than the rest and the clipper re-measures, which is written down where it matters.

### Tests
~~`_tileset_facts` blind to `tiles`, `wangsets`, presentation~~ -- widened 2026-09-03 to
`tiles`, `wangsets`, `phases`, the collection and the five presentation fields, which is
what caught the embedded-reader drift above. Written XML spellings still never asserted;
still no Tiled-authored fixture, which is `TODO.md` P7 and a human's.

---

## 8. Sirens

### Correctness
- ~~**[High]** Tempo/Speed/Rows sliders and name fields push a step per frame (T1).~~
- ~~**[High] The playhead is a row index into an imaginary single-pattern timeline.**~~
  Built 2026-09-03: `synth.render_marked` emits one `(sample offset, order index, pattern
  uid, row)` per row *as it was actually played*, `SongTab.mark_at` bisects it, and
  `playhead_row` is `None` while the song is in a pattern the grid is not showing. The
  map earns its keep three more times below: follow, play-from-caret, and knowing that an
  audition on the one channel is not the song.
- **[Medium]** A dragged release marker can land on step 0 and silence the held note
  (`sirens_envelopes.moved:154`); a loop dragged past the release vanishes from the
  graph but stays in the document.
- **[Medium]** Row-scoped effects (`1xx`, `2xx`, `4xy`, `Axy` last one row) differ
  silently from FamiTracker's persistent ones; the manual gives no persistence rule
  (`synth._apply_row:209`).
- ~~**[Medium]** The "instrument kind vs channel kind" refusal in `document.py:169` is
  documented, not enforced.~~ Settled 2026-09-03: the sentence went (T5). **[Low]** `read_wsng` collapses duplicate sample keys silently.

### UX (what a FamiTracker/Furnace/OpenMPT user cannot do)
- ~~**[High]** No delete/rename/reorder/retarget in the order list, no loop point other
  than 0, no duplicate pattern.~~ Built 2026-09-03: every row carries move/retarget/delete,
  the loop point is any entry (and follows an entry that moves under it —
  `sirens_orders.moved_loop`), patterns rename, duplicate (`SongDoc.duplicate_pattern`,
  one step) and delete with a confirm naming the order entries that go with them. "+ To
  order" refuses with the effect's name when the caret is on a sound effect's pattern.
- ~~**[High]** Grid click does not pick the column; click on `Fxx`, type, get a note.~~
  Built 2026-09-03: `sirens_patterns.column_at` measures the cell that was drawn, and the
  gap after a column belongs to it.
- ~~**[High]** No mute/solo.~~ Built 2026-09-03: a header strip over the grid names every
  channel (they had no name on screen at all) and carries mute and solo, which reach the
  mix through `synth.render_only` — `_stem_render`'s body, now the engine's, so a stem and
  a mute are one operation. Both are view state: a `.wsng` that remembered a mute would
  hand somebody else a song with a missing part. **Still open: channel pan/rename/kind
  have no UI** and `update_channel` still has no caller; the header is where they go.
- ~~**[Medium]** Playback cannot start from the caret, play one pattern, or loop.~~
  Built 2026-09-03: **From the caret** slices the rendered buffer at the row map's own
  offset (and says so rather than starting from the top when the order list never reaches
  that row); **This pattern** is `render_pattern`'s first caller, under its own key so the
  song's buffer is never touched; **Loop playback** passes `loops=-1` and the playhead
  wraps with it. The song's `loop_order` stays what the exported WAV's `smpl` chunk says
  — a property of the song, not of how it is being listened to.
- **[Medium]** No note preview on entry; no keyboard instrument selection; no
  Home/End, Insert/shift-rows, interpolate, or step increment.
- ~~**[Medium]** Follow mode hides the caret and edits go to an off-screen row.~~ Built
  2026-09-03: following moves the caret onto the sounding row, so what is under the
  highlight is what a keystroke writes to — the tracker answer, and one click off for
  anyone typing into bar 3 while bar 1 plays.
- **[Medium]** Channels past the pane width are invisible with no scroll or marker
  (`_grid:220`); Left/Right happily move the caret into them.

### Structure / performance
- **[Medium]** Split `sirens_mode.py` into `sirens_edit.py`, `sirens_play.py` and keys,
  keeping the re-export block as the compatibility surface.
- **[Low]** The envelope editor's pure half (`span`, `painted`, `moved`, `toggled`,
  `grabbed`) lives in an imgui-importing pane; move it under `studio/sirens/`.
- ~~**[Medium]** Every accepted render and every journal tick re-encodes every sample to
  WAV on the frame thread (T2).~~ **[Low]** `_cell_text` re-imports `synth` per cell
  (`sirens_patterns.py:133`).

### Tests
No test for undo granularity of a slider or long drag; `playhead_row`
against a multi-entry order; envelope marker invariants; effect persistence; grid click → column; paste
wider than the remaining channels. None of the six Sirens panes has a test.

---

## 9. Suggested order of work

1. ~~The "one gesture, one step" sweep (T1).~~ Done 2026-09-03.
2. ~~The frame-thread sweep (T2), then the task-thread writes (T3).~~ Done 2026-09-03.
3. ~~The three doc/code contradictions left in T5.~~ Done 2026-09-03.
4. ~~Plotter interop: embedded tileset reader, `_tileset_facts` parity. Packwright:
   failed-pack state, grid+trim, batch drop, replace-on-readd.~~ Done 2026-09-03. The
   *JSON zero coercion* was checked and is not a defect: `props.json_number` already
   fixed the JSON side, and the XML side never had it -- an attribute is a string, so
   `"0" or 1` is `"0"`. `tests/plotter/test_tmx.py` pins both syntaxes agreeing about a
   stored zero. **One Tiled-authored fixture is the human's** (`TODO.md` P7).
5. ~~Sirens playback: row map, follow-mode caret, order-list editing, mute/solo,
   play-from-caret.~~ Done 2026-09-03. What is left in section 8 is smaller and named
   there: channel pan/rename/kind, note preview on entry, Home/End and the row-insert
   verbs, the channels past the pane width, and the row-scoped effects question.
6. ~~Inker parity: zoom ceiling, snapped overlay lines, mirrored cursor, missing
   chords~~ — done 2026-09-03, along with the three section-5 **[High]** items (combine
   mode, the one-pixel selection, the stale `fg_slot`). The *footprint* cursor is the one
   piece left and is named in section 5. The "mirror fix" was withdrawn on 2026-09-03 as
   measured-and-wrong; see the top of section 5.
7. ~~Settings' three layout doors: the `save` latch, the dead "Sidebar width", and
   "Reset pane sizes".~~ Done 2026-09-03. All three were the same shape -- a control
   that redraws, toasts and persists nothing -- and all three lived in the seam between
   `layout.Layout` and `layouts.Library`. `tests/test_layouts.py` builds the pair,
   because a test holding only the `Layout` reproduces the old passing behaviour.
8. ~~The three Inker output paths: empty-cel regeneration, per-frame GIF palettes, and
   the `.ora` tile grid.~~ Done 2026-09-03. The GIF and `.ora` tests assert the *file*
   -- decoded frames, and a document byte-identical to its pre-load state -- since both
   defects are invisible to a test that inspects arguments.
9. ~~Extractions (T7) once the behaviour above is pinned, so the moves are pure.~~ Done
   2026-09-04, in that order and for that reason -- see T7 for what each file became and
   for the one door back. Section 8 (Sirens) is the only thing left in this document, and
   it is set aside rather than owed to this review.

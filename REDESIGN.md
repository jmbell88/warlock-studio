# REDESIGN.md — the "Apple-like" UI program

The staged plan every wave session cites. Written 2026-08-15, before wave 1.

## Context

A design review concluded Warlock Studio already has Apple-adjacent foundations
(Inter type, tokens, elevation, motion, themes, strong accessibility) but fails
on **information hierarchy**: 13 nav destinations, uniform grey buttons,
bordered boxes everywhere, developer chrome (FPS/VRAM readout, power icon), a
changelog-dominated Home, and toolbars that clip at 150 % scale. The fix is
structural, not cosmetic: show less chrome, reveal tools in context, let the
asset dominate.

### Decisions taken during planning (do not relitigate)

1. **Full program, staged waves** — all of the verdict, including the 2D/3D merge.
2. **Navigation = collapsible left icon rail** (~52 dp icons, expandable to ~188 dp with labels).
3. **Full restructure** — manual and profiles stop being modes (Help overlay;
   Profiles folded into the 2D flow). Settings stays a mode, in the rail footer.
4. **2D/3D merge designed to execution detail now**: one `create` mode with a
   stage rail Reference → Mesh → Rig → Pose → Export. Poser stays a separate
   workspace (it authors reusable template-scoped poses on its own Viewer; the
   Pose stage is the existing `pose_panel` on the shared viewer — different
   editing sessions by construction).

### Repo state at authoring time

`master` includes the merged inker update (baseline ~8177 passed / 22 skipped).
`plotter-wave-2` is an unmerged branch touching plotter panes — waves here touch
`plotter_canvas.py` / `plotter_tools.py` only lightly (empty-state unification);
expect a small merge conflict there and keep plotter churn minimal until
wave-2 merges.

### Execution conventions

One branch per wave, merged to master when the full suite is green (never edit
`src/` while pytest runs; a bare `uv sync` prunes the extras). Commit subjects
`Warlock v0.0.22` — no version bump unless asked.

## Wave order and why

1. **W1 Tokens + theme** — smallest diff, app-wide blast radius; everything else
   sits on the new metrics, so land and verify it alone.
2. **W2 Shared widget layer** — ghost buttons, overflow toolbar, empty-state
   unification, value formatting; no pane changes yet.
3. **W3 Shell** — Help overlay, Profiles fold-in, the rail, `_mode_switch`
   deletion. Owns `main._build_ui`'s outer structure.
4. **W4 Panes** — Settings, toolbars, Home, Library split view. Sits on W2's
   widgets inside W3's shell.
5. **W5 Create mode** — the `2d`/`3d` → `create` collapse with the stage rail.
   Last because it edits `MODES`/`RAIL_GROUPS` (a two-line nav edit thanks to
   W3's data-driven rail) and rewires `_sync_viewer`/dispatch that W3/W4
   stabilised.
6. **W6 Headings + polish** — sentence-case sweep, screenshot pass, prose sweep.

---

## Wave 1 — Tokens + theme (`tokens.py`, `theme.py`)

- **Radii**: `RADIUS_M` 6→8 (controls), `RADIUS_L` 10→12 (surfaces); `RADIUS_S`
  stays 4. In `theme.apply`: window/child/popup rounding → `sp(RADIUS_L)`;
  frame/grab/tab/scrollbar rounding → `sp(RADIUS_M)` (deletes the inline
  `RADIUS_S + 1` hack). Check other `RADIUS` readers (`card`,
  `push_surface_rounding`, chips) still read as intended.
- **Type**: `TEXT_SMALL` 11→12, `TEXT_BODY` 13→14; TITLE/HEADING/DISPLAY
  unchanged. Audit height constants tuned to 13 px body: `library.CARD_HEIGHT`
  92→~96, `COMPACT_HEIGHT` 46→~50, `_footer_px` seed. The line-height lever if
  cramped is `item_spacing`, not fonts.
- **Hairlines**: `child_border_size` `sp(BORDER)` → `0.0` (panes separate
  tonally: PANEL on BG). Keep `popup_border_size` and `card`'s EDGE border. If
  panes visually merge at splitters, widen workspace `item_spacing`, don't
  restore the border.
- **Secondary hover**: `button_hovered` ACCENT@0.75 → EDGE; `button_active`
  ACCENT → ACCENT@0.4 (accent is the press flash, not the hover). Same
  treatment for `header_hovered`/`tab_hovered`.
- Add `tokens.DISABLED_ALPHA = 0.6`, set `style.disabled_alpha` in `apply`.
- **Do NOT touch the palettes** — preserves the pinned composite 3.20
  (`test_accessibility.py:36-56`) and light-palette luminance ordering
  (`test_forms_and_layout.py:244-298`).
- Test fallout: `test_ux_phases.py:92` (`RADIUS_L > RADIUS_M`) holds; nothing
  pins radius values or hover colours.
- Verify: full suite + `scripts/screenshot_modes.py`, both palettes ×
  1.0/1.5/1.75, before starting W2.

## Wave 2 — Shared widget layer (`widgets.py`, new `toolbar.py`)

**B1 Ghost/borderless buttons.** `_glyph_button` (`widgets.py:1569`) gains
`borderless: bool = False` — transparent resting fill, hover fades in ELEV_2
(reuse the existing hover-mix machinery); thread through
`icon_button`/`small_icon_button`. New `ghost_button(label, size=(0,0), *,
enabled=True, reason="", tooltip="")` — labelled transparent button that keeps
standard `frame_padding` so `button_width()` stays correct (smoke tests rebuild
rows from it); delegates to `disabled_button` for the reason/tooltip contract.

**B2 Disabled primary.** `primary_button` (`widgets.py:1701`): branch on
`enabled` — disabled pushes neutral ELEV_2 (dimmed by `DISABLED_ALPHA`) instead
of accent. Add `reason=`/`tooltip=` forwarding. Read the AST matcher in
`test_ux_todo_fixes.py:122-155` first: if it keys only on the name
`disabled_button`, forwarding is API hygiene, not test-forced; fix any newly
swept call sites in the same commit.

**B3 Overflow toolbar — new `studio/toolbar.py`.** Generalises
`segmented_control`'s measure-then-swap:

- `Item(key, label, icon="", tooltip="", enabled=True, reason="", danger=False,
  priority=0, pinned=False)` — ids pinned to `key` (`##bar/key`), never the
  rendered label.
- Pure `plan(widths_full, widths_icon, priorities, pinned, avail, overflow_w)
  -> list[tier]` (`full|icon|menu`) — headless-testable.
- `toolbar(bar_id, items, on_click=None, *, trailing=None)` — `trailing` draws
  fixed-width non-collapsible controls (combos, sliders) measured first and
  subtracted.
- Rules: tiering per priority group all-or-nothing; collapse order label→icon→…
  menu (full labels return in the menu, never a second abbreviation); pinned
  (destructive, transport) never enters the menu; buttons render via
  ghost/borderless variants.
- Tests: pure `plan()` table + smoke coverage at 1.0/1.5/1.75 per rewritten row.

**B4 Empty states.**

- Fix `empty_state` hint wrapping (`widgets.py:1826` `centred()` measures
  unwrapped): wrap at `min(avail - margins, sp(360))` via
  `calc_text_size(wrap_width=)`.
- Add optional `action: tuple[str, Callable] | None` → centred `primary_button`
  under the hint.
- New `nothing_open(hint, actions, *, recent_paths=(), on_open=None)` unifying
  the four-way copy-paste: `inker_canvas.py:298-322`, `main.py:2696-2735`
  (`_clay_empty`), `plotter_canvas.py:164-177`, `packwright_preview.py:105-116`.
- Fallout: `test_ux_phase45.py:210` (three registers — additive params fine);
  `test_notifications.py:283-297` requires the literal `empty_state(` in six
  files — verify each swept file keeps it; `test_plotter_mode.py:1658` pins the
  droppable-suffix wording — keep it in the plotter hint;
  `test_studio_smoke.py:1413` rebuilds the clay block.

**B5 Value formatting.** `labeled_slider_float` gains `fmt: str | None = None`,
`percent: bool | None = None` (None = infer from 0–1 range → display ×100 with
`"%.0f%%"`, return `/100`). Pure `float_format(low, high, step=None)` modelled
on `clay_ops.format_for` (span ≥100 → `%.0f`, ≥10 → `%.1f`, else `%.2f`). Sweep
~10 call sites (`inker_tools`, `inker_layers`, `plotter_layers`,
`texture_panel` — all 0–1 → percent); give `_transform_row`'s X/Y scale sliders
`"%.2fx"`. Unit tests: format table + percent round-trip.

## Wave 3 — Shell (rail, Help overlay, Profiles fold-in)

Verified anchors: dispatch `main.py:2322-2359`; `_mode_switch` `:3652+`;
F1→manual `_shortcut :1846`; `pygame.QUIT`→`_request_quit :1603`; `modes.py`
`MODES :27-48`, `GROUP_BREAKS :102` (derived), `QUIT :133`.

### Step 3.1 Help overlay (independent of 3.2)

- `ManualState` gains `open: bool = False` (`state.py:646` — the docstring's
  "visibility is mode=='manual'" is deliberately inverted; say so in the field
  comment).
- New `manual/render.draw_overlay(ctx)`: centred frosted window
  (`window_shadow` + `window_backdrop` + `popover_enter`), `min(work − margins,
  sp(1040))` × ~80 % height, existing `draw_body` TOC+page split verbatim +
  close row. Drawn in `_overlays` before `palette.draw` (Ctrl+K floats above
  Help).
- `help_button(ctx, key)` keeps name/signature/call-site shape — body flips
  from `set_mode` to `open = True`; `open_at(*target)`. Both `tests/manual`
  gates stay green with zero call-site edits. Same for
  `open_at`/`troubleshooting_button`.
- F1 toggles the overlay; Esc-closes branch at the top of `_shortcut` (above
  the WORK_MODES dispatch — Inker consumes every key).
- Remove `manual` from `MODES` + its dispatch branch; palette gains a `help`
  command (hint "F1").
- Migrate: `test_mode_keys.py:98` (F1/Esc), `test_palette.py` /
  `test_studio_wiring.py` `go:manual` literals, frost-set test extension
  (`test_ux_phase45.py:506` — add positive asserts), `state.py` docstrings.

### Step 3.2 Profiles fold-in (independent of 3.1)

- `AppState.profiles_open: bool`; the manager becomes a frosted overlay sheet
  (~`sp(560)`) in `_overlays`, body = `profiles_panel.draw(ctx)` unchanged
  (keeps its file, list↔editor switch, heading,
  `help_button(ctx,"profiles")` — manual gates and HELP_TARGET stay alive).
- Entry: **Manage…** button in `settings_2d._profiles` (`:436-487`).
  `_journal_adopt` retargets to `set_mode("2d")` + `profiles_open=True`. Sheet
  close routes through `profiles_panel.guard` (a dirty draft still asks);
  quit-chain entry untouched.
- Remove `profiles` from `MODES` + dispatch; palette gains a `profiles` command
  ("Manage style profiles") replacing `go:profiles`.

### Step 3.3 The rail (the big commit; depends on 3.1 + 3.2 for final KEYS)

- `modes.py`: reorder `MODES` to rail order — home, 2d, 3d, library, review |
  inker, clay, poser, plotter, packwright | settings (11 modes; Review is
  visually primary while staying in `WORKSPACE_MODES` — those describe drawing,
  not nav). Delete `GROUP_BREAKS` and `QUIT`. Add hand-written `RAIL_GROUPS`
  (3 tuples as above) + new test: flatten == `KEYS` in order.
  `_SINGLE_PANE_MODES = ("home","settings","library")`.
- New `studio/rail.py` (beside `layout.py`, **not** in `panes/` — avoids the
  `help_button` coverage sweep). Drawn as the first column inside `##host`:
  rail child (borders) → `same_line` → content child (`doctor_banner` +
  dispatch unchanged). Items: `invisible_button` id `rail/{key}` + hand-drawn
  glyph, 48×44 dp hit area, selected pill slides via
  `motion.value("rail/pill",…)`; switching via `App._set_mode` (the crossfade at
  `:2282` fires unchanged; `test_mode_writes` clean).
- Expand/collapse: `RAIL_W=52`, `RAIL_EXPANDED_W=188` dp; eased via
  `motion.value("layout/rail",…, DUR_BASE)` (snaps under reduce-motion); labels
  alpha-fade with width. Toggle = footer chevron. Persisted as
  `"rail": "icons"|"labels"` in the existing Layout settings dict
  (`Layout.save()` replaces the whole dict — add the key there; unknown →
  `"icons"`).
- Responsive forced collapse: drawn target = `min(preference, room)`; when
  `work_width < sp(188) + 2·sp(SIDEBAR_MIN) + sp(CENTRE_MIN) + spacing`, draw
  collapsed (preference untouched). At `MIN_SIZE`×1.5 expanded does not fit —
  this rule is the guard; pin with a test.
- Layout integration: `layout.RAIL_RESERVED: float = 0.0` module state, set by
  `rail.tick()` before `layout.tick()`/`measure()` at `main.py:2251`;
  `measure()` subtracts it. `fit()` stays pure; headless tests unaffected
  (default 0).
- Tooltips: icon-only mode shows label tooltips (accessible-name rule);
  suppressed when expanded except health.
- Keyboard: the rail takes no arrow keys (doctrine at `modes.py:108-116` —
  mouse + Ctrl+K; a shell nav-key claim would collide with `NAV_KEY_MODES`
  surfaces). Record the decision + the future OR'd-flag shape in `rail.py`'s
  docstring.
- Footer: health badge only when failing (ERR/WARN ladder, tooltip = failing
  checks, click → `_diagnostics_popup`; add a palette `diagnostics` command so
  it is reachable when green) · Help · Settings · chevron.
  `shortcuts_requested` consumption moves here (Ctrl+/ unchanged). If
  `open_popup` inside the rail child hits id-stack scoping, hoist to host scope
  after `end_child` via the one-shot-flag pattern.
- Delete `_mode_switch` entirely. No header row remains — readout gone (F10
  `overlay.fps_meter` is the developer path; `Resources` in `state.py`
  untouched), quit icon gone. Re-route `pygame.QUIT` (`main.py:1603`) from
  `_request_quit()` to `_ask_quit()` so the preflight summary survives on the
  now-only interactive quit path.
- After the commit, grep for orphaned consumers: `shortcuts_requested` (a
  writer with no reader?) and `_diagnostics_popup` registration.

### Step 3.4 Docs/comments sweep

`modes.py` docstring, `_shortcut` comments, `state.py` mode lists, manual
chapters 02/12/14/17 prose (no renumbering — `tests/manual` gates both
directions).

### Step 3.5 Screenshots

`screenshot_modes.py` gains captures: Help overlay open, profiles sheet open,
rail expanded. Run and eyeball.

### Test migration (condensed)

Delete+replace the three header smoke tests (segment-fit `:2697`, right-strip
`:1805`, health-label `:1880`) with rail equivalents (fits `MIN_SIZE` at 3
scales; items inside the child, non-overlapping; badge only-on-failure);
rewrite the quit tests (`test_studio_state.py:473`, `test_ux_phases.py:427`) as
"quit is never a mode and has no shell control; `pygame.QUIT` routes to
`_ask_quit`"; delete the `GROUP_BREAKS` test, add a `RAIL_GROUPS` partition
test; keep partition/`KEYS[0]`/wiring-parity/palette-derivation tests (they
derive from `MODES`). Extend the `SP_SWEPT` equivalent to `rail.py`. New: rail
persistence round-trip; Help overlay state machine; profiles-sheet
Esc-through-guard.

### Risks

1. Rail width must be measured **before** `layout.measure()` or the columns
   disagree by a frame.
2. Missed `_mode_switch` consumers fail subtly — grep after.
3. Esc ordering: Help close → profiles-sheet close (via guard) → workspace
   `handle_key` → `_escape_mode`, in that order at the top of `_shortcut`.

## Wave 4 — Panes

### 4.1 Settings (`app_settings.py`)

Centred surface — `content_w = min(avail, sp(640))`, centred child. Categories
via `segmented_control("settings-cat")` under `pane_title("Settings")`,
selection in `state.preview["settings_category"]` (session-scoped): Appearance
(= Interface), Models (unchanged, width-bounded), Storage (new: `ctx.cache`
storage figure, trash summary via `library.measure_trash`, and **Prune…** /
**Clean library…** moved here under a Maintenance sub-section —
`library._ask_prune`/`_ask_clean` become public; the confirm bodies stay in
`library.py`), Advanced (= Layout + config table). `help_button(ctx,
"app-settings")` stays after the title.

### 4.2 Toolbars (all on W2's `toolbar.py`)

- Inker file actions move to the bridge (resolves the Inker-vs-others
  asymmetry): `inker_bridge.draw` gains the file group mirroring
  `plotter_bridge`'s shape (New/Open/Save/Save as/Export PNG full-width
  `disabled_button`s with busy reasons + recent section; popup ownership moves).
  `_file_row` slims to undo/redo (borderless icons) + view controls + status via
  `toolbar()`.
- `_transform_row` → toolbar: flips/rotates compact to icons, **Apply** = the
  row's one primary (pinned), **Cancel** ghost (pinned), sliders as `trailing`.
- `inker_timeline._transport` (the 17-`same_line` worst case) → toolbar:
  transport glyphs pinned (check the lucide cmap has play/skip glyphs first —
  the vendored subset is partial; else keep ASCII in icon-sized buttons),
  counter pinned, frame-ops priority 1, toggles 2, exports 3 (first into
  overflow); ms input + `help_button` trailing (`help_button` must follow an
  item on the line).
- `library._bulk` → toolbar: count + Clear pinned; **Delete** pinned —
  destructive actions never hide behind ….
- Stretch: `sheet_panel`/`clay_tools`/`plotter_tools` adopt opportunistically
  (keep plotter minimal until `plotter-wave-2` merges).
- Smoke tests rewritten to drive rows by `Item.key` + clip detection via
  `plan()`.

### 4.3 Home (`landing.py`)

Single column replacing the 50/50 split.

- Dismissible **What's New** card (only when `changelog.current(version)`
  exists and `settings["news_seen_version"] != version`; headline + 3 bullets +
  ×; full history via a footer ghost link → popup with the existing
  collapsing-header rendering). Pure `news_should_show()` helper.
- One primary **New…** button opening a menu of the six creation types (reuses
  `start_*` unchanged); the 3-across grid dies.
- **Resume** becomes the dominant thumbnail grid: cells ~`sp(168)`,
  `widgets.card` each — thumb `sp(136)` (assets: the same texture path as
  library cards; documents: `thumb_placeholder` + mode glyph — future work: a
  document preview cache), name, ago. `Row`/`rows()`/`open_row`/`activate`
  unchanged (most of `test_panes_home.py` survives). The keyboard ring keeps ±1
  wrapping. Thumb lookup extracted to new `panes/thumbs.py`
  (`job_thumb(ctx, job, side)`) shared with Library.
- Status → one quiet muted line: keep health + queue + review (actionable);
  drop the library row. `status_rows(ctx)` stays the pure shared source (the
  rail health badge reads the same data — no forking).
- Migrate `test_panes_home.py:218-299` (status set), smoke rebuilds of
  `landing.draw`.

### 4.4 Library split view

`library.py` stays the core (filters/actions/cards/bulk + the compact sidebar
list, unchanged API); new `panes/library_full.py` composes the full-window
mode:

- Left rail `sp(220)`: query + prefix chips (made public), collections as
  selectables (Status/Kind/Favourites/Trash writing the same
  `ctx.state.filters`), sort at the bottom, storage as one quiet line
  (maintenance now lives in Settings).
- Centre: thumbnail grid (cell ~`sp(160)`, `thumbs.job_thumb`, status pill
  overlay, star badge), date-group headings under newest sort, `_load_more`,
  bulk toolbar beneath.
- Right `sp(340)`: `inspector.draw(ctx)` unchanged, drawn only when a selection
  resolves.
- Keyboard: extend `library.select_relative` — left/right ±1, up/down
  ±columns; the column count is published per-frame in
  `state.preview["library_columns"]`.
- `main.py` library dispatch → `library_full.draw`. `library_full.py` needs
  `help_button` (reuse key `"library"`) — read `tests/manual/test_coverage.py`
  first for per-file rules, exemption if needed. Grid empty paths reuse
  `_empty`/`no_matches`.

## Wave 5 — Create mode (the 2D/3D merge)

Design summary: one `create` mode replaces `2d` + `3d`; a stage rail (new
`widgets.stage_rail` — segmented-control idiom, compact fallback, three segment
states: done / current / blocked-with-reason) sits atop the left settings
column; stages are derived availability, volatile position
(`AppState.create_stage`, never persisted); the lineage
(`parent_id`/`rerun_of`/`candidate_group` + `DERIVED_PARAMS` discipline) threads
the stages — no new object, nothing moves in `service/`.

- New imgui-free `studio/create_stages.py`: `STAGES` (grows across steps:
  `("reference","mesh")` → +rig, pose → +export), `reached(job, rig_meta)`,
  `available(stage, job, ctx)` (reuses service validation wording verbatim;
  rig-unavailable is disabled-with-reason, not hidden), `go(ctx, stage, *,
  select=None)` — the ONE stage switch (routes through `set_mode`; may move
  selection along the lineage; every exit from the pose stage routes through
  `pose_panel.guard`).
- Stage↔pane mapping: Reference = `settings_2d` (the profiles picker stays;
  **Manage…** from W3) | Mesh = `settings_3d` + candidates/quality/verdict in
  the inspector | Rig = new `panes/stage_rig.py` (lifts `library._skeleton` +
  the rig action; joins `SP_SWEPT`; `HELP_TARGETS["settings-rig"]` + a manual
  anchor in the same commit) | Pose = `pose_panel` re-hosted in the left
  column, `sheet_panel` inspector-side | Export = `_downloads` grid. The
  inspector drops its `tab_bar` in create mode (the stage rail drives; the
  inspector = stage-scoped evidence).
- `settings_2d.py`/`settings_3d.py` keep their files, names, `FOCUS_PANE` keys,
  help keys and `SP_SWEPT` rows — the wave re-routes *around* the repo's
  densest test surface, never through it.
- `_sync_viewer` re-keys mode→stage (~4 lines; model jobs carry their own
  `input.png`, so Reference can show a mesh's source without a parent lookup);
  `overlay.PLACEHOLDERS` gains `create/{stage}` entries;
  `offers_inker`/`shows_tiled` re-key to the reference stage.
- Steps (each a green checkpoint): **5.0** pure `create_stages` + tests →
  **5.1** `stage_rail` widget + `create_stage` state → **5.2a** helper
  indirection (replace `mode == "2d"`/`"3d"` reads in
  overlay/inspector/library/landing/palette with `create_stages.in_*()` helpers
  whose bodies still read the old strings — a pure refactor) → **5.2b THE
  FLIP** (one atomic commit: `modes.py` MODES/WORK_MODES/VIEWPORT_MODES =
  `{"create"}`, `RAIL_GROUPS`, `_build_ui` else-branch draws the stage rail +
  dispatches on stage, `_sync_viewer`, helper bodies re-keyed,
  `set_mode("2d"/"3d")` call sites → `create_stages.go`, all test literals —
  ships honestly as a two-segment rail with rig/pose/export still inspector
  tabs) → **5.3** Rig & Pose stages → **5.4** Export stage → **5.5** flow sugar
  (lineage links, the Poser cross-link "Author poses in Poser →", per-stage
  screenshots, empty-state copy).
- Test fallout highlights: partition/wiring/palette/mode-keys literals →
  `"create"`; smoke 2d/3d builds → create × stage;
  `test_api`/`test_service`/`test_guidance` `"2d"` hits are the *platform
  taxonomy* value, not the mode — do not touch; new `test_create_stages.py`
  pins gates, lineage, growth-only `STAGES`; new test: no `service` module
  imports `create_stages` (UI stage names must never leak into verdict/corpus
  records — mesh ≠ model on purpose).
- Top risks: pose-stage exits vs an unsaved pose (guard every `go()`);
  selection-moves-with-stage stranding `source_job`/staged verdict tags (`go()`
  is the only mover, reuses `library.select`, moves only when the target stage
  cannot show the current job); stage-enum leakage into the corpus.

## Wave 6 — Headings + polish

- Sentence-case sweep (~60 `section()` literals: `"tools"`→`"Tools"` etc.);
  `field_label` keeps `.upper()` (test-pinned); `pane_title` adopted by the
  Library full view and any full-window pane missing one. No test pins heading
  content; `test_ux_phases.py:103` pins `section`'s *implementation* —
  untouched.
- Final `screenshot_modes.py` pass: all modes × both palettes × 1.0/1.5/1.75
  (+ per-stage create captures, overlays open). Fix what it shows.

## Cross-wave contracts

- **Profiles**: W3 owns the sheet + the **Manage…** entry in
  `settings_2d._profiles`; W5 inherits it automatically (`settings_2d` = the
  Reference stage panel).
- **`modes.py`**: W3 removes manual/profiles + adds `RAIL_GROUPS`; W5 edits
  `MODES` + `RAIL_GROUPS` together (2d, 3d → create) — the rail redraws itself.
- **Home ↔ create**: W4's **New…** menu calls `start_2d`/`start_3d`; W5
  retargets those two entries to `create_stages.go`.
- **Health**: the rail badge (W3) and the Home status line (W4) both read
  `status_rows(ctx)`/runtime checks — one shared source, no forks.
- **`main.py`** is touched by W3 (shell structure), W4 (library dispatch) and
  W5 (create dispatch) — the waves are serial precisely to avoid collisions.

## Verification (every wave)

1. `uv run pytest` full suite green before merge (baseline ~8177/22; never edit
   `src/` mid-run; one suite at a time).
2. `uv run ruff check .`
3. `python scripts/screenshot_modes.py`, both palettes; eyeball at 1.0/1.5
   minimum — the repo's definition of "somebody looked at it".
4. Live app smoke per wave: **W3** — rail collapse/expand at `MIN_SIZE`,
   F1/Esc ordering in Inker, dirty-profile sheet close, quit via the native X;
   **W4** — Inker at 150 % / 1600 w (nothing clips), Library grid keyboard nav;
   **W5** — the full journey prompt→reference→mesh→rig→pose→export on one
   asset, a reroll at reference invalidates nothing it shouldn't, the
   pose-stage exit guard.
5. `uv run pytest -m gpu` before merging W5 (it touches conditioning-adjacent
   promote paths only by routing, but the lane is cheap insurance).

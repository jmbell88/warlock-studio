Warlock Studio — UX/UI Review Report
  
 Context                                                                                                       

 The user asked for a UX/UI review of Warlock Studio with a report of findings. Three parallel exploration passes covered (1) the shared UI
 framework and chrome, (2) navigation/flows/discoverability, and (3) the five workspace modes plus the viewer/pose editor. Findings were
 cross-checked against docs/LEFTOVERS.md (all 43 UX items H67–P124 are recorded as landed; UX.md no longer exists). The headline bugs below
 were verified directly against the source, not just taken from the survey.

 Executive summary

 The codebase is unusually disciplined — the shared kit, motion system, and confirm doctrine are coherent and most findings are gaps
 against the app's own stated standards rather than against generic heuristics. The review found:

 - 7 verified behavior bugs (worst: Plotter's space-to-pan latches on permanently; global shortcuts dead while typing; Clay documents can
 never be closed)
 - ~10 destructive/feedback holes (permanent deletes with no confirm, a frame-thread freeze, invisible armed state in Review)
 - A systemic accessibility layer: toast contrast fails WCAG badly, keyboard focus covers 2 of ~30 panes, two animations ignore Reduce
 motion
 - A systemic 1.5-UI-scale layer: ~20 unscaled pixel literals (known class per LEFTOVERS appendix; these are new instances)
 - Discoverability debt now load-bearing since Alt+digit mode keys were removed: the palette is the only keyboard route to modes, yet it's
 missing many commands and its disabled rows never say why

 A. Verified bugs (checked against source)

 1. Plotter space-to-pan latches forever. plotter_mode.py:657 returns for every non-KEYDOWN event, so the space_held = True set at :680-682
 is never cleared (KEYUP never reaches it; clear_drag doesn't reset it; no other writer). After one space press, every left-drag pans for
 the rest of the session and painting is impossible. Fix: mirror inker_mode.py:1034-1037 — handle both edges before the KEYDOWN filter.
 2. Global shortcuts are dead while any text field has focus. main.py:1368 gates all dispatch on not io.want_text_input, so Ctrl+K, F1,
 F10, Ctrl+Enter don't fire from the 2D prompt box, library filter, or any rename field — contradicting `manual/14-shortcuts.md` ("These
 four work in every mode") and settings_2d.py:996 ("Ctrl+Enter still works from anywhere"). Fix: let modifier-held keys and F-keys
 through; keep the plain-key block.
 3. Clay documents can be opened but never closed, and multiple docs are invisible. ClayState.close (clay_state.py:240) has zero callers
 (verified: only inker/plotter/packwright close callers exist); no Ctrl+W branch in clay_mode._ctrl_key; no tab bar
 (Inker/Plotter/Packwright each draw one). Ctrl+Tab switches docs with no visible indication; dirty-doc quit prompts ask about documents
 the user cannot see. Fix: copy plotter_canvas._tabs + plotter_mode.close_tab pattern.
 4. Inker's symmetry guide ignores the symmetry axis. Verified inker_canvas.py:929-942 draws mirror lines at width/2/height/2
 unconditionally while strokes mirror about state.symmetry_axis (brush.py:105-138) — the guide is wrong exactly when the feature is
 configured. Also a half-pixel default mismatch ((w-1)/2 vs w/2, visible at 32×).
 5. Two toasts pass level "warning" which doesn't exist (verified: landing.py:163, inker_bridge.py:234; TOAST_LEVELS knows warn). Both
 intended warnings render grey, glyph-less, non-sticky, and mouse-pass-through — indistinguishable from "settings copied". Two-word fix;
 add a test asserting literals ∈ TOAST_LEVELS.
 6. Radial symmetry is implemented end-to-end and unreachable. brush.py:42,123-132 implements it, radial_count state and a "Ways" slider
 exist (inker_tools.py:319-325), but SYMMETRY_LABELS (inker_tools.py:45) omits radial, so the slider is dead code.
 7. Review's armed negative sign is invisible. pending_negative (review_mode.py:1228) is never drawn anywhere; press R, glance away, and 4
 files −4 or +4 on hidden state; navigation silently disarms. One warn-coloured legend line fixes it.

 B. Destructive actions & feedback holes

 - Permanent deletes with no confirm: saved pose (pose_panel.py:314-318, sits beside "Save GLB..."), rendered sprite sheet
 (sheet_panel.py:323-330 — costs rows×yaws Blender renders). Both should route through ctx.confirms.ask per the house pattern
 (_review_delete_button).
 - Pose editor has no undo at all; bare "Reset all" (pose_panel.py:185) and joints "Revert" (:261) are unguarded while the preset-apply
 path is guarded; Ctrl+Z no-ops in this mode alone.
 - Inker sheet export freezes the frame thread: export_sheet:399 snapshots (a flatten per frame) before the busy flag at :428 — a 40-frame
 2048² clip is a multi-second dead-looking hang. Use the incremental step_sheet_strip model or flag-then-draw-one-frame.
 - "Fix matte" can silently do nothing (inker_mode._cut_matte:293-308 logs and swallows on the explicit-button path).
 - Clickable lies while busy: Plotter layer controls (plotter_layers.py:58,67,70,72) and Packwright settings (packwright_settings.py:49-80
 — slider moves then snaps back) draw live and discard the click; the anti-pattern is documented and fixed in clay_tools/clay_menu. Fix:
 begin_disabled(tab.busy).
 - No "Saving..." indication in Packwright/Plotter bridges (packwright_bridge.py:42, plotter_bridge.py:43); clay_bridge._facts:62-65 is the
 model. "Render sheet" lacks the spinner its neighbours have (sheet_panel._submit:255).
 - A filed Review verdict auto-advances with no confirmation of what was filed and no visible undo route (record:696-729; the "Recorded"
 line describes the next unit). Recovery exists (Left + re-grade) but nothing says so.
 - Ctrl+Enter Generate silently no-ops on an invalid form from the palette/shortcut path (palette.py:139-145 → settings_2d.py:1104); the
 button path shows problems, the keyboard path shows nothing.
 - Raw str(exc) toasts at 5 sites (packwright_mode.py:229/274, review_mode.py:1027, candidates_panel.py:94, plotter_tools.py:117) vs the
 house style (plain sentence + action="log"); Invalid without field wrapping raw OS errors at settings_2d.py:1129, settings_3d.py:626.
 - Autosave failure surfaces as an unexplained generic red toast (inker_mode._write_autosave:1293 → generic collector message).

 C. Accessibility

 - Toast contrast fails WCAG hard: TEXT on success/warn fills measures 1.82:1 (dark); all six level/palette combos below AA. Fix: fill with
 ELEV_2 (as info already does), spend level colour on glyph + accent bar (widgets.py:1723,1752-1754).
 - text_disabled = MUTED @ 0.6 alpha → 3.19:1 dark / 2.55:1 light (theme.py:142) — paints every hint and disabled label.
 - Toggle knob hardcoded white (widgets.py:1355, the only bare hex in the shared kit) — 1.50:1 on the off-track in light theme; all 16
 switches' off state near-invisible. Same class as the known viewer/env.BACKGROUND_HEX item, unrecorded site.
 - Keyboard focus ring covers 2 panes of ~30 (only settings_2d/settings_3d adopt focus.pump/begin; imgui nav deliberately off). Largest
 single a11y gap; library list + inspector are the highest-traffic next adopters.
 - Two animations bypass motion.py's "honoured here and nowhere else" contract: the indeterminate marquee (widgets.py:360) and spinner
 (widgets.py:377) keep moving under Reduce motion — during the app's longest waits. Gate on motion.REDUCED (drop_flash shows the pattern).
 - Sub-AA badge colours (dark ACCENT/ERR on ELEV_2 ≈ 3.7:1); icon-only controls (compact segmented control at 1.5 scale, icon_button) carry
 meaning solely in hover tooltips.

 D. UI-scale 1.5 (new instances of the known class)

 Unscaled literals: both primary CTAs (-1, 34) (settings_2d.py:992, settings_3d.py:339); (240, 0) buttons (packwright_preview.py:105,
 plotter_canvas.py:106); unscaled set_next_item_width (inker_tools.py:333, plotter_layers.py:179/200, retarget_panel.py:86); unscaled
 dummies (dialogs.py:219 — in the file that documents this exact mistake; settings_2d.py:977; settings_3d.py:320/527); three of four
 workspace empty states unscaled (_clay_empty:2049, plotter_canvas._empty:102, packwright_preview._empty:101 — only Inker's uses sp()). The
 viewport toolbar never wraps — panes/overlay.py has 12 same_line(), 0 same_line_or_wrap; tail controls vanish at 1.5 scale
 (inspector/settings_3d/app_settings/landing same shape, lower risk). Verify any fixes with scripts/screenshot_modes.py at 1.5.

 E. Discoverability & navigation (load-bearing since Alt+digit removal)

 - The shortcuts popup itself has no shortcut and no palette command (main.py:3042, mouse-only ? button).
 - Palette gaps: no New map / New atlas (Home has all six); no Save/Save as/Export/Undo/Redo/Quit/Open log/Show trash/Empty trash;
 quick-open doesn't match tags. Disabled rows never say why (contradicting palette.py:20-24's own docstring); Enter on a greyed row
 silently returns.
 - disabled_button promises a tooltip it doesn't draw — docstring says "a greyed one with a tooltip says why"; takes no reason, 86 call
 sites, 0 explanations. One reason="" param unlocks all of them (_glyph_button already shows the allow_when_disabled pattern). Worst case:
 "Download selected (0)".
 - Five list_filter panes have no "no matches" state (sweeps list, profiles, clay outliner, inker layers, poses) — a filtered-to-empty
 panel looks like a lost panel; widgets.list_filter is the one place to fix all five.
 - A weights refusal rings a control two collapsed folds deep: field errors open "More options" but not the nested "Advanced" header
 holding base_model/style_lora (settings_2d.py:838; request_open("2d/advanced") is never called) — ring lands on an undrawn control.
 Related: field-error rings survive mode switches, and platform names two different controls in 2D vs 3D, so a 2D refusal rings 3D's "Mesh
 resolution".
 - With no weights on disk, Generate is fully enabled — validate never checks model presence though ctx.model_rows answers it; first-run
 users hit a red toast pointing at a hidden control. Add a validate entry worded as a next step.
 - Diagnostics popup names missing weights but has no button to the downloads pane; Home's "Set up models" lands at the top of a
 four-section scroll with no scroll-to.
 - Manual search never sees prose (titles/headings only, render.py:141-148) though parsed blocks are already cached; overlay (the viewport
 toolbar) and clay_menu (the Clay op registry) are exempt from manual coverage despite being primary surfaces; library filter hint names 3
 of 6 query prefixes; unknown prefixes silently degrade to substring search.
 - Cancel is unreachable from Home: main.py:2864 suppresses the progress card on Home, whose own queue row has no Cancel/ETA.
 - A pending candidate group is announced nowhere outside 2D/3D (picker only in the inspector; rows hidden from library) — three meshes
 land while you're in Inker and nothing states a choice is pending.
 - The prompt modal silently refuses an empty name (dialogs.py:333-341 — Save does nothing, no message, modal sits there).

 F. Consistency

 - Redo: Ctrl+Shift+Z accepted in Inker/Clay only, Ctrl+Y-only in Plotter/Packwright. Ctrl+W closes docs in 3 of 4 editors (see A3). Ctrl+E
 export-to-library in 3 modes, absent in Inker (where Ctrl+Shift+E means Export PNG but Export .tmx/atlas elsewhere). Ctrl+1 = 100% zoom
 in four workspaces, front-view in Clay (defensible; needs a cue).
 - Undo/redo has on-screen controls only in Inker; Clay/Plotter/Packwright have none.
 - Dirty markers: Inker uses TabItemFlags_.unsaved_document (with a rationale comment); Plotter/Packwright prepend "* " to the title — the
 exact thing the comment argues against.
 - 16 widgets.toggle vs 22 raw imgui.checkbox (mixed within single panes); 20 raw sliders vs labeled_slider_* (app_settings.py:64
 hand-rolls the very slider the helper was written for); raw collapsing_header at 2 sites loses persist_key (section forgets open state);
 hand-rolled problem lines in both settings panes (third error register, no glyph); shortcuts button is ASCII "?" while help_marker
 migrated to icons.INFO; single-pane modes: only app_settings draws pane_title though Library/Profiles are now the same shape.
 - Card-vs-palette delete disagree: card trashes with no confirm (deliberate — trash is the confirm), palette's identical command raises a
 Confirm and its label promises a dialog.
 - Silent input truncation: input_text/multiline clamp out[:max_length] with no notice.
 - Shortcuts popup omissions: Inker animation keys entirely absent; Clay axis views absent; Plotter row advertises the broken space-pan;
 Packwright documents no pan.
 - Stale content: comments at widgets.py:317, inspector.py:942, review_mode.py:1118 still quote the retired AUC 0.115 "inverted" figure
 (LEFTOVERS §2 re-baseline: 0.756, corpus-dependent — behaviour stays, comments should update); widgets.py:324 cites TODO.md §2 which is
 now docs/LEFTOVERS.md.
 - hole_worst doctrine hole: review_mode.mesh_lines:1120 prints "see-through at worst view: X%" with no caveat at any value — Review is
 where corpus judgements are filed; mirror the inspector's "(a solid, featureless mesh scores this too)" line gated on AUDIT_UNINFORMATIVE.
 Also three different spellings of this one number across badge/inspector/review.

 G. Engine capabilities with no UI

 - Plotter tile flips (H/V/D) round-trip from Tiled, render, and draw correctly — but no UI can ever set one. Still open, deliberately: the terrain generator emits all 47 cases unflipped rather than 15-plus-flips, so the flag path stays import/export-only and the atlas stays readable in Tiled and Inker alike. Closing this by making the generator emit flipped variants would couple a new generator to the one code path with no UI and no user-visible test — the wrong two things to land together.
 - Inker "Animate" is a one-way door: drop_animation exists with no UI caller; merge/flatten permanently unavailable once animated (refusal
 is documented in LEFTOVERS §16; the missing exit is not). Fix: "Flatten to a still drawing" behind a confirm naming the loss.
 - planar_unwrap exists in clay/uv.py; only Box Unwrap is registered.
 - Clay's parameterised ops (bevel/inset/loop-cut/weld/subdivide) apply blind; inker_bridge._filter_popup is the in-house live-preview
 model.
 - Review grade/tag buttons carry no inline key hints while the label pass next door does ("Good (A)…"); units list has no filter while the
 sweep list beside it has one; Plotter's status bar lacks cursor cell + active tool (the two most useful numbers in a tilemap editor).
 - Splash: fixed 3 s floor on ~1 s warm starts; single unwrapped message line can run off both edges.

 Already-known backlog (not re-filed)

 LEFTOVERS §15: K93 tooltip floor, M106 sidebar presets, Ink9 quarter-turns, Clay21 no vertex extrude, viewer/env.BACKGROUND_HEX dark-only
 viewports. §16: animated merge/flatten refusal, cel thumbnails. §2: quality-badge green branch stays out. §14.1: box-unwrap island
 overlap. Appendix: the UI-scale-1.0 smoke-suite class (section D above are new instances of it).

 Recommended priority (if/when fixes are commissioned)

 1. One-line bugs on the primary feedback/input paths: A1 (plotter pan latch), A5 ("warning" typo ×2), A2 (shortcut gate on text input).
 2. Data-safety: B confirms on pose/sheet delete; pose-editor Reset/Revert guards.
 3. Review loop integrity: A7 (armed sign), verdict-filed feedback, inline key hints, mesh_lines caveat — the findings corpus depends on
 this loop.
 4. Clay tab bar + Ctrl+W (A3) — largest dead end, mostly copied from Plotter.
 5. disabled_button(reason=) signature change — one change, 86 sites become explainable; same pattern fixes palette's greyed rows and the
 empty-name modal.
 6. Accessibility batch: toast contrast, text_disabled, toggle knob, marquee/spinner reduce-motion, then focus-ring adoption in
 library/inspector.
 7. Scale-1.5 sweep (section D) + same_line_or_wrap in overlay.py, verified by scripts/screenshot_modes.py at 1.5.
 8. Discoverability batch: palette command parity, shortcuts-popup corrections, weights-aware validate, diagnostics→downloads link.
 9. Consistency batch: redo spelling, Ctrl+E, undo buttons in three bridges, dirty markers, toggle/checkbox unification, stale comments.

 Verification

 - This engagement's deliverable is the report itself; no code changes yet.
 - If fixes proceed: uv run pytest (full suite currently 4843/2-skipped), uv run ruff check ., and scripts/screenshot_modes.py at UI scale
 1.5 for anything in sections C/D; new toast-level literals pinned by a TOAST_LEVELS membership test.
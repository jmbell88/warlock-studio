# Pick up here — REDESIGN.md wave 6

## The prompt to open the next session with

> Continue executing `REDESIGN.md`. Waves 1–5 are merged to master; start wave
> 6 (headings + polish). It is the sentence-case sweep over ~60 `section()`
> literals plus the final screenshot pass at 1.0 / 1.5 / 1.75 in both palettes
> — `--asset` and `--seed` and `--overlays`, per-stage Create captures — and
> fixing what it shows. One branch, a green full suite before merging, and read
> `NEXT_SESSION.md` first.

## Where things stand

`REDESIGN.md` (repo root) is the six-wave plan. **Waves 1, 2, 3, 4 and 5 are
done and merged to `master`** — committed, not pushed.

| Wave | Commit | What landed |
| --- | --- | --- |
| 1 Tokens + theme | `68dee25` | RADIUS_M 6→8, RADIUS_L 10→12, type 11/13→12/14, `child_border_size`→0, hover moved off the accent onto EDGE, `tokens.DISABLED_ALPHA` |
| 2 Widget layer | `7b2b1b9` | `studio/toolbar.py` (pure `plan()` + overflow menu), `widgets.ghost_button`, borderless glyph buttons, neutral disabled primary, `widgets.nothing_open`, percent/format sliders |
| 3 Shell | `edbadac`, `f56f5df`, `2108924` | Manual → overlay, Profiles → sheet, `studio/rail.py`, `_mode_switch` + `modes.QUIT` deleted, `pygame.QUIT` → `_ask_quit` |
| 4 Panes | `1ab5037` … `3741c60` | Settings categories, four toolbars onto `toolbar()`, Home as one column, `panes/library_full.py` |
| 5.0 Stage model | `4809c99` | `studio/create_stages.py` — `STAGES`, `reached`, `shows`, `available`, `go`; imgui-free, nothing drawn |
| 5.1 Rail widget | `d3a30e9` | `widgets.stage_rail`, `AppState.create_stage`; both imgui fixtures run with no ini file |
| 5.2a Indirection | `e5026b4` | Every `mode == "2d"/"3d"` read → `create_stages.at`/`in_create`/`stage_for`; bodies unchanged |
| 5.2b **THE FLIP** | `a012dc6` | `2d` + `3d` → one `create` mode; rail drawn, dispatch on stage, `_sync_viewer` watches the stage, every `set_mode` call site → `go` |
| 5.3 Rig & Pose | `ad56d80` | `panes/stage_rig.py`, `pose_panel` re-hosted, inspector drops its tab bar in Create |
| 5.4 Export | `b920b43` | `inspector.downloads` public, drawn as the Export stage's column |
| 5.5 Flow sugar | `1f021cc`, `96af462` | Lineage links, "Author poses in Poser →", `widgets.fit_text`, `--asset` screenshots, the rail's fit ladder |

Suite: **8288 passed / 22 skipped**, ~6.3 min (`uv run pytest -q`). Ruff clean.
`uv run pytest -m gpu`: **18 passed**. Screenshots eyeballed in both palettes at
1.0 and 1.5, with `--overlays`, `--seed` and `--asset`.

## Start of wave 6

Read `REDESIGN.md` §"Wave 6 — Headings + polish". Two things:

1. The sentence-case sweep (~60 `section()` literals: `"tools"` → `"Tools"`).
   `field_label` keeps its `.upper()` — that is test-pinned — and
   `test_ux_phases.py:103` pins `section`'s *implementation*, not its
   arguments. No test pins heading content.
2. The final screenshot pass: all modes × both palettes × 1.0 / 1.5 / **1.75**,
   overlays open, per-stage Create captures. Fix what it shows.

`pane_title` should also be adopted by any full-window pane missing one — the
Library full view is the one the plan names.

## Things learned in wave 5 that the plan does not say

- **imgui writes `imgui.ini` into the working directory and reads it back next
  process.** A synthetic click that lands on a title bar twice inside the
  double-click window collapses the shared host window, and a collapsed window
  submits no items at all — so eighteen tests in files nobody had touched
  failed, reproducing from a gitignored artefact. Both smoke fixtures now call
  `io.set_ini_filename(None)`. Any new imgui fixture must too.
- **The screenshot harness was lying by omission again.** With nothing selected
  four of Create's five stages draw an empty state, so the Rig column, the Pose
  column, the export grid, the lineage links and a ticked rail had never been in
  a capture. `--asset` fixes it, and the first picture it took showed a real
  defect (below). Wave 4 learned the same lesson about `--seed`; assume the next
  new surface needs its own seeder.
- **`stage_rail` fits by a ladder of three**, each rung all-or-nothing: labels
  with ticks, labels alone, icons with tooltips. Five labels plus two ticks do
  not fit a 300 dp sidebar at 150 %, and the two-rung version traded every word
  on the rail for two check glyphs. In practice the ticks only show at 1.0, or
  in the "wide" sidebar.
- **`create_stage` is not cleared when the mode changes**, deliberately — coming
  back from Inker lands where you left. So `create_stages.at()` asks the mode
  *and* the stage; a pane that asked only the stage would grow Create's sections
  in Poser.
- **`reached()` returns None with nothing selected.** Standing at a stage and
  having completed it are different facts, and the rail draws its ticks off the
  second. Export is never reached at all: `save_artifact` copies a file
  somewhere the job row never hears about.
- **Job stages and UI stages are different vocabularies on purpose.** The corpus
  is keyed on `verdicts.STAGES` (`model`, `reference`, `blank`); the rail says
  `mesh`. `test_create_stages.py` pins that no `service` module imports
  `create_stages`.
- **`settings_2d.FOCUS_PANE` is still `"2d"` and `settings_3d`'s is `"3d"`.**
  The wave re-routed *around* the repo's densest test surface rather than
  through it; those keys, the `2d/reference` persist key and the `3d-source`
  drop slot are not mode names and were left alone.
- **`tests/test_studio_frame.py::_viewer_app` takes a `stage=` now**, and
  `tests/test_palette.py::_ctx` takes `stage=`. Any new fake state needs
  `create_stage` or `at()` raises `AttributeError`.
- **Flaky test**, third session running:
  `tests/test_panes_mtime_guard.py::test_a_missing_palette_directory_is_no_palettes_and_no_error`.
  It did **not** fail in any of the six full runs this session. Still worth ten
  minutes with `-p no:randomly` if it reappears.
- **`plotter-wave-2` is still an unmerged branch.** Wave 5 touched `plotter_*`
  not at all.

## Deliberate deviations from the plan, wave 5

1. **`go()` gained a `follow=False`.** The plan has `go` move the selection
   whenever the target stage cannot show the current job. That is wrong for the
   two arrivals that are about to *make* something — a dropped image and a
   promotion both carry their own source, and walking onto a mesh the current
   reference already has would describe the wrong asset while the form builds
   another.
2. **`reached()` takes three pieces of evidence, not two.** The plan's
   `reached(job, rig_meta)` cannot answer the pose stage: poses are files under
   `<job>/poses/` and nothing of theirs is in `files.LISTED`. The third argument
   is `state.preview["poses"]`, which `_refresh_rig_side_data` already fetches.
3. **The inspector keeps its tab bar outside Create**, and which tabs it shows
   is now a question about the *asset* rather than about the mode — the old
   gate offered Rig & Pose for a reference selected in 3D and withheld it for a
   mesh selected in the Library.
4. **The Export stage's inspector is `_reference` + `_settings`.** The plan
   maps Export to the `_downloads` grid and says nothing about the other
   column; the grid is the stage's own column, and the inspector answers the
   question a person actually has in front of it.
5. **`--asset` and the fit ladder are wave-5 work, not 5.5 sugar.** The plan
   lists per-stage screenshots under 5.5; the harness change is what found the
   rail's fit defect, so both landed together.

## Still owed, and named rather than buried

- **The manual was swept** (eleven chapters: every "the 2D pane" / "the 3D
  pane" / "in 3D mode" / "the 3D inspector", the mode list in chapter 01, the
  "2D and 3D" shortcut table, and the four "Rig & Pose tab" references from
  5.3). What has *not* been re-read end to end is whether any chapter's
  argument still assumes two destinations rather than one; the words are right,
  the shape of the prose was not re-examined. Chapter 01's mode list and
  chapter 14's tables are the two most likely to want it.
- **No live-app journey was driven.** The wave's verification asks for
  prompt→reference→mesh→rig→pose→export on one asset in the running app; this
  machine has no Blender, so the rig and pose halves refuse by design and the
  mesh half is two minutes of GPU per attempt. What was done instead: the full
  suite, the gpu lane, and a screenshot pass with a seeded rigged asset at both
  palettes and both scales.

## Verification per wave (unchanged)

1. `uv run pytest -q` — green before merge, and never edit `src/` while it runs.
2. `uv run ruff check .`
3. `uv run python scripts/screenshot_modes.py --out <dir> --scale 1.0` and
   `--scale 1.5`, both palettes; `--overlays` for the manual, the profile sheet
   and the expanded rail; `--seed` to populate Inker and Clay; `--asset` to
   populate Create's five stages. Point `WARLOCK_DATA_DIR` at a throwaway
   directory.
4. Live-app smoke for the wave, per `REDESIGN.md` §Verification.

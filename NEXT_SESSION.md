# Pick up here — REDESIGN.md wave 4

## The prompt to open the next session with

> Continue executing `REDESIGN.md`. Waves 1–3 are merged to master; start wave 4
> (Panes). Follow the plan's own order — 4.1 Settings, 4.2 toolbars, 4.3 Home,
> 4.4 the Library split view — one branch, a green full suite and a screenshot
> pass before merging, and read `NEXT_SESSION.md` first.

## Where things stand

`REDESIGN.md` (repo root) is the six-wave plan. **Waves 1, 2 and 3 are done and
merged to `master`** — committed, not pushed.

| Wave | Commit | What landed |
| --- | --- | --- |
| 1 Tokens + theme | `68dee25` | RADIUS_M 6→8, RADIUS_L 10→12, type 11/13→12/14, `child_border_size`→0, hover moved off the accent onto EDGE, `tokens.DISABLED_ALPHA` |
| 2 Widget layer | `7b2b1b9` | `studio/toolbar.py` (pure `plan()` + overflow menu), `widgets.ghost_button`, borderless glyph buttons, neutral disabled primary, `widgets.nothing_open`, percent/format sliders |
| 3 Shell | `edbadac`, `f56f5df`, `2108924` | Manual → overlay, Profiles → sheet, `studio/rail.py`, `_mode_switch` + `modes.QUIT` deleted, `pygame.QUIT` → `_ask_quit` |

Suite: **8215 passed / 22 skipped**, ~6 min (`uv run pytest -q`). Ruff clean.
Screenshots eyeballed in both palettes at 1.0/1.5 (and 1.75 for wave 1).

## Start of wave 4

Everything wave 4 needs already exists. In particular **`studio/toolbar.py` has
no callers yet** — wave 2 built it and wave 4 is what adopts it. The defect it
exists for is still visible: capture `dark-inker.png` at scale 1.5 and the file
row loses "Export PNG" off the right edge.

Read `REDESIGN.md` §"Wave 4 — Panes" for the detail. In the plan's order:

1. **4.1 Settings** (`panes/app_settings.py`) — centred `min(avail, sp(640))`
   surface, `segmented_control("settings-cat")` categories, Storage gains the
   maintenance actions moved out of the library footer.
2. **4.2 Toolbars** — Inker file actions to `inker_bridge`, `_file_row`,
   `_transform_row`, `inker_timeline._transport` (the 17-`same_line` worst
   case), `library._bulk`, all onto `toolbar.toolbar()`.
3. **4.3 Home** (`panes/landing.py`) — one column, dismissible What's New, one
   **New…** menu, Resume as a thumbnail grid, new `panes/thumbs.py`.
4. **4.4 Library** — new `panes/library_full.py` composing the split view;
   `library.py` keeps its API for the sidebar.

## Things learned in waves 1–3 that the plan does not say

- **The plan's `segmented_control` fit test is gone from the mode switch** and
  now covers the *Settings categories* shape (`test_studio_smoke.py`:
  `test_a_segmented_control_takes_its_compact_labelling_rather_than_clipping`).
  4.1 is what makes that test cover a real call site again.
- **`icons.py` is a vendored lucide subset.** There is no `CHEVRON_LEFT`, no
  `ARROW_RIGHT`, no `MENU`. Check the name exists before using it — a missing
  glyph renders as a blank square, and the rail hit this. Wave 4's timeline
  transport needs play/skip glyphs: `PLAY` exists, skip/step do not.
- **`imgui.calc_text_size`'s second positional is `text_end`, not the
  hide-after-`##` flag** — pass `(text, None, False, wrap)` for a wrapped
  measure.
- **A rail item's popup must be opened at host scope.** `rail.request(name)` /
  `rail.take(name)` is the one-shot; `main._build_ui` opens the popup after
  `end_child`. Any new footer popup follows that pattern.
- **`layout.RAIL_RESERVED`** is set by `rail.tick` before `layout.tick`. Any
  new column measured from the window must subtract it.
- **Flaky test**: `tests/test_panes_mtime_guard.py::test_a_missing_palette_directory_is_no_palettes_and_no_error`
  failed once in a full run and passed alone and on a full re-run. Re-run before
  investigating.
- **`plotter-wave-2` is still an unmerged branch** touching plotter panes. Wave 4
  should keep touching `plotter_tools.py` to a minimum (the plan says the same).

## Verification per wave (unchanged)

1. `uv run pytest -q` — green before merge, and never edit `src/` while it runs.
2. `uv run ruff check .`
3. `uv run python scripts/screenshot_modes.py --out <dir> --scale 1.0` and
   `--scale 1.5`, both palettes; `--overlays` for the manual, the profile sheet
   and the expanded rail; `--seed` to populate Inker and Clay. Point
   `WARLOCK_DATA_DIR` at a throwaway directory.
4. Live-app smoke for the wave, per `REDESIGN.md` §Verification.

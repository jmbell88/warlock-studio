# Pick up here — REDESIGN.md wave 5

## The prompt to open the next session with

> Continue executing `REDESIGN.md`. Waves 1–4 are merged to master; start wave 5
> (Create mode — the 2D/3D merge). Follow the plan's own step order — 5.0 pure
> `create_stages`, 5.1 the `stage_rail` widget, 5.2a the helper indirection,
> 5.2b THE FLIP as one atomic commit, then 5.3/5.4/5.5 — one branch, a green
> full suite and a screenshot pass before merging, `uv run pytest -m gpu` before
> the merge, and read `NEXT_SESSION.md` first.

## Where things stand

`REDESIGN.md` (repo root) is the six-wave plan. **Waves 1, 2, 3 and 4 are done
and merged to `master`** — committed, not pushed.

| Wave | Commit | What landed |
| --- | --- | --- |
| 1 Tokens + theme | `68dee25` | RADIUS_M 6→8, RADIUS_L 10→12, type 11/13→12/14, `child_border_size`→0, hover moved off the accent onto EDGE, `tokens.DISABLED_ALPHA` |
| 2 Widget layer | `7b2b1b9` | `studio/toolbar.py` (pure `plan()` + overflow menu), `widgets.ghost_button`, borderless glyph buttons, neutral disabled primary, `widgets.nothing_open`, percent/format sliders |
| 3 Shell | `edbadac`, `f56f5df`, `2108924` | Manual → overlay, Profiles → sheet, `studio/rail.py`, `_mode_switch` + `modes.QUIT` deleted, `pygame.QUIT` → `_ask_quit` |
| 4.1 Settings | `1ab5037` | Centred `CONTENT_W` column, four categories behind `segmented_control`, Storage gains the maintenance actions |
| 4.2 Toolbars | `1ffc49b`, `c8504ba` | Inker file actions → `inker_bridge`; canvas row, transform row, timeline transport and `library._bulk` onto `toolbar()`; `toolbar.Item.primary` |
| 4.3 Home | `cf003a6` | One column, dismissible What's New, one **New…** menu, Resume as a thumbnail grid, new `panes/thumbs.py` |
| 4.4 Library | `1593ef2`, `d63842d` | New `panes/library_full.py` (rail + grid + inspector), `library.select_grid`, `--seed` now animates the Inker canvas |

Suite: **8228 passed / 22 skipped**, ~6.5 min (`uv run pytest -q`). Ruff clean.
Screenshots eyeballed in both palettes at 1.0 and 1.5, with `--overlays`.

## Start of wave 5

Read `REDESIGN.md` §"Wave 5 — Create mode" for the detail; its step list
(5.0 → 5.5) is the commit plan and 5.2b is deliberately one atomic commit.

Everything wave 5 needs from earlier waves exists: the rail is data-driven
(`modes.MODES` + `RAIL_GROUPS`), the Profiles **Manage…** entry is in
`settings_2d._profiles`, and Home's **New…** menu is a table — `landing.NEW_ITEMS`
— whose first two entries are the two `start_*` functions wave 5 retargets at
`create_stages.go`.

## Things learned in wave 4 that the plan does not say

- **`widgets.card` is a *borderless* child**, so imgui gives it zero window
  padding and everything drawn in one sits flush against its left edge. Home's
  cells and the library's open a gutter by hand (`landing._card_margin`:
  a leading `dummy` then `indent`/`unindent`). Any new card does the same.
- **A grid cell must fit its label by measurement, not by a character count.**
  22 characters fit a 136 dp cell at 1.0 and run a third past it at 1.5, and a
  card has no scrollbar to absorb it. `landing._fit` / `library_full._fit`.
- **`toolbar` wraps its trailing block** (`same_line_or_wrap`) because `plan()`
  can legitimately return a row that does not fit — an all-pinned row has
  nothing left to give. Before that the timeline's frame-duration box was drawn
  past the strip's edge at 150 %.
- **`input_int` with a step draws its own -/+ buttons *inside* `CalcItemWidth`**,
  which is ~110 px of a 128 dp field at 1.5. Any hand-computed reservation for
  one has to include them.
- **`widgets.ring` takes `ImVec2`, not tuples** — it reads `.x`/`.y`.
- **`pane_title` ends in a spacer**, so a right-aligned `help_button` called
  after it floats alone on an empty row. Put the (?) on the first *real* row.
- **`scripts/screenshot_modes.py --seed` now animates the canvas.** The timeline
  strip only draws for an animated document, so before this the app's densest
  row had never been in a capture. Two real defects showed the first time it
  was.
- **`tests/manual/test_coverage.py` fails on any new file in `panes/`** with no
  `help_button` in it. `thumbs.py` is listed there with its reason; a new
  non-pane helper in that directory needs the same entry.
- **Flaky test**: `tests/test_panes_mtime_guard.py::test_a_missing_palette_directory_is_no_palettes_and_no_error`
  failed once in a full run again this session (it also failed once last
  session) and passes alone and on a re-run. Nothing in wave 4 touches it. It is
  worth ten minutes with `-p no:randomly` and a `--tb=long` full run next time
  rather than another shrug.
- **`plotter-wave-2` is still an unmerged branch** touching plotter panes. Wave
  4 touched `plotter_*` not at all, which is what the plan asked for.

## Deliberate deviations from the plan, wave 4

1. **The Inker `new-canvas` popup is registered by two panes.** A popup belongs
   to the window that begins it, so the bridge's **New** cannot open one
   registered in the canvas. `inker_canvas.new_popup` is public and both panes
   call it; the body is shared, the registration is not.
2. **The timeline is two toolbars, not one row with three priority groups.**
   The plan's single row would have put ~700 px of non-collapsible trailing
   (counter, duration box, two toggles, the scale combo, the (?)) on a strip
   that is ~540 px wide at 150 %. Split by what the controls are *about*: the
   frame you are on, and what leaves the app.
3. **The transport's steps stay ASCII.** `icons.py` is a transcription of
   lucide-static 0.525.0's codepoint table and its docstring forbids guessing
   one; the vendored subset has `play` and `square` and has no skip-back,
   skip-forward or chevron-left. The plan anticipated this fallback.
4. **`library.select_grid` is a second entry point rather than a flag on
   `select_relative`.** The same function is bound to Up/Down in the *sidebar*,
   where a row is one card — which of the two a key means is a property of the
   pane it was pressed in, so the pane's dispatcher chooses.
5. **The Settings (?) sits on the category switch's line, not under the title**
   (see `pane_title` above), and the section headings that merely repeat their
   own category are gone.

## Verification per wave (unchanged)

1. `uv run pytest -q` — green before merge, and never edit `src/` while it runs.
2. `uv run ruff check .`
3. `uv run python scripts/screenshot_modes.py --out <dir> --scale 1.0` and
   `--scale 1.5`, both palettes; `--overlays` for the manual, the profile sheet
   and the expanded rail; `--seed` to populate Inker and Clay. Point
   `WARLOCK_DATA_DIR` at a throwaway directory.
4. Live-app smoke for the wave, per `REDESIGN.md` §Verification.
5. **Wave 5 only**: `uv run pytest -m gpu` before merging.

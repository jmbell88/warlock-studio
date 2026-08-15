# REDESIGN.md is finished

All six waves of `REDESIGN.md` are done and merged to `master` — committed, not
pushed. There is no wave 7 and no pickup task; this file is a record now, not a
queue.

| Wave | Commit | What landed |
| --- | --- | --- |
| 1 Tokens + theme | `68dee25` | RADIUS_M 6→8, RADIUS_L 10→12, type 11/13→12/14, `child_border_size`→0, hover moved off the accent onto EDGE, `tokens.DISABLED_ALPHA` |
| 2 Widget layer | `7b2b1b9` | `studio/toolbar.py` (pure `plan()` + overflow menu), `widgets.ghost_button`, borderless glyph buttons, neutral disabled primary, `widgets.nothing_open`, percent/format sliders |
| 3 Shell | `edbadac`, `f56f5df`, `2108924` | Manual → overlay, Profiles → sheet, `studio/rail.py`, `_mode_switch` + `modes.QUIT` deleted, `pygame.QUIT` → `_ask_quit` |
| 4 Panes | `1ab5037` … `3741c60` | Settings categories, four toolbars onto `toolbar()`, Home as one column, `panes/library_full.py` |
| 5 Create | `4809c99` … `d3e91b0` | `studio/create_stages.py`, `widgets.stage_rail`, **the flip** (`2d` + `3d` → one `create` mode), `panes/stage_rig.py`, `inspector.downloads`, lineage links |
| 6 Headings + polish | *this wave* | The sentence-case sweep, `pane_title` on the Library, the final 1.0 / 1.5 / 1.75 screenshot pass and what it found |

Suite: **8292 passed / 22 skipped**, ~6.4 min (`uv run pytest -q`). Ruff clean.
`uv run pytest -m gpu`: **18 passed**. Screenshots eyeballed in both palettes at
1.0, 1.5 and 1.75 with `--seed --asset --review --overlays --floating`.

## What wave 6 actually did

**The sweep.** 48 `section()` literals sentence-cased across 20 files, plus
`palette.py`'s `heading.lower()` (a leftover of the older all-lowercase heading
widget) and `library.date_group`'s four return values, which are headings and
not data. `field_label` keeps its `.upper()`. `tests/test_headings.py` pins the
rule by AST walk so the sixty-first call site cannot be lowercase, and pins the
Library's `pane_title` — the one full-window pane that was named only by the
navigation chrome outside it.

**The screenshot pass found six real defects**, all fixed here:

1. `widgets.cost_note` called `muted`, which does not wrap. Every cost sentence
   in a 300 dp column was cut at the pane edge with no scrollbar to the rest —
   the Rig stage's read "Rigging is queued like a generation: it runs Blender
   in a". It calls `muted_wrapped` now, which exists for exactly this.
2. The library card's "| from a reference" was the fourth item on a badge row
   and drew as "| from a refere". It measures first now, and drops to its own
   line as "From a reference" — without the leading pipe, which is a separator
   and not a word.
3. The inspector's tab read **"Rig && Pose"**. Dear ImGui has no mnemonic
   escape; the doubling was a habit from toolkits that do.
4. The keyboard-shortcut overlay still had a **"2D / 3D"** table, one wave
   after those two modes stopped existing. It is "Create" now, and Clay's tool
   row no longer starts lowercase.
5. Three buttons and a refusal still said **"Send to 3D"** / "Open it in 3D
   mode" — Clay's bridge, Inker's bridge, the rigged-GLB import refusal. They
   say "Make 3D" and "Open it in Create", matching the Mesh stage's own button.
   Two manual chapters follow.
6. The profile sheet's `help_marker` was called before any control, so the (?)
   drew alone on an empty row between the heading and **New profile**. It is on
   the button now. `tag_toggles`' "GOOD:"/"BAD:" also lost the colon no other
   field label wears.

**The harness was lying twice more.** It drew every capture under a
"Recover unsaved work?" modal, because the journal lives in the user's home and
not under `WARLOCK_DATA_DIR` — `state.recovery_offered` is set before the first
frame now, so the offer is never made rather than dismissed a frame late. And
`--seed` opened a canvas and a model but no map and no atlas, so Plotter's and
Packwright's eight panes had never been photographed at any scale; both
`new_document` calls are synchronous and joined it. `--overlays` also captures
the shortcut popup now, which is how defect 4 was found.

**`docs/INVARIANTS.md` was two waves out of date** — beyond wave 6 as written,
and done because the file is the repo's authoritative record and it still
described eleven modes including `2d` and `3d`. Sixteen edits: the mode list and
count, `WORK_MODES`/`VIEWPORT_MODES`, the rail-grouping argument, the two
Review sentences that route "back into 3D", the deleted `mode_for_digit`
(claimed live in one paragraph and deleted in the next), Home's chooser tiles
(gone in wave 4), and a new paragraph stating Create's stage invariants: a stage
is a derived position, `reached()` takes three pieces of evidence, `at()` asks
the mode *and* the stage, `go(follow=False)`, and job stages ≠ UI stages.

## Things learned, for whoever is next

- **A `same_line` past the content edge clips silently**, and the repo has had
  `widgets.same_line_or_wrap` for it since UX Phase 4. Two of this wave's six
  defects were call sites that predate it. When adding an nth item to a row in
  a 300 dp column, measure.
- **The screenshot harness only shows what something seeded.** That is now four
  waves running: `--seed` (wave 4), `--review`, `--asset` (wave 5), and the map
  and atlas here. A new surface needs its own seeder in the same commit, or its
  first picture is an empty state.
- **`imgui.internal.close_popups_except_modals()`** is how a popup is closed
  from outside its own begin/end — `close_current_popup` is only legal inside.
  The `--overlays` shortcut capture needs it, or the light pass draws every
  mode under the dark pass's popup.
- **imgui writes `imgui.ini` into the working directory** (wave 5's lesson,
  still true): any new imgui fixture must call `io.set_ini_filename(None)`.
- **Flaky test, fourth session running:**
  `tests/test_panes_mtime_guard.py::test_a_missing_palette_directory_is_no_palettes_and_no_error`.
  It did not fail in any run this session either.
- **`plotter-wave-2` is still an unmerged branch.** No redesign wave touched
  `plotter_*` beyond the eleven headings swept here.

## Still owed

- **No live-app journey was driven**, in wave 5 or here. This machine has no
  Blender, so the rig and pose halves of prompt→reference→mesh→rig→pose→export
  refuse by design and the mesh half is two minutes of GPU per attempt. What
  stands in for it: the full suite, the gpu lane, and 120 screenshots across
  three scales and both palettes.
- **The manual's prose was re-read for the two-destination question** and is
  clean — chapter 01's mode list and window description both describe one
  Create mode with five stages, and chapter 14's Create table now matches the
  popup row for row. Chapter 13's two "the 2D and 3D forms" phrases were the
  last of it.

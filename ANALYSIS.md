# QA Audit — Warlock Studio

**Date:** 2026-08-07 · **Base:** `master` @ `4fc9927` (Warlock v0.0.10) plus the uncommitted
working tree (Clay object merge, library select-all, 2D form reset, `art_style` → "era style").

**Baseline before the audit:** `uv run pytest` → 3091 passed, 7 skipped. `uv run ruff check .` → clean.
**After the audit and the follow-up:** 3110 passed, 7 skipped, lint clean. Nineteen new tests, each
verified to fail before its fix and pass after.

The audit ran in two passes. The first fixed seven layout defects and left five items that needed a
decision rather than a patch; the second, after those decisions, closed all five. Both are recorded
here — *Fixed* is the first pass, *Resolved after review* is the second.

---

## Headline

The uncommitted work is sound. Every defect the first pass fixed is a **layout** bug of one shape,
and it is worth naming because it recurred seven times and none of it raises anything:

> `imgui.same_line()` after an item drawn at full width leaves the cursor **on** the content
> region's right edge, and an imgui child window *clips* rather than wraps. Whatever is drawn next
> is not squeezed — it is gone.

Four controls were living outside the panel when I started: the library's **favourites star**, the
new **select-all tick**, the **Prune** button, and every **evidence hint** and **(?)** in the two
generate panes. Nothing logged, nothing threw, and each one looked exactly like a feature nobody
had built. Notably, the favourites feature was unusable end to end — the filter star and the
per-card toggle were both off-panel.

The second pass then took the five deferred items. The one that mattered most was not a layout bug
at all: **`CLAUDE.md` asserted something about the filesystem that is false on Windows**, and the
job-list cache built on it could miss a new artifact permanently. That is measured, fixed and
written down under *Resolved after review*.

---

## Fixed

### 1. Clay op dialogs printed sub-millimetre parameters as `0.000`

`panes/clay_menu.py` drew every numeric parameter with `imgui.input_float(label, value, step)`,
whose implicit format is `"%.3f"`. **Both** weld distances in the registry default to `1e-4` — the
pre-existing **Weld** op and the new **Merge Objects...** — so the field read `0.000`, the step
arrows moved it by an amount no digit shown could express, and a user who typed into it got back a
different number than the one it was showing. The manual tells people to set this field.

Added `clay_ops.format_for(param)`, which widens the format *downwards* from the step and the
default, and is derived rather than declared: a `format` field on `Param` would be one more thing
to remember and the next sub-millimetre parameter would arrive without it. `%.4f` for the two weld
distances; every parameter that was legible at three decimals is unchanged.

*Tests:* `test_clay_ops.py` — the specific case, plus the whole-registry property (what a field
shows must round-trip to what the op is given).

### 2. The library filter row ran 127 px past a 300 px sidebar

Measured, not inferred: three 110 px combos plus two square buttons = **417 px** into a **290 px**
content region. The favourites star was ~94 px off the edge *before* this working tree (it has
apparently never been clickable), and the new select-all tick landed beside it in the dark.

Reflowed onto two rows, with widths taken from the live content region — the constant was the bug,
since `sp()` scales the sidebar with the monitor and `110` scaled with nothing.

### 3. A library card's action row started on the status pill's line

`_card_body` did `status_pill(...)` → `same_line()` → `quality_badge(job)`. `quality_badge` draws
**nothing** for a job with neither `mesh_report` nor `mesh_audit` — which is every reference — and a
`same_line` in front of a call that draws nothing is not spent, it is **inherited by whatever comes
next**. That was the entire action row: pulled up onto the pill's line and 73 px to the right,
leaving it 113 px of budget for a row that needs 138–180. The favourite star ended **25 px past the
card's edge**.

Fixed at the source: `quality_badge(job, inline=True)` issues its own `same_line`, only in the
branches where it actually draws. The inspector's call site is unaffected. The row now finishes
48 px inside the card.

### 4. Every evidence hint and (?) marker in both generate panes was off-panel

The findings hints are the whole visible payoff of the observation corpus. Both panes drew them
with a bare `same_line()` after a combo set to `-1` width:

| Where | Overflow |
|---|---|
| `settings_3d` hint after Detail / Budget / Background | **+147 px** past the panel |
| `settings_3d` `(?)` markers in the Mesh section | **+176 px** past the panel |
| `settings_2d` guidance grid hints | **+63 px** past the *column* (clipped by the table) |

Added `widgets.same_line_or_wrap(width)` — continue this line if it fits, start the next one if not
— and `widgets.hint_text(text)`, which also wraps: an evidence line reads `holes 3% · watertight
71% (21 meshes)`, wider than a cell of the two-column guidance grid, so a hint that merely moved
down a line would still have been cut off at the column. `help_marker` now goes through the same
check. The 2D pane already knew about this in exactly one place (its platform combo is narrowed by
30 px "to leave room for the marker"); this asks the layout instead of remembering per call site.

### 5. "Prune..." was invisible at launch

`_storage` called `same_line()` **outside** the `if storage:` branch. The measurement arrives on a
task thread, so `cache.storage` is `{}` for every frame between launch and the first reply — and
the `same_line` then attached to the full-width list child above it, putting Prune **82 px** past
the panel edge. Unclickable exactly while a new install has nothing else on screen. Moved inside
the branch.

### 6. Inker swatch rows clipped their last swatch

`_swatches` priced the gap between two swatches at a literal `6` while the style's item spacing is
`8`, so `per_row` came out one too high and the row overrun the panel (290 px of room, 300 px of
swatches). Now uses `get_style().item_spacing.x`.

### 7. The Clay snap grid disagreed with its own step button

`input_float("grid (m)", 0.0625, ...)` at the implicit `"%.3f"` drew 1/16 m as `0.063`. Now
`"%.4f"`. (I checked that imgui does *not* write a re-parsed value back when the field is idle, so
this was cosmetic only — the stored grid never drifted.)

### 8. A general guard for the whole class

`test_no_pane_continues_a_line_that_has_no_room_left` spies on `imgui.same_line` while building all
sixteen sidebar panes and fails on any call that leaves no room for what follows. The threshold is
"no room at all", not "not much room" — a tight row is a judgement call, a control drawn past the
edge is not. The one exempt caller is `same_line_or_wrap`, which asks the same question itself.
Verified: it catches each of the four bugs above when reverted.

---

## Reviewed and found sound

The Clay **object merge** (`clay/ops.py:join`, `ClayDoc.join_objects`, `clay_ops._join`, `Ctrl+J`) is
correct on the points that are easy to get wrong, and its own tests already pin most of them:

- The CSR `starts` concatenation is right, including the shared boundary offset, and `Mesh.__post_init__`
  re-coerces dtypes so the `int64` that `np.cumsum` introduces never reaches the GPU path.
- Removals are recorded in **descending** index order, so `CompoundEdit`'s reversed undo re-inserts
  ascending — the only order in which the stored indices are all still correct. Redo pops by uid, so
  it is order-independent.
- The generator freeze rides in the same `CompoundEdit`, matching `set_mesh`.
- `_forget_elements` walks compounds, so the undo drops the stale element selection.
- `bm.transformed` already reverses loop winding on a negative-determinant transform, so a
  mirrored or negatively-scaled object merges the right way out.
- `_into` refuses a zero-scaled target as a toast instead of letting `LinAlgError` out of the frame loop.
- `_MUTATING_CTRL` gained `"j"`, so `Ctrl+J` waits for a save like every other mutating shortcut.

Also verified clean: all 83 Lucide icon constants exist in the vendored `lucide.ttf` (including the
two new ones); the manual edits match the code, and the `HELP_TARGETS` ↔ call-site test still
passes in both directions; `_field_options` mutates a freshly built list, not a shared one;
`_reset` rebinds `state.form_2d` and nothing holds a long-lived reference to the old dict;
`warlock doctor` is healthy (the three WARNs are the known ones — no gltfpack, no host BiRefNet,
and port 17971 in use).

---


## Resolved after review

All five deferred items were taken, with the whole-mesh weld kept and its units fixed.

### A. The jobs cache could miss a new artifact permanently

The one item that was a correctness bug rather than a judgement call, and the one that made
`test_a_new_artifact_on_disk_is_noticed_without_a_status_change` fail about 1 run in 10.

`service/files.py:attach_files` stamps each row `(status, job-dir mtime)`, justified in both its
docstring and `CLAUDE.md` by "a file appearing or being removed … is exactly what moves a
directory's mtime". That is true about the event and silent about the **resolution**, which is what
the cache actually depends on. Measured, 200 trials on this machine:

```
directory mtime unchanged after adding a file: 155/200 times
when it changed, delta ms: min=0.993 median=1.002
```

**78% of the time the mtime did not move at all**, because Windows writes it from the system clock —
1 ms here, 15.6 ms by default when nothing has raised the timer resolution. So a write landing after
a listing but inside the stamped mtime's own tick is invisible to the stamp, and not for one tick:
every later comparison keeps matching, so that row serves a stale file list until the job's status
changes or another write happens to cross a tick boundary. Precisely the case the mtime half exists
for — a rig written into the **source** job's directory while that job stays `done`.

Fixed with git's racily-clean rule (`MTIME_RACE_NS = 50 ms`): a stamp is stored only once its mtime
is safely in the past, and a directory touched moments ago is answered correctly and simply not
remembered. **The clock is read after the listing, not before, and that ordering is the proof** —
the hazard needs a write later than the listing yet still in the mtime's tick, which cannot exist if
the listing already finished a tick after the mtime. Cost: one extra listing per write, on one row,
against a 500 ms poll.

The measurement is written up at `docs/measurements/2026-08-07-directory-mtime-granularity.md`, and
the `CLAUDE.md` clause and the docstring that asserted the false thing now say what is true.

*Verified:* the flaky test passed **20/20** consecutive runs. Two new tests pin the rule directly by
controlling the mtime with `os.utime` rather than hoping for a race, and both fail without the guard.

### B. The merge's weld distance was in local units but labelled metres

I was over-worried about the *scope* of the weld and under-worried about its *units*. UVs are stored
per face corner, so `merge_vertices` carries a texture seam through a weld untouched, and nothing in
Clay leaves two vertices at one position — so the whole-mesh weld is kept.

The real defect: `join` carries everything into the **target's local frame** and welds there, with
`eps` in that frame, while the dialog says `weld distance (m)`. A survivor scaled 2× welded at twice
the number on screen; one at 0.01 at a hundredth of it. Now divided by the largest absolute scale
component — `max` rather than a mean because that is the bound that holds *per axis*, so a
non-uniform scale never fuses points further apart than asked in any direction.

*Tests:* the same world gap decides the same way at scales 0.05, 1 and 20. Verified discriminating
in both directions — no conversion fails both new tests, and a `mean`-based conversion fails the
non-uniform one specifically.

The manual now states that the distance is world-space and that the weld applies to the whole merged
result, including the one case that bites: merge at 0 to keep separate shells, then merge *that*
object with a third, and the shells kept apart on purpose are welded.

### C. Merge silently absorbed hidden objects

Fixed at the source rather than in the op. `clay_mode._select_all`'s object branch took every object
while its own element branch three lines below had always skipped invisible ones — an asymmetry
against `CLAUDE.md`'s statement that `visible=False` means an object "cannot be picked". Both
`_select_all` and `_invert` now skip hidden objects in object mode, which also protects Delete and
Ctrl+D from the same surprise.

The merge gets the other half, for an object hidden *after* it was selected: `_join` filters on
`visible`, and the op's `enabled` counts visible selected objects so the row greys rather than
toasting — the op's own standard ("an enabled button that does nothing is worse than a greyed one").

### D. The library's (?) help button is back

Both halves restored — the `HELP_TARGETS` entry and the call site, which the manual integrity test
enforces as a pair. It takes its own short row because `render.help_button` right-aligns itself with
`same_line(cursor + avail - 26)`, which only computes where a line starts.

### E. The bulk bar now says how much of the selection is off screen

Honesty rather than restriction: selecting across a filter change is a real workflow, so
`state.checked` still survives one. What changed is that it says so — `12 selected (4 not shown)`
once some of what is ticked has been filtered away or scrolled past the newest-N window, and the
delete confirm repeats the number. The destructive path no longer describes a smaller act than it
performs.

---

## Verification

`uv run ruff check .` clean; `uv run pytest -q` → **3110 passed, 7 skipped** (from 3091 at the
start). Every new test was run against the unfixed code first and confirmed to fail for the stated
reason. The previously flaky cache test passed 20/20.

## The eyes-on pass — done, and it found one

The four remaining checks needed eyes rather than assertions, so each UI state was driven through
the real panes and the real backend and rendered to a PNG off the smoke fixtures' framebuffer. Three
passed as described:

- The Clay merge dialog reads **`0.0001`**, with its warn line and Apply/Cancel intact. (The snap
  grid field picks up the same derived format and reads `0.1250`.)
- `Ctrl+A` in object mode leaves the hidden object out — asserted, not just looked at:
  `uids[2] not in doc.selection` and `len(selection) == 2` — and **Merge Objects** draws visibly
  greyed while Duplicate, Bake Transform, Mirror X/Y/Z and Smooth beside it stay live.
- Ticking select-all and switching the status filter to *failed* makes the bulk bar read
  **`4 selected (2 not shown)`**, and the delete confirm repeats it.

**The library's (?) did not.** Item D restored the pair the manual integrity test enforces — the
`HELP_TARGETS` entry and the call site — and that test only asks whether both exist, so nothing
noticed *where* the button landed. `render.help_button` right-aligns with an unconditional
`same_line(cursor + avail - 26)`, and the filter row reserved width for two square buttons and had
already put the select-all tick at the right edge. Measured: tick at `(342, 166)-(367, 191)`, (?) at
`(341, 166)-(366, 191)`. The same pixels. The (?) was drawn second, so it took the tick's clicks —
pressing select-all opened the manual.

Fixed where the row is laid out rather than in `help_button`, which every other pane calls against a
line that has room: the sort row now reserves three squares and `_filters` draws the (?) itself,
before `_failures` (whose row is not always there). All three are now distinct and inside the
sidebar's edge.

This is the same bug shape as Pass 1 and it escaped Pass 1's guard, because
`test_no_pane_continues_a_line_that_has_no_room_left` asks whether a control is drawn past the right
edge and this one was drawn *on top of* another, and because
`test_the_library_filter_row_fits_the_sidebar` exercises `_filters` while the (?) was called from
`draw`. So the guard has a second half now:
`test_no_two_of_a_panes_icon_buttons_are_drawn_on_top_of_each_other` collects every icon button's
rect through a real frame and fails on any overlapping pair — confirmed red against the unfixed
code with exactly the rects above.

Nothing in this document is outstanding.

# Library pane relocation: left sidebar → right column

## Problem

In the 2D/3D generation modes, the three-column workspace (`App._workspace()` in
`src/warlock/studio/main.py`) currently stacks the job library list under the
generation settings in the left sidebar, split by the one draggable ratio
`layout.Layout.settings_share`. The right column holds a single `inspector`
pane. The user wants the library moved to the right column, under the
inspector, so the left column is settings-only.

## Design

**Left column** (`main.py`, workspace build site around lines 1019-1057):
becomes a single `layout_mod.pane_child("settings", (sidebar_w, 0))` that
dispatches to `settings_2d.draw(ctx)` / `settings_3d.draw(ctx)` based on
`ctx.state.mode`. The splitter between settings and library is removed — there
is nothing left below settings to split against.

**Right column**: becomes a two-pane vertical stack, following the existing
pattern already used by `_clay_workspace` and `_inker_workspace` for their
mirrored right-side stacks (both explicitly documented as reusing
`settings_share` rather than owning a second splitter, "so the two editors do
not drift"):

1. `layout_mod.pane_child("inspector", (0, avail_y * lay.settings_share))` →
   `inspector.draw(ctx)` (top — asset details, unchanged internals)
2. `layout_mod.splitter("sidebar-share", vertical=False, …)` — the same
   splitter instance/id already used on the left today, now drawn here
   instead, still writing to `lay.settings_share`
3. `layout_mod.pane_child("library", (0, 0))` → `library.draw(ctx)` (bottom —
   job list/management, unchanged internals)

**Centre viewport**: unaffected. `layout.centre_width()` already subtracts two
fixed `SIDEBAR_W` (300px) columns from the available width, so it requires no
change.

**No changes to `panes/library.py` or `panes/inspector.py` internals.** Both
already draw into whatever region `pane_child` gives them, and `library.py`'s
card/thumbnail sizing (`CARD_HEIGHT`, `THUMB_SIZE`) is already pinned to a
300px-wide column — that stays true on the right side.

**No new persisted state.** The split ratio continues to be the single
`layout.Layout.settings_share`, loaded/saved exactly as today
(`Layout.save()` persists `{"settings_share": …}`); no new splitter id, no new
settings key.

**Unaffected call site**: `panes/landing.py`'s use of `library.draw(ctx)` for
the home "open existing project" screen is a separate, standalone
`pane_child` call and is untouched by this change.

## Testing

This is a pure layout/wiring change with no new logic, so verification is
manual/visual rather than a new automated test:

- Launch the app, enter 2D mode: left column shows only settings, filling the
  full height; right column shows inspector on top, library on bottom.
- Repeat for 3D mode.
- Drag the splitter and confirm both the top (inspector) and bottom (library)
  panes resize together, matching how the left sidebar's settings/library
  split behaved before the move.
- Confirm the library's filter row, card list (scrolling), bulk-action bar,
  and storage/prune footer all still render correctly at the narrower
  right-column width (300px, unchanged from before).
- Confirm the inspector's Details/Rig & Pose/Export tabs still render
  correctly in the now-split (rather than full-height) top pane.
- Confirm the landing screen's "open existing project" library list is
  unaffected.

## Out of scope

- No change to `library.py` or `inspector.py` drawing logic.
- No new independent splitter for the right column (explicitly decided
  against — mirrors `settings_share` per existing Clay/Inker precedent).
- No change to Clay, Inker, or Review mode workspaces.

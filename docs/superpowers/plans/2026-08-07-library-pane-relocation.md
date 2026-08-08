# Library Pane Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the 2D/3D generation workspace, move the job library list from the left sidebar (stacked under settings) to the right sidebar (stacked under the inspector), leaving the left sidebar as settings alone.

**Architecture:** `App._workspace()` in `src/warlock/studio/main.py` currently builds a left column of two stacked panes (`settings` over `library`, split by `layout.Layout.settings_share` via a `layout.splitter` drag handle) and a right column of one pane (`inspector`). This plan flips that: the left column becomes a single `settings` pane filling the full height, and the right column becomes the two-stacked-pane shape (`inspector` over `library`, split by the same `settings_share`/splitter) that the left column used to own. This exact "two panes stacked, split by `settings_share`, one splitter" shape already exists three other times in this file (`_clay_workspace`'s left stack, `_inker_workspace`'s left stack, and the original `settings`/`library` left stack being replaced here) — this task relocates it rather than inventing a new pattern.

**Tech Stack:** Python 3.12+, imgui (imgui_bundle), the project's own `warlock.studio.layout` module for pane sizing primitives (`pane_child`, `splitter`, `Layout`). Tests run via `uv run pytest`.

## Global Constraints

- `layout.SIDEBAR_W = 300.0` (design px) is fixed and unchanged by this plan — both sidebars stay this width.
- The only persisted layout state is `Layout.settings_share` (via `Layout.save()`, which writes `{"settings_share": ...}`); this plan introduces no new persisted key and no new splitter id.
- No changes to `src/warlock/studio/panes/library.py` or `src/warlock/studio/panes/inspector.py` internals — both already draw into whatever region `pane_child` gives them.
- No changes to `src/warlock/studio/panes/landing.py`'s separate, standalone use of `library.draw(ctx)` for the home screen.
- No changes to Clay, Inker, or Review mode workspaces (`_clay_workspace`, `_inker_workspace`, `_review_workspace`).
- Design source of truth: `docs/superpowers/specs/2026-08-07-library-pane-relocation-design.md`.

---

### Task 1: Relocate the library pane to the right column, split under the inspector

**Files:**
- Modify: `src/warlock/studio/main.py:1019-1057` (the 2D/3D workspace build site, inside the method that builds `##host`'s three columns — search for the comment `# The sidebar is two scrollers, not one`)
- Modify: `tests/test_studio_smoke.py:568-598` (`test_the_whole_frame_builds_at_once`) — this test hand-mirrors the pane arrangement `main.py` builds (it cannot invoke `App._workspace()` directly, since `App` requires a real pygame window; this is the established pattern in this file, see the same shape in `_clay_workspace`/`_review_workspace` mirrors elsewhere in the suite)
- Create (new test in the same file): `tests/test_studio_smoke.py` — a new test pinning the inspector/library split geometry, next to `test_the_whole_frame_builds_at_once`

**Interfaces:**
- Consumes: `layout.pane_child(pane_id: str, size: tuple[float, float]) -> bool`, `layout.splitter(split_id: str, *, vertical: bool = True, length: float = 0.0) -> float`, `layout.Layout(settings: Any)` with `.settings_share: float` and `.save() -> None`, `layout.SHARE_MIN`/`SHARE_MAX`, `tokens.sp(px: float) -> float`, `tokens.SCALE`, `panes.library.draw(ctx)`, `panes.inspector.draw(ctx)`, `panes.settings_2d.draw(ctx)`, `panes.settings_3d.draw(ctx)` — all pre-existing, none of their signatures change.
- Produces: no new public names. The only externally-visible change is the on-screen arrangement (and the `test_the_right_sidebar_splits_between_inspector_and_library` test below, which nothing else depends on).

- [ ] **Step 1: Read the current workspace build site to confirm line numbers are unchanged**

Run: view `src/warlock/studio/main.py` lines 1019-1057. Confirm it still reads (mode has already been checked to be a non-single-pane, non-workspace mode, i.e. `2d` or `3d`):

```python
        # The sidebar is two scrollers, not one: sharing a single scroll region
        # meant the settings form pushed the library off the bottom of a
        # 950-pixel window, which made the whole asset list unreachable.
        from . import layout as layout_mod
        from . import tokens
        from .tokens import sp

        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)
        imgui.begin_group()
        avail_y = imgui.get_content_region_avail().y
        form_height = avail_y * lay.settings_share
        if layout_mod.pane_child("settings", (sidebar_w, form_height)):
            if ctx.state.mode == "2d":
                settings_2d.draw(ctx)
            else:
                settings_3d.draw(ctx)
        imgui.end_child()
        drag = layout_mod.splitter("sidebar-share", vertical=False, length=sidebar_w)
        if drag and avail_y > 0:
            lay.settings_share = min(
                max(lay.settings_share + drag * tokens.SCALE / avail_y, layout_mod.SHARE_MIN),
                layout_mod.SHARE_MAX,
            )
            lay.save()
        if layout_mod.pane_child("library", (sidebar_w, 0)):
            library.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        self._viewport_pane()
        imgui.same_line()

        if layout_mod.pane_child("inspector", (0, 0)):
            inspector.draw(ctx)
        imgui.end_child()
        imgui.end()
        self._overlays(viewport)
```

If the surrounding lines have drifted (e.g. due to unrelated edits landing first), locate the block by searching for the comment `"The sidebar is two scrollers, not one"` instead of trusting the line numbers.

- [ ] **Step 2: Replace the block with the relocated layout**

Replace the entire block from Step 1 with:

```python
        # The library used to share the left sidebar with settings, split by
        # settings_share; it shares the right sidebar with the inspector now
        # instead, so the left column is settings alone (nothing left to split
        # against) and the right column is the two-scroller stack that used to
        # live on the left.
        from . import layout as layout_mod
        from . import tokens
        from .tokens import sp

        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)
        if layout_mod.pane_child("settings", (sidebar_w, 0)):
            if ctx.state.mode == "2d":
                settings_2d.draw(ctx)
            else:
                settings_3d.draw(ctx)
        imgui.end_child()

        imgui.same_line()
        self._viewport_pane()
        imgui.same_line()

        imgui.begin_group()
        avail_y = imgui.get_content_region_avail().y
        inspector_height = avail_y * lay.settings_share
        if layout_mod.pane_child("inspector", (0, inspector_height)):
            inspector.draw(ctx)
        imgui.end_child()
        drag = layout_mod.splitter("sidebar-share", vertical=False, length=sidebar_w)
        if drag and avail_y > 0:
            lay.settings_share = min(
                max(lay.settings_share + drag * tokens.SCALE / avail_y, layout_mod.SHARE_MIN),
                layout_mod.SHARE_MAX,
            )
            lay.save()
        if layout_mod.pane_child("library", (0, 0)):
            library.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.end()
        self._overlays(viewport)
```

Notes on the two deliberate differences from a naive copy-paste of the old left-column code:
- `settings`'s size is `(sidebar_w, 0)`, not `(sidebar_w, form_height)` — it now fills the whole column, so height `0` (imgui's "fill remaining") is correct and no `avail_y`/`form_height` calculation is needed for it.
- `inspector` and `library` both size their width as `0` (fill remaining), not `sidebar_w` — this matches how the old `inspector` pane already sized itself (`(0, 0)`) at this same right-column position, since at this cursor position `imgui.get_content_region_avail().x` already equals `sidebar_w` (the centre pane already claimed everything else via `layout.centre_width()`). The explicit `length=sidebar_w` is kept on the `splitter(...)` call, matching the defensive style the original left-column call used (avoids relying on `begin_group()`'s effect on content-region-avail).

- [ ] **Step 3: Confirm the file still imports what it needs**

`library`, `inspector`, `settings_2d`, `settings_3d` must already be imported at module level in `main.py` (they were used identically before this change, just at different call sites within the same method) — no import changes needed. Confirm with:

Run: `grep -n "^from .panes import\|^from \.panes import" src/warlock/studio/main.py` (or open the top of the file) and confirm `library`, `inspector`, `settings_2d`, `settings_3d` are present among the pane imports.

- [ ] **Step 4: Update the mirror smoke test to match the new arrangement**

In `tests/test_studio_smoke.py`, find `test_the_whole_frame_builds_at_once` (currently lines 568-598) and replace its body:

```python
def test_the_whole_frame_builds_at_once(app_ctx, imgui_ctx):
    """The real layout: three panes side by side, as main.py assembles them.

    Through ``layout.pane_child`` rather than ``begin_child``, because that is
    what main.py calls now and a style var pushed and popped around begin is
    exactly the kind of thing that only fails inside a real frame. The right
    column is two stacked panes (inspector over library) since the library
    moved off the left sidebar onto the right one, under the inspector.
    """
    from warlock.studio import layout as layout_mod
    from warlock.studio.panes import inspector, library, overlay, settings_2d

    _seeded(app_ctx)
    imgui, renderer = imgui_ctx

    def build():
        layout_mod.pane_child("settings", (340, 0))
        settings_2d.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        layout_mod.pane_child("viewport", (400, 0))
        overlay.toolbar(app_ctx)
        overlay.placeholder(app_ctx)
        imgui.end_child()
        imgui.same_line()
        layout_mod.pane_child("inspector", (340, 250))
        inspector.draw(app_ctx)
        imgui.end_child()
        layout_mod.pane_child("library", (340, 0))
        library.draw(app_ctx)
        imgui.end_child()

    _frame(imgui_ctx, build)
    del renderer
```

(The `library` calls after `settings_2d.draw`/`imgui.separator()` are removed — `library.draw` is no longer nested inside the `settings` pane at all; the new `imgui.same_line()` before `layout_mod.pane_child("inspector", ...)` is retained since it was already there in the original test.)

- [ ] **Step 5: Run the smoke test to confirm it still builds cleanly**

Run: `uv run pytest tests/test_studio_smoke.py::test_the_whole_frame_builds_at_once -v`
Expected: PASS (this is a "does it build without an imgui begin/end mismatch" smoke test, not a geometry assertion — Step 6 below adds the geometry pin).

- [ ] **Step 6: Write a new test pinning the inspector/library split geometry**

Add this test directly after `test_the_whole_frame_builds_at_once` in `tests/test_studio_smoke.py`:

```python
def test_the_right_sidebar_splits_inspector_and_library_by_settings_share(app_ctx, imgui_ctx):
    """The right sidebar's split pins the same arithmetic the left sidebar's
    settings/library split used before the library moved -- inspector gets
    ``avail_y * settings_share`` and library gets whatever is left, so a
    future edit that hardcodes a 50/50 split (or swaps which pane is on top)
    shows up here rather than only on screen.
    """
    from warlock.studio import layout as layout_mod

    imgui, _renderer = imgui_ctx
    lay = layout_mod.Layout(app_ctx.settings)
    assert lay.settings_share == 0.55  # the untouched default this test relies on

    tops: list[float] = []
    bottoms: list[float] = []

    def build():
        imgui.begin_group()
        avail_y = imgui.get_content_region_avail().y
        inspector_height = avail_y * lay.settings_share
        top = imgui.get_cursor_screen_pos().y
        if layout_mod.pane_child("inspector", (300, inspector_height)):
            pass
        imgui.end_child()
        tops.append(imgui.get_item_rect_size().y)
        bottom_start = imgui.get_cursor_screen_pos().y
        if layout_mod.pane_child("library", (300, 0)):
            pass
        imgui.end_child()
        bottoms.append(imgui.get_item_rect_size().y)
        imgui.end_group()
        del top, bottom_start

    _frame(imgui_ctx, build)

    total = tops[0] + bottoms[0]
    assert tops[0] == pytest.approx(total * 0.55, abs=1.0)
    assert bottoms[0] == pytest.approx(total * 0.45, abs=1.0)
```

- [ ] **Step 7: Run the new test to verify it passes**

Run: `uv run pytest tests/test_studio_smoke.py::test_the_right_sidebar_splits_inspector_and_library_by_settings_share -v`
Expected: PASS. If it fails on the `pytest.approx` lines, print `tops[0]`, `bottoms[0]`, and `total` to check whether `PANE_PADDING`/border sizing is shifting the split by more than the 1.0px tolerance, and widen `abs=` slightly rather than changing the 0.55/0.45 split itself (which must stay tied to `lay.settings_share`).

- [ ] **Step 8: Run the full smoke + layout test files**

Run: `uv run pytest tests/test_studio_smoke.py tests/test_layout.py -v`
Expected: All PASS. In particular, confirm the pre-existing geometry tests that size a pane at `SIDEBAR_W` (e.g. `test_the_library_filter_row_fits_the_sidebar`, `test_a_library_cards_action_row_stays_inside_the_card`, `test_an_evidence_hint_stays_inside_the_pane`) still pass unmodified — they build `library`/`inspector`/`settings_2d`/`settings_3d` standalone at a 300px-wide `pane_child`, which is unaffected by which column that width happens to sit in.

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest`
Expected: All PASS, same pass/fail/skip counts as before this change (see `warlock-worktree-setup` project memory if running from a worktree rather than the main checkout — a worktree is expected to show a few extra skips for `vendor/`/`models/`-gated tests).

- [ ] **Step 10: Manual visual verification**

Launch the app (see the project's `run` skill, or `uv run python -m warlock.studio` per the project's normal entry point) and confirm:
- In 2D mode: left column shows only the generation settings form, filling the full sidebar height (no library list, no splitter below it).
- In 3D mode: same — left column is settings only.
- Right column, both modes: inspector on top, a horizontal splitter bar, library list (filters, cards, bulk-action bar, storage/prune footer) on the bottom.
- Dragging the splitter resizes the inspector/library split live, and the ratio persists (check by dragging, switching modes, and confirming the new ratio holds — `Layout.save()` fires on drag same as before).
- The library's filter row and cards are not clipped or overflowing at the 300px width (this was already covered by Step 8's automated tests, but confirm visually too since imgui clipping can look fine in a headless render and wrong on screen, or vice versa).
- The inspector's Details/Rig & Pose/Export tabs (3D mode) and Details/Export tabs (2D mode) still render correctly now that the inspector is a split top pane rather than the full column height.
- The home screen's "open existing project" library list (landing mode) is unaffected.

- [ ] **Step 11: Commit**

```bash
git add src/warlock/studio/main.py tests/test_studio_smoke.py
git commit -m "$(cat <<'EOF'
Warlock v0.0.11

Move the library pane from the left sidebar to the right sidebar, stacked
under the inspector and split by the same settings_share ratio the left
sidebar used to own; the left sidebar is settings alone now.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

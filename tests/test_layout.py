"""The three-column skeleton's measurements.

Two sidebars that the user could drag became two fixed ones, and the tests here
are about the *leftover*: a settings file written by the version that stored
widths is still on every machine that has ever run Warlock, and it must not
resurrect a width nothing reads or leave one behind for a future reader to find
and half-honour.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

from warlock.studio import layout as layout_mod


class _Settings:
    def __init__(self, stored: Any = None) -> None:
        self.store: dict[str, Any] = {"layout": stored} if stored is not None else {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def test_the_sidebars_are_one_of_three_named_sizes():
    """Named sizes rather than a drag (M106): a form has a width that reads
    well, and what the old free drag bought was a way to make the app look
    broken. ``SIDEBAR_W`` is the one in force -- module state, exactly as
    ``tokens.SCALE`` is, because eight call sites read it directly."""
    assert layout_mod.SIDEBAR_WIDTHS["default"] == 300.0
    assert layout_mod.SIDEBAR_W in layout_mod.SIDEBAR_WIDTHS.values()


def test_a_pane_is_inset_by_a_step_of_the_spacing_scale():
    """This used to read ``PANE_PADDING == 5.0``, which froze one afternoon's
    answer exactly as the spacing-scale test once did: 5 was not a step of
    anything, and the number is a *taste* call UX.md Phase 2 settled against a
    screenshot. What is not taste is that a pane's inset comes from the scale
    rather than being invented, which is the rule this asserts instead.
    """
    from warlock.studio import tokens

    steps = {v for k, v in vars(tokens).items() if k.startswith("SP_")}
    assert layout_mod.PANE_PADDING in steps
    # And it is tighter than the host window's own gutter: a pane sits inside
    # that gutter already, so matching it would double the inset on the two
    # sidebars, which are the width-constrained case.
    assert layout_mod.PANE_PADDING < tokens.SP_4


def test_an_unknown_stored_width_falls_back_rather_than_stopping_the_window():
    try:
        assert layout_mod.set_sidebar("enormous") == "default"
        assert layout_mod.SIDEBAR_W == 300.0
        assert layout_mod.set_sidebar("wide") == "wide"
        assert layout_mod.SIDEBAR_WIDTHS["wide"] == layout_mod.SIDEBAR_W
    finally:
        layout_mod.set_sidebar("default")


def test_a_stored_width_is_a_name_and_never_a_number():
    """So a settings file can never carry a size this build does not offer."""
    settings = _Settings({"sidebar": "wide", "settings_share": 0.4})
    try:
        lay = layout_mod.Layout(settings)
        assert lay.sidebar == "wide"
        lay.save()
        assert settings.store["layout"]["sidebar"] == "wide"
    finally:
        layout_mod.set_sidebar("default")


def test_stored_widths_are_ignored_and_only_the_share_is_read():
    lay = layout_mod.Layout(
        _Settings({"sidebar_w": 480.0, "inspector_w": 280.0, "settings_share": 0.4})
    )
    assert lay.settings_share == 0.4
    assert not hasattr(lay, "sidebar_w")
    assert not hasattr(lay, "inspector_w")


def test_a_save_writes_no_width_key_at_all():
    # Settings.set replaces the whole dict rather than merging into it, so the
    # stale keys die the first time anything saves rather than lingering.
    settings = _Settings({"sidebar_w": 480.0, "inspector_w": 280.0, "settings_share": 0.4})
    layout_mod.Layout(settings).save()
    assert settings.store["layout"] == {
        "settings_share": 0.4,
        # The per-split shares (empty here: nothing has been dragged). Written
        # every time for the same reason the rest of this dict is.
        "settings_shares": {},
        "sidebar": "default",
        # The navigation rail's labels-or-icons preference (the UI redesign, wave 3).
        # It has to be written here every time for the reason the whole test
        # exists: the dict is replaced, so a key ``save`` forgets is a
        # preference that silently resets the next time the other one changes.
        "rail": "icons",
    }


def test_the_rail_preference_round_trips_and_survives_nonsense():
    settings = _Settings({"rail": "icons"})
    assert layout_mod.Layout(settings).rail == "icons"
    # A fresh or unknown preference uses the compact editor-first default;
    # only an explicit stored "labels" keeps an existing expanded rail.
    assert layout_mod.Layout(_Settings({"rail": "enormous"})).rail == "icons"
    assert layout_mod.Layout(_Settings({})).rail == "icons"

    layout = layout_mod.Layout(settings)
    layout.set_rail("icons")
    assert settings.store["layout"]["rail"] == "icons"
    assert layout_mod.Layout(settings).rail == "icons"
    layout.set_rail("labels")
    assert settings.store["layout"]["rail"] == "labels"
    assert layout_mod.Layout(settings).rail == "labels"


def test_a_nonsense_share_falls_back_rather_than_raising():
    assert layout_mod.Layout(_Settings({"settings_share": "wide"})).settings_share == 0.55
    assert layout_mod.Layout(_Settings({"settings_share": 9.0})).settings_share == (
        layout_mod.SHARE_MAX
    )


def test_a_split_starts_at_the_shared_default_and_then_goes_its_own_way():
    """One number behind every workspace's split meant Inker's toolbox handle
    silently re-split Create, Clay, Plotter, Packwright, Troupe and Review."""
    lay = layout_mod.Layout(_Settings({"settings_share": 0.4}))
    assert lay.share("clay") == 0.4
    assert lay.share("inker-tools") == 0.4
    lay.set_share("inker-tools", 0.7)
    assert lay.share("inker-tools") == 0.7
    assert lay.share("clay") == 0.4, "a keyed drag must not move another split"
    assert lay.share("inker-tiles") == 0.4, "Inker's handles are separate splits"


def test_a_stored_split_is_clamped_and_junk_is_dropped():
    lay = layout_mod.Layout(
        _Settings(
            {
                "settings_shares": {
                    "clay-tools": 9.0,
                    "review-runs": "wide",
                    "troupe-cast": 0.42,
                }
            }
        )
    )
    assert lay.share("clay-tools") == layout_mod.SHARE_MAX
    assert lay.share("review-runs") == lay.settings_share
    assert lay.share("troupe-cast") == 0.42
    lay.set_share("packwright-items", -3.0)
    assert lay.share("packwright-items") == layout_mod.SHARE_MIN


def test_a_retired_workspace_key_seeds_both_splits_it_used_to_serve():
    """Clay, Plotter, Troupe and Packwright each stacked two panes on the left
    *and* two on the right and read one key for both, so one handle moved a
    column the user was not looking at. Two keys now -- seeded from the old
    one, so an existing profile opens on the proportion it had."""
    lay = layout_mod.Layout(_Settings({"settings_shares": {"clay": 0.31, "create": 0.62}}))
    assert lay.share("clay-tools") == 0.31
    assert lay.share("clay-outliner") == 0.31
    assert lay.share("create-inspector") == 0.62
    # Deleted, not left alongside: ``save`` writes ``shares`` wholesale, so a
    # key left in place would be re-seeded from on every launch for ever.
    assert "clay" not in lay.shares
    assert "create" not in lay.shares


def test_migration_never_overwrites_a_split_the_user_has_already_moved():
    lay = layout_mod.Layout(
        _Settings({"settings_shares": {"plotter": 0.30, "plotter-layers": 0.66}})
    )
    assert lay.share("plotter-tools") == 0.30
    assert lay.share("plotter-layers") == 0.66


# --- The keying, derived from the source rather than kept in step by hand ----
#
# Wave 0's whole finding was that ``Layout.shares`` had been keyed per split
# since the day it was written, and the callers passed the same string twice
# anyway. A list of expected keys maintained here would be one more thing to
# forget beside them; these two read ``main.py`` and check the property.


def _main_source() -> str:
    from warlock.studio import main as main_mod

    return pathlib.Path(main_mod.__file__).read_text(encoding="utf-8")


def _share_literals() -> list[str]:
    """Every string literal ``main.py`` names a split by.

    Both spellings count: ``lay.share("x")`` in hand-built code, and
    ``split_id="x"`` passed to ``_split_column``, which derives the ``share``
    call and the handle's id from it.
    """
    tree = ast.parse(_main_source())
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"share", "set_share"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                found.append(node.args[0].value)
            for kw in node.keywords:
                if kw.arg == "split_id" and isinstance(kw.value, ast.Constant):
                    found.append(kw.value.value)
        elif isinstance(node, ast.FunctionDef):
            # ``_right_column``'s ``share_key`` default -- Create names its
            # split in the signature and hands it straight to ``split_id``.
            args = node.args
            for name, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if name.arg == "share_key" and isinstance(default, ast.Constant):
                    found.append(default.value)
    return found


def _splitter_ids() -> list[str]:
    tree = ast.parse(_main_source())
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "splitter" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                found.append(arg.value)
            elif isinstance(arg, ast.JoinedStr):
                # ``f"{split_id}-share"`` -- the derived form. Recorded by the
                # suffix it contributes, since the id itself is not a literal.
                found.append("<derived>")
    return found


def _skeleton_share_keys() -> list[str]:
    """The share keys declared in ``skeletons.py``'s slot tables.

    Wave 5 moved two workspaces' columns out of ``main`` and into data, so the
    scan has two sources -- and the *union* is what the rules below are about.
    Declared, never derived: a key is in users' settings files forever, and
    computing one from ``f"{workspace}/{slot}"`` would silently reset every
    dragged proportion with no failure anywhere.
    """
    import ast
    import inspect

    from warlock.studio import skeletons

    tree = ast.parse(inspect.getsource(skeletons))
    found: list[str] = []
    for node in ast.walk(tree):
        named = isinstance(node, ast.keyword) and node.arg == "share_key"
        if named and isinstance(node.value, ast.Constant) and node.value.value:
            found.append(node.value.value)
    return found


def test_no_split_key_is_used_by_two_splits():
    """The defect this wave fixed, stated so it cannot come back by copy-paste:
    Clay, Plotter, Troupe, Packwright and Review each passed one key to both
    their left and their right column."""
    literals = _share_literals() + _skeleton_share_keys()
    duplicates = sorted({key for key in literals if literals.count(key) > 1})
    assert not duplicates, f"one key serving two splits: {duplicates}"


def test_every_split_has_a_handle_and_every_handle_a_split():
    """Six of seven workspaces drew a proportion with no way to change it.

    ``_split_column`` derives the handle's id from ``split_id``, so the check
    is that nothing builds a split outside it -- a hand-built ``lay.share``
    with no matching ``splitter`` is a proportion the user cannot drag, and a
    hand-built ``splitter`` with no share is a handle that moves nothing.
    """
    ids = _splitter_ids()
    # A *set*: Inker's timeline strip is a second hand-composed split (its
    # height is a drag along the bottom of the centre column, which no column
    # renderer owns), and it derives its handle from its key exactly the way
    # ``_split_column`` does. What matters is that no id is a bare literal.
    assert sorted(set(ids)) == ["<derived>"], (
        "every column's handle should come from _split_column, which derives "
        f"its id from split_id; hand-built splitters found: {sorted(set(ids))}"
    )
    # ``layout.column`` derives its handle the same way, from the slot's own
    # ``share_key``, so a declared key *is* a handle there too.
    keys = set(_share_literals()) | set(_skeleton_share_keys())
    assert keys == {
        "clay-tools",
        "clay-outliner",
        "create-inspector",
        "inker-colors",
        # Inker's right column stacks three shareable panes, and the strip
        # along the bottom of its centre column is a fourth split -- keyed
        # rather than fixed so its height is a drag that persists.
        "inker-tools",
        "inker-tiles",
        "inker-timeline",
        "packwright-sources",
        "packwright-items",
        # Two handles, one per column, since Plotter took Tiled's arrangement:
        # Properties over the map file on the left, the layer stack over the
        # tileset palette on the right. ``plotter-tools`` left this set with
        # the pane -- the tools are a strip inside the centre column now, and a
        # strip is not a slot. An orphaned share under the old key in a user's
        # settings file is inert; see ``skeletons.plotter``.
        "plotter-layers",
        "plotter-properties",
        "review-runs",
        "sirens-transport",
        # The right column stacks the instrument list over the envelope editor
        # over the song file, so it carries two split handles as Inker's does.
        "sirens-instruments",
        "sirens-envelopes",
        # The sound-effect list is the third shareable pane of that column
        # (Phase 4), so the right column carries three handles.
        "sirens-effects",
        "troupe-cast",
        "troupe-sheets",
    }

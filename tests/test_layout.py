"""The three-column skeleton's measurements.

Two sidebars that the user could drag became two fixed ones, and the tests here
are about the *leftover*: a settings file written by the version that stored
widths is still on every machine that has ever run Warlock, and it must not
resurrect a width nothing reads or leave one behind for a future reader to find
and half-honour.
"""

from __future__ import annotations

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
        "sidebar": "default",
        # The navigation rail's labels-or-icons preference (the UI redesign, wave 3).
        # It has to be written here every time for the reason the whole test
        # exists: the dict is replaced, so a key ``save`` forgets is a
        # preference that silently resets the next time the other one changes.
        "rail": "icons",
    }


def test_the_rail_preference_round_trips_and_survives_nonsense():
    settings = _Settings({"rail": "labels"})
    assert layout_mod.Layout(settings).rail == "labels"
    # A name, never a width, so a value written by a build that offered a third
    # state cannot become a size this one does not have -- and the fallback is
    # the state the rail is designed around rather than a degraded other one.
    assert layout_mod.Layout(_Settings({"rail": "enormous"})).rail == "icons"
    assert layout_mod.Layout(_Settings({})).rail == "icons"

    layout = layout_mod.Layout(settings)
    layout.set_rail("icons")
    assert settings.store["layout"]["rail"] == "icons"
    assert layout_mod.Layout(settings).rail == "icons"


def test_a_nonsense_share_falls_back_rather_than_raising():
    assert layout_mod.Layout(_Settings({"settings_share": "wide"})).settings_share == 0.55
    assert layout_mod.Layout(_Settings({"settings_share": 9.0})).settings_share == (
        layout_mod.SHARE_MAX
    )

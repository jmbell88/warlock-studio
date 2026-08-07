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


def test_the_sidebars_are_a_fixed_size():
    assert layout_mod.SIDEBAR_W == 300.0
    assert layout_mod.PANE_PADDING == 5.0


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
    assert settings.store["layout"] == {"settings_share": 0.4}


def test_a_nonsense_share_falls_back_rather_than_raising():
    assert layout_mod.Layout(_Settings({"settings_share": "wide"})).settings_share == 0.55
    assert layout_mod.Layout(_Settings({"settings_share": 9.0})).settings_share == (
        layout_mod.SHARE_MAX
    )

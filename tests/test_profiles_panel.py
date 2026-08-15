"""A profile draft is a document, and nothing treated it as one.

UX-17. Nine fields and an anchor image, edited in a pane with a Cancel button
that discarded the lot on one click and a Quit chain that walked six other
document guards and did not know this one existed.

The dirty test is a comparison against the origin rather than a flag, because
the editor writes every field on every frame -- a flag would be true from the
moment the pane opened.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import profiles
from warlock.studio.panes import profiles_panel
from warlock.studio.state import AppState


class _Settings:
    """The two methods ``profiles`` uses. Not the real one: it debounces to
    disk, and none of this is about persistence."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


class _Confirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


@pytest.fixture
def ctx() -> Any:
    return SimpleNamespace(
        state=AppState(),
        settings=_Settings(),
        confirms=_Confirms(),
        svc=SimpleNamespace(config=SimpleNamespace()),
    )


def test_nothing_open_is_not_dirty_and_proceeds(ctx):
    proceeded: list[str] = []
    assert profiles_panel.is_dirty(ctx) is False
    assert profiles_panel.guard(ctx, "quit", lambda: proceeded.append("quit")) is True
    assert proceeded == ["quit"]


def test_closing_the_sheet_with_nothing_typed_just_closes_it(ctx):
    profiles_panel.open_sheet(ctx)
    profiles_panel.close_sheet(ctx)
    assert ctx.state.profiles_open is False
    assert ctx.confirms.asked == []


def test_closing_the_sheet_over_a_draft_asks_first(ctx):
    """The manager was a *mode*, so the only way to lose a draft was to switch
    away -- where this guard already stood. A sheet can also be dismissed by
    Esc and by its own close control, so both of those go through it too."""
    profiles_panel.open_sheet(ctx)
    profiles_panel._open_draft(ctx, "", {})
    ctx.state.profile_draft_name = "Brass"

    profiles_panel.close_sheet(ctx)
    assert ctx.state.profiles_open is True, "it must not close before the answer"
    assert ctx.confirms.asked
    ctx.confirms.asked[0].on_confirm()
    assert ctx.state.profiles_open is False
    assert ctx.state.profile_draft is None


def test_a_recovered_draft_comes_back_with_the_pane_it_belongs_to(ctx, tmp_path):
    """The sheet is not a destination, so putting the reader back means
    restoring both halves: the 2D pane, and the manager over it."""
    import json

    path = tmp_path / "draft.profile.json"
    path.write_text(
        json.dumps({"fields": {"art_style": "painterly"}, "name": "Brass", "origin": ""}),
        encoding="utf-8",
    )
    assert profiles_panel._journal_adopt(ctx, path, {}) is True
    assert ctx.state.mode == "2d"
    assert ctx.state.profiles_open is True
    assert ctx.state.profile_draft_name == "Brass"


def test_an_unsaved_draft_asks_before_it_is_lost(ctx):
    profiles_panel._open_draft(ctx, "", {})
    ctx.state.profile_draft_name = "Brass"
    assert profiles_panel.is_dirty(ctx) is True

    proceeded: list[str] = []
    assert profiles_panel.guard(ctx, "quit", lambda: proceeded.append("quit")) is False
    assert proceeded == [], "it must not proceed until the question is answered"
    assert ctx.confirms.asked, "and it must actually ask"
    # Answering yes is what lets it through.
    ctx.confirms.asked[0].on_confirm()
    assert proceeded == ["quit"]


def test_a_draft_identical_to_its_origin_is_not_dirty(ctx):
    profiles.save_profile(ctx.settings, "Brass", profiles.capture(ctx.state.form_2d))
    saved = profiles.list_profiles(ctx.settings)["Brass"]
    profiles_panel._open_draft(ctx, "Brass", saved)
    assert profiles_panel.is_dirty(ctx) is False, (
        "opening a profile must not read as an edit -- the editor rewrites "
        "every field on every frame, which is why this is a comparison and not "
        "a flag"
    )

    ctx.state.profile_draft["negative_prompt"] = "blurry"
    assert profiles_panel.is_dirty(ctx) is True


def test_renaming_alone_counts_as_an_edit(ctx):
    profiles.save_profile(ctx.settings, "Brass", profiles.capture(ctx.state.form_2d))
    saved = profiles.list_profiles(ctx.settings)["Brass"]
    profiles_panel._open_draft(ctx, "Brass", saved)
    ctx.state.profile_draft_name = "Bronze"
    assert profiles_panel.is_dirty(ctx) is True


def test_the_guard_survives_a_state_that_has_never_opened_the_pane():
    """``docmodes.guard``'s rule: asking whether there is unsaved work must not
    require the state that says there is none. The quit chain runs against
    panes that have never been opened."""
    bare = SimpleNamespace(state=SimpleNamespace(), confirms=_Confirms())
    proceeded: list[str] = []
    assert profiles_panel.guard(bare, "quit", lambda: proceeded.append("quit")) is True
    assert proceeded == ["quit"]


def test_the_quit_chain_includes_the_profiles_guard():
    from pathlib import Path

    from warlock.studio import main

    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "profiles_panel.guard" in source


def test_cancel_goes_through_the_guard_rather_than_closing_outright():
    from pathlib import Path

    source = Path(profiles_panel.__file__).read_text(encoding="utf-8")
    cancel = source.index('imgui.button("Cancel"')
    after = source[cancel : cancel + 400]
    assert "guard(ctx" in after, "Cancel must ask before discarding a draft"

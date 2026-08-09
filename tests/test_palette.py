"""I80: the command palette's data half.

Everything about *which* command a query finds is here, headlessly -- the
drawing half owns no list, which is the whole point of the split.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import modes, palette


def _job(job_id: str, **over: Any) -> dict[str, Any]:
    row = {
        "id": job_id,
        "name": "",
        "prompt": "",
        "status": "done",
        "kind": "model",
        "stage": "model",
        "files": [],
    }
    row.update(over)
    return row


def _ctx(mode: str = "3d", jobs: list[Any] | None = None, selected: str | None = None) -> Any:
    rows = jobs or []
    by_id = {job["id"]: job for job in rows}
    return SimpleNamespace(
        state=SimpleNamespace(
            mode=mode,
            previous_mode=mode,
            mode_observed=mode,
            selected=selected,
            source_job=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
        ),
        cache=SimpleNamespace(jobs=rows, get=by_id.get),
        viewer=None,
    )


# --- matching ----------------------------------------------------------------


def test_a_substring_beats_a_scattered_match():
    """The first row is what Enter runs, so the ranking is a correctness
    property rather than a nicety."""
    contiguous = palette.match("clay", "Go to Clay")
    scattered = palette.match("clay", "Cancel the last yield")
    assert contiguous is not None and scattered is not None
    assert contiguous > scattered


def test_a_prefix_beats_a_hit_in_the_middle():
    assert palette.match("go", "Go to Clay") > palette.match("go", "Undo the last go")


def test_initials_find_a_command():
    assert palette.match("gtc", "Go to Clay") is not None
    assert palette.match("tw", "Toggle wireframe") is not None


def test_a_letter_that_is_not_there_does_not_match():
    assert palette.match("zq", "Go to Clay") is None


def test_an_empty_query_matches_everything_equally():
    assert palette.match("", "anything") == 0


def test_matching_ignores_case():
    assert palette.match("CLAY", "Go to Clay") is not None


def test_ranking_is_stable_within_a_score():
    """Commands arrive grouped and assets newest-first, so equal rows must keep
    the order they came in -- an unstable sort reshuffles the list under the
    cursor as a character is typed that changes nothing."""
    items = ["alpha one", "alpha two", "alpha three"]
    assert palette.rank("", items, lambda s: s) == items


def test_ranking_drops_what_does_not_match():
    items = ["Go to Clay", "Toggle wireframe"]
    assert palette.rank("wire", items, lambda s: s) == ["Toggle wireframe"]


# --- the command list --------------------------------------------------------


def test_every_mode_has_a_go_to_command_and_they_are_derived():
    """A palette is a second index of what the app can do, and a hand-written
    one drifts: a ninth mode would gain a switch segment and an Alt+9 binding
    and be missing from the one surface whose job is telling the user it
    exists."""
    keys = {command.key for command in palette.commands(_ctx())}
    assert {f"go:{key}" for key in modes.KEYS} <= keys


def test_a_go_to_command_advertises_no_key():
    """There is no per-mode binding to advertise since the positional Alt+digit
    scheme went away, and a hint naming Ctrl+K would be the palette telling you
    how to open the palette you are reading it in."""
    found = {c.key: c.hint for c in palette.commands(_ctx())}
    assert {found[f"go:{key}"] for key in modes.KEYS} == {""}


def test_going_somewhere_records_where_it_came_from():
    """Otherwise Esc out of the mode the palette put you in goes two steps
    back -- the palette is a mode switch like any other."""
    ctx = _ctx("3d")
    command = next(c for c in palette.commands(ctx) if c.key == "go:manual")
    command.run(ctx)
    assert ctx.state.mode == "manual"
    assert ctx.state.previous_mode == "3d"


def test_library_and_profiles_are_reachable_as_modes():
    """They were tiles on Home behind a sub-view enum, which is what a
    destination looks like when there is nowhere to put it. The palette derives
    its list from ``modes.MODES``, so this passes for free -- which is the
    property being asserted."""
    keys = {c.key for c in palette.commands(_ctx())}
    assert {"go:library", "go:profiles"} <= keys


@pytest.mark.parametrize("key", ["wireframe", "turntable", "frame"])
def test_the_viewport_commands_are_disabled_outside_a_viewport_mode(key):
    """Listed but greyed, not filtered out: a user searching for "wireframe"
    from Home learns nothing from an empty result and learns where to go from a
    row that is there."""
    command = next(c for c in palette.commands(_ctx("home")) if c.key == key)
    assert command.enabled(_ctx("home")) is False
    assert command.enabled(_ctx("3d")) is True


def test_generate_is_disabled_outside_the_generate_modes():
    command = next(c for c in palette.commands(_ctx()) if c.key == "generate")
    assert command.enabled(_ctx("2d")) is True
    assert command.enabled(_ctx("3d")) is True
    assert command.enabled(_ctx("inker")) is False


def test_the_asset_commands_need_a_selection():
    ctx = _ctx()
    for key in ("reroll", "delete"):
        command = next(c for c in palette.commands(ctx) if c.key == key)
        assert command.enabled(ctx) is False


def test_reroll_is_refused_for_a_hand_made_reference():
    """The same rule the library's overflow states: never offer an action the
    service will refuse."""
    hand_made = _job("j1", kind="image", stage="reference")
    ctx = _ctx(jobs=[hand_made], selected="j1")
    command = next(c for c in palette.commands(ctx) if c.key == "reroll")
    assert command.enabled(ctx) is False
    ctx = _ctx(jobs=[_job("j2", kind="model")], selected="j2")
    command = next(c for c in palette.commands(ctx) if c.key == "reroll")
    assert command.enabled(ctx) is True


def test_every_command_key_is_unique():
    keys = [c.key for c in palette.commands(_ctx())]
    assert len(keys) == len(set(keys))


def test_no_label_leaves_the_basic_latin_range():
    """imgui's default atlas is Basic Latin plus Latin-1; anything outside it
    renders as the missing-glyph box."""
    for command in palette.commands(_ctx()):
        for text in (command.label, command.hint, command.group):
            assert all(ord(ch) < 0x100 for ch in text), text


# --- quick-open --------------------------------------------------------------


def test_quick_open_is_empty_until_something_is_typed():
    """Eight asset names on every open would push the commands off screen, and
    the library is right there."""
    ctx = _ctx(jobs=[_job("abc", name="a chest")])
    assert palette.assets(ctx, "") == []
    assert palette.assets(ctx, "   ") == []


def test_quick_open_finds_by_name_prompt_and_id():
    jobs = [
        _job("aaa111", name="a wooden chest"),
        _job("bbb222", prompt="a rusty sword"),
        _job("ccc333"),
    ]
    ctx = _ctx(jobs=jobs)
    assert [j["id"] for j in palette.assets(ctx, "chest")] == ["aaa111"]
    assert [j["id"] for j in palette.assets(ctx, "sword")] == ["bbb222"]
    assert [j["id"] for j in palette.assets(ctx, "ccc333")] == ["ccc333"]


def test_quick_open_is_capped():
    jobs = [_job(f"job{n}", name="a chest") for n in range(30)]
    assert len(palette.assets(_ctx(jobs=jobs), "chest")) == palette.MAX_ASSETS


def test_the_palette_module_imports_nothing_that_needs_a_window():
    """Pure in the ``vram.py`` sense at *module* scope, so importing it and
    asking what a query finds costs no window.

    Only the top level is pinned: a command's ``run`` necessarily reaches the
    pane that performs it, and does so lazily -- which is the same shape every
    other module here uses for the same reason.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(palette))
    banned = {"imgui", "imgui_bundle", "moderngl", "pygame"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not banned & {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned

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


def _ctx(
    mode: str = "create",
    jobs: list[Any] | None = None,
    selected: str | None = None,
    stage: str = "mesh",
) -> Any:
    from warlock.studio.state import ManualState

    rows = jobs or []
    by_id = {job["id"]: job for job in rows}
    return SimpleNamespace(
        state=SimpleNamespace(
            mode=mode,
            previous_mode=mode,
            mode_observed=mode,
            create_stage=stage,
            selected=selected,
            source_job=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            # The real thing rather than a stub: the Manual is an overlay now,
            # so the palette's entry for it reads and writes this state.
            manual=ManualState(),
            profiles_open=False,
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
    ctx = _ctx("create")
    command = next(c for c in palette.commands(ctx) if c.key == "go:settings")
    command.run(ctx)
    assert ctx.state.mode == "settings"
    assert ctx.state.previous_mode == "create"


def test_the_manual_is_a_command_rather_than_a_destination():
    """It stopped being a mode in the UI redesign, wave 3, so ``_mode_commands`` no
    longer derives an entry -- and a reference reachable only by a function key
    is one most people never find."""
    ctx = _ctx("create")
    command = next(c for c in palette.commands(ctx) if c.key == "manual")
    assert command.hint == "F1"
    command.run(ctx)
    assert ctx.state.manual.open
    # Over the mode it was asked from, not instead of it.
    assert ctx.state.mode == "create"


def test_the_library_is_reachable_as_a_mode():
    """It was a tile on Home behind a sub-view enum, which is what a
    destination looks like when there is nowhere to put it. The palette derives
    its list from ``modes.MODES``, so this passes for free -- which is the
    property being asserted."""
    keys = {c.key for c in palette.commands(_ctx())}
    assert "go:library" in keys


def test_the_profile_manager_is_a_command_and_goes_where_it_belongs():
    """It stopped being a mode in the UI redesign, wave 3, so there is no
    ``go:profiles`` to derive. The manager is *about* the prompt form, so the
    command opens the Reference stage under it rather than raising a sheet over
    whatever happened to be on screen."""
    ctx = _ctx("clay")
    keys = {c.key for c in palette.commands(ctx)}
    assert "go:profiles" not in keys
    next(c for c in palette.commands(ctx) if c.key == "profiles").run(ctx)
    assert ctx.state.mode == "create"
    assert ctx.state.create_stage == "reference"
    assert ctx.state.profiles_open


@pytest.mark.parametrize("key", ["wireframe", "turntable", "frame"])
def test_the_viewport_commands_are_disabled_outside_a_viewport_mode(key):
    """Listed but greyed, not filtered out: a user searching for "wireframe"
    from Home learns nothing from an empty result and learns where to go from a
    row that is there."""
    command = next(c for c in palette.commands(_ctx("home")) if c.key == key)
    assert command.enabled(_ctx("home")) is False
    assert command.enabled(_ctx("create")) is True


def test_generate_is_disabled_outside_the_generate_modes():
    command = next(c for c in palette.commands(_ctx()) if c.key == "generate")
    assert command.enabled(_ctx("create", stage="reference")) is True
    assert command.enabled(_ctx("create", stage="mesh")) is True
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


# --- parity with what the app can actually do (A-7) ---------------------------
#
# The palette's whole case is discoverability: a searchable list of what the
# app does, reachable from one chord. That case only holds for what is *in* it,
# and four of the most-used actions in the application -- Save, Save as, Export
# and Undo -- were not, along with Quit, the log, the trash and this popup's
# own sibling, the shortcuts list.


def test_every_command_with_a_gate_says_why_it_is_shut():
    """``Command.why`` was built for exactly this and then not filled in: a
    greyed row that says which mode it belongs to was the whole argument for
    listing disabled commands at all, and a row that greys out in silence is
    the palette refusing to teach the thing it exists to teach."""
    silent = [
        command.key
        for command in palette.commands(_ctx())
        # A default ``enabled`` is the always-on lambda from the dataclass, and
        # a ``why`` on one of those would be text that can never be shown.
        if command.enabled is not type(command).__dataclass_fields__["enabled"].default
        and not command.why
    ]
    assert silent == []


@pytest.mark.parametrize(
    "key", ["save", "save-as", "export", "undo", "redo", "new-map", "quit", "open-log"]
)
def test_the_command_exists(key: str):
    """Named one at a time rather than as a set, so a removal says which."""
    assert key in {command.key for command in palette.commands(_ctx())}


@pytest.mark.parametrize("key", ["save", "save-as", "export", "undo", "redo"])
def test_the_document_commands_are_shut_outside_a_document_mode(key: str):
    """They act on "whichever document mode is in front", so in Create there is
    no document and the answer is a greyed row with a sentence -- not a command
    that runs against a tab from a mode the user left."""
    command = next(c for c in palette.commands(_ctx(mode="create")) if c.key == key)
    assert command.enabled(_ctx(mode="create")) is False
    assert command.why


def test_the_export_command_names_the_mode_it_will_act_on():
    """"Export" alone in a searchable list is a command whose result cannot be
    predicted from its label, and this list is read by searching."""
    labels = {
        mode: next(
            c.label for c in palette.commands(_ctx(mode=mode)) if c.key == "export"
        )
        for mode in ("inker", "clay", "plotter", "packwright")
    }
    assert len(set(labels.values())) == 4
    assert "PNG" in labels["inker"]
    assert ".tmx" in labels["plotter"]


def test_the_document_dispatch_covers_every_document_mode():
    """A mode added to the app and not to the table is four commands that
    quietly stop working in it -- which is how Plotter came to have no New."""
    from warlock.studio import modes as modes_mod

    assert set(palette._DOC_MODES) <= {key for key, _label, _icon in modes_mod.MODES}
    assert set(palette._DOC_MODES) == {"inker", "clay", "plotter", "packwright"}


def test_the_shortcuts_command_sets_a_flag_rather_than_opening_a_popup():
    """It cannot open one: ``imgui.open_popup`` names a popup registered in the
    *current* window, and the palette's window is closing on this frame. The
    header consumes the flag where the popup actually lives, which is also what
    makes Ctrl+/ and this command the same door."""
    ctx = _ctx()
    ctx.state.shortcuts_requested = False
    command = next(c for c in palette.commands(ctx) if c.key == "shortcuts")
    command.run(ctx)
    assert ctx.state.shortcuts_requested is True


def test_quit_goes_through_the_apps_guard_and_never_straight_out():
    """The guard is the chain that asks about unsaved pixels, geometry and a
    pose in turn. A palette entry that bypassed it would be the one way out of
    the app that loses work."""
    asked: list[bool] = []
    ctx = _ctx()
    ctx.ask_quit = lambda: asked.append(True)
    command = next(c for c in palette.commands(ctx) if c.key == "quit")
    command.run(ctx)
    assert asked == [True]


def test_quit_does_nothing_at_all_with_no_guard_attached():
    """``Ctx.ask_quit`` is None until the App attaches it, so a headless caller
    does nothing rather than killing a process it does not own."""
    ctx = _ctx()
    command = next(c for c in palette.commands(ctx) if c.key == "quit")
    command.run(ctx)  # no attribute, no exception, no exit


def test_empty_trash_is_shut_when_the_trash_is_empty():
    ctx = _ctx(jobs=[_job("a")])
    command = next(c for c in palette.commands(ctx) if c.key == "empty-trash")
    assert command.enabled(ctx) is False
    full = _ctx(jobs=[_job("a", deleted_at=1.0)])
    assert command.enabled(full) is True

"""The Aseprite-compatible many-to-many input registry."""

from __future__ import annotations

import json
from types import SimpleNamespace

from warlock.studio import inker_ops


def test_commands_tools_and_modifiers_share_one_binding_table():
    kinds = {binding.kind for binding in inker_ops.BINDINGS}
    assert kinds == {"command", "tool", "action_modifier"}
    assert inker_ops.resolve_binding("Ctrl+Z").target == "undo"
    assert inker_ops.resolve_binding("B").target == "brush"
    quick = inker_ops.resolve_binding("Alt", "FreehandTool", trigger="hold")
    assert quick is not None and (quick.kind, quick.target) == ("tool", "eyedropper")


def test_tools_have_aseprite_primary_bindings_and_compatibility_aliases():
    assert inker_ops.shortcut_for("tool", "spray") == "Shift+B or A"
    assert inker_ops.shortcut_for("tool", "gradient") == "Shift+G or K"
    assert inker_ops.shortcut_for("tool", "curve") == "Shift+L or F"
    assert inker_ops.shortcut_for("tool", "ellipse") == "Shift+U or J"
    assert inker_ops.shortcut_for("tool", "slice") == "Shift+C or C"


def test_a_context_binding_wins_over_a_global_tool_binding():
    assert inker_ops.resolve_binding("F", "Selection").target == "fill_selection"
    assert inker_ops.resolve_binding("F", "FreehandTool").target == "curve"


def test_an_override_replaces_only_its_target_and_can_have_many_chords():
    overrides = inker_ops.set_shortcuts({}, "command", "undo", ["Ctrl+U", "F12"])
    assert inker_ops.by_key("Ctrl+Z", overrides=overrides) is None
    assert inker_ops.by_key("Ctrl+U", overrides=overrides).name == "undo"
    assert inker_ops.by_key("F12", overrides=overrides).name == "undo"
    assert inker_ops.resolve_binding("B", overrides=overrides).target == "brush"


def test_a_user_override_wins_a_default_collision():
    overrides = inker_ops.set_shortcuts({}, "command", "undo", ["B"])
    binding = inker_ops.resolve_binding("B", overrides=overrides)
    assert binding is not None and (binding.kind, binding.target) == ("command", "undo")


def test_an_empty_override_really_unbinds_instead_of_falling_back():
    overrides = inker_ops.set_shortcuts({}, "tool", "brush", [])
    assert inker_ops.resolve_binding("B", overrides=overrides) is None


def test_shortcut_import_export_is_versioned_deterministic_and_validated():
    overrides = inker_ops.set_shortcuts({}, "tool", "brush", ["P", "B"])
    encoded = inker_ops.shortcuts_json(overrides)
    assert encoded == inker_ops.shortcuts_json(overrides)
    assert inker_ops.parse_shortcuts(encoded) == json.loads(encoded)["overrides"]

    try:
        inker_ops.parse_shortcuts('{"version": 99, "overrides": {}}')
    except ValueError as exc:
        assert "version 1" in str(exc)
    else:  # pragma: no cover - the assertion above is the expected path
        raise AssertionError("a future shortcut format was silently accepted")


def test_manifest_covers_every_registered_surface_exactly_once():
    contract = inker_ops.manifest()
    assert contract["target"] == {
        "application": "Aseprite",
        "version": "1.3.15.5",
        "platform": "Windows",
    }
    assert {row["id"] for row in contract["commands"]} == {op.name for op in inker_ops.OPS}
    assert {row["id"] for row in contract["tools"]} == {
        binding.target for binding in inker_ops.BINDINGS if binding.kind == "tool"
    }
    assert {row["id"] for row in contract["action_modifiers"]} == {
        modifier.name for modifier in inker_ops.ACTION_MODIFIERS
    }
    assert all(row["bindings"] for row in contract["tools"])


def test_quick_tools_restore_on_key_up():
    import pygame

    from warlock.studio import inker, inker_mode, inker_state

    state = inker_state.InkerState(tool="brush")
    state.add(inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="quick", title="Quick"))
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    down = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LALT, mod=pygame.KMOD_ALT)
    up = pygame.event.Event(pygame.KEYUP, key=pygame.K_LALT, mod=0)

    assert inker_mode.handle_key(ctx, down)
    assert state.tool == "eyedropper"
    assert inker_mode.handle_key(ctx, up)
    assert state.tool == "brush"


def test_shortcut_overrides_round_trip_through_studio_settings():
    from warlock.studio import inker_mode

    class Settings:
        def __init__(self, block):
            self.values = {"inker": block}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    changed = inker_ops.set_shortcuts({}, "tool", "brush", ["P"])
    settings = Settings({"shortcuts": changed})
    ctx = SimpleNamespace(state=SimpleNamespace(inker=None), settings=settings)

    state = inker_mode.ensure(ctx)
    assert inker_ops.resolve_binding("P", overrides=state.shortcut_overrides).target == "brush"
    assert inker_ops.resolve_binding("B", overrides=state.shortcut_overrides) is None

    inker_mode.persist(ctx)
    reopened = SimpleNamespace(state=SimpleNamespace(inker=None), settings=settings)
    restored = inker_mode.ensure(reopened)
    assert restored.shortcut_overrides == state.shortcut_overrides


def test_no_two_default_command_bindings_share_a_chord_and_context():
    """A chord that means two things means whichever one the table lists first.

    This is a ratchet rather than a migration: it passes against the table as
    it stands, and it is the test that would have caught the *reason* the
    resize dialog had to be split. "Scale image" and "Resize canvas" were two
    buttons on one popup because one ``Op`` cannot carry two keys, so Aseprite's
    Sprite Size (``Ctrl+Alt+I``) had nowhere to live -- a fact nothing checked
    and nothing surfaced until somebody went looking for the missing binding.

    Commands only. Tool bindings deliberately double up: ``_TOOL_BINDINGS`` and
    ``_QUICK_TOOL_BINDINGS`` are the same letters on press and on hold, which
    is the point of them, and ``_ACTION_BINDINGS`` carry modifiers as gestures.
    """
    from collections import Counter

    seen = Counter(
        (binding.chord, binding.context)
        for binding in inker_ops.BINDINGS
        if binding.kind == "command" and binding.chord
    )
    clashes = sorted(pair for pair, count in seen.items() if count > 1)
    assert not clashes, (
        f"{clashes} each mean two different commands; the second is unreachable"
    )

"""A mode the weights are missing for refuses to open, and says why.

Before this, the rail drew Create and Muse exactly like the eight workspaces
that need no weights at all: you found out by entering, typing a prompt and
pressing the button. The gate is drawn from ``modes.NEEDS_ROWS`` and enforced
at ``state.set_mode`` -- one door, so the greyed picture and the refusal cannot
disagree.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock import fetch
from warlock.studio import modes
from warlock.studio import state as state_mod
from warlock.studio.panes import model_gate
from warlock.studio.state import AppState, set_mode, set_mode_gate


def _ctx(present: bool, *, total: int = 0) -> SimpleNamespace:
    """A ctx whose snapshot answers for every gated row at once."""
    keys = {key for rows in modes.NEEDS_ROWS.values() for key in rows}
    return SimpleNamespace(
        model_rows=[
            {"row_key": key, "present": present, "size_gib": 8.0} for key in sorted(keys)
        ],
        cache=SimpleNamespace(total=total),
    )


def test_every_gated_row_is_a_row_the_registry_actually_has():
    """``modes.py`` may import only ``icons``, so the keys are literals.

    A renamed model must therefore fail here rather than silently ungating the
    mode it was protecting -- which would be invisible, because an unknown key
    is *skipped* by ``model_gate.missing`` on purpose.
    """
    for mode, rows in modes.NEEDS_ROWS.items():
        for key in rows:
            assert fetch.find(key) is not None, f"{mode} names an unknown row {key}"


def test_settings_is_never_gated():
    """It is where the download lives; gating it would be a locked door with
    the key behind it."""
    for key in ("settings", "home", "library", "review"):
        assert key not in modes.NEEDS_ROWS


def test_a_mode_with_its_weights_missing_is_blocked_and_priced():
    ctx = _ctx(present=False)
    assert model_gate.mode_block(ctx, "muse") == ("music:ace_step_v1",)
    reason = model_gate.mode_reason(ctx, "muse")
    assert "1 download" in reason
    assert "8 GB" in reason


def test_a_mode_with_its_weights_present_is_not_blocked():
    ctx = _ctx(present=True)
    assert model_gate.mode_block(ctx, "muse") == ()
    assert model_gate.mode_reason(ctx, "muse") == ""


def test_an_ungated_workspace_is_never_blocked():
    ctx = _ctx(present=False)
    for key in ("inker", "clay", "plotter", "packwright", "sirens", "settings"):
        assert model_gate.mode_block(ctx, key) == ()


def test_existing_work_is_never_locked_away():
    """The gate is for a fresh install, not for a library.

    Create's later stages export meshes that already exist and Muse plays takes
    it did not generate, so a user who removed a model to reclaim disk must not
    lose access to finished work.
    """
    ctx = _ctx(present=False, total=3)
    assert model_gate.mode_block(ctx, "create") == ()
    assert model_gate.mode_block(ctx, "muse") == ()


def test_an_empty_snapshot_gates_nothing():
    """``model_gate``'s doctrine. A headless ctx, and the frames before the
    answers land, must read as "nothing to say" rather than "everything is
    missing" -- which would lock every mode on a fully-installed host."""
    ctx = SimpleNamespace(model_rows=[], cache=SimpleNamespace(total=0))
    assert model_gate.mode_block(ctx, "create") == ()


# --- the one door ------------------------------------------------------------


def test_set_mode_refuses_a_gated_mode_and_says_it_did_not_move():
    state = AppState()
    state.mode = "home"
    try:
        set_mode_gate(lambda key: key != "muse")
        assert set_mode(state, "muse") is False
        assert state.mode == "home"
        assert set_mode(state, "inker") is True
        assert state.mode == "inker"
    finally:
        set_mode_gate(None)


def test_the_gate_never_traps_you_in_a_mode():
    """Leaving is always a different key, so the same-mode early return is
    reached first and a mode that became gated while you were in it is still
    one you can walk out of."""
    state = AppState()
    state.mode = "muse"
    try:
        set_mode_gate(lambda key: key != "muse")
        assert set_mode(state, "home") is True
        assert state.mode == "home"
    finally:
        set_mode_gate(None)


def test_no_gate_installed_means_every_mode_opens():
    assert state_mod._MODE_AVAILABLE is None
    state = AppState()
    state.mode = "home"
    assert set_mode(state, "muse") is True

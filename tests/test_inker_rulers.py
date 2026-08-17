"""The canvas rulers' step ladder, and the canvas-furniture persistence.

The step function is pure so it can be walked over the whole zoom range the
pane allows; the persistence round-trip runs against a dict-backed settings
fake, the ``test_packwright_mode`` pattern.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import inker_mode
from warlock.studio.inker_state import InkerState
from warlock.studio.panes.inker_canvas import RULER_LABEL_PX, ruler_minor, ruler_step

LADDER = {base * magnitude for base in (1, 2, 5) for magnitude in (1, 10, 100, 1000)}


def test_every_step_is_on_the_decimal_ladder_across_the_zoom_range() -> None:
    zoom = 0.25
    while zoom <= 10.0:
        step = ruler_step(zoom)
        assert step in LADDER, f"zoom {zoom}: {step} is not a 1/2/5 number"
        assert step * zoom >= RULER_LABEL_PX, f"zoom {zoom}: labels would collide"
        zoom = round(zoom + 0.05, 2)


def test_the_step_is_the_smallest_that_fits() -> None:
    # At 100% a 48px minimum wants the first ladder rung >= 48, which is 50.
    assert ruler_step(1.0) == 50
    # At 800% every 10 pixels is 80 screen pixels apart.
    assert ruler_step(8.0) == 10
    # At the pane's minimum zoom the rung has to reach 192 image pixels.
    assert ruler_step(0.25) == 200


def test_a_degenerate_zoom_does_not_hang() -> None:
    assert ruler_step(0.0) == 1
    assert ruler_step(-1.0) == 1


def test_minor_ticks_split_fifths_then_halves_then_not_at_all() -> None:
    assert ruler_minor(50) == 10
    assert ruler_minor(20) == 4
    assert ruler_minor(2) == 1
    assert ruler_minor(1) == 0


def test_the_grid_defaults_to_32_and_the_rulers_to_on() -> None:
    state = InkerState()
    assert state.grid_size == 32
    assert state.rulers is True


class _Settings:
    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


def _ctx(settings: _Settings, state: InkerState | None) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(inker=state), settings=settings)


def test_canvas_furniture_round_trips_through_persist() -> None:
    settings = _Settings()
    state = InkerState()
    state.grid, state.grid_size, state.grid_snap, state.rulers = True, 48, True, False
    inker_mode.persist(_ctx(settings, state))

    restored = inker_mode.ensure(_ctx(settings, None))
    assert restored.grid is True
    assert restored.grid_size == 48
    assert restored.grid_snap is True
    assert restored.rulers is False


def test_a_hand_edited_canvas_block_is_validated_not_trusted() -> None:
    settings = _Settings()
    settings.data["inker"] = {
        "canvas": {"grid": 1, "grid_size": True, "grid_snap": "yes", "rulers": 0}
    }
    state = inker_mode.ensure(_ctx(settings, None))
    # A bool is not a size, so the default holds; truthiness is coerced.
    assert state.grid_size == 32
    assert state.grid is True
    assert state.grid_snap is True
    assert state.rulers is False


def test_an_out_of_range_stored_size_is_clamped() -> None:
    settings = _Settings()
    settings.data["inker"] = {"canvas": {"grid_size": 100000}}
    assert inker_mode.ensure(_ctx(settings, None)).grid_size == 512

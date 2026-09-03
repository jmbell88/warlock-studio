"""The render budget. ``uv run pytest -m perf -n 0``.

A regenerate runs in a task, so the frame loop never waits on it, but the
user does: a slider nudged and a second later the timeline updates is the
whole feel of the feature. The budget is the nine-layer fireball at the
default 128px / 4x, measured at ~130 ms a frame on the development machine
after the two cost rules in ``prims/__init__`` (noise at logical resolution,
discs in their windows); the first draft was 1,100 ms. The floor is set at
three times the measurement so a modest CI box passes and a regression that
doubles the cost fails here rather than under somebody's cursor.

Excluded from the parallel run for the reason every ``perf`` case is.
"""

from __future__ import annotations

import time

import pytest
from _recipes import FIREBALL

from warlock.studio.inker import flourish

MAX_MS_PER_FRAME = 400.0


def _default_size_fireball() -> flourish.Recipe:
    raw = dict(FIREBALL)
    raw["size"] = [128, 128]
    raw["supersample"] = 4
    # Scale the geometry up with the canvas so the work is representative.
    scaled = []
    for layer in raw["layers"]:
        params = dict(layer.get("params", {}))
        for key in ("radius", "size", "width", "height"):
            if isinstance(params.get(key), (int, float)):
                params[key] = params[key] * 2.5
        if "x" in params:
            params["x"] = {"keys": [[0, -40], [1, 40]]}
        scaled.append({**layer, "params": params})
    raw["layers"] = scaled
    return flourish.from_dict(raw)


@pytest.mark.perf
def test_a_default_size_fireball_frame_renders_inside_the_budget():
    rec = _default_size_fireball()
    flourish.render_frame(rec, 5)  # warm the primitive imports
    frames = list(range(rec.frame_count))
    start = time.perf_counter()
    for f in frames:
        flourish.to_uint8(flourish.render_frame(rec, f), rec.supersample)
    per_frame = (time.perf_counter() - start) / len(frames) * 1000.0
    assert per_frame < MAX_MS_PER_FRAME, f"{per_frame:.0f} ms/frame"

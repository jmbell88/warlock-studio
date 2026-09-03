"""A keyframed value over a 0..1 parameter, with an easing between keys.

Everything an effect animates -- a radius, an opacity, an emission rate, a
colour's brightness -- is one of these. A curve is a sorted tuple of
``(t, value)`` keys and one easing name applied between every pair, which is
the shape a user can draw in a small panel and the shape a preset file can
hold in three lines. Per-segment easings are deliberately not a feature:
they double the editing surface for a distinction nobody reads at 128px.

The easing vocabulary is ``pipelines/sheet.EASINGS`` plus ``hold``, with the
same arithmetic -- ``smoothstep`` for ``ease``, the two quadratics for in and
out -- so a clip's spacing and an effect's spacing mean the same word. It is
restated here rather than imported because this package imports nothing from
``pipelines``; a test asserts the two tables agree.

Evaluation is vectorised, because a particle system asks the same curve for a
thousand ages at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EASINGS = ("linear", "ease", "ease_in", "ease_out", "hold")


def ease(t: Any, kind: str) -> Any:
    """Reshape a 0..1 segment parameter. Works on floats and arrays alike."""
    if kind == "linear":
        return t
    if kind == "ease":
        return t * t * (3.0 - 2.0 * t)
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return t * (2.0 - t)
    if kind == "hold":
        return np.zeros_like(t) if isinstance(t, np.ndarray) else 0.0
    raise ValueError(f"easing must be one of {list(EASINGS)}")


@dataclass(frozen=True)
class Curve:
    """``keys`` are ``(t, value)`` with ``t`` in 0..1, sorted; at least one."""

    keys: tuple[tuple[float, float], ...]
    easing: str = "linear"

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("a curve needs at least one key")
        if self.easing not in EASINGS:
            raise ValueError(f"easing must be one of {list(EASINGS)}")
        keys = tuple(sorted((float(t), float(v)) for t, v in self.keys))
        object.__setattr__(self, "keys", keys)

    # -- construction ------------------------------------------------------

    @classmethod
    def const(cls, value: float) -> Curve:
        return cls(((0.0, float(value)),))

    @classmethod
    def line(cls, start: float, end: float, easing: str = "linear") -> Curve:
        return cls(((0.0, float(start)), (1.0, float(end))), easing)

    @property
    def is_const(self) -> bool:
        return len(self.keys) == 1 or all(v == self.keys[0][1] for _, v in self.keys)

    @property
    def first(self) -> float:
        return self.keys[0][1]

    # -- evaluation --------------------------------------------------------

    def at(self, t: float) -> float:
        return float(self.sample(np.asarray([t], dtype=np.float32))[0])

    def sample(self, ts: np.ndarray) -> np.ndarray:
        """Values at every ``t`` in ``ts`` (float32 array out)."""
        ts = np.asarray(ts, dtype=np.float32)
        if len(self.keys) == 1:
            return np.full(ts.shape, self.keys[0][1], dtype=np.float32)
        kt = np.asarray([k[0] for k in self.keys], dtype=np.float32)
        kv = np.asarray([k[1] for k in self.keys], dtype=np.float32)
        clamped = np.clip(ts, kt[0], kt[-1])
        # Segment index for each sample: the key at or before it.
        idx = np.searchsorted(kt, clamped, side="right") - 1
        idx = np.clip(idx, 0, len(kt) - 2)
        t0 = kt[idx]
        t1 = kt[idx + 1]
        span = np.where(t1 > t0, t1 - t0, 1.0)
        u = np.clip((clamped - t0) / span, 0.0, 1.0)
        if self.easing == "hold":
            # A step at each key; the last key is reached, not approached.
            return np.where(u >= 1.0, kv[idx + 1], kv[idx]).astype(np.float32)
        u = ease(u, self.easing)
        return (kv[idx] + (kv[idx + 1] - kv[idx]) * u).astype(np.float32)

    # -- codec -------------------------------------------------------------

    def to_json(self) -> Any:
        if self.is_const:
            return self.keys[0][1]
        return {"keys": [[t, v] for t, v in self.keys], "easing": self.easing}

    @classmethod
    def from_json(cls, raw: Any) -> Curve:
        if isinstance(raw, (int, float)):
            return cls.const(float(raw))
        if isinstance(raw, dict):
            keys = raw.get("keys") or []
            easing = str(raw.get("easing") or "linear")
            return cls(tuple((float(k[0]), float(k[1])) for k in keys), easing)
        if isinstance(raw, (list, tuple)):
            return cls(tuple((float(k[0]), float(k[1])) for k in raw))
        raise ValueError(f"not a curve: {raw!r}")

    def clamped(self, lo: float, hi: float) -> Curve:
        return Curve(tuple((t, min(hi, max(lo, v))) for t, v in self.keys), self.easing)

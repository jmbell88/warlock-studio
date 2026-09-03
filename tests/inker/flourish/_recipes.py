"""The fixture recipes the render tests and the digest generator share.

Small on purpose -- 48px at 2x -- so the whole file of digests re-renders in
well under a second, and every primitive appears in at least one.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _texture() -> np.ndarray:
    """An 8x8 straight-alpha test texture: a white diamond with a red core."""
    tex = np.zeros((8, 8, 4), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            d = abs(x - 3.5) + abs(y - 3.5)
            if d <= 3.5:
                tex[y, x] = (255, 60 if d > 1.5 else 255, 60 if d > 1.5 else 255, 255)
    return tex


#: Asset id -> texture, for the sprite fixtures.
ASSETS: dict[str, np.ndarray] = {"tex": _texture()}

FIREBALL: dict[str, Any] = {
    "name": "fireball",
    "seed": 7,
    "size": [48, 48],
    "supersample": 2,
    "fps": 18,
    "phases": [
        {"name": "cast", "frames": 3},
        {"name": "projectile", "frames": 5, "loop": True},
        {"name": "impact", "frames": 2},
        {"name": "explosion", "frames": 4},
        {"name": "dissipate", "frames": 3},
    ],
    "layers": [
        {"kind": "smoke", "name": "Smoke", "params": {"count": 6, "size": 5.0}},
        {
            "kind": "trail",
            "name": "Trail",
            "params": {"x": {"keys": [[0, -16], [1, 16]]}, "radius": 4.0},
            "phases": ["projectile"],
        },
        {
            "kind": "flame",
            "name": "Outer flame",
            "params": {"x": {"keys": [[0, -16], [1, 16]]}, "width": 10.0, "height": 20.0},
            "phases": ["projectile"],
        },
        {
            "kind": "core",
            "name": "Core",
            "params": {"x": {"keys": [[0, -16], [1, 16]]}, "radius": 7.0},
            "phases": ["projectile"],
        },
        {
            "kind": "ring",
            "name": "Shockwave",
            "params": {"radius": {"keys": [[0, 2], [1, 22]]}, "unevenness": 0.5},
            "phases": ["impact", "explosion"],
        },
        {"kind": "flash", "name": "Flash", "params": {"radius": 14.0}, "phases": ["impact"]},
        {
            "kind": "particles",
            "name": "Sparks",
            "params": {"count": 30, "speed": 60.0, "streak": 0.5},
            "phases": ["explosion"],
        },
        {"kind": "glow", "name": "Glow", "params": {"radius": 3.0}},
        {"kind": "distortion", "name": "Heat", "params": {"radius": 20.0, "strength": 2.0}},
    ],
}

#: Every primitive alone on a 32px canvas, for per-kind assertions.
SOLO_PARAMS: dict[str, dict[str, Any]] = {
    "core": {"radius": 8.0},
    "flame": {"width": 10.0, "height": 18.0},
    "particles": {"count": 12, "speed": 40.0},
    "smoke": {"count": 4, "size": 4.0, "emission": "burst"},
    "ring": {"radius": {"keys": [[0, 2], [1, 12]]}},
    "flash": {"radius": 10.0},
    "trail": {"x": {"keys": [[0, -10], [1, 10]]}, "radius": 3.0},
    "glow": {},
    "distortion": {},
    "sprite": {"texture": "tex", "size": 14.0, "rotation": {"keys": [[0, 0], [1, 90]]}},
}


def solo(kind: str, **extra: Any) -> dict[str, Any]:
    layers = []
    if kind in ("glow", "distortion"):
        # Both read the composite beneath them; give them one.
        layers.append({"kind": "core", "name": "Core", "params": SOLO_PARAMS["core"]})
    layers.append({"kind": kind, "name": kind, "params": {**SOLO_PARAMS[kind], **extra}})
    return {
        "name": kind,
        "seed": 3,
        "size": [32, 32],
        "supersample": 2,
        "fps": 12,
        "phases": [{"name": "main", "frames": 6}],
        "layers": layers,
    }

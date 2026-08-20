"""Troupe: the headless character-sheet engine.

Imports no imgui, moderngl, pygame or ``service`` -- pinned by
``tests/troupe/test_troupe_imports.py``, the ``tests/inker/test_sheetout.py``
rule at its fourth instance.
"""

from __future__ import annotations

from . import spec, ulpc

__all__ = ["spec", "ulpc"]

"""Guided tours: the step model and the tours themselves, with nothing drawn.

Pure in the way ``studio/inker/``, ``clay/``, ``plotter/`` and ``packwright/``
are pure -- no imgui, no moderngl, no pygame, no ``service`` -- so every rule
about a tour is assertable headlessly, and a test can walk every step of every
tour without a GL context. ``tests/tour/test_tour_imports.py`` pins the outward
set.

The drawing half is ``studio/panes/tour.py`` and the per-frame state is
``state.TourState``.
"""

from __future__ import annotations

from .scripts import TOURS, find
from .steps import Condition, Step, Tour

__all__ = ["TOURS", "Condition", "Step", "Tour", "find"]

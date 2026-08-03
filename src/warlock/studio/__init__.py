"""The desktop app: one pygame window, one ModernGL context, imgui panels.

Importing this package must not import pygame or moderngl -- ``warlock doctor``
and the test suite reach for :mod:`warlock.studio.viewer.math3d` and friends on
machines with no display. The window lives in :mod:`warlock.studio.main` and is
only built when someone actually runs it.
"""

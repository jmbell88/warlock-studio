"""Flourish: procedural 2D effects, rendered from a recipe.

A spell is not a picture, it is a *recipe*: a seed, a few phases (cast,
projectile, impact, explosion, dissipate), and a stack of layers -- a core, a
flame, a glow, sparks, smoke, a trail, a distortion -- each with parameters and
animation curves. Every frame is a pure function of the recipe and a frame
number, so the same recipe renders the same bytes on every machine, a change to
one slider re-renders only what depends on it, and eight facing directions are
eight rotations of the *simulation* rather than eight drawings.

The alternative this package exists to refuse is "prompt -> diffusion model ->
sixteen frames that hopefully agree". ``pipelines/pixelize`` and
``pipelines/spritesynth`` already make that argument for characters; an effect
is the case where it is most obviously true, because a spark that moves three
pixels the wrong way is visible in every frame and there is no body to hide it
behind.

Headless, and deliberately narrower than its parent: numpy and nothing else at
module scope. No imgui, no pygame, no service, and **no scipy** -- the bar is
that a recipe renders byte-identical frames, and a blur whose kernel comes from
a dependency can change under a ``uv sync``. ``tests/inker/flourish/
test_flourish_imports.py`` pins that.

The modules, bottom up:

``curves``   a keyframed value with an easing, over a 0..1 parameter
``noise``    seeded value noise, fbm and domain warping
``prims``    one module per primitive -- each renders one layer for one frame
``recipe``   the document: phases, layers, their parameters, the JSON codec
``render``   a frame: every layer's premultiplied plane, and the composite

The Inker document that *hosts* a recipe -- a layer group whose cels are these
renders -- is ``_doc_flourish`` one level up; this package never learns what a
document is.
"""

from __future__ import annotations

from .curves import EASINGS, Curve, ease
from .recipe import (
    BLENDS,
    MODES,
    SCHEMA_VERSION,
    Layer,
    Phase,
    Recipe,
    clamp,
    dumps,
    from_dict,
    loads,
    new_uid,
    to_dict,
)
from .render import FrameCtx, composite, render_frame, to_uint8
from .render import render as render_layers

__all__ = [
    "BLENDS",
    "Curve",
    "EASINGS",
    "FrameCtx",
    "Layer",
    "MODES",
    "Phase",
    "Recipe",
    "SCHEMA_VERSION",
    "clamp",
    "composite",
    "dumps",
    "ease",
    "from_dict",
    "loads",
    "new_uid",
    "render_layers",
    "render_frame",
    "to_dict",
    "to_uint8",
]

"""Characters: what a species is, what a request for one is, and how to build it.

The layer sits between a prompt and a Troupe job. It owns four things and
deliberately not a fifth:

* :mod:`.family` -- the registries. Four **archetypes** (body plans, each owning
  a rig template, a clip library, sockets and material regions) and the
  **species** of each (channel defaults, themes, height, aliases).
* :mod:`.recipe` -- one flat, validating record of everything a character sheet
  needs. It refuses by name and never clamps.
* :mod:`.instantiate` -- recipe in, ``source.glb`` / ``model.glb`` /
  ``character.json`` out. Archetype-agnostic: it reads the species' baked asset
  and its mask file and knows nothing about how either was made.
* :mod:`.errors` -- one refusal type carrying the control it came from.

**The word-to-species resolver is :mod:`.resolve`**, and it reads nothing but
the public surface below -- no species list of its own, no archetype table of
its own. That separation is why it can be stdlib-only and why adding a species
is a registry edit rather than a resolver edit. The surface it codes against:

* ``family.families()`` -- ``key -> Family``; every species has ``aliases``
  (the spellings a prompt may use) and ``label`` (what to call it back).
* ``family.families_of(archetype)`` and ``family.ARCHETYPE_KEYS`` -- enough
  structure to answer "what else is shaped like this".
* ``Family.nearest`` -- ordered species keys to offer when the asked-for
  creature is not one we make. An unsupported creature is *offered* the nearest,
  labelled; it is never silently swapped, which is why the hint lives on the
  data and the substitution lives in the resolver.
* ``family.get_family(key)`` and ``recipe.Recipe.from_dict`` -- both raise
  :class:`errors.CharacterError` carrying a ``field``, so a refusal reaches the
  UI pointing at a control.

Nothing in this package imports ``warlock.service``, ``warlock.queue``, imgui,
moderngl, pygame, bpy or torch; ``tests/characters/test_characters_imports.py``
is the pin.
"""

from __future__ import annotations

from .errors import CharacterError
from .family import Archetype, Channel, Family, Socket, Theme, families, get_family
from .recipe import DEFAULT_RECIPE, THEMES, Recipe

__all__ = [
    "DEFAULT_RECIPE",
    "THEMES",
    "Archetype",
    "Channel",
    "CharacterError",
    "Family",
    "Recipe",
    "Socket",
    "Theme",
    "families",
    "get_family",
]

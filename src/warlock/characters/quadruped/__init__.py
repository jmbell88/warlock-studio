"""The quadruped archetype: one parameterised generator and its baked assets.

The generator is imported lazily by everything above it -- ``build`` pulls Clay,
trimesh and manifold3d in, and the only caller that needs them is
``scripts/author_quadruped.py``. Instantiating a character reads the baked
``<silhouette>.glb`` and ``<silhouette>.masks.npz`` beside this file and never
touches the generator at all.
"""

from __future__ import annotations

"""Clay's geometry engine -- the pure half.

Everything under this package is arrays and arithmetic. It may import numpy,
Pillow (lazily, for the texture members of a saved document), the shared history
engine :mod:`warlock.studio.undo` and the viewer's ``gltf`` and ``math3d``
modules, and nothing else: no ``imgui``, no ``moderngl``, no ``pygame``, nothing
from ``service``, ``queue`` or ``pipelines``. That is not tidiness for its own
sake. A user who knows the shape they want gets a path to it that is not
"install Blender", and the only way that path stays trustworthy is if every rule
about geometry -- what a face is, which way a normal points, what a mirror does
to winding, what an extrude does to a texture seam -- is assertable headlessly,
without a window and without a GPU.

That list is pinned by ``tests/clay/test_clay_imports.py``, the way the other
three pure packages pin theirs, so the next outward import is a decision rather
than a discovery. ``gltf`` in particular is reached for rather than mirrored
because a Clay material *is* a ``gltf.Material`` -- see :mod:`.document`.
"""

"""Makes Clay mode's tests a package, so their basenames cannot collide.

``conftest.py`` sets no ``importmode``, so under pytest's default prepend mode a
test module's *basename* must be unique across the whole tree -- a duplicate is
a hard ``import file mismatch`` collection **error**, not a skip. This marker
makes these modules import as ``clay.test_*`` instead of bare ``test_*``, which
is enough on its own; changing ``importmode`` would also fix it and has effects
well beyond this directory.

**The specific collision it was added for is gone, and the file is kept anyway.**
It was added because ``tests/inker/test_document.py`` existed and Clay's own
``studio/clay/document.py`` wanted ``tests/clay/test_document.py``. The inker
module was renamed to ``test_inker_document.py`` in d6fd407, so no basename in
the tree collides today (``tests/inker/``, ``tests/plotter/``,
``tests/packwright/`` and ``tests/manual/`` all get by with no ``__init__.py``
for exactly that reason). What keeps this one is that Clay's names are the most
generic in the tree -- ``test_document``, ``test_mesh``, ``test_ops``,
``test_picking``, ``test_serialize`` -- so it is the directory a new sibling
elsewhere is most likely to collide with, and the failure mode is a collection
error naming a file nobody touched.

The docstring previously justified ``tests/inker/`` going without one as keeping
the undo extraction's "the move was clean, no test changed" property checkable
from the diff. That property did not survive the rename, and citing it here was
pointing at evidence that no longer exists.
"""

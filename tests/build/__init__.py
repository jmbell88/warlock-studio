"""Makes Build mode's tests a package, so their basenames cannot collide.

``tests/`` has no ``__init__.py`` anywhere else and ``conftest.py`` sets no
``importmode``, so under pytest's default prepend mode a test module's *basename*
must be unique across the whole tree. ``tests/inker/test_document.py`` already
exists, and Build mode gets a ``studio/build/document.py`` of its own whose
natural test name is ``tests/build/test_document.py`` -- which without this file
is a hard ``import file mismatch`` collection **error**, not a skip.

This marker makes these modules import as ``build.test_*`` instead of bare
``test_*``, which is enough on its own. ``tests/inker/`` deliberately does not
get one: leaving it untouched is what keeps the undo extraction's "the move was
clean, no test changed" property checkable from the diff. Changing
``importmode`` would also fix it and has effects well beyond this directory.
"""

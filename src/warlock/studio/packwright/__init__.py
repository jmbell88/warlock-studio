"""Packwright -- the atlas packer's engine.

Pure in the way ``studio/inker/``, ``studio/clay/`` and ``studio/plotter/`` are:
no imgui, no moderngl, no pygame, no ``service``. Its outward imports are
:mod:`warlock.studio.undo` (the shared history engine),
:mod:`warlock.studio.inker` (the document type a clip is enumerated from),
:mod:`warlock.studio.plotter.tsx` (the one ``.tsx`` writer in the repo) and
:mod:`warlock.pipelines.sheet` (the authority on the atlas ceiling and on what
"trim" means) -- pinned exactly by ``tests/packwright/test_imports.py``.

Two of those four are the ``sheetout.py`` argument repeated: reach for the
module that *owns* a definition rather than restating it, so there is one answer
to "how big may an atlas be" and one to "where does the alpha stop".

**A layout is derived, never stored.** The document holds sources and settings;
the packer is deterministic, so re-deriving is what makes re-export of an
unchanged document byte-identical without a cache anybody has to invalidate.
"""

from __future__ import annotations

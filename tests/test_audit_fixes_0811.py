"""Regressions for the smaller fixes of the 2026-08-11 audit.

Each of these is a promise the codebase already made somewhere and did not keep
here: a staging file that outlived the failure it was staging against, and a
sweep unit admitted against half the door every other submission passes.

The retexture strand is pinned by source text rather than by behaviour, for the
reason ``test_vram.py``'s winjob scan is: the failure it guards needs a Blender
projection bake to have half-succeeded, which no fake can produce honestly, and
what actually has to stay true is a one-line structural property of the handler.
"""

from __future__ import annotations

import inspect

from warlock import queue as queue_mod
from warlock import rigging


def test_the_retexture_staging_file_is_cleaned_up_on_every_path():
    """``.retexture.tmp.glb`` lands in the *source* job's directory, beside the
    mesh it is a candidate replacement for. A swap that raised -- which it can
    now do, since an I/O failure surfaces as OSError rather than False -- used
    to leave a stale half-skinned GLB next to model.glb until the next
    re-texture happened to overwrite it."""
    source = inspect.getsource(queue_mod.Worker._retexture)
    at = source.index("rigging.RETEXTURE_GLB_TMP")
    window = source[at : at + 1000]
    assert "finally:" in window, "the staging name is not wrapped in a cleanup"
    assert "unlink" in window, "the cleanup does not remove the staging file"
    # Named here as well, so renaming the constant cannot quietly unpin this.
    assert rigging.RETEXTURE_GLB_TMP.endswith(".tmp.glb")

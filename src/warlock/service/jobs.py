"""Creating, resubmitting, editing and removing jobs -- a facade over five siblings.

This module was 1,446 lines covering five unrelated subjects, and it is now the
front of them: ``_jobs_create``, ``_jobs_list``, ``_jobs_lifecycle``,
``_jobs_resubmit`` and ``_jobs_rework``. Sibling *files* rather than a
``service/jobs/`` package, which is this codebase's naming everywhere else.

**Why a facade and not a rename.** Every caller in the repo -- panes, API,
sweeps and tests alike -- imports this module and calls by attribute
(``svc_jobs.create_job(...)``), and several tests monkeypatch names *on it*:
``create_job``, ``import_mesh``, ``list_jobs``, ``storage_sizes`` and
``MAX_LIST_LIMIT``. Re-exporting keeps every one of those patches landing where
it always did. The two that are read as module globals rather than called --
``MAX_LIST_LIMIT`` in ``list_jobs`` and ``prune_jobs`` -- resolve it back
through this module at call time for exactly that reason, with a comment at
each site.

**The accepted caveat**: intra-package calls bind at import, not through here.
``_jobs_lifecycle.update_job`` calls ``_jobs_list.get_job`` directly, and
``_jobs_resubmit.promote_candidates`` calls ``promote_to_model`` in its own
module -- so a facade patch of *those* would not redirect them. That was true
of the single-module version too (a patched ``jobs.get_job`` never redirected
``update_job``'s internal call either), no test relies on it, and the
alternative is a lazy self-import on every internal call for a redirection
nobody has ever asked for.

The names below that come from ``errors``, ``files`` and ``validation`` are not
part of the split: they have always been importable from here, and a caller
that learned that spelling should not have to learn another.
"""

from __future__ import annotations

import logging

from . import verdicts as verdicts_mod  # noqa: F401  -- historically importable
from ._jobs_create import (  # noqa: F401  -- the facade's re-export
    _normalize_guidance,
    _resolve_profile,
    create_job,
    import_mesh,
    import_reference,
    resolve_profile,
)
from ._jobs_lifecycle import (  # noqa: F401  -- the facade's re-export
    _refuse_if_busy,
    cancel_job,
    clean_jobs,
    delete_job,
    dependent_jobs,
    empty_trash,
    prune_jobs,
    restore_job,
    retained_job_ids,
    trash_job,
    trash_size,
    update_job,
    worker_is_inside,
)
from ._jobs_list import (  # noqa: F401  -- the facade's re-export
    get_job,
    list_jobs,
    storage,
    storage_sizes,
)
from ._jobs_resubmit import (  # noqa: F401  -- the facade's re-export
    keep_candidate,
    promote_candidates,
    promote_to_model,
    rerun_job,
)
from ._jobs_rework import (  # noqa: F401  -- the facade's re-export
    optimize_job,
    retexture_job,
    stale_rig_artifacts,
    stale_surface_artifacts,
)
from .core import WarlockService  # noqa: F401  -- historically importable
from .errors import (  # noqa: F401  -- historically importable from here
    Conflict,
    Failed,
    Invalid,
    NotFound,
    TooLarge,
    invalid_from,
)
from .files import (  # noqa: F401  -- historically importable from here
    ImageTooLarge,
    attach_files,
    dir_size,
    measure_storage,
    to_png,
)
from .validation import (  # noqa: F401  -- historically importable from here
    ALLOWED_RESOLUTIONS,
    CONDITIONING_PARAMS,
    DERIVED_PARAMS,
    MAX_JOB_NAME,
    MAX_LIST_LIMIT,
    MAX_MESH_BYTES,
    MAX_MESH_CANDIDATES,
    MAX_PROMPT,
    MAX_REFERENCE_COUNT,
    MAX_UPLOAD_BYTES,
    check_glb,
    check_job_id,
    check_seed,
    check_trellis_band,
    check_trellis_tex_res,
    check_vram,
    check_weights,
    normalize_tags,
    not_done_message,
    random_seed,
    valid_template,
)

# ``jobs.log`` has always been importable, and the sibling modules each have
# their own (``warlock.service._jobs_*``). Nothing asserts a logger name.
log = logging.getLogger(__name__)

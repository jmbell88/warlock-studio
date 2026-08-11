"""Reading jobs back: one page of history, one row, and what disk they hold.

Split out of ``service/jobs.py``, which had grown to 1,446 lines over five
unrelated subjects; ``jobs.py`` stays as the facade every caller still imports
and calls by attribute -- which matters here more than anywhere, because
``list_jobs`` and ``storage_sizes`` are the two the panes monkeypatch.

``list_jobs`` runs twice a second over a page of rows, so everything expensive
in it is either cached or paged: the ceiling on a single read is
``MAX_LIST_LIMIT`` and a longer history is reached by paging, never by asking
for more. It is read through the facade at call time -- ``tests/test_api.py``
patches it on ``service.jobs``, which is where it has always been patchable,
and an early-bound copy here would ignore that.
"""

from __future__ import annotations

import logging
from typing import Any

from .core import WarlockService
from .files import attach_files, measure_storage

log = logging.getLogger(__name__)


def list_jobs(
    svc: WarlockService,
    limit: int = 100,
    before: tuple[float, str] | None = None,
    *,
    files_cache: dict | None = None,
) -> list[dict[str, Any]]:
    """One page of history, newest first. ``before`` is the (created_at, id) of
    the last row of the previous page; MAX_LIST_LIMIT stays the ceiling on a
    single read, so a longer history is reached by paging rather than by asking
    for more at once.

    ``files_cache`` is handed straight to ``attach_files`` and is what makes
    this affordable to call twice a second from the frame loop -- see there.
    """
    # Through the facade, at call time: ``tests/test_api.py`` patches
    # MAX_LIST_LIMIT on ``service.jobs``, which is where it has always been
    # patchable, and an early-bound copy of the constant here would ignore it.
    from . import jobs as _facade

    limit = max(1, min(limit, _facade.MAX_LIST_LIMIT))
    jobs = svc.store.list(limit, before)
    for job in jobs:
        attach_files(job, svc.job_dir(job["id"]), cache=files_cache)
        svc.attach_progress(job)
    return jobs


def get_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    job = svc.require_job(job_id)
    attach_files(job, svc.job_dir(job_id))
    svc.attach_progress(job)
    return job


def storage(svc: WarlockService) -> dict[str, Any]:
    """How much disk the generated assets are using.

    Jobs and their artifacts accumulate forever otherwise -- at 5-20 MB per GLB
    that is real disk within weeks of regular use.
    """
    return measure_storage(svc.config.data_dir)


def storage_sizes(svc: WarlockService) -> dict[str, int]:
    """The same walk, per job directory -- what incremental accounting needs."""
    from .files import storage_sizes as _sizes

    return _sizes(svc.config.data_dir)

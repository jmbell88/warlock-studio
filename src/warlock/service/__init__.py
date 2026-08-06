"""The app's business logic, independent of any transport.

Everything here used to live inside ``app.py``'s route bodies. It is plain
synchronous Python that raises :mod:`~warlock.service.errors` exceptions
instead of ``HTTPException``, so the desktop UI can call it from a worker
thread and the (transitional) HTTP routes can map the same exceptions back to
status codes.

Functions take a :class:`~warlock.service.core.WarlockService` as their first
argument rather than hanging off it as methods: the split into modules is by
subject (jobs, rigs, sheets, ...) and a single 700-line class would only put
that structure back into one file.
"""

from .core import WarlockService
from .errors import (
    Conflict,
    Failed,
    Invalid,
    NotFound,
    NotReady,
    ServiceError,
    TooLarge,
)

__all__ = [
    "Conflict",
    "Failed",
    "Invalid",
    "NotFound",
    "NotReady",
    "ServiceError",
    "TooLarge",
    "WarlockService",
]

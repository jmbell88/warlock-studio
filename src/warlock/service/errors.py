"""What the service layer raises instead of ``HTTPException``.

One class per distinguishable outcome, not one per status code: the desktop UI
shows ``exc.message`` in a toast and mostly cares only that it failed, while
the HTTP shim maps each class to the code the route used to return. Keeping
the hierarchy this small is what makes that mapping mechanical -- ``_to_http``
is a dict lookup, so an extraction that changed a status code would be a
visible edit rather than a silent one.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base for every expected failure. ``message`` is user-facing."""

    status = 400

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        # Which form control was at fault, when the caller can know. The UI
        # highlights it; HTTP has nowhere to put it and drops it.
        self.field = field


class NotFound(ServiceError):
    """No such job / pose / sheet / file. Also covers a malformed id: a caller
    that supplies one cannot tell the difference, and saying so leaks less."""

    status = 404


class Invalid(ServiceError):
    """The request is well-formed but its values are not usable."""

    status = 400


class Conflict(ServiceError):
    """The object exists but is in the wrong state (running, at its limit)."""

    status = 409


class NotReady(NotFound):
    """The artifact is not on disk yet, or is still being written.

    A subclass of NotFound because that is the status the routes returned for
    it, and because "not ready" and "not there" are the same fact to a caller
    that can only retry. It exists separately so the UI can say *why* a
    download is disabled instead of just greying it out.
    """


class TooLarge(ServiceError):
    """An upload is over its byte cap, refused before it is decoded."""

    status = 413


class Failed(ServiceError):
    """A subprocess or conversion that should have worked, didn't.

    Blender is installed and the bake still died, gltfpack ran and returned
    garbage. The user's only useful response is to retry or report it, which is
    exactly what a 500 said.
    """

    status = 500

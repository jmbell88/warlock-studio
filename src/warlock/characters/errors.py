"""The one refusal type the character layer raises.

A ``ValueError`` so every existing ``except ValueError`` door still catches it,
and carrying a ``field`` so ``service.errors.invalid_from`` can hand the UI the
name of the control to point at. That passthrough is not incidental:
``invalid_from`` reads ``getattr(exc, "field", None)`` off whatever it wraps
(the S137 arrangement ``guidance.GuidanceError`` already relies on), so a
refusal raised three modules deep still lands on the right slider without any
service module restating the mapping.
"""

from __future__ import annotations


class CharacterError(ValueError):
    """A character request that was refused, and the control it came from.

    ``field`` is the recipe key -- ``"family"``, ``"theme"``, ``"appearance"``,
    ``"animations"`` -- never a wire-format path, because the only useful thing
    to do with it is to highlight a control the user can see.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field

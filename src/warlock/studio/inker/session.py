"""What the 2D reference editor keeps between frames.

Transitional: the standalone Paint mode replaces this with a multi-document
state of its own. It stays here, unchanged in shape, so the pane that uses it
did not have to be rewritten in the same commit as the engine under it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

RGBA = tuple[int, int, int, int]


@dataclass
class EditorSession:
    """Pinned to a job id *and* a path, so that changing the library selection
    while the editor is open does not quietly retarget the save."""

    job_id: str
    path: Path
    doc: Any
    tool: str = "brush"
    color: RGBA = (0, 0, 0, 255)
    brush_size: int = 8
    shape_filled: bool = False
    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)
    dirty: bool = False
    saving: bool = False
    has_original: bool = False
    drag_anchor: tuple[int, int] | None = None
    last_point: tuple[int, int] | None = None
    # What the current mouse drag means: "" (nothing), "paint", "shape",
    # "marquee" or "move". Decided on press, because a selection drag and a new
    # marquee start with the same button in the same tool.
    drag_kind: str = ""
    fitted: bool = False

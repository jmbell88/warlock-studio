"""The 2D editor's document model: pixels, history and a floating selection.

Pure Pillow, like ``pipelines/sheet.py`` is pure geometry -- no imgui, no GL,
no service. The pane above it decides *when* a stroke happens; everything about
*what* a stroke does to the image is here, which is what makes an editor
testable without a window.

Two decisions worth stating. Undo is whole-image snapshots rather than a diff
or a command log: a reference is at most a few megapixels, an SDXL flood fill
can touch nearly all of it, and a snapshot cannot be wrong about what it
restores. And the eraser has no colour picker -- ``erase_color`` is decided
once, when the document loads, from whether the image has any transparency at
all, because "erase" means transparent on a sprite and white on a photo and
guessing per stroke would silently change the semantics mid-edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RGBA = tuple[int, int, int, int]

TRANSPARENT: RGBA = (0, 0, 0, 0)
OPAQUE_WHITE: RGBA = (255, 255, 255, 255)

MIN_BRUSH = 1
MAX_BRUSH = 64

# Undo is snapshots, so the cap that matters is bytes rather than steps. The
# depth bounds are the other half: a huge image must still get a few levels,
# and a tiny one must not get hundreds.
UNDO_BYTES = 192 * 1024 * 1024
UNDO_MAX_DEPTH = 32
UNDO_MIN_DEPTH = 8

SHAPES = ("line", "rect", "ellipse")


def clamp_brush(size: int) -> int:
    return max(MIN_BRUSH, min(MAX_BRUSH, int(size)))


def normalise_rect(p0: tuple[int, int], p1: tuple[int, int]) -> tuple[int, int, int, int]:
    """Two corners in any order -> (x0, y0, x1, y1) with x0 <= x1."""
    x0, y0 = p0
    x1, y1 = p1
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class UndoStack:
    """Whole-image snapshots, bounded by bytes rather than by count."""

    def __init__(self, budget: int = UNDO_BYTES) -> None:
        self.budget = budget
        self._done: list[Any] = []
        self._undone: list[Any] = []

    def __len__(self) -> int:
        return len(self._done)

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    def _depth(self, image: Any) -> int:
        cost = max(image.width * image.height * 4, 1)
        return max(UNDO_MIN_DEPTH, min(UNDO_MAX_DEPTH, self.budget // cost))

    def push(self, image: Any) -> None:
        self._done.append(image.copy())
        # A new edit invalidates the redo branch: there is no tree here, and a
        # redo onto a different history is worse than no redo at all.
        self._undone.clear()
        while len(self._done) > self._depth(image):
            self._done.pop(0)

    def undo(self, current: Any) -> Any | None:
        if not self._done:
            return None
        self._undone.append(current.copy())
        return self._done.pop()

    def redo(self, current: Any) -> Any | None:
        if not self._undone:
            return None
        self._done.append(current.copy())
        return self._undone.pop()

    def drop(self) -> None:
        """Discard the most recent snapshot without restoring it.

        For an operation that undid itself by hand (a cancelled selection):
        the snapshot describes a state the user can already see, and leaving it
        on the stack makes the next Ctrl+Z appear to do nothing.
        """
        if self._done:
            self._done.pop()

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()


@dataclass
class Selection:
    """A rectangle lifted off the canvas and floating over it.

    ``chunk`` is the pixels that were there; the hole they left is already
    filled with ``erase_color``. Committing pastes them back at ``origin``,
    cancelling restores the snapshot taken at lift time -- which is why a
    cancel is exact rather than approximately exact.
    """

    chunk: Any
    origin: tuple[int, int]
    source: tuple[int, int]
    rev: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return (self.chunk.width, self.chunk.height)

    def moved(self, dx: int, dy: int) -> None:
        self.origin = (self.origin[0] + dx, self.origin[1] + dy)


@dataclass
class Document:
    image: Any
    erase_color: RGBA = TRANSPARENT
    rev: int = 0
    path: Path | None = None
    selection: Selection | None = None
    history: UndoStack = field(default_factory=UndoStack)
    _pre_selection: Any = None

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path: Path, *, budget: int = UNDO_BYTES) -> Document:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            image = im.convert("RGBA")
        return cls(
            image=image,
            erase_color=erase_color_for(image),
            path=Path(path),
            history=UndoStack(budget),
        )

    @property
    def size(self) -> tuple[int, int]:
        return (self.image.width, self.image.height)

    def png_bytes(self) -> bytes:
        """The document as a PNG. Blocking; the pane calls this off-thread."""
        import io

        buf = io.BytesIO()
        self.image.save(buf, "PNG")
        return buf.getvalue()

    # -- history -----------------------------------------------------------

    def checkpoint(self) -> None:
        """Snapshot before a mutation. Every editing entry point calls this
        first, so an operation is one undo step however many pixels it wrote."""
        self.history.push(self.image)

    def _touch(self) -> None:
        self.rev += 1

    def undo(self) -> bool:
        self.cancel_selection()
        restored = self.history.undo(self.image)
        if restored is None:
            return False
        self.image = restored
        self._touch()
        return True

    def redo(self) -> bool:
        self.cancel_selection()
        restored = self.history.redo(self.image)
        if restored is None:
            return False
        self.image = restored
        self._touch()
        return True

    # -- drawing -----------------------------------------------------------

    def _draw(self) -> Any:
        from PIL import ImageDraw

        # RGBA over RGBA has to *replace*, not composite: painting with a
        # half-transparent colour, or erasing to transparent, must set the
        # pixel rather than blend into what was under it.
        return ImageDraw.Draw(self.image, "RGBA")

    def stroke(
        self,
        p0: tuple[int, int],
        p1: tuple[int, int],
        color: RGBA,
        size: int,
        *,
        erase: bool = False,
    ) -> None:
        """One segment of a freehand stroke, with round caps.

        The caps are not decoration: ``line`` with a width joins two segments
        with a mitre gap, so a fast drag across the canvas comes out as a
        dotted line without them.
        """
        size = clamp_brush(size)
        paint = self.erase_color if erase else color
        draw = self._draw()
        draw.line([tuple(p0), tuple(p1)], fill=paint, width=size)
        if size > 1:
            radius = size / 2.0
            for x, y in (p0, p1):
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius], fill=paint
                )
        self._touch()

    def fill(self, xy: tuple[int, int], color: RGBA, *, thresh: int = 32) -> None:
        """Flood fill from a point.

        The threshold is the whole reason this is usable on generated art: an
        SDXL image has no flat regions, only nearly-flat ones behind an
        antialiased edge, and an exact-match fill stops after a few hundred
        pixels every time.
        """
        from PIL import ImageDraw

        if not self.in_bounds(xy):
            return
        ImageDraw.floodfill(self.image, (int(xy[0]), int(xy[1])), color, thresh=thresh)
        self._touch()

    def shape(
        self,
        kind: str,
        p0: tuple[int, int],
        p1: tuple[int, int],
        color: RGBA,
        size: int,
        *,
        filled: bool = False,
    ) -> None:
        if kind not in SHAPES:
            raise ValueError(f"unknown shape {kind!r}")
        size = clamp_brush(size)
        draw = self._draw()
        if kind == "line":
            draw.line([tuple(p0), tuple(p1)], fill=color, width=size)
        else:
            box = normalise_rect(p0, p1)
            method = draw.rectangle if kind == "rect" else draw.ellipse
            if filled:
                method(box, fill=color, outline=color, width=size)
            else:
                method(box, fill=None, outline=color, width=size)
        self._touch()

    def eyedrop(self, xy: tuple[int, int]) -> RGBA | None:
        if not self.in_bounds(xy):
            return None
        r, g, b, a = self.image.getpixel((int(xy[0]), int(xy[1])))
        return (r, g, b, a)

    def in_bounds(self, xy: tuple[int, int]) -> bool:
        x, y = int(xy[0]), int(xy[1])
        return 0 <= x < self.image.width and 0 <= y < self.image.height

    # -- selection ---------------------------------------------------------

    def _clip(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = rect
        x0 = max(0, min(int(x0), self.image.width))
        y0 = max(0, min(int(y0), self.image.height))
        x1 = max(0, min(int(x1), self.image.width))
        y1 = max(0, min(int(y1), self.image.height))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        return (x0, y0, x1, y1)

    def delete_rect(self, rect: tuple[int, int, int, int]) -> None:
        box = self._clip(rect)
        if box is None:
            return
        self.checkpoint()
        # PIL's rectangle bounds are inclusive; _clip returns a half-open box.
        self._draw().rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=self.erase_color)
        self._touch()

    def lift_selection(self, rect: tuple[int, int, int, int]) -> bool:
        """Cut a rectangle out and start floating it. One undo step."""
        self.commit_selection()
        box = self._clip(rect)
        if box is None:
            return False
        self.checkpoint()
        self._pre_selection = self.image.copy()
        chunk = self.image.crop(box)
        self._draw().rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=self.erase_color)
        self.selection = Selection(chunk=chunk, origin=(box[0], box[1]), source=(box[0], box[1]))
        self._touch()
        return True

    def move_selection(self, dx: int, dy: int) -> None:
        if self.selection is None:
            return
        self.selection.moved(int(dx), int(dy))
        self.selection.rev += 1

    def commit_selection(self) -> bool:
        """Paste the floating chunk where it now sits, and stop floating."""
        if self.selection is None:
            return False
        self.image.paste(self.selection.chunk, self.selection.origin)
        self.selection = None
        self._pre_selection = None
        self._touch()
        return True

    def cancel_selection(self) -> bool:
        """Put the canvas back exactly as it was before the lift."""
        if self.selection is None:
            return False
        if self._pre_selection is not None:
            self.image = self._pre_selection
            # The lift pushed a snapshot; undoing the lift by hand means that
            # snapshot no longer describes a step the user can see.
            self.history.drop()
        self.selection = None
        self._pre_selection = None
        self._touch()
        return True

    def delete_selection(self) -> bool:
        """Drop the floating chunk instead of pasting it: the hole it left is
        already erase-coloured, so this is just "don't put it back"."""
        if self.selection is None:
            return False
        self.selection = None
        self._pre_selection = None
        self._touch()
        return True


def erase_color_for(image: Any) -> RGBA:
    """Transparent if the image has any transparency, else opaque white.

    A sprite on a matted background must stay erasable to nothing, or the
    eraser breaks trellis's background detection; a photo has no alpha channel
    the user is thinking about, and erasing it to transparent would change the
    file's semantics without ever showing them the difference.
    """
    if image.mode != "RGBA":
        return OPAQUE_WHITE
    alpha = image.getchannel("A")
    return TRANSPARENT if alpha.getextrema()[0] < 255 else OPAQUE_WHITE


@dataclass
class EditorSession:
    """Everything the pane needs to keep between frames.

    Pinned to a job id *and* a path so that changing the library selection
    while the editor is open does not quietly retarget the save.
    """

    job_id: str
    path: Path
    doc: Document
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
    # "marquee" or "move". Decided on press, because a selection drag and a
    # new marquee start with the same button in the same tool.
    drag_kind: str = ""
    fitted: bool = False

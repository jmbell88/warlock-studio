"""Frame-rate measurement, kept pure so it can be asserted without a window.

The frame loop is the one thing in the app with a hard timing contract --
``clock.tick(TARGET_FPS)`` -- and nothing until now reported whether it was
being met. This measures it two ways at once, because they answer different
questions: a rolling mean says "is the loop keeping up right now", and the
worst frame in the same window says "did anything stall", which a mean over
120 frames hides almost completely (one 100 ms hitch moves a 60 fps mean to
59.5).

Fed the same ``dt`` the rest of the frame is, so it measures wall-clock frame
period including the sleep ``clock.tick`` does -- which is exactly what a
"locked to 60" claim is about.
"""

from __future__ import annotations

from collections import deque

# 120 frames is two seconds at target: long enough that the number is readable
# rather than flickering, short enough to react before a stall scrolls away.
WINDOW = 120


class FpsMeter:
    """A rolling window of frame durations, plus a session-long total."""

    def __init__(self, window: int = WINDOW) -> None:
        self._window = max(int(window), 1)
        self._samples: deque[float] = deque(maxlen=self._window)
        self.frames = 0
        self.elapsed = 0.0

    def record(self, dt: float) -> None:
        # A non-positive dt is a clock that did not advance (two frames inside
        # one timer tick), not a frame that took no time: counting it would
        # divide the window by less than it contains.
        if dt <= 0.0:
            return
        self._samples.append(dt)
        self.frames += 1
        self.elapsed += dt

    @property
    def fps(self) -> float:
        """Frames per second over the window, 0.0 before the first sample."""
        if not self._samples:
            return 0.0
        return len(self._samples) / sum(self._samples)

    @property
    def frame_ms(self) -> float:
        """Mean frame period over the window, in milliseconds."""
        if not self._samples:
            return 0.0
        return 1000.0 * sum(self._samples) / len(self._samples)

    @property
    def worst_ms(self) -> float:
        """The longest single frame in the window, in milliseconds."""
        if not self._samples:
            return 0.0
        return 1000.0 * max(self._samples)

    @property
    def average_fps(self) -> float:
        """Frames per second across the whole session, not just the window."""
        if self.elapsed <= 0.0:
            return 0.0
        return self.frames / self.elapsed

    def label(self) -> str:
        return f"{self.fps:5.1f} fps   {self.frame_ms:4.1f} ms   peak {self.worst_ms:4.1f} ms"

    def summary(self) -> str:
        """One line for the log at teardown: the evidence for a 60 fps claim."""
        return (
            f"{self.frames} frames in {self.elapsed:.1f}s "
            f"({self.average_fps:.1f} fps average)"
        )

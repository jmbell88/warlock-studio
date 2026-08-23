"""What the machine has left, sampled on a cadence and formatted as one line.

The status bar's meter, extracted from it for the reason ``status_bar.items``
is data rather than draw calls: the sampling and the formatting are testable
without a GL context, and the pane is left with a string and a position.

**This is the reversal of the wave-3 deletion of the shell header's resource
readout**, and the argument for it is recorded in full in
``tests/test_ux_phases.py``'s navigation-control test. In short: a different
surface (ambient state, not primary chrome), a different question ("can I
start a 7 GB generation right now", which ``check_vram`` forces on the user
and nothing on screen answered), opt-out and persisted, and it yields --
dropped first when the window narrows.

Imgui-free on purpose, the way ``status_bar.items`` is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .. import memlog, vram

#: How often the three readings are retaken. One second is the slowest cadence
#: at which a meter still reads as *live* while a generation is loading, and
#: the cost is bounded (see :meth:`Sampler.sample`).
TICK_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class Reading:
    """One sample. Every field is optional: a figure absent is a figure omitted.

    ``vram_*`` is None on any card NVML does not serve, and ``cpu`` is None on
    the first sample -- the reading is a delta and has no interval yet.
    """

    vram_used_gib: float | None = None
    vram_total_gib: float | None = None
    ram_used_gib: float | None = None
    ram_total_gib: float | None = None
    cpu: float | None = None

    def text(self) -> str:
        """The status-bar line. Empty when nothing could be read at all."""
        parts: list[str] = []
        if self.vram_total_gib:
            parts.append(f"VRAM {self.vram_used_gib:.1f}/{self.vram_total_gib:.0f}")
        if self.ram_total_gib:
            parts.append(f"RAM {self.ram_used_gib:.1f}/{self.ram_total_gib:.0f}")
        if self.cpu is not None:
            parts.append(f"CPU {self.cpu * 100:.0f}%")
        return "   ".join(parts)


class Sampler:
    """Holds the CPU baseline and the cadence. One per app.

    A class because :class:`memlog.CpuSampler` is one: the CPU figure is a
    delta between calls, so two owners sharing a baseline would each consume
    the other's interval.
    """

    def __init__(self) -> None:
        self._cpu = memlog.CpuSampler()
        self._last = 0.0
        self.reading = Reading()

    def sample(self) -> Reading:
        """Take all three readings now. Blocking only in the syscall sense.

        Two ctypes calls and one driver ioctl -- no filesystem, no network,
        nothing that can queue behind another process. That is what licenses
        calling it from the frame loop at all; ``_memory_ticker``'s docstring
        makes the same argument for its own two calls once per 30 s.

        **Measured at 0.047 ms** (RTX 5090, 2026-08-23, mean of 50). The first
        draft was 15.7 ms, all of it ``nvmlInit_v2``/``nvmlShutdown`` around
        each reading -- which is why ``vram._nvml`` holds the session open. If
        this ever exceeds ~1 ms again the sampler moves to the
        ``warlock-loop`` thread rather than getting a longer cadence.
        """
        gpu = vram.live_memory()
        ram = memlog.physical_memory()
        self.reading = Reading(
            vram_used_gib=None if gpu is None else max(gpu.total_gib - gpu.free_gib, 0.0),
            vram_total_gib=None if gpu is None else gpu.total_gib,
            ram_used_gib=None if ram is None else ram.used,
            ram_total_gib=None if ram is None else ram.total,
            cpu=self._cpu.sample(),
        )
        return self.reading

    def tick(self, now: float | None = None) -> Reading:
        """Resample if :data:`TICK_SECONDS` have passed; otherwise the last one.

        The caller gates on the *setting* before calling this, so the opt-out
        costs nothing at all rather than costing a comparison.
        """
        stamp = time.perf_counter() if now is None else now
        if stamp - self._last < TICK_SECONDS:
            return self.reading
        self._last = stamp
        return self.sample()

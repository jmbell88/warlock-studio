"""Live progress model for the generation pipeline.

trellis-server.exe prints a detailed stage trace to stdout; we pipe it, parse it,
and turn it into a single 0-100 percentage plus a human label. Two properties matter:

* **Monotonic.** The bar never goes backwards, even though the stage denominator
  switches from /6 to /7 mid-run and stage 4 contains two separate flow runs.
* **Never frozen.** Most of the wall clock is spent in stages that print nothing.
  A res-1024 run spends 64% of its time in stage 7, including one ~44 s silent
  gap inside xatlas packing. Discrete milestones alone would leave the bar stuck
  for most of the job, so every milestone also carries a nominal duration and
  ``snapshot()`` eases the reported percent toward the next milestone over time.

Progress is deliberately in-memory only: it is meaningless once a job ends, so
persisting it would only add DB traffic (and a reader-thread writer) for data
nobody reads back. ``JobStore`` serialises its own connection with an RLock,
but this module still never touches it -- see ``ProgressBus``.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any

log = logging.getLogger(__name__)

# --- phase weights ----------------------------------------------------------
# Fraction of the overall bar each phase owns, as (start, end).
# SDXL-Turbo is 4 steps (~1.5 s of GPU work); its cost is loading, not sampling.

#
# "optimize", "scale" and "audit" are zero-width on purpose. The gltfpack
# retarget is a two-second subprocess, rescaling the finished GLB (see
# pipelines/postprocess.normalize_glb) is sub-second JSON surgery and the mesh
# audit (meshaudit.hole_fraction) is a few seconds of CPU rasterisation --
# none of it is GPU work, so each gets a label at 100% rather than a slice of
# the bar. Reserving one would only make every job stall just short of done.
#
# Any phase named here must exist in both dicts: update() falls back to
# (0.0, 1.0) for an unknown phase, which would drag the bar back to zero.

PHASES_TEXT: dict[str, tuple[float, float]] = {
    # No download phase: weights are always local (HF_HUB_OFFLINE=1), so the
    # bar starts at the load.
    "t2i_load": (0.00, 0.16),
    "t2i_sample": (0.16, 0.20),
    "trellis": (0.20, 1.00),
    "optimize": (1.00, 1.00),
    "scale": (1.00, 1.00),
    "audit": (1.00, 1.00),
}
PHASES_IMAGE: dict[str, tuple[float, float]] = {
    "trellis": (0.00, 1.00),
    "optimize": (1.00, 1.00),
    "scale": (1.00, 1.00),
    "audit": (1.00, 1.00),
}
# Rig jobs are one Blender subprocess that reports its own 0..1 fraction, so
# the phase owns the whole bar and the worker's own milestones do the shaping.
PHASES_RIG: dict[str, tuple[float, float]] = {"rig": (0.00, 1.00)}
# A sheet job is the same shape -- one Blender subprocess reporting its own
# fraction -- plus a short packing tail on the host once the frames land.
PHASES_SHEET: dict[str, tuple[float, float]] = {
    "sheet": (0.00, 0.95),
    "pack": (0.95, 1.00),
}

_PHASES_BY_KIND: dict[str, dict[str, tuple[float, float]]] = {
    "text": PHASES_TEXT,
    "image": PHASES_IMAGE,
    "rig": PHASES_RIG,
    "sheet": PHASES_SHEET,
}


def phases_for(kind: str) -> dict[str, tuple[float, float]]:
    return _PHASES_BY_KIND.get(kind, PHASES_IMAGE)


# --- trellis stage weights --------------------------------------------------
# Keyed on the (k, n) pair exactly as printed, which sidesteps the 6 -> 7
# denominator switch without any monotonicity math.
#
# Calibrated from a measured 118.8 s cold res-1024 run (tests/fixtures/trellis_1024.log):
#   stages 1-3 19.8 s | 4: 9.1 s | 5: 5.6 s | 6: 8.0 s | 7: 76.3 s
# Stage 1 is inflated on a cold server because the ~8 GB CUDA load happens inside
# it. res 512/1536 shift stage 7 the most. These are estimates, not guarantees --
# the creep logic below is what keeps the bar honest when they are wrong.

TRELLIS_STAGES: dict[tuple[int, int], tuple[str, float]] = {
    (1, 6): ("Preprocessing image", 0.08),
    (2, 6): ("Conditioning", 0.03),
    (3, 6): ("Sparse structure", 0.05),
    (4, 7): ("Shape refinement", 0.08),
    (5, 7): ("Mesh decode", 0.05),
    (6, 7): ("Texture", 0.07),
    (7, 7): ("Cleanup + UV unwrap", 0.64),
}

# Cumulative offset of each stage within the trellis phase, in declaration order.
_STAGE_OFFSETS: dict[tuple[int, int], tuple[float, float]] = {}
_acc = 0.0
for _key, (_lbl, _w) in TRELLIS_STAGES.items():
    _STAGE_OFFSETS[_key] = (_acc, _acc + _w)
    _acc += _w
del _acc, _key, _lbl, _w

# Default seconds-to-next-event used when a stage has no finer breakdown.
DEFAULT_NOMINAL = 8.0

# How much of a stage the [flow] denoise loop accounts for. Stage 6 spends ~3.5 s
# in a silent PBR decode after its flow run finishes, so the flow must not run the
# bar to the end of the stage or the trailing milestone would sit below it.
FLOW_SPAN: dict[tuple[int, int], float] = {(6, 7): 0.55}

# On entering a stage the next real signal (a flow step or the first milestone)
# normally lands within a couple of seconds, so creep only a short way in.
ENTRY_CREEP = 0.10
ENTRY_NOMINAL = 4.0

# Stages 1 and 2 print nothing at all between their header and the next stage
# header -- no flow lines, no milestones. Stage 1 in particular absorbs the ~8 GB
# CUDA load on a cold server (~14 s measured, vs ~2 s warm), which is the first
# thing a user ever sees, so it gets a near-full creep budget instead.
ENTRY_OVERRIDES: dict[tuple[int, int], tuple[float, float]] = {
    (1, 6): (0.92, 14.0),
    (2, 6): (0.92, 3.0),
}


# --- line patterns ----------------------------------------------------------
# Verified against the exe's format strings. The flow bar is printed as
#   "%s      [flow] [%s] %2d/%d  %5.1fs  %-12s%s"
# so the step is right-aligned (hence \s*) and the status field is padded to 12.

RE_START = re.compile(r"^\[trellis-server\] generate: (\d+)-byte image, seed (\d+), res (\S+)")
RE_STAGE = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)$")
RE_FLOW = re.compile(r"^\s*\[flow\] \[[#.]*\]\s*(\d+)/(\d+)\s+([\d.]+)s\s+(.*?)\s*$")
RE_FLOWEND = re.compile(r"^\s*\[flow\] (\d+) steps, (\d+) forwards, ([\d.]+)s\s*$")
RE_ETA = re.compile(r"~(\d+)s left")
RE_DONE = re.compile(r"^done in ([\d.]+)s -> ")
RE_FAIL = re.compile(r"^\[trellis-server\] generate failed: (.*)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Milestone:
    """A line the server prints partway through a stage.

    ``frac`` is how far through the stage we are once it appears; ``nominal`` is
    how long the *next* piece of work typically takes, which drives the creep.
    """

    pattern: re.Pattern[str]
    label: str
    frac: float
    nominal: float


def _ms(pat: str, label: str, frac: float, nominal: float) -> Milestone:
    return Milestone(re.compile(pat), label, frac, nominal)


# Sub-stage milestones, ordered; first match wins and the index only moves
# forward. Fractions and nominals come from the measured run above.
# Note "decimated postproc:" must be tested after "decimate(" -- ordering handles it.
SUBSTEPS: dict[tuple[int, int], list[Milestone]] = {
    (5, 7): [
        _ms(r"^\s*decoded voxels @res", "decoding voxels", 0.65, 0.7),
        _ms(r"^\s*mesh V=\d+ F=\d+", "building mesh", 0.76, 1.4),
    ],
    (6, 7): [
        # The PBR decode after the flow run is ~3.5 s of silence.
        _ms(r"^\s*PBR voxels=", "PBR decode", 0.98, 0.5),
    ],
    (7, 7): [
        _ms(r"^\s*weld:", "welding vertices", 0.02, 3.5),
        _ms(r"^\s*fill_holes:", "filling holes", 0.07, 8.1),
        _ms(r"^\s*remesh(_dc:|: only )", "remeshing", 0.17, 5.9),
        _ms(r"^\s*\[clean\] faces=", "fixing winding", 0.25, 1.1),
        _ms(r"^\s*remesh postproc:", "cleanup", 0.26, 0.6),
        _ms(r"^\s*decimate(_qem)?(_gpu)?\(", "decimating", 0.27, 0.8),
        _ms(r"^\s*decimated postproc:", "cleanup", 0.28, 0.8),
        _ms(r"^\s*uv_bake: \d+ components", "uv unwrap", 0.28, 2.1),
        # xatlas packing: the longest silence in the pipeline and by far the most
        # variable -- two runs of the same image at res 1024 took 6 s and 44 s.
        # The nominal is deliberately between the two: tuned to the slow case the
        # bar barely moves on a fast run and then leaps ~28 points when the atlas
        # line lands; tuned to the fast case it saturates and sits still.
        _ms(r"^\s*uv_bake: \d+ merge clusters", "packing atlas", 0.31, 20.0),
        _ms(r"^\s*uv_bake: .*uncharted", "packing atlas", 0.88, 5.6),
        _ms(r"^\s*uv_bake: atlas", "baking texture", 0.95, 3.3),
        _ms(r"^\s*textured GLB", "writing glb", 0.99, 0.3),
    ],
}


# --- progress state ---------------------------------------------------------


@dataclass(slots=True)
class Progress:
    job_id: str
    kind: str
    phase: str = "queued"
    label: str = "Starting"
    detail: str = ""
    percent: float = 0.0
    stage_index: int | None = None
    stage_total: int | None = None
    step: int | None = None
    step_total: int | None = None
    stage_eta: float | None = None
    cold: bool = False
    error: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        # An explicit literal, not dataclasses.asdict: asdict walks the fields
        # through a recursive deep-copy machine, and this runs under the bus
        # lock on every snapshot -- which the frame loop takes every frame.
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "phase": self.phase,
            "label": self.label,
            "detail": self.detail,
            "percent": self.percent,
            "stage_index": self.stage_index,
            "stage_total": self.stage_total,
            "step": self.step,
            "step_total": self.step_total,
            "stage_eta": self.stage_eta,
            "cold": self.cold,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class _Creep:
    """Eases percent from ``floor`` toward ``ceil`` while no new line arrives."""

    floor: float = 0.0
    ceil: float = 0.0
    t0: float = field(default_factory=time.monotonic)
    tau: float = DEFAULT_NOMINAL

    def value(self) -> float:
        if self.ceil <= self.floor:
            return self.floor
        elapsed = time.monotonic() - self.t0
        # Asymptotic: approaches but never reaches ceil, so it can never overshoot
        # into a stage that hasn't started. Scaled so that one nominal duration
        # covers ~90% of the gap -- with a plain 1-exp(-t/nominal) it only reaches
        # 63%, which leaves a visible jump when the next milestone finally lands.
        tau = max(self.tau, 0.1) / 2.3
        return self.floor + (self.ceil - self.floor) * (1.0 - math.exp(-elapsed / tau))


class ProgressBus:
    """Thread-safe single-slot progress holder.

    Written by the trellis reader thread and by the text2image worker thread;
    read by the event loop. ``snapshot(job_id)`` returns ``None`` on an id
    mismatch, which prevents one job's progress ever being attributed to another.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._p: Progress | None = None
        self._creep = _Creep()

    def begin(self, job_id: str, kind: str, *, cold: bool = False) -> None:
        with self._lock:
            now = time.time()
            self._p = Progress(
                job_id=job_id,
                kind=kind,
                phase="starting",
                label="Starting",
                cold=cold,
                started_at=now,
                updated_at=now,
            )
            self._creep = _Creep(0.0, 0.0, time.monotonic(), DEFAULT_NOMINAL)

    def end(self, job_id: str) -> None:
        with self._lock:
            if self._p is not None and self._p.job_id == job_id:
                self._p = None

    def update(
        self,
        job_id: str,
        *,
        phase: str,
        label: str,
        inner: float,
        inner_next: float | None = None,
        nominal: float = DEFAULT_NOMINAL,
        **fields: Any,
    ) -> None:
        """Record progress at ``inner`` (0..1) within ``phase``.

        ``inner_next`` is where the next known milestone sits; the reported
        percent eases toward it over ``nominal`` seconds so the bar keeps moving
        through silent stretches.
        """
        with self._lock:
            p = self._p
            if p is None or p.job_id != job_id:
                return
            lo, hi = phases_for(p.kind).get(phase, (0.0, 1.0))
            span = hi - lo
            floor = (lo + span * _clamp01(inner)) * 100.0
            ceil = (lo + span * _clamp01(inner_next)) * 100.0 if inner_next is not None else floor

            floor = max(floor, self._creep.value())  # never regress
            p.percent = floor
            p.phase = phase
            p.label = label
            p.updated_at = time.time()
            for k, v in fields.items():
                setattr(p, k, v)
            self._creep = _Creep(floor, max(ceil, floor), time.monotonic(), nominal)

    def snapshot(self, job_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            p = self._p
            if p is None or (job_id is not None and p.job_id != job_id):
                return None
            d = p.as_dict()
            d["percent"] = round(min(100.0, max(p.percent, self._creep.value())), 2)
            return d


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# --- trellis stdout parser --------------------------------------------------

# emit(phase, label, inner, inner_next, nominal, fields)
Emit = Callable[[str, str, float, float | None, float, dict[str, Any]], None]


class TrellisProgressParser:
    """Turns trellis-server stdout lines into progress updates.

    Stateful across a run; ``reset()`` is called on the ``generate:`` header so a
    second job on the same warm server starts from zero.
    """

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self.reset()

    def reset(self) -> None:
        self._stage: tuple[int, int] | None = None
        self._stage_label = ""
        self._flow_run = 0
        self._flow_runs_expected = 1
        self._in_flow = False
        self._sub_idx = -1

    # -- helpers

    def _stage_span(self) -> tuple[float, float]:
        if self._stage is None:
            return (0.0, 0.0)
        return _STAGE_OFFSETS.get(self._stage, (0.0, 0.0))

    def _push(
        self,
        frac: float,
        *,
        detail: str = "",
        frac_next: float | None = None,
        nominal: float = DEFAULT_NOMINAL,
        **fields: Any,
    ) -> None:
        """Map a 0..1 position *within the current stage* onto the trellis phase."""
        lo, hi = self._stage_span()
        span = hi - lo
        inner = lo + span * _clamp01(frac)
        inner_next = lo + span * _clamp01(frac_next) if frac_next is not None else None
        idx, total = self._stage if self._stage else (None, None)
        self._emit(
            "trellis",
            self._stage_label or "Generating",
            inner,
            inner_next,
            nominal,
            {"detail": detail, "stage_index": idx, "stage_total": total, **fields},
        )

    # -- main entry

    def feed(self, line: str) -> None:
        if RE_START.match(line):
            self.reset()
            self._stage_label = "Starting 3D generation"
            self._emit("trellis", self._stage_label, 0.0, 0.01, 4.0, {"detail": ""})
            return

        m = RE_FAIL.match(line)
        if m:
            self._emit("trellis", "Failed", 0.0, None, DEFAULT_NOMINAL, {"error": m.group(1)})
            return

        if RE_DONE.match(line):
            self._stage = (7, 7)
            self._stage_label = "Finishing"
            self._push(1.0, detail="", step=None, step_total=None, stage_eta=None)
            return

        m = RE_STAGE.match(line)
        if m:
            self._enter_stage(int(m.group(1)), int(m.group(2)), m.group(3))
            return

        if self._stage is None:
            return

        m = RE_FLOW.match(line)
        if m:
            self._on_flow(int(m.group(1)), int(m.group(2)), m.group(4))
            return

        if RE_FLOWEND.match(line):
            if self._in_flow:
                self._flow_run += 1
                self._in_flow = False
            # Once every flow run in this stage is done, the next signal is a
            # milestone that may be seconds away (stage 6's PBR decode is ~3.5 s
            # of silence). Aim the creep at it so the bar doesn't sit still.
            if self._flow_run >= max(self._flow_runs_expected, 1):
                nxt = self._next_substep()
                self._push(
                    FLOW_SPAN.get(self._stage or (0, 0), 1.0),
                    detail="",
                    frac_next=nxt.frac if nxt else 1.0,
                    nominal=nxt.nominal if nxt else DEFAULT_NOMINAL,
                    step=None,
                    step_total=None,
                    stage_eta=None,
                )
            return

        self._on_substep(line)

    # -- handlers

    def _enter_stage(self, idx: int, total: int, text: str) -> None:
        key = (idx, total)
        if key not in TRELLIS_STAGES:
            return
        self._stage = key
        self._stage_label = TRELLIS_STAGES[key][0]
        self._flow_run = 0
        self._in_flow = False
        self._sub_idx = -1
        # Stage 4 is a cascade (two flow runs) at res > 512, one run otherwise.
        self._flow_runs_expected = 2 if "cascade" in text else 1
        # Creep only a little way into a freshly entered stage. Aiming at the
        # first milestone is wrong when that milestone sits near the end of the
        # stage (stage 6's only milestone is at 0.98 with a 0.5 s nominal): creep
        # would saturate within a second and every flow line afterwards would
        # fall below it, freezing the bar for the whole stage.
        first = SUBSTEPS.get(key, [])
        ceiling, nominal = ENTRY_OVERRIDES.get(
            key, (min(first[0].frac, ENTRY_CREEP) if first else ENTRY_CREEP, ENTRY_NOMINAL)
        )
        self._push(
            0.0,
            detail="",
            frac_next=ceiling,
            nominal=nominal,
            step=None,
            step_total=None,
            stage_eta=None,
        )

    def _next_substep(self) -> Milestone | None:
        steps = SUBSTEPS.get(self._stage or (0, 0), [])
        nxt = self._sub_idx + 1
        return steps[nxt] if nxt < len(steps) else None

    def _on_flow(self, step: int, step_total: int, status: str) -> None:
        self._in_flow = True
        runs = max(self._flow_runs_expected, 1)
        span = FLOW_SPAN.get(self._stage or (0, 0), 1.0)
        frac = span * (self._flow_run + step / max(step_total, 1)) / runs
        nxt = span * (self._flow_run + (step + 1) / max(step_total, 1)) / runs
        eta = RE_ETA.search(status)
        self._push(
            frac,
            detail=f"denoise {step}/{step_total}",
            frac_next=nxt,
            nominal=1.0,
            step=step,
            step_total=step_total,
            stage_eta=float(eta.group(1)) if eta else None,
        )

    def _on_substep(self, line: str) -> None:
        steps = SUBSTEPS.get(self._stage or (0, 0))
        if not steps:
            return
        for i in range(self._sub_idx + 1, len(steps)):
            if steps[i].pattern.match(line):
                self._sub_idx = i
                nxt = steps[i + 1] if i + 1 < len(steps) else None
                self._push(
                    steps[i].frac,
                    detail=steps[i].label,
                    frac_next=nxt.frac if nxt else 1.0,
                    nominal=steps[i].nominal,
                    step=None,
                    step_total=None,
                    stage_eta=None,
                )
                return


# --- stdout pump ------------------------------------------------------------

_SPLIT = re.compile(rb"\r\n|\r|\n")


def pump(
    stream: IO[bytes],
    on_bytes: Callable[[bytes], None] | None,
    on_line: Callable[[str], None],
) -> None:
    """Drain ``stream`` to EOF, mirroring raw bytes and emitting decoded lines.

    The raw chunk goes to ``on_bytes`` untouched so the log file stays
    byte-identical to what the OS used to write. An exception from ``on_line`` is
    logged and swallowed: if this loop ever stops early the child blocks on a
    full pipe and the whole generation hangs.
    """
    buf = b""
    while True:
        try:
            chunk = stream.read(65536)
        except (ValueError, OSError):  # stream closed under us during shutdown
            break
        if not chunk:
            break
        if on_bytes is not None:
            try:
                on_bytes(chunk)
            except Exception:
                log.exception("trellis log mirror failed")
        buf += chunk
        parts = _SPLIT.split(buf)
        buf = parts.pop()
        for part in parts:
            _emit_line(part, on_line)
    if buf:
        _emit_line(buf, on_line)


def _emit_line(raw: bytes, on_line: Callable[[str], None]) -> None:
    if not raw:
        return
    try:
        on_line(raw.decode("utf-8", errors="replace"))
    except Exception:
        log.exception("trellis progress parser failed on a line")

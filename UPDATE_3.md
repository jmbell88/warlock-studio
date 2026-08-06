# UPDATE_3 — Native rewrite: selection contours, plus the marching-ants walk

**Rank: #3 of 3. Depends on UPDATE_1's shared infrastructure** (`native/`
tree, `native/build.ps1` → `vendor/warlockc/warlockc.dll`,
`src/warlock/native.py` loader with `WARLOCK_NATIVE`/`WARLOCK_NATIVE_DLL`, ABI
guard, doctor row, never-delete-the-Python-fallback rule). If worked before
UPDATE_1, lift that section from UPDATE_1.md verbatim first.

Two halves, shipped separately: a C rewrite of contour extraction (the one
true per-pixel Python loop in the repo), and a **pure-Python vectorization**
of the per-frame dash walk that draws the ants. The second half needs no C —
do not force it into the DLL.

## Context

**Half 1 — `SelectionMask.contours()` + `_chain`**
(`src/warlock/studio/inker/selection.py`). For each of 4 edge directions it
builds `exposed = inside & ~np.roll(inside, ...)`, then iterates
`for y, x in zip(*np.nonzero(exposed))` — one Python iteration per boundary
pixel per direction — collecting unit segments, which `_chain` stitches into
closed loops via dict/set walks. On a 2048² lasso the boundary is tens of
thousands of pixels; this is a visible pause on every selection *change*
(cached per mask identity in `panes/inker_canvas.py::_contours`, so it does
not run per frame — the cost is the spike, not the steady state).

**Half 2 — `panes/inker_canvas.py::_ants`.** Every frame, for every loop, it
transforms every vertex to screen space in a Python list comprehension, then
walks every unit segment with an inner `while travelled < length` dash loop
issuing `draw_list.add_line` per dash. For the same big lasso that is tens of
thousands of Python iterations and imgui calls *per frame*, on the frame
thread, forever.

The contract (`selection.py` module docstring): `contours()` returns closed
polylines of integer lattice points in image coordinates, pixel-edge accurate,
around the `mask >= INSIDE` (128) region. The engine stays headless-pure — no
imgui/GL imports under `studio/inker/`.

## Half 1 — contour extraction in C

### Kernel

```c
// mask: (h, w) uint8, row stride in bytes. threshold: INSIDE (128).
// Emits closed loops of (x, y) int32 lattice points, unit steps, in image
// coordinates (vertices range 0..w, 0..h inclusive).
// points_out: caller-allocated int32 buffer (capacity cap_pts * 2);
// loop_lens_out: caller-allocated int32 buffer (capacity cap_loops).
// Returns number of loops, or -1 if either capacity was insufficient.
int64_t warlockc_contours(
    const uint8_t* mask, int64_t stride, int64_t h, int64_t w,
    uint8_t threshold,
    int32_t* points_out, int64_t cap_pts,
    int32_t* loop_lens_out, int64_t cap_loops);
```

Capacity rule on the Python side: total boundary edges are bounded by
`4 * inside_pixel_count` and loops by edges; allocate from a cheap numpy count
of the exposed edges (four vectorized comparisons — no roll needed, compare
shifted slices) and retry once with the exact requirement if `-1`. No
allocation inside C.

Algorithm: grid-edge tracing on the (w+1)×(h+1) vertex lattice, not
segment-soup-then-chain. Build the inside bitmap once (padded by one, as the
Python does), then scan for an unvisited boundary edge and follow it with the
**left-hand rule: walk so that inside pixels are always on the left**, marking
edges visited, until the walk returns to its start. Each closed loop comes out
already ordered. O(boundary) time, no hashing.

**Determinism rule for the ambiguous corner** (a checkerboard vertex where two
regions touch diagonally — 4 boundary edges meet): the Python `_chain` resolves
it by arbitrary dict order; a tracer must pick a rule. Use the standard one —
prefer the turn that keeps the walk on the same region (turn toward inside) —
and document it in the C. This can split what Python merged into one loop (or
vice versa), which is *more* correct rendering-wise and invisible to the ants.

### Parity bar — deliberately not "identical lists"

Loop order, starting vertex, and winding direction are unspecified by the
contract (`_chain` starts from `next(iter(set))`). The ambiguous-corner rule
above can legitimately change the loop *partition*. So the parity test
asserts:

1. The **set of unit edges** (each edge as a frozenset of its two endpoints)
   is identical between C and Python — this is the actual geometric content.
2. Every emitted loop is closed (last connects to first), has length ≥ 3
   (wait — a 1-pixel selection is a 4-edge loop of 4 vertices; the Python
   emits `len(loop) > 2`; match that floor), and steps only unit edges.
3. Loop count matches on all masks **without** diagonal corner-touching
   (rectangles, ellipses, wand results, the existing fixtures); on
   deliberately constructed checkerboard-corner masks, assert only 1 and 2.

The existing behavior tests in `tests/inker/test_selection.py` run unchanged
and must pass in both modes: a rectangle's contour is its four corners and
nothing inside; the 2×2 contour is exactly `[(3,3),(3,4),(4,3),(4,4)]`
sorted; two separate regions give two loops; empty gives `[]`. Note these pin
that **every lattice vertex is emitted, no collinear simplification** — the C
emits unit steps exactly like today; simplification is out of scope (it would
also change the ants' dash phase along the loop).

### Wiring

`SelectionMask.contours()`: `if native.available():` call kernel, convert to
`list[list[tuple[int, int]]]` (the public contract — the pane caches whatever
this returns); else the existing segment/`_chain` code, which stays as the
reference and fallback. `selection.py` may import `warlock.native` — it is
stdlib-ctypes-pure, so the "nothing under `studio/inker/` imports
imgui/moderngl/pygame/`service`" rule holds; add it to whatever test enforces
that rule if one scans imports.

## Half 2 — vectorize the ants walk (Python only, no C)

Rewrite `_ants` / `_contours` in `panes/inker_canvas.py`:

1. **Cache per loop, keyed by the mask identity (existing key):** a float32
   `(n, 2)` vertex array per loop, plus cumulative canvas-space arc length
   `(n,)` (`np.cumsum` of segment lengths, prepended 0). Unit-step loops make
   lengths 1.0 each, but do not assume it — diagonal-free is a property of
   today's tracer, not of the contract.
2. **Per frame, per loop, all numpy:** screen vertices =
   `verts * zoom + origin` (one affine, `to_screen` is uniform scale +
   offset — verify against `inker_state.to_screen` and reuse its constants
   rather than duplicating the formula). Screen arc length = canvas length ×
   zoom. Dash boundaries = `np.arange(-phase, total, DASH)`; clip to
   [0, total]; endpoints via `np.searchsorted` into the cumulative lengths +
   linear interpolation — two `(k, 2)` arrays of dash segment start/end
   points and a boolean on/off per dash, computed without a Python loop.
3. **Draw:** one Python loop over only the "on" dashes calling
   `draw_list.add_line` (imgui has no batched per-segment-colour API;
   halving the calls and deleting the per-segment `while` is the available
   win). Alternative worth trying while in there: `add_polyline` for the
   whole loop in the dark colour first, then only light "on" dashes on top —
   same visual, one call for the entire dark phase.

Behavior bar: same dash geometry (`DASH`, `ANT_SPEED`, phase from
`time.monotonic()`), same colours, same per-loop phase reset (`walked`
restarts at `-phase` per loop today — keep that). This is frame-thread UI
code with no headless test; verification is visual (below) plus a pure
function extracted for the dash math (`_dash_segments(cum_lengths, zoom,
phase) -> (starts, ends, on)`) that *does* get a unit test in
`tests/inker/`-adjacent style — put it in a small module-level helper in
`inker_canvas.py` and test it headlessly (imgui is not imported at module
scope for the helper's inputs; if `inker_canvas` imports imgui at top level,
put the helper in `studio/inker_state.py` or a new tiny module instead —
check first, do not move imports around to force it).

## Implementation order

1. Half 2 first — it is pure Python, independent of the DLL, and it is the
   per-frame cost. Extract + unit-test the dash math, rewrite `_ants`,
   verify visually.
2. Half 1: `native/contours.c`, bump `WARLOCKC_ABI`, wire
   `SelectionMask.contours()`.
3. Parity tests (below), both-modes runs, full suite, ruff. Commit each half
   separately (no version bump unless asked).

## Testing / verification

- `uv run pytest tests/inker/test_selection.py -q` with the DLL and with
  `WARLOCK_NATIVE=0`.
- New parity test (skipif not `native.available()`): edge-set equality per
  the parity bar, on: every existing fixture shape (rect, ellipse, polygon,
  wand output), a feathered-then-thresholded blob, a 1-pixel selection, a
  full-canvas selection (boundary hugs the border — exercises the pad), a
  checkerboard-corner mask (properties 1–2 only), and ~50 seeded random
  masks.
- New dash-math unit test: known cumulative lengths, zoom, phase → expected
  segment endpoints and on/off pattern; dash pattern is continuous across
  vertices (the old walk carried `walked` across segments — the vectorized
  version must too, which falls out of using total arc length).
- Visual: in the app, lasso a large freehand region at 2048² — ants animate
  identically (spacing, speed, colours), selection-change pause is gone,
  frame time with a huge selection visibly improved. Record before/after
  frame cost of `_ants` (a `time.perf_counter` around it in a scratch build)
  in the commit message.

## Risks / notes

- The pane cache keys contours by mask identity (`select()` always builds a
  new mask — that stays true; do not add mutation to `SelectionMask`).
- Winding direction may differ between C and Python loops. Nothing consumes
  winding today (the ants walk either way), but the parity test should not
  accidentally assert it.
- `np.searchsorted` side/off-by-one at exact dash boundaries: pick `side` so
  a dash boundary landing exactly on a vertex behaves like the old
  `while`-loop (which used `min(travelled + DASH, length)` per segment —
  boundaries at vertices were natural there). The unit test pins this.
- If `to_screen` turns out not to be a pure uniform affine (e.g. it rounds),
  match it exactly rather than "close" — ants must sit on the same pixels as
  before, or the selection looks like it moved.

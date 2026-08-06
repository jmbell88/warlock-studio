# UPDATE_2 — Native rewrite: the Inker blend/composite core

**Rank: #2 of 3. Depends on UPDATE_1's shared infrastructure** (`native/`
tree, `native/build.ps1` → `vendor/warlockc/warlockc.dll`, `src/warlock/native.py`
loader with `WARLOCK_NATIVE`/`WARLOCK_NATIVE_DLL`, ABI guard, doctor row,
never-delete-the-Python-fallback rule). If worked before UPDATE_1, lift that
section from UPDATE_1.md verbatim first.

## Context

`src/warlock/studio/inker/composite.py` (~170 LOC) holds every blend formula
of the raster editor: `blend` (W3C separable modes: normal, multiply, screen,
overlay, add), `over` (straight-alpha compositing), `paint_colour` (the one
formula every colour-writing tool shares), `stack_region` (bottom-first layer
fold), and the `to_float`/`to_uint8` conversions.

Why it is a rewrite candidate: the arithmetic is fully numpy-vectorized but
**memory-bandwidth-bound** — one `over()` call materializes roughly eight
full-region float32 temporaries (`ao`, `mixed`, `num`, the slices, the
`where`s), and `stack_region` calls it once per layer, allocating a fresh
output each time. The killer callers run **on the frame thread**:

- `Document.invalidate()` → `LayerStack.composite_region` — per stroke dab,
  bounded by the dirty rect, fine today.
- `Document.invalidate_all()` / `_rebuild()` — O(layers × canvas), fired on
  layer reorder, hide/show, layer-op undo, resize. At 2048² with several
  layers this is hundreds of MB of traffic on the one thread that owns the GL
  context — a visible hitch.
- `brush.py` `_resolve` → `paint_colour` per dab (small rects, latency-sensitive).
- `ora.py` and `flatten_onto` on save/export (task thread, less critical).

A fused C kernel does the whole per-pixel chain in registers: zero
temporaries, one read of each input, one write of the output. Expected win is
5–15× on the `invalidate_all` path, and it composes with UPDATE_3 (both are
"the frame thread stops doing avoidable work").

The module is deliberately headless-pure (no imgui/GL/pygame/service imports)
and fully covered by `tests/inker/test_composite.py` (11 behavior tests
pinning the SVG spec formulas, the uint8 round-trip, the matte rule) plus
`tests/inker/test_inker_document.py`'s
`test_the_cached_composite_matches_a_composite_from_scratch`. That coverage is
what makes a bit-exact native swap safe.

## Design

### Stage 1 — drop-in kernels for `over` and `paint_colour`

```c
// All buffers float32. h*w pixels, 4 channels, C-contiguous rows with an
// element stride so numpy views (rect slices of a larger canvas) pass
// without a copy: stride counts floats between row starts.
void warlockc_over_f32(
    const float* backdrop, int64_t backdrop_stride,
    const float* source,   int64_t source_stride,
    float* out,            int64_t out_stride,      // may alias backdrop
    int64_t h, int64_t w, float opacity, int32_t mode);

void warlockc_paint_colour_f32(
    const float* before, int64_t before_stride,
    const float* weight, int64_t weight_stride,     // (h, w) float32
    float* out,          int64_t out_stride,        // may alias before
    int64_t h, int64_t w, const float rgba[4]);     // colour, 0..255 domain
```

`mode` is an enum mirroring `BLEND_MODES` order; keep the mapping in one
place in `composite.py` next to `BLEND_MODES` so adding a mode is one Python
line + one C case, and an unknown mode still raises `ValueError` in Python
before the call (the existing `test_an_unknown_blend_mode_is_a_programming_error`).

**Bit-parity is the contract, and it dictates the C:**

- Compute in `float` (32-bit) with the *same expression shapes and operand
  order* as the numpy code: `ao = a_s + ab*(1-a_s)`;
  `num = a_s*(1-ab)*cs + a_s*ab*B + (1-a_s)*ab*cb`; divide guarded by
  `ao > 0` else 0; alpha channel = `ao`. numpy's elementwise float32 ops are
  per-element IEEE — a C `float` chain with the same order and no contraction
  is bit-identical. Build flags from UPDATE_1 (`/fp:precise`,
  `-ffp-contract=off`) are what guarantee "no contraction".
- `opacity` reaches numpy as a Python float multiplying a float32 array —
  which stays float32. In C: take `float opacity`, multiply in float.
  (Python side already does `float(opacity)`; pass
  `ctypes.c_float(opacity)` — the float64→float32 rounding happens once at
  the boundary, same as numpy's cast.)
- Overlay's branch is `backdrop <= 0.5` — same compare, per channel.
- **The fast path stays in Python.** `over`'s early-out
  (`opacity >= 1 and mode == "normal" and source alpha min >= 1`) returns
  `source.copy()`. Do *not* replicate it per-pixel in C — an opaque pixel
  inside a non-opaque source goes through the full formula in numpy, and a
  per-pixel shortcut would diverge in the last ulp. Python keeps the guard,
  then dispatches slow-path work to the kernel.

Python wrappers: inside `over`/`paint_colour`, after the existing early-outs,
`if native.available():` allocate `out = np.empty_like(backdrop)`, check
inputs are float32 with unit element stride within rows
(`arr.strides[-1] == 4` and `arr.strides[-2] == arr.shape[-1]*4` is *not*
required — pass the row stride), call, return; else the existing numpy body.
Signatures, docstrings, and the module's "straight alpha, float32, 0..1"
contract are untouched.

### Stage 2 — fused `stack_region` (optional, measure first)

`stack_region` still allocates one float32 output per layer fold and converts
each uint8 layer crop with `to_float`. A fused kernel takes the layer list in
one call:

```c
// layers: n pointers to uint8 RGBA rect crops (row stride each, may be
// views into full canvases); opacities/modes per layer; out float32.
void warlockc_stack_f32(
    const uint8_t** layers, const int64_t* strides,
    const float* opacities, const int32_t* modes, int64_t n,
    float* out, int64_t out_stride, int64_t h, int64_t w,
    const float* base, int64_t base_stride);       // NULL => zeros
```

Per pixel: load base (or 0), then fold each layer inline —
`to_float` (`x/255.0f`) fused into the load. Bit-parity holds because
`u8/255.0f` is exact per element and the fold order matches the Python loop.
Only build Stage 2 if Stage 1 timings show the remaining allocation/convert
overhead still matters on `invalidate_all` at 2048²; Stage 1 alone may be
enough. `LayerStack.composite_region` and `_entries` stay the seam — the
Python `stack_region` keeps its tuple-list signature and dispatches.

### Explicitly out of scope

- `to_uint8` / `to_float` as standalone calls (numpy is already fine),
  `flatten_onto` (export path, cold), `_draw_shape`, blur (Pillow, already C).
- The brush stamp pipeline (`brush.py`) beyond it calling the new
  `paint_colour` — resolving dabs stays numpy; that is UPDATE candidate #5
  and deliberately not here.
- Premultiplied alpha, SIMD intrinsics, threading inside the kernel. Straight
  scalar C first; it will already be memory-bound. Measure before adding
  anything clever.

## Implementation order

1. Confirm UPDATE_1's infra exists (`native.py`, build script, doctor row).
2. Add `native/composite.c` with `warlockc_over_f32` (all five modes) —
   bump `WARLOCKC_ABI`.
3. Wire `over()` behind `native.available()`. Run
   `uv run pytest tests/inker -q` with DLL and with `WARLOCK_NATIVE=0`.
4. Add the bit-parity test (below). Iterate until exact.
5. Add `warlockc_paint_colour_f32`, wire `paint_colour`, extend parity test.
6. Timing pass on `invalidate_all` (2048², 6 layers, all modes). Decide
   Stage 2 with numbers in hand; implement the same way if yes.
7. Full suite + ruff. Commit (no version bump unless asked).

## Testing / verification

- Existing: `tests/inker/test_composite.py` (11 tests) and the whole
  `tests/inker` suite must pass **in both modes** — with the DLL, and with
  `WARLOCK_NATIVE=0`. The document-level test
  `test_the_cached_composite_matches_a_composite_from_scratch` is the
  integration safety net: the below-cache and the from-scratch composite go
  through the same kernels.
- New `tests/inker/test_composite_native.py` (skipif not
  `native.available()`):
  - Randomized parity: seeded random `(H, W, 4)` float32 in [0,1] (plus
    exact-0 and exact-1 alpha bands, and denormal-adjacent tiny alphas), every
    mode × opacities {0.0, 0.37, 1.0}, assert `np.array_equal` — **bit-exact,
    not allclose**. If it fails, fix the C, never the assertion.
  - Strided views: composite a rect slice of a larger canvas through both
    paths (the `stack_region` crop shape).
  - `paint_colour` parity across weight edge values {0, 128/255, 1}.
- Behavior spot-check in the app (`uv run warlock studio` or the `run`
  skill): paint strokes in each blend mode, reorder/hide layers on a large
  multi-layer document, undo a layer op — no visual change, no hitch.
- Timing sanity recorded in the commit message: `invalidate_all` wall time
  at 2048² × 6 layers, native vs `WARLOCK_NATIVE=0`.

## Risks / notes

- The ORA interop promise ("reopened in Krita composites identically") is
  about the *formulas*, which do not change. Bit-parity with the numpy
  reference is a stronger property than the promise needs — hold the stronger
  line anyway, it is what makes the fallback undetectable.
- Frame-thread caller means a crash in the kernel takes the app down, not a
  job. The kernel has no allocations, no branches on data-dependent pointers,
  and Python validates shapes/dtypes/strides before every call — keep it that
  boring.
- `np.divide(..., where=ao > 0)` leaves `out` untouched where the mask is
  false and the code then `np.where`s zeros in; net semantics are
  "0 where ao<=0". C: `ao > 0.0f ? num/ao : 0.0f` — same result, simpler, and
  bit-identical (division only happens where taken).
- ctypes calls release the GIL; on the frame thread that is harmless (nothing
  else contends inside a frame) and it keeps task threads unblocked.

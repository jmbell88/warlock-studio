# UPDATE_1 — Native rewrite: meshaudit's rasterizer and flood fills

**Rank: #1 of 3.** The only genuinely bad algorithm in a hot path, and the best
effort-to-payoff ratio. This plan also establishes the shared native-library
infrastructure (`native/` sources, `vendor/warlockc/warlockc.dll`, the ctypes
loader, the doctor row) that UPDATE_2 and UPDATE_3 build on. **Work this one
first**; if you must start with 2 or 3, lift the "Shared infrastructure"
section from here verbatim.

## Context

`src/warlock/meshaudit.py` measures silhouette hole fractions on every finished
mesh (called from `queue.py` `Worker._audit_mesh` via `asyncio.to_thread`, and
from `sweep.py` offline). Three hot pieces:

1. `_coverage` / `_rasterise_batch` — a software triangle rasterizer using
   pixel-centre barycentric edge functions over `(n, k, k)` float64 tensors,
   with `_BATCH_MAX_SPAN` / `_BATCH_MAX_CELLS` chunking to cap a working set
   that otherwise holds ~a dozen full-size float64 temporaries.
2. `_enclosed_gaps` — an iterative fixpoint flood fill: `while True:` copy the
   whole boolean array, OR in four shifted copies, compare. O(image diameter)
   full-array passes.
3. `_count_blobs` — the same fixpoint shape but worse: int64 label propagation
   with `np.maximum` until convergence.

The module's own comments concede the fixpoint cost scales ~resolution³ and
that `REQUEST_PATH_RESOLUTION` was halved from 1024 to 512 purely to keep the
per-job audit at "seconds rather than tens of seconds". The goal: make the
audit sub-second at 512 and viable at 1024, with results **identical** to
today's (`hole_fraction`'s returned dict is stored on jobs and consumed by
observations/findings — the numbers must not drift).

Key discovery that shapes the plan: **`opencv-python-headless` is already a
core dependency** (pyproject names "flood fill, connected components" as the
reason it exists), and `cv2.connectedComponents` is already used in
`pipelines/reference.py` and `pipelines/asset2d.py`. The asymptotically bad
half of this module can move to native C++ with zero new infrastructure.

## Phase A — flood fills → cv2 (no toolchain, do first, ship independently)

Replace `_enclosed_gaps` and `_count_blobs` with **one**
`cv2.connectedComponents` call over the uncovered pixels:

```python
def _holes_and_blobs(covered: np.ndarray) -> tuple[np.ndarray, int]:
    free = (~covered).astype(np.uint8)
    count, labels = cv2.connectedComponents(free, connectivity=4)
    # Labels present on the border are background; label 0 is `covered`.
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.zeros(count, dtype=bool)
    outside[border] = True
    outside[0] = True
    holes = ~outside[labels]
    return holes, int(count - int(outside.sum()))
```

- **connectivity=4 is load-bearing**: both fixpoints grow 4-connected. 8 would
  change hole shapes and blob counts.
- Connected components are well-defined, so this is *exactly* equal to the
  fixpoint results — hole mask bit-for-bit, blob count identical. Pin that
  with a parity test (below) before deleting the fixpoints.
- `hole_fraction` keeps its exact return shape; only the internals of the two
  helpers change. `_count_blobs`'s answer falls out of the same
  `connectedComponents` call, so restructure the two helpers into one
  (`_enclosed_gaps` currently runs first and `_count_blobs` re-labels its
  output — one call now produces both).
- Import `cv2` inside the function, matching `pipelines/reference.py` style.

This phase alone removes the resolution³ term and is a plain Python diff. Ship
it before touching C.

## Phase B — the rasterizer → C (`warlockc.dll`)

### Shared infrastructure (established here, reused by UPDATE_2/3)

```
native/
  warlockc.h        # WARLOCKC_ABI version, exported prototypes
  meshaudit.c       # this plan's kernel
  build.ps1         # produces vendor/warlockc/warlockc.dll
src/warlock/native.py   # ctypes loader (stdlib only)
vendor/warlockc/warlockc.dll   # build artifact; vendor/ is already gitignored
```

**Build (`native/build.ps1`):** x64 DLL, no external dependencies, C11. Try
MSVC `cl` (via vswhere / VsDevShell), fall back to `clang-cl` or `zig cc` on
PATH. Flags are part of the correctness contract, not an optimization detail:

- MSVC: `/O2 /fp:precise /LD` — never `/fp:fast`.
- clang: `-O2 -ffp-contract=off -shared` — never `-ffast-math`; contraction
  off because an FMA changes rounding and breaks bit-parity with numpy.

**ABI guard:** the DLL exports `int warlockc_abi(void)` returning
`WARLOCKC_ABI` from `warlockc.h`. Bump it on any signature or semantic change.
The loader refuses a mismatched DLL (falls back to Python) — a stale local
build must degrade, never silently compute old behavior.

**Loader (`src/warlock/native.py`):** pure like `vram.py`/`memlog.py` — stdlib
ctypes only, no imports from `service`/`queue`/`studio`, never raises for a
missing/undecodable DLL (returns None; callers fall back).

- Default path: `<repo>/vendor/warlockc/warlockc.dll` (resolve the way
  `config.py` resolves `gltfpack_exe`).
- `WARLOCK_NATIVE=0` disables entirely (for A/B timing and CI).
- `WARLOCK_NATIVE_DLL=<path>` overrides the location (worktrees don't have
  `vendor/` — same reason the existing `WARLOCK_*` worktree env vars exist).
- Module-level `available() -> bool` and `lib() -> ctypes.CDLL | None`,
  cached after first probe.
- ctypes foreign calls release the GIL — meshaudit already runs on a task
  thread via `asyncio.to_thread`, and this keeps it from blocking anything.

**Doctor row:** follow `doctor._gltfpack_check` exactly — non-fatal, names the
remedy: `Check("warlockc (native kernels)", ok, path-or-"not built at <path>
-- run native\\build.ps1; Python fallbacks in use", fatal=False)`. Register in
`doctor.run_checks`.

**Fallback rule (applies to all three UPDATEs):** the numpy implementation is
never deleted. It is the reference the parity tests compare against and the
fallback when the DLL is absent. Every native call site is
`if native.available(): ... else: <existing code>`.

### The kernel

Keep the projection in Python (`_screen_basis`, the `positions @ right`
projection, degenerate-triangle cull, bbox/span math — all cheap numpy). Move
only the per-triangle fill:

```c
// Writes covered (resolution*resolution uint8, row-major) in place.
// a,b,c: projected pixel coords, float64 pairs, length n each.
// area2: signed doubled area, float64, length n (pre-culled, never 0).
void warlockc_rasterise(
    const double* ax, const double* ay,
    const double* bx, const double* by,
    const double* cx, const double* cy,
    const double* area2,
    int64_t n, int32_t resolution, uint8_t* covered);
```

Per triangle, reproduce `_coverage`'s semantics **exactly** — the parity bar
is a bit-identical `covered` mask, so every quirk below is deliberate:

1. Clipped bbox: `x0 = clip(floor(min(ax,bx,cx)), 0, res-1)`,
   `x1 = clip(ceil(max)), ...` — same for y. Spans from the *clipped* values.
2. Subpixel rule: if `span_x <= 1 && span_y <= 1`, mark the single pixel at
   the clipped integer centroid (`(ax+bx+cx)/3` truncated toward zero via
   `(int)` cast — numpy's `.astype(int)` truncates, and negative coords occur
   before the clip, so truncation vs floor matters).
3. Otherwise iterate a **square** `k × k` window, `k = max(span_x, span_y) + 1`,
   anchored at `(x0, y0)` — square even when the spans differ, because that is
   what the batched numpy path tests. Sample points `gx = x0 + i + 0.5`
   *unclamped*; edge functions `e0,e1,e2` with the same operand order as
   `_rasterise_batch`; inclusion `e*sign(area2) >= 0` for all three; write
   index `clip(x0+i, 0, res-1)` — the clamp on the *write*, not the sample,
   duplicates the numpy behavior at the border.
4. All arithmetic in `double`, same expression shapes, no reassociation.

The Python side of `_coverage` becomes: project and cull as today; if
`native.available()`, call the kernel once over all non-degenerate triangles
(the subpixel split moves into C); else run the existing subpixel/batch/chunk
machinery unchanged. `_BATCH_MAX_SPAN` / `_BATCH_MAX_CELLS` stay — they belong
to the fallback path.

## Implementation order

1. Phase A: rewrite `_enclosed_gaps`/`_count_blobs` on `cv2.connectedComponents`.
   Add a parity test comparing against the old fixpoints (keep the fixpoint
   code inside the test as the oracle) on random masks + the punched-cube
   fixture. Run `tests/test_meshaudit.py` (7 tests, all property-based:
   watertight sphere/cube report no holes, punched hole is found, higher
   resolution does not manufacture holes, chunking is bit-identical). Commit.
2. `native/warlockc.h`, `native/meshaudit.c`, `native/build.ps1`,
   `src/warlock/native.py`, doctor row. Build locally.
3. Wire `_coverage` to the kernel behind `native.available()`.
4. Parity + regression tests (below). Commit.
5. Only then, as a **separate decision with the user**: whether
   `REQUEST_PATH_RESOLUTION` goes back to 1024. `config.py`'s audit thresholds
   (~line 201) were calibrated against measurements at both resolutions and
   the existing test pins that fractions are comparable across resolutions —
   so keeping 512 is safe and raising it is a quality choice, not a bug fix.

## Testing / verification

- `uv run pytest tests/test_meshaudit.py -q` — all 7 existing tests pass with
  the DLL present *and* with `WARLOCK_NATIVE=0` (run both).
- New `test_native_rasteriser_matches_numpy_bit_for_bit`: same
  positions/faces/resolution through both paths, `np.array_equal(covered_a,
  covered_b)`. Fixtures: the sphere fixture, the punched cube, plus ~200
  random triangles including degenerate-adjacent slivers and
  partially-off-screen ones (the clip/clamp paths). Skip via
  `pytest.mark.skipif(not native.available(), ...)`.
- New `test_cv2_holes_match_the_fixpoint_oracle`: random boolean masks +
  fixture-derived coverage masks; hole mask and blob count equal.
- End-to-end: `hole_fraction` on the bench suite's "genuinely holed geometry"
  case (`bench/suites/core-v1.json` names it) returns the same dict as
  before, key for key.
- Timing sanity (not a committed test): time `hole_fraction` at 512 and 1024
  on a real trellis GLB, native vs `WARLOCK_NATIVE=0`. Expect the fixpoint
  removal (Phase A) to dominate; record numbers in the PR/commit message.
- `uv run pytest -q` full suite; `uv run ruff check .`.

## Risks / notes

- **Float parity**: the whole reason for `/fp:precise` + `-ffp-contract=off`
  and "same expression shapes". If the bit-parity test fails, diff the masks —
  a handful of border pixels means a contraction or reassociation snuck in;
  do not loosen the test to "close enough" without understanding why.
- `covered` dtype: numpy path uses `bool`; pass to C as `uint8` (same memory
  layout, `covered.view(np.uint8)` or allocate uint8 and view back).
- Arrays must be C-contiguous before the call (`np.ascontiguousarray` — the
  projected columns come out of `np.stack`, already contiguous, but assert).
- cv2 label ids differ from the fixpoint's label values — irrelevant, nothing
  consumes label values, only the hole mask and the count.
- Do not touch `meshreport.py` — the deliberate audit/report split
  (silhouette vs topology) stays.

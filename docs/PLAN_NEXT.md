# Where a fourth native kernel is (and is not) worth it

## Context

Three C kernels have shipped (ABI 4): the meshaudit rasteriser, the inker
compositor (`over` / `paint_colour` / fused `stack` / `to_uint8`), and the
selection contour tracer. Each was justified by a measured number, and each was
in a place numpy genuinely could not help — a scatter with overlap, an
eight-temporaries-per-layer fold, and a graph walk.

The question is what to rewrite next. The remaining compute-bearing modules were
read in full (`clay/mesh.py`, `clay/adjacency.py`, `clay/earclip.py`,
`clay/pick.py`, `viewer/picking.py`, `viewer/gltf.py`, `viewer/sheet.py`,
`viewer/env.py`, `inker/selection.py`, `inker/brush.py`, `inker/composite.py`,
`inker/document.py`, `inker/layers.py`, `meshreport.py`, `clay_view.py`,
`ants.py`) and the honest headline is:

**Most of what's left should not be rewritten in C, and three of the places
that look like C candidates are actually numpy/Python bugs that will get most of
the win for none of the ABI cost.** The codebase is already single-pass
whole-array numpy nearly everywhere. Two genuine kernel candidates remain — the
ones with the *same shape* as the ones that shipped — and only the smaller of
the two is in this plan's scope; the other is recorded under **Deferred**.

There is also no microbenchmark harness in the repo — the three shipped plans
each carried measured numbers from ad-hoc scripts. Every item below states what
to measure, because the bar this project has set is a number, not an argument.

---

## Tier 0 — fix these in Python/numpy first (no ABI bump, no new C)

These are the three places where a profile would show a hotspot and the correct
response is *not* a kernel. Doing these first is also what makes the Tier 1
measurement honest.

### 0a. `clay/mesh.py` — `np.add.at` in `render_arrays`

`src/warlock/studio/clay/mesh.py:345` and `:387`:

```python
np.add.at(accum, smooth_loops, raw[face_of_corner[smooth_corner]])
```

`np.add.at` is numpy's unbuffered ufunc path — routinely 10–50× slower than the
equivalent `bincount`. It is the single slowest primitive left in Clay's hot
path: `clay_view.ClayView.sync` (`clay_view.py:332`) rebuilds through
`_build` → `bd.to_primitives` → `render_arrays` on every mesh edit, keyed on
`id(obj.mesh)`, so every extrude / inset / element-drag commit pays it.

Replace with three `np.bincount(smooth_loops, weights=values[:, k], minlength=V)`
calls stacked. Both accumulate in input order, so the sum order is unchanged —
**assert bit-identity against the current output in a test** rather than assuming
it, since the result feeds `_normalize` and then the GPU.

### 0b. `clay/earclip.py` — `_earclip`'s inner loop

`src/warlock/studio/clay/earclip.py:179`. Two separate constant-factor problems
inside an inherently O(n²) search:

- `rest = [i for i in idx if i not in (a, b, c)]` rebuilds a list on **every**
  iteration of the ear search.
- `pts` is a numpy array and `_cross2`/`_inside` index it scalar-wise
  (`pts[a][0]`), so every one of the ~3n² cross products pays numpy scalar
  boxing (~100 ns each) instead of a float subtraction (~20 ns).

Convert `pts` to a list of plain float tuples once at the top of `_earclip`, and
carry the polygon as prev/next index arrays instead of rebuilding `rest`. This
is a local change to one function with an existing reference (`fan_corners`) and
existing tests. Expect roughly an order of magnitude before any C is written.

### 0c. `inker/selection.py` — `_draw_shape` supersamples the whole canvas

`src/warlock/studio/inker/selection.py:39`. It allocates `Image.new("L", (W*4, H*4))`
— for a 2048² document that is a **64 MB** buffer plus a 64 MB BOX-resize read —
regardless of how small the shape is. A 50-pixel lasso pays full canvas cost.

Rasterise the shape's own bounding box at 4× and paste the downsampled result
into a canvas-sized zero mask. Same output, bounded by the shape rather than the
document. This is the whole fix; the drawing itself is already Pillow's C.

---

## Tier 1 — the one kernel in scope

Same shape as the kernels that shipped: elementwise, a numpy reference that stays
as the fallback, and a **bit-identical** parity bar.

### `to_uint8` for values already in 0..255 — `warlockc_to_uint8_scaled_f32`

**Where:** four call sites hand-roll `np.clip(out + 0.5, 0, 255).astype(np.uint8)`
and none of them can use the existing kernel, because
`composite.to_uint8` / `warlockc_to_uint8_f32` multiplies by 255 first:

- `inker/brush.py:206` (`_resolve`)
- `inker/brush.py:238` (`_filter` — blur and smudge)
- `inker/document.py:318` (`write_colour`)
- `inker/document.py:482` (`gradient`)

Either add a scale parameter to the existing export (ABI bump, one call site to
update) or add a sibling that takes 0..255 floats. Trivial C, trivially
bit-identical (`v = x + 0.5f`, clamp, truncate — matching `.astype(uint8)`'s
truncation exactly). The measured rate on the existing narrowing kernel was
90 ms → 16.7 ms on a full 2048² RGBA.

All four sites become `cp.to_uint8_255(out)` (or whatever the seam is named),
with the numpy expression kept behind `native.available()` as the fallback and
the reference, exactly as `to_uint8` does today.

---

## Deferred (analysed, not in this scope)

**The fused brush dab — `warlockc_dab_u8`.** `brush.StrokeState._resolve`
(`brush.py:188`) issues ~8 numpy dispatches and ~6 allocations per dab, and
`to()` fires ~10 dabs per mouse-move (×4 under `symmetry="xy"`). At the default
8 px diameter the region is 64 pixels, so that is pure dispatch overhead — the
argument UPDATE_2's Stage 2 won when `to_float` turned out to be 312 ms of
`stack_region`'s 757 ms. A single call taking `(before_u8, coverage_f32,
clip_u8|NULL, colour, opacity, mode)` would replace the whole body. It is the
largest remaining kernel and the one most dependent on the brush actually showing
up in a profile, so it is deliberately left out here; the `to_uint8_255` seam
above lands inside `_resolve` anyway and is the natural place to measure from.

---

## Tier 2 — considered and explicitly rejected

Recorded so the question does not get re-asked. Each was read, not guessed.

**Already C — rewriting them is rewriting C in C:**
`SelectionMask.feathered` and `brush._filter`'s blur (Pillow GaussianBlur),
`_contiguous` (Pillow floodfill), `_draw_shape`'s fill (Pillow ImageDraw),
`ora.py` PNG encode (zlib), `meshreport.build` (trimesh predicates —
deliberately trimesh's, per the module docstring), `postprocess.normalize_glb`
(trimesh), `viewer/gltf.texture` (Pillow decode), the SDXL and trellis paths.

**Already single-pass whole-array numpy — no headroom:**
`clay/adjacency._build` (the one `np.unique` builds every table),
`clay/mesh.edges` / `_face_normals` / `reversed_corner_perm`,
`clay/pick.py` (project / marquee / `reduceat`), `viewer/picking.ray_triangles`
(vectorised Möller–Trumbore), `studio/ants.py` (already rewritten, whole-array),
`inker/composite.stack_region` (no obvious win left in that file),
`clay/adjacency.check_manifold` (lexsort, no per-face loop).

**Zero-copy or negligible N:**
`viewer/gltf._Reader.accessor` (`np.frombuffer` into the BIN chunk — nothing to
speed up; the interleaved gather path is rare and small),
`viewer/env.py` (32×16 gradient and a 64×32 probe, once at startup),
`viewer/sheet.StripRender` (GPU draw plus a synchronous `read_rgba` — the cost
is the readback, which is why it is already one cell per frame),
`clay/adjacency.boundary_loops` (Python walk, but only over boundary edges,
degree 1–2).

**A real Python walk, but low frequency:** `selection._chain` is now the
fallback/reference behind `native.contours` and should stay that way.

---

## Critical files

- `native/warlockc.h` — bump `WARLOCKC_ABI` (4 → 5) **together with**
  `src/warlock/native.py:49` `ABI`
- `native/composite.c` — the new narrowing kernel sits beside
  `warlockc_to_uint8_f32`, whose 256-entry-table idiom it should follow
- `src/warlock/native.py` — prototypes in `_bind` (ctypes defaults truncate
  64-bit pointers) and a typed view for the kernel
- `src/warlock/studio/inker/brush.py`, `src/warlock/studio/inker/document.py` —
  the `if native.available(): ... else: <numpy>` seams; the numpy bodies are
  never deleted
- `src/warlock/studio/clay/mesh.py`, `src/warlock/studio/clay/earclip.py`,
  `src/warlock/studio/inker/selection.py` — Tier 0
- `tests/inker/test_composite_native.py` — extend with the new kernel's parity
  cases; `tests/test_native.py` for the loader/ABI side

`native/build.ps1` globs `native/*.c`, so a new file needs no script change —
invoke it as `pwsh -NoProfile -File "native/build.ps1"` from Bash (a backslash
path silently becomes `nativebuild.ps1`).

## Verification

1. **A before/after number per item**, recorded the way UPDATE_1–3 did: a scratch
   script for `render_arrays` on an imported ~200k-corner GLB (0a), a dissolved
   n-gon mesh through `corner_triangles` (0b), a small-lasso `from_polygon` on a
   2048² document (0c), and a full-canvas narrowing for the kernel. If a change
   does not move its number, say so rather than keeping it.
2. `pwsh -NoProfile -File "native/build.ps1"`, then `uv run pytest` **both ways**
   (with the DLL, and `WARLOCK_NATIVE=0`), matching the 2901 / 2783 baseline
   recorded on 2026-08-06.
3. `uv run ruff check .`
4. Tier 0 items get bit-identity assertions against the current output, since
   each claims to change speed only.
5. Run the app and paint a stroke with a feathered selection active, and run a
   dissolve on a Clay object — the two paths the changes touch — to confirm no
   visible change.

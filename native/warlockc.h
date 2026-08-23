/* warlockc -- native kernels for the hot paths numpy cannot make cheap.
 *
 * Scope discipline, so this stays a leaf: no allocation, no I/O, no state, no
 * dependency beyond the C standard library. Every entry point takes
 * caller-owned buffers and writes into caller-owned buffers. The Python side
 * (src/warlock/native.py and its callers) validates shapes, dtypes and strides
 * before every call, which is what lets these be free of defensive checks --
 * they are on the frame thread's and the task thread's critical paths.
 *
 * Bit-parity with the numpy reference is the contract, not a goal. Every
 * kernel here has a Python implementation it must agree with exactly, and the
 * tests assert equality rather than closeness. What that costs at build time is
 * that contraction must be off in every compiler this is built with: a fused
 * multiply-add rounds once where numpy rounds twice, and a changed rounding
 * moves a triangle edge by one pixel. native/build.ps1 owns the flags, and it
 * is the one place that knows which flag each driver needs -- notably that a
 * driver taking MSVC-style options is not thereby MSVC, so /fp:precise alone
 * cannot be assumed to have disabled contraction. The contract is stated here;
 * the spelling of it lives with the build.
 *
 * Two kernels are deliberately not held to output identity, and in both cases
 * for the same reason: the reference itself leaves something unspecified, so
 * output identity is not a property the numpy path has either. The contour
 * tracer in contours.c leaves loop order and winding unspecified, and its bar
 * is the set of unit edges. The BVH build in bvh.c inherits np.argpartition's
 * unspecified permutation among equal keys, and its bar is the pick result
 * plus the structural invariants. A loosened bar is how a kernel quietly stops
 * being checked, so each licence is argued in docs/INVARIANTS.md and asserted
 * beside the tests that rest on it -- and neither is a precedent for a third.
 */

#ifndef WARLOCKC_H
#define WARLOCKC_H

#include <stdint.h>

#if defined(_WIN32)
#define WARLOCKC_API __declspec(dllexport)
#else
#define WARLOCKC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bump on ANY change to a signature or to what a kernel computes.
 *
 * The loader refuses a DLL whose value differs from the one compiled into the
 * Python side and falls back to numpy. vendor/ is gitignored, so a checkout
 * routinely carries a stale locally-built DLL next to newer sources -- without
 * this guard that DLL would silently compute the old behaviour, which is the
 * one failure mode a fallback path must never have. */
#define WARLOCKC_ABI 10

WARLOCKC_API int32_t warlockc_abi(void);

/* Rasterise triangles into an 8-bit coverage mask, pixel-centre sampled.
 *
 * Mirrors warlock.meshaudit._coverage's inner half exactly, quirks included:
 * a triangle whose clipped bounding box spans one pixel or less marks the
 * single pixel at its truncated integer centroid; every other triangle
 * iterates a square k x k window anchored at the clipped box origin, samples
 * at unclamped pixel centres, and clamps only the write coordinate.
 *
 * Coordinates are already projected to pixel space. area2 is the signed
 * doubled area, never zero (degenerate triangles are culled in Python).
 * `covered` is resolution*resolution bytes, row-major, written in place --
 * set to 1 where covered, never cleared, so repeated calls accumulate. */
WARLOCKC_API void warlockc_rasterise(const double *ax, const double *ay,
                                     const double *bx, const double *by,
                                     const double *cx, const double *cy,
                                     const double *area2, int64_t n,
                                     int32_t resolution, uint8_t *covered);

/* The blend modes: the numbers composite._MODE_IDS hands over.
 *
 * The names live in Python and the numbers live here, which means adding a
 * mode is one entry in BLEND_MODES, one in composite._MODE_IDS and one case in
 * blend_channel (or, for a non-separable one, in blend_nonseparable). The
 * *number* is the coupling, so it is spelled out in both places rather than
 * inferred in either -- and deliberately not the position in BLEND_MODES,
 * which is menu order and is free to be regrouped.
 *
 * **The four non-separable modes are the top of the range, and blend_rgb tests
 * for them with `mode >= WARLOCKC_BLEND_HUE`.** They read all three channels of
 * a pixel to decide one of them, so they cannot be a blend_channel case at all;
 * a mode added above LUMINOSITY without moving that test would be dispatched as
 * non-separable and read a garbage formula. Add separable modes below HUE. */
enum {
  /* Not a blend mode: "this layer replaces what is under it", which is what
   * over()'s early-out does for an opaque normal layer at full opacity. That
   * test is a reduction over the whole region rather than a per-pixel one, so
   * it stays in Python -- see the note on warlockc_over_f32 -- and reaches the
   * fused stack kernel as this sentinel. */
  WARLOCKC_BLEND_REPLACE = -1,
  WARLOCKC_BLEND_NORMAL = 0,
  WARLOCKC_BLEND_MULTIPLY = 1,
  WARLOCKC_BLEND_SCREEN = 2,
  WARLOCKC_BLEND_OVERLAY = 3,
  WARLOCKC_BLEND_ADD = 4,
  WARLOCKC_BLEND_DARKEN = 5,
  WARLOCKC_BLEND_LIGHTEN = 6,
  WARLOCKC_BLEND_COLOR_DODGE = 7,
  WARLOCKC_BLEND_COLOR_BURN = 8,
  WARLOCKC_BLEND_HARD_LIGHT = 9,
  WARLOCKC_BLEND_SOFT_LIGHT = 10,
  WARLOCKC_BLEND_DIFFERENCE = 11,
  /* Separable, and the two arithmetic ones follow Krita's definitions rather
   * than the W3C's, which has no word for them -- see composite.blend. */
  WARLOCKC_BLEND_EXCLUSION = 12,
  WARLOCKC_BLEND_SUBTRACT = 13,
  WARLOCKC_BLEND_DIVIDE = 14,
  /* Non-separable from here down; keep them last, see above. */
  WARLOCKC_BLEND_HUE = 15,
  WARLOCKC_BLEND_SATURATION = 16,
  WARLOCKC_BLEND_COLOR = 17,
  WARLOCKC_BLEND_LUMINOSITY = 18
};

/* Composite `source` onto `backdrop`, straight alpha, float32, 0..1, four
 * channels last -- warlock.studio.inker.composite.over's slow path.
 *
 * Strides count *floats* between row starts, so a rect slice of a larger
 * canvas passes without a copy; within a row the four channels of a pixel are
 * contiguous. `out` may alias `backdrop`. h and w are both > 0 and every
 * buffer holds h rows of w pixels -- Python checks that before every call.
 *
 * The `opacity >= 1 and mode == normal and every source alpha == 1` early-out
 * stays in Python and is deliberately not reproduced per pixel: an opaque
 * pixel inside a non-opaque source goes through the whole formula in numpy,
 * and a per-pixel shortcut would diverge in the last ulp. */
WARLOCKC_API void warlockc_over_f32(const float *backdrop,
                                    int64_t backdrop_stride,
                                    const float *source, int64_t source_stride,
                                    float *out, int64_t out_stride, int64_t h,
                                    int64_t w, float opacity, int32_t mode);

/* Write `rgba` over `before` at per-pixel `weight` -- composite.paint_colour.
 *
 * Straight alpha again, but in the 0..255 domain this one has always spoken;
 * `weight` is a single-channel (h, w) plane in 0..1 and `rgba` is four floats
 * with integral values. Strides as above, `weight_stride` counting floats
 * between row starts of the plane. `out` may alias `before`. */
WARLOCKC_API void warlockc_paint_colour_f32(const float *before,
                                            int64_t before_stride,
                                            const float *weight,
                                            int64_t weight_stride, float *out,
                                            int64_t out_stride, int64_t h,
                                            int64_t w, const float *rgba);

/* Fold `n` uint8 layer crops bottom-first onto an optional float32 base --
 * composite.stack_region, which is what invalidate_all spends its time in.
 *
 * `layers[i]` points at the top-left pixel of layer i's crop and `strides[i]`
 * counts bytes between its row starts, so the crops are views into full
 * canvases and nothing is copied. `base` is NULL for "start from transparent
 * black"; otherwise it is h by w float32 with `base_stride` floats per row.
 *
 * The whole point is that the u8 -> float conversion is fused into the load:
 * the reference materialises a 64 MB float32 temporary per layer and folds it
 * with a second full-size allocation, which at 2048 square by six layers is
 * more time than the arithmetic. Parity holds because x / 255.0f is exact per
 * element -- the table this builds holds the same 256 values the division
 * would produce -- and the fold order is the Python loop's. */
WARLOCKC_API void warlockc_stack_f32(const uint8_t **layers,
                                     const int64_t *strides,
                                     const float *opacities,
                                     const int32_t *modes, int64_t n,
                                     float *out, int64_t out_stride, int64_t h,
                                     int64_t w, const float *base,
                                     int64_t base_stride);

/* float32 0..1 -> uint8, `count` elements of each, both C-contiguous --
 * composite.to_uint8, which is what every composite crosses on its way to the
 * texture upload, the flatten and every export.
 *
 * A four-line expression that costs three full-size float32 temporaries: at
 * 2048 square the reference reads and writes about 260 MB to produce 16, and
 * it runs on the frame thread once per invalidate. Here it is one pass.
 *
 * NaN is out of contract on both sides rather than handled: numpy's clip
 * propagates it and .astype(uint8) of a NaN is undefined, exactly as the cast
 * below is. Nothing upstream can produce one -- every input traces back to a
 * uint8 layer -- and defining it here would be a divergence, not a fix. */
WARLOCKC_API void warlockc_to_uint8_f32(const float *pixels, uint8_t *out,
                                        int64_t count);

/* The same narrowing for floats that are *already* in 0..255 -- the hand
 * rolled `np.clip(out + 0.5, 0, 255).astype(np.uint8)` expressions that
 * `composite.to_uint8_255` now owns, none of which can use the kernel above
 * because that one multiplies by 255 first.
 *
 * Seven call sites, not four: inker.brush._resolve / _filter,
 * inker.document.write_colour / gradient / apply_matte, inker.filters._rejoin
 * and inker.transform._unpremultiplied. The last three were still spelling the
 * expression out by hand on 2026-08-11 -- which is what this comment is for,
 * and is also why it is worth keeping accurate: a site that does not go
 * through the helper is a site the parity tests do not cover.
 *
 * A sibling rather than a scale parameter: the two differ by one multiply, and
 * a per-element branch or a per-element multiply by 1.0f to unify them would
 * cost more than the duplication does. Parity is exact for the same reason the
 * scaled one's is -- add, clamp in numpy's NaN-leaving direction, truncate --
 * and it matters more here, because these sites write straight into a layer's
 * pixels and a half-level shift is a different file on disk. */
WARLOCKC_API void warlockc_to_uint8_255_f32(const float *pixels, uint8_t *out,
                                            int64_t count);

/* Closed boundary loops around `mask >= threshold` --
 * warlock.studio.inker.selection.SelectionMask.contours, which is the one true
 * per-pixel Python loop left in the package and runs on the frame thread every
 * time the selection changes.
 *
 * Emits loops of (x, y) int32 lattice points in image coordinates: unit steps,
 * vertices in 0..w and 0..h inclusive, the first point *not* repeated at the
 * end, and no loop shorter than three points -- the reference's contract, and
 * pixel-edge accurate rather than smoothed because the ants have to sit on the
 * boundary the fill used.
 *
 * `stride` counts bytes between the mask's row starts. `scratch` is
 * w * (h + 1) + (w + 1) * h bytes of *zeroed* caller memory, one flag per
 * lattice edge -- this file allocates nothing, and a retry after -1 has to zero
 * it again. `points_out` holds cap_pts pairs and `loop_lens_out` cap_loops
 * counts; the exact requirement is the number of boundary edges, which the
 * caller already counts in numpy.
 *
 * Returns the number of loops, or -1 if either capacity was too small -- in
 * which case the Python side answers, as it does when the DLL is absent.
 *
 * Loop order, starting vertex and winding are unspecified here exactly as they
 * are in the reference (_chain starts from next(iter(set))), and the two
 * genuinely differ at a checkerboard corner: see the turn rule in contours.c.
 * The invariant both hold to is the set of unit edges. */
WARLOCKC_API int64_t warlockc_contours(const uint8_t *mask, int64_t stride,
                                       int64_t h, int64_t w, uint8_t threshold,
                                       uint8_t *scratch, int32_t *points_out,
                                       int64_t cap_pts, int32_t *loop_lens_out,
                                       int64_t cap_loops);

/* Grow or shrink an 8-bit selection mask by `radius`, clamp-to-edge.
 *
 * Mirrors warlock.studio.inker.selection._morph / _spread exactly, including
 * the neighbourhood: passes alternate between the 4-neighbour cross and the
 * 8-neighbour box, starting with the cross, so radius r applies ceil(r/2)
 * crosses and floor(r/2) boxes interleaved -- an *octagon*, which is neither a
 * disc nor a square and is the whole reason this could not be one call to some
 * library's dilate. The mask is 0..255 and soft, not binary: a max filter over
 * an antialiased edge moves the edge and keeps it soft.
 *
 * `op` is 0 for max (grow) and 1 for min (shrink). `scratch` is h * w bytes of
 * caller memory, contents ignored -- this file allocates nothing, and the two
 * buffers are ping-ponged one pass at a time. `src` and `out` may not alias.
 * Strides count bytes between row starts.
 *
 * Integer min/max, so bit-identity with the reference is arithmetic rather than
 * a rounding argument. */
WARLOCKC_API void warlockc_morph_u8(const uint8_t *src, int64_t src_stride,
                                    uint8_t *scratch, uint8_t *out,
                                    int64_t out_stride, int64_t h, int64_t w,
                                    int64_t radius, int32_t op);

/* Serpentine Floyd-Steinberg error diffusion, in place on a float32 RGB work
 * plane -- warlock.studio.inker.dither._floyd_steinberg's loop.
 *
 * The one kernel here whose reference is a *Python* loop rather than a numpy
 * expression, and the reason is in the reference's own docstring: error
 * diffusion is sequential by definition, every pixel's input depends on its
 * neighbours' already-quantised output, so there is no vectorisation of it that
 * is still Floyd-Steinberg. What that costs in Python is about ten numpy
 * dispatches per pixel -- measured at ~10 us/px, which is 43 seconds for one
 * 2048-square conversion and is flat in palette size, because the cost is the
 * dispatch and not the arithmetic. Here the whole per-pixel chain is a handful
 * of instructions.
 *
 * Scalar-sequential also means this kernel has none of the bit-parity tension a
 * float *reduction* has: there is no order to choose, so the transcription is
 * operand-for-operand and the bar is np.array_equal as everywhere else.
 *
 * `work` is h by w by 3 float32 in the 0..255 domain, `work_stride` counting
 * floats between row starts and the three channels of a pixel contiguous; it is
 * both input and output. `visible` is the h by w uint8 alpha mask (numpy's
 * bool array): a zero pixel is neither quantised nor used as an error sink,
 * because its dead RGB is not a colour. `entries` is n_entries by 3 float32,
 * contiguous. The caller does the final clamp-and-narrow in numpy, which is
 * vectorised and not worth a seam.
 *
 * Nothing is clamped in here. The reference lets the diffused values run out of
 * range and clips once at the end, and a clamp per write would be a different
 * dither -- errors that fall off the bottom of the range are supposed to keep
 * being owed. */
WARLOCKC_API void warlockc_dither_fs(float *work, int64_t work_stride,
                                     const uint8_t *visible,
                                     int64_t visible_stride,
                                     const float *entries, int64_t n_entries,
                                     int64_t h, int64_t w);

/* Nearest palette entry per query colour, exact integer squared-Euclidean RGB.
 *
 * Three call sites each built an n by p by 3 int32 temporary to find one
 * argmin: indexed.snap's distinct-colours reduction, the nearest search inside
 * index_plane.resolve, and *both* searches in dither._ordered. On a
 * high-entropy image the distinct-colour count is tens of thousands and the
 * palette is tens, so the temporary is the whole cost -- allocated, written and
 * thrown away to read one index per row.
 *
 * Queries are **int32, not uint8**, and that is a correctness requirement
 * rather than a convenience. _ordered's second search is over the reflection
 * `2c - p1`, which spans -255..510 by construction; a uint8 signature would
 * silently clamp exactly the search that exists to find an entry on the far
 * side, and would pass every test that only fed in-gamut colours.
 *
 * Ties go to the **lowest palette index** -- numpy argmin's first-minimum rule,
 * which is what makes duplicate palette entries behave the same either way. The
 * arithmetic is integer throughout, so bit-parity is reachable without any care
 * about rounding and no SIMD is wanted: the parity bar forbids reassociation
 * anyway and the scalar loop is already memory-bound on the palette.
 *
 * `queries` is n by 3 int32, contiguous; `palette` is p by 3 int32, contiguous;
 * `out` is n int32. p >= 1 and n >= 0 are the caller's guarantees. */
WARLOCKC_API void warlockc_palette_nearest_i32(const int32_t *queries,
                                               const int32_t *palette,
                                               int32_t *out, int64_t n,
                                               int64_t n_palette);

/* Four-connected flood from one seed, bounded by a match mask.
 *
 * The Python reference (plotter/tools.flood_mask) is *not* a per-cell queue --
 * the deque was replaced by frontier dilation long ago -- so its cost is not
 * per-cell dispatch but the number of dilation passes, which is the path
 * distance to the furthest reachable cell. On an open room that is small; on a
 * serpentine corridor it approaches the cell count, and the work goes
 * quadratic. This restores O(cells) with a real queue.
 *
 * `match` is h by w uint8: nonzero means "this cell may be entered". `out` is h
 * by w uint8, written 1 for reached and 0 elsewhere; the caller need not clear
 * it. `scratch` is h*w int32 of caller-owned queue storage. Four-connected,
 * bounded by the array -- nothing wraps. A seed outside the array or on a
 * non-matching cell reaches nothing, which is the reference's own behaviour.
 *
 * The parity bar is the exact reached set, which is byte-identical to the
 * dilation result by construction: both compute the four-connected transitive
 * closure of the seed within `match`, and that set is unique. */
WARLOCKC_API void warlockc_flood_u8(const uint8_t *match, int64_t match_stride,
                                    uint8_t *out, int64_t out_stride,
                                    int32_t *scratch, int64_t h, int64_t w,
                                    int64_t seed_x, int64_t seed_y);

/* Median-split BVH over triangles -- viewer/picking.build_bvh's inner loop.
 *
 * `tri_lo`, `tri_hi` and `centroid` are each n by 3 float64, contiguous.
 * `order` is n int64 and is written by the kernel (the caller need not fill
 * it). `lo` and `hi` are max_nodes by 3 float64; `left`, `right`, `first` and
 * `count` are max_nodes int64; `stack` is 4 * max_nodes int64 of caller-owned
 * frame storage. Returns the node count, or -1 if max_nodes was too small --
 * in which case nothing has been read back and the caller falls back.
 *
 * Children are linked as they are created, so there is no second pass; an
 * interior node reports count 0, exactly as the reference's `_link` leaves it.
 *
 * **The one kernel besides contours.c not held to output identity.**
 * np.argpartition is introselect and its permutation among equal keys is
 * unspecified, so a C median split legitimately builds a different -- equally
 * valid -- tree. The bar is the pick result (the tie-break is already pinned
 * to the lowest triangle index so that tree and linear sweep agree) plus the
 * structural invariants. The licence is argued in docs/INVARIANTS.md. */
WARLOCKC_API int64_t warlockc_bvh_build(const double *tri_lo,
                                        const double *tri_hi,
                                        const double *centroid, int64_t n_tris,
                                        int64_t leaf_size, int64_t *order,
                                        double *lo, double *hi, int64_t *left,
                                        int64_t *right, int64_t *first,
                                        int64_t *count, int64_t max_nodes,
                                        int64_t *stack);

/* Source-over blit of many equally sized tiles -- plotter/render.render_layer.
 *
 * `out` is out_h by out_w by 4 uint8 with `out_stride` bytes per row; `atlas`
 * is a contiguous stack of tiles, each tile_h by tile_w by 4 uint8, indexed by
 * `tile_index`. `xs` and `ys` are the top-left destination of each cell and
 * may be negative or past the edge -- the kernel clips exactly as
 * render._blit_over does. Cells are drawn in the order given, which is the
 * document's own draw order and matters wherever tiles overlap.
 *
 * **Only render._over's masked-copy branch**: binary source alpha, full
 * opacity, normal mode. Everything else stays on the numpy path. The
 * both-clear quirk is reproduced -- see cells.c. */
WARLOCKC_API void warlockc_blit_cells_u8(uint8_t *out, int64_t out_h,
                                         int64_t out_w, int64_t out_stride,
                                         const uint8_t *atlas, int64_t tile_h,
                                         int64_t tile_w,
                                         const int32_t *tile_index,
                                         const int64_t *xs, const int64_t *ys,
                                         int64_t n_cells);

/* Nearest palette entry per query colour, Euclidean in float64 Oklab.
 *
 * The float sibling of warlockc_palette_nearest_i32, for
 * pipelines/pixel.map_palette. `queries` is n by 3 float64, contiguous;
 * `palette` is p by 3 float64, contiguous; `out` is n int32. Ties go to the
 * lowest index. The sqrt is part of the contract, not an oversight --
 * palettef.c says why. */
WARLOCKC_API void warlockc_palette_nearest_f64(const double *queries,
                                               const double *palette,
                                               int32_t *out, int64_t n,
                                               int64_t n_palette);

#ifdef __cplusplus
}
#endif

#endif /* WARLOCKC_H */

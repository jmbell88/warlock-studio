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
 * One kernel is deliberately not held to output identity -- the contour tracer
 * in contours.c, whose reference leaves loop order and winding unspecified. Its
 * bar is the set of unit edges, and the licence for that is written down beside
 * the tests that assert it.
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
#define WARLOCKC_ABI 7

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

/* Separable blend modes: the numbers composite._MODE_IDS hands over.
 *
 * The names live in Python and the numbers live here, which means adding a
 * mode is one entry in BLEND_MODES, one in composite._MODE_IDS and one case in
 * blend_channel. The *number* is the coupling, so it is spelled out in both
 * places rather than inferred in either -- and deliberately not the position in
 * BLEND_MODES, which is menu order and is free to be regrouped. */
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
  WARLOCKC_BLEND_DIFFERENCE = 11
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

#ifdef __cplusplus
}
#endif

#endif /* WARLOCKC_H */

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
 * tests assert equality rather than closeness. That is why the build uses
 * /fp:precise (MSVC) or -ffp-contract=off (clang): a fused multiply-add
 * changes rounding, and a changed rounding moves a triangle edge by one pixel.
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
#define WARLOCKC_ABI 2

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

/* Separable blend modes, in the order of composite.BLEND_MODES.
 *
 * The names live in Python and the numbers live here, which means adding a
 * mode is one entry in BLEND_MODES, one in composite._MODE_IDS and one case in
 * blend_channel. The order is the coupling, so it is spelled out in both
 * places rather than inferred in either. */
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
  WARLOCKC_BLEND_ADD = 4
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

#ifdef __cplusplus
}
#endif

#endif /* WARLOCKC_H */

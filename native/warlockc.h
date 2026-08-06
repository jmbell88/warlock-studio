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
#define WARLOCKC_ABI 1

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

#ifdef __cplusplus
}
#endif

#endif /* WARLOCKC_H */

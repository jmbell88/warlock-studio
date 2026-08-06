/* Silhouette coverage rasterisation -- the inner half of
 * warlock.meshaudit._coverage.
 *
 * The Python this replaces is vectorised but pays for it in memory: a batch of
 * n triangles sharing a k x k window materialises about a dozen (n, k, k)
 * float64 temporaries at once, which is why it needs _BATCH_MAX_SPAN and
 * _BATCH_MAX_CELLS to keep the working set from becoming a multi-gigabyte
 * commit spike. Per triangle the arithmetic is a handful of flops; the cost is
 * moving those temporaries. Here the whole per-pixel chain lives in registers
 * and nothing is allocated at all, so neither cap has a reason to exist.
 *
 * Every quirk of the Python is reproduced deliberately, because the test bar is
 * a bit-identical mask rather than a similar one. They are marked below. */

#include "warlockc.h"

#include <math.h>

int32_t warlockc_abi(void) { return WARLOCKC_ABI; }

/* floor/ceil, then clamp, then truncate -- in that order.
 *
 * np.clip(np.floor(v), 0, last).astype(int) rounds *before* clamping, and the
 * two orders genuinely differ: floor(-2.5) is -3, which clamps to 0, whereas
 * clamping first gives 0.0 and floors to 0. They agree here only because the
 * clamp is to a non-negative range, which is exactly the kind of coincidence
 * that stops being true when someone changes the range. */
static int64_t clamp_to_pixel(double v, int32_t last) {
    if (v < 0.0) {
        return 0;
    }
    if (v > (double)last) {
        return last;
    }
    return (int64_t)v;
}

static double min3(double p, double q, double r) {
    double m = p < q ? p : q;
    return m < r ? m : r;
}

static double max3(double p, double q, double r) {
    double m = p > q ? p : q;
    return m > r ? m : r;
}

void warlockc_rasterise(const double *ax, const double *ay, const double *bx,
                        const double *by, const double *cx, const double *cy,
                        const double *area2, int64_t n, int32_t resolution,
                        uint8_t *covered) {
    const int32_t last = resolution - 1;

    for (int64_t i = 0; i < n; ++i) {
        const double a0 = ax[i], a1 = ay[i];
        const double b0 = bx[i], b1 = by[i];
        const double c0 = cx[i], c1 = cy[i];

        const int64_t x0 = clamp_to_pixel(floor(min3(a0, b0, c0)), last);
        const int64_t x1 = clamp_to_pixel(ceil(max3(a0, b0, c0)), last);
        const int64_t y0 = clamp_to_pixel(floor(min3(a1, b1, c1)), last);
        const int64_t y1 = clamp_to_pixel(ceil(max3(a1, b1, c1)), last);

        const int64_t span_x = x1 - x0;
        const int64_t span_y = y1 - y0;

        /* A triangle smaller than a pixel cannot be caught by a pixel-centre
         * test, so it splats into the pixel holding its centroid. The cast is a
         * truncation toward zero, matching numpy's .astype(int) -- and the
         * centroid is computed before the clamp, so a triangle just off the
         * left edge truncates to 0 rather than flooring to -1. */
        if (span_x <= 1 && span_y <= 1) {
            const int64_t px = clamp_to_pixel((a0 + b0 + c0) / 3.0, last);
            const int64_t py = clamp_to_pixel((a1 + b1 + c1) / 3.0, last);
            covered[py * (int64_t)resolution + px] = 1;
            continue;
        }

        /* Square window, side max(span) + 1, anchored at the clipped box
         * origin. Square even when the spans differ: the batched Python path
         * groups triangles by a single k and iterates k in both axes, so a wide
         * flat triangle samples rows past its own bounding box. Those samples
         * fall outside the triangle and change nothing -- but they are what the
         * reference does, and the clamped write below is what keeps them from
         * running off the mask. */
        const int64_t k = (span_x > span_y ? span_x : span_y) + 1;
        const double sign = area2[i] < 0.0 ? -1.0 : 1.0;

        for (int64_t row = 0; row < k; ++row) {
            const double gy = (double)(y0 + row) + 0.5;
            int64_t iy = y0 + row;
            if (iy > last) {
                iy = last;
            }
            const int64_t row_base = iy * (int64_t)resolution;

            for (int64_t col = 0; col < k; ++col) {
                const double gx = (double)(x0 + col) + 0.5;

                /* Operand order is the reference's, expression for expression.
                 * Reassociating is algebraically free and numerically is not:
                 * a different rounding here moves an edge by a pixel. */
                const double e0 = (b0 - a0) * (gy - a1) - (b1 - a1) * (gx - a0);
                const double e1 = (c0 - b0) * (gy - b1) - (c1 - b1) * (gx - b0);
                const double e2 = (a0 - c0) * (gy - c1) - (a1 - c1) * (gx - c0);

                if (e0 * sign >= 0.0 && e1 * sign >= 0.0 && e2 * sign >= 0.0) {
                    int64_t ix = x0 + col;
                    if (ix > last) {
                        ix = last;
                    }
                    covered[row_base + ix] = 1;
                }
            }
        }
    }
}

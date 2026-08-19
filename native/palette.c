/* Nearest palette entry per query colour: exact integer squared-Euclidean RGB.
 *
 * The reference is numpy's `argmin(((q[:, None] - p[None]) ** 2).sum(2), 1)`,
 * and the whole reason this exists is the temporary in the middle of it: an
 * n by p by 3 int32 array built, written and thrown away to read one index per
 * row. On a high-entropy 512-square image n is tens of thousands.
 *
 * Integer arithmetic throughout, so bit-parity needs no care about rounding and
 * no SIMD is wanted -- the parity bar forbids reassociation and the scalar loop
 * is memory-bound on the palette anyway. See warlockc.h for the int32 query
 * signature and why a uint8 one would be wrong.
 */

#include "warlockc.h"

WARLOCKC_API void warlockc_palette_nearest_i32(const int32_t *queries,
                                               const int32_t *palette,
                                               int32_t *out, int64_t n,
                                               int64_t n_palette) {
    for (int64_t i = 0; i < n; ++i) {
        const int32_t qr = queries[i * 3 + 0];
        const int32_t qg = queries[i * 3 + 1];
        const int32_t qb = queries[i * 3 + 2];
        /* The query range is -255..510 (the reflection search), so a difference
         * is at most 765 and its square at most 585225; three of those fit an
         * int64 with room to spare and cannot overflow. */
        int64_t best = INT64_MAX;
        int32_t pick = 0;
        for (int64_t j = 0; j < n_palette; ++j) {
            const int64_t dr = (int64_t)(qr - palette[j * 3 + 0]);
            const int64_t dg = (int64_t)(qg - palette[j * 3 + 1]);
            const int64_t db = (int64_t)(qb - palette[j * 3 + 2]);
            const int64_t d2 = dr * dr + dg * dg + db * db;
            /* Strictly less: the first minimum wins, which is numpy argmin's
             * rule and what makes duplicate palette entries behave the same
             * through either path. */
            if (d2 < best) {
                best = d2;
                pick = (int32_t)j;
            }
        }
        out[i] = pick;
    }
}

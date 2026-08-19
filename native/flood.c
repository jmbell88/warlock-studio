/* Four-connected flood from one seed, bounded by a match mask.
 *
 * The Python reference (plotter/tools.flood_mask) is frontier dilation, not a
 * per-cell queue, so its cost is the number of passes -- the path distance to
 * the furthest reachable cell. An open room is a handful of passes; a
 * serpentine corridor is one pass per cell and the work goes quadratic. A real
 * queue restores O(cells).
 *
 * The parity bar is the exact reached set, and it holds by construction rather
 * than by transcription: both implementations compute the four-connected
 * transitive closure of the seed within `match`, and that set is unique.
 */

#include "warlockc.h"

WARLOCKC_API void warlockc_flood_u8(const uint8_t *match, int64_t match_stride,
                                    uint8_t *out, int64_t out_stride,
                                    int32_t *scratch, int64_t h, int64_t w,
                                    int64_t seed_x, int64_t seed_y) {
    for (int64_t y = 0; y < h; ++y) {
        uint8_t *row = out + y * out_stride;
        for (int64_t x = 0; x < w; ++x) {
            row[x] = 0;
        }
    }
    if (h <= 0 || w <= 0) {
        return;
    }
    /* A seed outside the array, or on a cell the mask refuses, reaches nothing
     * -- which is the reference's own behaviour and not an error. */
    if (seed_x < 0 || seed_y < 0 || seed_x >= w || seed_y >= h) {
        return;
    }
    if (!match[seed_y * match_stride + seed_x]) {
        return;
    }

    /* The queue holds a flat index per cell. It can never hold more than h*w
     * entries because a cell is marked in `out` before it is pushed and is
     * therefore pushed at most once, which is what makes the caller's h*w
     * scratch exactly enough. */
    int64_t head = 0;
    int64_t tail = 0;
    out[seed_y * out_stride + seed_x] = 1;
    scratch[tail++] = (int32_t)(seed_y * w + seed_x);

    while (head < tail) {
        const int32_t here = scratch[head++];
        const int64_t y = (int64_t)here / w;
        const int64_t x = (int64_t)here % w;
        const int64_t dx[4] = {1, -1, 0, 0};
        const int64_t dy[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            const int64_t nx = x + dx[k];
            const int64_t ny = y + dy[k];
            /* Bounded by the array: nothing wraps. */
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
                continue;
            }
            if (out[ny * out_stride + nx]) {
                continue;
            }
            if (!match[ny * match_stride + nx]) {
                continue;
            }
            out[ny * out_stride + nx] = 1;
            scratch[tail++] = (int32_t)(ny * w + nx);
        }
    }
}

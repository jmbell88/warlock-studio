/* Serpentine Floyd-Steinberg error diffusion -- the loop in
 * warlock.studio.inker.dither._floyd_steinberg.
 *
 * The odd one out among these kernels: every other reference here is a numpy
 * expression whose cost is the temporaries it materialises, and this one is a
 * *Python* loop whose cost is about ten numpy dispatches per pixel. That is not
 * an oversight in the reference and the kernel is not a reproach to it -- error
 * diffusion is sequential by definition (every pixel's input depends on its
 * neighbours' already-quantised output), so there is no vectorisation of it
 * that is still Floyd-Steinberg, and the reference says exactly that. The only
 * way to make it cheap is to stop paying for the dispatch.
 *
 * Measured before this file existed: ~10 us/px, near enough flat in palette
 * size (620 ms at 4 entries against 876 ms at 256, both at 256 square), which
 * is the signature of a dispatch-bound loop rather than an arithmetic-bound
 * one. At 1024 square that is 10.6 seconds and at 2048 square about 43 -- on a
 * conversion the user can trigger from a preview dropdown.
 *
 * Being scalar and sequential also means this kernel has none of the bit-parity
 * tension a float reduction has. There is no summation order to choose and
 * nothing to vectorise, so the transcription is operand for operand and the bar
 * is np.array_equal, the same as everywhere else in here. The one place order
 * mattered is the *diffusion*: four neighbours are accumulated into in a fixed
 * sequence, several of them again from the next pixel along, and float addition
 * is not associative -- so the writes below are in the reference's order and
 * must stay there.
 */

#include "warlockc.h"

/* As fractions of sixteen: the pixel ahead takes 7, the row below takes 3/5/1
 * across the three cells under and beside it. Every one is a power-of-two
 * denominator and therefore exact in float32, which is why the reference's
 * Python floats and these literals are the same numbers rather than nearly. */
#define FS_AHEAD (7.0f / 16.0f)
#define FS_BELOW_BACK (3.0f / 16.0f)
#define FS_BELOW (5.0f / 16.0f)
#define FS_BELOW_AHEAD (1.0f / 16.0f)

void warlockc_dither_fs(float *work, int64_t work_stride, const uint8_t *visible,
                        int64_t visible_stride, const float *entries, int64_t n_entries,
                        int64_t h, int64_t w) {
    for (int64_t y = 0; y < h; ++y) {
        /* Serpentine: alternate rows run right to left. A one-directional scan
         * piles its error up towards one edge and leaves a visible lean on any
         * large flat area, which is the classic artefact of the naive
         * implementation -- the reference's reasoning, kept here because the
         * direction is what makes the two loops the same loop. */
        const int rightwards = (y % 2) == 0;
        const int64_t step = rightwards ? 1 : -1;
        const int64_t first = rightwards ? 0 : w - 1;
        const int64_t past = rightwards ? w : -1;

        float *row = work + y * work_stride;
        float *below_row = (y + 1 < h) ? work + (y + 1) * work_stride : (float *)0;
        const uint8_t *seen = visible + y * visible_stride;

        for (int64_t x = first; x != past; x += step) {
            /* Transparent pixels are neither quantised nor used as error
             * sinks; their dead RGB is not a colour and diffusing into it
             * would be diffusing into nothing. */
            if (!seen[x]) {
                continue;
            }

            float *cell = row + x * 3;
            /* Read out before the write below: the reference copies for the
             * same reason, and the bug it prevents there (every error coming
             * out zero, which is nearest wearing this function's name) is the
             * same bug here. */
            const float old[3] = {cell[0], cell[1], cell[2]};

            /* np.argmin over the squared distances -- so the *first* minimum
             * on a tie, which a strict `<` keeps. Each square rounds to float32
             * before it is accumulated, and the accumulator starts at zero,
             * because that is what numpy's three-element pairwise sum does. */
            int64_t pick = 0;
            float best = 0.0f;
            for (int64_t p = 0; p < n_entries; ++p) {
                const float *entry = entries + p * 3;
                float total = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    const float delta = entry[c] - old[c];
                    const float square = delta * delta;
                    total = total + square;
                }
                if (p == 0 || total < best) {
                    best = total;
                    pick = p;
                }
            }

            const float *chosen = entries + pick * 3;
            const float error[3] = {old[0] - chosen[0], old[1] - chosen[1], old[2] - chosen[2]};
            cell[0] = chosen[0];
            cell[1] = chosen[1];
            cell[2] = chosen[2];

            /* The four writes, in the reference's order. Nothing is clamped:
             * values are allowed to run out of 0..255 and are clipped once at
             * the end, because an error that falls off the bottom of the range
             * is still owed to the pixels after it. */
            const int64_t ahead = x + step;
            const int64_t back = x - step;
            const int has_ahead = ahead >= 0 && ahead < w;
            const int has_back = back >= 0 && back < w;

            if (has_ahead) {
                float *target = row + ahead * 3;
                for (int c = 0; c < 3; ++c) {
                    target[c] = target[c] + error[c] * FS_AHEAD;
                }
            }
            if (below_row) {
                if (has_back) {
                    float *target = below_row + back * 3;
                    for (int c = 0; c < 3; ++c) {
                        target[c] = target[c] + error[c] * FS_BELOW_BACK;
                    }
                }
                {
                    float *target = below_row + x * 3;
                    for (int c = 0; c < 3; ++c) {
                        target[c] = target[c] + error[c] * FS_BELOW;
                    }
                }
                if (has_ahead) {
                    float *target = below_row + ahead * 3;
                    for (int c = 0; c < 3; ++c) {
                        target[c] = target[c] + error[c] * FS_BELOW_AHEAD;
                    }
                }
            }
        }
    }
}

/* Straight-alpha compositing and the separable blend modes -- the slow half of
 * warlock.studio.inker.composite.
 *
 * The Python is fully vectorised and still expensive, because the cost was
 * never the arithmetic: one `over()` materialises about eight full-region
 * float32 temporaries (ao, mixed, the three products num is made of, the
 * divide's output, the where's), and `stack_region` calls it once per layer.
 * At 2048 square by six layers that is hundreds of megabytes of traffic on the
 * one thread that owns the GL context. Here the whole per-pixel chain lives in
 * registers: each input is read once, the output is written once, and nothing
 * is allocated at all.
 *
 * Every expression below is the reference's, operand for operand. Reassociating
 * is algebraically free and numerically is not, and the test bar is
 * np.array_equal against the numpy path rather than np.allclose. Two places
 * where that dictated the shape rather than the other way round are marked. */

#include "warlockc.h"

#include <math.h>

/* B(Cb, Cs) for one channel. `mode` is an index into composite.BLEND_MODES,
 * which is where the mapping is kept; anything unrecognised is normal, and the
 * Python side never sends one (an unknown mode falls back to numpy, which
 * raises ValueError out of `blend` exactly as it always did). */
static float blend_channel(int32_t mode, float cb, float cs) {
    switch (mode) {
    case WARLOCKC_BLEND_MULTIPLY:
        return cb * cs;
    case WARLOCKC_BLEND_SCREEN:
        return (cb + cs) - cb * cs;
    case WARLOCKC_BLEND_OVERLAY:
        /* hard-light with the operands swapped, which is the spec's own
         * wording. NaN <= 0.5f is false and np.where(nan <= 0.5, ...) picks the
         * same branch, so the comparison direction is the reference's too. */
        return cb <= 0.5f ? (2.0f * cb) * cs : 1.0f - (2.0f * (1.0f - cb)) * (1.0f - cs);
    case WARLOCKC_BLEND_ADD: {
        const float sum = cb + cs;
        /* np.minimum propagates NaN, so the clamp has to be written as the
         * comparison that is *false* for NaN and therefore returns the sum.
         * `sum < 1.0f ? sum : 1.0f` would quietly turn a NaN into 1. */
        return sum > 1.0f ? 1.0f : sum;
    }
    case WARLOCKC_BLEND_DARKEN:
        /* np.minimum propagates NaN and a bare `cb < cs ? cb : cs` does not:
         * a NaN operand compares false and the *other* one comes back. Same
         * trap as the add clamp above, moved from the bound to the operand. */
        if (cb != cb) {
            return cb;
        }
        if (cs != cs) {
            return cs;
        }
        return cb < cs ? cb : cs;
    case WARLOCKC_BLEND_LIGHTEN:
        if (cb != cb) {
            return cb;
        }
        if (cs != cs) {
            return cs;
        }
        return cb > cs ? cb : cs;
    case WARLOCKC_BLEND_DIFFERENCE: {
        const float diff = cb - cs;
        return diff < 0.0f ? -diff : diff;
    }
    case WARLOCKC_BLEND_HARD_LIGHT:
        /* Overlay with the operands swapped, which is what it *is* -- and
         * saying so rather than writing the formula out inverted is what makes
         * the identity exact rather than exact-to-a-rounding. The reference
         * spells it the same way, for the same reason. */
        return blend_channel(WARLOCKC_BLEND_OVERLAY, cs, cb);
    case WARLOCKC_BLEND_COLOR_DODGE: {
        /* The spec's three cases in its order: an empty backdrop stays empty
         * even under a full source, so the Cb test comes first. The reference
         * guards the divisor rather than clipping an infinity afterwards,
         * because 1 - Cs is *exactly* zero for a saturated channel. */
        if (cb <= 0.0f) {
            return 0.0f;
        }
        if (cs >= 1.0f) {
            return 1.0f;
        }
        const float ratio = cb / (1.0f - cs);
        return ratio > 1.0f ? 1.0f : ratio;
    }
    case WARLOCKC_BLEND_COLOR_BURN: {
        if (cb >= 1.0f) {
            return 1.0f;
        }
        if (cs <= 0.0f) {
            return 0.0f;
        }
        const float ratio = (1.0f - cb) / cs;
        return 1.0f - (ratio > 1.0f ? 1.0f : ratio);
    }
    case WARLOCKC_BLEND_SOFT_LIGHT: {
        if (cs <= 0.5f) {
            return cb - ((1.0f - 2.0f * cs) * cb) * (1.0f - cb);
        }
        /* D(Cb). The clamp under the root matches np.maximum in the reference
         * and is not defensive about the spec: Cb is a composited channel and
         * over()'s divide is not clipped, so a value a hair below zero is
         * reachable and sqrtf of it is a NaN the reference does not produce. */
        const float d = cb <= 0.25f ? ((16.0f * cb - 12.0f) * cb + 4.0f) * cb
                                    : sqrtf(cb < 0.0f ? 0.0f : cb);
        return cb + (2.0f * cs - 1.0f) * (d - cb);
    }
    default:
        return cs;
    }
}

void warlockc_over_f32(const float *backdrop, int64_t backdrop_stride, const float *source,
                       int64_t source_stride, float *out, int64_t out_stride, int64_t h,
                       int64_t w, float opacity, int32_t mode) {
    for (int64_t y = 0; y < h; ++y) {
        const float *bp = backdrop + y * backdrop_stride;
        const float *sp = source + y * source_stride;
        float *op = out + y * out_stride;

        for (int64_t x = 0; x < w; ++x, bp += 4, sp += 4, op += 4) {
            /* Every input read before any output write: `out` is allowed to
             * alias `backdrop`, and the two need not share a stride. */
            const float cb[3] = {bp[0], bp[1], bp[2]};
            const float cs[3] = {sp[0], sp[1], sp[2]};
            const float ab = bp[3];

            const float a_s = sp[3] * opacity;
            const float ao = a_s + ab * (1.0f - a_s);

            /* Per pixel, not per channel -- these are the (H, W, 1) arrays the
             * reference broadcasts across the three colour channels. */
            const float k_src = a_s * (1.0f - ab);
            const float k_mix = a_s * ab;
            const float k_back = (1.0f - a_s) * ab;

            for (int c = 0; c < 3; ++c) {
                const float mixed = blend_channel(mode, cb[c], cs[c]);
                const float num = (k_src * cs[c] + k_mix * mixed) + k_back * cb[c];
                /* np.divide(..., where=ao > 0) leaves the output untouched
                 * where the mask is false and the reference then np.wheres
                 * zeros in; net semantics are "0 where ao <= 0", and the
                 * division only ever happens where it is taken. */
                op[c] = ao > 0.0f ? num / ao : 0.0f;
            }
            op[3] = ao;
        }
    }
}

void warlockc_paint_colour_f32(const float *before, int64_t before_stride, const float *weight,
                               int64_t weight_stride, float *out, int64_t out_stride, int64_t h,
                               int64_t w, const float *rgba) {
    /* The reference computes `colour[3] / 255.0` as a Python scalar -- so in
     * double -- and the multiply against the float32 weight rounds it once.
     * The other division is between a float32 array and a scalar and therefore
     * happens in float. The two are deliberately spelled differently here. */
    const float src_scale = (float)((double)rgba[3] / 255.0);

    for (int64_t y = 0; y < h; ++y) {
        const float *bp = before + y * before_stride;
        const float *wp = weight + y * weight_stride;
        float *op = out + y * out_stride;

        for (int64_t x = 0; x < w; ++x, bp += 4, ++wp, op += 4) {
            const float b[4] = {bp[0], bp[1], bp[2], bp[3]};

            const float src_a = src_scale * *wp;
            const float dst_a = b[3] / 255.0f;
            const float out_a = src_a + dst_a * (1.0f - src_a);
            const float share = out_a > 0.0f ? src_a / out_a : 0.0f;

            for (int c = 0; c < 3; ++c) {
                op[c] = b[c] + (rgba[c] - b[c]) * share;
            }
            op[3] = out_a * 255.0f;
        }
    }
}

void warlockc_stack_f32(const uint8_t **layers, const int64_t *strides, const float *opacities,
                        const int32_t *modes, int64_t n, float *out, int64_t out_stride,
                        int64_t h, int64_t w, const float *base, int64_t base_stride) {
    /* to_float is `x.astype(float32) / 255.0`, and there are only 256 answers.
     * Computed here rather than divided per channel per layer per pixel: at
     * 2048 square by six layers that division would be a hundred million of
     * them. Same expression, same operands, so the values are the division's
     * exactly -- this is a table of the reference, not an approximation of it.
     * Local rather than static because the header promises no state. */
    float from_u8[256];
    for (int32_t v = 0; v < 256; ++v) {
        from_u8[v] = (float)v / 255.0f;
    }

    for (int64_t y = 0; y < h; ++y) {
        const float *bp = base ? base + y * base_stride : (const float *)0;
        float *op = out + y * out_stride;

        for (int64_t x = 0; x < w; ++x) {
            float acc[4];
            if (base) {
                acc[0] = bp[x * 4 + 0];
                acc[1] = bp[x * 4 + 1];
                acc[2] = bp[x * 4 + 2];
                acc[3] = bp[x * 4 + 3];
            } else {
                acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
            }

            for (int64_t i = 0; i < n; ++i) {
                const uint8_t *lp = layers[i] + y * strides[i] + x * 4;
                const float cs[3] = {from_u8[lp[0]], from_u8[lp[1]], from_u8[lp[2]]};
                const float src_a = from_u8[lp[3]];

                if (modes[i] == WARLOCKC_BLEND_REPLACE) {
                    acc[0] = cs[0];
                    acc[1] = cs[1];
                    acc[2] = cs[2];
                    acc[3] = src_a;
                    continue;
                }

                const float ab = acc[3];
                const float a_s = src_a * opacities[i];
                const float ao = a_s + ab * (1.0f - a_s);
                const float k_src = a_s * (1.0f - ab);
                const float k_mix = a_s * ab;
                const float k_back = (1.0f - a_s) * ab;

                /* The backdrop is the accumulator, so every channel has to be
                 * read before any is written -- hence the third buffer. */
                float next[3];
                for (int c = 0; c < 3; ++c) {
                    const float mixed = blend_channel(modes[i], acc[c], cs[c]);
                    const float num = (k_src * cs[c] + k_mix * mixed) + k_back * acc[c];
                    next[c] = ao > 0.0f ? num / ao : 0.0f;
                }
                acc[0] = next[0];
                acc[1] = next[1];
                acc[2] = next[2];
                acc[3] = ao;
            }

            op[x * 4 + 0] = acc[0];
            op[x * 4 + 1] = acc[1];
            op[x * 4 + 2] = acc[2];
            op[x * 4 + 3] = acc[3];
        }
    }
}

void warlockc_to_uint8_f32(const float *pixels, uint8_t *out, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        const float scaled = pixels[i] * 255.0f + 0.5f;
        /* np.clip is min(max(a, 0), 255), and the comparisons are written in
         * the direction that leaves NaN alone for the same reason the `add`
         * blend's clamp is: both of numpy's are NaN-propagating. */
        const float clipped = scaled < 0.0f ? 0.0f : (scaled > 255.0f ? 255.0f : scaled);
        /* .astype truncates toward zero rather than rounding -- the +0.5f
         * above is what makes that a round, and doing it in one step here
         * would move every value by half a level. */
        out[i] = (uint8_t)clipped;
    }
}

void warlockc_to_uint8_255_f32(const float *pixels, uint8_t *out,
                               int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        /* No scale: the callers' floats are already levels, not fractions.
         * Everything else is warlockc_to_uint8_f32's reasoning verbatim --
         * clamp written so a NaN falls through both comparisons the way
         * numpy's clip propagates one, then truncate, because the reference's
         * .astype is the truncation and the +0.5f above is the round. */
        const float biased = pixels[i] + 0.5f;
        const float clipped = biased < 0.0f ? 0.0f : (biased > 255.0f ? 255.0f : biased);
        out[i] = (uint8_t)clipped;
    }
}

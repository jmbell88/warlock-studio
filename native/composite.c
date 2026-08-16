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
    case WARLOCKC_BLEND_EXCLUSION:
        /* `backdrop + source - 2.0 * backdrop * source`, in the association
         * Python gives it: (cb + cs) - ((2 * cb) * cs). Reassociating is
         * algebraically free and numerically is not. */
        return (cb + cs) - ((2.0f * cb) * cs);
    case WARLOCKC_BLEND_SUBTRACT: {
        /* np.maximum(cb - cs, 0.0), and its NaN rule is the darken case's:
         * numpy returns the *first* operand when it is NaN, so the guard
         * cannot be folded into the comparison. Clamped at the bottom rather
         * than wrapped -- the reference says why. */
        const float diff = cb - cs;
        if (diff != diff) {
            return diff;
        }
        /* `diff > 0` rather than `>= 0` so a -0.0 difference comes back as the
         * +0.0 numpy's maximum returns for it (it picks the second operand on
         * anything that is not strictly greater). */
        return diff > 0.0f ? diff : 0.0f;
    }
    case WARLOCKC_BLEND_DIVIDE: {
        /* Krita's zero convention, which the reference pins and this mirrors
         * branch for branch: a zero divisor gives white where there is
         * anything to divide and stays black where there is not. A NaN source
         * fails `> 0` and lands in the same arm, exactly as np.where does. */
        if (cs > 0.0f) {
            const float ratio = cb / cs;
            return ratio > 1.0f ? 1.0f : ratio;
        }
        return cb > 0.0f ? 1.0f : 0.0f;
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
        /* The divisor is guarded as the reference guards it --
         * `np.where(denom > 0.0, denom, 1.0)`. Past the `cs >= 1` return above
         * the only way here with a non-positive denominator is a NaN source,
         * where `>` is false, so numpy divides by one and returns Cb while an
         * unguarded divide returns NaN.
         *
         * **This difference is not observable and the guard is written for
         * shape, not for a bug.** `blend_channel` is reached only through
         * `warlockc_over_f32`, and a NaN Cs reaches the output there through
         * `k_src * cs` whatever B() returned -- 0 * NaN is NaN, so even a
         * fully opaque backdrop does not mask it. A test at the seam therefore
         * cannot tell the two apart, and 2026-08-11's attempt to write one
         * passed against both. It is here because every other branch in this
         * file spells its NaN behaviour out, and a reader comparing this one
         * against the reference should not have to re-derive that the
         * omission was harmless. */
        const float denom = 1.0f - cs;
        const float ratio = cb / (denom > 0.0f ? denom : 1.0f);
        return ratio > 1.0f ? 1.0f : ratio;
    }
    case WARLOCKC_BLEND_COLOR_BURN: {
        if (cb >= 1.0f) {
            return 1.0f;
        }
        if (cs <= 0.0f) {
            return 0.0f;
        }
        /* The dodge's guard, on the other operand: `np.where(source > 0.0,
         * source, 1.0)`. */
        const float ratio = (1.0f - cb) / (cs > 0.0f ? cs : 1.0f);
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

/* -- the non-separable four -------------------------------------------------
 *
 * hue / saturation / color / luminosity, transcribed from composite._lum,
 * _sat, _clip_colour, _set_lum, _set_sat -- which are themselves the W3C
 * compositing spec's pseudo-code rather than anybody's simplification of it.
 * They read all three channels of a pixel to decide one of them, so they are
 * not a blend_channel case: `blend_rgb` below routes them here with the whole
 * pixel.
 *
 * Two things make the parity argument here different from the separable ones,
 * and both are settled by measurement rather than by reading:
 *
 *   * The reference's `np.where` evaluates *both* arms and selects. This code
 *     evaluates only the taken one. That is identical wherever the reference
 *     defines a value, and the untaken arm is exactly where the reference
 *     guards its divisors -- so nothing that is selected here was computed
 *     differently there.
 *   * `_lum` is a three-element float32 *reduction*, which is where an order
 *     could be chosen and lost. numpy's pairwise sum takes its `n < 8` branch,
 *     which accumulates left to right from a zero seed; that was checked
 *     against this build's numpy over two million random triples before this
 *     was written, and it is why the seed below is spelled out rather than
 *     started at the first product.
 */

/* np.min / np.max over the channel axis, which propagate NaN where a bare
 * comparison chain would return the other operand -- the darken case's trap,
 * one axis over. */
static float min3(const float c[3]) {
    float m = c[0];
    for (int i = 1; i < 3; ++i) {
        if (m != m) {
            return m;
        }
        if (c[i] != c[i] || c[i] < m) {
            m = c[i];
        }
    }
    return m;
}

static float max3(const float c[3]) {
    float m = c[0];
    for (int i = 1; i < 3; ++i) {
        if (m != m) {
            return m;
        }
        if (c[i] != c[i] || c[i] > m) {
            m = c[i];
        }
    }
    return m;
}

/* The spec's luminance weights as float32, not Rec.709: this is a blend mode
 * and its answer has to match the editor on the other end of the file. */
static float lum3(const float c[3]) {
    const float p0 = c[0] * 0.30f;
    const float p1 = c[1] * 0.59f;
    const float p2 = c[2] * 0.11f;
    /* The zero seed is numpy's, not decoration -- see the note above. */
    return (((0.0f + p0) + p1) + p2);
}

static float sat3(const float c[3]) { return max3(c) - min3(c); }

/* The spec's ClipColor: pull an out-of-range colour back towards its own
 * luminance, so that fixing the range cannot change the luminance. */
static void clip_colour3(const float c[3], float out[3]) {
    const float lum = lum3(c);
    const float low = min3(c);
    const float high = max3(c);

    /* n and x are computed once, before C is touched, as the pseudo-code has
     * them -- so the second stage reads the *original* extremes and the
     * original luminance, not the first stage's. */
    float mid[3];
    if (low < 0.0f) {
        const float under = lum - low;
        const float denom = under != 0.0f ? under : 1.0f;
        for (int i = 0; i < 3; ++i) {
            mid[i] = lum + ((c[i] - lum) * lum) / denom;
        }
    } else {
        mid[0] = c[0];
        mid[1] = c[1];
        mid[2] = c[2];
    }

    if (high > 1.0f) {
        const float over_ = high - lum;
        const float denom = over_ != 0.0f ? over_ : 1.0f;
        for (int i = 0; i < 3; ++i) {
            out[i] = lum + ((mid[i] - lum) * (1.0f - lum)) / denom;
        }
    } else {
        out[0] = mid[0];
        out[1] = mid[1];
        out[2] = mid[2];
    }
}

static void set_lum3(const float c[3], float lum, float out[3]) {
    /* `colour + (lum - _lum(colour))`: the difference is a per-pixel scalar in
     * the reference too, broadcast across the channels, so it is computed once
     * and rounded once. */
    const float delta = lum - lum3(c);
    const float shifted[3] = {c[0] + delta, c[1] + delta, c[2] + delta};
    clip_colour3(shifted, out);
}

/* numpy's npy_float_LT, which is what argsort orders by: NaN sorts last, and
 * the comparison is strict so equal values keep their order. */
static int sort_lt(float a, float b) { return (a < b) || ((b != b) && (a == a)); }

/* The spec's SetSat. The reference says it with argsort; three elements is
 * below numpy's introsort threshold, so what argsort actually runs is an
 * insertion sort, which is *stable* -- and the reference's docstring leans on
 * that for the two-equal-channels case. This is that insertion sort, so the
 * tie-break is the same by construction rather than by coincidence. */
static void set_sat3(const float c[3], float sat, float out[3]) {
    int order[3] = {0, 1, 2};
    for (int i = 1; i < 3; ++i) {
        const int key = order[i];
        int j = i - 1;
        while (j >= 0 && sort_lt(c[key], c[order[j]])) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }

    const float low = c[order[0]];
    const float mid = c[order[1]];
    const float high = c[order[2]];
    const float span = high - low;

    /* On a grey pixel span is zero and every channel ends at zero, which is
     * the spec's else-branch. The reference's guarded divisor is the untaken
     * arm of the same test. */
    if (span > 0.0f) {
        out[order[0]] = 0.0f;
        out[order[1]] = ((mid - low) * sat) / span;
        out[order[2]] = sat;
    } else {
        out[order[0]] = 0.0f;
        out[order[1]] = 0.0f;
        out[order[2]] = 0.0f;
    }
}

static void blend_nonseparable(int32_t mode, const float cb[3], const float cs[3], float out[3]) {
    float tinted[3];
    switch (mode) {
    case WARLOCKC_BLEND_HUE:
        set_sat3(cs, sat3(cb), tinted);
        set_lum3(tinted, lum3(cb), out);
        return;
    case WARLOCKC_BLEND_SATURATION:
        set_sat3(cb, sat3(cs), tinted);
        set_lum3(tinted, lum3(cb), out);
        return;
    case WARLOCKC_BLEND_COLOR:
        set_lum3(cs, lum3(cb), out);
        return;
    default: /* WARLOCKC_BLEND_LUMINOSITY */
        set_lum3(cb, lum3(cs), out);
        return;
    }
}

/* Is this mode one that reads the whole pixel? The split is the enum's, and
 * warlockc.h says why HUE has to stay the bottom of the non-separable range. */
static int is_nonseparable(int32_t mode) { return mode >= WARLOCKC_BLEND_HUE; }

/* Co for one channel:
 *
 *     Co = [ as(1-ab)Cs + as*ab*B(Cb,Cs) + (1-as)ab*Cb ] / ao
 *
 * The *only* copy of the compositing arithmetic in this file, called from four
 * places -- both kernels, each on both sides of the separable/non-separable
 * branch. That branch is hoisted out to the call sites rather than taken inside
 * a `blend_rgb` helper because the separable path is then never made to
 * materialise a three-float `mixed` buffer at all, which measured about 3%
 * on an all-separable full-canvas fold and 5% with a non-separable layer in it.
 * Duplicating *this* expression to buy that would have been the wrong trade --
 * two copies of a formula held to bit-parity are two things to keep equal --
 * so it is written once and the cheap half is what got duplicated.
 *
 * `ao > 0` mirrors np.divide(..., where=ao > 0) followed by the reference's
 * np.where: the net semantics are "0 where ao <= 0", and the division only
 * ever happens where it is taken. */
static float combine_channel(float k_src, float cs, float k_mix, float mixed, float k_back,
                             float cb, float ao) {
    const float num = (k_src * cs + k_mix * mixed) + k_back * cb;
    return ao > 0.0f ? num / ao : 0.0f;
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

            if (is_nonseparable(mode)) {
                float mixed[3];
                blend_nonseparable(mode, cb, cs, mixed);
                for (int c = 0; c < 3; ++c) {
                    op[c] = combine_channel(k_src, cs[c], k_mix, mixed[c], k_back, cb[c], ao);
                }
            } else {
                for (int c = 0; c < 3; ++c) {
                    const float mixed = blend_channel(mode, cb[c], cs[c]);
                    op[c] = combine_channel(k_src, cs[c], k_mix, mixed, k_back, cb[c], ao);
                }
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
                 * read before any is written -- hence the third buffer. The
                 * non-separable modes need that anyway: they read all three
                 * of the accumulator's channels to decide one. */
                const float cb[3] = {acc[0], acc[1], acc[2]};
                float next[3];
                if (is_nonseparable(modes[i])) {
                    float mixed[3];
                    blend_nonseparable(modes[i], cb, cs, mixed);
                    for (int c = 0; c < 3; ++c) {
                        next[c] = combine_channel(k_src, cs[c], k_mix, mixed[c], k_back, cb[c], ao);
                    }
                } else {
                    for (int c = 0; c < 3; ++c) {
                        const float mixed = blend_channel(modes[i], cb[c], cs[c]);
                        next[c] = combine_channel(k_src, cs[c], k_mix, mixed, k_back, cb[c], ao);
                    }
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

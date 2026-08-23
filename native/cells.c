/* Tile-layer compositing -- plotter/render.render_layer's per-cell loop.
 *
 * The largest single number this programme measured: 4348 ms for a 200x200
 * three-layer map of 32px tiles, which is 36 us per cell across 120,000 cells.
 * That is far past dispatch overhead -- a 32-square tile is 1024 pixels, so
 * there is real per-cell numpy work -- and a C blit does it in about a
 * microsecond. Batch 2 named this kernel and left it unwritten because the
 * partial-alpha case it would have answered had no measured corpus. That
 * reason never covered the binary-alpha case, which is what a tileset actually
 * is and is still four seconds.
 *
 * **Only the binary-alpha, full-opacity, normal-mode case.** That is `_over`'s
 * own masked-copy branch, and every other combination falls back to the numpy
 * path unchanged -- the caller tests for it, because deciding "is this alpha
 * binary" is one vectorised pass per distinct tile rather than per cell.
 *
 * The exact-identity trap `_over` documents is reproduced rather than tidied
 * up: where source *and* destination are both fully clear, the reference
 * writes rgb 0, discarding whatever colour was stored under a zero alpha. A
 * kernel that left those pixels alone would be more sensible and would not be
 * the same function.
 */

#include "warlockc.h"

void warlockc_blit_cells_u8(uint8_t *out, int64_t out_h, int64_t out_w,
                            int64_t out_stride, const uint8_t *atlas,
                            int64_t tile_h, int64_t tile_w,
                            const int32_t *tile_index, const int64_t *xs,
                            const int64_t *ys, int64_t n_cells) {
  const int64_t tile_stride = tile_w * 4;
  const int64_t tile_size = tile_h * tile_stride;

  for (int64_t cell = 0; cell < n_cells; cell++) {
    const int64_t x0 = xs[cell];
    const int64_t y0 = ys[cell];

    /* The clip idiom `_blit_over` owns: a tile taller than its cell and a
     * layer nudged by a negative offset are the same arithmetic. */
    int64_t sx0 = x0 < 0 ? -x0 : 0;
    int64_t sy0 = y0 < 0 ? -y0 : 0;
    int64_t dx0 = x0 > 0 ? x0 : 0;
    int64_t dy0 = y0 > 0 ? y0 : 0;
    int64_t span_w = tile_w - sx0;
    int64_t span_h = tile_h - sy0;
    if (out_w - dx0 < span_w) {
      span_w = out_w - dx0;
    }
    if (out_h - dy0 < span_h) {
      span_h = out_h - dy0;
    }
    if (span_w <= 0 || span_h <= 0) {
      continue;
    }

    const uint8_t *src = atlas + tile_index[cell] * tile_size;
    for (int64_t row = 0; row < span_h; row++) {
      const uint8_t *s = src + (sy0 + row) * tile_stride + sx0 * 4;
      uint8_t *d = out + (dy0 + row) * out_stride + dx0 * 4;
      for (int64_t col = 0; col < span_w; col++) {
        if (s[3] == 255) {
          d[0] = s[0];
          d[1] = s[1];
          d[2] = s[2];
          d[3] = 255;
        } else if (d[3] == 0) {
          /* Both clear. The reference's `dead` mask writes rgb 0 here and
           * leaves alpha at 0; see the header note. */
          d[0] = 0;
          d[1] = 0;
          d[2] = 0;
        }
        s += 4;
        d += 4;
      }
    }
  }
}

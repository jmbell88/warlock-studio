/* Selection grow/shrink: greyscale morphology with an octagonal element.
 *
 * The reference (warlock.studio.inker.selection._morph + _spread) runs one
 * pass per unit of radius, and each pass is an np.pad of the whole mask plus
 * four or eight full-frame np.maximum/np.minimum calls, none of them with
 * out=, so a radius of 10 on a 2048-square mask is ~50 whole-array allocations
 * and was measured at 135 ms per click.
 *
 * This keeps two buffers and ping-pongs between them, one pass per radius unit,
 * reading each neighbourhood once. Same passes, same order, same alternation --
 * see warlockc_morph_u8 in warlockc.h for why the element is an octagon and why
 * that ruled out handing the whole thing to a library.
 *
 * BIT-PARITY is arithmetic here rather than a rounding argument: min and max
 * over uint8 are exact, so agreement with the reference needs only the same
 * neighbourhood, the same edge rule and the same number of passes.
 */

#include "warlockc.h"

#define MORPH_MAX 0
#define MORPH_MIN 1

/* Cells outside the mask replicate the edge, which is np.pad(mode="edge") --
 * the reference pads once per pass and takes a shifted window out of it. Here
 * that shows up as the clamped row indices below and the two end-column lines,
 * rather than as a bounds test per neighbour per pixel. */

/* One column of the neighbourhood, folded into an accumulator row.
 *
 * Split out so the interior loops below are branch-free straight-line code over
 * three known rows: a clamped accessor called per neighbour per pixel is ~40
 * unpredictable branches per output, which measured *slower* than the numpy
 * reference it replaced. The compiler vectorises these; uint8 min/max is exact,
 * so unlike a float reduction, vectorising cannot change the answer. */
#define FOLD_ROW(dst_row, src_row, lo, hi, op)                                 \
  do {                                                                         \
    if ((op) == MORPH_MAX) {                                                   \
      for (int64_t i = (lo); i < (hi); ++i) {                                  \
        if ((src_row)[i] > (dst_row)[i]) {                                     \
          (dst_row)[i] = (src_row)[i];                                         \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = (lo); i < (hi); ++i) {                                  \
        if ((src_row)[i] < (dst_row)[i]) {                                     \
          (dst_row)[i] = (src_row)[i];                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

/* One spread. `diagonal` adds the four corners, making the 8-neighbourhood. */
static void spread(const uint8_t *src, int64_t src_stride, uint8_t *dst,
                   int64_t dst_stride, int64_t h, int64_t w, int diagonal,
                   int32_t op) {
  for (int64_t y = 0; y < h; ++y) {
    uint8_t *row = dst + y * dst_stride;
    const uint8_t *mid = src + y * src_stride;
    /* Clamp-to-edge on the first and last row is the whole of the vertical
     * edge rule, so the neighbour rows are just these two indices. */
    const uint8_t *up = src + (y > 0 ? y - 1 : 0) * src_stride;
    const uint8_t *down = src + (y < h - 1 ? y + 1 : h - 1) * src_stride;

    /* The reference seeds with mask.copy() and folds neighbours in, so the
     * centre is always part of the answer. */
    for (int64_t x = 0; x < w; ++x) {
      row[x] = mid[x];
    }
    FOLD_ROW(row, up, 0, w, op);
    FOLD_ROW(row, down, 0, w, op);

    /* Horizontal neighbours. The interior runs unclamped; the two end columns
     * clamp onto themselves, which is what makes them a separate line rather
     * than a branch inside the loop. */
    if (w == 1) {
      continue;
    }
    FOLD_ROW(row + 1, mid, 0, w - 1, op);  /* left  neighbour of x is mid[x-1] */
    FOLD_ROW(row, mid + 1, 0, w - 1, op);  /* right neighbour of x is mid[x+1] */
    if (op == MORPH_MAX) {
      if (mid[0] > row[0]) row[0] = mid[0];
      if (mid[w - 1] > row[w - 1]) row[w - 1] = mid[w - 1];
    } else {
      if (mid[0] < row[0]) row[0] = mid[0];
      if (mid[w - 1] < row[w - 1]) row[w - 1] = mid[w - 1];
    }

    if (!diagonal) {
      continue;
    }
    for (int side = 0; side < 2; ++side) {
      const uint8_t *near = side == 0 ? up : down;
      FOLD_ROW(row + 1, near, 0, w - 1, op);
      FOLD_ROW(row, near + 1, 0, w - 1, op);
      if (op == MORPH_MAX) {
        if (near[0] > row[0]) row[0] = near[0];
        if (near[w - 1] > row[w - 1]) row[w - 1] = near[w - 1];
      } else {
        if (near[0] < row[0]) row[0] = near[0];
        if (near[w - 1] < row[w - 1]) row[w - 1] = near[w - 1];
      }
    }
  }
}

void warlockc_morph_u8(const uint8_t *src, int64_t src_stride, uint8_t *scratch,
                       uint8_t *out, int64_t out_stride, int64_t h, int64_t w,
                       int64_t radius, int32_t op) {
  /* Ping-pong so that the final pass lands in `out`. With an even radius the
   * first pass has to start in `scratch` for that to come out right, which is
   * cheaper than copying afterwards and is why the parity of the radius is
   * consulted here rather than at the end. */
  uint8_t *front = (radius % 2 == 0) ? scratch : out;
  uint8_t *back = (radius % 2 == 0) ? out : scratch;
  int64_t front_stride = (front == out) ? out_stride : w;
  int64_t back_stride = (back == out) ? out_stride : w;

  const uint8_t *read = src;
  int64_t read_stride = src_stride;

  for (int64_t step = 0; step < radius; ++step) {
    /* diagonal on odd steps: the reference's `step % 2 == 1`. */
    spread(read, read_stride, front, front_stride, h, w, (int)(step % 2 == 1),
           op);
    read = front;
    read_stride = front_stride;
    uint8_t *swap = front;
    int64_t swap_stride = front_stride;
    front = back;
    front_stride = back_stride;
    back = swap;
    back_stride = swap_stride;
  }
}

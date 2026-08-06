/* Selection contours: closed boundary loops around a thresholded mask.
 *
 * The reference (warlock.studio.inker.selection.contours + _chain) builds the
 * four exposed-edge planes with numpy and then runs one Python iteration per
 * boundary pixel per direction, collecting unit segments that a dict/set walk
 * stitches into loops. On a 2048-square lasso that is tens of thousands of
 * iterations plus a hash per edge, and it runs on the frame thread every time
 * the selection *changes*.
 *
 * This is a grid-edge tracer instead: it walks the boundary directly on the
 * (w+1) by (h+1) vertex lattice, so every loop comes out already ordered and
 * nothing is hashed. See warlockc_contours in warlockc.h for the contract.
 */

#include "warlockc.h"

/* Directions, indexed by the arrays below: +x, +y, -x, -y. Image coordinates,
 * so +y is down, which is what makes "left" (dy, -dx) come out as the table
 * turn_left below. */
static const int64_t STEP_X[4] = {1, 0, -1, 0};
static const int64_t STEP_Y[4] = {0, 1, 0, -1};

/* Left, straight, right, as offsets to add to the incoming direction. The
 * order *is* the ambiguous-corner rule -- see the walk below. */
static const int TURNS[3] = {3, 0, 1};

typedef struct {
  const uint8_t *mask;
  int64_t stride;
  int64_t h;
  int64_t w;
  uint8_t threshold;
  uint8_t *scratch;
  int64_t v_base; /* where the vertical edges start in scratch */
} tracer;

/* Cells outside the image are outside the selection, which is what the
 * reference's np.pad(inside, 1) buys and the only reason it pads at all. */
static int cell_inside(const tracer *t, int64_t x, int64_t y) {
  if (x < 0 || y < 0 || x >= t->w || y >= t->h) {
    return 0;
  }
  return t->mask[y * t->stride + x] >= t->threshold;
}

/* The boundary edge leaving vertex (x, y) in direction `dir`, or 0 if there is
 * none.
 *
 * Every boundary edge has exactly one direction it may be walked in: the one
 * that keeps inside pixels on the left. So this is both "is this edge part of
 * the boundary" and "does it lead away from here", in one test -- which is
 * what makes the walk below have no notion of winding to get wrong.
 *
 * Horizontal edge H(x, y) runs from vertex (x, y) to (x + 1, y) and separates
 * cell (x, y - 1) above from cell (x, y) below; vertical edge V(x, y) runs to
 * (x, y + 1) and separates cell (x - 1, y) from cell (x, y). */
static int edge_out(const tracer *t, int64_t x, int64_t y, int dir,
                    int64_t *index) {
  switch (dir) {
  case 0: /* right: inside is above */
    if (x >= t->w || !cell_inside(t, x, y - 1) || cell_inside(t, x, y)) {
      return 0;
    }
    *index = y * t->w + x;
    return 1;
  case 2: /* left: inside is below */
    if (x <= 0 || !cell_inside(t, x - 1, y) || cell_inside(t, x - 1, y - 1)) {
      return 0;
    }
    *index = y * t->w + (x - 1);
    return 1;
  case 1: /* down: inside is to the right */
    if (y >= t->h || !cell_inside(t, x, y) || cell_inside(t, x - 1, y)) {
      return 0;
    }
    *index = t->v_base + y * (t->w + 1) + x;
    return 1;
  default: /* up: inside is to the left */
    if (y <= 0 || !cell_inside(t, x - 1, y - 1) || cell_inside(t, x, y - 1)) {
      return 0;
    }
    *index = t->v_base + (y - 1) * (t->w + 1) + x;
    return 1;
  }
}

int64_t warlockc_contours(const uint8_t *mask, int64_t stride, int64_t h,
                          int64_t w, uint8_t threshold, uint8_t *scratch,
                          int32_t *points_out, int64_t cap_pts,
                          int32_t *loop_lens_out, int64_t cap_loops) {
  tracer t;
  int64_t npts = 0;
  int64_t nloops = 0;
  int64_t y;
  int64_t x;

  t.mask = mask;
  t.stride = stride;
  t.h = h;
  t.w = w;
  t.threshold = threshold;
  t.scratch = scratch;
  t.v_base = w * (h + 1);

  /* Every closed loop on the lattice contains at least one horizontal edge, so
   * scanning those finds every loop and the vertical ones need no scan.
   *
   * The two rows either side of the scan line are hoisted and the "is this a
   * boundary edge" test comes before the visited one, which between them are
   * most of the scan: the general cell_inside is four bounds checks and a
   * multiply, and consulting `scratch` first would touch every page of it on a
   * mask whose boundary is a thousandth of its area. */
  for (y = 0; y <= h; ++y) {
    const uint8_t *above_row = (y > 0) ? mask + (y - 1) * stride : 0;
    const uint8_t *below_row = (y < h) ? mask + y * stride : 0;

    for (x = 0; x < w; ++x) {
      int above = above_row && above_row[x] >= threshold;
      int below = below_row && below_row[x] >= threshold;
      int64_t edge;
      int dir;
      int64_t vx;
      int64_t vy;
      int64_t sx;
      int64_t sy;
      int64_t base;

      if (above == below) {
        continue;
      }
      edge = y * w + x;
      if (scratch[edge]) {
        continue;
      }
      dir = above ? 0 : 2;
      vx = above ? x : x + 1;
      vy = y;
      sx = vx;
      sy = vy;
      base = npts;
      scratch[edge] = 1;

      for (;;) {
        int k;
        int found = -1;
        int64_t next_edge = 0;

        if (npts >= cap_pts) {
          return -1;
        }
        points_out[npts * 2] = (int32_t)vx;
        points_out[npts * 2 + 1] = (int32_t)vy;
        ++npts;

        vx += STEP_X[dir];
        vy += STEP_Y[dir];
        if (vx == sx && vy == sy) {
          break;
        }

        /* Left, then straight, then right.
         *
         * At an ordinary boundary vertex exactly one of the three is a
         * boundary edge leading away, so the order decides nothing. It decides
         * everything at the ambiguous corner -- a checkerboard vertex where two
         * regions touch diagonally and four boundary edges meet -- and left
         * first is the standard rule: it keeps the walk on the region it was
         * already hugging, so the two regions come out as two loops.
         *
         * The reference resolves that corner by whatever order a dict happened
         * to hand back, which can merge the two into one loop instead. Both are
         * the same set of unit edges and the ants cannot tell; the parity test
         * asserts the edges rather than the partition for exactly this reason.
         */
        for (k = 0; k < 3; ++k) {
          int candidate = (dir + TURNS[k]) % 4;
          int64_t index = 0;
          if (edge_out(&t, vx, vy, candidate, &index) && !scratch[index]) {
            found = candidate;
            next_edge = index;
            break;
          }
        }
        if (found < 0) {
          /* Unreachable on a well-formed boundary: every lattice vertex has an
           * even number of boundary edges, so a walk can only run out of them
           * where it started. Bailing rather than looping is the cheap half of
           * being sure. */
          break;
        }
        scratch[next_edge] = 1;
        dir = found;
      }

      if (npts - base > 2) {
        if (nloops >= cap_loops) {
          return -1;
        }
        loop_lens_out[nloops] = (int32_t)(npts - base);
        ++nloops;
      } else {
        /* The reference drops loops of two points or fewer; nothing on this
         * lattice can produce one, and matching it costs a comparison. */
        npts = base;
      }
    }
  }
  return nloops;
}

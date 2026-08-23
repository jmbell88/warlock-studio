/* Median-split BVH build -- viewer/picking.build_bvh's inner loop.
 *
 * The reference is an explicit stack in Python doing roughly six small numpy
 * calls per node, and BVH_LEAF = 8 means a 200k-triangle mesh is tens of
 * thousands of iterations over spans small enough that the work is pure
 * dispatch. Measured: 1045 ms at 200k triangles, and it reruns on every mesh
 * edit because the cache is weak-keyed on the immutable Mesh -- so it lands
 * squarely on the interactive path.
 *
 * **This kernel is not held to output identity, and the licence is written
 * down.** The tree cannot be reproduced bit-for-bit: numpy's argpartition is
 * introselect, its permutation among equal keys is unspecified, and a C median
 * split therefore produces a different -- equally valid -- tree. That is the
 * contours.c situation exactly, so the bar moves to what is actually
 * guaranteed, which here is the *pick result*: the triangle a ray returns,
 * whose tie-break (lowest triangle index) is already pinned so that the tree
 * and the full linear sweep agree. See docs/INVARIANTS.md and
 * tests/viewer/test_bvh_native.py.
 *
 * The arithmetic that is *not* licensed to differ is the boxes: min and max
 * over doubles are exact, so a node's bounds are bit-identical whatever order
 * the triangles inside it ended up in.
 */

#include "warlockc.h"

/* Partition order[begin, end) around the k-th smallest key, in place.
 *
 * Hoare partitioning with a median-of-three pivot, which is what keeps this
 * O(n) on the sorted and reverse-sorted meshes an exporter actually produces
 * -- a first-element pivot degrades to O(n^2) on exactly those. The
 * post-condition is argpartition's: everything below k is <= key[k] and
 * everything above is >=, with the order among equal keys unspecified. */
static void select_nth(int64_t *order, int64_t begin, int64_t end, int64_t k,
                       const double *centroid, int64_t axis) {
  while (end - begin > 1) {
    int64_t lo = begin, hi = end - 1;
    int64_t mid = begin + (end - begin) / 2;
    double a = centroid[order[begin] * 3 + axis];
    double b = centroid[order[mid] * 3 + axis];
    double c = centroid[order[end - 1] * 3 + axis];
    double pivot = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
    while (lo <= hi) {
      while (centroid[order[lo] * 3 + axis] < pivot) {
        lo++;
      }
      while (centroid[order[hi] * 3 + axis] > pivot) {
        hi--;
      }
      if (lo <= hi) {
        int64_t swap = order[lo];
        order[lo] = order[hi];
        order[hi] = swap;
        lo++;
        hi--;
      }
    }
    /* Recurse into the half holding k, iterating on the other -- the tail call
     * is written as a loop so a degenerate mesh cannot blow the C stack any
     * more than it can blow Python's, which is why the reference uses an
     * explicit stack too. */
    if (k <= hi) {
      end = hi + 1;
    } else if (k >= lo) {
      begin = lo;
    } else {
      return; /* k sits in the equal-to-pivot gap: already in place */
    }
  }
}

int64_t warlockc_bvh_build(const double *tri_lo, const double *tri_hi,
                           const double *centroid, int64_t n_tris,
                           int64_t leaf_size, int64_t *order, double *lo,
                           double *hi, int64_t *left, int64_t *right,
                           int64_t *first, int64_t *count, int64_t max_nodes,
                           int64_t *stack) {
  int64_t n_nodes = 0;
  int64_t top = 0;

  for (int64_t i = 0; i < n_tris; i++) {
    order[i] = i;
  }

  /* Each stack frame is (begin, end, parent, which) -- `which` being 0 for a
   * left child and 1 for a right, so a node can fill in its own slot in its
   * parent the moment it is created. The reference instead appends blindly and
   * reconstructs the links afterwards in `_link`; doing it here removes that
   * whole second pass. */
  stack[0] = 0;
  stack[1] = n_tris;
  stack[2] = -1;
  stack[3] = 0;
  top = 1;

  while (top > 0) {
    top--;
    int64_t begin = stack[top * 4 + 0];
    int64_t end = stack[top * 4 + 1];
    int64_t parent = stack[top * 4 + 2];
    int64_t which = stack[top * 4 + 3];

    if (n_nodes >= max_nodes) {
      return -1; /* the caller sized the arrays; fall back rather than scribble */
    }
    int64_t node = n_nodes++;
    if (parent >= 0) {
      if (which == 0) {
        left[parent] = node;
      } else {
        right[parent] = node;
      }
    }

    double bx[3], bX[3], cmin[3], cmax[3];
    for (int64_t k = 0; k < 3; k++) {
      bx[k] = tri_lo[order[begin] * 3 + k];
      bX[k] = tri_hi[order[begin] * 3 + k];
      cmin[k] = centroid[order[begin] * 3 + k];
      cmax[k] = cmin[k];
    }
    for (int64_t i = begin + 1; i < end; i++) {
      int64_t t = order[i];
      for (int64_t k = 0; k < 3; k++) {
        double v = tri_lo[t * 3 + k];
        if (v < bx[k]) {
          bx[k] = v;
        }
        v = tri_hi[t * 3 + k];
        if (v > bX[k]) {
          bX[k] = v;
        }
        v = centroid[t * 3 + k];
        if (v < cmin[k]) {
          cmin[k] = v;
        }
        if (v > cmax[k]) {
          cmax[k] = v;
        }
      }
    }
    for (int64_t k = 0; k < 3; k++) {
      lo[node * 3 + k] = bx[k];
      hi[node * 3 + k] = bX[k];
    }
    left[node] = -1;
    right[node] = -1;
    first[node] = begin;
    count[node] = end - begin;

    if (end - begin <= leaf_size) {
      continue;
    }
    int64_t axis = 0;
    double best = cmax[0] - cmin[0];
    for (int64_t k = 1; k < 3; k++) {
      double extent = cmax[k] - cmin[k];
      if (extent > best) {
        best = extent;
        axis = k;
      }
    }
    if (!(best > 0.0)) {
      continue; /* every centroid coincident: splitting would not separate them */
    }
    int64_t middle = begin + (end - begin) / 2;
    select_nth(order, begin, end, middle, centroid, axis);
    count[node] = 0; /* an interior node holds no triangles of its own */

    if (top + 2 > max_nodes) {
      return -1;
    }
    stack[top * 4 + 0] = middle;
    stack[top * 4 + 1] = end;
    stack[top * 4 + 2] = node;
    stack[top * 4 + 3] = 1;
    top++;
    stack[top * 4 + 0] = begin;
    stack[top * 4 + 1] = middle;
    stack[top * 4 + 2] = node;
    stack[top * 4 + 3] = 0;
    top++;
  }
  return n_nodes;
}

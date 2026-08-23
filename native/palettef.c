/* Nearest palette entry per pixel in Oklab -- pipelines/pixel.map_palette.
 *
 * A float sibling of warlockc_palette_nearest_i32, which is int32 RGB and
 * wired only into inker. This site searches in Oklab float64 and is chunked at
 * 1 << 16 rows precisely because each chunk builds a (65536, p, 3) difference
 * array -- about 100 MB at a 64-entry palette, allocated, written and thrown
 * away to read one index per row. Measured: 2488 ms for a 1024-square frame
 * against 64 entries.
 *
 * **The sqrt is kept, and that is a correctness requirement rather than
 * carelessness.** It is tempting to compare squared distances, since sqrt is
 * monotonic -- but sqrt also *rounds*, so two distinct squared values can land
 * on the same double. numpy's argmin over the rounded norms then takes the
 * earlier index, while an argmin over the exact squares takes the strictly
 * smaller one, which may sit later. Same ordering, different pick, and the
 * difference is a visibly wrong colour on a ramp with near-equidistant
 * entries.
 *
 * The summation order matches np.linalg.norm's reduction over the last axis --
 * ((dl*dl + da*da) + db*db) -- for the same reason every kernel here pins one:
 * float addition is not associative and the bar is equality, not closeness.
 *
 * Ties go to the lowest palette index, matching np.argmin's first-minimum rule
 * and the int32 sibling.
 *
 * `queries` is n by 3 float64, contiguous; `palette` is p by 3 float64,
 * contiguous; `out` is n int32. p >= 1 and n >= 0 are the caller's guarantees.
 */

#include <math.h>

#include "warlockc.h"

void warlockc_palette_nearest_f64(const double *queries, const double *palette,
                                  int32_t *out, int64_t n, int64_t n_palette) {
  for (int64_t i = 0; i < n; i++) {
    const double l = queries[i * 3 + 0];
    const double a = queries[i * 3 + 1];
    const double b = queries[i * 3 + 2];
    double best = 0.0;
    int32_t pick = 0;
    for (int64_t p = 0; p < n_palette; p++) {
      const double dl = l - palette[p * 3 + 0];
      const double da = a - palette[p * 3 + 1];
      const double db = b - palette[p * 3 + 2];
      const double d = sqrt(dl * dl + da * da + db * db);
      if (p == 0 || d < best) {
        best = d;
        pick = (int32_t)p;
      }
    }
    out[i] = pick;
  }
}

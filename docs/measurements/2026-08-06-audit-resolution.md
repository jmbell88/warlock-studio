# Audit resolution 512 → 1024, 2026-08-06

`meshaudit.REQUEST_PATH_RESOLUTION` is what the worker measures every finished
mesh at, and every `params["mesh_audit"]` row in the corpus was produced at it.
Raising it changes what a stored number means, so it gets the same treatment
`Config.trellis_band` and `Config.mesh_hole_max` got: measured first, changed
second.

## Why it was 512

Cost, and nothing else. The module's own comment said so: the reachability half
was two iterative fixpoints — grow a boolean plane by four shifted ORs until it
converges, then propagate integer labels the same way — each O(image diameter)
full-array passes, so the total scaled with roughly resolution³. Full resolution
meant tens of seconds per mesh, and 512 was described as buying "seconds rather
than tens of seconds on a job that already took minutes".

That term is gone. Reachability is now one `cv2.connectedComponents` pass, and
the silhouette rasteriser has a C kernel:

| stage | 512 | 1024 |
|---|---|---|
| reachability, old fixpoints | 0.930 s | 7.351 s |
| reachability, connected components | 0.028 s | 0.010 s |
| rasterise one view, numpy | 0.075 s | 1.025 s |
| rasterise one view, warlockc | 0.008 s | 0.023 s |

Measured on a 19,884-face holed icosphere. A whole four-view audit at 1024 is
now **0.129 s** on that mesh and **0.78 s** on a real 290k-face trellis
reconstruction (`assets/be091c5c5557/model.glb`), against a job that took
minutes to produce. The reason to halve the resolution no longer exists.

## The question that actually gates the change

Not "is it fast enough" but "does the number move". Two thresholds are compared
against `mesh_audit["worst"]`:

- `Config.mesh_hole_max` = **0.07**, chosen as the midpoint of a band the
  2026-08-04 baseline run found completely empty from **0.0308 to 0.1010**.
- `meshreport.HOLE_WARN` = **0.02**, the cosmetic badge.

Both were calibrated on runs measured at 512. If 1024 shifted hole fractions by
even a hundredth, meshes would reclassify because the resolution changed rather
than because their geometry did — and old and new corpus rows would stop being
comparable.

## What was measured

Each mesh audited at both resolutions, all four default views, `worst` compared.
The synthetic ladder removes progressively more of an icosphere (faces whose
normal's |z| exceeds the cutoff) to land values across the whole usable range
rather than only near zero.

| case | worst @512 | worst @1024 | delta | side of 0.07 |
|---|---|---|---|---|
| real trellis mesh | 0.02472 | 0.02517 | +0.00045 | under → under |
| sphere intact | 0.00000 | 0.00000 | +0.00000 | under → under |
| sphere cut .97 | 0.05779 | 0.05755 | −0.00024 | under → under |
| sphere cut .9 | 0.19062 | 0.19050 | −0.00011 | over → over |
| sphere cut .8 | 0.35699 | 0.35712 | +0.00013 | over → over |
| sphere cut .6 | 0.64072 | 0.64067 | −0.00005 | over → over |
| sphere cut .4 | 0.83979 | 0.83979 | −0.00000 | over → over |

**Largest disagreement: 0.00045.** Verdicts flipped: **zero**.

## Conclusion

The drift is two orders of magnitude below the 0.07-wide empty band the
threshold sits in, and below the ~0.3% noise floor the band sweep already
established for this measurement (auto and band 2 are the same setting and
still disagreed by 728 faces). Rows measured at 512 and at 1024 are comparable
for every decision the codebase takes against them, so no threshold moves and
no stored row needs re-measuring.

The direction of the change is also the conservative one: 1024 resolves
sub-pixel gaps 512 could not see, so if anything it finds *more* of what the
measurement exists to find. `test_the_two_resolutions_agree_closely_enough_to_share_a_threshold`
pins the agreement, and `test_the_request_path_measures_at_full_resolution`
pins the constant — a corpus decision should take a test edit, not a one-line
tweak.

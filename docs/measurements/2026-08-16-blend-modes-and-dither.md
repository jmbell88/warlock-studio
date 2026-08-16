# Native kernel batch 3, part one — the blend-mode gap and Floyd–Steinberg

2026-08-16. Machine: Windows 11, MSVC (`native/build.ps1` picked `cl`), numpy
2.5.1, Python 3.13.13. Every figure is the minimum of 3–9 runs in a fresh
process; the spread on multi-megabyte float work is wide (±10 % here, and the
2026-08-09 batch saw ±30 %), so only minima are quoted and every comparison is
between two minima taken the same way. Where a difference was small enough that
run-to-run noise could explain it, the two builds were measured **interleaved in
the same session**, three rounds, and both are reported.

Two candidates from the C-rewrite review (K1 and K2). **Both cleared their gates
and both shipped.** ABI went 7 → 8, carried by one new symbol
(`warlockc_dither_fs`) and by a change to what `warlockc_over_f32` and
`warlockc_stack_f32` compute for seven mode numbers they previously never saw.

## Summary

| # | Candidate | Gate | Measured before | After | Verdict |
|---|---|---|---|---|---|
| K1 | the seven non-kernel blend modes | any stack-wide cliff | **5.4 → 49 ms** per dab invalidate | **6.3–7.1 ms** | **shipped**, §1 |
| K2 | `dither._floyd_steinberg` | >1 s per conversion | **10.6 s** @1024², ~43 s @2048² | **139 ms** @1024² | **shipped, 76×**, §2 |

One defect was found and fixed on the way, and it is the entry worth reading
first if you only read one: §3.

---

## §1 — One layer in the wrong mode cost the whole document

### What was measured

`composite._MODE_IDS` carried twelve of the nineteen `BLEND_MODES`. The seven
appended after `difference` — `exclusion`, `subtract`, `divide`, and the four
non-separable ones `hue`/`saturation`/`color`/`luminosity` — were numpy-only.

That was deliberate and written down, and the reasoning had one true premise and
one wrong conclusion. True: the four non-separable modes read all three channels
of a pixel to decide one of them, so they cannot be a per-*channel* C case.
Wrong: that they therefore could not be a C case at all. They are per-*pixel*
independent — a three-element sort and a luminance clip, no reductions across
pixels — so the kernel was always free to take the whole pixel and hand back
three floats.

What made it worth fixing is not the mode, it is `_stack_native` being
all-or-nothing: **one** layer in one of the seven put the **entire stack** on the
numpy fold. Six layers at 2048², minimum of 7 runs:

| stack | 256² dab invalidate | full canvas |
|---|---|---|
| all-native modes | 5.40 ms | 256 ms |
| + one `exclusion` | 24.4 ms (4.5×) | 1306 ms |
| + one `subtract` | 23.0 ms (4.3×) | 1244 ms |
| + one `divide` | 24.6 ms (4.5×) | 1312 ms |
| + one `hue` | 47.4 ms (8.8×) | 2745 ms |
| + one `saturation` | 48.7 ms (9.0×) | 2810 ms |
| + one `color` | 39.2 ms (7.3×) | 2051 ms |
| + one `luminosity` | 40.9 ms (7.6×) | 2098 ms |

A user fell off that cliff by picking an item from a menu. At 48.7 ms a dab, the
frame loop is at ~20 fps while painting, and a below-cache rebuild took 2.8
seconds.

### After

Same benchmark, ABI 8:

| stack | 256² dab | speedup | full canvas | speedup |
|---|---|---|---|---|
| all-native modes | 4.15 ms | — | 269 ms | — |
| + one `exclusion` | 4.51 ms | **5.4×** | 283 ms | **4.6×** |
| + one `subtract` | 3.95 ms | **5.8×** | 272 ms | **4.6×** |
| + one `divide` | 4.40 ms | **5.6×** | 274 ms | **4.8×** |
| + one `hue` | 7.07 ms | **6.7×** | 444 ms | **6.2×** |
| + one `saturation` | 6.33 ms | **7.7×** | 425 ms | **6.6×** |
| + one `color` | 6.59 ms | **5.9×** | 422 ms | **4.9×** |
| + one `luminosity` | 6.49 ms | **6.3×** | 414 ms | **5.1×** |

The cliff is gone. What remains is an honest ~1.6× for the non-separable four
over the separable ones, which is the sort and the luminance clip actually
costing something — as they should.

### The one implementation decision that needed a number

Routing the modes meant the kernels could no longer compute `mixed` a channel at
a time. The obvious shape is a `blend_rgb(mode, cb, cs, out[3])` helper called
once per pixel; it is one branch and one copy of the compositing expression.
Measured against hoisting the separable/non-separable test out to the call sites
(so the separable path never materialises a three-float buffer), interleaved,
three rounds, 2048²:

| build | all-separable | with a `color` layer |
|---|---|---|
| `blend_rgb` helper | 265.2 / 263.0 / 265.4 ms | 421.4 / 420.1 / 420.7 ms |
| hoisted branch | 256.6 / 257.1 / 257.4 ms | 399.9 / 392.8 / 398.5 ms |

~3 % on the separable path and ~5 % with a non-separable layer — small, but it
reproduced in every round and the separable path is the common one. The hoisted
form was taken, but **not** at the price the naive version of it charges: doing
it by duplicating the branch inline means four copies of the compositing
arithmetic in a file whose contract is bit-parity, which is four things to keep
equal. Instead the *cheap* half was duplicated and the arithmetic factored into
a single `combine_channel`. That build measured 254.7 / 255.9 / 264.2 ms
separable and 414 / 391 / 405 ms with `color` — indistinguishable from the
duplicated form, with one copy of the formula.

### Parity

`np.array_equal` against the untouched numpy bodies, and the existing sweeps in
`tests/inker/test_composite_native.py` are parametrised over `cp.BLEND_MODES`,
so the seven joined every one of them for free — including the NaN sweep. What
random floats do *not* produce got its own tests: two exactly equal channels
(the stable-sort tie `_set_sat` leans on), a fully grey pixel (`span == 0`),
and inputs that drive `_clip_colour` off both ends.

Three numpy semantics the transcription depends on were checked empirically
against this build's numpy before any C was written, over two million random
triples each:

* `.sum(axis=-1)` over a three-element axis accumulates left to right from a
  zero seed (numpy's pairwise sum takes its `n < 8` branch). `_lum` is the only
  reduction in the non-separable four.
* `np.argsort` over a three-element axis is stable — introsort falls to
  insertion sort well below sixteen elements — which is what makes the
  two-equal-channel case well-defined at all.
* A float32 array against a Python float stays float32 under NEP 50.

---

## §2 — Floyd–Steinberg: 76×

`dither._floyd_steinberg` is a per-pixel Python loop with ~10 numpy calls per
iteration. Its docstring defends the loop, correctly: error diffusion is
sequential by definition, so there is no vectorisation of it that is still
Floyd–Steinberg. That argument rules out numpy. It is also precisely the
argument *for* C.

Before, 32-entry palette, fully opaque:

| size | pixels | time | per pixel |
|---|---|---|---|
| 64² | 4 096 | 54.1 ms | 13.2 µs |
| 128² | 16 384 | 228 ms | 13.9 µs |
| 256² | 65 536 | 703 ms | 10.7 µs |
| 512² | 262 144 | 2 656 ms | 10.1 µs |
| 1024² | 1 048 576 | 10 567 ms | 10.1 µs |
| 2048² | 4 194 304 | ~43 s (extrapolated) | — |

After:

| size | before | after | speedup |
|---|---|---|---|
| 256² | 703 ms | 10.9 ms | **64×** |
| 512² | 2 656 ms | 42.9 ms | **62×** |
| 1024² | 10 567 ms | 139 ms | **76×** |
| 2048² | ~43 s | ~0.5 s | — |

**The palette-size curve is the useful diagnostic here**, and it flipped exactly
as predicted. Before, at 256², the cost barely moved with the size of the table
the inner `argmin` scans — 620 ms at 4 entries against 876 ms at 256, a 1.4×
spread over a 64× larger search. That is the signature of a loop paying for
dispatch rather than for arithmetic, and it is why the C win was expected to be
close to total rather than proportional. After: 4.5 ms at 4 entries against
22.0 ms at 256, a 4.9× spread — the arithmetic showing through, now that it is
what is left.

This is the first kernel in `native/` whose reference is a Python loop rather
than a numpy expression, and it is the easiest parity story in the directory:
being scalar and sequential, there is no summation order to choose and nothing
to vectorise, so the transcription is operand for operand. The one thing that
*did* need care is the diffusion order — four neighbours accumulated into in a
fixed sequence, several of them again from the next pixel along, and float
addition is not associative.

Parity is byte identity in `tests/inker/test_dither_native.py`, and it is a
stronger claim than usual: error diffusion is a chain, so a single differing ulp
anywhere propagates into a different picture rather than one wrong pixel. The
sweep covers single-row, single-column and single-pixel images (each removes one
of the four diffusion targets), four alpha patterns (invisible pixels are
neither quantised nor used as sinks, so holes change which neighbours an error
reaches), palette sizes 1–255, a deliberately tied `argmin`, and a 128² image —
capped there only because the *reference* runs at ~10 µs/px and the test has to
run it.

### What was not touched

`_ordered` — the vectorised sibling — measured **2 007 ms at 1024²** with 32
entries, which is 14× *slower* than the kernel that replaced the Python loop it
was supposed to be the fast alternative to. It is an O(P·N) full-canvas compare
per palette entry and belongs to the K3 palette-indexing family in the review,
not here. Recorded so the number exists before anyone reaches for it.

---

## §3 — The defect found on the way: `divide` was computing in float64

Worth reading even if you skip the rest.

`blend(..., "divide")` was the only one of the nineteen modes that did not
return float32. The cause:

```python
np.where(source > 0.0, ratio, np.where(backdrop > 0.0, 1.0, 0.0))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

The inner `where` is a bool array between two Python scalars. It has no array
operand to take a dtype from, so numpy answers **float64** — and the outer
`where` then promotes the whole result. Every other mode has an array in both
arms and stays float32.

The consequence was not the branch. `over` multiplies `mixed` into its `num`
chain, so a divide layer ran the *entire* composite at double width: three
full-size float64 temporaries in a module whose first paragraph says everything
speaks float32. It is visible in the "before" table above — `divide` is the
slowest of the three arithmetic modes despite being the simplest.

It also blocked the kernel: bit-parity with a float64 reference would have meant
carrying a double path in C for one mode out of nineteen.

The fix spells the zero-divisor answer as a cast of the mask —
`(backdrop > 0.0).astype(ratio.dtype)` — which is the same two numbers in the
arithmetic's own dtype, and follows a float64 caller up as well as a float32
caller down.

**This changes output**, and the house rule is that a kernel needing a changed
reference must beat the original reference, so the change is quantified rather
than asserted. Over 7.86 M channels of random compositing:

| | |
|---|---|
| float32 values that moved | 2 656 877 (33.8 %) |
| worst absolute delta | 1.19e-07 — one float32 ulp, 0.00003 of a 0–255 level |
| **uint8 bytes that moved after `to_uint8`** | **27 (0.0003 %)** |

So: a third of the intermediates shift by an ulp, and 27 bytes in 7.9 million
survive the narrowing — pixels that were sitting exactly on a rounding boundary.
The numpy fallback's own speed is unchanged within noise (626 → 618 ms at
2048², the float64 promotion being small against numpy's other temporaries), so
the change stands on correctness and on unblocking a kernel that beats the
original reference by 4.8×.

---

## Verification

* `pwsh native\build.ps1` clean under `/W4 /WX`; ABI 7 → 8 in
  `native/warlockc.h` and `src/warlock/native.py` together.
* Parity is `np.array_equal` against the untouched numpy/Python references, and
  both references remain in place as the fallback.
* Full suite run both ways — with the DLL and with `WARLOCK_NATIVE=0`.

# Per-cel z-index: what turning the below-cache off actually costs

2026-08-30. Machine: Windows 11 Pro 26200, Python 3.13.13, numpy from the
project's own `uv` environment, no native `warlockc.dll` present (so every
composite below is the numpy path). Every figure is from a **fresh process per
configuration**, driven by a throwaway script that calls
`Document.write_colour` directly — the same door the brush, the bucket and the
three shapes come through, so what is timed is the real invalidation path and
not a synthetic one.

Per row: sixty dabs, the **first reported separately** (it is the one that
builds the cache), the remaining fifty-five sorted and reported as min and
median, plus the wall-clock **total** of all sixty. Three repeats of the
headline configurations; the spread between repeats is under 4 % on the totals
and under 2 % on the steady-state minima, so single figures are quoted where
the three agreed and ranges where they did not.

## Why this document exists

`Document._below` is a full-canvas composite of the layers under the active
one. It is built lazily on the first dab of a stroke and then *patched* per
rectangle, so an ordinary dab re-blends only the active layer and what is above
it (`LayerStack.composite_region(box, below=...)`).

Per-cel z-index (`Animation.cel_z`, divergence #12, retired today) breaks the
premise: a cel can be lifted from below the active layer to above it, so the
rows under `active_index` are no longer finished business and the cached base
would be a picture with a hole in it. **Wave 14 refuses rather than repairs**:
any nonzero `cel_z` on the frame being drawn on sets `LayerStack.cel_z`, and
`Document.invalidate` then composites the whole stack for every dab.

That is a hot-path invariant, so a passing test is not sufficient evidence.
These are the numbers.

## The headline

2048² canvas, 32 px dab (the size a brush actually invalidates), active layer
in the middle of the stack, one slot lifted by `+1` in the `z` rows.

| Tracks | Cache | First dab (ms) | Steady min (ms) | Steady median (ms) | 60-dab total (ms) |
|---|---|---|---|---|---|
| 4 | on (no `cel_z`) | 59.6 – 60.7 | 0.079 – 0.081 | 0.083 – 0.085 | 65.7 – 66.2 |
| 4 | **off** (`cel_z` set) | **0.37 – 0.43** | 0.099 – 0.102 | 0.102 – 0.107 | **7.1 – 7.9** |
| 10 | on (no `cel_z`) | 205.3 – 212.4 | 0.164 – 0.166 | 0.169 – 0.172 | 216.1 – 223.1 |
| 10 | **off** (`cel_z` set) | **0.59 – 0.67** | 0.248 – 0.252 | 0.251 – 0.256 | **16.2 – 16.6** |

**The disabled cache is 1.3× to 1.5× more expensive per dab, and 13× cheaper
per stroke.** Both halves of that are real and neither is the whole story:

- **Per dab** the cache is doing its job. At ten tracks a dab costs 0.166 ms
  with it and 0.250 ms without — an extra 0.084 ms, or half of one percent of a
  16.7 ms frame. It is a cost, and it is not one a user can perceive.
- **Per stroke** the cache has to be *built*, and building it is a full-canvas
  composite of every layer below the active one: 205 ms at 2048² by ten tracks.
  A stroke that ends before that has been amortised is cheaper without it.

The crossover was measured rather than extrapolated, by running the same
configuration out to thousands of dabs:

| Dabs | Cache on, total (ms) | Cache off, total (ms) |
|---|---|---|
| 60 | 216 | 16 |
| 2 000 | 548.7 | **516.6** |
| 2 500 | **643.0** | 652.9 |
| 3 000 | **733.7** | 790.6 |

So at 2048² by ten tracks the cached path only starts winning at about **2 400
dabs in one stroke** — far past any single gesture, and the moment the user
lifts the pen and puts it down somewhere else the 205 ms build is paid again.

## Where the disabled path does cost real time

The steady-state gap scales with the *dab rectangle*, because that is what the
extra work is: re-blending the lower rows over the invalidated box.

| Canvas | Tracks | Dab | Cache on, steady min (ms) | Cache off, steady min (ms) | Ratio |
|---|---|---|---|---|---|
| 512² | 4 | 32 px | 0.079 | 0.098 | 1.24× |
| 512² | 10 | 32 px | 0.116 | 0.167 | 1.44× |
| 512² | 10 | 256 px | 3.52 | 5.71 | 1.62× |
| 2048² | 4 | 32 px | 0.080 | 0.100 | 1.25× |
| 2048² | 4 | 256 px | 2.15 | 2.95 | 1.37× |
| 2048² | 4 | 1024 px | 33.0 | 50.7 | 1.53× |
| 2048² | 10 | 32 px | 0.165 | 0.247 | 1.50× |
| 2048² | 10 | 256 px | 3.71 | 7.61 | 2.05× |
| 2048² | 10 | 1024 px | 63.7 | 114.9 | 1.80× |

The worst case in the table — a 1024 px dab on a 2048² ten-track document —
goes from 64 ms to 115 ms. **Both of those are already past the frame budget**,
with or without this wave: a half-canvas brush on a ten-layer 4-megapixel
document does not run at sixty frames a second either way, and nothing about
the number 115 makes a case the number 64 did not already make.

It is also **not the case being optimised**. The below-cache exists for the
dab-sized rectangle, and at that size the cost is 0.08 ms.

## The negative control, measured the same way

The tree was stashed back to `cc7ee724` and the *cached* configurations re-run
against it, so the "cache on" column above is a before-and-after and not an
after-only:

| Configuration | Before the wave | After the wave |
|---|---|---|
| 2048², 4 tracks, first dab | 59.2 – 60.7 ms | 59.6 – 60.6 ms |
| 2048², 4 tracks, steady min | 0.080 – 0.081 ms | 0.079 – 0.081 ms |
| 2048², 10 tracks, first dab | 203.6 – 206.2 ms | 205.3 – 212.4 ms |
| 2048², 10 tracks, steady min | 0.164 – 0.165 ms | 0.164 – 0.166 ms |

**Unchanged within the run-to-run spread.** What was added to the ordinary path
is one attribute read and one `is None` comparison per `_entries` call
(`LayerStack._order` returns `None` immediately when `cel_z` is), which is
below this method's resolution and is quoted as such rather than as a figure.

Bit-identity of the ordinary path is pinned separately and not by timing:
`tests/inker/test_cel_z.py` carries sha256 literals for the composite, the
flatten, an off-frame flatten, the `.ora` bytes and the `.aseprite` bytes of a
document with no `cel_z`, all captured by a throwaway script against `cc7ee724`
before this wave was written.

## What was *not* measured, and is not claimed

- **The native stack kernel.** `warlockc.dll` was not built on this machine, so
  every number here is the numpy fallback. The change cannot alter what the
  kernel receives for a document with no `cel_z` — `_entries` returns the
  identical list, which is asserted rather than timed — but no kernel timing
  was taken and none is quoted.
- **The GPU / studio frame loop.** These are engine-level timings through
  `Document`, with no imgui and no GL. What a frame costs in the app was not
  measured for this wave.
- **A document with many lifted cels.** Every "cache off" row above sets
  exactly one nonzero `cel_z`, deliberately: what is being measured is the
  cache being off, not the sort. The sort itself is one `sorted()` over a list
  of at most a few dozen ints per composite call and was not separately timed.

## The verdict

Refusing the cache is the right trade and the numbers say why: it is invisible
at dab size, it makes a *stroke* an order of magnitude cheaper by not paying
for a full-canvas base, and the configurations where it costs real milliseconds
were already over budget. The alternative — working out per dab which rows
crossed the active layer and re-splitting the stack — is a second ordering rule
to keep in step with the first, for a feature whose entire point is that the
split does not hold.

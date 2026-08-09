# The mesh grade scale, its backfill and its usable cut, 2026-08-09

**Status: scale declared, no graded labelling session taken.** A pre-registration
in the style of [`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md) and
[`2026-08-09-judge-threshold.md`](2026-08-09-judge-threshold.md), and it exists
under the same repo rule those two do: a constant the stored corpus is keyed on
gets a measurement document *before* it changes. Three constants are being
introduced at once here — the scale's endpoints, the value a binary row is
backfilled to, and the grade at which a mesh counts as usable — and every one of
them is a number a motivated reader would tune after seeing the first histogram.
Written now, this document is a commitment; written after the grades exist, it is
a rationalisation with a table in it.

## The question

The mesh-verdict corpus is binary accept/reject, and it has failed at its own
purpose. The 2026-08-07 review produced **3 accepts against 81 rejects**, which
`../LEFTOVERS.md` §8 calls "not a thin corpus, an unusable one" — and the corpus
has since grown to **151 latest-wins mesh verdicts, 26 accept / 125 reject**, a
better ratio and exactly the same shape of problem: a probe fitted to
it learns the word *reject* and scores 96% doing so, and `judge.fit` correctly
returns `None` rather than hand anybody that probe.

But the thinness is only half of it, and the other half does not go away by
labelling more meshes. **A binary row can say a mesh failed and can never say how
close it came.** Every one of those 81 rejects is the same row. A slab with no
geometry at all, a good shape whose texture is smeared, and a mesh a modeller
would fix in five minutes are one value, and the difference between them is
exactly the signal any ranking, any tie-break and any future regression target
would want. The corpus was recording the least informative bit available about
each mesh.

So the change is to grade rather than to classify: an integer **−5..+5**, model
(mesh) stage only, with an optional tag vocabulary attached at any grade.

## What the endpoints mean

**+5 — game-useable as-is.** Drop it in the scene; no fixup pass.
**−5 — completely unusable.** Nothing recoverable; the slab case.
**0 — no opinion either way**, and it is a real answer rather than a missing one:
a mesh that is neither usable nor worthless is the commonest honest verdict about
trellis output, and forcing it to a side is how a scale acquires a bimodal
histogram that says more about the reviewer than the meshes.

The interval is deliberately not claimed to be metric. Nothing here asserts that
the distance from +1 to +2 equals the distance from +4 to +5, and no arithmetic in
this change depends on it beyond a mean used **only as a tie-break and as a
displayed figure** (see below). An ordinal reading is the one the scale earns.

## Why the backfill is ±3 and not ±5

Migration 10 backfills every existing model-stage row: `accept` → **+3**,
`reject` → **−3**.

The argument is about what a binary reviewer actually asserted, and it is an
argument for the *mildest* grade that preserves the row's meaning. A reviewer who
pressed `A` asserted "this is usable" and asserted **nothing stronger** — they had
no key for "and it ships as-is", so reading one into their row would be inventing
evidence. ±5 would do exactly that, and it would do it to **every mesh verdict on
record** — 159 rows on this machine at the time of writing, 151 of them
latest-wins (26 accept / 125 reject). It would also poison the first histogram:
every real grade recorded afterwards would sit inside a distribution whose tails
are entirely synthetic.

±3 leaves **±4 and ±5 free for judgements a binary reviewer never made**. That is
the property that matters. When the first graded pass lands, a +5 in the data is a
human's +5 and nothing else.

The choice is coupled to the usable cut and that coupling is the point, not a
coincidence — see the round trip below.

## Why the usable cut is grade ≥ +3

A binary "usable" answer is still needed in four places that this change
deliberately does not disturb: prune retention, the judge service's label reads,
the `latest_verdicts` / `unverdicted_models` SQL, and every findings-v3 reader.
Those read the `verdict` TEXT column, which survives as a **derived** field
written at record time by one writer. `verdict_for_grade(grade)` is that
derivation, and the cut it applies is `grade >= 3`.

**The round trip is the proof, and it is one line.** A backfilled accept is +3.
`verdict_for_grade(+3)` must return `"accept"`, or migration 10 would silently
demote all 26 accepts it touched — a migration that changes the answer to a
question nobody asked it to change. A backfilled reject is −3 and
`verdict_for_grade(-3)` returns `"reject"` for the same reason. So the cut cannot
be +4 or higher without breaking its own backfill, and it cannot be +2 or lower
without asserting that some grade no reviewer ever recorded was already usable.
**The backfill value and the cut are one decision with two names**, and
`BINARY_GRADES["accept"] == USABLE_GRADE` is asserted in the test suite rather
than left as a comment.

The cut is also the only threshold in the change. There is no second one: no
grade band gets its own retention tier, no grade gates a UI affordance, and the
judge is untouched.

## Why the mean is a tie-break and not a ranking

`findings.json`'s `vectors` section ranks whole configurations. It sorts by the
**Wilson lower bound of the usable rate**, and that stays the primary key. The
mean grade becomes the *first tie-breaker* and is displayed beside it, and it is
deliberately not promoted above Wilson.

Two reasons, both of which would be invisible in a demo and fatal in the corpus.
A mean−SE bound **degenerates at n=1**: the standard deviation of one sample is
zero, so the bound equals the mean, and a single lucky +5 outranks a configuration
with nineteen +4s — which is precisely the lucky-5/5 pathology Wilson was adopted
to kill, re-created in a new coordinate system. And over the corpus as it stands
immediately after migration 10, **every grade is ±3**, so the mean is an affine
function of the usable rate and a mean-primary sort would be the Wilson sort with
its confidence correction thrown away.

Revisit when there is real spread: the condition is stated below.

## Tags

Optional at every grade, in one namespace, five per polarity:

- **bad** — `holes`, `bad-shape`, `bad-texture`, `wrong-style`, `broken`
- **good** — `clean-shape`, `good-texture`, `on-style`, `sharp-detail`,
  `good-topology`

**The five bad spellings are frozen**, because they are the existing `REASONS`
tuple and the stored corpus already carries those exact strings in the `reasons`
JSON column. Renaming one would not migrate evidence; it would split it, the way
`guidance._LEGACY_ALIASES` documents for taxonomy keys — and unlike a taxonomy key
there is no alias table here, so a rename is simply a loss. They are the same
column and the same strings; only the name of the concept changed, from *reasons a
reviewer rejected* to *tags describing what is true of this mesh*.

The good set mirrors the bad set where a mirror exists (`clean-shape` ↔
`bad-shape`, `good-texture` ↔ `bad-texture`, `on-style` ↔ `wrong-style`) and
supplies two that have no mirror: `sharp-detail` — detail survived reconstruction,
which is what trellis most often loses and what a bare "not broken" cannot say —
and `good-topology`, the constructive counterpart of `broken`, meaning it imports
and derives cleanly.

There is deliberately **no `game-ready` tag**. That is what grade +5 already
asserts, and a tag saying the same thing as a grade is two spellings of one fact,
which is the `judge.STAGES`/`verdicts.STAGES` hazard: they drift the first time one
of them is edited.

Five per polarity is not aesthetic — it is what makes `Ctrl+1..5` and
`Shift+1..5` map positionally onto the two vocabularies, so the keyboard needs no
second table naming which digit is which tag.

The two vocabularies are **disjoint strings**, which is what lets them share one
storage namespace: polarity is recoverable from the tag itself, so nothing has to
be written down twice and the findings writer splits by membership. That keeps
`bench/findings.py` pure-stdlib with no `warlock` imports, which is its own
standing invariant.

## What is *not* graded, and why

**Image-stage labels (`reference`, `blank`) stay binary and their `grade` stays
NULL, permanently.** Three reasons, and the first is decisive: those two labels
feed binary logistic probes, so a grade would have to be thresholded back to a bit
before `judge.fit` could use it — a scale introduced only to be discarded. Second,
the fast two-key loop is what makes a 100-image pass viable at all, and an
eleven-key decision per image is not a labelling pass, it is a review. Third, the
image question genuinely is binary in a way the mesh question is not: "will this
reconstruct" has no interesting middle, and
[`2026-08-09-judge-threshold.md`](2026-08-09-judge-threshold.md) pre-registered
binary-first for exactly this stage.

Nothing about `judge.py` or `service/judge.py` changes. Its trainable stages are
the image stages, and they are untouched.

## Predictions, recorded before the data

Stated now so they are predictions rather than post-hoc readings:

1. **The graded histogram will be left-heavy but not degenerate** — the corpus
   whose accept rate is ~17% should produce a mass around −3 to −1 with a thin
   right tail, rather than the two spikes at ±3 the backfill alone would show. If
   the first graded pass reproduces two spikes, the reviewer is grading binarily
   with extra steps and the scale has bought nothing.
2. **`sharp-detail` and `holes` will be the two most-used tags**, because they name
   the two things trellis output is most often distinguished by, and the tag
   tallies are the cheapest available check on whether the vocabulary fits the
   failures people actually see.
3. **`0` will be used**, and if it is not — if no reviewer ever records it over a
   full pass — that is evidence the scale should be even rather than odd, and it
   is written here so that finding is admissible.

## Revisit conditions

- **Promote the mean above Wilson** only once some configuration bucket holds
  grades with a standard deviation above zero at n ≥ 10, *and* a document shows
  the resulting ranking differs from the Wilson one on real data. Until both hold,
  the tie-break placement stands.
- **Move `USABLE_GRADE`** only together with a re-derivation of `BINARY_GRADES`,
  because the round trip above is what makes the pair coherent. Moving one alone
  is the failure this section exists to name.
- **Change a `BAD_TAGS` spelling** only with a written migration of the stored
  `reasons` column. There is no alias table; a bare rename splits the corpus
  silently.
- **Grade the image stages** only if a measurement shows the binary label is
  losing signal the probes could use — which cannot be shown until a probe exists,
  and no probe exists.

## Results

Not yet taken. The scale ships empty apart from the ±3 backfill; the first graded
pass is what fills it, and it is that pass — not this document — that will say
whether the predictions above hold.

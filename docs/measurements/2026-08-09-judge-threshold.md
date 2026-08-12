# The judge's threshold, and what "it works" has to mean, 2026-08-09

**Status: procedure written, no labelling session taken.** A pre-registration in
the style of [`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md), and the
strictest of the campaign's four: every number `../LEFTOVERS.md` §10 (deleted;
git history) asks for is one
a motivated reader would tune after seeing the ROC curve. Written before the
labels exist, this document is worth something; written after, it is a
rationalisation with a table in it.

## The question

`judge.py` fits a linear probe on DINOv2 CLS embeddings and `judge.score`
returns a probability. **There is no threshold anywhere in the module, and that
is deliberate** -- `service/judge.py` says so in its own docstring: a
probability-to-accept cut is a constant the stored corpus would then be keyed on,
and this repo's rule is that such a constant gets a measurement first.

`review_mode.SOURCE_AI = "ai:dino-probe"` is a constant nothing writes, and
`tests/test_review_mode.py` asserts no row carries it. The
`(job_id, source, stage)` seam is built and tested. **The day this document has
numbers in it, filing an `ai:` verdict is one call.** Until then it stays
unwritten, and that is not an oversight to be fixed.

## What has to happen first

A human labelling pass. Review -> *Teach the judge*, under the sweep list; `A`
is good, `R` is bad, no reason step; the pass owns the keyboard while open so a
keypress about a picture can never be filed as a verdict about a mesh.

**Both passes, over the same images.** `reference` asks "is this a good 2D
asset"; `blank` asks "will this reconstruct". They are opposed objectives on one
PNG -- a dramatic plate with pillars and a cast shadow is a better asset and a
worse blank -- and neither answer implies the other.

`judge.fit` returns `None` below `MIN_PER_CLASS = 8` of **each** class. After the
first labelling pass (2026-08-09, over the confirm sweep) the corpus stands at
**7 accept / 4 reject** at `reference` and **13 accept / 2 reject** at `blank`,
so neither probe is trainable and **both are short of negatives, not
positives**.

That asymmetry is the thing to read, and it is not a labelling error to be
corrected by hunting for rejects — a corpus assembled by seeking out one class is
biased in exactly the direction the held-out split cannot detect. It is a
consequence of what has been labelled: the images available were mostly ones that
passed the composition gate. The fix is more images across the quality range,
which the 50-unit re-baseline supplies.

The `blank` split is independently interesting at 13/2: most references look like
usable trellis input even when the mesh they produced was rejected. If that
holds, the blank question is the less discriminative of the two and the
`reference` probe is the one carrying the signal — worth stating now, before the
probe exists, so it is a prediction rather than a post-hoc explanation.

It is blocked on pixels as much as on the human: 100 of the 117 verdicts on
record name job directories that no longer exist, and a probe trains on pixels.

**Labels must be human.** The composition gate's own refusals are tempting free
labels and are disqualified: they are the rule's *output*, so a probe fitted to
them can at best reproduce `reference.py` exactly, blind spots included. The
canonical counter-example already exists -- `baseline s23` **passed** those rules
and is still a poor blank, which is precisely the case a learned judge exists to
catch.

**And the probe is fitted to pixels, never to the audit scalars.**
AUC(`hole_worst` -> reject) is 0.115, so a scalar-fitted probe would need a sign
flip to beat a coin and would then be fitting the slab artefact. The same
inversion disqualifies `hole_worst` as a sanity check on the probe.

## Decision rules, written in advance

**The split is by `prompt_hash`, never at random.** A random split leaks the same
subject into both halves and makes the scope risk unmeasurable by construction --
the exact question the held-out set exists to answer. The split is declared in
Results before any probe is fitted.

**False-reject rate is the gate.** Good assets the judge would have discarded.
The cut is adopted only at **FRR <= 5% on held-out**, and that is the number that
decides whether the probe may ever gate anything at all. False-accept rate is
reported and gates nothing: a bad asset waved through reaches Review anyway,
which is the asymmetry the whole advisory-first design rests on.

**Agreement with `reference.py`'s rules is computed per rule, not in aggregate**
-- one figure per `refused_<code>` (`empty`, `occupancy`, `edge`,
`multi_object`), which the observation rows already carry. **Agreement >= 95% is
pre-declared a null result**: the probe has learned to imitate the rules and has
added nothing, whatever its accuracy says. Value shows up as *disagreement a
human sides with the probe on*, and `baseline s23` is named here, before scoring,
as the canonical case.

**Two baselines, both stated in advance, because accuracy alone is
disqualified.** "Always reject" scores 96% on the corpus as it stands. So the
probe is reported against (a) a coin flip at 0.5 AUC -- and *not* against
`hole_worst`, whose 0.115 is a floor of no floor at all -- and (b) the
majority-class predictor. A headline accuracy figure without both is not
reported.

**Per-`prompt_hash` breakdown is mandatory.** No cut is adopted if any subject
with enough held-out labels to compute one shows an FRR more than **2x** the
pooled figure. A probe trained on this campaign learns "good SNES rogue", not
"good asset"; pointed at a wooden crate it is confidently wrong, and a judge with
no notion of subject is worse than no judge because it is trusted.

**Null, stated in advance:** below `MIN_PER_CLASS` in either class on the
held-out half, or a bootstrap AUC confidence interval containing 0.5, means **no
threshold is adopted**, `SOURCE_AI` stays unwritten, and this document says so.

**Three of §10's open questions are answered by this pre-registration rather
than after the fact:**

- Binary accept/reject first. The five `REASONS` are a multi-class problem and
  the first corpus supports nothing of the sort.
- One global probe per image stage, with the per-`prompt_hash` breakdown as the
  measurement of scope. Per-subject probes are a decision for a corpus that has
  several subjects with tens of labels each.
- Max-versus-mean pooling is the **mesh** probe's question and is deferred: there
  is no mesh corpus, and `bench/views.py`'s calibration
  ([`2026-08-04-view-calibration.md`](2026-08-04-view-calibration.md)) already
  establishes that a single-view mesh classifier would be learning camera pose.

**On feeding `findings.json`.** An `ai:` verdict does not enter the findings
aggregation in its first version. Letting it would make the corpus
self-referential -- the judge's own opinions becoming evidence for the hints that
shape the next generation -- and that is a decision that owes its own document.
Sorting Review by score is the whole of the authority granted here.
`by_score` sorts and never filters, because a judge that hid what it disliked
would make its own mistakes invisible.

## Results

Not yet taken. Blocked on a labelling session over images that do not exist yet.

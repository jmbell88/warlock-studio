# Judging what you made

Reconstruction is not reliable. Ask for twenty meshes and you will keep some, remesh some and throw
the rest away — that is the normal shape of the work, not a sign anything is wrong.

Review is where you say which is which. It exists because those judgements are worth more than the
meshes: they add up into an answer to "which settings actually work", they train the app's own
advisory scorer, and they are the only thing standing between you and re-learning the same lesson
every month.

This chapter is also where several numbers get their honest description. At least one of them means
considerably less than its name suggests, and knowing which is the point of reading this now.

## Grading is not accepting or rejecting

Open **Review** in the rail. On the left is a list of things to judge; in the middle the mesh, which
you can orbit; on the right a panel with what the app measured and what you have said.

A verdict is a **grade** from −5 to +5. Eleven values, and they mean:

| Grade | Meaning |
| --- | --- |
| **+5** | Ships as-is. |
| **+3** | Usable. This is the cut — at or above it the app counts the mesh as working. |
| **0** | No opinion. A real answer, not a refusal to give one. |
| **−3** | Not usable. |
| **−5** | Unusable in any form. |

The keyboard is the whole interface. `1` to `5` file a positive grade. `R` arms the negative sign, so
`R` then `4` files −4. `0` files "no opinion". `S` skips to the next unjudged item without filing
anything. Left and Right step by hand, so a grade you regret is one keypress away from being redone.

Optionally, before the grade lands, you can tag what is true of the mesh. `Ctrl+1` to `Ctrl+5` are
the five good tags — clean-shape, good-texture, on-style, sharp-detail, good-topology — and
`Shift+1` to `Shift+5` are the five bad ones — holes, bad-shape, bad-texture, wrong-style, broken.
Five of each is what lets the digits map positionally onto both vocabularies without a second table
to memorise.

Tags are legal at any grade, and that is deliberate. They describe the mesh rather than justify the
verdict, so "+4, holes" is a sensible thing to say: mostly good, one specific flaw.

The list you are judging is not something you have to build. Anything finished that nobody has
judged shows up automatically in a bucket called **Recent, unreviewed** — daily work and deliberate
experiments feed one pool.

## A judging pass

When there is a backlog, grading one item at a time is the slow way. **Start judging** walks
every outstanding item in one run, binary only: `A` accepts, `R` rejects, `S` skips, `Esc` ends the pass.

Underneath, `A` and `R` file +3 and −3 — the same grades, through the same door. The eleven-point
scale still works inside a pass if a particular mesh deserves a real number. The binary mode exists
because two keys and no decisions is how you clear forty meshes, and a scale you have to think about
is how you clear four and stop.

There is one thing to know before starting a pass over a **sweep** (see below): once every item in a
sweep has a verdict, that sweep's files are deleted automatically. The verdicts survive and the
measurements survive; the meshes and images do not. The entry card warns you up front, which is why
it happens with a toast rather than another dialogue. This does *not* apply to the Recent bucket —
those are ordinary library rows and are never auto-deleted.

## What the measurements are worth

The right-hand panel carries numbers from the mesh audit. They are useful and they are not a score.
One in particular needs its caveat stated rather than implied.

**`hole_worst`** is how see-through the mesh is from its worst viewing angle. A *high* reading is
real evidence of a real problem — the mesh has gaps you will see in a game. But a **low reading
proves nothing**, because a solid, featureless, detail-free lump scores just as well as a genuinely
clean model. The app prints that caveat under a low reading rather than letting the number speak for
itself, and this chapter repeats it because it is the single easiest number in the app to over-trust.

It is also worth knowing that `hole_worst` is measured against a corpus rather than an absolute
scale. It has been re-baselined before, and the same number has meant different things at different
times. Read it as one piece of evidence about one mesh, never as a quality percentage.

## The judge is advisory, and answers one question

With the right weights installed, the app will offer a score beside items in the list. It is worth
being precise about what that is.

It is a small model trained on **your own labels**, and it answers exactly one question: *will this
reference image reconstruct well?* It looks at the reference picture. It has never seen a mesh. It
has no opinion whatsoever about mesh quality, which is why the app spells its output out in full
rather than as a bare percentage — a number on its own would be read as "this mesh is 62% good",
which it is not.

It never hides, filters or auto-rejects anything. Sorting by score puts the promising ones first;
it does not remove the others. That restraint is on purpose: a scorer that filtered its own training
input would spend its life confirming its first guess.

**Teaching the judge** opens a labelling pass — a faster loop over *images* rather than meshes, two
keys, about two seconds each, answering "is this a good 2D asset" and "will this reconstruct". Those
labels are what the scorer learns from. It is the highest-value few minutes in the app if you intend
to generate at volume.

## Blinding

There is a **Blind** toggle, and it does more than its name suggests.

It hides each item's name behind a hash — and it also **reorders the list**. That second half is the
part that matters. When you launch a set of variations, the baseline is enqueued first, so position
alone tells you which arm is which. Hiding the caption while leaving the order intact would not be
blinding at all. Under blinding the app also hides its own score, on the same reasoning: an opinion
on screen anchors the independent judgement that blinding exists to collect.

## Sweeps, and what verdicts add up to

A **sweep** launches a family of jobs that vary one or more settings around a baseline. Capture your
current Create settings as that baseline, pick the axes to vary and their values, and the pane shows
you how many jobs that is before you commit any GPU time. They queue through exactly the same door
as any other job.

Grade the results and they roll up into **What works** — a ranked list of whole configurations that
produced usable meshes, with **Apply to forms** to load a winner straight back into Create.

One subtlety about that ranking, because it changes how you read it. A verdict credits *every*
setting in the job, so the marginal effect of any single one is confounded with everything that
happened to co-occur with it. The ranked list scores whole configurations for that reason. If your
question is "which one setting made the difference", the **Axis verdicts** line is the place that
answers it — it recovers matched pairs from the sweep's structure, which is the only genuinely
one-variable-at-a-time comparison available.

## One difference from Create

The viewport in Review looks like Create's and is not quite. You can orbit, pan and zoom exactly as
you would expect, but the toolbar over it — wireframe, turntable, screenshot, frame — is drawn only
in Create. If you reach for the Wireframe button here out of habit, that is why it is not there.

## What to read next

That is the pipeline end to end: make, find, judge. The remaining tutorial chapters are about the
eight workspaces, and they are independent of each other — read whichever matches what you want to
make. [Drawing](05-drawing.md) is the usual next stop, and needs no GPU at all.

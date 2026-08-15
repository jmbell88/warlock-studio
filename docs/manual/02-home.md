# The Home screen

Warlock Studio opens on Home every time you launch it, not just the first time. That is deliberate:
the workspace assumes you already know which of the two pipelines you are in and what you are
looking at, and neither of those is true a second after a launch.

Home used to be a grid of tiles, and most of those tiles were the navigation again — the same names
under the same glyphs, one click either way, beside a control that is drawn in every mode anyway. So
Home answers the three questions nothing else in the app answered instead: **what changed**, **what
is the machine doing**, and **what was I working on**.

Nothing about Home is remembered. There is no "last mode" setting, and no way to make the app skip
it — a stored mode would be a value with no reader, and the app would drift into disagreeing with
itself about where it opens.

## What's new

The left column opens with the changelog for this build, with older releases collapsed under it. It
is read from a `CHANGELOG.md` shipped inside the app, hand-written rather than generated: every
commit in this repository is titled `Warlock vN.N.N` and carries no detail, so a generated list
would be a column of version numbers. If the file is missing or unreadable the block says so and
nothing else on the screen is affected.

The version this build is running is printed beside the title, which is the only place in the UI it
appears.

## Status

Under the changelog is a short block about this machine rather than about the work. Each line that
has somewhere to go is clickable.

| Line | What it says | Where it goes |
|---|---|---|
| Diagnostics | "Everything checks out", "still checking" for the first second or two after launch, or "N things need attention" — amber for a warning, red for something fatal. | [App settings](17-app-settings.md), where the model list and its Download buttons are. |
| Queue | What is running or queued, with a percentage when the worker is reporting one, or "Queue idle". | — |
| Library | How many assets there are, how many failed, and how much disk they occupy. | The [Library](11-library-and-jobs.md). |
| Unreviewed | How many finished meshes nobody has judged, when there are any. | [Review](13-review.md). |

The diagnostics line is a different destination from the health dot in the top-right corner, which
opens the read-only diagnostics popup: the dot answers "what is wrong right now", and this line
answers "how do I fix it". It exists because a fresh install reaches Home with no weights
downloaded, presses New 2D, and is refused at the door with a download command in the message. That
refusal is correct, but Home offered every way to start work and no way to find out first whether
this install could do any of them.

The unreviewed count is asked for in the background on a timer, never on the frame the screen is
drawn — it is a table scan — so it can lag by up to half a minute. It is not shown at all until an
answer has come back, and not shown when the answer is zero.

## Resume

The right column starts with a **New …** row — one button per thing the app can start from nothing —
and under it a single list of everything you have touched recently, newest first, badged with the
mode that opens it.

That list is genuinely one list. Inker, Clay, Plotter and Packwright each used to keep their own
recent-files history, which meant "the six things you were working on" could not be answered at all:
four separate lists carry an order within themselves and none between them. They are now one
history with a timestamp per entry, and your generated assets are folded in beside them from the
library — an asset row opens in the pane that made it, 2D for a reference or a tile and 3D for
anything else.

Arrow keys move between the rows and Enter opens the highlighted one; hovering moves the highlight
too, so the mouse and the keyboard never disagree about what Enter would do.

If a row names a file that is no longer there, clicking it drops the row and says so rather than
failing silently. A recent list that keeps offering a moved file is worse than a short one.

## Failures

If a startup check failed fatally, or the GPU worker died, the message appears in red under the
title as well as in the banner across the top of every mode. Dismissing the banner does not destroy
the text: it moves into the diagnostics popup under a **Dismissed** heading, which is the only copy
there is.

See [Troubleshooting](18-troubleshooting.md) for what the individual checks mean.

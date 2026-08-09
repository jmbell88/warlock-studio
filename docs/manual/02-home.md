# The Home screen

Warlock Studio opens on Home every time you launch it, not just the first time. That is deliberate:
the workspace assumes you already know which of the two pipelines you are in and what you are
looking at, and neither of those is true a second after a launch. Home is a chooser, so the frame
starts with a decision rather than with a form.

Nothing about Home is remembered. There is no "last mode" setting, and no way to make the app skip
it — a stored mode would be a value with no reader, and the app would drift into disagreeing with
itself about where it opens. The mode switch along the top always has a Home entry, so the chooser
is one click away from anywhere.

## The six tiles

Each tile is a whole clickable card, not a button with a label beside it.

| Tile | What it does |
|---|---|
| New 2D Image | A clean [reference form](03-generating-references.md), wearing the active style profile. The seed is rolled fresh, so this is a genuine new start rather than the last form with the prompt cleared. |
| New 3D Model | A clean [mesh form](04-generating-meshes.md), with no source asset selected. |
| Inker | The [layered raster editor](07-inker.md), with whatever documents were already open. |
| Clay | The [primitive modeller](08-clay.md). If nothing is open it starts a new empty document, because a mode with nothing in it and no obvious way to begin is a dead end. |
| Open Existing | The [library](11-library-and-jobs.md) in full, with a Continue button that opens the selected asset in the pane that made it. |
| Profiles | The [style profiles](12-profiles.md) editor. The caption names the active profile when there is one. |

Inker and Clay keep whatever was open, and the two generate tiles do not. The difference is not an
inconsistency: in Inker and Clay the *documents are the work*, and there is no form to reset, while
a generate pane is a form and starting from someone else's half-filled one is worse than starting
empty.

## The diagnostics row

Under the tiles is a single line that counts what is wrong with this installation:

- **"Diagnostics — everything checks out"** when every check passed.
- **"Diagnostics — still checking"** for the first second or two after launch, while the slow
  probes are still running in the background.
- **"Diagnostics / Set up models — N things need attention"** otherwise, in amber for a warning and
  red when something fatal is missing.

Clicking it opens [app settings](17-app-settings.md), which is where the model list and its
Download buttons live. That is a different destination from the health dot in the top-right corner,
which opens the read-only diagnostics popup: the dot answers "what is wrong right now", and this
row answers "how do I fix it".

The row exists because a fresh install reaches Home with no weights downloaded, presses New 2D
Image, and is refused at the door with a download command in the message. That refusal is correct,
but Home offered four ways to start work and no way to find out first whether this install could do
any of them.

## Failures

If a startup check failed fatally, or the GPU worker died, the message appears in red below the
tiles as well as in the banner across the top of every mode. Dismissing the banner does not destroy
the text: it moves into the diagnostics popup under a **Dismissed** heading, which is the only copy
there is — each of the three things that write a banner message writes it exactly once.

See [Troubleshooting](18-troubleshooting.md) for what the individual checks mean.

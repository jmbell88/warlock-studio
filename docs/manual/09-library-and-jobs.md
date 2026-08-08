# The library and jobs

Everything the app has ever made is a **job**, and every job is a row in the library — the lower
half of the right sidebar, under the inspector. This chapter covers what a job goes through, how to
find one, how to run one again, how to save a reusable style, and how to keep the disk under
control.

## The job lifecycle

A job moves through a small set of states, shown as a coloured pill on its card:

| Status | Meaning |
| --- | --- |
| queued | Accepted and waiting its turn. |
| running | The worker is on it now. |
| done | Finished; its files are on disk. |
| error | It failed. The card offers **Try again**. |
| cancelled | You stopped it before it finished. |

The pill carries a short glyph as well as a colour, so the state is readable without relying on hue.

Jobs run **one at a time**. The GPU is a serial resource, so the queue is genuinely a queue, and a
queued card shows its position in it ("#3 in queue") rather than leaving you guessing.

While a job runs, its card shows a progress bar and a label for the current stage, and a floating
**progress card** narrates the same thing over every mode — including Inker, so a reconstruction
started before you switched is still visible. An estimated time remaining appears once the job is
warm and meaningfully underway; it is deliberately suppressed before that, because an estimate taken
from 3% of a cold start is a guess about weights loading rather than about your mesh.

Every card offers exactly one **primary action**, chosen in a fixed order of precedence so that the
obvious next step is always the button on offer:

- A queued or running job offers **Cancel**.
- A failed job offers **Try again**.
- A finished reference offers **Make 3D**.
- A finished mesh with no rig offers **Rig** (when Blender is available).
- Anything else finished offers **Open**.

Everything else lives behind the card's overflow menu. Beside it are a checkbox for bulk selection
and a star for favouriting.

When a job reaches a terminal state, a toast says so — finished, failed with its reason, or
cancelled. Errors linger about twice as long as ordinary toasts, because they usually say what to
do next. A failed job's inspector shows the one-line reason, and expanding **Details** offers the
reconstruction engine's log and a **Save error.log...** button for the full traceback.

## Selecting and filtering

Clicking anywhere on a card selects it — the whole card is the target, not just its title. The
selected card gets a raised background and an accent edge down its left side. Selecting a finished
reference also makes it the 3D pane's promotion source, so switching to 3D mode is enough to act on
it; selecting anything else leaves that source alone, so browsing your meshes never silently changes
what **Make 3D** would submit.

Above the list are four filters and a select-all:

- A free-text box, matched against the job's name, prompt, tags and id.
- **Status**: any status, done, running, or failed.
- **Kind**: any kind, references, meshes, rigs, or sheets. Note that "reference" and "mesh" are
  about what the job *produced*, not what was submitted — a text job that stopped at a reference and
  one that went on to a mesh are the same kind of job and two different things to look for.
- A star toggle for favourites only.
- A tick that selects every asset the filters are showing, for the bulk bar below. It says *shown*
  rather than *all* deliberately: the list is a window onto the newest N (see below), so a control
  claiming everything would leave the older jobs out of the delete that usually follows. Pressing it
  again once everything shown is ticked clears them.

The filters are remembered between sessions, because a workshop tends to be filtered the same way
every time.

Below them, when anything has failed and you are not already looking at the failures, a red
**N jobs failed - show** appears. Pressing it sets the status filter to *failed*. The count is what
the press will actually reveal, not how many failed overall: the other filters still apply
afterwards, so with **Kind** on *rigs* it counts failed rigs only. Sweep units are left out, exactly
as they are left out of the list — one failed sweep is dozens of rows the library never shows.

The library holds a window on your history — the newest 200 jobs by default. When there are more, a
line at the bottom says "Showing the newest N of M" and a **Load older** button widens the window.
This matters when searching: the filters apply to what is loaded, so a history longer than the
window will tell you rather than quietly missing what it never read.

Ticking cards enables the bulk bar: **Export zip...** writes the selected meshes to a single archive,
**Save to project** copies them into a configured export folder (shown only when one is configured),
and **Delete** removes them after a confirm.

**Ticks survive a filter change**, on purpose — ticking a few meshes, switching to references and
ticking a few more is a normal way to build up a selection. The count says so when it happens: it
reads "12 selected (4 not shown)" once some of what is ticked has scrolled past the newest-N window
or been filtered away, and the delete confirm repeats the number, so the bulk actions never describe
less than they are about to do. **Clear** empties the whole selection, shown or not.

A job's name and tags are editable at the top of the inspector, and **Rename...** is in the overflow
menu too. Tags are normalised on the way in — trimmed, lowercased, deduplicated and sorted — so
"Prop" and "prop " are one tag rather than two.

## Rerun and promotion

Three overflow-menu actions run something again, and they differ in what they reuse.

**Reroll** runs the job again with the same prompt and the same guidance, on a **new seed**. Because
generation is deterministic in its seed, this is the "that's close, give me another" button. A
reroll of a reference-stage job stops at a reference again, exactly as the original did — it does
not fall through and silently pay for a reconstruction you did not ask for. A hand-painted or
imported reference cannot be rerolled at all, and the menu item is not offered: there is no
generator behind it for a new seed to change.

**Remesh** reuses the existing reference image and reruns only the 3D stage on a new mesh seed. When
the image model drew something good and the reconstruction came out poor, this retries the second
half without paying for the first. It carries no image-conditioning settings across, because a
remesh never runs the image model and a row claiming an adapter that could not have run would be a
lie about how the asset was made.

**Make 3D** — promotion — is the ordinary path from a reference to a mesh, covered in
[Starting from a reference](04-generating-meshes.md#starting-from-a-reference).

All three share one rule: **derived values never survive into the new job.** Anything the worker
recorded about the finished job's *artifacts* is stripped before the new row is written — the
composed prompt, the applied scale factor, the mesh audit, the mesh report, the optimisation record,
the transform, the weighting method and bone count, the sheet id and cells, the reference report,
the control hint and the recipe. Without that, a rerun would wear a quality verdict about a mesh
that does not exist yet.

There is a fourth action that is not a rerun at all: **Copy settings to form** loads a job's
settings back into the 2D form so you can change one thing and generate. Reroll runs a job as it
was; this is the other half. It fills in the guidance fields, the model, the LoRA and the
conditioning strengths, which prompt history alone never did.

## Profiles

A **profile** is a saved house style — the *look* half of the 2D form, stored under a name, with an
optional anchor image every generation under it is conditioned on. It has a chapter of its own:
[Style profiles](10-profiles.md).

## Storage and pruning

Every job owns a directory under `assets/`, named for its job id, and the SQLite job store lives at
`assets/jobs.sqlite`. A job directory holds:

- `input.png` — the reference image the mesh was made from.
- `source.glb` — the raw reconstruction, kept forever.
- `model.glb` — the finished mesh, derived from `source.glb`.
- `rig.glb` and `rig.json` — once the mesh is rigged. The rig lives beside the mesh it was fitted
  to, not in the rig job's own directory.
- `poses/` — one JSON file per saved pose, plus its baked GLB once you have asked for one.
- `sheets/` — one PNG plus its JSON sidecar per rendered sprite sheet.
- Derived exports (`model.stl`, `model_obj.zip`, `model.fbx`, `collision.glb`, `textures.zip`) once
  they have been requested.
- `thumb.png`, and `error.log` on a failed job.

This accumulates. At 5 to 20 MB per GLB, regular use is real disk within weeks — which is why the
foot of the library shows a **storage meter**: how many job directories exist and how many bytes
they occupy. It is measured on a background thread, so it never stalls the window.

Beside it, **Prune...** deletes everything but the newest twenty jobs, after a confirm. Running jobs
are never touched. Pruning removes both the database rows and the directories on disk, and it walks
the whole history rather than only its first page — a history long enough to need pruning is exactly
the one a single-page prune would fail on.

To remove a single asset, use **Delete** from its overflow menu. It removes the job and everything
derived from it, and it refuses on a running job — cancel it first.

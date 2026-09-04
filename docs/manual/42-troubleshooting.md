# Troubleshooting

Most things that go wrong here announce themselves: an amber issue count appears in the status bar,
a compact summary appears across the top of the window, and the Issues popup names the check that
failed. This chapter is the other half — what each of those means and what to do about it.

The summary holds the issue count and the leading action on one line. **Review** opens Issues for
the complete checks, copy actions and troubleshooting route. Separately, a toast for a failure with no message
of its own carries an **Open log** button, so the log it tells you to read is one press away rather
than inside the Issues popup.

**Dismiss** takes the banner off the screen without destroying what it said. Each of the three
things that write a banner message writes it exactly once — the startup check sweep and the two
worker checks — so clearing the list would leave a one-line count as the only surviving evidence.
The text moves into the Issues popup under a **Dismissed** heading instead.

## The issue count and Issues

The status bar along the foot of the window says **N issue(s)** in amber whenever a check has
failed or an error has been recorded, and says nothing at all when everything passed — there is no
green state to read, because a healthy install has nothing to report. Hovering it offers to open
the health details.

Clicking it opens Issues, which holds four things:

- **Every check**, passing and failing, with its detail and its remedy.
- **Effective configuration** — every setting this process is running on, with the ones that came
  from an environment variable named and highlighted first. An install whose behaviour disagrees
  with the manual almost always disagrees because something in its environment says so, and this is
  the fastest way to see it. `warlock doctor` prints the same block.
- **Dismissed**, when a banner has been dismissed this session.
- **Copy details**, **Run checks again**, **Open the log** and **Troubleshooting**.

**Run checks again** is worth knowing about. Most of the checks are only computed once, at startup —
they cannot change without the disk changing — so having just installed something the popup says is
missing, nothing short of a restart would otherwise change its mind. This button re-runs everything.

A first run that has downloaded nothing yet is better served by the **Issues / Set up models**
row on the [Home screen](21-home.md), which opens the model list and its Download buttons rather
than this read-only list.

## Out of memory

**What you see.** A job fails partway through, or the app dies outright with nothing useful on
screen. Text-to-3D jobs are the usual victims, because they are the ones that need two models at
once.

**Why.** By default the reconstruction engine (about 16 GB) and the image model (about 7 GB) stay
resident together. That fits a 32 GB card and does not fit a smaller one. Geometry resolution 1536
raises the engine's own peak allocation, and an unusually large image model such as a local FLUX
raises the other side.

**Fix.** Set `WARLOCK_VRAM_EXCLUSIVE=1` and restart. Text jobs then run sequentially — the engine is
stopped, the image model loads, generates and unloads, and the engine restarts. It costs seconds per
job and buys back roughly 7 GB of headroom. See [VRAM modes](40-configuration.md#vram-modes).

If it still fails, drop the geometry resolution: **Mesh resolution** at the Mesh stage, choosing "2D" rather
than "3D" — see
[the mapping](23-generating-meshes.md#mesh-parameters).

**Afterwards.** A hard crash inside CUDA or the allocator never reaches a Python handler, so it will
not be in `warlock.log`. Look in `crash.log` instead — see
[Where everything lives](#where-everything-lives).

## Missing weights

**What you see.** Home's health row is amber and reads "*n* things need attention — see Health";
clicking it opens **Settings → Health**, which names the failing check and its detail. The
**Models** table beside it marks every download that is not on disk. At the Reference stage the
model combo shows an entry as "weights missing". Submitting anyway is refused, and the refusal
itself carries an **Install** button that ticks exactly the downloads that job needed.

**Why.** Every image model, style LoRA, IP-Adapter, ControlNet and metric model is an optional
one-time download. The app never fetches anything at runtime, by design — only a button you press
does, and that fetch runs in its own process.

**Fix.** Open **Settings → Models**. Every registered model is listed with its size, where it comes
from and whether it is on this card; tick what you need and press *Download selected*, which fetches
the whole selection as one transaction and shows a rate and an ETA. A download can be cancelled from
its own row, and cancelling installs nothing — the staging is swept the next time the pane opens.
See [Models](41-app-settings.md#models).

**Or from a terminal**, which is the only route on a headless box: `uv run warlock doctor` lists each
missing item individually with the exact command that fetches it, and the same commands are
collected in [Model weights](39-installation.md#model-weights).

Two of these rows are **fatal** rather than a note — `trellis-server.exe` and the TRELLIS GGUF
weights. Nothing degrades gracefully without a reconstruction engine, so those get a red banner and
have to be fixed before any mesh job will run. A third can join them on a small card: **VRAM
budget** is fatal when the budget cannot hold even a lone reconstruction, because there is nothing
to degrade to there either. On a card with room it is an ordinary green row.

The model combo marks an unavailable model rather than hiding it. Listing every registered model
regardless of its weights meant picking one and learning at job-failure time what `doctor` already
knew at startup.

## Rigging is unavailable

**What you see.** The rig controls are simply not there: no rig checkbox on the generate form, no
sprite sheets, and the FBX export button explains itself with "needs Blender". The **Pose** panel
is not hidden, though — it says "Posing needs Blender, which is not installed." **Settings →
Health** names the failing check, "Blender (rigging)", and prints its detail; `warlock doctor` says
the same thing in a terminal.

**Why.** Almost always the Python version. `bpy` ships CPython 3.13 wheels only, so on anything
else `uv sync --extra rig` installs nothing at all and the probe finds no `bpy`. The other cause is
a missing skeleton template directory, which the check names explicitly.

**Fix.** Create the environment on Python 3.13 and run `uv sync --extra rig`. The check is non-fatal
either way: everything except rigging, posing and sheets works exactly as before.

**Not an error.** A rig that reports envelope weights instead of bone-heat weights has not failed.
Bone-heat weighting gives up outright on the kind of non-manifold geometry a reconstruction sometimes
produces; the worker catches that, falls back to envelope weights, and records which was used in
`rig.json` rather than failing the job. The result is cruder around joints, not broken. See
[When rigging is unavailable](25-rigging-and-posing.md#when-rigging-is-unavailable).

## The GPU worker stopped

**What you see.** A red banner reading "The GPU worker stopped" or "The GPU worker is not running",
plus a toast. Jobs you queue afterwards sit at `queued` and never start.

**Why.** The worker thread that owns the GPU pipeline died — most often the tail end of a memory
exhaustion, sometimes a native crash. The app keeps running because the window and the job store are
in different threads from the worker, which is exactly why the banner exists: without it a
mid-session worker death was invisible outside the log file.

**Fix.** Restart the app. There is no in-place recovery, and the banner says so rather than
pretending there might be.

**Then find out why.** `warlock.log` has the run's logging, including the VRAM instrumentation, and
`crash.log` has native tracebacks. The Issues popup's **Open the log** button opens the first
of those directly.

## A port is already in use at startup

**What you see.** A red banner naming the trellis port check, with an "unavailable" detail. This is
the one non-fatal check that gets a fatal check's banner.

**Why.** An orphaned reconstruction-engine subprocess from a previous crash is still holding port
`17971`. The app is perfectly usable without ever running a mesh job, which is why the check is not
fatal — but every 3D job will fail until the port is free, or worse, be served by the orphan.

**Fix.** Usually nothing. The first 3D job after the banner looks at who holds the port: if it is a
`trellis-server.exe` started from this Warlock's own vendored copy — the only case where the answer
is certain — it is an orphan, and it is ended and replaced automatically, with a warning line in
`warlock.log` saying so. Anything else is left strictly alone and the job fails naming the process
and its path; end that program, or move Warlock by setting `WARLOCK_TRELLIS_PORT`.

**If the engine keeps failing to start.** Repeated failures are spaced out rather than retried at
once — each attempt waits longer than the last, up to five minutes, and the job that triggered it
still fails immediately with the reason. That is deliberate: a burst of identical restarts buries
the first failure, which is the only one that says what actually went wrong. `trellis.log` has it.

## Warlock says the previous session did not shut down cleanly

**What you see.** A warning in `warlock.log` at startup naming the previous run's process id and
start time.

**Why.** Every session writes a marker file when it starts and removes it on the way out. A marker
still present at the next launch means the last run never reached its shutdown — a crash, a forced
kill, or a power loss.

**Fix.** Nothing to repair; the message is evidence, not a fault. It is worth acting on only in that
it says where to look: `crash.log` for a native traceback with a matching session line, and
`warlock.log` for the run's final entries. A run that ended normally logs `teardown complete`, so
the absence of that line is the sharpest confirmation of a hard death.

A variant of the same warning says another Warlock **appears to be running**. Two instances share
one job database and one engine port, and the second will lose fights over both.

## Holes or artifacts in a mesh

**What you see.** The mesh audit reports visible openings, or the surface has thin perforated
patches you can see through when you orbit.

**Why, and what does not fix it.** The reference image is the lever. The reconstruction can only be
as good as the picture it was handed, and a subject that is cropped, partly occluded, sitting on a
busy background or ambiguous from one angle reconstructs badly no matter what the engine is told.

It is tempting to reach for `WARLOCK_TRELLIS_BAND` — the width of the band the mesh extraction runs
over — on the theory that a wider band closes holes. **It does not.** That was measured with
`warlock sweep` on one reference at a fixed seed, geometry resolution 1024, and the worst-view
see-through fraction came out like this:

| Band | See-through fraction | Faces | Time |
| --- | --- | --- | --- |
| auto | 0.0077 | 267,360 | 123 s |
| 2 | 0.0077 | 266,632 | 143 s |
| 4 | 0.0167 | 290,774 | 124 s |
| 8 | 0.0110 | 297,898 | 136 s |
| 16 | 0.0125 | 289,586 | 193 s |

Two conclusions, both against the guess. The engine's own heuristic is already the best of the
ladder, and widening the band makes the surface *more* perforated while adding faces and time. So
leave `WARLOCK_TRELLIS_BAND` unset. The run also puts a floor under what counts as a real difference:
`auto` and `2` are the same setting and still disagreed by 728 faces, so anything under about 0.3% is
noise.

**Fix.** Go back a stage. Generate a better reference — one complete subject, uncropped, on a plain
background — and promote that instead. A different mesh seed is worth one try; a different reference
is worth far more. Also check that `birefnet.gguf` is present, since without it the background
matting falls back to a threshold cutout that clips soft edges.

**Related.** The two mesh measurements answer different questions and disagreeing with each other is
normal — see
[Mesh audit and mesh report](23-generating-meshes.md#mesh-audit-and-mesh-report).

## The window feels sluggish

**What you see.** Panels lag behind the pointer, or the turntable stutters, and it is not obvious
whether the app is slow or the machine is busy.

**Why.** The frame loop is capped at 60 frames per second, so a healthy session sits at 60 and never
above it. Anything lower is the loop failing to keep up — usually a GPU job running alongside the
window, a very large Inker document, or a mesh being drawn at full reconstruction density.

**Fix.** Press **F10** for the frame-rate readout, bottom-left. A permanent strip along the top of
the window used to carry the rate, this process's memory and the card's VRAM; it was developer
chrome on screen in every mode whether or not anybody wanted it, and this is what it was a summary
of. It shows the rate over the last two
seconds, the mean frame time and the slowest single frame in that window — the last of those is the
one that catches a stutter, since one 100 ms stall barely moves an average. Green is at target, amber
is degraded, red is badly behind. Press F10 again to hide it; the choice is remembered.

**Afterwards.** The rate is also written to `warlock.log` every 30 seconds beside the memory sample,
and once more when the app closes, so a session that felt slow can be checked after the fact — see
[Where everything lives](#where-everything-lives).

## Where everything lives

Everything the app generates lives under `~/.warlock`, the `.warlock` folder inside your user
profile. When something needs investigating, these are the places to look:

- `~/.warlock/assets/warlock.log` — the rotating application log, 5 MB with three backups.
- `~/.warlock/assets/crash.log` — native crash tracebacks, appended, for the failures Python logging
  cannot catch. Each run writes a `=== session … ===` line on startup, so a traceback can be tied to
  the run that produced it.
- `~/.warlock/assets/session.marker` — present only while the app is running; left behind by a
  crash, which is what produces the unclean-shutdown warning above.
- `~/.warlock/assets/jobs.sqlite` — the job store: every job row, its parameters and its status.
- `~/.warlock/assets/` plus the job id — the job's own directory, with its images, meshes, rigs,
  poses and sheets.
- `~/.warlock/MIGRATED.txt` — written once, if an older install's data was moved here out of the
  project folder. It records what came from where.

All of those move with `WARLOCK_DATA_DIR` except the note, which sits at the top of `WARLOCK_HOME`,
and the store, which has its own `WARLOCK_DB`. The full layout, and the one-time move, are in
[Data locations](40-configuration.md#data-locations).

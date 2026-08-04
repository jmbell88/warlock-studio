# Troubleshooting

Most things that go wrong here announce themselves: the health dot turns amber or red, a banner
appears across the top of the window, and the diagnostics popup names the check that failed. This
chapter is the other half — what each of those means and what to do about it.

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
job and buys back roughly 7 GB of headroom. See [VRAM modes](10-configuration.md#vram-modes).

If it still fails, drop the geometry resolution: **Detail** in the 3D pane, choosing "Indie desktop"
or "Mobile / VR" rather than "Hero asset" — see
[the mapping](03-generating-meshes.md#mesh-parameters).

**Afterwards.** A hard crash inside CUDA or the allocator never reaches a Python handler, so it will
not be in `warlock.log`. Look in `crash.log` instead — see
[Where everything lives](#where-everything-lives).

## Missing weights

**What you see.** The health dot is amber. The diagnostics list has one `x` row per missing
download, and in the 2D pane the model combo shows an entry as "weights missing".

**Why.** Every image model, style LoRA, IP-Adapter, ControlNet and metric model is an optional
one-time manual download. The app never fetches anything at runtime, by design.

**Fix.** Run `uv run warlock doctor`. Each missing item is listed individually with the exact
command that fetches it, so you can copy the line and run it. The commands are also collected in
[Model weights](09-installation.md#model-weights).

Two of these rows are **fatal** rather than a note — `trellis-server.exe` and the TRELLIS GGUF
weights. Nothing degrades gracefully without a reconstruction engine, so those get a red banner and
have to be fixed before any mesh job will run.

The model combo marks an unavailable model rather than hiding it. Listing every registered model
regardless of its weights meant picking one and learning at job-failure time what `doctor` already
knew at startup.

## Rigging is unavailable

**What you see.** The rig controls are simply not there: no rig checkbox on the generate form, no
sprite sheets, and the FBX export button explains itself with "needs Blender". The **Pose** panel
is not hidden, though — it says "Posing needs Blender, which is not installed." Diagnostics shows
"Blender (rigging)" failed.

**Why.** Almost always the Python version. `bpy` ships CPython 3.13 wheels only, so on Python 3.12
`uv sync --extra rig` installs nothing at all and the probe finds no `bpy`. The other cause is a
missing skeleton template directory, which the check names explicitly.

**Fix.** Create the environment on Python 3.13 and run `uv sync --extra rig`. The check is non-fatal
either way: everything except rigging, posing and sheets works exactly as before.

**Not an error.** A rig that reports envelope weights instead of bone-heat weights has not failed.
Bone-heat weighting gives up outright on the kind of non-manifold geometry a reconstruction sometimes
produces; the worker catches that, falls back to envelope weights, and records which was used in
`rig.json` rather than failing the job. The result is cruder around joints, not broken. See
[When rigging is unavailable](04-rigging-and-posing.md#when-rigging-is-unavailable).

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
`crash.log` has native tracebacks. The diagnostics popup's **Open the log** button opens the first
of those directly.

## A port is already in use at startup

**What you see.** A red banner naming the trellis port check, with an "unavailable" detail. This is
the one non-fatal check that gets a fatal check's banner.

**Why.** An orphaned reconstruction-engine subprocess from a previous crash is still holding port
`17971`. The app is perfectly usable without ever running a mesh job, which is why the check is not
fatal — but every 3D job will fail until the port is free, or worse, be served by the orphan.

**Fix.** End the stray `trellis-server.exe` process, then restart Warlock. If something else on the
machine legitimately owns that port, move Warlock instead by setting `WARLOCK_TRELLIS_PORT`.

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
[Mesh audit and mesh report](03-generating-meshes.md#mesh-audit-and-mesh-report).

## Where everything lives

When something needs investigating, these are the four places to look:

- `assets/warlock.log` — the rotating application log, 5 MB with three backups.
- `assets/crash.log` — native crash tracebacks, appended, for the failures Python logging cannot
  catch.
- `assets/jobs.sqlite` — the job store: every job row, its parameters and its status.
- `assets/` plus the job id — the job's own directory, with its images, meshes, rigs, poses and
  sheets.

All four move with `WARLOCK_DATA_DIR` except the store, which has its own `WARLOCK_DB`. The full
layout is in [Data locations](10-configuration.md#data-locations).

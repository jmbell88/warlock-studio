# Configuration

Warlock Studio has no *engine* configuration file. Every path, port, timeout and mode is an
environment variable, read once when the process starts, with a default that works for the tested
machine. Changing one means setting it before launching the app; changing it while the app is
running has no effect until a restart.

One thing does persist to a file, and it is deliberately not on this page: the app's own UI
preferences — theme, UI scale, pane layout and the form fields it remembers — live in
`studio_settings.json` in the data directory, written by the app itself. They are edited in
[App settings](40-app-settings.md), never by hand, and nothing in the table below is stored there.

## Environment variables

Everything the app generates lives under one home directory, `~/.warlock`, and the defaults below
are relative to it. The exceptions are the two vendored binaries, whose defaults are relative to the
project root — the directory containing `pyproject.toml` — because they ship with the checkout.
Boolean variables accept `1`, `true` or `on`; anything else is off.

| Variable | Default | Effect |
| --- | --- | --- |
| `WARLOCK_HOME` | `~/.warlock` | The one directory the app owns on this machine. Moving it moves the library, the benchmark runs, the palettes and the model weights together. Every variable below overrides it individually. |
| `WARLOCK_DATA_DIR` | `~/.warlock/assets` | Where job directories and the log files live. Created at startup if absent. |
| `WARLOCK_DB` | `~/.warlock/assets/jobs.sqlite` | The SQLite job store. Set independently of the data directory, so moving one does not move the other. |
| `WARLOCK_EXPORT_DIR` | unset | A project folder assets can be copied straight into, such as a Godot project's `assets/`. Unset means the feature is off — writing outside the data directory is opt-in, never a default. |
| `WARLOCK_TRELLIS_EXE` | `vendor/trellis/trellis-server.exe` | The reconstruction engine binary. Missing it is a fatal check. |
| `WARLOCK_TRELLIS_MODELS` | `~/.warlock/models/trellis2-gguf` | Where the TRELLIS.2 GGUF weights and `birefnet.gguf` are looked for. |
| `WARLOCK_TRELLIS_PORT` | `17971` | The local port the engine subprocess listens on. |
| `WARLOCK_TRELLIS_IDLE` | `600` | Seconds of queue inactivity before resident models are evicted to free VRAM. |
| `WARLOCK_TRELLIS_WEBP` | `off` | Ask the engine for WebP textures instead of PNG. Off is correct: WebP output declares `EXT_texture_webp` as *required*, which Godot's glTF importer refuses rather than skips. |
| `WARLOCK_TRELLIS_TEX_RES` | `512` | Texture resolution. Pinned rather than left on the engine's `auto`, which bakes visible per-texel noise into the base colour atlas at 1024 and 1536. |
| `WARLOCK_TRELLIS_BAND` | unset | Width of the narrow band the mesh extraction runs over. Empty or `auto` omits the flag entirely and lets the engine apply its own heuristic. Measurement says leave it alone — see [Holes or artifacts in a mesh](41-troubleshooting.md#holes-or-artifacts-in-a-mesh). |
| `WARLOCK_GLTFPACK` | `vendor/gltfpack/gltfpack.exe` | The mesh optimiser binary. Vendored by hand like the engine — `vendor/` is git-ignored, so a fresh clone has neither (see [gltfpack](38-installation.md#gltfpack)). Point this elsewhere to use another copy; without it jobs ship the raw reconstruction rather than failing. |
| `WARLOCK_MESH_PROFILE` | `raw` | Default triangle profile for a new job. The decimating tiers all run now, but none has been qualified, so `raw` stays the default and the only tier the generate form offers. Set this to try one; the inspector's **Triangle budget** panel is the safer place to. |
| `WARLOCK_BENCH_DIR` | `~/.warlock/bench` | Where the benchmark writes its runs. Outside the data directory on purpose, so a run survives pruning. |
| `WARLOCK_T2I_ROOT` | `~/.warlock/models` | Where every image model lives, with style LoRAs under its `loras/` subdirectory. |
| `WARLOCK_T2I_DIR` | unset | Redirects the built-in `turbo` entry (by name; not the default model) at an arbitrary local diffusers directory. It changes *where* that entry loads from and nothing else. |
| `WARLOCK_T2I_MODEL` | `sdxl_cfg` | The base model key used when a job does not name one. |
| `WARLOCK_PALETTE_DIR` | `~/.warlock/palettes` | Where pixel-art palette files live (`.hex` from Lospec, `.gpl` from GIMP). Ships empty; a missing directory simply means the palette control offers nothing. |
| `WARLOCK_VRAM_EXCLUSIVE` | auto | Restores the sequential VRAM handoff for text jobs. Unset, the mode is chosen from the card's size; set, it is honoured verbatim. See [VRAM modes](#vram-modes). |
| `WARLOCK_VRAM_BUDGET` | unset | Overrides the measured VRAM budget (GiB) that admission control checks jobs against. For a card whose free figure reports low, or for pinning tests. |
| `WARLOCK_VRAM_TOTAL` | unset | Stands in for the device total (GiB) when no GPU is visible — the escape hatch that lets the VRAM planner and `warlock doctor` run on a torch-less install. |
| `WARLOCK_RANK` | `on` | Whether a finished reference is scored, for sorting candidates: its composition report, its style anchor when one exists, and a human-preference term when PickScore is installed. |
| `WARLOCK_REFERENCE_RETRIES` | `2` | Extra redraws a text job may spend when the composition report refuses its reference. It was `0`; the 2026-08-07 sweep refused 17 references in 100 and every one of those was a whole mesh job's GPU time lost to a picture that took four seconds to redraw. Set it to `0` to get the old behaviour. |
| `WARLOCK_MESH_RETRIES` | `0` | Extra trellis runs when the finished mesh audits worse than `WARLOCK_MESH_HOLE_MAX`. The best attempt is kept, not the last. |
| `WARLOCK_MESH_HOLE_MAX` | `0.07` | The worst-view see-through fraction past which a mesh is worth redoing. Measured, not guessed — see the hole-rate baseline in `docs/measurements/`. |
| `WARLOCK_RIG_TEMPLATE` | `humanoid` | Skeleton template a rig request falls back to when it does not name one. |
| `WARLOCK_RIG_TIMEOUT` | `1800` | Seconds a single Blender rigging subprocess may run before it is treated as hung. |
| `WARLOCK_POSE_TIMEOUT` | `300` | Seconds for one pose bake. Much tighter, because a bake runs inline rather than on the job queue. |
| `WARLOCK_SHEET_TIMEOUT` | `1800` | Seconds for one sprite-sheet render. Generous because the cell count is yours to choose, but still bounded. |
| `WARLOCK_LOG_LEVEL` | `INFO` | Logging level for the console and the rotating log file. |
| `WARLOCK_POSE_FIT` | `on` | Whether a rig may measure its joint positions off the reference image rather than taking the template's. Off falls back to the template everywhere. A kill switch, not an opt-in: any doubt already refuses the whole fit. |
| `WARLOCK_DEFORM_QA` | `on` | Whether a finished rig is rendered in a battery of test poses (`rig_qa.png` beside the rig). Nothing scores it — the point is a picture you look at. Off skips the render. |
| `WARLOCK_NATIVE` | `1` | Whether the optional native kernels are used at all. `0` forces the numpy fallbacks, which is what the parity tests and an A/B timing run want. The fallbacks are never deleted, so this changes speed and nothing else. |
| `WARLOCK_NATIVE_DLL` | unset | Path to the compiled kernel library, overriding `vendor/warlockc/warlockc.dll`. |

### Seeing which of these are actually set

`warlock doctor` prints an **Effective configuration** block after its checks, and the diagnostics
popup behind the health dot carries the same list. Both mark the rows that came from the
environment rather than from a default, which is the only part that diagnoses anything: an install
whose behaviour disagrees with this table almost always disagrees because something in its
environment says so.

Five variables are deliberately absent from that list, because they are not settings the app holds
— they are read once, where they are used, and nothing keeps them: `WARLOCK_LOG_LEVEL`,
`WARLOCK_NATIVE`, `WARLOCK_NATIVE_DLL`, and the two that only mean anything during the one-time
move described under [Data locations](#data-locations), `WARLOCK_NO_MIGRATE` and
`WARLOCK_MIGRATE_KEEP`. `warlock doctor`'s **warlockc** row reports the native pair directly.

The three timeouts are ceilings on hangs, not performance targets. Automatic weights on a
300,000-face mesh are genuinely minutes of CPU, and a hung Blender holds the single-worker queue
against every job behind it — which is what the ceiling exists to prevent.

## VRAM modes

The reconstruction engine is about 16 GB resident and an SDXL-class image model about 7 GB. How
those two share a card is the one setting most worth understanding.

Left unset, the mode is **chosen from the card** at startup: a budget (the device total minus a
headroom margin) that cannot hold the engine and an image model together selects exclusive, and
anything larger selects coexist. A 32 GB card gets coexist. Set the variable explicitly and that
choice is honoured verbatim — auto-detection never overrules it. `warlock doctor` prints which mode
was chosen and why.

**Coexist.** The engine subprocess starts on the first 3D job and stays resident in
VRAM alongside the image model. Neither is stopped for the other, and both are evicted after the
idle timeout (`WARLOCK_TRELLIS_IDLE`, ten minutes). On a 32 GB card this is simply faster: a text
job does not pay to restart the engine, and a mesh job does not pay to reload it.

**Exclusive (`WARLOCK_VRAM_EXCLUSIVE=1`).** Restores the sequential handoff for a text job: stop the
engine, load the image model, generate, unload it, then restart the engine. Every step costs
seconds, and the reason to accept that is the situation where coexisting does not fit:

- A card smaller than 32 GB.
- Geometry resolution 1536, which is where the engine's own peak allocation is highest.
- A resident FLUX or any other unusually large image model.

The flag is read once at startup. Setting it mid-session does nothing until you restart.

Note that only *one* base image model is resident at a time in either mode: switching base models
between jobs unloads the previous pipeline before building the next. Style LoRAs are adapters on
whatever pipeline is already resident and switch for free.

## Data locations

Everything the app produces lives under one home directory, `~/.warlock` — the `.warlock` folder
inside your user profile. It is deliberately outside the source tree: your generated work has to
survive a reinstall, a second checkout and a `git clean`, and it has to be findable by somebody who
never cloned the repository.

```text
~/.warlock/
  assets/                  the library (WARLOCK_DATA_DIR)
  bench/                   benchmark runs (WARLOCK_BENCH_DIR)
  palettes/                pixel-art palettes you supply (WARLOCK_PALETTE_DIR)
  models/                  every downloaded model weight (WARLOCK_T2I_ROOT)
  MIGRATED.txt             written once, if anything was moved here
```

`palettes/` and `models/` are created empty at startup, because both are directories you put files
into by hand and an empty folder is a clearer instruction than a paragraph in this manual. The
vendored binaries — `trellis-server.exe`, `gltfpack.exe`, `warlockc.dll` — stay under the checkout's
`vendor/`, which is git-ignored in full: the first two are one-time manual downloads and the third
is built locally by `native\build.ps1`, so none of them arrives with a clone.

### The one-time move

Warlock used to keep all four of those directories inside the project folder. If yours are still
there, the next start moves them: it copies each root to its new home, recounts both sides, and only
then deletes the original. Nothing is removed until the copy has been verified, and the copy lands
in a `.incoming` staging directory so an interrupted move leaves something to delete rather than a
half-populated library.

It refuses rather than proceeds if another Warlock is running (tested by taking an exclusive lock on
the old `jobs.sqlite`), or if the destination volume does not have the space — the refusal names both
paths and the figures. Each root is skipped if you have already pointed its own variable somewhere,
or if the destination already holds something: two libraries have no sensible merge.

It can be a slow start. The model weights alone are tens of gigabytes, and if the checkout and your
home directory are on different drives it is a byte copy rather than a rename. Progress is printed
to the console, one line per root, and afterwards `~/.warlock/MIGRATED.txt` records what came from
where.

| Variable | Effect |
| --- | --- |
| `WARLOCK_NO_MIGRATE` | Set to anything to skip the move entirely. Nothing is examined. |
| `WARLOCK_MIGRATE_KEEP` | `1` runs the copy and the verification but keeps the originals, so you can delete them yourself once you are satisfied. |

Setting `WARLOCK_HOME` to the project folder is the other way to keep everything exactly where it
is: source and destination are then the same directory and there is nothing to move.

### Inside the data directory

```text
assets/
  jobs.sqlite              the job store
  warlock.log              rotating log, 5 MB x 3 backups
  crash.log                native crash tracebacks, appended
  <job_id>/
    input.png              the reference image
    input.orig.png         the untouched original, once a reference has been hand-edited
    paint.ora              layered working state behind a hand-edited reference; never served
    source.glb             the raw reconstruction, never overwritten
    model.glb              the finished asset, derived from source.glb
    rig.glb, rig.json      once rigged -- beside the mesh they were fitted to
    poses/<pose_id>.json   one file per saved pose, with its baked .glb beside it
    sheets/<sheet_id>.png  one sprite sheet, with its JSON sidecar beside it
```

Two things about that layout are deliberate and easy to be surprised by. A rig is written into the
*source* job's directory rather than the rig job's own, because the rig belongs to the mesh it was
fitted to. And `source.glb` is kept permanently: `model.glb` is derived from it, which is what makes
rebuilding at a different triangle budget cheap.

The two log files answer different questions. `warlock.log` is ordinary Python logging, rotating at
5 MB with three backups. `crash.log` is written by `faulthandler` from a signal handler, and it
exists for the failures logging cannot catch — a hard crash inside torch, CUDA or the allocator
never unwinds to a Python `except`, so without it such a crash leaves no in-app record at all. Both
sit in the data directory, and the Issues popup has an **Open the log** button.

If the data directory is read-only or missing, file logging is skipped with a warning rather than
refusing to start; the console handler is always there.

## Using a different image model

`WARLOCK_T2I_DIR` points the built-in `turbo` entry at a local diffusers directory of your choosing.
It is the pre-registry override, still honoured so existing setups keep working, and its limit
matters: it redirects *where* that entry loads from, and the entry still runs at its own settings —
512 pixels, 4 steps, guidance 0. Those suit schnell-like distilled checkpoints and nothing else. A
25-step model run at 4 steps with no classifier-free guidance produces mush.

So the variable is the right tool for exactly one case: swapping in another distilled checkpoint
that wants the same sampler settings. A model that needs different settings wants a registry entry
in `models.py` instead, which carries its own image size, step count, guidance scale, variant,
scheduler and always-on step-distillation LoRA — because those are properties of the checkpoint, not
of the user's preference. [Adding an image model](44-extending.md#adding-an-image-model) is the
procedure.

Every other base model always resolves under `WARLOCK_T2I_ROOT`, by the directory name its registry
entry declares.

## In-app settings

The handful of preferences that are the app's rather than a job's -- UI scale, pane layout, and the
model list with its Download buttons -- have a chapter of their own: [App
settings](40-app-settings.md).

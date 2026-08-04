# Configuration

Warlock Studio has no configuration file. Every path, port, timeout and mode is an environment
variable, read once when the process starts, with a default that works for the tested machine.
Changing one means setting it before launching the app; changing it while the app is running has no
effect until a restart.

## Environment variables

Relative defaults below are relative to the project root — the directory containing `pyproject.toml`.
Boolean variables accept `1`, `true` or `on`; anything else is off.

| Variable | Default | Effect |
| --- | --- | --- |
| `WARLOCK_DATA_DIR` | `assets/` | Where job directories and the log files live. Created at startup if absent. |
| `WARLOCK_DB` | `assets/jobs.sqlite` | The SQLite job store. Set independently of the data directory, so moving one does not move the other. |
| `WARLOCK_EXPORT_DIR` | unset | A project folder assets can be copied straight into, such as a Godot project's `assets/`. Unset means the feature is off — writing outside the data directory is opt-in, never a default. |
| `WARLOCK_TRELLIS_EXE` | `vendor/trellis/trellis-server.exe` | The reconstruction engine binary. Missing it is a fatal check. |
| `WARLOCK_TRELLIS_MODELS` | `models/trellis2-gguf` | Where the TRELLIS.2 GGUF weights and `birefnet.gguf` are looked for. |
| `WARLOCK_TRELLIS_PORT` | `17971` | The local port the engine subprocess listens on. |
| `WARLOCK_TRELLIS_IDLE` | `600` | Seconds of queue inactivity before resident models are evicted to free VRAM. |
| `WARLOCK_TRELLIS_WEBP` | `off` | Ask the engine for WebP textures instead of PNG. Off is correct: WebP output declares `EXT_texture_webp` as *required*, which Godot's glTF importer refuses rather than skips. |
| `WARLOCK_TRELLIS_TEX_RES` | `512` | Texture resolution. Pinned rather than left on the engine's `auto`, which bakes visible per-texel noise into the base colour atlas at 1024 and 1536. |
| `WARLOCK_TRELLIS_BAND` | unset | Width of the narrow band the mesh extraction runs over. Empty or `auto` omits the flag entirely and lets the engine apply its own heuristic. Measurement says leave it alone — see [Holes or artifacts in a mesh](11-troubleshooting.md#holes-or-artifacts-in-a-mesh). |
| `WARLOCK_GLTFPACK` | `vendor/gltfpack/gltfpack.exe` | The mesh optimiser binary. Not yet vendored; without it jobs ship the raw reconstruction. |
| `WARLOCK_MESH_PROFILE` | `raw` | Default triangle profile for a new job. Every other tier needs `gltfpack`, so `raw` is the only one that can currently succeed. |
| `WARLOCK_BENCH_DIR` | `bench/` | Where the benchmark writes its runs. Outside the data directory on purpose, so a run survives pruning. |
| `WARLOCK_T2I_ROOT` | `models/` | Where every image model lives, with style LoRAs under its `loras/` subdirectory. |
| `WARLOCK_T2I_DIR` | unset | Redirects the built-in `turbo` entry at an arbitrary local diffusers directory. It changes *where* that entry loads from and nothing else. |
| `WARLOCK_T2I_MODEL` | `turbo` | The base model key used when a job does not name one. |
| `WARLOCK_VRAM_EXCLUSIVE` | `off` | Restores the sequential VRAM handoff for text jobs. See [VRAM modes](#vram-modes). |
| `WARLOCK_RIG_TEMPLATE` | `humanoid` | Skeleton template a rig request falls back to when it does not name one. |
| `WARLOCK_RIG_TIMEOUT` | `1800` | Seconds a single Blender rigging subprocess may run before it is treated as hung. |
| `WARLOCK_POSE_TIMEOUT` | `300` | Seconds for one pose bake. Much tighter, because a bake runs inline rather than on the job queue. |
| `WARLOCK_SHEET_TIMEOUT` | `1800` | Seconds for one sprite-sheet render. Generous because the cell count is yours to choose, but still bounded. |
| `WARLOCK_LOG_LEVEL` | `INFO` | Logging level for the console and the rotating log file. |

The three timeouts are ceilings on hangs, not performance targets. Automatic weights on a
300,000-face mesh are genuinely minutes of CPU, and a hung Blender holds the single-worker queue
against every job behind it — which is what the ceiling exists to prevent.

## VRAM modes

The reconstruction engine is about 16 GB resident and an SDXL-class image model about 7 GB. How
those two share a card is the one setting most worth understanding.

**Coexist (the default).** The engine subprocess starts on the first 3D job and stays resident in
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

Everything the app produces lives under the data directory (`assets/` by default):

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
sit in the data directory, and the diagnostics popup has an **Open the log** button.

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
of the user's preference. [Adding an image model](14-extending.md#adding-an-image-model) is the
procedure.

Every other base model always resolves under `WARLOCK_T2I_ROOT`, by the directory name its registry
entry declares.

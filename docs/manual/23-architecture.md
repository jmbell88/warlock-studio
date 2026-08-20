# Architecture

Warlock Studio is one operating-system process. There is no server, no browser and no local HTTP
API — the one exception is the reconstruction engine, which is a vendored binary that happens to
speak HTTP on a loopback port, and nothing else in the app does. What follows is how that single
process is arranged, and why each boundary inside it is where it is.

## One process, three threads

The window has a 16 ms frame budget and the work the app does is measured in minutes. Those two
facts are irreconcilable on one thread, so there are three, and the split between them is the whole
design.

| Thread | Runs | May block |
| --- | --- | --- |
| main (pygame) | events, imgui, the viewport, job-store reads | no |
| `warlock-loop` | the asyncio loop hosting the GPU worker | asyncio only |
| `TaskRunner` pool | service calls: exports, bakes, prune | yes |

**The frame loop** owns the window, the OpenGL context and every pixel. It reads the job store
directly, because that is a fast local SQLite query behind a lock and going through another thread
to get it would buy nothing. It never waits on anything else.

**The loop thread**, named `warlock-loop`, hosts an asyncio event loop, and the GPU worker lives on
it. That is
not decoration: the worker's `wake` is loop-affine and its cancellation path is a coroutine, so
moving the worker onto a plain thread would mean rewriting the queue rather than moving it.
`studio/runtime.py` is what starts and stops all of this — it is the old FastAPI lifespan ported
step for step, which is why the shape is familiar to anyone who saw the server version.

**The task pool** is four threads, and everything that can genuinely block goes there:
mesh exports, Blender bakes, the prune sweep, tokenizer loads. Four is a deliberate middle — enough
that a slow export does not stall a thumbnail decode, few enough that a handful of them cannot
starve the loop thread of CPU. Each task is submitted under a key, and the key is what a spinner
binds to and what deduplicates a double-clicked button into one export.

Progress needs no protocol at all as a result. The worker keeps its progress in memory, behind its
own lock, never touching the database from its reader side; the frame loop asks it for a snapshot
every frame. That is why the percentage eases smoothly rather than stepping — there is no poll
interval to be a multiple of.

The visible consequence of the split is that the app survives the worker dying. The window and the
job store are in different threads from the GPU pipeline, so a worker that fails takes the queue
with it and leaves the UI standing — which is exactly why there is a banner saying so. See
[The GPU worker stopped](22-troubleshooting.md#the-gpu-worker-stopped).

## The service layer

Everything that used to live in an HTTP route body now lives in `service/`, as plain synchronous
functions that take a `WarlockService` as their first argument and raise the exceptions in
`service/errors.py` — `Invalid`, `NotFound`, `Conflict`, `NotReady`, `Failed` and friends — instead
of returning status codes.

Two things follow from that shape. The panes and the tests call exactly the same functions, so a
test never exercises a different path from the one the button presses. And there is nothing between
the UI and the queue except this layer: no adapter, no serialisation, no protocol to keep in sync.

The functions are free functions rather than methods on a god object on purpose. The split into
modules is by subject — `jobs`, `rig`, `sheets`, `export`, `files`, `derive`, `system` — and folding
them into a single class would only put that structure back inside one file.

`WarlockService` itself holds the small amount of state the functions share: the config, the store,
the worker handle, the loop reference, the doctor-check cache, and the table of per-artifact
conversion locks described under [Derived artifacts](24-pipelines.md#derived-artifacts).

## The job store

`db.py` wraps exactly one `sqlite3` connection, opened with `check_same_thread=False`, and every
method that touches it takes an `RLock` first.

The single connection is not a simplification for its own sake — it is what makes the frame loop's
direct reads safe. The lock is what makes the writes safe, and the reason it cannot be skipped is
worth stating plainly: `asyncio.to_thread` does not serialise anything. Its default executor is a
multi-worker pool, so two concurrent operations genuinely land on different OS threads, and without
the explicit lock two `commit()` calls could interleave. Any code that reaches the connection takes
the lock too; that is the entire concurrency story for persistence.

The lock guards statements, though, not read-modify-write *sequences*. A job's parameters are one
JSON blob in one column, so a caller that reads a job, edits its params and writes them back
performs a last-write-wins update that silently discards anything committed in between. That is not
hypothetical: the worker records derived values onto a row while a retarget rewrites the same row's
triangle budget. So a partial parameter update goes through `merge_params`, which reads the blob,
applies the changes and writes it back under one hold of the lock. Neither writer touches the keys
the other cares about, so merging loses nothing.

The schema itself is append-only. `MIGRATIONS` is a list of statement batches, each applied in one
transaction and bumping SQLite's `user_version` by one; a fresh database gets the base schema and
then replays every entry, so a new install and a two-year-old one converge on the same shape. An
entry that has shipped is never edited, only followed by another.

One startup detail is a deliberate refusal to be clever: a job still marked `running` when the
process starts was orphaned by a crash, and it is surfaced as such rather than silently re-queued.
Re-running a two-minute GPU job on every launch is a worse failure than showing you an error.

## Offline by construction

The app never touches the network at runtime, and that is enforced in two independent places
because one of them would be a promise rather than a mechanism.

Every model load uses a local filesystem path with `local_files_only=True`. That is the real
guarantee. On top of it, `src/warlock/__init__.py` sets `HF_HUB_OFFLINE=1` and
`HF_HUB_DISABLE_TELEMETRY=1` as the very first thing the package does, before any import that could
pull in `huggingface_hub` — which reads those variables once, at its own import time. Because the
hub library is only ever imported lazily from modules inside the package, setting them in the
package `__init__` runs first for every entry point there is: the CLI, the app, the benchmark, the
tests. They are set with `setdefault`, so a deliberate override by the user still wins.

Missing weights therefore fail loudly with the exact one-time `hf download` command rather than
being fetched. That is the whole reason installation has a manual download step at all — see
[Model weights](19-installation.md#model-weights) and
[Offline by design](19-installation.md#offline-by-design).

There is one exception and it is deliberately shaped so that it changes nothing above. The Settings
pane's **Download** button spawns a separate process, which sets `HF_HUB_OFFLINE=0` in its own
environment, fetches one repository and exits. The app process never sets that variable to anything
but `1`, and nothing on the generation path can reach the fetcher. A subprocess rather than a
temporary flag flip precisely because `huggingface_hub` reads the variable at import time: in
process, "is this offline" would become a question about import order instead of about one line.

## The GL context

There is one moderngl context, and both the 3D viewport and the imgui panels draw through it.

`studio/imgui_backend.py` is a reimplementation of imgui's OpenGL3 backend on top of moderngl,
rather than imgui-bundle's own, which draws through PyOpenGL. The reason is state caching: moderngl
tracks what is currently bound and skips redundant GL calls, so raw `glBindTexture` and
`glUseProgram` calls made behind its back leave the viewport rendering with whatever the panels
happened to bind last. Two backends fighting over one context produce exactly the class of bug that
looks like a shader problem and is not.

The same bridge explains a small asymmetry in `widgets.texture_ref`, which *registers* a texture as
well as wrapping it for imgui. The id imgui carries is the raw GL name, but the renderer binds
through moderngl, which has no "bind this raw id" operation — so it needs the object as well as the
number. A texture the renderer has not been told about maps to no moderngl object, and the draw
leaves whatever was bound in place, which in practice means every image in the UI comes out as the
font atlas.

Registration has a matching half: a texture is forgotten before it is released, or the backend keeps
a dead object under a GL name the driver is free to hand to something else. The thumbnail cache goes
one step further and defers each release by a frame, never evicting an entry that was handed out
during the current frame — a card asks for its texture while the UI is being built, and the pixels
are not read until the backend draws, so freeing inside that window frees something the draw list
still points at.

Rendering parity is measured against three.js r170's stock pipeline rather than against an idea of
what PBR should look like, because the browser build this replaced never wrote a custom shader.
`viewer/programs.py` therefore reimplements what three does: the Stephen Hill ACES fit including its
`/0.6` pre-scale, the exact piecewise sRGB transfer function, and the same `DFGApprox` polynomial.
The viewport background is deliberately not tone-mapped, because three sets a clear colour straight
back to sRGB — so the background is the literal hex you asked for.

The GLB loader is hand-rolled for two reasons that no general-purpose loader satisfies: trimesh
discards a scene root's transform, which is precisely where the grounding transform is written, and
it has no notion of a skin. `viewer/gltf.py` keeps the node graph live after loading, because posing
*is* setting a joint node's local rotation and recomputing world matrices — see
[The pose contract](24-pipelines.md#the-pose-contract).

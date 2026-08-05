# Warlock Studio Manual

Warlock Studio is a local, fully offline desktop application that turns a text prompt or a
reference image into a game-ready, textured 3D asset on your own GPU. This manual is written for
three kinds of reader at once: the person making assets, the person installing and configuring the
app, and the person changing its code. Part I is the tour of the app itself, Part II covers getting
it running and keeping it running, and Part III explains how it is built.

## Using Warlock Studio

- [Overview](01-overview.md) — what the app is, the two-stage pipeline, and what each part of the window does.
- [Generating references](02-generating-references.md) — the prompt, the guidance selects, models, seeds and image conditioning.
- [Generating meshes](03-generating-meshes.md) — promoting a reference, mesh settings, triangle budgets, quality reports and exports.
- [Rigging and posing](04-rigging-and-posing.md) — fitting a skeleton, posing it with gizmos, and saving poses.
- [Sprite sheets](05-sprite-sheets.md) — baking poses and directions into a 2D sheet with a JSON sidecar.
- [Inker](06-inker.md) — the layered raster editor and the two directions it connects to the pipeline.
- [Clay](07-clay.md) — modelling from primitives, and the two ways a built document leaves the mode.
- [The library and jobs](08-library-and-jobs.md) — job status, filters, rerunning, style profiles, storage and pruning.
- [Keyboard shortcuts](09-shortcuts.md) — every binding the app answers to.

## Setup & operations

- [Installation](10-installation.md) — requirements, dependencies and the one-time model downloads.
- [Configuration](11-configuration.md) — environment variables, data locations and VRAM modes.
- [Troubleshooting](12-troubleshooting.md) — what the diagnostics say and what to do about it.

## Architecture

- [Architecture](13-architecture.md) — the process, the threads, the job store and the service layer.
- [Pipelines](14-pipelines.md) — how a job travels from prompt to GLB.
- [Extending](15-extending.md) — adding a model, a skeleton or a guidance field.

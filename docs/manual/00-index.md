# Warlock Studio Manual

Warlock Studio is a local, fully offline desktop application that turns a text prompt or a
reference image into a game-ready, textured 3D asset on your own GPU. This manual is written for
three kinds of reader at once: the person making assets, the person installing and configuring the
app, and the person changing its code. Part I is the tour of the app itself, Part II covers getting
it running and keeping it running, and Part III explains how it is built.

## Using Warlock Studio

- [Overview](01-overview.md) — what the app is, the two-stage pipeline, and what each part of the window does.
- [The Home screen](02-home.md) — the chooser the app opens on, and the diagnostics row under it.
- [Generating references](03-generating-references.md) — the prompt, models, seeds and image conditioning.
- [Generating meshes](04-generating-meshes.md) — promoting a reference, mesh settings, triangle budgets, quality reports and exports.
- [Rigging and posing](05-rigging-and-posing.md) — fitting a skeleton, posing it with gizmos, and saving poses.
- [Sprite sheets](06-sprite-sheets.md) — baking poses and directions into a 2D sheet with a JSON sidecar.
- [Inker](07-inker.md) — the layered raster editor and the two directions it connects to the pipeline.
- [Clay](08-clay.md) — modelling from primitives, and the two ways a built document leaves the mode.
- [Plotter](09-plotter.md) — tile maps: tilesets, layers, objects, and Tiled import and export.
- [Packwright](10-packwright.md) — packing sprites into an atlas, and the sidecar that describes it.
- [The library and jobs](11-library-and-jobs.md) — job status, filters, rerunning, style profiles, storage and pruning.
- [Style profiles](12-profiles.md) — saving a house style, and the anchor image that shows one.
- [Review](13-review.md) — judging finished meshes, parameter sweeps, and the findings the verdicts add up to.
- [Keyboard shortcuts](14-shortcuts.md) — every binding the app answers to.

## Setup & operations

- [Installation](15-installation.md) — requirements, dependencies and the one-time model downloads.
- [Configuration](16-configuration.md) — environment variables, data locations and VRAM modes.
- [App settings](17-app-settings.md) — UI scale, pane layout, and the model list with its downloads.
- [Troubleshooting](18-troubleshooting.md) — what the diagnostics say and what to do about it.

## Architecture

- [Architecture](19-architecture.md) — the process, the threads, the job store and the service layer.
- [Pipelines](20-pipelines.md) — how a job travels from prompt to GLB.
- [Extending](21-extending.md) — adding a model, a style or a skeleton.

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
- [The 3D viewport](05-the-3d-viewport.md) — the camera, the toolbar over it, and what else the scene carries.
- [Rigging and posing](06-rigging-and-posing.md) — fitting a skeleton to a mesh, and posing that asset.
- [Poser](07-poser.md) — authoring a pose against a skeleton, so it applies to every asset that shares it.
- [Sprite sheets](08-sprite-sheets.md) — baking poses and directions into a 2D sheet with a JSON sidecar.
- [Inker](09-inker.md) — the layered raster editor and the two directions it connects to the pipeline.
- [Inker: animation](10-inker-animation.md) — the timeline: cels, links, tags, onion skin, ranges and clip exports.
- [Clay](11-clay.md) — modelling from primitives, and the two ways a built document leaves the mode.
- [Plotter](12-plotter.md) — tile maps: tilesets, layers, objects, and Tiled import and export.
- [Packwright](13-packwright.md) — packing sprites into an atlas, and the sidecar that describes it.
- [Troupe](14-troupe.md) — character sprite sheets: a prompt to a rigged mesh to 256 animated cells.
- [The library and jobs](15-library-and-jobs.md) — job status, filters, rerunning, style profiles, storage and pruning.
- [Style profiles](16-profiles.md) — saving a house style, and the anchor image that shows one.
- [Review](17-review.md) — judging finished meshes, parameter sweeps, and the findings the verdicts add up to.
- [Keyboard shortcuts](18-shortcuts.md) — every binding the app answers to.

## Setup & operations

- [Installation](19-installation.md) — requirements, dependencies and the one-time model downloads.
- [Configuration](20-configuration.md) — environment variables, data locations and VRAM modes.
- [App settings](21-app-settings.md) — UI scale, pane layout, and the model list with its downloads.
- [Troubleshooting](22-troubleshooting.md) — what the diagnostics say and what to do about it.

## Architecture

- [Architecture](23-architecture.md) — the process, the threads, the job store and the service layer.
- [Pipelines](24-pipelines.md) — how a job travels from prompt to GLB.
- [Extending](25-extending.md) — adding a model, a style or a skeleton.

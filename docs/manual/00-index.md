# Warlock Studio Manual

Warlock Studio is a local, fully offline desktop application that turns a text prompt or a
reference image into a game-ready, textured 3D asset on your own GPU. This manual is written for
three kinds of reader at once: the person making assets, the person installing and configuring the
app, and the person changing its code.

Start with the tutorials. They walk one path through the app and explain what you are looking at as
you go, beginning with what the app needs before it will run at all. The parts after them are
reference: Part II describes every screen, Part III covers installing and operating the app, and
Part IV explains how it is built.

## Tutorials

- [Before you begin](01-before-you-begin.md) — hardware, the two downloads, and what works without them.
- [Your first asset](02-your-first-asset.md) — a prompt to a textured mesh, and why it stops halfway.
- [Finding your work again](03-finding-your-work.md) — Home, the library, and the four kinds of delete.
- [Judging what you made](04-judging-what-you-made.md) — grades, tags, and what the measurements are worth.
- [Drawing](05-drawing.md) — Inker: tools, inks, layers, colour and selections.
- [Animating](06-animating.md) — the timeline: frames, copy versus link, tags and onion skin.
- [Modelling](07-modelling.md) — Clay: primitives, element editing, and merge versus union.
- [Rigging and posing](08-rigging-and-posing.md) — fitting a skeleton, the A-pose trap, and the pose library.
- [Building a map](09-building-a-map.md) — Plotter: tilesets, terrain, objects and Tiled.
- [Packing an atlas](10-packing-an-atlas.md) — Packwright, and the power-of-two trap.
- [A character sprite sheet](11-a-character-sprite-sheet.md) — Troupe, and what in it is proven.
- [Tuning what you get](12-tuning-what-you-get.md) — seeds, LoRAs and conditioning.
- [Putting it in a game](13-putting-it-in-a-game.md) — exports, engines, and the interop caveats.
- [Making a soundtrack](14-making-a-soundtrack.md) — Sirens: a bassline, an envelope, a sound effect, a WAV.

## Using Warlock Studio

- [Overview](20-overview.md) — what the app is, the two-stage pipeline, and what each part of the window does.
- [The Home screen](21-home.md) — the chooser the app opens on, and the diagnostics row under it.
- [Generating references](22-generating-references.md) — the prompt, models, seeds and image conditioning.
- [Generating meshes](23-generating-meshes.md) — promoting a reference, mesh settings, triangle budgets, quality reports and exports.
- [The 3D viewport](24-the-3d-viewport.md) — the camera, the toolbar over it, and what else the scene carries.
- [Rigging and posing](25-rigging-and-posing.md) — fitting a skeleton to a mesh, and posing that asset.
- [Poser](26-poser.md) — authoring a pose against a skeleton, so it applies to every asset that shares it.
- [Sprite sheets](27-sprite-sheets.md) — baking poses and directions into a 2D sheet with a JSON sidecar.
- [Inker](28-inker.md) — the layered raster editor and the two directions it connects to the pipeline.
- [Inker: animation](29-inker-animation.md) — the timeline: cels, links, tags, onion skin, ranges and clip exports.
- [Clay](30-clay.md) — modelling from primitives, and the two ways a built document leaves the mode.
- [Plotter](31-plotter.md) — tile maps: tilesets, layers, objects, and Tiled import and export.
- [Packwright](32-packwright.md) — packing sprites into an atlas, and the sidecar that describes it.
- [Troupe](33-troupe.md) — character sprite sheets: a prompt to a rigged mesh to 256 animated cells.
- [Sirens](34-sirens.md) — the chiptune tracker: patterns, instruments, sound effects and WAV export.
- [The library and jobs](35-library-and-jobs.md) — job status, filters, rerunning, storage and pruning.
- [Review](36-review.md) — judging finished meshes, parameter sweeps, and the findings the verdicts add up to.
- [Keyboard shortcuts](37-shortcuts.md) — every binding the app answers to.

## Setup & operations

- [Installation](38-installation.md) — requirements, dependencies and the one-time model downloads.
- [Configuration](39-configuration.md) — environment variables, data locations and VRAM modes.
- [App settings](40-app-settings.md) — UI scale, pane layout, and the model list with its downloads.
- [Troubleshooting](41-troubleshooting.md) — what the diagnostics say and what to do about it.

## Architecture

- [Architecture](42-architecture.md) — the process, the threads, the job store and the service layer.
- [Pipelines](43-pipelines.md) — how a job travels from prompt to GLB.
- [Extending Warlock Studio](44-extending.md) — adding a model, a style or a skeleton.

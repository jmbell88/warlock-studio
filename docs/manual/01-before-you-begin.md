# Before you begin

Warlock Studio makes game art on your own machine. You give it a text prompt or a picture; it gives
you back a textured 3D model, a sprite sheet, a tile map or a drawing — whichever you asked for. It
is one desktop window with no server to start, no browser tab and no account, and after the initial
downloads it never touches the network again.

This is the first of the tutorial chapters. It is the only one about getting ready rather than about
making something, and it exists because Warlock asks for two large downloads and a particular kind
of graphics card before most of it will run. Reading it first saves finding that out one refusal at
a time.

The reference chapters that follow the tutorials describe every control in the app. These first
chapters instead walk one path through it and explain what you are looking at as you go.

## What you need

- **Windows, with an NVIDIA GPU.** The reconstruction engine has no CPU fallback — it either has a
  CUDA card with about 16 GB free or it does not run. The tested machine is an RTX 5090 with 32 GB.
- **[uv](https://docs.astral.sh/uv/)**, which manages the Python environment.
- **Python 3.12 or newer**, with one exception worth knowing now: **rigging needs exactly 3.13**,
  because `bpy` — Blender as a library — ships CPython 3.13 wheels and nothing else. On any other
  version the rig extra installs nothing at all, quietly. The app then hides its rig controls and
  `warlock doctor` reports rigging unavailable; everything else works unchanged. If posing and
  character sheets are why you are here, use 3.13.
- **About 23 GB of disk for weights**, plus room for what you make.

A machine that misses some of this still runs a useful amount of the app. There is a table of
exactly how much further down.

## Getting the app

Today Warlock runs from a source checkout:

```powershell
uv sync --extra studio --extra text2image --extra rig
```

The three extras are separable and each buys a different capability — `studio` is the window and its
renderer, `text2image` is text-to-3D, `rig` is rigging, posing and character sheets. A bare
`uv sync` prunes all three, so pass them.
[Python dependencies](39-installation.md#python-dependencies) explains what skipping each one costs.

One thing that is not a Python package: the reconstruction engine is a native binary that belongs in
`vendor/trellis/`, and it is a manual download.
[The trellis binary](39-installation.md#the-trellis-binary) has the link and the build this version
was tested against. Without it the app starts normally and every 3D job fails, so do it before you
go looking for a bug.

A Windows installer lives in the repository and produces the same layout — Warlock requires a
*checkout-shaped* root either way, which is why nothing in it needs an installed-only code path. It
has not yet been run end to end on real hardware, so the checkout above is the path to trust.

## The first launch

Start the app and the first thing you see is a panel titled **Set up this PC**. It appears once, on
the first launch against a given data directory, and not again once you answer it.

![Set up this PC: the GPU and VRAM readout, three readiness verdicts, and the required downloads](img/01-first-run.png)

It reports three verdicts, all measured on your actual machine while the app was starting rather
than guessed:

| Row | What it means |
| --- | --- |
| **3D reconstruction** | Whether there is a CUDA GPU with enough free VRAM. No CPU fallback exists, so this is a yes or a no. |
| **Image generation** | Whether the default image model fits the VRAM budget. Both "Ready" and "Ready (the image model and reconstruction run separately)" are fine — the second means they take turns rather than sharing the card. |
| **Rigging** | Whether Blender is importable, which is the Python 3.13 question from above. |

Under them it lists the two model packages the app needs, what they cost, and whether the volume has
room. Then two buttons. **Download models** takes you to Settings with both rows already ticked and
starts the fetch. **Not now** closes the panel and leaves you to it. Either way the panel is done;
the same rows are always reachable at Settings → Models.

The two packages are the TRELLIS.2 GGUF weights — about 16 GB, the reconstruction engine's own
weights — and SDXL 1.0, about 7 GB, the image model that draws your reference. Roughly 23 GB between
them.

That download is the only network use there is, and the mechanism is deliberate rather than
incidental. The app process sets `HF_HUB_OFFLINE=1` at import and keeps it for its entire life; the
Download button spawns a *separate* process which goes online in its own environment, fetches one
repository into a staging directory beside its destination, moves the files in only if it succeeded,
and exits. So a cancelled or failed fetch leaves no half-populated model directory, and the app
never becomes online-capable — not even briefly. To run the downloads yourself instead,
[Model weights](39-installation.md#model-weights) has the commands with their pinned revisions.

## What works before the downloads finish

Quite a lot, which is worth knowing if you are reading this while 23 GB arrives.

| Works right now | Needs weights, a GPU, or both |
| --- | --- |
| Inker — drawing and animation | Generating a reference image from a prompt |
| Clay — modelling from primitives | Reconstructing a 3D model from a reference |
| Plotter — tile maps | Re-texturing a finished model |
| Packwright — atlas packing | Starting a character in Troupe from a prompt |
| Poser, if Blender installed | Rigging, if Blender did not install |
| The Library, and every form in Create | |

The pattern is that the *editors* are yours immediately and the *generators* are what the weights
buy. With no NVIDIA card at all, everything in the left column still works, and so does the half of
the character-sheet pipeline that starts from a model you already have.

Every generator refuses at the door rather than half-running: press Generate with a model missing
and you get a sentence naming the model and an offer to install it, not a failed job that wastes
your time first. Triggering that refusal on purpose is the shortest route to the Models page.

## Where your work lives

Everything Warlock generates goes under one directory, `~/.warlock`, and not inside the checkout:

```text
~/.warlock/
  assets/          your jobs, their files, and jobs.sqlite
  models/          downloaded weights
  palettes/        your own palette files
```

That matters more than it sounds. Work surviving a `git clean`, a reinstall or a move to a new
checkout is exactly what putting it outside the source tree buys. An older install that kept things
inside the checkout has them moved here on the next start — copied first, verified, and only then
deleted. [Data locations](40-configuration.md#data-locations) covers the environment variables that
move any of it somewhere else.

## When something is wrong

Two places, both worth knowing before you need them.

The **status bar** along the foot of the window is the app's opinion of itself. When everything
checks out it says nothing about health at all; when something is wrong an amber **N issue(s)**
joins it, and clicking that lists what — a missing weight, a binary it cannot find, a GPU it cannot
see — each with the exact command that fixes it.

`uv run warlock doctor` asks the same questions from a terminal and prints the same answers, which
is the more useful of the two when the window will not open at all.
[Troubleshooting](42-troubleshooting.md) is organised by symptom.

## What to read next

[Your first asset](02-your-first-asset.md) makes something, and explains the one thing about
Warlock's pipeline that surprises nearly everyone the first time.

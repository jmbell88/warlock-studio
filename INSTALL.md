# Installing Warlock Studio

This guide walks you through downloading, installing, and confirming that Warlock Studio works on your PC. No coding experience or developer tools are required — just follow the steps in order.

This covers the packaged Windows installer only. If you want to run Warlock Studio from source or contribute to it, see `README.md` instead.

**The installer changed on 2026-09-04.** It used to stage every heavy dependency (image generation, rigging, music) up front. It now ships a slim **base runtime** — the app, the window, and nothing that needs a GPU-sized Python stack — and you add the pieces you actually want afterward from **Settings → Packs**, in-app. Model weights are a separate step again, from **Settings → Models**, exactly as before. This guide describes that new flow.

## What you'll need

- **Windows 10 or 11, 64-bit.** There is no macOS or Linux build, and no CPU-only mode — an NVIDIA GPU is required for image generation and 3D reconstruction, though the base app runs without one.
- **An NVIDIA GPU with CUDA, 16 GB VRAM or more**, for 3D reconstruction and image generation. Tested on an RTX 5090 (32 GB); a 4080/5080-class card or better is a comfortable fit. Keep your NVIDIA driver up to date via the NVIDIA app or GeForce Experience.
- **32 GB of system RAM**, recommended. Windows counts the GPU's memory allocation against your system RAM as well as VRAM, so a low-RAM machine can refuse to start jobs even when VRAM is free.
- **Free disk space** for four things, each sized separately below: the installer download, the installed base runtime, whichever dependency packs you choose, and whichever model weights you choose. All four together were roughly 35 GB under the old all-in-one installer; the base install alone is much smaller now, and the exact figures for this build are:
  - Installer download: **about 810 MB** (846,946,556 bytes)
  - Installed base runtime: **about 1.4 GB** — most of it the vendored TRELLIS and CUDA binaries (840 MB) and the bundled Python runtime (425 MB)
  - Dependency packs: see [Step 4](#step-4-add-dependency-packs), sized per pack
  - Model weights: see [First launch](#first-launch), sized per model
- **A 1920×1080 or larger display** at 100% scaling. The app window opens at 1600×950.
- You do **not** need to install Python, `uv`, or any other developer tools — the installer bundles its own Python runtime and everything else the base app needs to run.

## Step 1: Download Warlock Studio

1. Go to the **Releases** page of the [Warlock Studio GitHub repository](https://github.com/jmbell88/warlock-studio) and open the latest release.
2. Under **Assets**, download `WarlockSetup-v0.0.38.exe` (version numbers change between releases) — a single file, size **about 813 MB**. There is nothing to unzip and no other file to fetch alongside it. Its SHA-256 is **`56bf81b311811505b3d5944118831d59d08eca3bc61695e5cdfe0c425e84cd46`**; the release page lists the current build's actual hash.
3. Wait for the download to finish before opening it. Your browser may warn that a file this large is unusual; that's expected for an installer that carries its own Python runtime and GPU libraries.

## Step 2: Install Warlock Studio

1. Double-click the downloaded `.exe`.
2. **Windows will very likely show a blue "Windows protected your PC" screen. This is expected — it is not a virus warning.** Windows SmartScreen flags installers from publishers who haven't paid for code-signing, regardless of whether the software is safe. Click **More info**, then click **Run anyway** to continue.
3. The first wizard page is the **license agreement** (GPL-3.0-or-later). Read it if you like, then accept and continue.
4. Optional: tick **Create a desktop shortcut** if you'd like one. It's unchecked by default — a Start Menu shortcut is created either way.
5. You won't be asked for an administrator password. Warlock Studio installs just for your Windows user account, into `%LOCALAPPDATA%\Programs\Warlock Studio`.
6. Click **Install** and wait — it's unpacking the base runtime, which takes a few minutes depending on your disk speed.
7. Click **Finish**. Two new Start Menu shortcuts now exist:
   - **Warlock Studio** — the app itself
   - **Warlock Doctor** — a diagnostic tool that checks your install, GPU, packs, and models (more on this under Troubleshooting below)

## Step 3: Understand the two kinds of "download after install"

The base install you just did gets you a working window, the tile/atlas/pose-library workspaces, and diagnostics — but not yet image generation, rigging, or music. That's by design: two separate, optional downloads finish the picture, and they are not the same kind of thing.

- **Dependency packs** (Settings → Packs) are *code* — Python packages such as `torch` and `diffusers` that a workspace needs in order to run at all. Without the matching pack, a mode like Create, Poser, Troupe, or Muse says what it's missing instead of opening.
- **Model weights** (Settings → Models) are *data* — the actual trained checkpoints (SDXL, TRELLIS.2, and the rest) that a pack's code loads and runs. A pack with no weights fetched yet will tell you so at the door.

You need a pack *and* its weights, in either order, before the workspace it unlocks does anything. Both are downloaded once, kept under your Warlock home, and reused by every later install or upgrade that still matches their pinned digests — see [Where things are stored](#where-things-are-stored) below.

## Step 4: Add dependency packs

Open **Settings → Packs**. Three packs are offered, matching `src/warlock/packs.py`'s registry:

| Pack | Unlocks | What it costs |
|---|---|---|
| **Image generation** | Create's text-to-image path, host-side background matting, and candidate ranking | Multi-gigabyte (torch + diffusers + transformers stack); see the pane for this build's exact figure |
| **Rigging** | Poser's skeleton fitting, and Troupe's clip rendering | The cheapest of the three — well under a gigabyte |
| **Music generation** | Muse's text-to-music generation | Multi-gigabyte (its own torch + diffusers stack, pinned separately from Image generation); see the pane for this build's exact figure |

Install whichever ones match what you actually want to do — a pixel-art-only session never needs any of them. Each pack downloads to a wheel cache under your Warlock home, verifies every file's hash, and only then installs into the app's own runtime as a short-lived background process; the app stays open and usable throughout, and the mode it unlocks lights up automatically once it finishes (a restart is only asked for if the running process genuinely cannot pick up the change).

**A pack install cannot be safely cancelled once it starts writing into the runtime.** Cancel is offered only while the pack is still downloading; once installation begins, let it finish.

## First launch

Open the Start Menu and launch **Warlock Studio**.

The first time it opens, a **Set up this PC** panel appears:

![Set up this PC: the GPU and VRAM readout, three readiness verdicts, and the downloads generation needs](docs/manual/img/01-first-run.png)

It runs three live checks against your actual GPU, VRAM, and installed packs — not a guess:

| Check | What it means |
|---|---|
| 3D reconstruction | Whether your GPU has enough free VRAM for the 3D pipeline, and whether the weights are present |
| Image generation | Whether the Image generation pack is installed and the image model fits your VRAM budget |
| Rigging | Whether the Rigging pack is installed |

Below that, it lists the model downloads *generation* needs, with sizes, and tells you up front if your disk doesn't have room:

- **TRELLIS.2 GGUF weights** — about 16 GB
- **SDXL 1.0** — about 7 GB

Two buttons:

- **Download models** — starts both downloads in the background. The app stays fully usable (and offline apart from this and pack downloads) throughout.
- **Not now** — skip for good, not just for now. Nothing is owed: the app is fully usable without either download, and the same rows are always reachable at **Settings → Models**. The Home screen keeps one quiet line offering them again.

A few things worth knowing so you don't worry unnecessarily:

- There's no time estimate shown, since it depends entirely on your internet connection. It's fine to leave it running and come back later.
- A cancelled or failed model download is safe to retry — it doesn't leave a broken, half-downloaded mess behind.
- Most of the app — drawing, tile maps, the asset library, and more — works immediately with **zero downloads of any kind**. Only AI image and 3D generation need packs and models together; the rest of the app isn't "broken" while those finish. **Create** and **Muse** are greyed out in the left-hand rail until their pack and weights are both present — clicking a greyed one takes you to the right Settings screen with exactly what's missing already highlighted, rather than opening a workspace whose buttons would all refuse.

Separately, the Home screen offers a dismissible **"New here?"** guided tour with **Start** and **Not now** buttons. It's entirely optional and never launches on its own.

When it opens, you'll see a rail on the left with **Home**, **Library**, and **Create**, then eight creative workspaces — **Inker**, **Clay**, **Poser**, **Troupe**, **Plotter**, **Packwright**, **Muse**, and **Sirens** — with **Review** and **Settings** tucked into the footer.

## Where things are stored

Everything Warlock Studio downloads or creates lives under your Warlock home, `%USERPROFILE%\.warlock` by default (`WARLOCK_HOME` overrides it) — worth knowing if you ever want to back it up or check how much space it's using:

- `assets/`, `palettes/` — your own work.
- `models/` — model **weights** fetched from Settings → Models. Multi-gigabyte, and specific to the checkpoints you chose.
- `packs/` — the **wheel cache** for dependency packs fetched from Settings → Packs. This is separate from the app's own runtime (`%LOCALAPPDATA%\Programs\Warlock Studio`), where the packages actually get installed to run — the cache exists so that reinstalling or upgrading the app doesn't re-download a pack whose files still match what's already been verified.

## Upgrading

Installing a newer version over an existing one keeps `%USERPROFILE%\.warlock` — your assets, your model weights, and the pack wheel cache — untouched. **What it does not currently do is reinstall the dependency packs you had.** The upgrade replaces the app's own runtime, and that runtime is where packs are installed *into*; today nothing restores them automatically afterward. This is a known gap (tracked in `TODO.md`), not an intended behaviour.

**After upgrading, check Settings → Packs and reinstall anything that shows as not installed.** Because the wheel cache under `%USERPROFILE%\.warlock\packs` survives the upgrade, reinstalling a pack you'd already fetched should be fast — it re-verifies the cached wheels rather than downloading them again, as long as their pinned versions haven't changed in the new release.

## Uninstalling

1. Open Windows **Settings → Apps → Installed apps** (or **Control Panel → Programs → Uninstall a program**).
2. Find **Warlock Studio** and click **Uninstall**.
3. A confirmation message tells you your assets and downloaded models are being kept. Uninstalling does **not** delete `%USERPROFILE%\.warlock`, so you won't lose your work, your fetched model weights, or your pack wheel cache — reinstalling later and reinstalling the same packs from Settings → Packs should re-download little or nothing.

## Troubleshooting

- **Warlock Doctor**, in the Start Menu, re-runs the same first-run diagnostic checks any time you want to check your setup again, including which packs and which model weights are present.
- If the SmartScreen prompt during installation is what's worrying you, that's expected — see Step 2 above.
- **If a pack or model download fails**, the message you get is whatever the network gave us, which for a connection problem is not always readable — `ConnectError: [WinError 10054]` and similar mean the connection was cut, not that anything is wrong with your install. Antivirus software, a firewall, a workplace or school proxy, or a VPN are the usual causes; try pausing them, or try a different network. Nothing is left half-installed by a failed download, so it is always safe to press Install again.
- For anything else — crashes, out-of-memory errors, missing weights, a stuck GPU worker — see the full guide at `docs/manual/42-troubleshooting.md`.

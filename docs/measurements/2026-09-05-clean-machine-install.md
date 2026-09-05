# The first install on a machine that is not the development box — 2026-09-05

**Status: concluded 2026-09-05, over two runs.** `WarlockSetup-v0.0.35.exe` was
carried on a USB drive to a second Windows PC and installed there with no
network involved. The app launched, the modes were browsable, and a drawing was
made in Inker, saved, and reopened. This is the first time any part of this
project has run on a machine that is not the one that built it, and it closes
the install half of `TODO.md` P1 — but not P1 itself: the card is 8 GB, so
nothing was generated.

The machine was then uninstalled and reinstalled, and the second run is the
more informative one. **All three dependency packs installed successfully, and
every model download still failed.** That asymmetry is the finding: it rules
out the network, the machine and the download machinery in general, and points
at one transport.

## Why this document exists

P1 has said since 2026-08-26 that "everything below has never been seen on a
machine that is not this one", and that until a machine without `uv`, Python or
a CUDA toolkit had installed this and made something, the project had no
shippable artifact. Half of that sentence is now answered and half is not, and
the distinction is exactly the kind that gets lost if it lives only in a plan
file. The figures and verdicts are here because `TODO.md` is deleted when it
empties and this is not.

## The machine and the artifact

- **Target:** a normal Windows PC belonging to a household member. No Python, no
  `uv`, no CUDA toolkit, no Visual C++ build tooling — confirmed, not assumed.
  NVIDIA card with 8 GiB of VRAM.
- **Artifact:** `WarlockSetup-v0.0.35.exe`, the slim-base build. Its measured
  figures are in [`../../INSTALL.md`](../../INSTALL.md): 810 MB download
  (846,950,916 bytes), about 1.4 GB installed base, SHA-256 `254b3af9…`.
- **Transport:** a USB drive. The installer itself touched no network at any
  point, which is the strongest form of the claim P1 wanted tested — the
  bundled runtime either carries everything the base app needs or it does not,
  and there was no fallback available to hide a gap.

## What was proven

| P1 claim | Verdict |
|---|---|
| Installs with no Python, `uv` or CUDA toolkit present | **Proven.** Fully offline, from USB. |
| The bundled runtime reaches the window | **Proven.** The app launched and every mode was browsable. |
| The base install works with no torch and no `bpy` | **Proven.** This was the step P26 added to P1 on 2026-09-04. |
| The welcome dialog offers its three doors | **Proven.** Download, not now, and show me around. |
| The hardware scan reports the real GPU | **Proven.** It named the card correctly. |
| The fatal banners name the missing-weights rows | **Proven.** The model warnings appeared. |
| A document round-trips | **Proven.** Inker drew, saved, and reopened the file. |
| The installer stages `packs.json` and the bundled wheels into `{app}\packs` | **Proven.** Settings → Packs drew a real row with a size and a live `Install (N GB)` button, rather than `app_settings.pack_blocked`'s "this build carries no packs, so this one is installed with uv instead" fallback. That fallback is what a source checkout gets, so seeing the button is the build's staging step confirmed on a machine that has never had `uv`. |
| **A dependency pack installs end to end from Settings** | **Proven on the second run.** All three — Image Generation, Rigging and Music — installed successfully on a machine that has never had `uv`. This is P26's programme validated outside the test suite for the first time, and it includes `music`, the only pack that exercises the sdist build path and the bundled-wheel branch of `pack_worker.collect`. |
| An uninstall and reinstall cycle | **Proven.** The machine was uninstalled and reinstalled between the two runs. |
| SmartScreen's "More info → Run anyway" | **Not observed.** It did not appear. `INSTALL.md` says "very likely", so the guide is not wrong, but this path remains unwitnessed. |

## What failed: Hugging Face, and only Hugging Face

Every model download failed, on both runs, with

```
ConnectError: [WinError 10054] An existing connection was forcibly closed by the remote host
```

on a machine with a working internet connection and well over 50 GB free. The
rows were tickable and the Download button was live — this is not the
`Unavailable` state, and not the disk door refusing the plan. It was tried
against the reconstruction engine, the base models, the LoRAs and the
conditioners: everything on the page.

**The second run is what makes this diagnosable.** All three dependency packs —
Image Generation, Rigging and Music, several gigabytes each — installed
successfully on the same machine, on the same network, in the same session.
So the following are all eliminated: the network, DNS in general, the disk, the
`winjob` child-process machinery, the digest-verified staging design, and the
`Python-urllib/3.13` user-agent ban that v0.0.35 had just fixed.

What is left is the one thing that differs between them:

| | transport | host | result |
|---|---|---|---|
| Dependency packs | `pack_worker`, plain `httpx` | PyPI and the pinned cu128 index | **worked** |
| Every model row tried | `fetch_worker` → `huggingface_hub.snapshot_download` | `huggingface.co` and its storage backend | **reset, 10054** |

Note that the reconstruction engine is not the exception it looks like: it is
the repository `ilintar/trellis2-gguf` and goes through `snapshot_download`
like everything else. The *only* registry row that fetches from a non-Hugging
Face host by direct URL is `hdemucs_high` (0.32 GB, `download.pytorch.org`),
and it was not tried.

### The two hypotheses, and the experiments that separate them

1. **Hugging Face hosts are being reset on that machine** — antivirus, TLS
   interception, a DNS filter or an ISP-level block. Test: open
   `huggingface.co` in a browser on that machine, and fetch the `hdemucs_high`
   row, which is small and goes to `download.pytorch.org`. If hdemucs succeeds
   and HF rows fail, this is it.
2. **The Xet transport specifically.** `hf-xet` is a dependency of
   `huggingface_hub` on Windows AMD64, so it is installed in the base runtime,
   and modern `huggingface_hub` downloads Xet-backed repositories through
   `*.xethub.hf.co` using many parallel connections — a pattern that antivirus
   and firewall products reset far more often than a single HTTPS stream.
   Nothing in this project sets `HF_HUB_DISABLE_XET`. Test: set
   `HF_HUB_DISABLE_XET=1` as a user environment variable on that machine,
   relaunch, and retry a model download. If it succeeds, that is both the
   diagnosis and the shape of the fix.

Neither has been run yet. **Nothing should be changed in the fetch path on a
guess between these two** — they have different fixes, and the second is a
one-line default while the first is a documentation-and-messaging problem.

### What the failure did not cost

Nothing was left half-installed, and the app stayed usable throughout. The
download path stages and replaces rather than writing in place, and this is the
first time that design has been exercised against a real failure rather than a
stubbed one.

### And the message is a defect either way

The raw socket error reached a non-developer verbatim, in a transient toast,
with no log-file button and no remedy. That is `TODO.md`'s finding F1, it is
independent of which hypothesis above is right, and it is fixable today.

## What this machine can now close, and what it never will

The card is 8 GiB. `vram.TRELLIS_GIB` is 16.0, so `doctor._vram_check` is fatal
here and a reconstruction is out of reach by design. **P1's remaining half — a
non-developer install that generated an asset — cannot be closed on this
machine at all**, and neither can the three recovery paths that need a
successful pack install first (quit-during-install, Repair, and Restore packs
across an upgrade). Those need either a second clean machine with a bigger card
or this one plus a working download path.

What it *can* now close, because packs install there: the three recovery paths
of P1 step 5 — quit-during-install, Repair on a clean pack and on a
hand-damaged one, and Restore packs across an upgrade. None of them needs a
model weight or a card, all three have only ever run against fakes, and the
machine is now in exactly the state they need.

## What it changed

- `TODO.md` P1 keeps its number and narrows to the generation half; the
  install, launch and base-runtime steps are struck against this document.
- `TODO.md` P27 closed: the release candidate exists, and its figures are in
  `INSTALL.md`.
- Two findings opened against the tree, both buildable. F1 was rewritten after
  the second run: it is not "downloads fail" but "Hugging Face fails while PyPI
  works", which is a far narrower thing.

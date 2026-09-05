# The first install on a machine that is not the development box — 2026-09-05

**Status: concluded 2026-09-05, over two installs and four app sessions, with
her `warlock.log` copied back for the diagnosis.** `WarlockSetup-v0.0.35.exe` was
carried on a USB drive to a second Windows PC and installed there with no
network involved. The app launched, the modes were browsable, and a drawing was
made in Inker, saved, and reopened. This is the first time any part of this
project has run on a machine that is not the one that built it, and it closes
the install half of `TODO.md` P1 — but not P1 itself: the card is 8 GB, so
nothing was generated.

The machine was then uninstalled and reinstalled. **All three dependency packs
installed successfully, and every model download still failed** — and her log
turns that from a symptom into a diagnosis. The headline is not the network:
it is that a failed fetch deletes everything it has downloaded, so a 16 GB
model cannot be retried into existence on an imperfect line.

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

## What failed: the download path, diagnosed from her log

Her `warlock.log` was copied back on 2026-09-05 and is the first real evidence
this project has had from a machine it does not own. It covers 14:07 to 15:21
and four app sessions. What it shows:

### 1. Every model fetch fails, against more than one host

```
14:09:21  fetch of ilintar/trellis2-gguf failed: ConnectError: [WinError 10054] ...
14:13:32  fetch of stabilityai/stable-diffusion-xl-base-1.0 failed: ConnectError: [WinError 10054] ...
14:49:03  fetch of ACE-Step/ACE-Step-v1-3.5B failed: ConnectError: [WinError 10054] ...
14:49:04  fetch of  failed: URLError: <urlopen error [WinError 10054] ...>
```

The fourth line is the one that matters. An empty `repo_id` is the *other*
transport — `_fetch_url`, used by exactly one registry row, `hdemucs_high`,
which fetches from `download.pytorch.org` and never touches Hugging Face. It is
reset too. **So this is not a Hugging Face problem**, and the theory that the
first two runs suggested is wrong. Multiple unrelated hosts reset the
connection, while the dependency packs downloaded from PyPI without trouble in
the same session.

### 2. The Xet transport is confirmed in play, and it is where the long attempt died

The best attempt of the day ran for about eight minutes — the idle-tick lines
show the fetch child holding 5.0, 4.6, 3.7 then 1.4 GiB between 15:11 and
15:18 — and then:

```
15:18:59  fetch of ilintar/trellis2-gguf failed: ConnectionError: Network error:
          Request middleware error: error sending request for url
          (https://huggingface.co/api/models/ilintar/trellis2-gguf/xet-read-token/a573...)
```

That is `hf_xet` re-requesting a read token part-way through a 16.1 GB
download, and failing. `hf-xet` is a dependency of `huggingface_hub` on Windows
AMD64, so it is in the base runtime, and nothing in this project sets
`HF_HUB_DISABLE_XET`. This is what "it got most of the way and then faulted at
the end" was.

Note that this run still went through Xet, so **the `HF_HUB_DISABLE_XET=1`
experiment has still not actually been observed taking effect** — the log ends
before any session that demonstrably had it set. It remains worth running, but
it can no longer be the whole story, because hdemucs does not use Xet and fails
anyway.

### 3. The real blocker: a failed fetch discards everything it downloaded

`fetch_one` stages into `.{name}.fetch.part` beside the destination and its
unwind is `except BaseException: shutil.rmtree(staging)`. The promise that
motivates it — that a failed download never leaves a half-populated *model*
directory — is about `dest`, and keeping the staging tree does not weaken it,
since every presence probe looks at `dest`. But `huggingface_hub` keeps its
resume bookkeeping in `.cache/` *inside* `local_dir`, which is the staging
tree, so deleting it throws away the resume state along with the bytes.

The consequence on this machine is decisive: eight minutes and several
gigabytes of a 16.1 GB download were discarded by one failed token request, and
the next attempt started from zero. **On a connection that resets every few
minutes, the engine can never be downloaded, however many times the user
presses Install.** That is why every retry in this log fails, and it is a
design property rather than a network condition — it would do the same to any
beta user on a hotel, mobile or otherwise imperfect line.

There is already precedent for keeping the tree: `_stage_only` deliberately
exempts the no-publish path from the same unwind, on the grounds that there the
staging tree is the product rather than a temporary.

### 4. The health check races a pack install

Five tracebacks in twenty-one seconds, 14:44:50 to 14:45:11, all from
`service.system.current_checks` → `doctor._vram_check` → `vram.probe()` →
`import torch`:

```
OSError: [WinError 126] ... Error loading "...\torch\lib\caffe2_nvrtc.dll"
        or one of its dependencies.            (x4, 14:44:50 - 14:45:05)
PermissionError: [WinError 32] The process cannot access the file because it is
        being used by another process. Error loading "...\torch\lib\shm.dll"
        (14:45:11)
```

This is the health poll importing `torch` while `pack_worker`'s pip is still
writing it into the running `site-packages`. It cleared after the app was
restarted at 14:46:42 and never recurred, so **torch is not broken on that
machine** — but the poll should not import a package that is actively being
installed, and it should not emit a traceback per tick while it happens.

### What the failures did not cost

Nothing was left half-installed and the app stayed usable throughout — the
staging-and-replace design held against real failures rather than stubbed ones.
The cost is the opposite one: it holds *too* firmly, and item 3 above is that
bill.

### And the message is a defect independent of all of it

The raw socket error reached a non-developer verbatim in a transient toast with
no log-file button. Everything diagnosed above came out of a log file she had
to be told how to find.

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
- Four findings opened against the tree and **all four closed the same day**:
  the fetch path's discard-on-failure (the blocker — the tree is kept and
  resumed into, with retries over it), the unreadable socket message (five
  remedies where there was one stringified exception), the health poll racing a
  pack install (any failed torch import falls back to NVML), and the pack gate
  that was never written (a mode's door asks for the pack first, then the
  weights). `TODO.md`'s *Open findings* has what each one changed.
- What is still owed from this machine is not code: a rerun that says whether a
  download now outlasts the resets, and the `HF_HUB_DISABLE_XET=1` experiment
  that no session in the log demonstrably had set.

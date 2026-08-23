# The app cannot see its own children — 2026-08-22

What `TODO.md` §8's three defects actually are, measured on a live session plus
three direct probes. One of the three is refuted; one is confirmed and worse
than the figure the app ships; one is confirmed with a smaller fix than §8
assumed.

## The session

Windows 11, 31.8 GiB card, **77.0 GiB commit limit**. `warlock.log`, one
reference generation followed by its sprite-sheet follow-up. Idle ticks are the
app's own `memlog.summary` line.

| time | event | app private | commit |
|---|---|---|---|
| 20:42:25 | idle at startup | 2.3 GiB | 37.8 (49%) |
| 20:43:25 | `sdxl_cfg` resident | 11.1 GiB | 46.3 (60%) |
| 20:43:41 | `unload()`, VRAM 6.73 → 0.01 | — | — |
| 20:43:55 | idle | **7.2 GiB** | 42.4 (55%) |
| 20:44:55 | `flux_klein_distilled` generated | **28.3 GiB** | 66.1 (86%) |
| 20:44:56 | `unload()`, VRAM 0.01 → 0.01 | **28.2 GiB** | 66.0 (86%) |
| 20:45:16 | sprite-sheet dispatch | 28.2 GiB, `children 0.0` | **72.3 (94%)** |
| 20:46:26 | idle | 28.2 GiB, `children 0.0` | 73.0 (95%) |
| 20:47:26 | idle | 28.2 GiB, `children 0.0` | 73.5 (95%) |

It ended in the app refusing its own follow-up, correctly:

```
RuntimeError: host memory is 94% committed before loading SDXL 1.0
  (full CFG, structural control), at or past the 90% ceiling.
```

## D1 — confirmed, and the shipped figure understates it

`sdxl_cfg` cost +8.8 GiB of private commit and gave **4.9 GiB back**. Klein cost
**+21.1 GiB** and gave **0.1 GiB** back. The VRAM line klein logs (`0.01 -> 0.01`)
is true and irrelevant: an OFFLOAD spec's weights live in host memory by
definition, which is the whole reason `vram_gib` is 10 rather than 16.

`models.py` declares `host_peak_gib=16.0` for both klein entries. The measured
charge on this machine is **21.1 GiB**, and it is not a peak — it is a floor
that persists until the process exits.

## D2 — confirmed; the fix is one function, not four spawn sites

`memlog.summary` omits the children clause entirely when the reading is falsy
(`if kids:`), so the `children 0.0 GiB` in the table above is not "no children".
It is a **non-zero reading that rounded to zero** — the ~0.8 MB uv trampoline —
printed while a BiRefNet worker held ~6.3 GiB. `_q_sprite.py:331` runs the heavy
matte immediately before `_acquire_t2i` at line 345, which is exactly where that
6.3 GiB of commit appears in the table.

`sys.executable` under this venv is a trampoline, not an interpreter:

```
sys.executable : D:\Projects\Warlock\.venv\Scripts\python.exe
Popen.pid      : 8632
child says pid : 4256      <- not the same process
```

§8 proposed recording the real pid at every worker spawn. That is not needed.
The kill-on-close job **already holds the whole tree**, so the job itself is the
register — provided the child is assigned before it spawns its own child, which
is the ordering every spawn site already uses:

| ordering | `JobObjectBasicProcessIdList` |
|---|---|
| `Popen` → `sleep(2)` → `assign` | `[19780]` — grandchild missed |
| `Popen` → `assign` (what the code does) | `[30104, 35204, 31048]` — whole tree |

So the reading is a `QueryInformationJobObject` call, and no spawn site changes.
It is also indifferent to whether a trampoline exists at all, which §7's
installer layout requires (`sys.executable` becomes a real `python.exe` there,
and the job then holds exactly one pid).

## D3 — refuted

§8 predicted that `matting.unload()`'s `proc.kill()` would orphan the
grandchild, because Windows `TerminateProcess` does not cascade. Measured
directly — a 200 MB grandchild behind the trampoline, killed the way
`matting.unload()` kills it:

```
trampoline pid : 26124
real pid       : 5088
distinct?      : True
both alive     : True True
--- proc.kill() on the trampoline ---
t+ 0.5s  trampoline alive: False   real interpreter alive: False
t+ 1.5s  trampoline alive: False   real interpreter alive: False
t+ 3.0s  trampoline alive: False   real interpreter alive: False
```

`TerminateProcess` indeed does not cascade; the uv trampoline does the
cascading itself, holding its child in a kill-on-close job of its own. The
premise was right and the conclusion was wrong. **`matting.unload()` returns
the memory it claims to.** No fix is owed, and the reasoning is recorded here
so the refutation is not re-derived from the same true premise.

The caveat that survives: this is a property of *uv's* trampoline, not of
Windows. A worker spawned through some other shim would orphan exactly as §8
described, so the guarantee to rely on remains the job object, not the kill.

## D1 — built, and measured on both sides

The pipeline moved into `pipelines/text2image_worker.py`, held from the app by
`pipelines/t2i_client.Text2ImageClient`. Same machine, same checkpoint, same
sequence — construct, `load()`, one 1024x1024 sample, `unload()` — with the app's
own accounting (`process_memory().private` plus `children_private` over
`winjob.measured_pids()`, so the child is counted):

| path | charged by the load | returned by `unload()` | kept |
|---|---|---|---|
| in process (`WARLOCK_T2I_IN_PROCESS=1`) | 24.26 GiB | 0.26 GiB | **24.01 GiB** |
| child (the default) | 24.08 GiB | 24.08 GiB | **0.00 GiB** |

System commit tells the same story from outside: 37.19 → 61.30 → **37.21 GiB**.

Two things worth keeping from that table. **A bare load is not the case that
breaks** — load-then-unload with no sample kept only 2.8 GiB; it is the sample
that strands the arenas, so any future measurement of this has to generate.
And **the shipped `host_peak_gib` was wrong in the other direction too**: klein
declared 16.0, being the checkpoint's size, while the load reached 18.3 and the
sample took it to 24.1. Both klein entries now say 24.0.

## The deadlock that the child arrangement walked into

Worth recording because it cost more than the rest of the change and because
nothing about it is obvious.

The worker needs a *concurrent* stdin reader — a cancel has to be read while a
generate is running — where `matting_worker` reads on its main loop and never
blocks during an import. With a reader thread parked on the inherited stdin
pipe, the main thread's next native-extension import never returns:

```
[   5.0s] CHILD | PUMP: got '{"op": "load"}'
[   5.0s] CHILD | PIPE: constructed
[  45.1s] CHILD | Timeout (0:00:45)!
          Thread 0x24c8 (most recent call first):
            File "<frozen importlib._bootstrap_external>", line 1317 in create_module
            File "numpy\_core\multiarray.py", line 11 in <module>
```

Reduced to a 20-line reproduction, `import numpy` takes **0.1 s** with no such
thread and **never completes** with one. The cause is a *pending read on the
pipe*, not threads and not Python's IO layer:

| second thread | `import numpy` |
|---|---|
| none | 0.1 s |
| `threading.Event().wait()` | 0.1 s |
| blocking read on a *regular file* | 0.1 s |
| `for line in sys.stdin` | never |
| `sys.stdin.readline()` | never |
| `sys.stdin.buffer.readline()` | never |
| `os.read(0, ...)` | never |

Every blocking flavour fails alike, including the raw `os.read`, which puts it
below Python — in the interaction between an outstanding synchronous pipe read
and the Windows image loader.

The fix is `_lines_from`: `PeekNamedPipe` first, `os.read` only what is already
there, sleep 50 ms when the pipe is empty. No read is ever left outstanding for
an import to trip over. Verified on the same reproduction — numpy and torch
import at full speed *and* a line written during the import still arrives — and
both directions are pinned by tests, including one that fails if the underlying
interaction is ever fixed and the polling becomes unnecessary.

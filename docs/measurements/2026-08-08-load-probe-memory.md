# A load check cannot give its memory back — 2026-08-08

Why `pipelines/loadprobe.py` is a child process rather than four lines in
`doctor.py`.

## The question

NEXT_ROADMAP Phase 2's N112 asks for a one-time deferred load check on the host
matting model and the pose model, so their doctor rows stop saying *"weights
present — **not checked**: whether the model loads"*. A green row above a
pipeline that silently falls back is the one outcome those rows exist to
prevent, and only an attempted load settles it.

The obvious implementation calls `from_pretrained` on the CPU, once, on the
health-poll thread, and leaves the model in the module cache on the reasoning
that the load is not wasted — the first real export would find it resident.

## The measurement

Windows 11, Python 3.13, `psutil.Process().memory_info().rss`, one process per
run. `WARLOCK_T2I_ROOT` at its default.

| Step | RSS | Delta |
|---|---|---|
| baseline (warlock imported, no torch) | 23 MB | — |
| `import torch` | 486 MB | +463 |
| `matting._load(path, "cpu")` (BiRefNet) | 1961 MB | **+1475** |
| `pose2d._load(path)` (ViTPose base) | 1969 MB | +8 |

Then the release, in a fresh process:

| Step | Delta from before the load |
|---|---|
| after `probe()` returns, cache entry popped | **+1053 MB** |
| after `gc.collect()` | +1053 MB |

## What that says

**The memory does not come back.** Dropping every reference and collecting
recovers 422 MB of the 1475; the remaining 1053 MB is held by the allocator's
arenas, not by a live object, and no amount of collecting returns it. So an
in-process load check is a gigabyte of RSS spent on **every launch, for the life
of the process**, to make one diagnostic row say `loads` — on a user who may
never do a 2D export at all.

Warlock's worst crash to date is host-commit exhaustion (2026-08-03), and
`Worker._check_resources` refuses jobs past `memlog`'s commit ceiling. Spending
a gigabyte by default to answer a question is the wrong side of that trade.

A second finding, incidental and worth writing down: the in-process version also
imported torch into the app process, which is exactly what C29 moved *off* the
startup path. It would have arrived by a different door on the first health
poll.

**ViTPose is not the problem.** At 8 MB it could stay in process. It is spawned
the same way anyway, because "a diagnostic does not keep a working set it
created" is a rule rather than a size threshold, and one mechanism with one set
of failure modes beats two.

## The decision

The probe runs as `python -m warlock.pipelines.loadprobe <kind> <path>`, through
`winjob.run` so it dies with the app, and prints one line. Verified: the parent
grows **0 MB** and `torch` is never in `sys.modules` in the parent afterwards.

This is `doctor._probe_blender`'s shape and it is the same argument the repo
already makes twice — `rigging.py` keeps `bpy` out of the app process because it
is process-global and cannot be undone, and `fetch_worker` keeps
`HF_HUB_OFFLINE=0` out of it for the same reason. A cost that cannot be undone
in this process is paid in one that ends.

The cost paid instead is a subprocess spawn plus a torch import, once, on a
background thread — the same order as the bpy probe, which has a 120 s timeout.
`LOAD_PROBE_TIMEOUT` is 300 s, wider because two checkpoints of unknown size are
read from disk and a cold NVMe cache on a first launch is real.

## What would change this

Nothing about the checkpoint sizes. If BiRefNet is ever loaded on the job path
by default — it is not; `mask()` loads it on first use and `unload()` releases
it — the probe could ask whether it is already resident and skip the child. That
is the `resident` check the in-process version had, and it is not worth carrying
without a caller that makes it true.

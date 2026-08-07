# Directory mtime granularity, 2026-08-07

A measurement taken because a constant needed a number rather than a taste, and
because the prose it replaces asserted something that turned out to be false on
the only platform this app runs on.

`service.files.attach_files` caches each library row's file list under the stamp
`(status, the job directory's mtime)`. Its docstring justified the second half
with "a file appearing or being removed adds or removes a directory entry, which
is exactly what moves a directory's mtime", and `CLAUDE.md` repeated it. That is
true about the *event* and says nothing about the **resolution**, which is what
the cache actually depends on.

The suspicion came from a flake, not from reading: `tests/test_studio_frame.py::
test_a_new_artifact_on_disk_is_noticed_without_a_status_change` fails roughly one
run in ten, in isolation, and it does nothing but write a file and look for it.

## What was run

200 trials, each making a fresh directory, reading its mtime, adding one file,
and reading the mtime again. Windows 11 Pro 26200, NTFS, local SSD, under
`uv run python` (3.13) with nothing raising the system timer resolution.

```python
for i in range(200):
    d = root / f"d{i}"
    d.mkdir()
    before = d.stat().st_mtime_ns
    (d / "thumb.png").write_bytes(b"x")
    after = d.stat().st_mtime_ns
```

## Result

```
directory mtime unchanged after adding a file: 155/200 times
when it changed, delta ms: min=0.993 median=1.002
```

**Adding a file left the directory's mtime unchanged 78% of the time**, and the
smallest observed change was ~1 ms — the granularity of the clock the mtime is
written from, not of NTFS's 100 ns timestamp field. Windows' default system-clock
tick is 15.6 ms and is only finer while some process has requested it, so 1 ms is
the *favourable* case here, not the worst one.

## What follows

The cache can be permanently wrong, not briefly wrong. If a listing runs and a
write lands afterwards but still inside the stamped mtime's tick, every later
comparison keeps matching a stamp whose answer is stale — until the job's status
changes or another write happens to cross a tick boundary. The case this bites is
precisely the one the mtime half of the stamp was added for: a rig is written into
the **source** job's directory while that job stays `done`.

In the running app the exposure is small — a ~1–16 ms window against a 500 ms poll
— but the effect does not decay, and the fix is cheaper than the reasoning about
how often it matters.

## The constant

`files.MTIME_RACE_NS = 50_000_000` (50 ms). A stamp is stored only once its mtime
is older than that; a directory touched more recently is answered correctly and
simply not remembered, so the next tick asks the disk again. This is git's
racily-clean rule, for the same reason git needs it.

Sized from the measurement rather than from the observed 1 ms: the bound that
matters is the system clock tick, worst case 15.6 ms, and 50 ms clears it about
three times over while still being a tenth of the refresh interval — so a written
directory costs at most one extra listing, on one row.

The clock is read **after** the listing, not before, and that ordering is the
proof rather than a detail. The hazard requires a write later than our listing yet
still inside the mtime's tick; if the listing itself finished more than one tick
after the mtime, no such write can exist.

## Cross-check

With the guard in place the flaky test passed 15/15 consecutive runs, where it had
been failing about 1 in 10. Two further tests pin the rule directly by controlling
the mtime with `os.utime` instead of hoping for a race:
`test_a_directory_touched_moments_ago_is_answered_but_not_remembered` and
`test_a_write_that_never_moved_the_mtime_is_still_noticed`.

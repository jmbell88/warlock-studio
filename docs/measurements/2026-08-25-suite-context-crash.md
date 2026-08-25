# The suite's "one imgui context at a time" rule was upheld by luck

**Date:** 2026-08-25
**Tree:** `bd41c75` (v0.0.28) plus the pane-guard work in progress
**Question:** the hardening pass asked "what is flaky?" — this is what the first
hour of asking found, and it was not flaky, it was a native crash waiting for a
different deal of the cards.

## The finding

On **unmodified master**, in one process:

```
$ uv run pytest -n 0 tests/test_studio_smoke.py tests/test_studio_controls.py
...........................Windows fatal exception: access violation

Current thread 0x0000125c (most recent call first):
  File "D:\Projects\Warlock\tests\test_studio_controls.py", line 163
      in test_component_gallery_builds_every_state
```

Not a failure — a process death. Neither file crashes alone
(`test_studio_smoke.py` 157 passed; `test_studio_controls.py` 29 passed).

## Why it never fired

`tests/test_studio_smoke.py:62` declared its imgui context **`scope="session"`**.
Only that file uses the fixture, so the scope buys nothing — but a session-scoped
context is still alive after the file's last test, and the next file in that xdist
worker that builds its own then has **two imgui contexts over the one GL context**.
`tests/test_section_blocks.py`'s fixture docstring already names that exact
combination as an access violation, and the four files that build their own
context all carefully save and restore `prev_ctx` because of it. What none of them
could do is stop the *other* context existing.

The default run puts eight workers over the files with `--dist loadfile`, and
those two files happened to land in different workers. That is not an invariant.
It is a hash of the file list — so **adding one unrelated test file re-deals the
hand**. Adding `tests/test_pane_guard.py` did exactly that, and the first symptom
was a failure in `test_component_gallery_builds_every_state[1.0-light]`, a test
that has nothing to do with pane guards and was not modified.

That is the shape worth recording. The suite has two well-known cross-test
hazards already written down — `vram._published` (a module global nothing reset)
and the `imgui.ini` collapsed-flag artefact — and both had the same signature:
invisible under the default run, deterministic once the two participants met.
This is the third, and it is the only one that killed the interpreter.

## The fix

`scope="session"` → `scope="module"`. Since the fixture is file-local the two
scopes cost the same, and module scope destroys the context when the file ends,
so no later file can overlap it.

```
$ uv run pytest -n 0 tests/test_studio_smoke.py tests/test_studio_controls.py
186 passed in 13.53s
```

## A second leak found in the same fixture

The same fixture set `widgets.FORCE_SECTIONS_OPEN = True` and never put it back.
Session-scoped, that meant every later test in the worker drew its collapsing
sections forced open — the `vram._published` shape exactly, and invisible for the
same reason: the tests that would notice are in other files, which
`--dist loadfile` usually keeps elsewhere. It is restored on the way out now, as
is `type(gl).screen`.

## What this says about the next one

Neither of these was found by reading the code; both were found by *changing the
worker deal* and watching what broke somewhere unrelated. The suite's isolation
fixtures are excellent where they exist (`tests/conftest.py` pins host memory,
device memory and the published VRAM reading, each with the incident that
motivated it written down). The gap is that nothing enumerates the module globals
that have no such pin — there are 58 `global` statements across 25 modules in
`src/warlock`, and the two leaks above were both in *test* code rather than in
those.

The cheap continuing check is the one that found this: run the suite at more than
one worker count. A file→worker assignment that changes is the only thing that
makes this class visible at all.

# Tablet pressure, 2026-08-15

A spike rather than a measurement of a constant: Inker's brush has a
velocity-driven taper and no pressure input, and the question is whether a real
pen's pressure can reach `StrokeState` on this stack at a cost proportionate to
what it buys. The plan that raised it (phase Q, item b) named two routes — SDL3
pen events, or Windows Ink through a `WndProc` subclass — and explicitly allowed
the answer "the velocity taper stands". This document is the evidence for that
answer.

Everything below was run offline, in this worktree's venv, on the machine the app
is developed on: Windows 11 Pro 26200.

## What was inspected

### 1. The SDL under pygame

```
pygame-ce 2.5.7 (SDL 2.32.10, Python 3.13.13)
pygame.get_sdl_version() -> (2, 32, 10)
```

`pyproject.toml` pins `pygame-ce>=2.5`; the lockfile resolves 2.5.7. Searching
the whole `pygame` namespace for a pen constant:

```python
[n for n in dir(pygame) if 'PEN' in n.upper()]   # -> ['OPENGL', 'OPENGLBLIT']
[n for n in dir(pygame.locals) if 'PEN' in n]    # -> ['OPENGL', 'OPENGLBLIT']
```

Two substring accidents and no pen API. There is no `SDL3.dll` under `pygame/`;
the only SDL3 binary anywhere in the venv belongs to `bpy`, which by invariant
never loads in the app process (`../INVARIANTS.md`).

### 2. The vendored SDL2 binary's exports

`.venv/Lib/site-packages/pygame/SDL2.dll` parsed directly (PE32+, export
directory walked in python — no external tool):

```
dll name SDL2.dll
named exports: 845
matching /Pen|Tablet|Stylus/: ['SDL_IsTablet']
```

`SDL_IsTablet` answers "is this computer a tablet-form-factor device"; it says
nothing about a pen. Probing for the SDL3 pen entry points by name:

```
SDL_GetPens            False
SDL_PenConnected       False
SDL_GetPenStatus       False
SDL_GetPenCapabilities False
```

**The pressure signal does not exist anywhere in the binary the app links.** Any
route that keeps this wheel has to go round SDL, not through it.

### 3. What Win32 offers, and whether SDL2 will let us reach it

`user32.dll` on this host exports the whole modern pointer surface:

```
GetPointerPenInfo True   GetPointerInfo True   GetPointerType True
EnableMouseInPointer True   GetPointerPenInfoHistory True
SkipPointerFrameMessages True   SetWindowLongPtrW True   CallWindowProcW True
```

More usefully, the vendored SDL2 exports `SDL_SetWindowsMessageHook` — which the
plan did not consider, and which is the same route as a `WndProc` subclass
without the subclassing. It was installed from ctypes against pygame's own
`SDL2.dll`, on a real (hidden) pygame window, and it works:

```python
HOOK = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_uint, c_uint64, c_int64)
sdl.SDL_SetWindowsMessageHook(hook, None)
pygame.display.set_mode((320, 240), pygame.HIDDEN)
for _ in range(5):
    user32.PostMessageW(hwnd, WM_APP + 7, 0, 0)
pygame.event.pump()
```

```
wm_info keys: ['hdc', 'hinstance', 'window']
distinct messages seen: 2, total: 6
WM_APP+7 (0x8007) count: 5
top: [('0x8007', 5), ('0x31f', 1)]
```

All five posted messages arrived at the python callback, plus one real system
message (`0x31F`, `WM_DWMNCRENDERINGCHANGED`). No `SetWindowLongPtrW`, no
`CallWindowProcW`, nothing taken away from SDL's own window procedure.

Its cost, since the hook fires for *every* window message and runs inline on the
thread that pumps — which here is the frame loop (`studio/main.py:1602`):

```
9000 messages, pump timed with perf_counter
  no hook:     13.07 ms   ->  1.452 us/message
  ctypes hook: 14.52 ms   ->  1.613 us/message
```

**~0.16 µs of overhead per window message.** At a tablet's ~200 Hz report rate
that is ~32 µs per second of drawing, against a 16.7 ms frame budget. The hook
is not the expensive part of this idea.

(The first attempt posted 20 000 messages and the hook saw 10 000 — Windows'
default per-queue message limit, not a dropped callback. The run above stays
under it.)

### 4. Whether a pen can be tested at all on this machine

```
SM_DIGITIZER   0x0     SM_PENWINDOWS 0x0     SM_MAXIMUMTOUCHES 0
GetPointerDevices -> ok, count = 0
IsMouseInPointerEnabled -> False
ctypes.WinDLL('wintab32') -> FileNotFoundError
```

No digitizer, no pointer devices, no Wacom Wintab driver. **Nothing on this host
can produce a pressure sample**, so every claim about what a pen's messages look
like — whether `WM_POINTERUPDATE` reaches a window SDL2 has already registered
for touch, whether the promoted legacy mouse message stays in step with the
pointer frame, what `POINTER_PEN_INFO.pressure` reads at rest — would be
recited, not measured. That is the disqualifying fact for landing code today.

### 5. What the input path can carry, before pressure is even discussed

Two facts about the existing path, both measured.

**imgui already collapses sub-frame motion.** `studio/imgui_backend.py:379-381`
forwards every `pygame.MOUSEMOTION` to `io.add_mouse_pos_event`; the canvas reads
`imgui.get_mouse_pos()` exactly once per frame (`inker_canvas.py:467`) and calls
`doc.stroke_to` at most once from it (`inker_canvas.py:1468`). Five position
events queued inside one frame:

```
after 5 pos events in one frame: (140.0, 200.0)   # the fifth; 100..130 are gone
next frame:                      (500.0, 500.0)
```

**And the frame loop is capped at 60 Hz** (`main.py:68`, `TARGET_FPS = 60`;
`IDLE_FPS = 12` when nothing is live). A pen reporting at 133–240 Hz therefore
has more than half its samples discarded by the architecture *before* any
pressure question arises. Whatever route delivered pressure would deliver one
scalar per frame, per stroke segment — which is exactly the resolution the
velocity taper already runs at.

There is one encouraging detail: imgui 1.92.8 (via `imgui_bundle` 1.92.801)
already carries the carrier we would need. `imgui.IO` has `pen_pressure`
("Touch/Pen pressure (0.0 to 1.0 ...). Helper storage currently unused by Dear
ImGui"), and `add_mouse_source_event` with a `MouseSource` enum whose members are
`['mouse', 'touch_screen', 'pen', 'count']`. Verified settable and frame-stable:

```
io.pen_pressure default: 0.0
io.pen_pressure settable: 0.41999998688697815
io.pen_pressure survives new_frame: 0.41999998688697815
```

So the *transport* between a backend and a pane is already built and needs no
new state on `inker_state`.

## The engine half is not as free as the plan assumed

The plan's phrase is "engine half is nearly free (`StrokeState.to()` optional
pressure → existing `_taper`)". Reading the code, that is three-quarters true and
the missing quarter is the interesting part.

`StrokeState.to` (`brush.py:439`) computes a speed and hands it to `_advance`,
which calls `_taper` once per segment (`brush.py:477`) and uses the result as the
diameter for every dab in that segment. Adding `pressure: float | None = None` to
`to` and `_advance` is indeed two defaulted parameters. But:

1. **`_taper` early-returns.** `if self.speed_taper <= 0.0: return self.diameter`
   (`brush.py:495`). A pressure value must bypass that, so the function changes
   shape rather than gaining an argument — and `speed_taper` is 0.0 by default,
   which is the case pressure most wants to work in.
2. **The pixel nibs never reach `_taper` at all.** `_advance` returns from its
   `if self.pixel:` branch (`brush.py:464-471`) before the width is computed:
   a pixel walk stamps whole pixels along the line and has no per-segment
   diameter. Pressure→size would therefore silently do nothing on `pixel` and
   `pixel_hard` — the two nibs an Aseprite-parity user reaches for first. That is
   not a defaulted parameter; it is a second width mechanism.
3. **Pressure→opacity is a different axis.** `_stamp` reads `self.opacity`
   (`brush.py:602`) once per dab off the dataclass. Aseprite's dynamics offer
   pressure→size *and* pressure→opacity; the second one is not reachable by
   widening `to`.
4. **The smoothing constant is velocity-shaped.** `TAPER_SMOOTHING = 0.6`
   (`brush.py:97`) exists because a single fast frame otherwise drops one thin
   dab into a thick stroke. Pressure is not noisy in that way; running it through
   the same lag would add latency to a signal that does not need it, and running
   it around the lag means a second path through `_taper`.
5. **Every file involved is a merge hotspot.** `brush.py`, `_doc_paint.py` and
   `inker_canvas.py` are being edited by concurrent tracks this week. Landing
   parameters that no caller can pass a non-`None` value to, and that no test can
   exercise against real hardware, spends conflict budget on dead code.

None of these is hard. Together they are a design decision about brush dynamics,
not a parameter default — and the decision cannot be judged without a pen to
draw with.

## Routes, with their real cost here

**A. Move to SDL3 pen events.** Cost: replacing the SDL the whole app is built
on. The window, the GL context, the event pump, the IME `TEXTINPUT`/`TEXTEDITING`
split, `DROPFILE`, `WINDOWDISPLAYCHANGED`/`WINDOWMOVED` DPI resampling and the
imgui backend's entire event mapping are SDL2-shaped and all live on the one
frame loop. It also needs a wheel this project does not have and cannot fetch
offline. Disproportionate to one brush input.

**B. Windows pointer input through `SDL_SetWindowsMessageHook`.** The cheapest
real route, and cheaper than the `WndProc` subclass the plan named: measured
installable, measured at ~0.16 µs/message, no ownership of SDL's window
procedure and therefore nothing to restore on teardown or to get wrong on a
`set_mode` re-create (`main.py:1612` re-creates the window on every resize — a
subclass would have to be re-applied there; a hook would not). It stays inside
the invariants: no subprocess, so the `winjob` rule is untouched; it would live
in `studio/` beside `imgui_backend.py`, so the headless `studio/inker/` import
pin is untouched; the callback runs on the frame loop, which is where pygame
events are already read.
Its risks are the ones this machine cannot retire: whether pen `WM_POINTER`
messages reach a window SDL2 has already claimed for touch, whether the promoted
legacy mouse position and the pointer frame agree, and what happens on the
palm-rejection and barrel-button paths. Plus one hard edge — `EnableMouseInPointer`
is process-wide and irreversible once set, so it must **not** be used; the design
has to read pen messages that arrive anyway and leave SDL's mouse handling alone.

**C. Wintab.** A vendor DLL that is not installed here, a third input stack, and
Wacom-only. Rejected.

## Conclusion

**The velocity taper stands. No code lands from this spike.**

The stack cannot supply pressure without either swapping SDL or hand-rolling
Win32 pointer handling; the machine the app is developed on has no digitizer, so
neither could be verified rather than asserted; the input path throws away more
than half of a pen's samples before pressure is considered; and the engine-side
change is a brush-dynamics decision, not the two defaulted parameters the plan
priced it as. The existing `speed_taper` already produces a pen-like taper from a
signal the app definitely has, at exactly the resolution a pressure signal would
arrive at.

Nothing in `brush.py`, `_doc_paint.py` or `inker_canvas.py` was modified.

## If it is revisited

Route B, in this order, and only with a pen on the desk:

1. A `studio/pen.py` that installs `SDL_SetWindowsMessageHook` at window
   creation, watches `WM_POINTERUPDATE`/`WM_POINTERDOWN`/`WM_POINTERUP`, calls
   `GetPointerPenInfo`, and latches the last pressure as a float. No
   `EnableMouseInPointer`. One module, frame-loop-only, `studio/`-side of the
   import pin.
2. The latch published as `io.pen_pressure` from `imgui_backend.process_event`,
   with `add_mouse_source_event(MouseSource.pen)` alongside it — both already
   exist in the installed imgui and need no new `inker_state` field.
3. `inker_canvas._input` reads `io.pen_pressure` in the same place it reads
   `get_mouse_pos`, and passes it to `doc.stroke_to`.
4. Only then the engine question: whether pressure replaces or multiplies the
   velocity taper, whether it also drives opacity, and what the pixel nibs do
   with it — each of which wants a drawing to judge, and one of which
   (`_advance`'s pixel branch) needs a width mechanism that does not exist yet.

The first three steps are small and are the part that is now measured. The fourth
is the one this spike declines to guess at.

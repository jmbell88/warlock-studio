"""The one mtime-keyed cache rule the panes share.

Not a pane: no ``draw``, no imgui, nothing on screen. It lives here because two
panes need it and :mod:`.inspector` already imports :mod:`.sheet_panel`, so it
could not live in either of them without a cycle.

The rule is ``service.files.attach_files``' and the constant is its
``MTIME_RACE_NS``, imported rather than restated -- one judgement about one
class of hazard. It is stated once more here because every copy of it that was
missing was missing for the same reason: it looks unnecessary.

A file's (or a directory's) mtime on Windows is written from the system clock,
whose tick is 15.6 ms unless something has asked for better -- adding a file
left a directory's mtime unchanged 155 times in 200 on this machine (see
``docs/measurements/2026-08-07-directory-mtime-granularity.md``). So a write
landing *after* a read but still inside the stamped mtime's own tick is
invisible to that stamp, and invisible **forever**, because every later
comparison keeps matching the stale answer -- which is exactly the case these
caches exist for: a re-rig, a re-derived pixel sheet, a palette dropped into
the directory while the app is running. Git calls this racily-clean and answers
it the same way: a stamp is remembered only once its mtime is comfortably in
the past, so a thing touched moments ago is answered correctly and simply not
remembered, at the cost of re-reading it for another 50 ms.

The clock is read *after* whatever was read, and that ordering is the proof
rather than a detail: the hazard needs a write later than the read yet still
inside the mtime's tick, which cannot exist if the read already finished a tick
after the mtime. So :func:`storable` is called once the value has been
computed, never beside :func:`stamp_ns`.
"""

from __future__ import annotations

import time
from pathlib import Path

from ...service.files import MTIME_RACE_NS


def stamp_ns(path: Path) -> int | None:
    """``path``'s mtime in nanoseconds, or None when it is not there.

    None is a perfectly good stamp of its own -- "not there" is an answer about
    that path exactly as a number is, and it changes the moment the path
    appears.
    """
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def storable(stamp: int | None) -> bool:
    """Whether ``stamp`` may be *remembered*, per the rule in the module doc.

    None races with nothing and is always storable: a path appearing turns the
    stamp from None into a number whatever the clock happened to be doing. A
    backwards clock step makes the difference negative, which declines to cache
    -- slower, never wrong, which is the right way round.
    """
    return stamp is None or time.time_ns() - stamp > MTIME_RACE_NS

"""Reproduction harness for the stdin-reader / DLL-load deadlock.

Runs `text2image_worker._lines_from` on a daemon thread, then imports a native
extension on the main thread -- the exact arrangement that hung the worker
before the reader learned to peek. Prints one line per stage so the parent can
tell "finished" from "never got there".

Argument `--blocking` swaps in the naive `for line in stdin` reader, which is
what the fix replaced; it exists so the test can assert the reproduction still
reproduces, rather than trusting that the guard is guarding something.
"""

from __future__ import annotations

import sys
import threading
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from warlock.pipelines import text2image_worker as worker  # noqa: E402

BLOCKING = "--blocking" in sys.argv
seen: list[str] = []


def _pump() -> None:
    source = sys.stdin if BLOCKING else worker._lines_from(sys.stdin)
    for line in source:
        seen.append(line.strip())


threading.Thread(target=_pump, daemon=True).start()
time.sleep(0.3)

sys.stdout.write("START\n")
sys.stdout.flush()

import numpy  # noqa: E402, F401

sys.stdout.write("IMPORTED\n")
sys.stdout.flush()

# Give a line written during the import time to be picked up, then report it:
# the reader has to still work, not merely fail to deadlock.
deadline = time.monotonic() + 5.0
while not seen and time.monotonic() < deadline:
    time.sleep(0.05)
sys.stdout.write("SEEN {}\n".format(",".join(seen)))
sys.stdout.flush()

"""Does this checkpoint actually load? Answered in a child process (N112).

The doctor rows for the host matting model and the pose model used to say
"weights present -- **not checked**: whether the model loads", which is honest
and is the wrong amount of honesty: a green row above a pipeline silently
falling back is the one outcome those rows exist to prevent, and the only thing
that settles it is an attempted load.

**It runs out of process, and the reason is measured rather than assumed.**
Loading BiRefNet on the CPU costs **1475 MB** of RSS on this machine
(``docs/measurements/2026-08-08-load-probe-memory.md``), and the memory does not
come back: dropping every reference and calling ``gc.collect()`` leaves
1053 MB resident, because what is holding it is the allocator's arenas
rather than a live object. So an in-process check is a gigabyte spent on every
launch, for the life of the process, to make one row say "loads" -- on a user
who may never do a 2D export. Warlock's worst crash to date is host-commit
exhaustion; that is exactly the wrong trade.

A child pays the same gigabyte and then dies with it. This is
``doctor._probe_blender``'s shape, for ``doctor._probe_blender``'s reason -- and
the same one the module docstring in ``rigging.py`` gives for Blender and
``fetch_worker`` gives for ``HF_HUB_OFFLINE``: a cost or a global that cannot be
undone in this process is paid in one that ends.

The child prints a single line: ``ok <detail>`` or ``fail <detail>``. Nothing
structured, because there is nothing structured to say and a JSON contract
between a module and itself is a format to keep in agreement for no reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The probe kinds, and the module that owns each. Not a getattr on a
# caller-supplied name: this runs as ``-m`` with argv from a parent, so the set
# of things it will import is closed and written down.
KINDS = ("matting", "pose")


def probe(kind: str, path: Path) -> tuple[bool, str]:
    """Load the checkpoint at ``path``. -> (did it, why not).

    Never raises: every failure is a sentence. A probe that raises out of the
    child turns a red row into a traceback in a log nobody is reading yet.
    """
    try:
        if kind == "pose":
            from . import pose2d

            pose2d._load(path)
        else:
            from . import matting

            matting._load(path, "cpu")
    except Exception as exc:  # noqa: BLE001 -- the whole point is to report it
        # The type as well as the message: "No module named 'einops'" and a
        # shape mismatch are two entirely different repairs, and a bare str()
        # of an ImportError names neither.
        return False, f"{type(exc).__name__}: {exc}"
    return True, "loads"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in KINDS:
        print(f"fail usage: -m warlock.pipelines.loadprobe {'|'.join(KINDS)} <path>")
        return 2
    ok, detail = probe(args[0], Path(args[1]))
    print(f"{'ok' if ok else 'fail'} {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run Warlock from a console interpreter or an installer-owned pythonw."""

from __future__ import annotations

import importlib
import os
import sys
from typing import TextIO


def _stdio(stream: TextIO | None) -> TextIO:
    return stream if stream is not None else open(os.devnull, "w", encoding="utf-8")


sys.stdout = _stdio(sys.stdout)
sys.stderr = _stdio(sys.stderr)

raise SystemExit(importlib.import_module(".cli", __package__).main())

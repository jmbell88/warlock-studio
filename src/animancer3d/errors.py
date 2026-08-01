"""Turn raw exceptions into a sentence a user can act on.

The full traceback always goes to the job's error.log; only the short,
friendly sentence goes in the DB and the UI.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import httpx


def friendly(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "GPU out of memory — try resolution 512, or close other GPU apps."
    if isinstance(exc, httpx.TransportError):
        return "The 3D engine stopped unexpectedly. See assets/trellis.log."
    return str(exc) or exc.__class__.__name__


def write_error_log(job_dir: Path, exc: Exception) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "error.log").write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )

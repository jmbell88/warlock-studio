"""Offline sweep of trellis-server's --band flag against measured mesh quality.

The exe defaults the DC remesh band to res/512, which leaves the surface as a
crust of disconnected plates with see-through gaps between them. Widening the
band thickens the crust so the plates meet -- but only up to a point, and the
cost is geometry. This picks the value empirically: one generation per band on
a fixed image and seed, each scored by meshaudit.hole_fraction.

Two things force this to be a standalone command rather than a config knob.
--band is a *launch* flag (pipelines/trellis.py builds it into argv; the
per-request POST carries only image, seed and resolution), so changing it means
restarting the server -- which is exactly what this does, once per value.
And the run needs the real GPU, the vendored exe and the GGUF weights, so it
cannot be part of the test suite.

Deliberately bypasses Worker, JobStore and the HTTP app: nothing here should be
able to disturb queued jobs or leave rows behind. It must own the trellis port
outright -- see require_free_port -- so run it with the app stopped.

    warlock sweep --image assets/<job-id>/input.png --bands auto,2,4,8,16

Then set config.DEFAULT_TRELLIS_BAND to the winner.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from . import doctor
from .config import Config
from .pipelines.trellis import TrellisServer

log = logging.getLogger(__name__)

# The default ladder. "auto" (band=None, the exe's own res/512 heuristic) is
# always worth including as the baseline the numbers are being compared to.
# Note the heuristic *is* one of the explicit values -- at res 1024 it resolves
# to 2 -- so a ladder covering that value measures the baseline twice, which is
# a free read on run-to-run variance.
DEFAULT_BANDS = "auto,2,4,8,16"

# How much better than "auto" a band must measure before it is worth
# hard-coding. Deliberately coarse: the sweep runs each value once, and a
# single run cannot resolve anything finer.
MEANINGFUL_MARGIN = 0.20


def parse_bands(raw: str) -> list[int | None]:
    """"auto,2,4" -> [None, 2, 4], preserving order and dropping duplicates."""
    bands: list[int | None] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        value = None if part in ("auto", "none", "default") else int(part)
        if value not in bands:
            bands.append(value)
    if not bands:
        raise ValueError("no band values given")
    return bands


async def run_band(
    config: Config,
    band: int | None,
    image: Path,
    out_dir: Path,
    *,
    seed: int,
    resolution: int,
    audit_resolution: int,
) -> dict[str, Any]:
    """Generate one mesh at ``band`` and score it. -> a result row."""
    from . import meshaudit

    label = "auto" if band is None else str(band)
    glb_path = out_dir / f"band-{label}.glb"
    # A fresh server per value, and stopped before the next one starts: --band
    # is fixed at launch, and two 16 GB servers do not coexist.
    server = TrellisServer(
        config.trellis_server_exe,
        config.trellis_models_dir,
        config.trellis_port,
        log_path=out_dir / f"trellis-band-{label}.log",
        webp=config.trellis_webp,
        tex_res=config.trellis_tex_res,
        band=band,
    )
    started = time.monotonic()
    try:
        await server.generate(image, glb_path, seed=seed, resolution=resolution)
    finally:
        await asyncio.to_thread(server.stop)
    generate_s = time.monotonic() - started

    report = await asyncio.to_thread(
        meshaudit.hole_fraction, glb_path, meshaudit.DEFAULT_VIEWS, audit_resolution
    )
    return {
        "band": label,
        "worst": report["worst"],
        "mean": report["mean"],
        "faces": report["faces"],
        "seconds": generate_s,
        "glb": glb_path,
    }


def require_free_port(config: Config) -> None:
    """Refuse to run if something already holds the trellis port.

    ensure_started() polls /health and returns as soon as *a* server answers --
    it cannot tell one process from another. With the app (or a stray exe)
    already listening, every band would be measured against that server's
    settings and the sweep would return a confident table of identical, wrong
    numbers. Failing loudly is the only safe behaviour for a measurement tool.
    """
    check = doctor._port_check(config)
    if not check.ok:
        raise SystemExit(
            f"{check.detail}.\nStop the Warlock Studio server (and any stray "
            f"trellis-server.exe) first: the sweep needs to own port "
            f"{config.trellis_port}, or it will measure whatever is already "
            f"there instead of the band it just launched."
        )


async def sweep(
    config: Config,
    image: Path,
    bands: list[int | None],
    out_dir: Path,
    *,
    seed: int,
    resolution: int,
    audit_resolution: int,
) -> list[dict[str, Any]]:
    if not image.exists():
        raise SystemExit(f"reference image not found: {image}")
    require_free_port(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for band in bands:
        label = "auto" if band is None else str(band)
        print(f"[band {label}] generating...", flush=True)
        try:
            row = await run_band(
                config, band, image, out_dir,
                seed=seed, resolution=resolution, audit_resolution=audit_resolution,
            )
        except Exception as exc:
            # One bad value must not lose the values already measured -- a full
            # sweep is several minutes of GPU time per band.
            log.exception("band %s failed", label)
            rows.append({"band": label, "error": str(exc)})
            print(f"[band {label}] FAILED: {exc}", flush=True)
            continue
        rows.append(row)
        print(
            f"[band {label}] worst {row['worst']:.4f}  mean {row['mean']:.4f}  "
            f"faces {row['faces']}  {row['seconds']:.1f}s",
            flush=True,
        )
    return rows


def print_table(rows: list[dict[str, Any]], audit_resolution: int) -> None:
    print()
    print(f"band sweep (hole_fraction @ {audit_resolution}, lower is better)")
    print(f"{'band':>6}  {'worst':>8}  {'mean':>8}  {'faces':>9}  {'gen s':>7}")
    for row in rows:
        if "error" in row:
            print(f"{row['band']:>6}  {'failed':>8}  {row['error'][:40]}")
            continue
        print(
            f"{row['band']:>6}  {row['worst']:>8.4f}  {row['mean']:>8.4f}  "
            f"{row['faces']:>9d}  {row['seconds']:>7.1f}"
        )
    scored = [r for r in rows if "error" not in r]
    if not scored:
        print("\nevery band failed; see the trellis logs in the output directory")
        return
    best = min(scored, key=lambda r: r["worst"])
    print(f"\nlowest: band {best['band']} at {best['worst']:.4f} worst-view hole fraction")

    baseline = next((r for r in scored if r["band"] == "auto"), None)
    if baseline is None:
        print("no 'auto' row to compare against -- rerun including it before changing anything.")
        return
    if best is baseline:
        print("that is the exe's own heuristic: leave DEFAULT_TRELLIS_BAND = None.")
        return
    # One generation per band, so a narrow win is indistinguishable from run-to-run
    # noise -- the first sweep measured two runs of the *same* effective setting
    # disagreeing by 0.3% on face count. Overriding the exe's heuristic on that
    # basis would be picking a number out of the noise and calling it tuning.
    margin = baseline["worst"] - best["worst"]
    if margin < baseline["worst"] * MEANINGFUL_MARGIN:
        print(
            f"but 'auto' measured {baseline['worst']:.4f} -- a "
            f"{margin / baseline['worst'] * 100:.1f}% difference on a single run, "
            f"which is inside the noise.\nLeave DEFAULT_TRELLIS_BAND = None unless a "
            f"repeat sweep reproduces the gap."
        )
        return
    print(
        f"clearly better than 'auto' ({baseline['worst']:.4f}). Set "
        f"config.DEFAULT_TRELLIS_BAND = {best['band']}."
    )

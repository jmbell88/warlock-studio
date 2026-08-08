"""Planning a model download: what to fetch, where it lands, whether it fits.

Pure in the way ``vram.py`` and ``memlog.py`` are -- stdlib plus ``models`` and
``config``, no imports from ``service``/``queue``/``studio``, and **no network
of any kind**. Planning a fetch is not performing one: the performing half runs
out-of-process in ``pipelines/fetch_worker``, which is the whole reason the app
process can keep ``HF_HUB_OFFLINE=1`` for its entire life. Everything here is
therefore assertable headlessly, which is the point -- the path rule, the
dedupe and the disk refusal are exactly the parts that go wrong silently.

Two rules this file exists to own, both of which had already been got wrong
once as prose:

* **Paths resolve through one helper.** ``base_model_dir`` is the canonical
  rule -- only ``turbo`` honours ``WARLOCK_T2I_DIR``, everything else is
  ``t2i_model_root / dir_name`` -- and ``doctor`` imports it rather than
  keeping a second copy. The literal ``models/...`` in a download *string* is
  the README's spelling of the default root, never where a fetch actually
  goes.
* **Dedupe is on (repo_id, destination), never on model key.** ``dir_name`` is
  not unique: ``sdxl``, ``sdxl_cfg``, ``pixel`` and ``lightning`` are four
  recipes over one copy of ``sdxl-base-1.0``, and a key-keyed plan would fetch
  7 GB four times.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import models
from .config import Config

# Free space a fetch leaves behind, over and above what it declares. The
# declared sizes are approximate (see models.Fetch), and a disk filled to the
# last byte by a download is its own kind of failure.
DISK_HEADROOM_GIB = 2.0

_GIB = float(1024**3)

# What kinds of registry entry there are, and the doctor row-name prefix each
# one's presence is reported under. Written out rather than derived, for the
# reason models.py writes its blend-mode ids out: a prefix that stopped
# matching would silently mark a downloaded model missing forever, and a
# mismatch is far easier to see in a table than in a format string.
CHECK_PREFIXES: dict[str, str] = {
    "base": "image model: ",
    "lora": "style LoRA: ",
    "adapter": "IP-Adapter: ",
    "control": "ControlNet: ",
    "metric": "metric model: ",
    "pose": "pose model: ",
    "matting": "host matting: ",
}


@dataclass(frozen=True, slots=True)
class Entry:
    """One downloadable row: a registry record plus how to name and probe it."""

    kind: str
    key: str
    label: str
    spec: Any

    @property
    def fetch(self) -> tuple[models.Fetch, ...]:
        return tuple(getattr(self.spec, "fetch", ()) or ())

    @property
    def check_name(self) -> str:
        """The ``doctor.Check.name`` this row's presence is reported under."""
        return f"{CHECK_PREFIXES[self.kind]}{self.label}"

    def is_present(self, config: Config) -> bool:
        return present(config, self.kind, self.spec)

    @property
    def row_key(self) -> str:
        """A key unique across kinds -- ``lora:pixelxl`` and a base model of the
        same key would otherwise share a task key and a checkbox."""
        return f"{self.kind}:{self.key}"


def entries() -> list[Entry]:
    """Every registry entry that has something to download, in doctor's order."""
    tables = (
        ("base", models.BASE_MODELS),
        ("lora", models.STYLE_LORAS),
        ("adapter", models.IP_ADAPTERS),
        ("control", models.CONTROLNETS),
        ("metric", models.METRIC_MODELS),
        ("pose", models.POSE_MODELS),
        ("matting", models.MATTING_MODELS),
    )
    out: list[Entry] = []
    for kind, table in tables:
        for key, spec in table.items():
            out.append(Entry(kind, key, spec.label, spec))
    return out


def find(row_key: str) -> Entry | None:
    for entry in entries():
        if entry.row_key == row_key:
            return entry
    return None


# --- where things live -------------------------------------------------------


def base_model_dir(config: Config, spec: models.BaseModel) -> Path:
    """Where this checkpoint's weights are, honouring the one legacy override.

    ``WARLOCK_T2I_DIR`` relocates the ``turbo`` entry *only* -- it predates the
    registry and is kept so existing setups keep working. Every other model is
    ``t2i_model_root / dir_name``. This is the canonical rule; doctor and the
    download planner both call it rather than re-deriving it.
    """
    if spec.key == models.DEFAULT_BASE_MODEL and config.t2i_turbo_dir is not None:
        return config.t2i_turbo_dir
    return config.t2i_model_root / spec.dir_name


def destination(config: Config, entry: Entry, one: models.Fetch) -> Path:
    """Where ``one`` actually lands, as opposed to what its command string says.

    A base model's *own* directory goes through ``base_model_dir`` so the turbo
    override is honoured; anything else -- a step-distillation LoRA into the
    flat ``loras/``, an adapter, a metric -- resolves under the model root. Both
    halves relocate with ``WARLOCK_T2I_ROOT``, which the hardcoded
    ``--local-dir models/...`` in the command string never did.
    """
    spec = entry.spec
    if entry.kind == "base" and one.local_dir == spec.dir_name:
        return base_model_dir(config, spec)
    return config.t2i_model_root / one.local_dir


# --- the plan ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Job:
    """One resolved fetch, ready to hand to the worker."""

    repo_id: str
    dest: Path
    filenames: tuple[str, ...] = ()
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    rename: tuple[str, str] | None = None
    size_gib: float = 0.0

    def spec(self) -> dict[str, Any]:
        """The JSON the worker reads on stdin.

        ``filenames`` and ``allow_patterns`` are handed over separately rather
        than pre-merged, because an exact path and a glob read differently to a
        person reading a log line -- the worker unions them itself.
        """
        return {
            "repo_id": self.repo_id,
            "dest": str(self.dest),
            "filenames": list(self.filenames),
            "allow_patterns": list(self.allow_patterns),
            "ignore_patterns": list(self.ignore_patterns),
            "rename": list(self.rename) if self.rename else None,
            "size_gib": self.size_gib,
        }


def _merge(into: Job, one: models.Fetch) -> Job:
    """Fold a second record for the same (repo, destination) into the first.

    Union rather than replace, and order-preserving: the IP-Adapter lists its
    weights and its CLIP vision encoder as two records against one repository
    precisely so doctor prints two lines, and a fetch of only the second would
    produce the half-downloaded directory doctor's two-part probe exists to
    catch.
    """

    def _union(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
        out = list(a)
        out.extend(x for x in b if x not in out)
        return tuple(out)

    return Job(
        repo_id=into.repo_id,
        dest=into.dest,
        filenames=_union(into.filenames, one.filenames),
        allow_patterns=_union(into.allow_patterns, one.allow_patterns),
        ignore_patterns=_union(into.ignore_patterns, one.ignore_patterns),
        rename=into.rename or one.rename,
        size_gib=into.size_gib + one.size_gib,
    )


def plan(config: Config, chosen: list[Entry]) -> list[Job]:
    """Resolve entries into the set of fetches that actually has to run.

    Deduped on ``(repo_id, destination)``. Two entries naming the same
    repository into the same directory are one download, and the size is
    counted once -- which is the difference between "this needs 7 GB" and "this
    needs 28 GB" for the four SDXL 1.0 recipes.
    """
    order: list[tuple[str, Path]] = []
    jobs: dict[tuple[str, Path], Job] = {}
    # Which records have already been folded into each job. The same *record*
    # arriving again is the four-SDXL-recipes case and must change nothing --
    # merging it would union patterns that are already there and, worse, add
    # its size a second time, so ticking all four would claim 28 GB. A
    # genuinely different record against the same destination (the IP-Adapter's
    # two halves) still merges.
    seen: dict[tuple[str, Path], set[models.Fetch]] = {}
    for entry in chosen:
        for one in entry.fetch:
            dest = destination(config, entry, one)
            key = (one.repo_id, dest)
            if key in jobs:
                if one in seen[key]:
                    continue
                seen[key].add(one)
                jobs[key] = _merge(jobs[key], one)
                continue
            seen[key] = {one}
            order.append(key)
            jobs[key] = Job(
                repo_id=one.repo_id,
                dest=dest,
                filenames=one.filenames,
                allow_patterns=one.allow_patterns,
                ignore_patterns=one.ignore_patterns,
                rename=one.rename,
                size_gib=one.size_gib,
            )
    return [jobs[k] for k in order]


def total_gib(jobs: list[Job]) -> float:
    return sum(job.size_gib for job in jobs)


def free_gib(path: Path) -> float | None:
    """Free space on the volume ``path`` lives on, or None if it cannot be read.

    Walks up to the first parent that exists: a download's destination is
    routinely a directory that does not exist yet, and ``disk_usage`` on a
    missing path raises rather than answering about its volume.
    """
    probe = path
    for _ in range(64):
        if probe.exists():
            try:
                return shutil.disk_usage(probe).free / _GIB
            except OSError:
                return None
        if probe.parent == probe:
            return None
        probe = probe.parent
    return None


def disk_refusal(jobs: list[Job]) -> str | None:
    """Why this plan must not start, or None. Refusing is the point.

    A download that runs the volume dry leaves a partial model directory *and*
    a machine that cannot write anything else, and the failure surfaces
    somewhere unrelated. Unreadable free space is not a refusal: an answer
    nobody can obtain is not evidence there is no room.
    """
    if not jobs:
        return None
    need = total_gib(jobs) + DISK_HEADROOM_GIB
    free = free_gib(jobs[0].dest)
    if free is None or free >= need:
        return None
    return (
        f"Not enough disk space: this needs about {need:.1f} GB "
        f"(including {DISK_HEADROOM_GIB:.0f} GB headroom) and "
        f"{free:.1f} GB is free."
    )


# --- is it already here ------------------------------------------------------


def base_model_state(config: Config, spec: models.BaseModel) -> tuple[bool, Path | None]:
    """(is the checkpoint usable, which step-distillation LoRA is missing).

    ``model_index.json`` is the diffusers layout marker; the unet shard is the
    biggest file and the one a partial or wrong-variant download would miss. A
    spec may name its own ``probe`` files instead, because that formula is an
    SDXL fact -- FLUX.2 klein has no ``unet/`` at all, and its text encoder is
    half the download, so a probe that looked only at the transformer would
    call a half-fetched model present.

    The step-distillation LoRA counts as part of the checkpoint: ``_load_loras``
    *raises* on a missing one, where a style LoRA is merely skipped. Which one
    is missing is returned rather than folded in, because doctor's two details
    are different sentences and the user acts on them differently.
    """
    path = base_model_dir(config, spec)
    variant = f".{spec.variant}" if spec.variant else ""
    wanted = spec.probe or (f"unet/diffusion_pytorch_model{variant}.safetensors",)
    ok = (path / "model_index.json").exists() and all(
        (path / rel).exists() for rel in wanted
    )
    if ok and spec.base_lora:
        lora_path = config.t2i_model_root / "loras" / spec.base_lora
        if not lora_path.exists():
            return False, lora_path
    return ok, None


def present(config: Config, kind: str, spec: Any) -> bool:
    """Whether this row's files are all on disk.

    A *partial* directory is absent, deliberately: every probe here names more
    than one file where more than one matters, so a fetch interrupted halfway
    reads as "not downloaded" rather than as a model that will fail at load
    time with the checkpoint already in VRAM.

    Deliberately narrower than the doctor row of the same name, which also
    reports import failures and last-load errors. Those are real, and none of
    them is fixed by downloading -- so what a Download button is offered
    against is this question and not that one.
    """
    root = config.t2i_model_root
    if kind == "base":
        return base_model_state(config, spec)[0]
    if kind == "lora":
        return (root / "loras" / spec.filename).exists()
    if kind == "adapter":
        base = root / spec.dir_name
        # Both halves: weights without the CLIP vision encoder load fine and
        # then fail at the first call.
        return (base / spec.subfolder / spec.weight_name).exists() and (
            base / spec.image_encoder_dir / "config.json"
        ).exists()
    if kind == "control":
        base = root / spec.dir_name
        variant = f".{spec.variant}" if spec.variant else ""
        return (base / "config.json").exists() and (
            base / f"diffusion_pytorch_model{variant}.safetensors"
        ).exists()
    return (root / spec.dir_name / "config.json").exists()

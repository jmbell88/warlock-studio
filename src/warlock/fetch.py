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

# What kinds of registry entry there are, and everything a reader needs to say
# about one: which registry table it comes from, the prefix doctor reports its
# presence under, and the heading the app-Settings pane groups it beneath. One
# table rather than three, because all three were hand-copies of one list and
# each drifted differently -- the pane's copy dropped a kind by *omitting* it,
# silently, which is the failure a list of names cannot show you.
#
# The order is the order rows appear in, in doctor and in the pane alike.


@dataclass(frozen=True, slots=True, eq=False)
class Kind:
    key: str
    table: dict[str, Any]
    check_prefix: str
    """The ``doctor.Check.name`` prefix rows of this kind are reported under."""
    group: str
    """The app-Settings heading rows of this kind sit under. Repeats are
    deliberate: adapters and ControlNets are both "Conditioning"."""


KINDS: tuple[Kind, ...] = (
    Kind("base", models.BASE_MODELS, "image model: ", "Image models"),
    Kind("lora", models.STYLE_LORAS, "style LoRA: ", "Style LoRAs"),
    Kind("adapter", models.IP_ADAPTERS, "IP-Adapter: ", "Conditioning"),
    Kind("control", models.CONTROLNETS, "ControlNet: ", "Conditioning"),
    Kind("metric", models.METRIC_MODELS, "metric model: ", "Measurement"),
    Kind("pose", models.POSE_MODELS, "pose model: ", "Measurement"),
    Kind("matting", models.MATTING_MODELS, "host matting: ", "Measurement"),
)

CHECK_PREFIXES: dict[str, str] = {kind.key: kind.check_prefix for kind in KINDS}

GROUPS: tuple[tuple[str, str], ...] = tuple((kind.key, kind.group) for kind in KINDS)
"""(kind, heading) in display order, for the pane that draws the rows."""


def check_name(kind: str, label: str) -> str:
    """The ``doctor.Check.name`` a row of this kind and label is reported under.

    Doctor composes its own row names through this rather than repeating the
    prefix in seven f-strings: a prefix that stopped matching would silently
    mark a downloaded model missing forever.
    """
    return f"{CHECK_PREFIXES[kind]}{label}"


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
        return check_name(self.kind, self.label)

    def is_present(self, config: Config) -> bool:
        return present(config, self.kind, self.spec)

    @property
    def row_key(self) -> str:
        """A key unique across kinds -- ``lora:pixelxl`` and a base model of the
        same key would otherwise share a task key and a checkbox."""
        return f"{self.kind}:{self.key}"


def entries() -> list[Entry]:
    """Every registry entry that has something to download, in doctor's order.

    Walks ``KINDS`` rather than a second list of tables: a registry gaining a
    kind that this file has not heard of is then absent everywhere at once,
    which is visible, rather than absent from one of three hand-copied lists.
    """
    out: list[Entry] = []
    for kind in KINDS:
        for key, spec in kind.table.items():
            out.append(Entry(kind.key, key, spec.label, spec))
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

    The entry is named by name (``models.T2I_DIR_MODEL``), never the default:
    the override is about turbo's settings, so it must not follow the default
    to another checkpoint.
    """
    if spec.key == models.T2I_DIR_MODEL and config.t2i_turbo_dir is not None:
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


# --- taking one back out ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Removal:
    """What uninstalling a selection would actually delete.

    Pure, like ``plan``: nothing here touches the filesystem beyond resolving
    paths, so every rule below is assertable without a model on disk. The
    performing half is ``service.downloads.uninstall``.
    """

    paths: tuple[Path, ...]
    """Exactly what to delete, in claim order. Empty when ``blocked``."""
    freed_gib: float
    """Declared size of the fetches those paths stand for. 0.0 when blocked."""
    blocked: tuple[str, ...]
    """Why this must not run at all. Whole-plan, like ``disk_refusal``."""


def claims(config: Config, entry: Entry) -> tuple[Path, ...]:
    """Every path this row's *presence* stands on.

    The inverse of ``present``, and written beside nothing else on purpose:
    "what would make this row read as absent" and "what may be deleted for it"
    have to be the same list, or an uninstall leaves a row that still reports
    present or deletes something a different row was standing on.

    A base model claims its checkpoint directory **and** its step-distillation
    LoRA file, because ``base_model_state`` refuses without the latter -- but
    never ``loras/`` itself. Per-file claims only in there: the directory is
    flat and shared by every family, and the renames in ``models.py`` exist
    precisely so the filenames are distinct.
    """
    root = config.t2i_model_root
    spec = entry.spec
    if entry.kind == "base":
        out = [base_model_dir(config, spec)]
        if spec.base_lora:
            out.append(root / "loras" / spec.base_lora)
        return tuple(out)
    if entry.kind == "lora":
        return (root / "loras" / spec.filename,)
    # Everything else -- adapters, ControlNets, metrics, pose, matting -- is one
    # directory of its own under the model root.
    return (root / spec.dir_name,)


def removal_plan(config: Config, chosen: list[Entry]) -> Removal:
    """Which of these rows' paths may actually go, and what that frees.

    The reverse of ``plan``'s dedupe and keyed the same way -- on resolved
    paths, never on model key. A path is removable **iff no registry entry
    outside the selection claims it**, counted over every other entry rather
    than only the present ones: an entry whose weights are half-downloaded, or
    downloaded later, is still a claim, and deleting out from under it would
    turn "not installed yet" into "mysteriously broken".

    The practical shape of that: ``sdxl``, ``sdxl_cfg``, ``pixel`` and
    ``lightning`` are four recipes over one 7 GiB directory, so uninstalling
    any one of them frees only its own small distillation LoRA. The directory
    goes when all four are chosen together, and not before.

    Two whole-plan refusals, all-or-nothing in ``disk_refusal``'s style, because
    a partial delete is the outcome with no good description:

    * a chosen base whose directory is the ``WARLOCK_T2I_DIR`` override. That
      is a directory the user pointed us at, not one we fetched.
    * anything that does not resolve inside ``t2i_model_root``. Containment is
      checked rather than assumed: ``dir_name`` comes from the registry today,
      and this is the one function in the tree that deletes recursively.
    """
    chosen_keys = {entry.row_key for entry in chosen}
    others: set[Path] = set()
    for entry in entries():
        if entry.row_key in chosen_keys:
            continue
        others.update(claims(config, entry))

    wanted: list[Path] = []
    for entry in chosen:
        for path in claims(config, entry):
            if path not in wanted:
                wanted.append(path)

    blocked: list[str] = []
    override = config.t2i_turbo_dir
    if override is not None:
        pinned = [p for p in wanted if p == override]
        if pinned:
            blocked.append(
                "WARLOCK_T2I_DIR points at that model's directory. Warlock did "
                "not download it and will not delete it; unset the variable "
                "first if you really want it gone."
            )
    try:
        root = config.t2i_model_root.resolve()
        outside = [p for p in wanted if not p.resolve().is_relative_to(root)]
    except OSError:  # pragma: no cover -- a root that cannot be resolved
        outside = list(wanted)
    if outside:
        blocked.append(
            "that model's files are outside the model root "
            f"({config.t2i_model_root}), so removing them is refused."
        )
    if blocked:
        return Removal((), 0.0, tuple(blocked))

    removable = tuple(p for p in wanted if p not in others)
    freed = 0.0
    for job in plan(config, chosen):
        if any(p == job.dest or job.dest in p.parents for p in removable):
            freed += job.size_gib
    return Removal(removable, freed, ())


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

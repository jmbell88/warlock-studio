"""Planning a dependency-pack install: which packs, how big, whether it fits.

Pure in the way ``fetch.py`` is -- stdlib plus ``config`` and ``fetch``'s one
volume helper, no imports from ``service``/``queue``/``studio``, and **no
network of any kind**. Planning an install is not performing one: the
performing half runs out-of-process against a directory of already-downloaded
wheels, which is the whole reason this can be reasoned about with nothing on
disk.

The thing being planned is the *other* half of the download story. ``fetch.py``
plans model weights; this plans the Python distributions that can read them.
Today ``installer/build.ps1`` exports every extra at once
(``--extra studio --extra text2image --extra rig --extra music``) and stages the
lot, so a user who only ever draws pixel art still downloads torch, a CUDA
runtime and a lyric-language stack. A pack is the unit that changes: the base
installer carries the app and ``studio``, and everything heavy arrives later,
chosen.

Three rules this file exists to own, each of which is a way the same figure
goes silently wrong:

* **Dedupe is on wheel filename, never on pack.** ``text2image`` and ``music``
  share 27 distributions, torch among them. A pack-keyed sum tells a user who
  ticks both that they are about to download torch twice, and a pack-keyed
  *install* would fetch it twice. This is exactly ``fetch.plan``'s
  four-SDXL-recipes-over-one-directory rule wearing different clothes.
* **The manifest is the size authority, never the lock.** ``uv.lock`` declares
  a ``size`` for every wheel it resolves *except* the three that dominate the
  figure -- see ``SIZELESS_DISTS``.
* **Two volumes, not one.** The wheels land in a cache directory under the
  user's Warlock home; the packages land in the application runtime's
  ``site-packages``, which on a per-user install of a checkout-shaped app is
  routinely a different drive. Budgeting one and not the other is MDL-09 in a
  new place, so the volume rule is imported from ``fetch`` rather than
  rewritten here.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fetch import DISK_HEADROOM_GIB, volume_refusal

_GIB = float(1024**3)

MANIFEST_VERSION = 1
"""The one manifest shape this reader accepts. Bumped, never widened: a
manifest from a different generator is a build mistake, and reading it
half-successfully is how a pack installs the wrong wheels."""

MANIFEST_NAME = "packs.json"
"""What the generator writes beside the wheels it collected."""

# The three distributions ``uv.lock`` resolves with no declared ``size``, and
# they are the three that decide the download figure a user is shown.
#
# The cause is the source, not the lock format: every wheel from PyPI carries
# ``size = ...``, and the ones from ``download.pytorch.org/whl/cu128`` carry
# only ``url``, ``hash`` and ``upload-time``. So a generator that believes the
# lock produces a "1.4 GB" estimate for a 3 GB download -- wrong in the
# direction that runs a volume dry mid-install, and wrong *silently*, because
# a missing key sums as zero.
#
# The rule that follows is why this constant exists rather than a table of
# fallback numbers: **the generator stats the wheel files it collected.** It
# has them on disk -- it cannot build a pack without them -- so a hardcoded
# size would be a guess standing next to the truth. This tuple is pinned
# against ``uv.lock`` by ``tests/test_packs.py`` in both directions: a fourth
# sizeless distribution, or one of these three gaining a size, is a change to
# how the figure may be computed and has to be read by a person.
SIZELESS_DISTS: tuple[str, ...] = ("torch", "torchaudio", "torchvision")


class ManifestError(ValueError):
    """The pack manifest is malformed, or is not the shape this reader knows."""


# --- what a pack is ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pack:
    """One optional dependency set, and everything a reader needs to say about it.

    ``extras`` is the pack's definition -- the names ``uv sync --extra`` and
    ``uv export --extra`` take -- and every other field is how that definition
    is spoken about. One table rather than four, for ``fetch.KINDS``' reason:
    the install hints below already existed as three separate string literals
    in ``doctor``, ``cli`` and the queue's refusals, and a list of names cannot
    show you the one that drifted.
    """

    key: str
    label: str
    extras: tuple[str, ...]
    """The pyproject optional-dependency groups this pack resolves to."""
    modes: tuple[str, ...]
    """Which ``studio.modes`` keys stop working without it, in rail order.
    Strings rather than imports: ``studio`` may not be imported from here, and
    ``tests/test_packs.py`` pins them against the real list."""
    probe: tuple[str, ...]
    """Top-level module names whose presence means the pack is installed.
    Direct imports only -- what ``src/`` reaches for itself, not the transitive
    closure, which is the resolver's job and not a thing to hand-maintain."""
    summary: str
    """One line, for the pane and for doctor. What the user loses without it."""

    @property
    def install_hint(self) -> str:
        """The source-checkout remedy, composed rather than repeated.

        Doctor, the CLI's startup refusal and the queue's job refusals all
        print a ``uv sync --extra`` line; they were three literals and the
        music one was written a release after the extra it names.
        """
        return "uv sync " + " ".join(f"--extra {name}" for name in self.extras)


PACKS: tuple[Pack, ...] = (
    Pack(
        key="text2image",
        label="Image generation",
        extras=("text2image",),
        modes=("create",),
        # The direct imports, and the four at the end are not decoration:
        # BiRefNet's modelling code is loaded with ``trust_remote_code`` and
        # reaches for einops, kornia and timm from inside the checkpoint, where
        # no resolver can see them -- the defect ``doctor._MATTING_IMPORTS``
        # was written for. torchvision is there for the same reason one step
        # removed: transformers builds its fast image processors on it, and
        # its absence degrades candidate ranking with nothing on screen.
        probe=(
            "torch",
            "diffusers",
            "transformers",
            "accelerate",
            "peft",
            "torchvision",
            "einops",
            "kornia",
            "timm",
        ),
        summary=(
            "SDXL reference images, host matting and candidate ranking. "
            "Without it Create cannot make a reference, and every export's "
            "alpha comes from the corner flood fill instead."
        ),
    ),
    Pack(
        key="rig",
        label="Rigging",
        extras=("rig",),
        modes=("poser", "troupe"),
        probe=("bpy",),
        summary=(
            "Skeleton fitting and skinning. Without it Poser cannot rig a "
            "mesh, and Troupe has no clip to render a sheet from."
        ),
    ),
    Pack(
        key="music",
        label="Music generation",
        extras=("music",),
        modes=("muse",),
        probe=(
            "torch",
            "torchaudio",
            "diffusers",
            "transformers",
            "librosa",
            "loguru",
            "py3langid",
            "pypinyin",
            "num2words",
            "hangul_romanize",
            "cutlet",
            "fugashi",
            "spacy",
        ),
        summary=(
            "ACE-Step text-to-music and stem separation. Without it Muse "
            "refuses a take; auditioning and exporting a finished one still "
            "work, because those never touch torch."
        ),
    ),
)
"""Every installable pack, in the order rows appear in.

``studio`` is deliberately not here. It is not optional in the shipped product
-- it is the window -- and the base installer carries it; its extra exists so
that ``warlock doctor`` can run on a machine with no display.
"""

KEYS: tuple[str, ...] = tuple(pack.key for pack in PACKS)


def find(key: str) -> Pack | None:
    for pack in PACKS:
        if pack.key == key:
            return pack
    return None


# --- is it already here? ------------------------------------------------------


def missing_modules(names: Sequence[str]) -> list[str]:
    """Which of these do not resolve, without importing any of them.

    ``find_spec`` locates a top-level module without executing it, which is
    what keeps this cheap and keeps a startup check from dragging torch into
    the process. It raises rather than returns None for a dotted name whose
    parent is absent (and a package with broken metadata can raise anything at
    all), so every answer is wrapped: a probe that takes the app down at
    startup is strictly worse than a red row.
    """
    missing: list[str] = []
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except Exception:  # noqa: BLE001 -- a probe must never raise out of a check
            found = False
        if not found:
            missing.append(name)
    return missing


def missing(pack: Pack) -> list[str]:
    """Which of ``pack``'s direct imports are absent, in declaration order."""
    return missing_modules(pack.probe)


def installed(pack: Pack) -> bool:
    """Whether this pack's imports all resolve in the running interpreter.

    *All*, not any: the failure this exists to catch is the partial install --
    ``music`` shipped without ``fugashi`` reads as present on any "is torch
    here" test and fails at the first take with ``ModuleNotFoundError``, which
    is the shape of the defect that put Muse in the rail of a packaged build
    that could not run it.
    """
    return not missing(pack)


def chosen_packs(keys: Iterable[str]) -> list[Pack]:
    """Resolve pack keys to packs in ``PACKS`` order, refusing an unknown one.

    Order is the registry's rather than the caller's so that a plan reads the
    same however the checkboxes were ticked, and an unknown key raises rather
    than being skipped: a typo that silently plans *less* is how a pack goes
    missing from an installer for a release cycle.
    """
    wanted = list(keys)
    for key in wanted:
        if find(key) is None:
            raise KeyError(f"no such pack: {key!r}")
    return [pack for pack in PACKS if pack.key in set(wanted)]


# --- the manifest -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Wheel:
    """One collected wheel file, and which packs claim it."""

    filename: str
    """The wheel's own name, as PEP 427 spells it. The identity a plan dedupes
    on: two packs naming this file mean one download."""
    url: str
    """Where the file comes from -- the URL the lock recorded, which is PyPI or
    ``download.pytorch.org``. Empty when ``bundled``.

    Carrying it means a pack needs no hosting of its own: the installer ships
    ``packs.json`` and the wheels arrive from the same places ``uv`` would have
    got them, pinned by ``sha256`` rather than by trust in the host. https
    only, and enforced rather than assumed -- a manifest is a list of files
    about to be installed into the application runtime, and the digest is what
    makes the transport safe, not the other way round."""
    size_bytes: int
    """Stat of the collected file. Never the lock's declared size -- see
    ``SIZELESS_DISTS`` for the three that have none."""
    sha256: str
    bundled: bool = False
    """Whether this wheel ships with the installer instead of being fetched.

    True for exactly the distributions that publish no Windows wheel, which
    the build compiles itself. There is no URL such a file could be downloaded
    from, so the installer has to carry it -- which is a real cost worth seeing
    in the manifest rather than a detail: ``unidic-lite`` alone is most of it.
    """
    installed_bytes: int = 0
    """What the distribution occupies unpacked, summed from its ``RECORD`` by
    the generator, which has the installed tree in front of it. ``0`` means
    unmeasured, and is not a claim that it is free: ``disk_refusal`` declines
    to budget an install volume it has no figure for, the way ``fetch`` treats
    unreadable free space."""
    packs: tuple[str, ...] = ()
    """Which pack keys pull this wheel in. A wheel claimed by no pack is a
    generator bug and ``parse_manifest`` refuses it."""


@dataclass(frozen=True, slots=True)
class Manifest:
    """What a generator collected, read back with nothing trusted."""

    version: int
    wheels: tuple[Wheel, ...]

    def for_pack(self, key: str) -> tuple[Wheel, ...]:
        return tuple(w for w in self.wheels if key in w.packs)


def _int_field(raw: Mapping[str, Any], key: str, filename: str, *, required: bool) -> int:
    value = raw.get(key)
    if value is None and not required:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{filename}: {key} must be a non-negative integer")
    return value


def parse_manifest(payload: Any) -> Manifest:
    """Read a manifest object, refusing anything that is not exactly the shape.

    Written in ``installer/verify_runtime.py``'s style and for its reason: the
    thing on the other side of this parse is a list of files that are about to
    be installed into the application runtime, so "malformed" and "not the
    version I know" are refusals rather than defaults. A duplicate filename is
    one too: two entries for one name means the generator collected two builds
    of one distribution, and picking either silently is how a runtime ends up
    with a wheel nobody chose.
    """
    if not isinstance(payload, Mapping):
        raise ManifestError("pack manifest must be an object")
    if payload.get("version") != MANIFEST_VERSION:
        raise ManifestError(f"pack manifest must be version {MANIFEST_VERSION}")
    raw_wheels = payload.get("wheels")
    if not isinstance(raw_wheels, list) or not raw_wheels:
        raise ManifestError("pack manifest names no wheels")
    seen: set[str] = set()
    wheels: list[Wheel] = []
    for raw in raw_wheels:
        if not isinstance(raw, Mapping):
            raise ManifestError("each wheel must be an object")
        filename = raw.get("filename")
        if not isinstance(filename, str) or not filename.endswith(".whl"):
            raise ManifestError(f"wheel filename is not a wheel: {filename!r}")
        if "/" in filename or "\\" in filename or filename != Path(filename).name:
            raise ManifestError(f"wheel filename is not a bare name: {filename!r}")
        if filename in seen:
            raise ManifestError(f"pack manifest lists {filename} twice")
        seen.add(filename)
        url = raw.get("url")
        bundled = raw.get("bundled") is True
        if not isinstance(url, str):
            raise ManifestError(f"{filename}: url must be a string")
        # Exactly one of the two ways a wheel can arrive, never both and never
        # neither. ``docopt``, ``mojimoji`` and ``unidic-lite`` publish no
        # Windows wheel, so the build compiles them and they have to travel
        # with the installer; everything else is fetched. A bundled wheel with
        # a URL would invite a download of something that is not what was
        # built, and a fetched wheel without one fails on the user's machine
        # rather than here.
        if bundled and url:
            raise ManifestError(f"{filename}: a bundled wheel cannot also have a url")
        if not bundled and not url.startswith("https://"):
            raise ManifestError(f"{filename}: url must be https, not {url!r}")
        sha256 = raw.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ManifestError(f"{filename}: sha256 must be 64 hex characters")
        try:
            int(sha256, 16)
        except ValueError:
            raise ManifestError(f"{filename}: sha256 must be 64 hex characters") from None
        raw_packs = raw.get("packs")
        if not isinstance(raw_packs, list) or not raw_packs:
            raise ManifestError(f"{filename}: is claimed by no pack")
        packs: list[str] = []
        for key in raw_packs:
            if not isinstance(key, str) or find(key) is None:
                raise ManifestError(f"{filename}: names a pack that does not exist: {key!r}")
            if key not in packs:
                packs.append(key)
        wheels.append(
            Wheel(
                filename=filename,
                url=url,
                bundled=bundled,
                size_bytes=_int_field(raw, "size_bytes", filename, required=True),
                sha256=sha256.lower(),
                installed_bytes=_int_field(raw, "installed_bytes", filename, required=False),
                packs=tuple(packs),
            )
        )
    return Manifest(version=MANIFEST_VERSION, wheels=tuple(wheels))


def load_manifest(path: Path) -> Manifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read pack manifest {path}: {exc}") from exc
    return parse_manifest(payload)


# --- the plan -----------------------------------------------------------------


def plan(manifest: Manifest, chosen: Sequence[str] | Sequence[Pack]) -> list[Wheel]:
    """The wheels this selection actually has to install, each one once.

    Deduped on filename in manifest order. Ticking ``text2image`` and ``music``
    together is one torch, not two -- the 27 distributions they share are the
    difference between an honest figure and one nearly double it, and between
    an install that runs once and one that writes the same 3 GB twice.

    Order is the manifest's rather than the selection's, so that a plan for
    two packs is a stable list a person can diff against the pack directory.
    """
    keys = {p.key if isinstance(p, Pack) else p for p in chosen}
    for key in keys:
        if find(key) is None:
            raise KeyError(f"no such pack: {key!r}")
    return [wheel for wheel in manifest.wheels if keys.intersection(wheel.packs)]


def total_bytes(wheels: Sequence[Wheel]) -> int:
    """What this plan downloads. Bytes, because that is what the manifest
    measured -- rounding to GiB is the display's job and doing it here would
    make two plans that differ by a wheel compare equal."""
    return sum(wheel.size_bytes for wheel in wheels)


def installed_bytes(wheels: Sequence[Wheel]) -> int:
    """What this plan occupies unpacked, or 0 if any wheel is unmeasured.

    All-or-nothing on purpose. A partial sum is a number that looks like an
    answer: it would under-report by whatever the unmeasured wheels are, and
    the unmeasured one is overwhelmingly likely to be torch, which is most of
    the total. Better to have no figure and say so.
    """
    if any(wheel.installed_bytes <= 0 for wheel in wheels):
        return 0
    return sum(wheel.installed_bytes for wheel in wheels)


def gib(byte_count: int) -> float:
    return byte_count / _GIB


def canonical_name(name: str) -> str:
    """PEP 503 normalisation: lowercase, runs of ``-_.`` collapsed to one ``-``.

    By hand rather than through ``packaging``, to keep this module's import
    list what its docstring says it is. It is four lines and it is the thing
    that makes ``huggingface-hub``, ``huggingface_hub`` and ``Huggingface.Hub``
    one key rather than three.
    """
    out = name.strip().lower()
    for ch in "_.":
        out = out.replace(ch, "-")
    while "--" in out:
        out = out.replace("--", "-")
    return out


def wheel_dist(filename: str) -> tuple[str, str]:
    """The (canonical name, version) a wheel filename declares.

    PEP 503 normalisation by hand rather than through ``packaging``, to keep
    this module's import list what its docstring says it is. A wheel name is
    ``name-version-python-abi-platform.whl`` with the name's own hyphens
    escaped as underscores, so the first two fields are unambiguous.
    """
    stem = filename[:-4] if filename.endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 2:
        raise ValueError(f"{filename!r} is not a wheel filename")
    return canonical_name(parts[0]), parts[1]


def conflicts(wheels: Sequence[Wheel], installed: Mapping[str, str]) -> list[str]:
    """Distributions this plan would *change* rather than add, worst first.

    **A pack is a delta over the base runtime, not a self-contained set**, and
    this is that sentence made checkable. It was learned by running one: the
    first attempt to resolve the rig pack into an empty interpreter failed
    because ``bpy`` needs ``numpy``, which is one of the thirty distributions
    base+studio already ships. The pack deliberately does not carry it.

    The consequence for installing is the mirror image. Every wheel in a pack
    is either absent from the runtime or present at exactly the version the
    pack was built from, because pack and base come out of one lock. Anything
    else means the two were built from *different* locks, and installing it
    would quietly re-version a package the running application has already
    imported -- numpy under the app's feet, in the worst case. That is not a
    thing to do halfway through, so it is refused whole, before the first
    byte, in ``fetch.disk_refusal``'s style.

    A wheel already installed at the pack's own version is not a conflict; it
    is simply nothing to do, which is what makes an interrupted install safe
    to run again.
    """
    bad: list[str] = []
    for wheel in wheels:
        name, version = wheel_dist(wheel.filename)
        have = installed.get(name)
        if have is not None and have != version:
            bad.append(f"{name} {have} would be replaced by {version}")
    return sorted(bad)


def to_install(wheels: Sequence[Wheel], installed: Mapping[str, str]) -> list[Wheel]:
    """The subset that is not already present at the version the pack carries.

    What makes a re-run after a failure cheap, and what stops a second install
    of an overlapping pack rewriting torch: ``music`` after ``text2image`` is
    47 wheels, not 74.
    """
    out: list[Wheel] = []
    for wheel in wheels:
        name, version = wheel_dist(wheel.filename)
        if installed.get(name) != version:
            out.append(wheel)
    return out


def disk_refusal(
    wheels: Sequence[Wheel], *, cache_dir: Path, install_dir: Path
) -> str | None:
    """Why this install must not start, or None. Refusing is the point.

    Two budgets against two paths, because they are routinely two volumes: the
    wheels are downloaded under the user's Warlock home and the packages are
    written into the application runtime, and a per-user install puts those
    wherever the user put them. Budgeting only the first is how a roomy home
    drive approves a 3 GB write to a full system one (MDL-09, in a new place).

    Both budgets are charged to the same volume when the two paths share one,
    and that is not double-counting: the wheels are still on disk while pip
    unpacks them, so the peak really is the sum.

    The install figure is omitted rather than guessed when the manifest did not
    measure it -- ``installed_bytes`` says why -- which leaves the download
    budget, the one that is always known.
    """
    if not wheels:
        return None
    download = gib(total_bytes(wheels))
    unpacked = gib(installed_bytes(wheels))
    items: list[tuple[Path, float]] = [(cache_dir, download)]
    if unpacked > 0:
        items.append((install_dir, unpacked))
    short = volume_refusal(items)
    if short is None:
        return None
    where = f" on {short.volume}" if short.volumes > 1 else ""
    return (
        f"Not enough disk space{where}: installing these needs about "
        f"{short.need_gib:.1f} GB (including {DISK_HEADROOM_GIB:.0f} GB "
        f"headroom) and {short.free_gib:.1f} GB is free."
    )

"""Collect the optional dependency packs the installer no longer has to carry.

``installer/build.ps1`` exports every extra at once and stages the lot, so the
shipped runtime contains torch, a CUDA runtime and a lyric-language stack
whether or not the user will ever ask for a picture or a tune. This script
produces the other arrangement: a directory of wheels per pack plus the
``packs.json`` that ``warlock.packs`` reads, so the base installer can ship the
app and ``studio`` alone and everything heavy can arrive chosen.

Run it after ``uv pip sync`` has populated the staged runtime -- the installed
tree is where the unpacked sizes come from, and there is nowhere else to get
them.

    uv run python scripts/make_packs.py --out build\\packs --python build\\stage\\python\\python.exe

Three things this script exists to get right, because each is a way the pack
comes out wrong while looking finished:

* **The resolver's answer is not this machine's answer.** ``uv export`` emits
  the whole resolution with PEP 508 markers still attached -- every
  ``nvidia-*`` and ``triton`` distribution under ``sys_platform == 'linux'``
  among them. A generator that took the lines at face value would put Linux
  CUDA wheels in a Windows pack. So every requirement is evaluated against
  ``TARGET_ENVIRONMENT``, which describes the runtime the installer *ships*
  rather than the machine doing the building.
* **The lock is not a size authority.** Every wheel uv resolves from PyPI
  carries a declared ``size``; the three from ``download.pytorch.org`` --
  ``torch``, ``torchaudio``, ``torchvision``, pinned as
  ``warlock.packs.SIZELESS_DISTS`` -- carry none, and they are most of the
  download. A missing key sums as zero, so trusting the lock would advertise
  about 1.4 GB for a 3 GB download. Every size written here is a ``stat`` of a
  file this script has in its hand.
* **Three distributions have no Windows wheel at all.** ``docopt``,
  ``mojimoji`` and ``unidic-lite`` are sdist-only, and ``mojimoji`` is a C
  extension. They are built into wheels *here*, on the build host that already
  compiles them during ``uv pip sync`` today, rather than shipped as sdists --
  the alternative puts a compiler in every user's install, which is not a
  thing a game-asset tool may require.

The output is validated by reading it back through ``warlock.packs`` before it
is written: a manifest the app's own reader refuses is not a pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from warlock import packs  # noqa: E402 -- after the path insert, deliberately

# The agent this project downloads under, and the incident that made it a
# named constant, both live in one module rather than being restated here:
# the ban that motivated it hit this script and `pack_worker` alike.
from warlock.pipelines import download as _download  # noqa: E402 -- same

# The environment the *shipped* runtime is, which is not necessarily the one
# this script runs in. ``installer/build.ps1`` stages a uv-managed CPython 3.13
# x86_64 for Windows and ``runtime-manifest.json`` pins that; these values are
# that pin restated in PEP 508's vocabulary.
#
# ``platform_release`` and ``platform_version`` are empty rather than absent
# because ``Marker.evaluate`` raises on a name it is not given, and a marker
# that consulted the Windows build number would be one this script must not
# guess at -- an empty string makes such a marker false and visible, rather
# than making the evaluation explode at pack time.
TARGET_ENVIRONMENT: dict[str, str] = {
    "implementation_name": "cpython",
    "implementation_version": "3.13.0",
    "os_name": "nt",
    "platform_machine": "AMD64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Windows",
    "platform_version": "",
    "python_full_version": "3.13.0",
    "python_version": "3.13",
    "sys_platform": "win32",
    "extra": "",
}

# What a wheel filename has to say to be installable into that runtime.
#
# Written out rather than taken from ``packaging.tags``, which enumerates the
# tags of the interpreter *running it*. That happens to be the right answer
# today, because the build host is Windows on 3.13 -- and it would go on being
# the right answer silently until the day someone built a pack from a different
# machine and got a pack of wheels for it.
#
# ``cp313t`` is excluded by the ABI set on purpose: the free-threaded build
# publishes wheels whose interpreter tag is also ``cp313``, so torch offers two
# Windows wheels and only the ABI distinguishes them.
#
# The interpreter set is not a plain allowlist, and assuming it was is a defect
# this script had for an hour: a stable-ABI wheel is tagged with the *oldest*
# interpreter it supports, not the newest -- ``safetensors`` ships
# ``cp310-abi3``, ``tokenizers`` ``cp39-abi3``, ``psutil`` ``cp37-abi3``. All
# three install into 3.13 perfectly well, and rejecting them sent four
# distributions down the build-it-from-source path for no reason. So
# ``_tag_rank`` accepts any ``cpXY-abi3`` whose XY is at or below the target.
TARGET_PYTHON = (3, 13)
TARGET_ABIS = frozenset({"cp313", "abi3", "none"})
TARGET_PLATFORMS = frozenset({"win_amd64", "any"})


class PackError(RuntimeError):
    """The pack cannot be built, and building a partial one is not an answer."""


# --- what the resolution says --------------------------------------------------


def parse_export(text: str) -> dict[str, str]:
    """The distributions a ``uv export`` selects *for the shipped runtime*.

    Keyed by canonical name, valued by version. Comment lines are uv's ``# via``
    annotations and its header; everything else is a pinned requirement that may
    carry a marker, and the marker is the whole reason this function exists
    rather than a regex -- see the module docstring's first bullet.
    """
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirement = Requirement(stripped)
        if requirement.marker is not None and not requirement.marker.evaluate(
            TARGET_ENVIRONMENT
        ):
            continue
        pins = [
            spec.version for spec in requirement.specifier if spec.operator == "=="
        ]
        if len(pins) != 1:
            # An export is a resolution: every line is one exact version. A
            # range here would mean the pack was built from something that
            # still had a choice left in it, and collecting either end of it
            # is a guess about what the installer would have picked.
            raise PackError(f"export line is not pinned to one version: {stripped}")
        version = pins[0]
        found[canonicalize_name(requirement.name)] = version
    return found


def export(extras: Sequence[str]) -> dict[str, str]:
    """Run ``uv export`` for these extras and read the answer.

    ``--frozen`` so the lock is used as it stands: a pack built from a
    resolution nobody committed is a pack nobody can reproduce.
    """
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
    ]
    for extra in extras:
        command += ["--extra", extra]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise PackError(f"uv export failed for {extras}: {done.stderr.strip()}")
    return parse_export(done.stdout)


def claims(base: dict[str, str], per_pack: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Which packs claim each distribution, by canonical name.

    A distribution the base installer already ships is claimed by nobody: the
    point of the exercise is what the base does *not* have to carry. Everything
    else is claimed by every pack whose closure contains it, which is how the
    27 distributions ``text2image`` and ``music`` share end up as one download
    rather than two (``warlock.packs.plan``).
    """
    out: dict[str, list[str]] = {}
    # Registry order rather than call order, and only the packs actually asked
    # for: ``--pack`` exists so one pack can be collected on its own, and a
    # partial run must produce a manifest that says so rather than one with
    # empty rows for the packs nobody collected.
    for key in (k for k in packs.KEYS if k in per_pack):
        closure = per_pack[key]
        for name in closure:
            if name in base:
                continue
            out.setdefault(name, []).append(key)
    for name, keys in out.items():
        versions = {per_pack[key][name] for key in keys}
        if len(versions) > 1:
            raise PackError(
                f"{name} resolves to {sorted(versions)} across packs; one "
                f"directory cannot hold two versions"
            )
    return out


# --- what the lock has to offer ------------------------------------------------


@dataclass(frozen=True, slots=True)
class Source:
    """One downloadable file the lock names."""

    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return urllib.parse.unquote(self.url.rsplit("/", 1)[-1])


@dataclass(frozen=True, slots=True)
class LockEntry:
    name: str
    version: str
    wheels: tuple[Source, ...]
    sdist: Source | None


def _source(raw: dict[str, Any]) -> Source | None:
    url = raw.get("url")
    digest = str(raw.get("hash", ""))
    if not isinstance(url, str) or not digest.startswith("sha256:"):
        return None
    return Source(url, digest.split(":", 1)[1])


def load_lock(path: Path) -> dict[str, LockEntry]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[str, LockEntry] = {}
    for raw in payload.get("package", []):
        wheels = tuple(w for w in (_source(one) for one in raw.get("wheels", [])) if w)
        sdist = _source(raw["sdist"]) if isinstance(raw.get("sdist"), dict) else None
        entry = LockEntry(raw["name"], raw["version"], wheels, sdist)
        out[canonicalize_name(entry.name)] = entry
    return out


def _interpreter_rank(interpreter: str, abi: str) -> int | None:
    """How close a wheel's interpreter tag is to the target, None if unusable.

    ``py3`` is any Python 3; ``cp313`` is this interpreter exactly; and
    ``cpXY`` beside ``abi3`` is "3.XY or newer", which the target satisfies
    whenever XY is at or below 13 -- see ``TARGET_ABIS``' comment for what
    reading that as an exact match cost.
    """
    major, minor = TARGET_PYTHON
    if interpreter == f"cp{major}{minor}":
        return minor
    if interpreter in {"py3", f"py{major}", f"py{major}{minor}"}:
        return minor if interpreter.endswith(str(minor)) else 0
    if abi == "abi3" and interpreter.startswith(f"cp{major}"):
        try:
            wanted = int(interpreter[len(f"cp{major}") :])
        except ValueError:
            return None
        return wanted if wanted <= minor else None
    return None


def _tag_rank(tag: Any) -> tuple[int, int, int] | None:
    """How closely one tag fits the shipped runtime, None if it does not fit."""
    if tag.abi not in TARGET_ABIS or tag.platform not in TARGET_PLATFORMS:
        return None
    interpreter = _interpreter_rank(tag.interpreter, tag.abi)
    if interpreter is None:
        return None
    return (
        1 if tag.platform != "any" else 0,
        {"cp313": 2, "abi3": 1, "none": 0}[tag.abi],
        interpreter,
    )


def is_target_wheel(filename: str) -> bool:
    """Whether this wheel installs into the runtime the installer ships."""
    try:
        _name, _version, _build, tags = parse_wheel_filename(filename)
    except Exception:  # noqa: BLE001 -- an unparseable name is simply not ours
        return False
    return any(_tag_rank(tag) is not None for tag in tags)


def _specificity(filename: str) -> tuple[int, int, int]:
    """How closely a wheel fits the target, most specific first.

    Installers pick the most specific compatible wheel, and this is that rule
    written down. It is needed because publishing *both* an accelerated build
    and a pure-Python fallback is normal -- ``charset-normalizer`` ships
    ``cp313-cp313-win_amd64`` beside ``py3-none-any``, and taking the second
    would quietly install the slow path.
    """
    _name, _version, _build, tags = parse_wheel_filename(filename)
    ranks = [rank for rank in (_tag_rank(tag) for tag in tags) if rank is not None]
    return max(ranks) if ranks else (0, 0, 0)


def select_wheel(entry: LockEntry) -> Source | None:
    """The wheel for the target runtime, None if the lock publishes none.

    A tie is a refusal rather than a coin toss. The lock lists four torch
    wheels for one version -- two platforms times two ABIs -- and the
    difference between the right one and the free-threaded one is four
    characters in a filename; picking either silently is how a runtime ends up
    with a wheel that imports on nobody's interpreter. (That particular pair is
    separated by ``TARGET_ABIS`` before it gets here; the refusal is for the
    pair nobody has thought of yet.)
    """
    candidates = [w for w in entry.wheels if is_target_wheel(w.filename)]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda w: _specificity(w.filename), reverse=True)
    if len(ranked) > 1 and _specificity(ranked[0].filename) == _specificity(
        ranked[1].filename
    ):
        raise PackError(
            f"{entry.name} {entry.version} has two equally specific wheels for "
            f"the target runtime: {sorted(c.filename for c in ranked[:2])}"
        )
    return ranked[0]


# --- getting the files themselves ----------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: Source, into: Path, *, offline: bool) -> Path:
    """Fetch one file by the URL the lock recorded, verifying its hash.

    By URL and not through a resolver: the lock already decided which artifact
    this is, and asking an index again is asking a second question that can
    come back with a different answer. A file already present with the right
    hash is left alone, so a rebuild after a failure does not re-download 3 GB.
    """
    target = into / source.filename
    if target.exists() and _sha256(target) == source.sha256:
        return target
    if offline:
        raise PackError(f"{source.filename} is not collected and --offline was given")
    into.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".part")
    with (
        urllib.request.urlopen(_download.request(source.url)) as response,
        staging.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    got = _sha256(staging)
    if got != source.sha256:
        staging.unlink(missing_ok=True)
        raise PackError(f"{source.filename}: sha256 is {got}, lock says {source.sha256}")
    # Staged and replaced rather than written in place, for the reason every
    # other served name in this tree is: an interrupted download must not leave
    # something that looks collected.
    staging.replace(target)
    return target


def build_from_sdist(entry: LockEntry, into: Path, *, offline: bool) -> Path:
    """Build a wheel for a distribution that publishes none for Windows.

    ``docopt``, ``mojimoji`` and ``unidic-lite`` are the three, and they are
    already built on this host during ``uv pip sync`` -- so the compiler this
    needs is one the build host has and no user's machine does. Shipping the
    sdists instead would move that requirement onto every install.
    """
    if entry.sdist is None:
        raise PackError(f"{entry.name} {entry.version} has neither a wheel nor an sdist")
    archive = download(entry.sdist, into / "sdists", offline=offline)
    with tempfile.TemporaryDirectory() as work:
        unpacked = Path(work)
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(unpacked)
        else:
            with tarfile.open(archive) as tarred:
                tarred.extractall(unpacked, filter="data")
        roots = [p for p in unpacked.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise PackError(f"{archive.name} does not unpack to one source tree")
        done = subprocess.run(
            ["uv", "build", "--wheel", str(roots[0]), "--out-dir", str(into)],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise PackError(f"building {entry.name} from its sdist failed: {done.stderr.strip()}")
    built = sorted(into.glob(f"{entry.name.replace('-', '_')}-{entry.version}-*.whl"))
    if len(built) != 1:
        raise PackError(f"building {entry.name} produced {len(built)} wheels, expected one")
    return built[0]


# --- what it all costs unpacked -------------------------------------------------


def installed_sizes(python: Path) -> dict[str, int]:
    """What each distribution occupies inside the staged runtime.

    Asked of the target interpreter rather than computed here, because
    ``importlib.metadata`` is the thing that knows how a dist-info directory is
    named and this script should not own a second opinion about it. Sizes come
    from ``RECORD``, which is what the installer wrote and therefore what an
    uninstall would remove.

    A distribution whose ``RECORD`` is missing or unreadable is simply absent
    from the answer, and ``warlock.packs.installed_bytes`` then withholds the
    whole install figure rather than reporting a short one.
    """
    program = """
import json, sys
from importlib.metadata import distributions
out = {}
for dist in distributions():
    name = dist.metadata["Name"]
    files = dist.files or []
    total = 0
    for one in files:
        try:
            total += (dist.locate_file(one)).stat().st_size
        except OSError:
            total = 0
            break
    if name and total:
        out[name] = total
json.dump(out, sys.stdout)
"""
    done = subprocess.run(
        [str(python), "-c", program], capture_output=True, text=True
    )
    if done.returncode != 0:
        raise PackError(f"could not measure the staged runtime: {done.stderr.strip()}")
    raw = json.loads(done.stdout or "{}")
    return {canonicalize_name(name): int(size) for name, size in raw.items()}


# --- the manifest ---------------------------------------------------------------


def build_manifest(
    *,
    lock: dict[str, LockEntry],
    claimed: dict[str, list[str]],
    out_dir: Path,
    unpacked: dict[str, int],
    offline: bool,
) -> dict[str, Any]:
    """Collect every claimed distribution and describe what was collected."""
    wheels: list[dict[str, Any]] = []
    for name in sorted(claimed):
        entry = lock.get(name)
        if entry is None:
            raise PackError(f"{name} is in the resolution but not in the lock")
        source = select_wheel(entry)
        if source is not None:
            path = download(source, out_dir, offline=offline)
            url = source.url
        else:
            path = build_from_sdist(entry, out_dir, offline=offline)
            # A wheel built here exists nowhere else, so there is no URL the
            # app could fetch it from. It is marked bundled instead, which
            # means the installer has to carry this file: three of them, and
            # `unidic-lite` is most of the weight. Writing a plausible-looking
            # URL would move the failure to install time, on a user's machine,
            # three wheels from the end of a music pack.
            url = ""
        wheels.append(
            {
                "filename": path.name,
                "url": url,
                "bundled": not url,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "installed_bytes": unpacked.get(name, 0),
                "packs": claimed[name],
            }
        )
    return {"version": packs.MANIFEST_VERSION, "wheels": wheels}


def report(manifest: dict[str, Any], collected: Sequence[str]) -> str:
    """What was collected, per pack, in the figures a user will be shown.

    Only the packs actually collected are listed. A row of zeroes for a pack
    this run never looked at reads exactly like a pack that came out empty.
    """
    lines = []
    parsed = packs.parse_manifest(manifest)
    for key in (k for k in packs.KEYS if k in collected):
        plan = packs.plan(parsed, [key])
        unpacked = packs.installed_bytes(plan)
        measured = (
            f"{packs.gib(unpacked):>6.2f} GiB installed" if unpacked else "  (unmeasured)"
        )
        lines.append(
            f"{key:<12} {len(plan):>4} wheels  "
            f"{packs.gib(packs.total_bytes(plan)):>6.2f} GiB download  {measured}"
        )
    if not {"text2image", "music"} <= set(collected):
        return "\n".join(lines)
    both = packs.plan(parsed, ["text2image", "music"])
    lines.append(
        f"{'t2i+music':<12} {len(both):>4} wheels  "
        f"{packs.gib(packs.total_bytes(both)):>6.2f} GiB download  "
        f"(shared distributions counted once)"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "build" / "packs",
        help="where the wheels and packs.json are written",
    )
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="the staged runtime to measure unpacked sizes in",
    )
    parser.add_argument("--lock", type=Path, default=ROOT / "uv.lock")
    parser.add_argument(
        "--pack", action="append", choices=list(packs.KEYS), dest="chosen",
        help="collect only this pack; repeatable, and every pack by default",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="refuse to fetch anything not already collected",
    )
    args = parser.parse_args(argv)

    chosen = packs.chosen_packs(args.chosen or packs.KEYS)
    base = export(["studio"])
    per_pack = {pack.key: export(["studio", *pack.extras]) for pack in chosen}
    claimed = claims(base, per_pack)
    lock = load_lock(args.lock)
    unpacked = installed_sizes(args.python)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        lock=lock, claimed=claimed, out_dir=args.out, unpacked=unpacked, offline=args.offline
    )
    # Read back through the app's own reader before writing: a manifest
    # ``warlock.packs`` refuses is not a pack, and finding that out at install
    # time is finding it out from the user.
    packs.parse_manifest(manifest)
    (args.out / packs.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(report(manifest, [pack.key for pack in chosen]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

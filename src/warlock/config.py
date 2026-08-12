"""Central configuration. Every path/port is env-overridable with WARLOCK_* vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import DEFAULT_BASE_MODEL as DEFAULT_T2I_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def source_checkout() -> bool:
    """Whether this package is running from a source checkout.

    **Warlock is source-checkout-only, and that is a decision rather than an
    omission** (DST-01, D4). Every native default below resolves against
    ``PROJECT_ROOT`` -- ``Path(__file__).parents[2]`` -- which is the repository
    root for an editable install and, inside a wheel, points somewhere under the
    environment (``Lib/vendor``). The wheel carries the package, the manual and
    the changelog but *not* ``vendor/``, which is git-ignored in full: the
    binaries are one-time manual downloads. So a non-editable install can never
    find the reconstruction engine, and the failure would arrive as a missing
    file at the first job rather than as anything anyone could act on.

    Supporting wheels properly means installing pinned native assets under a
    versioned app-data location or shipping them as package resources resolved
    through ``importlib.resources``. That is a distribution project, not a
    bugfix, and nothing asks for it today -- so the supported model is stated
    and checked instead of half-implemented.

    The probe is the marker that only a checkout has: ``pyproject.toml`` beside
    the package's parent. Deliberately not "is vendor/ present" -- a checkout
    that has not downloaded the binaries yet is a *supported* state with its own
    Doctor rows, and conflating the two would refuse to start over a missing
    optional file.
    """
    return (PROJECT_ROOT / "pyproject.toml").is_file()

# None on purpose, and confirmed by measurement -- see Config.trellis_band.
DEFAULT_TRELLIS_BAND: int | None = None


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).resolve()


def _home() -> Path:
    """The one directory the app owns on this machine.

    A module-level helper rather than a field other factories read, because a
    ``default_factory`` lambda cannot see its siblings -- every root below
    resolves this independently, and ``WARLOCK_HOME`` therefore moves all of
    them at once while each per-root variable still wins over it.
    """
    return _env_path("WARLOCK_HOME", Path.home() / ".warlock")


# Every environment variable this process could not make sense of, as
# ``(name, raw value, what was expected)``. Collected rather than raised, and
# this is the whole of RUN-03's fix.
#
# The failure it replaces: several fields called bare ``int()``/``float()``
# inside their ``default_factory``, so one typo in a port, a timeout or a
# threshold raised out of ``get_config()`` -- which runs before ``studio.main``
# has a log handler, before Doctor exists, and before there is a window to say
# anything in. The user got a traceback naming ``int()`` and no indication of
# which variable was wrong. Meanwhile *other* fields (the optional VRAM floats)
# silently turned a malformed value into ``None``, so an explicit safety limit
# with a typo in it became "unset" -- the two halves of one policy disagreeing
# about whether bad input is fatal or invisible.
#
# So: malformed input never raises and never vanishes. It falls back to the
# documented default and is recorded here, and Doctor reports every one of them
# in a single actionable row.
INVALID_ENV: list[tuple[str, str, str]] = []


def _note_invalid(name: str, raw: str, expected: str) -> None:
    entry = (name, raw, expected)
    if entry not in INVALID_ENV:
        INVALID_ENV.append(entry)


def _env_int(name: str, default: int) -> int:
    """``int`` from the environment, or the default with a note. Never raises."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        _note_invalid(name, raw, "a whole number")
        return default


def _env_float(name: str, default: float) -> float:
    """``float`` from the environment, or the default with a note. Never raises."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        _note_invalid(name, raw, "a number")
        return default


def _env_opt_int(name: str, default: int | None) -> int | None:
    """Like int(os.environ[name]) but with an explicit "leave it to the exe" value.

    Empty or "auto" means None, i.e. omit the flag entirely rather than passing
    a number, so trellis-server.exe applies its own heuristic.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in ("", "auto"):
        return None
    try:
        return int(text)
    except ValueError:
        # Recorded and defaulted, like every other numeric field. This used to
        # raise out of ``get_config()`` -- before any log handler, Doctor or
        # window existed -- over a typo in a texture resolution (RUN-03).
        _note_invalid(name, raw, "a whole number, or 'auto'")
        return default


def _env_opt_bool(name: str) -> bool | None:
    """A tri-state flag: unset (None) is distinguishable from an explicit off.

    Auto-detection has to be able to tell "the user chose coexist" from "the
    user chose nothing", and a bool with a default cannot.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw == "":
        return None
    return raw in ("1", "true", "on", "yes")


def _env_opt_float(name: str) -> float | None:
    """An optional float: unset is None, and so is a malformed value.

    ``None`` is still the fallback -- these are the VRAM limits, where "unset"
    genuinely is the default and there is no better number to substitute. What
    changed is that a malformed one is no longer *silent*: an explicit safety
    limit with a typo in it used to become "unset" with nothing said anywhere,
    which is the same policy inconsistency RUN-03 names from the other side.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        _note_invalid(name, raw, "a number")
        return None


@dataclass(slots=True)
class Config:
    # Everything the app generates lives under here -- assets, bench runs,
    # palettes and the model weights. A user's work is not a part of the source
    # tree: it has to survive a reinstall, a second checkout and a `git clean`,
    # and it has to be findable by somebody who never cloned the repo. The one
    # exception is the vendored native binaries (trellis-server.exe,
    # gltfpack.exe, warlockc.dll), which ship *with* the checkout and so stay
    # repo-relative below.
    home: Path = field(default_factory=_home)
    data_dir: Path = field(
        default_factory=lambda: _env_path("WARLOCK_DATA_DIR", _home() / "assets")
    )
    # Configurable apart from data_dir on purpose, and documented as such in
    # docs/manual/16-configuration.md: the library and its index are two
    # separate choices (an SSD index over a spinning-disk library is the case).
    db_path: Path = field(
        default_factory=lambda: _env_path("WARLOCK_DB", _home() / "assets" / "jobs.sqlite")
    )
    trellis_server_exe: Path = field(
        default_factory=lambda: _env_path(
            "WARLOCK_TRELLIS_EXE",
            PROJECT_ROOT / "vendor" / "trellis" / "trellis-server.exe",
        )
    )
    # Optional: a project folder assets can be copied straight into (e.g. a
    # Godot project's assets/). Unset means the feature is off and its routes
    # 404 -- writing outside data_dir is opt-in, never a default.
    export_dir: Path | None = field(
        default_factory=lambda: (
            _env_path("WARLOCK_EXPORT_DIR", PROJECT_ROOT)
            if os.environ.get("WARLOCK_EXPORT_DIR")
            else None
        )
    )
    # Vendored like trellis-server.exe: a pinned native binary, never downloaded
    # at runtime. Missing it costs you the triangle budgets, not the app --
    # jobs then ship the raw reconstruction, which is what they did before.
    gltfpack_exe: Path = field(
        default_factory=lambda: _env_path(
            "WARLOCK_GLTFPACK", PROJECT_ROOT / "vendor" / "gltfpack" / "gltfpack.exe"
        )
    )
    # Default triangle profile for a new job. See pipelines/optimize.PROFILES.
    #
    # "raw" and not a named tier: gltfpack is vendored and present now, so the
    # named tiers can run -- but none of them has been qualified (kept UVs,
    # both PBR maps and material assignment on a chest, a sword and a rock),
    # and a default that decimates every mesh through an unqualified tier is a
    # quality regression nothing on screen would explain. It is the only tier
    # the UI offers for the same reason. Set WARLOCK_MESH_PROFILE=standard to
    # exercise one; the retarget control in the inspector is the qualification
    # path, since it offers the whole list on a job that already exists.
    mesh_profile: str = field(
        default_factory=lambda: os.environ.get("WARLOCK_MESH_PROFILE", "raw")
    )
    # Whether a finished reference is scored against the run's conditioning
    # reference -- ref.png, which is the active profile's style anchor when one
    # is set and otherwise whatever image the user attached -- as well as
    # against its own composition report. On by default: the composition half
    # costs nothing (the report is already measured) and the anchor half only
    # runs when there is a ref.png and DINOv2 is on disk.
    rank_candidates: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_RANK", "on").lower()
        not in ("0", "false", "off", "no")
    )
    # How many extra times a text job may redraw its reference when the
    # composition report refuses the one it just drew.
    #
    # Was 0 -- off -- on the reasoning that a retry is four seconds of GPU
    # nobody asked for and the report's rules are heuristics, so a refusal is a
    # strong hint rather than a fact. Both halves are still true and neither is
    # what the setting costs. The 2026-08-07 rogue sweep refused 17 of 100
    # units at the composition gate, and a model-stage refusal is not a hint at
    # all there: the job *fails*. Seed 11's baseline was one of them, and losing
    # it did not cost one mesh -- ``findings.comparisons`` pairs rows sharing a
    # sweep, a source and a seed, so that baseline was one side of nine
    # prospective pairs and every one went with it.
    #
    # 2, and the reroll holds ``mesh_seed`` fixed and moves only
    # the reference seed, which is the point: a unit's identity in the corpus
    # is its config vector and its mesh seed, and which of three reference
    # draws avoided a character sheet is not a setting anyone is measuring.
    # The ceiling still ends the loop rather than a verdict, so past it the
    # user gets the last draw and the model stage gets its ordinary refusal.
    reference_retries: int = field(
        default_factory=lambda: max(0, _env_int("WARLOCK_REFERENCE_RETRIES", 2))
    )
    # How many extra times the trellis stage may run when the finished mesh
    # audits worse than mesh_hole_max. 0 -- off -- because a retry is two
    # minutes of GPU the user did not ask for, and a remesh is a reroll rather
    # than a repair: the second reconstruction can perfectly well be worse than
    # the first, which is why the loop keeps whichever attempt measured best
    # rather than whichever came last.
    mesh_retries: int = field(
        default_factory=lambda: max(0, _env_int("WARLOCK_MESH_RETRIES", 0))
    )
    # The worst-view see-through fraction past which a mesh is worth redoing.
    # 0.07 is measured, not guessed, the same way trellis_band was settled by a
    # sweep: docs/measurements/2026-08-04-hole-rate-baseline.md ran the whole
    # core-v1 suite at two seeds and found the hole rate sharply bimodal, with
    # *nothing at all* between 0.0308 and 0.1010 -- 22 of 37 meshes below the
    # gap and 15 above it, out to 0.556. 0.07 is the midpoint of that empty gap
    # rather than a percentile, so it is the value furthest from either cluster
    # and the one least disturbed by a future sample shifting one of them; any
    # threshold inside the gap selects the identical fifteen meshes.
    mesh_hole_max: float = field(
        default_factory=lambda: _env_float("WARLOCK_MESH_HOLE_MAX", 0.07)
    )
    # Where `python -m warlock.bench` writes its runs. A sibling of data_dir
    # rather than a child, on purpose: a benchmark run copies its artifacts
    # rather than referencing them, precisely so it survives prune_jobs -- and
    # a prune (or a Clean) that walked the library would collect them if they
    # lived inside it.
    bench_dir: Path = field(
        default_factory=lambda: _env_path("WARLOCK_BENCH_DIR", _home() / "bench")
    )
    # Where pixel-art palette files live (.hex from Lospec, .gpl from GIMP).
    # Ships empty: a palette is the user's own art direction, and a bundled one
    # would be a default nobody chose. Absent or empty simply means the palette
    # control offers nothing, never an error -- the same rule every optional
    # model directory follows.
    palette_dir: Path = field(
        default_factory=lambda: _env_path("WARLOCK_PALETTE_DIR", _home() / "palettes")
    )
    trellis_models_dir: Path = field(
        default_factory=lambda: _env_path(
            "WARLOCK_TRELLIS_MODELS", _home() / "models" / "trellis2-gguf"
        )
    )
    trellis_port: int = field(
        default_factory=lambda: _env_int("WARLOCK_TRELLIS_PORT", 17971)
    )
    # Seconds of queue inactivity before the trellis server is stopped to free VRAM.
    trellis_idle_timeout: float = field(
        default_factory=lambda: _env_float("WARLOCK_TRELLIS_IDLE", 600.0)
    )
    # Where every image model lives: models.BASE_MODELS[k].dir_name resolves
    # against this, and style LoRAs against its loras/ subdirectory. All
    # downloaded once by hand (see README) -- the app never downloads, and
    # loads are local_files_only.
    t2i_model_root: Path = field(
        default_factory=lambda: _env_path("WARLOCK_T2I_ROOT", _home() / "models")
    )
    # Pre-registry override, still honoured: points the *turbo* entry at an
    # arbitrary local diffusers dir so existing setups keep working. Other base
    # models always resolve under t2i_model_root.
    t2i_turbo_dir: Path | None = field(
        default_factory=lambda: (
            _env_path("WARLOCK_T2I_DIR", PROJECT_ROOT)
            if os.environ.get("WARLOCK_T2I_DIR")
            else None
        )
    )
    # Base model used when a job doesn't name one. Sampler settings are not
    # configurable here on purpose -- they belong to the checkpoint (models.py).
    t2i_model: str = field(
        default_factory=lambda: os.environ.get(
            "WARLOCK_T2I_MODEL", DEFAULT_T2I_MODEL
        )
    )
    # trellis-server.exe's WebP textures declare EXT_texture_webp as required,
    # which Godot's glTF importer does not implement (it refuses the file
    # rather than skip the extension). Off is the correct default.
    trellis_webp: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_TRELLIS_WEBP", "off").lower()
        in ("1", "true", "on")
    )
    # trellis-server.exe's "auto" texture PBR resolution bakes visible per-texel
    # noise into the baseColor atlas at --res 1024/1536 (reproduced via
    # trellis-cli.exe: default auto is noise, explicit --tex-res 512 is clean).
    # Pin it to 512 until upstream fixes the auto heuristic.
    trellis_tex_res: int = field(
        default_factory=lambda: _env_int("WARLOCK_TRELLIS_TEX_RES", 512)
    )
    # Width of the narrow band the DC remesh runs over. The exe defaults it to
    # res/512 when the flag is absent, which is what None gives you.
    #
    # Measured 2026-08-01 (`warlock sweep`, one reference image, seed 42,
    # res 1024, hole_fraction @ 1024) -- worst-view see-through fraction:
    #
    #     auto  0.0077   267,360 faces   123 s   (res/512 == band 2 here)
    #     2     0.0077   266,632 faces   143 s
    #     4     0.0167   290,774 faces   124 s
    #     8     0.0110   297,898 faces   136 s
    #     16    0.0125   289,586 faces   193 s
    #
    # Two conclusions, both against the earlier guess that a wider band would
    # help: the heuristic is already the best of the ladder, and widening makes
    # the surface *more* perforated while adding faces and time. So the flag
    # stays off. The run also puts a floor under what counts as a real
    # difference: auto and 2 are the same setting, and still disagreed by 728
    # faces, so anything under ~0.3% is noise.
    #
    # This supersedes an earlier note claiming the default left "~1300
    # disconnected plates, 7-31% of the silhouette". Nothing in this sweep
    # measured worse than 1.7%, so whatever produced those numbers was not
    # this exe at these settings. Re-measure before acting on that claim.
    trellis_band: int | None = field(
        default_factory=lambda: _env_opt_int("WARLOCK_TRELLIS_BAND", DEFAULT_TRELLIS_BAND)
    )
    # Tri-state, and the None is the point. Unset means "decide from the card":
    # studio.runtime resolves it through vram.plan() at startup and writes a
    # plain bool back here before the Worker exists, so queue.py keeps reading
    # a bool and the "read once at startup" contract is unchanged. Set
    # explicitly (1/true/on or 0/false/off) it is honoured verbatim and
    # auto-detection never overrides it.
    #
    # False (coexist) keeps an SDXL-class pipe (~7 GB) and trellis-server (~16 GB)
    # resident together, which is right on a 32 GB card and is what a 32 GB
    # card still auto-selects. True restores the stop-trellis -> run-SDXL ->
    # unload -> restart handoff for OOM situations (resolution 1536, smaller
    # GPUs, a resident Flux).
    vram_exclusive: bool | None = field(
        default_factory=lambda: _env_opt_bool("WARLOCK_VRAM_EXCLUSIVE")
    )
    # Whether the value above came from the environment. Captured because the
    # resolve destroys the evidence: writing a plain bool back over the
    # tri-state makes "was this chosen or configured" underivable, so every
    # later plan() -- every health poll -- reported an auto-selected mode as
    # set by an environment variable nobody set. None means "not resolved yet",
    # i.e. infer it from vram_exclusive.
    vram_exclusive_explicit: bool | None = None
    # What one job may ask the card for, in GiB. None = device total minus
    # vram.HEADROOM_GIB, which is what you want unless the driver misreports.
    vram_budget_gib: float | None = field(
        default_factory=lambda: _env_opt_float("WARLOCK_VRAM_BUDGET")
    )
    # Pretend the card is this large. The escape hatch for a torch-less install
    # -- and what makes the whole admission gate testable without a GPU.
    vram_total_gib: float | None = field(
        default_factory=lambda: _env_opt_float("WARLOCK_VRAM_TOTAL")
    )
    # Whether a rig may read its joint positions off the reference image the
    # mesh was reconstructed from (pipelines/pose2d) instead of scaling the
    # template onto the bounding box.
    #
    # On, and it is a kill-switch rather than an opt-in because the thing it
    # switches off is already the fallback: with no weights on disk, a
    # non-humanoid template, no reference image, or a detection that fails any
    # sanity gate, a rig is byte-for-byte what it was before this existed. So
    # there is no state in which turning it *on* is a risk the user should have
    # to accept deliberately -- but a wrong skeleton is hard to see and easy to
    # blame on something else, so there has to be one flag that takes the whole
    # feature out of the picture while it is being judged.
    pose_fit: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_POSE_FIT", "on").lower()
        not in ("0", "false", "off", "no")
    )
    # Skeleton template a rig request falls back to when it doesn't name one.
    # Validated against rigging.templates() at request time, not here -- config
    # is imported by everything and must not pull the template registry in.
    rig_template: str = field(
        default_factory=lambda: os.environ.get("WARLOCK_RIG_TEMPLATE", "humanoid")
    )
    # Wall-clock ceiling for one Blender subprocess. Automatic weights on a
    # 300k-face mesh are minutes of CPU; anything past this is a hang, and a
    # hung bpy holds the single-worker queue against every other job.
    rig_timeout: float = field(
        default_factory=lambda: _env_float("WARLOCK_RIG_TIMEOUT", 1800.0)
    )
    # Baking a pose is an import, a handful of quaternion assignments and an
    # export -- seconds, not minutes. It gets its own much tighter ceiling
    # because it runs inline in a request rather than on the queue.
    pose_timeout: float = field(
        default_factory=lambda: _env_float("WARLOCK_POSE_TIMEOUT", 300.0)
    )
    # Render the deformation battery after a rig. A kill-switch rather than an
    # opt-in, for the reason pose_fit is one: with it off a rig is byte for
    # byte what it was before this existed, so there is no state in which
    # turning it *on* is a risk to accept deliberately -- but it is a dozen
    # extra EEVEE frames on the serial queue, and a user who does not review
    # rigs should be able to stop paying for them.
    deform_qa: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_DEFORM_QA", "on").lower()
        not in ("0", "false", "off", "no")
    )
    # A sheet is one EEVEE render per cell -- 8 yaws times however many poses.
    # Generous because the cell count is user-chosen, but still bounded: this
    # runs on the serial queue and a hang would block every later job.
    sheet_timeout: float = field(
        default_factory=lambda: _env_float("WARLOCK_SHEET_TIMEOUT", 1800.0)
    )

    @property
    def autosave_dir(self) -> Path:
        """Where Inker's crash-safety copies live.

        Under ``data_dir`` rather than beside each document: a document may
        have no path at all, and one that does may sit on a network share or a
        read-only directory -- neither of which is a good place to be writing
        every two minutes. One directory also means recovery is a listing
        rather than a search.
        """
        return self.data_dir / "autosave"

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id


# Which environment variable each field answers to (S140). A table rather than
# something derived, because the derivation does not exist: every default is a
# ``default_factory`` lambda closing over its own variable name, and there is no
# way to ask a dataclass field which string its factory read.
#
# The pairing is asserted in both directions by a test -- a field with no entry
# and an entry naming no field both fail -- which is what keeps this from
# becoming the usual stale second copy. ``vram_exclusive_explicit`` is the one
# deliberate omission: it is not configured, it is *derived* by the startup
# resolve, and listing it beside real settings would present a computed answer
# as something the user set.
SETTINGS: tuple[tuple[str, str], ...] = (
    ("home", "WARLOCK_HOME"),
    ("data_dir", "WARLOCK_DATA_DIR"),
    ("db_path", "WARLOCK_DB"),
    ("bench_dir", "WARLOCK_BENCH_DIR"),
    ("palette_dir", "WARLOCK_PALETTE_DIR"),
    ("export_dir", "WARLOCK_EXPORT_DIR"),
    ("trellis_server_exe", "WARLOCK_TRELLIS_EXE"),
    ("trellis_models_dir", "WARLOCK_TRELLIS_MODELS"),
    ("trellis_port", "WARLOCK_TRELLIS_PORT"),
    ("trellis_idle_timeout", "WARLOCK_TRELLIS_IDLE"),
    ("trellis_webp", "WARLOCK_TRELLIS_WEBP"),
    ("trellis_tex_res", "WARLOCK_TRELLIS_TEX_RES"),
    ("trellis_band", "WARLOCK_TRELLIS_BAND"),
    ("gltfpack_exe", "WARLOCK_GLTFPACK"),
    ("mesh_profile", "WARLOCK_MESH_PROFILE"),
    ("mesh_retries", "WARLOCK_MESH_RETRIES"),
    ("mesh_hole_max", "WARLOCK_MESH_HOLE_MAX"),
    ("reference_retries", "WARLOCK_REFERENCE_RETRIES"),
    ("rank_candidates", "WARLOCK_RANK"),
    ("t2i_model", "WARLOCK_T2I_MODEL"),
    ("t2i_model_root", "WARLOCK_T2I_ROOT"),
    ("t2i_turbo_dir", "WARLOCK_T2I_DIR"),
    ("vram_exclusive", "WARLOCK_VRAM_EXCLUSIVE"),
    ("vram_budget_gib", "WARLOCK_VRAM_BUDGET"),
    ("vram_total_gib", "WARLOCK_VRAM_TOTAL"),
    ("pose_fit", "WARLOCK_POSE_FIT"),
    ("pose_timeout", "WARLOCK_POSE_TIMEOUT"),
    ("rig_template", "WARLOCK_RIG_TEMPLATE"),
    ("rig_timeout", "WARLOCK_RIG_TIMEOUT"),
    ("deform_qa", "WARLOCK_DEFORM_QA"),
    ("sheet_timeout", "WARLOCK_SHEET_TIMEOUT"),
)

# The other half of the environment, and the reason the readout was incomplete.
#
# ``SETTINGS`` can only name ``Config`` fields -- the test above it asserts that
# in both directions -- so four variables that change what the process does were
# invisible to ``warlock doctor`` and to the Settings pane simply because no
# field carries them. They are read at their point of use, from ``os.environ``
# directly: ``migrate.py`` for the two home-migration switches, ``native.py``
# for the two kernel ones. A host that silently ran the numpy fallbacks, or that
# skipped a migration, gave the same readout as one that did not, which is
# exactly the failure ``from_env`` exists to prevent.
#
# They are a separate table rather than entries in ``SETTINGS`` because their
# value is the raw string in the environment, not a resolved field: there is no
# default to report, only set or unset. ``effective`` appends them last.
SWITCHES: tuple[tuple[str, str], ...] = (
    ("no_migrate", "WARLOCK_NO_MIGRATE"),
    ("migrate_keep", "WARLOCK_MIGRATE_KEEP"),
    ("native", "WARLOCK_NATIVE"),
    ("native_dll", "WARLOCK_NATIVE_DLL"),
)


@dataclass(frozen=True, slots=True)
class Setting:
    """One configured value, and where it came from."""

    name: str
    env: str
    value: str
    from_env: bool


def effective(config: Config | None = None) -> list[Setting]:
    """What this process is actually running on, field by field (S140).

    The point is the ``from_env`` column, not the values: "trellis port 17971"
    is not a diagnosis, and "trellis port 17971, set by WARLOCK_TRELLIS_PORT"
    is -- a host whose behaviour disagrees with the documentation almost always
    disagrees because something in its environment says so, and until this
    there was no way to see that from inside the app.

    Reads the environment rather than comparing against a freshly constructed
    ``Config``: the comparison would report "default" for a variable explicitly
    set to the default value, which is exactly the case where a user is asking
    whether their setting took.

    Both tables, ``SETTINGS`` first and then ``SWITCHES``: a field-backed
    setting reports its resolved value, an env-only switch reports the raw
    string or ``(unset)``, and the ``from_env`` column means the same thing in
    either case.

    Pure, in the ``vram.py`` sense -- stdlib only, no imports from ``service``,
    ``queue`` or ``studio`` -- because both the CLI and a pane read it.
    """
    config = config or get_config()
    out: list[Setting] = []
    for name, env in SETTINGS:
        value = getattr(config, name, None)
        out.append(
            Setting(
                name=name,
                env=env,
                value="(unset)" if value is None else str(value),
                from_env=env in os.environ,
            )
        )
    for name, env in SWITCHES:
        raw = os.environ.get(env)
        out.append(
            Setting(
                name=name,
                env=env,
                value="(unset)" if raw is None else raw,
                from_env=raw is not None,
            )
        )
    return out


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        cfg = Config()
        # Before any mkdir: the migration's "destination already exists and is
        # non-empty, so leave it alone" rule has to see the destination as the
        # user left it, and creating data_dir first would make every first run
        # look like a half-finished move.
        from . import migrate

        migrate.run(cfg)
        # palette_dir and t2i_model_root are created here and not lazily: on a
        # fresh install both are directories the user has to put files into by
        # hand, and an existing empty folder is the only instruction that
        # survives the README being closed. bench_dir stays lazy -- its two
        # writers already mkdir, and an empty bench/ says nothing to anyone.
        for d in (cfg.home, cfg.data_dir, cfg.palette_dir, cfg.t2i_model_root):
            d.mkdir(parents=True, exist_ok=True)
        _config = cfg
    return _config

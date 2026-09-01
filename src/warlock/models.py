"""The image-model registry: which base models and style LoRAs a job may pick.

Kept separate from guidance.py on purpose. That module owns *prompt fragments*
-- taxonomy that only ever ends up as text. Model identity is a different
concern: it decides what is resident in VRAM, what has to be downloaded by
hand, and what doctor.py reports on. Mixing the two would blur a boundary
guidance.py's docstring states explicitly.

Everything here is optional and independently skippable. A missing base model
fails that one job with its download command; a missing LoRA is skipped at load
time. Nothing here is ever downloaded by the app process -- see the offline
invariant in __init__.py; the one exception is an explicit, user-initiated
fetch, which runs out-of-process (``pipelines/fetch_worker``) from the ``Fetch``
records below.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The one blessed base download, and therefore what an unconfigured job runs.
#
# Not turbo, and measured rather than preferred: in the 2026-08-09 re-baseline
# `base_model=turbo` took 1 of 5 -- the weakest checkpoint that produced data --
# while `base_model=sdxl_cfg` took 3 of 3, the best survival rate of any arm
# with data. The two rank oppositely on refusals and on quality (both of that
# run's composition refusals fell on SDXL-family arms at full CFG), and this
# picks quality: a refusal is a sentence the user can act on, a poor mesh is two
# minutes of the serial worker spent on something they will discard.
#
# It is also the distribution decision. One 7 GiB download of
# stabilityai/stable-diffusion-xl-base-1.0 is shared by four recipes -- sdxl,
# sdxl_cfg, pixel, lightning -- so shipping this one entry unlocks the speed
# recipes for a small LoRA each, and pixel sheets for 0.2 GB more. Recorded in
# docs/measurements/2026-08-11-default-base-model.md.
DEFAULT_BASE_MODEL = "sdxl_cfg"

# The one key WARLOCK_T2I_DIR redirects. Pinned by name, not to the default:
# the variable predates the registry and docs/MODELS.md documents it as the
# FLUX-proper workaround against the turbo entry's settings specifically. Tying
# it to DEFAULT_BASE_MODEL made the redirect follow whichever model happened to
# be the default, which would silently point an existing setup's override at a
# checkpoint it was never about.
T2I_DIR_MODEL = "turbo"

# Which architecture a checkpoint is, and therefore which encode/sample path
# runs. Declared per model rather than sniffed, for the same reason
# BaseModel.controlnet is: a future checkpoint must not become "SDXL-shaped"
# because it happened to ship two text encoders and a unet/.
FAMILY_SDXL = "sdxl"
FAMILY_FLUX2_KLEIN = "flux2klein"
FAMILIES = (FAMILY_SDXL, FAMILY_FLUX2_KLEIN)

# How a pipeline is placed on the device.
#
# "resident" is .to("cuda") -- the whole pipe, which is what every SDXL-class
# checkpoint has always done. "offload" is enable_model_cpu_offload(), which
# keeps one submodule on the device at a time; the two are mutually exclusive
# (accelerate's hooks assume the modules start on the host, so a preceding
# .to("cuda") defeats the whole mechanism).
RESIDENT = "resident"
OFFLOAD = "offload"

# LoRA adapter names are per-pipeline, so the always-on step-distillation
# adapter needs a name no style LoRA key can collide with.
BASE_LORA_ADAPTER = "_base"

LORA_WEIGHT_MIN = 0.0
LORA_WEIGHT_MAX = 1.5
DEFAULT_LORA_WEIGHT = 0.9

# Conditioning strengths, bounded the same way LoRA weight is: the API's
# 400-on-out-of-range comes from one place, and the UI's sliders read their
# ends from here rather than repeating numbers.
IP_SCALE_MIN = 0.0
IP_SCALE_MAX = 1.5
DEFAULT_IP_SCALE = 0.6
CONTROL_SCALE_MIN = 0.0
CONTROL_SCALE_MAX = 2.0
DEFAULT_CONTROL_SCALE = 0.65
# How far an img2img denoise is taken. Below ~0.3 the init image survives
# essentially unchanged and above ~0.65 it stops constraining the result, which
# for a restyle is the whole point of starting from one. The effective step
# count is steps x strength, which is why a 4-step distilled base cannot do
# img2img at all: 0.45 of four steps is under two.
IMG2IMG_STRENGTH_MIN = 0.30
IMG2IMG_STRENGTH_MAX = 0.65
DEFAULT_IMG2IMG_STRENGTH = 0.45

# The re-texture door's own bounds and default, deliberately not a change to
# the shared values above: the pixel sheets run img2img with no geometry
# anchor and keep 0.45/0.65. A re-texture can afford 0.85 -- the 2026-08-08
# retexture measurement's positive control showed the projection carries what
# 0.85 invents faithfully, and with the depth ControlNet anchoring structure,
# invention is the point. The default is the 2026-08-15 retexture-visibility
# ladder's pick (docs/measurements/2026-08-15-retexture-visibility.md):
# monotone and pathology-free to 0.85 under the anchor, 0.65 sits in the
# pre-registered landing zone and is the old un-anchored ceiling made
# comfortable.
RETEXTURE_STRENGTH_MIN = IMG2IMG_STRENGTH_MIN
RETEXTURE_STRENGTH_MAX = 0.85
RETEXTURE_DEFAULT_STRENGTH = 0.65

# The style LoRA a pixel sheet restyle is fixed to in v1. Named here rather
# than in the sheet code so the registry stays the only place a model key is
# written down.
PIXEL_SHEET_LORA = "pixelxl"

# How far into the denoise the ControlNet keeps acting. Ending early lets the
# last steps add detail the hint image never had; 1.0 holds the structure to
# the final step and tends to look traced.
CONTROL_END_MIN = 0.0
CONTROL_END_MAX = 1.0
DEFAULT_CONTROL_END = 0.8


# Where a pasted command should send weights when there is no ``Config`` to
# resolve against -- the README's spelling of the default model root, which is
# ``Config.t2i_model_root``'s default (``~/.warlock/models``) written the way a
# PowerShell reader can paste it. ``$HOME`` rather than ``~`` because PowerShell
# expands the variable anywhere in an argument and only expands ``~`` at the
# start of a path.
#
# This is a *display* default and nothing resolves through it: an actual fetch
# goes through ``fetch.destination``, and anything rendering a command for a
# live app should go through ``fetch.download_text`` so the two cannot disagree.
DEFAULT_MODEL_ROOT_TEXT = "$HOME/.warlock/models"


@dataclass(frozen=True, slots=True)
class Fetch:
    """One repository fetch: what to get, where it lands, roughly how big.

    Structure rather than prose, because the prose has two readers that cannot
    both be satisfied by a string. Doctor prints it for a person to paste, and
    the in-app download button has to *execute* it -- and several entries are
    multi-command, one carries a rename and one carries a ``uv sync``, so
    executing the string blindly would be wrong. So the string is derived from
    this (``download_text``) and the executor reads the fields, which is what
    keeps doctor's text, the README and what the button actually does from
    drifting apart again.

    ``local_dir`` is relative to the model root. Rendering it needs to know
    where that root actually is, and this module cannot ask: ``config`` imports
    ``models``, not the other way round. So ``command``/``steps`` take the
    resolved destination as an argument and fall back to
    ``DEFAULT_MODEL_ROOT_TEXT`` when nobody supplies one --
    ``fetch.download_text(config, kind, spec)`` is the caller that does, and it
    resolves through ``fetch.destination`` so ``WARLOCK_T2I_ROOT`` and the
    ``turbo``-only ``WARLOCK_T2I_DIR`` override are both honoured.

    The literal used to be ``models/<local_dir>``, which was a *relative* path:
    pasting a command from the docs into any shell whose working directory was
    not the checkout put gigabytes somewhere the app never looks, and the
    one-time home migration skips a root that already holds something, so on any
    machine that had ever fetched in-app the stranding was permanent and doctor
    kept reporting the model missing.

    ``size_gib`` is approximate and exists for two things only: the progress
    bar's denominator and the free-disk refusal. Understating it weakens the
    refusal and never causes a wrong one, which is the direction to err in.
    """

    repo_id: str
    local_dir: str
    # The immutable commit this fetch pins, or "" for the repository's current
    # default branch.
    #
    # Unpinned, ``snapshot_download`` follows whatever ``main`` points at today,
    # so a Download click a year from now can retrieve different weights -- and,
    # for BiRefNet, different *Python*, which is loaded with
    # ``trust_remote_code=True``. The fetch child is crash and RSS isolation,
    # not a sandbox: it runs with the user's filesystem and network permissions,
    # and ``uv.lock`` says nothing about code downloaded at runtime (MDL-03).
    #
    # A string on the record rather than a lookup elsewhere, so the pin travels
    # with the thing it pins: the planner puts it in the job, the worker passes
    # it to ``snapshot_download``, and ``download_text`` renders it into the
    # paste-able command -- which is what stops a hand-run download from
    # quietly being a different one.
    #
    # Empty means "not pinned yet" and is honest about it rather than pretending
    # a default branch is a version. ``tests/test_fetch.py`` pins which entries
    # are expected to carry one.
    revision: str = ""
    # Explicit paths inside the repo, passed positionally to `hf download`.
    filenames: tuple[str, ...] = ()
    # The --include / --exclude sets.
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    # A post step: (name-as-downloaded, name-it-must-have). loras/ is flat, so
    # a repo's default-named adapter would otherwise overwrite another's.
    rename: tuple[str, str] | None = None
    size_gib: float = 0.0
    # A trailing non-shell instruction, reproduced verbatim. Only BiRefNet has
    # one, and it names a `uv sync` rather than a download.
    note: str = ""

    def dest_text(self, dest: str | None = None) -> str:
        """What ``--local-dir`` should say: the caller's resolved path, or the
        documented default home when there is no config to resolve against."""
        return dest if dest is not None else f"{DEFAULT_MODEL_ROOT_TEXT}/{self.local_dir}"

    def command(self, dest: str | None = None) -> str:
        """The one-line `hf download` this record stands for.

        --include is repeated per pattern deliberately; see the note on
        BaseModel below.
        """
        parts = [f"uvx hf download {self.repo_id}"]
        if self.revision:
            # Rendered into the pasted command as well as passed to the
            # in-app fetch: a hand-run download that silently took the branch
            # tip would defeat the pin for exactly the user most likely to be
            # reproducing something (MDL-03).
            parts.append(f"--revision {self.revision}")
        parts.extend(self.filenames)
        parts.extend(f'--include "{pat}"' for pat in self.allow_patterns)
        parts.extend(f'--exclude "{pat}"' for pat in self.ignore_patterns)
        parts.append(f"--local-dir {self.dest_text(dest)}")
        return " ".join(parts)

    def steps(self, dest: str | None = None) -> list[str]:
        """Every line a person has to run for this record, in order."""
        out = [self.command(dest)]
        if self.rename is not None:
            src, dst = self.rename
            out.append(f"then rename {self.dest_text(dest)}/{src} to {dst}")
        if self.note:
            out.append(self.note)
        return out


def fetch_dests(
    fetches: Iterable[Fetch],
    *,
    root: Path,
    dir_name: str | None = None,
    base_dir: Path | None = None,
) -> list[Path]:
    """Where each record lands, given *already resolved* directories.

    The rule, in one place: a base model's own weights go to that model's
    directory -- which is not always ``root / dir_name``, because the legacy
    ``WARLOCK_T2I_DIR`` relocates the ``turbo`` entry alone -- and everything
    else (a step-distillation LoRA into the flat ``loras/``, an adapter, a
    metric) resolves flat under the root.

    Taking resolved paths rather than a ``Config`` is what lets both callers
    share it: ``fetch.destination`` resolves them from config, and
    ``pipelines/text2image.py`` already holds its own two directories and must
    not import config (no pipeline does). A second copy of "where does turbo
    live" is exactly how the paste-able command and the button came to disagree
    in the first place.
    """
    out: list[Path] = []
    for one in fetches:
        own = dir_name is not None and base_dir is not None and one.local_dir == dir_name
        out.append(base_dir if own else root / one.local_dir)  # type: ignore[arg-type]
    return out


def download_text(fetches: Iterable[Fetch], dests: Iterable[str | None] | None = None) -> str:
    """The prose ``spec.download`` used to be a literal of.

    Two spaces of continuation indent, because doctor prints it after
    ``download with:\\n  `` and every line of a multi-command entry has to line
    up with the first.

    ``dests`` pairs one resolved destination with each fetch, in order. Only
    ``fetch.download_text`` passes it; ``spec.download`` renders the default
    home, which is what the docs say and what a reader with no app running can
    paste. See the note on ``Fetch``.
    """
    fetches = tuple(fetches)
    paths = tuple(dests) if dests is not None else (None,) * len(fetches)
    return "\n  ".join(
        step for f, dest in zip(fetches, paths, strict=True) for step in f.steps(dest)
    )


@dataclass(frozen=True, slots=True)
class BaseModel:
    """A diffusers text2image checkpoint plus the settings it must be run at.

    Sampler settings are part of the model's identity, not user preference: a
    4-step distilled model run at 25 steps with CFG produces mush, and Hyper-SD
    silently degrades unless its scheduler uses trailing timestep spacing. So
    steps/guidance/scheduler travel with the checkpoint.
    """

    # NOTE on the download strings below: --include must be repeated per
    # pattern. The space-separated form (--include "*.json" "*.txt" ...) is
    # accepted by the CLI but only the *last* pattern takes effect, which
    # fetches the safetensors and silently leaves out every config.json --
    # producing a directory that looks downloaded and fails the
    # model_index.json check. Verified against hf 0.36 on 2026-08-01.
    key: str
    label: str
    dir_name: str
    image_size: int
    steps: int
    guidance_scale: float
    variant: str | None = "fp16"
    # A name pipelines/text2image._scheduler knows (it raises on any other);
    # None keeps whatever the checkpoint's own scheduler_config.json specifies.
    scheduler: str | None = None
    # A step-distillation LoRA fused on at load, never user-facing.
    base_lora: str | None = None
    # Whether a ControlNet may be attached to this checkpoint. Explicit rather
    # than derived from guidance_scale: a ControlNet at guidance 0 on a 4-step
    # distilled base fights the structure hint instead of honouring it, and a
    # future base at CFG 1.5 must not silently become "controllable" because it
    # cleared a threshold nobody qualified it against.
    controlnet: bool = False
    # Perturbed-Attention Guidance (arXiv 2403.17377), a training-free sampling
    # upgrade: the self-attention map is perturbed to make a second "degraded"
    # prediction, and the sample is steered away from it -- structure gets
    # cleaner at no new weights. 0.0 means off, and off means *class-identical*:
    # the loader only asks diffusers for a PAG pipeline when this is set, so
    # every existing recipe's path is byte-for-byte untouched. SDXL-family
    # only (the perturbation targets UNet self-attention); the loader refuses
    # a non-SDXL spec that sets it rather than sampling something undefined.
    pag_scale: float = 0.0
    # diffusers' guidance_rescale (Lin et al. 2023, "Common Diffusion Noise
    # Schedules and Sample Steps are Flawed"): rescales the CFG output's
    # variance to fix the washed-out look of high-CFG sampling. 0.0 means the
    # kwarg is never passed -- the same absence-not-default rule as pag_scale,
    # for the same bit-identity reason.
    guidance_rescale: float = 0.0
    # Which architecture this is, from FAMILIES. Everything the image half of
    # the app does -- chunked CLIP encoding, the pooled embeddings, style
    # LoRAs, ControlNet, IP-Adapter, img2img -- is an SDXL fact, so a
    # non-"sdxl" family takes a different sample path and loses all of it.
    family: str = FAMILY_SDXL
    # RESIDENT or OFFLOAD. A 16 GB checkpoint fully resident cannot coexist
    # with trellis-server on a 32 GB card; offloaded, its peak is roughly the
    # larger of its two big submodules and it fits.
    residency: str = RESIDENT
    # Peak device footprint under ``residency``, in GiB -- what vram.estimate
    # charges a text job for this checkpoint. Deliberately conservative:
    # refusing a job is the good outcome.
    #
    # The checkpoint *alone*: adapters are priced separately by
    # ``vram._adapter_cost``, because which ones are on the device depends on
    # the job rather than on the spec. This number excluded them entirely until
    # 2026-08-12, which under-priced every distilled recipe by the ~0.8 GiB of
    # its required step-distillation LoRA -- the one direction the "deliberately
    # conservative" above forbids (MDL-17).
    vram_gib: float = 7.0
    # Peak *host* commit this checkpoint charges while it is loaded, in GiB.
    # A different question from vram_gib and not derivable from it: under
    # RESIDENT the weights are read and handed to the device, so the host charge
    # is transient and roughly the checkpoint's size; under OFFLOAD the whole
    # checkpoint stays in host memory for the pipe's life and streams one
    # submodule at a time, so vram_gib is small precisely *because* this is
    # large. FLUX.2 klein is the case: 10 GiB on the card, ~16 GiB on the host.
    #
    # Admission used to ask only whether commit was past a 90% ceiling, which
    # is a percentage and not a quantity -- a machine at 85% can have well under
    # 16 GiB of commit left, be admitted, and cross the limit mid-load, which
    # Windows answers by terminating the process (MDL-04). 0 means "no better
    # figure than the default", and the loader falls back to ``vram_gib``.
    host_peak_gib: float = 0.0
    # Files (relative to the model directory) whose presence means "downloaded",
    # for doctor. Empty keeps the default unet/-shaped formula, which is right
    # for every SDXL checkpoint and wrong for anything that has no unet/.
    probe: tuple[str, ...] = ()
    # What to fetch, and therefore also what doctor prints. A base model that
    # reuses another's weights lists the same Fetch again; warlock.fetch
    # dedupes on (repo_id, destination), never on model key.
    fetch: tuple[Fetch, ...] = ()
    #: One sentence for the Settings table's Description column, and the long
    #: form for its tooltip -- a blank line separates the two.
    #:
    #: **A field on the spec, not a table beside it.** ``fetch.KINDS``' own
    #: comment records what a parallel table costs: the download-kind
    #: vocabulary was hand-copied three times, "each drifted differently", and
    #: one copy silently dropped a kind. A descriptions dict keyed on the row
    #: key would be a fourth copy, where a renamed key leaves a blank cell and
    #: nothing fails. Last in the field order so no positional construction
    #: breaks, and defaulted so a new entry is legal before its prose is
    #: written.
    description: str = ""
    #: What the *weights* are licensed under, and whether output may be sold.
    #:
    #: **A field on the spec for ``description``'s reason exactly** -- a table
    #: keyed on the model key would be a copy where a rename leaves a blank
    #: cell and nothing fails. This one is worse than a blank cell: the blank
    #: is the permissive-looking answer.
    #:
    #: This exists because a tool whose entire purpose is producing assets
    #: people will sell shipped two checkpoints that restrict exactly that and
    #: said so nowhere -- not in ``docs/MODELS.md``, not at download time, not
    #: in the picker. SDXL-Turbo is Stability's non-commercial research licence
    #: and was promoted in the README as "the fast option"; Playground v2.5
    #: permits commercial use only below 1M monthly users and requires shipping
    #: its licence and attribution string.
    #:
    #: ``commercial`` is the field the UI actually branches on, because it is
    #: the only question a user generating a game asset is really asking.
    license: str = ""
    #: False when the licence forbids commercial use of generated output, or
    #: attaches conditions to it. Drives the picker's warning badge.
    commercial: bool = True
    #: The sentence the picker appends after the licence: the condition when
    #: ``commercial`` is True but not unconditionally so, or the reason when
    #: it is False (``turbo`` names its non-commercial clause here).
    license_note: str = ""

    @property
    def download(self) -> str:
        return download_text(self.fetch)


@dataclass(frozen=True, slots=True)
class StyleLora:
    key: str
    label: str
    filename: str
    # Trained trigger words. Prepended to the composed prompt in
    # text2image.generate(), alongside PROMPT_TEMPLATE -- these are model-facing
    # scaffolding, not creative direction, so they don't belong in guidance.py.
    trigger: str = ""
    default_weight: float = DEFAULT_LORA_WEIGHT
    # Which architecture's modules this adapter names, from FAMILIES. Declared
    # for the reason BaseModel.family is: an adapter must not become
    # "SDXL-shaped" because its key prefixes happen to look like one, and a
    # load attempt is not a cheap probe -- it raises with the base already
    # resident in VRAM. The default keeps every SDXL entry unedited.
    family: str = FAMILY_SDXL
    fetch: tuple[Fetch, ...] = ()
    description: str = ""

    @property
    def download(self) -> str:
        return download_text(self.fetch)


@dataclass(frozen=True, slots=True)
class IPAdapter:
    """An appearance/identity adapter plus the CLIP vision encoder it needs.

    The encoder is a separate download from the weights and loading one
    without the other succeeds and then fails at the first call, which is not
    a failure a user can read -- hence two download lines and a doctor row
    that checks both.
    """

    key: str
    label: str
    dir_name: str
    subfolder: str
    weight_name: str
    # Where the CLIP vision encoder sits *relative to dir_name*, matching the
    # layout the download command produces.
    image_encoder_dir: str
    default_scale: float = 0.6
    fetch: tuple[Fetch, ...] = ()
    description: str = ""
    # The architecture the adapter's weights name, like ``StyleLora.family``
    # and ``ControlNet.family``: every shipped adapter is an SDXL one, and the
    # door refuses it on any other base for the same reason it refuses a
    # cross-family LoRA -- the mismatch only surfaces after the checkpoint is
    # resident.
    family: str = FAMILY_SDXL

    @property
    def download(self) -> str:
        return download_text(self.fetch)

    @property
    def encoder_folder(self) -> str:
        """What diffusers' image_encoder_folder wants: a path *relative to the
        adapter root*, POSIX-separated. Built here so a Windows backslash can
        never reach load_ip_adapter.
        """
        return self.image_encoder_dir.replace("\\", "/")


@dataclass(frozen=True, slots=True)
class ControlNet:
    key: str
    label: str
    dir_name: str
    # Which preprocessor turns the reference into a hint image. Only "canny"
    # exists today (``pipelines/control.PREPROCESSORS``): a *monocular* depth
    # hint would need a torch model and therefore a module of its own, since
    # control.py must stay torch-free. ``None`` means no reference-derived
    # hint exists at all -- a pipeline renders the hint itself (the re-texture
    # stage's Blender depth pass) -- and such an entry must never surface on
    # the text-job path: ``catalog()`` filters it, ``guidance.normalize``
    # refuses it, and ``_q_generate._conditioning`` drops it, in that order of
    # defence.
    preprocessor: str | None
    variant: str | None = "fp16"
    default_scale: float = 0.65
    default_end: float = 0.8
    fetch: tuple[Fetch, ...] = ()
    description: str = ""

    @property
    def download(self) -> str:
        return download_text(self.fetch)


@dataclass(frozen=True, slots=True)
class EngineModel:
    """A native reconstruction engine's non-Python model payload."""

    key: str
    label: str
    probe: tuple[str, ...]
    fetch: tuple[Fetch, ...] = ()
    description: str = ""

    @property
    def download(self) -> str:
        return download_text(self.fetch)


def _table(*items):
    return {item.key: item for item in items}


TRELLIS_GGUF_FILES = (
    "birefnet.gguf",
    "dinov3.gguf",
    "shape_dec.gguf",
    "shape_flow_1024.gguf",
    "shape_flow_512.gguf",
    "ss_dec.gguf",
    "ss_flow.gguf",
    "tex_dec.gguf",
    "tex_flow_1024.gguf",
    "tex_flow_512.gguf",
)

ENGINE_MODELS: dict[str, EngineModel] = _table(
    EngineModel(
        "trellis_gguf",
        "TRELLIS.2 GGUF weights",
        TRELLIS_GGUF_FILES,
        fetch=(
            Fetch(
                "ilintar/trellis2-gguf",
                "trellis2-gguf",
                revision="a57397bd3d351599d9729fc144b3f87c3f87d65b",
                allow_patterns=("*.gguf",),
                ignore_patterns=("q4/*", "q8/*"),
                size_gib=16.1,
            ),
        ),
        description=(
            "The reconstruction engine: one image in, a textured mesh out.\n\n"
            "trellis-server.exe's own weights, quantised. This is the half of the "
            "app that makes geometry -- without it the Mesh stage has nothing to "
            "run, and every other model here is optional beside it."
        ),
    )
)


# The diffusers-layout include set every SDXL-class checkpoint takes: configs,
# tokenizer vocabularies, and the fp16 weight files only.
_SDXL_FILES = ("*.json", "*.txt", "*fp16.safetensors")

# Written once and referenced by four entries rather than repeated, because it
# genuinely *is* one download: sdxl, sdxl_cfg, pixel and lightning are four
# recipes over the same weights. Identity here is what makes the dedupe in
# warlock.fetch obvious rather than a coincidence of equal strings.
_SDXL_BASE_1_0 = Fetch(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-base-1.0",
    revision="462165984030d82259a11f4367a4eed129e94a7b",
    allow_patterns=_SDXL_FILES,
    size_gib=7.0,
)


BASE_MODELS: dict[str, BaseModel] = _table(
    BaseModel(
        "turbo",
        "SDXL-Turbo (fast)",
        "sdxl-turbo",
        image_size=512,
        steps=4,
        guidance_scale=0.0,
        fetch=(
            Fetch(
                "stabilityai/sdxl-turbo",
                "sdxl-turbo",
                revision="71153311d3dbb46851df1931d3ca6e939de83304",
                allow_patterns=_SDXL_FILES,
                ignore_patterns=("sd_xl_turbo_1.0*",),
                size_gib=7.0,
            ),
        ),
        description=(
            "The fast option: four steps at 512 px, guidance off.\n\n"
            "Its own checkpoint rather than a recipe over the base weights, and "
            "worth having when iteration speed matters more than fidelity -- a "
            "draft in seconds instead of most of a minute. Style LoRAs land "
            "weakly here: they are trained against full SDXL at 20-25 steps with "
            "CFG. This is also the entry WARLOCK_T2I_DIR redirects."
        ),
        license="Stability AI Non-Commercial Research Community License",
        commercial=False,
        license_note="Commercial use of generated images requires a paid Stability AI membership.",
    ),
    BaseModel(
        # The backend where style LoRAs behave as trained: they are fitted
        # against full SDXL at 20-25 steps with CFG, and Turbo's 4 steps at
        # guidance 0 applies them only weakly. Hyper-SD buys back the step
        # count without changing the base weights the LoRAs were fitted to.
        "sdxl",
        "SDXL 1.0 + Hyper-SD (best LoRA response)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=4,
        guidance_scale=0.0,
        scheduler="ddim_trailing",
        base_lora="Hyper-SDXL-4steps-lora.safetensors",
        fetch=(
            _SDXL_BASE_1_0,
            Fetch(
                "ByteDance/Hyper-SD",
                "loras",
                revision="bc08d970a87c74c71209491d64e3525845698863",
                filenames=("Hyper-SDXL-4steps-lora.safetensors",),
                size_gib=0.8,
            ),
        ),
        description=(
            "Full SDXL in four steps, where style LoRAs behave as trained.\n\n"
            "The base weights with Hyper-SD on top. Style LoRAs are fitted "
            "against full SDXL at 20-25 steps with CFG and apply only weakly at "
            "Turbo's four steps at guidance 0; Hyper-SD buys the step count back "
            "without changing the weights they were fitted to."
        ),
        license="OpenRAIL++-M (SDXL 1.0) + OpenRAIL++-M (Hyper-SD)",
        commercial=True,
    ),
    BaseModel(
        # Its own scheduler_config.json is already EDMDPMSolverMultistep, so no
        # override. The card recommends 50 steps; 25 is the point where extra
        # steps stop visibly changing a plain-background single object, and
        # this is a reference image for TRELLIS, not a final render.
        "playground",
        "Playground v2.5 (highest fidelity, slow)",
        "playground-v2.5",
        image_size=1024,
        steps=25,
        guidance_scale=3.0,
        controlnet=True,
        fetch=(
            Fetch(
                "playgroundai/playground-v2.5-1024px-aesthetic",
                "playground-v2.5",
                revision="1e032f13f2fe6db2dc49947dbdbd196e753de573",
                allow_patterns=_SDXL_FILES,
                size_gib=7.0,
            ),
        ),
        description=(
            "The highest-fidelity option, and correspondingly the slowest.\n\n"
            "Its own checkpoint, about 25 steps with CFG. Reach for it when the "
            "reference is the deliverable rather than a step on the way to a "
            "mesh."
        ),
        license="Playground v2.5 Community License",
        commercial=True,
        license_note=(
            "Free below 1M monthly active users. Requires shipping the licence "
            "text and its attribution string with anything you distribute."
        ),
    ),
    BaseModel(
        # The same weights as "sdxl", run the way the checkpoint was trained:
        # 30 steps with real classifier-free guidance and no Hyper-SD. This is
        # the only reason it exists as a separate entry -- a distilled 4-step
        # base at guidance 0 discards the negative prompt entirely
        # (text2image encodes it only when guidance_scale > 1.0) and gives a
        # ControlNet nothing to steer. No new weights: same dir_name.
        "sdxl_cfg",
        "SDXL 1.0 (full CFG, structural control)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=30,
        guidance_scale=7.0,
        controlnet=True,
        fetch=(_SDXL_BASE_1_0,),
        description=(
            "The default: full CFG, so structure is actually controllable.\n\n"
            "No extra download -- the same base weights as the entry above, run "
            "the way SDXL was trained. It is the only family that takes "
            "ControlNet, and the only one where the negative prompt carries full "
            "weight. Slower than the distilled options and measurably better at "
            "holding a silhouette."
        ),
        license="OpenRAIL++-M",
        commercial=True,
    ),
    BaseModel(
        # The same weights and recipe as sdxl_cfg with two training-free
        # sampling upgrades on top: PAG at the paper's 3.0, and CFG rescale at
        # the 0.7 its paper recommends against exactly the high-CFG washout a
        # 7.0 guidance produces. A separate row rather than fields on sdxl_cfg
        # because that row's unconditioned output is bit-identity-pinned: this
        # is the opt-in arm the bench compares against it, and a measured win
        # is what would flip DEFAULT_BASE_MODEL here -- with the
        # docs/measurements/ note that rule requires. No new download.
        "sdxl_cfg_pag",
        "SDXL 1.0 + PAG (full CFG, cleaner structure)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=30,
        guidance_scale=7.0,
        controlnet=True,
        pag_scale=3.0,
        guidance_rescale=0.7,
        fetch=(_SDXL_BASE_1_0,),
        description=(
            "The default recipe plus two training-free sampling upgrades.\n\n"
            "No download either: perturbed-attention guidance at 3.0 and CFG "
            "rescale at 0.7, which counters the washout high CFG produces. "
            "Cleaner structure for the same weights, and about a third more time "
            "per image."
        ),
        license="OpenRAIL++-M",
        commercial=True,
    ),
    BaseModel(
        # The pixel-art profile's null hypothesis: the same SDXL 1.0 weights as
        # "sdxl"/"sdxl_cfg" (no new checkpoint), run the way the pixel-art-xl
        # author documents -- LCM at 8 steps, guidance 1.0. Distinct from the
        # Hyper-SD arm because stacking a 1.0-weight distillation LoRA under a
        # 1.2-weight style LoRA is unproven; both arms exist so bench/pixel-v1
        # can decide which the preset keeps.
        #
        # No VAE override on purpose. sdxl-vae-fp16-fix patches fp16's
        # overflow-to-NaN, and everything here loads bfloat16 (see
        # pipelines/text2image; variant="fp16" only names the weight *files*),
        # which has fp32's exponent range -- so that failure mode does not
        # exist here. If bench images ever show VAE decode artifacts, add a
        # BaseModel.vae field then, with a docs/measurements/ note.
        "pixel",
        "SDXL 1.0 + LCM (pixel art)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=8,
        guidance_scale=1.0,
        scheduler="lcm",
        base_lora="lcm-lora-sdxl.safetensors",
        fetch=(
            _SDXL_BASE_1_0,
            Fetch(
                "latent-consistency/lcm-lora-sdxl",
                "loras",
                revision="a18548dd4956b174ec5b0d78d340c8dae0a129cd",
                filenames=("pytorch_lora_weights.safetensors",),
                # Renamed because the upstream filename is generic: any other
                # repo's default-named LoRA downloaded into the flat loras/
                # directory would silently overwrite it.
                rename=("pytorch_lora_weights.safetensors", "lcm-lora-sdxl.safetensors"),
                size_gib=0.4,
            ),
        ),
        description=(
            "The recipe the pixel-art LoRA was trained against.\n\n"
            "The base weights again with LCM on top: eight steps at guidance 1.0. "
            "Pair it with the pixel-art style LoRA -- alone it is simply a fast "
            "SDXL."
        ),
        license="OpenRAIL++-M (SDXL 1.0) + OpenRAIL-M (LCM-LoRA)",
        commercial=True,
    ),
    BaseModel(
        # The second distillation arm, and the reason it is worth a row: it is
        # a genuinely different method from "sdxl"'s Hyper-SD -- adversarial
        # rather than trajectory-consistency -- over the same base weights, so
        # the sweep/verdict machinery can compare the two with everything else
        # held fixed. 394 MB of new download, and no new checkpoint.
        #
        # Trailing timestep spacing is not optional, exactly as it is not for
        # Hyper-SD: on the default leading spacing a Lightning LoRA produces a
        # washed-out image and no error saying so. Euler rather than DDIM
        # because that is the pairing the model card documents.
        "lightning",
        "SDXL 1.0 + Lightning (4-step)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=4,
        guidance_scale=0.0,
        scheduler="euler_trailing",
        base_lora="sdxl_lightning_4step_lora.safetensors",
        fetch=(
            _SDXL_BASE_1_0,
            Fetch(
                "ByteDance/SDXL-Lightning",
                "loras",
                revision="c9a24f48e1c025556787b0c58dd67a091ece2e44",
                filenames=("sdxl_lightning_4step_lora.safetensors",),
                size_gib=0.4,
            ),
        ),
        description=(
            "A second four-step distillation, for comparison with Hyper-SD.\n\n"
            "Reuses the base weights. Adversarial where Hyper-SD is trajectory- "
            "consistency, so the two are directly comparable with everything else "
            "held fixed."
        ),
        license="OpenRAIL++-M (SDXL 1.0) + OpenRAIL++-M (SDXL-Lightning)",
        commercial=True,
    ),
    BaseModel(
        # A photoreal SDXL finetune, run at its card's own recipe: DPM++ 2M
        # Karras, 30-40 steps, CFG 3-7. 35/4.0 is the middle of both ranges.
        # Full CFG, so the negative prompt is live and a ControlNet has
        # something to steer.
        "juggernaut",
        "Juggernaut XL v9 (photoreal)",
        "juggernaut-xl-v9",
        image_size=1024,
        steps=35,
        guidance_scale=4.0,
        scheduler="dpm_karras",
        controlnet=True,
        fetch=(
            Fetch(
                "RunDiffusion/Juggernaut-XL-v9",
                "juggernaut-xl-v9",
                revision="cf419233522daa0b9ea36c3aff98fa2cab1fb0fb",
                allow_patterns=_SDXL_FILES,
                size_gib=6.9,
            ),
        ),
        description=(
            "A photoreal SDXL finetune.\n\n"
            "Its own checkpoint, DPM++ 2M Karras at 35 steps with CFG 4.0 -- the "
            "middle of the ranges its model card gives. Materials and lighting "
            "read as photographed rather than illustrated, which suits props that "
            "will be reconstructed and then lit in an engine."
        ),
        license="OpenRAIL-M",
        commercial=True,
    ),
    BaseModel(
        # The stylised counterpart to juggernaut, and the card's own snippet is
        # DEIS at 25 steps. It states no CFG, so this takes SDXL's own 7.0
        # rather than inventing one.
        "dreamshaper",
        "DreamShaper XL (stylised)",
        "dreamshaper-xl",
        image_size=1024,
        steps=25,
        guidance_scale=7.0,
        scheduler="deis",
        controlnet=True,
        fetch=(
            Fetch(
                "Lykon/dreamshaper-xl-1-0",
                "dreamshaper-xl",
                revision="41e6644752a8c9aa63930e6043c4fd83c7708420",
                allow_patterns=_SDXL_FILES,
                size_gib=6.9,
            ),
        ),
        description=(
            "The stylised counterpart to the photoreal finetune above.\n\n"
            "Its own checkpoint, DEIS at 25 steps per its card."
        ),
        license="OpenRAIL++-M",
        commercial=True,
    ),
    BaseModel(
        # The first non-SDXL architecture in the registry, and the reason
        # BaseModel has a ``family`` at all. One Qwen3 text encoder at 512
        # tokens instead of two CLIPs at 77, a DiT instead of a UNet, and no
        # pooled embedding -- so the chunker, the style LoRAs, the ControlNet,
        # the IP-Adapter and img2img are all inapplicable to it.
        #
        # The *base* variant rather than the distilled FLUX.2-klein-4B, and
        # deliberately: the distilled checkpoint registers is_distilled=True,
        # and Flux2KleinPipeline.do_classifier_free_guidance is
        # ``guidance_scale > 1 and not is_distilled``, so a negative prompt is
        # impossible on it. This one carries no is_distilled in its
        # model_index.json, takes the class default of False, and therefore
        # honours the negative prompt the app fills in by default.
        #
        # variant=None because the repo ships no *.fp16.safetensors, and a
        # ``probe`` because the default doctor formula looks for a unet/ this
        # has no equivalent of. Both halves of the probe matter: the text
        # encoder is half the download, and checking only the transformer would
        # call a half-fetched model present.
        #
        # OFFLOAD, not RESIDENT. Transformer 7.75 GB + text encoder 8.04 GB +
        # VAE 0.17 GB is ~16 GB fully resident -- the same as trellis-server,
        # so .to("cuda") would put coexist out of reach on a 32 GB card
        # (16 + 16 + 1.5 headroom > 32) and make this usable only under
        # WARLOCK_VRAM_EXCLUSIVE=1. Offloaded, the peak is roughly the larger
        # submodule plus activations; 10.0 is that, rounded up.
        "flux_klein",
        "FLUX.2 klein-base 4B (full CFG)",
        "flux2-klein-base-4b",
        image_size=1024,
        steps=50,
        guidance_scale=4.0,
        variant=None,
        family=FAMILY_FLUX2_KLEIN,
        residency=OFFLOAD,
        vram_gib=10.0,
        # The whole checkpoint stays in host memory for the pipe's life -- that
        # is what offload means, and it is why vram_gib above is 10 rather than
        # 16. Admission has to have the host number by name.
        #
        # 24.0, not the checkpoint's ~16: measured 2026-08-22 end to end, a load
        # charged 18.3 GiB of private commit and one 1024x1024 sample took it to
        # **24.1**, which is the figure admission is actually standing in front
        # of. The old number priced the weights and forgot the sample
        # (docs/measurements/2026-08-22-trampoline-child-pids.md).
        host_peak_gib=24.0,
        probe=(
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/model-00001-of-00002.safetensors",
        ),
        fetch=(
            Fetch(
                "black-forest-labs/FLUX.2-klein-base-4B",
                "flux2-klein-base-4b",
                revision="a3b4f4849157f664bdbc776fd7453c2783562f4d",
                allow_patterns=("*.json", "*.txt", "*.jinja", "*.safetensors"),
                # The repo carries a redundant 7.75 GB single-file checkpoint
                # beside the diffusers layout; nothing here reads it.
                ignore_patterns=("flux-2-klein-base-4b.safetensors",),
                size_gib=16.0,
            ),
        ),
        description=(
            "The one non-SDXL architecture, at full CFG.\n\n"
            "A 4B model with a Qwen3 text encoder, so it reads long prompts far "
            "more literally than SDXL does. The -base variant rather than the "
            "distilled one below, because a negative prompt is inert on the "
            "distilled weights. Large -- budget the VRAM before running it beside "
            "a reconstruction."
        ),
        license="Apache-2.0",
        commercial=True,
    ),
    BaseModel(
        # The distilled sibling of flux_klein above, and the other side of that
        # entry's argument. This one *does* register is_distilled=True, so
        # Flux2KleinPipeline.do_classifier_free_guidance is permanently False
        # and the negative prompt is inert on it -- which needs no new code,
        # because cfg_bases() filters on guidance_scale > 1.0 and 1.0 excludes
        # it. Pick klein-base when a negative prompt is wanted.
        #
        # It exists so the pixel-art style LoRA can run at the recipe it was
        # trained on: 4 steps at CFG 1.0. The two checkpoints are the same
        # architecture, so an adapter fitted to either loads onto both; which
        # one *expresses* a style is a recipe fact, measured rather than
        # assumed (docs/measurements/2026-08-10-pixel-art-klein.md).
        #
        # Same layout, same probe, same offload argument, same redundant
        # single-file checkpoint to ignore as klein-base.
        "flux_klein_distilled",
        "FLUX.2 klein 4B (distilled, 4-step)",
        "flux2-klein-4b",
        image_size=1024,
        steps=4,
        guidance_scale=1.0,
        variant=None,
        family=FAMILY_FLUX2_KLEIN,
        residency=OFFLOAD,
        vram_gib=10.0,
        # The whole checkpoint stays in host memory for the pipe's life -- that
        # is what offload means, and it is why vram_gib above is 10 rather than
        # 16. Admission has to have the host number by name.
        #
        # 24.0, not the checkpoint's ~16: measured 2026-08-22 end to end, a load
        # charged 18.3 GiB of private commit and one 1024x1024 sample took it to
        # **24.1**, which is the figure admission is actually standing in front
        # of. The old number priced the weights and forgot the sample
        # (docs/measurements/2026-08-22-trampoline-child-pids.md).
        host_peak_gib=24.0,
        probe=(
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/model-00001-of-00002.safetensors",
        ),
        fetch=(
            Fetch(
                "black-forest-labs/FLUX.2-klein-4B",
                "flux2-klein-4b",
                revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
                allow_patterns=("*.json", "*.txt", "*.jinja", "*.safetensors"),
                ignore_patterns=("flux-2-klein-4b.safetensors",),
                size_gib=16.0,
            ),
        ),
        description=(
            "The same architecture at four steps; no negative prompt.\n\n"
            "Guidance is baked in, so a negative prompt does nothing at all here "
            "-- pick klein-base when you want one. It is here because the FLUX.2 "
            "pixel-art LoRA was trained against these weights."
        ),
        license="Apache-2.0",
        commercial=True,
    ),
)

STYLE_LORAS: dict[str, StyleLora] = _table(
    StyleLora(
        "render3d",
        "3D render",
        "3d_render_style_xl.safetensors",
        trigger="3d style, 3d render",
        fetch=(
            Fetch(
                "goofyai/3d_render_style_xl",
                "loras",
                revision="5ec74a57db5e244a2157173781a7b29045f88237",
                filenames=("3d_render_style_xl.safetensors",),
                size_gib=0.2,
            ),
        ),
        description=(
            "Clean studio-lit 3D-render look. The general-purpose prop style.\n\n"
            "The safest starting point for something that will be reconstructed: "
            "even lighting, readable silhouette, and no painterly texture for the "
            "mesh to interpret as geometry."
        ),
    ),
    StyleLora(
        "redmond3d",
        "3D render (Redmond)",
        "3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
        trigger="3D Render Style, 3DRenderAF",
        fetch=(
            Fetch(
                "artificialguybr/3DRedmond-V1",
                "loras",
                revision="f4b4b980972566aea7c71af9d4e170d7fcb6c404",
                filenames=("3DRedmond-3DRenderStyle-3DRenderAF.safetensors",),
                size_gib=0.2,
            ),
        ),
        description=(
            "The same idea, warmer and more stylised.\n\n"
            "A second take on the render look, for when the first reads too "
            "clinical for the piece."
        ),
    ),
    StyleLora(
        # Pairs with the PS1-era art style: chunky untextured geometry
        # is exactly what TRELLIS reconstructs most cleanly.
        "ps1",
        "PS1 / low-poly game",
        "PS1Redmond-PS1Game-Playstation1Graphics.safetensors",
        trigger="Ps1 game graphics",
        fetch=(
            Fetch(
                "artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl",
                "loras",
                revision="74bb3a6e2efd47ead698ff3ac2695ab63bbd2d5c",
                filenames=("PS1Redmond-PS1Game-Playstation1Graphics.safetensors",),
                size_gib=0.2,
            ),
        ),
        description=(
            "Low-poly PlayStation-era game graphics.\n\n"
            "Chunky forms and flat texturing, which reconstruct unusually cleanly "
            "-- the style is already making the simplifications the mesh would "
            "have to."
        ),
    ),
    StyleLora(
        # The one LoRA that generates pixel art *natively* rather than being
        # downscaled into it. Its default weight is the author's documented
        # recipe (1.2), not this module's DEFAULT_LORA_WEIGHT -- below it the
        # output keeps SDXL's anti-aliased gradients and no downscale recovers
        # a clean grid from them.
        #
        # The trigger carries both spellings the model card uses; they are
        # model-facing scaffolding, which is exactly why guidance.py's fragments
        # may never contain the word (tests/test_guidance.py pins that).
        "pixelxl",
        "Pixel art (pixel-art-xl)",
        "pixel-art-xl.safetensors",
        trigger="pixel, pixel art",
        default_weight=1.2,
        fetch=(
            Fetch(
                "nerijs/pixel-art-xl",
                "loras",
                revision="8bf4a4d9ea283e00a51fafda8e0539f8248ea037",
                filenames=("pixel-art-xl.safetensors",),
                size_gib=0.2,
            ),
        ),
        description=(
            "Draws on a pixel grid rather than being downscaled onto one.\n\n"
            "The difference is the whole point: a downscaled render has soft, "
            "resampled edges, and this generates the blocks. SDXL only -- pair it "
            "with the LCM pixel recipe it was trained against."
        ),
    ),
    StyleLora(
        # The registry's first non-SDXL adapter, and the reason StyleLora has a
        # ``family`` at all. Trained against the *distilled* FLUX.2-klein-4B;
        # klein-base is the same architecture, so it loads onto both, and which
        # one expresses the style is a recipe fact rather than an architecture
        # one -- measured in docs/measurements/2026-08-10-pixel-art-klein.md.
        #
        # "pixelklein" names the architecture the way "pixelxl" does. A bare
        # "pixel" is banned: it is a BASE_MODELS key, and keys are adapter
        # names on one pipeline.
        #
        # The trigger is the card's declared trigger_word, so like every other
        # trigger here it is model-facing scaffolding and may never appear in
        # guidance.py's fragments -- which tests/test_guidance.py already pins
        # for the word "pixel", the word this one contains.
        #
        # The rename is mandatory rather than tidy: loras/ is flat and shared
        # across families, and "pytorch_lora_weights.safetensors" is also what
        # latent-consistency/lcm-lora-sdxl ships. Two generic names in one flat
        # directory is one file, silently the wrong weights under one of the
        # keys. fetch_worker performs the rename inside staging, before the
        # move, so the generic name never lands in loras/ at all.
        #
        # The repo also ships a .comfyui.safetensors variant, deliberately not
        # used: it carries no lora_adapter_metadata and no kohya .alpha keys,
        # so peft would load it at scale 1.0 against a trained alpha/sqrt(r) of
        # 16.0 -- a 16x under-application no strength slider could reach. The
        # diffusers file declares r=64, alpha=128, use_rslora=True in its own
        # header and is restored exactly.
        "pixelklein",
        "Pixel art (FLUX.2 klein)",
        "pixel-art-klein.safetensors",
        trigger="pixel art sprite",
        family=FAMILY_FLUX2_KLEIN,
        # Measured, and far below every other entry here for one declared
        # reason: this adapter trained with use_rslora=True, so peft restores a
        # scale of alpha/sqrt(r) = 128/8 = 16.0 where an ordinary alpha/r LoRA
        # restores about 2. The slider's meaning is unchanged -- it has always
        # been a multiplier on whatever the checkpoint declares -- so 0.0625 is
        # the number that puts this one at an effective 1.0.
        #
        # The model card's own 0.85-1.4 assumes a loader that ignores rslora;
        # honoured, that range is an effective 13.6-22.4 and every image in it
        # is a black frame. The usable band is 0.02-0.08, and by 0.125 the
        # lattice is already smearing. All of that is measured, per prompt and
        # per weight, in docs/measurements/2026-08-10-pixel-art-klein.md.
        default_weight=0.0625,
        fetch=(
            Fetch(
                "Limbicnation/pixel-art-lora",
                "loras",
                # Unconfirmed against local bytes: this adapter is in the
                # registry but has never been downloaded on the host the
                # pins were recovered from, so the SHA comes from the hub's
                # refs/main rather than from a .metadata beside real files.
                # Re-check it against the HF API the next time this box is
                # online; every other pin here was read off downloaded data.
                revision="0ac8e5c3400af68228811edc324721e25fc26777",
                filenames=("pytorch_lora_weights.safetensors",),
                rename=("pytorch_lora_weights.safetensors", "pixel-art-klein.safetensors"),
                size_gib=0.3,
            ),
        ),
        description=(
            "Pixel art for the FLUX.2 klein family.\n\n"
            "The one non-SDXL adapter here, and it is offered only on the two "
            "klein entries because a LoRA cannot cross architectures."
        ),
    ),
)


IP_ADAPTERS: dict[str, IPAdapter] = _table(
    IPAdapter(
        # "plus" rather than the base adapter: it conditions on 16 patch
        # tokens instead of one pooled embedding, which is the difference
        # between "same kind of object" and "this object".
        "plus",
        "Appearance reference (IP-Adapter Plus)",
        "ip-adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_dir="models/image_encoder",
        # Two records against one repository, and they stay two: the weights
        # and the CLIP vision encoder are separate lines in the README and in
        # doctor's text. The dedupe in warlock.fetch is what merges them back
        # into a single fetch when the button runs.
        fetch=(
            Fetch(
                "h94/IP-Adapter",
                "ip-adapter",
                revision="018e402774aeeddd60609b4ecdb7e298259dc729",
                filenames=("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",),
                size_gib=0.9,
            ),
            Fetch(
                "h94/IP-Adapter",
                "ip-adapter",
                revision="018e402774aeeddd60609b4ecdb7e298259dc729",
                allow_patterns=("models/image_encoder/*",),
                size_gib=2.6,
            ),
        ),
        description=(
            "Condition a generation on a reference image's appearance.\n\n"
            "Colour, material and general look carried across from an image you "
            "supply, rather than described in words. Both halves of the download "
            "are needed: the weights alone load fine and then fail at the first "
            "call."
        ),
    ),
)

CONTROLNETS: dict[str, ControlNet] = _table(
    ControlNet(
        "canny",
        "Edge / silhouette lock (Canny)",
        "controlnet-canny-sdxl",
        preprocessor="canny",
        fetch=(
            Fetch(
                "diffusers/controlnet-canny-sdxl-1.0",
                "controlnet-canny-sdxl",
                revision="eb115a19a10d14909256db740ed109532ab1483c",
                allow_patterns=("*.json", "*fp16.safetensors"),
                size_gib=2.5,
            ),
        ),
        description=(
            "Lock the silhouette to a reference image's edges.\n\n"
            "The strongest structural control available, and only on the full-CFG "
            "families -- the distilled ones do not respond to it. Use it when the "
            "shape is decided and only the treatment is in question."
        ),
    ),
    ControlNet(
        # Pipeline-fed (preprocessor=None): the re-texture stage renders the
        # hint itself from the mesh -- ground-truth depth through the exact
        # bake cameras, which no monocular estimator can match. Same org,
        # layout, fp16 variant and size class as the canny entry, and trained
        # on inverted relative depth, which is what `retexture.depth_hint`
        # produces.
        "depth",
        "Surface relief (rendered depth)",
        "controlnet-depth-sdxl",
        preprocessor=None,
        fetch=(
            Fetch(
                "diffusers/controlnet-depth-sdxl-1.0",
                "controlnet-depth-sdxl",
                revision="17bb97973f29801224cd66f192c5ffacf82648b4",
                allow_patterns=("*.json", "*fp16.safetensors"),
                size_gib=2.5,
            ),
        ),
        description=(
            "Anchors a re-texture's restyle passes to the mesh's own geometry.\n\n"
            "The depth hint is rendered by Blender from the mesh itself and never "
            "estimated from a photo, which is why this one belongs to the re- "
            "texture stage alone and is not offered in the reference-stage "
            "conditioning pickers."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class MetricModel:
    """A model used to *measure* an asset rather than make one.

    Same registry shape as the rest so doctor reports it the same way, but
    nothing in the app's generation path touches these -- only the benchmark
    does, and a missing one costs a metric, not a job.
    """

    key: str
    label: str
    dir_name: str
    fetch: tuple[Fetch, ...] = ()
    description: str = ""

    @property
    def download(self) -> str:
        return download_text(self.fetch)


METRIC_MODELS: dict[str, MetricModel] = _table(
    MetricModel(
        "dinov2",
        "DINOv2 base (identity metric)",
        "dinov2-base",
        fetch=(
            Fetch(
                "facebook/dinov2-base",
                "dinov2-base",
                revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
                allow_patterns=("*.json", "*.safetensors"),
                size_gib=0.4,
            ),
        ),
        description=(
            "The identity metric the bench scores candidates with.\n\n"
            "Answers 'is this the same object' between two images, which is what "
            "makes a parameter sweep comparable. A missing one costs a number, "
            "never a job. CPU."
        ),
    ),
    MetricModel(
        # PickScore v1: a CLIP-H fine-tuned on 500k human A/B preferences over
        # generated images (arXiv 2305.01569) -- the inference-time stand-in
        # for the preference post-training every hosted image service bakes
        # in. Scores an image *for a prompt*, so the candidate ranker can add
        # "which of these would a person pick" beside composition and the
        # style anchor. CPU like the DINOv2 anchor, and for the same rule: a
        # metric must not take VRAM from the models making the asset. One repo
        # carries the processor and tokenizer beside the weights; the pattern
        # set skips the redundant pytorch_model.bin duplicate of the
        # safetensors named explicitly.
        "pickscore",
        "PickScore v1 (human preference metric)",
        "pickscore-v1",
        # Two records against one repository, the IP-Adapter's shape and for a
        # sharper reason, learned the hard way on 2026-08-17: `hf download`
        # silently ignores --include when a positional filename is also given,
        # so a single record mixing the two downloads the weights alone and
        # doctor's probe then reports a model that cannot load. The in-app
        # fetch would have unioned them; the pasted command is what breaks.
        fetch=(
            Fetch(
                "yuvalkirstain/PickScore_v1",
                "pickscore-v1",
                revision="a4e4367c6dfa7288a00c550414478f865b875800",
                filenames=("model.safetensors",),
                size_gib=3.7,
            ),
            Fetch(
                "yuvalkirstain/PickScore_v1",
                "pickscore-v1",
                revision="a4e4367c6dfa7288a00c550414478f865b875800",
                allow_patterns=("*.json", "*.txt"),
                size_gib=0.1,
            ),
        ),
        description=(
            "A human-preference metric: which candidate would a person pick.\n\n"
            "A CLIP-H fine-tuned on human A/B choices over generated images. With "
            "it, ranking a submit's candidates weighs preference beside "
            "composition and the style anchor; without it the ranking is exactly "
            "what it was. CPU."
        ),
    ),
)


DEFAULT_POSE_MODEL = "vitpose"


@dataclass(frozen=True, slots=True)
class PoseModel:
    """A model that finds a subject's joints in a picture of it.

    Its own table for the same reason MattingModel is not a MetricModel: this
    one is read by the *generation* path -- the rig fitter asks it where the
    shoulders are -- while a metric only ever grades a finished asset. What
    they share is that every one of them is optional, none is ever downloaded
    at runtime, and a missing one costs quality rather than a job.
    """

    key: str
    label: str
    dir_name: str
    fetch: tuple[Fetch, ...] = ()
    description: str = ""

    @property
    def download(self) -> str:
        return download_text(self.fetch)


POSE_MODELS: dict[str, PoseModel] = _table(
    PoseModel(
        # The plain ViTPose-base, not one of the MoE variants: those need a
        # dataset_index passed with every forward and buy accuracy on datasets
        # this project never sees. COCO-17 on a single centred subject is the
        # easiest case a pose estimator has, and base clears it.
        "vitpose",
        "ViTPose base (rig joint placement)",
        "vitpose-base",
        fetch=(
            Fetch(
                "usyd-community/vitpose-base-simple",
                "vitpose-base",
                revision="a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45",
                allow_patterns=("*.json", "*.safetensors"),
                size_gib=0.4,
            ),
        ),
        description=(
            "Measures a humanoid's joints off the reference image.\n\n"
            "Without it a humanoid rig places joints by scaling the template onto "
            "the mesh's bounding box -- right for a subject standing in a T-pose "
            "and progressively wrong as it departs from one. With it the "
            "reference is read for real shoulders, elbows, hips, knees and "
            "ankles. It engages by itself when the weights are present, and falls "
            "back wholesale rather than partially. CPU, about a second per rig."
        ),
    ),
)


DEFAULT_MATTING = "birefnet"


@dataclass(frozen=True, slots=True)
class MattingModel:
    """A model that separates a subject from its background, host-side.

    Its own table rather than another MetricModel: a metric measures a finished
    asset and its absence costs a number, while this one *produces* an asset's
    alpha and its absence costs edge quality on every 2D export. Both are
    optional and neither is ever downloaded at runtime, which is all they have
    in common.

    ``remote_code`` is stated rather than implied, and it is **False for every
    entry now**. The published BiRefNet repo ships its own modelling code and
    transformers used to execute it on load: from the snapshot the user
    downloaded once, so nothing was fetched and the offline invariant held, but
    it was still third-party Python running in this process with no lockfile
    over it (MDL-03). That code is vendored at
    ``pipelines/birefnet/`` now, so nothing is executed out of a model
    directory at all.

    The field stays rather than being deleted, and so does doctor's sentence
    keyed off it: it is the thing that would have to be set again if a future
    entry wanted the old behaviour, and a registry with nowhere to say
    "this one runs downloaded code" is a registry that cannot warn about it.
    """

    key: str
    label: str
    dir_name: str
    remote_code: bool = False
    fetch: tuple[Fetch, ...] = ()
    description: str = ""

    @property
    def download(self) -> str:
        return download_text(self.fetch)


MATTING_MODELS: dict[str, MattingModel] = _table(
    MattingModel(
        # BiRefNet and not something smaller, because trellis-server already
        # uses BiRefNet internally for bg_removal="birefnet" -- so a 2D export
        # and the 3D input derived from the same reference agree about where
        # the subject ends, which two different matting models would not.
        "birefnet",
        "BiRefNet",
        "birefnet",
        # False since the modelling code was vendored. Nothing is executed out
        # of the downloaded directory any more; see ``pipelines/birefnet``.
        remote_code=False,
        fetch=(
            Fetch(
                "ZhengPeng7/BiRefNet",
                "birefnet",
                revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
                # ``*.py`` is gone from the download. Not merely unused -- left
                # in place it would be Python sitting in a model directory that
                # nothing runs, which is worse than either extreme: a reader
                # finding it would reasonably assume it *is* what runs.
                allow_patterns=("*.json", "*.safetensors"),
                size_gib=1.0,
                # The weights are not the whole download. The repo's own
                # modelling code -- which trust_remote_code runs out of the
                # checkpoint directory and is vendored here now -- builds its
                # backbone through packages no resolver can see, so a directory
                # doctor is happy with used to hold a model that could not
                # import. They are declared in the text2image extra
                # (einops/kornia/timm/torchvision), which is why this names the
                # sync rather than a bare pip install: the old "you may also
                # need" line was both optional-sounding and incomplete. The
                # dependency did not go away with the vendoring -- the same
                # imports are at the top of ``pipelines/birefnet/modeling.py``
                # -- so the note stays exactly as it was.
                #
                # A ``note`` rather than a second Fetch precisely because the
                # download button cannot do it: it is not a repository fetch,
                # so the pane shows it and the worker never runs it.
                note=(
                    "then: uv sync --extra text2image "
                    "-- BiRefNet's modelling code imports einops, kornia, timm and "
                    "torchvision, and that extra is what supplies them"
                ),
            ),
        ),
        description=(
            "Cuts the background out of 2D exports properly.\n\n"
            "Without it the alpha comes from a corner flood fill, with visibly "
            "rougher edges around hair, spokes and anything thin. Runs on the "
            "host rather than the card. Its own modelling code runs on load, so "
            "it also needs the text2image extra installed."
        ),
    ),
)


def controlnet_bases() -> list[str]:
    """Base models a ControlNet may be attached to -- the UI hides the whole
    Structure group when the chosen base is not one of these, rather than
    offering a control that cannot do anything."""
    return [m.key for m in BASE_MODELS.values() if m.controlnet]


def ip_adapter_bases() -> list[str]:
    """Base models some IP-Adapter is fitted to -- what the door names when it
    refuses a cross-family pairing, and what a picker greys on."""
    families = {a.family for a in IP_ADAPTERS.values()}
    return [m.key for m in BASE_MODELS.values() if m.family in families]


def cfg_bases() -> list[str]:
    """Base models where a negative prompt actually does something.

    text2image encodes the negative branch only when ``guidance_scale > 1.0``
    -- there is no unconditional branch to steer at CFG 0 -- so on a distilled
    4-step base the field is inert. Derived from the number rather than
    declared per model, unlike ``controlnet`` above: "does classifier-free
    guidance run" *is* the guidance scale, whereas "is a ControlNet qualified
    against this checkpoint" is a judgement no threshold can make.
    """
    return [m.key for m in BASE_MODELS.values() if m.guidance_scale > 1.0]


def lora_fits(base: BaseModel, lora: StyleLora) -> bool:
    """Whether this adapter can be loaded onto this checkpoint.

    The single owner of the pairing. Two *declared* families compared, never a
    file inspected: guidance.normalize's refusal, the pane's picker, the
    loader's filter and the worker's tolerance are all this one line, so they
    cannot come to disagree about a pair. Loading a mismatched adapter is a
    load error with the base already resident, not a weak result, which is why
    every one of those sites refuses rather than degrading the way a missing
    LoRA file does.
    """
    return lora.family == base.family


def loras_by_base() -> dict[str, list[str]]:
    """Which style LoRAs fit each base, keyed by base key -- the picker's question.

    Not "may this base have a style" but "which styles": a flat list of bases
    was enough only while every adapter in the registry named one
    architecture's modules.
    """
    return {
        m.key: [lora.key for lora in STYLE_LORAS.values() if lora_fits(m, lora)]
        for m in BASE_MODELS.values()
    }


def lora_bases() -> list[str]:
    """Base models that can take *some* style LoRA -- what greys the picker.

    Keeps its name, signature and meaning; only its derivation changed. The
    question is no longer "is this SDXL" but "does the registry hold an adapter
    fitted to this architecture", so no caller of it is a rename.

    Sharing a derivation with ``loras_by_base`` is right here and is *not* the
    ``tile_bases`` hazard below. That one is two independent questions with
    coincidentally equal answers; this is a single question -- whether the map's
    entry is non-empty -- so sharing is precisely what keeps the greyed picker
    and the populated picker in agreement rather than how they would drift.
    """
    return [key for key, loras in loras_by_base().items() if loras]


def tile_bases() -> list[str]:
    """Base models that can produce a seamless tile.

    Its own name rather than a second reader of ``lora_bases()``:
    seamlessness is circular padding over ``Conv2d``, and a LoRA is a set of
    some architecture's tensors. The two lists once agreed, because both were
    SDXL facts; they genuinely differ now that the registry carries a FLUX.2
    adapter, so the shared list this avoided would already have made a DiT
    "tileable" -- an image that looks seamless in a thumbnail and seams in a
    material.
    """
    return [m.key for m in BASE_MODELS.values() if m.family == FAMILY_SDXL]


def catalog() -> dict[str, Any]:
    """The two tables in guidance.catalog()'s field shape, for the same selects."""
    return {
        "base_model": [
            {"key": m.key, "label": m.label} for m in BASE_MODELS.values()
        ],
        "style_lora": [
            {"key": lora.key, "label": lora.label, "default_weight": lora.default_weight}
            for lora in STYLE_LORAS.values()
        ],
        "ip_adapter": [
            {"key": a.key, "label": a.label, "default_scale": a.default_scale}
            for a in IP_ADAPTERS.values()
        ],
        "control": [
            {
                "key": c.key,
                "label": c.label,
                "preprocessor": c.preprocessor,
                "default_scale": c.default_scale,
                "default_end": c.default_end,
            }
            for c in CONTROLNETS.values()
            # A pipeline-fed entry has no reference-derived hint, so a select
            # offering it would submit a job that fails at write_hint with the
            # checkpoint already in VRAM. Starved at the source: this list is
            # what feeds the settings combo.
            if c.preprocessor
        ],
    }

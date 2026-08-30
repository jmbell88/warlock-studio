"""Design guidance: the validated settings attached to a job.

trellis-server.exe takes only an image, a seed and a geometry resolution, so
guidance can act on exactly three things:

* the SDXL prompt -- text jobs only, since image jobs never touch SDXL;
* the per-request geometry resolution, which the platform preset supplies;
* the physical scale of the finished GLB (pipelines/postprocess.scale_glb).

Deliberately *not* texture resolution: --tex-res is a server launch flag, not a
per-request one, and config.py pins it to 512 because the vendored exe bakes
per-texel noise into the atlas above that.

The creative-direction taxonomy this module used to own (twelve tables of
subject adjectives) was retired on 2026-08-17 -- see
docs/measurements/2026-08-17-taxonomy-retirement.md. What survives is the
machinery selection: ``platform`` (geometry resolution), the model fields
(``base_model``/``style_lora``) and the conditioning fields
(``ip_adapter``/``control``), all validated and stored here so the API gets
its 400-on-unknown from one place. Retired keys arriving in stored params are
tolerated and ignored, never refused, so every pre-retirement job still
rerolls and promotes. The single-object/plain-background scaffolding that
keeps images TRELLIS-friendly lives in pipelines/prompt.PROMPT_TEMPLATE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import models


class GuidanceError(ValueError):
    """A refused guidance value, naming the field it came from.

    A ``ValueError`` subclass, deliberately: every caller in the repo already
    catches ``ValueError`` and this module has raised one since it was written,
    so the addition is purely ``field`` and nothing that reads it today has to
    change. What ``field`` buys is an *address* -- ``service.errors.Invalid``
    has carried a ``field`` since it was written and the UI highlights the
    control it names, but the guidance passthroughs could never supply one, so
    the twelve most common refusals in the app arrived as a toast pointing at
    nothing in particular (S137).

    Recovering the field by parsing the message was the alternative and is the
    translation-table hazard ``judge.STAGES``/``verdicts.STAGES`` is written
    down about: two spellings of one fact, drifting silently the first time a
    message is reworded.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


SIZE_MIN_M = 0.01
SIZE_MAX_M = 100.0

# Fallback when the caller names no size.
DEFAULT_SIZE_M = 1.0
DEFAULT_PLATFORM = "3d"

# How trellis-server mattes the input image. Not an Option table: these are
# server capabilities, not prompt fragments, so they never reach compose_prompt.
BG_REMOVAL = ("auto", "birefnet", "threshold")

# birefnet, and it is the one thing the 2026-08-07 review of the rogue sweep
# found any signal at all for. 3 accepts in 83; all three were
# bg_removal=birefnet and `auto` went 0 for 80, over matched pairs whose
# input.png hashed identically -- bg_removal is passed to trellis-server at
# reconstruction time, so nothing about the picture differed. The failure mode
# moved as well as the rate: 58 of 80 `auto` rejects were tagged `broken`, 0 of
# 4 birefnet were. The mechanism is in doctor.py's own words -- without the
# weights matting "falls back to a threshold cutout", and `auto` lets the
# server decide -- and a threshold cutout on a dark brief leaves background
# attached, which TRELLIS reconstructs into a solid slab.
#
# Gated, though, and default_bg_removal below is the gate: on a host that never
# downloaded birefnet.gguf there is nothing to load, and `auto` is still right
# there. n=4 is thin and the blind confirmation the roadmap asks for has not
# been run; what tips it is that the alternative is a default already measured
# at 0 for 80.
DEFAULT_BG_REMOVAL = "birefnet"
FALLBACK_BG_REMOVAL = "auto"

# The weights DEFAULT_BG_REMOVAL needs, named once. doctor._birefnet_check
# reports this same file, and a second spelling of it would let the app pick a
# matte the doctor is telling the user is unavailable.
BIREFNET_WEIGHTS = "birefnet.gguf"


def default_bg_removal(trellis_models_dir: Path) -> str:
    """The matte to use when the caller named none: the learned one when its
    weights are on disk, ``auto`` when they are not."""
    return (
        DEFAULT_BG_REMOVAL
        if (trellis_models_dir / BIREFNET_WEIGHTS).exists()
        else FALLBACK_BG_REMOVAL
    )

# What a TRELLIS reference image must not be. A second subject or a cropped one
# is the single most common cause of a mesh that reconstructs into nonsense, so
# this is a default rather than an empty field the user has to discover.
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, multiple objects, cropped, cut off, "
    "text, watermark, signature, busy background, human hands"
)
MAX_NEGATIVE_PROMPT = 1000


@dataclass(frozen=True, slots=True)
class Option:
    key: str
    label: str
    prompt: str
    # Unused since the taxonomy retirement (the categories carried one); kept
    # so ``catalog`` needs no shape change if a sized table ever returns.
    default_size_m: float | None = None
    # Platforms only: the geometry resolution passed to trellis-server.
    resolution: int | None = None
    # Kept in the table but not offered in the UI. ``normalize`` still accepts
    # the key -- the option is only withdrawn from the selects ``catalog``
    # builds. No current instance since the taxonomy retirement; the mechanism
    # stays for the next option that loses its measurement.
    hidden: bool = False


def _table(*options: Option) -> dict[str, Option]:
    return {opt.key: opt for opt in options}


# What the asset is *for*, which is the only thing the user actually knows at
# the point of asking: a 2D asset is going to be seen flat and small, a 3D one
# in a scene. The geometry resolution follows from that rather than from a
# hardware tier the user has to translate.
PLATFORMS = _table(
    Option(
        "2d", "2D",
        "clean readable silhouette, simple flat forms",
        resolution=512,
    ),
    Option(
        "3d", "3D",
        "moderate surface detail, clear material separation",
        resolution=1024,
    ),
)

# Whether -- and toward what -- the local GPT-2 prompt expander enriches the
# user's prompt before SDXL sees it (pipelines/expand.py). The DALL-E 3
# principle, offline: every hosted image service rewrites short prompts into
# dense descriptive ones before its image model runs. Not prompt *fragments*
# (the option prompts are empty and _PROMPT_FIELDS stays empty): the field
# selects machinery, and the expansion itself is recorded by the worker as
# the derived ``expanded_prompt``.
#
# "off" is a real option rather than only the absent default so the UI combo
# has a first entry to fall back to; normalize stores the key only when a mode
# is actually on, so no stored vector re-keys.
EXPAND_MODES = _table(
    Option("off", "Off", ""),
    Option("asset", "3D asset", ""),
    Option("scene", "General 2D", ""),
)
DEFAULT_EXPAND = "off"

_OPTION_TABLES: dict[str, dict[str, Option]] = {
    "platform": PLATFORMS,
    "expand": EXPAND_MODES,
}

# Keys that used to exist, mapped onto the ones that replaced them. Every job
# row already on disk carries the old spellings, and rerun/promotion re-run
# normalize() over stored params -- without this a stored ``platform: "hero"``
# would 400 rather than reroll. normalize() writes the canonical key back, so
# params migrate the first time a job is touched. The retired taxonomy keys
# need no aliases: normalize() never reads them, so they are simply ignored.
_LEGACY_ALIASES: dict[str, dict[str, str]] = {
    "platform": {"mobile": "2d", "desktop": "3d", "hero": "3d"},
}


def _canonical(field: str, value: str) -> str:
    """The current key for a value, which for anything current is itself."""
    return _LEGACY_ALIASES.get(field, {}).get(value, value)

# The model-selection fields are validated and stored here so the API gets its
# 400-on-unknown from the same place as everything else, but they are owned by
# models.py and are deliberately absent from _PROMPT_FIELDS: a checkpoint is
# not a prompt fragment, and a LoRA's trigger words are model scaffolding that
# belongs next to PROMPT_TEMPLATE, not creative direction.
_TABLES: dict[str, dict[str, Any]] = {
    **_OPTION_TABLES,
    "base_model": models.BASE_MODELS,
    "style_lora": models.STYLE_LORAS,
    # Conditioning selections. Same reasoning as base_model/style_lora: they
    # are validated here so the 400 comes from one place, and are absent from
    # _PROMPT_FIELDS because they are not prompt fragments -- compose_prompt's
    # output is byte-identical with and without any of them.
    "ip_adapter": models.IP_ADAPTERS,
    "control": models.CONTROLNETS,
}

# Empty since the 2026-08-17 taxonomy retirement: no stored field composes
# into the prompt any more. compose_prompt survives as a shell (the workers
# still call it) and the tuple survives so its emptiness is a stated fact
# rather than an accident a future field lands in unnoticed.
_PROMPT_FIELDS: tuple[str, ...] = ()


def form_fields() -> tuple[str, ...]:
    """Every field name the API accepts as a taxonomy value."""
    return tuple(_TABLES)


def _lookup(field: str, value: Any) -> Any | None:
    """None for 'not specified'; ValueError for a value we do not know."""
    if value is None or value == "":
        return None
    table = _TABLES[field]
    option = table.get(_canonical(field, str(value)))
    if option is None:
        raise GuidanceError(
            f"unknown {field} {value!r}; expected one of {sorted(table)}", field=field
        )
    return option


def _number(raw: dict[str, Any], field: str, *, default: float, low: float, high: float) -> float:
    """A bounded float field, with the same missing-or-empty-means-default rule
    lora_weight has had since it was the only one."""
    value = raw.get(field)
    if value in (None, ""):
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise GuidanceError(f"{field} must be a number, got {value!r}", field=field) from exc
    if not low <= value <= high:
        raise GuidanceError(
            f"{field} must be between {low} and {high}, got {value!r}", field=field
        )
    return value


def normalize(raw: dict[str, Any], *, bg_default: str | None = None) -> dict[str, Any]:
    """Validate submitted guidance and fill in what it implies.

    Returns the dict stored in jobs.params, so it also carries the derived
    ``resolution`` the worker actually sends to trellis-server. Raises
    ValueError (which the API turns into a 400) on any unrecognised value.

    ``bg_default`` is the matte to use when ``raw`` names none -- what
    ``default_bg_removal`` resolved against the host's weights directory. It is
    a passed value rather than a lookup because this module is pure and holds
    no Config; ``None`` means "no gate was applied", which is the stated
    preference and what the validation-only callers (bench recipes, the sweep
    parser) want, since they never reach a server.
    """
    chosen = {field: _lookup(field, raw.get(field)) for field in _TABLES}

    platform = chosen["platform"] or PLATFORMS[DEFAULT_PLATFORM]
    resolution = raw.get("resolution")
    # An explicit resolution stays an override; otherwise the platform supplies
    # it. The override is bounded by the resolutions ``vram`` prices
    # (TRELLIS_RES_MULT): admission re-buckets anything it does not know to the
    # 1024 baseline, so an unpriced value here would be *sent* to trellis-server
    # verbatim while being *charged* as the cheap case -- accounting as the
    # optimistic half, which the VRAM invariant forbids.
    if resolution in (None, ""):
        resolution = platform.resolution
    else:
        from . import vram

        try:
            resolution = int(resolution)
        except (TypeError, ValueError) as exc:
            raise GuidanceError(
                f"resolution must be a number, got {resolution!r}",
                field="platform",
            ) from exc
        if resolution not in vram.TRELLIS_RES_MULT:
            raise GuidanceError(
                f"resolution must be one of {sorted(vram.TRELLIS_RES_MULT)}, "
                f"got {resolution!r}",
                field="platform",
            )

    size_m = raw.get("size_m")
    if size_m in (None, ""):
        size_m = DEFAULT_SIZE_M
    else:
        try:
            size_m = float(size_m)
        except (TypeError, ValueError) as exc:
            raise GuidanceError(
                f"size_m must be a number, got {size_m!r}", field="size_m"
            ) from exc
        if not SIZE_MIN_M <= size_m <= SIZE_MAX_M:
            raise GuidanceError(
                f"size_m must be between {SIZE_MIN_M} and {SIZE_MAX_M} metres, "
                f"got {size_m!r}",
                field="size_m",
            )

    base_model = chosen["base_model"] or models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    style_lora = chosen["style_lora"]

    lora_weight = raw.get("lora_weight")
    if lora_weight in (None, ""):
        # The LoRA's own tuned default, so picking a style is a one-click action.
        lora_weight = style_lora.default_weight if style_lora else models.DEFAULT_LORA_WEIGHT
    else:
        try:
            lora_weight = float(lora_weight)
        except (TypeError, ValueError) as exc:
            raise GuidanceError(
                f"lora_weight must be a number, got {lora_weight!r}", field="lora_weight"
            ) from exc
        if not models.LORA_WEIGHT_MIN <= lora_weight <= models.LORA_WEIGHT_MAX:
            raise GuidanceError(
                f"lora_weight must be between {models.LORA_WEIGHT_MIN} "
                f"and {models.LORA_WEIGHT_MAX}, got {lora_weight!r}",
                field="lora_weight",
            )

    if style_lora is not None and not models.lora_fits(base_model, style_lora):
        # Symmetric with the ControlNet refusal below, and refused for a
        # stronger reason: an adapter names one architecture's modules, so
        # loading it onto another raises at load time with the checkpoint
        # already in VRAM rather than merely doing nothing.
        #
        # Two controls are in conflict and only one can be highlighted, so it
        # is the one the message tells the user to change -- and which one that
        # is depends on whether a remedy exists on the style side. When some
        # adapter fits this base, the base is fine and the style is wrong; when
        # none does, the style is fine and the base is. Ringing the other would
        # highlight a control the sentence does not mention, which is worse
        # than ringing nothing.
        fitting = models.loras_by_base().get(base_model.key) or []
        if fitting:
            raise GuidanceError(
                f"style_lora {style_lora.key!r} is fitted to {style_lora.family} "
                f"and base_model {base_model.key!r} is {base_model.family}; "
                f"pick one of {fitting}",
                field="style_lora",
            )
        raise GuidanceError(
            f"base_model {base_model.key!r} cannot take a style LoRA; "
            f"pick one of {models.lora_bases()}",
            field="base_model",
        )

    ip_adapter = chosen["ip_adapter"]
    control = chosen["control"]
    if ip_adapter is not None and ip_adapter.family != base_model.family:
        # The third conditioning half, gated like the other two. Without this
        # the only refusal was ``Text2Image._conditioned``'s, which fires with
        # the checkpoint already resident -- the failure ``lora_fits`` exists
        # to keep in front of the load.
        raise GuidanceError(
            f"ip_adapter {ip_adapter.key!r} is fitted to {ip_adapter.family} "
            f"and base_model {base_model.key!r} is {base_model.family}; "
            f"pick one of {models.ip_adapter_bases()}",
            field="ip_adapter",
        )
    if control is not None and not base_model.controlnet:
        raise GuidanceError(
            f"base_model {base_model.key!r} cannot run a ControlNet; "
            f"pick one of {models.controlnet_bases()}",
            field="base_model",
        )
    if control is not None and control.preprocessor is None:
        # Pipeline-fed: the re-texture stage renders this hint itself from the
        # mesh. An image job has no mesh to render one from, and accepting the
        # key here would fail at write_hint with the checkpoint already in
        # VRAM. catalog() already hides it from the combo; this is the door
        # for params that arrive from anywhere else.
        raise GuidanceError(
            f"control {control.key!r} takes a hint rendered by a pipeline, "
            "not one derived from the reference; it cannot be chosen here",
            field="control",
        )

    # The selection's own tuned default, so picking an adapter is one click --
    # the same rule lora_weight follows.
    ip_scale = _number(
        raw, "ip_scale",
        default=ip_adapter.default_scale if ip_adapter else models.DEFAULT_IP_SCALE,
        low=models.IP_SCALE_MIN, high=models.IP_SCALE_MAX,
    )
    control_scale = _number(
        raw, "control_scale",
        default=control.default_scale if control else models.DEFAULT_CONTROL_SCALE,
        low=models.CONTROL_SCALE_MIN, high=models.CONTROL_SCALE_MAX,
    )
    control_end = _number(
        raw, "control_end",
        default=control.default_end if control else models.DEFAULT_CONTROL_END,
        low=models.CONTROL_END_MIN, high=models.CONTROL_END_MAX,
    )
    # img2img: the reference is the picture the denoise *starts from*. A flag
    # and a strength, gated like the other two halves -- a strength with no
    # start image would read as a live setting on rerun -- and refused off
    # the SDXL family here rather than in ``_conditioned`` with the checkpoint
    # already resident.
    init_image = bool(raw.get("init_image"))
    if init_image and base_model.family != models.FAMILY_SDXL:
        raise GuidanceError(
            f"base_model {base_model.key!r} cannot start from an image; "
            f"pick an SDXL-family model",
            field="init_image",
        )
    init_strength = _number(
        raw, "init_strength",
        default=models.DEFAULT_IMG2IMG_STRENGTH,
        low=models.IMG2IMG_STRENGTH_MIN, high=models.IMG2IMG_STRENGTH_MAX,
    )

    bg_removal = raw.get("bg_removal")
    if bg_removal in (None, ""):
        bg_removal = bg_default or DEFAULT_BG_REMOVAL
    # Validated after the fallback rather than only on the explicit branch: the
    # default is a value a caller hands in now, and an unknown one would
    # otherwise reach trellis-server unchecked -- the one path where "the
    # server decides" is not a safe answer.
    if str(bg_removal) not in BG_REMOVAL:
        raise GuidanceError(
            f"bg_removal must be one of {list(BG_REMOVAL)}, got {bg_removal!r}",
            field="bg_removal",
        )

    # Unlike bg_removal, a missing key and an explicit "" are NOT the same
    # thing here: missing means "use the game-asset default", an explicit
    # empty string means the user deliberately wants no negative prompt at all.
    negative = raw.get("negative_prompt")
    if negative is None:
        negative = DEFAULT_NEGATIVE_PROMPT
    negative = str(negative).strip()
    if len(negative) > MAX_NEGATIVE_PROMPT:
        raise GuidanceError(
            f"negative_prompt must be at most {MAX_NEGATIVE_PROMPT} characters",
            field="negative_prompt",
        )

    out: dict[str, Any] = {
        "resolution": resolution,
        "size_m": size_m,
        "platform": platform.key,
        "bg_removal": str(bg_removal),
        "negative_prompt": negative,
        # Always present, unlike the optional fields below: the worker needs a
        # base model for every text job and must never have to guess one.
        "base_model": base_model.key,
    }
    # Only carried when a mode is actually on: "off" and absent are one state,
    # and storing "off" on every job would re-key every same-settings vector
    # (``expand`` is in vectors.VECTOR_PARAMS).
    expand = chosen["expand"]
    if expand is not None and expand.key != DEFAULT_EXPAND:
        out["expand"] = expand.key
    # Only carried when a style was actually chosen -- a stored lora_weight with
    # no lora would read as "a LoRA at 0.9" on rerun.
    if style_lora is not None:
        out["style_lora"] = style_lora.key
        out["lora_weight"] = lora_weight
    # Same rule again for both conditioning halves: a scale with nothing to
    # scale reads as a live setting on rerun.
    if ip_adapter is not None:
        out["ip_adapter"] = ip_adapter.key
        out["ip_scale"] = ip_scale
    if control is not None:
        out["control"] = control.key
        out["control_scale"] = control_scale
        out["control_end"] = control_end
    if init_image:
        out["init_image"] = True
        out["init_strength"] = init_strength
    return out


def compose_prompt(
    user_prompt: str, params: dict[str, Any], fields: tuple[str, ...] | None = None
) -> str:
    """The subject clause of the SDXL prompt.

    A shell since the taxonomy retirement: ``_PROMPT_FIELDS`` is empty, so the
    output is the stripped user prompt. The function survives because every
    worker path calls it, and because it is the one place a future prompt-
    composing field would re-enter. ``fields`` keeps its meaning (a restriction
    over ``_PROMPT_FIELDS``) and is inert while the tuple is empty. Unknown or
    stale params keys are never read, so a pre-retirement job composes cleanly
    with its fragments simply gone.
    """
    parts = [user_prompt.strip()]
    for field in _PROMPT_FIELDS:
        if fields is not None and field not in fields:
            continue
        option = _TABLES[field].get(_canonical(field, str(params.get(field, ""))))
        if option is not None:
            parts.append(option.prompt)
    return ", ".join(p for p in parts if p)


def catalog(*, bg_default: str | None = None) -> dict[str, Any]:
    """The taxonomy as JSON, so the UI builds its selects from one source.

    ``bg_default`` is ``normalize``'s, for the same reason and with the same
    meaning: the form's initial matte must be the one a submit would pick, or
    the pane shows a setting the job will not run at.
    """
    return {
        "fields": {
            field: [
                {
                    "key": opt.key,
                    "label": opt.label,
                    **({"default_size_m": opt.default_size_m} if opt.default_size_m else {}),
                    **({"resolution": opt.resolution} if opt.resolution else {}),
                }
                for opt in table.values()
                if not opt.hidden
            ]
            for field, table in _OPTION_TABLES.items()
        }
        | models.catalog(),
        "bg_removal": list(BG_REMOVAL),
        # Not a select, so not under "fields": what the UI checks the chosen
        # base against before it draws the Structure group at all.
        "controlnet_bases": models.controlnet_bases(),
        # Same shape and the same purpose as controlnet_bases: what the UI
        # checks the chosen base against before it presents the negative
        # prompt as a live control rather than an inert one.
        "cfg_bases": models.cfg_bases(),
        # And again for the style LoRA: what the UI checks before it offers the
        # picker as a live control rather than a disabled one with a reason.
        "lora_bases": models.lora_bases(),
        # And which ones, once the picker is live. Deliberately not a family on
        # each style_lora option: that would be a second spelling of the
        # pairing, and the UI never needs an adapter's family, only whether the
        # chosen base is offered it.
        "loras_by_base": models.loras_by_base(),
        "defaults": {
            "platform": DEFAULT_PLATFORM,
            "expand": DEFAULT_EXPAND,
            "size_m": DEFAULT_SIZE_M,
            "base_model": models.DEFAULT_BASE_MODEL,
            "lora_weight": models.DEFAULT_LORA_WEIGHT,
            "bg_removal": bg_default or DEFAULT_BG_REMOVAL,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "init_strength": models.DEFAULT_IMG2IMG_STRENGTH,
        },
        "size_range_m": [SIZE_MIN_M, SIZE_MAX_M],
        "lora_weight_range": [models.LORA_WEIGHT_MIN, models.LORA_WEIGHT_MAX],
        "ip_scale_range": [models.IP_SCALE_MIN, models.IP_SCALE_MAX],
        "control_scale_range": [models.CONTROL_SCALE_MIN, models.CONTROL_SCALE_MAX],
        "control_end_range": [models.CONTROL_END_MIN, models.CONTROL_END_MAX],
        "init_strength_range": [models.IMG2IMG_STRENGTH_MIN, models.IMG2IMG_STRENGTH_MAX],
    }

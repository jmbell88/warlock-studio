"""The config-vector vocabulary, and the machine evidence read off a job.

This lives at the package top level rather than in ``service/`` for one
reason: the worker (queue.py) writes an observation row when a model job
finishes, and queue.py deliberately imports no ``service`` module. The
vocabulary both halves must agree on -- what a vector is made of, how it is
canonicalized, how it is keyed -- therefore sits where both can reach it.
``service.findings`` re-exports the three names that were already spelled
against it (``VECTOR_PARAMS``, ``config_vector``, ``vector_key``), so callers
that learned the old import path never notice the move; ``prompt_hash`` and
``observation_metrics`` are new here and are imported from here. Pure stdlib,
and it must stay that way: it is imported before torch exists and inside the
frame loop's reach.

``observation_metrics`` is the defensive half: it extracts the numbers an
observation row carries from ``params`` written by ``_audit_mesh`` and
``meshreport.build``. Both measurements are advisory -- either can be absent,
partial, or (after a failure someone hand-edited around) the wrong shape --
so every malformed piece costs its own key and nothing else. It runs on the
worker's completion path, where a diagnostic must never fail a job whose
mesh is already on disk.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# The gate's own vocabulary, not a second copy of it: a rule added to
# reference.py would otherwise record a rate under a name nothing aggregates.
# Safe against this module's stdlib-only rule -- pipelines/reference.py imports
# nothing but stdlib at module scope (cv2/numpy/Pillow are imported inside the
# functions that use them), which is the same discipline that lets it be
# measured headlessly.
from .pipelines.reference import REFUSAL_CODES

# What a config vector is made of: every param that is an *input* to a
# generation and could plausibly be varied. An allowlist rather than "params
# minus DERIVED_PARAMS" because the interesting property is that adding a
# derived value to queue.py can never silently widen it.
#
# Deliberately absent, each for its own reason: ``prompt`` (a property of the
# subject, not of the settings -- grouping on it would make every bucket n=1),
# every seed (the thing a repeat is *for*), everything in
# ``validation.DERIVED_PARAMS`` (measurements of one run's artifacts), the rig
# and pose keys (they describe a follow-up job, not this one's settings), and
# ``hand_edited``/``imported``/``built``/``rerun_of`` (provenance, not config).
VECTOR_PARAMS = (
    "category",
    "silhouette",
    "material",
    "condition",
    "rarity",
    "emissive",
    "setting",
    "genre",
    "mood",
    "art_style",
    "palette",
    "platform",
    "base_model",
    "style_lora",
    "lora_weight",
    "negative_prompt",
    "ip_adapter",
    "ip_scale",
    "control",
    "control_scale",
    "control_end",
    "resolution",
    "size_m",
    "bg_removal",
    "reference_prep",
    "profile",
    "custom_triangles",
    "trellis_band",
    "trellis_tex_res",
)


def config_vector(job: dict[str, Any]) -> dict[str, Any]:
    """The settings that produced this job, canonicalized.

    Canonical in three ways, all of which exist so two runs of the same
    configuration land in the same bucket. Unset is *absent* rather than
    ``None``/``""`` -- a form sends empty strings for every taxonomy field the
    user left alone, and keeping them would make "no style chosen" a distinct
    value from "the key was never there". Floats are rounded to six decimals,
    because an imgui float32 slider hands back ``0.6000000238418579`` for the
    0.6 a spec file wrote. And ``stage`` comes along, because a reference and a
    mesh judged under the same settings are not the same thing.
    """
    params = job.get("params") or {}
    out: dict[str, Any] = {}
    for key in VECTOR_PARAMS:
        if key not in params:
            continue
        value = params[key]
        if value is None or value == "":
            continue
        if isinstance(value, float):
            value = round(value, 6)
        out[key] = value
    out["stage"] = job.get("stage") or "model"
    return out


def vector_key(vector: dict[str, Any]) -> str:
    """A short stable id for a vector -- the same settings always hash the
    same, whatever order the dict was built in."""
    blob = json.dumps(vector, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def prompt_hash(text: str | None) -> str:
    """A short id for a prompt, or ``""`` for none -- comparisons use it only
    to count how many distinct prompts an axis finding spans."""
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def observation_metrics(params: dict[str, Any]) -> dict[str, Any]:
    """The machine measurements an observation row snapshots, keys absent
    when unmeasured. ``faces``/``resolution`` stay behind on purpose -- they
    are audit internals; ``triangles`` is the number anyone acts on."""
    out: dict[str, Any] = {}
    audit = params.get("mesh_audit")
    if isinstance(audit, dict):
        for src, dst in (("worst", "hole_worst"), ("mean", "hole_mean")):
            value = audit.get(src)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[dst] = float(value)
    report = params.get("mesh_report")
    if isinstance(report, dict):
        watertight = report.get("watertight")
        if isinstance(watertight, bool):
            out["watertight"] = watertight
        triangles = report.get("triangles")
        if isinstance(triangles, int) and not isinstance(triangles, bool):
            out["triangles"] = triangles
        status = report.get("status")
        if isinstance(status, str):
            out["ready"] = status == "ready"
    out.update(_refusal_metrics(params.get("reference_report")))
    return out


def _refusal_metrics(report: Any) -> dict[str, float]:
    """Whether the composition gate refused this job, and for what.

    Recorded as 0.0/1.0 numbers rather than as a flag on the refused jobs
    alone, and that is the whole design: the mean of ``refused`` over a bucket
    is only the refusal *rate* if the jobs that passed contribute zeros. With
    the refusals alone every bucket would read 100%.

    The 17 refusals in the 2026-08-07 rogue sweep wrote nothing at all, so a
    reader saw each checkpoint's accept rate *among the references that
    survived* -- which flatters exactly the checkpoints that fail most often.
    ``sdxl_cfg`` refused 3 of 5 and ``playground`` 0 of 5, and findings.json
    could not say so.

    Defensive in the way the two measurements above are: a report that crossed
    a disk can be any shape, and a partial one costs its own keys and no more.
    An absent ``ok`` says nothing rather than "it passed".
    """
    if not isinstance(report, dict) or not isinstance(report.get("ok"), bool):
        return {}
    refused = not report["ok"]
    out: dict[str, float] = {"refused": float(refused)}
    codes = report.get("codes")
    if refused and not isinstance(codes, list):
        # A refusal recorded before ``codes`` existed. The aggregate rate is
        # still true and still worth having; which rule fired is not knowable,
        # and a zero for every reason would be a claim that none of them did.
        return out
    named = set(codes or ()) if isinstance(codes, list) else set()
    out.update({f"refused_{code}": float(code in named) for code in REFUSAL_CODES})
    return out

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
    return out

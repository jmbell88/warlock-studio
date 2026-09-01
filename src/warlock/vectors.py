"""The config-vector vocabulary, and the machine evidence read off a job.

This lives at the package top level rather than in ``service/`` for one
reason: the worker (queue.py) writes an observation row when a model job
finishes, and queue.py deliberately imports no ``service`` module. The
vocabulary both halves must agree on -- what a vector is made of, how it is
canonicalized, how it is keyed -- therefore sits where both can reach it.
``service.findings`` re-exports the three names that were already spelled
against it (``VECTOR_PARAMS``, ``config_vector``, ``vector_key``), so callers
that learned the old import path never notice the move; ``prompt_hash`` and
``observation_metrics`` are new here and are imported from here.

**What this module may import, exactly**: the standard library, and the one
name it takes from ``pipelines.reference`` (``REFUSAL_CODES``, whose own module
is stdlib-only at module scope). Never ``service``, never ``studio``, never
torch, never imgui -- it is imported before torch exists and inside the frame
loop's reach, and it is what the worker writes observations through. The rule
is not prose: ``tests/test_vectors_evidence.py`` imports this module in a
subprocess and refuses any of those having been dragged in.

The mesh-verdict **grade scale** and its **tag vocabulary** are here for exactly
the same reason and not a related one: ``service/verdicts.py`` imports
``service/findings.py`` at module top, so a constant both need cannot live in
the importer. ``service.verdicts`` re-exports all of it.

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
# The twelve taxonomy keys were removed on 2026-08-17 (see
# docs/measurements/2026-08-17-taxonomy-retirement.md): same-settings vectors
# re-key and evidence accumulation restarts per configuration, the accepted
# cost that document records. A stale key in old stored params is skipped by
# config_vector's membership test, never an error.
VECTOR_PARAMS = (
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
    "init_image",
    "init_strength",
    "resolution",
    "size_m",
    "bg_removal",
    # Whether the reference carried a matte the user approved. An *input*, and
    # a deliberate one: it is set at the door by ``service.matte.approve`` and
    # never by the worker, so it stays out of ``validation.DERIVED_PARAMS`` and
    # survives a reroll. Here so findings can compare an approved matte against
    # the server's own -- which is the whole question the promote preview and
    # the Inker hand-off exist to let a user answer.
    "matte",
    "reference_prep",
    "profile",
    "custom_triangles",
    "trellis_band",
    "trellis_tex_res",
)


# --- The mesh-verdict grade scale, and the tag vocabulary beside it. ---------
#
# Here rather than in ``service/verdicts.py`` for the reason ``VECTOR_PARAMS`` is
# here: ``verdicts.py`` imports ``service.findings`` at module top, so a constant
# both of them need cannot live in the importer. ``service.verdicts`` re-exports
# every name below, so callers that learned the old spelling never notice.
#
# An integer scale rather than a bit, because the binary corpus failed at its own
# purpose: 3 accepts against 81 rejects on 2026-08-07, in which a slab with no
# geometry, a smeared texture and a mesh a modeller would fix in five minutes are
# all the same row. See ``docs/measurements/2026-08-09-grade-scale.md``, which is
# where every number here is argued -- the repo rule is that a constant the stored
# corpus is keyed on gets its document before it changes.
#
# Model (mesh) stage only. Image labels stay binary and keep ``grade`` NULL: they
# feed binary logistic probes, so a grade would be thresholded straight back to a
# bit, and the two-key loop is what makes a hundred-image pass viable at all.
GRADE_MIN, GRADE_MAX = -5, 5

# The one binary cut, and the only threshold in the whole scale. ``verdict`` TEXT
# survives as a *derived* column written from this by one writer, which is what
# keeps prune retention, the judge's label reads, the ``latest_verdicts`` SQL and
# every findings-v3 reader unchanged.
USABLE_GRADE = 3

# What migration 10 backfills a binary row to, and the fallback for a row that
# has no grade. Deliberately the mildest grade that preserves the row's meaning:
# a reviewer who pressed Accept asserted "usable" and had no key for anything
# stronger, so +-5 would invent evidence 84 times over and leave every real grade
# recorded later sitting inside synthetic tails. +-3 keeps +-4/+-5 free for
# judgements a binary reviewer never made.
#
# ``BINARY_GRADES["accept"] == USABLE_GRADE`` is not a coincidence and is
# asserted rather than commented: the backfill value and the cut are one decision
# with two names. A cut above +3 would silently demote every row the migration
# just wrote; a cut below it would claim a grade nobody recorded was already
# usable.
BINARY_GRADES = {"accept": USABLE_GRADE, "reject": -USABLE_GRADE}

# Tags, optional at every grade, in one namespace.
#
# The five bad spellings are **frozen**: they are the old ``REASONS`` tuple and
# the stored corpus already carries those exact strings in the ``reasons`` JSON
# column. There is no alias table here as there is for taxonomy keys, so a rename
# would not migrate evidence, it would split it. Only the name of the concept
# changed -- from "reasons a reviewer rejected" to "tags describing what is true
# of this mesh" -- which is why they are legal at any grade now.
BAD_TAGS = ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")

# Mirrors of the bad set where a mirror exists, plus the two that have none:
# ``sharp-detail`` (detail survived reconstruction -- what trellis most often
# loses, and what a bare "not broken" cannot say) and ``good-topology`` (the
# constructive counterpart of ``broken``: it imports and derives cleanly).
#
# Deliberately no ``game-ready``: that is what grade +5 already asserts, and a tag
# saying the same thing as a grade is two spellings of one fact.
#
# Five per polarity is load-bearing rather than tidy -- it is what makes
# Ctrl+1..5 and Shift+1..5 map *positionally* onto the two vocabularies, so the
# keyboard needs no second table naming which digit is which tag.
GOOD_TAGS = ("clean-shape", "good-texture", "on-style", "sharp-detail", "good-topology")

# One namespace, stored in the existing ``reasons`` column. Sound only because the
# two vocabularies are disjoint strings: polarity is recoverable from the tag
# itself, so nothing is written down twice and the findings writer splits by
# membership -- which is what keeps ``bench/findings.py`` pure-stdlib with no
# ``warlock`` imports.
TAGS = GOOD_TAGS + BAD_TAGS


def verdict_for_grade(grade: int) -> str:
    """The derived binary answer for a grade -- the *only* place the cut lives.

    One writer owns this, which is what stops ``verdict`` and ``grade`` from
    becoming the two-spellings hazard: the column can never disagree with the
    grade it was derived from, because nothing else writes it.
    """
    return "accept" if grade >= USABLE_GRADE else "reject"


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
        # The welded figure when the report carries one: the raw one counts
        # every xatlas UV-seam split as a boundary, so on a textured mesh it
        # measures seams rather than holes. Rows written before the welded
        # analysis existed keep contributing their raw flag under the same key.
        watertight = report.get("welded_watertight", report.get("watertight"))
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

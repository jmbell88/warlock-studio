# Asset-Consistency Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loops between what Warlock already measures and what it does — honest UI about inert controls, a per-profile style anchor that conditions every generation in a set, an advisory rank score on reference candidates, and two bounded, opt-in retries driven by measurements the app already takes.

**Architecture:** Every new decision is a pure function in a module with no imgui / service / queue imports (`pipelines/rank.py`, new helpers in `bench/imageprep.py`, `models.py`, `studio/profiles.py`), so it is testable headlessly. The worker (`queue.py`) calls them off the event loop through `asyncio.to_thread`, records their answers into `params`, and every such key joins `service.validation.DERIVED_PARAMS`. Nothing new fails a job whose artifact is already on disk; the two retries are bounded, config-gated and off by default.

**Tech Stack:** Python 3.12+, uv, pytest, ruff. imgui-bundle for the panes, Pillow/NumPy for image measurement, DINOv2 through `transformers` (optional weights, CPU only in the app path).

## Global Constraints

- Run every command from `D:\Projects\Warlock`. Verify with `uv run pytest` and `uv run ruff check .` — both must pass before each commit.
- Commit subject format is exactly `Warlock v0.0.7` (project name + fixed version). **Do not bump the version.** Put the detail in the commit body. End the message with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- **Fully offline.** No code may make a runtime network call. Model weights load from local paths with `local_files_only=True`; a missing weight degrades the feature and reports the one-time manual `hf download`, never downloads.
- **The frame loop never blocks.** Anything that reads a file, loads a model or measures pixels runs on the TaskRunner (`ctx.submit(...)`) or in the worker via `asyncio.to_thread`. Pane code may only read already-computed values.
- **Anything the worker records about a finished job's artifacts must be added to `service.validation.DERIVED_PARAMS`** (`src/warlock/service/validation.py:60`), or a reroll or promotion inherits a verdict about an artifact that does not exist yet.
- **The single sqlite connection is serialized by `JobStore`'s own lock.** Write params through `self.store.set_params` / `merge_params` as the existing code does; never reach `_conn`.
- **A control belongs to exactly one pane.** The 2D pane owns everything that composes the SDXL prompt; the 3D pane owns mesh/rig/pose/sheet.
- New config values are `Config` fields with a `WARLOCK_*` env override and a default that preserves today's behaviour exactly.
- Docstrings in this codebase explain *why*, at length, and prose is British-inflected with `--` for dashes. Match the surrounding style; do not add terse one-liners to files full of essays.

---

### Task 1: UI honesty — the negative prompt and the Structure group name their models

Two footguns. `pipelines/text2image.py` only encodes a negative prompt when `spec.guidance_scale > 1.0`, so on `turbo` and `sdxl` (both CFG 0) the field is silently inert. And the Structure group already refuses non-ControlNet bases but does not say which bases would work.

**Files:**
- Modify: `src/warlock/models.py` (add `cfg_bases()` beside `controlnet_bases()` at line 320)
- Modify: `src/warlock/guidance.py:514` (add `cfg_bases` to the catalog beside `controlnet_bases`)
- Modify: `src/warlock/studio/panes/settings_2d.py` (`_reference`, `_advanced`, two new pure helpers)
- Modify: `src/warlock/studio/panes/profiles_panel.py:133` (the profile editor's own Negative field)
- Test: `tests/test_models.py`, `tests/test_guidance.py`, `tests/test_upload_pane.py` is the pane-test precedent — new file `tests/test_settings_2d_notes.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `models.cfg_bases() -> list[str]`; catalog key `"cfg_bases"`; `settings_2d.negative_prompt_note(ctx, form) -> str | None` and `settings_2d.structure_note(ctx, form) -> str | None`, both pure and taking a ctx that only needs `.guidance` (a dict) and `.base_models` (a list of `(key, label)`).

- [ ] **Step 1: Write the failing tests for the registry half**

Add to `tests/test_models.py`:

```python
def test_cfg_bases_are_exactly_the_ones_that_encode_a_negative_prompt():
    # text2image only encodes the negative prompt when guidance_scale > 1.0,
    # so this list is what the UI may present the field as live for.
    bases = models.cfg_bases()
    assert bases
    assert all(models.BASE_MODELS[b].guidance_scale > 1.0 for b in bases)
    assert all(
        m.key in bases for m in models.BASE_MODELS.values() if m.guidance_scale > 1.0
    )
    assert "turbo" not in bases
```

Add to `tests/test_guidance.py`:

```python
def test_catalog_publishes_the_cfg_bases_the_ui_gates_on():
    from warlock import models

    catalog = guidance.catalog()
    assert catalog["cfg_bases"] == models.cfg_bases()
```

(`tests/test_guidance.py` already imports `guidance`; check the existing import block and reuse it rather than adding a duplicate.)

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_models.py -k cfg_bases tests/test_guidance.py -k cfg_bases -v`
Expected: FAIL with `AttributeError: module 'warlock.models' has no attribute 'cfg_bases'`.

- [ ] **Step 3: Implement the registry half**

In `src/warlock/models.py`, immediately after `controlnet_bases()` (line 324):

```python
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
```

In `src/warlock/guidance.py`, in `catalog()` directly after the `controlnet_bases` entry (line 514):

```python
        # Same shape and the same purpose as controlnet_bases: what the UI
        # checks the chosen base against before it presents the negative
        # prompt as a live control rather than an inert one.
        "cfg_bases": models.cfg_bases(),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models.py tests/test_guidance.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing pane-helper tests**

Create `tests/test_settings_2d_notes.py`:

```python
"""What the 2D pane says about a control that cannot act.

Both notes are pure functions of (catalog, form) so the wording is assertable
without a GL context -- the pane's draw code only decides where to put them.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock import guidance, models
from warlock.studio.panes import settings_2d


def _ctx():
    catalog = guidance.catalog()
    return SimpleNamespace(
        guidance=catalog,
        base_models=[(m["key"], m["label"]) for m in catalog["fields"]["base_model"]],
    )


def test_a_cfg_base_gets_no_negative_prompt_note():
    form = {"base_model": models.cfg_bases()[0]}
    assert settings_2d.negative_prompt_note(_ctx(), form) is None


def test_a_distilled_base_is_told_the_negative_prompt_is_inert():
    note = settings_2d.negative_prompt_note(_ctx(), {"base_model": "turbo"})
    assert note is not None
    assert "no effect" in note


def test_the_note_names_a_model_the_user_could_switch_to():
    # A refusal that doesn't say what would work is a dead end -- the label,
    # not the key, because the key is not what the picker shows.
    note = settings_2d.negative_prompt_note(_ctx(), {"base_model": "turbo"})
    label = models.BASE_MODELS[models.cfg_bases()[0]].label
    assert label in note


def test_an_unset_base_is_treated_as_the_default_which_is_distilled():
    # "" means "use the configured default", which is turbo -- so the field is
    # inert and saying nothing would be the same silence the note replaces.
    assert settings_2d.negative_prompt_note(_ctx(), {"base_model": ""}) is not None


def test_a_controlnet_base_gets_no_structure_note():
    form = {"base_model": models.controlnet_bases()[0]}
    assert settings_2d.structure_note(_ctx(), form) is None


def test_the_structure_note_names_the_bases_that_can_run_one():
    note = settings_2d.structure_note(_ctx(), {"base_model": "turbo"})
    assert note is not None
    for key in models.controlnet_bases():
        assert models.BASE_MODELS[key].label in note
```

- [ ] **Step 6: Run them and watch them fail**

Run: `uv run pytest tests/test_settings_2d_notes.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'negative_prompt_note'`.

- [ ] **Step 7: Implement the helpers**

In `src/warlock/studio/panes/settings_2d.py`, after `_range` (line 305):

```python
def _base_labels(ctx: Any, keys: list[str]) -> str:
    """The picker's own labels for a set of base-model keys.

    Labels rather than keys: "sdxl_cfg" is not what the combo shows, and a
    message naming something the user cannot find in the list is worse than no
    message at all.
    """
    labels = [label for key, label in (ctx.base_models or []) if key in keys]
    return ", ".join(labels or keys)


def negative_prompt_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why the negative prompt is inert here, or None when it is live.

    A distilled base runs at guidance 0, and text2image encodes the negative
    branch only above 1.0 -- so on turbo the field accepted text, stored it in
    params and changed nothing about the image. That silence is the bug; this
    is the sentence that ends it.
    """
    bases = ctx.guidance.get("cfg_bases") or []
    if (form.get("base_model") or "") in bases:
        return None
    return (
        "This model runs at guidance 0, so the negative prompt has no effect. "
        f"It does on: {_base_labels(ctx, bases)}."
    )


def structure_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Which bases could run the ControlNet this one cannot, or None."""
    bases = ctx.guidance.get("controlnet_bases") or []
    if (form.get("base_model") or "") in bases:
        return None
    return (
        "Structure control needs a full-CFG model -- pick one of "
        f"{_base_labels(ctx, bases)} under Advanced."
    )
```

- [ ] **Step 8: Wire the notes into the pane**

In `_reference` (line 272), replace the existing muted refusal:

```python
    widgets.section("Structure")
    note = structure_note(ctx, form)
    if note is not None:
        widgets.muted(note)
        return
```

In `_advanced` (line 319), wrap the Negative field:

```python
    inert = negative_prompt_note(ctx, form)
    if inert is not None:
        # Disabled rather than hidden, and with the reason underneath: the
        # field holds text the user typed under another base, and hiding it
        # would make that text vanish without saying why.
        imgui.begin_disabled()
    before = form["negative_prompt"]
    form["negative_prompt"] = widgets.multiline("Negative", before, 54, MAX_PROMPT)
    if form["negative_prompt"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
    if inert is not None:
        imgui.end_disabled()
        widgets.muted(inert)
```

In `src/warlock/studio/panes/profiles_panel.py`, the editor's Negative field (line 133) gets the same treatment — a profile is saved once and applied for weeks, so the same silence applies there:

```python
    inert = settings_2d.negative_prompt_note(ctx, draft)
    if inert is not None:
        imgui.begin_disabled()
    draft["negative_prompt"] = widgets.multiline(
        "Negative", draft.get("negative_prompt", ""), 54, validation.MAX_PROMPT
    )
    if inert is not None:
        imgui.end_disabled()
        widgets.muted(inert)
```

with `from . import settings_2d` added to the imports at the top of `profiles_panel.py`.

- [ ] **Step 9: Run the whole suite and the linter**

Run: `uv run pytest && uv run ruff check .`
Expected: all tests PASS, ruff reports no issues.

- [ ] **Step 10: Commit**

```bash
git add src/warlock/models.py src/warlock/guidance.py src/warlock/studio/panes/settings_2d.py src/warlock/studio/panes/profiles_panel.py tests/test_models.py tests/test_guidance.py tests/test_settings_2d_notes.py
git commit -m "Warlock v0.0.7

Say when the negative prompt is inert and name the bases that can run a
ControlNet, instead of a silently dead field and an unexplained refusal.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Style anchor storage in user profiles

A profile carries words today. An anchor image is what actually holds a set together: every asset generated under the profile is IP-Adapter-conditioned on one picture.

The anchor is a **file** under `<data_dir>/profiles/<12 hex>.png`, and the profile records only the filename. Two consequences drive the design and each has a test below: a rename must carry the anchor across (the editor saves under the new name and deletes the old), and deleting a profile must not unlink a file another profile still points at.

**Files:**
- Modify: `src/warlock/studio/profiles.py`
- Test: create `tests/test_profiles_anchor.py`

**Interfaces:**
- Consumes: `models.IP_ADAPTERS`, `models.DEFAULT_IP_SCALE` from `src/warlock/models.py`.
- Produces, all in `warlock.studio.profiles`:
  - `ANCHOR_FIELDS: tuple[str, ...]` == `("anchor", "anchor_scale")`
  - `ANCHOR_ADAPTER: str` — the IP-Adapter key an anchor is applied through
  - `anchor_dir(config) -> Path`
  - `anchor_path(config, fields: dict) -> Path | None`
  - `set_anchor(settings, config, name: str, png: bytes, scale: float | None = None) -> None`
  - `clear_anchor(settings, config, name: str) -> None`
  - `active_anchor(settings, config) -> tuple[Path, float] | None`
  - `save_profile(settings, name, fields)` — unchanged signature, now preserves anchor fields
  - `delete_profile(settings, name, config=None)` — new optional third argument

- [ ] **Step 1: Write the failing tests**

Create `tests/test_profiles_anchor.py`:

```python
"""A profile's style anchor: one image on disk, referenced by filename.

The image is a file rather than base64 in studio_settings.json because that
file is rewritten on a one-second debounce for every UI preference, and a
megabyte of PNG in it would be rewritten with them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock import models
from warlock.studio import profiles


class FakeSettings:
    """The two methods profiles.py uses, with no disk behind them."""

    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


@pytest.fixture
def env(tmp_path):
    return SimpleNamespace(
        settings=FakeSettings(), config=SimpleNamespace(data_dir=tmp_path)
    )


PNG = b"\x89PNG\r\n\x1a\n-not-really-a-png-but-bytes-are-bytes"


def test_the_anchor_adapter_is_a_real_registry_key():
    # The whole feature turns into a submit-time refusal if this drifts.
    assert profiles.ANCHOR_ADAPTER in models.IP_ADAPTERS


def test_setting_an_anchor_writes_a_file_and_records_only_its_name(env):
    profiles.save_profile(env.settings, "house", {"base_model": "turbo"})
    profiles.set_anchor(env.settings, env.config, "house", PNG)

    stored = profiles.list_profiles(env.settings)["house"]
    assert stored["anchor"].endswith(".png")
    assert "/" not in stored["anchor"] and "\\" not in stored["anchor"]
    path = profiles.anchor_path(env.config, stored)
    assert path is not None and path.read_bytes() == PNG


def test_an_anchor_carries_its_own_strength(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG, scale=0.85)
    assert profiles.list_profiles(env.settings)["house"]["anchor_scale"] == 0.85


def test_an_anchor_without_a_strength_takes_the_registry_default(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    stored = profiles.list_profiles(env.settings)["house"]
    assert stored["anchor_scale"] == models.DEFAULT_IP_SCALE


def test_saving_a_profile_again_keeps_the_anchor_it_already_had(env):
    # capture() reads the *form*, which has no anchor field -- so a plain
    # re-save would drop it every time the user edited a taxonomy select.
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    profiles.save_profile(env.settings, "house", {"base_model": "sdxl"})

    stored = profiles.list_profiles(env.settings)["house"]
    assert stored["base_model"] == "sdxl"
    assert profiles.anchor_path(env.config, stored) is not None


def test_an_explicit_anchor_in_the_saved_fields_wins(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    profiles.save_profile(env.settings, "house", {"anchor": "", "anchor_scale": 0.5})
    assert profiles.list_profiles(env.settings)["house"].get("anchor") == ""


def test_clearing_an_anchor_removes_the_file(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    path = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])

    profiles.clear_anchor(env.settings, env.config, "house")

    assert not path.exists()
    assert not profiles.list_profiles(env.settings)["house"].get("anchor")


def test_replacing_an_anchor_removes_the_old_file(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    old = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])

    profiles.set_anchor(env.settings, env.config, "house", b"second-image")

    assert not old.exists()
    new = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])
    assert new.read_bytes() == b"second-image"


def test_deleting_a_profile_removes_its_anchor(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    path = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])

    profiles.delete_profile(env.settings, "house", env.config)

    assert not path.exists()


def test_deleting_a_profile_keeps_an_anchor_another_profile_still_points_at(env):
    # This is what a rename does: the editor saves under the new name and
    # deletes the old entry, and for a moment both name the same file.
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    stored = profiles.list_profiles(env.settings)["house"]
    profiles.save_profile(env.settings, "house2", dict(stored))
    path = profiles.anchor_path(env.config, stored)

    profiles.delete_profile(env.settings, "house", env.config)

    assert path.exists()
    assert profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house2"])


def test_delete_without_a_config_still_removes_the_profile(env):
    # The old two-argument call site is still valid; it just cannot tidy up.
    profiles.save_profile(env.settings, "house", {})
    profiles.delete_profile(env.settings, "house")
    assert profiles.list_profiles(env.settings) == {}


def test_a_hand_edited_anchor_name_is_refused_before_it_becomes_a_path(env):
    # The same rule rigging.pose_path follows: a caller-supplied string that
    # names a file is validated, not joined.
    assert profiles.anchor_path(env.config, {"anchor": "../../secrets.png"}) is None
    assert profiles.anchor_path(env.config, {"anchor": "nope.txt"}) is None
    assert profiles.anchor_path(env.config, {}) is None


def test_a_recorded_anchor_whose_file_is_gone_reads_as_no_anchor(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    stored = profiles.list_profiles(env.settings)["house"]
    profiles.anchor_path(env.config, stored).unlink()
    assert profiles.anchor_path(env.config, stored) is None


def test_the_active_profiles_anchor_is_what_active_anchor_returns(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG, scale=0.7)
    profiles.set_active(env.settings, "house")

    found = profiles.active_anchor(env.settings, env.config)

    assert found is not None
    path, scale = found
    assert Path(path).read_bytes() == PNG
    assert scale == 0.7


def test_no_active_profile_means_no_anchor(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    assert profiles.active_anchor(env.settings, env.config) is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_profiles_anchor.py -v`
Expected: FAIL with `AttributeError: module 'warlock.studio.profiles' has no attribute 'ANCHOR_ADAPTER'`.

- [ ] **Step 3: Implement the storage**

In `src/warlock/studio/profiles.py`, extend the imports and add the anchor half. Replace the import block and `save_profile` / `delete_profile`, and add the new functions:

```python
from __future__ import annotations

import contextlib
import re
import secrets
from pathlib import Path
from typing import Any

from .. import models

KEY = "user_profiles"
ACTIVE_KEY = "active_profile"

# Where the anchor images live, relative to the data dir. Beside the job
# directories rather than inside one: an anchor outlives every job it was
# taken from, and prune_jobs walks that root.
ANCHOR_DIR = "profiles"

# The two keys a profile carries that are *not* form fields, so capture()
# never sees them and save_profile has to preserve them by hand.
ANCHOR_FIELDS = ("anchor", "anchor_scale")

# Which adapter an anchor is applied through. "plus" conditions on 16 patch
# tokens rather than one pooled embedding, which is the difference between
# "same kind of object" and "this look" -- exactly what an anchor is for.
ANCHOR_ADAPTER = "plus"

# Exactly what _new_anchor_name generates. A profile is a JSON blob a user can
# edit by hand, and this string becomes a path.
_ANCHOR_RE = re.compile(r"^[0-9a-f]{12}\.png$")
```

(keep `_FIXED`, `TAXONOMY`, `profile_fields`, `capture`, `apply`, `list_profiles`, `get_active`, `set_active`, `active_fields` exactly as they are.)

```python
def save_profile(settings: Any, name: str, fields: dict[str, Any]) -> None:
    name = (name or "").strip()
    if not name:
        return
    profiles = list_profiles(settings)
    merged = dict(fields)
    existing = profiles.get(name) or {}
    for key in ANCHOR_FIELDS:
        # Preserved rather than captured: the anchor is not a form field, so
        # capture() cannot see it and an ordinary re-save would drop it every
        # time the user changed a select. An explicit value still wins, which
        # is what makes clear_anchor a plain save.
        if key not in merged and key in existing:
            merged[key] = existing[key]
    # A fresh dict rather than a mutation in place: Settings.set compares the
    # old value to the new one to decide whether anything is dirty, and an
    # object edited under it compares equal to itself.
    profiles[name] = merged
    settings.set(KEY, profiles)


def delete_profile(settings: Any, name: str, config: Any = None) -> None:
    profiles = list_profiles(settings)
    removed = profiles.pop(name, None)
    if removed is None:
        return
    settings.set(KEY, profiles)
    if config is not None:
        # After the write, and only when nothing else points at it: the editor
        # renames by saving under the new name and deleting the old, so both
        # entries name the same file for exactly one call.
        _drop_anchor_file(settings, config, removed.get("anchor"))
    if get_active(settings) == name:
        set_active(settings, None)


# --- the style anchor -------------------------------------------------------


def anchor_dir(config: Any) -> Path:
    return Path(config.data_dir) / ANCHOR_DIR


def anchor_path(config: Any, fields: dict[str, Any] | None) -> Path | None:
    """The anchor image on disk, or None.

    None covers all three ways there is no usable anchor: the profile never
    had one, the recorded name is not one this module wrote (studio_settings
    .json is a file a user can edit, and this string becomes a path), or the
    file has since been deleted -- in which case a stale name must read as no
    anchor rather than as a missing-file crash at submit time.
    """
    name = str((fields or {}).get("anchor") or "")
    if not _ANCHOR_RE.match(name):
        return None
    path = anchor_dir(config) / name
    return path if path.exists() else None


def set_anchor(
    settings: Any, config: Any, name: str, png: bytes, scale: float | None = None
) -> None:
    """Point a profile at a new anchor image, replacing any it had."""
    profiles = list_profiles(settings)
    if name not in profiles:
        return
    previous = profiles[name].get("anchor")
    filename = f"{secrets.token_hex(6)}.png"
    directory = anchor_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_bytes(png)
    save_profile(
        settings,
        name,
        {
            **profiles[name],
            "anchor": filename,
            "anchor_scale": (
                models.DEFAULT_IP_SCALE if scale is None else float(scale)
            ),
        },
    )
    _drop_anchor_file(settings, config, previous)


def clear_anchor(settings: Any, config: Any, name: str) -> None:
    profiles = list_profiles(settings)
    if name not in profiles:
        return
    previous = profiles[name].get("anchor")
    save_profile(settings, name, {**profiles[name], "anchor": "", "anchor_scale": ""})
    _drop_anchor_file(settings, config, previous)


def active_anchor(settings: Any, config: Any) -> tuple[Path, float] | None:
    """-> (image, strength) for the active profile's anchor, or None."""
    fields = active_fields(settings)
    path = anchor_path(config, fields)
    if path is None:
        return None
    try:
        scale = float(fields.get("anchor_scale") or models.DEFAULT_IP_SCALE)
    except (TypeError, ValueError):
        scale = models.DEFAULT_IP_SCALE
    return (path, scale)


def _drop_anchor_file(settings: Any, config: Any, filename: Any) -> None:
    """Unlink an anchor image nothing points at any more.

    The reference count is the point. A rename saves the profile under its new
    name and deletes the old entry, so for one call two entries name the same
    file -- unlinking on sight would delete the anchor of the profile that was
    just created.
    """
    name = str(filename or "")
    if not _ANCHOR_RE.match(name):
        return
    if any(p.get("anchor") == name for p in list_profiles(settings).values()):
        return
    with contextlib.suppress(OSError):
        (anchor_dir(config) / name).unlink()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_profiles_anchor.py -v`
Expected: PASS (17 tests).

- [ ] **Step 5: Run the whole suite and the linter**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS. `delete_profile`'s existing two-argument call sites in `profiles_panel.py` still type-check because `config` is optional; Task 3 updates them.

- [ ] **Step 6: Commit**

```bash
git add src/warlock/studio/profiles.py tests/test_profiles_anchor.py
git commit -m "Warlock v0.0.7

Store a style anchor image per user profile: a file under profiles/, named by
the profile, reference-counted so a rename cannot delete it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The anchor conditions every generation under its profile

Storage is inert until something reads it. The 2D pane's submit consults the active profile's anchor when the user has attached no reference of their own, and the profile manager gains the controls to set and clear one.

**Files:**
- Modify: `src/warlock/studio/panes/settings_2d.py` (`_reference`, `generate`, new `anchor_kwargs`)
- Modify: `src/warlock/studio/panes/profiles_panel.py` (anchor controls, `delete_profile` call sites)
- Test: create `tests/test_anchor_submit.py`

**Interfaces:**
- Consumes: `profiles.active_anchor`, `profiles.ANCHOR_ADAPTER`, `profiles.set_anchor`, `profiles.clear_anchor` (Task 2); `settings_2d.submit_kwargs`.
- Produces: `settings_2d.anchor_kwargs(ctx, form, kwargs) -> str` — mutates `kwargs` in place to carry the anchor's conditioning and returns the reference *path* to read, or `""`. The path is read on a task thread, never on the frame thread.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_anchor_submit.py`:

```python
"""A profile's anchor becomes the job's IP-Adapter reference.

The pane's rule everywhere else is that a manual attachment wins: the anchor
is what the *set* has in common, and the reference the user just dropped is
what this one asset needs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import profiles
from warlock.studio.panes import settings_2d
from warlock.studio.state import default_form_2d

from .test_profiles_anchor import PNG, FakeSettings


@pytest.fixture
def ctx(tmp_path):
    return SimpleNamespace(
        settings=FakeSettings(),
        svc=SimpleNamespace(config=SimpleNamespace(data_dir=tmp_path)),
    )


def _with_anchor(ctx, scale=0.7):
    profiles.save_profile(ctx.settings, "house", {})
    profiles.set_anchor(ctx.settings, ctx.svc.config, "house", PNG, scale=scale)
    profiles.set_active(ctx.settings, "house")


def test_no_profile_means_the_kwargs_are_untouched(ctx):
    form = default_form_2d()
    kwargs = settings_2d.submit_kwargs(form)
    before = dict(kwargs)

    assert settings_2d.anchor_kwargs(ctx, form, kwargs) == ""
    assert kwargs == before


def test_an_anchor_supplies_the_reference_path_and_the_adapter(ctx):
    _with_anchor(ctx)
    form = default_form_2d()
    kwargs = settings_2d.submit_kwargs(form)

    path = settings_2d.anchor_kwargs(ctx, form, kwargs)

    assert path and open(path, "rb").read() == PNG
    assert kwargs["guidance_fields"]["ip_adapter"] == profiles.ANCHOR_ADAPTER
    assert kwargs["ip_scale"] == 0.7


def test_a_manual_reference_wins_over_the_anchor(ctx):
    _with_anchor(ctx)
    form = default_form_2d()
    form["ref_path"] = "C:/somewhere/else.png"
    kwargs = settings_2d.submit_kwargs(form)

    assert settings_2d.anchor_kwargs(ctx, form, kwargs) == ""
    assert "ip_adapter" not in kwargs["guidance_fields"]


def test_a_profile_without_an_anchor_changes_nothing(ctx):
    profiles.save_profile(ctx.settings, "house", {})
    profiles.set_active(ctx.settings, "house")
    form = default_form_2d()
    kwargs = settings_2d.submit_kwargs(form)

    assert settings_2d.anchor_kwargs(ctx, form, kwargs) == ""


def test_generate_reads_the_anchor_on_a_task_thread(ctx, monkeypatch):
    # The same contract the manual picker follows: the form is read on the
    # frame thread because it is UI state, the file inside the task because a
    # large one would freeze the window.
    _with_anchor(ctx)
    seen = {}
    monkeypatch.setattr(
        settings_2d.svc_jobs, "create_job", lambda svc, **kw: seen.update(kw) or "id"
    )
    submitted = []
    ctx.state = SimpleNamespace(
        form_2d=default_form_2d(), preview_dirty_at=0.0, remember_prompt=lambda _p: None
    )
    ctx.submit = lambda key, fn, *a, **k: (submitted.append((key, fn)), True)[1]
    ctx.state.form_2d["prompt"] = "a barrel"

    settings_2d.generate(ctx, ctx.state.form_2d)

    assert seen == {}  # nothing read yet
    _key, run = submitted[0]
    run()
    assert seen["reference"] == PNG
    assert seen["ip_scale"] == 0.7
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_anchor_submit.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'anchor_kwargs'`.

- [ ] **Step 3: Implement the submit half**

In `src/warlock/studio/panes/settings_2d.py`, add after `submit_kwargs` (line 425):

```python
def anchor_kwargs(ctx: Any, form: dict[str, Any], kwargs: dict[str, Any]) -> str:
    """Fold the active profile's style anchor into a submit, in place.

    -> the path to read as the conditioning reference, or "" for none.

    A manual attachment wins and this returns "" for it: the anchor is what a
    whole set has in common, and the image the user just dropped is what this
    one asset needs. The path is *returned* rather than read here because this
    runs on the frame thread and generate() reads files in its task.
    """
    if form.get("ref_path"):
        return ""
    found = profiles.active_anchor(ctx.settings, ctx.svc.config)
    if found is None:
        return ""
    path, scale = found
    # setdefault, not assignment: a form that already names an adapter chose
    # it, and the anchor only supplies one where there was none.
    kwargs.setdefault("guidance_fields", {}).setdefault(
        "ip_adapter", profiles.ANCHOR_ADAPTER
    )
    kwargs["ip_scale"] = float(scale)
    return str(path)
```

and change `generate` (line 428) so `ref_path` comes from either source:

```python
    kwargs = submit_kwargs(form)
    ref_path = form.get("ref_path") or anchor_kwargs(ctx, form, kwargs)
```

(the rest of `generate` is unchanged — it already reads `ref_path` inside `run()`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_anchor_submit.py -v`
Expected: PASS.

- [ ] **Step 5: Show the anchor in the Reference section**

In `_reference` (line 240), after `path = form["ref_path"]` and before the `if path:` branch:

```python
    if not path:
        found = profiles.active_anchor(ctx.settings, ctx.svc.config)
        if found is not None:
            active = profiles.get_active(ctx.settings)
            widgets.muted(
                f"The profile {active} has a style anchor; every generation "
                "under it is conditioned on that image. Attaching one here "
                "replaces it for this asset."
            )
```

- [ ] **Step 6: Add the anchor controls to the profile manager**

In `src/warlock/studio/panes/profiles_panel.py`:

Add `from ...service.validation import MAX_UPLOAD_BYTES` and `from .. import dialogs` (already imported) to the header, then add a section drawn from `_editor`, just before the Save/Cancel row (line 146):

```python
    _anchor(ctx, name)
```

and the function itself, after `_editor`:

```python
def _anchor(ctx: Any, name: str) -> None:
    """The style anchor: one image every asset in this set is conditioned on.

    Only offered on a profile that has been saved once, because the anchor is
    stored against the name -- there is nowhere to put it while the editor is
    still holding an unnamed draft.
    """
    widgets.section("Style anchor")
    saved = profiles.list_profiles(ctx.settings)
    if name not in saved:
        widgets.muted("Save the profile once, then attach an anchor image to it.")
        return
    fields = saved[name]
    path = profiles.anchor_path(ctx.svc.config, fields)
    if path is not None:
        if ctx.textures is not None:
            texture = ctx.textures.get(f"anchor:{name}", path)
            if texture is not None:
                imgui.image(widgets.texture_ref(texture), (96, 96))
        changed, value = imgui.slider_float(
            "Strength##anchor", float(fields.get("anchor_scale") or 0.6), 0.0, 1.5
        )
        if changed:
            profiles.save_profile(
                ctx.settings, name, {**fields, "anchor_scale": float(value)}
            )
        if imgui.small_button("Remove anchor"):
            profiles.clear_anchor(ctx.settings, ctx.svc.config, name)
        imgui.same_line()
    busy = ctx.busy("anchor-pick")
    if widgets.disabled_button(
        "Choose an image..." if path is None else "Replace...", not busy
    ):
        ctx.submit("anchor-pick", _pick_anchor, ctx, name)
    if path is None:
        widgets.muted(
            "Every generation under this profile is conditioned on the anchor, "
            "which is what keeps a set of assets looking like one set."
        )


def _pick_anchor(ctx: Any, name: str) -> None:
    """Runs on a task thread: both the dialog and the read block."""
    chosen = dialogs.open_file("Choose a style anchor", dialogs.IMAGE_FILTER)
    if chosen is None:
        return
    data = Path(chosen).open("rb").read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        ctx.toast("That image is over 20 MB.", "error")
        return
    profiles.set_anchor(ctx.settings, ctx.svc.config, name, data)
    ctx.toast(f"Anchor set for {name}.")
```

with `from pathlib import Path` added to the imports.

- [ ] **Step 7: Pass the config to both delete call sites**

In `profiles_panel.py`, the confirm at line 70 becomes:

```python
                    on_confirm=lambda n=name: profiles.delete_profile(
                        ctx.settings, n, ctx.svc.config
                    ),
```

and in `_save` (line 167):

```python
        profiles.delete_profile(ctx.settings, origin, ctx.svc.config)
```

Confirm the rename path keeps the anchor by hand: `_save` calls `profiles.save_profile(ctx.settings, name, profiles.capture(draft))` first, which has no anchor to preserve under the *new* name. Fix it in the same edit — read the origin's fields and carry them:

```python
def _save(ctx: Any) -> None:
    name = ctx.state.profile_draft_name.strip()
    origin = ctx.state.profile_draft_origin
    fields = profiles.capture(ctx.state.profile_draft)
    if origin and origin != name:
        # A rename moves the anchor with the profile: save_profile preserves
        # anchor fields under the *same* name, and this is the one path where
        # the name changes underneath them.
        carried = profiles.list_profiles(ctx.settings).get(origin) or {}
        fields.update({k: carried[k] for k in profiles.ANCHOR_FIELDS if k in carried})
    profiles.save_profile(ctx.settings, name, fields)
    if origin and origin != name:
        profiles.delete_profile(ctx.settings, origin, ctx.svc.config)
    profiles.set_active(ctx.settings, name)
    ctx.toast(f"Saved the profile {name}.")
    _close(ctx)
```

- [ ] **Step 8: Add the rename test**

Append to `tests/test_profiles_anchor.py`:

```python
def test_a_rename_keeps_the_anchor_and_its_file(env):
    # The editor's own sequence: save under the new name, delete the old.
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    carried = profiles.list_profiles(env.settings)["house"]

    profiles.save_profile(
        env.settings, "manor", {k: carried[k] for k in profiles.ANCHOR_FIELDS}
    )
    profiles.delete_profile(env.settings, "house", env.config)

    stored = profiles.list_profiles(env.settings)["manor"]
    assert profiles.anchor_path(env.config, stored) is not None
```

- [ ] **Step 9: Run everything**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/warlock/studio/panes/settings_2d.py src/warlock/studio/panes/profiles_panel.py tests/test_anchor_submit.py tests/test_profiles_anchor.py
git commit -m "Warlock v0.0.7

Condition every generation under a profile on its style anchor, with the
manual reference still winning, and give the profile manager the controls.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Rank reference candidates against what was already measured

`count > 1` fans out N candidates and presents them in submission order. Every one already carries a `reference_report`; when a style anchor was used, `ref.png` sits next to `input.png` and DINOv2 can say which candidate looks most like the set. The score is advisory — the user still picks.

The scoring is two pure pieces: `pipelines/rank.py` (no torch, no I/O) and a new reference-against-reference pairing in `bench/imageprep.py`. `bench/metrics.py:dino_cosine` cannot be reused as-is: `prepare_pair` takes the *render* side's subject from its alpha channel, and two opaque SDXL images have none — it would return `None` for every pair.

**Files:**
- Create: `src/warlock/pipelines/rank.py`
- Modify: `src/warlock/bench/imageprep.py` (add `prepare_references`)
- Modify: `src/warlock/bench/metrics.py` (add `reference_cosine`, thread a `device` through `_dino_model` / `_embed`)
- Modify: `src/warlock/queue.py` (record `rank` on a finished reference)
- Modify: `src/warlock/service/validation.py` (`DERIVED_PARAMS`)
- Modify: `src/warlock/config.py` (`rank_candidates`)
- Modify: `src/warlock/studio/state.py` (`Filters.sort`, `Filters.order`), `src/warlock/studio/jobs_cache.py`, `src/warlock/studio/panes/library.py`
- Test: create `tests/test_rank.py`; extend `tests/test_queue.py`, `tests/test_studio_state.py`, `tests/test_bench_metrics.py`

**Interfaces:**
- Consumes: `reference.Report.as_dict()`'s shape — `ok`, `reasons`, `warnings`, `occupancy`, `components`, `touches` (`src/warlock/pipelines/reference.py:113`).
- Produces:
  - `rank.composition_score(report: dict | None) -> float` in 0..1
  - `rank.score(report: dict | None, anchor_cosine: float | None = None) -> dict` with keys `score`, `composition`, `anchor`
  - `imageprep.prepare_references(a: Path, b: Path, *, size=PAIR_SIZE) -> tuple[Any, Any]`
  - `metrics.reference_cosine(a: Path, b: Path, config=None, device: str | None = None) -> float | None`
  - `params["rank"]` on a finished reference job: `{"score": float, "composition": float, "anchor": float | None}`
  - `state.Filters.sort: str` (`"newest"` | `"best"`) and `Filters.order(jobs) -> list`

- [ ] **Step 1: Write the failing tests for the pure scorer**

Create `tests/test_rank.py`:

```python
"""The candidate score: what the reference report already knew, as a number.

Advisory by construction -- nothing rejects a candidate. Its whole job is to
put the most likely one first in a strip of eight.
"""

from __future__ import annotations

from warlock.pipelines import rank, reference


def _report(**kwargs):
    return reference.Report(**kwargs).as_dict()


def test_a_refused_reference_scores_zero():
    assert rank.composition_score(_report(ok=False, reasons=("too small",))) == 0.0


def test_a_clean_reference_at_the_target_occupancy_scores_one():
    assert rank.composition_score(
        _report(occupancy=reference.DEFAULT_OCCUPANCY, components=1)
    ) == 1.0


def test_missing_the_target_occupancy_costs_something_but_not_everything():
    near = rank.composition_score(_report(occupancy=0.70, components=1))
    far = rank.composition_score(_report(occupancy=0.20, components=1))
    assert 0.0 < far < near < 1.0


def test_a_second_object_is_penalised():
    one = rank.composition_score(_report(occupancy=0.78, components=1))
    two = rank.composition_score(_report(occupancy=0.78, components=2))
    assert two < one


def test_running_off_the_edge_is_penalised():
    clean = rank.composition_score(_report(occupancy=0.78, components=1))
    cropped = rank.composition_score(
        _report(occupancy=0.78, components=1, touches=("left",))
    )
    assert cropped < clean


def test_warnings_cost_less_than_reasons():
    warned = rank.composition_score(
        _report(occupancy=0.78, components=1, warnings=("close to the edge",))
    )
    assert 0.0 < warned < 1.0


def test_no_report_at_all_is_a_middling_score_not_a_zero():
    # A job whose measurement failed is unknown, not bad -- scoring it zero
    # would sort it below a candidate that was actually measured and refused.
    assert 0.0 < rank.composition_score(None) < 1.0


def test_the_score_is_the_composition_when_there_is_no_anchor():
    out = rank.score(_report(occupancy=0.78, components=1))
    assert out["score"] == out["composition"] == 1.0
    assert out["anchor"] is None


def test_an_anchor_cosine_moves_the_score_and_is_recorded():
    report = _report(occupancy=0.78, components=1)
    close = rank.score(report, anchor_cosine=0.9)
    far = rank.score(report, anchor_cosine=0.1)
    assert close["score"] > far["score"]
    assert close["anchor"] == 0.9


def test_every_score_stays_inside_zero_and_one():
    for cosine in (-1.0, 0.0, 1.0):
        for report in (None, _report(ok=False), _report(occupancy=0.01, components=5)):
            out = rank.score(report, anchor_cosine=cosine)
            assert 0.0 <= out["score"] <= 1.0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_rank.py -v`
Expected: FAIL with `ImportError: cannot import name 'rank' from 'warlock.pipelines'`.

- [ ] **Step 3: Implement `pipelines/rank.py`**

Create `src/warlock/pipelines/rank.py`:

```python
"""Which of N reference candidates to look at first.

Nothing here rejects anything. A fan-out of eight candidates arrives in
submission order, which is the order the seeds happened to be drawn in, and
every one of them already carries a measurement nobody reads -- the
composition report the reference stage takes anyway. This turns that report,
plus an optional DINOv2 similarity against the profile's style anchor, into
one number the gallery can sort by.

Pure: no torch, no I/O, no imports from service/queue/studio. The cosine is
computed by the caller (bench.metrics) and handed in, so this module stays
testable without weights and the weighting stays a single readable formula
rather than something buried in the worker.
"""

from __future__ import annotations

from typing import Any

from .reference import DEFAULT_OCCUPANCY

# How the two halves trade off. Composition dominates because it is the one
# that predicts whether the mesh stage can succeed at all; the anchor is about
# whether this candidate belongs with the others, which only matters among
# candidates that can all reconstruct.
COMPOSITION_WEIGHT = 0.6
ANCHOR_WEIGHT = 0.4

# What an unmeasured reference scores. Deliberately mid-range: a job whose
# report is missing is unknown, not bad, and scoring it zero would sort it
# below a candidate that was measured and refused.
UNMEASURED = 0.5

# Per-defect costs, all applied to a base of 1.0.
WARNING_COST = 0.10
COMPONENT_COST = 0.15
TOUCH_COST = 0.10
# How much of the score the occupancy distance can take. Scaled by how far the
# subject is from DEFAULT_OCCUPANCY as a fraction of the whole range, so a
# candidate at 0.70 against a target of 0.78 loses very little.
OCCUPANCY_COST = 0.40


def composition_score(report: dict[str, Any] | None) -> float:
    """0..1 for how well framed one reference is, from its own report."""
    if not isinstance(report, dict):
        return UNMEASURED
    if report.get("ok") is False:
        # The report already said this cannot reconstruct. Nothing below can
        # rescue it, and a candidate that is going to be refused at promotion
        # belongs last.
        return 0.0
    score = 1.0
    try:
        occupancy = float(report.get("occupancy") or 0.0)
    except (TypeError, ValueError):
        occupancy = 0.0
    score -= OCCUPANCY_COST * min(1.0, abs(occupancy - DEFAULT_OCCUPANCY))
    score -= WARNING_COST * len(report.get("warnings") or ())
    score -= COMPONENT_COST * max(0, int(report.get("components") or 1) - 1)
    score -= TOUCH_COST * min(1, len(report.get("touches") or ()))
    return max(0.0, min(1.0, score))


def score(
    report: dict[str, Any] | None, anchor_cosine: float | None = None
) -> dict[str, Any]:
    """The whole verdict for one candidate, as it is stored in params.

    ``anchor_cosine`` is a DINOv2 cosine in -1..1 and is rescaled to 0..1 here
    rather than by its producer, so the raw number stays comparable with every
    other cosine in the codebase.
    """
    composition = composition_score(report)
    if anchor_cosine is None:
        return {"score": composition, "composition": composition, "anchor": None}
    anchor = max(0.0, min(1.0, (float(anchor_cosine) + 1.0) / 2.0))
    combined = COMPOSITION_WEIGHT * composition + ANCHOR_WEIGHT * anchor
    return {
        "score": max(0.0, min(1.0, combined)),
        "composition": composition,
        "anchor": float(anchor_cosine),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rank.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing tests for the reference-pair metric**

Append to `tests/test_bench_metrics.py` (read its existing imports and helpers first — it already builds synthetic PNGs; reuse whatever it has rather than adding a second helper):

```python
def test_two_opaque_references_pair_up(tmp_path):
    # prepare_pair takes the render side's subject from its alpha, which two
    # SDXL images do not have -- so the reference pipeline needs its own.
    from warlock.bench import imageprep

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for path, box in ((a, (20, 20, 60, 90)), (b, (40, 30, 70, 100))):
        im = Image.new("RGB", (128, 128), (200, 200, 200))
        ImageDraw.Draw(im).rectangle(box, fill=(20, 20, 20))
        im.save(path)

    left, right = imageprep.prepare_references(a, b)

    assert left is not None and right is not None
    assert left.size == right.size == (imageprep.PAIR_SIZE, imageprep.PAIR_SIZE)
    assert left.mode == right.mode == "RGB"


def test_a_reference_with_no_subject_pairs_to_nothing(tmp_path):
    from warlock.bench import imageprep

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(a)
    im = Image.new("RGB", (64, 64), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((10, 10, 40, 50), fill=(0, 0, 0))
    im.save(b)

    assert imageprep.prepare_references(a, b) == (None, None)
```

(Add `from PIL import Image, ImageDraw` to that file's imports if it is not already there.)

- [ ] **Step 6: Run them and watch them fail**

Run: `uv run pytest tests/test_bench_metrics.py -v`
Expected: FAIL with `AttributeError: module 'warlock.bench.imageprep' has no attribute 'prepare_references'`.

- [ ] **Step 7: Implement the pairing and the metric**

In `src/warlock/bench/imageprep.py`, after `prepare_pair`:

```python
def prepare_references(
    a_path: Path, b_path: Path, *, size: int = PAIR_SIZE
) -> tuple[Any, Any]:
    """-> two references, cropped to their subjects and made comparable.

    The same crop-and-square treatment ``prepare_pair`` gives a render, but
    both sides measured with ``reference_mask``: two SDXL images are opaque,
    so the alpha route that identifies a rendered subject finds nothing at all
    and would score every pair as None.

    The background is taken from the *first* image, so a candidate is judged
    against the anchor's background rather than the two differing in a way the
    metric can see.
    """
    from PIL import Image

    prepared = []
    background = None
    for path in (a_path, b_path):
        with Image.open(path) as im:
            im.load()
            if background is None:
                background = border_colour(im)
            mask = reference_mask(im)
            box = _bbox(mask) if mask is not None else None
            if box is None:
                return (None, None)
            prepared.append(
                _square_crop(im.convert("RGB"), box, background).resize(
                    (size, size), Image.LANCZOS
                )
            )
    return (prepared[0], prepared[1])
```

In `src/warlock/bench/metrics.py`, thread an explicit device through and add the reference-pair entry point. Replace `_dino_model` and `_embed`, and add `reference_cosine`:

```python
def _dino_model(config: Any = None, device: str | None = None):
    """Load DINOv2 once per (path, device) -- a 160-item run would otherwise
    pay for it 1280 times.

    ``device`` is explicit rather than "cuda if available" for one caller: the
    app scores candidates on the job queue, beside a resident trellis and a
    resident SDXL pipe, and a metric has no business taking VRAM away from
    the models that are producing the asset. The benchmark passes None and
    keeps the old behaviour.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"{_dino_dir(config)}|{device}"
    hit = _model_cache.get(key)
    if hit is not None:
        return hit

    path = _dino_dir(config)
    if not path.exists():
        from .. import models

        raise RuntimeError(
            f"DINOv2 not found at {path}. Download once with:\n"
            f"  {models.METRIC_MODELS['dinov2'].download}"
        )
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(str(path), local_files_only=True)
    model.eval()
    model = model.to(device)
    _model_cache[key] = (processor, model, device)
    return _model_cache[key]


def _embed(image: Any, config: Any = None, device: str | None = None):
    import torch

    processor, model, resolved = _dino_model(config, device)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(resolved) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # The CLS token, which is DINOv2's whole-image descriptor.
    cls = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(cls, dim=-1)
```

`dino_cosine` keeps its body but passes `device=None` through to `_embed`. Then, after it:

```python
def reference_cosine(
    a_path: Path, b_path: Path, config: Any = None, device: str | None = None
) -> float | None:
    """Cosine similarity between two *references*, higher is more alike.

    The A-against-A caveat in this module's docstring is not merely satisfied
    here, it is the whole point: both sides are SDXL output on a plain
    background, so the style gap that makes ``dino_cosine`` unreadable in
    absolute terms is absent, and comparing candidates of one submit against
    one anchor is exactly the comparison the number supports.
    """
    import torch

    a, b = imageprep.prepare_references(a_path, b_path)
    if a is None or b is None:
        return None
    return float(torch.sum(_embed(a, config, device) * _embed(b, config, device)).item())
```

- [ ] **Step 8: Run the metric tests**

Run: `uv run pytest tests/test_bench_metrics.py -v`
Expected: PASS. (The cosine tests themselves are skipped without weights, exactly as the existing DINO tests are — check how `tests/test_bench_metrics.py` guards them and follow the same guard.)

- [ ] **Step 9: Write the failing worker test**

Append to `tests/test_queue.py`:

```python
async def test_a_finished_reference_carries_a_rank(worker):
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    rank = worker.store.get(job_id)["params"]["rank"]
    assert 0.0 <= rank["score"] <= 1.0
    # No ref.png, so nothing to compare against -- the composition half stands
    # on its own rather than the whole thing being absent.
    assert rank["anchor"] is None


async def test_a_failing_rank_does_not_fail_the_job(worker, monkeypatch):
    # Same rule as the mesh audit: a diagnostic must never be able to fail a
    # job whose artifact is already on disk.
    import warlock.pipelines.rank as rank_mod

    monkeypatch.setattr(
        rank_mod, "score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "done"
    assert "rank" not in worker.store.get(job_id)["params"]


async def test_a_mesh_job_is_not_ranked(worker):
    # The score is about choosing between reference candidates; a mesh job has
    # nothing to choose between and the measurement would be noise in params.
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert "rank" not in worker.store.get(job_id)["params"]
```

Add to `tests/test_service.py` (find the existing DERIVED_PARAMS test and extend it, or add):

```python
def test_a_rank_never_survives_into_a_new_job():
    from warlock.service.validation import DERIVED_PARAMS

    assert "rank" in DERIVED_PARAMS
```

- [ ] **Step 10: Run them and watch them fail**

Run: `uv run pytest tests/test_queue.py -k rank tests/test_service.py -k rank -v`
Expected: FAIL — `KeyError: 'rank'` and the DERIVED_PARAMS assertion.

- [ ] **Step 11: Implement the worker hook**

In `src/warlock/config.py`, after `mesh_profile` (line 101):

```python
    # Whether a finished reference is scored against the active profile's
    # style anchor as well as its own composition report. On by default: the
    # composition half costs nothing (the report is already measured) and the
    # anchor half only runs when there is a ref.png and DINOv2 is on disk.
    rank_candidates: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_RANK", "on").lower()
        not in ("0", "false", "off", "no")
    )
```

In `src/warlock/service/validation.py`, add `"rank"` to `DERIVED_PARAMS` (after `"reference_report"`), with a comment:

```python
    "reference_report",
    # Advisory, and about *this* run's image: a reroll that inherited it would
    # wear a verdict about pixels it is about to replace.
    "rank",
```

In `src/warlock/queue.py`, add the ranking method next to `_audit_mesh`:

```python
    def _rank_reference(self, job_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Score a finished reference. Blocking -- called through to_thread.

        The anchor half is opportunistic in three separate ways, and every one
        of them is a "leave the number out", never a failure: ranking can be
        switched off, ref.png only exists when something conditioned the run,
        and DINOv2 is an optional download. What is left is the composition
        score, which is free -- the report was measured either way.
        """
        from .bench import metrics
        from .pipelines import rank

        report = params.get("reference_report")
        cosine = None
        anchor = job_dir / "ref.png"
        if self.config.rank_candidates and anchor.exists():
            try:
                if metrics.dino_available(self.config):
                    # CPU deliberately: this runs on the job queue beside a
                    # resident trellis and a resident SDXL pipe, and a metric
                    # must not take VRAM from the models making the asset.
                    cosine = metrics.reference_cosine(
                        anchor, job_dir / "input.png", self.config, device="cpu"
                    )
            except Exception:
                log.exception("anchor similarity failed; ranking on composition alone")
        return rank.score(report, cosine)
```

and call it in `_generate`'s text branch, inside the existing `if job.get("stage") == "reference":` block at line 711, right after `params["reference_report"] = ...`:

```python
                    try:
                        params["rank"] = await asyncio.to_thread(
                            self._rank_reference, job_dir, params
                        )
                    except Exception:
                        # Advisory: the image is on disk and fine. The UI shows
                        # no score rather than a wrong one.
                        log.exception("ranking failed for job %s", job_id)
```

- [ ] **Step 12: Run the worker tests**

Run: `uv run pytest tests/test_queue.py -k rank tests/test_service.py -v`
Expected: PASS.

- [ ] **Step 13: Write the failing sort tests**

Append to `tests/test_studio_state.py`:

```python
def _ranked(job_id, score):
    return {"id": job_id, "created_at": 0.0, "params": {"rank": {"score": score}}}


def test_the_default_order_is_the_order_it_was_given():
    filters = statelib.Filters()
    jobs = [_ranked("a", 0.1), _ranked("b", 0.9)]
    assert [j["id"] for j in filters.order(jobs)] == ["a", "b"]


def test_sorting_by_best_puts_the_highest_score_first():
    filters = statelib.Filters(sort="best")
    jobs = [_ranked("a", 0.1), _ranked("b", 0.9), _ranked("c", 0.5)]
    assert [j["id"] for j in filters.order(jobs)] == ["b", "c", "a"]


def test_unranked_jobs_sort_last_and_keep_their_own_order():
    # Most of a workshop predates the score. Sorting them to the top on a
    # missing value would make "best first" show the oldest assets.
    filters = statelib.Filters(sort="best")
    jobs = [{"id": "a", "params": {}}, _ranked("b", 0.4), {"id": "c", "params": {}}]
    assert [j["id"] for j in filters.order(jobs)] == ["b", "a", "c"]


def test_a_malformed_rank_is_treated_as_unranked():
    filters = statelib.Filters(sort="best")
    jobs = [{"id": "a", "params": {"rank": "nonsense"}}, _ranked("b", 0.2)]
    assert [j["id"] for j in filters.order(jobs)] == ["b", "a"]
```

- [ ] **Step 14: Run them and watch them fail**

Run: `uv run pytest tests/test_studio_state.py -k order -v`
Expected: FAIL with `TypeError: Filters.__init__() got an unexpected keyword argument 'sort'`.

- [ ] **Step 15: Implement the sort**

In `src/warlock/studio/state.py`, add the field to `Filters` (after `favorites_only`) and the method after `matches`:

```python
    # newest | best. Persisted with the rest of the filter bar, because a
    # workshop is browsed the same way every session.
    sort: str = "newest"
```

```python
    def order(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The list in the order the bar asks for.

        Stable and non-destructive: "newest" returns the query's own order
        untouched, and "best" is a stable sort, so candidates that share a
        score stay in submission order rather than shuffling every refresh.
        """
        if self.sort != "best":
            return list(jobs)

        def key(job: dict[str, Any]) -> tuple[int, float]:
            rank = (job.get("params") or {}).get("rank")
            if not isinstance(rank, dict):
                # Unranked sorts last rather than at zero: most of a workshop
                # predates the score, and burying it under refused candidates
                # would make "best first" mean "oldest first".
                return (1, 0.0)
            try:
                return (0, -float(rank.get("score") or 0.0))
            except (TypeError, ValueError):
                return (1, 0.0)

        return sorted(jobs, key=key)
```

In `src/warlock/studio/jobs_cache.py`, `visible` becomes:

```python
    def visible(self, filters: Any) -> list[dict[str, Any]]:
        return filters.order([j for j in self.jobs if filters.matches(j)])
```

In `src/warlock/studio/panes/library.py`, add the picker to `_filters` after the kind combo:

```python
    imgui.same_line()
    filters.sort = widgets.combo(
        "##sort",
        filters.sort,
        [("newest", "newest first"), ("best", "best first")],
        width=110,
    )
```

and show the number on the card, in `_card_body` right after `widgets.quality_badge(job)`:

```python
    rank = (job.get("params") or {}).get("rank")
    if isinstance(rank, dict) and rank.get("score") is not None:
        imgui.same_line()
        widgets.muted(f"{float(rank['score']) * 100:.0f}%")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "How well framed this reference is, and -- when the profile "
                "has a style anchor -- how close it looks to it. Advisory."
            )
```

Check how `Filters` is persisted and restored in `src/warlock/studio/main.py` (grep for `filters`): if the restore is field-by-field, add `sort` alongside `status` and `kind`; if it round-trips the dataclass through `asdict`, nothing more is needed.

- [ ] **Step 16: Run everything**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS.

- [ ] **Step 17: Commit**

```bash
git add src/warlock/pipelines/rank.py src/warlock/bench/imageprep.py src/warlock/bench/metrics.py src/warlock/queue.py src/warlock/config.py src/warlock/service/validation.py src/warlock/studio/state.py src/warlock/studio/jobs_cache.py src/warlock/studio/panes/library.py tests/test_rank.py tests/test_queue.py tests/test_service.py tests/test_studio_state.py tests/test_bench_metrics.py
git commit -m "Warlock v0.0.7

Score a finished reference from its own composition report, plus a CPU DINOv2
similarity against the profile style anchor when there is one, and let the
library sort by it. Advisory: nothing is rejected.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Opt-in auto-reroll when the reference fails its own report

Today a reference that cannot reconstruct is either shown to the user as a finished asset (reference stage) or fails the job after the image was drawn (model stage, the gate at `queue.py:789`). Either way the four seconds of SDXL are already spent; drawing it again with a fresh seed costs the same four seconds and often lands.

The retry loop sits **inside** the existing `try` that wraps `t2i.generate`, so the VRAM ordering is untouched: one load, N samples, one unload in the `finally`.

**Files:**
- Modify: `src/warlock/config.py` (`reference_retries`)
- Modify: `src/warlock/queue.py` (`_generate`, text branch)
- Modify: `src/warlock/service/validation.py` (`DERIVED_PARAMS`)
- Modify: `src/warlock/studio/panes/inspector.py` (show the attempts)
- Test: `tests/test_queue.py`

**Interfaces:**
- Consumes: `reference.measure_file(path) -> Report` (`src/warlock/pipelines/reference.py`), `Config.reference_retries`.
- Produces: `params["reference_attempts"]` — a list of `{"seed": int, "ok": bool, "reasons": list[str]}`, oldest first, present only when more than one attempt ran.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_queue.py`:

```python
def _bad_then_good(monkeypatch, failures: int):
    """reference.measure_file that refuses the first `failures` calls."""
    import warlock.pipelines.reference as reference_mod

    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        ok = calls["n"] > failures
        return reference_mod.Report(
            ok=ok, reasons=() if ok else ("the subject runs off the frame",)
        )

    monkeypatch.setattr(reference_mod, "measure_file", fake)
    return calls


async def test_without_the_setting_a_bad_reference_is_not_rerolled(worker, monkeypatch):
    _bad_then_good(monkeypatch, failures=99)
    job_id = worker.store.create("text", "a barrel", {"seed": 5}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.seeds == [5]
    assert "reference_attempts" not in worker.store.get(job_id)["params"]


async def test_a_bad_reference_is_rerolled_once_with_a_fresh_seed(
    tmp_path, fake_pipelines, monkeypatch
):
    from warlock.config import Config
    from warlock.db import JobStore

    _bad_then_good(monkeypatch, failures=1)
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        reference_retries=1,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("text", "a barrel", {"seed": 5}, stage="reference")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    seeds = w._text2image.seeds
    assert len(seeds) == 2
    assert seeds[0] == 5 and seeds[1] != 5
    attempts = store.get(job_id)["params"]["reference_attempts"]
    assert [a["ok"] for a in attempts] == [False, True]
    assert [a["seed"] for a in attempts] == seeds
    # One load and one unload around both samples: the retry must not repeat
    # the VRAM handoff, which is the whole reason it lives inside the try.
    assert w._text2image.unload_calls == 0
    store.close()


async def test_the_retry_budget_is_a_ceiling_not_a_loop(
    tmp_path, fake_pipelines, monkeypatch
):
    from warlock.config import Config
    from warlock.db import JobStore

    _bad_then_good(monkeypatch, failures=99)
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        reference_retries=2,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("text", "a barrel", {"seed": 5}, stage="reference")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    # Three samples, then it stops and hands the user the last one -- the
    # report is a heuristic, and refusing to finish would be worse.
    assert len(w._text2image.seeds) == 3
    assert store.get(job_id)["status"] == "done"
    store.close()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_queue.py -k reroll -v` and `uv run pytest tests/test_queue.py -k retry_budget -v`
Expected: FAIL — `TypeError: Config.__init__() got an unexpected keyword argument 'reference_retries'`.

- [ ] **Step 3: Add the config field**

In `src/warlock/config.py`, after `rank_candidates`:

```python
    # How many extra times a text job may redraw its reference when the
    # composition report refuses the one it just drew. 0 -- off -- because a
    # retry is another four seconds of GPU the user did not ask for, and the
    # report's rules are heuristics: a refusal is a strong hint, not a fact.
    # 1 is the setting that pays for itself.
    reference_retries: int = field(
        default_factory=lambda: max(0, int(os.environ.get("WARLOCK_REFERENCE_RETRIES", "0")))
    )
```

- [ ] **Step 4: Implement the retry loop**

In `src/warlock/queue.py`, add `import secrets` to the stdlib imports and, near `DEFAULT_REFERENCE_PREP`:

```python
def _fresh_seed() -> int:
    """A new 31-bit seed for a retry.

    Local rather than imported from service.validation: queue.py imports only
    top-level modules by design, and this is the same three-line contract --
    31-bit so it round-trips through an sqlite INTEGER unchanged.
    """
    return secrets.randbelow(2**31)
```

Then replace the body of the `try:` at line 694 (the `t2i.generate` call and what follows it) with the loop:

```python
            attempts: list[dict[str, Any]] = []
            seed = reference_seed
            retries = max(0, int(self.config.reference_retries))
            is_reference = job.get("stage") == "reference"
            try:
                while True:
                    await asyncio.to_thread(
                        functools.partial(
                            t2i.generate,
                            composed,
                            image_path,
                            seed=seed,
                            lora=style_lora,
                            lora_weight=lora_weight,
                            negative_prompt=str(params.get("negative_prompt") or ""),
                            conditioning=cond,
                            on_state=lambda s: self._t2i_state(job_id, s),
                            on_step=lambda i, n: self._t2i_step(job_id, i, n),
                            cancel_event=self._cancel.event,
                        )
                    )
                    params["composed_prompt"] = t2i.last_prompt or composed
                    if not (is_reference or retries):
                        # Nothing to measure it for: a model-stage job with the
                        # retry off is measured a few lines further down by
                        # reference.prepare anyway, and paying for a second
                        # flood fill here would be pure cost.
                        break
                    # Measure only, and never a rejection: the user is judging
                    # the image, and the mesh stage is where the cost is. This
                    # is what promote_to_model's soft check reads.
                    report = await asyncio.to_thread(reference.measure_file, image_path)
                    if is_reference:
                        params["reference_report"] = report.as_dict()
                    attempts.append(
                        {"seed": seed, "ok": report.ok, "reasons": list(report.reasons)}
                    )
                    if (
                        report.ok
                        or len(attempts) > retries
                        or self._cancel.event.is_set()
                    ):
                        # A budget, not a loop: the report's rules are
                        # heuristics, so past the ceiling the user gets the
                        # last attempt rather than a job that refuses to end.
                        break
                    seed = _fresh_seed()
                    log.info(
                        "job %s: rerolling the reference (%s)",
                        job_id,
                        "; ".join(report.reasons),
                    )
                if len(attempts) > 1:
                    # Only when it actually retried: a single-attempt job's
                    # provenance is already the seed in params.
                    params["reference_attempts"] = attempts
                    params["reference_seed"] = seed
                if is_reference:
                    try:
                        params["rank"] = await asyncio.to_thread(
                            self._rank_reference, job_dir, params
                        )
                    except Exception:
                        log.exception("ranking failed for job %s", job_id)
                if t2i.last_recipe:
                    params.setdefault("recipe", {})["reference"] = t2i.last_recipe
                await asyncio.to_thread(self.store.set_params, job_id, params)
            finally:
                ...unchanged...
```

Leave the `finally:` block (the handoff unload / `trim`) exactly as it is.

- [ ] **Step 5: Record the key as derived**

In `src/warlock/service/validation.py`, next to `"rank"`:

```python
    # Provenance for a reroll that already happened. It describes this run's
    # attempts, so a rerun inheriting it would claim retries it never made.
    "reference_attempts",
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_queue.py -v`
Expected: PASS, including the pre-existing seed and composed-prompt tests — they assert `seeds[-1]` and are unaffected by a loop that runs once.

- [ ] **Step 7: Show the attempts in the inspector**

In `src/warlock/studio/panes/inspector.py`, in `_reference` after the report block (line 194):

```python
    attempts = params.get("reference_attempts")
    if isinstance(attempts, list) and len(attempts) > 1:
        widgets.muted(
            f"redrawn {len(attempts) - 1} time(s): "
            + "; ".join(
                f"seed {a.get('seed')} {'kept' if a.get('ok') else 'refused'}"
                for a in attempts
            )
        )
```

- [ ] **Step 8: Run everything**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/warlock/config.py src/warlock/queue.py src/warlock/service/validation.py src/warlock/studio/panes/inspector.py tests/test_queue.py
git commit -m "Warlock v0.0.7

WARLOCK_REFERENCE_RETRIES: redraw a reference the composition report refuses,
inside the existing load/unload so the VRAM ordering is untouched. Off by
default, bounded, and every attempt is recorded.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Measure the hole-rate baseline before building anything that reacts to it

**This task needs the real machine:** a GPU, `vendor/trellis/trellis-server.exe`, the model weights, and `bpy` for the render half. It produces a measurement and a document, no code.

Project memory records that the old "7–31% of the silhouette is holes" claim is stale — the 2026-08-01 band sweep never measured worse than 1.7%. Task 7 builds a retry around `mesh_audit`, and a retry threshold set against a phantom is worse than no retry.

**Files:**
- Create: `docs/measurements/2026-08-04-hole-rate-baseline.md`

**Interfaces:**
- Consumes: `src/warlock/bench/suites/` (list them, pick the core suite), `params["mesh_audit"]["worst"]` on finished jobs.
- Produces: the decision Task 7 reads — a threshold value, or the finding that no threshold is warranted.

- [ ] **Step 1: See what suites and runs exist**

Run:
```bash
uv run python -m warlock.bench suites
uv run python -m warlock.bench recipes
```
Record the output. If a suite named `core-v1` exists, use it; otherwise use whichever suite the `suites` listing describes as the general one and say in the document which you used and why.

- [ ] **Step 2: Run the suite to the mesh stage**

Run:
```bash
uv run python -m warlock.bench run --suite <suite> --recipe <default recipe> --stage model
```
This is slow — minutes of GPU per item. Let it finish. If it fails, read `assets/trellis.log` and stop: a broken bench run is a separate problem from this plan, and guessing a baseline from a partial run is exactly the mistake this task exists to prevent.

- [ ] **Step 3: Collect the audit numbers**

Run this against the data dir the run wrote into:

```bash
uv run python -c "
import json, sqlite3, statistics
from warlock.config import get_config
c = get_config()
rows = sqlite3.connect(c.db_path).execute('select id, params from jobs').fetchall()
worst = []
for job_id, raw in rows:
    audit = (json.loads(raw) or {}).get('mesh_audit')
    if audit and audit.get('worst') is not None:
        worst.append((float(audit['worst']), job_id))
worst.sort(reverse=True)
print(f'n={len(worst)}')
for value, job_id in worst[:20]:
    print(f'  {value:.4f}  {job_id}')
if worst:
    values = [w for w, _ in worst]
    print('mean', round(statistics.fmean(values), 4))
    print('median', round(statistics.median(values), 4))
"
```

- [ ] **Step 4: Write the finding down**

Create `docs/measurements/2026-08-04-hole-rate-baseline.md` with: the exact commands run, the suite and recipe, `n`, the mean, the median, the worst twenty rows, and a verdict in one of these two forms.

> **Warranted.** k of n meshes exceeded X. A retry threshold of X catches those and nothing else. Task 7 proceeds with `WARLOCK_REMESH_HOLE_MAX` defaulting to X.

> **Not warranted.** The worst mesh measured Y, below any threshold that would fire on a real defect. Task 7 is not built; a retry with no defect to catch would spend two minutes of GPU on a mesh that was already fine. Revisit if the audit distribution changes.

Pick the threshold, if any, as roughly halfway between the tail of the healthy cluster and the worst outliers — and say in the document how you chose it.

- [ ] **Step 5: Commit the measurement**

```bash
git add docs/measurements/2026-08-04-hole-rate-baseline.md
git commit -m "Warlock v0.0.7

Measure the current mesh hole-rate baseline before building a retry around it,
the same way Config.trellis_band was settled.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Opt-in remesh when the audit says the mesh is see-through

**Entry condition: Task 6's document says "Warranted".** If it says "Not warranted", skip this task entirely, note the skip in the commit for Task 6, and move to Task 8.

`_audit_mesh` measures every finished mesh and records the number. Nothing acts on it. With a threshold from Task 6, a job whose worst view is more perforated than that retries the trellis stage with a fresh `mesh_seed` and keeps whichever GLB audited best.

In-worker, not a new linked job: the user sees one job that healed itself. It must respect the cancel event and, under exclusive mode, must not reorder anything — the retry re-enters only the trellis half, which by then has already had its handoff.

**Files:**
- Modify: `src/warlock/config.py` (`mesh_retries`, `mesh_hole_max`)
- Modify: `src/warlock/queue.py` (`_generate` mesh half, `_audit_mesh` returns its summary)
- Modify: `src/warlock/service/validation.py` (`DERIVED_PARAMS`)
- Test: `tests/test_queue.py`

**Interfaces:**
- Consumes: `params["mesh_audit"]["worst"]`, `Config.mesh_retries`, `Config.mesh_hole_max`.
- Produces: `params["mesh_attempts"]` — `[{"seed": int, "worst": float | None}, ...]`, oldest first, present only when more than one attempt ran. `Worker._audit_mesh` now returns `dict | None` (the stored summary) instead of `None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_queue.py`:

```python
def _audits(monkeypatch, worsts: list[float]):
    """meshaudit.hole_fraction returning a scripted sequence."""
    import warlock.meshaudit as meshaudit_mod

    seen = {"n": 0}

    def fake(path, views, resolution):
        value = worsts[min(seen["n"], len(worsts) - 1)]
        seen["n"] += 1
        return {
            "worst": value, "mean": value / 2, "faces": 1000,
            "resolution": resolution, "views": [],
        }

    monkeypatch.setattr(meshaudit_mod, "hole_fraction", fake)
    return seen


def _retry_worker(tmp_path, **config_kwargs):
    from warlock.config import Config
    from warlock.db import JobStore

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        **config_kwargs,
    )
    store = JobStore(config.db_path)
    return Worker(config, store), store


async def test_without_the_setting_a_holey_mesh_is_kept(worker, monkeypatch):
    _audits(monkeypatch, [0.5])
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert len(worker.trellis.generate_calls) == 1
    assert "mesh_attempts" not in worker.store.get(job_id)["params"]


async def test_a_holey_mesh_is_remeshed_with_a_fresh_seed(
    tmp_path, fake_pipelines, monkeypatch
):
    _audits(monkeypatch, [0.5, 0.01])
    w, store = _retry_worker(tmp_path, mesh_retries=1, mesh_hole_max=0.1)
    job_id = store.create("image", None, {"seed": 3, "mesh_seed": 3, "resolution": 512})
    job_dir = w.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    seeds = [c["seed"] for c in w.trellis.generate_calls]
    assert len(seeds) == 2 and seeds[0] == 3 and seeds[1] != 3
    params = store.get(job_id)["params"]
    assert [a["worst"] for a in params["mesh_attempts"]] == [0.5, 0.01]
    assert params["mesh_audit"]["worst"] == 0.01
    store.close()


async def test_a_mesh_that_passes_is_never_remeshed(tmp_path, fake_pipelines, monkeypatch):
    _audits(monkeypatch, [0.01])
    w, store = _retry_worker(tmp_path, mesh_retries=2, mesh_hole_max=0.1)
    job_id = store.create("image", None, {"seed": 3, "resolution": 512})
    job_dir = w.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert len(w.trellis.generate_calls) == 1
    store.close()


async def test_the_best_attempt_is_the_one_kept_even_when_it_is_the_first(
    tmp_path, fake_pipelines, monkeypatch
):
    # A reroll can be worse. Keeping the newest would then have spent two
    # minutes of GPU to make the asset worse than it already was.
    _audits(monkeypatch, [0.3, 0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=1, mesh_hole_max=0.1)
    job_id = store.create("image", None, {"seed": 3, "resolution": 512})
    job_dir = w.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert store.get(job_id)["params"]["mesh_audit"]["worst"] == 0.3
    store.close()


async def test_a_cancel_stops_the_retry(tmp_path, fake_pipelines, monkeypatch):
    _audits(monkeypatch, [0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=3, mesh_hole_max=0.1)
    job_id = store.create("image", None, {"seed": 3, "resolution": 512})
    job_dir = w.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: w.trellis.running)
    await w.request_cancel(job_id)
    await _wait_until(lambda: store.get(job_id)["status"] == "cancelled")
    await w.shutdown()

    assert len(w.trellis.generate_calls) == 1
    store.close()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_queue.py -k remesh -v`
Expected: FAIL — `TypeError: Config.__init__() got an unexpected keyword argument 'mesh_retries'`.

- [ ] **Step 3: Add the config fields**

In `src/warlock/config.py`, after `reference_retries`:

```python
    # How many extra times the trellis stage may run when the finished mesh
    # audits worse than mesh_hole_max. 0 -- off -- because a retry is two
    # minutes of GPU, and the 2026-08-04 baseline measurement is what decides
    # whether the defect it catches is real. See
    # docs/measurements/2026-08-04-hole-rate-baseline.md.
    mesh_retries: int = field(
        default_factory=lambda: max(0, int(os.environ.get("WARLOCK_MESH_RETRIES", "0")))
    )
    # The worst-view see-through fraction past which a mesh is worth redoing.
    # From the baseline measurement, not from a guess.
    mesh_hole_max: float = field(
        default_factory=lambda: float(os.environ.get("WARLOCK_MESH_HOLE_MAX", "<X from Task 6>"))
    )
```

Replace `<X from Task 6>` with the threshold the measurement document settled on, as a string literal (e.g. `"0.12"`).

- [ ] **Step 4: Make `_audit_mesh` report what it stored**

In `src/warlock/queue.py`, change `_audit_mesh`'s signature to `-> dict[str, Any] | None`, `return None` on both of its existing early exits and both `except` paths, and end it with `return params["mesh_audit"]` after the final `set_params`. Every existing caller ignores the value, so nothing else changes.

- [ ] **Step 5: Implement the retry**

In `_generate`, replace the three trailing calls (lines 800–802) with:

```python
        best: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        retries = max(0, int(self.config.mesh_retries))
        keep = job_dir / "best.glb"
        while True:
            await self._optimize(job_id, source_glb, glb_path, params)
            await self._apply_scale(job_id, glb_path, params)
            audit = await self._audit_mesh(job_id, glb_path, params)
            worst = None if audit is None else audit.get("worst")
            attempts.append({"seed": mesh_seed, "worst": worst})
            if best is None or (
                worst is not None
                and best["worst"] is not None
                and worst < best["worst"]
            ):
                # Kept aside rather than trusted to be last: a reroll can be
                # worse, and then two minutes of GPU would have made the asset
                # worse than it already was.
                best = {"worst": worst, "audit": audit, "params": dict(params)}
                await asyncio.to_thread(shutil.copyfile, glb_path, keep)
            if (
                worst is None
                or worst <= self.config.mesh_hole_max
                or len(attempts) > retries
                or self._cancel.event.is_set()
            ):
                break
            mesh_seed = _fresh_seed()
            log.info(
                "job %s: mesh audited %.3f open, past %.3f -- remeshing at seed %d",
                job_id, worst, self.config.mesh_hole_max, mesh_seed,
            )
            await self.trellis.generate(
                trellis_input,
                source_glb,
                seed=mesh_seed,
                resolution=resolution,
                bg_removal=str(params.get("bg_removal") or "auto"),
            )
        if len(attempts) > 1:
            if best is not None and best["worst"] != attempts[-1]["worst"]:
                # An earlier attempt won: put its GLB and its measurements back.
                await asyncio.to_thread(shutil.copyfile, keep, glb_path)
                params.update(best["params"])
            params["mesh_attempts"] = attempts
            params["mesh_seed"] = mesh_seed
            await asyncio.to_thread(self.store.set_params, job_id, params)
        with contextlib.suppress(OSError):
            keep.unlink()
```

Add `import shutil` to the stdlib imports at the top of `queue.py`.

- [ ] **Step 6: Record the key as derived and clean up the scratch file**

In `src/warlock/service/validation.py`, next to `"reference_attempts"`:

```python
    "mesh_attempts",
```

In `Worker._discard_artifacts` (line 552), add `job_dir / "best.glb"` to the non-rig `paths` list, so a cancelled retry leaves no scratch GLB behind.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_queue.py -v`
Expected: PASS, including the existing `test_mesh_audit_runs_after_scaling` and `test_trellis_output_is_kept_as_source_glb` — the loop runs exactly once with the default `mesh_retries=0`.

- [ ] **Step 8: Show the attempts in the inspector**

In `src/warlock/studio/panes/inspector.py`, `_quality`, after the audit line (line 237):

```python
    attempts = params.get("mesh_attempts")
    if isinstance(attempts, list) and len(attempts) > 1:
        widgets.muted(
            f"remeshed {len(attempts) - 1} time(s); kept the best of "
            + ", ".join(
                "unmeasured" if a.get("worst") is None else f"{float(a['worst']) * 100:.1f}%"
                for a in attempts
            )
        )
```

- [ ] **Step 9: Run everything**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/warlock/config.py src/warlock/queue.py src/warlock/service/validation.py src/warlock/studio/panes/inspector.py tests/test_queue.py
git commit -m "Warlock v0.0.7

WARLOCK_MESH_RETRIES: remesh in-worker when the audit says the mesh is more
see-through than the measured baseline allows, keeping the best attempt. Off by
default; threshold from docs/measurements/2026-08-04-hole-rate-baseline.md.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Answer calibrate.py's own question

**This task needs the real machine:** finished meshes in the data dir, and `bpy` for the render sweep. It produces a measurement and a document.

`bench/views.py` says `REFERENCE_YAW` / `REFERENCE_ELEVATION` are **UNCALIBRATED**, and `bench/calibrate.py` exists to settle whether a camera-matched view of a reconstruction exists at all. Task 9 is only buildable if it does.

**Files:**
- Create: `docs/measurements/2026-08-04-view-calibration.md`
- Modify (only if the verdict is "stable"): `src/warlock/bench/views.py`

**Interfaces:**
- Consumes: `bench.calibrate.verdict_lines`, `bench.calibrate.format_table`, `bench/calibrate.STABLE_YAW_SPREAD`.
- Produces: the verdict Task 9 reads, and (when stable) `views.REFERENCE_YAW` / `views.REFERENCE_ELEVATION` set to the mode.

- [ ] **Step 1: Check there are enough finished meshes**

Run:
```bash
uv run python -c "
from pathlib import Path
from warlock.config import get_config
root = Path(get_config().data_dir)
found = [d.name for d in root.iterdir() if (d/'model.glb').exists() and (d/'input.png').exists()]
print(len(found)); print(found)
"
```
`calibrate.verdict` returns "inconclusive" below three jobs and the plan calls for 5–10 spanning categories. If there are fewer, generate more first (Task 6's bench run produces them) — do not run the sweep on two jobs and read a mode into it.

- [ ] **Step 2: Run the sweep**

Run:
```bash
uv run python -m warlock.bench calibrate --all
```
It renders 144 views per job and scores every one. Capture the full output.

- [ ] **Step 3: Write the finding down**

Create `docs/measurements/2026-08-04-view-calibration.md` with the command, the job ids swept, the per-job `format_table` output, and the verdict lines verbatim. Then state the conclusion in one of these two forms:

> **Stable.** The argmax yaws are `[...]`, spread `S` degrees, inside `STABLE_YAW_SPREAD`. `REFERENCE_YAW` is set to the mode `M` and `REFERENCE_ELEVATION` to `E`. Task 9 proceeds.

> **Scattered.** The argmax yaws are `[...]`, spread `S` degrees. There is no fixed matched view, so a request-path fidelity score would be measuring the camera, not the mesh. Task 9 stops here; `views.py` keeps its UNCALIBRATED note, amended to say the sweep was run and what it found.

- [ ] **Step 4: If stable, set the constants**

In `src/warlock/bench/views.py`, replace the `UNCALIBRATED` paragraph (lines 30–37) with the measured table — the pattern `Config.trellis_band` follows: the numbers, the date, the command, the sample size, and then `REFERENCE_YAW` / `REFERENCE_ELEVATION` set to the mode.

If scattered, amend the same paragraph to say the sweep ran on `2026-08-04`, over N jobs, and scattered by S degrees — so the next reader does not repeat it.

- [ ] **Step 5: Run the suite and commit**

Run: `uv run pytest && uv run ruff check .`

```bash
git add docs/measurements/2026-08-04-view-calibration.md src/warlock/bench/views.py
git commit -m "Warlock v0.0.7

Run the yaw/elevation sweep bench/calibrate.py exists for and record whether a
camera-matched view of a reconstruction exists.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Fidelity score on the request path

**Entry condition: Task 8's document says "Stable".** If it says "Scattered", skip this task; the finding is already recorded and there is nothing honest to build.

With a calibrated view, a finished mesh can be rendered once at low resolution from that view and its silhouette compared against the reference the job was built from. Advisory, stored and displayed like `mesh_report`: nothing rejects a mesh whose GLB is already on disk.

**Files:**
- Modify: `src/warlock/queue.py` (`_measure_fidelity`, called from `_generate` after the audit)
- Modify: `src/warlock/config.py` (`fidelity_check`)
- Modify: `src/warlock/service/validation.py` (`DERIVED_PARAMS`)
- Modify: `src/warlock/studio/panes/inspector.py` (`_quality`)
- Test: `tests/test_queue.py`

**Interfaces:**
- Consumes: `rigging.sheet_spec(model, out_dir, cells, frame_size=, elevation=, lighting=)` and `rigging.run_worker(spec, timeout=)` (`src/warlock/bench/calibrate.py:105` is the working call site to copy); `bench.metrics.silhouette_iou(reference, render) -> float | None`; `bench.views.REFERENCE_YAW` / `REFERENCE_ELEVATION` / `VIEW_LIGHTING`.
- Produces: `params["fidelity"]` — `{"iou": float, "yaw": float, "elevation": float, "size": int}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_queue.py`:

```python
async def test_a_finished_mesh_carries_a_fidelity_score(
    tmp_path, fake_pipelines, monkeypatch
):
    import warlock.rigging as rigging_mod
    from warlock.bench import metrics as metrics_mod
    from warlock.config import Config
    from warlock.db import JobStore

    rendered = {}

    def fake_run_worker(spec, **kwargs):
        # The worker writes one PNG per cell into the scratch dir it was given.
        out = Path(spec["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "0000.png").write_bytes(b"fake-render")
        rendered["spec"] = spec
        return {}

    monkeypatch.setattr(rigging_mod, "run_worker", fake_run_worker)
    monkeypatch.setattr(metrics_mod, "silhouette_iou", lambda ref, render: 0.62)

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        fidelity_check=True,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("image", None, {"seed": 1, "resolution": 512})
    job_dir = config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert store.get(job_id)["params"]["fidelity"]["iou"] == 0.62
    # The calibrated view, not an arbitrary one.
    from warlock.bench import views

    assert rendered["spec"]["elevation"] == views.REFERENCE_ELEVATION
    store.close()


async def test_a_failing_fidelity_measurement_does_not_fail_the_job(
    tmp_path, fake_pipelines, monkeypatch
):
    import warlock.rigging as rigging_mod
    from warlock.config import Config
    from warlock.db import JobStore

    monkeypatch.setattr(
        rigging_mod,
        "run_worker",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no blender")),
    )
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        fidelity_check=True,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("image", None, {"seed": 1, "resolution": 512})
    job_dir = config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert store.get(job_id)["status"] == "done"
    assert "fidelity" not in store.get(job_id)["params"]
    store.close()
```

Before writing this, open `src/warlock/rigging.py` and read `sheet_spec`'s actual return shape — the test asserts on `spec["out_dir"]` and `spec["elevation"]`, and the real key names must be used instead if they differ.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_queue.py -k fidelity -v`
Expected: FAIL — `TypeError: Config.__init__() got an unexpected keyword argument 'fidelity_check'`.

- [ ] **Step 3: Add the config field**

In `src/warlock/config.py`, after `mesh_hole_max`:

```python
    # Whether a finished mesh is rendered once from the calibrated view and
    # compared against the reference it was built from. Off by default: it is
    # a Blender launch per job, and the number is advisory -- see
    # docs/measurements/2026-08-04-view-calibration.md for the sweep that
    # established the view it renders from.
    fidelity_check: bool = field(
        default_factory=lambda: os.environ.get("WARLOCK_FIDELITY", "").lower()
        in ("1", "true", "on", "yes")
    )
```

- [ ] **Step 4: Implement the measurement**

In `src/warlock/queue.py`, after `_audit_mesh`:

```python
    # Small on purpose: this is a silhouette IoU, which is a coarse question
    # about shape, and the render is one more Blender launch per job.
    FIDELITY_SIZE = 256

    async def _measure_fidelity(
        self, job_id: str, glb_path: Path, job_dir: Path, params: dict[str, Any]
    ) -> None:
        """Does the mesh have the shape the reference showed?

        One render from the calibrated view (bench/views.py, settled by the
        2026-08-04 sweep) against input.png's subject, both cropped to their
        own bounding boxes so this measures shape rather than framing.

        Advisory, and swallowed on failure like _audit_mesh and the report: the
        GLB is on disk and correct, and a diagnostic that can fail a finished
        job is worse than no diagnostic.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        reference_png = job_dir / "input.png"
        if not (self.config.fidelity_check and reference_png.exists()):
            return
        self.progress.update(
            job_id, phase="audit", label="Checking fidelity", inner=0.0,
            inner_next=1.0, nominal=8.0, detail="",
        )
        try:
            from .bench import metrics, views

            with tempfile.TemporaryDirectory(prefix="warlock-fid-") as scratch:
                out = Path(scratch)
                cells = [
                    {
                        "index": 0, "row": 0, "column": 0,
                        "yaw": views.REFERENCE_YAW, "frame": 0,
                        "pose": None, "bones": {},
                    }
                ]
                spec = rigging.sheet_spec(
                    glb_path,
                    out,
                    cells,
                    frame_size=self.FIDELITY_SIZE,
                    elevation=views.REFERENCE_ELEVATION,
                    lighting=views.VIEW_LIGHTING,
                )
                await asyncio.to_thread(
                    functools.partial(
                        rigging.run_worker, spec, timeout=self.config.pose_timeout
                    )
                )
                render = out / "0000.png"
                if not render.exists():
                    return
                iou = await asyncio.to_thread(
                    metrics.silhouette_iou, reference_png, render
                )
        except Exception:
            log.exception("fidelity measurement failed for job %s", job_id)
            return
        if iou is None:
            return
        params["fidelity"] = {
            "iou": float(iou),
            "yaw": views.REFERENCE_YAW,
            "elevation": views.REFERENCE_ELEVATION,
            "size": self.FIDELITY_SIZE,
        }
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info("job %s fidelity: silhouette IoU %.3f", job_id, iou)
```

Call it in `_generate` as the last line of the mesh half, after the audit loop:

```python
        await self._measure_fidelity(job_id, glb_path, job_dir, params)
```

(`tempfile` is already imported in `queue.py`; confirm before adding it again.)

- [ ] **Step 5: Record the key as derived**

In `src/warlock/service/validation.py`:

```python
    "fidelity",
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_queue.py -k fidelity -v`
Expected: PASS.

- [ ] **Step 7: Show it in the inspector**

In `src/warlock/studio/panes/inspector.py`, `_quality`, after the audit line:

```python
    fidelity = params.get("fidelity")
    if isinstance(fidelity, dict) and fidelity.get("iou") is not None:
        widgets.muted(f"matches the reference: {float(fidelity['iou']) * 100:.0f}%")
        widgets.help_marker(
            "Silhouette overlap between the mesh, rendered from the matched "
            "view, and the reference it was built from. Advisory -- a low "
            "score on a stylised reference is expected."
        )
```

- [ ] **Step 8: Run everything**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/warlock/config.py src/warlock/queue.py src/warlock/service/validation.py src/warlock/studio/panes/inspector.py tests/test_queue.py
git commit -m "Warlock v0.0.7

WARLOCK_FIDELITY: render a finished mesh once from the calibrated view and
record its silhouette IoU against the reference. Advisory, off by default.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] `uv run pytest` — full suite green (expect ~427 passing with 6 skipped in a worktree without `vendor/` and `models/`, or ~433 in the main checkout; see the worktree note in project memory).
- [ ] `uv run ruff check .` — clean.
- [ ] Every new `params` key appears in `service.validation.DERIVED_PARAMS`: `rank`, `reference_attempts`, and (where built) `mesh_attempts`, `fidelity`.
- [ ] Launch the app: `uv run warlock-studio` (confirm the entry point in `pyproject.toml` first). Then, by hand:
  - Pick `turbo` under Advanced in the 2D pane and confirm the Negative field is greyed with the reason under it; switch to `playground` and confirm it becomes live and the Structure group appears.
  - Create a profile, attach a style anchor, set it active, and generate four references at `count = 4`. Confirm each card shows a score, that "best first" reorders them, and that `assets/<job>/ref.png` is the anchor image.
  - Attach a different reference in the 2D pane's Reference section and confirm the anchor note says it is being replaced and the submitted `ref.png` is the manual one.
  - Rename the profile and confirm the anchor thumbnail survives and its file is still on disk.
- [ ] With `WARLOCK_REFERENCE_RETRIES=1`, generate against a prompt that produces an off-frame subject and confirm the inspector's Reference section reports the redraw.
- [ ] The two measurement documents exist under `docs/measurements/` and each states a verdict, including the ones that say a feature was not built.

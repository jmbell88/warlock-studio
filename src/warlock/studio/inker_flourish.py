"""Flourish in the editor: predicates, the pending recipe, and the render task.

The headless half of the feature's UI, ``inker_sheet``'s shape: everything
the ops registry, the inspector and the tests ask is answered here with no
imgui in reach, so ``inker_ops`` can grey a row with the same sentence the
panel shows and a test can drive the whole loop with a fake ``ctx``.

**Renders run in a task; the document is written on the frame thread.** A
slider reports on every frame of a drag, and a bake is a hundred milliseconds
of numpy, so the inspector writes its edits to ``state.flourish_pending`` and
this module submits *one* render once the value has rested for
``DEBOUNCE_SECONDS``. The result comes back through ``inker_mode.on_task_done``
on the ``inker-flourish`` prefix and lands as one undo step
(``Document.apply_flourish``). A result for a tab that has closed, a group
that has been dissolved, or a recipe the user has since moved past is dropped
rather than shown -- ``viewer_embed``'s pending-marker rule, applied to cels.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

FLOURISH_POPUP = "inker-flourish-insert"
SNIPPET_POPUP = "inker-flourish-snippet"
TEXTURE_POPUP = "inker-flourish-texture"
TEXTURE_KEY = "inker-flourish-texture"
TEXTURE_LAND_KEY = "inker-flourish-texture-land"
PROMPT_KEY = "inker-flourish-prompt"
RESTYLE_POPUP = "inker-flourish-restyle"
RESTYLE_KEY = "inker-flourish-restyle"
RESTYLE_LAND_KEY = "inker-flourish-restyle-land"
RESTYLE_PENDING = "A restyle of this effect is already running."
#: The words around the user's own, for a keyframe the pixel model repaints.
RESTYLE_PROMPT_TEMPLATE = "{subject}, 2D game VFX frame, centered, transparent background"
#: How long the text model may take before the words fall back to the mapper.
PROMPT_TIMEOUT_S = 120.0
RENDER_KEY = "inker-flourish"
INSERT_KEY = "inker-flourish-insert"

#: How long a slider value has to rest before the render for it is submitted.
DEBOUNCE_SECONDS = 0.25


def clock() -> float:
    """The one clock the debounce runs on. Monotonic rather than imgui's,
    because ``land`` runs from the task-completion path where no imgui
    context need exist, and a due time compared across two clocks is never
    due or always due."""
    return time.monotonic()

NO_EFFECT = "The active layer is not part of a Flourish effect."
BUSY = "The document is busy -- a save, an export or playback is still running."
RENDERING = "A render of this effect is still running."
NO_CONFLICTS = "No cells of this effect are flagged."
NO_SELECTION = "Select the pixels to use as a texture first."
TEXTURE_PENDING = "A texture is already being generated."
#: The words around the user's own, for a texture the pixel model paints.
TEXTURE_PROMPT_TEMPLATE = (
    "single {subject}, centered, on a plain black background, 2D game VFX texture, "
    "no objects, no text, high contrast"
)
#: How often the door asks the store about a pending texture.
TEXTURE_POLL_S = 0.5
#: A generated texture is brought down to this on its long side.
TEXTURE_MAX_PX = 256


def render_key(tab: Any, group_uid: int) -> str:
    return f"{RENDER_KEY}:{tab.uid}:{int(group_uid)}"


def insert_key(tab: Any) -> str:
    return f"{INSERT_KEY}:{tab.uid}"


# -- predicates (the ops registry reads these) ------------------------------------


def active_group(state: Any, tab: Any) -> int | None:
    doc = getattr(tab, "doc", None)
    if doc is None or not hasattr(doc, "flourish_group_of_active"):
        return None
    return doc.flourish_group_of_active()


def has_effect(state: Any, tab: Any) -> bool:
    return active_group(state, tab) is not None


def can_insert(state: Any, tab: Any) -> bool:
    return tab is not None and not getattr(tab, "busy", False)


def insert_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    return BUSY if getattr(tab, "busy", False) else ""


def can_regenerate(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def regenerate_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    return BUSY if getattr(tab, "busy", False) else ""


def has_conflicts(state: Any, tab: Any) -> bool:
    group = active_group(state, tab)
    return group is not None and bool(tab.doc.flourish_conflicts(group))


def conflicts_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    return "" if has_conflicts(state, tab) else NO_CONFLICTS


# -- the pending recipe ---------------------------------------------------------------


def current_recipe(state: Any, tab: Any, group_uid: int) -> Any:
    """What the inspector shows: the pending edit if there is one, else the
    document's own."""
    pending = state.flourish_pending.get(int(group_uid))
    if pending is not None:
        return pending
    held = tab.doc.flourish_state(group_uid)
    return None if held is None else held.recipe


def set_pending(state: Any, group_uid: int, recipe: Any, *, now: float) -> None:
    """Record an edit and restart its debounce clock."""
    state.flourish_pending[int(group_uid)] = recipe
    state.flourish_due[int(group_uid)] = float(now) + DEBOUNCE_SECONDS


def due(state: Any, *, now: float) -> list[int]:
    """Groups whose pending edit has rested long enough to render."""
    return [g for g, at in state.flourish_due.items() if float(now) >= at]


def in_flight(ctx: Any, tab: Any, group_uid: int) -> bool:
    return bool(ctx.busy(render_key(tab, group_uid)))


# -- tasks --------------------------------------------------------------------------


def submit_render(ctx: Any, tab: Any, group_uid: int, recipe: Any, *, force: bool = False) -> bool:
    """Bake ``recipe`` off-thread for ``group_uid``. -> whether it was accepted.
    The group's textures go with it, read once here on the frame thread."""
    from .inker.flourish import bake as flourish_bake

    key = render_key(tab, group_uid)
    tab_uid = tab.uid
    group_uid = int(group_uid)
    held = tab.doc.flourish_state(group_uid)
    assets = dict(held.assets) if held is not None else {}

    def work() -> dict[str, Any]:
        def progress(done: int, total: int) -> None:
            ctx.tasks.set_progress(key, 100.0 * done / max(total, 1), f"{done}/{total}")

        baked = flourish_bake.bake(recipe, progress=progress, assets=assets)
        return {"tab": tab_uid, "group": group_uid, "baked": baked, "force": force}

    return bool(ctx.submit(key, work))


def submit_insert(ctx: Any, tab: Any, recipe: Any) -> bool:
    from .inker.flourish import bake as flourish_bake

    key = insert_key(tab)
    tab_uid = tab.uid

    def work() -> dict[str, Any]:
        def progress(done: int, total: int) -> None:
            ctx.tasks.set_progress(key, 100.0 * done / max(total, 1), f"{done}/{total}")

        return {"tab": tab_uid, "baked": flourish_bake.bake(recipe, progress=progress)}

    return bool(ctx.submit(key, work))


def tick(ctx: Any, state: Any, tab: Any, *, now: float) -> int:
    """Submit every render that has become due for ``tab``. -> how many."""
    if tab is None or getattr(tab, "busy", False):
        return 0
    sent = 0
    for group in due(state, now=now):
        recipe = state.flourish_pending.get(group)
        if recipe is None or tab.doc.flourish_state(group) is None:
            state.flourish_due.pop(group, None)
            state.flourish_pending.pop(group, None)
            continue
        if in_flight(ctx, tab, group):
            # Let it rest until the running render lands; ``land`` re-arms the
            # clock when the pending recipe has moved past what it rendered.
            continue
        if submit_render(ctx, tab, group, recipe):
            state.flourish_due.pop(group, None)
            sent += 1
    return sent


def _tab_by_uid(state: Any, uid: str) -> Any:
    for tab in getattr(state, "docs", []):
        if tab.uid == uid:
            return tab
    return None


def land(ctx: Any, state: Any, done: Any, *, now: float) -> bool:
    """Frame thread: put a finished bake onto its document. -> whether it did."""
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast(f"The effect could not be rendered: {done.error or 'no result'}", "warn")
        return False
    tab = _tab_by_uid(state, result.get("tab", ""))
    if tab is None:
        return False  # the tab closed; nothing to land on and nobody to tell
    baked = result["baked"]
    if done.key.startswith(INSERT_KEY):
        group = tab.doc.insert_flourish(baked)
        state.flourish_layer[group] = baked.recipe.layers[0].uid if baked.recipe.layers else 0
        ctx.toast(f"Inserted {baked.recipe.name}: {baked.frame_count} frames.", "success")
        return True
    group = int(result["group"])
    if tab.doc.flourish_state(group) is None:
        ctx.toast("That effect was detached while it rendered; nothing landed.", "info")
        state.flourish_pending.pop(group, None)
        return False
    counts = tab.doc.apply_flourish(group, baked, force=bool(result.get("force")))
    pending = state.flourish_pending.get(group)
    if pending is not None and pending == baked.recipe:
        state.flourish_pending.pop(group, None)
    elif pending is not None:
        # The user kept editing while this rendered: render the newer one next.
        state.flourish_due[group] = float(now)
    ctx.toast(counts.sentence(), "warn" if counts.conflicts else "success")
    return True


# -- export and engine snippets -------------------------------------------------------------


def can_export(state: Any, tab: Any) -> bool:
    return (
        has_effect(state, tab)
        and not getattr(tab, "busy", False)
        and bool(getattr(tab.doc.anim, "tags", None))
    )


def export_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    if getattr(tab, "busy", False):
        return BUSY
    return "" if getattr(tab.doc.anim, "tags", None) else "This document has no tags to export by."


def tag_names(tab: Any) -> list[str]:
    anim = getattr(tab.doc, "anim", None)
    return [] if anim is None else [tag.name for tag in anim.tags]


def snippet_info(tab: Any, tag_name: str) -> dict[str, Any] | None:
    """What ``engines.snippet`` needs for one exported phase: the file the
    per-tag export writes (``sheetout.DEFAULT_TAG_TEMPLATE`` over the
    document's title), the frames the tag spans, the rate from the tag's
    first frame, the loop flag, and the origin -- the canvas centre, which is
    where ``bake`` puts an effect by construction."""
    from .inker import sheetout
    from .inker.flourish import engines

    anim = getattr(tab.doc, "anim", None)
    if anim is None:
        return None
    tag = next((t for t in anim.tags if t.name == tag_name), None)
    if tag is None:
        return None
    first, last = sheetout.tag_span(anim, tag)
    duration = max(1, int(anim.frames[first].duration_ms))
    title = Path(str(getattr(tab, "title", "") or "effect")).stem or "effect"
    stem = sheetout.filename_for(sheetout.DEFAULT_TAG_TEMPLATE, title=title, tag=tag.name)
    width, height = tab.doc.size
    return engines.describe(
        name=f"{title} {tag.name}",
        image=f"{stem}.png",
        frame_width=width,
        frame_height=height,
        frames=last - first + 1,
        fps=max(1, round(1000.0 / duration)),
        loop=bool(tag.loop),
        origin=(width // 2, height // 2),
    )


def snippet_text(tab: Any, tag_name: str, engine: str) -> str:
    from .inker.flourish import engines

    info = snippet_info(tab, tag_name)
    if info is None:
        return ""
    return engines.snippet(engine, info)


# -- textures --------------------------------------------------------------------------------


def can_texture_selection(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and getattr(tab.doc, "mask", None) is not None


def texture_selection_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return "Nothing is open."
    if not has_effect(state, tab):
        return NO_EFFECT
    return "" if getattr(tab.doc, "mask", None) is not None else NO_SELECTION


def can_texture_generate(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def texture_generate_reason(state: Any, tab: Any) -> str:
    return regenerate_reason(state, tab)


def texture_from_selection(ctx: Any, state: Any, tab: Any) -> str | None:
    """The selection's pixels become a texture of the active effect, and the
    inspector's layer takes it if it has a ``texture`` parameter. One step."""
    group = active_group(state, tab)
    if group is None:
        return None
    cutout = tab.doc.selection_cutout()
    if cutout is None or not cutout[..., 3].any():
        state.say(NO_SELECTION)
        return None
    asset_id = tab.doc.add_flourish_asset(group, cutout)
    _assign_texture(state, tab, group, asset_id)
    return asset_id


def _assign_texture(state: Any, tab: Any, group: int, asset_id: str) -> None:
    """Point the inspector's current layer at ``asset_id`` when it can take
    one, as a pending edit -- the render that lands it is one step."""
    from .inker.flourish import prims

    recipe = current_recipe(state, tab, group)
    if recipe is None or not recipe.layers:
        return
    uid = state.flourish_layer.get(group)
    layer = next((each for each in recipe.layers if each.uid == uid), recipe.layers[-1])
    if "texture" not in prims.params_of(layer.kind):
        return
    edited = recipe.replace_layer(layer.with_param("texture", asset_id))
    set_pending(state, group, edited, now=clock())


def key_out_black(pixels: np.ndarray) -> np.ndarray:
    """A straight-alpha cutout of a picture painted on black: alpha from the
    brightest channel. The offline fallback when no matting model is present,
    and the *right* answer for an additive VFX texture, whose black *is*
    transparency."""
    rgb = pixels[..., :3].astype(np.float32)
    alpha = rgb.max(axis=-1)
    out = np.empty(pixels.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = pixels[..., :3]
    out[..., 3] = np.clip(np.rint(alpha), 0, 255).astype(np.uint8)
    return out


def submit_texture(ctx: Any, state: Any, tab: Any, subject: str) -> bool:
    """Ask the pixel model for a texture. The job goes to the queue like any
    reference job (``inker_bridge.submit_inpaint``'s door); ``poll_texture``
    watches for it and ``land_texture`` puts the cutout on the effect."""
    group = active_group(state, tab)
    if group is None or tab.busy:
        return False
    if state.flourish_texture_pending is not None:
        state.say(TEXTURE_PENDING)
        return False
    prompt = TEXTURE_PROMPT_TEMPLATE.format(subject=subject.strip() or "magical flame")
    pending = {
        "tab_uid": tab.uid,
        "group": int(group),
        "layer": state.flourish_layer.get(group),
        "job_id": "",
        "next_poll": 0.0,
        "subject": subject.strip(),
    }
    key = f"{TEXTURE_KEY}:{tab.uid}"

    def run() -> Any:
        from ..service import jobs as svc_jobs

        return svc_jobs.create_job(
            ctx.svc,
            kind="text",
            prompt=prompt,
            negative="photo, realistic, text, watermark, frame, border",
            output="reference",
            count=1,
        )

    if not ctx.submit(key, run):
        return False
    state.flourish_texture_pending = pending
    ctx.toast("Generating a texture...")
    return True


def on_texture_queued(ctx: Any, state: Any, done: Any) -> None:
    pending = state.flourish_texture_pending
    if pending is None:
        return
    result = done.result
    job_id = ""
    if done.error is None and isinstance(result, dict):
        job_id = str(result.get("id") or "")
        if not job_id:
            ids = result.get("ids") or result.get("jobs") or []
            job_id = str(ids[0]) if ids else ""
    if not job_id:
        state.flourish_texture_pending = None
        ctx.toast(f"The texture was not queued: {done.error or 'no job id'}.", "warn")
        return
    pending["job_id"] = job_id


def poll_texture(ctx: Any, state: Any, *, now: float) -> None:
    """Frame thread, cheap: once every ``TEXTURE_POLL_S`` ask whether the job
    is done, and hand the decode to a task."""
    pending = state.flourish_texture_pending
    if pending is None or not pending.get("job_id"):
        return
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + TEXTURE_POLL_S
    try:
        job = ctx.svc.store.get(pending["job_id"])
    except Exception:  # noqa: BLE001 -- the store answers next tick
        return
    if job is None:
        state.flourish_texture_pending = None
        return
    status = job.get("status")
    if status in ("queued", "running"):
        return
    state.flourish_texture_pending = None
    if status != "done":
        ctx.toast(f"The texture {status}: {job.get('error') or 'no result'}.", "warn")
        return
    image_path = ctx.svc.job_dir(pending["job_id"]) / "input.png"
    key = f"{TEXTURE_LAND_KEY}:{pending['tab_uid']}"
    ctx.submit(key, decode_texture, pending, image_path, ctx.svc)


def decode_texture(
    pending: dict[str, Any], image_path: Any, svc: Any = None
) -> dict[str, Any] | None:
    """Task thread. The picture as a cutout, brought down to ``TEXTURE_MAX_PX``:
    the matting model where the machine has one, black-keyed otherwise."""
    from PIL import Image

    from ..pipelines import matting

    try:
        with Image.open(image_path) as im:
            im.load()
            picture = im.convert("RGBA")
    except OSError:
        return None
    picture.thumbnail((TEXTURE_MAX_PX, TEXTURE_MAX_PX), Image.Resampling.LANCZOS)
    pixels = np.asarray(picture, dtype=np.uint8).copy()
    config = getattr(svc, "config", None)
    source = "black-key"
    if config is not None and matting.available(config):
        try:
            found, source = matting.mask(picture, config)
            cut = pixels.copy()
            cut[..., 3] = np.where(np.asarray(found, dtype=bool), cut[..., 3], 0)
            pixels = cut
        except Exception:  # noqa: BLE001 -- the fallback is always right, if rougher
            source = "black-key"
            pixels = key_out_black(pixels)
    else:
        pixels = key_out_black(pixels)
    return {"pending": pending, "pixels": pixels, "source": source}


def land_texture(ctx: Any, state: Any, done: Any) -> bool:
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast("The generated texture could not be read.", "warn")
        return False
    pending = result["pending"]
    tab = _tab_by_uid(state, pending["tab_uid"])
    if tab is None:
        return False
    group = int(pending["group"])
    if tab.doc.flourish_state(group) is None:
        ctx.toast("That effect was detached while its texture generated.", "info")
        return False
    asset_id = tab.doc.add_flourish_asset(group, result["pixels"], stem="gen")
    if pending.get("layer") is not None:
        state.flourish_layer[group] = pending["layer"]
    _assign_texture(state, tab, group, asset_id)
    ctx.toast(f"Texture {asset_id} added ({result.get('source', '')}).", "success")
    return True


# -- the prompt field --------------------------------------------------------------------------


#: The directory under the model root the prompt field looks in. **Not a
#: registry entry**: every ``models`` entry carries a fetch pinned to a
#: revision, and the pin comes from the measurement that picks the model
#: (still owed, on the human's list). Until then a user who wants to try one puts an instruct
#: model's ``config.json`` and safetensors here by hand, and doctor says so.
TEXT_MODEL_DIR = "text-instruct"


def text_model_dir(config: Any) -> Path | None:
    """Where the text model would be, or None with no config in reach."""
    if config is None:
        return None
    return Path(config.t2i_model_root) / TEXT_MODEL_DIR


def text_model_present(config: Any) -> bool:
    """Weights on disk: ``config.json`` and at least one safetensors file, the
    same two facts ``fetch.present`` asks of every helper model."""
    base = text_model_dir(config)
    if base is None or not (base / "config.json").exists():
        return False
    return any(base.rglob("*.safetensors"))


def text_model_available(config: Any) -> bool:
    """Weights on disk *and* the packages to run them, checked before any torch
    import -- ``tests/test_offline.py``'s ordering rule."""
    if not text_model_present(config):
        return False
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


def can_prompt(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def prompt_reason(state: Any, tab: Any) -> str:
    return regenerate_reason(state, tab)


def ask_words(recipe: Any, text: str, *, model_dir: Path | None) -> tuple[Any, list[str], str]:
    """Task thread. -> ``(recipe, notes, source)``: the text model when there
    is one and it answers, the keyword mapper otherwise -- always something,
    and the source says which, because a change the user cannot attribute is
    a change they cannot trust."""
    from .inker.flourish import keywords

    if model_dir is not None:
        diff, why = run_text_model(recipe, text, model_dir)
        if diff is not None:
            changed, notes = keywords.apply_diff(recipe, diff)
            return changed, notes, "model"
        changed, notes = keywords.apply(recipe, text)
        return changed, [f"model: {why}; used the keyword mapper", *notes], "keywords"
    changed, notes = keywords.apply(recipe, text)
    return changed, notes, "keywords"


def run_text_model(recipe: Any, text: str, model_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """One child, one answer. -> ``(diff, reason-if-none)``."""
    import json
    import subprocess
    import sys

    from .. import winjob
    from .inker.flourish import keywords

    request = {
        "model_dir": str(model_dir),
        "recipe": keywords.describe_for_model(recipe),
        "request": text,
        "schema": keywords.DIFF_SCHEMA,
    }
    argv = [sys.executable, "-m", "warlock.pipelines.recipe_worker"]
    try:
        proc = winjob.run(
            argv,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROMPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, f"no answer within {PROMPT_TIMEOUT_S:.0f} s"
    except OSError as exc:
        return None, f"could not start the text model: {exc}"
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None, "the text model said nothing"
    try:
        answer = json.loads(line[-1])
    except json.JSONDecodeError:
        return None, "the text model's answer was not JSON"
    if not isinstance(answer, dict) or answer.get("kind") != "ok":
        reason = answer.get("error") if isinstance(answer, dict) else "malformed answer"
        return None, str(reason or "the text model failed")
    diff = answer.get("diff")
    return (diff if isinstance(diff, dict) else None), "the diff was not an object"


def submit_prompt(ctx: Any, state: Any, tab: Any, text: str) -> bool:
    """Words -> a pending recipe, off-thread; ``land_prompt`` sets it pending
    and the ordinary render then lands it as one step."""
    group = active_group(state, tab)
    text = (text or "").strip()
    if group is None or not text or tab.busy:
        return False
    recipe = current_recipe(state, tab, group)
    if recipe is None:
        return False
    key = f"{PROMPT_KEY}:{tab.uid}:{group}"
    config = getattr(getattr(ctx, "svc", None), "config", None)
    model_dir = text_model_dir(config) if text_model_available(config) else None
    tab_uid = tab.uid

    def work() -> dict[str, Any]:
        changed, notes, source = ask_words(recipe, text, model_dir=model_dir)
        return {"tab": tab_uid, "group": group, "recipe": changed, "notes": notes, "source": source}

    if not ctx.submit(key, work):
        state.say("The last prompt is still being read.")
        return False
    return True


def land_prompt(ctx: Any, state: Any, done: Any, *, now: float) -> bool:
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast(f"The prompt could not be applied: {done.error or 'no result'}", "warn")
        return False
    tab = _tab_by_uid(state, result.get("tab", ""))
    if tab is None:
        return False
    group = int(result["group"])
    if tab.doc.flourish_state(group) is None:
        return False
    notes = list(result.get("notes") or [])
    recipe = result["recipe"]
    if recipe == current_recipe(state, tab, group):
        ctx.toast("; ".join(notes) or "Nothing changed.", "info")
        return False
    set_pending(state, group, recipe, now=now)
    source = "model" if result.get("source") == "model" else "keywords"
    ctx.toast(f"[{source}] " + "; ".join(notes[:6]), "success")
    return True


# -- restyled keyframes -----------------------------------------------------------------------


def can_restyle(state: Any, tab: Any) -> bool:
    return has_effect(state, tab) and not getattr(tab, "busy", False)


def restyle_reason(state: Any, tab: Any) -> str:
    return regenerate_reason(state, tab)


def phase_names(state: Any, tab: Any) -> list[str]:
    group = active_group(state, tab)
    if group is None:
        return []
    held = tab.doc.flourish_state(group)
    return [p.name for p in held.recipe.phases] if held is not None else []


def _phase_span(state_held: Any, anim: Any, phase_name: str) -> tuple[int, int] | None:
    """The flat frame span of one phase of the effect, from its tag."""
    from .inker import sheetout

    for tag in anim.tags:
        if tag.name == phase_name or tag.name.startswith(phase_name + "/"):
            return sheetout.tag_span(anim, tag)
    return None


def submit_restyle(
    ctx: Any,
    state: Any,
    tab: Any,
    *,
    phase: str,
    subject: str,
    strength: float = 0.55,
    anchors: int = 3,
) -> bool:
    """Send ``anchors`` frames of ``phase`` -- the effect's own composite --
    through the image model as img2img, one reference job each; ``poll_restyle``
    collects them and ``land_restyle`` interpolates the rest and lands a
    snapshot track. Opt-in, never default: whether this beats the procedural
    frames is a measurement, not a setting."""
    from .inker import sheetout
    from .inker.flourish import keyframes

    group = active_group(state, tab)
    if group is None or tab.busy:
        return False
    if state.flourish_restyle_pending is not None:
        state.say(RESTYLE_PENDING)
        return False
    held = tab.doc.flourish_state(group)
    anim = tab.doc.anim
    if held is None or anim is None:
        return False
    span = _phase_span(held, anim, phase)
    if span is None:
        state.say(f"This effect has no phase called {phase!r}.")
        return False
    first, last = span
    frames = keyframes.anchor_frames(first, last, anchors)
    track_uids = [uid for uid in held.tracks.values()]
    uids = sheetout.frame_uids(tab.doc)
    pictures: dict[int, bytes] = {}
    for index in frames:
        plane = sheetout.flatten_subset(tab.doc, uids[index], track_uids)
        pictures[index] = _png_bytes(plane)
    prompt = RESTYLE_PROMPT_TEMPLATE.format(subject=subject.strip() or "painted magical effect")
    pending = {
        "tab_uid": tab.uid,
        "group": int(group),
        "phase": phase,
        "span": [first, last],
        "jobs": {},  # frame index -> job id
        "frames": list(frames),
        "next_poll": 0.0,
        "subject": subject.strip(),
    }
    key = f"{RESTYLE_KEY}:{tab.uid}"
    strength = min(0.95, max(0.1, float(strength)))

    def run() -> Any:
        from ..service import jobs as svc_jobs

        ids: dict[int, str] = {}
        for index, png in pictures.items():
            result = svc_jobs.create_job(
                ctx.svc,
                kind="text",
                prompt=prompt,
                negative="photo, text, watermark, frame, border",
                reference=png,
                init_image=True,
                init_strength=strength,
                output="reference",
                count=1,
                reference_prep=False,
            )
            job_id = str(result.get("id") or "") if isinstance(result, dict) else ""
            if not job_id and isinstance(result, dict):
                found = result.get("ids") or result.get("jobs") or []
                job_id = str(found[0]) if found else ""
            if not job_id:
                raise RuntimeError(f"frame {index} was not queued")
            ids[index] = job_id
        return ids

    if not ctx.submit(key, run):
        return False
    state.flourish_restyle_pending = pending
    ctx.toast(f"Restyling {len(frames)} keyframes of {phase}...")
    return True


def _png_bytes(plane: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(plane, dtype=np.uint8), "RGBA").save(buf, "PNG")
    return buf.getvalue()


def on_restyle_queued(ctx: Any, state: Any, done: Any) -> None:
    pending = state.flourish_restyle_pending
    if pending is None:
        return
    if done.error is not None or not isinstance(done.result, dict) or not done.result:
        state.flourish_restyle_pending = None
        ctx.toast(f"The restyle was not queued: {done.error or 'no jobs'}.", "warn")
        return
    pending["jobs"] = {int(k): str(v) for k, v in done.result.items()}


def poll_restyle(ctx: Any, state: Any, *, now: float) -> None:
    pending = state.flourish_restyle_pending
    if pending is None or not pending.get("jobs"):
        return
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + TEXTURE_POLL_S
    statuses: dict[int, str] = {}
    for index, job_id in pending["jobs"].items():
        try:
            job = ctx.svc.store.get(job_id)
        except Exception:  # noqa: BLE001 -- the store answers next tick
            return
        if job is None:
            state.flourish_restyle_pending = None
            return
        statuses[index] = str(job.get("status"))
        if statuses[index] not in ("queued", "running", "done"):
            state.flourish_restyle_pending = None
            ctx.toast(f"The restyle {statuses[index]}: {job.get('error') or 'no result'}.", "warn")
            return
    if any(s in ("queued", "running") for s in statuses.values()):
        return
    state.flourish_restyle_pending = None
    paths = {
        index: ctx.svc.job_dir(job_id) / "input.png" for index, job_id in pending["jobs"].items()
    }
    tab = _tab_by_uid(state, pending["tab_uid"])
    held = tab.doc.flourish_state(int(pending["group"])) if tab is not None else None
    recipe = held.recipe if held is not None else None
    size = tab.doc.size if tab is not None else None
    key = f"{RESTYLE_LAND_KEY}:{pending['tab_uid']}"
    ctx.submit(key, decode_restyle, pending, paths, recipe, size)


def decode_restyle(
    pending: dict[str, Any], paths: dict[int, Any], recipe: Any, size: tuple[int, int] | None
) -> dict[str, Any] | None:
    """Task thread: read every anchor, key it out, interpolate the span."""
    from PIL import Image

    from .inker.flourish import keyframes

    if recipe is None or size is None:
        return None
    anchors: dict[int, np.ndarray] = {}
    for index, path in paths.items():
        try:
            with Image.open(path) as im:
                im.load()
                picture = im.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        except OSError:
            return None
        pixels = np.asarray(picture, dtype=np.uint8).copy()
        if not (pixels[..., 3] < 255).any():
            pixels = key_out_black(pixels)  # the model painted on an opaque ground
        anchors[int(index)] = pixels
    first, last = pending["span"]
    field = keyframes.field_from_recipe(recipe)
    planes = keyframes.interpolate(anchors, int(first), int(last), field)
    cels = {int(first) + i: plane for i, plane in enumerate(planes)}
    return {"pending": pending, "cels": cels}


def land_restyle(ctx: Any, state: Any, done: Any) -> bool:
    result = done.result
    if done.error is not None or not isinstance(result, dict):
        ctx.toast("The restyled keyframes could not be read.", "warn")
        return False
    pending = result["pending"]
    tab = _tab_by_uid(state, pending["tab_uid"])
    if tab is None:
        return False
    group = int(pending["group"])
    if tab.doc.flourish_state(group) is None:
        ctx.toast("That effect was detached while its keyframes restyled.", "info")
        return False
    name = f"Restyled {pending['phase']}"
    tab.doc.insert_flourish_track(group, name, result["cels"])
    ctx.toast(
        f"{name}: {len(result['cels'])} frames from {len(pending['frames'])} keyframes.",
        "success",
    )
    return True

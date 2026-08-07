"""The handle every pane draws through.

One object holding the app's parts, passed down rather than imported: a pane
that reached for a module-level singleton could not be built twice (a compare
view, a test), and the frame loop would have no single place to see what a
frame touched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..service import derive as svc_derive
from ..service import files as svc_files
from . import dialogs
from .state import AppState

log = logging.getLogger(__name__)


def pixel_prefs(settings: Any) -> tuple[int, int, str | None, bool]:
    """The pixel-art (size, colours, palette, dither) preference, defensively
    coerced.

    The settings JSON is user-editable and every reader runs on the frame
    thread, so a bad value falls back to the default rather than raising --
    the same posture restore_form takes. Clamped against the literal choice
    tuples, which is also what keeps the size from ever composing a filename
    outside the MEDIA allowlist.
    """
    try:
        size = int(settings.get("pixel_size") or 128)
    except (TypeError, ValueError):
        size = 128
    if size not in svc_files.PIXEL_ARTIFACTS.values():
        size = 128
    try:
        colors = int(settings.get("pixel_colors") or 0)
    except (TypeError, ValueError):
        colors = 0
    if colors not in svc_files.PIXEL_COLOR_CHOICES:
        colors = 0
    # The palette is *not* validated against the directory here: this runs on
    # the frame thread and a directory listing is a syscall per frame. A name
    # whose file has since been deleted is refused by service.palettes on the
    # task thread, which is where the user can be told about it.
    palette = settings.get("pixel_palette") or None
    if not isinstance(palette, str) or not palette.strip():
        palette = None
    dither = bool(settings.get("pixel_dither"))
    return size, colors, palette, dither


def save_key(job_id: str, name: str) -> str:
    """The task key for "derive this artifact and ask where to put it"."""
    return f"save:{job_id}:{name}"


def derive_key(job_id: str, name: str) -> str:
    """The task key for "derive this artifact and leave it where it lands".

    A namespace of its own rather than a borrowed ``save:``, because the app
    claims a finished save *by prefix* and toasts "Saved to <result>"
    (``main._on_task_done``). A derivation returns the path inside the job
    directory and is never None, so a preview submitted under the save key told
    the user a file had been saved -- naming an internal path, with no dialog
    ever shown. Callers that offer both check both (``Ctx.artifact_busy``), so
    the two still cannot describe different files.
    """
    return f"derive:{job_id}:{name}"


@dataclass
class Ctx:
    svc: Any
    runtime: Any
    state: AppState
    cache: Any
    tasks: Any
    settings: Any
    viewer: Any = None
    textures: Any = None
    confirms: dialogs.ConfirmQueue = field(default_factory=dialogs.ConfirmQueue)
    prompts: dialogs.PromptQueue = field(default_factory=dialogs.PromptQueue)
    # Answers from doctor + the rig-template probe, read once at startup: none
    # of them can change while the app runs without a restart.
    rigging_available: bool = False
    rig_templates: list[dict[str, Any]] = field(default_factory=list)
    rig_default: str = ""
    export_dir: str | None = None
    # Set by the App, which owns the task keys these results come back on.
    load_presets: Any = lambda _template: None
    refresh_rig_data: Any = lambda: None
    # Clay's "send to 3D": an offscreen GL draw on the frame thread, which is
    # the App's business rather than a pane's. None until the App attaches it,
    # so a headless caller gets a clear refusal rather than a half-drawn frame.
    clay_send_to_3d: Any = None
    guidance: dict[str, Any] = field(default_factory=dict)
    sheet_options: dict[str, Any] = field(default_factory=dict)
    # The monitor scale sampled at startup, kept apart from tokens.SCALE so the
    # settings pane can multiply a user preference onto it without compounding
    # its own previous answer.
    dpi_scale: float = 1.0
    # The App's Layout, so the settings pane can reset pane sizes.
    layout: Any = None
    base_models: list[tuple[str, str]] = field(default_factory=list)
    style_loras: list[tuple[str, str]] = field(default_factory=list)

    # -- shorthands --------------------------------------------------------

    def toast(self, text: str, level: str = "info") -> None:
        self.state.toast(text, level)

    def job(self, job_id: str | None = None) -> dict[str, Any] | None:
        return self.cache.get(self.state.selected if job_id is None else job_id)

    def job_dir(self, job_id: str) -> Path:
        return self.svc.job_dir(job_id)

    def busy(self, key: str) -> bool:
        return self.tasks.is_busy(key)

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        return self.tasks.submit(key, fn, *args, tag=tag, **kwargs)

    def artifact_busy(self, job_id: str, name: str) -> bool:
        """Whether *either* half is working on this artifact.

        Both keys, always: a preview derivation and an export of one name run
        the same ``get_file`` under the same per-artifact lock, so a control
        that watched only its own key would offer a button that then blocked
        invisibly on the other one.
        """
        return self.busy(save_key(job_id, name)) or self.busy(derive_key(job_id, name))

    # -- common actions ----------------------------------------------------

    def save_artifact(self, job_id: str, name: str) -> None:
        """Derive the artifact if it does not exist, then ask where to put it.

        Both halves are off the frame thread and in that order: a save dialog
        that opens before the two-second export has run would make the user
        wait with a modal in front of them for no reason.
        """
        key = save_key(job_id, name)
        # Captured on the frame thread and closed over: the task thread never
        # touches Settings. For a pixel artifact this is what makes the export
        # button and the inspector preview produce the same file under the
        # same lock.
        pixel = (
            pixel_prefs(self.settings)
            if name in svc_files.PIXEL_ARTIFACTS
            else (0, None, None, False)
        )
        pixel_colors, palette, dither = pixel[1], pixel[2], pixel[3]

        def run() -> Path | None:
            source = svc_derive.get_file(
                self.svc,
                job_id,
                name,
                pixel_colors=pixel_colors,
                pixel_palette=palette,
                pixel_dither=dither,
            )
            dest = dialogs.save_file(f"Save {name}", f"{job_id}_{name}", dialogs.filters_for(name))
            if dest is None:
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
            return dest

        if not self.submit(key, run, tag={"name": name}):
            return
        self.toast(f"Preparing {name}...")

    def capture_thumbnail(self, job_id: str) -> None:
        """Store the current viewport as the job's card image.

        Rendered by the viewer rather than by a pipeline, for the same reason
        the browser did it: the viewport already has the model framed, so the
        snapshot is free, while a Blender render would need a place on the
        serial GPU queue for something purely cosmetic.
        """
        if self.viewer is None or not self.viewer.has_model:
            return
        try:
            data = self.viewer.thumbnail_png()
        except Exception:
            log.exception("could not capture a thumbnail")
            return
        self.submit(f"thumb:{job_id}", svc_files.save_thumbnail, self.svc, job_id, data)

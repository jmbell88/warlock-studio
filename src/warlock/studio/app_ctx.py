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
    guidance: dict[str, Any] = field(default_factory=dict)
    sheet_options: dict[str, Any] = field(default_factory=dict)
    base_models: list[tuple[str, str]] = field(default_factory=list)
    style_loras: list[tuple[str, str]] = field(default_factory=list)
    quit_requested: bool = False

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

    # -- common actions ----------------------------------------------------

    def save_artifact(self, job_id: str, name: str) -> None:
        """Derive the artifact if it does not exist, then ask where to put it.

        Both halves are off the frame thread and in that order: a save dialog
        that opens before the two-second export has run would make the user
        wait with a modal in front of them for no reason.
        """
        key = f"save:{job_id}:{name}"

        def run() -> Path | None:
            source = svc_derive.get_file(self.svc, job_id, name)
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

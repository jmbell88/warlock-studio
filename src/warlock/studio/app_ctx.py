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
from . import atomic, dialogs
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


def download_key(row_key: str) -> str:
    """The task key for "fetch this model's weights".

    Its own namespace, and the prefix is load-bearing: ``any_busy("download:")``
    is what disables every other row's button while one fetch is running, which
    is the whole of the app-Settings pane's concurrency policy.
    """
    return f"download:{row_key}"


def remove_key(row_key: str) -> str:
    """The task key for "delete this model's weights".

    Its own namespace beside ``download_key``, and the prefix is load-bearing
    in exactly the same way: the app-Settings pane disables every row's buttons
    while ``any_busy("download:")`` *or* ``any_busy("remove:")`` is true, and
    the App's task-done handler switches on the prefix to decide which toast to
    show after the same wholesale re-probe.
    """
    return f"remove:{row_key}"


def pack_key(key: str) -> str:
    """The task key for "install this dependency pack".

    Its own namespace beside ``download_key`` and for the same reason: the
    app-Settings pane disables every pack's button while ``any_busy("pack:")``
    is true -- two packs share distributions, and two children installing torch
    into one ``site-packages`` at once is the worst concurrency in the tree --
    and the App's task-done handler claims the result by this prefix.
    """
    return f"pack:{key}"


@dataclass
class Ctx:
    svc: Any
    runtime: Any
    state: AppState
    cache: Any
    tasks: Any
    settings: Any
    viewer: Any = None
    # Poser's own Viewer instance, attached by the App when the mode is first
    # entered. Separate from ``viewer`` on purpose: ``adopt_model`` there exits
    # pose mode unconditionally, so sharing one instance would let loading the
    # Poser preview silently discard unsaved inspector pose edits.
    poser_viewer: Any = None
    # Clay's live viewport, attached by the App when the mode's viewport is
    # first built and cleared again at teardown. The App owns the instance;
    # this is how clay_mode and the panes reach it -- the drag keyboard, the
    # axis views and the per-tab camera all read ``getattr(ctx, "clay_view",
    # None)``, and a field nothing assigned left every one of them silently
    # inert, because getattr-with-a-default is exactly the spelling that turns
    # a missing wire into None rather than an AttributeError.
    clay_view: Any = None
    # The status bar's resource sampler (``studio/resources.Sampler``), owned
    # by the App and attached here for the pane to read. Attached rather than
    # constructed: the CPU figure is a delta between calls, so one owner ticks
    # it and everything else reads the last reading -- two samplers would each
    # eat the other's interval.
    resources: Any = None
    textures: Any = None
    confirms: dialogs.ConfirmQueue = field(default_factory=dialogs.ConfirmQueue)
    prompts: dialogs.PromptQueue = field(default_factory=dialogs.PromptQueue)
    # Answers from doctor + the rig-template probe. Written at startup and
    # again when a later health poll first reports Blender working: the bpy
    # probe is deferred (C30), so the *startup* answer is routinely "not yet
    # known" rather than "no" -- ``App._on_task_done`` re-asks for the
    # templates on the poll that changes its mind. What is fixed for the life
    # of the process is whether bpy is installed, not whether this ctx has
    # heard about it yet.
    #
    # The *model* answers below are not fixed either, and used to say they
    # were. A download
    # started from the app-Settings pane makes weights appear while the app
    # runs, so ``base_models``/``style_loras``/``model_rows`` are recomputed
    # from a fresh ``doctor.run_checks`` when a fetch finishes -- the same
    # wholesale replacement the health poll already does to ``runtime.checks``.
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
    # The quit chain, for the same reason: the palette's Quit must go through
    # the App's guard -- painted pixels, then geometry, then a pose -- and not
    # straight to the exit, which is the one way out that loses work. None
    # until the App attaches it, so a headless caller does nothing rather than
    # quitting a process it does not own.
    ask_quit: Any = None
    # "Clear the canvas", for the same reason again: emptying the viewport is
    # only half the job -- the other half is pinning ``viewer.path`` so the
    # App's own ``_sync_viewer`` does not refill it on the next cache tick, and
    # that is the App's mechanism rather than a pane's. None until attached, so
    # a headless caller does nothing instead of leaving a half-cleared viewer.
    clear_viewport: Any = None
    guidance: dict[str, Any] = field(default_factory=dict)
    sheet_options: dict[str, Any] = field(default_factory=dict)
    # The monitor scale sampled at startup, kept apart from tokens.SCALE so the
    # settings pane can multiply a user preference onto it without compounding
    # its own previous answer.
    dpi_scale: float = 1.0
    # The App's Layout, so the settings pane can reset pane sizes.
    layout: Any = None
    # Named v2 workspace arrangements (widths, vertical shares and panes).
    layouts: Any = None
    base_models: list[tuple[str, str]] = field(default_factory=list)
    style_loras: list[tuple[str, str]] = field(default_factory=list)
    # Every downloadable registry row, with a real ``present`` flag rather than
    # the word "missing" inside a label -- see ``service.downloads.rows``.
    model_rows: list[dict[str, Any]] = field(default_factory=list)
    # Every dependency pack, with a real ``present`` flag and what installing
    # it would cost -- see ``service.packs.rows``. Refreshed wholesale beside
    # ``model_rows``, because an install changes what this interpreter can
    # import and every pack answer in the ctx is derived from that.
    pack_rows: list[dict[str, Any]] = field(default_factory=list)
    # Which rows the app-Settings pane has ticked. Frame-thread state and
    # deliberately not persisted: it is a selection for one action, and a
    # remembered one would offer to re-download something already on disk.
    model_picks: set[str] = field(default_factory=set)
    # The measured size of the model store, once the walk lands. ``None`` is
    # "not looked yet" and reads differently from an empty store.
    model_storage: dict[str, Any] | None = None
    # The label of the best base model this card can actually hold, or "" when
    # no VRAM plan has been resolved. A label rather than a key, and computed by
    # the App rather than by the pane: ``vram.recommended_base`` wants a Plan,
    # and a pane that reached for one would be a pane doing service work.
    recommended_base_label: str = ""
    # Installer onboarding. ``first_run`` is sampled exactly once while the
    # Ctx is built; the dict is likewise a startup snapshot so drawing the
    # modal never probes hardware, disk or the model store on the frame thread.
    first_run: bool = False
    first_run_info: dict[str, Any] = field(default_factory=dict)
    gpu_name: str = ""

    # -- shorthands --------------------------------------------------------

    def toast(
        self,
        text: str,
        level: str = "info",
        action: str | None = None,
        action_arg: str | None = None,
    ) -> None:
        self.state.toast(text, level, action, action_arg)

    def toast_once(
        self,
        text: str,
        level: str = "info",
        action: str | None = None,
        action_arg: str | None = None,
    ) -> bool:
        """:meth:`toast`'s coalescing twin, forwarded for the same reason: a
        caller holding a ctx should not have to know the state is where toasts
        live. -> whether one was raised."""
        return self.state.toast_once(text, level, action, action_arg)

    def open_log(self) -> None:
        """Hand ``warlock.log`` to whatever the user reads text files in.

        One owner, because callers want the same behavior: the log command and
        the toast for a failure whose message defers to the log. Silent
        when there is no log yet -- it is written on the first line logged, so
        its absence means there is nothing to show rather than a fault.

        Deliberately outside the kill-on-close job, and deliberately not a
        subprocess spawn: ``startfile`` hands the path to the shell, which
        opens it in the user's own editor, and that editor is not ours to kill
        when Warlock closes. Which is also why the every-spawn-is-in-the-job
        scan does not see this line.
        """
        import os

        log_path = Path(self.runtime.config.data_dir) / "warlock.log"
        if log_path.exists():
            self.submit("open-log", os.startfile, str(log_path))

    def job(self, job_id: str | None = None) -> dict[str, Any] | None:
        return self.cache.get(self.state.selected if job_id is None else job_id)

    def job_dir(self, job_id: str) -> Path:
        return self.svc.job_dir(job_id)

    def busy(self, key: str) -> bool:
        return self.tasks.is_busy(key)

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        return self.tasks.submit(key, fn, *args, tag=tag, **kwargs)

    def progress(self, key: str) -> dict[str, Any] | None:
        """How far a running task has got, or None. Same shape as
        ``Runtime.progress``, so a pane draws either with one call."""
        return self.tasks.progress(key)

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
            atomic.write_bytes(dest, source.read_bytes())
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
            # Only the GL readback happens here; the PNG encode -- the slow
            # half -- runs on the task thread with the save (D41).
            image = self.viewer.screenshot()
        except Exception:
            # Toasted, not merely logged (E47): the user pressed a button and
            # the only evidence of the outcome is the card's picture, which
            # already looked like whatever it looked like a moment ago. Points
            # at the log because a GL readback failing is not something the
            # message could usefully summarise.
            log.exception("could not capture a thumbnail")
            self.toast("Could not capture the thumbnail.", "error", "log")
            return

        def run() -> Any:
            import io

            buf = io.BytesIO()
            image.convert("RGB").save(buf, "PNG")
            return svc_files.save_thumbnail(self.svc, job_id, buf.getvalue())

        self.submit(f"thumb:{job_id}", run)

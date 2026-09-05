"""The selected asset: what it is, what it cost, and what you can take away.

The downloads section is the interesting part. Each artifact has three states
-- ready, derivable, impossible -- and the difference matters: a greyed "FBX"
with "needs Blender" under it is information, while a missing one is a mystery.
*Which* artifacts appear is a function of the job: a reference was being shown
the mesh grid, so all eight of its buttons were permanently greyed, which reads
as a broken asset rather than as a 2D one.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ... import followups
from ...service import derive as svc_derive
from ...service import files as svc_files
from ...service import jobs as svc_jobs
from ...service import system as svc_system
from .. import controls, create_stages, fonts, forms, quality, theme, verbs, widgets
from ..app_ctx import derive_key, pixel_prefs
from ..manual import render as manual_render
from ..tokens import sp
from . import (
    candidates_panel,
    overlay,
    pose_panel,
    retarget_panel,
    sheet_panel,
    sprite_panel,
    stamps,
    texture_panel,
)

#: How tall the trellis log readout is, in design pixels. Enough lines to see
#: a stack trace's shape without the pane becoming a log viewer.
_LOG_HEIGHT = 160.0

# Matches the library card's thumbnail, so the two read as the same kind of
# object rather than as two different image widgets. Design px: every use of it
# goes through ``sp()``, as the library's own does -- a raw literal here was
# the one size in the pane that ignored the UI scale.
THUMB_SIZE = 96

# How tall a single reference image may get. The reference now fills the pane's
# width, and at full width a portrait is taller than the pane -- three thumbs is
# enough to read the composition by and leaves the sections under it reachable
# without a scroll to the bottom.
REFERENCE_MAX_THUMBS = 3


# What the inspector shows at each Create stage. **The stage rail is the tab
# bar now** (the UI redesign, wave 5): a stage switch and a tab click were two
# controls doing one thing, sitting a foot apart, and the tab strip could
# disagree with the rail about which step you were on. So in Create the
# inspector carries no tabs at all and is *stage-scoped evidence* -- what this
# step produced, beside the column that produced it.
#
# The tab bar survives in every other host (the Library's full-window
# inspector), because there is no rail there to drive it.
_STAGE_SECTIONS: dict[str, tuple[str, ...]] = {
    "reference": (
        "_edit_actions",
        "_lineage",
        "_settings",
        "_reference",
        "_pixel",
        "sprites",
        "_seam",
    ),
    "mesh": (
        "_edit_actions",
        "_lineage",
        "_settings",
        "_reference",
        "_pixel",
        "_seam",
        "_quality",
        "_verdict",
    ),
    # The rig's *products*: how it was weighted, how many joints, and the
    # deformation sheet. The decision and the button are in the stage's own
    # column; retarget and retexture join them because both act on the mesh a
    # rig was fitted to and both are read *after* looking at it.
    "rig": ("_weighting", "_bones", "_deform_qa", "retarget", "remesh", "retexture"),
    # Posing produces poses, and the thing made *of* poses is a sprite sheet.
    "pose": ("sheet",),
    # The grid itself is the stage's *column*; this side answers the question
    # a person actually has in front of it -- is this the right asset, and
    # what was it made from.
    "export": ("_reference", "_settings"),
}


def _stage_body(ctx: Any, job: Any) -> None:
    from . import remesh_panel, retarget_panel, sheet_panel, sprite_panel, texture_panel

    named = {
        "_edit_actions": lambda: _edit_actions(ctx, job),
        "_lineage": lambda: _lineage(ctx, job),
        "_settings": lambda: _settings(ctx, job),
        "_reference": lambda: _reference(ctx, job),
        "_pixel": lambda: _pixel(ctx, job),
        "_seam": lambda: _seam(ctx, job),
        "_quality": lambda: _quality(ctx, job),
        "_verdict": lambda: _verdict(ctx, job),
        "_weighting": lambda: _weighting(ctx, job),
        "_bones": lambda: _bones(ctx, job),
        "_deform_qa": lambda: _deform_qa(ctx, job),
        "sprites": lambda: sprite_panel.draw(ctx, job),
        "retarget": lambda: retarget_panel.draw(ctx, job),
        "remesh": lambda: remesh_panel.draw(ctx, job),
        "retexture": lambda: texture_panel.draw(ctx, job),
        "sheet": lambda: sheet_panel.draw(ctx, job),
    }
    for name in _STAGE_SECTIONS.get(ctx.state.create_stage, ()):
        named[name]()


def draw(ctx: Any) -> None:
    from .. import icons

    if create_stages.at(ctx.state, "mesh"):
        # Above the header, and above the "select an asset" empty state, on
        # purpose: an undecided candidate group is a question being asked, and
        # its rows are hidden from the library -- so this is the only way back
        # to them, and it has to be visible whatever is (or is not) selected.
        candidates_panel.draw(ctx)
    job = ctx.job()
    if job is None:
        widgets.empty_state(icons.BOX, "Select an asset.", "Its details and exports live here.")
        return

    # Identity stays above whatever follows: whichever stage or tab is open,
    # the header answers "what am I looking at". Everything below it moved from
    # one nine-header scroll column into three tabs, because reaching the
    # sprite sheet on a rigged mesh meant scrolling past the whole pose editor;
    # in Create the stage rail does that job instead.
    _header(ctx, job)
    manual_render.help_button(ctx, "inspector")
    # Said out loud, because the two halves of the screen are not in step yet.
    # The inspector follows the selection immediately; the viewport parses off
    # the frame thread and adopts when the parse lands. That gap is bounded and
    # short now that a selection change triggers the sync -- but it exists, and
    # an unlabelled stale mesh beside a new inspector is what makes an export or
    # a compare decision untrustworthy (UX-03). Naming the asset is the point:
    # "loading" alone would not say *which*.
    viewer = getattr(ctx, "viewer", None)
    if viewer is not None and getattr(viewer, "pending", None) is not None:
        widgets.muted(f"Loading {job.get('name') or job['id']}...")
    _meta(ctx, job)
    if job.get("status") == "error":
        _error(ctx, job)
    _followup_failures(job)

    if create_stages.in_create(ctx.state):
        _stage_body(ctx, job)
        return
    # Every other host (the Library's full-window inspector) keeps the tabs,
    # because there is no stage rail there to drive them. Which tabs is a
    # question about the *asset* rather than about the mode it is being looked
    # at from -- it used to be "are we in 3D", which offered Rig & Pose for a
    # reference selected there and withheld it for a mesh selected anywhere
    # else.
    tabs: list[tuple[str, Any]] = [("Details", lambda: _details_tab(ctx, job))]
    if "model.glb" in (job.get("files") or []):
        # One ampersand. Dear ImGui has no mnemonic escape -- the doubling is a
        # habit from toolkits that do, and it drew "Rig && Pose" on the tab for
        # as long as nobody looked at the Library's inspector in a screenshot.
        tabs.append(("Rig & Pose", lambda: _rig_tab(ctx, job)))
    tabs.append(("Export", lambda: downloads(ctx, job)))
    widgets.tab_bar("inspector-tabs", tabs)


def can_edit_in_clay(job: Any) -> bool:
    """Whether "Open in Clay" belongs on this job.

    The library's overflow menu gates on ``model.glb`` being listed and this is
    the same question asked from the same cached row -- no filesystem call,
    because the inspector asks it every frame. ``build.wblk`` is deliberately
    not consulted: it is never listed, and ``clay_mode.edit_asset_in_clay``
    prefers it on the task thread where a stat is free.
    """
    if not isinstance(job, dict):
        return False
    return job.get("status") == "done" and "model.glb" in (job.get("files") or [])


def offers_inker(ctx: Any, job: Any) -> bool:
    """Whether *this* pane is the one that offers Open in Inker.

    Not simply ``can_edit_job``: the 2D viewport toolbar offers the same button
    on the same job, so in 2D both were on screen at once -- two controls calling
    one function on one object, one of them right above the pixels and one in the
    sidebar. The toolbar wins where it exists, because adjacency to the image is
    the whole of its advantage; this pane covers the rest, which is where the
    button is not otherwise reachable at all (a reference selected in 3D, whose
    toolbar deliberately hides it because the thing on screen is a mesh).

    Defined as the *complement* of ``overlay.offers_inker`` rather than as its
    own reading of the mode, so the two can never both be true and never both be
    false: a second spelling of "is it 2D" is exactly how one action grew two
    buttons in the first place.
    """
    from .. import inker_mode

    if not inker_mode.can_edit_job(ctx, job):
        return False
    return not overlay.offers_inker(ctx, job)


def _edit_actions(ctx: Any, job: Any) -> None:
    """Take this asset somewhere it can be edited, from where it was made.

    Both halves already existed and were reachable only from the library's
    overflow menu -- which is the wrong place for the moment they are wanted,
    the one straight after a generation finishes with the result still on
    screen. Wiring only: the Inker gate and the Clay open are the functions
    those two modes own, called verbatim, so the confirm Clay puts in front of
    a 200k-triangle import still fires here.
    """
    from .. import clay_mode, icons, inker_mode

    params = (
        job.get("params")
        if isinstance(job, dict) and isinstance(job.get("params"), dict)
        else {}
    )
    if params.get("asset_intent") == "tileset" and "input.png" in (job.get("files") or []):
        from .. import packwright_mode, plotter_mode

        if controls.button(verbs.add_to("plotter")):
            plotter_mode.use_as_tileset(ctx, job)
        if controls.button(verbs.add_to("packwright")):
            packwright_mode.add_job_source(ctx, job)
        widgets.hint_text("Use the generated grid as a map tileset or atlas source.")

    if offers_inker(ctx, job):
        if controls.button(f"{icons.BRUSH} {verbs.open_in('inker')}"):
            inker_mode.open_job_reference(ctx, job)
        widgets.hint_text("Paint over the reference; saving updates this asset.")
    # Independent, not an else: the two gates are disjoint today (a reference
    # has no mesh and a mesh has no editable reference), and an else would hide
    # one of them without saying so if that ever stopped being true.
    if can_edit_in_clay(job):
        if controls.button(f"{icons.BOX} {verbs.open_in('clay')}"):
            clay_mode.edit_asset_in_clay(ctx, job)
        widgets.hint_text("Opens the authored document when there is one, else the mesh.")

    from .. import troupe_mode

    if troupe_mode.can_send_to_troupe(ctx, job):
        if controls.button(f"{icons.PERSON_STANDING} {verbs.send_to('troupe')}"):
            troupe_mode.send_to_troupe(ctx, job)
        # The library's menu item has carried this since it existed and the
        # button beside the mesh had nothing at all -- so the one surface where
        # the user is *looking at the mesh* was the one that did not say a
        # humanoid rig is required, or that an unrigged mesh buys minutes of
        # CPU behind a button not called "Rig".
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Render a character sheet from this mesh. Needs a humanoid rig"
                " -- Troupe's clip library has no clips for the other six"
                " skeletons -- and rigs it first if it is not rigged yet."
            )
        # The hint has to say the rig happens: for an unrigged mesh this is
        # minutes of CPU behind a button that is not called "Rig", and a user
        # who is not told will read the quiet as a hang.
        widgets.hint_text(
            "Render a character sheet -- rigging the mesh first if it is not rigged."
            if "rig.glb" in (job.get("files") or [])
            else "Rigs the mesh as a humanoid, then renders a character sheet."
        )


def _label_of(job: Any) -> str:
    return str(job.get("name") or job.get("prompt") or job.get("id") or "asset")


def _lineage(ctx: Any, job: Any) -> None:
    """Where this asset came from, and what came of it.

    The lineage has been threaded through ``parent_id`` since promotion existed
    and nothing on screen said so: a reference and the three meshes made from
    it were four unrelated cards in a list sorted by date. These are the
    edges, walkable -- and through ``create_stages.go``, so following one lands
    at the stage that asset belongs to rather than leaving the rail pointing at
    a step this row has not reached.
    """
    from .. import create_stages, icons

    came_from = create_stages.parent(ctx, job)
    made = create_stages.promotions(ctx, job)
    if came_from is None and not made:
        return
    widgets.section("Lineage")
    width = imgui.get_content_region_avail().x
    if came_from is not None:
        label = widgets.fit_text(
            f"{icons.ARROW_LEFT} From {_label_of(came_from)}", width - sp(24)
        )
        if widgets.ghost_button(label, (-1, 0), tooltip=_label_of(came_from)):
            create_stages.go(ctx, "reference", select=came_from["id"])
    for mesh in made:
        label = widgets.fit_text(f"{icons.BOX} {_label_of(mesh)}", width - sp(24))
        if widgets.ghost_button(f"{label}##lineage/{mesh['id']}", (-1, 0), tooltip=_label_of(mesh)):
            create_stages.go(ctx, "mesh", select=mesh["id"])
    if made:
        widgets.hint_text(
            "One mesh from this reference." if len(made) == 1 else f"{len(made)} meshes from this."
        )


def _details_tab(ctx: Any, job: Any) -> None:
    _edit_actions(ctx, job)
    _settings(ctx, job)
    _reference(ctx, job)
    _pixel(ctx, job)
    sprite_panel.draw(ctx, job)
    _seam(ctx, job)
    if create_stages.at(ctx.state, "mesh"):
        _quality(ctx, job)
        _verdict(ctx, job)


def _rig_tab(ctx: Any, job: Any) -> None:
    _weighting(ctx, job)
    _bones(ctx, job)
    _deform_qa(ctx, job)
    retarget_panel.draw(ctx, job)
    texture_panel.draw(ctx, job)
    pose_panel.draw(ctx, job)
    sheet_panel.draw(ctx, job)


def rig_meta(ctx: Any, job: Any) -> dict[str, Any] | None:
    """The selected asset's ``rig.json``, or None -- cached per job by mtime.

    Why the file at all, when 0d put ``weighting`` on a row: it put it on the
    **rig** job's row, and a rig writes its artifacts into the *source* job's
    directory because the rig belongs to the mesh. The Rig & Pose tab opens on
    that source job -- it is the asset a user selects -- so the row on screen
    has no ``weighting`` of its own and the tab showed nothing at all. The
    alternative was making ``queue._rig`` a second writer onto the model job's
    row, which is refused: that row has one owner and ``set_params`` is
    last-write-wins.

    Why the racily-clean rule: a re-rig landing after this read but inside the
    stamped mtime's own tick would be invisible **forever**, because every later
    comparison keeps matching. :mod:`.stamps` is where that rule and its
    constant live -- one judgement about one class of hazard, shared with the
    three other stamped caches in the panes -- and the ordering it depends on is
    visible here: the clock is read (by ``storable``) *after* the file.
    """
    job_id = job.get("id")
    if not job_id:
        return None
    job_dir = ctx.svc.job_dir(job_id)
    # A missing directory stamps as None, which is an answer of its own: it is
    # "no rig", and it changes the moment a directory appears.
    stamp = stamps.stamp_ns(job_dir)
    cache = ctx.state.rig_meta_cache
    hit = cache.get(job_id)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    from ... import rigging

    meta = rigging.read_rig(job_dir)
    if stamps.storable(stamp):
        cache[job_id] = (stamp, meta)
    return meta


def rig_weighting(ctx: Any, job: Any) -> tuple[int, str] | None:
    """The weighting line for the asset on screen, from wherever it is recorded.

    The row first, then the rig file. The order matters and only in one
    direction: a *rig* job selected in the library carries the params and has no
    ``rig.json`` of its own, while the model job it rigged has the file and no
    params -- so the two sources are disjoint in practice, and preferring the
    row means the answer is always about the row the user is looking at.
    """
    verdict = weighting_verdict(job.get("params") or {})
    if verdict is not None:
        return verdict
    return weighting_verdict(rig_meta(ctx, job) or {})


def weighting_verdict(params: Any) -> tuple[int, str] | None:
    """The one line that says how the mesh is bound to its skeleton, or None.

    A pure function for the reason ``seam_verdict`` is one: the wording is the
    feature and it has to be assertable without a GL context. Envelope is a
    *degraded* outcome -- the bone-heat solve failed and a capsule falloff
    stood in for it -- so it is named as one rather than reported as a second
    kind of success, which is how a silently degraded rig reached the user
    looking exactly like a good one.
    """
    if not isinstance(params, dict):
        return None
    weighting = params.get("weighting")
    if not weighting:
        return None
    if weighting == "envelope":
        return (theme.WARN, "weighting: envelope - needs review")
    return (theme.MUTED, f"weighting: {weighting}")


def rig_bones(ctx: Any, job: Any) -> int | None:
    """How many bones the rig on screen has, from wherever it is recorded.

    ``rig_weighting``'s two sources and its order, for the same reason: the
    **rig** job's row carries ``bone_count`` (``queue._rig`` records it beside
    ``weighting``) and the **model** job the user selects carries ``rig.json``,
    so the two are disjoint in practice and preferring the row means the answer
    is always about the row on screen. ``queue.py`` wrote that param and nothing
    read it, which is how a rig could report *how* it was bound and never *what*
    it was bound to.
    """
    count = bone_count_of(job.get("params") or {})
    if count is not None:
        return count
    return bone_count_of(rig_meta(ctx, job) or {})


def bone_count_of(source: Any) -> int | None:
    """The bone count a rig record carries, in either of its two spellings.

    A pure function for the reason ``weighting_verdict`` is one. The two
    spellings are not a mistake to be normalised away: the row records the
    *number* because a card must not open a file per asset to draw a badge,
    while ``rig.json`` carries the joints themselves because a pose editor needs
    their names and positions. Read defensively throughout -- rows written
    before ``bone_count`` existed simply do not carry it, and a bool is an int
    as far as ``isinstance`` is concerned.
    """
    if not isinstance(source, dict):
        return None
    raw = source.get("bone_count")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    bones = source.get("bones")
    if isinstance(bones, list) and bones:
        return len(bones)
    return None


def _bones(ctx: Any, job: Any) -> None:
    count = rig_bones(ctx, job)
    if count is None:
        return
    widgets.muted(f"{count} bone" if count == 1 else f"{count} bones")
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "How many joints the skeleton template fitted onto this mesh. It is "
            "the template's count, not a measurement of the mesh."
        )


def _weighting(ctx: Any, job: Any) -> None:
    source: Any = job.get("params") or {}
    verdict = weighting_verdict(source)
    if verdict is None:
        # The ordinary case, not the exception: the tab opens on the mesh, and
        # the mesh's rig records itself in a file beside it.
        source = rig_meta(ctx, job) or {}
        verdict = weighting_verdict(source)
    if verdict is None:
        return
    widgets.text_colored(*verdict)
    reason = source.get("weighting_reason")
    if reason and imgui.is_item_hovered():
        imgui.set_tooltip(str(reason))
    if verdict[0] == theme.WARN:
        widgets.hint_text("Blender's bone-heat solve did not take; hover for why.")


def deform_qa_path(svc: Any, job: Any, *, cache: dict | None = None) -> Any:
    """The deformation review sheet for this asset, or None.

    Gated on the **sidecar** rather than on its own existence, which is
    ``rig.glb``'s rule and the same hazard: ``sheetlib`` writes the PNG and then
    the JSON that says what its cells are, so a sheet without its sidecar is a
    render in progress. Deliberately not in the inspector's download grid --
    that grid is a hardcoded list of mesh artifacts, which is exactly why
    ``rig_qa.png`` reached ``job["files"]`` and appeared nowhere.

    ``cache`` is an optional ``{job_id: (stamp, path)}`` the caller owns, and it
    is ``attach_files``' argument at two stats rather than ten: both files are
    answered by *existence*, and a file appearing or going moves the job
    directory's own mtime, so one stat on the directory stands in for both --
    which matters because this ran twice per frame for as long as the Rig & Pose
    tab was open. Optional rather than mandatory because the gate is also a
    plain question a test or a one-off caller asks once, and a cache is only
    worth its subtleties on the frame thread. Those subtleties are
    :mod:`.stamps`': the answer is remembered only once the directory's mtime is
    safely in the past, or a QA sheet landing inside that mtime's own tick would
    be invisible for the life of the process.
    """
    job_id = job.get("id")
    if not job_id:
        return None
    job_dir = svc.job_dir(job_id)
    if cache is None:
        return _deform_qa_probe(job_dir)
    stamp = stamps.stamp_ns(job_dir)
    hit = cache.get(job_id)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    # No directory is no sheet, and there is nothing to stat inside one.
    found = None if stamp is None else _deform_qa_probe(job_dir)
    if stamps.storable(stamp):
        cache[job_id] = (stamp, found)
    return found


def _deform_qa_probe(job_dir: Any) -> Any:
    sheet = job_dir / "rig_qa.png"
    if not (job_dir / "rig_qa.json").exists() or not sheet.exists():
        return None
    return sheet


def _deform_qa(ctx: Any, job: Any) -> None:
    """The QA battery as a thumbnail, beside the weighting line it explains.

    **Nothing scores it, and the wording must not imply anything does.** The
    battery renders a rig in poses nobody would ship an asset in (squat, arms
    overhead, elbow and knee at 90 degrees, a torso twist) precisely so a person
    can see where the skin tears. A metric here would be a claim about
    deformation quality that nothing in the repo has earned yet.
    """
    path = deform_qa_path(ctx.svc, job, cache=ctx.state.preview.setdefault("deform_qa", {}))
    if path is None:
        return
    widgets.field_label("Deformation review")
    texture = ctx.textures.get(f"rigqa:{job['id']}", path)
    if texture is None:
        return
    side = min(imgui.get_content_region_avail().x, float(sp(THUMB_SIZE * 2)))
    imgui.image(widgets.texture_ref(texture), (side, side))
    if imgui.is_item_hovered():
        imgui.set_tooltip(str(path))
    widgets.hint_text("The rig in five test poses. Look for tearing; nothing scores it.")


# --- pieces -----------------------------------------------------------------

# Field text the queue refused, keyed by submit key. Frame thread only, and
# cleared the moment a submit is accepted, so it holds at most one entry per
# field being typed into right now. See ``_write_field``.
_unsent: dict[str, str] = {}


def _write_field(ctx: Any, key: str, job_id: str, payload: Any, typed: str, current: str) -> None:
    """Submit an edited field, retrying until the queue takes it.

    ``TaskRunner.submit`` refuses a key already in flight and nothing re-arms
    it, and ``widgets.input_text`` hands back the *passed* value on a frame
    with no keystroke -- so the frame after a refusal compared the stale cached
    job against itself, found no difference, and the keystrokes typed during
    the write were simply gone. Fast typing committed a prefix.

    The unsent text is held here instead and retried every frame until a submit
    is accepted, which is also what clears it.
    """
    if typed != current:
        _unsent[key] = typed
    pending = _unsent.get(key)
    if pending is None:
        return
    if ctx.submit(key, svc_jobs.update_job, ctx.svc, job_id, payload(pending)):
        _unsent.pop(key, None)


def _header(ctx: Any, job: Any) -> None:
    job_id = job["id"]
    # The name *is* this pane's header (UX.md Phase 2), so it is drawn at
    # heading size rather than at body size like the tag field under it. It
    # stays an editable field rather than becoming a label with a pencil beside
    # it: renaming an asset is one of the two things anybody does up here, and
    # the input is what already carries the "Name" hint when there is none.
    with fonts.heading(imgui):
        name = widgets.input_text("##name", job.get("name") or "", max_length=120, hint="Name")
    _write_field(
        ctx, f"name:{job_id}", job_id, lambda v: {"name": v}, name, job.get("name") or ""
    )
    tags = widgets.input_text("##tags", job.get("tags") or "", max_length=400, hint="tags, comma")
    _write_field(
        ctx,
        f"tags:{job_id}",
        job_id,
        lambda v: {"tags": v.split(",")},
        tags,
        job.get("tags") or "",
    )
    widgets.status_pill(job["status"])
    imgui.same_line()
    favourite = bool(job.get("favorite"))
    if controls.small_button("Unfavourite" if favourite else "Favourite"):
        ctx.submit(
            f"fav:{job_id}", svc_jobs.update_job, ctx.svc, job_id, {"favorite": not favourite}
        )


def _meta(ctx: Any, job: Any) -> None:
    # The same badge-and-"yesterday" a Home tile wears; the id, the internal
    # kind/stage names and the exact timestamp are one foldout down.
    widgets.asset_summary(job)
    if ctx.viewer is not None and ctx.viewer.has_model and create_stages.at(ctx.state, "mesh"):
        stats = ctx.viewer.stats()
        if stats:
            size = stats["size"]
            # Counted off what is actually loaded, not read from mesh_report:
            # the viewer may be showing rig.glb or a baked pose, and the number
            # under the model should describe the model under it.
            widgets.muted(
                f"{stats['triangles']:,} tris - {stats['vertices']:,} verts - "
                f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m"
            )


def _error(ctx: Any, job: Any) -> None:
    message = job.get("error") or "It failed."
    widgets.text_colored(theme.ERR, message)
    # Copy (N114). The one thing anybody does with a failure message is send it
    # somewhere, and the text is drawn rather than in an input, so it could not
    # be selected -- retyping a friendly() sentence and its job id was the whole
    # of "report this". The id goes with it because the message alone is not
    # enough to find the job's error.log or its row.
    if controls.small_button("Copy error"):
        imgui.set_clipboard_text(f"job {job['id']}\n{message}")
    imgui.same_line()
    if not imgui.tree_node("Details"):
        return
    key = "trellis-log"
    if widgets.disabled_button("Read the trellis log", not ctx.busy(key)):
        ctx.submit(key, svc_system.trellis_log, ctx.svc)
    log_text = ctx.state.preview.get("trellis_log") if ctx.state.preview else None
    if log_text:
        controls.input_text_multiline(
            "##log",
            log_text[-4000:],
            (-1, sp(_LOG_HEIGHT)),
            imgui.InputTextFlags_.word_wrap.value,
        )
    if "error.log" in (job.get("files") or []) and controls.button("Save error.log..."):
        ctx.save_artifact(job["id"], "error.log")
    imgui.tree_pop()


def followup_failure_lines(job: Any) -> list[tuple[str, str]]:
    """User-facing follow-up failures for one cached row."""
    params = job.get("params") if isinstance(job, dict) else None
    stage = job.get("stage") if isinstance(job, dict) else None
    return [
        (f"{item['label']} was not queued.", item["message"])
        for item in followups.records(params, stage)
    ]


def _followup_failures(job: Any) -> None:
    for title, message in followup_failure_lines(job):
        widgets.text_colored(theme.WARN, title)
        widgets.muted(message)


def _settings(ctx: Any, job: Any) -> None:
    params = job.get("params") or {}
    if not params:
        return
    if not widgets.header("Generation settings", default_open=False):
        return
    rows = [
        ("seed", params.get("seed")),
        ("reference seed", params.get("reference_seed")),
        ("mesh seed", params.get("mesh_seed")),
        ("platform", params.get("platform")),
        ("resolution", params.get("resolution")),
        ("size (m)", params.get("size_m")),
        ("model", params.get("base_model")),
        ("style LoRA", params.get("style_lora")),
        ("appearance ref", params.get("ip_adapter")),
        ("appearance strength", params.get("ip_scale")),
        ("structure", params.get("control")),
        ("structure strength", params.get("control_scale")),
        ("start image", "yes" if params.get("init_image") else None),
        ("start strength", params.get("init_strength")),
        ("background", params.get("bg_removal")),
        ("profile", params.get("profile")),
    ]
    with forms.Form("generation-settings") as form_ui:
        for label, value in rows:
            if value in (None, ""):
                continue
            form_ui.readonly(label.replace(" ", "_"), label, value)
    composed = params.get("composed_prompt")
    if composed and imgui.tree_node("Prompt sent"):
        imgui.text_wrapped(str(composed))
        imgui.tree_pop()


def _attempt_verdict(attempt: Any) -> str:
    """What one reroll attempt is to be called in the attempts line.

    Three words rather than two, because "not measured" is a third state and
    not a flavour of "refused": the composition report can fail to run at all
    (queue.py records the attempt with measured False, following
    reference.unmeasured), and calling that a refusal would put a verdict in
    the UI that nothing ever reached. Written against .get throughout, so an
    attempt row from before the third state existed still renders.
    """
    if not isinstance(attempt, dict):
        return "?"
    if attempt.get("measured") is False:
        return "not measured"
    return "kept" if attempt.get("ok") else "refused"


def reference_max_height() -> float:
    """The tallest a reference image may be drawn, in scaled pixels."""
    return sp(THUMB_SIZE * REFERENCE_MAX_THUMBS)


def reference_fit(size: tuple[int, int], avail: float) -> tuple[float, float]:
    """The box a reference image is drawn in: fill the width, keep the shape.

    The same fit-to-box arithmetic ``main._draw_reference`` uses for the
    viewport, and pure for the reason ``pixel_scale`` is -- it decides what the
    user sees and it should not need a GL context to assert. It used to be a
    hardcoded ``(THUMB_SIZE, THUMB_SIZE)``, which ignored the aspect ratio
    outright: a 2:1 sheet was drawn squashed into a square, and the image never
    reacted to the width of the pane it was in.

    The height cap is the other half. Filling the width of a 300 px sidebar
    with a 1:4 portrait is a thousand-pixel image, and there are up to four of
    them, which pushes every section under Reference off the screen.
    """
    width = float(max(size[0], 1))
    height = float(max(size[1], 1))
    scale = min(max(avail, 1.0) / width, reference_max_height() / height)
    return (width * scale, height * scale)


def _reference(ctx: Any, job: Any) -> None:
    """What the mesh engine was actually handed, and what it made of it.

    The three images answer the question a bad mesh always raises -- was it
    the reconstruction, or was it what we sent? -- which nothing in the UI
    could answer before: trellis was streamed input.png verbatim and no
    measurement of it was kept.
    """
    params = job.get("params") or {}
    report = params.get("reference_report")
    files = set(job.get("files") or [])
    shown = [n for n in ("input.png", "ref.png", "reference.png", "control.png") if n in files]
    if not isinstance(report, dict) and len(shown) <= 1:
        return
    # Open by default now that the images are legible: the section is only
    # drawn at all when there is a composition report or more than one image,
    # so it is never a collapsed heading over nothing, and the question it
    # answers -- was it the reconstruction or was it what we sent? -- is the
    # first one a bad mesh raises. It carries no persist_key, so this changes
    # the default rather than overriding anything a user has closed.
    if not widgets.header("Reference"):
        return

    if isinstance(report, dict):
        for reason in report.get("reasons") or []:
            widgets.text_colored(theme.ERR, reason)
        for warning in report.get("warnings") or []:
            widgets.text_colored(theme.WARN, warning)
        if report.get("occupancy"):
            widgets.muted(f"the subject fills {float(report['occupancy']) * 100:.0f}% of the frame")
        if report.get("measured") is False:
            widgets.muted("not measured")
        elif report.get("normalised"):
            widgets.muted("recentred and rescaled before upload")

    attempts = params.get("reference_attempts")
    if isinstance(attempts, list) and len(attempts) > 1:
        # Only ever present on a job the reroll actually ran for, so the line
        # doubles as the answer to "why is this not the seed I asked for".
        widgets.muted(
            f"redrawn {len(attempts) - 1} time(s): "
            + "; ".join(f"seed {a.get('seed')} {_attempt_verdict(a)}" for a in attempts)
        )

    hint = params.get("control_hint")
    if isinstance(hint, dict) and hint.get("edge_fraction") is not None:
        # The number that answers "my silhouette lock did nothing": near zero
        # means the detector found no structure, not that it was ignored.
        widgets.muted(f"{hint['kind']} hint: {float(hint['edge_fraction']) * 100:.1f}% edges")

    if ctx.textures is None:
        return
    for name in shown:
        # Keyed by job *and* file: the cache keys on (id, mtime), so sharing
        # the bare job id across four images would show whichever one was
        # decoded first for all of them.
        texture = ctx.textures.get(f"{job['id']}:{name}", ctx.job_dir(job["id"]) / name)
        if texture is None:
            continue
        widgets.muted(name)
        # Measured per image rather than once: the label above each one costs a
        # line, and a scrollbar appearing partway down the list changes what is
        # left for the images under it.
        imgui.image(
            widgets.texture_ref(texture),
            reference_fit(texture.size, imgui.get_content_region_avail().x),
        )


def seam_verdict(report: Any) -> tuple[int, str] | None:
    """The one line that says whether the tile actually tiles, or None.

    A pure function of the stored report, so the wording is assertable without
    a GL context -- and so the number and the word never come from two places.
    The ratio is "how much worse than the picture's own grain the wrap seam is",
    which is why the threshold is quoted alongside it: a bare 3.4 is a number
    nobody can calibrate against, while "3.4, over 2.00" says which way it went
    and by how far.
    """
    if not isinstance(report, dict):
        return None
    # Which number to quote is decided by the row, never by today's constant.
    # A row carrying ``metric == "dominance"`` was judged by the seam-against-
    # interior-maximum statistic (2026-08-30); a row with no ``metric`` at all
    # predates that change and was judged by the edge-against-mean-grain ratio,
    # and describing it in the new vocabulary would be stating a number the row
    # does not contain. docs/measurements/2026-08-30-seam-dominance.md R8.
    if report.get("metric") == "dominance":
        worst = float(report.get("dominance") or 0.0)
        wording = "seam/worst join"
    else:
        worst = float(report.get("worst") or 0.0)
        wording = "edge/grain"
    if report.get("seamless"):
        # "likely", not "seamless", and the hedge is measured rather than
        # cautious. It was introduced because
        # ``docs/measurements/2026-08-09-seam-threshold-cfg.md`` found the
        # edge-energy ratio does not separate seamless tiles from seamed ones
        # on a CFG base, and it *stays* under the dominance statistic for a
        # narrower and better-measured reason:
        # ``2026-08-30-seam-dominance.md`` records dominance missing 4 of 44
        # visibly seamed control units -- a picture whose interior already
        # contains a step as hard as its seam ties at 1.0 and passes. The
        # false-alarm direction is what improved (0 of 72 confirmed-seamless
        # tiles flagged, against the ratio's 18); the miss direction is why
        # this still says "likely" and still sends the reader to the wrapped
        # view below, which is the real check.
        return (theme.OK, f"likely seamless -- {wording} {worst:.2f}; check the wrap")
    threshold = float(report.get("threshold") or 0.0)
    return (theme.WARN, f"visible seam -- {wording} {worst:.2f}, over {threshold:.2f}")


def _seam(ctx: Any, job: Any) -> None:
    """Whether the tile actually tiles, and what it looks like wrapped.

    The ratio alone is a number nobody can calibrate against by eye, which is
    what the wrapped view is for: rolled by half, what was the wrap seam runs
    through the middle of the frame, where a discontinuity is obvious. It is a
    derived artifact like every other export rather than something this pane
    writes -- the roll reads and rewrites a full-size PNG, which is emphatically
    not frame-thread work, and going through ``derive.get_file`` means the
    button here and the Export tab's produce the same file under the same lock.
    """
    verdict = seam_verdict((job.get("params") or {}).get("seam_report"))
    if verdict is None:
        return
    if not widgets.header("Seam"):
        return
    report = job["params"]["seam_report"]
    widgets.text_colored(*verdict)
    widgets.muted(
        f"left/right {float(report.get('horizontal') or 0.0):.2f} - "
        f"top/bottom {float(report.get('vertical') or 0.0):.2f}"
    )
    if ctx.textures is None:
        return
    job_id = job["id"]
    # The cache answers "is it on disk yet" as a side effect of answering "give
    # me the texture": a missing file is a None, which is exactly the state in
    # which the button should be offered.
    texture = ctx.textures.get(f"{job_id}:wrap", ctx.job_dir(job_id) / "wrap_preview.png")
    key = f"wrap:{job_id}"
    if texture is None:
        busy = ctx.busy(key)
        if busy:
            # The same idiom the Export grid uses beside a busy artifact:
            # rolling a 1024-square PNG is several hundred milliseconds, and a
            # button that only greys out reads as one that did nothing.
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button("Show it wrapped", not busy):
            ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, "wrap_preview.png")
        return
    imgui.image(widgets.texture_ref(texture), (sp(THUMB_SIZE * 2), sp(THUMB_SIZE * 2)))


def pixel_provenance(manifest: Any, name: str) -> str | None:
    """What actually cut the pixel artifact on disk, or None without an entry.

    Read off the manifest rather than off the knob, because the two can
    disagree -- switching sizes shows a file derived under an older palette
    setting -- and the line exists to say what the file *is*. Pure for the
    reason seam_verdict is: the wording is assertable without a GL context.
    """
    if not isinstance(manifest, dict):
        return None
    entry = (manifest.get("artifacts") or {}).get(name)
    if not isinstance(entry, dict):
        return None
    size = svc_files.PIXEL_ARTIFACTS.get(name)
    palette = entry.get("palette")
    if entry.get("palette_file"):
        colours = str(entry["palette_file"])
        if entry.get("dither"):
            colours += " (dithered)"
    else:
        colours = f"{palette} colours" if palette else "full colour"
    parts = [f"{size} px", colours]
    grid = entry.get("grid")
    if isinstance(grid, dict) and grid.get("scale"):
        # Only said when it happened: an ordinary render has no lattice, and
        # claiming one would be a claim about the generator, not the file.
        parts.append(f"{grid['scale']}px source grid")
    from ...pipelines import pixel as pixelmod

    sentence = pixelmod.verdict(entry.get("qa") or {})
    if sentence:
        parts.append(sentence)
    return " - ".join(parts)


def pixel_scale(size: tuple[int, int], avail: int) -> int:
    """The integer factor a pixel artifact is drawn at.

    A whole multiple or nothing: a fractional factor samples some source
    pixels twice and others once, which reads as banding -- the thing NEAREST
    sampling exists to avoid.
    """
    return max(1, avail // max(size[0], size[1], 1))


def palette_names(ctx: Any, door: Any = None) -> list[str]:
    """The palette directory's stems, listed once per version of the directory.

    One stat per frame rather than one walk: this is drawn on the frame thread,
    and a directory that changes when a user drops a file into it moves its own
    mtime, so a file added while the app runs still appears on the next frame.
    A missing directory is an empty list, never an error -- palettes are opt-in
    and there is nothing to report about not having any.

    "Moves its own mtime" is true only to a 15.6 ms tick, which is why the
    remembering goes through :mod:`.stamps`: dropping a palette in is precisely
    a write that can land inside the stamp's own tick, and an unguarded stamp
    would then hide that palette for the life of the process -- from the one
    control whose whole purpose is to offer it.

    ``door`` is the service call to ask -- ``sprites.sprite_palettes`` or
    ``tilesheets.tile_sheet_palettes`` for the two Sheet arms, whose forms name
    their own door rather than reaching past it. Every one of them is
    ``palettes.available`` over one directory, which is why they share **one**
    cache slot and why the default is that function: the question "which
    palettes are installed" has one answer, and a second remembering of it
    would be a second thing to invalidate when a file lands.

    Public because it is the only listing of that directory on the frame
    thread. ``settings_2d`` draws the same picker for the two sheet arms, and a
    second copy of the stamp rule above is exactly the copy that comes back as
    a palette nobody can select until the app restarts.
    """
    from ...service import palettes as svc_palettes

    directory = ctx.svc.config.palette_dir
    key = stamps.stamp_ns(directory)
    if key is None:
        ctx.state.palettes = None
        return []
    cached = ctx.state.palettes
    if cached is not None and cached[0] == key:
        return cached[1]
    names = (
        svc_palettes.available(ctx.svc.config) if door is None else list(door(ctx.svc))
    )
    # After the listing, never beside the stat: that ordering is what makes the
    # stored stamp's tick provably older than the read it describes.
    if stamps.storable(key):
        ctx.state.palettes = (key, names)
    return names


def _pixel_handoffs(
    ctx: Any, job_id: str, name: str, size: int, prefs: dict[str, Any]
) -> None:
    """Where the pixels go from here: the Inker, or a file.

    Neither is a new *capability* -- ``downloads`` already exports these
    artifacts and the Inker already opens documents. What both add is **place
    and specificity**, which is ``_edit_actions``' own argument: this is where
    the user is looking at the pixels, and it acts on the size selected in
    this section, which the download grid cannot express.

    The three preferences are read on the frame thread and handed over, so the
    preview above, the export and the open all describe one file.
    """
    from .. import inker_mode

    if controls.button(verbs.open_in("inker")):
        inker_mode.open_pixel_artifact(
            ctx,
            job_id,
            name,
            title=f"{job_id[:8]} pixels {size}",
            pixel_colors=int(prefs["pixel_colors"]),
            pixel_palette=prefs["pixel_palette"],
            pixel_dither=bool(prefs["pixel_dither"]),
        )
    imgui.same_line()
    if controls.button("Export as PNG"):
        ctx.save_artifact(job_id, name)


def _pixel(ctx: Any, job: Any) -> None:
    """The pixel-art cutout as it will actually export, crisp on screen.

    The knob is derive-time state, not a job param: input.png stays at full
    resolution (a promotion feeds trellis the same pixels it always did) and
    changing it re-derives the artifact, exactly as a hand edit does. The
    re-derive goes through ``derive.get_file`` under its own ``derive:`` key
    and reports busy on either key, so this preview, its spinner and the
    export button can never describe different files -- while the "Saved to"
    toast the app hangs off every ``save:`` key stays with the one control
    that actually opens a dialog.
    """
    if job.get("stage") != "reference" or job.get("status") != "done":
        return
    if "input.png" not in set(job.get("files") or []):
        return
    if not widgets.header("Pixel art", default_open=False):
        return
    job_id = job["id"]
    size, colors, palette_name, dither = pixel_prefs(ctx.settings)
    submit_kwargs = {
        "pixel_colors": colors,
        "pixel_palette": palette_name,
        "pixel_dither": dither,
    }

    sizes = sorted(svc_files.PIXEL_ARTIFACTS.values())
    chosen = int(widgets.labeled_combo("Size", str(size), [(str(s), f"{s} px") for s in sizes]))
    if chosen != size:
        ctx.settings.set("pixel_size", chosen)
    # The name is always one of the three literals: ``chosen`` comes from
    # PIXEL_ARTIFACTS itself, so the MEDIA allowlist property holds.
    name = f"pixel_{chosen}.png"
    key = derive_key(job_id, name)

    picked = int(
        widgets.labeled_combo(
            "Colours",
            str(colors),
            [(str(n), "Off" if n == 0 else f"{n} colours") for n in svc_files.PIXEL_COLOR_CHOICES],
        )
    )
    if picked != colors:
        ctx.settings.set("pixel_colors", picked)
        colors = submit_kwargs["pixel_colors"] = picked
        # Re-derive the size on screen straight away; the other sizes catch up
        # when they are next asked for, which is what _pixel_current is for.
        ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)

    names = palette_names(ctx)
    if names:
        options = [("", "None")] + [(n, n) for n in names]
        chosen_palette = widgets.labeled_combo("Palette", palette_name or "", options)
        chosen_palette = chosen_palette or None
        changed = chosen_palette != palette_name
        if changed:
            ctx.settings.set("pixel_palette", chosen_palette or "")
            palette_name = submit_kwargs["pixel_palette"] = chosen_palette
        if palette_name:
            toggled, dither = controls.checkbox("Dither", dither)
            if toggled:
                ctx.settings.set("pixel_dither", dither)
                submit_kwargs["pixel_dither"] = dither
                changed = True
        if changed:
            ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)

    # Before the texture early-out below, deliberately: an export that only
    # appears once "Preview pixels" has been pressed is a discoverability trap,
    # and both doors derive the artifact themselves if it is absent.
    _pixel_handoffs(ctx, job_id, name, chosen, submit_kwargs)

    if ctx.textures is None:
        return
    busy = ctx.artifact_busy(job_id, name)
    texture = ctx.textures.get(f"{job_id}:{name}", ctx.job_dir(job_id) / name, nearest=True)
    if texture is None:
        if busy:
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button("Preview pixels", not busy):
            ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)
        return
    factor = pixel_scale(texture.size, int(sp(THUMB_SIZE * 2)))
    imgui.image(widgets.texture_ref(texture), (texture.size[0] * factor, texture.size[1] * factor))
    if busy:
        widgets.spinner()
        return
    manifest = _manifest(ctx, job_id)
    line = pixel_provenance(manifest, name)
    if line:
        widgets.muted(line)
    entry = ((manifest or {}).get("artifacts") or {}).get(name)
    stale = isinstance(entry, dict) and (
        (entry.get("palette_file") or None) != palette_name
        or bool(entry.get("dither")) != dither
        or (
            # The median-cut cap only means anything while no palette file is
            # in play; with one, the entry records ``palette: None`` by design.
            not palette_name and entry.get("palette") != (colors or None)
        )
    )
    # The file on screen was cut under a different palette setting -- switching
    # sizes surfaces one derived before the knob changed.
    if stale and controls.button("Rebuild with these colours"):
        ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)


def watertight_value(report: Any) -> bool | None:
    """Whether the mesh is sealed, read the way the badge reads it.

    ``meshreport`` measures watertightness on a *welded* copy, because it loads
    with ``process=False`` so the UV and material checks see the unwelded mesh
    -- and every xatlas seam split is then a boundary edge, which makes the raw
    flag say "not watertight" about every textured mesh ever reconstructed.
    ``widgets.quality_badge`` was moved onto ``welded_watertight`` and this
    panel was not, so it printed ``watertight: False`` directly under a badge
    saying ``watertight``.

    The fallback is what keeps rows recorded before the change readable, and it
    is the identical expression ``vectors.py`` uses.
    """
    if not isinstance(report, dict):
        return None
    value = report.get("welded_watertight", report.get("watertight"))
    return value if isinstance(value, bool) else None


def watertight_line(report: Any) -> str | None:
    """The one line the panel prints about watertightness, or None.

    The label names which measurement it is: a welded reading says so, and a
    row that only ever carried the raw flag must not be relabelled as one --
    that would be the same overclaim in reverse, and it is exactly the mistake
    the badge made with ``mesh_audit``.
    """
    value = watertight_value(report)
    if value is None:
        return None
    if isinstance(report, dict) and isinstance(report.get("welded_watertight"), bool):
        return f"watertight (welded): {value}"
    return f"watertight: {value}"


def _quality(ctx: Any, job: Any) -> None:
    params = job.get("params") or {}
    report = params.get("mesh_report")
    audit = params.get("mesh_audit")
    if not (report or audit):
        return
    if not widgets.header("Mesh quality"):
        return
    if isinstance(report, dict):
        widgets.quality_badge(job)
        for reason in report.get("reasons") or []:
            widgets.muted(f"- {reason}")
        # Only the report may use the word watertight: the audit is a
        # silhouette check and never proved it.
        for label, key in (
            ("triangles", "triangles"),
            ("materials", "materials"),
        ):
            if key in report:
                widgets.muted(f"{label}: {report[key]}")
        sealed = watertight_line(report)
        if sealed:
            widgets.muted(sealed)
        if "grounded" in report:
            widgets.muted(f"pivot at feet: {report['grounded']}")
    if isinstance(audit, dict) and audit.get("worst") is not None:
        ratio = float(audit["worst"])
        widgets.muted(f"visible openings: {ratio * 100:.1f}%")
        # P120, and only where the number actually misleads. A *high* reading
        # is real evidence of a hole and needs no caveat; a low one is what a
        # solid slab measures, which is the dominant failure mode.
        #
        # ``hole_worst`` is **corpus-dependent and is not a quality scale** --
        # the invariant, not a fact about one number. It read AUC 0.115 against
        # reject over the 2026-08-07 rogue corpus (backwards rather than merely
        # weak; 48 of 81 rejected meshes measured exactly 0.0), and it
        # re-baselined to 0.756 once the matte was fixed -- see
        # ``docs/measurements/2026-08-09-rebaseline.md``, whose "`hole_worst`
        # is no longer inverted, and the matte is why" section is the authority.
        # The caveat below stays through both readings, because what it warns
        # about is reading *any* single figure as a grade.
        #
        # A second muted line rather than a tooltip on the first: this function
        # is called with ``widgets.muted`` stubbed and no imgui frame at all
        # (``tests/test_quality_badge.py``), so raw imgui here is an access
        # violation rather than a failure -- and a caveat nobody hovers is a
        # caveat nobody reads anyway.
        caveat = quality.caveat_for(ratio)
        if caveat:
            widgets.muted(caveat)

    attempts = params.get("mesh_attempts")
    if isinstance(attempts, list):
        # Only ever present on a job the remesh actually ran for, so the line
        # doubles as the answer to "why is this not the mesh seed I asked for".
        # ``quality.remesh_line`` owns the wording: this used to say "kept the
        # best of 12.3%, 4.5%", which is ``hole_worst`` presented as a ranking
        # -- the one thing INVARIANTS says it may never be.
        for line in quality.remesh_line(attempts):
            widgets.muted(line)


# --- verdicts ---------------------------------------------------------------
#
# The same graded verdict a sweep unit gets in Review, on any finished asset --
# which is the point: daily use and a deliberate sweep feed one findings pool,
# and the vast majority of assets anyone ever looks at were never part of a
# sweep. The two halves that are easy to get wrong (which asset the staged tags
# belong to, and what a recorded verdict does) are plain functions so they can be
# asserted without a GL context.


def staged_tags(state: Any, job_id: str) -> list[str]:
    """The tags staged for this asset, and ``[]`` for any other.

    Answering per job id is what makes the drop automatic: nothing has to notice
    a selection change, because a different id simply has nothing staged. The
    two fields are read as a pair for exactly that reason -- a bare list would
    ride onto whatever was selected next.
    """
    if state.inspector_tags_job != job_id:
        return []
    return state.inspector_tags


def toggle_tag(state: Any, job_id: str, tag: str) -> None:
    """Stage or unstage one tag against this asset.

    The toggle rule itself is ``review_mode.toggled`` -- including its refusal
    to stage a tag the vocabulary does not carry, which would otherwise make
    ``record_verdict`` refuse the whole verdict and lose a grade to a typo. Only
    the "which asset is this staged against" half lives here, because that is
    the part the inspector genuinely does differently.
    """
    from .. import review_mode

    if state.inspector_tags_job != job_id:
        state.inspector_tags_job = job_id
        state.inspector_tags = []
    state.inspector_tags = review_mode.toggled(state.inspector_tags, tag)


def clear_tags(state: Any) -> None:
    state.inspector_tags_job = None
    state.inspector_tags = []


def record_verdict(ctx: Any, job_id: str, grade: int, tags: tuple[str, ...] = ()) -> bool:
    """File a graded verdict and queue the findings recompute. -> whether it
    landed.

    Inline on the frame thread for the reason Review's is: one INSERT under the
    store's lock. The recompute that follows reads every verdict and writes a
    file, so it is a task.
    """
    from ...service import verdicts as svc_verdicts
    from ...service.errors import ServiceError
    from .. import review_mode

    try:
        svc_verdicts.record_verdict(ctx.svc, job_id, grade=grade, reasons=tags)
    except (ServiceError, OSError):
        ctx.toast("Could not record that verdict.", "error")
        return False
    clear_tags(ctx.state)
    # Through Review's own request, not a second submit under a copy of its
    # key: two spellings of one task key are two things to keep in step.
    review_mode.refresh_findings(ctx)
    ctx.toast(f"Recorded: {review_mode.grade_text(grade)}.")
    return True


def _verdict(ctx: Any, job: Any) -> None:
    if job.get("status") != "done" or job.get("stage") != "model":
        return
    if not widgets.header("Was this any good?", default_open=False):
        return
    job_id = job["id"]
    widgets.muted("Feeds the same findings a sweep does.")
    pending = staged_tags(ctx.state, job_id)

    grade = widgets.grade_buttons(f"inspector-{job_id}", True)
    if grade is not None:
        record_verdict(ctx, job_id, grade, tuple(pending))
    widgets.muted_wrapped("+5 ships as-is, +3 usable, -5 unusable. Tags are optional.")

    tag = widgets.tag_toggles(f"inspector-{job_id}", pending, True)
    if tag is not None:
        toggle_tag(ctx.state, job_id, tag)


def downloads(ctx: Any, job: Any) -> None:
    """The Export stage's grid of artifacts, two columns wide.

    Public since the UI redesign, wave 5.4, because it is drawn from two places
    now: the Export stage's own column, and the inspector's Export *tab* in
    every host that still has tabs. One function, because "what you can take
    away from this asset" has exactly one right answer and a second grid would
    be a second place for an artifact to be forgotten.

    A blocked artifact keeps its button and explains itself in a tooltip --
    the old layout wrote the reason as an indented line under each of eight
    disabled buttons, which for a plain reference job was a column of noise.
    """
    from .. import icons

    widgets.section("Downloads")
    job_id = job["id"]
    files = set(job.get("files") or [])
    # Both notes under the grid are about cutouts -- which matte cut them, and
    # what each one came out as -- so they belong to the stage that has any. A
    # tile's manifest holds the recipe and no artifact entries at all.
    cutouts = job.get("stage") == "reference"
    if not imgui.begin_table("downloads", 2):
        return
    for name, label in widgets.artifacts_for(job):
        # job["files"] is the sanctioned answer; a raw exists() check here used
        # to re-enable buttons the service would then refuse.
        ready = name in files
        blocked = _why_blocked(ctx, job, name, ready, _derivable(job, files, name))
        # Either key: the pixel panel can have a derivation of this very name
        # in flight, and it takes the same per-artifact lock this button would.
        busy = ctx.artifact_busy(job_id, name)
        imgui.table_next_column()
        if busy:
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button(
            f"{icons.DOWNLOAD} {label}##{name}", not blocked and not busy, (-1, 0)
        ):
            ctx.save_artifact(job_id, name)
        if blocked and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(blocked)
    imgui.end_table()
    if cutouts:
        # Read once and handed to both: the two notes answer halves of the same
        # question, and asking for the manifest twice in a frame would make the
        # cheap-stat property depend on how many readers there happen to be.
        manifest = _manifest(ctx, job_id)
        _matte_note(ctx, manifest)
        _manifest_summary(manifest)


def _derivable(job: Any, files: set[str], name: str) -> bool:
    """Whether this job could produce ``name`` if the button were pressed.

    A 2D export comes from the reference's own pixels, so it is gated on those
    and not on a ``model.glb`` a reference will never have -- which is the old
    gate, and which greyed out every one of them. But it *is* gated on the job
    having finished: the mesh half got that for free, because model.glb is not
    listed until the worker is done with it, whereas ``derivable_2d`` answers a
    question about the name alone and would light six buttons on a queued job
    whose only outcome is an error toast. The conditions are deliberately the
    same three ``service.files.ready`` applies, restated here in terms of the
    listing rather than of the disk, because the frame thread may not stat --
    ``ready`` takes a job dir and would be a stat per artifact per frame. The
    two are pinned equal across the whole stage x status x files matrix by
    ``test_the_pane_agrees_with_the_service_about_every_artifact``, which is
    what stops the restatement drifting.

    Which names each image stage may ask for is deliberately *not* restated: a
    reference has the cutouts and a tile has the wrapped view, and that split is
    a fact about the artifacts rather than about the disk, so it is read
    straight off ``derivable_2d`` at no cost.
    """
    stage = job.get("stage")
    if stage in ("reference", "tile"):
        return (
            job.get("status") == "done"
            and "input.png" in files
            and svc_derive.derivable_2d(name, stage)
        )
    return "model.glb" in files and svc_derive.derivable(name)


def _why_blocked(ctx: Any, job: Any, name: str, ready: bool, derivable: bool) -> str | None:
    if ready:
        return None
    if name == "model.fbx" and not ctx.rigging_available:
        return "needs Blender"
    if derivable:
        return None
    if job.get("status") in ("queued", "running"):
        # Correctly disabled, and it used to be misleadingly explained: every
        # export on an unfinished job read "not available for this asset",
        # which says the asset cannot have the file rather than that it has not
        # got it yet. An errored job keeps the other wording, because there
        # "not yet" would promise something that is not coming.
        return "not finished yet"
    return "not available for this asset"


def _matte_note(ctx: Any, manifest: Any) -> None:
    """Why the cutouts will look the way they do, before any of them exist.

    Not a reason a button is blocked, which is why it sits under the grid
    rather than in ``_why_blocked``: the exports always work. The question a
    user has when the edges come out ragged is whether that is the model or the
    fallback, and nothing in the UI could answer it -- the doctor row says the
    same thing in a place nobody opens mid-export.

    Deliberately only for the case where nothing has been derived yet. Once
    artifacts exist, each manifest entry records the matte that actually cut
    it, and ``_manifest_summary`` says so per artifact -- which is the truthful
    answer where this one is merely the current one: an icon cut by the corner
    fill before the weights were installed would, on this note alone, claim
    nothing at all the moment they were.
    """
    from ...pipelines import matting

    if manifest is not None and (manifest.get("artifacts") or {}):
        return
    if matting.available(ctx.svc.config):
        return
    widgets.muted(
        "Cutouts use the corner fill -- edges are rougher than the matting "
        "model's. See the matting row under Settings for the one-time download."
    )


def _manifest_summary(manifest: Any) -> None:
    """The pivot, the alpha QA and the matte, read off the manifest as written.

    Read from the file rather than recomputed: the manifest is the thing an
    importer will consume, so showing anything else here would let the two
    disagree about the asset. That is also why the matte is reported per
    artifact -- ``derive`` records which one cut each file, and a file on disk
    was cut by whatever was installed when it was made, not by what is
    installed now.

    The hand-edited caveat travels with the recipe for the reason
    ``derive._write_manifest`` records it: the recipe hash names a seed and a
    model, which after a hand edit is no longer the whole story of the pixels.
    """
    if manifest is None:
        return
    if manifest.get("hand_edited"):
        widgets.muted("hand-edited -- the recipe describes the generated image, not these pixels")
    for name, entry in sorted((manifest.get("artifacts") or {}).items()):
        if not isinstance(entry, dict):
            continue
        bits = [name]
        if entry.get("pivot"):
            bits.append(f"pivot {entry['pivot'][0]:.0f},{entry['pivot'][1]:.0f}")
        alpha = entry.get("alpha") or {}
        if alpha.get("islands", 0) > 1:
            bits.append(f"{alpha['islands']} separate pieces")
        if entry.get("matte") == "flood":
            bits.append("corner fill")
        widgets.muted(" - ".join(bits))


def _manifest(ctx: Any, job_id: str) -> dict[str, Any] | None:
    """The job's parsed manifest.json, or None if there isn't a readable one.

    A stat every frame and a parse only when the file has changed. That split
    is the whole point: the export tab redraws sixty times a second, a
    derivation running on the TaskRunner rewrites the manifest underneath it,
    and re-reading on an mtime is how ``ThumbnailCache`` already reconciles
    those two facts. The slot lives on ``AppState`` rather than in a module
    global so the pane itself stays stateless.

    A failure to parse is cached too, as ``(key, None)``. "Unreadable" is an
    answer about that version of the file exactly as a parsed dict is, and a
    truncated or hand-mangled manifest is the one state that would otherwise be
    read and parsed *every* frame, for as long as it stayed mangled -- which is
    the per-frame read this cache exists to remove. A missing file needs no
    sentinel: the stat fails, so nothing is read either way.

    The remembering goes through :mod:`.stamps` for the reason ``rig_meta``'s
    does. "A derivation rewrites the manifest underneath it" is exactly a write
    that can land inside the stamped mtime's own 15.6 ms tick, and an unguarded
    stamp keeps matching it forever -- so the export tab would go on describing
    the artifacts of the derivation before last.
    """
    import json

    path = ctx.job_dir(job_id) / "manifest.json"
    mtime = stamps.stamp_ns(path)
    if mtime is None:
        return None
    key = (job_id, mtime)
    cached = ctx.state.manifest
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        manifest = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        manifest = None
    if not isinstance(manifest, dict):
        manifest = None
    if stamps.storable(mtime):
        ctx.state.manifest = (key, manifest)
    return manifest

"""Review's panes: the labelling grid, the sweep list, the form, the verdict.

A **mixin on** :class:`~.main.App`, which is this repository's idiom for a body
of drawing that belongs to the shell -- ``ClayView`` is assembled the same way
from ``CacheOps``/``BoundsOps``/``PickOps``/``OverlayOps``/``DragOps``, and
``Document`` from six more. So ``self`` here is the App and every method's body
is unchanged; what moved is nine hundred lines out of a six-and-a-half-thousand
line module.

Lifted out of ``studio/main`` on 2026-09-04 (T7 of the 2026-09-02 review), and
last of the behavioural work deliberately: the review's own suggested order puts
the extractions after everything they touch is pinned, so the move is code
motion over tested behaviour rather than a rewrite with a rewrite's risk.

**Not a pane module in the ``panes/`` sense**: it draws inside the host window
through ``layout.pane``/``guard`` like the rest of ``_build_ui``, and it reaches
``self.viewer``, ``self.layout`` and ``self.app_ctx``. The ``panes/`` package is
for surfaces that take a context and nothing else.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import tokens

log = logging.getLogger(__name__)


class ReviewPanes:
    """Review's drawing, mixed into :class:`~.main.App`.

    The four shell names this reaches -- ``REVIEW_MESH_KEY``, the two label
    tables and the two column helpers -- are imported *inside* the methods that
    use them. ``main`` imports this module to build the class, so a module-scope
    import back would be a cycle; the lazy one is the same shape ``main``'s own
    ~150 function-local imports have and for a related reason (see its
    docstring).
    """

    def _review_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses:

            [ review-runs  ]              [ review-verdict ]
            [ review-units ]  the mesh    [ the reference  ]

        The centre borrows the *shared* asset viewer rather than a second one:
        one GL context, one framebuffer, and a sweep unit's model.glb is an
        ordinary GLB. Leaving Review needs no cleanup because ``_sync_viewer``
        compares ``viewer.path`` against what the selection implies and reloads
        the moment 3D is on screen again.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import review_mode
        from .main import _column_boundary, _split_column

        ctx = self.app_ctx
        state = review_mode.ensure(ctx)
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")

        _split_column(
            ctx,
            lay,
            split_id="review-runs",
            handle_length=left_w,
            width=left_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=(
                "review-runs",
                layout_mod.PaneRole.SIDEBAR,
                lambda _ctx: self._review_runs(ctx, state, review_mode),
            ),
            bottom=(
                "review-units",
                layout_mod.PaneRole.SIDEBAR,
                lambda _ctx: self._review_units(state, review_mode),
            ),
        )

        _column_boundary(self.layouts, "review", "left")
        width = layout_mod.centre_width()
        # The labelling grid replaces the viewport rather than sitting beside it:
        # a mesh on screen under a question about a *picture* is the mismatch that
        # files an accept about the wrong artifact. It also scrolls, so it must
        # not inherit the viewport's no-scroll flag.
        labelling = state.labels is not None
        flags = 0 if labelling else imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "review-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                if labelling:
                    self._review_labels(ctx, state, review_mode)
                else:
                    self._review_viewport(state, review_mode, width)

        _column_boundary(self.layouts, "review", "right")
        with layout_mod.pane(
            "review-verdict",
            (right_w, 0),
            layout_mod.PaneRole.INSPECTOR,
            edge=layout_mod.PaneEdge.LEFT,
        ) as visible:
            if visible:
                if labelling:
                    self._review_label_panel(ctx, state, review_mode)
                else:
                    self._review_verdict(ctx, state, review_mode)

    # How wide a labelling cell is, in design px. ``tokens.THUMB_CELL``: the
    # same picture-you-judge-by that Home's Resume grid draws, which had its
    # own answer four pixels away from this one.
    _LABEL_CELL = tokens.THUMB_CELL

    def _review_labels(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """The labelling grid: images, two keys, no reason step.

        **One thumbnail upload per frame.** ``review_mode.next_thumbnail`` hands
        back at most one row per call, and the rest draw a placeholder until
        their turn comes -- ``viewer/sheet.StripRender``'s rule at a larger scale,
        because a synchronous upload per cell over a hundred cells is a freeze
        measured in seconds rather than frames.
        """
        from imgui_bundle import imgui

        from . import icons, theme, widgets
        from .main import _LABEL_QUESTIONS, _LABEL_TITLES
        from .tokens import sp

        labels = state.labels
        widgets.section(_LABEL_TITLES.get(labels.stage, labels.stage))
        widgets.hint_text(_LABEL_QUESTIONS.get(labels.stage, ""))
        if labels.loading:
            widgets.muted("Reading...")
            return
        if not labels.rows:
            widgets.empty_state(
                icons.CHECK,
                "Nothing left to label",
                "Every image has an answer for this question.",
            )
            return

        # Exactly one upload admitted per frame, claimed before the loop so which
        # cell gets it does not depend on where the scroll happens to be.
        review_mode.next_thumbnail(labels)
        side = float(sp(self._LABEL_CELL))
        per_row = max(int(imgui.get_content_region_avail().x // (side + sp(8))), 1)
        for i, row in enumerate(labels.rows):
            if i % per_row:
                imgui.same_line()
            imgui.begin_group()
            texture = None
            # ``ctx.textures`` is None until a GL context exists (app_ctx
            # defaults it), which is the state a headless or pre-init draw is
            # in -- every pane guards it and these three Review sites did not.
            if i < labels.uploaded and ctx.textures is not None:
                texture = ctx.textures.get(review_mode.cache_id_for_label(row), row["image"])
            if texture is not None:
                imgui.image(widgets.texture_ref(texture), (side, side))
            else:
                # A placeholder rather than nothing: the grid must not reflow as
                # the uploads land, or a click lands on a cell that moved.
                imgui.dummy((side, side))
            if imgui.is_item_clicked():
                labels.index = i
            mark = {"accept": icons.CHECK, "reject": icons.X}.get(row["verdict"] or "", "")
            colour = theme.ACCENT if i == labels.index else theme.MUTED
            widgets.text_colored(colour, f"{mark} {i + 1}")
            imgui.end_group()

    def _review_label_panel(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """What is being labelled, and what the probe knows so far."""
        from imgui_bundle import imgui

        from . import controls, widgets

        labels = state.labels
        row = review_mode.current_label(state)
        widgets.section("Label")
        if row is None:
            widgets.muted("Nothing selected.")
        else:
            widgets.muted(str(row["prompt"])[:120])
            if row.get("status") == "error":
                # The most informative negatives in the corpus, and worth saying
                # so: this image was refused at the composition gate.
                widgets.hint_text("This job was refused; the picture is still judgeable.")
            texture = (
                None
                if ctx.textures is None
                else ctx.textures.get(review_mode.cache_id_for_label(row), row["image"])
            )
            if texture is not None:
                side = min(imgui.get_content_region_avail().x, 220.0)
                imgui.image(widgets.texture_ref(texture), (side, side))
        imgui.separator()
        # One sentence for the three of them: they share a gate, and three
        # spellings of "there is nothing on screen to judge" would read as
        # three different problems. The ``_VIEWPORT_WHY`` pattern.
        no_row = "There is nothing left to label in this pass."
        if widgets.primary_button("Good (A)", enabled=row is not None):
            review_mode.record_label(ctx, "accept")
        imgui.same_line()
        if widgets.disabled_button("Bad (R)", row is not None, reason=no_row):
            review_mode.record_label(ctx, "reject")
        imgui.same_line()
        if widgets.disabled_button("Skip (S)", row is not None, reason=no_row):
            review_mode.advance_labels(labels)
        if controls.button("Done", role=controls.ButtonRole.GHOST):
            review_mode.close_labels(ctx)

        # The snapshot the listing task read, kept current by ``record_label``.
        # Never a live ``judge.status`` call: that is a whole-table scan plus a
        # stat, and this panel draws every frame.
        status = labels.status
        imgui.separator()
        widgets.section("The probe")
        answered = sum(1 for r in labels.rows if r["verdict"])
        widgets.muted(f"{answered} labelled this session")
        widgets.muted(
            f"{status.get('positives', 0)} good / {status.get('negatives', 0)} bad, "
            f"{status.get('needed', 0)} of each needed"
        )
        if status.get("trained"):
            widgets.muted(f"trained on {status.get('trained_labels', 0)} label(s)")
        else:
            widgets.muted("no probe yet")
        widgets.hint_text(
            "Advisory only. A trained probe scores each unit and sorts the "
            "review best-first; it never hides, refuses or deletes anything, "
            "and it files no verdict of its own yet."
        )

    def _review_runs(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """The sweep list, and the form that launches a new one."""
        from imgui_bundle import imgui

        from . import controls, icons, widgets
        from .main import _LABEL_TITLES
        from .manual import render as manual_render

        self._review_judging_card(ctx, state, review_mode)
        widgets.section("Sweeps")
        manual_render.help_button(ctx, "review")
        if widgets.disabled_button(
            f"{icons.REFRESH} Rescan",
            not state.scanning,
            reason="A scan is already running.",
        ):
            review_mode.scan(ctx)
        if state.scanning:
            imgui.same_line()
            widgets.muted("Reading...")
        # Blinding is a session control rather than a per-sweep one, and it is
        # here because it belongs beside the list it re-presents. It renames and
        # *reorders*: see review_mode's docstring on why hiding the label alone
        # blinds nothing.
        changed, blind = widgets.toggle("Blind", state.blind, tag="review-blind")
        if changed:
            review_mode.set_blind(ctx, blind)
        widgets.hint_text("Hides which settings each unit ran, and the order.")
        if not state.sweeps and not state.scanning:
            # H73. An empty Sweeps heading with a Rescan button under it says
            # nothing about *why* -- and the two reasons (no sweep has ever run
            # here, versus the bench directory is somewhere else) want different
            # responses.
            widgets.empty_state(
                icons.LIST,
                "No sweep runs found",
                "Launch one below, or check that the bench directory is where you expect.",
            )
        # **Above the filter on purpose.** Below it, a button naming a count
        # sitting under a search box reads as "remove the ones I have filtered
        # to", which it is not -- and the confirm says so in as many words.
        self._review_remove_reviewed(ctx, state, review_mode)
        # J86: a bench directory accumulates a run per experiment and nothing
        # ever removes one, so this is the panel list that grows fastest.
        needle = widgets.list_filter(ctx, "sweeps", len(state.sweeps))
        shown = 0
        for sweep in state.sweeps:
            # Blinded, like every other on-screen spelling of this row --
            # ``sweep["label"]`` is the raw DB value and matching or showing it
            # here would un-blind the row the toggle above just promised to hide.
            named = review_mode.bucket_label(state, sweep)
            if needle and needle not in named.lower():
                continue
            shown += 1
            todo = sweep["todo"]
            total = len(sweep["units"])
            selected = sweep["id"] == state.sweep_id
            if controls.selectable(f"{named}##sweep-{sweep['id']}", selected)[0]:
                # Picking a sweep by hand leaves the pass: the pass is a walk
                # over every outstanding bucket in a stated order, and a user
                # who jumps out of that order is no longer on the walk its
                # header is counting. Cleared here rather than inside
                # ``open_sweep``, which the pass itself calls.
                state.judging = None
                review_mode.open_sweep(ctx, sweep["id"])
            if not total and sweep["id"] != review_mode.RECENT_ID:
                # "0/0 reviewed" is a sentence about nothing. A sweep whose
                # units went through the ordinary job lifecycle keeps its row
                # for ever -- nothing but ``_remove_units``' tail ever deletes
                # one -- so this is a common state and not an odd one.
                widgets.muted("   no units left")
            else:
                widgets.muted(f"   {total - todo}/{total} reviewed")
            # What the run actually varied, under the name the user typed for
            # it at the time -- which is routinely "test2" by the time anyone
            # comes back to judge it.
            summary = review_mode.spec_summary(sweep.get("spec"))
            if summary:
                widgets.muted(f"   {summary}")
            if selected and sweep["id"] != review_mode.RECENT_ID:
                self._review_delete_button(ctx, state, review_mode, sweep)
        widgets.no_matches(needle, shown)
        imgui.separator()
        # The labelling passes, beside the sweep list rather than in a mode of
        # their own: the judge is meant to improve as the corpus is reviewed,
        # which is the whole reason the loop lives here.
        widgets.section("Teach the judge")
        for stage, title in _LABEL_TITLES.items():
            open_here = state.labels is not None and state.labels.stage == stage
            if controls.selectable(f"{title}##label-{stage}", open_here)[0]:
                if open_here:
                    review_mode.close_labels(ctx)
                else:
                    review_mode.open_labels(ctx, stage)
        imgui.separator()
        self._review_form(ctx, state, review_mode)

    def _review_judging_card(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """The offer to start a guided pass, or the report from the last one.

        A card at the top of the column rather than a modal, deliberately: the
        pass is a *convenience* over the loop that is already on screen, and a
        dialog in front of Review would make judging feel like something you
        have to commit to before you can look at anything.
        """
        from imgui_bundle import imgui

        from . import widgets

        if state.judging_report is not None:
            self._review_judging_report(ctx, state, review_mode)
            return
        if state.judging is not None or state.scanning:
            # Nothing to offer: the pass is running (its controls are in the
            # verdict pane, beside the mesh they are about) or the list is still
            # being read and its counts are not yet true.
            return
        outstanding = review_mode.todo_total(state)
        if outstanding <= 0:
            return
        widgets.section("Judging")
        widgets.muted(f"{outstanding} unit(s) across every bucket have no verdict.")
        if widgets.primary_button("Start judging", (-1, 0)):
            review_mode.start_judging(ctx)
        # The up-front warning, and the only one there is. The user chose no
        # dialog, so this sentence is carrying the whole of the notice that a
        # judged sweep's files are about to go -- which is why it says what
        # survives as well as what does not.
        widgets.hint_text(
            "One at a time, Accept or Reject. Once every unit of a sweep has "
            "been judged its images and meshes are removed automatically; the "
            "verdicts and findings they produced are kept."
        )
        imgui.separator()

    def _review_judging_report(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """What the pass that just ended did.

        Drawn from the stored dict and never recomputed: the numbers were
        tallied once, in memory, at the moment the pass ended -- recomputing
        per frame would be a table scan behind the one serialized connection,
        every frame, for a card that says the same thing each time.
        """
        from imgui_bundle import imgui

        from . import controls, widgets

        report = state.judging_report
        widgets.section("Judging pass")
        # Wrapped, not ``muted``: these are sentences with a sweep's own name in
        # them, in a 300 px sidebar, and the unwrapped form clipped both the
        # average grade off the end of every row and the word "do" off the
        # overall line -- so the two numbers the card exists to report were the
        # two the reader could not see.
        for row in report["sweeps"]:
            line = (
                f"{row['label']}: {row['accepted']} accepted / "
                f"{row['rejected']} rejected of {row['total']}"
            )
            if row["mean_grade"] is not None:
                line += f", avg {row['mean_grade']:+.1f}"
            widgets.muted_wrapped(line)
        widgets.muted_wrapped(
            f"{report['filed']} filed this pass - {report['accepted']} accepted, "
            f"{report['rejected']} rejected, {report['remaining']} still to do."
        )
        if controls.button("Dismiss", role=controls.ButtonRole.GHOST):
            review_mode.dismiss_report(ctx)
        imgui.separator()

    def _retention_tick(self, ctx: Any, state: Any, retained: int) -> Any:
        """The ``Confirm.body`` for a removal that could override retention.

        ``None`` when there is nothing retained to override -- a checkbox that
        would change nothing teaches the reader that the checkbox does nothing.

        The hint says what those pixels *are*, which is the whole burden this
        one control carries. ``retained_job_ids`` guards accepted meshes
        because ``tiercheck`` and the mesh probe are measured against them, and
        labelled images of **both** classes because ``judge.fit`` embeds pixels
        and refuses below ``MIN_PER_CLASS`` of each. On 2026-08-09 a bulk
        button whose confirmation truthfully promised the verdicts would be
        kept left 100 of 117 verdicts naming directories that no longer
        existed; they were kept, and the pixels three blocked items needed were
        not. So this says which promise is being broken, ``ask_clean``'s rule.
        """
        from . import controls, widgets

        if not retained:
            return None

        def body() -> None:
            changed, value = controls.checkbox(
                f"Also delete the {retained} unit(s) I accepted or labelled",
                state.drop_retained,
            )
            if changed:
                state.drop_retained = bool(value)
            widgets.hint_text(
                "This is the one rule the app otherwise keeps for you. Those "
                "pictures are what the quality judge and the tier checks are "
                "measured against; the verdict rows survive with nothing "
                "behind them."
            )

        return body

    def _review_remove_reviewed(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """Clear out every sweep there is nothing left to judge in.

        The complaint this answers is about a *class* of rows rather than one
        row, which is why it is a list-level control and not a tidier trash
        button: a Review list only ever grew, because ``store.delete_sweep`` is
        reached from exactly one place and the lifecycle paths that actually
        take a sweep's units never look at the ``sweeps`` table.
        """
        from . import dialogs, icons, widgets

        ids = review_mode.removable_ids(state)
        if not ids:
            return
        plan = review_mode.removal_plan(state, ids)
        if not widgets.disabled_button(
            f"{icons.TRASH} Remove {len(ids)} reviewed sweep(s)...",
            not state.scanning,
            (-1, 0),
            reason="A scan is already running.",
            tooltip="Clear out the sweeps you have finished judging.",
        ):
            return
        state.drop_retained = False
        retained = int(plan["retained"])
        shown = plan["labels"][:6]
        listing = "\n".join(f"  {name}" for name in shown)
        if len(plan["labels"]) > len(shown):
            listing += f"\n  and {len(plan['labels']) - len(shown)} more"
        dialogs.ask_delete(
            ctx,
            title="Remove the sweeps you have finished with?",
            message=(
                f"{plan['sweeps']} sweep(s) with nothing left to judge go from "
                f"this list, along with {plan['units']} job(s), their meshes "
                "and their reference images. The filter above does not narrow "
                "this.\n\n"
                f"{listing}\n\n"
                "Every verdict and observation they produced is kept, and so "
                "is every finding computed from them: each row carries its own "
                "copy of the settings it judged and does not need its sweep to "
                "be found again.\n\n"
                "A sweep with a unit you have not judged yet is left alone. A "
                "unit that errored or was cancelled can never be judged, so it "
                "does not hold its sweep back."
            ),
            body=self._retention_tick(ctx, state, retained),
            on_confirm=lambda: review_mode.remove_reviewed(
                ctx, ids, drop_retained=state.drop_retained
            ),
        )

    def _review_delete_button(self, ctx: Any, state: Any, review_mode: Any, sweep: Any) -> None:
        """Delete a sweep's jobs and meshes, keeping what they taught.

        Behind the same confirm an asset delete goes through
        (``panes/library.py``), because it is the same kind of act. What the
        message has to say is the part that is *not* obvious: the verdicts and
        the findings they feed survive, because each verdict carries its own
        snapshot of the settings it was filed against.
        """
        from imgui_bundle import imgui

        from . import dialogs, icons, widgets

        sweep_id = sweep["id"]
        units = len(sweep.get("units") or ())
        # ``.get``: ``test_studio_smoke``'s harness builds these dicts by hand
        # and a scan from before this field existed has no key either.
        retained = int(sweep.get("retained") or 0)
        if widgets.icon_button(
            f"{icons.TRASH}##delete-{sweep_id}",
            "Delete this sweep's jobs and meshes",
            danger=True,
            enabled=not state.scanning,
        ):
            state.drop_retained = False
            if not units:
                # The common case for an old row, and "its 0 job(s) ... are
                # deleted" is a sentence that reads as a bug.
                message = (
                    f"{sweep['label']}: its jobs and meshes are already gone "
                    "-- only the list entry is left.\n\n"
                    "The verdicts and observations it produced stay exactly "
                    "where they are. Nothing that feeds findings lives in this "
                    "row."
                )
            else:
                message = (
                    f"{sweep['label']}: its {units} job(s), their meshes "
                    "and their reference images are deleted.\n\n"
                    "The verdicts you recorded are kept, and so are the findings "
                    "they feed -- each one carries its own copy of the settings it "
                    "was filed against."
                )
                if retained:
                    message += (
                        "\n\nUnits you accepted, and any image you labelled, are "
                        "kept with their files unless you say otherwise below: a "
                        "verdict's copy of the settings cannot stand in for the "
                        "picture it was filed against."
                    )
            dialogs.ask_delete(
                ctx,
                title="Delete this sweep?",
                message=message,
                body=self._retention_tick(ctx, state, retained),
                on_confirm=lambda: review_mode.delete(
                    ctx, sweep_id, drop_retained=state.drop_retained
                ),
            )
        imgui.dummy((0, 0))

    def _review_form(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """New sweep: a prompt, a baseline captured from the generate forms,
        seeds, and the axes to vary."""
        from imgui_bundle import imgui

        from ..service import sweeps as sweeps_mod
        from . import controls, widgets

        # The label table for every guidance field, which is where a param's
        # human name already lives. Resolved here rather than in ``review_mode``
        # for that module's own rule: it may not import a pane.
        from .panes import settings_2d

        if not widgets.header("New sweep", default_open=False):
            return
        form = state.form
        widgets.field_label("prompt")
        form.prompt = widgets.multiline("##sweep-prompt", form.prompt, 60, 1000)
        widgets.field_label("name")
        form.label = widgets.input_text("##sweep-label", form.label, max_length=120)
        widgets.field_label("seeds")
        form.seeds = widgets.input_text("##sweep-seeds", form.seeds, max_length=120)

        if controls.button("Start from current 2D/3D settings"):
            form.base = review_mode.capture_base(ctx)
            form.base_note = f"{len(form.base)} setting(s) captured"
            ctx.toast("Captured the current settings as this sweep's baseline.")
        widgets.muted(
            form.base_note
            # Names the button, because "the defaults" is a fact about a sweep
            # that is unreproducible rather than merely unconfigured -- and the
            # remedy is one control away and was not being pointed at.
            or "No baseline captured; units use the defaults. Press "
            '"Start from current 2D/3D settings" above to use your own.'
        )

        # "what to vary", not "vary": the old label was a verb with no object,
        # over a combo of thirty raw param names.
        widgets.field_label("what to vary")
        rows = {row["param"]: row for row in review_mode.axis_options(ctx)}
        options = [("", "-")] + [(p, settings_2d.field_label(p)) for p in sweeps_mod.axis_params()]
        for i, row in enumerate(form.axes):
            imgui.push_id(f"axis-{i}")
            row["param"] = widgets.combo("##param", row.get("param", ""), options, width=-1)
            self._review_axis_values(row, rows.get(row.get("param") or ""))
            imgui.pop_id()
        if controls.button("Add axis"):
            form.axes.append({"param": "", "values": ""})
        if len(form.axes) > 1:
            imgui.same_line()
            if controls.button("Remove axis", role=controls.ButtonRole.GHOST):
                form.axes.pop()

        planned = review_mode.preview_units(state)
        if planned < 0:
            widgets.muted("Fill in the prompt and one axis.")
        else:
            labels = {p: settings_2d.field_label(p) for p in rows}
            widgets.muted_wrapped(review_mode.preview_line(state, labels))
            widgets.muted(f"Roughly two minutes of GPU each - {planned * 2} minutes in all.")
        enabled = planned > 0 and not form.submitting and not state.scanning
        if widgets.primary_button("Launch sweep", (-1, 0), enabled=enabled):
            review_mode.launch(ctx)

    def _review_axis_values(self, row: dict[str, Any], spec: Any) -> None:
        """One axis row's values, drawn as whatever the param actually accepts.

        **Every kind writes back the same comma-separated string.** That is the
        whole design: ``build_plan`` and ``_coerce`` parse one representation
        and are untouched, so this is a better *control* over the existing
        field rather than a second way of storing an axis -- and a param the
        catalog cannot resolve falls back to the free-text field the row has
        always been, which is why an unknown param is less discoverable rather
        than broken.
        """
        from . import controls, widgets

        kind = (spec or {}).get("kind", "text")
        if spec and spec.get("help"):
            widgets.help_marker(spec["help"])
        if kind in ("options", "bool"):
            entries = (
                spec["options"]
                if kind == "options"
                else [{"key": "true", "label": "on"}, {"key": "false", "label": "off"}]
            )
            chosen = [v.strip() for v in (row.get("values") or "").split(",") if v.strip()]
            for entry in entries:
                key = entry["key"]
                changed, _ticked = controls.checkbox(f"{entry['label']}##{key}", key in chosen)
                if changed:
                    # Rebuilt from the *entry order* rather than by appending, so
                    # the string the user sees back is stable however they
                    # clicked -- and so a value typed by hand and then unticked
                    # cannot leave a duplicate behind.
                    picked = set(chosen) ^ {key}
                    chosen = [e["key"] for e in entries if e["key"] in picked]
                    row["values"] = ", ".join(chosen)
            return
        hint = "comma-separated"
        if kind == "number" and spec.get("range"):
            low, high = spec["range"]
            default = spec.get("default")
            hint = f"{low}-{high}" + (f", e.g. {default}" if default is not None else "")
        row["values"] = widgets.input_text(
            "##values", row.get("values", ""), max_length=200, hint=hint
        )

    def _review_units(self, state: Any, review_mode: Any) -> None:
        from . import controls, icons, widgets

        widgets.section("Units")
        if not state.units:
            widgets.muted("Nothing to review here.")
            return
        for i, unit in enumerate(state.units):
            # The grade, which says more than the tick it replaces -- but a
            # unit judged before migration 10, or by an older build, has a
            # verdict and no grade, so the icons stay as the fallback rather
            # than that row going blank.
            mark = review_mode.grade_text(unit.get("grade")) or {
                "accept": icons.CHECK,
                "reject": icons.X,
            }.get(unit["verdict"] or "", " ")
            if controls.selectable(
                f"{mark} {review_mode.label(state, unit)}##unit-{unit['job_id']}",
                i == state.index,
            )[0]:
                review_mode.step(state, i - state.index)

    def _review_judging_controls(
        self, ctx: Any, state: Any, review_mode: Any, enabled: bool
    ) -> None:
        """Accept / Reject / Finish, for a pass that is running.

        The keys are named on the buttons, which is the whole of what licenses
        binding ``A`` again: the objection was never to the key, it was to a
        *silent* remap onto a grade the reviewer had not chosen. A button
        labelled "Accept (A)" in a mode entered on purpose says exactly what it
        files.
        """
        from imgui_bundle import imgui

        from ..vectors import BINARY_GRADES
        from . import controls, widgets

        reason = "A scan is running; the queue is being rebuilt."
        if widgets.disabled_button("Accept (A)", enabled, reason=reason):
            review_mode.record(ctx, BINARY_GRADES["accept"], state.pending_tags)
        imgui.same_line()
        if widgets.disabled_button("Reject (R)", enabled, reason=reason):
            review_mode.record(ctx, BINARY_GRADES["reject"], state.pending_tags)
        imgui.same_line()
        if controls.button("Finish", role=controls.ButtonRole.GHOST):
            review_mode.end_judging(ctx)
        widgets.hint_text(
            f"Files {BINARY_GRADES['accept']:+d} or {BINARY_GRADES['reject']:+d}. "
            "Use the grades below to say more; Esc ends the pass."
        )

    def _review_viewport(self, state: Any, review_mode: Any, width: float) -> None:
        """The unit's mesh, in the shared viewer.

        **What decides whether to load is ``viewer.path``, not a remembered
        unit key** -- the same comparison ``_sync_viewer`` makes, and for a
        stronger reason here. Unit keys repeat across runs of one sweep spec,
        so a key-keyed marker said "already showing that" when the mesh on
        screen belonged to a *different run*, and a verdict was then filed
        against a mesh nobody had looked at. The same marker also survived a
        trip through 3D, which loads a library asset into this same viewer, so
        coming back drew that asset under Review's verdict buttons. Comparing
        paths fixes both, structurally, and needs no reset anywhere.
        """
        from imgui_bundle import imgui

        from . import widgets
        from .panes import overlay

        ctx = self.app_ctx
        if ctx.state.comparing:
            # 3D's Escape handler does exactly this pair (main.py's
            # ``_shortcut``), but Review draws no compare UI of its own and
            # its Escape branch returns before that handler runs -- so a
            # split entered in 3D and never exited stays armed forever once
            # the mode switches. ``_draw_viewport_image`` halves the width
            # for any mode whenever ``comparing`` is set, so without this a
            # sweep unit's mesh renders next to a stale compare texture.
            # Checked every frame Review draws (not just on entry), so it
            # also covers 3D -> Review -> 3D -> Review re-entry.
            ctx.state.comparing = None
            self.viewer.exit_compare()

        unit = review_mode.current(state)
        if self.viewer.pose_mode:
            # The pose editor owns the viewer and holds unsaved rotations;
            # loading over it would discard them without the confirm every
            # other exit goes through (``pose_panel.guard``). ``_sync_viewer``
            # refuses on exactly this condition -- this is the same refusal.
            widgets.muted("Finish or close the pose editor to review a mesh.")
            return

        self._review_load(unit, review_mode)

        image_pos = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        height = max(avail.y, 64)
        if unit is None:
            # Before ``has_model``: arriving from 3D leaves an asset loaded,
            # and asking the viewer first drew that asset with no unit selected
            # -- a mesh on screen that no button on the right refers to.
            overlay.placeholder(self.app_ctx)
        elif self.viewer.has_model:
            self._draw_viewport_image(image_pos, width, height)
        else:
            widgets.muted(f"No mesh for this unit (status: {unit['status']}).")

    def _review_load(self, unit: Any, review_mode: Any) -> None:
        """Show the unit's mesh if the viewer is not already showing it.

        ``viewer.path`` is set even when there is nothing to show, so a unit
        whose job errored (or whose GLB will not open) is tried once rather
        than re-attempted -- and re-toasted -- on every frame.

        Parsed off-thread, ``_sync_viewer``'s split under its own key: this is
        reached from the draw of the Review pane, so a blocking load here was
        a frozen frame per arrow press through a pass -- and a judging pass is
        forty arrow presses. ``_adopt_review_model`` is the other half.
        """
        from .main import REVIEW_MESH_KEY

        wanted = None if unit is None else review_mode.model_path(unit)
        if self.viewer.path == wanted or self.viewer.pending == wanted:
            return
        if wanted is None or not wanted.exists():
            self.viewer.clear()
            self.viewer.path = wanted
            return
        self.viewer.pending = wanted
        if not self.app_ctx.submit(REVIEW_MESH_KEY, self.viewer.parse_model, wanted, tag=wanted):
            self.viewer.pending = None

    def _adopt_review_model(self, done: Any) -> None:
        """Take a parsed sweep-unit mesh. Frame thread only.

        ``_adopt_model``'s shape without its thumbnail capture, which is the
        Create selection's and would stamp this mesh onto whichever job that
        is. A parse that failed still sets ``viewer.path`` to what was wanted,
        which is the "tried once" rule above.
        """
        wanted = done.tag
        if wanted is None or self.viewer.pending != wanted:
            self.viewer.pending = None
            return
        self.viewer.pending = None
        try:
            if not done.ok:
                raise RuntimeError(str(done.error))
            self.viewer.adopt_model(done.result, wanted)
        except Exception:
            log.exception("could not open %s", wanted)
            self.viewer.clear()
            self.viewer.path = wanted
            self.app_ctx.toast("Could not open that sweep unit's mesh.", "error")

    def _review_verdict(self, ctx: Any, state: Any, review_mode: Any) -> None:
        from imgui_bundle import imgui

        from . import forms, widgets

        unit = review_mode.current(state)
        if unit is None:
            widgets.muted("Pick a sweep on the left.")
            self._review_findings(ctx)
            return

        if state.judging is not None:
            # The pass's own position, above the unit's. It counts *filed*
            # against the total outstanding when the pass started, which is a
            # different question from "where in this sweep am I" -- and the one
            # a reviewer who has agreed to do twenty of these is asking.
            widgets.section(f"Judging {state.judging.filed + 1} of {state.judging.total}")
            widgets.muted(review_mode.label(state, unit))
        else:
            widgets.section(review_mode.label(state, unit))
        # ``review_mode.label`` again rather than the raw job_id: under blinding
        # that id is the very thing being withheld, and printing it in full here
        # would hand back what the truncated ``#abcdef`` above was hiding.
        named = review_mode.label(state, unit)
        widgets.muted(f"{state.index + 1} of {len(state.units)}  -  {named}")

        reference = review_mode.reference_path(unit)
        if reference is not None and ctx.textures is not None:
            texture = ctx.textures.get(review_mode.cache_id(unit), reference)
            if texture is not None:
                side = min(imgui.get_content_region_avail().x, 220.0)
                imgui.image(widgets.texture_ref(texture), (side, side))

        for line in review_mode.mesh_lines(unit):
            widgets.muted(line)

        # Below the measurements and named as a judgement, because it is one and
        # the measurements are not. Empty when there is no probe, when the judge
        # had nothing to say about this row, and always under blinding.
        judged = review_mode.score_line(state, unit)
        if judged:
            widgets.muted(judged)

        imgui.separator()
        enabled = not state.scanning

        if state.pending_negative:
            # R is a *sign*, held until the next digit, and nothing on screen
            # said it was held: the reviewer who pressed R and then walked
            # away came back and pressed 4 expecting +4. Warn-coloured because
            # the consequence of not noticing is the opposite verdict, and it
            # says how to drop it -- Esc, which ``_disarm`` already answers.
            from . import theme

            widgets.text_colored(
                theme.WARN, "Negative armed: the next digit files a minus. Esc drops it."
            )

        if state.judging is not None:
            # Above the grade row, not instead of it. The binary pair is the
            # fast path; the eleven-point scale below is the power path and
            # keeps working, files a grade and advances the pass exactly as
            # these two do.
            self._review_judging_controls(ctx, state, review_mode, enabled)

        with forms.Form("review-verdict") as form_ui:
            with form_ui.field(
                "grade",
                "Grade",
                help_text="A digit grades; press R first for a negative grade.",
                helper="+5 ships as-is, +3 is usable, and -5 is unusable.",
            ):
                grade = widgets.grade_buttons("review", enabled)
            if grade is not None:
                review_mode.record(ctx, grade, state.pending_tags)

            with form_ui.field("tags", "Tags", helper="Optional; S skips without filing a grade."):
                tag = widgets.tag_toggles("review", state.pending_tags, enabled)
            if tag is not None:
                review_mode.toggle_tag(state, tag)

            if widgets.disabled_button(
                "Skip (S)",
                enabled,
                reason="A scan is running; the queue is being rebuilt.",
            ):
                review_mode.advance(state)

        if unit["verdict"]:
            # ``grade_text`` rather than the verdict word: the word is the
            # derived cut and the grade is what was actually said, so showing
            # the word here would answer a coarser question than the one the
            # buttons above ask.
            recorded = review_mode.grade_text(unit.get("grade")) or unit["verdict"]
            if unit.get("tags"):
                recorded += " - " + ", ".join(unit["tags"])
            widgets.muted(f"Recorded: {recorded}")

        self._review_findings(ctx)

    def _review_findings(self, ctx: Any) -> None:
        """What the verdicts add up to, and the one-click way to reuse it.

        Two answers, most conclusive first. Axis verdicts are matched pairs
        recovered from sweep structure -- same prompt, same seed, one param
        differing -- the only all-else-equal comparison in the pool. The
        ranked vectors are whole configurations ordered by their Wilson lower
        bound (the "floor" percentage), because the per-parameter marginals
        are confounded and a raw rate lets a lucky 5/5 outrank a 19/20.
        """
        from imgui_bundle import imgui

        from ..bench import findings as findings_lib
        from ..service import findings as svc_findings
        from . import controls, review_mode, widgets

        imgui.separator()
        if not widgets.header("What works", default_open=False):
            return
        # Everything below the header guard (B21), the load included: it is
        # mtime-cached but still a stat per frame, for a section that is
        # closed by default -- and the lines are formatted from scratch.
        doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
        top = svc_findings.presets(doc or {})
        axis_lines = findings_lib.comparison_lines(doc)
        if axis_lines:
            widgets.muted("Axis verdicts (matched pairs, all else equal):")
            for line in axis_lines:
                if line.startswith("    "):
                    widgets.muted(line)
                else:
                    imgui.text_wrapped(line)
            imgui.separator()
        if not top:
            widgets.muted(
                f"No whole configuration has {svc_findings.PRESET_MIN_N} verdicts yet."
                if axis_lines
                else (
                    f"Nothing yet: a configuration needs "
                    f"{svc_findings.PRESET_MIN_N} verdicts to rank, and axis "
                    "verdicts need matched pairs from sweeps sharing seeds."
                )
            )
            return
        for entry in top[:5]:
            summary = review_mode.describe_vector(entry["vector"])
            imgui.text_wrapped(f"{findings_lib.vector_line(entry)}  -  {summary}")
            measured = findings_lib.metrics_line(entry.get("metrics"))
            if measured:
                widgets.muted(measured)
            tagged = findings_lib.tag_line(entry)
            if tagged:
                widgets.muted(tagged)
            vector = entry["vector"]
            if controls.button(f"Apply to forms##apply-{entry['key']}"):
                review_mode.apply_vector(ctx.state, vector)
                ctx.toast("Applied those settings to the 2D and 3D forms.")
            imgui.separator()

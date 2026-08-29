"""Everything the UI remembers between frames.

Plain dataclasses with no imgui in sight, so the interesting parts -- which
action a card offers, how the ETA is smoothed, what a filter matches -- are
testable as ordinary Python. imgui's own widget state (a text buffer's cursor,
a combo's scroll) stays inside imgui; this is the application's state, and the
line between the two is that anything here would still make sense if the app
were driven by a script.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

# The prompt history the 2D pane offers. Twenty is what the browser kept: long
# enough to find yesterday's phrasing, short enough to scan.
MAX_HISTORY = 20

def default_form_2d() -> dict[str, Any]:
    """What a submit is composed from.

    Split across the two panes deliberately: a control belongs to exactly one
    of them. ``platform`` is the 3D pane's alone since the taxonomy retirement
    -- it is a geometry resolution, and this form carries no prompt-fragment
    fields any more.

    The seed is rolled here rather than being a constant: it is deliberately
    not persisted (settings.VOLATILE), so a literal default meant every launch
    opened on the same seed and a first Generate reproduced last week's image.
    """
    from .. import models
    from ..service.sprites import DEFAULT_SPRITE_OUTLINE
    from ..service.tilesheets import DEFAULT_MODE as DEFAULT_TILE_MODE
    from ..service.validation import random_seed
    from .create_assets import DEFAULT_ASSET_TYPE

    return {
        "prompt": "",
        # Prompt expansion mode: "off" | "asset" | "scene". "off" rather than
        # "" so the combo shows its own first entry; submit sends it and
        # normalize stores nothing for it.
        "expand": "off",
        "negative_prompt": "",
        # Explicit rather than empty: the combo and the stored form must say
        # the same model guidance.normalize will run.  An empty selection used
        # to draw the registry's first row while silently submitting a
        # different default.
        "base_model": models.DEFAULT_BASE_MODEL,
        "style_lora": "",
        "lora_weight": 0.9,
        "seed": random_seed(),
        "seed_locked": False,
        "count": 1,
        # reference | tile | sheet. What this pane submits. A tile is the same
        # pipeline with circular padding and a different framing template, so it
        # belongs to the pane that owns the prompt rather than to a mode of its
        # own -- and it is persisted, unlike the seed, because someone making a
        # texture set is making several. A sheet is a different *door* (its own
        # job kind), but it is composed from the same prompt, which is the whole
        # reason it is a third value here rather than a fourth mode in the rail.
        # Authoritative user intent. The fields below remain as a compatibility
        # bridge to the generation services and are synchronised from this by
        # create_assets.sync_legacy_fields.
        "asset_type": DEFAULT_ASSET_TYPE,
        # New normalized vocabulary. ``asset_type`` remains in the form as a
        # compatibility alias for profiles and old service adapters.
        "generation_type": DEFAULT_ASSET_TYPE,
        "quality": "quality",
        "model_mode": "auto",
        "model_override": "",
        "output": "reference",
        # The Sheet output's three fields. Strings, including the tile size,
        # because ``restore_form`` gates on ``type(value) is type(default)`` and
        # the segmented controls that carry all three hand back strings -- an
        # int default here would make every persisted size fail to restore
        # silently. They are converted at the submit boundary, once.
        #
        # tile | sprite. Which door the Sheet output opens.
        "sheet_type": "tile",
        "tile_size": "32",
        # materials | terrain | grid -- which *layout* the tile arm draws, and
        # therefore which shape the request takes at
        # ``service.tilesheets.create_tile_sheet``. The door's own default, read
        # from it rather than spelled here, because the door is what refuses a
        # mode it does not build.
        #
        # ``grid`` is deliberately not the default and is deliberately still
        # offered: it paints one frame through a guide whose sixty-four cells
        # are identical (docs/measurements/2026-08-18-tile-sheet-grid.md), and
        # it is the only layout that draws a 3/4 or an isometric tile. The door
        # refuses it unless the request says ``allow_grid``, which the pane
        # sends only when this field says so.
        "tile_mode": DEFAULT_TILE_MODE,
        # The materials layout's list, one surface per line. One string rather
        # than a list because ``restore_form`` gates on
        # ``type(value) is type(default)`` and the control is a multiline text
        # field; it is split into lines once, at the submit boundary.
        "materials": "",
        # How many times each material line is drawn. A string for the reason
        # ``tile_size`` is one, and converted at the same boundary.
        "variants": "1",
        # The terrain layout's two surfaces, in precedence order: ``inner`` is
        # the one the forty-seven blob cases are pictures *of*.
        "inner_terrain": "",
        "outer_terrain": "",
        # Context both terrain materials share, and never an instruction to
        # draw an edge -- see ``service.tilesheets.create_tile_sheet``.
        "boundary": "",
        # The sprite arm's two, named apart from the tile arm's on purpose:
        # "cell size" and "tile size" are the same measurement of different
        # things, and one key serving both would carry 48 (legal for a sprite
        # cell, not offered for a tile) across a switch of arms.
        "sheet_layout": "turnaround",
        "cell_size": "64",
        # The pixel look, shared by *both* sheet arms -- unlike the pair above,
        # because "which authored palette" and "dither against it" are the same
        # question with the same answers whichever arm asks, and the palettes
        # come from one directory. Named for the service fields they become
        # (``palette``, ``dither``, ``outline``) so a refusal's ``field=`` rings
        # the control it is about.
        #
        # ``palette`` is also a *profile* field (``studio.profiles._FIXED``):
        # two sheets of one character matching is what a profile is for, and a
        # palette was the only piece of that it did not already carry.
        "palette": "",
        # Not gated on the palette, and that is measured rather than assumed:
        # ``tilesheet.quantize_tiles`` branches on ``not entries and not
        # dither`` and ``queue`` maps the sprite atlas through ``map_palette``
        # either way, so a dithered sheet naming no palette dithers against the
        # table the median cut derived. (``pipelines.asset2d`` is the path where
        # dither *does* need a palette, and it is a different pipeline.)
        "dither": False,
        # The sprite arm's alone: a tile is opaque edge to edge, so an outline
        # finds every cell's border and draws a grid line around all of them --
        # the tile door refuses one by name, and this form must not offer what
        # that door refuses. Seeded from the pipeline rather than spelled here.
        "outline": DEFAULT_SPRITE_OUTLINE,
        # top_down | three_quarter | isometric -- the *view*, meaning where
        # the camera is. The key is still ``projection`` because it is a
        # persisted form field and a stored profile carries it; a value of
        # "orthogonal" from before the vocabulary widened reads as "top_down"
        # (``service.tilesheets.LEGACY_VIEWS``).
        #
        # Still deliberately not Plotter's five. Oblique was refused for being
        # a button that changed nothing about the picture, and 3/4 shares its
        # lattice -- but what SDXL is handed is the subject clause and the
        # guide, and both of those differ, which is the difference between the
        # two cases.
        "projection": "top_down",
        "target_cell_px": "",
        # Conditioning. Every number is a float literal on purpose:
        # restore_form gates on `type(value) is type(default)`, so an int here
        # would make a persisted 0.6 fail to restore.
        "ref_path": "",
        "ip_adapter": "",
        "ip_scale": 0.6,
        "control": "",
        "control_scale": 0.65,
        "control_end": 0.8,
    }

def form_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """A 2D form filled from a finished job's params -- "another like this".

    Only keys the form already has, which is what keeps a derived value the
    worker recorded (``composed_prompt``, ``reference_report``, a mesh verdict)
    from ever becoming a submitted field: the form is the allowlist.

    Numbers are coerced rather than type-matched. Params arrive from JSON,
    where a strength saved as 0.6 can come back as an int 0 or 1, and the
    persisted-settings merge's ``type(value) is type(default)`` rule would
    silently drop exactly those fields.
    """
    form = default_form_2d()
    for key, default in list(form.items()):
        value = params.get(key)
        if value is None:
            continue
        try:
            if isinstance(default, bool):
                form[key] = bool(value)
            elif isinstance(default, float):
                form[key] = float(value)
            elif isinstance(default, int):
                form[key] = int(value)
            elif isinstance(default, str):
                form[key] = str(value)
        except (TypeError, ValueError):
            continue
    from . import create_assets

    _restore_sheet_block(form, params)
    form["asset_type"] = (
        create_assets.asset_type_from_params(params) or form["asset_type"]
    )
    form["generation_type"] = form["asset_type"]
    create_assets.sync_legacy_fields(form)
    return form


def _restore_sheet_block(form: dict[str, Any], params: dict[str, Any]) -> None:
    """Put a finished tile set's own request back into the form, in place.

    A tile set is the one kind whose request does not live in ``params`` as flat
    fields: ``create_tile_sheet`` writes one nested ``sheet`` block, because it
    is a *document description* rather than a settings vector. So the loop above
    -- which matches param keys to form keys -- finds none of it, and "another
    like this" on a tile set reopened the form at its defaults with nothing said.

    **A block with no mode is the grid.** That is the door's own reading of a
    version-2 block, and it is why the sheet version exists: a mode key alone
    could not tell "a sheet from before the seamless layouts" from "a sheet
    somebody asked the grid for".
    """
    from ..service.tilesheets import MODE_GRID, MODE_MATERIALS, MODE_TERRAIN, TILE_MODES

    block = params.get("sheet")
    if not isinstance(block, dict):
        return
    mode = str(block.get("mode") or MODE_GRID)
    if mode not in TILE_MODES:
        return
    form["tile_mode"] = mode
    for key, source in (("tile_size", "tile_w"), ("projection", "projection")):
        value = block.get(source)
        if value not in (None, ""):
            form[key] = str(value)
    cells = [cell for cell in (block.get("materials") or ()) if isinstance(cell, dict)]
    if mode == MODE_MATERIALS:
        # The *lines*, not the cells: the block holds one record per
        # ``(line, variant)`` pair, expanded line-major, so the first draw of
        # each line is the list the user typed.
        form["materials"] = "\n".join(
            str(cell.get("prompt") or "")
            for cell in cells
            if int(cell.get("variant") or 1) == 1
        )
        form["variants"] = str(int(block.get("variants") or 1))
    elif mode == MODE_TERRAIN and len(cells) == 2:
        # Two subjects in ``(inner, outer)`` order, which is not a convention:
        # it is which of the two the forty-seven pictures are of.
        form["inner_terrain"] = str(cells[0].get("prompt") or "")
        form["outer_terrain"] = str(cells[1].get("prompt") or "")
        form["boundary"] = str(block.get("boundary") or "")


DEFAULT_FORM_3D: dict[str, Any] = {
    "backend": "trellis_single_view",
    "texture_mode": "pbr",
    "view_assets": {},
    "hunyuan_license_ack": False,
    "platform": "",
    "profile": "raw",
    # Deliberately without a widget. A triangle budget only means anything for
    # profile "custom", and the generate form offers ``raw`` alone until a tier
    # has been qualified -- so a control here would offer a number that "raw"
    # ignores. The plumbing through _payload is kept because it is correct the
    # moment a tier is exposed; the retarget control on a finished mesh is where
    # a budget is actually chosen today.
    "custom_triangles": 0,
    "size_m": 0.0,
    "bg_removal": "",
    "mesh_seed": 0,
    "mesh_seed_locked": False,
    # How many meshes one Make 3D queues. 1 is the old behaviour exactly.
    # Reliability rather than variety: trellis is deterministic in its seed and
    # its failure mode is a lottery, so three attempts and a picker is the
    # cheapest answer to a hole through the shoulder. Bounded at
    # validation.MAX_MESH_CANDIDATES, because each one is two minutes of a
    # serial worker.
    "candidates": 1,
    "rig": False,
    "rig_template": "",
    # Whether the host recentres and rescales the subject before the trellis
    # upload. False, matching queue.DEFAULT_REFERENCE_PREP: whether
    # normalising here helps or fights the exe's own preprocessing is
    # unmeasured, so the pane offers the switch rather than turning it on for
    # everyone. Flip both together once the occupancy sweep says which way.
    "reference_prep": False,
}


# The library's sort keys and what the combo calls them (J85). ``newest`` keeps
# its name rather than becoming ``date``: it is a *persisted* value, so a
# rename would need an alias table, and the direction toggle beside the combo
# is what turns it into oldest-first.
SORTS: list[tuple[str, str]] = [
    ("newest", "date"),
    ("name", "name"),
    ("kind", "kind"),
    ("duration", "time taken"),
    ("size", "size on disk"),
    ("best", "score"),
]

# The field prefixes the filter box understands (J87). Each maps onto a
# predicate over one job row. Free text keeps its old meaning exactly -- a
# substring of name, prompt, tags or id -- so a query with no colon in it
# behaves character for character as it did.
QUERY_FIELDS = ("tag", "status", "kind", "stage", "id", "name")


def parse_query(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Split a filter box into ``(free words, [(field, value)])`` (J87).

    ``tag:wood status:error rusty`` is two constraints and one word. A prefix
    this does not recognise is **not** a constraint -- ``http://x`` stays a
    free-text search for ``http://x`` -- because the box has always been plain
    text and silently reinterpreting a colon somebody typed on purpose is how a
    search starts returning nothing with no explanation.

    Values may be quoted for a space: ``name:"a wooden chest"``.
    """
    import shlex

    try:
        words = shlex.split(text)
    except ValueError:
        # An unbalanced quote is not an error the user should see: they are
        # mid-typing, and the half-written query is `name:"a wo`. Closing it
        # for them reads it the way they clearly meant it; a plain split would
        # hand `"a` to the field and leave `wo` as a separate word, which
        # matches nothing and looks like the box being broken.
        try:
            words = shlex.split(text + '"')
        except ValueError:
            words = text.split()
    terms: list[str] = []
    fields: list[tuple[str, str]] = []
    for word in words:
        head, sep, tail = word.partition(":")
        if sep and head.lower() in QUERY_FIELDS and tail:
            fields.append((head.lower(), tail.lower()))
        elif word:
            terms.append(word.lower())
    return terms, fields


def _field_matches(job: dict[str, Any], field: str, value: str) -> bool:
    """One ``field:value`` constraint against one row.

    Tags match a whole comma-separated entry rather than a substring, because
    ``tag:wood`` finding ``driftwood`` makes the prefix useless for the one
    thing it is for -- narrowing. Everything else is a substring, which is what
    the plain-text box has always done.
    """
    if field == "tag":
        tags = [t.strip().lower() for t in str(job.get("tags") or "").split(",")]
        return value in tags
    if field == "kind":
        return card_kind(job) == value
    haystack = {
        "status": job.get("status"),
        "stage": job.get("stage"),
        "id": job.get("id"),
        "name": job.get("name") or job.get("prompt"),
    }.get(field)
    return value in str(haystack or "").lower()


@dataclass
class Filters:
    """The library's filter bar. Persisted, because a workshop is filtered the
    same way every session."""

    text: str = ""
    status: str = "all"  # all | done | running | error
    # all | reference | tile | model | rig | sheet | sprite -- the same seven
    # the library's combo offers, in its order. ``tile`` is a reference the
    # seamless path produced, and it is a kind of its own because its next step
    # is an export rather than a mesh (see ``card_kind``/``card_action``);
    # ``sprite`` is a sprite-sheet draft, which is 2D and goes to Inker.
    kind: str = "all"
    favorites_only: bool = False
    # One of SORTS. Persisted with the rest of the filter bar, because a
    # workshop is browsed the same way every session.
    sort: str = "newest"
    # Which way that sort runs. True is each key's *own* natural order --
    # newest, largest, longest, best, and A to Z -- rather than "numerically
    # descending", because there is no descending order of a name that anybody
    # means by default. False reverses whatever that is, which is the only
    # description of this flag that stays true for every key (J85). The default
    # is the old behaviour for both keys that existed before.
    descending: bool = True
    # Whether the list is showing the trash instead of the workshop (J91). A
    # *view*, not a filter on top of the others: everything else narrows what
    # you are looking at, and this changes which set you are looking at. Which
    # is also why it is the one field here that is never persisted -- see
    # ``VOLATILE_FILTERS``.
    trash: bool = False

    def matches(self, job: dict[str, Any]) -> bool:
        # First, and above the sweep/candidate rules: a trashed job is out of
        # the workshop entirely, and the trash view is out of everything else.
        # Asked as one equality so the two views can never both show a row or
        # both hide one.
        if bool(job.get("deleted_at")) != self.trash:
            return False
        if job.get("sweep_id"):
            # A sweep's units are dozens of near-identical rows whose whole
            # purpose is to be compared against each other in Review. Left in,
            # one launched sweep buries a workshop's actual assets. They are
            # reachable by their sweep, and deleting the sweep deletes them.
            return False
        if job.get("candidate_group"):
            # And the same rule for a mesh candidate nobody has picked yet:
            # three attempts at one asset are three near-identical cards, and
            # the choice between them belongs in the picker rather than in a
            # list of finished work. The column goes NULL the moment the user
            # keeps one (``service.jobs.keep_candidate`` dissolves the whole
            # group), so nothing stays hidden without a picker to reach it by.
            return False
        if self.favorites_only and not job.get("favorite"):
            return False
        if self.status != "all" and job.get("status") != self.status:
            return False
        if self.kind != "all" and card_kind(job) != self.kind:
            return False
        if self.text:
            terms, fields = parse_query(self.text)
            # Field constraints are **in addition to** the combos above, never
            # instead of them: `status:error` with the status combo on "done"
            # is a contradiction and correctly shows nothing, which is honest
            # about what the bar is set to. A term that quietly overrode a
            # visible control would be worse.
            for field, value in fields:
                if not _field_matches(job, field, value):
                    return False
            if terms:
                haystack = " ".join(
                    str(job.get(k) or "") for k in ("name", "prompt", "tags", "id")
                ).lower()
                # Every word, not any: typing more narrows, which is what a
                # filter box is for.
                if not all(term in haystack for term in terms):
                    return False
        return True

    def failures(self, jobs: list[dict[str, Any]]) -> int:
        """How many rows switching the status filter to "error" would reveal.

        Derived by re-running :meth:`matches` under that one substitution
        rather than by counting ``status == "error"``, so the number is exactly
        what the click produces: the kind, text and favourites filters still
        apply afterwards, and a count that ignored them would offer a jump to
        an empty list. Sweep units stay excluded for the same reason they are
        excluded from the list -- one failed sweep is dozens of rows nobody
        can see.
        """
        probe = replace(self, status="error")
        return sum(1 for job in jobs if probe.matches(job))

    def order(
        self, jobs: list[dict[str, Any]], sizes: dict[str, int] | None = None
    ) -> list[dict[str, Any]]:
        """The list in the order the bar asks for (J85).

        Every key is a **stable** sort, so rows that tie -- two assets with the
        same name, a hundred with no score -- keep the query's own newest-first
        order rather than shuffling on every refresh half a second apart.

        Every key also sorts its *unanswerable* rows into a second bucket at
        the end, in both directions. That is the rule "best" already followed
        for an unranked job, generalised: an asset whose directory has not been
        measured yet has no size, and putting it at zero would file the whole
        unmeasured backlog under "smallest" until the storage walk lands.

        ``sizes`` is ``{job dir name: bytes}`` from the storage walk, which is
        deferred (C32) and may simply not have happened -- hence the bucket.
        """
        key = self._sort_key(sizes or {})
        if key is None:
            # "newest" descending *is* the query's order; returning it
            # untouched is both free and exactly stable.
            return list(jobs) if self.descending else list(reversed(jobs))
        ordered = sorted(jobs, key=key)
        if not self.descending:
            # A plain ``reversed`` would put the unanswerable bucket *first*,
            # which is wrong in a way that is easy to miss: "unknown" is not a
            # value at one end of a scale, it is the absence of one, and it
            # belongs last whichever way the answered rows run.
            answered = [job for job in ordered if key(job)[0] == 0]
            unknown = [job for job in ordered if key(job)[0] != 0]
            ordered = list(reversed(answered)) + unknown
        return ordered

    def _sort_key(self, sizes: dict[str, int]):
        """The comparison for the current sort, or ``None`` for "as queried".

        Every key returns ``(bucket, value)`` where bucket 1 means "cannot
        answer" -- see :meth:`order`. Values are negated where the natural
        reading is largest-first, so ``descending`` is a plain reverse.
        """
        sort = self.sort
        if sort == "newest":
            return None
        if sort == "name":
            def key(job: dict[str, Any]) -> tuple[int, Any]:
                # The *displayed* name, which is the prompt when there is no
                # title -- sorting by a column the card does not show would
                # look like no sort at all.
                shown = (job.get("name") or job.get("prompt") or "").strip().lower()
                return (0, shown) if shown else (1, "")

            return key
        if sort == "kind":
            return lambda job: (0, (card_kind(job), -float(job.get("created_at") or 0.0)))
        if sort == "duration":
            def key(job: dict[str, Any]) -> tuple[int, Any]:
                started, finished = job.get("started_at"), job.get("finished_at")
                if not started or not finished or finished < started:
                    # Queued, running, or a row that never recorded one of the
                    # two: no duration exists, rather than a duration of zero.
                    return (1, 0.0)
                return (0, -(float(finished) - float(started)))

            return key
        if sort == "size":
            def key(job: dict[str, Any]) -> tuple[int, Any]:
                measured = sizes.get(job["id"])
                return (1, 0.0) if measured is None else (0, -float(measured))

            return key
        if sort == "best":
            def key(job: dict[str, Any]) -> tuple[int, Any]:
                rank = (job.get("params") or {}).get("rank")
                if not isinstance(rank, dict):
                    # Most of a workshop predates the score, so treating "no
                    # score" as a middling one would scatter hundreds of old
                    # assets through the ranked candidates this sort exists to
                    # surface -- and treating it as zero would tie it with the
                    # refused, which the bucket keeps distinct.
                    return (1, 0.0)
                try:
                    return (0, -float(rank.get("score") or 0.0))
                except (TypeError, ValueError):
                    return (1, 0.0)

            return key
        return None


# The filter-bar fields that must never survive a restart, in the shape
# ``settings.VOLATILE`` already established for the forms.
#
# ``trash`` is here because it is a *view* rather than a filter: everything
# else on the bar narrows what you are looking at, and remembering that is the
# whole reason the bar is persisted at all. This one changes *which set* you
# are looking at, so a session that ended while emptying the trash reopened in
# the trash -- an empty-looking library with no obvious way back, and the one
# reading of "restore my filters" nobody wants.
VOLATILE_FILTERS: tuple[str, ...] = ("trash",)


def filters_to_store(filters: Filters) -> dict[str, Any]:
    """The filter bar as it should be written to the settings file."""
    return {k: v for k, v in vars(filters).items() if k not in VOLATILE_FILTERS}


def filters_from_stored(stored: Any) -> Filters:
    """The inverse. Unknown, volatile and wrongly-typed keys are dropped.

    Volatile keys are dropped on the way *in* as well as on the way out, so a
    settings file written by an older build -- which really does carry
    ``trash: true`` -- opens on the library like every other launch.

    **Types are checked, not only names.** The name filter survived as whatever
    JSON held it: ``{"text": 5}`` built a ``Filters`` whose ``text`` is an int,
    which then reached ``parse_query`` and imgui's ``input_text`` -- and by then
    it is the frame loop's problem rather than a bad byte in a file. Checked
    against the *default's* type, the rule ``settings.restore_form`` has applied
    to every remembered form field since it was written, and for its stated
    reason: settings are untrusted JSON read before any control can clamp them.
    An int where a float belongs is the one widening allowed, because JSON does
    not distinguish ``0`` from ``0.0`` and a stored ``0`` is not corruption.
    """
    values = stored if isinstance(stored, dict) else {}
    blank = Filters()
    keep: dict[str, Any] = {}
    for key, value in values.items():
        if key not in Filters.__annotations__ or key in VOLATILE_FILTERS:
            continue
        default = getattr(blank, key)
        if type(value) is type(default):
            keep[key] = value
        elif isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            keep[key] = float(value)
    return Filters(**keep)


def set_mode(state: AppState, key: str) -> bool:
    """Switch modes, recording where Esc should go back to. -> whether it moved.

    **The one implementation.** ``App._set_mode`` and the command palette's
    ``go:`` commands both call this, because they were two spellings of it and
    had already drifted: the palette's copy did not early-return on the mode it
    was already in, so re-selecting the current mode from the palette left
    ``previous_mode == mode`` -- and ``_escape_mode`` treats that as no history
    at all and falls back to Home. Escaping a pass-through mode is documented
    as going back to *the mode it came from*, so the drift was a wrong answer
    rather than a cosmetic difference.

    Takes the state rather than being an ``AppState`` method so it stays
    callable against any object carrying the four fields, which is what the
    palette's tests hand it.
    """
    if key == state.mode:
        return False
    state.previous_mode = state.mode
    state.mode_observed = key
    state.mode = key
    return True


def card_kind(job: dict[str, Any]) -> str:
    """What the filter calls this row.

    Not simply job["kind"]: a text job that stops at a reference, one that
    stops at a tile and one that goes on to a mesh are all the same kind and
    three different things to look for.
    """
    if job.get("kind") in ("rig", "sheet"):
        return job["kind"]
    if job.get("kind") == "charsheet":
        # Filed under "sheet" with the pose sheets rather than given a kind of
        # its own: what the filter is for is "show me the sprite sheets this
        # mesh has", and a character sheet is one -- it lands in the same
        # ``sheets/`` directory, opens in the same viewer and exports through
        # the same writers. The thing that makes it a Troupe sheet is the
        # frame table it was laid out on, which is a property of the artifact
        # and not of the row.
        return "sheet"
    if job.get("kind") == "sprite_synthesis":
        # A kind of its own rather than the "model" every other follow-up job
        # falls through to: a sprite draft is 2D and its next step is Inker, so
        # a workshop filtered to meshes should not be showing it.
        return "sprite"
    if job.get("kind") == "tile_sheet":
        # Its own kind for the sprite draft's reason exactly: a tile sheet is
        # 2D, its next step is a map rather than a mesh, and a workshop filtered
        # to meshes has no business showing one.
        return "tilesheet"
    stage = job.get("stage")
    if stage in ("reference", "tile"):
        return stage
    return "model"


# How long each level of toast stays up, in seconds (H68). Four levels, and
# the ladder is the *reading time* each one asks for rather than a severity
# score: a confirmation is read at a glance, a warning is a sentence, and an
# error usually says what to do next and four seconds is not long enough to
# read a sentence and act on it.
#
# ``success`` and ``warn`` are new, and the gap they fill is real: everything
# that finished, and everything that half-worked, has been arriving in the same
# neutral grey as "settings copied to the form". A level with no entry here
# falls back to ``info`` -- the same rule ``Toast.action`` follows, and what
# lets a level be introduced from the calling side.
TOAST_LEVELS: dict[str, float] = {
    "info": 4.0,
    "success": 4.0,
    "warn": 6.0,
    "error": 8.0,
}

# The levels that take mouse input and offer a close button: a message the user
# may need to act on must not be un-clickable, and must not vanish under the
# cursor. See ``widgets.toasts``.
TOAST_STICKY = frozenset({"warn", "error"})


@dataclass
class Toast:
    text: str
    level: str = "info"  # one of TOAST_LEVELS
    born: float = field(default_factory=time.monotonic)
    # A route the toast offers alongside its text. Only "log" today: an
    # unexpected exception's toast says "see the log for details" and the one
    # button that opens it lives in the global command surfaces, which is
    # further away than eight seconds. Named rather than a bool so the widget's label
    # is a function of the toast; an unrecognised name simply draws nothing.
    action: str | None = None
    # What the action acts *on* -- a job id for "show", a sweep id for
    # "review". A field rather than a name like ``show:abc123``: the label
    # comes from ``TOAST_ACTIONS[action]``, so an id spliced into the name
    # would make every toast's action unrecognised and silently unlabelled.
    action_arg: str | None = None

    @property
    def ttl(self) -> float:
        # A level the UI has not learned is an ordinary notice rather than an
        # exception: the same rule ``action`` follows, and the reason a new
        # level can be introduced from one side.
        return TOAST_LEVELS.get(self.level, TOAST_LEVELS["info"])

    def expired(self, now: float | None = None) -> bool:
        return (time.monotonic() if now is None else now) - self.born > self.ttl


@dataclass
class ManualState:
    """The Manual: which chapter it shows, where in it, and whether it is up.

    **This flag is deliberately the inversion of what this docstring used to
    say.** The Manual was a mode, and the argument was that a second "am I
    visible" flag would be a way for the two to disagree -- which was right
    while it *was* a mode. It is an overlay now (the UI redesign, wave 3): help is
    something you consult *about the screen you are on*, and making it a
    destination meant clicking a (?) replaced the pane whose control you were
    asking about, so the answer arrived with the question gone. There is no
    ``mode == "manual"`` left for this to disagree with; this is the only thing
    that decides.
    """

    open: bool = False
    chapter: str = "00-index"
    # Set alongside a navigation, consumed by the renderer on the frame the
    # heading is drawn -- scrolling needs a cursor position that only exists
    # mid-draw.
    pending_anchor: str | None = None
    # Which section of ``chapter`` the reader is currently looking at, so the
    # TOC tree can light that row. **Written by the renderer, not by
    # navigation** -- it is derived from the scroll position every frame
    # (``loader.active_anchor``), which is what makes the tree follow a reader
    # who scrolled rather than only one who clicked. ``pending_anchor`` is the
    # opposite direction, and the two are deliberately not one field: a
    # navigation says "put me there" exactly once, and this says "I am here"
    # continuously.
    anchor: str | None = None
    search: str = ""

    def open_at(self, chapter: str, anchor: str | None = None) -> None:
        self.chapter = chapter
        self.pending_anchor = anchor
        # Assigned unconditionally, both halves of which matter. A section
        # navigation lights that row on the frame it is clicked rather than on
        # the frame the scroll lands, because a highlight one frame behind the
        # click reads as a click that missed. A *chapter* navigation clears it,
        # so the section you were reading in the chapter you just left does not
        # stay lit under the one you arrived at; the renderer recomputes from
        # the new scroll position on the same frame.
        self.anchor = anchor


@dataclass
class TourState:
    """The guided tour: which one is running and how far through it is.

    Deliberately not a mode and not a modal. A tour points at real controls and
    waits for the reader to use them, so the app underneath has to stay live and
    clickable -- which is why nothing here reaches ``App._modal_open`` and why
    the overlay's scrim takes no input.

    ``index`` can outlive the tour it counted: it comes back from settings, and
    a tour that lost a step between releases would otherwise take the frame that
    restored it. ``Tour.step`` returns ``None`` past the end rather than
    raising, and :meth:`current` is where that is turned into "the tour is
    over".
    """

    #: Which tour, or ``""`` for none running.
    key: str = ""
    index: int = 0
    #: Tour keys the reader has reached the end of. Persisted, so the offer on
    #: Home can stop offering something already done.
    finished: tuple[str, ...] = ()
    #: Set for one frame when a step's condition is met, so the card can say so
    #: before it moves on rather than vanishing mid-sentence.
    satisfied: bool = False

    @property
    def running(self) -> bool:
        return bool(self.key)

    def start(self, key: str) -> None:
        self.key = key
        self.index = 0
        self.satisfied = False

    def stop(self) -> None:
        self.key = ""
        self.index = 0
        self.satisfied = False

    def complete(self) -> None:
        """End the running tour and remember that it was finished."""

        if self.key and self.key not in self.finished:
            self.finished = (*self.finished, self.key)
        self.stop()


@dataclass
class AppState:
    """The whole UI's mutable state."""

    # The one thing that decides what a pane shows, one of ``modes.KEYS`` --
    # which is the authoritative list, not restated here: a copy of it in this
    # comment went stale twice (it still said "2d | 3d"). It defaults
    # to the Home screen, which is what makes the chooser appear on every
    # launch rather than only the first ever; no mode is ever persisted
    # (``test_no_mode_is_persisted_anywhere`` pins that).
    mode: str = "home"
    # Where Esc goes from a mode you only pass through (Home, the Manual, app
    # Settings): back to the work you left, not to the chooser. ``mode_observed``
    # is the other half and exists because ``mode`` is assigned from a dozen
    # places (landing tiles, library cards, Inker's own open path) -- rather
    # than funnel all of them through a setter, ``App._note_mode`` samples the
    # pair once per key event, which is both the only place ``previous_mode``
    # is read and late enough to have seen a change made by a click, by a drop
    # or by F1 earlier in the very same frame. Neither is persisted: no mode is
    # (``test_no_mode_is_persisted_anywhere``), so a remembered one would be a
    # write with no reader across launches.
    previous_mode: str = "home"
    mode_observed: str = "home"
    selected: str | None = None
    # The job the library list should scroll into view on the next frame it is
    # drawn, set by arrow-key navigation and cleared by the card that answers
    # it. A one-shot rather than a standing position: a sticky value would drag
    # the list back every frame while the user is dragging the scrollbar.
    library_scroll_to: str | None = None
    # The asset currently being dragged, if any. imgui's Python drag-and-drop
    # payload is an integer, so the job id travels here instead; one drag is in
    # flight at a time by construction. See ``library.DRAG_JOB``.
    dragging_job: str | None = None
    # Which slot last received a drop, and when (H70). A dropped file used to
    # be acknowledged only by a toast in the far corner of the window, while
    # the control that actually changed -- in 2D, inside a section that is
    # collapsed by default -- showed nothing at all. See ``widgets.drop_flash``.
    drop_flash_slot: str = ""
    drop_flash_at: float = 0.0
    # The little search boxes over the panel lists (J86), keyed by pane tag.
    # One dict rather than a field per pane: they are all the same control with
    # the same lifetime, and five near-identical fields is five places to
    # forget to clear. Not persisted -- a filter you cannot see the box for,
    # because the list it belongs to is short today, would be a panel that
    # silently hides half its contents on launch.
    list_filters: dict[str, str] = field(default_factory=dict)
    # Which form control the last refusal named, and what it said (UX.md Phase
    # 3). ``service.errors.ServiceError`` has carried a ``field`` since it was
    # written, documented as "the UI highlights it", and nothing in ``studio/``
    # read it: a refusal about the seed and a refusal about the style LoRA both
    # arrived as the same red toast in the far corner of the window, and
    # finding which of fifteen controls it meant was the reader's problem.
    #
    # A dict rather than one slot because a single submit can only be refused
    # once but the two generate panes are both live, and keying by field name
    # is what lets the *control* ask ("is anything wrong with me") rather than
    # the pane switch on a remembered name. Not persisted: a refusal describes
    # a submit, and a submit does not survive the session that made it.
    field_errors: dict[str, str] = field(default_factory=dict)
    # ``ServiceError.rows`` for the same refusals, and what installing exactly
    # those costs. Carried beside the message rather than parsed back out of
    # it, which is the reason the service field exists at all: the offer under
    # the ring pre-ticks exact registry rows.
    field_error_rows: dict[str, tuple[str, ...]] = field(default_factory=dict)
    field_error_gib: dict[str, float] = field(default_factory=dict)
    # The keyboard focus ring (UX.md Phase 3), per pane: which control the
    # cursor is on, what this frame's tab order is, and whether the cursor just
    # moved -- see :mod:`.focus`, which owns every rule about them. Here rather
    # than on the panes because a pane is a module of functions with no
    # instance to hang state on, which is the same reason ``list_filters`` is
    # here. Not persisted: a focus ring describes where somebody's hands are.
    focus_key: dict[str, str] = field(default_factory=dict)
    focus_order: dict[str, list[str]] = field(default_factory=dict)
    focus_moved: bool = False
    # One-shot: "somebody asked for the shortcuts list this frame". A flag
    # rather than a call because ``imgui.open_popup`` names a popup registered
    # inside the *current* window, so neither the Ctrl+/ handler nor the
    # palette's command can open it directly -- the header consumes this where
    # the popup actually lives. Cleared on read, so a request that arrives
    # while the header is not drawn is dropped rather than firing later.
    shortcuts_requested: bool = False
    # What a previous session left, scanned once on the first frame and offered
    # by the home screen. ``None`` means "not looked yet" and is distinct from
    # an empty list: the autosave directory is *also* where this session's own
    # copies land, so a rescan would list the user's open documents back at them
    # as though they had been crashed out of. ``journal.snapshot`` is the only
    # thing that fills it; ``journal.take``/``discard`` are the only things that
    # shorten it. Typed loosely because ``state`` is imported by everything and
    # must not pull the journal in with it.
    recovery: list[Any] | None = None
    # Set when the UI scale changes, consumed by the frame loop *between*
    # frames (K99): the atlas rebuild invalidates every ImFont handle, and a
    # rebuild inside a frame would leave the rest of it drawing through freed
    # pointers.
    fonts_dirty: bool = False
    # Whether the pose editor is holding rotations nobody has saved. Written
    # only from ``Viewer.on_pose_dirty`` -- so it costs a frame nothing, and it
    # is a mirror rather than the authority: ``pose_panel.guard`` still asks
    # the editor itself, because a guard that trusted a cached bool would let
    # unsaved work through if one write were ever missed. What this is for is
    # the *indicator*, which has to be visible from outside the pose pane.
    pose_dirty: bool = False
    # Which Resume row the keyboard is on. Not persisted: the app opens on Home
    # every launch, and a remembered cursor would put Enter on whichever row
    # was last hovered a week ago -- which, now that the list is derived from
    # what has actually been touched, is not even the same row.
    home_index: int = 0
    # How many finished meshes nobody has judged, for Home's status block, and
    # when it was last asked. A count, not a list: the question is "is there a
    # review pass waiting", and Review itself owns the list. ``None`` means
    # nothing has come back yet, which draws as nothing rather than as zero --
    # "0 unreviewed" and "not asked yet" are different sentences.
    #
    # A cached answer refreshed on a cadence rather than read on the frame
    # thread, because it is a table scan behind the one serialized connection.
    home_unreviewed: int | None = None
    home_unreviewed_at: float = 0.0
    # The command palette (I80). Three plain fields rather than a state object:
    # it holds a query, a cursor and whether it is up, and nothing about it
    # survives being closed -- reopening on the last query would make Ctrl+K
    # act on a search the user has forgotten typing.
    palette_open: bool = False
    palette_query: str = ""
    palette_index: int = 0
    comparing: str | None = None
    # The other side of a comparison, and the parse that is on its way.
    #
    # ``compare_baseline`` is recorded when the context menu *opens*, because
    # right-clicking a card selects it first -- so reading ``selected`` at the
    # moment the menu item is clicked gives the target, and "compare with
    # selected" compared a mesh with itself (UX-04).
    #
    # ``compare_pending`` is the compare half of ``viewer.pending``: the parse
    # runs on a task thread and the result is checked against this before it is
    # adopted, so a selection that moves while a large GLB is being read cannot
    # drop a stale mesh into the right-hand pane.
    compare_baseline: str | None = None
    compare_pending: Any = None
    # Which asset the inspector's tag toggles are staged against, and what is
    # staged. Keyed by job id rather than being a bare list, so selecting a
    # different asset drops them -- staged state belongs to the thing that was
    # on screen when it was chosen, exactly as Review's ``pending_tags`` does,
    # and a tag surviving a selection change is filed against a mesh nobody was
    # looking at when they picked it. The two fields move together and are
    # cleared together; there is no arming any more, because a grade is one
    # keypress and a tag is optional at every grade.
    inspector_tags_job: str | None = None
    inspector_tags: list[str] = field(default_factory=list)
    form_2d: dict[str, Any] = field(default_factory=default_form_2d)
    form_3d: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FORM_3D))
    filters: Filters = field(default_factory=Filters)
    history: list[str] = field(default_factory=list)
    checked: set[str] = field(default_factory=set)
    toasts: list[Toast] = field(default_factory=list)
    # The composed-prompt preview, refreshed off-thread as the prompt is typed.
    preview: dict[str, Any] = field(default_factory=dict)
    preview_dirty_at: float = 0.0
    # The frame-rate overlay (F10). Persisted, because someone watching for a
    # stall wants it to survive the restart they are about to do. It is the
    # *detailed* view -- mean and worst frame -- and the always-on strip beside
    # the mode switch is the summary; see ``overlay.fps_meter``.
    show_fps: bool = False
    # The status bar's live VRAM / RAM / CPU meter. Persisted beside the
    # frame-rate readout above and, like it, **opt-out rather than a fixture**:
    # a permanent resource readout was deleted from the shell header in wave 3
    # as developer chrome, and the four grounds on which this one is different
    # are recorded in ``tests/test_ux_phases.py``. Default on, because the
    # question it answers -- "can I start a 7 GB generation right now" -- is
    # one the app forces on the user at the door (``check_vram``) and nothing
    # on screen answered.
    show_resources: bool = True
    # Suppress every animation in the app (accessibility: vestibular
    # sensitivity). Persisted, because it is a property of the person rather
    # than of the session. The flag itself lives in ``motion.REDUCED`` -- this
    # is what the checkbox binds to and what the settings file remembers, and
    # the two are kept in step by the one function that sets both
    # (``app_settings._apply_reduce_motion``).
    reduce_motion: bool = False
    wireframe: bool = False
    turntable: bool = False
    # Draw a tile repeated in the 2D viewport rather than once. Off by default,
    # and that is a decision rather than an oversight: every other view of an
    # asset in this app -- the thumbnail, the exports, the Inker -- shows one
    # cell, so a viewport that silently showed four would make the texture look
    # a quarter of its size. What repetition answers that nothing else does is
    # whether the pattern *reads* as repeating, which is a question the user
    # asks deliberately; the seam question already has the wrapped view.
    tile_preview: bool = False
    source_job: str | None = None  # the 2D asset the 3D pane starts from
    # Where the user is standing in Create mode, one of
    # ``create_stages.STAGES`` (the UI redesign, wave 5). **Volatile and never
    # persisted**, which is the same rule ``mode`` follows and for a stronger
    # reason: a stage is a *derived* position over the selected asset, so a
    # remembered one would be restored against whatever happens to be selected
    # next launch and would routinely name a stage that asset has not reached.
    # Written only by ``create_stages.go`` -- the one switch.
    create_stage: str = "reference"
    # The Simple/Advanced disclosure is workspace state, not a creative
    # setting. It deliberately resets each launch and is never persisted.
    create_advanced: bool = False
    # Every outstanding failure the banner is showing, oldest first. A list
    # rather than the single ``last_error`` slot it replaces: three writers
    # (a failed doctor check, a dead worker, a worker that never started) all
    # assigned to that slot, so a launch that failed two checks reported
    # whichever wrote last and lost the other with nothing to say it had.
    errors: list[str] = field(default_factory=list)
    # What Dismiss took off screen, kept rather than dropped (F59). Every writer
    # of ``errors`` fires exactly once -- the startup doctor sweep, and the two
    # one-shot worker checks -- so clearing the list destroyed the only copy of
    # the text, and the sole remaining trace of a launch that failed two checks
    # was a coloured dot. Keeping these makes Dismiss "put it away" rather than
    # "forget it" for the session log and support tooling.
    dismissed_errors: list[str] = field(default_factory=list)
    # Whether the style-profile manager is up (the UI redesign, wave 3). It was a
    # mode, which put a shelf of saved settings in the top-level navigation
    # beside the six creative workspaces -- and made "manage my styles" a place
    # you travel to rather than something you do to the form in front of you.
    # It is a sheet over the 2D pane now, opened from the profile picker it is
    # about. Not persisted, for the reason no mode is: reopening the app into a
    # manager nobody asked for is a setting nobody chose.
    profiles_open: bool = False
    profile_draft: dict[str, Any] | None = None
    profile_draft_name: str = ""
    # The name the draft was opened under, so renaming one in the editor moves
    # it rather than leaving the old name behind as a duplicate.
    profile_draft_origin: str = ""
    # The journal's three marks for the open draft (UX-05). Here rather than on
    # the draft dict because that dict is replaced wholesale every time a
    # different profile is opened, and a mark travelling with it would mint a
    # new crash-copy filename per open.
    profile_journal_name: str = ""
    profile_journal_head: Any = None
    profile_journal_at: float = 0.0
    # Inker mode's open documents and tool settings, built on first use.
    # Typed Any so state.py keeps no import of the editor or of Pillow, and
    # lazy so a session that never draws pays nothing for it.
    inker: Any = None
    # Clay's own multi-document state, built on first use by
    # ``clay_mode.ensure``. Untyped and None here for the reason ``inker`` is:
    # AppState is the shared frame state and deliberately knows nothing about
    # what a mode keeps, so a session that never opens Clay pays nothing.
    clay: Any = None
    # Review mode's sweep runs and where in one it is, built on first use by
    # ``review_mode.ensure`` and untyped here for the reason ``clay`` is.
    # Nothing in it is persisted: a stored run directory would outlive the
    # sweep it names.
    review: Any = None
    # Plotter's open maps and tool settings, and Packwright's open atlases,
    # both built on first use by their modes' ``ensure``. Untyped and None for
    # the reason ``inker`` and ``clay`` are: AppState is the shared frame state
    # and deliberately knows nothing about what a mode keeps.
    plotter: Any = None
    packwright: Any = None
    # Sirens' open songs, its caret and the last audio each tab rendered to,
    # built on first use by ``sirens_mode.ensure``. Untyped and None for the
    # reason the four above are.
    sirens: Any = None
    # Troupe's selection and its preview clock, built on first use by
    # ``troupe_mode.ensure``. Untyped and None for the reason the others are --
    # and the only one of them that holds no document: it is a selection over
    # sheets a worker published, so there is nothing here to lose.
    troupe: Any = None
    # Poser's authoring session -- which template, what the library holds --
    # built on first use by ``poser_mode.ensure``. Untyped and None for the
    # reason the four above are. Its Viewer lives on the App/Ctx, not here:
    # AppState carries no GL objects.
    poser: Any = None
    # Whether ``findings.json`` is behind the evidence in the DB. A flag rather
    # than a submit, because ``TaskRunner.submit`` *refuses* a key already in
    # flight and nothing re-arms it: five verdicts in a second used to run one
    # recompute over the set as it stood at the first press and silently drop
    # the rest until the next verdict. It is also what lets the worker's
    # observations reach the file at all -- they are appended off the frame
    # thread by a component that cannot submit anything, so the frame loop
    # notices the job finishing and marks this instead.
    findings_dirty: bool = False
    # Which probe question needs retraining, or None. The ``findings_dirty``
    # pattern exactly, and for the identical reason -- ``TaskRunner.submit``
    # refuses a key already in flight and nothing re-arms it, so a burst of
    # labels would train once on the set as it stood at the first press and drop
    # the rest. A stage string rather than a bool because a labelling pass is
    # about one question at a time, and the training run needs to know which.
    judge_dirty: str | None = None
    # Whether the open review's units need scoring by the judge. The same flag
    # pattern for the third time, and the third time for the same reason: a
    # score request follows every scan and every retrain, and a direct submit
    # would drop whichever one arrived while the previous run was still going.
    review_scores_dirty: bool = False
    # The promote flow's matte preview and its cutout cache, built on first use
    # by ``matte_preview.ensure``. Untyped and None here for the reason
    # ``clay``/``review`` are, and never persisted: a stored cutout would be a
    # claim about a file that has had a whole session to change.
    matte: Any = None
    manual: ManualState = field(default_factory=ManualState)
    tour: TourState = field(default_factory=TourState)
    # The selected asset's parsed manifest.json, held as ((job id, mtime), data)
    # so the Export tab reads and parses it once per version of the file rather
    # than once per frame. The same (id, mtime) idiom ThumbnailCache uses, for
    # the same reason: the stat that decides whether to re-read is the cheap
    # half, and a derivation rewrites the manifest under a tab that is open.
    # One slot, because one asset is inspected at a time.
    manifest: Any = None
    # The palette directory's listing, held as (mtime, [names]) for exactly the
    # reason above: the combo is drawn every frame and a directory walk per
    # frame is a syscall storm for a folder that changes when a user drops a
    # file in it. A palette added while the app runs appears on the next frame,
    # because dropping a file moves the directory's own mtime.
    palettes: Any = None
    # The selected asset's ``rig.json``, as ``{job id: (mtime stamp, data)}``.
    # A rig is recorded on the *rig* job's row but written into the **source**
    # job's directory, and the Rig & Pose tab opens on that source job -- so the
    # file is the only place the selection on screen can learn how its own mesh
    # was bound. Read on the frame thread and therefore cached, under the same
    # racily-clean rule ``files.attach_files`` documents: a re-rig lands inside
    # a directory whose mtime may not move, so a stamp is only remembered once
    # it is safely in the past. Keyed by job rather than one slot, because the
    # inspector is not the only reader and a one-slot cache thrashes the moment
    # two are alive. Never persisted: it describes a file, not a preference.
    rig_meta_cache: dict[str, Any] = field(default_factory=dict)

    # -- toasts ------------------------------------------------------------

    def toast(
        self,
        text: str,
        level: str = "info",
        action: str | None = None,
        action_arg: str | None = None,
    ) -> None:
        entry = Toast(text=text, level=level, action=action, action_arg=action_arg)
        self.toasts.append(entry)

    def toast_once(
        self,
        text: str,
        level: str = "info",
        action: str | None = None,
        action_arg: str | None = None,
    ) -> bool:
        """:meth:`toast`, unless an identical toast is already on screen.

        For the warnings a gesture can repeat: painting under a hidden layer
        says one sentence per press, and a multi-click shape tool presses once
        per vertex, so the unconditional append stacked one copy per click for
        as long as the user kept clicking. Identity is text plus level over the
        *live* toasts only -- once the copy on screen has expired the next
        press earns a fresh one, which is what keeps this a coalesce rather
        than a once-per-session gag. -> whether a toast was raised.
        """
        now = time.monotonic()
        for entry in self.toasts:
            if entry.text == text and entry.level == level and not entry.expired(now):
                return False
        self.toast(text, level, action, action_arg)
        return True

    def expire_toasts(self) -> None:
        now = time.monotonic()
        self.toasts = [t for t in self.toasts if not t.expired(now)]

    # -- the error banner --------------------------------------------------

    def note_error(self, text: str | None) -> None:
        """Add a failure to the banner, once.

        Deduplicated because the conditions are re-derived rather than edged:
        a doctor check that is still failing is reported again on the next
        refresh, and a banner that grew a line each time would bury the first.
        """
        text = (text or "").strip()
        if text and text not in self.errors:
            self.errors.append(text)

    def dismiss_errors(self) -> None:
        """Off the banner, into Settings -> Health -- never out of existence (F59)."""
        self.dismissed_errors.extend(m for m in self.errors if m not in self.dismissed_errors)
        self.errors.clear()

    @property
    def error_text(self) -> str:
        """Every outstanding failure as one block -- what Copy details copies."""
        return "\n".join(self.errors)

    # -- field-level refusals ----------------------------------------------

    def note_field_error(
        self,
        field_name: str,
        message: str,
        rows: tuple[str, ...] = (),
        gib: float = 0.0,
    ) -> bool:
        """Remember that ``field_name`` is why the last submit was refused.

        -> whether it was recorded. A refusal that names no control is not one
        of these and must keep going to the toast, which is the whole of what
        the caller does with the answer.
        """
        if not field_name or not message:
            return False
        self.field_errors[str(field_name)] = str(message)
        # Only when there is something to offer: an empty tuple must not
        # displace rows a previous refusal on the same control recorded and
        # then leave a button that installs nothing.
        if rows:
            self.field_error_rows[str(field_name)] = tuple(rows)
            self.field_error_gib[str(field_name)] = float(gib)
        else:
            self.field_error_rows.pop(str(field_name), None)
            self.field_error_gib.pop(str(field_name), None)
        return True

    def clear_field_error(self, field_name: str) -> None:
        """Forget one, because the user has just changed it.

        Editing the control the refusal named is the clearest possible "I have
        dealt with that", and a ring that outlived it would be an app arguing
        about a value that is no longer there.
        """
        self.field_errors.pop(field_name, None)
        self.field_error_rows.pop(field_name, None)
        self.field_error_gib.pop(field_name, None)

    def clear_field_errors(self) -> None:
        """Forget all of them: a new submit is about to be judged on its own."""
        self.field_errors.clear()
        self.field_error_rows.clear()
        self.field_error_gib.clear()

    # -- history -----------------------------------------------------------

    def remember_prompt(self, prompt: str) -> None:
        """Most recent first, deduplicated, bounded."""
        prompt = (prompt or "").strip()
        if not prompt:
            return
        self.history = [prompt] + [p for p in self.history if p != prompt]
        del self.history[MAX_HISTORY:]

    # -- selection ---------------------------------------------------------

    def select(self, job_id: str | None) -> None:
        if job_id != self.selected:
            self.selected = job_id
            # A comparison is between the selection and something else; keeping
            # it across a selection change would compare two jobs neither of
            # which the user just clicked.
            self.comparing = None
            # And the parse in flight for it is no longer wanted. Left set, a
            # result landing after the selection moved would be adopted into a
            # comparison that no longer exists.
            self.compare_pending = None

    def toggle_check(self, job_id: str) -> None:
        self.checked.symmetric_difference_update({job_id})


# --- the primary-action ladder ----------------------------------------------
#
# Ported from the browser's card renderer. One button per card, and which one
# is a function of what the job *is*, in a fixed order of precedence -- the
# point being that the obvious next step is always the one on offer.

ACTIONS = {
    "cancel": "Cancel",
    "retry": "Try again",
    "promote": "Make 3D",
    "rig": "Rig",
    "open": "Open",
    "inker": "Open in Inker",
    "clay": "Open in Clay",
    "plotter": "Add to Plotter",
    "variation": "Variation",
    "revise": "Revise with prompt",
    "settings": "Use settings",
    "reference": "Use as reference",
}


def result_actions(job: dict[str, Any], *, rigging_available: bool = True) -> tuple[str, ...]:
    """Actions a finished result may expose in its result surface.

    This is intentionally a pure policy function.  The card and a future
    result viewer can render the same actions without each inventing a subtly
    different set of valid follow-ups.
    """
    if job.get("status") != "done":
        return ()
    files = set(job.get("files") or ())
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    actions: list[str] = []
    if "input.png" in files:
        actions.extend(("variation", "revise", "settings", "reference"))
    actions.append(primary_action(job, rigging_available=rigging_available) or "open")
    if params.get("asset_intent") == "tileset" or job.get("kind") == "tile_sheet":
        actions.append("plotter")
    elif "model.glb" in files and rigging_available:
        actions.append("clay")
    # Preserve order while avoiding a duplicate primary action.
    return tuple(dict.fromkeys(actions))


def primary_action(job: dict[str, Any], *, rigging_available: bool = True) -> str | None:
    """The one action a job's card offers, or None."""
    status = job.get("status")
    if status in ("queued", "running"):
        return "cancel"
    # "Open" for a finished tile sheet, which is the answer a tile gets further
    # down for the same reason: the deliverable *is* the published image, and
    # Open selects the row and shows the Export tab. Without this arm the ladder
    # walks past the stage checks, finds no ``model.glb`` and offers a finished
    # sheet no action at all.
    #
    # This kind *is* rerollable -- the whole request is a prompt and a seed, so
    # "draw me another" is exactly meaningful and ``rerun_job`` allows it. So
    # the ``status == "error" -> retry`` default below is the right answer here,
    # and this arm deliberately answers only the one case it owns rather than
    # returning for the kind. A finished sheet with no ``input.png`` falls
    # through the rest of the ladder to None, which is the same answer by a
    # longer road -- ``stage`` is "tilesheet", so neither image arm claims it
    # and there is no ``model.glb``.
    if (
        job.get("kind") == "tile_sheet"
        and status == "done"
        and "input.png" in (job.get("files") or [])
    ):
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        return "plotter" if params.get("asset_intent") == "tileset" else "open"
    if status == "error":
        return "retry"
    if status != "done":
        return None
    files = job.get("files") or []
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    intent = params.get("asset_intent")
    if intent == "tileset" and "input.png" in files:
        return "plotter"
    if intent in ("refine_2d", "sprite") and "input.png" in files:
        return "inker"
    if (
        intent == "reconstruct_3d"
        and job.get("stage") == "reference"
        and "input.png" in files
    ):
        # The reference is explicitly an approval/editing stage. Make 3D is
        # still available from Create after the image has been inspected; the
        # library card's contextual next action is the editor beside it.
        return "inker"
    if job.get("stage") == "tile":
        # No mesh, no rig: a tile's next step is to be exported, which the
        # inspector's Export tab is. "Open" selects it and shows that tab.
        return "inker" if "input.png" in files and intent == "refine_2d" else (
            "open" if "input.png" in files else None
        )
    if job.get("stage") == "reference":
        # A finished reference's next step is the mesh it exists for.
        return "promote" if "input.png" in files else None
    if "model.glb" not in files:
        return None
    if intent == "reconstruct_3d":
        return "clay"
    if rigging_available and job.get("kind") != "rig" and "rig.glb" not in files:
        return "rig"
    return "open"


# --- progress ---------------------------------------------------------------


class Eta:
    """A smoothed estimate of how much longer the running job has.

    Exponentially weighted (0.7 old, 0.3 new) because the raw estimate jumps
    every time a stage boundary changes the rate, and a number that flickers
    between "2 min" and "20 s" is worse than none. Suppressed until the job is
    warm and meaningfully underway, for the same reason: an estimate from 3%
    of a cold start is a guess about the weights loading.
    """

    ALPHA = 0.3

    def __init__(self) -> None:
        self.value: float | None = None
        self._job: str | None = None

    def update(
        self, job_id: str | None, percent: float, elapsed: float, cold: bool
    ) -> float | None:
        if job_id != self._job:
            self._job, self.value = job_id, None
        if job_id is None or cold or not (10.0 <= percent < 100.0) or elapsed <= 5.0:
            return None
        remaining = elapsed * (100.0 - percent) / percent
        self.value = (
            remaining
            if self.value is None
            else (1 - self.ALPHA) * self.value + self.ALPHA * remaining
        )
        return self.value


def format_duration(seconds: float | None) -> str:
    """A clock a person reads at a glance: 45s, 2m 10s, 1h 04m."""
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def format_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(count) < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"

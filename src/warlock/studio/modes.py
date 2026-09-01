"""The top-level modes, as data.

One list, in the order the switch draws them. Deliberately data and nothing
else -- the dispatch that turns ``state.mode`` into a pane stays hand-coded in
:mod:`.main`, because "``state.mode`` is the only thing that decides what a
pane shows" is only true while there is exactly one place doing the deciding.
A table of callbacks here would be a second one.

The module imports :mod:`.icons` and nothing else, so it stays importable from
anywhere without dragging imgui in.
"""

from __future__ import annotations

from . import icons

# (key, label, icon). The key is what lands in ``AppState.mode``.
#
# **The order is the rail's order** (the UI redesign, wave 3): where you start and
# what you look at, then the seven creative workspaces, then Settings. It used to
# be the *segmented control's* order, grouped by a predicate over
# ``WORK_MODES`` -- a rule that rendered correctly and explained nothing, and
# which put Library and Review on the far side of a break from the panes they
# are about. The grouping is written out in ``RAIL_GROUPS`` now, and this list
# is the flattening of it (a test asserts exactly that), so the two cannot
# drift while remaining two spellings of one fact.
#
# Two modes left this list in wave 3. ``manual`` is still here in spirit: help
# is consulted *about* a screen, so taking that screen away to show it answered
# the question by removing it -- it is ``manual.render.draw_overlay`` now,
# raised by F1 and by every ``help_button``. ``profiles`` was a shelf of saved
# settings in the top-level navigation beside seven creative workspaces, which
# said that "manage my styles" is a place you travel to; it became a sheet over
# the Reference stage and was then deleted outright -- a recipe is copied off a
# finished result, not curated in a second store.
MODES: list[tuple[str, str, str]] = [
    ("home", "Home", icons.HOUSE),
    # A real mode rather than a sub-view of Home. The Library and Review were
    # tiles on the chooser and a ``state.landing_view`` enum behind it, which
    # is what a destination looks like when there is nowhere to put it; Home
    # stopped being a tile grid, so they went where everything else already
    # was. The glyph is the one ``landing._SUBVIEW_ICONS`` already assigned --
    # moved, not re-picked, because a screen the user has seen should not
    # change its pictures for a refactor.
    #
    # It sits *before* Create because that is the order of the question: what
    # do I have, then make another one.
    ("library", "Library", icons.FOLDER_OPEN),
    # **One mode, not two** (the UI redesign, wave 5). "2D" and "3D" were the two
    # halves of a single journey -- you write a prompt, you get a picture, you
    # turn the picture into a mesh -- presented as two destinations you had to
    # know to travel between. Worse, the names described the *artifact* rather
    # than the act: a user who wants a barrel does not first decide to do some
    # 2D. The halves are stages of Create now (``create_stages.STAGES``), drawn
    # as a rail above the settings column, and which one you are on is a
    # property of the asset in front of you rather than a place in the
    # navigation. The glyph is neither of the two it replaces, deliberately:
    # IMAGE and BOX went with the stages that kept their meanings.
    ("create", "Create", icons.SPARKLES),
    ("inker", "Inker", icons.PEN_TOOL),
    ("clay", "Clay", icons.RULER),
    ("poser", "Poser", icons.PERSON_STANDING),
    # Troupe (the Troupe programme's own mode). A workspace of its own rather
    # than a panel in Create for the reason Poser is one: what happens here is
    # *watching* -- a walk cycle plays continuously and you judge it -- and
    # that is a use of the whole window, not of a 300px column beside a form.
    # The glyph is Poser's, deliberately: both are about a human figure, and
    # the rail distinguishes them by label.
    ("troupe", "Troupe", icons.PERSON_STANDING),
    ("plotter", "Plotter", icons.GRID),
    ("packwright", "Packwright", icons.LAYERS),
    # Sirens: the twelfth mode, and the first thing in this app that makes a
    # sound. A workspace rather than a stage of Create for the reason Plotter
    # and Packwright are: it owns a document type (``.wsng``), it has its own
    # tabs and its own undo stack, and nothing it produces is a job row. It
    # sits at the end of the Workspaces group because it is the newest and
    # because the group has no other ordering to respect -- the six before it
    # are not a pipeline.
    ("sirens", "Sirens", icons.AUDIO_WAVEFORM),
    # Review is footer matter, beside Settings, and shares its glyph history
    # with the Library above (both were Home tiles). It is the one place you
    # go to *judge* rather than to make, and it is entered rarely and left
    # again -- which is the same shape as Settings and not the shape of the
    # seven workspaces it used to sit among.
    ("review", "Review", icons.CIRCLE_CHECK),
    ("settings", "Settings", icons.SETTINGS),
]

# The rail's sections, hand-written. **Not derived**, and that is the reversal
# of what ``GROUP_BREAKS`` used to be: deriving the gaps from "is this a work
# mode" was right while the only claim being made was *workspaces are not
# places*, and it is wrong now, because no predicate over ``WORK_MODES`` can
# derive this grouping: the first group is one work mode and two that are not,
# and the footer is one of each. Home, the Library and Create are where an
# asset *begins* -- what do I have, and make another one -- and that is a
# claim about the user's question, not about whether a pane has a form in it.
# A rule that cannot state the grouping is not a better version of stating it.
#
# The last group is the rail's *footer*, and now that it holds two items it
# needs a meaning: it is the end matter -- the two destinations where you are
# not making something. Review is where you judge what came out and Settings
# is where you configure the machine; both are entered rarely and left again.
# It used to be drawn against the bottom edge beside a health badge and an
# expand toggle; the editor shell moved both of those to the status bar and
# the Window menu, so the footer is now simply the last group in the one
# column -- distinguished by carrying no caption rather than by a separate
# drawing path.
RAIL_GROUPS: tuple[tuple[str, ...], ...] = (
    ("home", "library", "create"),
    ("inker", "clay", "poser", "troupe", "plotter", "packwright", "sirens"),
    ("review", "settings"),
)

#: What each group is called, when the rail is wide enough to say so. One entry
#: per group in :data:`RAIL_GROUPS`, footer included -- its own is the empty
#: string and is never drawn, because a caption over a single item is a label
#: for a label. ``rail._caption`` returns on an empty label rather than drawing
#: nothing in a full row, and ``rail.draw``'s offset arithmetic skips the row
#: for the same groups: the two loops have to agree or the selection pill lands
#: a caption-height off the item it names.
#:
#: The grouping above is a *claim* ("these four are one pipeline;
#: these seven are workspaces") and until these existed the only thing asserting
#: it was a gap, which at a glance reads as an accident of spacing.
RAIL_GROUP_LABELS: tuple[str, ...] = ("Pipeline", "Workspaces", "")

# The modes that own a viewport or a form, and so have work in them. Home, the
# Manual and Settings are places you pass through: they have no form to
# submit and no viewport to frame, which is why they take no keyboard
# shortcuts at all.
WORK_MODES = frozenset(
    {"create", "inker", "clay", "poser", "review", "plotter", "packwright", "troupe", "sirens"}
)

# The subset that draws the *asset* viewport, and therefore the only modes
# whose selection is worth loading a mesh for. One member since wave 5, and
# still a set rather than an ``== "create"``: it is the *question* "does this
# mode frame the selected asset" and the answer has been one, two and one
# again. Which of Create's stages a viewport shows is a second question, asked
# of ``create_stages`` -- see ``_sync_viewer``.
#
# Inker and Clay each own their
# own centre pane -- Clay's draws a live document rather than a file, so
# ``_sync_viewer`` has nothing to do for it and returns early. Review is not
# here either, and deliberately: it *borrows* the shared viewer, but for a
# sweep unit's mesh rather than for the library selection, so leaving it out is
# what stops ``_sync_viewer`` reloading the selected asset over it.
VIEWPORT_MODES = frozenset({"create"})

# Neither one pane nor the asset viewport: a mode that fills the window with
# its own three-column workspace. Inker, Clay, Poser, Review, Plotter,
# Packwright, Troupe and Sirens are the eight; Library and Profiles are single
# panes, not workspaces, and join Home/Manual/Settings there. The three categories
# partition KEYS exactly -- which matters because ``_build_ui``'s dispatch ends
# in a bare ``else``, so an unlisted mode would draw one of these rather than
# fail.
WORKSPACE_MODES = frozenset(
    {"inker", "clay", "poser", "review", "plotter", "packwright", "troupe", "sirens"}
)

# The modes that bind the arrow keys or Space themselves, and so keep them from
# imgui's keyboard navigation (UX-02). Home and the Library move a selection
# with Up/Down, Review steps units with Left/Right, and Inker and Plotter hold
# Space to pan -- for all five, one press must not also step a focus ring.
#
# The rule is stated in ``imgui_backend._NAV_KEYS``: Tab traverses everywhere,
# the arrows belong to the surface. This is the "which surface" half, listed
# here beside the other mode groupings rather than inside the backend, because
# it is a fact about the modes and not about the input door.
# Troupe joins them for its own version of the same clash: Space toggles
# playback and Left/Right step one frame of a clip, so one press must not also
# move a focus ring through the direction buttons.
# Sirens joins them for the sharpest version of the clash: all four arrows move
# the pattern caret, Space starts and stops playback, and Page Up/Down move a
# bar -- so every key imgui would use to walk a focus ring is a key the grid
# has already spoken for.
NAV_KEY_MODES = frozenset(
    {"home", "library", "review", "inker", "plotter", "troupe", "sirens"}
)

KEYS = tuple(key for key, _label, _icon in MODES)

#: One line saying what each mode is *for*, shown as the rail item's tooltip.
#:
#: The rail is the primary navigation and seven of its twelve labels -- Inker,
#: Clay, Poser, Troupe, Plotter, Packwright, Sirens -- are invented names. A new user
#: hovering one got a word and an icon, because ``rail._item`` suppresses its
#: accessible-name tooltip once the label is legible (correctly: a tooltip
#: repeating a word already on screen is noise) and no call site had anything
#: more to say. This is the something more.
#:
#: A table beside ``MODES`` rather than a fourth element of it, so that the
#: order-and-grouping tuple stays the thing every reader already knows, and a
#: mode with nothing useful to add can simply be absent.
PURPOSE: dict[str, str] = {
    "home": "Start here: recent work, what needs attention, and what to make next.",
    "library": "Every asset you have made, searchable and filterable.",
    "create": "Prompt to reference image to 3D model — the main pipeline.",
    "inker": "Pixel-art and image editor: layers, animation, tilesets.",
    "clay": "Assemble and edit meshes from primitives and booleans.",
    "poser": "Rig a mesh to a skeleton and author animation clips.",
    "troupe": "Render a 3D character to a 256-cell sprite sheet.",
    "plotter": "Paint tile maps and export them to Tiled.",
    "packwright": "Pack loose sprites into an atlas with a manifest.",
    "sirens": "Write chiptune music and sound effects in a tracker grid.",
    "review": "Judge and grade finished assets side by side.",
    "settings": "Models, folders, appearance and hardware.",
}

#: Modes whose maturity the rail says out loud, and the word it uses.
#:
#: **Troupe is code-complete and a user really can get a rendered sheet**, but
#: three of its own phases are unstarted, its 22 keyframes are provisional, and
#: its palette claim rests on a textured base mesh that does not exist.
#: ``docs/manual/11`` and ``33`` are candid about every bit of that -- and the
#: app was not, which is the gap this closes: the manual is read by people who
#: already know to be careful, and the rail is read by everyone.
#:
#: The wording is the manual's own ("provisional", "untested"), deliberately,
#: so a user who follows the tooltip into the chapter finds the same words
#: rather than a second, differently-hedged account.
# **Sirens' chip was narrowed at every landing and it did not come off at the
# last one.** It went on because the mode shipped in halves and the note was
# rewritten each time a gap closed, because a chip whose sentence names what is
# already there teaches the reader to ignore the chip. By phase 5 the note said
# only "no manual chapter or guided tour", and both of those now exist
# (``docs/manual/14`` and ``34``, and the ``sirens-basics`` tour) -- but writing
# the chapter is what found the gap that was never in the note at all: **four of
# a cell's five columns cannot be typed into.** ``sirens_mode.handle_key``
# answers the piano row and the note-off in column 0 and nothing anywhere else,
# so the instrument, volume, effect and parameter columns are drawn, are
# reachable with Left/Right, and take no input; ``write_cell`` has exactly two
# callers and both pass ``column=0``. Every effect the synth implements --
# ``Fxx``, ``Bxx``, the slides, the arpeggio -- is therefore unreachable from
# the UI, and so is ``notes.NOTE_RELEASE``: the backtick writes ``NOTE_OFF``,
# which cuts, so an instrument's release half is reachable mid-song only from a
# document that already carries a ``~~~``. That is a bigger absence than either
# of the two the note used to name, and a chip removed on the same day it was
# found would be the badge doing the opposite of its job.
MATURITY: dict[str, str] = {
    "troupe": "Experimental",
    "sirens": "Experimental",
}

#: What the chip's own tooltip adds, past the word.
MATURITY_NOTE: dict[str, str] = {
    "troupe": (
        "The chain runs end to end, but the shipped animation keyframes are"
        " provisional and humanoid reconstruction quality is untested."
        " See the manual (Troupe)."
    ),
    # Narrowed rather than removed at every landing, and narrowest here. The
    # sound-effect pane, the WAV export, the manual chapter, the tour and -- as
    # of the sixth landing -- a keyboard for all five columns exist now. What
    # is left is one thing and it is nameable in a sentence, which is the bar
    # this note has been held to since the second landing.
    "sirens": (
        "Every column can be typed into and every effect played, but a block"
        " selection can only be transposed or cleared: there is no copy and no"
        " paste, so a repeated bar is retyped. See the manual (Sirens)."
    ),
}

# **There is no positional Alt+digit binding, and there deliberately is not.**
# It existed while there were ten modes and ten digits, on the argument that the
# binding was the picture on screen rather than a second table. That argument
# stopped holding the moment Library and Profiles became modes: twelve segments
# against ten digits means either two modes with no key, or a second table
# saying which two -- and the second table is exactly what the positional
# scheme existed to avoid. So mode switching is a mouse action and a palette
# (Ctrl+K) action, and the digits go back to the workspace modes that were
# already reaching for them.


# **Quit has no control anywhere in the shell, and that is the decision.**
#
# It was never a mode -- it does not land in ``AppState.mode``, it has no pane,
# and the three categories above partition ``KEYS`` exactly. What it *had* was
# a place to be drawn: an eleventh segment of the switch until UX.md Phase 2,
# then a power icon in the header's right-hand strip. Both were a destructive
# action one click from every mode, mitigated by an unconditional confirm
# rather than fixed, and the second was only less bad than the first.
#
# The rail has no strip for it, so the ``QUIT`` tuple is gone with the header
# that read it. The ways out are the window's own X -- which routes through
# ``App._ask_quit`` (the UI redesign, wave 3; it used to bypass the preflight
# summary and go straight to ``_request_quit``, which was survivable only while
# the icon existed to carry it) -- and the palette's Quit command, which calls
# the same guard. Both ask about unsaved work; neither is a button somebody's
# pointer can find by accident.

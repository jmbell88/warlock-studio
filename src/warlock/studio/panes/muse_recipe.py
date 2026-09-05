"""Muse's recipe column: *how* a take is drawn, never *what* it is.

``settings_2d``'s recipe section, scoped down to the six knobs ACE-Step
actually has. The split against ``muse_brief`` is the same one Create keeps and
is enforced by hand rather than by a mechanism: **no control appears in both**.
The bar owns the tags, the lyrics, the duration, the count and the button; this
column owns everything a user changes once and then leaves alone.

Every field is named for the model's own parameter, deliberately -- so that a
setting here, the value in a finished take's stored recipe, and ACE-Step's own
documentation are three views of one word rather than three vocabularies.
"""

from __future__ import annotations

from typing import Any

from .. import focus, muse_mode, widgets
from ..manual import render as manual_render
from . import model_gate

#: This pane's key in the focus ring.
FOCUS_PANE = "muse-recipe"

#: The scheduler and guidance choices, and their bounds are
#: ``_jobs_music._SCHEDULERS`` / ``_CFG_TYPES``. Listed here as (key, label)
#: because the door's tuples are keys alone -- it validates, it does not name
#: things for a person.
_SCHEDULERS: tuple[tuple[str, str], ...] = (
    ("euler", "Euler"),
    ("heun", "Heun"),
    ("pingpong", "Ping-pong"),
)
_CFG_TYPES: tuple[tuple[str, str], ...] = (
    ("apg", "APG"),
    ("cfg", "CFG"),
    ("cfg_star", "CFG*"),
)


def draw(ctx: Any) -> None:
    state = muse_mode.ensure(ctx)
    form = state.form

    focus.pump(ctx.state, FOCUS_PANE)
    focus.begin(ctx.state, FOCUS_PANE)

    widgets.pane_title("Recipe")
    manual_render.help_button(ctx, "muse-recipe")

    # Ahead of every control, for ``model_gate``'s reason: Muse has no
    # fallback -- a take either generates or it does not -- so learning about
    # an 8 GB download only after writing a prompt and pressing Generate was
    # the worst moment to learn it. The refusal at the door is still the
    # authority; this is the courtesy in front of it.
    from ...service import jobs as svc_jobs

    model_gate.draw(ctx, svc_jobs.MUSIC_ROWS, what="Generating music")

    with focus.item(ctx.state, FOCUS_PANE, "infer_step"):
        _, form["infer_step"] = widgets.labeled_slider_int(
            "Steps",
            int(form["infer_step"]),
            1,
            200,
            help_text=(
                "How many sampling steps. More is slower and, past about 60, "
                "not better -- it is the first thing to lower when you are "
                "auditioning ideas rather than finishing one."
            ),
        )

    with focus.item(ctx.state, FOCUS_PANE, "guidance_scale"):
        _, form["guidance_scale"] = widgets.labeled_slider_float(
            "Guidance",
            float(form["guidance_scale"]),
            0.0,
            30.0,
            help_text=(
                "How closely the model follows your tags. Low wanders and can "
                "surprise you; high obeys and can flatten."
            ),
        )

    with focus.item(ctx.state, FOCUS_PANE, "scheduler_type"):
        form["scheduler_type"] = widgets.labeled_combo(
            "Scheduler",
            str(form["scheduler_type"]),
            list(_SCHEDULERS),
            help_text=(
                "Which sampler walks the steps. Euler is the default and the "
                "one every other setting here was chosen against."
            ),
        )

    with focus.item(ctx.state, FOCUS_PANE, "cfg_type"):
        form["cfg_type"] = widgets.labeled_combo(
            "Guidance type",
            str(form["cfg_type"]),
            list(_CFG_TYPES),
            help_text=(
                "How guidance is applied. APG is the model's own default and "
                "holds up best at the higher guidance values."
            ),
        )

    with focus.item(ctx.state, FOCUS_PANE, "omega_scale"):
        _, form["omega_scale"] = widgets.labeled_slider_float(
            "Omega",
            float(form["omega_scale"]),
            0.0,
            100.0,
            help_text=(
                "ACE-Step's granularity term. Leave it alone unless a take is "
                "muddy in a way guidance does not fix."
            ),
        )

    widgets.divider()
    _seed(ctx, form)


def _seed(ctx: Any, form: dict[str, Any]) -> None:
    """The seed, and the checkbox that decides whether there is one.

    ``None`` means "a fresh seed per take", which is a generative mode's
    default and is why the value is nullable rather than a number with a lock
    beside it: a locked number and an unlocked number are two states, and
    "there is no seed" is a third that the other two cannot express.

    With a number set, it applies to the **first** take of a press and the rest
    walk from it -- so "four takes at this seed" is four different pieces with
    one of them reproducible, rather than four copies of one file. The door is
    where that walk happens; this only says which case is being asked for.
    """
    pinned = form["seed"] is not None
    with focus.item(ctx.state, FOCUS_PANE, "seed_pinned"):
        changed, pinned = widgets.toggle("Pin the seed", pinned)
    if changed:
        form["seed"] = 0 if pinned else None
    if form["seed"] is None:
        widgets.muted_wrapped("Each take gets its own seed.")
        return
    with focus.item(ctx.state, FOCUS_PANE, "seed"):
        _, form["seed"] = widgets.labeled_drag_int(
            "Seed", int(form["seed"]), 0, _max_seed()
        )


def _max_seed() -> int:
    from ...service.validation import MAX_SEED

    return MAX_SEED


__all__ = ["FOCUS_PANE", "draw"]

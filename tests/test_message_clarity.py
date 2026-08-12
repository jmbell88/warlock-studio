"""The S-section of NEXT_ROADMAP Phase 2: what a message has to carry.

The pattern across all five items is the same one: the app already knew the
answer and said a smaller part of it than it had. A unit label without the
offending value, a button with no stated cost, a consequence reported after the
action rather than before it, and a configuration nothing could see.
"""

from __future__ import annotations

import dataclasses

import pytest

from warlock import guidance
from warlock.config import SETTINGS, Config, effective
from warlock.service.errors import Invalid

# --- S136: a sweep refusal names the unit, the field and the value ----------


def test_a_bad_sweep_unit_names_itself_and_the_offending_value(svc):
    from warlock.service import sweeps as sweeps_mod

    plan = sweeps_mod.SweepPlan(
        label="bad",
        prompt="a rock",
        base={},
        seeds=(23,),
        axes=(sweeps_mod.Axis("bg_removal", ("magic",)),),
        stage="reference",
    )
    with pytest.raises(Invalid) as caught:
        sweeps_mod._validate(svc, plan, sweeps_mod.expand(plan))
    message = caught.value.message
    assert "bg_removal=magic" in message  # the unit
    assert "magic" in message  # the value
    assert caught.value.field == "bg_removal"  # the control


def test_a_bounded_guidance_refusal_says_what_it_was_given():
    """A range without the value is a rule, not a diagnosis: it leaves the
    reader to work out which of their settings was out of it."""
    with pytest.raises(guidance.GuidanceError) as caught:
        guidance.normalize({"size_m": 900.0})
    assert "900" in str(caught.value)

    with pytest.raises(guidance.GuidanceError) as caught:
        guidance.normalize({"lora_weight": 12.0})
    assert "12" in str(caught.value)


# --- S140: the effective configuration, and the table that cannot drift -----


def test_every_config_field_is_either_listed_or_deliberately_not():
    """Both directions. A field with no entry vanishes from the readout the
    moment it is added, and an entry naming no field is a row of ``(unset)``
    that looks like a setting nobody has."""
    fields = {f.name for f in dataclasses.fields(Config)}
    named = {name for name, _ in SETTINGS}
    # Derived by the startup VRAM resolve rather than configured; listing it
    # beside real settings would present a computed answer as a user's choice.
    assert fields - named == {"vram_exclusive_explicit"}
    assert named - fields == set()


def test_every_entry_names_a_warlock_variable():
    assert all(env.startswith("WARLOCK_") for _, env in SETTINGS)
    assert len({env for _, env in SETTINGS}) == len(SETTINGS)


def test_a_set_variable_is_reported_as_set_even_at_its_default(monkeypatch):
    """The case the obvious implementation gets wrong. Comparing against a
    fresh ``Config`` would call this "default" -- and it is exactly the moment
    a user is asking whether their setting took effect."""
    monkeypatch.setenv("WARLOCK_TRELLIS_PORT", "17971")
    config = Config()
    row = next(s for s in effective(config) if s.name == "trellis_port")
    assert row.from_env is True
    assert row.value == "17971"


def test_an_unset_optional_reads_as_unset_rather_than_none(monkeypatch):
    monkeypatch.delenv("WARLOCK_EXPORT_DIR", raising=False)
    row = next(s for s in effective(Config()) if s.name == "export_dir")
    assert row.value == "(unset)"
    assert row.from_env is False


# --- the env-only switches, which no ``Config`` field can carry --------------


def test_every_switch_is_actually_read_somewhere():
    """The counterpart of the both-directions test above. ``SWITCHES`` has no
    dataclass to check itself against, so the check is that each variable is
    read by the module that owns it -- an entry naming a variable nothing reads
    is a row in the readout that means nothing."""
    import inspect

    from warlock import migrate, native
    from warlock.config import SWITCHES

    sources = inspect.getsource(migrate) + inspect.getsource(native)
    for _, env in SWITCHES:
        assert env in sources, f"{env} is listed in SWITCHES but nothing reads it"


def test_a_switch_is_not_also_a_config_field():
    """The two tables partition the environment; an overlap would print the
    same variable twice with two different answers."""
    from warlock.config import SWITCHES

    fields = {f.name for f in dataclasses.fields(Config)}
    assert {name for name, _ in SWITCHES} & fields == set()
    assert {env for _, env in SWITCHES} & {env for _, env in SETTINGS} == set()


def test_an_unset_switch_reads_as_unset_and_a_set_one_carries_its_value(monkeypatch):
    monkeypatch.delenv("WARLOCK_NATIVE_DLL", raising=False)
    monkeypatch.setenv("WARLOCK_NATIVE", "0")
    rows = {s.name: s for s in effective(Config())}
    assert rows["native_dll"].value == "(unset)"
    assert rows["native_dll"].from_env is False
    assert rows["native"].value == "0"
    assert rows["native"].from_env is True


def test_effective_is_pure():
    """The ``vram.py`` rule: both the CLI and a pane read this, and the CLI runs
    on a machine with no display."""
    import inspect

    from warlock import config as config_mod

    source = inspect.getsource(config_mod)
    for forbidden in ("from .service", "from .studio", "from .queue"):
        assert forbidden not in source

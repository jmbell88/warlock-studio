"""A mode's door asks for the pack as well as the weights, and in that order.

Finding F4, 2026-09-05. On the clean-machine install both halves of Create were
missing at once -- no ``text2image`` pack and no SDXL weights -- and only the
weights half was checked. ``packs.Pack.modes`` already named which mode each
pack gates and was read in exactly one place, a label inside Settings, so the
rail sent the user to Models to download about 23 GB that still would not let
Create generate anything, because ``torch`` was not installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio.panes import model_gate


class _State:
    def __init__(self):
        self.preview: dict[str, str] = {}
        self.mode = "home"


def _ctx(*, packs=(), models_=(), library=0):
    """A context of the shape the panes read, and nothing else."""
    return SimpleNamespace(
        pack_rows=list(packs),
        model_rows=list(models_),
        cache=SimpleNamespace(total=library),
        model_picks=set(),
        state=_State(),
    )


def _pack(key, *, modes, present, gib=1.0, label=None):
    return {
        "key": key,
        "label": label or key.title(),
        "modes": list(modes),
        "present": present,
        "download_gib": gib,
    }


def _model(row_key, *, present, gib=7.0):
    return {"row_key": row_key, "present": present, "size_gib": gib}


BASE_INSTALL = dict(
    packs=[_pack("text2image", modes=["create"], present=False, gib=2.4,
                 label="Image generation")],
    models_=[_model("engine:trellis_gguf", present=False, gib=16.1),
             _model("base:sdxl_cfg", present=False, gib=7.0)],
)


def test_a_base_install_is_sent_to_packs_not_models():
    """The exact 2026-09-05 state: no pack, no weights, nothing in the library."""
    where, keys = model_gate.mode_gate(_ctx(**BASE_INSTALL), "create")
    assert where == "packs", "23 GB of weights was offered before the code that reads them"
    assert keys == ("text2image",)


def test_the_reason_names_the_pack_and_its_own_size():
    said = model_gate.mode_reason(_ctx(**BASE_INSTALL), "create")
    assert "Image generation" in said
    assert "2.4 GB" in said, "the pack's size, not the weights'"
    assert "23" not in said


def test_clicking_through_opens_settings_at_packs():
    from warlock.studio.panes import app_settings

    ctx = _ctx(**BASE_INSTALL)
    model_gate.request_for_mode(ctx, "create")
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "packs"
    assert ctx.state.mode == "settings"


def test_once_the_pack_is_in_the_door_asks_for_the_weights():
    """The second half, and the routing has to move with it."""
    from warlock.studio.panes import app_settings

    ctx = _ctx(
        packs=[_pack("text2image", modes=["create"], present=True)],
        models_=BASE_INSTALL["models_"],
    )
    where, keys = model_gate.mode_gate(ctx, "create")
    assert where == "models"
    assert set(keys) == {"engine:trellis_gguf", "base:sdxl_cfg"}
    model_gate.request_for_mode(ctx, "create")
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "models"
    assert ctx.model_picks == set(keys), "the rows were not pre-ticked"


def test_a_fully_equipped_mode_is_not_gated():
    ctx = _ctx(
        packs=[_pack("text2image", modes=["create"], present=True)],
        models_=[_model("engine:trellis_gguf", present=True),
                 _model("base:sdxl_cfg", present=True)],
    )
    assert model_gate.mode_gate(ctx, "create") == ("", ())
    assert model_gate.mode_reason(ctx, "create") == ""


def test_a_library_with_work_in_it_opens_the_mode_anyway():
    """The escape both halves share.

    Create's later stages act on jobs that already exist. Gating on a missing
    pack alone would lock a user out of their own finished work -- the failure
    ``mode_block``'s docstring has always warned about, which the pack half
    must not reintroduce.
    """
    ctx = _ctx(**BASE_INSTALL, library=3)
    assert model_gate.mode_gate(ctx, "create") == ("", ())


def test_a_pack_that_gates_another_mode_does_not_gate_this_one():
    ctx = _ctx(
        packs=[_pack("music", modes=["muse"], present=False)],
        models_=[_model("engine:trellis_gguf", present=True),
                 _model("base:sdxl_cfg", present=True)],
    )
    assert model_gate.mode_gate(ctx, "create") == ("", ())
    assert model_gate.mode_gate(ctx, "muse")[0] == "packs"


def test_an_empty_snapshot_reports_nothing_rather_than_everything():
    """``missing``'s rule, applied to the new half.

    A headless ctx, or the first frame before the answers land, must read as
    "nothing to say" -- reading it as "no packs present" would grey both
    generation modes on a fully installed machine.
    """
    ctx = SimpleNamespace(state=_State(), model_picks=set())
    assert model_gate.mode_gate(ctx, "create") == ("", ())


@pytest.mark.parametrize("mode", ["home", "library", "settings", "inker", "review"])
def test_ungated_modes_stay_ungated(mode):
    """Settings above all: it is where the pack and the download both live."""
    assert model_gate.mode_gate(_ctx(**BASE_INSTALL), mode) == ("", ())


def test_every_pack_gates_a_mode_the_rail_actually_has():
    """``Pack.modes`` is now load-bearing rather than a label, so it must resolve."""
    from warlock import packs as packs_mod
    from warlock.studio import modes as modes_mod

    known = {key for key, _label, _icon in modes_mod.MODES}
    for pack in packs_mod.PACKS:
        for key in pack.modes:
            assert key in known, f"{pack.key} gates {key!r}, which is not a mode"

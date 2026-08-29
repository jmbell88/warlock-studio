"""The pixelisation options every pixel path shares, validated once.

``service/troupe._check_options`` was the only copy of these four refusals and
is now a thin wrapper over ``pixelopts.check_pixel_options``. The bar for that
lift is that nothing a user can see moved: the same values are refused, in the
same sentences, on the same ``field=`` -- because a refusal is a string
somebody reads and a field a form highlights.
"""

from __future__ import annotations

import pytest

from warlock.service import troupe
from warlock.service.errors import Invalid
from warlock.service.pixelopts import check_pixel_options

# A caller that is *not* Troupe, which is the whole point of the parameters.
SIZES = (16, 32, 64)
COLORS = (4, 8, 16)


def _check(svc, entries, **kwargs):
    return check_pixel_options(
        svc,
        entries,
        sizes=SIZES,
        size_default=32,
        colors=COLORS,
        colors_default=8,
        **kwargs,
    )


def test_the_defaults_are_what_an_empty_request_means(svc):
    assert _check(svc, {}) == {
        "logical_size": 32,
        "colors": 8,
        "outline": "none",
        "reduce_mode": "box",
        "dither": False,
        "palette": "",
    }


@pytest.mark.parametrize(
    ("entries", "field", "fragment"),
    [
        ({"logical_size": "wide"}, "logical_size", "must be a whole number"),
        ({"logical_size": 24}, "logical_size", "must be one of [16, 32, 64]"),
        ({"colors": "many"}, "colors", "must be a whole number"),
        ({"colors": 64}, "colors", "must be one of [4, 8, 16]"),
        ({"outline": "glow"}, "outline", "outline must be one of"),
        ({"reduce_mode": "lanczos"}, "reduce_mode", "reduce_mode must be one of"),
    ],
)
def test_a_bad_value_is_refused_on_its_own_field(svc, entries, field, fragment):
    with pytest.raises(Invalid) as excinfo:
        _check(svc, entries)
    assert excinfo.value.field == field
    assert fragment in excinfo.value.message


def test_the_ladders_named_in_the_message_are_the_callers_own(svc):
    """The parameterisation, visible where it matters: a caller with a
    different ladder must not be told Troupe's."""
    with pytest.raises(Invalid) as excinfo:
        _check(svc, {"colors": 64})
    assert "64" not in excinfo.value.message.split("of", 1)[1]


def test_a_path_with_no_outline_pass_returns_no_outline(svc):
    """A request that said nothing gets the key dropped, not defaulted: a params
    blob carrying a setting nothing reads is the dead field this gate exists to
    prevent."""
    out = _check(svc, {}, allow_outline=False, allow_reduce_mode=False)
    assert "outline" not in out
    assert "reduce_mode" not in out


@pytest.mark.parametrize("asked", ["inner", "outer", "glow"])
def test_a_path_with_no_outline_pass_refuses_one_that_was_asked_for(svc, asked):
    """Dropping is the *weaker* answer, and it used to be the one given here --
    which is why ``create_tile_sheet`` had to grow an ``outline`` parameter whose
    only job was to be refused by name after this function had already seen it
    and said nothing. A caller that hands an outline mode to a path with no
    outline pass has made a mistake, and diagnosing a mistake and then swallowing
    it is how the caller comes to believe the setting took."""
    with pytest.raises(Invalid) as excinfo:
        _check(svc, {"outline": asked}, allow_outline=False)
    assert excinfo.value.field == "outline"


def test_the_refusal_carries_the_callers_own_reason(svc):
    """*Why* there is no outline pass is a fact about the caller's kind rather
    than about this function -- the tile sheet's reason is
    ``pixelize._edge_mask`` -- so the sentence is passed in and the field is
    fixed here."""
    with pytest.raises(Invalid) as excinfo:
        _check(
            svc,
            {"outline": "inner"},
            allow_outline=False,
            outline_refusal="a tile is opaque edge to edge",
        )
    assert excinfo.value.message == "a tile is opaque edge to edge"


def test_no_outline_is_not_an_outline_request(svc):
    """The word every one of these doors spells "off". Refusing it would refuse
    a request for exactly what a path with no outline pass does anyway."""
    assert "outline" not in _check(svc, {"outline": "none"}, allow_outline=False)


def test_a_path_with_one_reduction_refuses_a_choice_of_it(svc):
    """The outline rule's twin, and there is no second argument for it."""
    assert "reduce_mode" not in _check(svc, {}, allow_reduce_mode=False)
    with pytest.raises(Invalid) as excinfo:
        _check(svc, {"reduce_mode": "point"}, allow_reduce_mode=False)
    assert excinfo.value.field == "reduce_mode"


@pytest.mark.parametrize("field", ["logical_size", "colors"])
def test_zero_is_a_value_and_not_an_absence(svc, field):
    """``entries.get(k) or default`` made 0 mean "the form said nothing", so a
    request for zero colours came back as eight and nobody was told. That is the
    failure ``service/jobs`` already names on the tile size -- "make me 96px
    tiles" answered with 32px tiles -- and 0 is on no ladder any path publishes,
    so it falls through to the ladder check and is refused by name.

    ``None`` and ``""`` still mean absent: one is a key the request did not have
    and the other is a form field nobody typed in."""
    with pytest.raises(Invalid) as excinfo:
        _check(svc, {field: 0})
    assert excinfo.value.field == field
    assert "must be one of" in excinfo.value.message
    for absent in (None, ""):
        assert _check(svc, {field: absent})[field] == (32 if field == "logical_size" else 8)


def test_a_named_palette_is_loaded_at_the_door_and_thrown_away(svc, tmp_path):
    """The reason it is loaded here at all: an unreadable palette should cost
    the request, not a minute of GPU and a sheet that came back wrong."""
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    (directory / "broken.hex").write_text("this is not a palette\n")
    with pytest.raises(Invalid) as excinfo:
        _check(svc, {"palette": "broken"})
    assert excinfo.value.field == "palette"

    (directory / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    # Thrown away: the name is what is carried forward, never the colours --
    # the worker re-reads the file, so an edit between queueing and running is
    # the user's edit rather than a stale snapshot.
    assert _check(svc, {"palette": " duo "})["palette"] == "duo"


def test_troupe_still_answers_exactly_what_it_answered_before(svc):
    """The wrapper, pinned. Troupe's own ladders and its ``outer`` default --
    which is not ``check_pixel_options``' ``none`` -- have to survive the lift.
    """
    assert troupe._check_options(svc, {}) == {
        "logical_size": troupe.DEFAULT_TROUPE_LOGICAL_SIZE,
        "colors": troupe.DEFAULT_TROUPE_COLORS,
        "outline": troupe.DEFAULT_TROUPE_OUTLINE,
        "reduce_mode": troupe.TROUPE_REDUCE_MODES[0],
        "dither": False,
        "palette": "",
    }
    with pytest.raises(Invalid) as excinfo:
        troupe._check_options(svc, {"logical_size": 7})
    assert excinfo.value.field == "logical_size"
    assert str(list(troupe.TROUPE_LOGICAL_SIZES)) in excinfo.value.message

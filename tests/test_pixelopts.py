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


def test_a_path_with_no_outline_pass_neither_validates_nor_returns_one(svc):
    out = _check(svc, {"outline": "glow"}, allow_outline=False)
    assert "outline" not in out
    out = _check(svc, {"reduce_mode": "lanczos"}, allow_reduce_mode=False)
    assert "reduce_mode" not in out


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

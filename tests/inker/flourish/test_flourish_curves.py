"""Curves: the easing vocabulary, sampling, clamping and the codec."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.pipelines import sheet
from warlock.studio.inker.flourish import curves


def test_the_easing_vocabulary_is_the_clip_easings_plus_hold():
    """Same words, same arithmetic as ``pipelines/sheet``: a clip's spacing and
    an effect's spacing must mean the same thing, and the package may not
    import ``pipelines`` to get them."""
    assert set(curves.EASINGS) == set(sheet.EASINGS) | {"hold"}
    for kind in sheet.EASINGS:
        for t in (0.0, 0.2, 0.5, 0.9, 1.0):
            assert curves.ease(t, kind) == pytest.approx(sheet._ease(t, kind))  # noqa: SLF001


def test_a_constant_curve_is_constant_everywhere():
    c = curves.Curve.const(3.5)
    assert c.is_const
    assert c.at(0.0) == c.at(0.5) == c.at(1.0) == 3.5
    assert c.to_json() == 3.5


def test_a_line_interpolates_and_clamps_outside_its_keys():
    c = curves.Curve.line(0.0, 10.0)
    assert c.at(0.5) == pytest.approx(5.0)
    assert c.at(-1.0) == 0.0
    assert c.at(2.0) == 10.0


def test_keys_are_sorted_on_construction():
    c = curves.Curve(((1.0, 1.0), (0.0, 0.0), (0.5, 4.0)))
    assert [k[0] for k in c.keys] == [0.0, 0.5, 1.0]
    assert c.at(0.5) == pytest.approx(4.0)


def test_hold_steps_at_each_key():
    c = curves.Curve(((0.0, 1.0), (0.5, 2.0), (1.0, 3.0)), "hold")
    assert c.at(0.25) == 1.0
    assert c.at(0.5) == 2.0
    assert c.at(0.99) == 2.0
    assert c.at(1.0) == 3.0


def test_sampling_is_vectorised_and_matches_scalar_evaluation():
    c = curves.Curve(((0.0, 0.0), (0.4, 2.0), (1.0, -1.0)), "ease")
    ts = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    got = c.sample(ts)
    assert got.dtype == np.float32
    for t, v in zip(ts, got, strict=True):
        assert c.at(float(t)) == pytest.approx(float(v), abs=1e-6)


def test_the_codec_round_trips_and_accepts_bare_numbers():
    c = curves.Curve(((0.0, 1.0), (1.0, 0.0)), "ease_out")
    assert curves.Curve.from_json(c.to_json()) == c
    assert curves.Curve.from_json(2) == curves.Curve.const(2.0)
    assert curves.Curve.from_json([[0, 1], [1, 2]]).at(0.5) == pytest.approx(1.5)


def test_an_unknown_easing_or_empty_curve_is_refused():
    with pytest.raises(ValueError):
        curves.Curve(((0.0, 1.0),), "bounce")
    with pytest.raises(ValueError):
        curves.Curve(())


def test_clamped_narrows_every_key():
    c = curves.Curve(((0.0, -5.0), (1.0, 50.0))).clamped(0.0, 10.0)
    assert [v for _, v in c.keys] == [0.0, 10.0]

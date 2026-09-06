"""The filesystem-free half of ``characters.effects``.

Everything here is arithmetic: which effect kind a theme declares, which socket
it hangs on, what seed it renders from, how big it is, where the compositor
puts it and which of the two planes ends up on top. The half that reads and
writes PNGs is exercised end to end through the real worker in
``tests/troupe/test_troupe_chain.py``; splitting them is what keeps the claims
above checkable in milliseconds instead of behind a Blender fake.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.characters import effects, family


def _theme(key: str = "fire", family_key: str = "elemental"):
    fam = family.get_family(family_key)
    found = next((t for t in fam.themes if t.key == key), None)
    assert found is not None, f"{family_key} has no {key} theme"
    return found


# -- what the registry actually declares -------------------------------------


def test_every_effect_the_registry_declares_is_one_this_module_can_draw():
    """The pin that makes ``EFFECT_KINDS`` a contract rather than a note.

    A species row that names ``"lightning"`` would otherwise ship as a theme
    that silently draws nothing -- the failure the registry's own "every region
    must appear" rule exists to prevent, one layer out. Widening the table is a
    function beside ``flame_recipe``, and this test is what asks for it.
    """
    declared = {
        kind
        for fam in family.families().values()
        for theme in fam.themes
        for kind in theme.effects
    }
    assert declared, "no theme declares an effect at all; this test is vacuous"
    assert declared <= set(effects.EFFECT_KINDS), sorted(declared - set(effects.EFFECT_KINDS))


def test_every_theme_that_declares_an_effect_has_a_socket_to_hang_it_on():
    """An effect kind and an archetype that share no socket name is a theme
    that costs a render and draws nothing -- and the four archetypes
    deliberately do *not* share a socket name, so the join table is the only
    thing standing between them."""
    for fam in family.families().values():
        for theme in fam.themes:
            kind = effects.effect_kind(theme)
            if kind is None:
                continue
            assert effects.pick_socket(kind, fam.arch.sockets) is not None, (
                f"{fam.key}/{theme.key} emits {kind} and {fam.archetype} has nowhere to put it"
            )


def test_a_theme_that_declares_nothing_draws_nothing():
    """Most themes are a palette and nothing more, and they must stay that way:
    an effect that appeared on a stone elemental because the code guessed would
    be a character nobody asked for."""
    fam = family.get_family("elemental")
    quiet = next(t for t in fam.themes if not t.effects)
    assert effects.effect_kind(quiet) is None


def test_an_effect_kind_this_build_cannot_draw_costs_the_effect_and_not_the_sheet():
    """``blender_worker``'s unknown-bone rule, restated one layer up: a theme
    from a newer registry must not raise into a job that has already spent a
    minute in Blender."""
    from dataclasses import replace

    theme = replace(_theme(), effects=("lightning",))
    assert effects.effect_kind(theme) is None


def test_the_first_kind_this_build_knows_wins_over_a_later_one():
    from dataclasses import replace

    theme = replace(_theme(), effects=("lightning", "embers"))
    assert effects.effect_kind(theme) == "embers"


# -- the socket the effect hangs on ------------------------------------------


@pytest.mark.parametrize(
    "archetype,expected",
    [("amorphous", "core"), ("humanoid", "crown"), ("winged", "crown"), ("quadruped", "crown")],
)
def test_the_socket_is_the_first_preference_the_archetype_actually_declares(
    archetype, expected
):
    """Ordered rather than exhaustive on purpose: an amorphous elemental has a
    ``core`` and no head worth crowning, a dragon has a ``crown`` and no core."""
    arch = family.get_archetype(archetype)
    found = effects.pick_socket("embers", arch.sockets)
    assert found is not None
    index, socket = found
    assert socket.name == expected
    assert arch.sockets[index] is socket


def test_nothing_hangs_on_an_archetype_that_declares_no_preferred_socket():
    from warlock.characters.family import Socket

    assert effects.pick_socket("embers", (Socket("tail", "tail.001"),)) is None


def test_only_one_socket_ever_burns():
    """Five flames on a humanoid is a bonfire, not a character -- and the five
    would composite over each other in cell order, which is not a drawing
    anybody chose."""
    arch = family.get_archetype("humanoid")
    found = effects.pick_socket("embers", arch.sockets)
    assert isinstance(found, tuple) and len(found) == 2


# -- the seed ----------------------------------------------------------------


def test_two_sockets_of_one_character_get_two_different_flames():
    a = effects.effect_seed(1234, 0)
    b = effects.effect_seed(1234, 1)
    assert a != b


def test_the_seed_is_a_pure_function_of_the_recipe_and_the_socket():
    """The whole "the same sheet renders the same bytes twice" property rests
    on this being arithmetic and not a counter."""
    assert effects.effect_seed(1234, 3) == effects.effect_seed(1234, 3)
    assert effects.effect_seed(1234, 3) != effects.effect_seed(1235, 3)


def test_the_seed_survives_the_recipe_codecs_own_masking():
    """``recipe.clamp`` masks a seed to 31 bits. A value that did not already
    fit would come back as a *different* flame from the one this function
    named, silently."""
    from warlock.studio.inker.flourish import recipe as flourish_recipe

    seed = effects.effect_seed(2**31 - 1, 5)
    assert 0 <= seed <= 0x7FFFFFFF
    assert flourish_recipe.clamp(flourish_recipe.Recipe(seed=seed)).seed == seed


# -- the recipe --------------------------------------------------------------


def _flame(**kw):
    fam = family.get_family("elemental")
    theme = _theme()
    _index, socket = effects.pick_socket("embers", fam.arch.sockets)
    args = {"seed": 99, "cell_px": 32, "height_px": 11, "frames": 4, "fps": 7}
    args.update(kw)
    return effects.flame_recipe(theme, socket, **args)


def test_the_flame_rises_regardless_of_which_way_the_character_faces():
    """Fire rises in world space. A sheet whose flames lean east when the
    character faces east is a sheet where gravity turns with the camera -- and
    a character sheet's eight directions are eight cameras, not eight worlds."""
    recipe = _flame()
    assert recipe.layers[0].params["rise"] == pytest.approx(-90.0)


def test_the_flame_takes_its_colours_from_the_theme_it_belongs_to():
    """A correction to the plan, and worth stating: ``Theme.effect_params`` is
    typed ``Mapping[str, float]``, so a colour cannot live there. The theme's
    own materials are the honest source -- a fire theme's ``accent`` is the
    orange it was authored with."""
    theme = _theme()
    base, tip = effects.effect_colors(theme)
    assert base == theme.materials["core"]
    assert tip == theme.materials["accent"]
    layer = _flame().layers[0]
    assert layer.params["color_base"] == base
    assert layer.params["color_tip"] == tip


def test_a_string_in_effect_params_still_wins_if_the_registry_ever_grows_one():
    from dataclasses import replace

    theme = replace(_theme(), effect_params={"color_tip": "#123456", "rate": 24.0})
    assert effects.effect_colors(theme)[1] == "#123456"


def test_a_theme_with_neither_region_falls_back_to_the_primitives_own_flame():
    """Nothing rather than a stock flame would be the worse answer: an effect
    that vanishes because a palette had no ``accent`` is a bug nobody can see."""
    from dataclasses import replace

    from warlock.studio.inker.flourish import prims

    theme = replace(_theme(), materials={}, effect_params={})
    base, tip = effects.effect_colors(theme)
    params = prims.params_of("flame")
    assert base == params["color_base"].default
    assert tip == params["color_tip"].default


def test_the_flame_is_one_phase_exactly_as_long_as_the_movement():
    """A four-frame idle wants four distinct flames and an eight-frame walk
    eight, or the flame restarts in every cell instead of animating with it."""
    recipe = _flame(frames=8)
    assert len(recipe.phases) == 1
    assert recipe.phases[0].frames == 8
    assert recipe.frame_count == 8


def test_the_flame_is_painterly_and_supersampled_because_the_sheet_quantises_after_it():
    """A flame that had already cut itself to sixteen colours would be
    quantised twice, and the second cut would be over a palette the first one
    chose."""
    recipe = _flame()
    assert recipe.mode == "painterly"
    assert recipe.supersample == 4
    assert recipe.palette is None


def test_the_flame_never_leaves_the_canvas_it_is_drawn_on():
    """The base sits at the canvas centre, so a flame taller than half the
    canvas is cut off by the *plane* rather than by the cell -- the same pixels
    lost for a much less obvious reason."""
    recipe = _flame(cell_px=32, height_px=400)
    assert recipe.width == recipe.height == 32
    assert recipe.layers[0].params["height"] <= 16


def test_the_flame_is_placed_at_the_canvas_centre_and_moved_by_the_compositor():
    """The alternative -- placing it with the layer's own x/y -- works equally
    well and is only checkable by rendering, where an offset is arithmetic a
    test can state outright."""
    layer = _flame().layers[0]
    assert layer.params["x"] == pytest.approx(0.0)
    assert layer.params["y"] == pytest.approx(0.0)


def test_the_recipe_does_not_depend_on_how_many_recipes_came_before_it():
    """A ``new_uid()`` here would make the bytes a function of the process's
    history, which is the one thing this whole path is built not to be."""
    first = _flame()
    second = _flame()
    assert first.layers[0].uid == second.layers[0].uid
    assert first == second


# -- how big ------------------------------------------------------------------


def test_the_reach_is_the_effects_size_budget():
    """``Socket.reach`` is the radius a prop may occupy before it clips the
    body, which is exactly what an effect should be allowed to spend: a flame
    the size of the thing it is allowed to hang there does not swallow the
    character."""
    from warlock.characters.family import Socket

    theme = _theme()
    small = effects.effect_height_px(Socket("a", "b", reach=1.0), theme, logical=32)
    big = effects.effect_height_px(Socket("a", "b", reach=3.0), theme, logical=32)
    assert big > small


def test_the_species_scales_the_budget_and_the_archetype_sets_it():
    """"How big" stays a species decision (``effect_params["rise"]``) and "how
    much room is there" stays the archetype's (``Socket.reach``)."""
    from dataclasses import replace

    from warlock.characters.family import Socket

    socket = Socket("a", "b", reach=1.0)
    quiet = replace(_theme(), effect_params={"rise": 0.1})
    loud = replace(_theme(), effect_params={"rise": 0.9})
    assert effects.effect_height_px(socket, quiet, logical=64) < effects.effect_height_px(
        socket, loud, logical=64
    )


def test_a_flame_is_never_zero_pixels_tall():
    from warlock.characters.family import Socket

    tiny = effects.effect_height_px(Socket("a", "b", reach=0.01), _theme(), logical=16)
    assert tiny >= 2


# -- the offset arithmetic ----------------------------------------------------


def _dot(size, x, y, colour=(255, 0, 0, 255)):
    plane = np.zeros((size, size, 4), dtype=np.float32)
    plane[y, x] = np.array(colour, dtype=np.float32) / 255.0
    return plane


def test_the_shift_puts_the_canvas_centre_on_the_socket():
    """The one line of arithmetic the whole placement rests on: the flame's
    base is the canvas centre, so shifting by ``socket - centre`` lands the
    base on the socket."""
    size = 16
    centre = size // 2
    plane = _dot(size, centre, centre)
    moved = effects._shift_into(plane, (size, size), dx=3 - centre, dy=11 - centre)
    assert moved[11, 3, 3] == pytest.approx(1.0)
    assert moved[centre, centre, 3] == pytest.approx(0.0)


def test_a_flame_at_the_edge_is_half_a_flame_and_never_a_stripe_down_the_other_side():
    """Clipped rather than wrapped. A wrap would put a socket's flame on the
    cell's opposite edge, which in a dense atlas is the *next character*."""
    size = 8
    plane = _dot(size, 1, 1)
    moved = effects._shift_into(plane, (size, size), dx=-4, dy=0)
    assert moved[..., 3].sum() == pytest.approx(0.0)


def test_a_shift_of_nothing_is_the_plane_it_was_given():
    size = 8
    plane = _dot(size, 2, 5)
    assert np.array_equal(effects._shift_into(plane, (size, size), dx=0, dy=0), plane)


def test_the_shift_fills_a_canvas_of_the_bodys_size_not_the_flames():
    """The cells are square today; the compositor must not assume it, because
    ``Plan`` grew ``frame_w``/``frame_h`` for a reason."""
    plane = _dot(8, 4, 4)
    moved = effects._shift_into(plane, (5, 11), dx=0, dy=0)
    assert moved.shape == (5, 11, 4)


# -- over and under -----------------------------------------------------------


def _solid(colour):
    plane = np.zeros((2, 2, 4), dtype=np.float32)
    plane[...] = np.array(colour, dtype=np.float32)
    return plane


def test_an_opaque_top_plane_hides_what_is_under_it():
    under = _solid((0.0, 0.0, 1.0, 1.0))
    top = _solid((1.0, 0.0, 0.0, 1.0))
    out = effects._over(under, top)
    assert out[0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_a_socket_behind_the_body_is_the_same_two_planes_in_the_other_order():
    """The single bit the worker measures view depth for. A back-mounted flame
    drawn over the body is a character standing in front of their own fire."""
    body = _solid((0.5, 0.5, 0.5, 1.0))
    flame = _solid((1.0, 0.4, 0.0, 1.0))
    assert effects._over(body, flame)[0, 0, 0] == pytest.approx(1.0)
    assert effects._over(flame, body)[0, 0, 0] == pytest.approx(0.5)


def test_a_clear_top_plane_leaves_the_body_exactly_as_it_was():
    under = _solid((0.25, 0.5, 0.75, 1.0))
    out = effects._over(under, np.zeros_like(under))
    assert out == pytest.approx(under)


def test_a_clear_pixel_carries_no_colour_at_all():
    """``render.to_uint8``'s rule, and the sheet's: a transparent pixel with
    colour in it becomes a fringe the moment anything resamples the atlas."""
    clear = np.zeros((2, 2, 4), dtype=np.float32)
    out = effects._over(clear, clear)
    assert out.sum() == pytest.approx(0.0)


def test_a_half_transparent_flame_mixes_rather_than_replaces():
    under = _solid((0.0, 0.0, 0.0, 1.0))
    top = _solid((1.0, 1.0, 1.0, 0.5))
    out = effects._over(under, top)
    assert out[0, 0, 0] == pytest.approx(0.5, abs=1e-3)
    assert out[0, 0, 3] == pytest.approx(1.0)


# -- the rendered flame, with no filesystem ----------------------------------


def test_the_flame_actually_draws_something():
    """A recipe that clamps to nothing visible would pass every test above."""
    from warlock.studio.inker.flourish import render as flourish_render

    recipe = _flame(cell_px=64, height_px=24, frames=4)
    plane = flourish_render.render_frame(recipe, 0)
    rgba = flourish_render.to_uint8(plane, recipe.supersample)
    assert rgba.shape == (64, 64, 4)
    assert int(rgba[..., 3].sum()) > 0


def test_the_flames_base_sits_at_the_canvas_centre_and_it_rises_from_there():
    """``rise = -90`` in screen degrees is straight up. If it were +90 the
    flame would hang below the socket, which on a crown is the character's face.
    """
    from warlock.studio.inker.flourish import render as flourish_render

    recipe = _flame(cell_px=64, height_px=24, frames=4)
    alpha = flourish_render.to_uint8(
        flourish_render.render_frame(recipe, 0), recipe.supersample
    )[..., 3]
    above = int(alpha[:32].sum())
    below = int(alpha[33:].sum())
    assert above > below


def test_two_frames_of_one_movement_are_two_different_flames():
    from warlock.studio.inker.flourish import render as flourish_render

    recipe = _flame(cell_px=64, height_px=24, frames=4)
    first, second = (
        flourish_render.to_uint8(flourish_render.render_frame(recipe, f), recipe.supersample)
        for f in (0, 2)
    )
    assert not np.array_equal(first, second)


def test_two_runs_of_one_recipe_are_the_same_bytes():
    from warlock.studio.inker.flourish import render as flourish_render

    a = flourish_render.render_frame(_flame(cell_px=64, height_px=24), 1)
    b = flourish_render.render_frame(_flame(cell_px=64, height_px=24), 1)
    assert np.array_equal(a, b)

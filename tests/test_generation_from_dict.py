"""``GenerationRequest.from_dict``'s sub-document coercion.

The module's docstring calls the request "safe to use from settings
migration, API validation, and worker planning code" -- but until the
2026-09-06 audit (finding create2-07), ``from_dict`` only cast its own
top-level scalar fields; ``TileSettings``, ``SpriteSettings`` and
``ModelSettings`` were built from raw sub-dict values with no casting at
all. A string-typed numeric field arriving from JSON (a form field, a
migrated settings row, an external caller) survived construction and then
made ``validate_request`` crash with an uncaught ``TypeError`` instead of
returning a ``CompatibilityIssue``. ``from_dict`` has no test home before
this file -- part of why the gap survived undetected.
"""

from __future__ import annotations

from warlock import generation


def test_from_dict_coerces_string_typed_tile_and_sprite_numerics():
    """The two working reproductions from the audit, as one regression."""
    tile_raw = {
        "generation_type": "tileset",
        "prompt": "a knight",
        "tile": {"mode": "collection", "prompt_items": ["grass"], "variants": "2"},
    }
    tile_req = generation.GenerationRequest.from_dict(tile_raw)
    assert tile_req.tile.variants == 2
    assert isinstance(tile_req.tile.variants, int)
    tile_issues = generation.validate_request(tile_req)
    assert all(issue.field != "tile.variants" for issue in tile_issues)

    sprite_raw = {
        "generation_type": "sprite_sheet",
        "prompt": "a knight",
        "sprite": {
            "mode": "action",
            "action": "walk",
            "directions": "8",
            "candidate_count": "1",
        },
    }
    sprite_req = generation.GenerationRequest.from_dict(sprite_raw)
    assert sprite_req.sprite.directions == 8
    assert isinstance(sprite_req.sprite.directions, int)
    assert sprite_req.sprite.candidate_count == 1
    assert isinstance(sprite_req.sprite.candidate_count, int)
    issues = generation.validate_request(sprite_req)
    assert all(
        issue.field not in ("sprite.directions", "sprite.candidate_count")
        for issue in issues
    )


def test_from_dict_coerces_model_settings_custom_triangles():
    """The finding's ``model`` sub-document, uncovered by either probe."""
    raw = {
        "generation_type": "3d_model",
        "prompt": "a knight",
        "model": {"output_profile": "raw", "custom_triangles": "5000"},
    }
    req = generation.GenerationRequest.from_dict(raw)
    assert req.model.custom_triangles == 5000
    assert isinstance(req.model.custom_triangles, int)


def test_a_non_numeric_tile_variants_refuses_instead_of_crashing():
    """Coercion is not acceptance: ``{"variants": "banana"}`` must reach
    ``validate_request`` as a ``CompatibilityIssue``, not an uncaught
    ``ValueError`` from ``int()`` -- that would just move the crash the
    audit found, not fix it."""
    raw = {
        "generation_type": "tileset",
        "prompt": "a knight",
        "tile": {"mode": "collection", "prompt_items": ["grass"], "variants": "banana"},
    }
    req = generation.GenerationRequest.from_dict(raw)
    issues = generation.validate_request(req)
    assert any(issue.field == "tile.variants" for issue in issues)


def test_a_non_numeric_sprite_directions_refuses_instead_of_crashing():
    raw = {
        "generation_type": "sprite_sheet",
        "prompt": "a knight",
        "sprite": {"mode": "action", "action": "walk", "directions": "banana"},
    }
    req = generation.GenerationRequest.from_dict(raw)
    issues = generation.validate_request(req)
    assert any(issue.field == "sprite.directions" for issue in issues)

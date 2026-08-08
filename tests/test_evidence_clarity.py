"""The N/D/O/P section of NEXT_ROADMAP Phase 2: what the evidence is allowed
to claim.

P120 is the one worth reading twice. Every other item here adds something the
app knew and did not say; that one *removes* a claim, because the claim was
backwards -- and a metric that is believed and inverted is worse than one that
is missing.
"""

from __future__ import annotations

import pytest

from warlock import guidance, vram
from warlock.service.errors import NotFound

# --- P120: hole_worst is inverted, and nothing may imply otherwise ----------


def test_no_reader_paints_a_low_hole_fraction_green():
    """AUC(hole_worst -> reject) is 0.115 over the reviewed corpus: 48 of 81
    rejected meshes measured *exactly* 0.0, because the dominant failure mode
    is a solid slab and a slab has no visible openings. 0 of the 3 accepted
    ones did.

    Scanned rather than exercised, because the failure is a colour constant in
    a branch and the branch is three lines long -- the mistake is easy to
    reintroduce and expensive to notice, since it looks right.
    """
    import inspect

    from warlock.studio import widgets

    source = inspect.getsource(widgets.quality_badge)
    audit_half = source.split('audit = params.get("mesh_audit")', 1)[1]
    # Code only: the comment above the branch names ``theme.OK`` precisely
    # because it is explaining why the branch no longer paints it.
    audit_half = "\n".join(
        line for line in audit_half.splitlines() if not line.lstrip().startswith("#")
    )
    assert "theme.OK" not in audit_half, (
        "the silhouette audit must not paint a good verdict: a featureless "
        "slab scores 0.0 and would wear it"
    )
    # The escalation is kept: a *high* reading is still real evidence.
    assert "theme.ERR" in audit_half


def test_the_review_line_reports_the_reading_that_exists():
    """``mesh_audit`` has no ``verdict`` key -- the worker stores worst, mean,
    faces and resolution -- so the branch that read one was dead from the day
    it was typed, exactly as ``report["verdict"]`` was in the badge."""
    from warlock.studio import review_mode

    lines = review_mode.mesh_lines({"params": {"mesh_audit": {"worst": 0.0304, "mean": 0.01}}})
    assert any("3.0%" in line for line in lines)
    assert not any("verdict" in line for line in lines)


def test_the_band_sweep_table_does_not_state_the_inverted_rule_generally(capsys):
    from warlock import sweep as sweep_mod

    sweep_mod.print_table(
        [
            {"band": "auto", "worst": 0.02, "mean": 0.01, "faces": 100, "seconds": 1.0},
            {"band": "8", "worst": 0.01, "mean": 0.005, "faces": 100, "seconds": 1.0},
        ],
        1024,
    )
    out = capsys.readouterr().out
    assert "slab" in out  # the caveat is stated where the ranking is read
    assert "one subject at one seed" in out


# --- P124: the taxonomy says what it is contributing ------------------------


def test_a_style_and_a_palette_that_disagree_are_named():
    params = guidance.normalize({"art_style": "snes", "palette": "mono"})
    (conflict,) = guidance.colour_conflicts(params)
    assert "SNES era" in conflict
    assert "Monochrome" in conflict


def test_a_style_and_a_palette_that_agree_are_silent():
    params = guidance.normalize({"art_style": "snes", "palette": "vibrant"})
    assert guidance.colour_conflicts(params) == []


def test_the_documented_case(  # the one that produced the item
):
    """TODO.md §5: the brief asked for "black and silver and blue" and the
    composed prompt contributed "vivid saturated colours" on top of it."""
    params = guidance.normalize({"art_style": "snes"})
    (conflict,) = guidance.colour_conflicts(params, "a rogue in black and silver and blue")
    assert "saturated" in conflict
    for colour in ("black", "silver", "blue"):
        assert colour in conflict


def test_one_named_colour_is_not_a_palette():
    """"A red sword" names a subject's colour and asks for no scheme; flagging
    it would make the warning noise, and a warning that is usually noise is one
    nobody reads when it is not."""
    params = guidance.normalize({"art_style": "snes"})
    assert guidance.colour_conflicts(params, "a red sword") == []


def test_a_stated_palette_answers_the_brief_and_ends_the_matter():
    """One disagreement, one sentence: a chosen palette *is* the user's colour
    decision, so reporting the brief on top of it would be two sentences about
    one thing."""
    params = guidance.normalize({"art_style": "snes", "palette": "mono"})
    conflicts = guidance.colour_conflicts(params, "a rogue in black and silver and blue")
    assert len(conflicts) == 1
    assert "Monochrome" in conflicts[0]


def test_a_style_with_no_colour_claim_says_nothing():
    params = guidance.normalize({"art_style": "ps1"})
    assert guidance.colour_conflicts(params, "black and silver and blue") == []


def test_the_preview_carries_them(svc):
    from warlock.service import system as svc_system

    out = svc_system.prompt_preview(
        svc, {"art_style": "snes", "palette": "mono"}, "a rogue"
    )
    assert out["conflicts"]


# --- N113: one set of remedies, two checks ----------------------------------


def test_the_dispatch_refusal_offers_the_job_s_own_remedies():
    """The check that fires *after* the user has waited in the queue used to be
    the one with the least to say about what to change."""
    message = vram.dispatch_shortfall_message(
        20.0, 12.0, 5.0, {"control": "depth", "ip_adapter": "plus"}, exclusive=False
    )
    assert "20.0 GiB" in message and "12.0 GiB" in message
    assert "close other GPU applications" in message
    assert "ControlNet" in message
    assert "IP-Adapter" in message
    assert "WARLOCK_VRAM_EXCLUSIVE" in message


def test_both_refusals_share_one_remedy_list():
    params = {"control": "depth"}
    plan = vram.Plan(
        total_gib=12.0, budget_gib=8.0, exclusive=False, reason="test", explicit=False
    )
    door = vram.shortfall_message(20.0, plan, params)
    dispatch = vram.dispatch_shortfall_message(20.0, 8.0, 8.0, params, exclusive=False)
    for remedy in ("turn off ControlNet conditioning", "WARLOCK_VRAM_EXCLUSIVE=1"):
        assert remedy in door
        assert remedy in dispatch


# --- D42: a texture that could not be loaded is not a texture that was absent -


def test_the_loader_counts_images_it_could_not_use():
    """Counted per image *source*, not per material reference: ``_images``
    memoizes, so one unreadable atlas shared by six primitives is one loss."""
    from warlock.studio.viewer import gltf

    reader = gltf._Reader({"images": [], "bufferViews": []}, b"")
    assert reader.skipped == 0
    assert reader._image_bytes({"uri": "textures/albedo.png"}) is None
    assert reader.skipped == 1
    assert reader._image_bytes({"uri": "data:image/png;base64,!!!not base64!!!"}) is None
    assert reader.skipped == 2


def test_a_model_reports_its_losses():
    from warlock.studio.viewer import gltf

    model = gltf.Model([], [], [], [], skipped_textures=3)
    assert model.skipped_textures == 3


def test_a_model_defaults_to_no_losses():
    """Every existing construction site passes four arguments and must keep
    meaning "nothing was lost"."""
    from warlock.studio.viewer import gltf

    assert gltf.Model([], [], [], []).skipped_textures == 0


# --- O117: an empty name is not an ambiguity worth protecting ---------------


def test_an_empty_file_name_is_distinguishable_from_an_unknown_one(svc):
    from warlock.service import derive as svc_derive

    with pytest.raises(NotFound) as empty:
        svc_derive.get_file(svc, "0123456789ab", "")
    with pytest.raises(NotFound) as unknown:
        svc_derive.get_file(svc, "0123456789ab", "secrets.txt")
    assert empty.value.message != unknown.value.message
    assert empty.value.message == "no file was named"


def test_an_empty_job_id_is_distinguishable_from_a_pruned_one():
    from warlock.service.validation import check_job_id

    with pytest.raises(NotFound) as empty:
        check_job_id("")
    with pytest.raises(NotFound) as malformed:
        check_job_id("../../etc/passwd")
    assert empty.value.message == "no job was named"
    # Still deliberately ambiguous: saying which leaks what the store holds.
    assert malformed.value.message == "no such job"

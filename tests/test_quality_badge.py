"""The audit fallback reads the keys the audit actually writes.

``queue._audit_mesh`` stores ``{worst, mean, faces, resolution}`` -- there has
never been a ``hole_ratio`` key, so a badge or a caption keyed on one is dead
code that silently shows nothing for every job with an audit but no report.
"""

from __future__ import annotations

from warlock.studio import widgets


def _job(**audit):
    return {"params": {"mesh_audit": {"mean": 0.01, "faces": 4, "resolution": 512, **audit}}}


def test_the_badge_falls_back_to_the_audits_worst_fraction(monkeypatch):
    drawn: list[str] = []
    monkeypatch.setattr(widgets, "text_colored", lambda colour, label: drawn.append(label))

    widgets.quality_badge(_job(worst=0.05))

    assert drawn == ["5.0% open"]


def test_the_badge_reads_the_welded_watertight_flag_not_the_raw_one(monkeypatch):
    """The report wins over the audit, and within the report the welded flag
    wins over the raw one: ``meshreport`` loads with ``process=False`` so the
    UV checks see split vertices, and the raw flag therefore says "not
    watertight" about every textured mesh trellis has ever produced."""
    drawn: list[str] = []
    monkeypatch.setattr(widgets, "text_colored", lambda colour, label: drawn.append(label))

    widgets.quality_badge(
        {
            "params": {
                "mesh_audit": {"worst": 0.05},
                "mesh_report": {"watertight": False, "welded_watertight": True},
            }
        }
    )

    assert drawn == ["watertight"]


def test_a_report_verdict_key_means_nothing_to_the_badge(monkeypatch):
    """There used to be a branch above both of the ones this file pins,
    painting ``report["verdict"]`` as good / usable / anything-else. Nothing
    has ever written that key -- ``meshreport.build`` writes ``status``, whose
    values are ready / review / invalid -- so it was dead from the day it was
    typed and every mesh fell past it. It is deleted rather than re-pointed at
    ``status``: that word pools the triangle budget, the UV set, both PBR maps
    and the pivot into one verdict, which is the inspector's section (where the
    reasons are listed beside it), and as the first branch it would have
    suppressed the welded-watertight badge for every mesh.
    """
    drawn: list[str] = []
    monkeypatch.setattr(widgets, "text_colored", lambda colour, label: drawn.append(label))

    widgets.quality_badge(
        {"params": {"mesh_audit": {"worst": 0.05}, "mesh_report": {"verdict": "good"}}}
    )

    assert drawn == ["5.0% open"]


def test_an_audit_without_a_measurement_draws_no_badge(monkeypatch):
    drawn: list[str] = []
    monkeypatch.setattr(widgets, "text_colored", lambda colour, label: drawn.append(label))

    widgets.quality_badge({"params": {"mesh_audit": {"faces": 4}}})

    assert drawn == []


def test_the_inspector_reports_visible_openings_from_the_audit(monkeypatch):
    from warlock.studio.panes import inspector

    lines: list[str] = []
    monkeypatch.setattr(widgets, "header", lambda label, **kwargs: True)
    monkeypatch.setattr(widgets, "muted", lines.append)

    inspector._quality(None, _job(worst=0.05))

    assert "visible openings: 5%" in lines

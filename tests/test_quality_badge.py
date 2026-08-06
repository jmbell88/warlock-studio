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

    assert "visible openings: 5.0%" in lines

"""The retarget panel's two restatements, removed.

Its labels said "Draft (20k)", "Standard (50k)", "Detailed (100k)" as literals
while ``pipelines/optimize.PROFILES`` said 20 000, 50 000 and 100 000 -- the one
lapse in a file that already renders ``CUSTOM_MIN``/``CUSTOM_MAX`` from the
module rather than typing them, and ``validate``'s docstring already states the
rule the labels broke: "``optimize.resolve`` is the authority on the range;
restating the numbers here rather than the rule would let the two drift". A
label that drifts is worse than a refusal that drifts, because it offers a
budget the button then does not use.

And ``_gltfpack_available`` called ``Path.exists`` on every frame the section
was open, for an answer that is fixed for the life of the process: the path
comes from Config and a vendored binary does not arrive while the app runs.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.pipelines import optimize
from warlock.studio.panes import retarget_panel

# --- the tier list ------------------------------------------------------------


def test_every_named_tier_is_one_the_pipeline_carries():
    named = {key for key, _label in retarget_panel.TIERS} - {"custom"}
    assert named == set(optimize.PROFILES)


def test_raw_is_first_because_it_is_the_one_that_needs_no_binary():
    """``TIERS[0]`` is the whole option list when gltfpack is absent, so its
    position is stated rather than derived from the dict's order."""
    assert retarget_panel.TIERS[0][0] == "raw"
    assert optimize.PROFILES["raw"] is None


def test_custom_is_last_and_is_not_a_profile():
    assert retarget_panel.TIERS[-1][0] == "custom"
    assert "custom" not in optimize.PROFILES


def test_each_label_states_its_own_budget():
    """The regression itself: the labels are read off ``PROFILES`` rather than
    typed beside it."""
    labels = dict(retarget_panel.TIERS)
    for key, budget in optimize.PROFILES.items():
        if budget is None:
            continue
        assert f"{budget // 1000}k" in labels[key], key


def test_a_new_tier_would_be_labelled_without_an_edit_here():
    assert retarget_panel.tier_label("coarse", 5_000) == "Coarse (5k)"
    assert retarget_panel.tier_label("raw", None) == "Raw (full density)"


def test_a_budget_that_is_not_a_round_thousand_is_printed_in_full():
    """Rounding it into "25k" would be a label naming a number the pipeline is
    not using -- the drift this whole change is about, reintroduced by the fix."""
    assert retarget_panel.tier_label("odd", 25_500) == "Odd (25,500)"


# --- the binary probe ---------------------------------------------------------


class _Probe:
    """A path that counts the syscall, which is the thing under test.

    A real ``Path`` would need ``pathlib`` monkeypatched to be counted, and that
    counts every other ``exists`` in the process with it.
    """

    def __init__(self, name: str, answer: bool) -> None:
        self.name = name
        self.answer = answer
        self.calls = 0

    def exists(self) -> bool:
        self.calls += 1
        return self.answer

    def __str__(self) -> str:
        return self.name


def _ctx(path):
    return SimpleNamespace(svc=SimpleNamespace(config=SimpleNamespace(gltfpack_exe=path)))


def test_the_binary_is_stat_ed_once_per_path_not_once_per_frame(monkeypatch):
    monkeypatch.setattr(retarget_panel, "_gltfpack_seen", {})
    probe = _Probe("vendor/gltfpack/gltfpack.exe", True)
    ctx = _ctx(probe)

    for _ in range(20):
        assert retarget_panel._gltfpack_available(ctx) is True

    assert probe.calls == 1


def test_an_absent_binary_is_remembered_too(monkeypatch):
    """Both answers, or the machine with no gltfpack -- the common one -- keeps
    paying the syscall this memo exists to remove."""
    monkeypatch.setattr(retarget_panel, "_gltfpack_seen", {})
    probe = _Probe("nowhere/gltfpack.exe", False)
    ctx = _ctx(probe)

    for _ in range(20):
        assert retarget_panel._gltfpack_available(ctx) is False

    assert probe.calls == 1


def test_two_services_do_not_inherit_each_others_answer(monkeypatch):
    """Keyed by the path rather than kept as a single flag: a test's config, or
    a second window's, names a different binary."""
    monkeypatch.setattr(retarget_panel, "_gltfpack_seen", {})

    assert retarget_panel._gltfpack_available(_ctx(_Probe("a.exe", True))) is True
    assert retarget_panel._gltfpack_available(_ctx(_Probe("b.exe", False))) is False


def test_a_config_that_raises_is_no_binary_rather_than_a_frame_that_dies(monkeypatch):
    monkeypatch.setattr(retarget_panel, "_gltfpack_seen", {})

    class _Exploding:
        @property
        def gltfpack_exe(self):
            raise RuntimeError("no config")

    ctx = SimpleNamespace(svc=SimpleNamespace(config=_Exploding()))
    assert retarget_panel._gltfpack_available(ctx) is False

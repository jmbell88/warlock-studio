"""The driver's pure half.

Import-level only, deliberately: pressing a control needs a real window and a
GL context, which is the whole reason ``exercise_mode.py`` is a script rather
than a test. What *can* be pinned here is everything that decides what the
driver does -- the refusal list, the digest, and the verdict classifier -- and
those are the parts whose being wrong would make a whole run's findings wrong
without anything looking broken.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registered before execution: a dataclass in the module resolves its
        # string annotations through ``sys.modules``, so a module that is not
        # there yet fails at its own decorator.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def driver():
    return _load("exercise_mode")


def test_the_refusal_list_is_not_empty_and_names_the_two_hazards(driver):
    assert driver.REFUSED
    assert driver.refused("Quit")
    assert driver.refused("Save As...")
    assert driver.refused("open…")
    # And nothing broad: a pattern that swallowed ordinary controls would skip
    # exactly what the pass exists to test.
    assert not driver.refused("Save")
    assert not driver.refused("Bucket")
    assert not driver.refused("Export sheet")


def test_delta_names_only_the_components_that_moved(driver):
    before = ("inker", "", None, "brush", 4, (), 0, False, ())
    after = ("inker", "", None, "bucket", 4, (), 0, False, ())
    assert driver.describe_delta(before, after) == ["tool: 'brush' -> 'bucket'"]
    assert driver.describe_delta(before, before) == []
    assert len(driver.DIGEST_NAMES) == len(before)


def _verdict(driver, **kwargs):
    base = dict(
        raised=None,
        enabled=True,
        reason="",
        toast_levels=(),
        submitted=(),
        state_delta=[],
        pixel_delta=0.0,
    )
    return driver.verdict(**(base | kwargs))


def test_verdict_maps_each_signal_to_its_label(driver):
    assert _verdict(driver) == "inert"
    assert _verdict(driver, pixel_delta=0.5) == "pixels-changed"
    assert _verdict(driver, state_delta=["tool: a -> b"]) == "state-changed"
    assert _verdict(driver, submitted=("export:1",)) == "submitted"
    assert _verdict(driver, enabled=False, reason="Open a drawing first.") == "disabled"
    assert _verdict(driver, enabled=False) == "disabled-no-reason"
    assert _verdict(driver, toast_levels=("error",)) == "toast-error"
    assert _verdict(driver, raised="Traceback...") == "raised"


def test_verdict_orders_by_severity_not_by_convenience(driver):
    # A control that crashed is a crash whatever else it also did.
    assert (
        _verdict(
            driver,
            raised="boom",
            submitted=("x",),
            state_delta=["mode: a -> b"],
            toast_levels=("error",),
        )
        == "raised"
    )
    # And an error toast outranks the work that produced it.
    assert _verdict(driver, submitted=("x",), toast_levels=("error",)) == "toast-error"


def test_a_pixel_flicker_below_the_threshold_is_still_inert(driver):
    assert driver.PIXEL_EPSILON > 0
    assert _verdict(driver, pixel_delta=driver.PIXEL_EPSILON) == "inert"


def test_always_look_covers_every_verdict_a_reader_must_judge(driver):
    for name in ("raised", "inert", "toast-error", "disabled-no-reason", "hard-reset"):
        assert name in driver.ALWAYS_LOOK


def test_keys_are_stable_and_disambiguate_a_repeated_label(driver):
    from warlock.studio.probe import Control

    def _one(name, pane="inker_tools"):
        return Control(label=name, kind="button", rect=(0, 0, 1, 1), pane=pane)

    seen: dict[str, int] = {}
    first = driver.key_for(_one("Add"), seen)
    second = driver.key_for(_one("Add"), seen)
    assert first == "inker_tools/button/Add#0"
    assert second == "inker_tools/button/Add#1"
    assert driver.key_for(_one("Add", pane=""), {}) == "floating/button/Add#0"
    assert "/" not in driver.safe_name(first)


def test_the_harness_is_shared_rather_than_reimplemented(driver):
    """Two scripts booting the app two ways is the drift the extraction ends."""

    harness = _load("_appharness")
    for name in (
        "boot",
        "capture",
        "close_popups",
        "seed",
        "seed_asset",
        "seed_review",
        "seed_tile",
        "seed_matte",
        "WARMUP",
        "SETTLE_FRAMES",
    ):
        assert hasattr(harness, name), name
    shots = (SCRIPTS / "screenshot_modes.py").read_text(encoding="utf-8")
    assert "from _appharness import" in shots
    # The boot sequence must live in exactly one place.
    assert "app.setup_window()" not in shots
    assert "app.setup_window(size_override=size)" in (SCRIPTS / "_appharness.py").read_text(
        encoding="utf-8"
    )


def test_the_driver_reports_its_own_blind_spot(driver):
    from test_probe import RAW_IMGUI_CONTROLS

    assert driver.raw_imgui_controls() == RAW_IMGUI_CONTROLS

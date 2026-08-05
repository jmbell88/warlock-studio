"""A profile's style anchor: one image on disk, referenced by filename.

The image is a file rather than base64 in studio_settings.json because that
file is rewritten on a one-second debounce for every UI preference, and a
megabyte of PNG in it would be rewritten with them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock import models
from warlock.studio import profiles


class FakeSettings:
    """The two methods profiles.py uses, with no disk behind them."""

    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


@pytest.fixture
def env(tmp_path):
    return SimpleNamespace(
        settings=FakeSettings(), config=SimpleNamespace(data_dir=tmp_path)
    )


PNG = b"\x89PNG\r\n\x1a\n-not-really-a-png-but-bytes-are-bytes"


def test_the_anchor_adapter_is_a_real_registry_key():
    # The whole feature turns into a submit-time refusal if this drifts.
    assert profiles.ANCHOR_ADAPTER in models.IP_ADAPTERS


def test_setting_an_anchor_writes_a_file_and_records_only_its_name(env):
    profiles.save_profile(env.settings, "house", {"base_model": "turbo"})
    profiles.set_anchor(env.settings, env.config, "house", PNG)

    stored = profiles.list_profiles(env.settings)["house"]
    assert stored["anchor"].endswith(".png")
    assert "/" not in stored["anchor"] and "\\" not in stored["anchor"]
    path = profiles.anchor_path(env.config, stored)
    assert path is not None and path.read_bytes() == PNG


def test_an_anchor_carries_its_own_strength(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG, scale=0.85)
    assert profiles.list_profiles(env.settings)["house"]["anchor_scale"] == 0.85


def test_an_anchor_without_a_strength_takes_the_registry_default(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    stored = profiles.list_profiles(env.settings)["house"]
    assert stored["anchor_scale"] == models.DEFAULT_IP_SCALE


def test_saving_a_profile_again_keeps_the_anchor_it_already_had(env):
    # capture() reads the *form*, which has no anchor field -- so a plain
    # re-save would drop it every time the user edited a taxonomy select.
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    profiles.save_profile(env.settings, "house", {"base_model": "sdxl"})

    stored = profiles.list_profiles(env.settings)["house"]
    assert stored["base_model"] == "sdxl"
    assert profiles.anchor_path(env.config, stored) is not None


def test_an_explicit_anchor_in_the_saved_fields_wins(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    profiles.save_profile(env.settings, "house", {"anchor": "", "anchor_scale": 0.5})
    assert profiles.list_profiles(env.settings)["house"].get("anchor") == ""


def test_clearing_an_anchor_removes_the_file(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    path = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])

    profiles.clear_anchor(env.settings, env.config, "house")

    assert not path.exists()
    assert not profiles.list_profiles(env.settings)["house"].get("anchor")


def test_replacing_an_anchor_removes_the_old_file(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    old = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])

    profiles.set_anchor(env.settings, env.config, "house", b"second-image")

    assert not old.exists()
    new = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])
    assert new.read_bytes() == b"second-image"


def test_deleting_a_profile_removes_its_anchor(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    path = profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house"])

    profiles.delete_profile(env.settings, "house", env.config)

    assert not path.exists()


def test_deleting_a_profile_keeps_an_anchor_another_profile_still_points_at(env):
    # This is what a rename does: the editor saves under the new name and
    # deletes the old entry, and for a moment both name the same file.
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    stored = profiles.list_profiles(env.settings)["house"]
    profiles.save_profile(env.settings, "house2", dict(stored))
    path = profiles.anchor_path(env.config, stored)

    profiles.delete_profile(env.settings, "house", env.config)

    assert path.exists()
    assert profiles.anchor_path(env.config, profiles.list_profiles(env.settings)["house2"])


def test_delete_without_a_config_still_removes_the_profile(env):
    # The old two-argument call site is still valid; it just cannot tidy up.
    profiles.save_profile(env.settings, "house", {})
    profiles.delete_profile(env.settings, "house")
    assert profiles.list_profiles(env.settings) == {}


def test_a_hand_edited_anchor_name_is_refused_before_it_becomes_a_path(env):
    # The same rule rigging.pose_path follows: a caller-supplied string that
    # names a file is validated, not joined.
    assert profiles.anchor_path(env.config, {"anchor": "../../secrets.png"}) is None
    assert profiles.anchor_path(env.config, {"anchor": "nope.txt"}) is None
    assert profiles.anchor_path(env.config, {}) is None


def test_a_recorded_anchor_whose_file_is_gone_reads_as_no_anchor(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    stored = profiles.list_profiles(env.settings)["house"]
    profiles.anchor_path(env.config, stored).unlink()
    assert profiles.anchor_path(env.config, stored) is None


def test_the_active_profiles_anchor_is_what_active_anchor_returns(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG, scale=0.7)
    profiles.set_active(env.settings, "house")

    found = profiles.active_anchor(env.settings, env.config)

    assert found is not None
    path, scale = found
    assert Path(path).read_bytes() == PNG
    assert scale == 0.7


def test_no_active_profile_means_no_anchor(env):
    profiles.save_profile(env.settings, "house", {})
    profiles.set_anchor(env.settings, env.config, "house", PNG)
    assert profiles.active_anchor(env.settings, env.config) is None

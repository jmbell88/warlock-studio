"""The dependency-pack registry and planner, headlessly.

Everything here is pure. Nothing in this file may install or download
anything: ``packs`` plans, and the performing half is somebody else's problem
-- the same split, and the same reason, as ``test_fetch.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from warlock import fetch, packs

ROOT = Path(__file__).resolve().parents[1]

SHA_A = "a" * 64
SHA_B = "b" * 64


def wheel(filename: str, size: int, *, packs_: tuple[str, ...], installed: int = 0) -> dict:
    return {
        "filename": filename,
        "size_bytes": size,
        "sha256": SHA_A,
        "installed_bytes": installed,
        "packs": list(packs_),
    }


def manifest(*wheels: dict) -> packs.Manifest:
    return packs.parse_manifest({"version": packs.MANIFEST_VERSION, "wheels": list(wheels)})


# --- the registry ------------------------------------------------------------


def test_every_pack_probe_resolves_in_a_fully_synced_environment():
    """The typo pin, and it is the whole value of ``probe``.

    A misspelt module name never raises: ``find_spec`` simply does not find it,
    so the pack reads as missing forever on a machine where it is installed --
    the pane offers a download that is already there and the mode stays greyed.
    Every documented install and every CI lane syncs all four extras, so a
    failure here is a wrong name rather than a lean environment.
    """
    for pack in packs.PACKS:
        assert packs.installed(pack), (
            f"{pack.key}: {packs.missing(pack)} did not resolve; either the "
            f"probe names are wrong or this environment is missing "
            f"{pack.install_hint}"
        )


def test_a_pack_composes_its_own_install_hint():
    assert packs.find("music").install_hint == "uv sync --extra music"
    assert packs.find("rig").install_hint == "uv sync --extra rig"


def test_the_modes_a_pack_names_are_real_modes():
    """``Pack.modes`` is strings because ``studio`` may not be imported from
    ``warlock.packs``. Strings drift, so they are pinned here instead."""
    from warlock.studio import modes

    known = {key for key, _label, _icon in modes.MODES}
    for pack in packs.PACKS:
        assert pack.modes, f"{pack.key} unlocks nothing"
        for key in pack.modes:
            assert key in known, f"{pack.key} names a mode that does not exist: {key}"


def test_the_extras_a_pack_names_are_real_extras():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    declared = set(re.findall(r"^([a-z0-9_-]+) = \[", block, re.M))
    for pack in packs.PACKS:
        for extra in pack.extras:
            assert extra in declared, f"{pack.key} names a missing extra: {extra}"


def test_studio_is_not_a_pack():
    """It is the window, and the base installer carries it. Offering it as an
    optional download would offer to install the thing doing the offering."""
    assert "studio" not in packs.KEYS
    assert all("studio" not in pack.extras for pack in packs.PACKS)


def test_an_unknown_pack_key_raises_rather_than_planning_less():
    with pytest.raises(KeyError):
        packs.chosen_packs(["text2image", "text2img"])
    with pytest.raises(KeyError):
        packs.plan(manifest(wheel("a-1-py3-none-any.whl", 10, packs_=("rig",))), ["nope"])


def test_chosen_packs_answers_in_registry_order():
    picked = packs.chosen_packs(["music", "rig"])
    assert [p.key for p in picked] == ["rig", "music"]


# --- the size the lock cannot tell you ---------------------------------------


def test_the_sizeless_lock_entries_are_exactly_the_declared_three():
    """Pinned in both directions, because both are how the figure goes wrong.

    A fourth distribution resolving with no declared ``size`` would be summed
    as zero by anything that trusted the lock; one of these three gaining a
    size would mean the rule could be relaxed. Either is a change a person has
    to read, which is what this test is for -- see ``packs.SIZELESS_DISTS``.
    """
    text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    sizeless = set()
    for block in text.split("[[package]]")[1:]:
        name = re.search(r'^name = "(.+?)"', block, re.M)
        wheels = re.findall(r'\{ url = "[^"]+\.whl"[^}]*\}', block)
        if name is None or not wheels:
            continue
        if any("size = " not in one for one in wheels):
            sizeless.add(name.group(1))
    assert sizeless == set(packs.SIZELESS_DISTS)


# --- reading a manifest -------------------------------------------------------


def test_a_manifest_round_trips_what_the_generator_measured():
    one = manifest(
        wheel(
            "torch-2.8.0-cp313-win_amd64.whl", 3_000, packs_=("text2image",), installed=9_000
        )
    )
    assert one.version == packs.MANIFEST_VERSION
    assert one.wheels[0].filename == "torch-2.8.0-cp313-win_amd64.whl"
    assert one.wheels[0].size_bytes == 3_000
    assert one.wheels[0].installed_bytes == 9_000
    assert one.wheels[0].packs == ("text2image",)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"version": 2, "wheels": [wheel("a-1-py3-none-any.whl", 1, packs_=("rig",))]},
        {"version": packs.MANIFEST_VERSION, "wheels": []},
        # A name that is a path, not a wheel: the performing half joins this
        # onto the pack directory, so a traversal must never survive the parse.
        {
            "version": packs.MANIFEST_VERSION,
            "wheels": [wheel("../../evil-1-py3-none-any.whl", 1, packs_=("rig",))],
        },
        {"version": packs.MANIFEST_VERSION, "wheels": [wheel("a.tar.gz", 1, packs_=("rig",))]},
        # Claimed by no pack, and by a pack that does not exist.
        {
            "version": packs.MANIFEST_VERSION,
            "wheels": [wheel("a-1-py3-none-any.whl", 1, packs_=())],
        },
        {
            "version": packs.MANIFEST_VERSION,
            "wheels": [wheel("a-1-py3-none-any.whl", 1, packs_=("wobble",))],
        },
        {
            "version": packs.MANIFEST_VERSION,
            "wheels": [{"filename": "a-1-py3-none-any.whl", "size_bytes": 1, "packs": ["rig"]}],
        },
        {
            "version": packs.MANIFEST_VERSION,
            "wheels": [
                {
                    "filename": "a-1-py3-none-any.whl",
                    "size_bytes": -1,
                    "sha256": SHA_A,
                    "packs": ["rig"],
                }
            ],
        },
    ],
)
def test_a_manifest_that_is_not_exactly_the_shape_is_refused(payload):
    with pytest.raises(packs.ManifestError):
        packs.parse_manifest(payload)


def test_one_filename_listed_twice_is_refused():
    """Two entries for one name means the generator collected two builds of one
    distribution; installing either silently is how a runtime ends up carrying
    a wheel nobody chose."""
    with pytest.raises(packs.ManifestError):
        manifest(
            wheel("torch-2.8.0-cp313-win_amd64.whl", 3_000, packs_=("text2image",)),
            wheel("torch-2.8.0-cp313-win_amd64.whl", 3_000, packs_=("music",)),
        )


def test_load_manifest_refuses_a_file_that_is_not_json(tmp_path):
    path = tmp_path / packs.MANIFEST_NAME
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(packs.ManifestError):
        packs.load_manifest(path)


# --- the plan -----------------------------------------------------------------


def _shared() -> packs.Manifest:
    """The real overlap in miniature: torch is claimed by two packs, and each
    pack has one wheel of its own."""
    return manifest(
        wheel("torch-2.8.0-cp313-win_amd64.whl", 3_000, packs_=("text2image", "music")),
        wheel("diffusers-0.31-py3-none-any.whl", 100, packs_=("text2image", "music")),
        wheel("timm-1.0-py3-none-any.whl", 50, packs_=("text2image",)),
        wheel("librosa-0.10-py3-none-any.whl", 20, packs_=("music",)),
        wheel("bpy-5.2-cp313-win_amd64.whl", 900, packs_=("rig",)),
    )


def test_ticking_both_torch_packs_downloads_torch_once():
    """The registry's own four-SDXL-recipes case. ``text2image`` and ``music``
    share 27 distributions; a pack-keyed sum would tell the user to expect
    nearly twice the download and a pack-keyed install would write torch
    twice."""
    plan = packs.plan(_shared(), ["text2image", "music"])
    names = [w.filename for w in plan]
    assert names.count("torch-2.8.0-cp313-win_amd64.whl") == 1
    assert packs.total_bytes(plan) == 3_000 + 100 + 50 + 20


def test_a_plan_leaves_the_other_packs_wheels_alone():
    plan = packs.plan(_shared(), ["rig"])
    assert [w.filename for w in plan] == ["bpy-5.2-cp313-win_amd64.whl"]
    assert packs.total_bytes(plan) == 900


def test_a_plan_reads_in_manifest_order_however_it_was_ticked():
    one = packs.plan(_shared(), ["music", "text2image"])
    two = packs.plan(_shared(), ["text2image", "music"])
    assert [w.filename for w in one] == [w.filename for w in two]


def test_packs_may_be_named_by_object_or_by_key():
    by_key = packs.plan(_shared(), ["rig"])
    by_pack = packs.plan(_shared(), packs.chosen_packs(["rig"]))
    assert by_key == by_pack


def test_an_unmeasured_wheel_withholds_the_installed_figure_entirely():
    """All-or-nothing, because a partial sum looks like an answer. The
    unmeasured wheel is overwhelmingly likely to be the largest one."""
    measured = manifest(
        wheel("a-1-py3-none-any.whl", 10, packs_=("rig",), installed=30),
        wheel("b-1-py3-none-any.whl", 20, packs_=("rig",), installed=60),
    )
    assert packs.installed_bytes(measured.wheels) == 90
    partial = manifest(
        wheel("a-1-py3-none-any.whl", 10, packs_=("rig",), installed=30),
        wheel("b-1-py3-none-any.whl", 20, packs_=("rig",)),
    )
    assert packs.installed_bytes(partial.wheels) == 0


# --- the refusal --------------------------------------------------------------


@pytest.fixture
def free(monkeypatch):
    """Free space by volume anchor, so a test can put two paths on two drives."""
    space: dict[str, float] = {}

    def fake_free_gib(path):
        return space.get(Path(path).anchor)

    monkeypatch.setattr(fetch, "free_gib", fake_free_gib)
    return space


GIB = 1024**3


def test_a_plan_that_fits_is_not_refused(free):
    free["C:\\"] = 500.0
    free["/"] = 500.0
    plan = packs.plan(_shared(), ["rig"])
    assert packs.disk_refusal(
        plan, cache_dir=Path("C:/home/packs"), install_dir=Path("C:/app/lib")
    ) is None


def test_an_empty_plan_is_never_refused(free):
    assert packs.disk_refusal([], cache_dir=Path("C:/a"), install_dir=Path("C:/b")) is None


def test_the_wheels_and_the_unpacked_bytes_are_charged_to_one_volume_together(free):
    """Not double-counting: the wheels are still on disk while the packages are
    written out of them, so the peak really is the sum."""
    free["C:\\"] = 12.0
    plan = [packs.Wheel("torch.whl", 5 * GIB, SHA_B, installed_bytes=6 * GIB, packs=("rig",))]
    said = packs.disk_refusal(
        plan, cache_dir=Path("C:/home/packs"), install_dir=Path("C:/app/lib")
    )
    assert said is not None
    # 5 downloaded + 6 unpacked + 2 headroom = 13, against 12 free.
    assert "13.0 GB" in said and "12.0 GB is free" in said
    # One volume, so the message does not name a drive nobody had to choose.
    assert " on " not in said


def test_a_roomy_cache_drive_does_not_approve_a_write_to_a_full_runtime_one(free):
    """MDL-09 in a new place. A per-user install routinely puts the Warlock
    home and the application runtime on two drives."""
    free["D:\\"] = 500.0
    free["C:\\"] = 1.0
    plan = [packs.Wheel("torch.whl", 3 * GIB, SHA_B, installed_bytes=8 * GIB, packs=("rig",))]
    said = packs.disk_refusal(
        plan, cache_dir=Path("D:/home/packs"), install_dir=Path("C:/app/lib")
    )
    assert said is not None
    assert "C:\\" in said, said


def test_an_unmeasured_plan_still_budgets_the_download_it_does_know(free):
    """No install-volume claim without a figure, but the wheels are always
    known and a cache drive can be full on its own."""
    free["D:\\"] = 1.0
    free["C:\\"] = 0.5
    plan = [packs.Wheel("torch.whl", 3 * GIB, SHA_B, packs=("rig",))]
    said = packs.disk_refusal(
        plan, cache_dir=Path("D:/home/packs"), install_dir=Path("C:/app/lib")
    )
    assert said is not None
    # The nearly-empty install volume is not what was refused: it was never
    # budgeted, because nothing measured what would land there.
    assert "5.0 GB" in said  # 3 downloaded + 2 headroom
    assert "D:\\" not in said and "C:\\" not in said


def test_unreadable_free_space_is_not_a_refusal(free):
    plan = packs.plan(_shared(), ["rig"])
    assert packs.disk_refusal(
        plan, cache_dir=Path("Z:/nowhere"), install_dir=Path("Z:/nowhere")
    ) is None


# --- the rule the model downloader and the pack installer share ---------------


def test_the_volume_rule_is_one_implementation(monkeypatch):
    """``fetch.disk_refusal`` and ``packs.disk_refusal`` phrase two different
    subjects and must not decide differently about which drive is short."""
    free = {"C:\\": 1.0, "D:\\": 500.0}
    monkeypatch.setattr(fetch, "free_gib", lambda path: free.get(Path(path).anchor))
    short = fetch.volume_refusal([(Path("D:/a"), 1.0), (Path("C:/b"), 10.0)])
    assert short is not None
    assert short.volume == "C:\\"
    assert short.need_gib == pytest.approx(12.0)
    assert short.free_gib == pytest.approx(1.0)
    assert short.volumes == 2

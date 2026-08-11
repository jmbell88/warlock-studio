"""``downloads.uninstall``: the only code in the tree that deletes weights.

Every test here runs against the ``svc`` fixture's throwaway model root, which
``conftest._materialize_generative_weights`` fills with empty files. Nothing in
this file may be pointed at a real install -- the fixture pins
``WARLOCK_T2I_ROOT`` under ``tmp_path`` for exactly that reason, and the
containment rule in ``fetch.removal_plan`` is the second line of defence.
"""

from __future__ import annotations

import pytest

from warlock import fetch
from warlock.service import downloads as svc_downloads
from warlock.service.errors import Conflict, Failed, Invalid, NotFound


def _lora(svc, name: str):
    return svc.config.t2i_model_root / "loras" / name


def test_removing_a_style_lora_takes_one_file_and_says_what_it_freed(svc):
    path = _lora(svc, "pixel-art-xl.safetensors")
    assert path.exists()

    result = svc_downloads.uninstall(svc, ["lora:pixelxl"])

    assert not path.exists()
    assert result["removed"] == [str(path)]
    assert result["freed_gib"] == pytest.approx(0.2)
    assert not fetch.find("lora:pixelxl").is_present(svc.config)


def test_one_shared_recipe_leaves_the_checkpoint_the_others_stand_on(svc):
    """The rule the whole feature turns on. Removing ``sdxl`` must take its
    Hyper-SD adapter and *not* the 7 GiB directory ``sdxl_cfg``, ``pixel`` and
    ``lightning`` are all still standing on."""
    base_dir = fetch.base_model_dir(svc.config, fetch.find("base:sdxl").spec)
    hyper = _lora(svc, "Hyper-SDXL-4steps-lora.safetensors")

    result = svc_downloads.uninstall(svc, ["base:sdxl"])

    assert not hyper.exists()
    assert base_dir.is_dir(), "the shared checkpoint must survive"
    assert result["freed_gib"] == pytest.approx(0.8)
    # And the three siblings still read as present.
    for key in ("base:sdxl_cfg", "base:pixel", "base:lightning"):
        assert fetch.find(key).is_present(svc.config), key


def test_all_four_recipes_together_do_take_the_checkpoint(svc):
    base_dir = fetch.base_model_dir(svc.config, fetch.find("base:sdxl").spec)
    result = svc_downloads.uninstall(
        svc, ["base:sdxl", "base:sdxl_cfg", "base:pixel", "base:lightning"]
    )
    assert not base_dir.exists()
    assert result["freed_gib"] == pytest.approx(8.6)
    # The directory went via a rename, and nothing of it is left behind.
    leftovers = [
        p.name
        for p in svc.config.t2i_model_root.iterdir()
        if p.name.startswith(svc_downloads.TRASH_PREFIX)
    ]
    assert leftovers == []


def test_an_unknown_row_is_a_not_found(svc):
    with pytest.raises(NotFound):
        svc_downloads.uninstall(svc, ["lora:no-such-thing"])


def test_a_queued_job_blocks_every_removal(svc):
    """Conservative and whole-queue: a running job may have loaded weights its
    own row never mentions."""
    svc.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    path = _lora(svc, "pixel-art-xl.safetensors")

    with pytest.raises(Conflict, match="queued or running"):
        svc_downloads.uninstall(svc, ["lora:pixelxl"])
    assert path.exists()


def test_the_env_var_override_is_refused_with_nothing_deleted(svc, monkeypatch, tmp_path):
    elsewhere = svc.config.t2i_model_root / "elsewhere"
    elsewhere.mkdir(parents=True, exist_ok=True)
    (elsewhere / "model_index.json").write_bytes(b"")
    monkeypatch.setattr(svc.config, "t2i_turbo_dir", elsewhere)

    with pytest.raises(Invalid, match="WARLOCK_T2I_DIR"):
        svc_downloads.uninstall(svc, ["base:turbo"])
    assert elsewhere.is_dir()


def test_a_selection_that_frees_nothing_is_refused_rather_than_reported_done(svc):
    """``base:sdxl_cfg`` shares its whole footprint with three siblings and has
    no adapter of its own, so there is genuinely nothing to delete."""
    with pytest.raises(Invalid, match="nothing to remove"):
        svc_downloads.uninstall(svc, ["base:sdxl_cfg"])


def test_the_resident_image_model_is_unloaded_before_anything_is_deleted(svc):
    """Windows will not delete a mapped safetensors, and this process is the
    one holding the mapping. Recorded on a fake worker, through the same
    ``call_on_loop`` the real one is reached by."""
    calls: list[str] = []

    class FakeWorker:
        async def unload_text2image(self) -> None:
            calls.append("unload")
            assert _lora(svc, "pixel-art-xl.safetensors").exists(), (
                "the unload must happen before the delete, not after"
            )

    svc.worker = FakeWorker()
    svc_downloads.uninstall(svc, ["lora:pixelxl"])
    assert calls == ["unload"]


def test_a_locked_file_fails_loudly_and_the_trash_is_swept_next_time(svc, monkeypatch):
    """The Windows case: the rename succeeded, the rmtree did not. The model is
    already gone as far as any reader is concerned -- that is the point of the
    rename -- and the next uninstall clears the residue."""
    import shutil as shutil_mod

    base_dir = fetch.base_model_dir(svc.config, fetch.find("base:juggernaut").spec)
    real = shutil_mod.rmtree
    boom = {"armed": True}

    def once(path, *args, **kwargs):
        if boom["armed"]:
            boom["armed"] = False
            raise PermissionError(13, "in use")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(svc_downloads.shutil, "rmtree", once)
    with pytest.raises(Failed, match="another process holds"):
        svc_downloads.uninstall(svc, ["base:juggernaut"])

    # Renamed out of the way, so the row already reads absent.
    assert not base_dir.exists()
    assert not fetch.find("base:juggernaut").is_present(svc.config)
    trash = [
        p for p in svc.config.t2i_model_root.iterdir()
        if p.name.startswith(svc_downloads.TRASH_PREFIX)
    ]
    assert len(trash) == 1

    # The next removal sweeps it, opportunistically and without being asked.
    svc_downloads.uninstall(svc, ["lora:pixelxl"])
    assert not any(
        p.name.startswith(svc_downloads.TRASH_PREFIX)
        for p in svc.config.t2i_model_root.iterdir()
    )


def test_removing_the_default_base_model_is_allowed(svc):
    """No special case: it degrades to the friendly refusal every other missing
    checkpoint gets, and a rule the user cannot see would be worse."""
    from warlock import models
    from warlock.service import jobs as svc_jobs

    # Everything sharing sdxl-base-1.0, so the directory actually goes.
    svc_downloads.uninstall(
        svc, ["base:sdxl", "base:sdxl_cfg", "base:pixel", "base:lightning"]
    )
    assert models.DEFAULT_BASE_MODEL == "sdxl_cfg"
    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(svc, kind="text", prompt="a rock", output="reference")
    assert "Settings" in caught.value.message
    assert caught.value.rows == ("base:sdxl_cfg",)


def test_rows_carry_what_a_removal_would_free(svc):
    rows = {r["row_key"]: r for r in svc_downloads.rows(svc)}
    # One of four recipes: 0.8, not 7. The honesty risk the feature carries.
    assert rows["base:sdxl"]["removable"] is True
    assert rows["base:sdxl"]["freed_gib"] == pytest.approx(0.8)
    # And the recipe with no adapter of its own frees nothing, so no button.
    assert rows["base:sdxl_cfg"]["removable"] is False
    assert rows["base:sdxl_cfg"]["freed_gib"] == pytest.approx(0.0)
    assert rows["lora:pixelxl"]["freed_gib"] == pytest.approx(0.2)


def test_an_absent_row_is_offered_nothing_to_remove(svc):
    _lora(svc, "pixel-art-xl.safetensors").unlink()
    row = next(r for r in svc_downloads.rows(svc) if r["row_key"] == "lora:pixelxl")
    assert row["present"] is False
    assert "removable" not in row and "freed_gib" not in row

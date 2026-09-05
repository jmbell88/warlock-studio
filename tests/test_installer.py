"""The checkout-shaped Windows installer and its pinned native payload."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "installer"
MANIFEST = INSTALLER / "runtime-manifest.json"


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "warlock_installer_verify", INSTALLER / "verify_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_manifest_covers_every_shipped_vendor_file_once() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    named = [entry["path"] for entry in payload["files"]]
    assert len(named) == len(set(named))
    assert set(named) == {
        "vendor/trellis/cublas64_13.dll",
        "vendor/trellis/cublasLt64_13.dll",
        "vendor/trellis/cudart64_13.dll",
        "vendor/trellis/ggml-base.dll",
        "vendor/trellis/ggml-cpu.dll",
        "vendor/trellis/ggml-cuda.dll",
        "vendor/trellis/ggml.dll",
        "vendor/trellis/trellis-cli.exe",
        "vendor/trellis/trellis-server.exe",
        "vendor/gltfpack/gltfpack.exe",
        "vendor/warlockc/warlockc.dll",
    }
    assert payload["python"] == {
        "implementation": "CPython",
        "distribution": "python-build-standalone via uv",
        "major_minor": "3.13",
        "architecture": "x86_64",
        "torch_cuda": "12.8",
    }
    assert payload["roots"] == [
        "vendor/trellis",
        "vendor/gltfpack",
        "vendor/warlockc",
    ]
    assert all(entry["size"] > 0 for entry in payload["files"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) for entry in payload["files"])


def test_the_provisioned_native_runtime_matches_every_size_and_hash_pin() -> None:
    verifier = _verifier()
    payload = verifier.load_manifest(MANIFEST)
    targets = [ROOT / entry["path"] for entry in payload["files"]]
    if not any(path.exists() for path in targets):
        pytest.skip("the ignored installer runtime is not provisioned on this checkout")
    assert all(path.is_file() for path in targets), "a provisioned runtime may not be partial"
    actual = {
        path.relative_to(ROOT).as_posix()
        for directory in ("trellis", "gltfpack", "warlockc")
        for path in (ROOT / "vendor" / directory).rglob("*")
        if path.is_file()
    }
    assert actual == {entry["path"] for entry in payload["files"]}
    verified = verifier.verify_runtime(ROOT, MANIFEST)
    assert len(verified) == 11


def test_runtime_verification_refuses_a_tampered_file(tmp_path: Path) -> None:
    verifier = _verifier()
    runtime = tmp_path / "vendor" / "tool.exe"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"right")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "python": {
                    "implementation": "CPython",
                    "distribution": "python-build-standalone via uv",
                    "major_minor": "3.13",
                    "architecture": "x86_64",
                    "torch_cuda": "12.8",
                },
                "roots": ["vendor"],
                "files": [
                    {
                        "path": "vendor/tool.exe",
                        "size": 5,
                        "sha256": verifier.digest(runtime),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    verifier.verify_runtime(tmp_path, manifest)
    runtime.write_bytes(b"wrong")
    with pytest.raises(verifier.ManifestError, match="SHA-256 differs"):
        verifier.verify_runtime(tmp_path, manifest)
    runtime.write_bytes(b"right")
    (runtime.parent / "extra.dll").write_bytes(b"extra")
    with pytest.raises(verifier.ManifestError, match="unpinned files"):
        verifier.verify_runtime(tmp_path, manifest)


def test_runtime_verification_refuses_a_path_outside_the_stage(tmp_path: Path) -> None:
    verifier = _verifier()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "python": {
                    "implementation": "CPython",
                    "distribution": "python-build-standalone via uv",
                    "major_minor": "3.13",
                    "architecture": "x86_64",
                    "torch_cuda": "12.8",
                },
                "roots": ["vendor"],
                "files": [
                    {"path": "../escape.dll", "size": 1, "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(verifier.ManifestError, match="safe relative"):
        verifier.verify_runtime(tmp_path, manifest)


def test_build_script_stages_and_verifies_the_checkout_without_downloading_models() -> None:
    source = (INSTALLER / "build.ps1").read_text(encoding="utf-8")
    for required in (
        "uv version --short",
        "uv python find --managed-python --no-python-downloads --no-project 3.13",
        "uv export --frozen --no-dev --no-emit-project",
        "uv pip sync --python $StagedPython",
        "torch.version.cuda == '12.8'",
        'Set-Content -LiteralPath (Join-Path $SitePackages "warlock_app.pth")',
        'runtime_manifest_sha256 = $ManifestHash',
        'uv_lock_sha256 = $LockHash',
        "-m compileall",
        "-m warlock doctor",
        '"/DAppVersion=$Version"',
        '"/DStageDir=$Stage"',
    ):
        assert required in source
    assert source.count("$Verifier --root") == 2

    assert not any(
        forbidden.lower() in source.lower()
        for forbidden in ("snapshot_download", "hf download", "Invoke-WebRequest", "curl.exe")
    )


def test_the_build_collects_the_packs_and_then_ships_the_base_runtime() -> None:
    """The whole point of the packs: the installer stages ``--extra studio``
    alone, and torch, bpy and the music stack arrive from Settings -> Packs.

    The *order* is what this pins. The full resolution has to be installed when
    ``make_packs`` runs -- unpacked sizes exist nowhere but an installed tree,
    and the CUDA 12.8 assertion above it is what proves the collected wheels
    are the cu128 ones -- and the prune has to happen after it. A build that
    lost either half would still produce an installer: one carrying 6.61 GB of
    extras *and* a pack directory, or one whose packs advertise no install
    size at all.
    """
    source = (INSTALLER / "build.ps1").read_text(encoding="utf-8")
    collect = source.index("scripts/make_packs.py")
    prune = source.index("--extra studio -o $BaseRequirements")
    cuda = source.index("staged PyTorch CUDA 12.8 check")
    assert cuda < collect < prune, "collect against the full runtime, then prune it"
    # And the prune is asserted rather than assumed, because a sync that
    # quietly kept the previous resolution is invisible in every other output.
    assert "find_spec('torch')" in source
    assert "packs.json" in source and "bundled" in source


def test_the_installer_carries_the_wheels_that_cannot_be_downloaded() -> None:
    """``docopt``, ``mojimoji`` and ``unidic-lite`` publish no Windows wheel,
    so there is no URL a user's machine could fetch them from: the build
    compiles them and the installer has to carry them. An upgrade clears the
    previous version's copies, whose filenames no manifest will name again."""
    source = (INSTALLER / "build.ps1").read_text(encoding="utf-8")
    assert 'Join-Path $Stage "packs"' in source
    iss = (INSTALLER / "warlock.iss").read_text(encoding="utf-8")
    assert r'Type: filesandordirs; Name: "{app}\packs"' in iss


def test_inno_setup_is_per_user_relocatable_and_leaves_user_data_alone() -> None:
    source = (INSTALLER / "warlock.iss").read_text(encoding="utf-8")
    assert re.search(r"AppId=\{\{[0-9A-F-]{36}\}", source)
    assert "PrivilegesRequired=lowest" in source
    assert r"DefaultDirName={localappdata}\Programs\Warlock Studio" in source
    assert "Compression=lzma2" in source and "SolidCompression=yes" in source
    # Inverted on 2026-08-26, by the first compile this file ever described.
    # The spanning was a prediction -- a ~4 GB payload was assumed not to fit
    # one executable, so the slices were set at 2.1 GB. Measured, the payload
    # compresses to a single 2.91 GB exe in 855 s, and a one-file download is
    # strictly better for the people this installer exists for: the three-file
    # variant fails partway through if any .bin is renamed or left behind.
    # DiskSliceSize is deliberately left in warlock.iss and deliberately not
    # asserted here -- it is inert while spanning is off, and keeping it makes
    # the decision one line to reverse if the payload ever outgrows one file.
    assert "DiskSpanning=no" in source
    assert r'Filename: "{app}\python\pythonw.exe"; Parameters: "-m warlock"' in source
    assert "Flags: unchecked" in source
    assert r'Name: "{app}\python"' in source
    assert ".warlock" in source and "downloaded models remain" in source
    # Inverted on 2026-08-24. This used to assert ``LicenseFile`` was *absent*,
    # which was an accurate record of the fact that no licence had been chosen
    # -- not a decision that the wizard should show none. The project is
    # GPL-3.0-or-later now, and for everyone who installs rather than clones
    # this page is the only place the terms appear at all.
    assert r"LicenseFile={#ProjectRoot}\LICENSE" in source

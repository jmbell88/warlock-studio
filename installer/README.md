# Windows installer build

`pwsh installer\build.ps1` produces the checkout-shaped, per-user Warlock
Studio installer. The build requires Windows, `uv`, an already installed
uv-managed CPython 3.13, and Inno Setup 6 (`iscc.exe` on `PATH`, or supplied as
`-Iscc`). It does not download Python, native tools, or model weights.

The staged application contains the locked Python runtime and dependencies,
`src\warlock`, the manual, and the three native runtime directories under
`vendor`. Model weights remain first-run downloads under the user's Warlock
home. The project is GPL-3.0-or-later: `LICENSE` is shown as the wizard's
first page (`LicenseFile=`) and staged into the install root alongside
`THIRD-PARTY-NOTICES.md`, which also goes beside `vendor\` because MIT and
the NVIDIA redistributable EULA require the notice to travel with the binary.

## Native runtime pins

`runtime-manifest.json` is the build boundary for vendored executables and
DLLs. Every file is pinned by relative path, byte size, and SHA-256. The build
runs `verify_runtime.py` against both the checkout and `build\stage`; a missing,
extra, replaced, or truncated native runtime therefore fails before ISCC.

When a native runtime is deliberately upgraded, update the binary and its
manifest entry together, review the upstream version/source separately, and run:

```powershell
uv run pytest tests/test_installer.py -n 0
```

The produced `install.json` records the application/Python versions and the
SHA-256 of the complete runtime manifest. `dist\` and `build\` are build
outputs, not source inputs.

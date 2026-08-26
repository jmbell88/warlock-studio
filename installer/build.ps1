param(
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$BuildRoot = Join-Path $Root "build"
$Stage = Join-Path $BuildRoot "stage"
$Requirements = Join-Path $BuildRoot "requirements.txt"
$Manifest = Join-Path $PSScriptRoot "runtime-manifest.json"
$Verifier = Join-Path $PSScriptRoot "verify_runtime.py"
$Iss = Join-Path $PSScriptRoot "warlock.iss"

function Assert-LastExit([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

function Assert-UnderRoot([string]$Path, [string]$Parent) {
    $ResolvedPath = [IO.Path]::GetFullPath($Path)
    $ResolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    if (-not $ResolvedPath.StartsWith($ResolvedParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing a build operation outside $ResolvedParent`: $ResolvedPath"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to build Warlock Studio"
}
if (-not $Iscc) {
    $Compiler = Get-Command iscc -ErrorAction SilentlyContinue
    if ($null -eq $Compiler) {
        throw "Inno Setup 6 is required; put iscc.exe on PATH or pass -Iscc"
    }
    $Iscc = $Compiler.Source
}
if (-not (Test-Path -LiteralPath $Iscc -PathType Leaf)) {
    throw "iscc.exe was not found at $Iscc"
}

$VersionOutput = (& uv version --short | Out-String).Trim()
Assert-LastExit "uv version"
$Version = ($VersionOutput -split '\s+')[-1]
$InitText = Get-Content -LiteralPath (Join-Path $Root "src\warlock\__init__.py") -Raw
$RuntimeMatch = [regex]::Match($InitText, '(?m)^__version__ = "([^"]+)"')
if (-not $RuntimeMatch.Success -or $RuntimeMatch.Groups[1].Value -ne $Version) {
    throw "version mismatch: uv says $Version and src\warlock\__init__.py says $($RuntimeMatch.Groups[1].Value)"
}

$ManagedPython = (& uv python find --managed-python --no-python-downloads --no-project 3.13 | Out-String).Trim()
Assert-LastExit "uv python find 3.13"
if (-not (Test-Path -LiteralPath $ManagedPython -PathType Leaf)) {
    throw "uv did not return a usable managed CPython 3.13"
}
# `uv python find` returns the project virtualenv when it is run from inside the
# checkout -- `--no-project` stops uv reading pyproject.toml, it does not stop
# .venv discovery. A venv's parent directory is `Scripts`, which holds console
# shims and no stdlib, and staging that produced a `stage\python` that satisfied
# every guard below while being unusable. Resolve the real installation through
# sys.base_prefix, which is a no-op when uv already returned an installation.
$ManagedRoot = (& $ManagedPython -c "import sys; print(sys.base_prefix)" | Out-String).Trim()
Assert-LastExit "managed Python base prefix"
$ManagedPython = Join-Path $ManagedRoot "python.exe"
if (-not (Test-Path -LiteralPath $ManagedPython -PathType Leaf)) {
    throw "no python.exe under the resolved interpreter root $ManagedRoot"
}
# The stdlib is what separates an installation from a `Scripts` directory, and
# its absence is the assertion whose absence let the venv through.
if (-not (Test-Path -LiteralPath (Join-Path $ManagedRoot "Lib\os.py") -PathType Leaf)) {
    throw "the resolved interpreter root has no stdlib (Lib\os.py): $ManagedRoot"
}
& $ManagedPython -c "import platform, sys; assert sys.version_info[:2] == (3, 13); assert platform.machine().lower() in ('amd64', 'x86_64')"
Assert-LastExit "managed Python 3.13 check"

# Refuse an accidental or locally replaced native payload before copying a byte.
& $ManagedPython $Verifier --root $Root --manifest $Manifest
Assert-LastExit "checkout runtime verification"

New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
Assert-UnderRoot $Stage $BuildRoot
if (Test-Path -LiteralPath $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

New-Item -ItemType Directory -Path (Join-Path $Stage "python") -Force | Out-Null
Copy-Item -Path (Join-Path $ManagedRoot "*") -Destination (Join-Path $Stage "python") -Recurse -Force
$StagedPython = Join-Path $Stage "python\python.exe"
if (-not (Test-Path -LiteralPath $StagedPython -PathType Leaf)) {
    throw "the managed Python copy did not produce stage\python\python.exe"
}
if (-not (Test-Path -LiteralPath (Join-Path $Stage "python\Lib\os.py") -PathType Leaf)) {
    throw "the managed Python copy did not produce a stdlib at stage\python\Lib\os.py"
}
# A uv-managed CPython carries Lib\EXTERNALLY-MANAGED, and `uv pip sync` refuses
# to install into an interpreter that declares it. The marker is a true statement
# about uv's own copy under AppData and a false one about this staged tree, which
# is Warlock's private application runtime and is never managed by uv again --
# leaving it would also refuse anyone installing into the shipped runtime later.
$ExternallyManaged = Join-Path $Stage "python\Lib\EXTERNALLY-MANAGED"
if (Test-Path -LiteralPath $ExternallyManaged -PathType Leaf) {
    Remove-Item -LiteralPath $ExternallyManaged -Force
}

& uv export --frozen --no-dev --no-emit-project --extra studio --extra text2image --extra rig -o $Requirements
Assert-LastExit "uv export"

& uv pip sync --python $StagedPython $Requirements
$NeedsCudaRetry = $LASTEXITCODE -ne 0
if (-not $NeedsCudaRetry) {
    & $StagedPython -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda"
    $NeedsCudaRetry = $LASTEXITCODE -ne 0
}
if ($NeedsCudaRetry) {
    Write-Host "Retrying dependency sync against the pinned PyTorch CUDA 12.8 index"
    & uv pip sync --python $StagedPython $Requirements `
        --index https://download.pytorch.org/whl/cu128 `
        --index-strategy unsafe-best-match
    Assert-LastExit "uv pip sync with CUDA 12.8 index"
}
& $StagedPython -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda"
Assert-LastExit "staged PyTorch CUDA 12.8 check"

New-Item -ItemType Directory -Path (Join-Path $Stage "src") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "src\warlock") -Destination (Join-Path $Stage "src") -Recurse -Force
New-Item -ItemType Directory -Path (Join-Path $Stage "docs") -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "docs\manual") -Destination (Join-Path $Stage "docs") -Recurse -Force
New-Item -ItemType Directory -Path (Join-Path $Stage "vendor") -Force | Out-Null
foreach ($RuntimeDir in @("trellis", "gltfpack", "warlockc")) {
    Copy-Item -LiteralPath (Join-Path $Root "vendor\$RuntimeDir") -Destination (Join-Path $Stage "vendor") -Recurse -Force
}
# LICENSE and THIRD-PARTY-NOTICES.md are not optional paperwork here. This
# installer packs GPL-3.0 `bpy` and eleven vendored binaries -- MIT trellis.cpp
# and ggml, MIT gltfpack, and three NVIDIA CUDA redistributables -- into one
# executable. MIT requires its notice to travel *with the binary*, the NVIDIA
# redistributable EULA carries its own terms, and the GPL requires the licence
# to reach whoever receives the program. Until 2026-08-24 the binaries were
# copied bare and no licence text was staged at all.
foreach ($Document in @("pyproject.toml", "CHANGELOG.md", "README.md", "LICENSE", "THIRD-PARTY-NOTICES.md")) {
    Copy-Item -LiteralPath (Join-Path $Root $Document) -Destination $Stage -Force
}
# And a copy beside the DLLs themselves, so a user who opens vendor\ finds the
# terms without knowing to look at the install root.
Copy-Item -LiteralPath (Join-Path $Root "THIRD-PARTY-NOTICES.md") -Destination (Join-Path $Stage "vendor") -Force

# warlock.iss points UninstallDisplayIcon and both shortcuts at {app}\warlock.ico.
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "warlock.ico") -Destination $Stage -Force

# Build caches are never an input. compileall below creates only the caches that
# match the interpreter being shipped.
Get-ChildItem -LiteralPath (Join-Path $Stage "src\warlock") -Directory -Filter "__pycache__" -Recurse |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }

$SitePackages = Join-Path $Stage "python\Lib\site-packages"
New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null
Set-Content -LiteralPath (Join-Path $SitePackages "warlock_app.pth") -Value "../../../src" -Encoding ascii

Copy-Item -LiteralPath $Manifest -Destination (Join-Path $Stage "runtime-manifest.json") -Force
$ManifestHash = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
$LockHash = (Get-FileHash -LiteralPath (Join-Path $Root "uv.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
$PythonVersion = (& $StagedPython -c "import platform; print(platform.python_version())" | Out-String).Trim()
Assert-LastExit "staged Python version"
$InstallRecord = [ordered]@{
    schema = 1
    product = "Warlock Studio"
    version = $Version
    python = $PythonVersion
    runtime_manifest_sha256 = $ManifestHash
    uv_lock_sha256 = $LockHash
    built_utc = [DateTime]::UtcNow.ToString("o")
}
$InstallRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Stage "install.json") -Encoding utf8

New-Item -ItemType Directory -Path (Join-Path $Stage "bin") -Force | Out-Null
$Doctor = @'
@echo off
setlocal
"%~dp0..\python\python.exe" -m warlock doctor %*
'@
Set-Content -LiteralPath (Join-Path $Stage "bin\warlock-doctor.cmd") -Value $Doctor -Encoding ascii

& $StagedPython -m compileall -q (Join-Path $Stage "src\warlock")
Assert-LastExit "compileall"
& $StagedPython $Verifier --root $Stage --manifest (Join-Path $Stage "runtime-manifest.json")
Assert-LastExit "staged runtime verification"

$SmokeHome = Join-Path $BuildRoot "smoke-home"
Assert-UnderRoot $SmokeHome $BuildRoot
if (Test-Path -LiteralPath $SmokeHome) {
    Remove-Item -LiteralPath $SmokeHome -Recurse -Force
}
New-Item -ItemType Directory -Path $SmokeHome -Force | Out-Null
$PreviousWarlockHome = $env:WARLOCK_HOME
try {
    $env:WARLOCK_HOME = $SmokeHome
    $DoctorOutput = (& $StagedPython -m warlock doctor 2>&1 | Out-String)
    $DoctorExit = $LASTEXITCODE
    if ($DoctorExit -ne 1) {
        throw "staged doctor should report missing required models (exit 1), got $DoctorExit"
    }
    foreach ($Expected in @("trellis-server.exe", "TRELLIS GGUF weights", "SDXL 1.0")) {
        if (-not $DoctorOutput.Contains($Expected)) {
            throw "staged doctor output did not mention $Expected"
        }
    }
    & $StagedPython -c "import warlock.studio.main"
    Assert-LastExit "staged Studio import"
}
finally {
    if ($null -eq $PreviousWarlockHome) {
        Remove-Item Env:WARLOCK_HOME -ErrorAction SilentlyContinue
    }
    else {
        $env:WARLOCK_HOME = $PreviousWarlockHome
    }
}

New-Item -ItemType Directory -Path (Join-Path $Root "dist") -Force | Out-Null
& $Iscc "/DAppVersion=$Version" "/DStageDir=$Stage" $Iss
Assert-LastExit "Inno Setup"

Write-Host "Built dist\WarlockSetup-v$Version.exe"

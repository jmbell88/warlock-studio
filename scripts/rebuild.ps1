<#
.SYNOPSIS
    Run what GitHub Actions runs, then build the installer.

.DESCRIPTION
    One command before a push. Steps 1-5 are `.github/workflows/windows-ci.yml`
    line for line, so a red Action is discovered on the workstation instead of
    ten minutes into a runner; the rest is the half CI never runs --
    `installer\build.ps1` and the prune that stops `dist\` drifting releases
    behind the tree.

    Nothing here reimplements either half. Every step shells out to the script
    or command that already owned it.

.PARAMETER SkipTests
    Skip the suite. The fast lane for "does this still package?".

.PARAMETER CiWorkers
    Run the suite with -n 4 as the runner does, rather than the workstation's
    -n 8, when reproducing a runner-shaped xdist death.

.PARAMETER Native
    Rebuild vendor\warlockc\warlockc.dll before the installer. Opt-in, and it
    invalidates the manifest pin -- see the comment on the step.

.PARAMETER SkipInstaller
    Stop after the CI mirror. No ISCC required, and dist\ is not pruned.

.PARAMETER NoPrune
    Leave older setup executables and wheels in dist\.

.PARAMETER Iscc
    Path to iscc.exe, when Inno Setup 6 is not on PATH.
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$CiWorkers,
    [switch]$Native,
    [switch]$SkipInstaller,
    [switch]$NoPrune,
    [string]$Iscc = ""
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Dist = Join-Path $Root "dist"
$Timings = [System.Collections.Generic.List[object]]::new()
$StepNumber = 0

function Assert-LastExit([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

# The same guard `installer\build.ps1` puts around its stage wipe, for the same
# reason: the prune below is the one part of this script that can destroy
# something a human wanted.
function Assert-UnderRoot([string]$Path, [string]$Parent) {
    $ResolvedPath = [IO.Path]::GetFullPath($Path)
    $ResolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    if (-not $ResolvedPath.StartsWith($ResolvedParent, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to delete outside $ResolvedParent`: $ResolvedPath"
    }
}

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Name,
        [string]$CiStep = "",
        [Parameter(Mandatory)][scriptblock]$Body
    )
    $script:StepNumber++
    $Number = $script:StepNumber
    Write-Host ""
    Write-Host "== [$Number] $Name" -ForegroundColor Cyan
    $Watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        & $Body
    }
    catch {
        $Watch.Stop()
        Write-Host ""
        Write-Host "FAILED at step $Number`: $Name" -ForegroundColor Red
        if ($CiStep) {
            Write-Host "This is the '$CiStep' step in .github/workflows/windows-ci.yml -- the push would have failed here." -ForegroundColor Red
        }
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host "build\ is left as it stands, for inspection."
        exit 1
    }
    $Watch.Stop()
    $script:Timings.Add([pscustomobject]@{
        Step    = "$Number. $Name"
        Seconds = [math]::Round($Watch.Elapsed.TotalSeconds, 1)
    })
}

# --- step 0: every precondition, before any work ------------------------------
#
# `installer\build.ps1` checks for ISCC itself -- at minute zero of *its* run,
# which is minute twenty of this one. A missing Inno Setup should cost a second,
# not a suite and a 6 GB dependency sync.

try {

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to build Warlock Studio"
}

$VersionOutput = (& uv version --short | Out-String).Trim()
Assert-LastExit "uv version"
$Version = ($VersionOutput -split '\s+')[-1]

$ManagedPython = (& uv python find --managed-python --no-python-downloads --no-project 3.13 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "no uv-managed CPython 3.13 (uv python install 3.13)"
}
# Run from inside the checkout, `uv python find` returns the project
# virtualenv: --no-project stops uv reading pyproject.toml, it does not stop
# .venv discovery. installer\build.ps1 documents the same trap and resolves it
# the same way, and this is the interpreter that will be *staged* -- so the
# check that it exists at all belongs here, at second zero, not twenty minutes
# in. The stdlib is what separates an installation from a Scripts directory.
$ManagedRoot = (& $ManagedPython -c "import sys; print(sys.base_prefix)" | Out-String).Trim()
Assert-LastExit "managed Python base prefix"
if (-not (Test-Path -LiteralPath (Join-Path $ManagedRoot "Lib\os.py") -PathType Leaf)) {
    throw "the resolved interpreter root has no stdlib (Lib\os.py): $ManagedRoot"
}
$ManagedPython = Join-Path $ManagedRoot "python.exe"

if (-not $SkipInstaller) {
    if (-not $Iscc) {
        $Compiler = Get-Command iscc -ErrorAction SilentlyContinue
        if ($null -eq $Compiler) {
            # PATH is not where Inno Setup puts itself. Its installer offers a
            # non-admin install by default, which lands under
            # %LOCALAPPDATA%\Programs and adds nothing to PATH -- so "Inno Setup
            # is required" was told to a machine that had had 6.7.3 installed
            # all along, and the answer was to go and find it by hand. Look in
            # the three places it actually installs to before saying that.
            $Candidates = @(
                (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
                (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
                (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
            )
            $Iscc = $Candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
            if (-not $Iscc) {
                throw "Inno Setup 6 is required to build the installer; install it, pass -Iscc, or run with -SkipInstaller"
            }
        }
        else {
            $Iscc = $Compiler.Source
        }
    }
    if (-not (Test-Path -LiteralPath $Iscc -PathType Leaf)) {
        throw "iscc.exe was not found at $Iscc"
    }
    # The fresh-checkout case the native step used to cover by building
    # unconditionally. /vendor/ is gitignored and all three runtime directories
    # are pinned, so a checkout that has never been provisioned fails inside
    # installer\build.ps1's first verify_runtime -- twenty minutes of nothing.
    # Said here instead, in a second, and with what to do about it.
    if (-not $Native) {
        $Dll = Join-Path $Root "vendor\warlockc\warlockc.dll"
        if (-not (Test-Path -LiteralPath $Dll -PathType Leaf)) {
            throw "no vendor\warlockc\warlockc.dll, which runtime-manifest.json pins: build it with ``pwsh native\build.ps1``, then update its manifest entry and run ``uv run pytest tests/test_installer.py -n 0``"
        }
    }
}

Write-Host "Warlock Studio $Version" -ForegroundColor Green
Write-Host "  python    $ManagedPython"
if (-not $SkipInstaller) { Write-Host "  iscc      $Iscc" }

}
catch {
    # A missing tool is a one-line answer, not a PowerShell stack trace with the
    # `throw` statement quoted back at the reader.
    Write-Host ""
    Write-Host "Cannot start: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# --- steps 1-5: the CI mirror -------------------------------------------------

Invoke-Step "Sync development environment" "Sync development environment" {
    # --frozen is the point of copying CI's line rather than writing `uv sync`:
    # a pyproject.toml edited without relocking is a CI-only failure, and
    # --frozen is what makes it a local one. The four extras are not optional
    # either -- a bare `uv sync` prunes them and breaks collection.
    & uv sync --frozen --extra studio --extra text2image --extra rig --extra music
    Assert-LastExit "uv sync"
}

Invoke-Step "Version lockstep and ruff" "Version lockstep" {
    & uv run python scripts/preflight.py --fast
    Assert-LastExit "preflight"
}

if ($SkipTests) {
    Write-Host ""
    Write-Host "== [skipped] Default test suite (-SkipTests)" -ForegroundColor Yellow
}
else {
    Invoke-Step "Default test suite" "Default test suite" {
        # The repo's own addopts (-n 8) unless asked otherwise. CI's -n 4 is a
        # 16 GB runner constraint, documented in the workflow, not a
        # correctness choice -- so it is available and not the default.
        $PytestArgs = @('-q', '--timeout', '300')
        if ($CiWorkers) { $PytestArgs += @('-n', '4') }
        & uv run pytest @PytestArgs
        Assert-LastExit "pytest"
    }
}

Invoke-Step "Build wheel and sdist" "Build wheel and sdist" {
    & uv build
    Assert-LastExit "uv build"
}

Invoke-Step "Wheel install smoke test" "Wheel install smoke test" {
    # The clean venv lands under build\ rather than CI's .smoke: /build/ is
    # gitignored and .smoke is not, so the runner's name would leave a venv
    # sitting in `git status` after every local run.
    & uv run python scripts/wheel_smoke.py
    Assert-LastExit "wheel smoke test"
}

# --- steps 6-8: the half CI does not run --------------------------------------

# **Rebuilding the DLL is opt-in, and this driver had it the other way round
# for exactly one run.** The reasoning for building it by default was that
# runtime-manifest.json pins vendor\warlockc\warlockc.dll and /vendor/ is
# gitignored, so a fresh checkout has no DLL for verify_runtime to find. True,
# and it misses the corollary: MSVC embeds a build timestamp, so recompiling
# *identical* sources yields an identical 123392 bytes with a different
# SHA-256. A default rebuild therefore breaks the pin on every machine that
# already had a good DLL -- which is what it did here, and the installer
# refused with "runtime file SHA-256 differs: vendor/warlockc/warlockc.dll".
#
# The pin is the point: a native binary and its manifest entry are upgraded
# together by a human, and tests\test_installer.py asserts they agree. So the
# default is to leave the DLL alone, and -Native says "I am about to update
# the manifest too".
if ($Native) {
    Invoke-Step "Native kernels" "" {
        & pwsh -NoProfile -File (Join-Path $Root "native\build.ps1")
        Assert-LastExit "native\build.ps1"
        Write-Host ""
        Write-Host "warlockc.dll was rebuilt, so its runtime-manifest.json pin is now stale." -ForegroundColor Yellow
        Write-Host "Update the entry and run: uv run pytest tests/test_installer.py -n 0" -ForegroundColor Yellow
    }
}

if ($SkipInstaller) {
    Write-Host ""
    Write-Host "== [skipped] Installer (-SkipInstaller)" -ForegroundColor Yellow
}
else {
    Invoke-Step "Installer" "" {
        & pwsh -NoProfile -File (Join-Path $Root "installer\build.ps1") -Iscc $Iscc
        Assert-LastExit "installer\build.ps1"
    }

    if ($NoPrune) {
        Write-Host ""
        Write-Host "== [skipped] Prune dist (-NoPrune)" -ForegroundColor Yellow
    }
    else {
        Invoke-Step "Prune dist" "" {
            # After the build, never before: pruning first would leave nothing
            # at all behind a build that then failed. The version is the one
            # preflight has already proven agrees across all four files.
            $Keep = @(
                "WarlockSetup-v$Version.exe",
                "warlock-$Version-py3-none-any.whl",
                "warlock-$Version.tar.gz"
            )
            $Stale = @(Get-ChildItem -LiteralPath $Dist -File | Where-Object {
                ($_.Name -like 'WarlockSetup-v*.exe' -or
                 $_.Name -like 'warlock-*.whl' -or
                 $_.Name -like 'warlock-*.tar.gz') -and $Keep -notcontains $_.Name
            })
            foreach ($File in $Stale) {
                Assert-UnderRoot $File.FullName $Dist
                Write-Host "  removing $($File.Name)"
                Remove-Item -LiteralPath $File.FullName -Force
            }
            if (-not $Stale) { Write-Host "  nothing stale in dist\" }
        }
    }
}

# --- step 9: what was built ---------------------------------------------------

Write-Host ""
$Timings | Format-Table -AutoSize | Out-String | Write-Host

$Artifacts = @(
    (Join-Path $Dist "WarlockSetup-v$Version.exe"),
    (Join-Path $Dist "warlock-$Version-py3-none-any.whl"),
    (Join-Path $Dist "warlock-$Version.tar.gz")
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }

foreach ($Artifact in $Artifacts) {
    $Item = Get-Item -LiteralPath $Artifact
    $Size = "{0:N1} MB" -f ($Item.Length / 1MB)
    Write-Host ("{0,-40} {1,12}" -f $Item.Name, $Size)
    # INSTALL.md carries the setup executable's size and SHA-256 as
    # "[TBD -- measure from release-candidate build]". Filling them in is a
    # human edit; this is where the two numbers come from.
    if ($Item.Extension -eq '.exe') {
        $Hash = (Get-FileHash -LiteralPath $Item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host ("{0,-40} {1}" -f "  sha256", $Hash)
    }
}

Write-Host ""
Write-Host "Green. Nothing in windows-ci.yml is left to surprise you." -ForegroundColor Green

<#
.SYNOPSIS
    Build vendor/warlockc/warlockc.dll from the C sources in this directory.

.DESCRIPTION
    Vendored the way trellis-server.exe and gltfpack.exe are: a native binary
    under vendor/, which is gitignored, so every checkout builds its own and a
    missing one is never fatal -- warlock.native falls back to the numpy
    implementations and `warlock doctor` says so.

    The floating-point flags are part of the correctness contract rather than a
    tuning choice. Every kernel here must agree bit for bit with a numpy
    reference, and a fused multiply-add rounds once where numpy rounds twice.
    /fp:precise (MSVC) and -ffp-contract=off (clang) are what forbid that; do
    not "optimise" them to /fp:fast or -ffast-math. clang-cl needs both spelled
    out -- its /fp:precise still allows contraction.

.PARAMETER Clean
    Delete intermediate objects after a successful build.
#>
[CmdletBinding()]
param([switch]$Clean)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$outDir = Join-Path (Split-Path -Parent $here) 'vendor\warlockc'
$outDll = Join-Path $outDir 'warlockc.dll'
$sources = @(Get-ChildItem -Path $here -Filter '*.c' | ForEach-Object { $_.FullName })

if (-not $sources) { throw "no C sources found in $here" }
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

function Find-VcVars {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $install = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null
    if (-not $install) { return $null }
    $vcvars = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
    return (Test-Path $vcvars) ? $vcvars : $null
}

$objDir = Join-Path $here 'obj'
if (-not (Test-Path $objDir)) { New-Item -ItemType Directory -Path $objDir -Force | Out-Null }

$built = $false
$vcvars = Find-VcVars

if ($vcvars) {
    Write-Host "MSVC: $vcvars"
    # cl only works inside the developer environment, and vcvars64.bat sets that
    # up for a cmd session -- so the compile goes through cmd. Via a generated
    # batch file rather than `cmd /c "<one long string>"`: the arguments carry
    # paths with spaces *and* trailing backslashes (/Fo wants one), and cmd's
    # quote handling mangles that combination. A file has no such layer.
    $bat = Join-Path $objDir '_build.bat'
    $srcArgs = ($sources | ForEach-Object { "`"$_`"" }) -join ' '
    @(
        '@echo off'
        "call `"$vcvars`" >nul || exit /b 1"
        "cl /nologo /O2 /fp:precise /W4 /WX /LD /std:c11 /I`"$here`" " +
        "/Fo`"$objDir\\`" /Fe`"$outDll`" $srcArgs /link /IMPLIB:`"$objDir\warlockc.lib`""
        'exit /b %ERRORLEVEL%'
    ) | Set-Content -Path $bat -Encoding ascii

    & cmd.exe /c "`"$bat`""
    if ($LASTEXITCODE -eq 0) { $built = $true } else { Write-Warning "cl failed ($LASTEXITCODE)" }
}

if (-not $built) {
    # clang-cl / clang / zig cc, in that order. Any of them produces a DLL the
    # ctypes loader can use; none of them is required.
    foreach ($alt in @('clang-cl', 'clang', 'zig')) {
        $exe = (Get-Command $alt -ErrorAction SilentlyContinue)
        if (-not $exe) { continue }
        Write-Host "fallback compiler: $($exe.Source)"
        if ($alt -eq 'clang-cl') {
            # /clang:-ffp-contract=off is not belt-and-braces beside /fp:precise:
            # on clang 14+ /fp:precise means -ffp-contract=on, which *permits*
            # the FMAs the bit-parity contract forbids -- so this branch alone
            # could produce a DLL that fails the np.array_equal parity tests
            # while the ABI handshake says it is current. /std:c11 and /W4 /WX
            # match the MSVC line for the same reason: one compiler quietly
            # building to a different standard, with warnings, is how a kernel
            # stops being the same kernel.
            & clang-cl /O2 /fp:precise /clang:-ffp-contract=off /std:c11 /W4 /WX `
                /LD /I"$here" /Fe:"$outDll" @sources
        } elseif ($alt -eq 'clang') {
            & clang -O2 -ffp-contract=off -std=c11 -shared -I"$here" -o "$outDll" @sources
        } else {
            & zig cc -O2 -ffp-contract=off -std=c11 -shared -I"$here" -o "$outDll" @sources
        }
        if ($LASTEXITCODE -eq 0) { $built = $true; break }
        Write-Warning "$alt failed ($LASTEXITCODE)"
    }
}

if (-not $built) {
    throw @"
no usable C compiler found.

Install one of:
  * Visual Studio Build Tools with the "Desktop development with C++" workload
    (winget install Microsoft.VisualStudio.2022.BuildTools)
  * LLVM/clang (winget install LLVM.LLVM)
  * zig (winget install zig.zig)

The app runs without this DLL -- meshaudit falls back to numpy and
`warlock doctor` reports the kernels as unavailable.
"@
}

if ($Clean) { Remove-Item -Recurse -Force $objDir -ErrorAction SilentlyContinue }

Write-Host "built $outDll" -ForegroundColor Green

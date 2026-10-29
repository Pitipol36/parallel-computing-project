# Configure and build (Release) with Ninja inside a Visual Studio developer environment.
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 [-Clean]
param([switch]$Clean)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "build"
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with C++ tools not found" }

# Import the x64 developer environment (cl, link, cmake, ninja) into this session.
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

cmake -S $root -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build $build
exit $LASTEXITCODE

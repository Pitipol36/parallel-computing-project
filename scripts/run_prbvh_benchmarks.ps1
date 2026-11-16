# PRBVH (parallel reinsertion) experiments. Writes, in results\:
#   prbvh.csv          build time + tree quality of each GPU binary-tree builder with and without
#                      --optimize prbvh (tag base / prbvh), and a rounds sweep (tag prbvh_iters)
#   prbvh_render.csv   render time on the same trees (dragon and mixed), base and prbvh runs
#                      back to back, --quality-bench method (warm-up, interleaved, median + IQR)
#   powershell -ExecutionPolicy Bypass -File scripts\run_prbvh_benchmarks.ps1 [-Quick]
# The full-SAH reference cost comes from results\build.csv (scripts\run_build_benchmarks.ps1).
param([switch]$Quick, [switch]$BuildOnly)  # -BuildOnly: redo prbvh.csv, keep prbvh_render.csv
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "build\raytracer.exe"
if (-not (Test-Path $exe)) { throw "build first: scripts\build.ps1" }
New-Item -ItemType Directory -Force results | Out-Null
$csv = "results\prbvh.csv"; $renderCsv = "results\prbvh_render.csv"
foreach ($f in $csv, $renderCsv) {
    if ((Test-Path $f) -and -not ($BuildOnly -and $f -eq $renderCsv)) { Remove-Item $f }
}

$runs = if ($Quick) { 1 } else { 5 }
$qruns = if ($Quick) { 3 } else { 11 }
$builders = "gpu_lbvh,gpu_ploc,gpu_hploc,gpu_trbvh"
$models = @(
    @{ label = "bunny";  file = "data\bun_zipper.ply"; up = "y"; cam = "orbit" },
    @{ label = "dragon"; file = "data\dragon_vrip.ply"; up = "y"; cam = "orbit" },
    @{ label = "buddha"; file = "data\happy_vrip.ply";  up = "y"; cam = "orbit" },
    @{ label = "mixed";  file = "mixed";                up = "y"; cam = "inside" }
)
$present = $models | Where-Object { $_.file -eq "mixed" -or (Test-Path $_.file) }
$dragon = $models | Where-Object { $_.label -eq "dragon" }
$mixed = $models | Where-Object { $_.label -eq "mixed" }

function Run([hashtable]$m, [string[]]$extra) {
    $a = @("--model", $m.file, "--label", $m.label, "--up", $m.up, "--cam", $m.cam) + $extra
    Write-Host "`n>>> $($m.label) $($extra -join ' ')" -ForegroundColor Cyan
    & $exe @a
    if ($LASTEXITCODE -ne 0) { Write-Warning "run failed: $($m.label) $($extra -join ' ')" }
}

# 1. With and without PRBVH (8 rounds), every model.
foreach ($m in $present) {
    foreach ($o in "none", "prbvh") {
        $tag = if ($o -eq "none") { "base" } else { "prbvh" }
        Run $m @("--build-bench", "--build", $builders, "--optimize", $o, "--runs", $runs, "--warmup", "1",
                 "--validate", "--tag", $tag, "--build-csv", $csv)
    }
}

# 2. Convergence: SAH cost and time against the number of rounds (dragon).
foreach ($i in 1, 2, 4, 8, 16, 32) {
    Run $dragon @("--build-bench", "--build", $builders, "--prbvh-iters", "$i", "--runs", $runs, "--warmup", "1",
                  "--tag", "prbvh_iters", "--build-csv", $csv)
}

# 3. Render time on the trees, base then PRBVH, 16 AO at 1080p.
foreach ($m in $dragon, $mixed) {
    if ($BuildOnly) { break }
    foreach ($o in "none", "prbvh") {
        $tag = if ($o -eq "none") { "base" } else { "prbvh" }
        Run $m @("--quality-bench", "--build", $builders, "--optimize", $o, "--width", "1920", "--height", "1080",
                 "--ao", "16", "--mode", "gpu_bvh_opt", "--runs", $qruns, "--tag", $tag, "--csv", $renderCsv)
    }
}
Write-Host "`nwrote $csv and $renderCsv"

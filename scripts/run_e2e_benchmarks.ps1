# Total time to image per BVH builder: one wall clock from "mesh in host memory" to "image in
# host memory" (mesh upload, allocations, build, tree upload for CPU builders, gpu_bvh_opt render,
# image download, frees). Builders interleaved round-robin, median of 5 runs after a warm-up;
# an untimed GPU LBVH build + render before each timed run keeps the GPU clocked up.
# Writes results\e2e.csv.
#   powershell -ExecutionPolicy Bypass -File scripts\run_e2e_benchmarks.ps1 [-Quick]
param([switch]$Quick)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "build\raytracer.exe"
if (-not (Test-Path $exe)) { throw "build first: scripts\build.ps1" }
New-Item -ItemType Directory -Force results | Out-Null
$csv = "results\e2e.csv"
if (Test-Path $csv) { Remove-Item $csv }

$runs = if ($Quick) { 2 } else { 5 }
$builders = "cpu_sah,cpu_sah_full,gpu_lbvh,gpu_lbvh+prbvh,gpu_ploc,gpu_ploc+prbvh,gpu_hploc,gpu_hploc+prbvh," +
            "gpu_trbvh,gpu_trbvh+prbvh"
$models = @(
    @{ label = "bunny";  file = "data\bun_zipper.ply"; up = "y"; cam = "orbit" },
    @{ label = "dragon"; file = "data\dragon_vrip.ply"; up = "y"; cam = "orbit" },
    @{ label = "buddha"; file = "data\happy_vrip.ply";  up = "y"; cam = "orbit" },
    @{ label = "mixed";  file = "mixed";                up = "y"; cam = "inside" }
)
foreach ($m in $models) {
    if ($m.file -ne "mixed" -and -not (Test-Path $m.file)) { Write-Warning "$($m.label) missing, skipped"; continue }
    foreach ($ao in 4, 16, 64) {
        $a = @("--model", $m.file, "--label", $m.label, "--up", $m.up, "--cam", $m.cam, "--e2e-bench",
               "--build", $builders, "--width", "1920", "--height", "1080", "--ao", "$ao", "--runs", "$runs",
               "--warmup", "1", "--tag", "ao$ao", "--e2e-csv", $csv)
        Write-Host "`n>>> $($m.label) $ao AO" -ForegroundColor Cyan
        & $exe @a
        if ($LASTEXITCODE -ne 0) { Write-Warning "run failed: $($m.label) $ao AO" }
    }
}
Write-Host "`nwrote $csv; plot with: py scripts\plot_build.py"

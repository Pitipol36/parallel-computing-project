# Runs every experiment from the project plan and writes results\results.csv.
#   powershell -ExecutionPolicy Bypass -File scripts\run_benchmarks.ps1 [-Quick]
# -Quick uses 1 timed run everywhere (for a smoke test). Close games/GPU-heavy apps first:
# other GPU work makes the kernel timings noisy.
param([switch]$Quick)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "build\raytracer.exe"
if (-not (Test-Path $exe)) { throw "build first: scripts\build.ps1" }
$csv = "results\results.csv"
New-Item -ItemType Directory -Force results, renders | Out-Null
if (Test-Path $csv) { Remove-Item $csv }

$runs = if ($Quick) { 1 } else { 5 }
$slowCpu = if ($Quick) { 1 } else { 3 }   # CPU runs for the heaviest settings (noted in README)
$gpu = "gpu_bvh_naive,gpu_bvh_opt,gpu_bvh_persistent"

$models = @(
    @{ label = "bunny";  file = "data\bun_zipper.ply"; up = "y" },
    @{ label = "dragon"; file = "data\dragon_vrip.ply"; up = "y" },
    @{ label = "buddha"; file = "data\happy_vrip.ply";  up = "y" },
    @{ label = "lucy";   file = "data\lucy.ply";        up = "z" }
)

function Run([string]$tag, [hashtable]$m, [string[]]$extra) {
    $args_ = @("--model", $m.file, "--label", $m.label, "--up", $m.up, "--tag", $tag, "--csv", $csv, "--runs", $runs) + $extra
    Write-Host "`n>>> $tag $($m.label) $($extra -join ' ')" -ForegroundColor Cyan
    & $exe @args_
    if ($LASTEXITCODE -ne 0) { Write-Warning "run failed: $tag $($m.label)" }
}

$present = $models | Where-Object { Test-Path $_.file }
$dragon = $models | Where-Object { $_.label -eq "dragon" }
$bunny = $models | Where-Object { $_.label -eq "bunny" }
foreach ($m in $models) { if (-not (Test-Path $m.file)) { Write-Warning "$($m.file) missing, skipped" } }

# 1. Speedup vs mesh size (1080p, 4 AO). Also saves the showcase renders.
foreach ($m in $present) {
    Run "size" $m @("--width", "1920", "--height", "1080", "--ao", "4", "--mode", "cpu,$gpu",
                     "--cpu-runs", $slowCpu, "--out", "renders\$($m.label).png")
}
# 2. Speedup vs workload (AO samples), dragon 1080p.
foreach ($ao in 0, 1, 4, 16, 64) {
    $cr = if ($ao -ge 16) { $slowCpu } else { $runs }
    Run "ao" $dragon @("--width", "1920", "--height", "1080", "--ao", "$ao", "--mode", "cpu,gpu_bvh_opt,gpu_bvh_persistent", "--cpu-runs", $cr)
}
# 3. Speedup vs resolution, dragon 4 AO.
foreach ($res in @(@(512, 512), @(1920, 1080), @(3840, 2160))) {
    $cr = if ($res[0] -ge 3840) { $slowCpu } else { $runs }
    Run "resolution" $dragon @("--width", "$($res[0])", "--height", "$($res[1])", "--ao", "4", "--mode", "cpu,gpu_bvh_opt,gpu_bvh_persistent", "--cpu-runs", $cr)
}
# 4. Ablation: BVH kernels on the dragon; brute force only on the bunny at low resolution.
Run "ablation" $dragon @("--width", "1920", "--height", "1080", "--ao", "4", "--mode", $gpu)
Run "ablation_brute" $bunny @("--width", "320", "--height", "240", "--ao", "4", "--mode", "cpu,gpu_brute,$gpu")
# 5. Block-size tuning.
foreach ($b in "8x8", "16x16", "32x8", "32x32") {
    Run "block" $dragon @("--width", "1920", "--height", "1080", "--ao", "4", "--mode", "gpu_bvh_naive,gpu_bvh_opt,gpu_bvh_persistent", "--block", $b)
}
Write-Host "`nwrote $csv; plot with: py scripts\plot_results.py"

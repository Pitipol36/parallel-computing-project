# GPU BVH construction experiments. Writes, in results\:
#   build.csv          build time per stage + tree quality, every builder x model
#   build_render.csv   render time on each builder's tree (quality and rays-per-build sweeps)
#   traversal.csv      box / triangle tests per ray on each tree (needs build-stats\, see below)
#   dynamic.csv        per-frame times and SAH cost of the exploding-dragon scene
# The previous run's CSVs are kept once in results\v1\ for comparison.
#   powershell -ExecutionPolicy Bypass -File scripts\run_build_benchmarks.ps1 [-Quick]
# The traversal counters need the counting build: scripts\build.ps1 -Stats (skipped when
# build-stats\raytracer.exe is missing). Close GPU-heavy apps first.
#
# Measurement method: quality renders use --quality-bench (all trees built first, 3 s GPU
# warm-up, runs interleaved round-robin across builders, median and IQR of 11 runs) at 16 AO,
# so kernels run for tens of ms. GPU clocks could not be locked (nvidia-smi -lgc needs
# administrator rights), which the warm-up and interleaving are there to compensate for.
param([switch]$Quick)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root "build\raytracer.exe"
$statsExe = Join-Path $root "build-stats\raytracer.exe"
if (-not (Test-Path $exe)) { throw "build first: scripts\build.ps1" }
New-Item -ItemType Directory -Force results, renders | Out-Null
$buildCsv = "results\build.csv"; $renderCsv = "results\build_render.csv"
$travCsv = "results\traversal.csv"; $dynCsv = "results\dynamic.csv"
if (-not (Test-Path "results\v1")) {
    New-Item -ItemType Directory -Force "results\v1" | Out-Null
    foreach ($f in $buildCsv, $renderCsv, $travCsv, $dynCsv) { if (Test-Path $f) { Copy-Item $f "results\v1\" } }
}
foreach ($f in $buildCsv, $renderCsv, $travCsv, $dynCsv) { if (Test-Path $f) { Remove-Item $f } }

$runs = if ($Quick) { 1 } else { 5 }        # build timings
$qruns = if ($Quick) { 3 } else { 11 }      # render timings (median + IQR)
$slowCpu = if ($Quick) { 1 } else { 3 }
$builders = "cpu_sah,cpu_sah_full,cpu_lbvh,gpu_lbvh,cpu_ploc,gpu_ploc,cpu_hploc,gpu_hploc,cpu_trbvh,gpu_trbvh"
$trees = "cpu_sah,cpu_sah_full,gpu_lbvh,gpu_ploc,gpu_hploc,gpu_trbvh"   # cpu_* twins build identical trees
$models = @(
    @{ label = "bunny";  file = "data\bun_zipper.ply"; up = "y"; cam = "orbit" },
    @{ label = "dragon"; file = "data\dragon_vrip.ply"; up = "y"; cam = "orbit" },
    @{ label = "buddha"; file = "data\happy_vrip.ply";  up = "y"; cam = "orbit" },
    @{ label = "lucy";   file = "data\lucy.ply";        up = "z"; cam = "orbit" },
    @{ label = "mixed";  file = "mixed";                up = "y"; cam = "inside" }   # procedural, from dragon + bunny
)
$present = $models | Where-Object { $_.file -eq "mixed" -and (Test-Path "data\dragon_vrip.ply") -and (Test-Path "data\bun_zipper.ply") -or (Test-Path $_.file) }
$dragon = $models | Where-Object { $_.label -eq "dragon" }
foreach ($m in $models) { if (-not ($present -contains $m)) { Write-Warning "$($m.label) missing, skipped" } }

function Run([string]$exePath, [hashtable]$m, [string[]]$extra) {
    $a = @("--model", $m.file, "--label", $m.label, "--up", $m.up, "--cam", $m.cam) + $extra
    Write-Host "`n>>> $($m.label) $($extra -join ' ')" -ForegroundColor Cyan
    & $exePath @a
    if ($LASTEXITCODE -ne 0) { Write-Warning "run failed: $($m.label) $($extra -join ' ')" }
}

# 1 + 2. Build time vs mesh size, with the per-stage breakdown and tree quality.
foreach ($m in $present) {
    Run $exe $m @("--build-bench", "--build", $builders, "--runs", $runs, "--warmup", "1", "--validate",
                  "--tag", "size", "--build-csv", $buildCsv)
}

# PLOC radius sweep (dragon): build time and SAH cost against the search radius r.
foreach ($r in 1, 2, 4, 8, 16, 32, 64) {
    Run $exe $dragon @("--build-bench", "--build", "gpu_ploc", "--ploc-radius", "$r", "--runs", $runs, "--warmup", "1",
                       "--tag", "radius", "--build-csv", $buildCsv)
}
# H-PLOC radius x threshold sweep (dragon).
foreach ($r in 4, 8, 16) {
    foreach ($t in 8, 16) {
        Run $exe $dragon @("--build-bench", "--build", "gpu_hploc", "--hploc-radius", "$r", "--hploc-threshold", "$t",
                           "--runs", $runs, "--warmup", "1", "--tag", "hploc_sweep", "--build-csv", $buildCsv)
    }
}

# TRBVH sweeps (dragon): passes 0-4 with 7-leaf treelets, 5-leaf treelets, and one thread per
# treelet instead of one warp.
foreach ($p in 0, 1, 2, 3, 4) {
    Run $exe $dragon @("--build-bench", "--build", "gpu_trbvh", "--trbvh-passes", "$p", "--runs", $runs, "--warmup", "1",
                       "--tag", "trbvh_passes", "--build-csv", $buildCsv)
}
Run $exe $dragon @("--build-bench", "--build", "gpu_trbvh", "--trbvh-treelet", "5", "--runs", $runs, "--warmup", "1",
                   "--tag", "trbvh_treelet", "--build-csv", $buildCsv)
Run $exe $dragon @("--build-bench", "--build", "gpu_trbvh", "--trbvh-thread", "--runs", $runs, "--warmup", "1",
                   "--tag", "trbvh_thread", "--build-csv", $buildCsv)

# 3. Quality: render every tree, interleaved round-robin, 16 AO, median + IQR.
foreach ($m in $present) {
    Run $exe $m @("--quality-bench", "--build", $trees, "--width", "1920", "--height", "1080", "--ao", "16",
                  "--mode", "gpu_bvh_opt,cpu", "--runs", $qruns, "--cpu-runs", $slowCpu, "--tag", "quality",
                  "--csv", $renderCsv)
    if (Test-Path $statsExe) {
        foreach ($b in $trees.Split(",")) {
            Run $statsExe $m @("--build", $b, "--width", "1920", "--height", "1080", "--ao", "16",
                               "--mode", "gpu_bvh_opt", "--runs", "1", "--warmup", "0",
                               "--tag", "quality", "--stats-csv", $travCsv)
        }
    }
}

# 5. Rays per build: render time vs AO samples on each tree (dragon, 1080p), interleaved.
foreach ($ao in 0, 1, 4, 16, 64) {
    Run $exe $dragon @("--quality-bench", "--build", "cpu_sah,gpu_lbvh,gpu_ploc,gpu_hploc,gpu_trbvh", "--width", "1920",
                       "--height", "1080", "--ao", "$ao", "--mode", "gpu_bvh_opt", "--runs", $qruns,
                       "--tag", "rays", "--csv", $renderCsv)
}

# 4. Dynamic scene: exploding dragon, rebuilds (CPU SAH, GPU LBVH, PLOC, H-PLOC, TRBVH) vs refit.
$frames = if ($Quick) { 10 } else { 60 }
Run $exe $dragon @("--dynamic", "--frames", "$frames", "--width", "1280", "--height", "720", "--ao", "4",
                   "--dynamic-csv", $dynCsv, "--out", "renders\explode.png")

Write-Host "`nwrote results\build*.csv, traversal.csv, dynamic.csv; plot with: py scripts\plot_build.py"

#!/usr/bin/env bash
# Runs every experiment from the project plan and writes results/results.csv.
#   scripts/run_benchmarks.sh [--quick]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
exe="$root/build/raytracer"
[[ -x "$exe" ]] || { echo "build first (see README)"; exit 1; }
csv="results/results.csv"
mkdir -p results renders
rm -f "$csv"

runs=5; slow_cpu=3
if [[ "${1:-}" == "--quick" ]]; then runs=1; slow_cpu=1; fi
gpu="gpu_bvh_naive,gpu_bvh_opt,gpu_bvh_persistent"

declare -A file=( [bunny]=data/bun_zipper.ply [dragon]=data/dragon_vrip.ply [buddha]=data/happy_vrip.ply [lucy]=data/lucy.ply )
declare -A up=( [bunny]=y [dragon]=y [buddha]=y [lucy]=z )

run() {  # run TAG LABEL extra-args...
  local tag="$1" label="$2"; shift 2
  [[ -f "${file[$label]}" ]] || { echo "skip $label (missing ${file[$label]})"; return; }
  echo; echo ">>> $tag $label $*"
  "$exe" --model "${file[$label]}" --label "$label" --up "${up[$label]}" --tag "$tag" --csv "$csv" --runs "$runs" "$@" \
    || echo "run failed: $tag $label"
}

for m in bunny dragon buddha lucy; do
  run size "$m" --width 1920 --height 1080 --ao 4 --mode "cpu,$gpu" --cpu-runs "$slow_cpu" --out "renders/$m.png"
done
for ao in 0 1 4 16 64; do
  cr=$runs; (( ao >= 16 )) && cr=$slow_cpu
  run ao dragon --width 1920 --height 1080 --ao "$ao" --mode cpu,gpu_bvh_opt,gpu_bvh_persistent --cpu-runs "$cr"
done
for res in 512x512 1920x1080 3840x2160; do
  w=${res%x*}; h=${res#*x}; cr=$runs; (( w >= 3840 )) && cr=$slow_cpu
  run resolution dragon --width "$w" --height "$h" --ao 4 --mode cpu,gpu_bvh_opt,gpu_bvh_persistent --cpu-runs "$cr"
done
run ablation dragon --width 1920 --height 1080 --ao 4 --mode "$gpu"
run ablation_brute bunny --width 320 --height 240 --ao 4 --mode "cpu,gpu_brute,$gpu"
for b in 8x8 16x16 32x8 32x32; do
  run block dragon --width 1920 --height 1080 --ao 4 --mode "$gpu" --block "$b"
done
echo; echo "wrote $csv; plot with: python3 scripts/plot_results.py"

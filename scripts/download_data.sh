#!/usr/bin/env bash
# Downloads the Stanford 3D Scanning Repository models into data/.
# Tries graphics.stanford.edu first and falls back to a mirror of the original PLY files
# (Harvey Mudd CS155 course archive) when Stanford is unreachable.
#   scripts/download_data.sh          # bunny, dragon, buddha
#   scripts/download_data.sh --lucy   # also Lucy (~28M triangles; Stanford only)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
data="$root/data"
tmp="$data/_download"
mkdir -p "$data" "$tmp"

stanford="http://graphics.stanford.edu/pub/3Dscanrep"
mirror="https://www.cs.hmc.edu/courses/2009/fall/cs155/archive/models"
models=(
  "$stanford/bunny.tar.gz|bunny/reconstruction/bun_zipper.ply|bun_zipper.ply|$mirror/bunny/reconstruction/bun_zipper.ply"
  "$stanford/dragon/dragon_recon.tar.gz|dragon_recon/dragon_vrip.ply|dragon_vrip.ply|$mirror/dragon/dragon_vrip.ply"
  "$stanford/happy/happy_recon.tar.gz|happy_recon/happy_vrip.ply|happy_vrip.ply|$mirror/happy_buddah/happy_vrip.ply"
)
if [[ "${1:-}" == "--lucy" ]]; then
  models+=("http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz|lucy.ply|lucy.ply|")
fi

for entry in "${models[@]}"; do
  IFS="|" read -r url member out mirror_url <<< "$entry"
  if [[ -f "$data/$out" ]]; then echo "have $out"; continue; fi
  archive="$tmp/$(basename "$url")"
  echo "downloading $url"
  if curl -fL --retry 2 --connect-timeout 30 -o "$archive" "$url" && tar -xzf "$archive" -C "$tmp" "$member"; then
    mv "$tmp/$member" "$data/$out"
  elif [[ -n "$mirror_url" ]]; then
    echo "  Stanford unavailable; trying mirror $mirror_url"
    curl -fL --retry 3 -o "$data/$out" "$mirror_url"
  else
    echo "  failed and no mirror for $out" >&2
    continue
  fi
  echo "  -> data/$out"
done
rm -rf "$tmp"
ls -lh "$data"/*.ply

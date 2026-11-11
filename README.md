# BVH Ray Tracing of Large Triangle Meshes: Sequential CPU vs CUDA

Parallel Computing (CUDA) term project. Pitipol Tangcherdchar.

This project renders 3D-scanned triangle meshes with up to about a million triangles. Each pixel
casts one **primary ray**, one **shadow ray** toward a directional light, and *N* **ambient-occlusion
(AO) rays**. Each ray has to find its closest (or any) hit among all the triangles. A
**binned-SAH bounding volume hierarchy (BVH)** brings that down from O(T) to about O(log T) per
ray. The program compares a tuned single-threaded CPU baseline against four CUDA kernel versions
that all use the *same* BVH, camera, rays and random numbers, so their outputs can be checked
against each other pixel for pixel.

![dragon](renders/dragon_gpu_bvh_opt.png)

## Dataset

[Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/):

| Model | File | Triangles | Format |
|---|---|---|---|
| Bunny | `bun_zipper.ply` | 69,451 | ASCII PLY |
| Dragon | `dragon_vrip.ply` | 871,414 | ASCII PLY |
| Happy Buddha | `happy_vrip.ply` | 1,087,716 | ASCII PLY |
| Lucy *(optional)* | `lucy.ply` | ~28M | binary PLY |

`graphics.stanford.edu` could not be reached from the development machine, so the download
script falls back to an unmodified copy of the original PLY files in the
[Harvey Mudd CS155 model archive](https://www.cs.hmc.edu/courses/2009/fall/cs155/archive/models/).
**Lucy is only on the Stanford server**, so as the plan allows, the largest model in the reported
results is the **Happy Buddha** (1.09M triangles). If you can reach Stanford, run the script with
`-Lucy` / `--lucy` and the benchmark script picks it up automatically (`--up z`). Lucy needs
about 1.8 GB of GPU memory and about 4 GB of host RAM.

A procedural sphere-on-ground scene (20K triangles) is built into the program for unit tests and
`--verify`, so neither needs a download.

## Requirements

| | Used for the reported results |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 6 GB (Turing, sm_75, 30 SMs) |
| CUDA | 13.3 (any CUDA ≥ 11 should work; CUDA 13 needs sm_75 or newer) |
| Host compiler | MSVC 19.51 (Visual Studio 2026); GCC/Clang with C++17 on Linux |
| CMake | ≥ 3.24 (VS ships one), Ninja |
| CPU | Intel Core i7-9700F @ 3.0 GHz (baseline uses one core) |
| OS | Windows 10 Pro 22H2 |
| Plots | Python 3 + matplotlib (`py -m pip install matplotlib`) |

The only third-party code is `include/stb_image_write.h` (public domain, PNG output). There is
no OptiX, Embree or other ray-tracing library: the BVH, traversal and intersection code is all
hand-written.

## Build

The GPU architecture defaults to `native` (the GPU in the build machine). Override it with
`-DCMAKE_CUDA_ARCHITECTURES=75` (or 86, 89, …) when building for a different GPU.

**Windows** (finds Visual Studio, loads its x64 environment, then runs CMake + Ninja):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

Manual build from a *x64 Native Tools* prompt:
`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` then `cmake --build build`.

**Linux:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

The build produces `build/raytracer(.exe)` and `build/rt_tests(.exe)` (`ctest --test-dir build`
runs the unit tests).

## Download the data

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1     # add -Lucy for Lucy
```
```bash
scripts/download_data.sh                                              # add --lucy for Lucy
```

The meshes go in `data/`, which Git ignores.

## Run

```bash
# One model, CPU baseline vs optimized GPU kernel, writes renders/dragon_<mode>.png
build/raytracer --model data/dragon_vrip.ply --width 1920 --height 1080 --ao 4 \
                --mode cpu,gpu_bvh_opt --out renders/dragon.png

# Every GPU kernel, 32x8 blocks, timings appended to a CSV
build/raytracer --model data/happy_vrip.ply --mode gpu_all --block 32x8 --csv results/mine.csv

# Correctness: unit tests + CPU vs every GPU kernel on the test scene, bunny and dragon
build/raytracer --verify
```

Modes are `cpu`, `gpu_brute`, `gpu_bvh_naive`, `gpu_bvh_opt` and `gpu_bvh_persistent`, plus
`gpu_all` (the four GPU modes) and `all`. Run `raytracer --help` for every option (camera
`--yaw/--pitch/--fov/--up`, `--ao-radius`, `--bins`, `--leaf`, `--runs`, `--warmup`,
`--validate`, …).

Each mode does 1 warm-up run and then reports the **median of 5** timed runs (`--warmup`,
`--runs`). The timings are:

* **CPU**: wall-clock time of the render loop (`std::chrono`).
* **GPU kernel**: `cudaEvent` time around the kernel launch(es) only.
* **GPU end to end**: wall time of device allocation + host-to-device upload of the BVH and
  triangles + kernel + device-to-host copy of the image + free.
* **BVH build**: timed on its own. It is shared by both versions and runs on the CPU.

"Mrays/s" uses the nominal ray count *W·H·(2 + AO)*.

### Benchmarks and plots

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_benchmarks.ps1   # ~10 min; -Quick for 1 run each
py scripts\plot_results.py
```
```bash
scripts/run_benchmarks.sh && python3 scripts/plot_results.py
```

These write `results/results.csv`, plus `results/plots/*.png` for the slides:

| Experiment (`experiment` column) | Setting |
|---|---|
| `size` | bunny, dragon, buddha (lucy) at 1920×1080, 4 AO |
| `ao` | dragon 1080p with 0, 1, 4, 16, 64 AO samples |
| `resolution` | dragon, 4 AO, 512², 1080p, 4K |
| `ablation` | naive / opt / persistent kernels on the dragon |
| `ablation_brute` | CPU and all four kernels on the bunny at 320×240 (brute force is far too slow for more) |
| `block` | 8×8, 16×16, 32×8, 32×32 blocks |

The CPU uses 3 timed runs instead of 5 for the heaviest settings (16/64 AO, 4K, and the size
sweep) to keep the script's run time reasonable. The GPU comparison is never reduced.
**Close games and other GPU-heavy programs before benchmarking.** On a shared GPU, kernel times
for the same setting vary by up to 4×.

## Design

### Sequential baseline (CPU), and why it is the best *sequential* version

1. **Custom PLY loader** (`src/ply_loader.cpp`) for ASCII, binary little- and big-endian files,
   with any element/property layout. Polygons are fan-triangulated.
2. **Binned SAH BVH** (`src/bvh_build.cpp`): 16 bins per axis (configurable up to 32), cost
   *C_trav + (A_L·N_L + A_R·N_R)/A*. A node becomes a leaf at ≤ 4 triangles when splitting is
   not cheaper. Depth is capped at 62 so the 64-entry traversal stack can never overflow.
   Build cost is O(T log T).
3. **Flattened depth-first layout**: 32-byte nodes (two `float4`s). The left child is always
   the next node, so only the right child index (or first triangle + count) is stored.
4. **Iterative ordered traversal** (`include/trace.h`): visits the nearer child first, keeps a
   stack of entry distances, and skips any subtree whose entry is beyond the current closest
   hit. Uses a precomputed reciprocal ray direction.
5. **Möller–Trumbore** on precomputed `v0, e1, e2` (48-byte triangles).
6. **Any-hit traversal** for shadow and AO rays, which stops at the first hit within `tMax`.
7. **Shading**: albedo × (0.7 · Lambert · shadow + 0.3 · (1 − occlusion)), with sqrt gamma.

Brute force costs O(P·T). The BVH gives O(P log T) + O(T log T) for the build, and SAH is the
standard measure of BVH quality (PBRT and the research literature). The CPU code is compiled
`/O2 /arch:AVX2` (or `-O3 -march=native`). The hot path avoids MSVC's out-of-line `fminf`
(swapping it for compare-select made both the CPU render and the BVH build about 6× faster). It
is single-threaded on purpose. "Best possible" here means the best known acceleration structure
with single-ray traversal tuned for one core. SIMD packet tracing and wide BVHs (BVH4/8) are
faster still, but they are a different algorithm.

### Parallel version (CPU builds, GPU traces)

* One thread per pixel (primary + shadow + N AO rays), 2D grid of 2D blocks (default 16×16).
* Per-thread traversal stack (64 entries) in local memory.
* Deterministic **PCG hash** RNG keyed by (pixel, sample, seed). AO directions use
  rejection-sampled cosine weighting (no `sin`/`cos`), so the CPU and GPU generate
  **bit-identical** rays.
* The ray, AABB, triangle, traversal and shading code is written once as `__host__ __device__`
  in `include/vecmath.h` and `include/trace.h`. The CPU and `gpu_bvh_opt` compile literally the
  same functions.

| Kernel | What it does | Targets |
|---|---|---|
| `gpu_brute` | each ray tests every triangle | shows that the algorithm matters, not only raw parallelism |
| `gpu_bvh_naive` | same BVH, but 40-byte array-of-structs nodes, indexed vertices (edges recomputed), plain loads, a division per slab test, fixed child order | GPU BVH baseline |
| `gpu_bvh_opt` | `float4`-aligned nodes/triangles via `__ldg`, precomputed inverse direction, nearer-child-first with distance culling | memory layout and access |
| `gpu_bvh_persistent` | `gpu_bvh_opt` traversal; launches just enough warps to fill the GPU (occupancy API), each warp pulls 8×4 pixel tiles from a global `atomicAdd` counter | load imbalance between background and dense-geometry pixels |

`gpu_brute` is split into strips of 8 rows per launch so that no single launch trips the
Windows display watchdog (TDR, ~2 s).

## Correctness

`raytracer --verify` runs:

* **Unit tests** (98 checks): ray–triangle (hit, miss, exact edge/vertex, parallel, behind),
  ray–AABB (zero direction components, `tMax` culling, origin inside, misses), BVH closest-hit and
  any-hit against brute force on 4000 random rays for three meshes and build settings, ASCII and
  binary PLY round trips, RNG determinism and cosine distribution, and the LBVH tests listed under
  [GPU BVH construction](#gpu-bvh-construction-lbvh-ploc).
* **BVH validation**: every triangle appears in exactly one leaf, every node is reachable once,
  child boxes lie inside their parent, leaves contain their triangles, and depth fits the stack.
* **Hit-ID test**: the primary-ray triangle ID per pixel must match the CPU in ≥ 99.99 % of pixels.
* **Image test**: max absolute difference and PSNR (must be ≥ 40 dB). Fast-math is never used.

Equal-distance hits are broken by the smaller triangle index, so the result does not depend on
traversal order. The only remaining CPU/GPU difference is the GPU's fused multiply-add, which
changes a handful of pixels exactly on triangle edges.

Results at 640×480, 4 AO (RTX 2060, current build):

| Scene | Kernel | Hit-ID mismatches | Pixels differing | PSNR |
|---|---|---|---|---|
| test scene (20K) | all four | 0 / 307,200 | 1 | 67.2 dB |
| bunny (69K) | all four | 3 / 307,200 (0.0010 %) | 2 | 64.6 dB |
| dragon (871K) | naive, opt, persistent | 7 / 307,200 (0.0023 %) | 8 | 56.7 dB |

(An earlier build, used for the presentation slides, measured 1 / 2 / 5 mismatches. The
difference comes from code generation of the edge cases, not from the tree: the commit before the
LBVH work also gives 7 on the dragon.)

## Results

These numbers are from `scripts/run_benchmarks.ps1` on the hardware above, with a median of 5
GPU runs (3 CPU runs for heavy settings). Raw data is in `results/results.csv` and the charts
are in `results/plots/`. The desktop was still running light background programs, so repeated
GPU measurements of the *same* setting vary by about ±25 % (dragon 1080p / 4 AO measured
6.6–11.1 ms across experiments). Read small differences between kernels as noise.

**Speedup vs mesh size** (1080p, 4 AO; kernel time, speedup over the 1-thread CPU):

| Model | Triangles | BVH build | CPU | gpu_bvh_naive | gpu_bvh_opt | gpu_bvh_persistent | opt end to end |
|---|---|---|---|---|---|---|---|
| bunny | 69K | 0.06 s | 743 ms | 13.5 ms (55×) | 4.1 ms (182×) | 4.6 ms (161×) | 8.4 ms (88×) |
| dragon | 871K | 0.77 s | 974 ms | 28.0 ms (35×) | 6.7 ms (146×) | 6.6 ms (147×) | 23.0 ms (42×) |
| buddha | 1.09M | 0.92 s | 937 ms | 27.2 ms (34×) | 8.3 ms (113×) | 8.9 ms (105×) | 27.7 ms (34×) |

**Speedup vs workload** (dragon, 1080p, gpu_bvh_opt): 0 AO 92×, 1 AO 119×, 4 AO 88×,
16 AO 132×, 64 AO 101× (CPU 8.7 s vs GPU 87 ms for 145M rays).

**Speedup vs resolution** (dragon, 4 AO, gpu_bvh_opt): 512² 47×, 1080p 113×, 4K **145×**.
End to end, the speedup goes from 11× to 79×, because the fixed upload cost is amortized
over more pixels.

**Ablation** (dragon 1080p, 4 AO): naive 19.7 ms → opt 6.7 ms (**3.0×** from the memory layout
and access changes); persistent 6.6 ms (no measurable gain on Turing). On the bunny at 320×240,
**brute force takes 3.6 s against 0.48 ms with the BVH (7,500×)**. At that size the 1-thread CPU
with a BVH (38 ms) is about 100× faster than the GPU without one.

**Block size** (dragon, gpu_bvh_opt): 8×8 6.6 ms, 32×8 7.0 ms, 16×16 8.3 ms, 32×32 8.6 ms.
Narrow or small blocks win slightly because pixels close together in a block trace more
coherent paths.

**What the results say:**

* The GPU is **35–180× faster** than the sequential BVH code on every setting, and
  4 orders of magnitude faster than brute force on the same GPU.
* The gap **widens as the ray workload grows** (resolution: 47× → 145×), because the GPU
  needs many rays in flight to hide memory latency.
* The gap **narrows as the mesh grows** (182× → 113×). A BVH makes the CPU's cost only about
  log T, but on the GPU a bigger BVH overflows the 3 MB L2 cache, so traversal becomes
  memory-bound. End to end, the upload of ~80 MB of nodes and triangles also starts to matter.
* Persistent threads did not help: on modern GPUs the hardware block scheduler already balances
  cheap background pixels against expensive geometry pixels. (Aila & Laine's gains were on older
  architectures.)
* The BVH build on the CPU (0.8–0.9 s) takes **longer than the GPU render plus transfer** for
  the big models (see `time_breakdown.png`). That makes a GPU BVH build the obvious next step.

## GPU BVH construction (LBVH, PLOC)

Follow-up to the presentation (`gpu-bvh-build-plan.md`). Goal 1: build the BVH on the GPU with
Karras's LBVH, measure the speedup, and state what it costs in tree quality. Goal 2: win that
quality back with higher-quality parallel builders, PLOC (done) and TRBVH (in progress).

**Why.** With the CPU SAH build, the build is most of the time to the first image
(1080p, 4 AO, `gpu_bvh_opt`, end to end):

| Model | CPU SAH build | GPU render, end to end | Build share |
|---|---|---|---|
| bunny | 57 ms | 8.8 ms | 87 % |
| dragon | 771 ms | 25.6 ms | 97 % |
| buddha | 941 ms | 28.6 ms | 97 % |

The best builder minimises build + render. Few rays per build (previews, dynamic scenes, low
sample counts) favour a fast builder; many rays per build favour a high-quality one. Scenes
with incoherent motion (explosions, particles, destruction) need a full rebuild every frame,
because refitting a fixed topology collapses in quality (measured below).

### Builders (`--build`)

| Builder | What it is |
|---|---|
| `cpu_sah` | the existing binned SAH builder: the quality reference (100 %) |
| `cpu_lbvh` | Karras 2012 LBVH, single-threaded: same algorithm as the GPU, so the build speedup comes from parallelism, not a cheaper algorithm |
| `gpu_lbvh` | the same LBVH as CUDA kernels; **node-for-node identical** to `cpu_lbvh` |
| `cpu_ploc` | PLOC (Meister & Bittner 2018), single-threaded |
| `gpu_ploc` | the same PLOC as CUDA kernels; **node-for-node identical** to `cpu_ploc` |

All builders emit the same flattened 32-byte node layout, so every render mode runs unchanged
on any tree, and "which builder" is the only variable in every experiment.

**Shared stages** (`src/gpu_build_common.cu`), one thread per element:

1. **Bounds**: triangle AABBs and centroids, then a hand-written min/max reduction (warp
   shuffles + shared memory, two passes) for the centroid bounds.
2. **Morton codes**: 63-bit (21 bits per axis), so even Lucy-sized meshes rarely produce
   duplicate keys. Reference code commonly uses 30-bit codes.
3. **Sort**: `cub::DeviceRadixSort::SortPairs` on (code, triangle index). It is stable, so
   equal codes keep index order exactly as the CPU's `(code, index)` sort does.
4. **Collapse + flatten** (after the builder-specific stages): every binary-tree builder
   produces one triangle per leaf. Bottom-up, each node gets an SAH cost,
   `min(C_trav·A + cost(left) + cost(right), C_isect·N·A if N ≤ 4)`, and becomes a leaf when that
   is cheaper (`--collapse count` instead collapses every subtree with ≤ 4 triangles). Each kept
   node then finds its depth-first position by walking to the root, adding 1 per level plus the
   left sibling's collapsed size whenever it comes from a right child, so no global
   synchronisation is needed. Triangles find their output slot the same way, so a subtree's
   triangles need not be contiguous in the builder's order (PLOC's are not).

**LBVH** (`src/bvh_build_lbvh_gpu.cu`):

* **Hierarchy**: one thread per internal node finds its range and split (Karras 2012, Fig. 4).
  δ(i, j) = `clz(key_i ^ key_j)`; equal keys fall back to 64 + the common prefix of the indices.
* **Bottom-up fit**: one thread per leaf climbs; at each node it does `__threadfence()` and
  `atomicAdd` on an arrival counter. The first thread to arrive stops, the second unions the
  children (read with `__ldcg`, because the sibling may have been written on another SM) and
  computes the triangle count, SAH cost and collapsed size.
* `GpuLbvhBuilder` keeps its buffers between builds, builds from device-resident vertices, and
  can **refit** a finished tree (new positions, same topology) with the same atomic pattern.

**PLOC** (`src/bvh_build_ploc_gpu.cu`): clusters start as one triangle each, in Morton order.
Each iteration runs four data-parallel kernels over the current clusters:

1. **Nearest neighbour**: one thread per cluster searches the r clusters on each side for the
   one whose merged box has the smallest surface area. The block's clusters plus an r-wide halo
   on each side are staged in shared memory.
2. **Flags**: mutual nearest neighbours merge; the lower index creates the node. Each cluster
   packs `(creates a node) << 32 | (survives)` into one 64-bit value.
3. **Scan**: a hand-written exclusive scan of those values (warp-shuffle block scan, recursive
   scan of the block sums, add back: the Lab 6 pattern). The low half gives each survivor its
   new slot, the high half gives each new node its id, in creation order.
4. **Merge**: new internal nodes are written (box, children, triangle count, SAH cost), and
   survivors are scattered in order.

The CPU reads back one 8-byte total per iteration (the next cluster count). Distances use
round-to-nearest arithmetic with no contraction, and ties are broken by a total order on
pairs, so the CPU and GPU agree bit for bit and the globally closest pair is always mutual
(every iteration makes progress).

The per-element logic (Morton, δ, the split, the PLOC distance and tie-break, the collapse and
the converter) is `__host__ __device__` code in `include/morton.h`, `include/ploc.h` and
`include/bvh_common.h`, shared by the CPU and GPU builders.

### Running it

```bash
# Build-time benchmark: median per stage + tree quality, appended to results/build.csv
build/raytracer --model data/dragon_vrip.ply --build-bench --build cpu_sah,gpu_lbvh,gpu_ploc --validate

# Render with a GPU-built tree; PLOC with a different search radius
build/raytracer --model data/dragon_vrip.ply --build gpu_ploc --ploc-radius 16 --mode gpu_bvh_opt

# Exploding dragon: CPU SAH rebuild, GPU LBVH rebuild, refit and GPU PLOC rebuild, every frame
build/raytracer --model data/dragon_vrip.ply --dynamic --frames 60 --width 1280 --height 720 --out renders/explode.png

# Box/triangle tests per ray need the counting build (never use it for timing)
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Stats
build-stats/raytracer --model data/dragon_vrip.ply --build gpu_ploc --mode gpu_bvh_opt

# Everything, then the charts
powershell -ExecutionPolicy Bypass -File scripts\run_build_benchmarks.ps1
py scripts\plot_build.py
```

### Correctness

* **Unit tests**: Morton expansion and bit order; δ with duplicate keys; hand-checked
  hierarchies for n = 2, 3 and 4 identical keys; the Karras tree against an independent
  top-down reference for 60 random sorted key sets with many duplicates (n = 2 … 2,233); the
  PLOC tie-break and distance; every builder against brute force on random rays; PLOC at least
  as good as LBVH by SAH cost; GPU LBVH and GPU PLOC node-for-node equal to their CPU versions on
  five meshes each (including a single triangle and a mesh with every triangle duplicated) and,
  for PLOC, the same number of iterations.
* **`--verify`** on the test scene, bunny and dragon: every tree passes structure validation,
  each GPU tree equals its CPU twin byte for byte, rendering any tree gives exactly the same
  per-pixel triangle IDs as the SAH tree (the closest hit does not depend on the BVH), and the
  device-resident render matches the host-copy render.
* **`compute-sanitizer`** `memcheck`, `racecheck`, `synccheck` and `initcheck` on both GPU
  builders: 0 errors, 0 hazards.

### Results

Median of 5 builds after a warm-up (`results/build.csv`, PLOC radius 8). GPU times are
`cudaEvent` time of the build kernels; the mesh upload (10–15 ms for the big models) is extra.

| Model | CPU SAH | CPU LBVH | GPU LBVH | CPU PLOC | GPU PLOC | GPU LBVH vs CPU SAH | GPU PLOC vs CPU SAH |
|---|---|---|---|---|---|---|---|
| bunny (69K) | 57.3 ms | 22.4 ms | 0.51 ms | 43.9 ms | 3.5 ms | 111× | 16× |
| dragon (871K) | 800 ms | 335 ms | 4.8 ms | 735 ms | 13.4 ms | 167× | 60× |
| buddha (1.09M) | 967 ms | 426 ms | 6.0 ms | 966 ms | 13.7 ms | 162× | 71× |

Against its own single-threaded version, GPU LBVH is 44–72× faster and GPU PLOC 12–71× faster.

![Build time](results/plots/build_time_vs_size.png)

**Stages** (`build_stages.png`): GPU LBVH is dominated by the sort, the bottom-up fit and the
collapse/flatten (each 1–2.4 ms on the Buddha). GPU PLOC spends 70 % of its time clustering
(9.4 ms, 64 iterations on the Buddha), where the per-iteration launches and the 8-byte readback
add up once only a few thousand clusters remain.

**Quality** (`build_quality.png`, `pareto.png`; SAH cost with C_trav = 1.2, C_isect = 1.0;
counters from the `-Stats` build at 1080p, 4 AO):

| Model | SAH cost: LBVH / PLOC (% of SAH) | Box tests / ray: SAH, LBVH, PLOC | Triangle tests / ray | 1-thread CPU render: SAH, LBVH, PLOC |
|---|---|---|---|---|
| bunny | 118 % / 111 % | 24.5, 27.8, 28.8 | 1.85, 2.00, 1.97 | 787, 826, 848 ms |
| dragon | 122 % / 111 % | 30.9, 36.9, 34.9 | 2.21, 2.38, 2.29 | 1,054, 1,150, 1,136 ms |
| buddha | 128 % / 110 % | 30.5, 37.2, 34.0 | 2.21, 2.38, 2.27 | 972, 1,151, 1,099 ms |

* PLOC wins back about half of the LBVH's quality loss: 10–11 % above SAH instead of 18–28 %,
  with 5–9 % fewer box tests per ray than the LBVH on the big models. On the procedural test
  scene, PLOC's tree is even **11 % better than the binned SAH builder's**.
* The 1-thread CPU render is the steadiest measure of tree quality (the GPU kernel times in this
  run vary by ±25 % with background load): on the Buddha the LBVH tree renders 18 % slower than
  the SAH tree, the PLOC tree 13 % slower.
* The SAH-based leaf collapse matters: the earlier ≤ 4-triangle rule gave the LBVH 3.45
  triangle tests per ray on the dragon; the SAH rule gives 2.38.

**PLOC radius** (`ploc_radius.png`, dragon): r = 1 gives 117 % of SAH in 8.1 ms; r = 2–16 gives
110–112 % and costs 9.5–12.9 ms; larger radii only add build time (17.9 ms at r = 64) and
quality no longer improves. r = 8 (the default) is a good trade-off.

![Build speed vs quality](results/plots/pareto.png)

**Time to image** (`time_to_image.png`, dragon): build + transfers + render as a function of
the rays traced with one tree, from a linear fit to the AO sweep (0–64 samples at 1080p). The
GPU trees win until roughly **8–12 G rays per build (600–1,000 frames at 1080p, 4 AO)**; only
then does the SAH tree's faster rendering pay for its 800 ms build. Between the two GPU
builders the gap in render speed is within the timing noise, so their crossover (≈ 22 frames)
is not reliable.

**Dynamic scene** (`dynamic_refit_vs_rebuild.png`, `explode.gif`): the dragon explodes over 60
frames (1280×720, 4 AO); every triangle flies off in its own random direction.

| Per frame (update + render) | Mean time | Frame rate | SAH cost, frame 0 → last |
|---|---|---|---|
| CPU SAH rebuild + upload | 1,085 ms | 0.9 fps | 57 → 169 |
| GPU LBVH rebuild | 18.2 ms | 55 fps | 70 → 180 |
| GPU PLOC rebuild | 26.6 ms | 38 fps | 64 → 184 |
| GPU refit of the frame-0 LBVH | 15.5 s | 0.06 fps | 70 → 80,663 |

Refitting is the cheapest update (1.1 ms), but under incoherent motion the old topology's boxes
grow to span the whole explosion: the SAH cost rises over 1,000-fold and the last frame takes
41 s to render. Rebuilding every frame on the GPU keeps the tree close to SAH quality at
38–55 fps. Once the pieces are scattered, PLOC's advantage over the LBVH disappears (184 vs 180),
so for this scene the faster LBVH is the better choice.

**The trade-off:**
* **GPU LBVH** builds **~165× faster than the CPU SAH builder** (44–71× faster than the same
  algorithm on one core); its tree has **18–28 % higher SAH cost** and renders up to ~18 % slower.
* **GPU PLOC** builds **60–71× faster than the CPU SAH builder** on the big models, about 2.5×
  slower than the GPU LBVH, and halves the quality loss (**10–11 % above SAH**).
* Both GPU trees beat the SAH tree on total time to image below ~600 frames per build, and only
  a GPU rebuild keeps a fully dynamic scene interactive.

Lucy (28M triangles) was not available (only on the Stanford server, unreachable from here), so
the Buddha is the largest model.

### References

* T. Karras. *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees.*
  HPG 2012.
* T. Karras. *Thinking Parallel, Part III: Tree Construction on the GPU.* NVIDIA Developer Blog,
  2012.
* D. Meister, J. Bittner. *Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy
  Construction.* IEEE TVCG 2018.
* ToruNiina/lbvh (MIT), an LBVH implementation named in the plan as a read-only reference. The
  kernels here were written from the papers; no code was taken from it.

## Repository layout

```
include/   vecmath.h (shared math, named so it never shadows <math.h>), trace.h (shared
           traversal + shading + optional traversal counters), bvh.h, mesh.h, camera.h, rng.h,
           ply_loader.h, render.h, builders.h (builder API, tree stats), morton.h (Morton
           codes, delta, Karras split), bvh_common.h (binary tree + converter), animate.h
src/       main.cpp (CLI), ply_loader.cpp, bvh_build.cpp (CPU SAH), bvh_build_lbvh_cpu.cpp,
           bvh_build_lbvh_gpu.cu (GPU LBVH + refit), bvh_convert_cpu.cpp, builders.cpp,
           animate.cu, render_cpu.cpp, render_gpu.cu (all render kernels + launch code),
           camera.cpp, verify.cpp, image.cpp, mesh.cpp
tests/     unit-test runner (procedural meshes only)
scripts/   build.ps1 [-Stats], download_data.ps1/.sh, run_benchmarks.ps1/.sh, plot_results.py,
           run_build_benchmarks.ps1, plot_build.py
results/   results.csv, build.csv, build_render.csv, traversal.csv, dynamic.csv and plots/
renders/   output images
```

## Limitations and future work

* ~~The BVH is built on the CPU~~: done, see [GPU BVH construction](#gpu-bvh-construction-lbvh-ploc).
  Next: high-quality parallel builders (PLOC, TRBVH) to win back the LBVH's 21–30 % SAH-cost loss.
* Sorting AO rays by direction or octant would make traversal within a warp more coherent.
* Use Nsight Compute / `-DRT_PTXAS_VERBOSE=ON` to report register use, occupancy and local-memory
  traffic for the traversal stack.

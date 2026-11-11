// Shared GPU build stages: bounds + hand-written reduction, Morton codes, CUB sort, and the
// collapse + depth-first flatten converter (see gpu_build_common.cuh).
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>

#include "builders.h"
#include "gpu_build_common.cuh"

void DeviceBvh::release() {
    for (void** p : {&nodes, &tris, &triOrder}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
    numNodes = numTris = 0;
}

namespace gpub {
namespace {

// ---------------------------------------------------------------- 1. bounds

__global__ void k_tri_bounds(const vec3* __restrict__ verts, const Face* __restrict__ faces, int n,
                             AABB* __restrict__ boxes, vec3* __restrict__ cents) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Face f = faces[i];
    AABB b = aabb_empty();
    aabb_grow(b, verts[f.a]);
    aabb_grow(b, verts[f.b]);
    aabb_grow(b, verts[f.c]);
    boxes[i] = b;
    cents[i] = (b.bmin + b.bmax) * 0.5f;
}

__device__ __forceinline__ void warp_reduce_box(AABB& b) {
    for (int o = 16; o > 0; o >>= 1) {
        b.bmin.x = fminf(b.bmin.x, __shfl_down_sync(0xffffffffu, b.bmin.x, o));
        b.bmin.y = fminf(b.bmin.y, __shfl_down_sync(0xffffffffu, b.bmin.y, o));
        b.bmin.z = fminf(b.bmin.z, __shfl_down_sync(0xffffffffu, b.bmin.z, o));
        b.bmax.x = fmaxf(b.bmax.x, __shfl_down_sync(0xffffffffu, b.bmax.x, o));
        b.bmax.y = fmaxf(b.bmax.y, __shfl_down_sync(0xffffffffu, b.bmax.y, o));
        b.bmax.z = fmaxf(b.bmax.z, __shfl_down_sync(0xffffffffu, b.bmax.z, o));
    }
}

// Block-wide min/max: warp shuffles, then one value per warp through shared memory.
__device__ AABB block_reduce_box(AABB b) {
    __shared__ AABB warpBox[BLOCK / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    warp_reduce_box(b);
    if (lane == 0) warpBox[warp] = b;
    __syncthreads();
    b = threadIdx.x < BLOCK / 32 ? warpBox[threadIdx.x] : aabb_empty();
    if (warp == 0) warp_reduce_box(b);
    return b;  // valid in thread 0
}

// Pass 1: grid-stride over centroids, one partial box per block.
__global__ void k_reduce_centroids(const vec3* __restrict__ cents, int n, AABB* __restrict__ partial) {
    AABB b = aabb_empty();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) aabb_grow(b, cents[i]);
    b = block_reduce_box(b);
    if (threadIdx.x == 0) partial[blockIdx.x] = b;
}

// Pass 2: one block folds the partials and writes the Morton frame.
__global__ void k_reduce_partials(const AABB* __restrict__ partial, int count, MortonFrame* __restrict__ frame) {
    AABB b = aabb_empty();
    for (int i = threadIdx.x; i < count; i += blockDim.x) aabb_grow(b, partial[i]);
    b = block_reduce_box(b);
    if (threadIdx.x == 0) *frame = make_morton_frame(b);
}

// ---------------------------------------------------------------- 2. Morton codes

__global__ void k_morton(const vec3* __restrict__ cents, int n, const MortonFrame* __restrict__ frame,
                         uint64_t* __restrict__ keys, uint32_t* __restrict__ vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = morton_code(*frame, cents[i]);
    vals[i] = uint32_t(i);
}

// ---------------------------------------------------------------- convert

__global__ void k_convert_nodes(DeviceTree tree, const float4* __restrict__ boxLo, const float4* __restrict__ boxHi,
                                BVHNode* __restrict__ out, int* __restrict__ maxDepth,
                                int* __restrict__ leafCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= 2 * tree.n - 1) return;
    const BinaryTreeView t = tree.view();
    if (!conv_is_kept(t, x)) return;
    uint32_t idx, off;
    int depth;
    conv_offsets(t, x, idx, off, depth);
    float4 a = boxLo[x], b = boxHi[x];
    AABB box{{a.x, a.y, a.z}, {b.x, b.y, b.z}};
    out[idx] = conv_make_node(t, x, idx, off, box);
    atomicMax(maxDepth, depth);
    if (t.dfsCount[x] == 1u) atomicAdd(leafCount, 1);
}

__global__ void k_place_tris(DeviceTree tree, const uint32_t* __restrict__ order, const vec3* __restrict__ verts,
                             const Face* __restrict__ faces, Tri* __restrict__ tris, uint32_t* __restrict__ triOrder) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= tree.n) return;
    const BinaryTreeView t = tree.view();
    uint32_t idx, pos;
    int depth;
    conv_offsets(t, lbvh_leaf_id(tree.n, k), idx, pos, depth);
    const uint32_t f = order[k];
    const Face fc = faces[f];
    tris[pos] = make_tri(verts[fc.a], verts[fc.b], verts[fc.c], f);
    triOrder[pos] = f;
}

// ---------------------------------------------------------------- scan

// Exclusive scan of one block (one element per thread): warp-level inclusive scan with
// shuffles, then the warp totals are scanned by warp 0 through shared memory.
__global__ void k_scan_block(const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int n,
                             uint64_t* __restrict__ blockSums) {
    __shared__ uint64_t warpSums[BLOCK / 32];
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint64_t v = i < n ? in[i] : 0;
    uint64_t x = v;
    for (int o = 1; o < 32; o <<= 1) {
        uint64_t y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warpSums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint64_t s = lane < BLOCK / 32 ? warpSums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            uint64_t y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        if (lane < BLOCK / 32) warpSums[lane] = s;
    }
    __syncthreads();
    const uint64_t incl = x + (warp > 0 ? warpSums[warp - 1] : 0);
    if (i < n) out[i] = incl - v;
    if (threadIdx.x == BLOCK - 1) blockSums[blockIdx.x] = incl;
}

__global__ void k_scan_add(uint64_t* __restrict__ out, int n, const uint64_t* __restrict__ scannedSums) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) out[i] += scannedSums[blockIdx.x];
}

}  // namespace

cudaError_t scan_exclusive_u64(const uint64_t* in, uint64_t* out, int n, ScanTemp& tmp, int level) {
    if (n <= 0) return cudaSuccess;
    const int blocks = blocks_for(n);
    while (int(tmp.sums.size()) <= level) {
        tmp.sums.push_back(new Buf);
        tmp.scanned.push_back(new Buf);
    }
    cudaError_t e;
    if ((e = tmp.sums[level]->ensure(size_t(blocks) * 8)) != cudaSuccess) return e;
    if ((e = tmp.scanned[level]->ensure(size_t(blocks) * 8)) != cudaSuccess) return e;
    uint64_t* sums = tmp.sums[level]->as<uint64_t>();
    uint64_t* scanned = tmp.scanned[level]->as<uint64_t>();
    k_scan_block<<<blocks, BLOCK>>>(in, out, n, sums);
    if (blocks > 1) {
        if ((e = scan_exclusive_u64(sums, scanned, blocks, tmp, level + 1)) != cudaSuccess) return e;
        k_scan_add<<<blocks, BLOCK>>>(out, n, scanned);
    }
    return cudaGetLastError();
}

cudaError_t MortonSort::ensure(int n) {
    cudaError_t e;
    if ((e = boxes.ensure(size_t(n) * sizeof(AABB))) != cudaSuccess) return e;
    if ((e = cents.ensure(size_t(n) * sizeof(vec3))) != cudaSuccess) return e;
    if ((e = partial.ensure(REDUCE_BLOCKS * sizeof(AABB))) != cudaSuccess) return e;
    if ((e = frame.ensure(sizeof(MortonFrame))) != cudaSuccess) return e;
    if ((e = keys.ensure(size_t(n) * 8)) != cudaSuccess) return e;
    if ((e = keysOut.ensure(size_t(n) * 8)) != cudaSuccess) return e;
    if ((e = vals.ensure(size_t(n) * 4)) != cudaSuccess) return e;
    if ((e = valsOut.ensure(size_t(n) * 4)) != cudaSuccess) return e;
    size_t sortBytes = 0;
    e = cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, keys.as<uint64_t>(), keysOut.as<uint64_t>(),
                                        vals.as<uint32_t>(), valsOut.as<uint32_t>(), n, 0, 63);
    if (e != cudaSuccess) return e;
    return sortTemp.ensure(sortBytes);
}

cudaError_t MortonSort::run(const vec3* verts, const Face* faces, int n, Events& ev) {
    cudaError_t e = ensure(n);
    if (e != cudaSuccess) return e;
    k_tri_bounds<<<blocks_for(n), BLOCK>>>(verts, faces, n, boxes.as<AABB>(), cents.as<vec3>());
    const int rb = std::min(REDUCE_BLOCKS, blocks_for(n));
    k_reduce_centroids<<<rb, BLOCK>>>(cents.as<vec3>(), n, partial.as<AABB>());
    k_reduce_partials<<<1, BLOCK>>>(partial.as<AABB>(), rb, frame.as<MortonFrame>());
    ev.mark();
    k_morton<<<blocks_for(n), BLOCK>>>(cents.as<vec3>(), n, frame.as<MortonFrame>(), keys.as<uint64_t>(),
                                       vals.as<uint32_t>());
    ev.mark();
    size_t sortBytes = sortTemp.bytes;
    e = cub::DeviceRadixSort::SortPairs(sortTemp.p, sortBytes, keys.as<uint64_t>(), keysOut.as<uint64_t>(),
                                        vals.as<uint32_t>(), valsOut.as<uint32_t>(), n, 0, 63);
    ev.mark();
    return e;
}

cudaError_t convert_tree(const DeviceTree& t, const float4* boxLo, const float4* boxHi, const uint32_t* order, const vec3* verts, const Face* faces, BVHNode* outNodes, Tri* outTris,
                         uint32_t* outOrder, int* counters) {
    k_convert_nodes<<<blocks_for(2 * t.n - 1), BLOCK>>>(t, boxLo, boxHi, outNodes, counters, counters + 1);
    k_place_tris<<<blocks_for(t.n), BLOCK>>>(t, order, verts, faces, outTris, outOrder);
    return cudaGetLastError();
}

}  // namespace gpub

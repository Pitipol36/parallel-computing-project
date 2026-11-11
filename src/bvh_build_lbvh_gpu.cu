// GPU LBVH builder (Karras 2012) in hand-written kernels plus CUB's radix sort.
//   bounds     one thread per triangle: AABB + centroid; hand-written min/max reduction
//   morton     63-bit codes (21 bits per axis)
//   sort       cub::DeviceRadixSort::SortPairs on (code, triangle index); stable, so equal
//              codes keep index order exactly like the CPU builder
//   hierarchy  one thread per internal node (shared lbvh_internal_node)
//   fit        one thread per leaf climbs; the second arrival at a node unions its children
//   convert    leaf collapse + depth-first flatten, one thread per node / per triangle
// The result is node-for-node identical to build_lbvh_cpu.
#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <chrono>

#include "builders.h"
#include "bvh_common.h"
#include "morton.h"

void DeviceBvh::release() {
    for (void** p : {&nodes, &tris, &triOrder}) {
        if (*p) cudaFree(*p);
        *p = nullptr;
    }
    numNodes = numTris = 0;
}

namespace {

constexpr int BLOCK = 256;
constexpr int REDUCE_BLOCKS = 240;  // 30 SMs x 8; the second pass reduces these partials

int blocks_for(int n) { return (n + BLOCK - 1) / BLOCK; }

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

// ---------------------------------------------------------------- 4. hierarchy

__global__ void k_hierarchy(const uint64_t* __restrict__ keys, int n, int* __restrict__ left,
                            int* __restrict__ right, int* __restrict__ parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;
    int l, r;
    lbvh_internal_node(keys, n, i, l, r);
    left[i] = l;
    right[i] = r;
    parent[l] = i;
    parent[r] = i;
    if (i == 0) parent[0] = -1;
}

// ---------------------------------------------------------------- 5. bottom-up fit

// Node boxes are stored as two float4 arrays so that they can be read with __ldcg (L2 only):
// a sibling's box was written by another thread, possibly on another SM, whose L1 is not
// coherent with ours.
__device__ __forceinline__ AABB load_box_cg(const float4* lo, const float4* hi, int x) {
    float4 a = __ldcg(lo + x), b = __ldcg(hi + x);
    return {{a.x, a.y, a.z}, {b.x, b.y, b.z}};
}

__global__ void k_fit(const AABB* __restrict__ triBoxes, const uint32_t* __restrict__ order, int n,
                      const int* __restrict__ left, const int* __restrict__ right, const int* __restrict__ parent,
                      int maxLeaf, unsigned int* __restrict__ visited, float4* boxLo, float4* boxHi,
                      uint32_t* triCount, uint32_t* dfsCount) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int x = lbvh_leaf_id(n, k);
    AABB b = triBoxes[order[k]];
    boxLo[x] = make_float4(b.bmin.x, b.bmin.y, b.bmin.z, 0.0f);
    boxHi[x] = make_float4(b.bmax.x, b.bmax.y, b.bmax.z, 0.0f);
    triCount[x] = 1;
    dfsCount[x] = 1;
    for (int p = parent[x]; p >= 0; p = parent[p]) {
        __threadfence();  // publish this subtree before announcing it
        if (atomicAdd(&visited[p], 1u) == 0) return;  // first arrival: the sibling finishes the node
        const int l = left[p], r = right[p];
        AABB bl = load_box_cg(boxLo, boxHi, l), br = load_box_cg(boxLo, boxHi, r);
        aabb_grow(bl, br);
        boxLo[p] = make_float4(bl.bmin.x, bl.bmin.y, bl.bmin.z, 0.0f);
        boxHi[p] = make_float4(bl.bmax.x, bl.bmax.y, bl.bmax.z, 0.0f);
        const uint32_t tl = __ldcg(triCount + l), tr = __ldcg(triCount + r);
        const uint32_t t = tl + tr;
        triCount[p] = t;
        dfsCount[p] = collapsed_count(t, __ldcg(dfsCount + l), __ldcg(dfsCount + r), maxLeaf);
    }
}

// ---------------------------------------------------------------- 6. convert

struct DeviceTree {
    int n, root;
    const int *left, *right, *parent;
    const uint32_t *triCount, *dfsCount;
    __device__ BinaryTreeView view() const {
        BinaryTreeView v;
        v.n = n; v.root = root; v.left = left; v.right = right; v.parent = parent;
        v.triCount = triCount; v.dfsCount = dfsCount;
        return v;
    }
};

__global__ void k_convert_nodes(DeviceTree tree, const float4* __restrict__ boxLo, const float4* __restrict__ boxHi,
                                int maxLeaf, BVHNode* __restrict__ out, int* __restrict__ maxDepth,
                                int* __restrict__ leafCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= 2 * tree.n - 1) return;
    const BinaryTreeView t = tree.view();
    if (!conv_is_kept(t, x, maxLeaf)) return;
    uint32_t idx, off;
    int depth;
    conv_offsets(t, x, idx, off, depth);
    float4 a = boxLo[x], b = boxHi[x];
    AABB box{{a.x, a.y, a.z}, {b.x, b.y, b.z}};
    out[idx] = conv_make_node(t, x, idx, off, box, maxLeaf);
    atomicMax(maxDepth, depth);
    if (t.triCount[x] <= uint32_t(maxLeaf)) atomicAdd(leafCount, 1);
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

// ---------------------------------------------------------------- host side

struct Buf {
    void* p = nullptr;
    ~Buf() { if (p) cudaFree(p); }
    cudaError_t alloc(size_t bytes) { return cudaMalloc(&p, bytes ? bytes : 4); }
    void* take() { void* q = p; p = nullptr; return q; }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

struct Events {
    std::vector<cudaEvent_t> ev;
    ~Events() { for (cudaEvent_t e : ev) cudaEventDestroy(e); }
    cudaEvent_t mark() {
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e);
        ev.push_back(e);
        return e;
    }
};

float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

#define CU(call)                                                                  \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            err = std::string(#call) + " failed: " + cudaGetErrorString(e_);      \
            return false;                                                         \
        }                                                                         \
    } while (0)

}  // namespace

bool build_lbvh_gpu(const Mesh& mesh, const BuildParams& params, BVH* hostOut, DeviceBvh* devOut,
                    BuildReport* report, std::string& err) {
    const int n = int(mesh.faces.size());
    const int maxLeaf = std::max(1, params.maxLeaf);
    if (n < 1) { err = "empty mesh"; return false; }
    if (!hostOut && !devOut) { err = "no output requested"; return false; }
    BuildReport rep;
    const int total = 2 * n - 1;

    // Upload the mesh (timed separately from the build).
    auto tUp = Clock::now();
    Buf dVerts, dFaces;
    CU(dVerts.alloc(mesh.vertices.size() * sizeof(vec3)));
    CU(dFaces.alloc(mesh.faces.size() * sizeof(Face)));
    CU(cudaMemcpy(dVerts.p, mesh.vertices.data(), mesh.vertices.size() * sizeof(vec3), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dFaces.p, mesh.faces.data(), mesh.faces.size() * sizeof(Face), cudaMemcpyHostToDevice));

    // Working buffers.
    Buf dBoxes, dCents, dPartial, dFrame, dKeys, dKeysOut, dVals, dValsOut, dLeft, dRight, dParent, dVisited,
        dBoxLo, dBoxHi, dTriCount, dDfsCount, dCounters, dSortTemp;
    CU(dBoxes.alloc(size_t(n) * sizeof(AABB)));
    CU(dCents.alloc(size_t(n) * sizeof(vec3)));
    CU(dPartial.alloc(REDUCE_BLOCKS * sizeof(AABB)));
    CU(dFrame.alloc(sizeof(MortonFrame)));
    CU(dKeys.alloc(size_t(n) * 8));
    CU(dKeysOut.alloc(size_t(n) * 8));
    CU(dVals.alloc(size_t(n) * 4));
    CU(dValsOut.alloc(size_t(n) * 4));
    CU(dLeft.alloc(size_t(std::max(n - 1, 1)) * 4));
    CU(dRight.alloc(size_t(std::max(n - 1, 1)) * 4));
    CU(dParent.alloc(size_t(total) * 4));
    CU(dVisited.alloc(size_t(std::max(n - 1, 1)) * 4));
    CU(dBoxLo.alloc(size_t(total) * sizeof(float4)));
    CU(dBoxHi.alloc(size_t(total) * sizeof(float4)));
    CU(dTriCount.alloc(size_t(total) * 4));
    CU(dDfsCount.alloc(size_t(total) * 4));
    CU(dCounters.alloc(3 * sizeof(int)));
    size_t sortTempBytes = 0;
    CU(cub::DeviceRadixSort::SortPairs(nullptr, sortTempBytes, dKeys.as<uint64_t>(), dKeysOut.as<uint64_t>(),
                                       dVals.as<uint32_t>(), dValsOut.as<uint32_t>(), n, 0, 63));
    CU(dSortTemp.alloc(sortTempBytes));
    // Outputs: the collapsed tree has at most 2n-1 nodes, so no mid-build sizing sync is needed.
    Buf dNodes, dTris, dOrder;
    CU(dNodes.alloc(size_t(total) * sizeof(BVHNode)));
    CU(dTris.alloc(size_t(n) * sizeof(Tri)));
    CU(dOrder.alloc(size_t(n) * 4));
    CU(cudaMemset(dCounters.p, 0, 3 * sizeof(int)));
    CU(cudaMemset(dVisited.p, 0, size_t(std::max(n - 1, 1)) * 4));
    CU(cudaDeviceSynchronize());
    rep.uploadMs = ms_since(tUp);

    Events ev;
    cudaEvent_t e0 = ev.mark();
    k_tri_bounds<<<blocks_for(n), BLOCK>>>(dVerts.as<vec3>(), dFaces.as<Face>(), n, dBoxes.as<AABB>(), dCents.as<vec3>());
    const int rb = std::min(REDUCE_BLOCKS, blocks_for(n));
    k_reduce_centroids<<<rb, BLOCK>>>(dCents.as<vec3>(), n, dPartial.as<AABB>());
    k_reduce_partials<<<1, BLOCK>>>(dPartial.as<AABB>(), rb, dFrame.as<MortonFrame>());
    cudaEvent_t e1 = ev.mark();
    k_morton<<<blocks_for(n), BLOCK>>>(dCents.as<vec3>(), n, dFrame.as<MortonFrame>(), dKeys.as<uint64_t>(),
                                       dVals.as<uint32_t>());
    cudaEvent_t e2 = ev.mark();
    CU(cub::DeviceRadixSort::SortPairs(dSortTemp.p, sortTempBytes, dKeys.as<uint64_t>(), dKeysOut.as<uint64_t>(),
                                       dVals.as<uint32_t>(), dValsOut.as<uint32_t>(), n, 0, 63));
    cudaEvent_t e3 = ev.mark();
    if (n > 1)
        k_hierarchy<<<blocks_for(n - 1), BLOCK>>>(dKeysOut.as<uint64_t>(), n, dLeft.as<int>(), dRight.as<int>(),
                                                 dParent.as<int>());
    else
        CU(cudaMemsetAsync(dParent.p, 0xff, sizeof(int)));  // parent[0] = -1
    cudaEvent_t e4 = ev.mark();
    k_fit<<<blocks_for(n), BLOCK>>>(dBoxes.as<AABB>(), dValsOut.as<uint32_t>(), n, dLeft.as<int>(), dRight.as<int>(),
                                    dParent.as<int>(), maxLeaf, dVisited.as<unsigned int>(), dBoxLo.as<float4>(),
                                    dBoxHi.as<float4>(), dTriCount.as<uint32_t>(), dDfsCount.as<uint32_t>());
    cudaEvent_t e5 = ev.mark();
    DeviceTree tree{n, 0, dLeft.as<int>(), dRight.as<int>(), dParent.as<int>(), dTriCount.as<uint32_t>(),
                    dDfsCount.as<uint32_t>()};
    k_convert_nodes<<<blocks_for(total), BLOCK>>>(tree, dBoxLo.as<float4>(), dBoxHi.as<float4>(), maxLeaf,
                                                  dNodes.as<BVHNode>(), dCounters.as<int>(), dCounters.as<int>() + 1);
    k_place_tris<<<blocks_for(n), BLOCK>>>(tree, dValsOut.as<uint32_t>(), dVerts.as<vec3>(), dFaces.as<Face>(),
                                           dTris.as<Tri>(), dOrder.as<uint32_t>());
    cudaEvent_t e6 = ev.mark();
    CU(cudaGetLastError());
    CU(cudaEventSynchronize(e6));

    rep.stages = {{"bounds", elapsed(e0, e1)},  {"morton", elapsed(e1, e2)}, {"sort", elapsed(e2, e3)},
                  {"hierarchy", elapsed(e3, e4)}, {"fit", elapsed(e4, e5)},   {"convert", elapsed(e5, e6)}};
    rep.buildMs = elapsed(e0, e6);

    int counters[2] = {0, 0};
    uint32_t numNodes = 0;  // the root's collapsed count (root = node 0)
    CU(cudaMemcpy(counters, dCounters.p, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&numNodes, dDfsCount.p, 4, cudaMemcpyDeviceToHost));

    if (hostOut) {
        auto tDown = Clock::now();
        BVH& b = *hostOut;
        b = BVH();
        b.nodes.resize(numNodes);
        b.tris.resize(n);
        b.triOrder.resize(n);
        CU(cudaMemcpy(b.nodes.data(), dNodes.p, size_t(numNodes) * sizeof(BVHNode), cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(b.tris.data(), dTris.p, size_t(n) * sizeof(Tri), cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(b.triOrder.data(), dOrder.p, size_t(n) * 4, cudaMemcpyDeviceToHost));
        b.maxDepth = counters[0];
        b.leafCount = counters[1];
        b.buildMs = rep.buildMs;
        rep.downloadMs = ms_since(tDown);
    }
    if (devOut) {
        devOut->release();
        devOut->numNodes = numNodes;
        devOut->numTris = uint32_t(n);
        devOut->maxDepth = counters[0];
        devOut->leafCount = counters[1];
        devOut->nodes = dNodes.take();
        devOut->tris = dTris.take();
        devOut->triOrder = dOrder.take();
    }
    if (report) *report = rep;
    return true;
}

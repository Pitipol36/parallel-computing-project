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

// ---------------------------------------------------------------- refit

// Parent of every node of a flattened tree (left child = i + 1, right = leftOrFirst).
__global__ void k_flat_parents(const BVHNode* __restrict__ nodes, int numNodes, int* __restrict__ parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;
    if (i == 0) parent[0] = -1;
    const BVHNode nd = nodes[i];
    if (nd.count == 0) {
        parent[i + 1] = i;
        parent[nd.leftOrFirst] = i;
    }
}

// Refit: each leaf recomputes its triangles and box from the current vertices, then climbs;
// the second child to arrive at a node writes the union (same pattern as k_fit). The
// topology words (.w) are never changed.
__global__ void k_refit(float4* nodes, int numNodes, const int* __restrict__ parent, unsigned int* __restrict__ visited,
                        const vec3* __restrict__ verts, const Face* __restrict__ faces,
                        const uint32_t* __restrict__ triOrder, Tri* __restrict__ tris) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;
    const float4 lo = nodes[2 * i], hi = nodes[2 * i + 1];
    const uint32_t count = __float_as_uint(hi.w);
    if (count == 0) return;  // interior nodes are written by the climbing leaves
    const uint32_t first = __float_as_uint(lo.w);
    AABB b = aabb_empty();
    for (uint32_t pos = first; pos < first + count; ++pos) {
        const uint32_t f = triOrder[pos];
        const Face fc = faces[f];
        const vec3 v0 = verts[fc.a], v1 = verts[fc.b], v2 = verts[fc.c];
        aabb_grow(b, v0);
        aabb_grow(b, v1);
        aabb_grow(b, v2);
        tris[pos] = make_tri(v0, v1, v2, f);
    }
    nodes[2 * i] = make_float4(b.bmin.x, b.bmin.y, b.bmin.z, lo.w);
    nodes[2 * i + 1] = make_float4(b.bmax.x, b.bmax.y, b.bmax.z, hi.w);
    for (int p = parent[i]; p >= 0; p = parent[p]) {
        __threadfence();
        if (atomicAdd(&visited[p], 1u) == 0) return;
        const float4 plo = nodes[2 * p], phi = nodes[2 * p + 1];  // .w only: topology, never written
        const int l = p + 1, r = int(__float_as_uint(plo.w));
        const float4 llo = __ldcg(nodes + 2 * l), lhi = __ldcg(nodes + 2 * l + 1);
        const float4 rlo = __ldcg(nodes + 2 * r), rhi = __ldcg(nodes + 2 * r + 1);
        nodes[2 * p] = make_float4(fminf(llo.x, rlo.x), fminf(llo.y, rlo.y), fminf(llo.z, rlo.z), plo.w);
        nodes[2 * p + 1] = make_float4(fmaxf(lhi.x, rhi.x), fmaxf(lhi.y, rhi.y), fmaxf(lhi.z, rhi.z), phi.w);
    }
}

// ---------------------------------------------------------------- host side

struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    ~Buf() { release(); }
    void release() { if (p) cudaFree(p); p = nullptr; bytes = 0; }
    // Grows (never shrinks) to at least `need` bytes.
    cudaError_t ensure(size_t need) {
        need = need ? need : 4;
        if (p && bytes >= need) return cudaSuccess;
        release();
        bytes = need;
        return cudaMalloc(&p, need);
    }
    void* take() { void* q = p; p = nullptr; bytes = 0; return q; }
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

struct GpuLbvhBuilder::Impl {
    Buf boxes, cents, partial, frame, keys, keysOut, vals, valsOut, left, right, parent, visited, boxLo, boxHi,
        triCount, dfsCount, counters, sortTemp;
    Buf nodes, tris, order;           // outputs
    Buf flatParent, flatVisited;      // refit
    bool flatParentsValid = false;
    DeviceBvh tree;                   // view onto nodes/tris/order
    int maxLeaf = 4;

    bool ensure(int n, std::string& err) {
        const size_t total = size_t(2 * n - 1), inner = size_t(std::max(n - 1, 1));
        CU(boxes.ensure(size_t(n) * sizeof(AABB)));
        CU(cents.ensure(size_t(n) * sizeof(vec3)));
        CU(partial.ensure(REDUCE_BLOCKS * sizeof(AABB)));
        CU(frame.ensure(sizeof(MortonFrame)));
        CU(keys.ensure(size_t(n) * 8));
        CU(keysOut.ensure(size_t(n) * 8));
        CU(vals.ensure(size_t(n) * 4));
        CU(valsOut.ensure(size_t(n) * 4));
        CU(left.ensure(inner * 4));
        CU(right.ensure(inner * 4));
        CU(parent.ensure(total * 4));
        CU(visited.ensure(inner * 4));
        CU(boxLo.ensure(total * sizeof(float4)));
        CU(boxHi.ensure(total * sizeof(float4)));
        CU(triCount.ensure(total * 4));
        CU(dfsCount.ensure(total * 4));
        CU(counters.ensure(4 * sizeof(int)));
        size_t sortBytes = 0;
        CU(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, keys.as<uint64_t>(), keysOut.as<uint64_t>(),
                                           vals.as<uint32_t>(), valsOut.as<uint32_t>(), n, 0, 63));
        CU(sortTemp.ensure(sortBytes));
        // The collapsed tree has at most 2n-1 nodes, so no mid-build sizing sync is needed.
        CU(nodes.ensure(total * sizeof(BVHNode)));
        CU(tris.ensure(size_t(n) * sizeof(Tri)));
        CU(order.ensure(size_t(n) * 4));
        return true;
    }
};

GpuLbvhBuilder::GpuLbvhBuilder() : impl_(new Impl) {}
GpuLbvhBuilder::~GpuLbvhBuilder() { delete impl_; }
const DeviceBvh& GpuLbvhBuilder::tree() const { return impl_->tree; }
bool GpuLbvhBuilder::reserve(int numTris, std::string& err) { return impl_->ensure(std::max(numTris, 1), err); }

DeviceBvh GpuLbvhBuilder::take_tree() {
    DeviceBvh t = impl_->tree;
    t.nodes = impl_->nodes.take();
    t.tris = impl_->tris.take();
    t.triOrder = impl_->order.take();
    impl_->tree = DeviceBvh();
    impl_->flatParentsValid = false;
    return t;
}

bool GpuLbvhBuilder::build(const void* dVertsV, const void* dFacesV, int n, const BuildParams& params,
                           BuildReport* report, std::string& err) {
    if (n < 1) { err = "empty mesh"; return false; }
    Impl& I = *impl_;
    const vec3* dVerts = static_cast<const vec3*>(dVertsV);
    const Face* dFaces = static_cast<const Face*>(dFacesV);
    const int maxLeaf = std::max(1, params.maxLeaf);
    const int total = 2 * n - 1;
    if (!I.ensure(n, err)) return false;
    size_t sortBytes = I.sortTemp.bytes;
    CU(cudaMemsetAsync(I.counters.p, 0, 4 * sizeof(int)));
    CU(cudaMemsetAsync(I.visited.p, 0, size_t(std::max(n - 1, 1)) * 4));
    I.flatParentsValid = false;
    I.maxLeaf = maxLeaf;

    Events ev;
    cudaEvent_t e0 = ev.mark();
    k_tri_bounds<<<blocks_for(n), BLOCK>>>(dVerts, dFaces, n, I.boxes.as<AABB>(), I.cents.as<vec3>());
    const int rb = std::min(REDUCE_BLOCKS, blocks_for(n));
    k_reduce_centroids<<<rb, BLOCK>>>(I.cents.as<vec3>(), n, I.partial.as<AABB>());
    k_reduce_partials<<<1, BLOCK>>>(I.partial.as<AABB>(), rb, I.frame.as<MortonFrame>());
    cudaEvent_t e1 = ev.mark();
    k_morton<<<blocks_for(n), BLOCK>>>(I.cents.as<vec3>(), n, I.frame.as<MortonFrame>(), I.keys.as<uint64_t>(),
                                       I.vals.as<uint32_t>());
    cudaEvent_t e2 = ev.mark();
    CU(cub::DeviceRadixSort::SortPairs(I.sortTemp.p, sortBytes, I.keys.as<uint64_t>(), I.keysOut.as<uint64_t>(),
                                       I.vals.as<uint32_t>(), I.valsOut.as<uint32_t>(), n, 0, 63));
    cudaEvent_t e3 = ev.mark();
    if (n > 1)
        k_hierarchy<<<blocks_for(n - 1), BLOCK>>>(I.keysOut.as<uint64_t>(), n, I.left.as<int>(), I.right.as<int>(),
                                                 I.parent.as<int>());
    else
        CU(cudaMemsetAsync(I.parent.p, 0xff, sizeof(int)));  // parent[0] = -1
    cudaEvent_t e4 = ev.mark();
    k_fit<<<blocks_for(n), BLOCK>>>(I.boxes.as<AABB>(), I.valsOut.as<uint32_t>(), n, I.left.as<int>(),
                                    I.right.as<int>(), I.parent.as<int>(), maxLeaf, I.visited.as<unsigned int>(),
                                    I.boxLo.as<float4>(), I.boxHi.as<float4>(), I.triCount.as<uint32_t>(),
                                    I.dfsCount.as<uint32_t>());
    cudaEvent_t e5 = ev.mark();
    DeviceTree tree{n, 0, I.left.as<int>(), I.right.as<int>(), I.parent.as<int>(), I.triCount.as<uint32_t>(),
                    I.dfsCount.as<uint32_t>()};
    k_convert_nodes<<<blocks_for(total), BLOCK>>>(tree, I.boxLo.as<float4>(), I.boxHi.as<float4>(), maxLeaf,
                                                  I.nodes.as<BVHNode>(), I.counters.as<int>(), I.counters.as<int>() + 1);
    k_place_tris<<<blocks_for(n), BLOCK>>>(tree, I.valsOut.as<uint32_t>(), dVerts, dFaces, I.tris.as<Tri>(),
                                           I.order.as<uint32_t>());
    cudaEvent_t e6 = ev.mark();
    CU(cudaGetLastError());
    CU(cudaEventSynchronize(e6));

    BuildReport rep;
    rep.stages = {{"bounds", elapsed(e0, e1)},  {"morton", elapsed(e1, e2)}, {"sort", elapsed(e2, e3)},
                  {"hierarchy", elapsed(e3, e4)}, {"fit", elapsed(e4, e5)},   {"convert", elapsed(e5, e6)}};
    rep.buildMs = elapsed(e0, e6);

    int counters[2] = {0, 0};
    uint32_t numNodes = 0;  // the root's collapsed count (root = node 0)
    CU(cudaMemcpy(counters, I.counters.p, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&numNodes, I.dfsCount.p, 4, cudaMemcpyDeviceToHost));
    I.tree.nodes = I.nodes.p;
    I.tree.tris = I.tris.p;
    I.tree.triOrder = I.order.p;
    I.tree.numNodes = numNodes;
    I.tree.numTris = uint32_t(n);
    I.tree.maxDepth = counters[0];
    I.tree.leafCount = counters[1];
    if (report) *report = rep;
    return true;
}

bool GpuLbvhBuilder::refit(const void* dVerts, const void* dFaces, double& ms, std::string& err) {
    Impl& I = *impl_;
    const int numNodes = int(I.tree.numNodes);
    if (!I.tree.nodes || numNodes == 0) { err = "refit before build"; return false; }
    Events ev;
    cudaEvent_t e0 = ev.mark();
    if (!I.flatParentsValid) {  // once per topology; counted in the first refit's time
        CU(I.flatParent.ensure(size_t(numNodes) * 4));
        CU(I.flatVisited.ensure(size_t(numNodes) * 4));
        k_flat_parents<<<blocks_for(numNodes), BLOCK>>>(static_cast<const BVHNode*>(I.tree.nodes), numNodes,
                                                         I.flatParent.as<int>());
        I.flatParentsValid = true;
    }
    CU(cudaMemsetAsync(I.flatVisited.p, 0, size_t(numNodes) * 4));
    k_refit<<<blocks_for(numNodes), BLOCK>>>(static_cast<float4*>(I.tree.nodes), numNodes, I.flatParent.as<int>(),
                                             I.flatVisited.as<unsigned int>(), static_cast<const vec3*>(dVerts),
                                             static_cast<const Face*>(dFaces),
                                             static_cast<const uint32_t*>(I.tree.triOrder), static_cast<Tri*>(I.tree.tris));
    cudaEvent_t e1 = ev.mark();
    CU(cudaGetLastError());
    CU(cudaEventSynchronize(e1));
    ms = elapsed(e0, e1);
    return true;
}

bool GpuLbvhBuilder::download(BVH& b, std::string& err) const {
    const DeviceBvh& t = impl_->tree;
    if (!t.nodes) { err = "no tree"; return false; }
    b = BVH();
    b.nodes.resize(t.numNodes);
    b.tris.resize(t.numTris);
    b.triOrder.resize(t.numTris);
    CU(cudaMemcpy(b.nodes.data(), t.nodes, size_t(t.numNodes) * sizeof(BVHNode), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.tris.data(), t.tris, size_t(t.numTris) * sizeof(Tri), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(b.triOrder.data(), t.triOrder, size_t(t.numTris) * 4, cudaMemcpyDeviceToHost));
    b.maxDepth = t.maxDepth;
    b.leafCount = t.leafCount;
    return true;
}

bool build_lbvh_gpu(const Mesh& mesh, const BuildParams& params, BVH* hostOut, DeviceBvh* devOut,
                    BuildReport* report, std::string& err) {
    const int n = int(mesh.faces.size());
    if (n < 1) { err = "empty mesh"; return false; }
    if (!hostOut && !devOut) { err = "no output requested"; return false; }

    // Upload the mesh and allocate the builder's buffers (timed separately from the build).
    auto tUp = Clock::now();
    Buf dVerts, dFaces;
    CU(dVerts.ensure(mesh.vertices.size() * sizeof(vec3)));
    CU(dFaces.ensure(mesh.faces.size() * sizeof(Face)));
    CU(cudaMemcpy(dVerts.p, mesh.vertices.data(), mesh.vertices.size() * sizeof(vec3), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dFaces.p, mesh.faces.data(), mesh.faces.size() * sizeof(Face), cudaMemcpyHostToDevice));
    GpuLbvhBuilder builder;
    if (!builder.reserve(n, err)) return false;
    CU(cudaDeviceSynchronize());
    const double uploadMs = ms_since(tUp);

    BuildReport rep;
    if (!builder.build(dVerts.p, dFaces.p, n, params, &rep, err)) return false;
    rep.uploadMs = uploadMs;
    if (hostOut) {
        auto tDown = Clock::now();
        if (!builder.download(*hostOut, err)) return false;
        hostOut->buildMs = rep.buildMs;
        rep.downloadMs = ms_since(tDown);
    }
    if (devOut) {
        devOut->release();
        *devOut = builder.take_tree();
    }
    if (report) *report = rep;
    return true;
}

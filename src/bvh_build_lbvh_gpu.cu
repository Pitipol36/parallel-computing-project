// GPU LBVH builder (Karras 2012). Shared stages (bounds, Morton, CUB sort, collapse +
// flatten) live in gpu_build_common.cu; this file adds the LBVH-specific parts:
//   hierarchy  one thread per internal node (shared lbvh_internal_node)
//   fit        one thread per leaf climbs; the second arrival at a node unions its children
//   refit      the same climb on a finished flattened tree, for moving geometry
//   trbvh      optional treelet restructuring passes (Karras & Aila 2013), one thread per
//              treelet, followed by a re-fit
// The result is node-for-node identical to build_lbvh_cpu.
#include <algorithm>

#include "builders.h"
#include "gpu_build_common.cuh"
#include "trbvh.h"

using namespace gpub;

namespace {
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
                      int maxLeaf, bool sah, unsigned int* __restrict__ visited, float4* boxLo, float4* boxHi,
                      uint32_t* triCount, uint32_t* dfsCount, float* cost) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int x = lbvh_leaf_id(n, k);
    AABB b = triBoxes[order[k]];
    boxLo[x] = make_float4(b.bmin.x, b.bmin.y, b.bmin.z, 0.0f);
    boxHi[x] = make_float4(b.bmax.x, b.bmax.y, b.bmax.z, 0.0f);
    triCount[x] = 1;
    dfsCount[x] = 1;
    cost[x] = leaf_cost(1, box_area_rn(b.bmin, b.bmax));
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
        float c;
        dfsCount[p] = collapse_node(t, box_area_rn(bl.bmin, bl.bmax), __ldcg(cost + l), __ldcg(cost + r),
                                    __ldcg(dfsCount + l), __ldcg(dfsCount + r), maxLeaf, sah, c);
        cost[p] = c;
    }
}

// ---------------------------------------------------------------- TRBVH

// One restructuring pass: every leaf climbs; the second thread to reach a node with at least
// gamma triangles restructures the treelet rooted there. Both subtrees are final by then, and
// no other thread touches them again in this pass.
__global__ void k_trbvh_pass(TreeArrays t, uint32_t gamma, int treelet, int maxLeaf, bool sah,
                             unsigned int* __restrict__ visited) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= t.n) return;
    for (int p = ldcg_i(t.parent + lbvh_leaf_id(t.n, k)); p >= 0; p = ldcg_i(t.parent + p)) {
        __threadfence();  // publish this subtree (including any restructuring) before announcing it
        if (atomicAdd(&visited[p], 1u) == 0) return;
        if (ldcg_u(t.triCount + p) >= gamma) trbvh_restructure(t, p, treelet, maxLeaf, sah);
    }
}

// Warp-cooperative pass (Karras & Aila): the climb is the same, but whenever lanes of a warp
// reach treelet roots, the whole warp processes those treelets one at a time. Lane 0 grows the
// treelet; all 32 lanes compute subset areas, then solve the subsets of each size in parallel
// (tables in shared memory); lane 0 rewires. Lanes never leave the loop early, so the
// warp-wide shuffles and ballots always see the full warp.
constexpr int TRBVH_WARPS = BLOCK / 32;
constexpr unsigned FULL_MASK = 0xffffffffu;

__global__ void k_trbvh_pass_warp(TreeArrays t, uint32_t gamma, int treelet, int maxLeaf, bool sah,
                                  unsigned int* __restrict__ visited) {
    __shared__ Treelet sTl[TRBVH_WARPS];
    __shared__ float sArea[TRBVH_WARPS][1 << TRBVH_MAX_LEAVES], sCost[TRBVH_WARPS][1 << TRBVH_MAX_LEAVES];
    __shared__ uint32_t sTris[TRBVH_WARPS][1 << TRBVH_MAX_LEAVES];
    __shared__ uint8_t sPart[TRBVH_WARPS][1 << TRBVH_MAX_LEAVES];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    Treelet& tl = sTl[warp];
    float* area = sArea[warp];
    float* cost = sCost[warp];
    uint32_t* tris = sTris[warp];
    uint8_t* part = sPart[warp];

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = k < t.n;
    int x = active ? lbvh_leaf_id(t.n, k) : 0;
    while (__any_sync(FULL_MASK, active)) {
        int p = -1;
        bool work = false;
        if (active) {
            p = ldcg_i(t.parent + x);
            if (p < 0) {
                active = false;
            } else {
                __threadfence();
                if (atomicAdd(&visited[p], 1u) == 0) active = false;
                else work = ldcg_u(t.triCount + p) >= gamma;
            }
        }
        unsigned m = __ballot_sync(FULL_MASK, work);
        while (m) {
            const int leader = __ffs(int(m)) - 1;
            m &= m - 1u;
            const int root = __shfl_sync(FULL_MASK, p, leader);
            if (lane == 0) trbvh_form(t, root, treelet, tl);
            __syncwarp();
            const uint32_t full = (1u << tl.nl) - 1u;
            for (uint32_t S = lane + 1; S <= full; S += 32) trbvh_subset_init(tl, S, area, tris, cost, part);
            __syncwarp();
            for (int size = 2; size <= tl.nl; ++size) {
                for (uint32_t S = lane + 1; S <= full; S += 32)
                    if (popc32(S) == size) trbvh_subset_solve(S, area, tris, cost, part, maxLeaf, sah);
                __syncwarp();
            }
            if (lane == 0) trbvh_rewire(t, tl, tris, cost, part);
            __syncwarp();
        }
        __threadfence();  // publish the rewired treelets before anyone climbs past them
        if (active) x = p;
    }
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

}  // namespace

struct GpuLbvhBuilder::Impl {
    MortonSort ms;
    Buf left, right, parent, visited, boxLo, boxHi, triCount, dfsCount, cost, counters;
    Buf nodes, tris, order;           // outputs
    Buf flatParent, flatVisited;      // refit
    bool flatParentsValid = false;
    DeviceBvh tree;                   // view onto nodes/tris/order
    int maxLeaf = 4;

    bool ensure(int n, std::string& err) {
        const size_t total = size_t(2 * n - 1), inner = size_t(std::max(n - 1, 1));
        CU(ms.ensure(n));
        CU(left.ensure(inner * 4));
        CU(right.ensure(inner * 4));
        CU(parent.ensure(total * 4));
        CU(visited.ensure(inner * 4));
        CU(boxLo.ensure(total * sizeof(float4)));
        CU(boxHi.ensure(total * sizeof(float4)));
        CU(triCount.ensure(total * 4));
        CU(dfsCount.ensure(total * 4));
        CU(cost.ensure(total * 4));
        CU(counters.ensure(4 * sizeof(int)));
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
                           BuildReport* report, std::string& err, bool restructure) {
    if (n < 1) { err = "empty mesh"; return false; }
    Impl& I = *impl_;
    const vec3* dVerts = static_cast<const vec3*>(dVertsV);
    const Face* dFaces = static_cast<const Face*>(dFacesV);
    const int maxLeaf = std::max(1, params.maxLeaf);
    if (!I.ensure(n, err)) return false;
    CU(cudaMemsetAsync(I.counters.p, 0, 4 * sizeof(int)));
    CU(cudaMemsetAsync(I.visited.p, 0, size_t(std::max(n - 1, 1)) * 4));
    I.flatParentsValid = false;
    I.maxLeaf = maxLeaf;

    Events ev;
    cudaEvent_t e0 = ev.mark();
    CU(I.ms.run(dVerts, dFaces, n, ev));  // records 3 events: after bounds, Morton, sort
    if (n > 1)
        k_hierarchy<<<blocks_for(n - 1), BLOCK>>>(I.ms.sortedKeys(), n, I.left.as<int>(), I.right.as<int>(),
                                                 I.parent.as<int>());
    else
        CU(cudaMemsetAsync(I.parent.p, 0xff, sizeof(int)));  // parent[0] = -1
    cudaEvent_t e4 = ev.mark();
    k_fit<<<blocks_for(n), BLOCK>>>(I.ms.triBoxes(), I.ms.order(), n, I.left.as<int>(), I.right.as<int>(),
                                    I.parent.as<int>(), maxLeaf, params.sahCollapse, I.visited.as<unsigned int>(),
                                    I.boxLo.as<float4>(), I.boxHi.as<float4>(), I.triCount.as<uint32_t>(),
                                    I.dfsCount.as<uint32_t>(), I.cost.as<float>());
    cudaEvent_t e5 = ev.mark();
    restructure = restructure && n > 1 && params.trbvhPasses > 0;
    if (restructure) {
        TreeArrays ta{n, I.left.as<int>(), I.right.as<int>(), I.parent.as<int>(), I.boxLo.as<f4>(), I.boxHi.as<f4>(),
                      I.triCount.as<uint32_t>(), I.cost.as<float>()};
        for (int pass = 0; pass < params.trbvhPasses; ++pass) {
            CU(cudaMemsetAsync(I.visited.p, 0, size_t(n - 1) * 4));
            const uint32_t gamma = trbvh_gamma(params.trbvhTreelet, pass);
            if (params.trbvhWarp)
                k_trbvh_pass_warp<<<blocks_for(n), BLOCK>>>(ta, gamma, params.trbvhTreelet, maxLeaf,
                                                            params.sahCollapse, I.visited.as<unsigned int>());
            else
                k_trbvh_pass<<<blocks_for(n), BLOCK>>>(ta, gamma, params.trbvhTreelet, maxLeaf, params.sahCollapse,
                                                       I.visited.as<unsigned int>());
        }
        CU(cudaMemsetAsync(I.visited.p, 0, size_t(n - 1) * 4));
        k_fit<<<blocks_for(n), BLOCK>>>(I.ms.triBoxes(), I.ms.order(), n, I.left.as<int>(), I.right.as<int>(),
                                        I.parent.as<int>(), maxLeaf, params.sahCollapse, I.visited.as<unsigned int>(),
                                        I.boxLo.as<float4>(), I.boxHi.as<float4>(), I.triCount.as<uint32_t>(),
                                        I.dfsCount.as<uint32_t>(), I.cost.as<float>());
    }
    cudaEvent_t e5b = ev.mark();
    DeviceTree tree{n, 0, I.left.as<int>(), I.right.as<int>(), I.parent.as<int>(), I.triCount.as<uint32_t>(),
                    I.dfsCount.as<uint32_t>()};
    CU(convert_tree(tree, I.boxLo.as<float4>(), I.boxHi.as<float4>(), I.ms.order(), dVerts, dFaces,
                    I.nodes.as<BVHNode>(), I.tris.as<Tri>(), I.order.as<uint32_t>(), I.counters.as<int>()));
    cudaEvent_t e6 = ev.mark();
    CU(cudaGetLastError());
    CU(cudaEventSynchronize(e6));

    const cudaEvent_t e1 = ev.ev[1], e2 = ev.ev[2], e3 = ev.ev[3];
    BuildReport rep;
    rep.stages = {{"bounds", elapsed(e0, e1)}, {"morton", elapsed(e1, e2)}, {"sort", elapsed(e2, e3)},
                  {"hierarchy", elapsed(e3, e4)}, {"fit", elapsed(e4, e5)}};
    if (restructure) rep.stages.push_back({"restructure", elapsed(e5, e5b)});
    rep.stages.push_back({"convert", elapsed(e5b, e6)});
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
                    BuildReport* report, std::string& err, bool restructure) {
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
    if (!builder.build(dVerts.p, dFaces.p, n, params, &rep, err, restructure)) return false;
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

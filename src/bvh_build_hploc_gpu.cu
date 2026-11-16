// GPU H-PLOC (Benthin et al. 2024): after the shared bounds / Morton / sort stages, the whole
// binary hierarchy is built by ONE kernel launch (see hploc.h for the algorithm):
//   * one thread per leaf climbs the implicit Morton hierarchy (Apetrei's bottom-up method);
//     at each parent the first arrival stores its range boundary with atomicExch and stops,
//     the second continues with the merged range ("second arrival continues": no thread ever
//     waits on another, so there is no deadlock risk);
//   * lanes of a warp that hold a node to process are served one at a time (__ballot_sync +
//     __shfl_sync): the whole warp loads that node's two cluster lists (<= 32 clusters, one per
//     lane) into shared memory and runs PLOC iterations on them — nearest neighbour by lane,
//     mutual merges, compaction with __ballot_sync + __popc — until the list is back to the
//     threshold (1 at the root). New node ids come from a global atomic counter.
// No per-iteration launches and no readbacks; the host reads the root id once at the end.
#include <algorithm>

#include "builders.h"
#include "gpu_build_common.cuh"
#include "hploc.h"

using namespace gpub;

namespace {

constexpr unsigned FULL = 0xffffffffu;
constexpr int WARPS = BLOCK / 32;

__global__ void k_hploc_init(HplocTree t, const AABB* __restrict__ triBoxes, const uint32_t* __restrict__ order,
                             int* __restrict__ listId, int* __restrict__ listSize) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= t.n) return;
    hploc_init_leaf(t, k, triBoxes[order[k]]);
    listId[k] = t.n - 1 + k;
    listSize[k] = 1;
}

__global__ void k_hploc(HplocTree t, const uint64_t* __restrict__ keys, int* listId, int* listSize,
                        int* __restrict__ pb, int* __restrict__ nodeCounter, int* __restrict__ rootOut, int threshold,
                        int radius, int maxLeaf, bool sah) {
    __shared__ int sIdA[WARPS][HPLOC_MAX_LIST], sNnA[WARPS][HPLOC_MAX_LIST];
    __shared__ vec3 sLoA[WARPS][HPLOC_MAX_LIST], sHiA[WARPS][HPLOC_MAX_LIST];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int* sId = sIdA[warp];
    int* sNn = sNnA[warp];
    vec3* sLo = sLoA[warp];
    vec3* sHi = sHiA[warp];
    const int n = t.n;

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = k < n;
    int l = k, r = k;
    while (__any_sync(FULL, active)) {
        bool work = false;
        int ls = 0, rs = 0, nL = 0, nR = 0;
        if (active) {
            bool isLeft;
            const int p = hploc_parent(keys, n, l, r, isLeft);
            __threadfence();  // publish this node's list before announcing it
            const int other = atomicExch(&pb[p], isLeft ? l : r);
            if (other < 0) {
                active = false;  // first arrival: the sibling finishes the parent
            } else {
                ls = isLeft ? l : other;
                rs = p + 1;
                nL = ls;
                nR = isLeft ? other : r;
                const int sL = ldcg_i(listSize + ls), sR = ldcg_i(listSize + rs);
                const bool isRoot = nL == 0 && nR == n - 1;
                if (sL + sR > (isRoot ? 1 : threshold)) {
                    work = true;  // needs merging: the whole warp serves this node below
                } else {
                    // Fast path: no merging. The left list already starts at nL; move the right list
                    // up behind it (forward copy is safe: nL + sL <= rs).
                    for (int i = 0; i < sR; ++i) listId[nL + sL + i] = ldcg_i(listId + rs + i);
                    listSize[nL] = sL + sR;
                    l = nL;  // (never the root: the root always merges down to one cluster)
                    r = nR;
                }
            }
        }
        unsigned m = __ballot_sync(FULL, work);
        while (m) {
            const int leader = __ffs(int(m)) - 1;
            m &= m - 1u;
            const int L0 = __shfl_sync(FULL, ls, leader), R0 = __shfl_sync(FULL, rs, leader);
            const int NL = __shfl_sync(FULL, nL, leader), NR = __shfl_sync(FULL, nR, leader);
            const int sL = ldcg_i(listSize + L0), sR = ldcg_i(listSize + R0);
            int s = sL + sR;
            if (lane < s) {
                const int id = lane < sL ? ldcg_i(listId + L0 + lane) : ldcg_i(listId + R0 + lane - sL);
                sId[lane] = id;
                sLo[lane] = xyz(ldcg_f4(t.lo + id));
                sHi[lane] = xyz(ldcg_f4(t.hi + id));
            }
            __syncwarp();
            const bool isRoot = NL == 0 && NR == n - 1;
            const int target = isRoot ? 1 : threshold;
            while (s > target) {
                int j = -1;
                if (lane < s) {
                    j = hploc_nn(sLo, sHi, s, lane, radius);
                    sNn[lane] = j;
                }
                __syncwarp();
                bool valid = false;
                int newId = 0;
                vec3 nlo, nhi;
                if (lane < s) {
                    const bool mutual = sNn[j] == lane;
                    valid = !(mutual && lane > j);
                    newId = sId[lane];
                    nlo = sLo[lane];
                    nhi = sHi[lane];
                    if (mutual && lane < j) {
                        const int q = atomicAdd(nodeCounter, 1);
                        nlo = vmin(nlo, sLo[j]);
                        nhi = vmax(nhi, sHi[j]);
                        hploc_make_node(t, q, sId[lane], sId[j], nlo, nhi, maxLeaf, sah);
                        newId = q;
                    }
                }
                const unsigned vm = __ballot_sync(FULL, valid);  // also orders the reads above
                __syncwarp();
                if (valid) {
                    const int pos = __popc(vm & ((1u << lane) - 1u));
                    sId[pos] = newId;
                    sLo[pos] = nlo;
                    sHi[pos] = nhi;
                }
                s = __popc(vm);
                __syncwarp();
            }
            if (lane < s) listId[NL + lane] = sId[lane];
            if (lane == 0) {
                listSize[NL] = s;
                if (isRoot) *rootOut = sId[0];
            }
            __syncwarp();
        }
        __threadfence();  // publish the merged lists and new nodes before anyone climbs past them
        if (work) {
            l = nL;
            r = nR;
            if (l == 0 && r == n - 1) active = false;  // root done
        }
    }
}

struct HplocBuffers {
    MortonSort ms;
    Buf left, right, parent, lo, hi, triCount, dfsCount, cost, listId, listSize, pb, counters;
    Buf nodes, tris, order;
    bool ensure(int n, std::string& err) {
        const size_t total = size_t(2 * n - 1), inner = size_t(std::max(n - 1, 1));
        CU(ms.ensure(n));
        CU(left.ensure(inner * 4));
        CU(right.ensure(inner * 4));
        CU(parent.ensure(total * 4));
        CU(lo.ensure(total * sizeof(f4)));
        CU(hi.ensure(total * sizeof(f4)));
        CU(triCount.ensure(total * 4));
        CU(dfsCount.ensure(total * 4));
        CU(cost.ensure(total * 4));
        CU(listId.ensure(size_t(n) * 4));
        CU(listSize.ensure(size_t(n) * 4));
        CU(pb.ensure(inner * 4));
        CU(counters.ensure(8 * sizeof(int)));
        CU(nodes.ensure(total * sizeof(BVHNode)));
        CU(tris.ensure(size_t(n) * sizeof(Tri)));
        CU(order.ensure(size_t(n) * 4));
        return true;
    }
};

}  // namespace

struct GpuHplocBuilder::Impl {
    HplocBuffers B;
    DeviceBvh tree;
};

GpuHplocBuilder::GpuHplocBuilder() : impl_(new Impl) {}
GpuHplocBuilder::~GpuHplocBuilder() { delete impl_; }
const DeviceBvh& GpuHplocBuilder::tree() const { return impl_->tree; }
bool GpuHplocBuilder::reserve(int numTris, std::string& err) { return impl_->B.ensure(std::max(numTris, 1), err); }

bool GpuHplocBuilder::build(const void* dVertsV, const void* dFacesV, int n, const BuildParams& params,
                            BuildReport* report, std::string& err) {
    if (n < 1) { err = "empty mesh"; return false; }
    HplocBuffers& B = impl_->B;
    if (!B.ensure(n, err)) return false;
    const int maxLeaf = std::max(1, params.maxLeaf);
    const int radius = std::max(1, std::min(params.hplocRadius, HPLOC_MAX_LIST - 1));
    const int threshold = std::max(1, std::min(params.hplocThreshold, HPLOC_MAX_LIST / 2));
    const vec3* verts = static_cast<const vec3*>(dVertsV);
    const Face* faces = static_cast<const Face*>(dFacesV);
    CU(cudaMemsetAsync(B.parent.p, 0xff, size_t(2 * n - 1) * 4));  // every parent -1 until set
    CU(cudaMemsetAsync(B.pb.p, 0xff, size_t(std::max(n - 1, 1)) * 4));
    CU(cudaMemsetAsync(B.counters.p, 0, 8 * sizeof(int)));

    int* counters = B.counters.as<int>();  // [0] max depth, [1] leaf count, [2] node counter, [3] root
    HplocTree tree{n, B.left.as<int>(), B.right.as<int>(), B.parent.as<int>(), B.lo.as<f4>(), B.hi.as<f4>(),
                   B.triCount.as<uint32_t>(), B.dfsCount.as<uint32_t>(), B.cost.as<float>()};
    Events ev;
    cudaEvent_t e0 = ev.mark();
    CU(B.ms.run(verts, faces, n, ev));
    k_hploc_init<<<blocks_for(n), BLOCK>>>(tree, B.ms.triBoxes(), B.ms.order(), B.listId.as<int>(), B.listSize.as<int>());
    if (n > 1)
        k_hploc<<<blocks_for(n), BLOCK>>>(tree, B.ms.sortedKeys(), B.listId.as<int>(), B.listSize.as<int>(),
                                          B.pb.as<int>(), counters + 2, counters + 3, threshold, radius, maxLeaf,
                                          params.sahCollapse);
    cudaEvent_t e4 = ev.mark();
    CU(cudaGetLastError());
    int host[4] = {0, 0, 0, 0};
    CU(cudaMemcpy(host, counters, 4 * sizeof(int), cudaMemcpyDeviceToHost));  // root id for the converter
    int root = n > 1 ? host[3] : 0;
    if (n > 1 && host[2] != n - 1) {
        err = "H-PLOC created " + std::to_string(host[2]) + " nodes, expected n-1";
        return false;
    }
    cudaEvent_t e5 = ev.mark();
    PrbvhStats ps;
    double optMs = 0;
    if (params.prbvhIters > 0 && n > 2 &&
        !prbvh_optimize_gpu(n, root, B.left.as<int>(), B.right.as<int>(), B.parent.as<int>(), B.lo.as<f4>(),
                            B.hi.as<f4>(), B.triCount.as<uint32_t>(), B.dfsCount.as<uint32_t>(), B.cost.as<float>(),
                            params, &ps, optMs, err))
        return false;
    cudaEvent_t e5b = ev.mark();
    DeviceTree dt{n, root, B.left.as<int>(), B.right.as<int>(), B.parent.as<int>(), B.triCount.as<uint32_t>(),
                  B.dfsCount.as<uint32_t>()};
    CU(convert_tree(dt, reinterpret_cast<const float4*>(B.lo.p), reinterpret_cast<const float4*>(B.hi.p),
                    B.ms.order(), verts, faces, B.nodes.as<BVHNode>(), B.tris.as<Tri>(), B.order.as<uint32_t>(),
                    counters));
    cudaEvent_t e6 = ev.mark();
    CU(cudaEventSynchronize(e6));
    const cudaEvent_t e1 = ev.ev[1], e2 = ev.ev[2], e3 = ev.ev[3];
    BuildReport rep;
    rep.stages = {{"bounds", elapsed(e0, e1)}, {"morton", elapsed(e1, e2)}, {"sort", elapsed(e2, e3)},
                  {"cluster", elapsed(e3, e4)}};
    if (params.prbvhIters > 0 && n > 2) rep.stages.push_back({"optimize", elapsed(e5, e5b)});
    rep.stages.push_back({"convert", elapsed(e5b, e6)});
    rep.buildMs = elapsed(e0, e4) + elapsed(e5, e6);  // the root readback between them is not build work
    rep.optimizeMs = optMs;
    rep.prbvhRounds = ps.iterations;
    rep.prbvhMoves = ps.moves;

    CU(cudaMemcpy(host, counters, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    uint32_t numNodes = 0;
    CU(cudaMemcpy(&numNodes, B.dfsCount.as<uint32_t>() + root, 4, cudaMemcpyDeviceToHost));
    DeviceBvh& t = impl_->tree;
    t = DeviceBvh();
    t.nodes = B.nodes.p;
    t.tris = B.tris.p;
    t.triOrder = B.order.p;
    t.numNodes = numNodes;
    t.numTris = uint32_t(n);
    t.maxDepth = host[0];
    t.leafCount = host[1];
    if (report) *report = rep;
    return true;
}

bool GpuHplocBuilder::download(BVH& b, std::string& err) const {
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

DeviceBvh GpuHplocBuilder::take_tree() {
    DeviceBvh t = impl_->tree;
    t.nodes = impl_->B.nodes.take();
    t.tris = impl_->B.tris.take();
    t.triOrder = impl_->B.order.take();
    impl_->tree = DeviceBvh();
    return t;
}

bool build_hploc_gpu(const Mesh& mesh, const BuildParams& params, BVH* hostOut, DeviceBvh* devOut,
                     BuildReport* report, std::string& err) {
    const int n = int(mesh.faces.size());
    if (n < 1) { err = "empty mesh"; return false; }
    if (!hostOut && !devOut) { err = "no output requested"; return false; }
    auto tUp = Clock::now();
    Buf dVerts, dFaces;
    CU(dVerts.ensure(mesh.vertices.size() * sizeof(vec3)));
    CU(dFaces.ensure(mesh.faces.size() * sizeof(Face)));
    CU(cudaMemcpy(dVerts.p, mesh.vertices.data(), mesh.vertices.size() * sizeof(vec3), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dFaces.p, mesh.faces.data(), mesh.faces.size() * sizeof(Face), cudaMemcpyHostToDevice));
    GpuHplocBuilder builder;
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

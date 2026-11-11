// GPU PLOC (Meister & Bittner 2018). Shared stages (bounds, Morton, CUB sort, collapse +
// flatten) come from gpu_build_common.cu. Each clustering iteration is four data-parallel
// steps over the current c clusters:
//   nn      one thread per cluster scans +-r neighbours, staged in a shared-memory tile
//           with an r-wide halo on each side
//   flags   mutual nearest neighbours; packs (creates a node) << 32 | (still valid)
//   scan    hand-written exclusive scan of the packed flags: the low half is each
//           survivor's new slot, the high half the id offset of each new node
//   merge   new internal nodes are written, survivors scattered in order
// The CPU only reads back one 8-byte total per iteration (the next cluster count).
#include <algorithm>

#include "builders.h"
#include "gpu_build_common.cuh"
#include "ploc.h"

using namespace gpub;

namespace {

__global__ void k_ploc_init(const AABB* __restrict__ triBoxes, const uint32_t* __restrict__ order, int n,
                            int* __restrict__ cNode, float4* __restrict__ cLo, float4* __restrict__ cHi,
                            float4* __restrict__ nodeLo, float4* __restrict__ nodeHi, uint32_t* __restrict__ triCount,
                            uint32_t* __restrict__ dfsCount, float* __restrict__ cost) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int x = n - 1 + k;
    const AABB b = triBoxes[order[k]];
    const float4 lo = make_float4(b.bmin.x, b.bmin.y, b.bmin.z, 0.0f), hi = make_float4(b.bmax.x, b.bmax.y, b.bmax.z, 0.0f);
    nodeLo[x] = lo;
    nodeHi[x] = hi;
    triCount[x] = 1;
    dfsCount[x] = 1;
    cost[x] = leaf_cost(1, box_area_rn(b.bmin, b.bmax));
    cNode[k] = x;
    cLo[k] = lo;
    cHi[k] = hi;
}

__device__ __forceinline__ vec3 v3(float4 a) { return {a.x, a.y, a.z}; }

// Nearest neighbour of every cluster within +-r. The block's clusters plus an r-wide halo on
// each side are staged in shared memory (2 float4 per cluster).
__global__ void k_ploc_nn(const float4* __restrict__ cLo, const float4* __restrict__ cHi, int c, int r,
                          int* __restrict__ nn) {
    extern __shared__ float4 sh[];
    const int T = BLOCK + 2 * r;
    float4* shLo = sh;
    float4* shHi = sh + T;
    const int base = blockIdx.x * BLOCK - r;
    for (int t = threadIdx.x; t < T; t += BLOCK) {
        const int g = base + t;
        if (g >= 0 && g < c) {
            shLo[t] = cLo[g];
            shHi[t] = cHi[g];
        }
    }
    __syncthreads();
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= c) return;
    const vec3 alo = v3(shLo[i - base]), ahi = v3(shHi[i - base]);
    float bd = RT_INF;
    int bj = -1;
    const int j0 = max(0, i - r), j1 = min(c - 1, i + r);
    for (int j = j0; j <= j1; ++j) {
        if (j == i) continue;
        const float d = ploc_merged_area(alo, ahi, v3(shLo[j - base]), v3(shHi[j - base]));
        if (bj < 0 || ploc_better(d, i, j, bd, bj)) {
            bd = d;
            bj = j;
        }
    }
    nn[i] = bj;
}

__global__ void k_ploc_flags(const int* __restrict__ nn, int c, uint64_t* __restrict__ flags) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c) return;
    const int j = nn[i];
    const bool mutual = nn[j] == i;
    const uint64_t create = mutual && i < j, valid = !(mutual && i > j);
    flags[i] = (create << 32) | valid;
}

__global__ void k_ploc_merge(int c, int base, int maxLeaf, bool sah, const int* __restrict__ nn,
                             const uint64_t* __restrict__ flags, const uint64_t* __restrict__ offs,
                             const int* __restrict__ cNodeIn, const float4* __restrict__ cLoIn,
                             const float4* __restrict__ cHiIn, int* __restrict__ cNodeOut, float4* __restrict__ cLoOut,
                             float4* __restrict__ cHiOut, int* __restrict__ left, int* __restrict__ right,
                             int* __restrict__ parent, float4* __restrict__ nodeLo, float4* __restrict__ nodeHi,
                             uint32_t* __restrict__ triCount, uint32_t* __restrict__ dfsCount,
                             float* __restrict__ cost, uint64_t* __restrict__ totals) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c) return;
    const uint64_t f = flags[i], o = offs[i];
    if (i == c - 1) *totals = o + f;  // (nodes created) << 32 | (clusters left)
    if (!(f & 0xffffffffull)) return;
    const int pos = int(uint32_t(o));
    if (f >> 32) {
        const int j = nn[i];
        const int p = base + int(o >> 32);
        const int l = cNodeIn[i], r = cNodeIn[j];
        const float4 al = cLoIn[i], ah = cHiIn[i], bl = cLoIn[j], bh = cHiIn[j];
        const float4 lo = make_float4(fminf(al.x, bl.x), fminf(al.y, bl.y), fminf(al.z, bl.z), 0.0f);
        const float4 hi = make_float4(fmaxf(ah.x, bh.x), fmaxf(ah.y, bh.y), fmaxf(ah.z, bh.z), 0.0f);
        left[p] = l;
        right[p] = r;
        parent[l] = p;
        parent[r] = p;
        nodeLo[p] = lo;
        nodeHi[p] = hi;
        const uint32_t t = triCount[l] + triCount[r];
        triCount[p] = t;
        float cp;
        dfsCount[p] = collapse_node(t, box_area_rn(make_vec3(lo.x, lo.y, lo.z), make_vec3(hi.x, hi.y, hi.z)), cost[l],
                                    cost[r], dfsCount[l], dfsCount[r], maxLeaf, sah, cp);
        cost[p] = cp;
        cNodeOut[pos] = p;
        cLoOut[pos] = lo;
        cHiOut[pos] = hi;
    } else {
        cNodeOut[pos] = cNodeIn[i];
        cLoOut[pos] = cLoIn[i];
        cHiOut[pos] = cHiIn[i];
    }
}

struct PlocBuffers {
    MortonSort ms;
    Buf cNode[2], cLo[2], cHi[2], nn, flags, offs, totals;
    Buf left, right, parent, nodeLo, nodeHi, triCount, dfsCount, cost, counters;
    Buf nodes, tris, order;
    ScanTemp scan;

    bool ensure(int n, std::string& err) {
        const size_t total = size_t(2 * n - 1), inner = size_t(std::max(n - 1, 1));
        CU(ms.ensure(n));
        for (int b = 0; b < 2; ++b) {
            CU(cNode[b].ensure(size_t(n) * 4));
            CU(cLo[b].ensure(size_t(n) * sizeof(float4)));
            CU(cHi[b].ensure(size_t(n) * sizeof(float4)));
        }
        CU(nn.ensure(size_t(n) * 4));
        CU(flags.ensure(size_t(n) * 8));
        CU(offs.ensure(size_t(n) * 8));
        CU(totals.ensure(8));
        CU(left.ensure(inner * 4));
        CU(right.ensure(inner * 4));
        CU(parent.ensure(total * 4));
        CU(nodeLo.ensure(total * sizeof(float4)));
        CU(nodeHi.ensure(total * sizeof(float4)));
        CU(triCount.ensure(total * 4));
        CU(dfsCount.ensure(total * 4));
        CU(cost.ensure(total * 4));
        CU(counters.ensure(4 * sizeof(int)));
        CU(nodes.ensure(total * sizeof(BVHNode)));
        CU(tris.ensure(size_t(n) * sizeof(Tri)));
        CU(order.ensure(size_t(n) * 4));
        return true;
    }
};

bool ploc_build(PlocBuffers& B, const vec3* dVerts, const Face* dFaces, int n, const BuildParams& params,
                BuildReport& rep, DeviceBvh& tree, std::string& err) {
    const int maxLeaf = std::max(1, params.maxLeaf);
    const int r = std::max(1, std::min(params.plocRadius, 256));
    if (!B.ensure(n, err)) return false;
    CU(cudaMemsetAsync(B.counters.p, 0, 4 * sizeof(int)));
    CU(cudaMemsetAsync(B.parent.p, 0xff, size_t(2 * n - 1) * 4));  // every parent -1 until set

    Events ev;
    cudaEvent_t e0 = ev.mark();
    CU(B.ms.run(dVerts, dFaces, n, ev));
    k_ploc_init<<<blocks_for(n), BLOCK>>>(B.ms.triBoxes(), B.ms.order(), n, B.cNode[0].as<int>(), B.cLo[0].as<float4>(),
                                          B.cHi[0].as<float4>(), B.nodeLo.as<float4>(), B.nodeHi.as<float4>(),
                                          B.triCount.as<uint32_t>(), B.dfsCount.as<uint32_t>(), B.cost.as<float>());
    int c = n, base = 0, cur = 0, iterations = 0;
    const size_t shBytes = size_t(BLOCK + 2 * r) * 2 * sizeof(float4);
    while (c > 1) {
        ++iterations;
        const int nb = blocks_for(c);
        k_ploc_nn<<<nb, BLOCK, shBytes>>>(B.cLo[cur].as<float4>(), B.cHi[cur].as<float4>(), c, r, B.nn.as<int>());
        k_ploc_flags<<<nb, BLOCK>>>(B.nn.as<int>(), c, B.flags.as<uint64_t>());
        CU(scan_exclusive_u64(B.flags.as<uint64_t>(), B.offs.as<uint64_t>(), c, B.scan));
        k_ploc_merge<<<nb, BLOCK>>>(c, base, maxLeaf, params.sahCollapse, B.nn.as<int>(), B.flags.as<uint64_t>(), B.offs.as<uint64_t>(),
                                    B.cNode[cur].as<int>(), B.cLo[cur].as<float4>(), B.cHi[cur].as<float4>(),
                                    B.cNode[1 - cur].as<int>(), B.cLo[1 - cur].as<float4>(), B.cHi[1 - cur].as<float4>(),
                                    B.left.as<int>(), B.right.as<int>(), B.parent.as<int>(), B.nodeLo.as<float4>(),
                                    B.nodeHi.as<float4>(), B.triCount.as<uint32_t>(), B.dfsCount.as<uint32_t>(),
                                    B.cost.as<float>(), B.totals.as<uint64_t>());
        uint64_t totals = 0;
        CU(cudaMemcpy(&totals, B.totals.p, 8, cudaMemcpyDeviceToHost));
        const int created = int(totals >> 32), left = int(uint32_t(totals));
        if (created == 0) { err = "PLOC made no progress"; return false; }
        base += created;
        c = left;
        cur = 1 - cur;
    }
    cudaEvent_t e4 = ev.mark();
    const int root = n > 1 ? n - 2 : 0;
    DeviceTree dt{n, root, B.left.as<int>(), B.right.as<int>(), B.parent.as<int>(), B.triCount.as<uint32_t>(),
                  B.dfsCount.as<uint32_t>()};
    CU(convert_tree(dt, B.nodeLo.as<float4>(), B.nodeHi.as<float4>(), B.ms.order(), dVerts, dFaces,
                    B.nodes.as<BVHNode>(), B.tris.as<Tri>(), B.order.as<uint32_t>(), B.counters.as<int>()));
    cudaEvent_t e5 = ev.mark();
    CU(cudaGetLastError());
    CU(cudaEventSynchronize(e5));
    const cudaEvent_t e1 = ev.ev[1], e2 = ev.ev[2], e3 = ev.ev[3];
    rep.stages = {{"bounds", elapsed(e0, e1)}, {"morton", elapsed(e1, e2)}, {"sort", elapsed(e2, e3)},
                  {"cluster", elapsed(e3, e4)}, {"convert", elapsed(e4, e5)}};
    rep.buildMs = elapsed(e0, e5);
    rep.iterations = iterations;

    int counters[2] = {0, 0};
    uint32_t numNodes = 0;
    CU(cudaMemcpy(counters, B.counters.p, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&numNodes, B.dfsCount.as<uint32_t>() + root, 4, cudaMemcpyDeviceToHost));
    tree.nodes = B.nodes.p;
    tree.tris = B.tris.p;
    tree.triOrder = B.order.p;
    tree.numNodes = numNodes;
    tree.numTris = uint32_t(n);
    tree.maxDepth = counters[0];
    tree.leafCount = counters[1];
    return true;
}

}  // namespace

struct GpuPlocBuilder::Impl {
    PlocBuffers B;
    DeviceBvh tree;
};

GpuPlocBuilder::GpuPlocBuilder() : impl_(new Impl) {}
GpuPlocBuilder::~GpuPlocBuilder() { delete impl_; }
const DeviceBvh& GpuPlocBuilder::tree() const { return impl_->tree; }
bool GpuPlocBuilder::reserve(int numTris, std::string& err) { return impl_->B.ensure(std::max(numTris, 1), err); }

bool GpuPlocBuilder::build(const void* dVerts, const void* dFaces, int n, const BuildParams& params,
                           BuildReport* report, std::string& err) {
    if (n < 1) { err = "empty mesh"; return false; }
    BuildReport rep;
    impl_->tree = DeviceBvh();
    if (!ploc_build(impl_->B, static_cast<const vec3*>(dVerts), static_cast<const Face*>(dFaces), n, params, rep,
                    impl_->tree, err))
        return false;
    if (report) *report = rep;
    return true;
}

bool GpuPlocBuilder::download(BVH& b, std::string& err) const {
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

DeviceBvh GpuPlocBuilder::take_tree() {
    DeviceBvh t = impl_->tree;
    t.nodes = impl_->B.nodes.take();
    t.tris = impl_->B.tris.take();
    t.triOrder = impl_->B.order.take();
    impl_->tree = DeviceBvh();
    return t;
}

bool build_ploc_gpu(const Mesh& mesh, const BuildParams& params, BVH* hostOut, DeviceBvh* devOut,
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
    GpuPlocBuilder builder;
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

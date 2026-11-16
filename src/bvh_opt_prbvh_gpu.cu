// PRBVH on the GPU (see prbvh.h). Per round: one thread per node searches its best reinsertion;
// moves claim their six nodes with atomicMax on a 64-bit priority key; a second kernel marks the
// moves that hold all their locks; a third applies them (disjoint node sets, so no races); a
// bottom-up refit (arrival counters, as in the LBVH fit) recomputes boxes, collapse decisions and
// the plain SAH cost. The host reads back the root and its cost once per round to decide whether
// to keep the round (undo it if the cost went up) and whether to continue.
#include <algorithm>

#include "builders.h"
#include "gpu_build_common.cuh"
#include "prbvh.h"

using namespace gpub;

namespace {

__global__ void k_prbvh_search(PrbvhTree t, int total, PrbvhMove* __restrict__ mv) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= total) return;
    mv[x] = prbvh_search(t, x);
}

__global__ void k_prbvh_lock(PrbvhTree t, int total, const PrbvhMove* __restrict__ mv, float eps,
                             unsigned long long* __restrict__ lock) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= total || !(mv[x].gain > eps)) return;
    int nodes[6];
    prbvh_move_nodes(t, x, mv[x].target, nodes);
    const unsigned long long key = prbvh_key(mv[x].gain, prbvh_rank(t, x));
    for (int q : nodes)
        if (q >= 0) atomicMax(lock + q, key);
}

__global__ void k_prbvh_winners(PrbvhTree t, int total, const PrbvhMove* __restrict__ mv, float eps,
                                const unsigned long long* __restrict__ lock, int* __restrict__ win) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= total) return;
    int w = 0;
    if (mv[x].gain > eps) {
        int nodes[6];
        prbvh_move_nodes(t, x, mv[x].target, nodes);
        const unsigned long long key = prbvh_key(mv[x].gain, prbvh_rank(t, x));
        w = 1;
        for (int q : nodes)
            if (q >= 0 && lock[q] != key) w = 0;
    }
    win[x] = w;
}

__global__ void k_prbvh_apply(PrbvhTree t, int total, const PrbvhMove* __restrict__ mv, const int* __restrict__ win,
                              int* __restrict__ root, int* __restrict__ applied) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= total || !win[x]) return;
    const int r = prbvh_apply(t, x, mv[x].target);
    if (r >= 0) *root = r;
    atomicAdd(applied, 1);
}

__global__ void k_prbvh_refit(PrbvhTree t, int maxLeaf, bool sah, unsigned int* __restrict__ visited) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= t.n) return;
    const int x = t.n - 1 + k;
    prbvh_refit_leaf(t, x);
    for (int p = ldcg_i(t.parent + x); p >= 0; p = ldcg_i(t.parent + p)) {
        __threadfence();
        if (atomicAdd(visited + p, 1u) == 0) return;
        prbvh_refit_node(t, p, maxLeaf, sah);
    }
}

}  // namespace

bool prbvh_optimize_gpu(int n, int& root, int* left, int* right, int* parent, f4* lo, f4* hi, uint32_t* triCount,
                        uint32_t* dfsCount, float* cost, const BuildParams& params, PrbvhStats* stats, double& ms,
                        std::string& err) {
    PrbvhStats st;
    ms = 0;
    if (n < 3 || params.prbvhIters <= 0) {
        if (stats) *stats = st;
        return true;
    }
    const int total = 2 * n - 1, inner = n - 1;
    const int maxLeaf = std::max(1, params.maxLeaf);
    Buf plain, mv, lock, win, visited, dRoot, applied, sLeft, sRight, sParent;
    CU(plain.ensure(size_t(total) * 4));
    CU(mv.ensure(size_t(total) * sizeof(PrbvhMove)));
    CU(lock.ensure(size_t(total) * 8));
    CU(win.ensure(size_t(total) * 4));
    CU(visited.ensure(size_t(inner) * 4));
    CU(dRoot.ensure(4));
    CU(applied.ensure(4));
    CU(sLeft.ensure(size_t(inner) * 4));
    CU(sRight.ensure(size_t(inner) * 4));
    CU(sParent.ensure(size_t(total) * 4));
    PrbvhTree t{n, left, right, parent, lo, hi, triCount, dfsCount, cost, plain.as<float>()};

    Events ev;
    cudaEvent_t e0 = ev.mark();
    CU(cudaMemsetAsync(visited.p, 0, size_t(inner) * 4));
    k_prbvh_refit<<<blocks_for(n), BLOCK>>>(t, maxLeaf, params.sahCollapse, visited.as<unsigned int>());
    CU(cudaMemcpyAsync(dRoot.p, &root, 4, cudaMemcpyHostToDevice));
    float prev = 0;
    CU(cudaMemcpy(&prev, plain.as<float>() + root, 4, cudaMemcpyDeviceToHost));
    st.costBefore = prev;
    for (int it = 0; it < params.prbvhIters; ++it) {
        f4 rlo, rhi;
        CU(cudaMemcpy(&rlo, lo + root, sizeof(f4), cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(&rhi, hi + root, sizeof(f4), cudaMemcpyDeviceToHost));
        const float eps = 1e-6f * box_area_rn(xyz(rlo), xyz(rhi));
        k_prbvh_search<<<blocks_for(total), BLOCK>>>(t, total, mv.as<PrbvhMove>());
        CU(cudaMemsetAsync(lock.p, 0, size_t(total) * 8));
        k_prbvh_lock<<<blocks_for(total), BLOCK>>>(t, total, mv.as<PrbvhMove>(), eps, lock.as<unsigned long long>());
        k_prbvh_winners<<<blocks_for(total), BLOCK>>>(t, total, mv.as<PrbvhMove>(), eps, lock.as<unsigned long long>(),
                                                      win.as<int>());
        CU(cudaMemcpyAsync(sLeft.p, left, size_t(inner) * 4, cudaMemcpyDeviceToDevice));
        CU(cudaMemcpyAsync(sRight.p, right, size_t(inner) * 4, cudaMemcpyDeviceToDevice));
        CU(cudaMemcpyAsync(sParent.p, parent, size_t(total) * 4, cudaMemcpyDeviceToDevice));
        CU(cudaMemsetAsync(applied.p, 0, 4));
        k_prbvh_apply<<<blocks_for(total), BLOCK>>>(t, total, mv.as<PrbvhMove>(), win.as<int>(), dRoot.as<int>(),
                                                    applied.as<int>());
        int nApplied = 0;
        CU(cudaMemcpy(&nApplied, applied.p, 4, cudaMemcpyDeviceToHost));
        if (nApplied == 0) break;
        const int sRoot = root;
        CU(cudaMemcpy(&root, dRoot.p, 4, cudaMemcpyDeviceToHost));
        CU(cudaMemsetAsync(visited.p, 0, size_t(inner) * 4));
        k_prbvh_refit<<<blocks_for(n), BLOCK>>>(t, maxLeaf, params.sahCollapse, visited.as<unsigned int>());
        float now = 0;
        CU(cudaMemcpy(&now, plain.as<float>() + root, 4, cudaMemcpyDeviceToHost));
        ++st.iterations;
        if (now > prev) {  // moves interacted badly: undo this round and stop
            CU(cudaMemcpyAsync(left, sLeft.p, size_t(inner) * 4, cudaMemcpyDeviceToDevice));
            CU(cudaMemcpyAsync(right, sRight.p, size_t(inner) * 4, cudaMemcpyDeviceToDevice));
            CU(cudaMemcpyAsync(parent, sParent.p, size_t(total) * 4, cudaMemcpyDeviceToDevice));
            root = sRoot;
            CU(cudaMemcpyAsync(dRoot.p, &root, 4, cudaMemcpyHostToDevice));
            CU(cudaMemsetAsync(visited.p, 0, size_t(inner) * 4));
            k_prbvh_refit<<<blocks_for(n), BLOCK>>>(t, maxLeaf, params.sahCollapse, visited.as<unsigned int>());
            st.rolledBack = true;
            break;
        }
        st.moves += nApplied;
        const bool tiny = prev - now < 1e-4f * prev;
        prev = now;
        if (tiny) break;
    }
    cudaEvent_t e1 = ev.mark();
    CU(cudaGetLastError());
    CU(cudaEventSynchronize(e1));
    ms = elapsed(e0, e1);
    float after = 0;
    CU(cudaMemcpy(&after, plain.as<float>() + root, 4, cudaMemcpyDeviceToHost));
    st.costAfter = after;
    if (stats) *stats = st;
    return true;
}

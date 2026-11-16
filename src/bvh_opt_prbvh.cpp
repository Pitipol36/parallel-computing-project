// PRBVH on the CPU: the same rounds as the GPU (search all nodes -> lock by priority -> apply the
// winners -> refit -> keep the round only if the SAH cost went down), run sequentially. Every step
// is a pure function of the tree at the start of the round, so the result equals the GPU's.
#include <algorithm>
#include <cstring>
#include <vector>

#include "builders.h"
#include "prbvh.h"

namespace {

void refit_all(const PrbvhTree& t, int maxLeaf, bool sah) {
    const int n = t.n;
    std::vector<uint8_t> visited(std::max(n - 1, 1), 0);
    for (int k = 0; k < n; ++k) {
        const int x = n - 1 + k;
        prbvh_refit_leaf(t, x);
        for (int p = t.parent[x]; p >= 0; p = t.parent[p]) {
            if (visited[p]++ == 0) break;
            prbvh_refit_node(t, p, maxLeaf, sah);
        }
    }
}

}  // namespace

int prbvh_optimize_cpu(int n, int root, int* left, int* right, int* parent, f4* lo, f4* hi, uint32_t* triCount,
                       uint32_t* dfsCount, float* cost, const BuildParams& params, PrbvhStats* stats) {
    PrbvhStats st;
    if (n < 3 || params.prbvhIters <= 0) {
        if (stats) *stats = st;
        return root;
    }
    const int total = 2 * n - 1;
    const int maxLeaf = std::max(1, params.maxLeaf);
    std::vector<float> plain(total);
    PrbvhTree t{n, left, right, parent, lo, hi, triCount, dfsCount, cost, plain.data()};
    refit_all(t, maxLeaf, params.sahCollapse);
    st.costBefore = plain[root];
    float prev = plain[root];

    std::vector<PrbvhMove> mv(total);
    std::vector<uint64_t> lock(total);
    std::vector<int> sLeft(std::max(n - 1, 1)), sRight(std::max(n - 1, 1)), sParent(total);
    for (int it = 0; it < params.prbvhIters; ++it) {
        const AABB rb = prbvh_box(t, root);
        const float eps = 1e-6f * area_of(rb);
        for (int x = 0; x < total; ++x) mv[x] = prbvh_search(t, x);
        std::fill(lock.begin(), lock.end(), 0ull);
        for (int x = 0; x < total; ++x) {
            if (!(mv[x].gain > eps)) continue;
            int nodes[6];
            prbvh_move_nodes(t, x, mv[x].target, nodes);
            const uint64_t key = prbvh_key(mv[x].gain, prbvh_rank(t, x));
            for (int q : nodes)
                if (q >= 0) lock[q] = std::max(lock[q], key);
        }
        std::copy(left, left + std::max(n - 1, 1), sLeft.begin());
        std::copy(right, right + std::max(n - 1, 1), sRight.begin());
        std::copy(parent, parent + total, sParent.begin());
        const int sRoot = root;
        int applied = 0;
        // Winners are found before any move is applied (the GPU decides them in a separate kernel).
        std::vector<int> winners;
        for (int x = 0; x < total; ++x) {
            if (!(mv[x].gain > eps)) continue;
            int nodes[6];
            prbvh_move_nodes(t, x, mv[x].target, nodes);
            const uint64_t key = prbvh_key(mv[x].gain, prbvh_rank(t, x));
            bool owns = true;
            for (int q : nodes)
                if (q >= 0 && lock[q] != key) owns = false;
            if (owns) winners.push_back(x);
        }
        for (int x : winners) {
            const int r = prbvh_apply(t, x, mv[x].target);
            if (r >= 0) root = r;
            ++applied;
        }
        if (applied == 0) break;
        refit_all(t, maxLeaf, params.sahCollapse);
        const float now = plain[root];
        ++st.iterations;
        if (now > prev) {  // moves interacted badly: undo this round and stop
            std::copy(sLeft.begin(), sLeft.end(), left);
            std::copy(sRight.begin(), sRight.end(), right);
            std::copy(sParent.begin(), sParent.end(), parent);
            root = sRoot;
            refit_all(t, maxLeaf, params.sahCollapse);
            st.rolledBack = true;
            break;
        }
        st.moves += applied;
        const bool tiny = prev - now < 1e-4f * prev;
        prev = now;
        if (tiny) break;
    }
    st.costAfter = plain[root];
    if (stats) *stats = st;
    return root;
}

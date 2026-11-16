// H-PLOC (Benthin, Meister, Barczak, Mehalwal, Tsakok, Kensler, HPG 2024), shared by the CPU
// twin and the GPU kernel.
//
// The LBVH's Morton-code hierarchy is walked bottom-up without being stored (Apetrei 2014):
// each thread starts at a leaf with the key range [k, k] and repeatedly moves to the parent,
// whose split is the neighbouring boundary with the longer common prefix. The first thread to
// reach a parent stops; the second continues with the merged range. Every node carries a list
// of clusters (BVH node ids) for its range, stored at the start of the range. At each node the
// two children's lists are concatenated; when the result holds more than `threshold` clusters
// (16, half a warp), PLOC iterations run on it until it is back to `threshold`; at the root
// they run until one cluster remains. A PLOC iteration inside a list: every cluster finds its
// nearest neighbour within +-radius list positions (merged-box surface area, PLOC tie-break),
// mutual pairs merge (the lower position creates the node and keeps its place), and the list
// is compacted in order.
//
// Merging decisions depend only on the list's contents and order, so the topology is the same
// whatever order nodes are processed in; only the internal node ids (taken from a counter, an
// atomic on the GPU) differ. The flattened output does not depend on ids, so CPU and GPU trees
// compare equal after the converter.
#pragma once

#include <cstdint>

#include "bvh_common.h"
#include "morton.h"
#include "ploc.h"
#include "trbvh.h"  // ldcg_* loads

constexpr int HPLOC_MAX_LIST = 32;  // two children of at most 16 clusters each

// Parent of the node covering keys [l, r] (0 <= l <= r < n, not the root): returns the split
// index p (the parent covers keys up to p and from p + 1) and whether this node is the left child.
HD int hploc_parent(const uint64_t* keys, int n, int l, int r, bool& isLeft) {
    const int dl = l > 0 ? lbvh_delta(keys, n, l - 1, l) : -1;
    const int dr = r < n - 1 ? lbvh_delta(keys, n, r, r + 1) : -1;
    isLeft = dr > dl;  // the deeper (longer-prefix) boundary is the parent's split
    return isLeft ? r : l - 1;
}

// Nearest neighbour of list entry i within +-radius positions (PLOC distance and tie-break).
HD int hploc_nn(const vec3* lo, const vec3* hi, int s, int i, int radius) {
    float bd = RT_INF;
    int bj = -1;
    const int j0 = i - radius > 0 ? i - radius : 0, j1 = i + radius < s - 1 ? i + radius : s - 1;
    for (int j = j0; j <= j1; ++j) {
        if (j == i) continue;
        const float d = ploc_merged_area(lo[i], hi[i], lo[j], hi[j]);
        if (bj < 0 || ploc_better(d, i, j, bd, bj)) {
            bd = d;
            bj = j;
        }
    }
    return bj;
}

// Binary tree being built (ids as in bvh_common.h; boxes as two f4 per node, w unused).
struct HplocTree {
    int n;
    int* left;
    int* right;
    int* parent;
    f4* lo;
    f4* hi;
    uint32_t* triCount;
    uint32_t* dfsCount;
    float* cost;
};

// Writes internal node p with children l and r and box [lo, hi], including the SAH collapse.
HD void hploc_make_node(const HplocTree& t, int p, int l, int r, vec3 lo, vec3 hi, int maxLeaf, bool sah) {
    t.left[p] = l;
    t.right[p] = r;
    t.parent[l] = p;
    t.parent[r] = p;
    t.lo[p] = {lo.x, lo.y, lo.z, 0.0f};
    t.hi[p] = {hi.x, hi.y, hi.z, 0.0f};
    const uint32_t tris = ldcg_u(t.triCount + l) + ldcg_u(t.triCount + r);
    t.triCount[p] = tris;
    float c;
    t.dfsCount[p] = collapse_node(tris, box_area_rn(lo, hi), ldcg_f(t.cost + l), ldcg_f(t.cost + r),
                                  ldcg_u(t.dfsCount + l), ldcg_u(t.dfsCount + r), maxLeaf, sah, c);
    t.cost[p] = c;
}

HD void hploc_init_leaf(const HplocTree& t, int k, const AABB& box) {
    const int x = t.n - 1 + k;
    t.lo[x] = {box.bmin.x, box.bmin.y, box.bmin.z, 0.0f};
    t.hi[x] = {box.bmax.x, box.bmax.y, box.bmax.z, 0.0f};
    t.triCount[x] = 1;
    t.dfsCount[x] = 1;
    t.cost[x] = leaf_cost(1, box_area_rn(box.bmin, box.bmax));
}

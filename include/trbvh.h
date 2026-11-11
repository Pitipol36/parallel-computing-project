// TRBVH treelet restructuring (Karras & Aila 2013), shared by the CPU and GPU builders.
//
// Starting from an LBVH with one triangle per leaf, bottom-up passes visit every node whose
// subtree holds at least gamma triangles. At each such node a treelet is grown: start with
// the node's two children and repeatedly replace the treelet leaf with the largest surface
// area by its two children, until there are up to 7 treelet leaves. A dynamic program over
// all subsets of those leaves then finds the SAH-optimal binary topology for them, and the
// treelet's internal nodes (the root keeps its id) are rewired to that topology.
//
//   cost(S) = min( C_trav * A(S) + min over splits {P, S \ P} of cost(P) + cost(S \ P),
//                  C_isect * N(S) * A(S)                       if N(S) <= maxLeaf )
//
// With 7 leaves that is 127 subsets and 3^7 = 2,187 split evaluations. The GPU spreads each
// treelet over a warp, as in the paper; the CPU runs the same steps sequentially. Reads of data
// written by other threads go through ldcg_* (L2, bypassing the non-coherent L1) on the device.
#pragma once

#include <cstdint>

#include "bvh_common.h"

constexpr int TRBVH_MAX_LEAVES = 7;

HD int ldcg_i(const int* p) {
#ifdef __CUDA_ARCH__
    return __ldcg(p);
#else
    return *p;
#endif
}
HD uint32_t ldcg_u(const uint32_t* p) {
#ifdef __CUDA_ARCH__
    return __ldcg(reinterpret_cast<const unsigned int*>(p));
#else
    return *p;
#endif
}
HD float ldcg_f(const float* p) {
#ifdef __CUDA_ARCH__
    return __ldcg(p);
#else
    return *p;
#endif
}
HD f4 ldcg_f4(const f4* p) {
#ifdef __CUDA_ARCH__
    float4 v = __ldcg(reinterpret_cast<const float4*>(p));
    return {v.x, v.y, v.z, v.w};
#else
    return *p;
#endif
}

HD int popc32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    int c = 0;
    for (; x; x &= x - 1) ++c;
    return c;
#endif
}
HD int lowest_bit(uint32_t x) {  // index of the lowest set bit (x != 0)
#ifdef __CUDA_ARCH__
    return __ffs(int(x)) - 1;
#else
    int i = 0;
    while (!(x & 1u)) { x >>= 1; ++i; }
    return i;
#endif
}

// Mutable binary tree (ids as in bvh_common.h): boxes as two f4 per node (w unused).
struct TreeArrays {
    int n;
    int* left;
    int* right;
    int* parent;
    f4* lo;
    f4* hi;
    uint32_t* triCount;
    float* cost;  // SAH cost of each subtree, including the leaf-collapse option
};

HD float node_area(const TreeArrays& t, int x) {
    const f4 a = ldcg_f4(t.lo + x), b = ldcg_f4(t.hi + x);
    return box_area_rn(xyz(a), xyz(b));
}

// The restructure is split into four steps so that the GPU can run the dynamic program with a
// whole warp (lanes solve subsets of the same size in parallel) while the CPU runs the same
// steps one after another. Every subset's result is a pure function of smaller subsets, so the
// evaluation order does not change the answer: CPU and GPU trees stay identical.

struct Treelet {
    int nl;                                  // treelet leaves
    int leaves[TRBVH_MAX_LEAVES];
    int internals[TRBVH_MAX_LEAVES - 1];     // internals[0] = root
    vec3 lo[TRBVH_MAX_LEAVES], hi[TRBVH_MAX_LEAVES];
    uint32_t tris[TRBVH_MAX_LEAVES];
    float cost[TRBVH_MAX_LEAVES];
};

// 1. Grow the treelet: repeatedly expand the internal treelet leaf with the largest area.
HD void trbvh_form(const TreeArrays& t, int root, int treeletLeaves, Treelet& tl) {
    float leafArea[TRBVH_MAX_LEAVES];
    int nl = 2, ni = 1;
    tl.internals[0] = root;
    tl.leaves[0] = ldcg_i(t.left + root);
    tl.leaves[1] = ldcg_i(t.right + root);
    leafArea[0] = node_area(t, tl.leaves[0]);
    leafArea[1] = node_area(t, tl.leaves[1]);
    const int maxL = treeletLeaves < TRBVH_MAX_LEAVES ? treeletLeaves : TRBVH_MAX_LEAVES;
    while (nl < maxL) {
        int best = -1;
        float ba = -1.0f;
        for (int i = 0; i < nl; ++i)
            if (tl.leaves[i] < t.n - 1 && leafArea[i] > ba) {  // only internal nodes can be expanded
                ba = leafArea[i];
                best = i;
            }
        if (best < 0) break;
        const int x = tl.leaves[best];
        tl.internals[ni++] = x;
        tl.leaves[best] = ldcg_i(t.left + x);
        tl.leaves[nl] = ldcg_i(t.right + x);
        leafArea[best] = node_area(t, tl.leaves[best]);
        leafArea[nl] = node_area(t, tl.leaves[nl]);
        ++nl;
    }
    tl.nl = nl;
    for (int i = 0; i < nl; ++i) {
        tl.lo[i] = xyz(ldcg_f4(t.lo + tl.leaves[i]));
        tl.hi[i] = xyz(ldcg_f4(t.hi + tl.leaves[i]));
        tl.tris[i] = ldcg_u(t.triCount + tl.leaves[i]);
        tl.cost[i] = ldcg_f(t.cost + tl.leaves[i]);
    }
}

HD void trbvh_subset_box(const Treelet& tl, uint32_t S, vec3& lo, vec3& hi) {
    lo = make_vec3(RT_INF, RT_INF, RT_INF);
    hi = make_vec3(-RT_INF, -RT_INF, -RT_INF);
    for (int i = 0; i < tl.nl; ++i)
        if (S & (1u << i)) {
            lo = vmin(lo, tl.lo[i]);
            hi = vmax(hi, tl.hi[i]);
        }
}

// 2. Area and triangle count of subset S (1 <= S <= full); singletons get their cost here.
HD void trbvh_subset_init(const Treelet& tl, uint32_t S, float* area, uint32_t* tris, float* cost, uint8_t* part) {
    if (popc32(S) == 1) {
        const int i = lowest_bit(S);
        tris[S] = tl.tris[i];
        cost[S] = tl.cost[i];
        area[S] = 0.0f;
        part[S] = 0;
        return;
    }
    vec3 lo, hi;
    trbvh_subset_box(tl, S, lo, hi);
    uint32_t nt = 0;
    for (int i = 0; i < tl.nl; ++i)
        if (S & (1u << i)) nt += tl.tris[i];
    area[S] = box_area_rn(lo, hi);
    tris[S] = nt;
}

// 3. Optimal split of subset S (|S| >= 2); all proper subsets must be solved already.
HD void trbvh_subset_solve(uint32_t S, const float* area, const uint32_t* tris, float* cost, uint8_t* part, int maxLeaf,
                           bool sah) {
    const uint32_t low = S & (0u - S);
    float best = RT_INF;
    uint32_t bp = 0;
    for (uint32_t P = (S - 1u) & S; P; P = (P - 1u) & S) {
        if (!(P & low)) continue;  // each split once: P holds the lowest leaf
        const float c = fadd_rn(cost[P], cost[S ^ P]);
        if (c < best) {
            best = c;
            bp = P;
        }
    }
    const float interior = fadd_rn(fmul_rn(COLLAPSE_C_TRAV, area[S]), best);
    float c = interior;
    if (tris[S] <= uint32_t(maxLeaf)) {
        const float leaf = leaf_cost(tris[S], area[S]);
        if (!sah || leaf <= interior) c = leaf;
    }
    cost[S] = c;
    part[S] = uint8_t(bp);
}

// 4. Rewire the treelet's internal nodes to the optimal topology (the root keeps its id).
HD void trbvh_rewire(const TreeArrays& t, const Treelet& tl, const uint32_t* tris, const float* cost,
                     const uint8_t* part) {
    int stackNode[TRBVH_MAX_LEAVES];
    uint32_t stackSet[TRBVH_MAX_LEAVES];
    int sp = 0, nextFree = 1;
    stackNode[sp] = tl.internals[0];
    stackSet[sp++] = (1u << tl.nl) - 1u;
    while (sp > 0) {
        --sp;
        const int node = stackNode[sp];
        const uint32_t S = stackSet[sp];
        const uint32_t sets[2] = {uint32_t(part[S]), S ^ uint32_t(part[S])};
        int child[2];
        for (int k = 0; k < 2; ++k) {
            if (popc32(sets[k]) == 1) {
                child[k] = tl.leaves[lowest_bit(sets[k])];
            } else {
                child[k] = tl.internals[nextFree++];
                stackNode[sp] = child[k];
                stackSet[sp++] = sets[k];
            }
        }
        vec3 lo, hi;
        trbvh_subset_box(tl, S, lo, hi);
        t.left[node] = child[0];
        t.right[node] = child[1];
        t.parent[child[0]] = node;
        t.parent[child[1]] = node;
        t.lo[node] = {lo.x, lo.y, lo.z, 0.0f};
        t.hi[node] = {hi.x, hi.y, hi.z, 0.0f};
        t.triCount[node] = tris[S];
        t.cost[node] = cost[S];
    }
}

// The whole restructure, one step after another (CPU, and the one-thread GPU variant).
HD void trbvh_restructure(const TreeArrays& t, int root, int treeletLeaves, int maxLeaf, bool sah) {
    Treelet tl;
    trbvh_form(t, root, treeletLeaves, tl);
    float area[1 << TRBVH_MAX_LEAVES], cost[1 << TRBVH_MAX_LEAVES];
    uint32_t tris[1 << TRBVH_MAX_LEAVES];
    uint8_t part[1 << TRBVH_MAX_LEAVES];
    const uint32_t full = (1u << tl.nl) - 1u;
    for (uint32_t S = 1; S <= full; ++S) trbvh_subset_init(tl, S, area, tris, cost, part);
    for (uint32_t S = 1; S <= full; ++S)
        if (popc32(S) >= 2) trbvh_subset_solve(S, area, tris, cost, part, maxLeaf, sah);
    trbvh_rewire(t, tl, tris, cost, part);
}

// Minimum subtree size for restructuring in pass p: gamma doubles each pass.
HD uint32_t trbvh_gamma(int treeletLeaves, int pass) { return uint32_t(treeletLeaves) << pass; }

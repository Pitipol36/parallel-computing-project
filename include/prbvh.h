// Parallel reinsertion (PRBVH, after Meister & Bittner, EG 2018), shared by the CPU twin and
// the GPU kernels. It optimises any binary tree with one triangle per leaf before the SAH collapse.
//
// One round:
//   1. search: for every node X (whose parent is not the root) find the reinsertion target Y
//      that reduces the tree's SAH cost the most. Removing X deletes its parent P (the sibling S
//      takes P's place); inserting X next to Y creates a new node N = (X, Y) where Y was (P's id is
//      reused for N). The change of cost is computed exactly for a single move:
//        - P disappears: -A(P);
//        - P's ancestors below the lowest common ancestor (LCA) of P and Y shrink (they lose X);
//        - Y's ancestors below the LCA grow by X, and N adds A(X u Y);
//        - nodes at and above the LCA keep their boxes (they contain X before and after).
//      Moving X up next to one of its own ancestors a_i is the special case Y = a_i. The search is
//      a branch-and-bound walk of each subtree hanging off X's ancestor path: the cost of inserting
//      anywhere below a node b is at least the growth so far + A(X), which prunes most of the tree.
//   2. lock: every move with a positive gain tries to claim the six nodes whose pointers it
//      changes (X, P, S, P's parent, Y, Y's parent) with an atomic max of its priority key
//      (gain, then smaller node id): a strict total order, so the winners do not depend on timing.
//   3. apply: a move that holds all six of its locks is applied; winners touch disjoint nodes.
//   4. refit all boxes bottom-up, and undo the round if the total SAH cost went up (moves applied
//      together can interact through shared ancestors' boxes).
// Every step is deterministic, so the CPU twin and the GPU produce the same tree.
#pragma once

#include <cstdint>

#include "bvh_common.h"
#include "trbvh.h"  // ldcg_* loads, f4 helpers

constexpr int PRBVH_MAX_DEPTH = 64;   // longer ancestor paths are skipped (never in practice)
constexpr int PRBVH_STACK = 64;       // branch-and-bound stack; deeper candidates are skipped

struct PrbvhTree {
    int n;
    int* left;
    int* right;
    int* parent;
    f4* lo;
    f4* hi;
    uint32_t* triCount;
    uint32_t* dfsCount;
    float* cost;   // collapse-aware SAH cost (bvh_common.h), for the final collapse
    float* plain;  // plain SAH cost: C_trav * A(internal) + C_isect * A(leaf), summed bottom-up
};

HD AABB prbvh_box(const PrbvhTree& t, int x) { return {xyz(ldcg_f4(t.lo + x)), xyz(ldcg_f4(t.hi + x))}; }
HD AABB box_union(const AABB& a, const AABB& b) { return {vmin(a.bmin, b.bmin), vmax(a.bmax, b.bmax)}; }
HD float area_of(const AABB& b) { return box_area_rn(b.bmin, b.bmax); }

struct PrbvhMove {
    float gain;  // SAH cost reduction / C_trav (areas only: leaves never change)
    int target;  // Y, or -1
};

// Best reinsertion of node X (see the file comment). gain <= 0 means "no move".
// Position of X in the in-order traversal (leaves even, internal nodes odd). It depends only on the
// topology and child order, not on node ids, so builders that number nodes differently (H-PLOC's
// GPU atomic counter vs its CPU twin) still break ties the same way. O(depth).
HD uint32_t prbvh_rank(const PrbvhTree& t, int X) {
    uint32_t before = 0;  // leaves left of X's subtree
    for (int c = X, p = ldcg_i(t.parent + X); p >= 0; c = p, p = ldcg_i(t.parent + p))
        if (ldcg_i(t.right + p) == c) before += ldcg_u(t.triCount + ldcg_i(t.left + p));
    return X < t.n - 1 ? 2u * (before + ldcg_u(t.triCount + ldcg_i(t.left + X))) - 1u : 2u * before;
}

HD PrbvhMove prbvh_search(const PrbvhTree& t, int X) {
    PrbvhMove best = {0.0f, -1};
    const int P = ldcg_i(t.parent + X);
    if (P < 0) return best;
    const int G = ldcg_i(t.parent + P);
    if (G < 0) return best;  // P is the root: removing X would make S the root (not handled)
    const int S = ldcg_i(t.left + P) == X ? ldcg_i(t.right + P) : ldcg_i(t.left + P);
    const AABB bx = prbvh_box(t, X);
    const float aX = area_of(bx);

    // Ancestor path a_0 = P, a_1 = G, ..., a_m = root, with each a_j's shrink when X is removed.
    int chain[PRBVH_MAX_DEPTH];
    float shrink[PRBVH_MAX_DEPTH];
    int m = 0;
    chain[0] = P;
    shrink[0] = 0.0f;
    AABB cur = prbvh_box(t, S);  // what now sits where P was
    while (true) {
        const int a = chain[m];
        const int up = ldcg_i(t.parent + a);
        if (up < 0) break;
        if (m + 1 >= PRBVH_MAX_DEPTH) return best;
        const int sib = ldcg_i(t.left + up) == a ? ldcg_i(t.right + up) : ldcg_i(t.left + up);
        cur = box_union(cur, prbvh_box(t, sib));
        ++m;
        chain[m] = up;
        shrink[m] = fsub_rn(area_of(prbvh_box(t, up)), area_of(cur));
    }
    const float aP = area_of(prbvh_box(t, P));

    // (a) Move X up next to its ancestor a_i (i >= 1): N = (X, a_i'), whose box is a_i's old box.
    float removed = aP;  // A(P) + shrinks of a_1 .. a_i
    for (int i = 1; i <= m; ++i) {
        removed = fadd_rn(removed, shrink[i]);
        const float g = fsub_rn(removed, area_of(prbvh_box(t, chain[i])));
        if (g > best.gain || (g == best.gain && best.target >= 0 && prbvh_rank(t, chain[i]) < prbvh_rank(t, best.target))) {
            best.gain = g;
            best.target = chain[i];
        }
    }

    // (b) Insert inside the subtree hanging off a_i (LCA = a_i): branch and bound.
    int stackNode[PRBVH_STACK];
    float stackInd[PRBVH_STACK];
    float rbelow = aP;  // A(P) + shrinks strictly below the LCA
    for (int i = 0; i <= m; ++i) {
        if (i >= 2) rbelow = fadd_rn(rbelow, shrink[i - 1]);
        int sp = 0;
        if (i == 0) {
            // LCA = P: S takes P's place; S itself is the original position, so start below it.
            if (S >= t.n - 1) continue;  // S is a leaf
            const AABB bs = prbvh_box(t, S);
            const float growS = fsub_rn(area_of(box_union(bs, bx)), area_of(bs));
            stackNode[sp] = ldcg_i(t.right + S); stackInd[sp++] = growS;
            stackNode[sp] = ldcg_i(t.left + S); stackInd[sp++] = growS;
        } else {
            const int a = chain[i], below = chain[i - 1];
            const int c = ldcg_i(t.left + a) == below ? ldcg_i(t.right + a) : ldcg_i(t.left + a);
            stackNode[sp] = c;
            stackInd[sp++] = 0.0f;
        }
        while (sp > 0) {
            --sp;
            const int b = stackNode[sp];
            const float induced = stackInd[sp];
            const AABB bb = prbvh_box(t, b);
            const float aBX = area_of(box_union(bb, bx));
            const float g = fsub_rn(rbelow, fadd_rn(induced, aBX));
            if (g > best.gain || (g == best.gain && best.target >= 0 && prbvh_rank(t, b) < prbvh_rank(t, best.target))) {
                best.gain = g;
                best.target = b;
            }
            if (b >= t.n - 1) continue;  // leaf: no descendants
            const float ind2 = fadd_rn(induced, fsub_rn(aBX, area_of(bb)));
            // Any target below b costs at least ind2 + A(X): prune when that cannot beat best.
            if (fsub_rn(rbelow, fadd_rn(ind2, aX)) <= best.gain) continue;
            if (sp + 2 > PRBVH_STACK) continue;
            stackNode[sp] = ldcg_i(t.right + b); stackInd[sp++] = ind2;
            stackNode[sp] = ldcg_i(t.left + b); stackInd[sp++] = ind2;
        }
    }
    if (best.target < 0) best.gain = 0.0f;
    return best;
}

// Priority of a move: larger gain first, then smaller in-order rank. Positive floats order like their bits.
HD uint64_t prbvh_key(float gain, uint32_t rank) {
    return (uint64_t(float_as_uint(gain)) << 32) | uint64_t(0xffffffffu - rank);
}

// The six nodes whose pointers the move (X -> next to Y) changes; -1 entries are ignored.
HD void prbvh_move_nodes(const PrbvhTree& t, int X, int Y, int nodes[6]) {
    const int P = t.parent[X];
    nodes[0] = X;
    nodes[1] = P;
    nodes[2] = t.left[P] == X ? t.right[P] : t.left[P];  // S
    nodes[3] = t.parent[P];                               // G
    nodes[4] = Y;
    nodes[5] = t.parent[Y];                               // Q (-1 if Y is the root)
}

// Applies the move (pointers only; boxes are refitted afterwards). Returns the new root if the
// move put N at the top, else -1.
HD int prbvh_apply(const PrbvhTree& t, int X, int Y) {
    const int P = t.parent[X];
    const int S = t.left[P] == X ? t.right[P] : t.left[P];
    const int G = t.parent[P];
    // Remove: S takes P's place under G.
    if (t.left[G] == P) t.left[G] = S;
    else t.right[G] = S;
    t.parent[S] = G;
    // Insert: P becomes N = (X, Y) where Y was.
    const int Q = t.parent[Y];
    int newRoot = -1;
    if (Q >= 0) {
        if (t.left[Q] == Y) t.left[Q] = P;
        else t.right[Q] = P;
        t.parent[P] = Q;
    } else {
        t.parent[P] = -1;
        newRoot = P;
    }
    t.left[P] = X;
    t.right[P] = Y;
    t.parent[X] = P;
    t.parent[Y] = P;
    return newRoot;
}

// Bottom-up refit step for internal node p once both children are final.
HD void prbvh_refit_node(const PrbvhTree& t, int p, int maxLeaf, bool sah) {
    const int l = ldcg_i(t.left + p), r = ldcg_i(t.right + p);
    const AABB b = box_union(prbvh_box(t, l), prbvh_box(t, r));
    t.lo[p] = {b.bmin.x, b.bmin.y, b.bmin.z, 0.0f};
    t.hi[p] = {b.bmax.x, b.bmax.y, b.bmax.z, 0.0f};
    const uint32_t tris = ldcg_u(t.triCount + l) + ldcg_u(t.triCount + r);
    t.triCount[p] = tris;
    const float a = area_of(b);
    float c;
    t.dfsCount[p] = collapse_node(tris, a, ldcg_f(t.cost + l), ldcg_f(t.cost + r), ldcg_u(t.dfsCount + l),
                                  ldcg_u(t.dfsCount + r), maxLeaf, sah, c);
    t.cost[p] = c;
    t.plain[p] = fadd_rn(fmul_rn(COLLAPSE_C_TRAV, a), fadd_rn(ldcg_f(t.plain + l), ldcg_f(t.plain + r)));
}

HD void prbvh_refit_leaf(const PrbvhTree& t, int x) {
    const float a = area_of(prbvh_box(t, x));
    t.triCount[x] = 1;
    t.dfsCount[x] = 1;
    t.cost[x] = leaf_cost(1, a);
    t.plain[x] = leaf_cost(1, a);
}

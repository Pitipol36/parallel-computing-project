// Binary-tree intermediate format produced by the parallel builders (LBVH now; PLOC and
// TRBVH later), and the converter that turns it into the flattened depth-first BVHNode
// layout the render kernels already use. The per-node converter logic is __host__
// __device__, so the CPU and GPU converters are the same code.
//
// Node ids: [0, n-1) internal, [n-1, 2n-1) leaves; leaf n-1+k holds triangle k of the
// builder's triangle order (one triangle per leaf).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bvh.h"
#include "vecmath.h"

struct BinaryTreeView {
    int n = 0;     // triangles = leaves
    int root = 0;  // node id of the root (0 for LBVH; n-1 when n == 1)
    const int* left = nullptr;    // [n-1] children of internal nodes
    const int* right = nullptr;   // [n-1]
    const int* parent = nullptr;  // [2n-1], -1 at the root
    const uint32_t* triCount = nullptr;  // [2n-1] triangles under each node
    const uint32_t* dfsCount = nullptr;  // [2n-1] nodes the subtree becomes after leaf collapse (1 = leaf)
};

// Leaf collapse (shared by every binary-tree builder). Bottom-up, each node gets an SAH cost
//   leaf:     C_isect * N * area               (only when N <= maxLeaf)
//   interior: C_trav * area + cost(left) + cost(right)
// and becomes a leaf when that is cheaper (sah = true) or whenever N <= maxLeaf (sah = false).
// A node's collapsed size is 1 when it becomes a leaf, so dfsCount == 1 marks the leaves of
// the final tree: single triangles and collapsed subtrees. Arithmetic is round-to-nearest
// with no contraction, so CPU and GPU builders decide identically.
constexpr float COLLAPSE_C_TRAV = 1.2f;
constexpr float COLLAPSE_C_ISECT = 1.0f;

HD float leaf_cost(uint32_t tris, float area) { return fmul_rn(fmul_rn(COLLAPSE_C_ISECT, float(tris)), area); }

// Returns the node's collapsed size; writes its SAH cost.
HD uint32_t collapse_node(uint32_t tris, float area, float costLeft, float costRight, uint32_t dfsLeft,
                          uint32_t dfsRight, int maxLeaf, bool sah, float& cost) {
    const float interior = fadd_rn(fadd_rn(fmul_rn(COLLAPSE_C_TRAV, area), costLeft), costRight);
    if (tris <= uint32_t(maxLeaf)) {
        const float leaf = leaf_cost(tris, area);
        if (!sah || leaf <= interior) {
            cost = leaf;
            return 1u;
        }
    }
    cost = interior;
    return 1u + dfsLeft + dfsRight;
}

// A node survives the collapse when no ancestor became a leaf (a leaf's subtree is dropped).
HD bool conv_is_kept(const BinaryTreeView& t, int x) {
    while (x != t.root) {
        x = t.parent[x];
        if (t.dfsCount[x] == 1u) return false;
    }
    return true;
}

// Walks from node x to the root and returns its depth-first position among kept nodes,
// the offset of its first triangle in the output order, and its depth. Ancestors of a
// kept node are always kept interior nodes, so the sums need no global synchronisation.
HD void conv_offsets(const BinaryTreeView& t, int x, uint32_t& dfsIndex, uint32_t& triOffset, int& depth) {
    uint32_t idx = 0, off = 0;
    int d = 0;
    while (x != t.root) {
        const int p = t.parent[x];
        idx += 1;
        if (t.right[p] == x) {
            const int l = t.left[p];
            idx += t.dfsCount[l];
            off += t.triCount[l];
        }
        ++d;
        x = p;
    }
    dfsIndex = idx;
    triOffset = off;
    depth = d;
}

// Builds the output node for kept node x at position dfsIndex.
HD BVHNode conv_make_node(const BinaryTreeView& t, int x, uint32_t dfsIndex, uint32_t triOffset, AABB box) {
    BVHNode out;
    out.bmin[0] = box.bmin.x; out.bmin[1] = box.bmin.y; out.bmin[2] = box.bmin.z;
    out.bmax[0] = box.bmax.x; out.bmax[1] = box.bmax.y; out.bmax[2] = box.bmax.z;
    if (t.dfsCount[x] == 1u) {
        out.leftOrFirst = triOffset;
        out.count = t.triCount[x];
    } else {
        out.leftOrFirst = dfsIndex + 1 + t.dfsCount[t.left[x]];  // right child; left is dfsIndex + 1
        out.count = 0;
    }
    return out;
}

// Precomputed triangle as stored in BVH order (same as the SAH builder writes).
HD Tri make_tri(vec3 v0, vec3 v1, vec3 v2, uint32_t faceId) {
    Tri tr;
    vec3 e1 = v1 - v0, e2 = v2 - v0;
    tr.v0 = {v0.x, v0.y, v0.z, uint_as_float(faceId)};
    tr.e1 = {e1.x, e1.y, e1.z, 0.0f};
    tr.e2 = {e2.x, e2.y, e2.z, 0.0f};
    return tr;
}

// CPU stages 1-3 shared by the Morton-based builders: triangle boxes, Morton codes, and the
// (code, index) sort. order[k] = face at sorted position k; sortedKeys[k] = its code.
// Appends "bounds", "morton" and "sort" timings to stages.
void cpu_morton_sort(const Mesh& mesh, std::vector<AABB>& boxes, std::vector<uint32_t>& order,
                     std::vector<uint64_t>& sortedKeys, std::vector<std::pair<std::string, double>>& stages);

// CPU converter: collapse + depth-first flatten of a fitted binary tree.
//   boxes[2n-1]: node bounds; order[n]: original face index of each leaf's triangle.
BVH convert_binary_tree_cpu(const BinaryTreeView& t, const AABB* boxes, const uint32_t* order, const Mesh& mesh);

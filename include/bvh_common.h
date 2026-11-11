// Binary-tree intermediate format produced by the parallel builders (LBVH now; PLOC and
// TRBVH later), and the converter that turns it into the flattened depth-first BVHNode
// layout the render kernels already use. The per-node converter logic is __host__
// __device__, so the CPU and GPU converters are the same code.
//
// Node ids: [0, n-1) internal, [n-1, 2n-1) leaves; leaf n-1+k holds triangle k of the
// builder's triangle order (one triangle per leaf).
#pragma once

#include <cstdint>

#include "bvh.h"
#include "vecmath.h"

struct BinaryTreeView {
    int n = 0;     // triangles = leaves
    int root = 0;  // node id of the root (0 for LBVH; n-1 when n == 1)
    const int* left = nullptr;    // [n-1] children of internal nodes
    const int* right = nullptr;   // [n-1]
    const int* parent = nullptr;  // [2n-1], -1 at the root
    const uint32_t* triCount = nullptr;  // [2n-1] triangles under each node
    const uint32_t* dfsCount = nullptr;  // [2n-1] nodes the subtree becomes after leaf collapse
};

// After collapse, a node with at most maxLeaf triangles becomes one leaf.
HD uint32_t collapsed_count(uint32_t tris, uint32_t dfsLeft, uint32_t dfsRight, int maxLeaf) {
    return tris <= uint32_t(maxLeaf) ? 1u : 1u + dfsLeft + dfsRight;
}

// A node survives the collapse when it is the root or its parent stays an interior node.
HD bool conv_is_kept(const BinaryTreeView& t, int x, int maxLeaf) {
    return x == t.root || t.triCount[t.parent[x]] > uint32_t(maxLeaf);
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
HD BVHNode conv_make_node(const BinaryTreeView& t, int x, uint32_t dfsIndex, uint32_t triOffset, AABB box,
                          int maxLeaf) {
    BVHNode out;
    out.bmin[0] = box.bmin.x; out.bmin[1] = box.bmin.y; out.bmin[2] = box.bmin.z;
    out.bmax[0] = box.bmax.x; out.bmax[1] = box.bmax.y; out.bmax[2] = box.bmax.z;
    const uint32_t tris = t.triCount[x];
    if (tris <= uint32_t(maxLeaf)) {
        out.leftOrFirst = triOffset;
        out.count = tris;
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

// CPU converter: collapse + depth-first flatten of a fitted binary tree.
//   boxes[2n-1]: node bounds; order[n]: original face index of each leaf's triangle.
BVH convert_binary_tree_cpu(const BinaryTreeView& t, const AABB* boxes, const uint32_t* order, const Mesh& mesh,
                            int maxLeaf);

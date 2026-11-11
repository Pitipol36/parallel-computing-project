// CPU converter from a fitted binary tree to the flattened depth-first BVH. Uses the same
// per-node functions (bvh_common.h) as the GPU converter kernels.
#include <algorithm>

#include "bvh_common.h"

BVH convert_binary_tree_cpu(const BinaryTreeView& t, const AABB* boxes, const uint32_t* order, const Mesh& mesh,
                            int maxLeaf) {
    const int n = t.n;
    const int total = 2 * n - 1;
    BVH bvh;
    bvh.nodes.resize(t.dfsCount[t.root]);
    bvh.tris.resize(n);
    bvh.triOrder.resize(n);
    int maxDepth = 0, leaves = 0;

    for (int x = 0; x < total; ++x) {
        if (!conv_is_kept(t, x, maxLeaf)) continue;
        uint32_t idx, off;
        int depth;
        conv_offsets(t, x, idx, off, depth);
        bvh.nodes[idx] = conv_make_node(t, x, idx, off, boxes[x], maxLeaf);
        maxDepth = std::max(maxDepth, depth);
        if (t.triCount[x] <= uint32_t(maxLeaf)) ++leaves;
    }
    for (int k = 0; k < n; ++k) {
        uint32_t idx, pos;
        int depth;
        conv_offsets(t, n - 1 + k, idx, pos, depth);
        const uint32_t f = order[k];
        const Face& fc = mesh.faces[f];
        bvh.tris[pos] = make_tri(mesh.vertices[fc.a], mesh.vertices[fc.b], mesh.vertices[fc.c], f);
        bvh.triOrder[pos] = f;
    }
    bvh.maxDepth = maxDepth;
    bvh.leafCount = leaves;
    return bvh;
}

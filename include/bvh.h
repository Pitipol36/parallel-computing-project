// Flattened BVH shared by the CPU baseline and every GPU kernel.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mesh.h"
#include "vecmath.h"

// 32 bytes = two float4s. Nodes are stored in depth-first order, so an interior node's
// left child is always the next node (index + 1) and only the right child is stored.
//   interior: count == 0, leftOrFirst = index of the right child
//   leaf:     count  > 0, leftOrFirst = first triangle in BVH order
struct alignas(16) BVHNode {
    float bmin[3];
    uint32_t leftOrFirst;
    float bmax[3];
    uint32_t count;
};
static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes");

// Triangle in BVH order, precomputed for Möller–Trumbore: 48 bytes = three float4s.
//   v0.w holds the original face index (as raw bits) so hit IDs can be reported.
struct alignas(16) Tri {
    f4 v0, e1, e2;
};
static_assert(sizeof(Tri) == 48, "Tri must be 48 bytes");

constexpr int BVH_STACK_SIZE = 64;
constexpr int BVH_MAX_DEPTH = BVH_STACK_SIZE - 2;

struct BuildParams {
    int bins = 16;     // SAH bins per axis (16..32)
    int maxLeaf = 4;   // a node with this many triangles or fewer may become a leaf
    int plocRadius = 8;  // PLOC nearest-neighbour search: r clusters on each side
    bool sahCollapse = true;
    int trbvhPasses = 3;   // TRBVH restructuring passes (gamma doubles each pass)
    int trbvhTreelet = 7;  // TRBVH treelet leaves (2..7)
    bool trbvhWarp = true;
    int hplocRadius = 8;      // H-PLOC nearest-neighbour search within a cluster list
    int hplocThreshold = 16;  // H-PLOC: merge a node's list when it holds more clusters (<= 16)  // GPU TRBVH: one warp per treelet (false: one thread per treelet)  // binary-tree builders: collapse a subtree into a leaf only when SAH says so
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<Tri> tris;          // precomputed triangles in BVH order
    std::vector<uint32_t> triOrder; // triOrder[i] = original face index of tris[i]
    int maxDepth = 0;
    int leafCount = 0;
    double buildMs = 0.0;
};

BVH build_bvh(const Mesh& mesh, const BuildParams& params);

// Every triangle in exactly one leaf, children inside parents, depth within the stack.
bool validate_bvh(const BVH& bvh, std::string& report);

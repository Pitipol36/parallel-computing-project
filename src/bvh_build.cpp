// Binned SAH BVH builder (single-threaded, shared by the CPU and GPU renderers).
#include <algorithm>
#include <chrono>
#include <sstream>

#include "bvh.h"

namespace {

constexpr int MAX_BINS = 32;
constexpr float C_TRAV = 1.0f;  // traversal cost relative to one triangle test

struct Builder {
    const std::vector<AABB>& boxes;
    const std::vector<vec3>& cents;
    std::vector<uint32_t>& idx;
    std::vector<BVHNode>& nodes;
    BuildParams params;
    int maxDepth = 0;
    int leafCount = 0;

    static int bin_of(float c, float lo, float scale, int bins) {
        int b = int((c - lo) * scale);
        return b < bins - 1 ? b : bins - 1;
    }

    uint32_t build(uint32_t first, uint32_t count, int depth) {
        uint32_t me = uint32_t(nodes.size());
        nodes.emplace_back();
        maxDepth = std::max(maxDepth, depth);

        AABB nb = aabb_empty(), cb = aabb_empty();
        for (uint32_t i = first; i < first + count; ++i) {
            aabb_grow(nb, boxes[idx[i]]);
            aabb_grow(cb, cents[idx[i]]);
        }
        BVHNode& n = nodes[me];
        n.bmin[0] = nb.bmin.x; n.bmin[1] = nb.bmin.y; n.bmin[2] = nb.bmin.z;
        n.bmax[0] = nb.bmax.x; n.bmax[1] = nb.bmax.y; n.bmax[2] = nb.bmax.z;

        auto make_leaf = [&]() {
            nodes[me].leftOrFirst = first;
            nodes[me].count = count;
            ++leafCount;
            return me;
        };
        if (count <= 1 || depth >= BVH_MAX_DEPTH) return make_leaf();

        // ---- evaluate SAH over bins on each axis ----
        const int B = std::clamp(params.bins, 2, MAX_BINS);
        const float parentArea = aabb_area(nb);
        float bestCost = RT_INF;
        int bestAxis = -1, bestSplit = -1;
        for (int axis = 0; axis < 3; ++axis) {
            float lo = comp(cb.bmin, axis), hi = comp(cb.bmax, axis);
            if (!(hi > lo)) continue;
            float scale = float(B) / (hi - lo);
            uint32_t binCount[MAX_BINS] = {};
            AABB binBox[MAX_BINS];
            for (int b = 0; b < B; ++b) binBox[b] = aabb_empty();
            for (uint32_t i = first; i < first + count; ++i) {
                uint32_t p = idx[i];
                int b = bin_of(comp(cents[p], axis), lo, scale, B);
                ++binCount[b];
                aabb_grow(binBox[b], boxes[p]);
            }
            // Sweep from the left and from the right; plane s sits after bin s.
            float leftArea[MAX_BINS], rightArea[MAX_BINS];
            uint32_t leftN[MAX_BINS], rightN[MAX_BINS];
            AABB lb = aabb_empty(), rb = aabb_empty();
            uint32_t ln = 0, rn = 0;
            for (int s = 0; s < B - 1; ++s) {
                ln += binCount[s];
                aabb_grow(lb, binBox[s]);
                leftN[s] = ln;
                leftArea[s] = aabb_area(lb);
                rn += binCount[B - 1 - s];
                aabb_grow(rb, binBox[B - 1 - s]);
                rightN[B - 2 - s] = rn;
                rightArea[B - 2 - s] = aabb_area(rb);
            }
            for (int s = 0; s < B - 1; ++s) {
                if (leftN[s] == 0 || rightN[s] == 0) continue;
                float cost = C_TRAV + (leftArea[s] * leftN[s] + rightArea[s] * rightN[s]) / parentArea;
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = s;
                }
            }
        }

        const float leafCost = float(count);
        if (count <= uint32_t(params.maxLeaf) && (bestAxis < 0 || bestCost >= leafCost)) return make_leaf();

        uint32_t mid;
        if (bestAxis < 0) {
            mid = first + count / 2;  // all centroids coincide: split the list in half
        } else {
            float lo = comp(cb.bmin, bestAxis), hi = comp(cb.bmax, bestAxis);
            float scale = float(B) / (hi - lo);
            auto it = std::partition(idx.begin() + first, idx.begin() + first + count, [&](uint32_t p) {
                return bin_of(comp(cents[p], bestAxis), lo, scale, B) <= bestSplit;
            });
            mid = uint32_t(it - idx.begin());
            if (mid == first || mid == first + count) mid = first + count / 2;
        }

        build(first, mid - first, depth + 1);  // left child lands at me + 1
        uint32_t right = build(mid, first + count - mid, depth + 1);
        nodes[me].leftOrFirst = right;
        nodes[me].count = 0;
        return me;
    }
};

}  // namespace

BVH build_bvh(const Mesh& mesh, const BuildParams& params) {
    auto t0 = std::chrono::high_resolution_clock::now();
    const size_t T = mesh.faces.size();
    std::vector<AABB> boxes(T);
    std::vector<vec3> cents(T);
    std::vector<uint32_t> idx(T);
    for (size_t i = 0; i < T; ++i) {
        const Face& f = mesh.faces[i];
        AABB b = aabb_empty();
        aabb_grow(b, mesh.vertices[f.a]);
        aabb_grow(b, mesh.vertices[f.b]);
        aabb_grow(b, mesh.vertices[f.c]);
        boxes[i] = b;
        cents[i] = (b.bmin + b.bmax) * 0.5f;
        idx[i] = uint32_t(i);
    }

    BVH bvh;
    bvh.nodes.reserve(2 * T);
    Builder builder{boxes, cents, idx, bvh.nodes, params};
    builder.build(0, uint32_t(T), 0);
    bvh.nodes.shrink_to_fit();
    bvh.maxDepth = builder.maxDepth;
    bvh.leafCount = builder.leafCount;

    // Triangles in BVH order with precomputed edges.
    bvh.tris.resize(T);
    bvh.triOrder = idx;
    for (size_t i = 0; i < T; ++i) {
        const Face& f = mesh.faces[idx[i]];
        vec3 v0 = mesh.vertices[f.a], v1 = mesh.vertices[f.b], v2 = mesh.vertices[f.c];
        vec3 e1 = v1 - v0, e2 = v2 - v0;
        bvh.tris[i].v0 = {v0.x, v0.y, v0.z, uint_as_float(idx[i])};
        bvh.tris[i].e1 = {e1.x, e1.y, e1.z, 0.0f};
        bvh.tris[i].e2 = {e2.x, e2.y, e2.z, 0.0f};
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    bvh.buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return bvh;
}

bool validate_bvh(const BVH& bvh, std::string& report) {
    std::ostringstream os;
    const size_t N = bvh.nodes.size(), T = bvh.tris.size();
    bool ok = !bvh.nodes.empty();
    std::vector<uint8_t> covered(T, 0), visited(N, 0);
    size_t badContain = 0, badRange = 0, badId = 0;
    int deepest = 0;

    struct Item { uint32_t node; int depth; };
    std::vector<Item> stack = {{0, 0}};
    auto inside = [](const BVHNode& c, const BVHNode& p) {
        for (int k = 0; k < 3; ++k)
            if (c.bmin[k] < p.bmin[k] || c.bmax[k] > p.bmax[k]) return false;
        return true;
    };
    while (ok && !stack.empty()) {
        Item it = stack.back();
        stack.pop_back();
        if (it.node >= N || visited[it.node]) { ok = false; os << "  node " << it.node << " out of range or shared\n"; break; }
        visited[it.node] = 1;
        deepest = std::max(deepest, it.depth);
        const BVHNode& n = bvh.nodes[it.node];
        if (n.count > 0) {
            if (size_t(n.leftOrFirst) + n.count > T) { ++badRange; continue; }
            for (uint32_t i = n.leftOrFirst; i < n.leftOrFirst + n.count; ++i) {
                ++covered[i];
                // the leaf box must contain the triangle
                vec3 v0 = xyz(bvh.tris[i].v0), v1 = v0 + xyz(bvh.tris[i].e1), v2 = v0 + xyz(bvh.tris[i].e2);
                for (vec3 v : {v0, v1, v2})
                    for (int k = 0; k < 3; ++k) {
                        float c = comp(v, k);
                        // e1/e2 were rounded once, so allow a few ulps here
                        float tol = 1e-5f * (fabsf(c) + 1.0f);
                        if (c < n.bmin[k] - tol || c > n.bmax[k] + tol) ++badContain;
                    }
                if (float_as_uint(bvh.tris[i].v0.w) != bvh.triOrder[i]) ++badId;
            }
        } else {
            uint32_t l = it.node + 1, r = n.leftOrFirst;
            if (l >= N || r >= N) { ok = false; os << "  child index out of range\n"; break; }
            if (!inside(bvh.nodes[l], n) || !inside(bvh.nodes[r], n)) ++badContain;
            stack.push_back({r, it.depth + 1});
            stack.push_back({l, it.depth + 1});
        }
    }
    size_t missing = 0, dup = 0, unvisited = 0;
    for (uint8_t c : covered) { if (c == 0) ++missing; if (c > 1) ++dup; }
    for (uint8_t v : visited) if (!v) ++unvisited;
    std::vector<uint8_t> seen(T, 0);
    for (uint32_t f : bvh.triOrder) { if (f >= T || seen[f]) { ++badId; continue; } seen[f] = 1; }

    ok = ok && missing == 0 && dup == 0 && unvisited == 0 && badContain == 0 && badRange == 0 && badId == 0 &&
         deepest <= BVH_MAX_DEPTH;
    os << "  nodes=" << N << " leaves=" << bvh.leafCount << " tris=" << T << " depth=" << deepest
       << " (stack " << BVH_STACK_SIZE << ")\n"
       << "  missing=" << missing << " duplicated=" << dup << " unreachable=" << unvisited
       << " containment-errors=" << badContain << " range-errors=" << badRange << " id-errors=" << badId << "\n";
    report = os.str();
    return ok;
}

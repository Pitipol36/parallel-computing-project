// Full-sweep SAH builder (CPU, single-threaded): the quality reference. At every node the
// triangles are sorted by centroid on each axis and every split position is evaluated, with
// prefix and suffix bounding boxes; the cheapest split over all axes wins. The tree is built
// down to one triangle per leaf and then goes through the same SAH collapse and converter as
// the GPU builders, so leaf decisions are optimal for its topology. O(n log^2 n); it is a
// reference for tree quality, not a speed baseline (cpu_sah, binned, is that).
#include <algorithm>
#include <chrono>
#include <numeric>

#include "builders.h"
#include "bvh_common.h"

namespace {

using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct FullSah {
    int n;
    int maxLeaf;
    bool sah;
    const std::vector<AABB>& boxes;
    const std::vector<vec3>& cents;
    std::vector<uint32_t>& idx;                 // triangle ids, partitioned in place
    std::vector<uint32_t> work[3];              // per-axis sorted copies of the current slice
    std::vector<float> rightArea;               // suffix areas of the current slice
    std::vector<int> left, right, parent;
    std::vector<AABB> nodeBox;
    std::vector<uint32_t> triCount, dfsCount, order;
    std::vector<float> cost;
    int nextInternal = 0;

    FullSah(int n_, int maxLeaf_, bool sah_, const std::vector<AABB>& b, const std::vector<vec3>& c,
            std::vector<uint32_t>& i)
        : n(n_), maxLeaf(maxLeaf_), sah(sah_), boxes(b), cents(c), idx(i) {
        for (auto& w : work) w.resize(n);
        rightArea.resize(n + 1);
        left.resize(std::max(n - 1, 0));
        right.resize(std::max(n - 1, 0));
        parent.assign(2 * n - 1, -1);
        nodeBox.resize(2 * n - 1);
        triCount.resize(2 * n - 1);
        dfsCount.resize(2 * n - 1);
        cost.resize(2 * n - 1);
        order.resize(n);
    }

    int build(uint32_t first, uint32_t count) {
        if (count == 1) {
            const int x = n - 1 + int(first);  // leaf k = sorted position first
            order[first] = idx[first];
            nodeBox[x] = boxes[idx[first]];
            triCount[x] = 1;
            dfsCount[x] = 1;
            cost[x] = leaf_cost(1, box_area_rn(nodeBox[x].bmin, nodeBox[x].bmax));
            return x;
        }
        const int me = nextInternal++;
        double bestCost = 1e300;
        int bestAxis = 0;
        uint32_t bestSplit = 0;  // left = first bestSplit+1 entries
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<uint32_t>& w = work[axis];
            std::copy(idx.begin() + first, idx.begin() + first + count, w.begin() + first);
            std::sort(w.begin() + first, w.begin() + first + count, [&](uint32_t a, uint32_t b) {
                const float ca = comp(cents[a], axis), cb = comp(cents[b], axis);
                return ca != cb ? ca < cb : a < b;
            });
            AABB rb = aabb_empty();
            for (uint32_t i = count; i-- > 1;) {
                aabb_grow(rb, boxes[w[first + i]]);
                rightArea[i] = box_area_rn(rb.bmin, rb.bmax);
            }
            AABB lb = aabb_empty();
            for (uint32_t i = 0; i + 1 < count; ++i) {
                aabb_grow(lb, boxes[w[first + i]]);
                const double c = double(box_area_rn(lb.bmin, lb.bmax)) * double(i + 1) +
                                 double(rightArea[i + 1]) * double(count - i - 1);
                if (c < bestCost) {
                    bestCost = c;
                    bestAxis = axis;
                    bestSplit = i;
                }
            }
        }
        std::copy(work[bestAxis].begin() + first, work[bestAxis].begin() + first + count, idx.begin() + first);
        const int l = build(first, bestSplit + 1);
        const int r = build(first + bestSplit + 1, count - bestSplit - 1);
        left[me] = l;
        right[me] = r;
        parent[l] = parent[r] = me;
        AABB b = nodeBox[l];
        aabb_grow(b, nodeBox[r]);
        nodeBox[me] = b;
        triCount[me] = triCount[l] + triCount[r];
        dfsCount[me] = collapse_node(triCount[me], box_area_rn(b.bmin, b.bmax), cost[l], cost[r], dfsCount[l],
                                     dfsCount[r], maxLeaf, sah, cost[me]);
        return me;
    }
};

}  // namespace

BVH build_sah_full_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report) {
    const int n = int(mesh.faces.size());
    BuildReport rep;
    auto t0 = Clock::now();
    std::vector<AABB> boxes(n);
    std::vector<vec3> cents(n);
    for (int i = 0; i < n; ++i) {
        const Face& f = mesh.faces[i];
        AABB b = aabb_empty();
        aabb_grow(b, mesh.vertices[f.a]);
        aabb_grow(b, mesh.vertices[f.b]);
        aabb_grow(b, mesh.vertices[f.c]);
        boxes[i] = b;
        cents[i] = (b.bmin + b.bmax) * 0.5f;
    }
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    FullSah fs(n, std::max(1, params.maxLeaf), params.sahCollapse, boxes, cents, idx);
    const int root = fs.build(0, uint32_t(n));
    rep.stages.push_back({"sweep", ms_since(t0)});
    auto t1 = Clock::now();
    BinaryTreeView view;
    view.n = n;
    view.root = root;  // 0 for n > 1 (first internal node), else the only leaf
    view.left = fs.left.data();
    view.right = fs.right.data();
    view.parent = fs.parent.data();
    view.triCount = fs.triCount.data();
    view.dfsCount = fs.dfsCount.data();
    BVH bvh = convert_binary_tree_cpu(view, fs.nodeBox.data(), fs.order.data(), mesh);
    rep.stages.push_back({"convert", ms_since(t1)});
    rep.buildMs = ms_since(t0);
    bvh.buildMs = rep.buildMs;
    if (report) *report = rep;
    return bvh;
}

// Single-threaded LBVH (Karras 2012): the same algorithm and the same shared per-node code
// as the GPU builder, run one element at a time. It gives the apples-to-apples build
// speedup (parallelism, not a cheaper algorithm) and the reference tree that the GPU
// build must match node for node.
#include <algorithm>
#include <chrono>
#include <numeric>

#include "builders.h"
#include "bvh_common.h"
#include "morton.h"
#include "trbvh.h"

namespace {
using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

void cpu_morton_sort(const Mesh& mesh, std::vector<AABB>& boxes, std::vector<uint32_t>& order,
                     std::vector<uint64_t>& sorted, std::vector<std::pair<std::string, double>>& stages) {
    const int n = int(mesh.faces.size());
    // 1. Triangle bounds, centroids and the centroid bounds of the scene.
    auto t = Clock::now();
    boxes.assign(n, AABB());
    std::vector<vec3> cents(n);
    AABB cb = aabb_empty();
    for (int i = 0; i < n; ++i) {
        const Face& f = mesh.faces[i];
        AABB b = aabb_empty();
        aabb_grow(b, mesh.vertices[f.a]);
        aabb_grow(b, mesh.vertices[f.b]);
        aabb_grow(b, mesh.vertices[f.c]);
        boxes[i] = b;
        cents[i] = (b.bmin + b.bmax) * 0.5f;
        aabb_grow(cb, cents[i]);
    }
    stages.push_back({"bounds", ms_since(t)});

    // 2. 63-bit Morton codes.
    t = Clock::now();
    const MortonFrame frame = make_morton_frame(cb);
    std::vector<uint64_t> keys(n);
    for (int i = 0; i < n; ++i) keys[i] = morton_code(frame, cents[i]);
    stages.push_back({"morton", ms_since(t)});

    // 3. Sort (key, index) pairs. Ties keep index order, like the GPU's stable radix sort.
    t = Clock::now();
    order.resize(n);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return keys[a] != keys[b] ? keys[a] < keys[b] : a < b;
    });
    sorted.resize(n);
    for (int k = 0; k < n; ++k) sorted[k] = keys[order[k]];
    stages.push_back({"sort", ms_since(t)});
}

BVH build_lbvh_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report, bool restructure) {
    const int n = int(mesh.faces.size());
    const int maxLeaf = std::max(1, params.maxLeaf);
    BuildReport rep;
    auto tAll = Clock::now();
    std::vector<AABB> boxes;
    std::vector<uint32_t> order;
    std::vector<uint64_t> sorted;
    cpu_morton_sort(mesh, boxes, order, sorted, rep.stages);

    // 4. Hierarchy: one step per internal node.
    auto t = Clock::now();
    const int total = 2 * n - 1;
    std::vector<int> left(std::max(n - 1, 0)), right(std::max(n - 1, 0)), parent(total, -1);
    for (int i = 0; i < n - 1; ++i) {
        lbvh_internal_node(sorted.data(), n, i, left[i], right[i]);
        parent[left[i]] = i;
        parent[right[i]] = i;
    }
    rep.stages.push_back({"hierarchy", ms_since(t)});

    // 5. Bottom-up fit with arrival counters (the second child to arrive does the union).
    t = Clock::now();
    std::vector<AABB> nodeBox(total);
    std::vector<uint32_t> triCount(total), dfsCount(total);
    std::vector<float> cost(total);
    std::vector<uint8_t> visited(std::max(n - 1, 0), 0);
    auto fit = [&]() {
        std::fill(visited.begin(), visited.end(), uint8_t(0));
        for (int k = 0; k < n; ++k) {
            int x = lbvh_leaf_id(n, k);
            nodeBox[x] = boxes[order[k]];
            triCount[x] = 1;
            dfsCount[x] = 1;
            cost[x] = leaf_cost(1, box_area_rn(nodeBox[x].bmin, nodeBox[x].bmax));
            for (int p = parent[x]; p >= 0; p = parent[p]) {
                if (visited[p]++ == 0) break;  // first arrival: the sibling is not ready yet
                const int l = left[p], r = right[p];
                AABB b = nodeBox[l];
                aabb_grow(b, nodeBox[r]);
                nodeBox[p] = b;
                triCount[p] = triCount[l] + triCount[r];
                dfsCount[p] = collapse_node(triCount[p], box_area_rn(b.bmin, b.bmax), cost[l], cost[r], dfsCount[l],
                                            dfsCount[r], maxLeaf, params.sahCollapse, cost[p]);
            }
        }
    };
    fit();
    rep.stages.push_back({"fit", ms_since(t)});

    // TRBVH: treelet restructuring passes (same climb; the second arrival restructures), then
    // a re-fit for the new topology's collapse decisions.
    if (restructure && n > 1) {
        t = Clock::now();
        std::vector<f4> lo(total), hi(total);
        for (int x = 0; x < total; ++x) {
            lo[x] = {nodeBox[x].bmin.x, nodeBox[x].bmin.y, nodeBox[x].bmin.z, 0.0f};
            hi[x] = {nodeBox[x].bmax.x, nodeBox[x].bmax.y, nodeBox[x].bmax.z, 0.0f};
        }
        TreeArrays ta{n, left.data(), right.data(), parent.data(), lo.data(), hi.data(), triCount.data(), cost.data()};
        for (int pass = 0; pass < params.trbvhPasses; ++pass) {
            const uint32_t gamma = trbvh_gamma(params.trbvhTreelet, pass);
            std::fill(visited.begin(), visited.end(), uint8_t(0));
            for (int k = 0; k < n; ++k) {
                for (int p = parent[lbvh_leaf_id(n, k)]; p >= 0; p = parent[p]) {
                    if (visited[p]++ == 0) break;
                    if (triCount[p] >= gamma) trbvh_restructure(ta, p, params.trbvhTreelet, maxLeaf, params.sahCollapse);
                }
            }
        }
        fit();
        rep.stages.push_back({"restructure", ms_since(t)});
    }

    // 6. Collapse small subtrees into leaves and flatten depth-first.
    t = Clock::now();
    BinaryTreeView view;
    view.n = n;
    view.root = n > 1 ? 0 : 0;  // internal 0, or the only leaf (id n-1 = 0)
    view.left = left.data();
    view.right = right.data();
    view.parent = parent.data();
    view.triCount = triCount.data();
    view.dfsCount = dfsCount.data();
    BVH bvh = convert_binary_tree_cpu(view, nodeBox.data(), order.data(), mesh);
    rep.stages.push_back({"convert", ms_since(t)});

    rep.buildMs = ms_since(tAll);
    bvh.buildMs = rep.buildMs;
    if (report) *report = rep;
    return bvh;
}

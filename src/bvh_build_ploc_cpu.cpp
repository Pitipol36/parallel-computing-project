// Single-threaded PLOC (Meister & Bittner 2018), the same algorithm and shared per-element code
// (ploc.h) as the GPU builder, so the two produce node-for-node identical trees.
#include <algorithm>
#include <chrono>

#include "builders.h"
#include "bvh_common.h"
#include "ploc.h"

namespace {
using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

BVH build_ploc_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report) {
    const int n = int(mesh.faces.size());
    const int maxLeaf = std::max(1, params.maxLeaf);
    const int r = std::max(1, params.plocRadius);
    BuildReport rep;
    auto tAll = Clock::now();
    std::vector<AABB> triBoxes;
    std::vector<uint32_t> order;
    std::vector<uint64_t> keys;
    cpu_morton_sort(mesh, triBoxes, order, keys, rep.stages);

    // Clustering.
    auto t = Clock::now();
    const int total = 2 * n - 1;
    std::vector<int> left(std::max(n - 1, 0)), right(std::max(n - 1, 0)), parent(total, -1);
    std::vector<AABB> nodeBox(total);
    std::vector<uint32_t> triCount(total), dfsCount(total);
    std::vector<float> cost(total);
    std::vector<int> cNode(n), nn(n);
    std::vector<AABB> cBox(n);
    for (int k = 0; k < n; ++k) {
        const int x = n - 1 + k;
        nodeBox[x] = triBoxes[order[k]];
        triCount[x] = dfsCount[x] = 1;
        cost[x] = leaf_cost(1, box_area_rn(nodeBox[x].bmin, nodeBox[x].bmax));
        cNode[k] = x;
        cBox[k] = nodeBox[x];
    }
    int c = n, nextId = 0, iterations = 0;
    std::vector<int> nNode;
    std::vector<AABB> nBox;
    while (c > 1) {
        ++iterations;
        // 1. Nearest neighbour within +-r.
        for (int i = 0; i < c; ++i) {
            float bd = RT_INF;
            int bj = -1;
            const int j0 = std::max(0, i - r), j1 = std::min(c - 1, i + r);
            for (int j = j0; j <= j1; ++j) {
                if (j == i) continue;
                const float d = ploc_merged_area(cBox[i].bmin, cBox[i].bmax, cBox[j].bmin, cBox[j].bmax);
                if (bj < 0 || ploc_better(d, i, j, bd, bj)) { bd = d; bj = j; }
            }
            nn[i] = bj;
        }
        // 2 + 3. Merge mutual pairs (lower index creates the node) and compact in order.
        nNode.clear();
        nBox.clear();
        for (int i = 0; i < c; ++i) {
            const int j = nn[i];
            const bool mutual = nn[j] == i;
            if (mutual && i > j) continue;  // merged into cluster j
            if (mutual) {
                const int p = nextId++;
                const int l = cNode[i], rr = cNode[j];
                AABB b = cBox[i];
                aabb_grow(b, cBox[j]);
                left[p] = l;
                right[p] = rr;
                parent[l] = parent[rr] = p;
                nodeBox[p] = b;
                triCount[p] = triCount[l] + triCount[rr];
                dfsCount[p] = collapse_node(triCount[p], box_area_rn(b.bmin, b.bmax), cost[l], cost[rr], dfsCount[l],
                                            dfsCount[rr], maxLeaf, params.sahCollapse, cost[p]);
                nNode.push_back(p);
                nBox.push_back(b);
            } else {
                nNode.push_back(cNode[i]);
                nBox.push_back(cBox[i]);
            }
        }
        c = int(nNode.size());
        cNode.swap(nNode);
        cBox.swap(nBox);
    }
    rep.stages.push_back({"cluster", ms_since(t)});

    int root = n > 1 ? n - 2 : 0;
    if (params.prbvhIters > 0 && n > 2) {  // PRBVH reinsertion before the collapse
        t = Clock::now();
        std::vector<f4> lo(total), hi(total);
        for (int x = 0; x < total; ++x) {
            lo[x] = {nodeBox[x].bmin.x, nodeBox[x].bmin.y, nodeBox[x].bmin.z, 0.0f};
            hi[x] = {nodeBox[x].bmax.x, nodeBox[x].bmax.y, nodeBox[x].bmax.z, 0.0f};
        }
        PrbvhStats ps;
        root = prbvh_optimize_cpu(n, root, left.data(), right.data(), parent.data(), lo.data(), hi.data(),
                                  triCount.data(), dfsCount.data(), cost.data(), params, &ps);
        for (int x = 0; x < total; ++x) nodeBox[x] = {xyz(lo[x]), xyz(hi[x])};
        rep.optimizeMs = ms_since(t);
        rep.prbvhRounds = ps.iterations;
        rep.prbvhMoves = ps.moves;
        rep.stages.push_back({"optimize", rep.optimizeMs});
    }

    t = Clock::now();
    BinaryTreeView view;
    view.n = n;
    view.root = root;
    view.left = left.data();
    view.right = right.data();
    view.parent = parent.data();
    view.triCount = triCount.data();
    view.dfsCount = dfsCount.data();
    BVH bvh = convert_binary_tree_cpu(view, nodeBox.data(), order.data(), mesh);
    rep.stages.push_back({"convert", ms_since(t)});
    rep.iterations = iterations;
    rep.buildMs = ms_since(tAll);
    bvh.buildMs = rep.buildMs;
    if (report) *report = rep;
    return bvh;
}

// Single-threaded H-PLOC: the same bottom-up walk, list merging and shared code (hploc.h) as
// the GPU kernel, run one leaf at a time. Its flattened tree equals the GPU's.
#include <algorithm>
#include <chrono>

#include "builders.h"
#include "bvh_common.h"
#include "hploc.h"

namespace {
using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

BVH build_hploc_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report) {
    const int n = int(mesh.faces.size());
    const int maxLeaf = std::max(1, params.maxLeaf);
    const int radius = std::max(1, std::min(params.hplocRadius, HPLOC_MAX_LIST - 1));
    const int threshold = std::max(1, std::min(params.hplocThreshold, HPLOC_MAX_LIST / 2));
    BuildReport rep;
    auto tAll = Clock::now();
    std::vector<AABB> triBoxes;
    std::vector<uint32_t> order;
    std::vector<uint64_t> keys;
    cpu_morton_sort(mesh, triBoxes, order, keys, rep.stages);

    auto t = Clock::now();
    const int total = 2 * n - 1;
    std::vector<int> left(std::max(n - 1, 1)), right(std::max(n - 1, 1)), parent(total, -1);
    std::vector<f4> lo(total), hi(total);
    std::vector<uint32_t> triCount(total), dfsCount(total);
    std::vector<float> cost(total);
    HplocTree tree{n, left.data(), right.data(), parent.data(), lo.data(), hi.data(), triCount.data(), dfsCount.data(),
                   cost.data()};
    std::vector<int> listId(n), listSize(n), pb(std::max(n - 1, 1), -1);
    for (int k = 0; k < n; ++k) {
        hploc_init_leaf(tree, k, triBoxes[order[k]]);
        listId[k] = n - 1 + k;
        listSize[k] = 1;
    }
    int nextId = 0, root = n > 1 ? -1 : 0;
    int sId[HPLOC_MAX_LIST], nn[HPLOC_MAX_LIST], nId[HPLOC_MAX_LIST];
    vec3 sLo[HPLOC_MAX_LIST], sHi[HPLOC_MAX_LIST], nLo[HPLOC_MAX_LIST], nHi[HPLOC_MAX_LIST];
    for (int k = 0; k < n && n > 1; ++k) {
        int l = k, r = k;
        while (true) {
            bool isLeft;
            const int p = hploc_parent(keys.data(), n, l, r, isLeft);
            const int other = pb[p];
            pb[p] = isLeft ? l : r;
            if (other < 0) break;  // first arrival: the sibling is not finished yet
            const int ls = isLeft ? l : other, rs = p + 1;
            const int nL = ls, nR = isLeft ? other : r;
            const int sL = listSize[ls], sR = listSize[rs];
            int s = sL + sR;
            for (int i = 0; i < s; ++i) {
                sId[i] = i < sL ? listId[ls + i] : listId[rs + i - sL];
                sLo[i] = xyz(lo[sId[i]]);
                sHi[i] = xyz(hi[sId[i]]);
            }
            const bool isRoot = nL == 0 && nR == n - 1;
            const int target = isRoot ? 1 : threshold;
            while (s > target) {
                for (int i = 0; i < s; ++i) nn[i] = hploc_nn(sLo, sHi, s, i, radius);
                int ns = 0;
                for (int i = 0; i < s; ++i) {
                    const int j = nn[i];
                    const bool mutual = nn[j] == i;
                    if (mutual && i > j) continue;  // merged into entry j
                    nId[ns] = sId[i];
                    nLo[ns] = sLo[i];
                    nHi[ns] = sHi[i];
                    if (mutual) {
                        const int q = nextId++;
                        nLo[ns] = vmin(sLo[i], sLo[j]);
                        nHi[ns] = vmax(sHi[i], sHi[j]);
                        hploc_make_node(tree, q, sId[i], sId[j], nLo[ns], nHi[ns], maxLeaf, params.sahCollapse);
                        nId[ns] = q;
                    }
                    ++ns;
                }
                s = ns;
                for (int i = 0; i < s; ++i) {
                    sId[i] = nId[i];
                    sLo[i] = nLo[i];
                    sHi[i] = nHi[i];
                }
            }
            for (int i = 0; i < s; ++i) listId[nL + i] = sId[i];
            listSize[nL] = s;
            if (isRoot) {
                root = sId[0];
                break;
            }
            l = nL;
            r = nR;
        }
    }
    rep.stages.push_back({"cluster", ms_since(t)});

    t = Clock::now();
    std::vector<AABB> nodeBox(total);
    for (int x = 0; x < total; ++x) nodeBox[x] = {xyz(lo[x]), xyz(hi[x])};
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
    rep.buildMs = ms_since(tAll);
    bvh.buildMs = rep.buildMs;
    if (report) *report = rep;
    return bvh;
}

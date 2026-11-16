// Builder selection, tree-quality metrics (SAH cost, depth, leaf sizes) and tree comparison.
#include <algorithm>
#include <cstring>
#include <sstream>

#include "builders.h"

const char* builder_name(BuilderKind b) {
    switch (b) {
        case BuilderKind::CpuSah: return "cpu_sah";
        case BuilderKind::CpuLbvh: return "cpu_lbvh";
        case BuilderKind::GpuLbvh: return "gpu_lbvh";
        case BuilderKind::CpuPloc: return "cpu_ploc";
        case BuilderKind::GpuPloc: return "gpu_ploc";
        case BuilderKind::CpuTrbvh: return "cpu_trbvh";
        case BuilderKind::GpuTrbvh: return "gpu_trbvh";
        case BuilderKind::CpuSahFull: return "cpu_sah_full";
    }
    return "?";
}

bool parse_builder(const std::string& s, BuilderKind& b) {
    for (BuilderKind k : {BuilderKind::CpuSah, BuilderKind::CpuLbvh, BuilderKind::GpuLbvh, BuilderKind::CpuPloc,
                          BuilderKind::GpuPloc, BuilderKind::CpuTrbvh, BuilderKind::GpuTrbvh,
                          BuilderKind::CpuSahFull})
        if (s == builder_name(k)) { b = k; return true; }
    return false;
}

bool build_with(BuilderKind kind, const Mesh& mesh, const BuildParams& params, BVH& out, BuildReport& report,
                std::string& err) {
    report = BuildReport();
    switch (kind) {
        case BuilderKind::CpuSah:
            out = build_bvh(mesh, params);
            report.buildMs = out.buildMs;
            report.stages.push_back({"sah", out.buildMs});
            return true;
        case BuilderKind::CpuLbvh:
            out = build_lbvh_cpu(mesh, params, &report);
            return true;
        case BuilderKind::GpuLbvh:
            return build_lbvh_gpu(mesh, params, &out, nullptr, &report, err);
        case BuilderKind::CpuPloc:
            out = build_ploc_cpu(mesh, params, &report);
            return true;
        case BuilderKind::GpuPloc:
            return build_ploc_gpu(mesh, params, &out, nullptr, &report, err);
        case BuilderKind::CpuTrbvh:
            out = build_lbvh_cpu(mesh, params, &report, true);
            return true;
        case BuilderKind::GpuTrbvh:
            return build_lbvh_gpu(mesh, params, &out, nullptr, &report, err, true);
        case BuilderKind::CpuSahFull:
            out = build_sah_full_cpu(mesh, params, &report);
            return true;
    }
    err = "unknown builder";
    return false;
}

static double node_area(const BVHNode& n) {
    double ex = double(n.bmax[0]) - n.bmin[0], ey = double(n.bmax[1]) - n.bmin[1], ez = double(n.bmax[2]) - n.bmin[2];
    if (ex < 0) return 0;
    return 2.0 * (ex * ey + ey * ez + ez * ex);
}

TreeStats tree_stats(const BVH& bvh, float cTrav, float cIsect) {
    TreeStats s;
    if (bvh.nodes.empty()) return s;
    double interiorArea = 0, leafCost = 0;
    size_t leafTris = 0;
    struct Item { uint32_t node; int depth; };
    std::vector<Item> stack = {{0, 0}};
    while (!stack.empty()) {
        Item it = stack.back();
        stack.pop_back();
        const BVHNode& n = bvh.nodes[it.node];
        ++s.nodes;
        s.maxDepth = std::max(s.maxDepth, it.depth);
        if (n.count > 0) {
            ++s.leaves;
            leafTris += n.count;
            leafCost += node_area(n) * n.count;
        } else {
            ++s.interior;
            interiorArea += node_area(n);
            stack.push_back({n.leftOrFirst, it.depth + 1});
            stack.push_back({it.node + 1, it.depth + 1});
        }
    }
    const double rootArea = node_area(bvh.nodes[0]);
    s.sahCost = rootArea > 0 ? (cTrav * interiorArea + cIsect * leafCost) / rootArea : 0;
    s.avgLeafSize = s.leaves ? double(leafTris) / double(s.leaves) : 0;
    return s;
}

bool same_bvh(const BVH& a, const BVH& b, std::string& why) {
    std::ostringstream os;
    if (a.nodes.size() != b.nodes.size()) os << "node count " << a.nodes.size() << " vs " << b.nodes.size() << "; ";
    if (a.tris.size() != b.tris.size()) os << "triangle count " << a.tris.size() << " vs " << b.tris.size() << "; ";
    if (os.tellp() == 0) {
        size_t badNodes = 0, badTris = 0, badOrder = 0, firstBad = size_t(-1);
        for (size_t i = 0; i < a.nodes.size(); ++i)
            if (std::memcmp(&a.nodes[i], &b.nodes[i], sizeof(BVHNode)) != 0) {
                if (firstBad == size_t(-1)) firstBad = i;
                ++badNodes;
            }
        for (size_t i = 0; i < a.tris.size(); ++i) {
            if (std::memcmp(&a.tris[i], &b.tris[i], sizeof(Tri)) != 0) ++badTris;
            if (a.triOrder[i] != b.triOrder[i]) ++badOrder;
        }
        if (badNodes) os << badNodes << " nodes differ (first " << firstBad << "); ";
        if (badTris) os << badTris << " triangles differ; ";
        if (badOrder) os << badOrder << " order entries differ; ";
    }
    why = os.str();
    return why.empty();
}

// BVH builders selectable with --build, their stage timings, and tree-quality metrics.
//   cpu_sah   binned SAH, single-threaded (bvh_build.cpp): the quality reference
//   cpu_lbvh  Karras 2012 LBVH, single-threaded: the apples-to-apples build baseline
//   gpu_lbvh  the same LBVH as CUDA kernels; node-for-node identical to cpu_lbvh
//   cpu_ploc  PLOC bottom-up clustering, single-threaded
//   gpu_ploc  the same PLOC as CUDA kernels; node-for-node identical to cpu_ploc
//   cpu_trbvh LBVH + TRBVH treelet restructuring (Karras & Aila 2013), single-threaded
//   gpu_trbvh the same on the GPU, one warp per treelet; identical to cpu_trbvh
//   cpu_sah_full full-sweep SAH (every split on every axis): the tree-quality reference
//   cpu_hploc / gpu_hploc  H-PLOC (Benthin et al. 2024); one kernel launch on the GPU
// Every builder emits the same flattened BVH (BVHNode + Tri), so all render modes run
// unchanged on any tree.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "bvh.h"
#include "mesh.h"

enum class BuilderKind { CpuSah, CpuLbvh, GpuLbvh, CpuPloc, GpuPloc, CpuTrbvh, GpuTrbvh, CpuSahFull, CpuHploc, GpuHploc };

const char* builder_name(BuilderKind b);
bool parse_builder(const std::string& s, BuilderKind& b);

struct BuildReport {
    std::vector<std::pair<std::string, double>> stages;  // per-stage time in ms, in order
    double buildMs = 0;     // total build time (GPU: sum of kernel stages, cudaEvent timed)
    double uploadMs = 0;    // GPU: mesh upload before the build
    double downloadMs = 0;  // GPU: copying the finished tree back to the host
    int iterations = 0;     // PLOC: clustering iterations
    int prbvhRounds = 0;    // PRBVH: rounds kept
    int prbvhMoves = 0;     // PRBVH: reinsertions applied
    double optimizeMs = 0;  // PRBVH time (part of buildMs)
};

// A finished BVH resident on the GPU, in the render kernels' format.
struct DeviceBvh {
    void* nodes = nullptr;     // BVHNode[numNodes]
    void* tris = nullptr;      // Tri[numTris]
    void* triOrder = nullptr;  // uint32_t[numTris]
    uint32_t numNodes = 0, numTris = 0;
    int maxDepth = 0;
    int leafCount = 0;
    void release();
};

// restructure = true runs TRBVH treelet restructuring (params.trbvhPasses/trbvhTreelet) on the
// LBVH before the leaf collapse: that is the cpu_trbvh / gpu_trbvh builder.
BVH build_lbvh_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report, bool restructure = false);

// Reusable GPU LBVH builder: keeps its working and output buffers between builds, builds
// from vertices and faces that are already on the device, and can refit the last tree
// (new triangle positions, same topology). Used per frame by the dynamic-scene experiment.
class GpuLbvhBuilder {
public:
    GpuLbvhBuilder();
    ~GpuLbvhBuilder();
    GpuLbvhBuilder(const GpuLbvhBuilder&) = delete;
    GpuLbvhBuilder& operator=(const GpuLbvhBuilder&) = delete;

    // Allocates the buffers for numTris triangles ahead of time (build() also does it).
    bool reserve(int numTris, std::string& err);
    // dVerts: vec3[], dFaces: Face[numTris], both device pointers.
    bool build(const void* dVerts, const void* dFaces, int numTris, const BuildParams& params, BuildReport* report,
               std::string& err, bool restructure = false);
    // Recomputes every box and precomputed triangle of the last tree from new vertex
    // positions, bottom-up, keeping the topology. ms = kernel time.
    bool refit(const void* dVerts, const void* dFaces, double& ms, std::string& err);
    // The current tree (owned by the builder; do not release).
    const DeviceBvh& tree() const;
    bool download(BVH& out, std::string& err) const;
    // Hands the tree's buffers to the caller (who releases them); the builder forgets them.
    DeviceBvh take_tree();

private:
    struct Impl;
    Impl* impl_;
};

// GPU LBVH. hostOut (optional) receives a host copy of the tree; devOut (optional) keeps the
// device-resident tree (caller releases it). At least one must be given.
bool build_lbvh_gpu(const Mesh& mesh, const BuildParams& params, BVH* hostOut, DeviceBvh* devOut,
                    BuildReport* report, std::string& err, bool restructure = false);

// Reusable GPU PLOC builder (buffers persist; builds from device-resident vertices).
class GpuPlocBuilder {
public:
    GpuPlocBuilder();
    ~GpuPlocBuilder();
    GpuPlocBuilder(const GpuPlocBuilder&) = delete;
    GpuPlocBuilder& operator=(const GpuPlocBuilder&) = delete;
    bool reserve(int numTris, std::string& err);
    bool build(const void* dVerts, const void* dFaces, int numTris, const BuildParams& params, BuildReport* report,
               std::string& err);
    const DeviceBvh& tree() const;  // owned by the builder
    bool download(BVH& out, std::string& err) const;
    DeviceBvh take_tree();          // caller releases

private:
    struct Impl;
    Impl* impl_;
};

// PLOC (Meister & Bittner 2018), radius params.plocRadius. CPU and GPU trees are identical.
BVH build_ploc_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report);
bool build_ploc_gpu(const Mesh& mesh, const BuildParams& params, BVH* hostOut, DeviceBvh* devOut,
                    BuildReport* report, std::string& err);

// Reusable GPU H-PLOC builder (buffers persist; builds from device-resident vertices).
class GpuHplocBuilder {
public:
    GpuHplocBuilder();
    ~GpuHplocBuilder();
    GpuHplocBuilder(const GpuHplocBuilder&) = delete;
    GpuHplocBuilder& operator=(const GpuHplocBuilder&) = delete;
    bool reserve(int numTris, std::string& err);
    bool build(const void* dVerts, const void* dFaces, int numTris, const BuildParams& params, BuildReport* report,
               std::string& err);
    const DeviceBvh& tree() const;  // owned by the builder
    bool download(BVH& out, std::string& err) const;
    DeviceBvh take_tree();          // caller releases

private:
    struct Impl;
    Impl* impl_;
};

// H-PLOC: CPU twin and single-launch GPU builder; their flattened trees are identical.
BVH build_hploc_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report);
bool build_hploc_gpu(const Mesh& mesh, const BuildParams& params, BVH* hostOut, DeviceBvh* devOut,
                     BuildReport* report, std::string& err);

// PRBVH (parallel reinsertion) on a binary tree with one triangle per leaf, before the collapse.
// Arrays as in bvh_common.h (f4 boxes); returns the new root. cost/dfsCount are left ready for the
// converter. CPU twin and GPU version make the same moves.
struct PrbvhStats {
    int iterations = 0, moves = 0;
    bool rolledBack = false;
    double costBefore = 0, costAfter = 0;  // plain SAH sums (unnormalised)
};
struct f4;
int prbvh_optimize_cpu(int n, int root, int* left, int* right, int* parent, f4* lo, f4* hi, uint32_t* triCount,
                       uint32_t* dfsCount, float* cost, const BuildParams& params, PrbvhStats* stats);
bool prbvh_optimize_gpu(int n, int& root, int* left, int* right, int* parent, f4* lo, f4* hi, uint32_t* triCount,
                        uint32_t* dfsCount, float* cost, const BuildParams& params, PrbvhStats* stats, double& ms,
                        std::string& err);

// Full-sweep SAH reference builder (CPU, O(n log^2 n)).
BVH build_sah_full_cpu(const Mesh& mesh, const BuildParams& params, BuildReport* report);

// Builds with any builder into a host BVH (GPU builders download the tree).
bool build_with(BuilderKind kind, const Mesh& mesh, const BuildParams& params, BVH& out, BuildReport& report,
                std::string& err);

// ---- tree quality ----
constexpr float SAH_C_TRAV = 1.2f;   // cost of visiting an interior node
constexpr float SAH_C_ISECT = 1.0f;  // cost of one ray-triangle test

struct TreeStats {
    double sahCost = 0;  // [C_trav * sum SA(interior) + C_isect * sum SA(leaf) * N(leaf)] / SA(root)
    size_t nodes = 0, interior = 0, leaves = 0;
    int maxDepth = 0;
    double avgLeafSize = 0;
};

TreeStats tree_stats(const BVH& bvh, float cTrav = SAH_C_TRAV, float cIsect = SAH_C_ISECT);

// True when two BVHs are bit-for-bit identical (nodes, triangles and order).
bool same_bvh(const BVH& a, const BVH& b, std::string& why);

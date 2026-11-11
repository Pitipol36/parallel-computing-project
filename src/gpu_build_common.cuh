// Stages and helpers shared by every GPU BVH builder (LBVH, PLOC, ...):
//   MortonSort    triangle bounds + centroid reduction, 63-bit Morton codes, CUB radix sort
//   convert_tree  leaf collapse + depth-first flatten of a fitted binary tree into BVHNode/Tri
// plus small RAII/timing helpers. Internal to src/ (not a public header).
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <string>
#include <vector>

#include "bvh.h"
#include "bvh_common.h"
#include "mesh.h"
#include "morton.h"

namespace gpub {

constexpr int BLOCK = 256;
constexpr int REDUCE_BLOCKS = 240;  // 30 SMs x 8; a second pass folds these partials

inline int blocks_for(int n) { return (n + BLOCK - 1) / BLOCK; }

struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    Buf() = default;
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() { release(); }
    void release() {
        if (p) cudaFree(p);
        p = nullptr;
        bytes = 0;
    }
    // Grows (never shrinks) to at least `need` bytes.
    cudaError_t ensure(size_t need) {
        need = need ? need : 4;
        if (p && bytes >= need) return cudaSuccess;
        release();
        bytes = need;
        return cudaMalloc(&p, need);
    }
    void* take() {
        void* q = p;
        p = nullptr;
        bytes = 0;
        return q;
    }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

struct Events {
    std::vector<cudaEvent_t> ev;
    Events() = default;
    Events(const Events&) = delete;
    ~Events() { for (cudaEvent_t e : ev) cudaEventDestroy(e); }
    cudaEvent_t mark() {
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e);
        ev.push_back(e);
        return e;
    }
};

inline float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms;
}

using Clock = std::chrono::high_resolution_clock;
inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Stages 1-3 of every Morton-based builder. Buffers persist between builds.
struct MortonSort {
    Buf boxes, cents, partial, frame, keys, keysOut, vals, valsOut, sortTemp;
    cudaError_t ensure(int n);
    // Enqueues bounds -> Morton -> sort; records an event after each stage into ev
    // (3 events). After it: triBoxes()[f] is triangle f's box, order()[k] the face of
    // sorted position k, sortedKeys()[k] its code.
    cudaError_t run(const vec3* verts, const Face* faces, int n, Events& ev);
    const AABB* triBoxes() const { return boxes.as<AABB>(); }
    const uint32_t* order() const { return valsOut.as<uint32_t>(); }
    const uint64_t* sortedKeys() const { return keysOut.as<uint64_t>(); }
};

// A binary tree on the device (node ids as in bvh_common.h).
struct DeviceTree {
    int n, root;
    const int *left, *right, *parent;
    const uint32_t *triCount, *dfsCount;
    __host__ __device__ BinaryTreeView view() const {
        BinaryTreeView v;
        v.n = n; v.root = root; v.left = left; v.right = right; v.parent = parent;
        v.triCount = triCount; v.dfsCount = dfsCount;
        return v;
    }
};

// Flatten (leaves = nodes with dfsCount 1). boxLo/boxHi: node boxes [2n-1]; order[k]: face of leaf k. Writes
// outNodes (dfsCount[root] nodes), outTris/outOrder (n each); counters[0] = max depth,
// counters[1] = leaf count (both must be zeroed by the caller).
cudaError_t convert_tree(const DeviceTree& t, const float4* boxLo, const float4* boxHi, const uint32_t* order, const vec3* verts, const Face* faces, BVHNode* outNodes, Tri* outTris,
                         uint32_t* outOrder, int* counters);

// Hand-written exclusive prefix sum of 64-bit values (warp-shuffle block scan, recursive scan
// of the block sums, add back): out[i] = in[0] + ... + in[i-1]. Temp buffers persist.
struct ScanTemp {
    std::vector<Buf*> sums, scanned;  // one pair per recursion level
    ScanTemp() = default;
    ScanTemp(const ScanTemp&) = delete;
    ~ScanTemp() {
        for (Buf* b : sums) delete b;
        for (Buf* b : scanned) delete b;
    }
};
cudaError_t scan_exclusive_u64(const uint64_t* in, uint64_t* out, int n, ScanTemp& tmp, int level = 0);

}  // namespace gpub

#define CU(call)                                                                  \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            err = std::string(#call) + " failed: " + cudaGetErrorString(e_);      \
            return false;                                                         \
        }                                                                         \
    } while (0)

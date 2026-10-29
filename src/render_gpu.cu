// CUDA renderers. Four kernel variants for the ablation study:
//   gpu_brute          one thread per pixel, every ray tests every triangle
//   gpu_bvh_naive      BVH with array-of-structs nodes, indexed vertices, plain loads,
//                      per-test division, fixed child order
//   gpu_bvh_opt        float4 nodes/triangles via __ldg, precomputed inverse direction,
//                      nearer child first (the exact trace.h code the CPU runs)
//   gpu_bvh_persistent gpu_bvh_opt traversal, but a GPU-filling grid of warps pulls 8x4
//                      pixel tiles from a global atomic counter (load balancing)
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <string>

#include "render.h"
#include "trace.h"

namespace {

// ---------------- gpu_bvh_naive data layout and traversal ----------------

struct NodeAoS {  // 40 bytes, not 16-byte aligned
    AABB box;
    int left, right, first, count;
};

// Slab test that divides by the direction every time (no precomputed reciprocal).
HD float ray_aabb_div(const Ray& r, const AABB& b, float tmax) {
    float tx1 = (b.bmin.x - r.o.x) / r.d.x, tx2 = (b.bmax.x - r.o.x) / r.d.x;
    float t0 = rmin(tx1, tx2), t1 = rmax(tx1, tx2);
    float ty1 = (b.bmin.y - r.o.y) / r.d.y, ty2 = (b.bmax.y - r.o.y) / r.d.y;
    t0 = rmax(t0, rmin(ty1, ty2));
    t1 = rmin(t1, rmax(ty1, ty2));
    float tz1 = (b.bmin.z - r.o.z) / r.d.z, tz2 = (b.bmax.z - r.o.z) / r.d.z;
    t0 = rmax(t0, rmin(tz1, tz2));
    t1 = rmin(t1, rmax(tz1, tz2));
    if (t1 >= t0 && t1 > 0.0f && t0 <= tmax) return rmax(t0, 0.0f);
    return RT_INF;
}

struct NaiveTracer {
    const NodeAoS* nodes;
    const vec3* verts;
    const Face* faces;      // in BVH order
    const uint32_t* faceIds;

    HD void tri(int i, vec3& v0, vec3& e1, vec3& e2) const {
        Face f = faces[i];
        v0 = verts[f.a];
        e1 = verts[f.b] - v0;
        e2 = verts[f.c] - v0;
    }

    HD bool closest(const Ray& r, float tmax, Hit& h) const {
        h.t = tmax;
        h.prim = -1;
        int stack[BVH_STACK_SIZE];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const NodeAoS& n = nodes[stack[--sp]];
            if (ray_aabb_div(r, n.box, h.t) == RT_INF) continue;
            if (n.count > 0) {
                for (int i = n.first; i < n.first + n.count; ++i) {
                    vec3 v0, e1, e2;
                    tri(i, v0, e1, e2);
                    float t;
                    if (ray_triangle(r.o, r.d, v0, e1, e2, t)) hit_update(h, t, i);
                }
            } else {
                stack[sp++] = n.right;
                stack[sp++] = n.left;
            }
        }
        return h.prim >= 0;
    }

    HD bool any(const Ray& r, float tmax) const {
        int stack[BVH_STACK_SIZE];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0) {
            const NodeAoS& n = nodes[stack[--sp]];
            if (ray_aabb_div(r, n.box, tmax) == RT_INF) continue;
            if (n.count > 0) {
                for (int i = n.first; i < n.first + n.count; ++i) {
                    vec3 v0, e1, e2;
                    tri(i, v0, e1, e2);
                    float t;
                    if (ray_triangle(r.o, r.d, v0, e1, e2, t) && t < tmax) return true;
                }
            } else {
                stack[sp++] = n.right;
                stack[sp++] = n.left;
            }
        }
        return false;
    }

    HD vec3 normal(int prim) const {
        vec3 v0, e1, e2;
        tri(prim, v0, e1, e2);
        return cross(e1, e2);
    }
    HD int face_id(int prim) const { return int(faceIds[prim]); }
};

// ---------------- gpu_brute ----------------

struct BruteTracer {
    const f4* tris;
    int numTris;

    HD bool closest(const Ray& r, float tmax, Hit& h) const {
        h.t = tmax;
        h.prim = -1;
        for (int i = 0; i < numTris; ++i) {
            f4 v0 = ld4(tris + 3 * i), e1 = ld4(tris + 3 * i + 1), e2 = ld4(tris + 3 * i + 2);
            float t;
            if (ray_triangle(r.o, r.d, xyz(v0), xyz(e1), xyz(e2), t)) hit_update(h, t, i);
        }
        return h.prim >= 0;
    }
    HD bool any(const Ray& r, float tmax) const {
        for (int i = 0; i < numTris; ++i) {
            f4 v0 = ld4(tris + 3 * i), e1 = ld4(tris + 3 * i + 1), e2 = ld4(tris + 3 * i + 2);
            float t;
            if (ray_triangle(r.o, r.d, xyz(v0), xyz(e1), xyz(e2), t) && t < tmax) return true;
        }
        return false;
    }
    HD vec3 normal(int prim) const {
        return cross(xyz(ld4(tris + 3 * prim + 1)), xyz(ld4(tris + 3 * prim + 2)));
    }
    HD int face_id(int prim) const { return int(float_as_uint(ld4(tris + 3 * prim).w)); }
};

// ---------------- kernels ----------------

template <class Tracer>
__global__ void k_per_pixel(Tracer tr, ShadeParams sp, int y0, int y1, uint32_t* __restrict__ out,
                            int* __restrict__ ids) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sp.cam.width || y >= y1) return;
    int face;
    uint32_t c = shade_pixel(tr, sp, x, y, face);
    size_t i = size_t(y) * sp.cam.width + x;
    out[i] = c;
    if (ids) ids[i] = face;
}

constexpr int TILE_W = 8, TILE_H = 4;  // one warp = one 8x4 tile

__global__ void k_persistent(BvhTracer tr, ShadeParams sp, int tilesX, int numTiles, int* counter,
                             uint32_t* __restrict__ out, int* __restrict__ ids) {
    const int lane = threadIdx.x & 31;
    while (true) {
        int tile = 0;
        if (lane == 0) tile = atomicAdd(counter, 1);
        tile = __shfl_sync(0xffffffffu, tile, 0);
        if (tile >= numTiles) return;
        int x = (tile % tilesX) * TILE_W + (lane % TILE_W);
        int y = (tile / tilesX) * TILE_H + (lane / TILE_W);
        if (x < sp.cam.width && y < sp.cam.height) {
            int face;
            uint32_t c = shade_pixel(tr, sp, x, y, face);
            size_t i = size_t(y) * sp.cam.width + x;
            out[i] = c;
            if (ids) ids[i] = face;
        }
    }
}

// ---------------- host helpers ----------------

struct DevBuf {
    void* p = nullptr;
    size_t bytes = 0;
    ~DevBuf() { release(); }
    void release() {
        if (p) cudaFree(p);
        p = nullptr;
    }
    cudaError_t alloc(size_t n) {
        bytes = n;
        return cudaMalloc(&p, n ? n : 4);
    }
    cudaError_t upload(const void* src, size_t n) {
        cudaError_t e = alloc(n);
        if (e != cudaSuccess) return e;
        return cudaMemcpy(p, src, n, cudaMemcpyHostToDevice);
    }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

#define CU(call)                                                                  \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            err = std::string(#call) + " failed: " + cudaGetErrorString(e_);      \
            return false;                                                         \
        }                                                                         \
    } while (0)

using Clock = std::chrono::high_resolution_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

}  // namespace

const char* gpu_mode_name(GpuMode m) {
    switch (m) {
        case GpuMode::Brute: return "gpu_brute";
        case GpuMode::BvhNaive: return "gpu_bvh_naive";
        case GpuMode::BvhOpt: return "gpu_bvh_opt";
        case GpuMode::Persistent: return "gpu_bvh_persistent";
    }
    return "?";
}

bool parse_gpu_mode(const std::string& s, GpuMode& m) {
    for (GpuMode c : {GpuMode::Brute, GpuMode::BvhNaive, GpuMode::BvhOpt, GpuMode::Persistent})
        if (s == gpu_mode_name(c)) { m = c; return true; }
    return false;
}

bool gpu_info(std::string& desc, size_t& freeBytes, size_t& totalBytes) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    cudaMemGetInfo(&freeBytes, &totalBytes);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (sm_%d%d, %d SMs, %.1f GB)", p.name, p.major, p.minor,
                  p.multiProcessorCount, double(p.totalGlobalMem) / (1 << 30));
    desc = buf;
    return true;
}

size_t gpu_bytes_needed(const BVH& bvh, const Mesh& mesh, GpuMode mode, int width, int height, bool hitIds) {
    size_t px = size_t(width) * height;
    size_t img = px * 4 + (hitIds ? px * 4 : 0);
    switch (mode) {
        case GpuMode::Brute: return img + bvh.tris.size() * sizeof(Tri);
        case GpuMode::BvhNaive:
            return img + bvh.nodes.size() * sizeof(NodeAoS) + mesh.vertices.size() * sizeof(vec3) +
                   bvh.tris.size() * (sizeof(Face) + 4);
        default: return img + bvh.nodes.size() * sizeof(BVHNode) + bvh.tris.size() * sizeof(Tri);
    }
}

bool render_gpu(const BVH& bvh, const Mesh& mesh, GpuMode mode, const ShadeParams& sp,
                const GpuLaunch& launch, std::vector<uint32_t>& pixels, std::vector<int>* hitIds,
                GpuTimings& tm, std::string& err) {
    const int W = sp.cam.width, H = sp.cam.height;
    const size_t px = size_t(W) * H;

    // Naive layout is prepared on the host before timing starts (it is part of "build").
    std::vector<NodeAoS> aos;
    std::vector<Face> orderedFaces;
    if (mode == GpuMode::BvhNaive) {
        aos.resize(bvh.nodes.size());
        for (size_t i = 0; i < bvh.nodes.size(); ++i) {
            const BVHNode& n = bvh.nodes[i];
            NodeAoS& a = aos[i];
            a.box = {{n.bmin[0], n.bmin[1], n.bmin[2]}, {n.bmax[0], n.bmax[1], n.bmax[2]}};
            if (n.count > 0) { a.left = a.right = -1; a.first = int(n.leftOrFirst); a.count = int(n.count); }
            else             { a.left = int(i + 1); a.right = int(n.leftOrFirst); a.first = 0; a.count = 0; }
        }
        orderedFaces.resize(bvh.tris.size());
        for (size_t i = 0; i < bvh.tris.size(); ++i) orderedFaces[i] = mesh.faces[bvh.triOrder[i]];
    }

    cudaEvent_t ev0, ev1;
    CU(cudaEventCreate(&ev0));
    CU(cudaEventCreate(&ev1));
    struct EvGuard { cudaEvent_t a, b; ~EvGuard() { cudaEventDestroy(a); cudaEventDestroy(b); } } evg{ev0, ev1};

    auto tStart = Clock::now();
    DevBuf dNodes, dTris, dVerts, dFaces, dIds, dOut, dHit, dCounter;
    switch (mode) {
        case GpuMode::Brute:
            CU(dTris.upload(bvh.tris.data(), bvh.tris.size() * sizeof(Tri)));
            break;
        case GpuMode::BvhNaive:
            CU(dNodes.upload(aos.data(), aos.size() * sizeof(NodeAoS)));
            CU(dVerts.upload(mesh.vertices.data(), mesh.vertices.size() * sizeof(vec3)));
            CU(dFaces.upload(orderedFaces.data(), orderedFaces.size() * sizeof(Face)));
            CU(dIds.upload(bvh.triOrder.data(), bvh.triOrder.size() * sizeof(uint32_t)));
            break;
        default:
            CU(dNodes.upload(bvh.nodes.data(), bvh.nodes.size() * sizeof(BVHNode)));
            CU(dTris.upload(bvh.tris.data(), bvh.tris.size() * sizeof(Tri)));
            break;
    }
    CU(dOut.alloc(px * 4));
    if (hitIds) CU(dHit.alloc(px * 4));
    CU(cudaDeviceSynchronize());
    tm.uploadMs = ms_since(tStart);

    uint32_t* out = dOut.as<uint32_t>();
    int* ids = hitIds ? dHit.as<int>() : nullptr;
    dim3 block(launch.blockX, launch.blockY);
    auto grid_for = [&](int rows) { return dim3((W + block.x - 1) / block.x, (rows + block.y - 1) / block.y); };

    CU(cudaEventRecord(ev0));
    switch (mode) {
        case GpuMode::Brute: {
            BruteTracer tr{dTris.as<f4>(), int(bvh.tris.size())};
            int step = launch.bruteRowsPerLaunch > 0 ? launch.bruteRowsPerLaunch : H;
            for (int y0 = 0; y0 < H; y0 += step) {
                int y1 = y0 + step < H ? y0 + step : H;
                k_per_pixel<<<grid_for(y1 - y0), block>>>(tr, sp, y0, y1, out, ids);
            }
            break;
        }
        case GpuMode::BvhNaive: {
            NaiveTracer tr{dNodes.as<NodeAoS>(), dVerts.as<vec3>(), dFaces.as<Face>(), dIds.as<uint32_t>()};
            k_per_pixel<<<grid_for(H), block>>>(tr, sp, 0, H, out, ids);
            break;
        }
        case GpuMode::BvhOpt: {
            BvhTracer tr{dNodes.as<f4>(), dTris.as<f4>()};
            k_per_pixel<<<grid_for(H), block>>>(tr, sp, 0, H, out, ids);
            break;
        }
        case GpuMode::Persistent: {
            BvhTracer tr{dNodes.as<f4>(), dTris.as<f4>()};
            int threads = launch.blockX * launch.blockY;
            threads = (threads + 31) / 32 * 32;
            int perSM = 0, dev = 0, sms = 0;
            CU(cudaGetDevice(&dev));
            CU(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
            CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k_persistent, threads, 0));
            int tilesX = (W + TILE_W - 1) / TILE_W, tilesY = (H + TILE_H - 1) / TILE_H;
            CU(dCounter.alloc(sizeof(int)));
            CU(cudaMemsetAsync(dCounter.p, 0, sizeof(int)));
            k_persistent<<<sms * (perSM > 0 ? perSM : 1), threads>>>(tr, sp, tilesX, tilesX * tilesY,
                                                                     dCounter.as<int>(), out, ids);
            break;
        }
    }
    CU(cudaGetLastError());
    CU(cudaEventRecord(ev1));
    CU(cudaEventSynchronize(ev1));
    float kms = 0;
    CU(cudaEventElapsedTime(&kms, ev0, ev1));
    tm.kernelMs = kms;

    auto tDown = Clock::now();
    pixels.resize(px);
    CU(cudaMemcpy(pixels.data(), out, px * 4, cudaMemcpyDeviceToHost));
    if (hitIds) {
        hitIds->resize(px);
        CU(cudaMemcpy(hitIds->data(), ids, px * 4, cudaMemcpyDeviceToHost));
    }
    tm.downloadMs = ms_since(tDown);

    for (DevBuf* b : {&dNodes, &dTris, &dVerts, &dFaces, &dIds, &dOut, &dHit, &dCounter}) b->release();
    tm.e2eMs = ms_since(tStart);
    return true;
}

// Renderer entry points: the sequential CPU baseline and the CUDA kernel variants.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bvh.h"
#include "camera.h"
#include "mesh.h"

// ---- CPU (single-threaded baseline) ----
// hitIds may be null. Returns render-loop wall time in milliseconds.
double render_cpu(const BVH& bvh, const ShadeParams& sp, std::vector<uint32_t>& pixels,
                  std::vector<int>* hitIds);

// ---- GPU ----
enum class GpuMode { Brute, BvhNaive, BvhOpt, Persistent };

const char* gpu_mode_name(GpuMode m);
bool parse_gpu_mode(const std::string& s, GpuMode& m);

struct GpuTimings {
    double kernelMs = 0;    // cudaEvent time of the kernel launch(es) only
    double uploadMs = 0;    // alloc + host-to-device copy of scene data
    double downloadMs = 0;  // device-to-host copy of the image (and hit IDs)
    double e2eMs = 0;       // wall time of upload + kernel + download + free
};

struct GpuLaunch {
    int blockX = 16, blockY = 16;  // 2D block for the per-pixel kernels;
                                   // blockX*blockY threads per block for persistent
    int bruteRowsPerLaunch = 8;    // brute force is split into strips (Windows TDR watchdog)
};

// Uploads the scene, runs one kernel variant over the full image, downloads the result.
bool render_gpu(const BVH& bvh, const Mesh& mesh, GpuMode mode, const ShadeParams& sp,
                const GpuLaunch& launch, std::vector<uint32_t>& pixels, std::vector<int>* hitIds,
                GpuTimings& timings, std::string& err);

// Device name, compute capability and memory; false if no CUDA device.
bool gpu_info(std::string& desc, size_t& freeBytes, size_t& totalBytes);

// Rough device memory needed for a mode, in bytes (for the "does it fit" check).
size_t gpu_bytes_needed(const BVH& bvh, const Mesh& mesh, GpuMode mode, int width, int height,
                        bool hitIds);

// ---- Image helpers ----
bool write_png(const std::string& path, int w, int h, const std::vector<uint32_t>& rgba);

// ---- Verification ----
int run_unit_tests();  // returns number of failures

struct Comparison {
    size_t pixels = 0;
    size_t idMismatches = 0;  // primary-ray triangle IDs that differ
    size_t diffPixels = 0;    // pixels whose RGB differs at all
    int maxAbsDiff = 0;       // largest per-channel difference (0..255)
    double psnr = 0;          // dB over RGB; +inf when identical
};

Comparison compare_renders(const std::vector<uint32_t>& a, const std::vector<int>& idsA,
                           const std::vector<uint32_t>& b, const std::vector<int>& idsB);

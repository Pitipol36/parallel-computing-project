// Sequential CPU baseline: one thread, one pixel at a time, same traversal and shading
// code (trace.h) as the optimized GPU kernels.
#include <chrono>

#include "render.h"
#include "trace.h"

double render_cpu(const BVH& bvh, const ShadeParams& sp, std::vector<uint32_t>& pixels,
                  std::vector<int>* hitIds) {
    const int W = sp.cam.width, H = sp.cam.height;
    pixels.assign(size_t(W) * H, 0);
    if (hitIds) hitIds->assign(size_t(W) * H, -1);
    BvhTracer tr{reinterpret_cast<const f4*>(bvh.nodes.data()), reinterpret_cast<const f4*>(bvh.tris.data())};

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int face;
            size_t i = size_t(y) * W + x;
            pixels[i] = shade_pixel(tr, sp, x, y, face);
            if (hitIds) (*hitIds)[i] = face;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

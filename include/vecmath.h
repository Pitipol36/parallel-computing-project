// Shared math for the CPU baseline and the CUDA kernels.
// Everything here is __host__ __device__ so both sides run literally the same code.
// (Named vecmath.h rather than math.h so it never shadows the system <math.h>.)
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

constexpr float RT_INF = 1e30f;

// min/max: hardware fminf/fmaxf on the device; on the host MSVC turns fminf into a library
// call, so a compare-select is used instead (same result for the non-NaN values used here).
HD float rmin(float a, float b) {
#ifdef __CUDA_ARCH__
    return fminf(a, b);
#else
    return b < a ? b : a;
#endif
}
HD float rmax(float a, float b) {
#ifdef __CUDA_ARCH__
    return fmaxf(a, b);
#else
    return a < b ? b : a;
#endif
}

struct vec3 {
    float x, y, z;
};

HD vec3 make_vec3(float x, float y, float z) { return {x, y, z}; }
HD vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
HD vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
HD vec3 operator-(vec3 a) { return {-a.x, -a.y, -a.z}; }
HD vec3 operator*(vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
HD vec3 operator*(float s, vec3 a) { return {a.x * s, a.y * s, a.z * s}; }
HD vec3 operator*(vec3 a, vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
HD float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD vec3 cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
HD float length(vec3 a) { return sqrtf(dot(a, a)); }
HD vec3 normalize(vec3 a) { return a * (1.0f / sqrtf(dot(a, a))); }
HD vec3 vmin(vec3 a, vec3 b) { return {rmin(a.x, b.x), rmin(a.y, b.y), rmin(a.z, b.z)}; }
HD vec3 vmax(vec3 a, vec3 b) { return {rmax(a.x, b.x), rmax(a.y, b.y), rmax(a.z, b.z)}; }
HD float comp(vec3 a, int axis) { return axis == 0 ? a.x : (axis == 1 ? a.y : a.z); }

// 16-byte aligned 4-float vector, layout-compatible with CUDA's float4.
struct alignas(16) f4 {
    float x, y, z, w;
};
HD vec3 xyz(f4 a) { return {a.x, a.y, a.z}; }

HD uint32_t float_as_uint(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
#endif
}
HD float uint_as_float(uint32_t u) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}

// Read-only load: __ldg (read-only data cache) on the device, a plain load on the host.
HD f4 ld4(const f4* p) {
#ifdef __CUDA_ARCH__
    float4 v = __ldg(reinterpret_cast<const float4*>(p));
    return {v.x, v.y, v.z, v.w};
#else
    return *p;
#endif
}

struct AABB {
    vec3 bmin, bmax;
};

HD AABB aabb_empty() { return {{RT_INF, RT_INF, RT_INF}, {-RT_INF, -RT_INF, -RT_INF}}; }
HD void aabb_grow(AABB& b, vec3 p) {
    b.bmin = vmin(b.bmin, p);
    b.bmax = vmax(b.bmax, p);
}
HD void aabb_grow(AABB& b, const AABB& o) {
    b.bmin = vmin(b.bmin, o.bmin);
    b.bmax = vmax(b.bmax, o.bmax);
}
HD float aabb_area(const AABB& b) {
    vec3 e = b.bmax - b.bmin;
    if (e.x < 0.0f) return 0.0f;  // empty box
    return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
}

// A ray plus its precomputed reciprocal direction (zero components nudged to a tiny
// signed value so the slab test never computes 0 * inf).
struct Ray {
    vec3 o, d, inv;
};

HD float safe_rcp(float x) {
    const float eps = 1e-20f;
    if (fabsf(x) < eps) x = x < 0.0f ? -eps : eps;
    return 1.0f / x;
}

HD Ray make_ray(vec3 o, vec3 d) {
    Ray r;
    r.o = o;
    r.d = d;
    r.inv = {safe_rcp(d.x), safe_rcp(d.y), safe_rcp(d.z)};
    return r;
}

// Slab test with the precomputed reciprocal. Returns the entry distance, or RT_INF on a
// miss or when the box starts beyond tmax. "<=" (not "<") keeps nodes whose entry equals
// tmax, so exact-tie hits are always visited and the tie-break below is order independent.
HD float ray_aabb(const Ray& r, vec3 bmin, vec3 bmax, float tmax) {
    float tx1 = (bmin.x - r.o.x) * r.inv.x, tx2 = (bmax.x - r.o.x) * r.inv.x;
    float tmin_ = rmin(tx1, tx2), tmax_ = rmax(tx1, tx2);
    float ty1 = (bmin.y - r.o.y) * r.inv.y, ty2 = (bmax.y - r.o.y) * r.inv.y;
    tmin_ = rmax(tmin_, rmin(ty1, ty2));
    tmax_ = rmin(tmax_, rmax(ty1, ty2));
    float tz1 = (bmin.z - r.o.z) * r.inv.z, tz2 = (bmax.z - r.o.z) * r.inv.z;
    tmin_ = rmax(tmin_, rmin(tz1, tz2));
    tmax_ = rmin(tmax_, rmax(tz1, tz2));
    if (tmax_ >= tmin_ && tmax_ > 0.0f && tmin_ <= tmax) return rmax(tmin_, 0.0f);
    return RT_INF;
}

// Mรถllerโ€“Trumbore with precomputed edges e1 = v1 - v0, e2 = v2 - v0.
// Returns true and writes t when the ray hits the triangle at 0 < t.
HD bool ray_triangle(vec3 o, vec3 d, vec3 v0, vec3 e1, vec3 e2, float& t) {
    vec3 pvec = cross(d, e2);
    float det = dot(e1, pvec);
    if (fabsf(det) < 1e-30f) return false;  // ray parallel to the triangle plane
    float inv_det = 1.0f / det;
    vec3 tvec = o - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    vec3 qvec = cross(tvec, e1);
    float v = dot(d, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    t = dot(e2, qvec) * inv_det;
    return t > 0.0f;
}

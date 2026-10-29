// BVH traversal and shading shared by the CPU baseline and the gpu_bvh_opt /
// gpu_bvh_persistent kernels. The CPU and GPU compile this exact code.
#pragma once

#include "bvh.h"
#include "camera.h"
#include "rng.h"
#include "vecmath.h"

struct Hit {
    float t;
    int prim;  // index into the BVH-ordered triangle array, -1 = none
};

// Closest-hit update with a deterministic tie-break (smaller prim wins on equal t), so the
// result does not depend on traversal order: every renderer reports the same triangle.
HD void hit_update(Hit& h, float t, int prim) {
    if (t < h.t || (t == h.t && h.prim >= 0 && prim < h.prim)) {
        h.t = t;
        h.prim = prim;
    }
}

// Flattened BVH: float4-aligned nodes and triangles, read-only loads, precomputed inverse
// direction, nearer child first, and stacked entry distances to skip subtrees beyond the
// current closest hit.
struct BvhTracer {
    const f4* nodes;  // 2 per node
    const f4* tris;   // 3 per triangle

    HD bool closest(const Ray& r, float tmax, Hit& h) const {
        h.t = tmax;
        h.prim = -1;
        uint32_t stackNode[BVH_STACK_SIZE];
        float stackDist[BVH_STACK_SIZE];
        int sp = 0;

        uint32_t node = 0;
        f4 a = ld4(nodes), b = ld4(nodes + 1);
        if (ray_aabb(r, xyz(a), xyz(b), h.t) == RT_INF) return false;

        while (true) {
            uint32_t lf = float_as_uint(a.w), count = float_as_uint(b.w);
            if (count > 0) {
                for (uint32_t i = lf; i < lf + count; ++i) {
                    f4 v0 = ld4(tris + 3 * i), e1 = ld4(tris + 3 * i + 1), e2 = ld4(tris + 3 * i + 2);
                    float t;
                    if (ray_triangle(r.o, r.d, xyz(v0), xyz(e1), xyz(e2), t)) hit_update(h, t, int(i));
                }
            } else {
                uint32_t c1 = node + 1, c2 = lf;
                f4 a1 = ld4(nodes + 2 * c1), b1 = ld4(nodes + 2 * c1 + 1);
                f4 a2 = ld4(nodes + 2 * c2), b2 = ld4(nodes + 2 * c2 + 1);
                float d1 = ray_aabb(r, xyz(a1), xyz(b1), h.t);
                float d2 = ray_aabb(r, xyz(a2), xyz(b2), h.t);
                if (d2 < d1) {
                    float td = d1; d1 = d2; d2 = td;
                    uint32_t tc = c1; c1 = c2; c2 = tc;
                    f4 t4 = a1; a1 = a2; a2 = t4;
                    t4 = b1; b1 = b2; b2 = t4;
                }
                if (d1 != RT_INF) {
                    if (d2 != RT_INF) {
                        stackNode[sp] = c2;
                        stackDist[sp] = d2;
                        ++sp;
                    }
                    node = c1;
                    a = a1;
                    b = b1;
                    continue;
                }
            }
            // Pop the next subtree that can still contain a closer hit.
            while (true) {
                if (sp == 0) return h.prim >= 0;
                --sp;
                if (stackDist[sp] <= h.t) {
                    node = stackNode[sp];
                    a = ld4(nodes + 2 * node);
                    b = ld4(nodes + 2 * node + 1);
                    break;
                }
            }
        }
    }

    // Any hit with 0 < t < tmax (shadow and AO rays): exits on the first one found.
    HD bool any(const Ray& r, float tmax) const {
        uint32_t stack[BVH_STACK_SIZE];
        int sp = 0;
        uint32_t node = 0;
        f4 a = ld4(nodes), b = ld4(nodes + 1);
        if (ray_aabb(r, xyz(a), xyz(b), tmax) == RT_INF) return false;

        while (true) {
            uint32_t lf = float_as_uint(a.w), count = float_as_uint(b.w);
            if (count > 0) {
                for (uint32_t i = lf; i < lf + count; ++i) {
                    f4 v0 = ld4(tris + 3 * i), e1 = ld4(tris + 3 * i + 1), e2 = ld4(tris + 3 * i + 2);
                    float t;
                    if (ray_triangle(r.o, r.d, xyz(v0), xyz(e1), xyz(e2), t) && t < tmax) return true;
                }
            } else {
                uint32_t c1 = node + 1, c2 = lf;
                f4 a1 = ld4(nodes + 2 * c1), b1 = ld4(nodes + 2 * c1 + 1);
                f4 a2 = ld4(nodes + 2 * c2), b2 = ld4(nodes + 2 * c2 + 1);
                bool h1 = ray_aabb(r, xyz(a1), xyz(b1), tmax) != RT_INF;
                bool h2 = ray_aabb(r, xyz(a2), xyz(b2), tmax) != RT_INF;
                if (h1 || h2) {
                    if (h1 && h2) stack[sp++] = c2;
                    if (h1) { node = c1; a = a1; b = b1; }
                    else    { node = c2; a = a2; b = b2; }
                    continue;
                }
            }
            if (sp == 0) return false;
            node = stack[--sp];
            a = ld4(nodes + 2 * node);
            b = ld4(nodes + 2 * node + 1);
        }
    }

    HD vec3 normal(int prim) const {
        return cross(xyz(ld4(tris + 3 * prim + 1)), xyz(ld4(tris + 3 * prim + 2)));
    }
    HD int face_id(int prim) const { return int(float_as_uint(ld4(tris + 3 * prim).w)); }
};

// Cosine-weighted direction around unit normal n. The disk point is drawn by rejection
// sampling (no sin/cos), so the CPU and GPU produce bit-identical directions.
HD vec3 cosine_sample(vec3 n, Rng& rng) {
    float x = 0.0f, y = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float cx = 2.0f * rng.next() - 1.0f, cy = 2.0f * rng.next() - 1.0f;
        if (cx * cx + cy * cy < 1.0f) { x = cx; y = cy; break; }
    }
    float z = sqrtf(rmax(0.0f, 1.0f - x * x - y * y));
    // Branchless orthonormal basis (Duff et al. 2017).
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float bb = n.x * n.y * a;
    vec3 t = {1.0f + sign * n.x * n.x * a, sign * bb, -sign * n.x};
    vec3 s = {bb, sign + n.y * n.y * a, -n.y};
    return normalize(t * x + s * y + n * z);
}

HD uint32_t pack_rgba8(vec3 c) {
    // Gamma 2.0 (sqrt) instead of 2.2 keeps CPU and GPU bit-identical: sqrtf is IEEE exact.
    uint32_t r = uint32_t(sqrtf(rmin(rmax(c.x, 0.0f), 1.0f)) * 255.0f + 0.5f);
    uint32_t g = uint32_t(sqrtf(rmin(rmax(c.y, 0.0f), 1.0f)) * 255.0f + 0.5f);
    uint32_t b = uint32_t(sqrtf(rmin(rmax(c.z, 0.0f), 1.0f)) * 255.0f + 0.5f);
    return r | (g << 8) | (b << 16) | (255u << 24);
}

// One pixel: primary ray, shadow ray toward the light, aoSamples AO rays.
// Shading = albedo * (0.7 * lambert * shadow + 0.3 * (1 - occlusion)).
template <class Tracer>
HD uint32_t shade_pixel(const Tracer& tr, const ShadeParams& sp, int px, int py, int& hitFace) {
    Ray r = camera_ray(sp.cam, px, py);
    Hit h;
    if (!tr.closest(r, RT_INF, h)) {
        hitFace = -1;
        float k = 0.5f * (dot(r.d, sp.worldUp) + 1.0f);
        vec3 sky = make_vec3(0.95f, 0.95f, 0.97f) * (1.0f - k) + make_vec3(0.55f, 0.68f, 0.90f) * k;
        return pack_rgba8(sky);
    }
    hitFace = tr.face_id(h.prim);
    vec3 n = normalize(tr.normal(h.prim));
    if (dot(n, r.d) > 0.0f) n = -n;
    vec3 p = r.o + r.d * h.t + n * sp.eps;

    float direct = 0.0f;
    float ndl = dot(n, sp.lightDir);
    if (ndl > 0.0f && !tr.any(make_ray(p, sp.lightDir), RT_INF)) direct = ndl;

    float ambient = 1.0f;
    if (sp.aoSamples > 0) {
        uint32_t pixel = uint32_t(py) * uint32_t(sp.cam.width) + uint32_t(px);
        int occluded = 0;
        for (int s = 0; s < sp.aoSamples; ++s) {
            Rng rng = make_rng(pixel, uint32_t(s), sp.seed);
            vec3 dir = cosine_sample(n, rng);
            if (tr.any(make_ray(p, dir), sp.aoDist)) ++occluded;
        }
        ambient = 1.0f - float(occluded) / float(sp.aoSamples);
    }
    vec3 albedo = make_vec3(0.82f, 0.78f, 0.72f);
    return pack_rgba8(albedo * (0.7f * direct + 0.3f * ambient));
}

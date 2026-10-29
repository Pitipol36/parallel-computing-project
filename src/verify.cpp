// Unit tests (ray-triangle, ray-AABB, BVH, PLY, RNG) and CPU/GPU result comparison.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

#include "ply_loader.h"
#include "render.h"
#include "trace.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("    FAIL: %s\n", what);
    }
}

void test_ray_triangle() {
    std::printf("  ray-triangle\n");
    vec3 v0{0, 0, 0}, v1{1, 0, 0}, v2{0, 1, 0};
    vec3 e1 = v1 - v0, e2 = v2 - v0;
    float t = 0;
    check(ray_triangle({0.25f, 0.25f, 1}, {0, 0, -1}, v0, e1, e2, t) && fabsf(t - 1.0f) < 1e-6f, "hit inside");
    check(ray_triangle({0.25f, 0.25f, -1}, {0, 0, 1}, v0, e1, e2, t) && fabsf(t - 1.0f) < 1e-6f, "hit back face");
    check(!ray_triangle({0.8f, 0.8f, 1}, {0, 0, -1}, v0, e1, e2, t), "miss beyond hypotenuse");
    check(!ray_triangle({-0.1f, 0.5f, 1}, {0, 0, -1}, v0, e1, e2, t), "miss outside edge");
    check(ray_triangle({0.0f, 0.5f, 1}, {0, 0, -1}, v0, e1, e2, t), "hit exactly on edge u=0");
    check(ray_triangle({0.5f, 0.0f, 1}, {0, 0, -1}, v0, e1, e2, t), "hit exactly on edge v=0");
    check(ray_triangle({0.0f, 0.0f, 1}, {0, 0, -1}, v0, e1, e2, t), "hit exactly on vertex");
    check(!ray_triangle({-1, 0.25f, 0}, {1, 0, 0}, v0, e1, e2, t), "parallel ray in plane");
    check(!ray_triangle({0.25f, 0.25f, 1}, {1, 0, 0}, v0, e1, e2, t), "parallel ray above plane");
    check(!ray_triangle({0.25f, 0.25f, 1}, {0, 0, 1}, v0, e1, e2, t), "triangle behind origin");
}

void test_ray_aabb() {
    std::printf("  ray-AABB\n");
    vec3 bmin{-1, -1, -1}, bmax{1, 1, 1};
    Ray r = make_ray({0, 0, -5}, {0, 0, 1});
    check(fabsf(ray_aabb(r, bmin, bmax, RT_INF) - 4.0f) < 1e-6f, "axis-aligned hit (zero dir components)");
    check(ray_aabb(r, bmin, bmax, 3.0f) == RT_INF, "culled by tmax");
    check(ray_aabb(r, bmin, bmax, 4.0f) != RT_INF, "entry == tmax is kept");
    check(ray_aabb(make_ray({0, 0, -5}, {0, 0, -1}), bmin, bmax, RT_INF) == RT_INF, "box behind ray");
    check(ray_aabb(make_ray({0, 0, 0}, {0.3f, 0.5f, 0.81f}), bmin, bmax, RT_INF) == 0.0f, "origin inside box");
    check(ray_aabb(make_ray({0, 2, -5}, {0, 0, 1}), bmin, bmax, RT_INF) == RT_INF, "parallel ray misses slab");
    vec3 d = normalize({1, 1, 1});
    check(ray_aabb(make_ray({-3, -3, -3}, d), bmin, bmax, RT_INF) != RT_INF, "diagonal hit");
    check(ray_aabb(make_ray({-3, 3.5f, -3}, d), bmin, bmax, RT_INF) == RT_INF, "diagonal miss");
}

// Reference closest hit by testing every triangle, with the same tie-break.
bool brute_closest(const BVH& bvh, const Ray& r, Hit& h) {
    h.t = RT_INF;
    h.prim = -1;
    for (size_t i = 0; i < bvh.tris.size(); ++i) {
        const Tri& tr = bvh.tris[i];
        float t;
        if (ray_triangle(r.o, r.d, xyz(tr.v0), xyz(tr.e1), xyz(tr.e2), t)) hit_update(h, t, int(i));
    }
    return h.prim >= 0;
}

void test_bvh(const char* name, const Mesh& mesh, BuildParams bp) {
    std::printf("  BVH on %s (%zu tris, bins=%d, leaf=%d)\n", name, mesh.faces.size(), bp.bins, bp.maxLeaf);
    BVH bvh = build_bvh(mesh, bp);
    std::string rep;
    bool ok = validate_bvh(bvh, rep);
    if (!ok) std::printf("%s", rep.c_str());
    check(ok, "BVH structure valid");

    // Random rays from a shell around the scene: BVH must match brute force exactly.
    AABB b = mesh.bounds();
    vec3 c = (b.bmin + b.bmax) * 0.5f;
    float R = length(b.bmax - b.bmin);
    BvhTracer tr{reinterpret_cast<const f4*>(bvh.nodes.data()), reinterpret_cast<const f4*>(bvh.tris.data())};
    int mismatchClosest = 0, mismatchAny = 0, hits = 0;
    const int N = 4000;
    for (int i = 0; i < N; ++i) {
        Rng rng = make_rng(uint32_t(i), 7, 99);
        vec3 o = c + normalize({rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f}) * R;
        vec3 target = c + vec3{rng.next() - 0.5f, rng.next() - 0.5f, rng.next() - 0.5f} * R * 0.6f;
        Ray r = make_ray(o, normalize(target - o));
        Hit hb, hr;
        bool a = tr.closest(r, RT_INF, hb), ref = brute_closest(bvh, r, hr);
        hits += ref;
        if (a != ref || (a && (hb.prim != hr.prim || hb.t != hr.t))) ++mismatchClosest;
        float tmax = ref ? hr.t * 1.001f : RT_INF;
        if (tr.any(r, tmax) != ref) ++mismatchAny;
        if (ref && tr.any(r, hr.t * 0.999f)) ++mismatchAny;  // nothing can be closer than the closest
    }
    std::printf("    %d rays, %d hits, closest mismatches=%d, any-hit mismatches=%d\n", N, hits,
                mismatchClosest, mismatchAny);
    check(hits > N / 10, "enough rays hit the mesh");
    check(mismatchClosest == 0, "BVH closest hit == brute force");
    check(mismatchAny == 0, "BVH any hit consistent with closest hit");
}

void test_ply() {
    std::printf("  PLY round trip\n");
    Mesh m = make_icosphere(2);
    auto dir = std::filesystem::temp_directory_path();
    for (int binary = 0; binary < 2; ++binary) {
        std::string path = (dir / (binary ? "rt_test_bin.ply" : "rt_test_ascii.ply")).string();
        bool saved = binary ? save_ply_binary(path, m) : save_ply_ascii(path, m);
        check(saved, "PLY written");
        Mesh l;
        std::string err;
        bool loaded = load_ply(path, l, err);
        if (!loaded) std::printf("    %s\n", err.c_str());
        check(loaded, binary ? "binary PLY loads" : "ASCII PLY loads");
        check(l.vertices.size() == m.vertices.size() && l.faces.size() == m.faces.size(), "PLY counts match");
        float maxErr = 0;
        for (size_t i = 0; i < l.vertices.size() && i < m.vertices.size(); ++i)
            maxErr = fmaxf(maxErr, length(l.vertices[i] - m.vertices[i]));
        check(maxErr < 1e-6f, "PLY vertex positions match");
        bool facesOk = l.faces.size() == m.faces.size();
        for (size_t i = 0; facesOk && i < l.faces.size(); ++i)
            facesOk = l.faces[i].a == m.faces[i].a && l.faces[i].b == m.faces[i].b && l.faces[i].c == m.faces[i].c;
        check(facesOk, "PLY faces match");
        std::filesystem::remove(path);
    }
}

void test_rng() {
    std::printf("  RNG and hemisphere sampling\n");
    Rng a = make_rng(123, 4, 5), b = make_rng(123, 4, 5), c = make_rng(124, 4, 5);
    check(a.next() == b.next(), "RNG deterministic");
    check(a.next() != c.next(), "RNG differs per pixel");
    vec3 n = normalize({0.3f, -0.8f, 0.5f});
    bool ok = true;
    double mean = 0;
    for (int i = 0; i < 2000; ++i) {
        Rng r = make_rng(uint32_t(i), 0, 1);
        vec3 d = cosine_sample(n, r);
        float cosT = dot(d, n);
        ok = ok && fabsf(length(d) - 1.0f) < 1e-5f && cosT >= -1e-6f;
        mean += cosT;
    }
    mean /= 2000;
    check(ok, "AO directions are unit length and in the hemisphere");
    check(fabs(mean - 2.0 / 3.0) < 0.03, "AO directions are cosine distributed (mean cos ~ 2/3)");
}

}  // namespace

int run_unit_tests() {
    g_failures = 0;
    g_checks = 0;
    std::printf("Unit tests\n");
    test_ray_triangle();
    test_ray_aabb();
    test_bvh("icosphere(4)", make_icosphere(4), BuildParams{});
    test_bvh("test scene", make_test_scene(3), BuildParams{32, 2});
    test_bvh("icosphere(5)", make_icosphere(5), BuildParams{8, 8});
    test_ply();
    test_rng();
    std::printf("Unit tests: %d/%d checks passed -> %s\n", g_checks - g_failures, g_checks,
                g_failures ? "FAIL" : "PASS");
    return g_failures;
}

Comparison compare_renders(const std::vector<uint32_t>& a, const std::vector<int>& idsA,
                           const std::vector<uint32_t>& b, const std::vector<int>& idsB) {
    Comparison c;
    c.pixels = a.size();
    double se = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (idsA[i] != idsB[i]) ++c.idMismatches;
        bool diff = false;
        for (int ch = 0; ch < 3; ++ch) {
            int va = int((a[i] >> (8 * ch)) & 255), vb = int((b[i] >> (8 * ch)) & 255);
            int d = va > vb ? va - vb : vb - va;
            if (d) diff = true;
            if (d > c.maxAbsDiff) c.maxAbsDiff = d;
            se += double(d) * d;
        }
        c.diffPixels += diff;
    }
    double mse = se / (3.0 * double(a.size()));
    c.psnr = mse == 0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(255.0 * 255.0 / mse);
    return c;
}

// Unit tests (ray-triangle, ray-AABB, BVH, PLY, RNG) and CPU/GPU result comparison.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

#include <algorithm>

#include "builders.h"
#include "morton.h"
#include "ploc.h"
#include "hploc.h"
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

void check_bvh(const Mesh& mesh, const BVH& bvh);

void test_bvh(const char* name, const Mesh& mesh, BuildParams bp) {
    std::printf("  BVH on %s (%zu tris, bins=%d, leaf=%d)\n", name, mesh.faces.size(), bp.bins, bp.maxLeaf);
    check_bvh(mesh, build_bvh(mesh, bp));
}

// Structure check plus random rays against brute force, for a tree from any builder.
void check_bvh(const Mesh& mesh, const BVH& bvh) {
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

// ---------------------------------------------------------------- LBVH

void test_morton() {
    std::printf("  Morton codes\n");
    check(morton_expand21(0) == 0 && morton_expand21(1) == 1 && morton_expand21(2) == 8 && morton_expand21(3) == 9,
          "expand21 small values");
    check(morton_expand21(0x1fffff) == 0x1249249249249249ULL, "expand21 all 21 bits");
    check(morton_expand21(0xffffffffu) == 0x1249249249249249ULL, "expand21 ignores bits above 21");
    check(morton3d(1, 0, 0) == 4 && morton3d(0, 1, 0) == 2 && morton3d(0, 0, 1) == 1, "x, y, z bit order");
    check(morton3d(0x1fffff, 0x1fffff, 0x1fffff) == 0x7fffffffffffffffULL, "63-bit maximum");
    check(quantize21(-5.0f, 0.0f, 1.0f) == 0 && quantize21(7.0f, 0.0f, 1.0f) == 2097151u, "quantize clamps");
    check(quantize21(0.5f, 0.0f, 1.0f) == 1048576u, "quantize midpoint");
    check(quantize21(3.0f, 3.0f, 0.0f) == 0, "flat axis maps to 0");
    MortonFrame f = make_morton_frame({{0, 0, 0}, {2, 4, 0}});
    check(f.inv.x == 0.5f && f.inv.y == 0.25f && f.inv.z == 0.0f, "frame reciprocal extents");
}

void test_lbvh_delta() {
    std::printf("  LBVH delta with duplicate keys\n");
    const uint64_t keys[4] = {5, 5, 5, 9};
    check(lbvh_delta(keys, 4, 0, -1) == -1 && lbvh_delta(keys, 4, 3, 4) == -1, "out of range is -1");
    check(lbvh_delta(keys, 4, 0, 3) == clz64(5 ^ 9), "different keys use the key prefix");
    check(lbvh_delta(keys, 4, 0, 1) == 64 + 31 && lbvh_delta(keys, 4, 0, 2) == 64 + 30,
          "equal keys fall back to 64 + index prefix");
    check(lbvh_delta(keys, 4, 1, 2) == 64 + 30, "index prefix of 1 and 2");
    check(lbvh_delta(keys, 4, 0, 1) > lbvh_delta(keys, 4, 0, 3), "duplicates bind tighter than distinct keys");
}

struct RefNode { int first, last, split; };

// Independent top-down construction (Karras, "Thinking Parallel III"): the split of a range
// is the highest position where the common prefix with the first key changes.
void ref_build(const uint64_t* keys, int n, int first, int last, std::vector<RefNode>& out) {
    if (first == last) return;
    const int common = lbvh_delta(keys, n, first, last);
    int split = first, step = last - first;
    do {
        step = (step + 1) >> 1;
        int ns = split + step;
        if (ns < last && lbvh_delta(keys, n, first, ns) > common) split = ns;
    } while (step > 1);
    out.push_back({first, last, split});
    ref_build(keys, n, first, split, out);
    ref_build(keys, n, split + 1, last, out);
}

// Builds the Karras tree for sorted keys and checks it against the top-down reference:
// every internal node's range and split must appear in the reference, and vice versa.
bool karras_matches_reference(const std::vector<uint64_t>& keys) {
    const int n = int(keys.size());
    std::vector<int> left(n - 1), right(n - 1), first(2 * n - 1), last(2 * n - 1), parent(2 * n - 1, -1);
    for (int i = 0; i < n - 1; ++i) {
        lbvh_internal_node(keys.data(), n, i, left[i], right[i]);
        parent[left[i]] = i;
        parent[right[i]] = i;
    }
    for (int k = 0; k < n; ++k) first[n - 1 + k] = last[n - 1 + k] = k;
    // Ranges bottom-up: post-order from the root.
    std::vector<std::pair<int, bool>> st = {{0, false}};
    int visitedInternal = 0;
    while (!st.empty()) {
        auto [x, done] = st.back();
        st.pop_back();
        if (x >= n - 1) continue;
        if (!done) {
            ++visitedInternal;
            st.push_back({x, true});
            st.push_back({left[x], false});
            st.push_back({right[x], false});
        } else {
            first[x] = first[left[x]];
            last[x] = last[right[x]];
            if (last[left[x]] + 1 != first[right[x]]) return false;  // children must be adjacent
        }
    }
    if (visitedInternal != n - 1) return false;  // every internal node reachable from the root
    for (int x = 1; x < 2 * n - 1; ++x)
        if (parent[x] < 0) return false;
    std::vector<RefNode> ref;
    ref_build(keys.data(), n, 0, n - 1, ref);
    if (int(ref.size()) != n - 1) return false;
    std::vector<std::pair<int, int>> mine, theirs;
    for (int i = 0; i < n - 1; ++i) mine.push_back({first[i] * 100003 + last[i], last[left[i]]});
    for (const RefNode& r : ref) theirs.push_back({r.first * 100003 + r.last, r.split});
    std::sort(mine.begin(), mine.end());
    std::sort(theirs.begin(), theirs.end());
    return mine == theirs;
}

void test_lbvh_hierarchy() {
    std::printf("  LBVH hierarchy (hand-checked and against a top-down reference)\n");
    {   // n = 2: root with two leaves.
        const uint64_t k[2] = {1, 2};
        int l, r;
        lbvh_internal_node(k, 2, 0, l, r);
        check(l == 1 && r == 2, "n=2: root -> leaf 0, leaf 1");
    }
    {   // n = 3, keys 001 010 100: root splits off leaf 2; internal node 1 holds leaves 0 and 1.
        const uint64_t k[3] = {1, 2, 4};
        int l0, r0, l1, r1;
        lbvh_internal_node(k, 3, 0, l0, r0);
        lbvh_internal_node(k, 3, 1, l1, r1);
        check(l0 == 1 && r0 == 2 + 2, "n=3: root -> internal 1, leaf 2");
        check(l1 == 2 + 0 && r1 == 2 + 1, "n=3: internal 1 -> leaf 0, leaf 1");
    }
    {   // n = 4, all keys identical: the index fallback gives a balanced tree.
        const uint64_t k[4] = {7, 7, 7, 7};
        int l0, r0, l1, r1, l2, r2;
        lbvh_internal_node(k, 4, 0, l0, r0);
        lbvh_internal_node(k, 4, 1, l1, r1);
        lbvh_internal_node(k, 4, 2, l2, r2);
        check(l0 == 1 && r0 == 2, "identical keys: root -> internal 1, internal 2");
        check(l1 == 3 && r1 == 4 && l2 == 5 && r2 == 6, "identical keys: leaves paired by index");
    }
    // n = 8 hand-checkable, then random keys with many duplicates.
    check(karras_matches_reference({0, 1, 2, 3, 4, 5, 6, 7}), "n=8 consecutive keys");
    check(karras_matches_reference({3, 3, 3, 3, 3, 3, 3, 3}), "n=8 identical keys");
    bool ok = true;
    for (int trial = 0; trial < 60 && ok; ++trial) {
        int n = trial < 20 ? 2 + trial : 50 + 37 * trial;
        Rng rng = make_rng(uint32_t(trial), 11, 3);
        std::vector<uint64_t> keys(n);
        uint64_t range = trial % 3 == 0 ? 4 : (trial % 3 == 1 ? 1000 : (1ull << 62));
        for (auto& k : keys) k = uint64_t(double(rng.next()) * double(range));
        std::sort(keys.begin(), keys.end());
        ok = karras_matches_reference(keys);
        if (!ok) std::printf("    mismatch at trial %d (n=%d)\n", trial, n);
    }
    check(ok, "random sorted keys (with duplicates) match the reference");
}

void test_lbvh_builders() {
    std::string gdesc, err;
    size_t fb, tb;
    const bool gpu = gpu_info(gdesc, fb, tb);
    struct Case { const char* name; Mesh mesh; int leaf; };
    std::vector<Case> cases;
    cases.push_back({"icosphere(4)", make_icosphere(4), 4});
    cases.push_back({"test scene", make_test_scene(3), 2});
    cases.push_back({"icosphere(5)", make_icosphere(5), 8});
    Mesh one;
    one.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    one.faces = {{0, 1, 2}};
    cases.push_back({"single triangle", one, 4});
    Mesh dup = make_icosphere(2);  // the same triangles twice: duplicate Morton codes
    size_t nf = dup.faces.size();
    for (size_t i = 0; i < nf; ++i) dup.faces.push_back(dup.faces[i]);
    cases.push_back({"duplicated icosphere(2)", dup, 4});
    for (Case& c : cases) {
        BuildParams bp;
        bp.maxLeaf = c.leaf;
        std::printf("  LBVH on %s (%zu tris, leaf=%d)\n", c.name, c.mesh.faces.size(), c.leaf);
        BVH cpu = build_lbvh_cpu(c.mesh, bp, nullptr);
        if (c.mesh.faces.size() > 1) check_bvh(c.mesh, cpu);
        else {
            std::string rep;
            check(validate_bvh(cpu, rep), "single-triangle LBVH is valid");
        }
        if (!gpu) continue;
        BVH g;
        BuildReport rep;
        bool built = build_lbvh_gpu(c.mesh, bp, &g, nullptr, &rep, err);
        if (!built) std::printf("    %s\n", err.c_str());
        check(built, "GPU LBVH builds");
        std::string why;
        bool same = built && same_bvh(cpu, g, why);
        if (built && !same) std::printf("    %s\n", why.c_str());
        check(same, "GPU LBVH is node-for-node identical to the CPU LBVH");
        check(!built || (g.maxDepth == cpu.maxDepth && g.leafCount == cpu.leafCount), "GPU depth and leaf count match");
    }
}

void test_ploc_builders() {
    std::string gdesc, err;
    size_t fb, tb;
    const bool gpu = gpu_info(gdesc, fb, tb);
    // Tie-break: a strict total order on pairs, so the smallest pair is always mutual.
    check(ploc_better(1.0f, 5, 6, 2.0f, 4), "smaller distance wins");
    check(ploc_better(1.0f, 5, 4, 1.0f, 6), "equal distance: pair with the smaller low index wins");
    check(!ploc_better(1.0f, 5, 6, 1.0f, 4), "equal distance: (4,5) beats (5,6)");
    check(ploc_merged_area({0, 0, 0}, {1, 1, 1}, {1, 0, 0}, {2, 1, 1}) == 10.0f, "merged area of two unit cubes");
    struct Case { const char* name; Mesh mesh; int leaf, radius; };
    std::vector<Case> cases;
    cases.push_back({"icosphere(4)", make_icosphere(4), 4, 8});
    cases.push_back({"test scene", make_test_scene(3), 2, 1});
    cases.push_back({"icosphere(5), r=32", make_icosphere(5), 4, 32});
    Mesh one;
    one.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    one.faces = {{0, 1, 2}};
    cases.push_back({"single triangle", one, 4, 8});
    Mesh dup = make_icosphere(2);  // every triangle twice: zero-area merges and exact ties
    size_t nf = dup.faces.size();
    for (size_t i = 0; i < nf; ++i) dup.faces.push_back(dup.faces[i]);
    cases.push_back({"duplicated icosphere(2)", dup, 4, 4});
    for (Case& c : cases) {
        BuildParams bp;
        bp.maxLeaf = c.leaf;
        bp.plocRadius = c.radius;
        std::printf("  PLOC on %s (%zu tris, leaf=%d, r=%d)\n", c.name, c.mesh.faces.size(), c.leaf, c.radius);
        BuildReport crep;
        BVH cpu = build_ploc_cpu(c.mesh, bp, &crep);
        if (c.mesh.faces.size() > 1) check_bvh(c.mesh, cpu);
        else {
            std::string rep;
            check(validate_bvh(cpu, rep), "single-triangle PLOC is valid");
        }
        if (c.mesh.faces.size() > 100) {
            BVH lb = build_lbvh_cpu(c.mesh, bp, nullptr);
            double sp = tree_stats(cpu).sahCost, sl = tree_stats(lb).sahCost;
            std::printf("    %d iterations, SAH cost %.2f (LBVH %.2f)\n", crep.iterations, sp, sl);
            check(sp <= sl, "PLOC tree is at least as good as LBVH by SAH cost");
        }
        if (!gpu) continue;
        BVH g;
        BuildReport rep;
        bool built = build_ploc_gpu(c.mesh, bp, &g, nullptr, &rep, err);
        if (!built) std::printf("    %s\n", err.c_str());
        check(built, "GPU PLOC builds");
        std::string why;
        bool same = built && same_bvh(cpu, g, why);
        if (built && !same) std::printf("    %s\n", why.c_str());
        check(same, "GPU PLOC is node-for-node identical to the CPU PLOC");
        check(!built || rep.iterations == crep.iterations, "GPU and CPU take the same number of iterations");
    }
}

void test_hploc_builders() {
    std::string gdesc, err;
    size_t fb, tb;
    const bool gpu = gpu_info(gdesc, fb, tb);
    // Bottom-up parent choice: for keys 1, 2, 4 the leaves 0 and 1 pair up first (Apetrei).
    {
        const uint64_t k[3] = {1, 2, 4};
        bool left0, left1, left2;
        check(hploc_parent(k, 3, 0, 0, left0) == 0 && left0, "leaf 0 is the left child of split 0");
        check(hploc_parent(k, 3, 1, 1, left1) == 0 && !left1, "leaf 1 is the right child of split 0");
        check(hploc_parent(k, 3, 2, 2, left2) == 1 && !left2, "leaf 2 is the right child of split 1");
        bool l01;
        check(hploc_parent(k, 3, 0, 1, l01) == 1 && l01, "node [0,1] is the left child of split 1");
    }
    struct Case { const char* name; Mesh mesh; int leaf, radius, threshold; };
    std::vector<Case> cases;
    Mesh one;
    one.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    one.faces = {{0, 1, 2}};
    cases.push_back({"n = 1", one, 4, 8, 16});
    Mesh two = one;
    two.vertices.push_back({5, 5, 5});
    two.vertices.push_back({6, 5, 5});
    two.vertices.push_back({5, 6, 5});
    two.faces.push_back({3, 4, 5});
    cases.push_back({"n = 2", two, 1, 8, 16});
    Mesh three = two;
    three.vertices.push_back({0, 0, 9});
    three.vertices.push_back({1, 0, 9});
    three.vertices.push_back({0, 1, 9});
    three.faces.push_back({6, 7, 8});
    cases.push_back({"n = 3", three, 1, 8, 16});
    Mesh same;  // 200 copies of one triangle: every Morton key identical
    same.vertices = one.vertices;
    for (int i = 0; i < 200; ++i) same.faces.push_back({0, 1, 2});
    cases.push_back({"200 identical triangles", same, 4, 8, 16});
    cases.push_back({"icosphere(4)", make_icosphere(4), 4, 8, 16});
    cases.push_back({"icosphere(4), threshold 1", make_icosphere(4), 4, 8, 1});
    cases.push_back({"test scene, radius 2, threshold 4", make_test_scene(3), 2, 2, 4});
    cases.push_back({"icosphere(5), radius 16", make_icosphere(5), 4, 16, 16});
    Mesh dup = make_icosphere(2);
    size_t nf = dup.faces.size();
    for (size_t i = 0; i < nf; ++i) dup.faces.push_back(dup.faces[i]);
    cases.push_back({"duplicated icosphere(2)", dup, 4, 8, 16});
    for (Case& c : cases) {
        BuildParams bp;
        bp.maxLeaf = c.leaf;
        bp.hplocRadius = c.radius;
        bp.hplocThreshold = c.threshold;
        std::printf("  H-PLOC on %s (%zu tris, leaf=%d, r=%d, threshold=%d)\n", c.name, c.mesh.faces.size(), c.leaf,
                    c.radius, c.threshold);
        BVH cpu = build_hploc_cpu(c.mesh, bp, nullptr);
        if (c.mesh.faces.size() > 100 && c.mesh.faces.size() != 200) {
            check_bvh(c.mesh, cpu);
            const double sh = tree_stats(cpu).sahCost, sl = tree_stats(build_lbvh_cpu(c.mesh, bp, nullptr)).sahCost;
            std::printf("    SAH cost %.2f (LBVH %.2f)\n", sh, sl);
        } else {
            std::string rep;
            bool v = validate_bvh(cpu, rep);
            if (!v) std::printf("%s", rep.c_str());
            check(v, "small H-PLOC tree is valid");
        }
        if (!gpu) continue;
        BVH g;
        BuildReport rep;
        bool built = build_hploc_gpu(c.mesh, bp, &g, nullptr, &rep, err);
        if (!built) std::printf("    %s\n", err.c_str());
        check(built, "GPU H-PLOC builds");
        std::string why;
        bool same_ = built && same_bvh(cpu, g, why);
        if (built && !same_) std::printf("    %s\n", why.c_str());
        check(same_, "GPU H-PLOC flattens to the same tree as the CPU twin");
    }
}

void test_trbvh_builders() {
    std::string gdesc, err;
    size_t fb, tb;
    const bool gpu = gpu_info(gdesc, fb, tb);
    struct Case { const char* name; Mesh mesh; int leaf, treelet, passes; };
    std::vector<Case> cases;
    cases.push_back({"icosphere(4)", make_icosphere(4), 4, 7, 3});
    cases.push_back({"test scene, treelet 5", make_test_scene(3), 2, 5, 2});
    cases.push_back({"icosphere(5), 1 pass", make_icosphere(5), 4, 7, 1});
    Mesh tiny;  // 3 triangles: treelets smaller than the requested size
    tiny.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {5, 0, 0}, {6, 0, 0}, {5, 1, 0}, {0, 0, 5}, {1, 0, 5}, {0, 1, 5}};
    tiny.faces = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    cases.push_back({"three triangles", tiny, 1, 7, 3});
    Mesh dup = make_icosphere(2);
    size_t nf = dup.faces.size();
    for (size_t i = 0; i < nf; ++i) dup.faces.push_back(dup.faces[i]);
    cases.push_back({"duplicated icosphere(2)", dup, 4, 7, 3});
    for (Case& c : cases) {
        BuildParams bp;
        bp.maxLeaf = c.leaf;
        bp.trbvhTreelet = c.treelet;
        bp.trbvhPasses = c.passes;
        std::printf("  TRBVH on %s (%zu tris, leaf=%d, treelet=%d, passes=%d)\n", c.name, c.mesh.faces.size(), c.leaf,
                    c.treelet, c.passes);
        BVH cpu = build_lbvh_cpu(c.mesh, bp, nullptr, true);
        if (c.mesh.faces.size() > 10) {
            check_bvh(c.mesh, cpu);
        } else {
            std::string rep;
            check(validate_bvh(cpu, rep), "small TRBVH is valid");
        }
        BVH lb = build_lbvh_cpu(c.mesh, bp, nullptr, false);
        double st = tree_stats(cpu).sahCost, sl = tree_stats(lb).sahCost;
        std::printf("    SAH cost %.2f (LBVH %.2f)\n", st, sl);
        check(st <= sl * 1.0001, "restructuring never makes the tree worse by SAH cost");
        if (!gpu) continue;
        BVH g;
        BuildReport rep;
        bool built = build_lbvh_gpu(c.mesh, bp, &g, nullptr, &rep, err, true);
        if (!built) std::printf("    %s\n", err.c_str());
        check(built, "GPU TRBVH builds");
        std::string why;
        bool same = built && same_bvh(cpu, g, why);
        if (built && !same) std::printf("    %s\n", why.c_str());
        check(same, "GPU TRBVH is node-for-node identical to the CPU TRBVH");
    }
}

void test_prbvh() {
    std::string gdesc, err;
    size_t fb, tb;
    const bool gpu = gpu_info(gdesc, fb, tb);
    struct Case { const char* name; Mesh mesh; int leaf, iters; };
    std::vector<Case> cases;
    cases.push_back({"icosphere(4)", make_icosphere(4), 4, 8});
    cases.push_back({"test scene", make_test_scene(3), 2, 8});
    cases.push_back({"icosphere(5), 2 rounds", make_icosphere(5), 8, 2});
    Mesh tiny;
    tiny.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {5, 0, 0}, {6, 0, 0}, {5, 1, 0}, {0, 0, 5}, {1, 0, 5}, {0, 1, 5}};
    tiny.faces = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    cases.push_back({"three triangles", tiny, 1, 8});
    Mesh dup = make_icosphere(2);
    size_t nf = dup.faces.size();
    for (size_t i = 0; i < nf; ++i) dup.faces.push_back(dup.faces[i]);
    cases.push_back({"duplicated icosphere(2)", dup, 4, 8});
    const std::pair<BuilderKind, BuilderKind> pairs[] = {{BuilderKind::CpuLbvh, BuilderKind::GpuLbvh},
                                                         {BuilderKind::CpuPloc, BuilderKind::GpuPloc},
                                                         {BuilderKind::CpuHploc, BuilderKind::GpuHploc},
                                                         {BuilderKind::CpuTrbvh, BuilderKind::GpuTrbvh}};
    for (Case& c : cases) {
        for (auto [ck, gk] : pairs) {
            BuildParams bp;
            bp.maxLeaf = c.leaf;
            BVH base, cpu;
            BuildReport r0, r1;
            build_with(ck, c.mesh, bp, base, r0, err);
            bp.prbvhIters = c.iters;
            build_with(ck, c.mesh, bp, cpu, r1, err);
            const double s0 = tree_stats(base).sahCost, s1 = tree_stats(cpu).sahCost;
            std::printf("  PRBVH on %s after %s (%zu tris): SAH %.2f -> %.2f, %d rounds, %d moves\n", c.name,
                        builder_name(ck), c.mesh.faces.size(), s0, s1, r1.prbvhRounds, r1.prbvhMoves);
            if (c.mesh.faces.size() > 10) {
                check_bvh(c.mesh, cpu);
            } else {
                std::string rep;
                check(validate_bvh(cpu, rep), "small PRBVH tree is valid");
            }
            check(s1 <= s0 * 1.0001, "PRBVH never makes the tree worse by SAH cost");
            if (!gpu) continue;
            BVH g;
            BuildReport rg;
            const bool built = build_with(gk, c.mesh, bp, g, rg, err);
            if (!built) std::printf("    %s\n", err.c_str());
            check(built, "GPU builder with PRBVH builds");
            std::string why;
            const bool same = built && same_bvh(cpu, g, why);
            if (built && !same) std::printf("    %s\n", why.c_str());
            check(same, "GPU PRBVH is node-for-node identical to the CPU twin");
            check(rg.prbvhMoves == r1.prbvhMoves && rg.prbvhRounds == r1.prbvhRounds, "same PRBVH rounds and moves");
        }
    }
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
    test_morton();
    test_lbvh_delta();
    test_lbvh_hierarchy();
    test_lbvh_builders();
    test_ploc_builders();
    test_trbvh_builders();
    test_hploc_builders();
    test_prbvh();
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

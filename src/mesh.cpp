#include "mesh.h"

#include <map>
#include <utility>

Mesh make_icosphere(int subdiv) {
    Mesh m;
    const float t = 1.6180339887f;
    const vec3 base[12] = {{-1, t, 0}, {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                           {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
    for (const vec3& v : base) m.vertices.push_back(normalize(v));
    m.faces = {{0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
               {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
               {3, 8, 9},  {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};

    for (int s = 0; s < subdiv; ++s) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> mid;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
            auto it = mid.find(key);
            if (it != mid.end()) return it->second;
            uint32_t idx = uint32_t(m.vertices.size());
            m.vertices.push_back(normalize((m.vertices[a] + m.vertices[b]) * 0.5f));
            mid[key] = idx;
            return idx;
        };
        std::vector<Face> next;
        next.reserve(m.faces.size() * 4);
        for (const Face& f : m.faces) {
            uint32_t ab = midpoint(f.a, f.b), bc = midpoint(f.b, f.c), ca = midpoint(f.c, f.a);
            next.push_back({f.a, ab, ca});
            next.push_back({f.b, bc, ab});
            next.push_back({f.c, ca, bc});
            next.push_back({ab, bc, ca});
        }
        m.faces.swap(next);
    }
    return m;
}

Mesh make_test_scene(int subdiv) {
    Mesh m = make_icosphere(subdiv);
    // Ground quad touching the bottom of the sphere (y = -1), 2 triangles.
    uint32_t base = uint32_t(m.vertices.size());
    const float s = 3.0f, y = -1.0f;
    m.vertices.push_back({-s, y, -s});
    m.vertices.push_back({s, y, -s});
    m.vertices.push_back({s, y, s});
    m.vertices.push_back({-s, y, s});
    m.faces.push_back({base, base + 2, base + 1});
    m.faces.push_back({base, base + 3, base + 2});
    return m;
}

// ---------------------------------------------------------------- mixed-size scene

namespace {
void append_scaled(Mesh& out, const Mesh& m, float scale, vec3 offset) {
    const uint32_t base = uint32_t(out.vertices.size());
    AABB b = m.bounds();
    const vec3 c = {(b.bmin.x + b.bmax.x) * 0.5f, b.bmin.y, (b.bmin.z + b.bmax.z) * 0.5f};  // bottom centre
    for (const vec3& v : m.vertices) out.vertices.push_back((v - c) * scale + offset);
    for (const Face& f : m.faces) out.faces.push_back({f.a + base, f.b + base, f.c + base});
}

void append_box(Mesh& out, vec3 lo, vec3 hi) {
    const uint32_t b = uint32_t(out.vertices.size());
    for (int i = 0; i < 8; ++i)
        out.vertices.push_back({(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z});
    const uint32_t q[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    for (auto& f : q) {
        out.faces.push_back({b + f[0], b + f[1], b + f[2]});
        out.faces.push_back({b + f[0], b + f[2], b + f[3]});
    }
}

uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
float rnd(uint32_t& s) { s = hash32(s + 0x9e3779b9u); return float(s >> 8) * (1.0f / 16777216.0f); }
}  // namespace

Mesh make_mixed_scene(const Mesh& dragon, const Mesh& bunny) {
    Mesh out;
    const AABB db = dragon.bounds();
    const float D = length(db.bmax - db.bmin);  // dragon diagonal: the scene's unit
    // Ground: a 6D x 6D square made of 8 big triangles (a 2 x 2 grid of quads).
    const float g = 3.0f * D;
    const uint32_t gb = 0;
    for (int j = 0; j <= 2; ++j)
        for (int i = 0; i <= 2; ++i) out.vertices.push_back({-g + i * g, 0.0f, -g + j * g});
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            const uint32_t a = gb + j * 3 + i, b2 = a + 1, c = a + 3, d = a + 4;
            out.faces.push_back({a, c, d});
            out.faces.push_back({a, d, b2});
        }
    // Dragon at full size in the middle, bunnies at three very different scales.
    append_scaled(out, dragon, 1.0f, {0, 0, 0});
    const float bd = length(bunny.bounds().bmax - bunny.bounds().bmin);
    append_scaled(out, bunny, 0.08f * D / bd, {0.55f * D, 0, 0.25f * D});
    append_scaled(out, bunny, 0.6f * D / bd, {-1.1f * D, 0, -0.6f * D});
    append_scaled(out, bunny, 1.6f * D / bd, {1.6f * D, 0, -1.7f * D});
    // 150 thin pillars of random height, scattered over the ground (deterministic).
    uint32_t s = 12345u;
    for (int p = 0; p < 150; ++p) {
        const float x = (rnd(s) * 2 - 1) * 2.7f * D, z = (rnd(s) * 2 - 1) * 2.7f * D;
        if (std::fabs(x) < 0.7f * D && std::fabs(z) < 0.5f * D) continue;  // keep clear of the dragon
        const float w = (0.006f + 0.02f * rnd(s)) * D, h = (0.2f + 1.4f * rnd(s)) * D;
        append_box(out, {x - w, 0.0f, z - w}, {x + w, h, z + w});
    }
    return out;
}

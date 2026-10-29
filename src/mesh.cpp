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

// Indexed triangle mesh as loaded from disk (or generated procedurally).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vecmath.h"

struct Face {
    uint32_t a, b, c;
};

struct Mesh {
    std::vector<vec3> vertices;
    std::vector<Face> faces;

    AABB bounds() const {
        AABB b = aabb_empty();
        for (const vec3& v : vertices) aabb_grow(b, v);
        return b;
    }
};

// Icosphere with 20 * 4^subdiv triangles, centred at the origin, radius 1.
Mesh make_icosphere(int subdiv);

// Small test scene: an icosphere sitting on a ground quad, so shadows and AO show up.
Mesh make_test_scene(int subdiv);

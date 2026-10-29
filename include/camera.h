// Pinhole camera plus the render settings that every renderer shares.
#pragma once

#include "vecmath.h"

struct Camera {
    vec3 pos, fwd, right, up;  // right/up are pre-scaled by tan(fov/2) (and aspect)
    int width, height;
};

struct RenderParams {
    int width = 1920;
    int height = 1080;
    int aoSamples = 4;
    float aoRadius = 0.1f;      // as a fraction of the scene diagonal (CLI value)
    float fovDeg = 35.0f;
    float yawDeg = 25.0f;       // orbit around the up axis
    float pitchDeg = 15.0f;     // elevation
    int upAxis = 1;             // 1 = +Y up, 2 = +Z up
    uint32_t seed = 1234u;
};

// Values derived from the scene bounds; passed to the kernels by value.
struct ShadeParams {
    Camera cam;
    vec3 lightDir;     // unit vector toward the (directional) light
    vec3 worldUp;
    float aoDist;      // absolute AO ray length
    float eps;         // absolute self-intersection offset
    int aoSamples;
    uint32_t seed;
};

ShadeParams make_shade_params(const AABB& bounds, const RenderParams& rp);

HD Ray camera_ray(const Camera& c, int px, int py) {
    float sx = (2.0f * (float(px) + 0.5f) / float(c.width) - 1.0f);
    float sy = (1.0f - 2.0f * (float(py) + 0.5f) / float(c.height));
    vec3 d = normalize(c.fwd + c.right * sx + c.up * sy);
    return make_ray(c.pos, d);
}

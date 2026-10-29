#include "camera.h"

ShadeParams make_shade_params(const AABB& bounds, const RenderParams& rp) {
    const float deg = 3.14159265f / 180.0f;
    vec3 center = (bounds.bmin + bounds.bmax) * 0.5f;
    float diag = length(bounds.bmax - bounds.bmin);
    float radius = 0.5f * diag;

    // World frame: U = up, Z0 = axis the camera looks along at yaw 0, X0 = side axis.
    vec3 U, Z0, X0;
    if (rp.upAxis == 2) { U = {0, 0, 1}; Z0 = {0, -1, 0}; X0 = {1, 0, 0}; }
    else                { U = {0, 1, 0}; Z0 = {0, 0, 1};  X0 = {1, 0, 0}; }

    float yaw = rp.yawDeg * deg, pitch = rp.pitchDeg * deg, halfFov = 0.5f * rp.fovDeg * deg;
    vec3 offs = (X0 * sinf(yaw) + Z0 * cosf(yaw)) * cosf(pitch) + U * sinf(pitch);
    // Fits the bounding sphere (with a little margin for perspective) in the shorter side.
    float dist = radius / sinf(halfFov) * 0.95f;

    ShadeParams sp;
    Camera& c = sp.cam;
    c.width = rp.width;
    c.height = rp.height;
    c.pos = center + offs * dist;
    c.fwd = normalize(center - c.pos);
    vec3 right = normalize(cross(c.fwd, U));
    vec3 up = cross(right, c.fwd);
    float aspect = float(rp.width) / float(rp.height);
    float th = tanf(halfFov);
    // Fit the bounding sphere along the shorter image side.
    if (aspect >= 1.0f) { c.right = right * (th * aspect); c.up = up * th; }
    else                { c.right = right * th; c.up = up * (th / aspect); }

    sp.lightDir = normalize(right * -0.45f + up * 0.85f - c.fwd * 0.55f);
    sp.worldUp = U;
    sp.aoDist = rp.aoRadius * diag;
    sp.eps = 2e-5f * diag;
    sp.aoSamples = rp.aoSamples;
    sp.seed = rp.seed;
    return sp;
}

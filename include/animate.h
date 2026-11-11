// Incoherent motion for the dynamic-scene experiment: the mesh explodes, and every triangle
// flies off as a rigid piece in its own random direction. Neighbouring triangles move apart,
// which is the case where refitting a fixed topology degrades and a rebuild is needed.
// The motion is one __host__ __device__ function used by both sides (positions can differ
// in the last bit, because the GPU contracts multiply-adds; nothing here compares them).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mesh.h"
#include "rng.h"
#include "vecmath.h"

struct Explosion {
    vec3 center;   // explosion origin (mesh bounds centre)
    float scale;   // mesh bounds diagonal
    uint32_t seed = 2024u;
};

// Rigid offset of triangle k at time t in [0, 1]: half radial, half random direction,
// random speed, a little gravity. At t = 1 the fastest pieces have travelled ~0.6 diagonals.
HD vec3 explode_offset(const Explosion& e, uint32_t k, vec3 centroid0, int upAxis, float t) {
    Rng rng = make_rng(k, 0u, e.seed);
    vec3 radial = centroid0 - e.center;
    float rl = length(radial);
    radial = rl > 0.0f ? radial * (1.0f / rl) : make_vec3(0.0f, 1.0f, 0.0f);
    vec3 rnd = make_vec3(2.0f * rng.next() - 1.0f, 2.0f * rng.next() - 1.0f, 2.0f * rng.next() - 1.0f);
    vec3 dir = normalize(radial + rnd * 1.5f);
    float speed = e.scale * (0.1f + 0.5f * rng.next());
    vec3 up = upAxis == 2 ? make_vec3(0.0f, 0.0f, 1.0f) : make_vec3(0.0f, 1.0f, 0.0f);
    return dir * (speed * t) - up * (0.15f * e.scale * t * t);
}

// "Unwelds" a mesh so every triangle owns its three vertices (3k, 3k+1, 3k+2).
Mesh unweld(const Mesh& m);

// verts[3k+j] = verts0[3k+j] + explode_offset(k, centroid of triangle k at rest, t).
void animate_cpu(const std::vector<vec3>& verts0, std::vector<vec3>& verts, const Explosion& e, int upAxis,
                 float t);

// Copies a host array into a new device buffer (nullptr and err on failure); device_free(nullptr) is a no-op.
void* device_upload(const void* src, size_t bytes, std::string& err);
void device_free(void* p);

// Same on the GPU (device pointers, 3 vertices per triangle). Returns kernel time in ms.
bool animate_gpu(const void* dVerts0, void* dVerts, int numTris, const Explosion& e, int upAxis, float t,
                 double& ms, std::string& err);

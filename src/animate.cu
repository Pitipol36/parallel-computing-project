// Exploding-mesh animation on the CPU and GPU (same per-triangle function).
#include <cuda_runtime.h>

#include "animate.h"

Mesh unweld(const Mesh& m) {
    Mesh out;
    out.vertices.resize(m.faces.size() * 3);
    out.faces.resize(m.faces.size());
    for (size_t k = 0; k < m.faces.size(); ++k) {
        const Face& f = m.faces[k];
        out.vertices[3 * k] = m.vertices[f.a];
        out.vertices[3 * k + 1] = m.vertices[f.b];
        out.vertices[3 * k + 2] = m.vertices[f.c];
        out.faces[k] = {uint32_t(3 * k), uint32_t(3 * k + 1), uint32_t(3 * k + 2)};
    }
    return out;
}

void* device_upload(const void* src, size_t bytes, std::string& err) {
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes ? bytes : 4);
    if (e == cudaSuccess) e = cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        err = cudaGetErrorString(e);
        if (p) cudaFree(p);
        return nullptr;
    }
    return p;
}

void device_free(void* p) {
    if (p) cudaFree(p);
}

static HD vec3 centroid3(vec3 a, vec3 b, vec3 c) { return (a + b + c) * (1.0f / 3.0f); }

void animate_cpu(const std::vector<vec3>& verts0, std::vector<vec3>& verts, const Explosion& e, int upAxis,
                 float t) {
    const size_t n = verts0.size() / 3;
    verts.resize(verts0.size());
    for (size_t k = 0; k < n; ++k) {
        vec3 a = verts0[3 * k], b = verts0[3 * k + 1], c = verts0[3 * k + 2];
        vec3 off = explode_offset(e, uint32_t(k), centroid3(a, b, c), upAxis, t);
        verts[3 * k] = a + off;
        verts[3 * k + 1] = b + off;
        verts[3 * k + 2] = c + off;
    }
}

__global__ void k_animate(const vec3* __restrict__ v0, vec3* __restrict__ v, int n, Explosion e, int upAxis, float t) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    vec3 a = v0[3 * k], b = v0[3 * k + 1], c = v0[3 * k + 2];
    vec3 off = explode_offset(e, uint32_t(k), centroid3(a, b, c), upAxis, t);
    v[3 * k] = a + off;
    v[3 * k + 1] = b + off;
    v[3 * k + 2] = c + off;
}

bool animate_gpu(const void* dVerts0, void* dVerts, int numTris, const Explosion& e, int upAxis, float t,
                 double& ms, std::string& err) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    k_animate<<<(numTris + 255) / 256, 256>>>(static_cast<const vec3*>(dVerts0), static_cast<vec3*>(dVerts), numTris,
                                              e, upAxis, t);
    cudaEventRecord(b);
    cudaError_t ce = cudaEventSynchronize(b);
    if (ce == cudaSuccess) ce = cudaGetLastError();
    float f = 0;
    cudaEventElapsedTime(&f, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    ms = f;
    if (ce != cudaSuccess) {
        err = std::string("animate: ") + cudaGetErrorString(ce);
        return false;
    }
    return true;
}

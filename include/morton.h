// Morton codes and the Karras (2012) radix-tree step, shared by the CPU and GPU LBVH
// builders. Everything is __host__ __device__, so both builders run identical logic and
// produce node-for-node identical trees.
#pragma once

#include <cstdint>

#include "vecmath.h"

#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#include <intrin.h>
#endif

HD int clz64(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __clzll(static_cast<long long>(x));
#elif defined(_MSC_VER)
    unsigned long idx;
    return _BitScanReverse64(&idx, x) ? 63 - int(idx) : 64;
#else
    return x ? __builtin_clzll(x) : 64;
#endif
}

HD int clz32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __clz(static_cast<int>(x));
#elif defined(_MSC_VER)
    unsigned long idx;
    return _BitScanReverse(&idx, x) ? 31 - int(idx) : 32;
#else
    return x ? __builtin_clz(x) : 32;
#endif
}

// Spread the low 21 bits of v so that two zero bits separate each pair of bits.
HD uint64_t morton_expand21(uint32_t v) {
    uint64_t x = v & 0x1fffffu;
    x = (x | x << 32) & 0x1f00000000ffffULL;
    x = (x | x << 16) & 0x1f0000ff0000ffULL;
    x = (x | x << 8) & 0x100f00f00f00f00fULL;
    x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
    x = (x | x << 2) & 0x1249249249249249ULL;
    return x;
}

// 63-bit Morton code from three 21-bit grid coordinates (x in the highest bit of each triple).
HD uint64_t morton3d(uint32_t x, uint32_t y, uint32_t z) {
    return (morton_expand21(x) << 2) | (morton_expand21(y) << 1) | morton_expand21(z);
}

// Maps scene centroid bounds to the 2^21 grid: q = (c - lo) * inv, with inv = 1 / (hi - lo)
// (0 on a flat axis). inv is computed once by whoever owns the bounds; the expression has
// no multiply-add to contract, so host and device round identically.
struct MortonFrame {
    vec3 lo;
    vec3 inv;
};

HD MortonFrame make_morton_frame(const AABB& centroidBounds) {
    MortonFrame f;
    f.lo = centroidBounds.bmin;
    vec3 e = centroidBounds.bmax - centroidBounds.bmin;
    f.inv = {e.x > 0.0f ? 1.0f / e.x : 0.0f, e.y > 0.0f ? 1.0f / e.y : 0.0f, e.z > 0.0f ? 1.0f / e.z : 0.0f};
    return f;
}

HD uint32_t quantize21(float c, float lo, float inv) {
    float t = (c - lo) * inv;
    t = rmin(rmax(t, 0.0f), 1.0f);
    uint32_t q = uint32_t(t * 2097152.0f);  // 2^21, exact scaling
    return q > 2097151u ? 2097151u : q;
}

HD uint64_t morton_code(const MortonFrame& f, vec3 c) {
    return morton3d(quantize21(c.x, f.lo.x, f.inv.x), quantize21(c.y, f.lo.y, f.inv.y),
                    quantize21(c.z, f.lo.z, f.inv.z));
}

// Karras's delta: length of the longest common prefix of sorted keys i and j, or -1 when j
// is out of range. Equal keys fall back to 64 + the common prefix of the indices, so every
// key is effectively unique and duplicates still form a valid tree.
HD int lbvh_delta(const uint64_t* keys, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    uint64_t a = keys[i], b = keys[j];
    if (a != b) return clz64(a ^ b);
    return 64 + clz32(uint32_t(i) ^ uint32_t(j));
}

// Node ids used by every binary-tree builder: [0, n-1) are internal nodes, [n-1, 2n-1) are
// leaves, and leaf id n-1+k holds sorted triangle k.
HD int lbvh_leaf_id(int n, int k) { return n - 1 + k; }

// Karras 2012, Figure 4: the children and key range of internal node i (0 <= i < n-1).
// Node 0 is the root.
HD void lbvh_internal_node(const uint64_t* keys, int n, int i, int& left, int& right) {
    const int d = (lbvh_delta(keys, n, i, i + 1) - lbvh_delta(keys, n, i, i - 1)) >= 0 ? 1 : -1;
    const int dmin = lbvh_delta(keys, n, i, i - d);

    // Upper bound for the range length, then the exact other end j = i + l*d.
    int lmax = 2;
    while (lbvh_delta(keys, n, i, i + lmax * d) > dmin) lmax *= 2;
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2)
        if (lbvh_delta(keys, n, i, i + (l + t) * d) > dmin) l += t;
    const int j = i + l * d;

    // Split position: the last key that shares more than delta(i, j) bits with key i.
    const int dnode = lbvh_delta(keys, n, i, j);
    int s = 0;
    int t = l;
    do {
        t = (t + 1) / 2;
        if (lbvh_delta(keys, n, i, i + (s + t) * d) > dnode) s += t;
    } while (t > 1);
    const int gamma = i + s * d + (d < 0 ? d : 0);

    const int first = i < j ? i : j, last = i < j ? j : i;
    left = first == gamma ? lbvh_leaf_id(n, gamma) : gamma;
    right = last == gamma + 1 ? lbvh_leaf_id(n, gamma + 1) : gamma + 1;
}

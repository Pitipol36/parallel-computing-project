// Deterministic per-pixel, per-sample random numbers (PCG hash), identical on CPU and GPU.
#pragma once

#include "vecmath.h"

HD uint32_t pcg_hash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct Rng {
    uint32_t state;
    HD float next() {
        state = pcg_hash(state);
        return float(state >> 8) * (1.0f / 16777216.0f);  // [0, 1) with 24 bits
    }
};

HD Rng make_rng(uint32_t pixel, uint32_t sample, uint32_t seed) {
    return Rng{pcg_hash(pixel ^ pcg_hash(sample * 0x9E3779B9u + seed))};
}

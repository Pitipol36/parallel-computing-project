// PLOC (Meister & Bittner 2018), shared by the CPU and GPU builders.
//
// Clusters start as one triangle each, in Morton order. Each iteration:
//   1. every cluster finds its nearest neighbour among the r clusters on each side, where the
//      distance is the surface area of the merged box;
//   2. mutual nearest neighbours merge (the lower index creates the new internal node and
//      keeps its slot, the other is removed);
//   3. the survivors are compacted in order (an exclusive scan of the "still valid" flags).
// New internal nodes get ids in creation order (0, 1, ...), so the root is node n-2 and every
// child id is smaller than its parent's. Node ids otherwise follow bvh_common.h.
//
// Determinism: the area uses round-to-nearest multiplies and adds with no contraction, and
// ties are broken by a total order on pairs. With that, the CPU and GPU builders produce
// node-for-node identical trees, and a mutual pair always exists (the globally smallest pair
// is mutual), so every iteration merges at least once.
#pragma once

#include "vecmath.h"

// Surface area of the union of two boxes, bit-identical on host and device.
HD float ploc_merged_area(vec3 alo, vec3 ahi, vec3 blo, vec3 bhi) {
    const float ex = fsub_rn(rmax(ahi.x, bhi.x), rmin(alo.x, blo.x));
    const float ey = fsub_rn(rmax(ahi.y, bhi.y), rmin(alo.y, blo.y));
    const float ez = fsub_rn(rmax(ahi.z, bhi.z), rmin(alo.z, blo.z));
    return 2.0f * fadd_rn(fadd_rn(fmul_rn(ex, ey), fmul_rn(ey, ez)), fmul_rn(ez, ex));
}

// For cluster i: is candidate j at distance d better than the current best bj at bd?
// Pairs are ordered by (distance, index gap, parity of the smaller index, smaller index, larger
// index), a strict total order. On exact distance ties (identical boxes) the gap and parity
// terms make neighbours pair up as (0,1), (2,3), ... so ties merge in balanced rounds instead
// of one pair per iteration (which would build a chain as deep as the number of triangles).
HD bool ploc_better(float d, int i, int j, float bd, int bj) {
    if (d != bd) return d < bd;
    const int lo = i < j ? i : j, hi = i < j ? j : i;
    const int blo = i < bj ? i : bj, bhi = i < bj ? bj : i;
    if (hi - lo != bhi - blo) return hi - lo < bhi - blo;
    if ((lo & 1) != (blo & 1)) return (lo & 1) < (blo & 1);
    return lo != blo ? lo < blo : hi < bhi;
}

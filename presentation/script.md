# Presentation script: BVH Ray Tracing of Large Triangle Meshes on CUDA

About 12 minutes of talking, then Q&A, out of a 15-minute slot. About 1,800 spoken words, so it
needs a brisk ~150 words per minute; at a calmer 130 it runs about 14 minutes, so time a rehearsal. Lines in *[brackets and italics]* are stage directions. Items in plain `[brackets]` are
facts only you know, so fill them in before rehearsing.

| Part | Slides | Time |
|---|---|---|
| Why this topic | 1 | 1:30 |
| The problem and the data | 2–3 | 1:00 |
| BVH in depth | 4–5 | 3:45 |
| Parallel design and kernels | 6–7 | 1:45 |
| Correctness | 8 | 0:40 |
| Results | 9–13 | 2:45 |
| Wrap-up | 14–15 | 0:35 |

If you run long, shorten the results section (slides 12–13), not the BVH section.

---

## Slide 1: Cover (1:30)

Hi, I’m Pitipol. My project is ray tracing big 3D-scanned meshes on the GPU, and I want to start
with why I picked this topic.

Before this course I wrote a ray tracer of my own, RT-Project, in Python. My goal was to render a
black hole: an accretion disk, a few planets, a little space probe. The black hole is faked. The
engine only shoots straight rays, so near the black hole I bend each ray toward the singularity
with a pull of mass over distance squared, like gravity.

It worked, and I was proud of it. But it taught me something. To find what a ray hits, my scene
loops over every object, tests every one, and keeps the closest. With about ten spheres, rings and
quads, that’s fine. I even split the image into rows and ran them on all my CPU cores. Even so, a
4K render at 1024 samples per pixel took `[how long it took]`.

So I wondered what it would take to render a *real* object: not ten spheres, but a laser-scanned
statue with a million triangles. With my old loop, every ray would test a million triangles.
That needs two things my old project didn’t have. The first is a smarter way to find the closest
hit: a bounding volume hierarchy, or BVH. The second is far more parallel hardware: the GPU. Those
two things are this project.

*[Point to the dragon.]* This is the Stanford dragon, 871 thousand triangles. The GPU renders it in
about 7 milliseconds.

## Slide 2: The problem (0:40)

Every pixel shoots rays. One primary ray finds what the pixel sees. One shadow ray asks whether
that point can see the light. And N ambient-occlusion rays ask how enclosed the point is, which
gives the soft darkening in creases.

At full HD with 4 AO samples, that’s about 12 million rays per frame, and each one has to search
up to a million triangles. The rays don’t depend on each other, so it’s very parallel. But each
search walks an irregular tree. That’s a different pattern from our labs: not a stencil, a scan or
a reduction, but tree traversal.

## Slide 3: Dataset (0:20)

The models come from the Stanford 3D Scanning Repository: the bunny for development, the dragon
as the main benchmark, and the Happy Buddha at 1.09 million triangles as the largest. The
28-million-triangle Lucy was only on Stanford’s server, which I couldn’t reach, so the Buddha is
my biggest model.

## Slide 4: The BVH, in depth (3:00)

Now the core of the talk: how a BVH works and why it wins.

**The cost of the naive way.** Finding the closest hit naively means testing every triangle:
rays times triangles. For the dragon, that’s 12 million rays times 871 thousand triangles, about
ten to the thirteen ray-triangle tests per frame. That’s my RT-Project loop at a much bigger scale.

**The key idea.** If a ray misses a box, it misses everything inside that box. Testing a ray
against an axis-aligned box is very cheap: a few subtractions and multiplications per axis to find
where the ray enters and leaves each pair of slabs, then a comparison. So we put boxes around
groups of triangles, and when a ray misses a box, one cheap test rules out thousands of triangles.

**The structure.** Do it recursively and you get a tree. The root box holds the whole model. Each
node splits its triangles into two groups and puts a tight box around each group. You keep
splitting until a leaf holds only a few triangles. For the dragon that gives 959 thousand nodes,
about 480 thousand leaves with fewer than two triangles each, and a depth of 26.

Notice that we split the *objects*, not the space. Every triangle lives in exactly one leaf, and
two sibling boxes may overlap. That’s the difference from a kd-tree or a grid, which cut space
itself and have to copy triangles that straddle a cut.

**How to choose a split: the Surface Area Heuristic.** This is where the quality comes from. You
could split a node down the middle of its box, or at the median triangle. But the best split
depends on the rays. Here’s the insight: if a random ray hits a parent box, the chance it also
hits a child box is about the child’s surface area divided by the parent’s. So the expected cost
of a split is:

> cost = one traversal step + (area_left / area_parent) × triangles_left
> + (area_right / area_parent) × triangles_right

We try candidate splits and keep the cheapest. If no split beats just testing all the triangles,
the node becomes a leaf. SAH favours small, tight boxes with many triangles and avoids big empty
boxes that rays hit for nothing.

**Making the build fast: binning.** Trying every possible split is slow, so I use binning. Along
each axis I drop triangle centres into 16 bins, then sweep once from the left and once from the
right. That scores all 15 split planes in linear time per node, so the whole build is N log N. The
dragon builds in about 0.8 seconds on one core.

## Slide 5: Traversal, and why it’s the best sequential version (0:45)

**Traversal.** A ray starts at the root and tests both children’s boxes. It goes into the
*nearer* child first and pushes the farther one on a small stack, along with the distance where it
enters it. Once the ray has hit a triangle at distance t, any box it enters beyond t can’t hold a
closer hit, so we skip it, both when we first meet it and when we pop it from the stack.

Shadow and AO rays only need *any* hit, not the closest, so they stop at the first triangle they
hit.

**Why this is the best sequential version.** SAH is the standard acceleration structure in
production renderers and research. The nodes are flattened into an array in depth-first order, 32
bytes each, so the left child is always the next node. Triangle edges and the ray’s inverse
direction are precomputed. One fun finding: Visual Studio compiles `fminf` into a function call,
and swapping it for a plain comparison made both the render and the build about six times faster.

**Why it’s better: the numbers.** Each ray now visits roughly log-of-N nodes, about the tree’s
depth of 26, instead of 871 thousand triangles. The total goes from rays × triangles to rays × log
of triangles. *[Point to the box on slide 4.]* Measured on the bunny: brute force on the **GPU**
takes 3.6 seconds, and the BVH on **one CPU core** takes 38 milliseconds. A good algorithm on one
core beats a bad algorithm on two thousand GPU cores, by a factor of about a hundred. That’s why I
built the BVH first and parallelized second.

And it’s why GPUs now have ray-tracing hardware: NVIDIA’s RT cores walk exactly this kind of tree.
In this project I wrote the traversal by hand in CUDA, with no ray-tracing libraries.

## Slide 6: Parallel design (1:00)

So how does it run on the GPU? I build the BVH once on the CPU, with the same code as the
baseline, and upload it. Then every GPU thread owns one pixel and traces its primary, shadow and
AO rays, each with its own small stack.

To make the comparison fair and checkable, the ray, box, triangle, traversal and shading code is
written *once*, as `__host__ __device__` functions, so the CPU and the GPU run literally the same
code. The random numbers for AO come from a hash of the pixel and sample number, so both sides
shoot exactly the same rays.

There’s no communication between threads. The real costs are divergence, when threads in a warp
take different paths down the tree, and scattered memory reads.

## Slide 7: Four kernels (0:45)

To see what each optimization buys, I wrote four kernels. Brute force tests every triangle. The
naive BVH kernel uses the same tree in a plain struct layout. The optimized kernel uses aligned
`float4` loads through the read-only cache, the inverse direction and nearer-child-first ordering.
The persistent-threads kernel launches just enough warps to fill the GPU and has them grab tiles of
pixels from a global atomic counter.

## Slide 8: Correctness (0:40)

`--verify` checks that the CPU and GPU agree. For every pixel, the primary ray must hit the
same triangle. On the dragon, 5 pixels out of 307 thousand differ, all exactly on triangle edges,
where the GPU’s fused multiply-add rounds differently. The images match at 57.6 decibels, the test
scene is bit-identical, and all 44 unit tests pass, including the BVH checked against brute force
on random rays.

## Slide 9: Headline (0:20)

The headline: on every setting I measured, the GPU is **35 to 180 times** faster than the tuned
single-core version, with the same tree and the same rays.

## Slide 10: Speedup vs mesh size (0:40)

This one surprised me. I expected the gap to grow with mesh size, but it *shrinks*: 182 times on
the bunny, 113 on the Buddha. The BVH is the reason. It made the CPU’s cost grow only with the log
of the triangle count. On the GPU, a bigger tree no longer fits in the 3-megabyte L2 cache, so
threads wait on memory.

## Slide 11: Speedup vs ray workload (0:30)

Where the gap *does* grow is with the number of rays: 47 times at 512 by 512, 145 times at 4K. The
GPU needs lots of rays in flight to hide memory latency. And end to end, the fixed upload cost gets
shared by more pixels.

## Slide 12: Ablation (0:40)

Breaking it down: the algorithm is by far the biggest win, with the BVH 7,500 times faster than
brute force. The memory layout gives another 3 times. Persistent threads gave no measurable gain.
On this GPU, the hardware scheduler already balances cheap background pixels against expensive
ones.

## Slide 13: Where the time goes (0:35)

Once rendering takes a few milliseconds, the single-threaded BVH build becomes the biggest cost:
almost a second on the Buddha. And small or wide blocks, like 8 by 8, are slightly faster, because
nearby pixels take similar paths through the tree.

## Slide 14: Limitations and future work (0:25)

The obvious next step is to build the BVH on the GPU too, with Morton codes, radix sort and scan,
which are the patterns from our labs. I’d also sort AO rays by direction to cut divergence, and try
the 28-million-triangle Lucy.

## Slide 15: Questions (0:10)

To sum up: going from my ten-sphere Python ray tracer to a million triangles took two steps. A BVH
made each ray’s search logarithmic, and the GPU ran millions of those searches at once. Thank you,
and I’m happy to take questions.

---

## Likely questions

- **Why a BVH and not a kd-tree or a grid?** Every triangle is stored once, so memory is bounded
  (at most 2N − 1 nodes). It builds fast, and it handles uneven scan density well. A uniform grid
  wastes cells on empty space. A kd-tree has to copy triangles that straddle split planes.
- **Why build the BVH on the CPU?** It keeps the comparison fair: both versions use the identical
  tree. A GPU build (LBVH) is the future-work item.
- **Why didn’t persistent threads help?** The technique came from GPUs whose block scheduler handled
  uneven work poorly. Turing’s scheduler already refills SMs quickly, so the atomic counter only
  adds overhead.
- **Is 180× fair, given the CPU has 8 cores?** The assignment compares against sequential code, and
  the baseline is deliberately single-threaded. Perfect scaling over 8 cores would still leave the
  GPU about 15–20× ahead.
- **How many nodes does a ray actually visit?** Roughly the depth (about 20–26) along each path,
  plus side branches it can’t rule out. A few dozen box tests and a handful of triangle tests,
  against 871 thousand triangle tests without the tree.

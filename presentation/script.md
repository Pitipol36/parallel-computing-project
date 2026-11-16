# Presentation script: BVH Ray Tracing of Large Triangle Meshes on CUDA

The original talk (slides 1–16, 25–26) is about 12 minutes; the GPU-build follow-up (slides
17–24) adds about 5¾ minutes, so the whole deck runs about 18 minutes plus Q&A. If the slot is
15 minutes, either present the follow-up as its own block, or drop slides 15–16 and 22 (the
follow-up's points survive without them). About 2,500 spoken words at ~150 words per minute; time
a rehearsal. Lines in *[brackets and italics]* are stage directions. Items in plain `[brackets]` are
facts only you know, so fill them in before rehearsing.

| Part | Slides | Time |
|---|---|---|
| Why this topic | 1 | 1:30 |
| The problem and the data | 2–3 | 1:00 |
| Brute force and the BVH in depth | 4–8 | 4:10 |
| Parallel design and kernels | 9–10 | 1:55 |
| Correctness | 11 | 0:40 |
| Results | 12–16 | 2:45 |
| Follow-up: building the BVH on the GPU | 17–24 | 5:40 |
| Wrap-up | 25–26 | 0:35 |

If you run long, shorten the results section (slides 15–16), not the BVH section.

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

## Slide 4: Brute force (0:35)

*[Point to the loop.]* First, the obvious way. For each pixel, make a camera ray, then loop over
every triangle, run a ray-triangle test, and keep the closest hit. Shadow and AO rays use the same
loop but stop at the first hit. It’s simple and always correct, and it’s exactly my RT-Project
loop.

The problem is the cost: rays times triangles. For the dragon, that’s 12 million rays times 871
thousand triangles, about ten to the thirteen ray-triangle tests per frame. Even on the GPU, the
small bunny at 320 by 240 takes 3.6 seconds. So before parallelizing anything, we need a better
algorithm.

## Slide 5: The BVH, in depth (0:40)

Now the core of the talk: how a BVH works and why it wins.

**The key idea.** If a ray misses a box, it misses everything inside that box. Testing a ray
against an axis-aligned box is very cheap: a few subtractions and multiplications per axis to find
where the ray enters and leaves each pair of slabs, then a comparison. So we put boxes around
groups of triangles, and when a ray misses a box, one cheap test rules out thousands of triangles.

## Slide 6: How a BVH is built (1:30)

*[The diagram: twelve triangles, root, boxes A and B, four leaves; the tree on the right.]*

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

## Slide 7: How a ray walks the BVH (0:50)

**Traversal.** *[Follow the ray on the diagram, step by step.]* A ray starts at the root and tests both children’s boxes. It goes into the
*nearer* child first and pushes the farther one on a small stack, along with the distance where it
enters it. Once the ray has hit a triangle at distance t, any box it enters beyond t can’t hold a
closer hit, so we skip it, both when we first meet it and when we pop it from the stack.

Shadow and AO rays only need *any* hit, not the closest, so they stop at the first triangle they
hit.

## Slide 8: Why it’s the best sequential version (0:35)

**Why this is the best sequential version.** SAH is the standard acceleration structure in
production renderers and research. The nodes are flattened into an array in depth-first order, 32
bytes each, so the left child is always the next node. Triangle edges and the ray’s inverse
direction are precomputed. One fun finding: Visual Studio compiles `fminf` into a function call,
and swapping it for a plain comparison made both the render and the build about six times faster.

**Why it’s better: the numbers.** Each ray now visits roughly log-of-N nodes, about the tree’s
depth of 26, instead of 871 thousand triangles. The total goes from rays × triangles to rays × log
of triangles. *[Go back to slide 4, or just quote it.]* Measured on the bunny: brute force on the **GPU**
takes 3.6 seconds, and the BVH on **one CPU core** takes 38 milliseconds. A good algorithm on one
core beats a bad algorithm on two thousand GPU cores, by a factor of about a hundred. That’s why I
built the BVH first and parallelized second.

And it’s why GPUs now have ray-tracing hardware: NVIDIA’s RT cores walk exactly this kind of tree.
In this project I wrote the traversal by hand in CUDA, with no ray-tracing libraries.

## Slide 9: Parallel design (1:00)

So how does it run on the GPU? I build the BVH once on the CPU, with the same code as the
baseline, and upload it. Then every GPU thread owns one pixel and traces its primary, shadow and
AO rays, each with its own small stack.

To make the comparison fair and checkable, the ray, box, triangle, traversal and shading code is
written *once*, as `__host__ __device__` functions, so the CPU and the GPU run literally the same
code. The random numbers for AO come from a hash of the pixel and sample number, so both sides
shoot exactly the same rays.

There’s no communication between threads. The real costs are divergence, when threads in a warp
take different paths down the tree, and scattered memory reads.

## Slide 10: Four kernels (0:55)

To see what each optimization buys, I wrote four kernels. Brute force tests every triangle. The
naive BVH kernel uses the same tree in a plain struct layout. The optimized kernel uses aligned
`float4` loads through the read-only cache, the inverse direction and nearer-child-first ordering.
The persistent-threads kernel launches just enough warps to fill the GPU and has them grab tiles of
pixels from a global atomic counter.

*[Point to the table.]* On the biggest model, the 1.09-million-triangle Buddha at full HD, the
optimized kernel takes 8.3 milliseconds; one CPU core takes 937.

## Slide 11: Correctness (0:40)

`--verify` checks that the CPU and GPU agree. For every pixel, the primary ray must hit the
same triangle. On the dragon, 7 pixels out of 307 thousand differ, all exactly on triangle edges,
where the GPU’s fused multiply-add rounds differently. The images match at 56.7 decibels, on the
test scene every pixel hits the same triangle, and all 216 unit-test checks pass, including every
BVH checked against brute force on random rays.

## Slide 12: Headline (0:20)

The headline: on every setting I measured, the GPU is **35 to 180 times** faster than the tuned
single-core version, with the same tree and the same rays.

## Slide 13: Speedup vs mesh size (0:40)

This one surprised me. I expected the gap to grow with mesh size, but it *shrinks*: 182 times on
the bunny, 113 on the Buddha. The BVH is the reason. It made the CPU’s cost grow only with the log
of the triangle count. On the GPU, a bigger tree no longer fits in the 3-megabyte L2 cache, so
threads wait on memory.

## Slide 14: Speedup vs ray workload (0:30)

Where the gap *does* grow is with the number of rays: 47 times at 512 by 512, 145 times at 4K. The
GPU needs lots of rays in flight to hide memory latency. And end to end, the fixed upload cost gets
shared by more pixels.

## Slide 15: Ablation (0:40)

Breaking it down: the algorithm is by far the biggest win, with the BVH 7,500 times faster than
brute force. The memory layout gives another 3 times. Persistent threads gave no measurable gain.
On this GPU, the hardware scheduler already balances cheap background pixels against expensive
ones.

## Slide 16: Where the time goes (0:35)

Once rendering takes a few milliseconds, the single-threaded BVH build becomes the biggest cost:
almost a second on the Buddha. And small or wide blocks, like 8 by 8, are slightly faster, because
nearby pixels take similar paths through the tree.

## Slide 17: Follow-up: building the BVH on the GPU (0:40)

*[Point to the CPU line at the top of the chart.]* That last result is where the follow-up starts:
the single-threaded build was now most of the time to the first image. So I built the BVH on the
GPU, first with Karras's LBVH. Every triangle gets a Morton code, a radix sort puts them along a
space-filling curve, one thread per node builds the hierarchy, and a bottom-up pass with atomic
counters fits the boxes. On the Buddha it takes 6 milliseconds instead of almost a second: 160
times faster than the CPU SAH build, and 67 times faster than the same algorithm on one core. Every
GPU builder has a single-threaded twin, and the GPU tree is checked to be identical to it.

## Slide 18: The price: a looser tree (0:40)

The speed has a price. Morton order ignores surface area, so the boxes are looser: on the scans
the tree costs 19 to 30 percent more by SAH, rays do about a fifth more box tests, and rendering is
10 to 24 percent slower. *[Point to the "mixed sizes" row.]* On a scene with mixed triangle sizes
it is much worse. But it only matters when a tree serves many rays: the SAH tree catches up after
about 5 billion rays, roughly 400 full-HD frames.

## Slide 19: Winning the quality back: PLOC and TRBVH (0:45)

Two higher-quality GPU builders win that back. PLOC clusters bottom-up: every cluster finds its
nearest neighbour among its Morton-order neighbours, mutual pairs merge, and a scan I wrote myself,
the scan lab's pattern, compacts the list. That gets within 12 percent of the best tree. TRBVH starts
from the LBVH and, under every large node, finds the optimal shape for a treelet of seven subtrees
with a small dynamic program, one warp per treelet. That gets within 6 percent, in 58
milliseconds.

## Slide 20: H-PLOC: PLOC quality in one kernel launch (0:40)

PLOC's weakness is its loop: 68 iterations on the dragon, each with several kernel launches and a
readback to the CPU. H-PLOC, from last year, builds the whole tree in one launch. Threads climb the
Morton hierarchy from the leaves; each node carries a short list of clusters, and when two
children's lists together exceed sixteen, the whole warp merges them in shared memory. Same
quality as PLOC, 1.3 to almost 4 times faster. The paper gets within about 15 percent of the LBVH;
mine takes about twice the LBVH's time, and I know where that goes.

## Slide 21: A harder scene and a stronger reference (0:45)

*[Point to the image.]* Scanned meshes are the easy case, because every triangle is about the same
size. So I added a scene with mixed sizes: huge ground triangles, thin pillars, and objects at very
different scales. And I replaced the binned SAH reference with a full sweep that tries every split.
On the scans binning is within one percent of it; on the mixed scene it is 11 percent worse, and
**TRBVH beats the CPU SAH builder** while building 19 times faster. I also made the timing fairer:
warm-up, eleven interleaved runs and medians. The clocks can't be locked without admin rights, so
the quality claims rest on SAH cost and traversal counters, which are deterministic.

## Slide 22: Build speed vs tree quality (0:40)

All the builders on one chart: build time against tree quality, lower left is better. Every GPU
builder is on the front. LBVH is the fastest and the loosest, H-PLOC is the middle ground, and
TRBVH gets closest to the reference while still building 16 times faster than the CPU. So the right
builder depends on how many rays one tree will serve.

## Slide 23: Parallel reinsertion: past the CPU reference (0:40)

The last step works on any of these trees. Parallel reinsertion asks, for every node at once:
where else could this node go so the tree's SAH cost drops the most? Moves that don't touch the
same nodes are applied together: each one locks its six nodes with an atomic max, the winners
move, and the boxes are refitted. After eight rounds every GPU tree is within a few percent of the
full-sweep reference on the scans, and TRBVH plus PRBVH beats it. On the mixed scene all four end up
5 to 15 percent **better** than the full sweep, which is still greedy at the top. H-PLOC plus PRBVH
builds the dragon in 240 milliseconds with a better tree than the CPU builder's 760. Renders get 8
to 43 percent faster, so the extra time pays off after 11 to 65 frames.

## Slide 24: Dynamic scenes: rebuild every frame (0:30)

*[Let the animation play.]* Where a fast build really matters is a scene that changes every frame.
The dragon explodes. Refitting the old tree costs a millisecond, but its boxes stretch across the
explosion and rendering slows to tens of seconds per frame. Rebuilding on the GPU every frame keeps
the tree good at 12 to 34 frames per second; the CPU needs one and a half seconds per frame.

## Slide 25: Limitations and future work (0:25)

Most of PRBVH's time is its exact search over the whole tree; searching only near each node would
trade a little quality for a lot of time. H-PLOC could get closer to the LBVH's speed by folding the
leaf collapse into the build. And I'd still like to try the 28-million-triangle Lucy.

## Slide 26: Questions (0:10)

To sum up: going from my ten-sphere Python ray tracer to a million triangles took two steps. A BVH
made each ray's search logarithmic, and the GPU ran millions of those searches at once. And now the
GPU builds that tree too: in milliseconds close to the best CPU builder's quality, or in a quarter
of a second better than it. Thank you,
and I'm happy to take questions.

---

## Likely questions

- **Why a BVH and not a kd-tree or a grid?** Every triangle is stored once, so memory is bounded
  (at most 2N − 1 nodes). It builds fast, and it handles uneven scan density well. A uniform grid
  wastes cells on empty space. A kd-tree has to copy triangles that straddle split planes.
- **Why build the BVH on the CPU in the main results?** It keeps the comparison fair: both versions
  use the identical tree. The follow-up (slides 17–24) then builds it on the GPU.
- **How do you know the GPU builders are correct?** Each one has a single-threaded CPU twin running the
  same shared code, and the GPU tree must equal it byte for byte; every tree must also give the same
  triangle per pixel as the SAH tree, and `compute-sanitizer` must report no errors or races.
- **Why is your H-PLOC slower than the paper's?** About half the extra time is the warp merging itself,
  the rest is the flatten step: node ids come from an atomic counter, so the tree is scattered in
  memory. The paper fuses the leaf collapse into the build, which I didn't.
- **How can a GPU tree beat the full-sweep SAH reference?** The full sweep is greedy: it picks the
  best split at each node from the top down and never revisits it. Reinsertion measures the whole
  tree's cost and moves subtrees across those early splits. The mixed scene, with huge and tiny
  triangles meeting near the root, is where the greedy choice is worst.
- **Why are PRBVH's CPU and GPU trees identical?** Each round is a pure function of the tree at the
  start of the round. Ties are broken by the node's in-order position, not by its id, so the
  H-PLOC tree (whose GPU ids come from an atomic counter) makes the same moves on both.
- **Why is binned SAH not 100 %?** Binning only tries 15 split planes per axis. On uniform scans that
  costs about 1 %, but on mixed triangle sizes 11 %; the full sweep tries every split.
- **Why didn’t persistent threads help?** The technique came from GPUs whose block scheduler handled
  uneven work poorly. Turing’s scheduler already refills SMs quickly, so the atomic counter only
  adds overhead.
- **Is 180× fair, given the CPU has 8 cores?** The assignment compares against sequential code, and
  the baseline is deliberately single-threaded. Perfect scaling over 8 cores would still leave the
  GPU about 15–20× ahead.
- **How many nodes does a ray actually visit?** Roughly the depth (about 20–26) along each path,
  plus side branches it can’t rule out. A few dozen box tests and a handful of triangle tests,
  against 871 thousand triangle tests without the tree.

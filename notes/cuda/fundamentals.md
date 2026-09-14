# Fundamentals

## Single Global Index

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
**blockIdx.x:** the block's index within the grid

**blockDim.x:** number of threads per block

**threadIdx.x:** the thread's position within its own block

### Block 0 example

- For a block with 8 threads, we have the following representation.

```text

Block 0:

threadIdx.x
    0  1  2  3  4  5  6  7
    ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
   T0 T1 T2 T3 T4 T5 T6 T7

```

- The formula becomes

```c
int i = 0 * 8 + threadIdx.x;
```

- threadIdx.x maps to

```text

thread 0 → i = 0
thread 1 → i = 1
thread 2 → i = 2
...
thread 7 → i = 7

```

- Suppose we launch `<<<4,8>>>`, that is, 4 blocks with 8 threads each.
- Visually we have

```text
Block 0              Block 1              Block 2              Block 3

T0 T1 T2 T3 T4 T5 T6 T7
                     T0 T1 T2 T3 T4 T5 T6 T7
                                          T0 T1 T2 T3 T4 T5 T6 T7
                                                               T0 T1 T2 T3 T4 T5 T6 T7
```

- Equivalently

```text

Block 0: threadIdx = 0 1 2 3 4 5 6 7
Block 1: threadIdx = 0 1 2 3 4 5 6 7
Block 2: threadIdx = 0 1 2 3 4 5 6 7
Block 3: threadIdx = 0 1 2 3 4 5 6 7

```

- Each block handles a contiguous, non-overlapping range of `blockDim.x` global indices, offset by `blockIdx.x * blockDim.x`: Block 0 covers indices 0–7, Block 1 covers 8–15, Block 2 covers 16–23, and Block 3 covers 24–31.

```text
                    GLOBAL INDEX

Block 0                 Block 1                 Block 2                 Block 3
--------                --------                --------                --------
0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | 16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31
↑                    ↑ | ↑                     ↑ | ↑                     ↑ | ↑                     ↑
T0                  T7 | T0                   T7 | T0                   T7 | T0                   T7
```

## Coalescing

**Memory coalescing** is a GPU memory-controller optimization that combines the individual global-memory accesses issued by the threads of a single warp into as few physical memory transactions as possible. It applies specifically to global memory accesses made by threads within the same warp: if thread 0 accesses location $n$, thread 1 accesses location $n+1$, and so on up to thread 31 accessing location $n+31$, the hardware merges all 32 accesses into a single, efficient transaction instead of issuing 32 separate ones. The more consecutive and aligned the accesses across a warp, the fewer transactions are needed and the better the memory bandwidth utilization.

- Suppose we have `kernel<<<N / 256, 256>>>`
- The inside of the kernel uses the global index.
- A warp is a group of 32 threads.
- Each block has 256 / 32 = 8 warps.
- Threads in the same warp execute the same instruction.
- Imagine one warp

```text
Block 0

ThreadIdx:

0  1  2  3  4  5 ... 31
↓  ↓  ↓  ↓  ↓  ↓     ↓
0  1  2  3  4  5 ... 31

```
- The warp's threads produce consecutive global indices (0-31)
- These indices access x[0] through x[31]
- so for the eight warps we get

```text
Warp 0 → x[0:31]
Warp 1 → x[32:63]
Warp 2 → x[64:95]
...
```
### AoS - Array of Structures vs SoA - Structure of Arrays

- Suppose we have a particle with the following attributes

```c
struct Particle {
    float x;
    float y;
    float z;
    float mass;
};
```

#### AoS
- With AoS, we create an array of Particle:

```c
Particle particles[32]
```

- Memory looks conceptually like:

```text
particle 0                particle 1                particle 2
┌────┬────┬────┬────┐     ┌────┬────┬────┬────┐     ┌────┬────┬────┬────┐
│ x  │ y  │ z  │ m  │     │ x  │ y  │ z  │ m  │     │ x  │ y  │ z  │ m  │
└────┴────┴────┴────┘     └────┴────┴────┴────┘     └────┴────┴────┴────┘
```

```text
x0 y0 z0 m0  x1 y1 z1 m1  x2 y2 z2 m2  x3 y3 z3 m3 ...
```

- Each particle is 16 bytes (4 floats x 4 bytes each).
- If each thread in a warp reads the `x` field of its own particle (`particles[threadIdx.x].x`), thread 0 reads `x0`, thread 1 reads `x1`, thread 2 reads `x2`, and so on:

```text
Thread       Address

T0 ───────→  x0
T1 ───────────────→ x1
T2 ───────────────────────→ x2
T3 ─────────────────────────────→ x3
...
```

```
x0 y0 z0 m0 | x1 y1 z1 m1 | x2 y2 z2 m2 | x3 y3 z3 m3
^            ^              ^              ^
T0           T1             T2             T3
```
- The threads are accessing 4 floats.
- Although the threads are processing consecutive particles, their `x` values aren't adjacent in memory.

```text
T0 → address 0
T1 → address 4 floats
T2 → address 8 floats
T3 → address 12 floats
...
```
- This converts to a stride of 16 bytes.
- This can still be reasonably coalesced because the addresses are regular and close enough that multiple requests can be covered by the same cache lines/sectors. But it is not as compact as having the 32 requested floats laid out contiguously.

#### SoA
- With SoA, we instead store each attribute in its own contiguous array:

```c
struct ParticlesSoA {
    float x[32];
    float y[32];
    float z[32];
    float mass[32];
};
```

- Memory looks conceptually like:

```text
x array                          y array                          z array                          mass array
┌───┬───┬───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┬───┬───┐
│x0 │x1 │x2 │x3 │...│...│x31│    │y0 │y1 │y2 │y3 │...│...│y31│    │z0 │z1 │z2 │z3 │...│...│z31│    │m0 │m1 │m2 │m3 │...│...│m31│
└───┴───┴───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┴───┴───┘
```

- When each thread in a warp reads `x[threadIdx.x]`, thread 0 reads `x0`, thread 1 reads `x1`, thread 2 reads `x2`, and so on, up through thread 31 reading `x31`:

```text
Thread       Address

T0 ───────→  x0
T1 ──────→   x1
T2 ─────→    x2
T3 ────→     x3
...
T31 →        x31
```

```
x0 x1 x2 x3 ... x31
^  ^  ^  ^      ^
T0 T1 T2 T3 ... T31
```
- Here the 32 threads' accesses land on 32 consecutive 4-byte floats with **no stride at all** — exactly the pattern the coalescing rule above describes (thread $i$ accesses location $n + i$). This maps to a single, fully coalesced memory transaction per field access, rather than the strided one seen in AoS.

### Comparing AoS and SoA coalescing
- **AoS**: accessing one field (e.g., `x`) across a warp means jumping in 16-byte steps (the full size of a `Particle`), since the other fields (`y`, `z`, `mass`) sit between consecutive particles' `x` values in memory. The accesses are still regular and can often be covered by a handful of cache-line-sized transactions, but a meaningful fraction of each fetched cache line is data the warp doesn't actually need right now (the other fields).
- **SoA**: accessing the same field across a warp means reading 32 truly contiguous floats with zero stride, since all `x` values are packed together. This is the ideal coalescing pattern, uses the full width of each memory transaction productively, and typically delivers better effective bandwidth than AoS for this kind of field-at-a-time access.
- The trade-off is that AoS keeps all of one particle's data together, which is convenient if a thread needs *every* field of *one* particle, whereas SoA is better when many threads each need *one* field across *many* particles — which is the far more common access pattern in data-parallel GPU kernels, and why SoA is generally preferred in CUDA code for performance-sensitive kernels.
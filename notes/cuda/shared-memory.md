# Shared Memory

## Shared-memory architecture
- **Shared memory** is a small, fast, on-chip, programmer-managed memory space — much lower latency and much higher bandwidth than global memory.
- **Scope and lifetime**: shared memory is allocated per thread block and lives for exactly as long as that block does. It is visible to (and shareable between) every thread in the block, but not to threads in other blocks.
- **Allocation** comes in two flavors:
  - **Static** — size is fixed at compile time (`__shared__ int data[256];`). This is capped at 48 KB per block on most architectures and lets the compiler optimize access patterns more aggressively.
  - **Dynamic** — size is only known at kernel-launch time, declared with `extern __shared__ ...` and passed as a launch parameter. Only one dynamic array is allowed per block; multiple logical regions must be carved out of it with pointer offsets.
- **Capacity**: shared memory is a limited, per-SM resource (physically it shares an on-chip pool with the L1 cache, split between them per kernel/architecture); how much of it a modern GPU makes available per block or per SM varies by architecture and can be raised above the 48 KB default via `cudaFuncSetAttribute`.
- **Occupancy implications**: since shared memory is finite per SM, a kernel that requests a lot of it per block leaves room for fewer concurrently resident blocks, which can lower occupancy (the ratio of active warps to the SM's maximum). Balancing shared-memory usage against occupancy — much like balancing register usage against occupancy — is a common tuning trade-off.
- **Latency vs. bandwidth**: shared memory has low latency for any single access, but its real strength is aggregate bandwidth — many threads accessing it in parallel, across many banks, in the same cycle. That aggregate bandwidth is only realized when accesses are spread across banks (see below); when they collide, bandwidth collapses toward serial, single-bank latency.

## Simplified GPU Model
- Shared memory belongs to SM
- Threads in the same block can access the block's shared-memory allocation

```text
                 GPU
                  |
       +----------+----------+
       |          |          |
      SM 0       SM 1       SM 2 ...
       |          |          |
   shared mem  shared mem  shared mem
       |          |          |
   threads      threads      threads
```

## Banks

- Shared memory isn't one giant monolithic RAM array.
- Shared memory is divided into banks
- For an architecture with 32 banks in shared memory, we have the following conceptual model

```text
Bank 0
Bank 1
Bank 2
Bank 3
...
Bank 31
```
- This allows the ideal case:
```text
Thread 0  -> Bank 0
Thread 1  -> Bank 1
Thread 2  -> Bank 2
...
Thread 31 -> Bank 31
```
- This allows threads to access shared memory efficiently.
- Suppose we have:

```c
__shared__ int data[32];
```
- Assuming int = 4 bytes, the conceptual mapping is

```text
Element       Address       Bank

data[0]       0             0
data[1]       4             1
data[2]       8             2
data[3]       12            3
...
data[31]      124           31
data[32]      128           0
data[33]      132           1
...
```
- So approximately:

$$
    bank = \left(\frac{address}{4} \right) mod \ 32
$$

- The bank structure repeats.
```text
Elements:

0   1   2   3   ... 30  31
|   |   |   |        |   |
B0  B1  B2  B3       B30 B31

32  33  34  35  ... 62  63
|   |   |   |        |   |
B0  B1  B2  B3       B30 B31
```
- Assuming a warp has 32 threads and shared memory has 32 banks, the conceptual mapping is a follows:

```c
shared[threadIdx.x]
```
```text
Thread       Element       Bank

T0           shared[0]      B0
T1           shared[1]      B1
T2           shared[2]      B2
...
T31          shared[31]     B31
```

- Every thread accesses a different bank.
- This is known as conflict-free access.
- If multiple threads try to access the same bank, it creates a bank conflict.

```c
shared[0]
```

```text
T0  -> B0
T1  -> B0
T2  -> B0
...
T31 -> B0
```
- In the above example, all the thirty-two threads are trying to access the same bank.
- The hardware cannot simply service all those distinct addresses through the same bank simultaneously in the normal way, so the accesses are effectively serialized into multiple bank transactions.
- Conceptually,

```text
Cycle 1:
T0 -> B0

Cycle 2:
T1 -> B0

Cycle 3:
T2 -> B0

...

Cycle 32:
T31 -> B0
```
- This is why shared memory can become slow despite having enormous bandwidth.
- Conflicts scale with how many threads collide on a bank in the same request — a **2-way conflict** (2 threads hitting the same bank) takes 2 serialized transactions, a **4-way conflict** takes 4, and so on up to the worst case, a **32-way conflict** (all 32 threads on one bank), which takes 32 serialized transactions — roughly 32x slower than the conflict-free case.
- If all threads are reading the same address, the hardware is able to broadcast the same value to the threads.
- So the broadcast case can be handled efficiently

```text
T0  ─┐
T1  ─┤
T2  ─┤
...   ├──> shared[0]
T31 ──┘
```
- Therefore, Same bank does not automatically mean a bad bank conflict. A request only becomes a true bank conflict when threads in the same bank ask for *different* addresses; if they all ask for the exact same address, it's a free broadcast instead.
- The ability of the banks to service many accesses in parallel is known as bandwidth.
- With tremendous parallelism we expect something like

```text
Bank 0   Bank 1   Bank 2   ... Bank 31
  |        |        |             |
  ↓        ↓        ↓             ↓
 T0       T1       T2            T31
```
- The following case throws away the parallelism
  
```text
Bank 0
 | | | | | | | |
 ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
T0 T1 T2 T3 ... 
```
- Bandwidth isn't just a property of the memory itself, it also depends on the access pattern.

## Access patterns: rows vs. columns

### What "fast-varying" means
- Within a block, CUDA numbers threads so that `threadIdx.x` cycles through all its values first, before `threadIdx.y` ever increments — `threadIdx.x` is therefore the **fast-varying index**, and `threadIdx.y` is the **slow-varying index**.
- Concretely, for a 2D block with `blockDim = (4, 4)`:

```text
Thread   threadIdx.x   threadIdx.y
T0            0             0
T1            1             0
T2            2             0
T3            3             0
T4            0             1
T5            1             1
T6            2             1
T7            3             1
...
```
- Notice `threadIdx.x` runs 0→3 (fast) while `threadIdx.y` only ticks up once every 4 threads (slow). This is also true for a real 32-wide warp: `threadIdx.x` sweeps through consecutive values across the warp, while `threadIdx.y` stays fixed for the whole warp (as long as the block's x-dimension is a multiple of 32).

### Setup for the examples
Take a small tile so the whole thing is easy to draw, stored row-major:

```c
__shared__ int tile[4][4];
```

```text
Row-major memory layout (element index)

Row 0: tile[0][0] tile[0][1] tile[0][2] tile[0][3]   →   0  1  2  3
Row 1: tile[1][0] tile[1][1] tile[1][2] tile[1][3]   →   4  5  6  7
Row 2: tile[2][0] tile[2][1] tile[2][2] tile[2][3]   →   8  9 10 11
Row 3: tile[3][0] tile[3][1] tile[3][2] tile[3][3]   →  12 13 14 15
```

To keep the toy example self-consistent, assume this GPU has **4 banks** instead of 32 (same "stride collides with bank count" idea, just scaled down so it fits on screen):

```text
element:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
bank:      0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3     (bank = element mod 4)
```

### Row access: `tile[threadIdx.y][threadIdx.x]`
Here the **row** is picked by the slow-varying `threadIdx.y` (fixed across most of the warp) and the **column** by the fast-varying `threadIdx.x`. Looking at threads T0–T3 (`threadIdx.y = 0`, `threadIdx.x = 0..3`):

```text
T0 -> tile[0][0] -> element 0  -> bank 0
T1 -> tile[0][1] -> element 1  -> bank 1
T2 -> tile[0][2] -> element 2  -> bank 2
T3 -> tile[0][3] -> element 3  -> bank 3
```

```text
Row 0 in memory:   [ e0 ][ e1 ][ e2 ][ e3 ]
                      ↑     ↑     ↑     ↑
threads:              T0    T1    T2    T3
banks:                B0    B1    B2    B3
```
Each thread lands on consecutive elements *within the same row* → consecutive addresses → 4 different banks (B0–B3). Conflict-free, exactly like the 1D `shared[threadIdx.x]` example earlier.

### Column access: `tile[threadIdx.x][threadIdx.y]`
Now the **row** is picked by the fast-varying `threadIdx.x`, and the **column** by the slow-varying `threadIdx.y`. Looking at the same T0–T3 (`threadIdx.x = 0..3`, `threadIdx.y = 0`):

```text
T0 -> tile[0][0] -> element 0  -> bank 0
T1 -> tile[1][0] -> element 4  -> bank 0
T2 -> tile[2][0] -> element 8  -> bank 0
T3 -> tile[3][0] -> element 12 -> bank 0
```

```text
Column 0, down the rows:
Row 0: [ e0  ]  ← T0
Row 1: [ e4  ]  ← T1
Row 2: [ e8  ]  ← T2
Row 3: [ e12 ]  ← T3

banks:    B0      B0      B0      B0
```
Each thread now jumps a full **row stride** (4 elements) to reach its target instead of moving to the next element — and because that stride (4) is a multiple of the bank count (4), every thread lands back on bank 0. All 4 threads collide → a 4-way bank conflict (a 32-way conflict at real scale, with a 32×32 tile and 32 banks).

### Why 32×32 is special
It's the same "stride equals bank count" coincidence as the toy example above, just at full size: a `[32][32]` tile has a row stride of 32 words, which is also the real bank count on the GPU, so a full-column access (`tile[threadIdx.x][threadIdx.y]`) always collides on a single bank.

### Why 32×33 fixes it (padding)
Allocating the tile as `__shared__ int tile[32][33]` (one extra "padding" column per row) changes the row stride from 32 to 33 words. Since 33 shares no common factor with 32, each successive row's element for a given column now lands on a *different* bank — rotating column accesses across all 32 banks and eliminating the conflict. The one wasted `int` per row costs a little shared memory but restores full bank parallelism.

To see exactly why, replay the toy example — same `tile[threadIdx.x][threadIdx.y]` column access, same 4 toy banks, but pad the tile from `[4][4]` to `[4][5]` (one extra element per row, mirroring the real `[32][33]` fix):

```c
__shared__ int tile[4][5];   // padded: 5 elements per row instead of 4
```

```text
Row-major memory layout (element index)

Row 0: tile[0][0] tile[0][1] tile[0][2] tile[0][3] tile[0][4]   →   0  1  2  3  4
Row 1: tile[1][0] tile[1][1] tile[1][2] tile[1][3] tile[1][4]   →   5  6  7  8  9
Row 2: tile[2][0] tile[2][1] tile[2][2] tile[2][3] tile[2][4]   →  10 11 12 13 14
Row 3: tile[3][0] tile[3][1] tile[3][2] tile[3][3] tile[3][4]   →  15 16 17 18 19
```

The row stride is now **5** instead of 4 — one column (`tile[*][4]`) is pure padding, never touched by useful data. Re-run the same column-0 access, `tile[threadIdx.x][threadIdx.y]` with `threadIdx.y = 0`, across T0–T3:

```text
T0 -> tile[0][0] -> element 0  -> bank 0   (0  mod 4 = 0)
T1 -> tile[1][0] -> element 5  -> bank 1   (5  mod 4 = 1)
T2 -> tile[2][0] -> element 10 -> bank 2   (10 mod 4 = 2)
T3 -> tile[3][0] -> element 15 -> bank 3   (15 mod 4 = 3)
```

```text
Column 0, down the padded rows:
Row 0: [ e0  ]  ← T0
Row 1: [ e5  ]  ← T1
Row 2: [ e10 ]  ← T2
Row 3: [ e15 ]  ← T3

banks:    B0      B1      B2      B3
```

Compare this to the unpadded `[4][4]` case above, where every thread landed on B0 — here, because the stride (5) and the bank count (4) share no common factor, each thread's jump down a row also nudges it one bank over, so T0–T3 spread across B0–B3 instead of piling onto one bank. Conflict-free, at the cost of one unused element per row. This is exactly the same mechanism that makes `[32][33]` conflict-free at full scale.

### Bank conflicts in matrix multiplication
Tiled matrix-multiply kernels commonly load `A`/`B` sub-tiles into shared memory and then read them column-wise (or in some transposed order) during the multiply-accumulate step — the same row-stride-equals-bank-count collision shown above can occur there too, and the same `[TILE][TILE+1]` padding trick is the standard fix.

## Global memory relationship
- **Alignment and memory transactions**: coalescing (see the earlier note on this topic) is about how a warp's accesses to **global** memory get merged into physical DRAM transactions, based on address alignment and cache-line/sector boundaries and global-memory bandwidth.
- **Bank conflicts** are a completely separate phenomenon that happens on **shared** memory, governed by which of the (typically 32) banks an address maps to, not by DRAM alignment or cache lines.
- **Why they're different problems**: a kernel can have perfectly coalesced global-memory loads into shared memory, and still suffer badly from bank conflicts when that data is later read back out of shared memory in a different pattern (e.g., row-in/column-out, as in a transpose) — and vice versa. They have to be diagnosed and fixed independently.

## The transpose example
- Say we have the following tile

```text
__shared__ int tile[32][32];
```
- A warp loads something like:

```text
Global memory

T0  -> A[0][0]
T1  -> A[0][1]
T2  -> A[0][2]
...
T31 -> A[0][31]
```

- Then we put it into

```text
tile[0][0]
tile[0][1]
tile[0][2]
...
tile[0][31]
```

- Consider

```c
tile[threadIdx.x][threadIdx.y]
```

- **Warping with `__shared__ int tile[32][32]`**: this expression uses `threadIdx.x` to pick the *row* and `threadIdx.y` to pick the *column* — this is exactly the column-style access described above. Across a warp (which varies `threadIdx.x`), each thread jumps a full 32-element row stride to reach its element, so every thread in the warp lands on the same bank: a 32-way bank conflict, even though the earlier global-memory load into the tile was fully coalesced.
- **The optimization: `__shared__ int tile[32][33]`**: padding the tile to 32×33 changes the row stride from 32 to 33 words. Because 33 is not a multiple of 32, the same `tile[threadIdx.x][threadIdx.y]` access now spreads across all 32 distinct banks, restoring conflict-free, full-bandwidth shared-memory access — at the cost of one wasted `int` per row.
- Putting it together, a typical **tiled transpose** kernel: (1) each block coalesced-loads a tile of the input matrix from global memory into shared memory (row-wise), (2) calls `__syncthreads()` so every thread's write is visible before any thread reads, (3) reads the tile back out of shared memory in transposed order and writes it to the transposed location in global memory (also coalesced, since the output indices are recomputed to be row-wise for the write) — with the shared-memory tile padded (`[32][33]`) so that step (3)'s transposed access pattern stays bank-conflict-free.

### Benchmarking the variants: naive vs. shared-memory transpose
This benchmark ran a 2048×2048 `int` matrix transpose (`BLOCK_SIZE = 32`, so a 32×32 grid of 32×32 blocks) with `nvprof` on an NVIDIA Quadro P2000 — a Pascal-generation GPU (GP106, compute capability 6.1, 8 SMs, 5 GB GDDR5, ~140 GB/s peak memory bandwidth).

**`matrix_transpose_naive`** — no shared memory at all, reads and writes global memory directly:
```c
int inputIndex  = y * N + x;   // read
int outputIndex = x * N + y;   // write
output[outputIndex] = input[inputIndex];
```
- The **read**, `input[y * N + x]`, is coalesced: `x` (fast-varying across the warp) is the fast-varying part of the address too, so consecutive threads touch consecutive elements.
- The **write**, `output[x * N + y]`, is not: as `x` varies across the warp, the address jumps by a full row (`N = 2048` elements = 8192 bytes) each time. Every thread in the warp writes to a different, widely-separated location — this is about as uncoalesced as a global-memory access pattern gets, and it's a *global-memory coalescing* problem, unrelated to shared-memory banks (there's no shared memory here to have banks in).

**`matrix_transpose_shared`** — stages each tile through shared memory instead of transposing directly in global memory:
```c
tile[threadIdx.y][threadIdx.x] = input[y * N + x];      // load into shared mem
__syncthreads();
output[transposedY * N + transposedX] = tile[threadIdx.x][threadIdx.y];  // store from shared mem
```
- The global **read** is the same coalesced `input[y * N + x]` pattern as the naive kernel.
- The **write into shared memory**, `tile[threadIdx.y][threadIdx.x]`, is a *row access* (row = `threadIdx.y`, column = the fast-varying `threadIdx.x`) — conflict-free, exactly like the row-access example earlier in this note.
- The **read back out of shared memory**, `tile[threadIdx.x][threadIdx.y]`, is the *column access* case — and since `tile` is declared as plain `[BLOCK_SIZE][BLOCK_SIZE]` = `[32][32]` with **no padding**, this is exactly the 32-way bank conflict worked through above.
- The global **write**, `output[transposedY * N + transposedX]`, is coalesced: `transposedX` depends on the fast-varying `threadIdx.x`, so consecutive threads in the warp write consecutive addresses.

So the shared-memory kernel trades the naive kernel's badly-strided *global*-memory write for a same-warp *shared*-memory bank conflict on the read-back — reordering the data on-chip (where a 32-way conflict, while wasteful, is still relatively cheap) instead of reordering it directly against much slower, much less forgiving DRAM.

**Measured kernel times (`nvprof`):**

| Kernel | Time | Achieved BW* | % of peak (140 GB/s) |
|---|---|---|---|
| `matrix_transpose_naive`  | 1.1260 ms | ≈ 29.8 GB/s | ≈ 21% |
| `matrix_transpose_shared` | 887.33 µs | ≈ 37.8 GB/s | ≈ 27% |

*\*Achieved bandwidth = total bytes moved (one read + one write of the full 2048×2048 `int` matrix ≈ 33.55 MB) ÷ kernel time.*

- The shared-memory version is about **1.27× faster** (kernel time drops ~21%), even though its shared-memory read-back is still bank-conflicted. That's consistent with the reasoning above: an on-chip 32-way bank conflict is still cheaper than the badly-strided global-memory write it replaced, so the kernel comes out ahead overall.
- Both kernels are still far from the card's 140 GB/s peak (21% and 27% respectively) — expected, since neither is achieving fully efficient memory access on every step.
- The `[CUDA memcpy HtoD]`/`[CUDA memcpy DtoH]` times (roughly 3–4 ms each in both runs) are essentially constant across the two profiles, as expected — they depend on PCIe transfer of the same-sized matrix either way, not on which transpose kernel runs. The kernel time is the only column that isolates the effect of the algorithm change.
- **Natural next step**: since the shared kernel's remaining bottleneck is the unpadded `tile[32][32]`'s 32-way bank conflict, padding it to `tile[32][33]` (as described above) should remove that conflict entirely while keeping both global-memory accesses coalesced — the expected result is a further drop in kernel time, worth re-measuring with `nvprof`/Nsight Compute to confirm.
- Note: `nvprof` is NVIDIA's legacy command-line profiler; it still ships with recent CUDA toolkits for compatibility, but new projects are generally pointed toward Nsight Systems (timeline-level) and Nsight Compute (kernel-level, used throughout the section above) instead.

## Profiling with Nsight Compute
- **Nsight Compute** is NVIDIA's kernel-level profiler, and is the standard tool for confirming whether bank conflicts (or coalescing issues) are actually hurting a given kernel, rather than guessing from source code alone.
- **Shared-memory bank-conflict metrics**: look at the SOL (Speed of Light) memory breakdown for shared-memory data bank reads/writes; if these are elevated relative to peak, bank conflicts are inflating the number of shared-memory transactions the kernel issues.
- **Global-memory efficiency**: separately, the global-memory/L2 throughput and sector-efficiency metrics show whether global-memory accesses are well coalesced (close to the ideal number of sectors per request) or wasteful (many more sectors touched than bytes actually used).
- **Load/store throughput and memory transactions**: Nsight Compute reports achieved memory throughput against the theoretical peak, plus the actual number of memory transactions issued versus the ideal minimum — a big gap between the two is a direct, quantitative signal of coalescing or bank-conflict overhead.
- **Warp stall reasons**: the Warp State Statistics view breaks down *why* warps aren't issuing instructions each cycle. Two stall reasons are especially relevant here — "Stall MIO Throttle" (the shared-memory instruction queue is backed up) and "Stall Short Scoreboard" (a warp is waiting on the extra latency caused by a conflicting shared-memory access); both rising together is a strong sign of bank conflicts specifically.
- **Distinguishing a real bottleneck from a scary-looking metric**: a nonzero bank-conflict or stall metric isn't automatically a problem — check it against the kernel's overall behavior (e.g., is it compute-bound or memory-bound overall in the roofline/SOL summary?). If the kernel is already saturating a different resource (compute throughput, global-memory bandwidth), fixing bank conflicts may free up very little real time; the metric is only worth chasing when it lines up with the stall reasons that are actually dominating the kernel's cycles.
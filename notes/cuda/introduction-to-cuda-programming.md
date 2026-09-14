# Introduction to CUDA Programming

- CUDA is short for **Compute Unified Device Architecture** — NVIDIA's parallel computing platform and programming model for using a GPU for general-purpose computation.

## Definitions
- **Floating-Point Operations per Second (FLOPS)** is the fundamental unit for measuring the theoretical peak performance of a compute processor.
  - **MegaFLOP** = $10^6$ FLOPS
  - **GigaFLOP** = $10^9$ FLOPS
  - **TeraFLOP** = $10^{12}$ FLOPS
  - **PetaFLOP** = $10^{15}$ FLOPS
- **Instruction-Level Parallelism (ILP)** is the ability of code-independent instructions to execute at the same time. For operations to run in parallel, they must be independent of one another (i.e., neither depends on the other's result).
- **Latency** is the time it takes to complete a single operation, from start to finish (e.g., the time for one memory access or one instruction to finish).
- **Throughput** is the amount of work completed per unit of time (e.g., operations per second, or bytes moved per second) — it measures overall processing rate rather than the speed of any single operation.
- **Bandwidth** is the rate at which data can be moved to or from memory, typically measured in GB/s; it's the main factor limiting memory throughput.
- **Kernel** is a function written to run on the GPU, launched from and controlled by the CPU (host).
- **Thread, block, grid** describe CUDA's execution hierarchy: individual **threads** are grouped into **thread blocks**, and blocks are grouped into a **grid** — this is how a kernel's work is distributed across the GPU's cores.
- **Warp** is a group of (typically 32) threads that the GPU schedules and executes together in lockstep, following the same instruction at the same time (SIMT execution).
- **Arithmetic intensity** is the ratio of compute operations to bytes of memory transferred for a given piece of code; it's used to determine whether that code is likely to be compute-bound or memory-bound on a given GPU.

## Heterogeneous computing
- A GPU is not a replacement for a CPU — it's used to **accelerate** the parts of an application that are parallel in nature, which is why a GPU is also referred to as an **accelerator**.
- CPUs and GPUs are optimized for different goals:
  - **CPU** architecture is optimized for **low-latency** access — fast completion of individual tasks — and is therefore described as **latency-bound**.
  - **GPU** architecture is optimized for **data-parallel throughput** — completing as much total work as possible per unit of time.
- This difference shows up in how each hides latency:
  - CPUs hide latency by relying on a large, multi-level cache hierarchy (L1, L2, L3) — cache size decreases and latency drops as you move from L3 down to L1, keeping frequently used data close to the processor.
  - GPUs hide latency differently: rather than relying on large caches, they keep many threads in flight and switch to a different thread's computation whenever one thread is stalled waiting on memory.
  - Correspondingly, GPUs dedicate proportionally more of their hardware to compute units and registers (to support many concurrent threads) than to cache, which is the opposite trade-off from a CPU.
- **Throughput** is the amount of work a processor completes per unit of time (e.g., FLOPS for compute, or GB/s for data movement) — as opposed to latency, which measures how long a single operation takes.
- A piece of code's performance is typically described as:
  - **Compute-bound**, when it's limited by the processor's arithmetic throughput (FLOPS) — i.e., there's a lot of computation relative to the data being moved.
  - **Memory-bound**, when it's limited by memory bandwidth rather than compute — i.e., there's a lot of data movement relative to the computation performed on it.
  - **Throughput-bound** is the general case of being limited by a resource's sustained processing rate (compute or memory) rather than by the latency of any single operation; in practice this usually shows up as either the compute-bound or memory-bound case above.
  - Which category a piece of code falls into can be estimated using its **arithmetic intensity** relative to the GPU's own compute-to-bandwidth ratio.
- Combining a highly efficient (low-latency) CPU with a high-throughput GPU, and offloading the parallel portions of a program from the CPU onto the GPU, is called **heterogeneous computing**, and results in improved overall performance.
- **Amdahl's Law** explains why this split makes sense, and also why it has diminishing returns: if $P$ is the fraction of a program that can be parallelized (and offloaded to the GPU) and $N$ is the number of parallel processing units applied to it, the maximum theoretical speedup is
$$
S(N) = \frac{1}{(1-P) + \dfrac{P}{N}}
$$
  The remaining serial fraction, $(1-P)$, is executed on the CPU and cannot be sped up no matter how many GPU cores are thrown at the problem — so even a very fast GPU gives only limited overall speedup if a large share of the program must stay serial.
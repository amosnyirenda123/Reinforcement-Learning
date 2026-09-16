# CUDA Programming Model

## Grids and Blocks
- All threads spawned by a single kernel launch are collectively called a **grid**, and all threads in a grid share the same (global) memory space.
- A grid is made up of many **thread blocks**. A block is a group of threads that can cooperate with each other via block-local synchronization and block-local shared memory.
- Threads within the same block can easily communicate with each other; threads that belong to different blocks cannot cooperate.

## Thread and Block Indexing
- To distinguish themselves from one another, threads rely on two built-in coordinate variables, which are assigned to each thread at runtime and are pre-initialized when accessed inside a kernel function:
  - **blockIdx** — the block's index within the grid.
  - **threadIdx** — the thread's index within its block.
- Both are of type `uint3`, a CUDA built-in vector type: a structure holding three unsigned integers, whose components are accessed via the fields `x`, `y`, and `z` — i.e., `blockIdx.x/y/z` and `threadIdx.x/y/z`.

## Grid and Block Dimensions
- CUDA organizes both grids and blocks in up to three dimensions. Their sizes are given by:
  - **blockDim** — block dimensions, measured in threads.
  - **gridDim** — grid dimensions, measured in blocks.
- Both are of type `dim3`, which is based on `uint3`; any component left unspecified is initialized to 1 and effectively ignored. Components are accessed the same way as the index variables, e.g., `blockDim.x/y/z`.
- A grid is usually organized as a 2D array of blocks, and a block as a 3D array of threads.
- For a given data size, the general steps to pick grid and block dimensions are:
  1. Decide on a block size (number of threads per block).
  2. Calculate the grid dimensions from the application's data size and that block size.
- Choosing the block dimension itself usually comes down to two considerations: the kernel's own performance characteristics, and the limits of the available GPU resources.

## Kernel Functions
- A kernel call is **asynchronous** with respect to the host thread: once a kernel is invoked, control returns to the host side immediately, without waiting for the kernel to finish.
- A kernel function must have a `void` return type.
- Function execution/callability is controlled by three qualifiers:
  - **`__global__`** — runs on the device, and is callable from the host. On devices of compute capability 3.5 or higher, a `__global__` function can also be called from the device itself (this is known as **dynamic parallelism** — a kernel launching another kernel).
  - **`__device__`** — runs on the device, and is callable from the device only.
  - **`__host__`** — runs on the host, and is callable from the host only.
  - `__host__` and `__device__` can be combined on the same function, in which case it is compiled separately for both the host and the device.

### CUDA kernel restrictions
CUDA kernels are subject to several restrictions:
- Access to device memory only.
- Must have a `void` return type.
- No support for a variable number of arguments.
- No support for static variables.
- No support for function pointers.
- Exhibit asynchronous execution behavior (as noted above).
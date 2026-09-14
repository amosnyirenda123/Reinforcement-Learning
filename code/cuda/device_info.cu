#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int count;
    cudaGetDeviceCount(&count);

    printf("CUDA devices: %d\n\n", count);

    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device: %s\n", prop.name);
        printf("Compute capability: %d.%d\n",
               prop.major, prop.minor);
        printf("Global memory: %.2f GB\n",
               prop.totalGlobalMem / 1e9);
        printf("Multiprocessors: %d\n",
               prop.multiProcessorCount);
        printf("Max threads/block: %d\n",
               prop.maxThreadsPerBlock);
    }

    return 0;
}